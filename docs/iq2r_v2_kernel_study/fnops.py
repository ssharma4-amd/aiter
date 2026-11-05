"""Opcode histogram of whole kernels matching SUB minus their innermost loop. usage: fnops.py FILE.s SUB [dump-tag]"""
import re, sys
from collections import Counter
txt = open(sys.argv[1]).read()
out = {}
for m in re.finditer(r"\n(_ZN5aiter\S*" + re.escape(sys.argv[2]) + r"\S*):[^\n]*\n(.*?)s_endpgm", txt, re.S):
    body = m.group(2).split("\n")
    starts = [i for i, l in enumerate(body) if re.match(r"\.LBB\S+:", l) and i + 1 < len(body) and "Inner Loop Header" in body[i + 1]]
    s = starts[0]; lab = body[s].split(":")[0]
    e = next(i for i in range(s + 1, len(body)) if re.search(r"s_cbranch\S* " + re.escape(lab) + r"\b", body[i]))
    name = re.sub(r"_ZN5aiter12_GLOBAL__N_1\d+", "", m.group(1)).split("EEEv")[0]
    keep = lambda ls: [l.strip() for l in ls if l.strip() and not l.strip().startswith((";", "."))]
    out[name] = (keep(body[:s]), keep(body[e + 1:]))
names = list(out)
for part, idx in (("before loop", 0), ("after loop", 1)):
    cs = {n: Counter(l.split()[0] for l in out[n][idx]) for n in names}
    print(f"== {part}: totals " + " ".join(str(sum(c.values())) for c in cs.values()))
    ops = sorted(set().union(*cs.values()))
    for o in ops:
        v = [cs[n][o] for n in names]
        if len(set(v)) > 1:
            print(f"{o:34}" + " ".join(f"{x:8d}" for x in v))
if len(sys.argv) > 3:
    for n in names:
        open(f"fn_{sys.argv[3]}_{n[-8:]}.txt", "w").write("\n".join(out[n][0] + ["<<LOOP>>"] + out[n][1]))
