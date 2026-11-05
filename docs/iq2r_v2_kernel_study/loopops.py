"""Opcode histogram of the innermost loop of kernels matching SUB. usage: loopops.py FILE.s SUB [dump]"""
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
    lines = [l.strip() for l in body[s:e + 1] if l.strip() and not l.strip().startswith((";", "."))]
    out[name] = lines
names = list(out)
cs = {n: Counter(l.split()[0] for l in out[n]) for n in names}
ops = sorted(set().union(*cs.values()))
print(" " * 34 + " ".join(f"{n[-12:]:>12}" for n in names))
for o in ops:
    v = [cs[n][o] for n in names]
    if len(set(v)) > 1 or len(sys.argv) > 3:
        print(f"{o:34}" + " ".join(f"{x:12d}" for x in v))
if len(sys.argv) > 3:
    for n in names:
        open(f"loop_{sys.argv[3]}_{n[-8:]}.txt", "w").write("\n".join(out[n]))
