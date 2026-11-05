"""Instruction mix of the innermost loop of each glm53 kernel matching SUB.
usage: loopmix.py FILE.s SUB"""
import re, sys
from collections import Counter
txt = open(sys.argv[1]).read()
for m in re.finditer(r"\n(_ZN5aiter\S*" + re.escape(sys.argv[2]) + r"\S*):[^\n]*\n(.*?)s_endpgm", txt, re.S):
    body = m.group(2).split("\n")
    # innermost loop: from a label whose comment says Inner Loop Header (or Depth=2) to the backward branch to it
    starts = [i for i, l in enumerate(body) if re.match(r"\.LBB\S+:", l) and i + 1 < len(body) and "Inner Loop Header" in body[i + 1]]
    for s in starts[:1]:
        lab = body[s].split(":")[0]
        e = next(i for i in range(s + 1, len(body)) if re.search(r"s_cbranch\S* " + re.escape(lab) + r"\b", body[i]))
        c = Counter()
        for l in body[s:e + 1]:
            l = l.strip()
            if not l or l.startswith(";") or l.startswith("."): continue
            op = l.split()[0]
            k = ("mfma" if "mfma" in op else "valu" if op.startswith("v_") else "salu" if op.startswith("s_") and "waitcnt" not in op and "nop" not in op else
                 "ds" if op.startswith("ds_") else "vmem" if op.startswith(("global_", "buffer_")) else "wait/nop")
            c[k] += 1
        name = re.sub(r"_ZN5aiter12_GLOBAL__N_1\d+", "", m.group(1)).split("EEEv")[0]
        print(f"{name[:60]:60} lines {e-s:4d} " + " ".join(f"{k}={c[k]}" for k in ("valu", "mfma", "ds", "vmem", "salu", "wait/nop")))
