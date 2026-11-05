"""Count memory ops and vmcnt waits per glm53 kernel in an .s file (crude scheduling check).
usage: waits.py FILE.s [SUBSTRING]"""
import re, sys
from collections import Counter
txt = open(sys.argv[1]).read()
sub = sys.argv[2] if len(sys.argv) > 2 else "glm53"
for m in re.finditer(r"\n(_ZN5aiter\S*" + r"glm53\S*):[^\n]*\n(.*?)s_endpgm", txt, re.S):
    name, body = m.group(1), m.group(2)
    if sub not in name: continue
    c = Counter()
    for line in body.split("\n"):
        line = line.strip()
        w = re.match(r"s_waitcnt vmcnt\((\d+)\)", line)
        if w: c["vmcnt0" if w.group(1) == "0" else "vmcntN"] += 1
        for op in ("global_load", "buffer_load", "ds_read", "ds_write", "s_barrier", "v_mfma", "scratch_"):
            if line.startswith(op): c[op] += 1
    n = re.sub(r"_ZN5aiter12_GLOBAL__N_1\d+", "", name).split("EEEv")[0]
    print(f"{n:70} " + " ".join(f"{k}={c[k]}" for k in ("global_load", "buffer_load", "vmcnt0", "vmcntN", "ds_read", "ds_write", "s_barrier", "v_mfma", "scratch_")))
