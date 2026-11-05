"""usage: icount.py FILE.s PATTERN [OPS...] -> per kernel matching PATTERN: VALU/SALU/ds/mfma totals + listed op counts"""
import re, sys
f, pat, ops = sys.argv[1], sys.argv[2], sys.argv[3:] or ["v_mul_lo_u32", "v_mul_u32_u24", "v_readfirstlane_b32", "s_and_saveexec_b64", "v_cmp_eq_u64"]
name, body = None, []
def flush():
    if not name or not re.search(pat, name): return
    ins = [l.split()[0] for l in body if l.strip() and not l.strip().startswith((";", ".", "_")) and not l.strip().endswith(":")]
    c = lambda p: sum(1 for i in ins if i.startswith(p))
    n = re.sub(r"_ZN5aiter12_GLOBAL__N_1\d+", "", name).split("EEEv")[0]
    print(f"{n:60} total {len(ins):5} v_ {c('v_')-c('v_mfma'):5} s_ {c('s_'):4} ds_ {c('ds_'):3} mfma {c('v_mfma'):3} " + " ".join(f"{o} {sum(1 for i in ins if i == o)}" for o in ops))
for line in open(f):
    m = re.match(r"^(_Z\S+):\s*(;.*)?$", line)
    if m:
        flush(); name, body = m.group(1), []
    elif name is not None:
        if line.strip().startswith("s_endpgm"):
            body.append(line); flush(); name = None
        else: body.append(line)
