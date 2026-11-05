"""Resource metadata of every glm53 kernel instantiation in an .s file.
usage: kres.py FILE.s [SUBSTRING]"""
import re, sys
s = open(sys.argv[1]).read()
sub = sys.argv[2] if len(sys.argv) > 2 else "glm53"
for m in re.finditer(r"\n  - \.agpr_count.*?(?=\n  - \.agpr_count|\n\.end_amdgpu_metadata)", s, re.S):
    meta = m.group(0)
    name = re.search(r"\.name:\s+(\S+)", meta).group(1)
    if sub not in name: continue
    g = lambda k: (re.search(r"\." + k + r":\s+(\d+)", meta) or [None, "?"])[1]
    print(f"{name[:110]:110} v{g('vgpr_count')} a{g('agpr_count')} spill{g('vgpr_spill_count')} lds{g('group_segment_fixed_size')} priv{g('private_segment_fixed_size')}")
