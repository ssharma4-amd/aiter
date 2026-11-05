"""Per glm53 kernel: VGPRs, LDS, occupancy (waves/SIMD) from the asm comments.
usage: occ.py FILE.s"""
import re, sys
name = None
for line in open(sys.argv[1]):
    m = re.match(r"\s*\.type\s+([^,]+),@function", line)
    if m: name, info = m.group(1), {}
    elif name and line.startswith("; "):
        k = re.match(r"; (NumVgprs|NumAgprs|LDSByteSize|Occupancy|ScratchSize):\s*(\d+)", line)
        if k: info[k.group(1)] = k.group(2)
        if line.startswith("; Occupancy") and "glm53" in name:
            n = re.sub(r"_ZN5aiter12_GLOBAL__N_1\d+", "", name).split("EEEv")[0]
            print(f"{n:72} vgpr {info.get('NumVgprs')} agpr {info.get('NumAgprs')} lds {info.get('LDSByteSize')} scratch {info.get('ScratchSize')} occ {info['Occupancy']}")
