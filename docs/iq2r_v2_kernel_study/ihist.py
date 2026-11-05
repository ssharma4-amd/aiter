"""usage: ihist.py PATTERN A.s B.s -> opcode count diff (B - A) for the first kernel matching PATTERN"""
import re, sys
from collections import Counter
def hist(f, pat):
    name, c = None, Counter()
    for line in open(f):
        m = re.match(r"^(_Z\S+):\s*(;.*)?$", line)
        if m:
            if name and re.search(pat, name): return c
            name, c = m.group(1), Counter(); continue
        t = line.strip()
        if name and t and not t.startswith((";", ".", "_")) and not t.endswith(":"):
            c[re.sub(r"_e(32|64)$", "", t.split()[0])] += 1
    return c
pat, a, b = sys.argv[1], hist(sys.argv[2], sys.argv[1]), hist(sys.argv[3], sys.argv[1])
for k in sorted(set(a) | set(b), key=lambda k: -abs(b[k] - a[k])):
    if a[k] != b[k]: print(f"{k:28} {a[k]:5} {b[k]:5} {b[k]-a[k]:+5}")
