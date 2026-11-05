"""usage: cmpn.py BASE A B ...  -> down us per M, ratio vs BASE"""
import sys
from cmp import load
import io, contextlib
files = sys.argv[1:]
data = [load(f) for f in files]
print(f"{'M':>5} " + " ".join(f"{f.split('/')[-1][:14]:>14}" for f in files))
for m in sorted(data[0]):
    base = data[0][m][1].get("down", 0)
    row = []
    for d in data:
        if m not in d: row.append(f"{'-':>14}"); continue
        v = d[m][1].get("down", 0)
        row.append(f"{v:7.2f} {v/base:5.3f}{'X' if d[m][3] else ' '}")
    print(f"{m:5d} " + " ".join(row))
