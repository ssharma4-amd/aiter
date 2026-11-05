"""Compare res2_* variants with ablation.py's c4/c2 on the same matrices (weighted mean relg)."""
import glob, json, collections
rows = collections.defaultdict(dict)
for f in glob.glob("res_*.jsonl") + glob.glob("res2_*.jsonl"):
    for l in open(f):
        r = json.loads(l)
        rows[r["set"]][(r["layer"], r["expert"], r["proj"])] = r
base = rows["c4"]
print(f"{'set':6} {'n':>3} {'relg':>8} {'vs c4 same mats':>16}  gate/up/down ratio vs c4")
for s in sorted(rows):
    keys = [k for k in rows[s] if k in base]
    if not keys: continue
    mean = sum(rows[s][k]["relg"] for k in keys) / len(keys)
    ref = sum(base[k]["relg"] for k in keys) / len(keys)
    per = {}
    for p in ("gate_proj", "up_proj", "down_proj"):
        ks = [k for k in keys if k[2] == p]
        if ks: per[p[:-5]] = sum(rows[s][k]["relg"] / base[k]["relg"] for k in ks) / len(ks)
    print(f"{s:6} {len(keys):3} {mean:8.4f} {100*(mean/ref-1):+15.1f}%  " + " ".join(f"{p} {v:.3f}" for p, v in per.items()))
