"""usage: pavg.py DIR [DIR...] -> mean counter value per kernel dispatch (and mean duration ns) for each prof dir."""
import csv, sys, collections
rows = {}
for d in sys.argv[1:]:
    acc = collections.defaultdict(list); dur = {}
    for r in csv.DictReader(open(f"prof/{d}/run_counter_collection.csv")):
        acc[r["Counter_Name"]].append(float(r["Counter_Value"]))
        dur[r["Dispatch_Id"]] = int(r["End_Timestamp"]) - int(r["Start_Timestamp"])
    acc["duration_ns"] = list(dur.values())
    rows[d] = {k: sum(v) / len(v) for k, v in acc.items()}
names = sorted({k for r in rows.values() for k in r})
print(f"{'counter':34s}" + "".join(f"{d:>16s}" for d in rows))
for n in names:
    print(f"{n:34s}" + "".join(f"{rows[d].get(n, float('nan')):16.4g}" for d in rows))
