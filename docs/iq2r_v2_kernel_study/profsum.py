"""Mean per dispatch of each counter in prof/NAME (rocprofv3 csv). usage: profsum.py NAME..."""
import glob, sys, pandas as pd
for n in sys.argv[1:]:
    fs = glob.glob(f"prof/{n}/**/*counter_collection.csv", recursive=True)
    d = pd.concat(pd.read_csv(f) for f in fs)
    g = d.groupby(["Dispatch_Id", "Counter_Name"]).Counter_Value.sum().unstack()
    print(n, f"dispatches={len(g)}", " ".join(f"{c}={g[c].mean():.4g}" for c in g.columns))
