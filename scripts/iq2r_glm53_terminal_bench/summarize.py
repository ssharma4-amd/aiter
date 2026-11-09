"""Summarize one or more Harbor job directories (the batches of one run_tb2.sh run).

usage: python3 summarize.py JOB_DIR [JOB_DIR ...]
Prints one line per trial (task, reward, exception, agent minutes, tokens) and a last line with the totals.
Score = resolved / all trials; trials that raised count as unresolved (Terminal-Bench leaderboard convention).
"""

import glob
import json
import sys
from datetime import datetime
from pathlib import Path


def minutes(step):
    if not step or not step.get("started_at") or not step.get("finished_at"):
        return float("nan")
    t = [
        datetime.fromisoformat(step[k].replace("Z", ""))
        for k in ("started_at", "finished_at")
    ]
    return (t[1] - t[0]).total_seconds() / 60


rows = []
for job in sys.argv[1:]:
    for f in sorted(glob.glob(f"{job}/*/result.json")):
        r = json.loads(Path(f).read_text())
        reward = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        exc = (r.get("exception_info") or {}).get("exception_type")
        a = r.get("agent_result") or {}
        rows.append(
            (
                r["task_name"],
                reward,
                exc,
                minutes(r.get("agent_execution")),
                a.get("n_input_tokens"),
                a.get("n_output_tokens"),
            )
        )
for t, rew, exc, m, i, o in sorted(rows):
    print(f"{t:40s} reward {rew}  {exc or '-':28s} agent {m:6.1f} min  in {i}  out {o}")
n = len(rows)
solved = sum(1 for r in rows if r[1] == 1.0)
errs = sum(1 for r in rows if r[2])
print(
    f"TOTAL {solved}/{n} resolved ({100 * solved / max(n, 1):.1f}%), {errs} trials with exceptions"
)
