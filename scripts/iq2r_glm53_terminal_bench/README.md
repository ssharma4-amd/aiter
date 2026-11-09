# Terminal-Bench 2.0 on GLM-5.3 IQ2R-v2 (ATOM, native kernels)

Runs [Terminal-Bench 2.0](https://www.tbench.ai/) (89 tasks, via Harbor's `terminus-2` agent) against GLM-5.3
served by ATOM with the native IQ2R-v2 MoE kernels of this branch.

```
scripts/iq2r_glm53_terminal_bench/run_tb2.sh                          # all 89 tasks, IQ2R-v2 (arm fp8)
ARM=mxfp4 STORE=/models/zai-org/GLM-5.3 scripts/iq2r_glm53_terminal_bench/run_tb2.sh   # MXFP4 baseline
TASKS="fix-git gcode-to-text" scripts/iq2r_glm53_terminal_bench/run_tb2.sh             # a subset
SERVER=stub TASKS="fix-git" scripts/iq2r_glm53_terminal_bench/run_tb2.sh              # plumbing test, no GPU
```

Needs one MI355X node with 8 GPUs (TP8) and a local Docker engine for the task sandboxes. Fleet dev sessions have no
Docker engine, so the harness cannot run there. All settings are environment variables, listed at the top of
`run_tb2.sh`.

## What runs

| Piece | File | Setting |
|---|---|---|
| Server | `serve.sh` | ATOM, arm `fp8`: experts as stored (native IQ2R-v2), shared expert stored block FP8; non-expert layers online PTPC FP8; TP8, FP8 KV, prefix cache, `--max-num-seqs 16`, `--max-model-len 262144` |
| Server gate | `check_server.py` | no load errors in the log, and 3 sanity prompts answered correctly |
| Thinking budget | `budget_proxy.py` | 16K thinking tokens, then 4K to answer (the SWE-bench runs' setting) |
| Agent | Harbor `terminus-2` | temperature 0.95 (GLM-5.3 model card), leaderboard timeouts |
| Results | `summarize.py` | per task: reward, exception, agent minutes, tokens; total = resolved / all trials (errors count as unresolved) |
| Test server | `stub_server.py` | CPU stand-in that returns a valid terminus-2 action |

The server and proxy are the ones used for the GLM-5.3 IQ2R-v2 AIME and SWE-bench runs. The defaults use ports
8011/9011.

## Files outside the repo

Everything a run creates goes under `WORK` (default `~/iq2r_glm53_tb2`):
- `venv/` (`harbor==0.25.0`, installed on first run);
- `terminal-bench/` (the dataset, downloaded on first run);
- `jobs/<JOB>-b<N>/` (one Harbor job per batch);
- `logs/` (server, server check, proxy JSONL, Harbor, summary).

**Disk:** the task images are prebuilt on Docker Hub (`tb2_images.txt`, 48 GB compressed for all 89), and Harbor
does not remove images it pulled. So the run goes in batches of `BATCH` (30) tasks, removes the images each batch
pulled (images present before the run are kept), and stops if the Docker root has less than `MIN_FREE_GB` (40)
free. Rerunning with the same `JOB` skips finished batches.

## Validation so far (CPU only)

- Harbor's `oracle` agent (reference solutions, no model): 3/3 tasks solved on the local Docker engine.
- `SERVER=stub`: terminus-2 -> proxy -> stub server on 4 tasks in 2 batches. 0 exceptions, the commands ran in
  the sandboxes, the verifiers ran, and the pulled image was removed after its batch.
- Not yet run: the real model. That includes the proxy's forced-answer path (when thinking hits the budget,
  shared with the SWE-bench runs) and the wall time of a full run.
