# Handoff: IQ2R-v2 recipe v3 (1,024-entry codebook + LDLQ + decision corpus)

Status 2026-11-17: measured on the study format (v2-B-rw) with decode-to-FP8 serving, **not yet in this code**. It
builds on [iq2r_v2_ldlq_handoff.md](iq2r_v2_ldlq_handoff.md), which covers LDLQ itself; read that first. The study
write-up is in the research repo: `research/glm53-2bit-quantization/08_targeted_error_reduction/`
(`PLAN_V3_SCALES_AND_DECISION_CORPUS.md` for this step, `PLAN.md` for the whole study).

It is the best 2-bit recipe so far on every measure we ran: held-out output error, AIME-60 (level with MXFP4), and a
3-task Terminal-Bench 2.0 subset. The full Terminal-Bench 2.0 has not been run on it yet.

## Recipe (three changes over the LDLQ handoff)

1. **1,024-entry learned codebook per matrix (10-bit index, +0.125 b/w).** Scales are unchanged.
   - Seed: antirez's 256-entry IQ2_XXS grid (levels 8 / 25 / 43) stacked at x1, x2^(1/4), x2^(1/2), x2^(3/4), rounded to
     E4M3. The 512-entry book was seeded with x1 and x sqrt(2).
   - Learn: the same importance-weighted k-means as the 512 book (4 rounds, 4 inner reassignments, a round kept only
     if the cost drops), seed `layer*100000 + expert*10 + proj`.
   - The learned 1,024 entries are 1,024 distinct shapes, not scaled copies of each other. The median nearest-neighbour
     distance between entries, after removing size, is 0.192 vs 0.189 for the 512 book.
   - The extra entries cover spiky vectors (a few large weights, several small ones), which are most of the weights.
     Smallest / largest value inside an entry, 512 book vs 1,024 book (layer 40, expert 0, gate):
     - median entry 0.208 vs 0.181;
     - entries below 0.15: 14 vs 225.
   - Shape-only error on 200K weight vectors of that matrix: 5.9% -> 4.8%.
2. **LDLQ on gate/up**, as in the LDLQ handoff: one layer-level Hessian per MoE layer,
   `H = H_all/tr(H_all) + beta * H_decision/tr(H_decision)`, beta 1.0, damping 1.0. Down is not changed.
3. **A separate, larger decision corpus for `H_decision`.** The corpus-v2 bucket had 201 positions; this one has
   4,972.
   - Text: 76 SWE-bench Verified agent trajectories of non-Lite instances (the last 8,192 tokens of each, 620K tokens).
     They are rendered with the serving chat template + bash tool. The 8 instances used as sensitivity evaluation text
     are excluded, so SWE-bench Lite, Terminal-Bench 2.0 and AIME stay unseen.
   - Positions: FP8 teacher-forced, scored positions (assistant tokens + the end-of-turn token after them) where FP8's
     top-1 next token is end of turn (`<|endoftext|>`, `<|user|>`, `<|observation|>`: 1,073), `</think>` (2,689) or
     `<tool_call>` (1,210).
   - `H_decision` = mean x x^T over those positions (x = MoE input, 6144), one per MoE layer. In the build it is the
     count-weighted mean of the two halves.
   - `H_all` stays the corpus-v2 capture, as do the importance statistics.

Everything is decided at encode time. Serving reads the same kind of record: scales, an index, 8 signs and a
per-matrix codebook.

## The decision corpus must cover every workload (measured 2026-11-18)

The v3 bucket comes from one source, SWE-bench Verified mini-swe-agent trajectories. Decision Hessians captured
separately from three sources (40 windows of 8,192 tokens each; layers 10/40/70) show that a bucket mostly helps its
own workload. Setup:
- 18 gate/up matrices; LDLQ fit on half A, scored on half B.
- Relative output error `tr(D H D^T)/tr(W H W^T)` at held-out decision positions of each source, as a ratio to no
  LDLQ.

| LDLQ fit | SWE-agent decisions | Terminal-agent decisions | Math decisions |
|---|---|---|---|
| `H_all` only | 0.623 | 0.629 | 0.693 |
| + SWE bucket | **0.296** | 0.572 | 0.692 |
| + terminal-agent bucket | 0.597 | **0.317** | 0.707 |
| + math bucket | 0.640 | 0.636 | **0.167** |
| **+ equal mix of the three** | **0.330** | **0.322** | **0.183** |
| + v3 bucket (SWE, 2,300 positions) | 0.295 | 0.565 | 0.689 |

- Each bucket cuts its own source's decision error ~2x more than `H_all` alone, and the others' little or not at
  all. The v3 bucket is in effect SWE-only.
- An equal mix (each source normalized by its trace, then averaged) comes within ~10% of each specialist, at about
  the same cost on ordinary tokens as any single bucket.
- Build the shipped decision corpus as a balanced mix of the served workloads: SWE-style agents, terminal agents,
  reasoning (`</think>`-heavy), code and chat.
- The terminal-agent bucket above used Terminal-Bench 2.0 text as a diagnostic only. A shipped bucket needs
  terminal-agent trajectories generated on non-evaluation tasks.
- terminus-2 never emits `<tool_call>`; its decisions are end of turn and `</think>` only.

Study write-up: `research/glm53-2bit-quantization/08_targeted_error_reduction/DECISION_CORPUS_SOURCE_STUDY.md`.

**Whole-model check (2026-11-19).** v3 was re-encoded with an SWE + math mixed bucket, keeping v3's books and scales
and redoing only the gate/up LDLQ picks. Teacher-forced against FP8 on held-out text:
- **AIME traces:** P(end of turn after the final answer) moves from 1/5.5 of FP8's (dlogP -1.71, 13 top-1 flips) to
  FP8's level (-0.03, 0 flips); 90% interval of the change +1.13 to +2.27.
- **SWE text:** KL and stop / tool-call probabilities are unchanged.

Mixing works end to end for the sources in the mix, and costs nothing on the others.

**The decision corpus (v1, 2026-11-19)** is `corpus/iq2r_v2_decision_corpus_v1.jsonl` in the private repository
`ssharma4-amd/glm53-iq2r-v2-private`, next to calibration corpus v2:
- sha256 `775778c8aba84c46c0ea55e4f1aa997e54f935bbae4bb17d61673c626fa8d0b2`;
- per-kind counts in `iq2r_v2_decision_corpus_v1_report.json`;
- scripts in `corpus/decision_v1_build/`.
- 431 windows, 2.13M tokens.
- No SWE-bench Lite, Terminal-Bench, AIME or GPQA text, and no Terminal-Bench-family dataset used for generation.

| Kind | Windows | Decision positions | Text |
|---|---|---|---|
| `swe_agent` | 76 | 4,972 | mini-swe-agent trajectories, SWE-bench Verified non-Lite |
| `math_reasoning` | 80 | 983 | FP8 reasoning traces, MATH-500 level 4-5 |
| `terminal_agent` | 155 | 707 | FP8 terminus-2 on LiteCoder-RL, Aider-Polyglot and SWE-bench Verified non-Lite (Harbor) |
| `chat_code_multilingual` | 120 | 91 | corpus v2's chat / code / multilingual rows, with the end-of-turn token after complete replies |

Recipe:
- one `H_decision` per kind (decision positions as above), each divided by its trace, averaged with equal weights;
- `H = H_all/tr(H_all) + 1.0 * mean_k(H_decision,k / tr)`.

Acceptance check (`decision_v1_build/srccheck.py --full`). Held-out error at each kind's decision positions, as a ratio
to no LDLQ:

| Fit | SWE | math | terminal | chat |
|---|---|---|---|---|
| `H_all` only | 0.63 | 0.70 | 0.63 | 0.57 |
| own-kind bucket | 0.31 | 0.12 | 0.26 | 0.49 |
| **equal mix of the 4 kinds** | **0.35** | **0.16** | **0.29** | **0.47** |

The full-model re-encode with this 4-kind bucket and its eval have not been run yet.
Tools: `EXP/targeted-error-20261111/tools/hdec_by_source.py` (capture), `tools/srccheck.py` (cross-check).

## Measurements

**Format screen.** 27 matrices (layers 10 / 40 / 70 x experts 0 / 77 / 200 x gate, up, down). The metric is held-out
output error `tr(dW H dW^T)/tr(W H W^T)` after the 0.98 gain, as a ratio to the 512-book encoder, both with LDLQ:

| Option | Extra b/w | Error ratio | Gain per extra bit |
|---|---|---|---|
| 6-bit scale per 32 | +0.0625 | 0.995 | 0.08 |
| 4-bit scale per 16 | +0.125 | 0.926 | 0.59 |
| 2-bit quarter-octave class per 8 | +0.25 | 0.878 | 0.49 |
| 2-bit odd multiplier per 8 | +0.25 | 0.877 | 0.49 |
| **1,024-entry codebook** | **+0.125** | **0.832** | **1.35** |

The limit is codebook coverage, not scale resolution. LDLQ + 1,024 book gives 0.52x the original encoder's error;
LDLQ alone gives 0.63x.

**Decision corpus, held-out check.** 18 gate/up matrices; ratio to the encoder without LDLQ:

| `H` used by LDLQ | Corpus-v2 expert H | Eval texts, all positions | Eval texts, decision positions | New corpus decisions, unseen half |
|---|---|---|---|---|
| `H_all` only | 0.646 | 0.595 | 0.648 | 0.626 |
| + corpus-v2 bucket, beta 1 | 0.691 | 0.636 | 0.528 | 0.507 |
| **+ new bucket, beta 1** | **0.687** | **0.627** | **0.498** | **0.305** |
| + new bucket, beta 5 | 0.755 | 0.692 | 0.510 | 0.290 |

Earlier full-model runs with the corpus-v2 bucket: beta 5 scored 4/9 on the TB2 gate and beta 20 scored 2/8. Use
beta 1.

**Full model.** Same serving and settings for every row: ATOM decode-to-FP8 at load, TP4, FP8 KV, max-num-seqs 16,
budget proxy 16K/4K. TB2 is Harbor terminus-2 at T 0.95 with leaderboard timeouts. AIME is T 1.0, top-p 0.95, 24K think +
8K forced answer.

AIME-60, 2 passes, paired with LDLQ beta 1 on the same day:

| Model | b/w (routed) | AIME-60 x2 | Median output tokens |
|---|---|---|---|
| 2-bit, no LDLQ | 2.31 | 53, 54 | 23.8K |
| LDLQ beta 1 | 2.31 | 55, 56 (earlier runs 55, 55, 55, 54) | 16.8K, 16.7K |
| **v3 (this recipe)** | **2.44** | **55, 56** | **14.5K, 15.7K** |
| MXFP4 A8W4 (same setup, earlier) | 4.25 | 54, 58 | ~12K |

**Terminal-Bench 2.0, 3-task subset.** These are not the full benchmark. The three tasks (cobol-modernization,
write-compressor, torch-tensor-parallelism) were picked as tasks that earlier 2-bit models failed while MXFP4 solved
them. They are a stress test for the 2-bit failure mode, not an estimate of the overall score.

All runs below are paired (both models served at the same time, same settings); solved / valid trials (setup failures
excluded):

| Run (11-17) | v3 | MXFP4 / other arm |
|---|---|---|
| s3dec, 3 x 3 | 8/9 | LDLQ beta 1: 2/8 |
| s3conf, 3 x 5 | 4/8 (7 setup failures) | MXFP4 12/15 |
| s3conf2, 3 x 2 | 4/6 | MXFP4 4/6 |
| clean, 3 x 5 | 10/15 | MXFP4 13/15 |
| clean-tb1, 3 x 5, v3 with a `</think>` logit offset +1.0 | 4/14 | MXFP4 9/15 |

- Earlier 2-bit models on the same subset: 2-bit without LDLQ 7/28, LDLQ beta 1 9/24 (mixed settings).
- One run of this gate varies by about +-4 tasks: MXFP4 scored 13/15 and 9/15 in two identical clean runs. Differences
  of a few tasks between models are within that range.
- The full benchmark (89 tasks) was last run with earlier 2-bit models: MXFP4 56, 60; 2-bit 49, 52. v3 has not been
  run on it.
- Analysis of the subset runs (`08_targeted_error_reduction/ANALYSIS_DECISION_VS_GENERAL.md`):
  - teacher-forced KL vs FP8 on these conversations: v3 0.080, MXFP4 0.048;
  - decode-to-FP8 serving runs at 52 tok/s vs MXFP4's 60;
  - the `</think>` logit offset did not change thinking length.

**Size.** The v3 store is 207 GiB of routed experts (2.44 b/w). The rest of the model stays FP8 (19.6 GiB, including
the 2.7 GiB shared expert), so ~227 GiB in total, against 684 GiB of FP8 routed experts in the source checkpoint.

Caveat: the codebook and the decision corpus were changed together, so their contributions are not separated.

## Porting to the shipped format

The shipped format (`aiter/ops/iq2r_v2.py`) keeps its scales: E8M0 `d` per row and 256 inputs, a 4-bit level per 32
weights selecting one of four pre-scaled codebook copies, and FP32 gains.

**The shipped scaling is the right base, and the 1,024 gain carries over to it (measured 2026-11-17).** Same 27
matrices and importance as the format screen. Each format learns its own codebook per matrix with its own learner; no
LDLQ. Held-out output error after the 0.98 gain, as a ratio to today's shipped format (c4e4q, 512 entries):

| Scaling | Codebook | b/w | Error ratio | LDS per gate/up codebook |
|---|---|---|---|---|
| shipped c4e4q (today) | 512 | 2.28 | 1.000 | 16 KiB |
| study (FP16 `d`, `d*(2*ls+1)`) | 512 | 2.31 | 1.001 | - |
| shipped c8e2 | 512 | 2.28 | 0.978 | 32 KiB |
| study (FP16 `d`, `d*(2*ls+1)`) | 1,024 | 2.44 | 0.839 | - |
| **shipped c4e4q** | **1,024** | **2.41** | **0.822** | **32 KiB** |
| shipped c8e2 | 1,024 | 2.41 | 0.802 | 64 KiB |

- The study format's FP16 `d` and odd multipliers give no gain over the shipped scaling at 512 entries, and cost
  0.03 b/w more.
- With 1,024 entries the shipped c4e4q scaling is slightly better than the study format, at lower b/w. Port the
  codebook size, not the study scales.
- The shipped learner was seeded with the IQ2_XXS grid x 2^(j/4), j = 0..3 (native units).
- Tool: `EXP/targeted-error-20261111/tools/scale_compare.py --book1024`. It loads `aiter/ops/iq2r_v2.py` from this
  branch and swaps only `iq2r_v2_seed_codebook`.

The changes:

- **Codebook:** 1,024 entries instead of 512.
  - `iq2r_v2_learn_codebook` already works with any codebook size; only `iq2r_v2_seed_codebook` (512) and the
    512-specific constants (`IQ2R_CODEBOOK_ENTRIES`, the format name, the auxiliary layout) need to change.
  - Seed: the IQ2_XXS grid at 2^(j/4), j = 0..3.
  - Check that the learned entries times the copy multipliers stay inside `IQ2R_V2_CODEBOOK_MIN_BYTE..MAX_BYTE`.
- **Records:** 10-bit indices. Each 8-weight vector grows by 1 bit (+0.125 b/w). The study format's packing
  (`v2_codec10.py`: `[fp16 d | 8 x 4-bit ls | 32 x 10-bit indices LSB-first | 32 sign bytes]`, 78 bytes per 256
  weights) is only a reference; the native record layout is the kernel owner's choice.
- **Kernels:**
  - the LDS codebook doubles: four copies x 1,024 x 8 E4M3 = 32 KiB per gate/up codebook instead of 16 KiB;
  - the index unpack moves from 9 to 10 bits;
  - the MFMA stream and scale handling (E8M0 `d`, level exponent, class -> copy) are unchanged.
- **Calibrate:**
  - the layer-level `H_all` (LDLQ handoff, step 1);
  - a `H_decision` capture over the decision corpus `iq2r_v2_decision_corpus_v1.jsonl`, one capture per `kind`
    (see "The decision corpus must cover every workload"), each normalized by its trace and then averaged (~11 GB at
    fp32 per kind for 75 layers). The reference capture is `decision_v1_build/hdec_by_source.py`.
  - Keep that corpus separate from the calibration corpus and away from evaluation text.
- **Compile:**
  - learn the 1,024 book;
  - normal encode;
  - LDLQ on gate/up with `H_all/tr + 1.0 * H_decision/tr`, damping 1.0;
  - recompute the gains.
- **Validate:** as in the LDLQ handoff, plus a bit-exact round trip of every matrix. The study build checks it for
  all 172,800 matrices.

## Reference implementations (experiment host, `EXP = ~/2-bit-Upstream/GLM5.3_Integration/experiments`)

| What | Where |
|---|---|
| Full-model build (book + encode + LDLQ + pack + bit-exact check; 236 min on 8 GPUs) | `EXP/targeted-error-20261111/tools/s2_encode_s3.py` |
| 1,024 book seed + learning | `v3_screen.py: book1024`; `EXP/iq2r-v2-20261102/learn_codebook.py: learn` |
| LDLQ (`layer_factor`, `ldlq_fast`; `TE_BETA`, `TE_HDEC`) | `EXP/targeted-error-20261111/tools/s2_encode_ldlq.py` |
| 10-bit packing / decode, tests | `tools/v2_codec10.py`, `tools/v2_on_atom10/`, `tools/test_codec10.py` |
| Decision corpus capture | `tools/decision_corpus_capture.py` (output `/models/zai-org/glm53-te-hess-dec`) |
| Per-source decision Hessians and cross-source check | `tools/hdec_by_source.py`, `tools/srccheck.py` |
| Held-out checks | `tools/v3_screen.py`, `tools/deccorpus_check.py` |
| Study vs shipped scaling, 1,024 entries in the shipped format | `tools/scale_compare.py` (`--book1024`) |
| Paired evaluation | `tools/run_eval_pair.sh`, `tools/tb2_tally.py` |
| Store | `/models/zai-org/GLM-5.3-IQ2Rv2-te-s3ldlq-dec` |
