# Handoff: IQ2R-v2 recipe v3 (1,024-entry codebook + LDLQ + decision corpus)

Status 2026-11-17: measured on the study format (v2-B-rw) with decode-to-FP8 serving, **not yet in this code**. It
builds on [iq2r_v2_ldlq_handoff.md](iq2r_v2_ldlq_handoff.md), which covers LDLQ itself; read that first. The study
write-up is in the research repo: `research/glm53-2bit-quantization/08_targeted_error_reduction/`
(`PLAN_V3_SCALES_AND_DECISION_CORPUS.md` for this step, `PLAN.md` for the whole study).

This is the first 2-bit recipe that solves the three hard Terminal-Bench 2.0 tasks at close to MXFP4's rate.

## Recipe (three changes over the LDLQ handoff)

1. **1,024-entry learned codebook per matrix (10-bit index, +0.125 b/w).** Scales are unchanged.
   - Seed: antirez's 256-entry IQ2_XXS grid (levels 8 / 25 / 43) stacked at x1, x2^(1/4), x2^(1/2), x2^(3/4), rounded to
     E4M3. The 512-entry book was seeded with x1 and x sqrt(2).
   - Learn: the same importance-weighted k-means as the 512 book (4 rounds, 4 inner reassignments, a round kept only
     if the cost drops), seed `layer*100000 + expert*10 + proj`.
   - The learned book reaches shapes the 512 book cannot. On layer 40, expert 0, gate it has 38 distinct magnitudes
     down to 0.69 raw (512 book: 30, smallest 5.0), so vectors with one or two near-zero weights next to large ones
     no longer round the small ones up. Neighbouring entries are no closer together (median nearest-neighbour
     distance 0.192 vs 0.189).
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

| Model | b/w (routed) | cobol | write-compressor | torch-TP | TB2 3 hard tasks | AIME-60 x2 | Median output tokens |
|---|---|---|---|---|---|---|---|
| 2-bit, no LDLQ | 2.31 | 0/10 | 3/9 | 4/9 | 7/28 | 53, 54 | 23.8K |
| LDLQ beta 1 (all runs) | 2.31 | 3/8 | 0/7 | 6/9 | 9/24 | 55, 55, 55, 54, 55, 56 | 15.8-18.6K |
| **v3 (this recipe)** | **2.44** | **3/3** | **2/3** | **3/3** | **8/9** | **55, 56** | **14.5K, 15.7K** |
| MXFP4 A8W4 | 4.25 | 5/5 | 4/5 | 4/5 | 13/15 | 54, 58 | ~12K |

- The v3 and LDLQ rows marked 55, 56 were a paired run on the same day (LDLQ scored 2/8 on TB2 in that pair).
- The v3 store is 207 GiB of routed experts.
- Caveats: 3 trials per task, and the codebook and decision corpus were changed together, so the gain is not yet
  attributed. Next: a same-day paired run against MXFP4 (3 tasks x 5), then the two changes one at a time.

## Porting to the shipped format

The shipped format (`aiter/ops/iq2r_v2.py`) keeps its scales: E8M0 `d` per row and 256 inputs, a 4-bit level per 32
weights selecting one of four pre-scaled codebook copies, and FP32 gains. The changes:

- **Codebook:** 1,024 entries instead of 512, seeded with the four quarter-octave grid copies above. Check that the
  learned entries times the copy multipliers stay inside `IQ2R_V2_CODEBOOK_MIN_BYTE..MAX_BYTE`. Note that the learned
  book uses smaller magnitudes than the 512 book.
- **Records:** 10-bit indices. Each 8-weight vector grows by 1 bit (+0.125 b/w). The study format's packing
  (`v2_codec10.py`: `[fp16 d | 8 x 4-bit ls | 32 x 10-bit indices LSB-first | 32 sign bytes]`, 78 bytes per 256
  weights) is only a reference; the native record layout is the kernel owner's choice.
- **Kernels:** the LDS codebook doubles (four copies x 1,024 x 8 E4M3 = 32 KiB per gate/up pair instead of 16 KiB),
  and the index unpack moves from 9 to 10 bits. The MFMA stream and scale handling are unchanged.
- **Calibrate:**
  - the layer-level `H_all` (LDLQ handoff, step 1);
  - a `H_decision` capture over the decision corpus described above (~11 GB at fp32 for 75 layers).
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
| Held-out checks | `tools/v3_screen.py`, `tools/deccorpus_check.py` |
| Paired evaluation | `tools/run_eval_pair.sh`, `tools/tb2_tally.py` |
| Store | `/models/zai-org/GLM-5.3-IQ2Rv2-te-s3ldlq-dec` |
