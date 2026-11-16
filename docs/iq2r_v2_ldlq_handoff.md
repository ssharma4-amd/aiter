# Handoff: LDLQ (Hessian error feedback) for the IQ2R-v2 recipe

Status 2026-11-16: measured on the study format (v2-B-rw), **not yet in this code**. This note is for porting it into
the shipped recipe (`aiter.quantization.iq2r.glm5.calibrate` / `compile`, format in `aiter/ops/iq2r_v2.py`) and its
native kernels. The full study write-up is in the research repo:
`research/glm53-2bit-quantization/08_targeted_error_reduction/EXPLAINER_ERROR_IMPORTANCE_DIRECTIONS_LDLQ.md`.
That folder also holds `PLAN.md`, `STEP1_SENSITIVITY_MAP.md` and `STEP2_IMPORTANCE_AND_CANDIDATES.md`.

## Why

The shipped encoder minimises an importance-weighted weight error. It uses only the per-column input energy (the
diagonal of the input second-moment matrix H = E[x x^T]). Measured on GLM-5.3 (layers 10-77):
- the per-column importance is almost flat, at stop/act decision moments too;
- the 2-bit error is about 6.5x MXFP4 (squared), spread evenly over the columns, so there is nothing to re-prioritise;
- with the full H the error is 10-30% below the diagonal estimate, because inputs move together. An encoder that
  aims for cancellation along those directions has room to gain.

LDLQ (QuIP, Chee et al. 2023; the same error feedback as GPTQ) minimises the full expected output error
`trace(dW H dW^T)`. It quantizes along the input dimension and pushes each step's error onto the not-yet-quantized
columns through the Cholesky factor of H^-1. Only the chosen codes change: same format, same kernels, nothing extra
at runtime.

## What was measured (study format, decode-to-FP8 serving)

Held-out relative output error, as a ratio to the normal encoder. 18 matrices (layers 10 / 40 / 70, experts
0 / 77 / 200, gate and up); Hessian fitted on one half of the data, scored on the other.

| Fit | corpus-v2 expert H | all positions of test texts | decision positions |
|---|---|---|---|
| per-expert H, damping 1.0 | 0.63 | 0.75 | 0.84 |
| layer-level H (shared by all experts), damping 1.0 | 0.73 | 0.54 | 0.49 |
| layer-level H + decision bucket (beta 1) | 0.76 | 0.58 | 0.32 |
| expert + layer blend, damping 1.0 | 0.60 | 0.58 | 0.55 |
| (MXFP4) | 0.16 | 0.16 | 0.16 |

Damping matters: at 0.01 LDLQ over-fits (in-sample 0.013 vs held-out 0.082 against shipped 0.060). Use 1.0.

Full model, re-encoded with the layer-level H + decision bucket (beta 1, damping 1.0):
- Hessians captured on corpus v2 only: 160K positions, 201 decision positions.
- gate/up re-encoded; down unchanged.
- Re-encode took 8 min on 8 GPUs.

| | AIME-60, 2 passes (T 1.0, top-p 0.95, 24K think + 8K forced answer) | ran out of thinking budget | median output tokens | TB2, 3 hard tasks x 3 |
|---|---|---|---|---|
| base (same format, no LDLQ) | 53, 54 | 30, 28 | 23.9K, 23.6K | 4/9 |
| **LDLQ** | **55, 55** | **23, 22** | **15.9K, 15.8K** | 4/9 (+1 setup error) |
| MXFP4 (earlier runs, TP8) | 55, 55, 55, 54 | 16-21 | ~12-14K | (6/6 in the full runs; 3x3 run pending) |

- LDLQ removes about 3/4 of the extra thinking (1.83x MXFP4's tokens -> 1.22x) and matches MXFP4's accuracy.
- The 3 Terminal-Bench tasks are not yet solved like MXFP4 solves them. **Agentic parity is not reached.**

## How to port it into this recipe

Format mapping (`aiter/ops/iq2r_v2.py`):
- E8M0 `d` per row and 256 inputs;
- a 4-bit level `l` per 32 weights: value scale `m[l % 4] * 2^(l / 4)`; the mantissa `m[c]` selects one of four
  pre-scaled E4M3 codebook copies;
- FP32 output gain per projection.

LDLQ keeps all scale choices fixed and re-chooses only the 9-bit codeword index and the 8 signs of each 8-weight
vector.

1. **Calibrate** (`glm5/calibrate.py`): add an optional pass that accumulates, per MoE layer, the layer-level input
   second-moment matrix:
   - `H_all = sum x x^T / n` over all scored positions (assistant tokens + the end-of-turn token after them);
   - `H_decision` over positions where FP8's top-1 next token is end of turn (`<|endoftext|>`, `<|user|>`,
     `<|observation|>`), `</think>` or `<tool_call>`.
   - x is the MoE input hidden state (the gate/up input of every expert, 6144).
   - Size: 75 layers x 2 x 151 MB fp32 = ~22 GB. Record the corpus hash as for the importance statistics.
   - Reference implementation: `EXP/targeted-error-20261111/tools/hess_layer_capture.py`.
2. **Compile** (`glm5/compile.py`): `--ldlq` for gate and up.
   - Normal encode first (scales, levels, codebook learning unchanged).
   - Then per matrix: `H = H_all/tr + beta * H_decision/tr`, `+ damp * mean(diag) * I` (beta 1.0, damp 1.0); Cholesky
     factor `U` of `H^-1`, computed once per layer.
   - Walk K in 8-column vectors (GPTQ lazy batches of 128 columns). For each row choose the nearest codeword of the
     row-group's codebook copy to `|target| / scale`, signs from the target. Error `E = (target - q) @ inv(U_jj)`,
     propagated with `U[j, later]`.
   - Gate and up are interleaved in this format: run LDLQ on each projection's rows (same H; the rows are
     independent).
   - Recompute the per-projection gains (`iq2r_v2_gains`, target 0.98) after LDLQ.
   - Down: unchanged. LDLQ on down did not help in the earlier screen, and down's input is per expert.
   - References:
     - study-format encoder: `EXP/targeted-error-20261111/tools/s2_encode_ldlq.py` (`ldlq_fast`);
     - an earlier LDLQ for *this* format, with codebook copies and levels: `EXP/iq2r-v2-20261102/final_study/screen.py`
       (`ldlq`, `v0_scales`; held-out gate/up 0.63x with per-expert H).
3. **Validate:**
   - pack/unpack/materialize tests;
   - the held-out check (fit on half A, score on half B) on a few layers before a full compile;
   - KL and stop/act shifts with `EXP/targeted-error-20261111/tools/sens_map.py`;
   - native-kernel serving;
   - AIME-60 and the agentic sets (TB2 3 hard tasks, SWE 15 disputed tasks, then full TB2 / Lite-150).

Do not capture the Hessians on evaluation text: corpus v2 only, so TB2 / SWE / AIME stay unseen.

## Open: why the 2-bit model still loses the hard agentic tasks

This is the current priority, ahead of the port. MXFP4 solves the three Terminal-Bench tasks (cobol-modernization,
write-compressor, torch-tensor-parallelism) that the 2-bit model, with or without LDLQ, mostly times out on.
Next in the study:
- MXFP4 at 3 attempts per task in the same setup;
- a trajectory comparison of where the 15 minutes go.
