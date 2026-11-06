# IQ2R-v2 GLM-5.3 MoE kernels (gfx950)

How `iq2r_glm53_moe_out` runs IQ2R-v2 weights, why it is built this way, and what was measured. The format and
recipe are in [iq2r_glm53.md](iq2r_glm53.md).

## What runs

IQ2R-v2 uses the IQ2R (v1) launch structure and dispatch table: sort/quantize, gate+up GEMM with SwiGLU and MXFP8
requantization, down GEMM, weighted top-k reduction. The kernels detect the format from the packed auxiliary
size. IQ2R behaviour is unchanged:
- route9 and the nobarrier gate are upstream's kernel bodies;
- the other kernels take IQ2R-v2 paths only under `Classes > 0`;
- the one shared change on the IQ2R path is the study's cheaper sign spread in the decode helpers
  (`glm53_sign_pair` / `glm53_sign_word`), which produces the same bits.

**256 experts, top-8.** The IQ2R-v2 checkpoint keeps the shared expert in FP8 outside the MoE, so the MoE gets
`[256, ...]` stacks with `[tokens, 8]` ids and weights. The kernels read top-8 routing directly; nothing is padded
on the host:

- Routes keep nine slots per token (`token * 9 + k`). With top-8, slot 8 is empty: the sort kernels
  (`TopK` template, `glm53_route_id`) read `[tokens, 8]` ids and put slot 8 into the invalid bucket.
- They emit no tasks for those slots (see below). The skipped tasks unbalanced the gate's work split;
  dropping them was worth 12-22% of the whole MoE at M512-1024.
- The reductions read `[tokens, 8]` weights; the route9 down runs 8 route waves (2 per SIMD) instead of 9.
- 257-expert / top-9 stacks still run (IQ2R, and IQ2R-v2 with a fused shared expert).

**Invalid routes and the shared expert in the task lists.**

- **The sort:** it groups routes by expert, in the order routed experts 0-255, the shared expert 256 (fused
  stacks only), then invalid routes (id -1: dropped or masked routes, and the empty ninth slot with top-8). From
  these groups it builds the gate and down task lists (`{first sorted row, rows, expert}`) and writes their
  count, which the GEMM kernels loop over.
- **Shared expert:**
  - IQ2R (257-expert stack): the shared expert is real work, every token with weight 1, and keeps its tasks.
  - IQ2R-v2: it is not in the MoE at all (it runs as an FP8 MLP outside), so id 256 never occurs and has no
    tasks.
- **Invalid routes:**
  - Top-9: they still get tasks marked expert -1, which the kernels skip, as IQ2R always did.
  - Top-8: the sorts emit no tasks for them (`glm53_sort_routes<TopK>`, `glm53_sort_expert_prefix<TopK>`).
  - Either way they get `scatter = -1`, so the reductions skip them.

**IQ2R-v2 kernel changes.** Ported from the kernel study (`fork/glm-5.3-iq2r-v2-kernel-exp`, its
`docs/iq2r_v2_kernel_findings.md`) to the one-codebook format, with every experiment switch resolved to its
fastest value and the other variants removed:

| Stage | Kernel | What it does differently from IQ2R |
|---|---|---|
| M1 gate | `glm53_gate_m1_kernel` | Loads the 4 KB book (copy 0) and forms copies 1-3 in LDS (GG=2); non-temporal weight loads |
| Decode gate | `glm53_gate_decode_nobarrier_v2_kernel` | Per-XCD item ranges; book formed in LDS from copy 0 after the item's loads are issued, then one barrier with the loads still in flight; buffer-descriptor K loop with SGPR offsets (ABUF); MFMA scale `op_sel` per atom (OPSEL); carried levels; non-temporal weight loads; 2 workgroups/CU |
| Prefill gate | `glm53_gate_prefill_kernel` | Book formed from copy 0; launch bounds ask for 2 waves per SIMD |
| M1-M4 down | `glm53_down_route9_v2_kernel` | Each route wave forms its own book in LDS ("gen7"), activation row by DMA, scalar scales; 2 triplets per CTA at M4; 8 waves for top-8 |
| Decode down | `glm53_down_tp4_kernel`, `glm53_down_tp8_ordered_kernel` | K-outer triplets (S8KO/T8KO), levels and scale bytes once per K tile (T8LV), 32-bit offsets (A32); 16 KB copies staged by DMA |
| Prefill down | `glm53_down_prefill_tp4/tp8_kernel` | Book formed from copy 0; levels and scale bytes once per K tile; TP4 items cover 768 columns (TP8 384) |

**Copy formation.** The packed auxiliary stores all four 4 KB codebook copies `E4M3(codebook * 2^(c/4))`
(16 KB). Most v2 kernels read only copy 0 and form copies 1-3 in LDS with integer byte adds
(`iq2r_v2_copy_words`): for a normal E4M3 byte `b`, copy c's byte is `b + delta_c[b & 7]`. This cuts the codebook
traffic per expert switch from 16 KB to 4 KB. It is exact only for bytes 0x08..0x78 (2^-6 to 256).
`iq2r_v2_learn_codebook` now clamps codebook entries to at least 2^-6. `iq2r_glm53_pack_v2` rejects any
codebook byte outside the range (`iq2r_v2_check_codebook`).

**Tuned table.** `aiter/configs/iq2r_glm53_tuned.csv` has its own rows for 256 experts / top-8
(`iq2r_glm53_config(tokens, intermediate, experts)`). The tuner (`iq2r_glm53_tune.py`) builds IQ2R-v2 synthetic
layers for them.

## Checkpoint compatibility

- **Packed layout:** unchanged.
- **Encoder output bytes:** change only where the old learner would have produced a codebook entry below
  2^-6, which is now clamped to 2^-6.
  - Such entries are zero or near-zero weights. The 36 real GLM-5.3 books in the kernel study spanned
    0x21..0x4f, so most layers are unaffected.
  - A synthetic matrix with zeroed and tiny weights produced a 0x05 byte before the change.
  - A checkpoint built before this change is usable only if every codebook byte is in 0x08..0x78. Otherwise
    `iq2r_glm53_pack_v2` raises, and the affected layers must be re-encoded.

## Correctness

- **Tests:** `op_tests/test_iq2r_glm53.py` and `op_tests/test_iq2r_glm53_v2.py` pass on MI355X. New tests cover:
  - top-8 against the dense reference;
  - top-8 equal to top-9 with the ninth route dropped;
  - routing/stack mismatches;
  - the formation formula on every byte in 0x08..0x78 (and that 0x07 breaks it);
  - the learner's clamp on zero and tiny weights;
  - the packer's rejection of out-of-range books.
- **Bit-exact against the pre-port kernels (OPSEL, GG=2, ABUF and the rest):**
  - Reference: the branch head before the port (`b5c7638cd`: staged copies, no OPSEL/GG/ABUF).
  - Ported: the kernels, on the same synthetic 257-expert IQ2R-v2 layer, with codebooks of random bytes over
    the whole 0x08..0x78 range.
  - Compared bit for bit (`torch.equal`): the gate stage's FP8 output and scales (valid routes) and the final
    output.
  - Cases: every valid explicit launch config, at TP4 and TP8, M = 1, 2, 3, 4, 5, 8, 16, 48, 64, 100, 256, 512,
    1024, 2048, 4096. Result: 188 of 188 runs identical.
- **Top-8 vs top-9:** native top-8 on a 256-expert stack against top-9 with route 8 dropped: identical in the
  same 188 runs.

## Measurements

One MI355X (gfx950, 256 CUs), ROCm 7.2.4, PyTorch 2.10. Whole MoE (`iq2r_glm53_moe_out`), the tuner's method:

- CUDA-graph replay;
- every call reads its own copy of the layer, at least 1 GiB of expert weights per replay, so experts come
  from HBM;
- 50 warmup and 500 timed calls;
- synthetic weights; the best of several rounds.

Baseline: IQ2R v1 from upstream `main` (no IQ2R, tuner or JIT changes between `796ff6a35` and the commit timed)
with its tuned CSV, 257 experts / top-9. IQ2R-v2: this branch, 256 experts / top-8, retuned. Both arms ran on
the same GPU of the same node, alternating.

Whole MoE, µs: best of 6 (v1) / 6 (v2) rounds; median in brackets.

| M | TP4 v1 | TP4 v2 | Δ | TP8 v1 | TP8 v2 | Δ |
|---|---|---|---|---|---|---|
| 1 | 22.62 (22.68) | 17.71 (17.75) | -21.7% | 18.97 (19.02) | 16.36 (16.41) | -13.8% |
| 2 | 25.60 (25.83) | 19.88 (20.01) | -22.3% | 20.01 (20.07) | 17.80 (17.88) | -11.0% |
| 4 | 36.53 (36.73) | 26.70 (26.84) | -26.9% | 23.38 (23.47) | 20.25 (20.35) | -13.4% |
| 8 | 52.09 (52.31) | 47.30 (47.50) | -9.2% | 35.40 (35.50) | 32.66 (32.85) | -7.7% |
| 16 | 75.64 (76.44) | 69.94 (70.92) | -7.5% | 50.82 (50.99) | 45.23 (45.66) | -11.0% |
| 64 | 143.40 (144.82) | 135.22 (135.53) | -5.7% | 84.22 (84.84) | 78.96 (79.04) | -6.2% |
| 256 | 187.37 (195.75) | 169.81 (170.44) | -9.4% | 109.84 (111.44) | 102.64 (102.78) | -6.6% |
| 1024 | 292.94 (296.23) | 254.31 (255.25) | -13.2% | 210.55 (213.81) | 164.15 (164.44) | -22.0% |
| 4096 | 731.81 (736.20) | 699.14 (700.45) | -4.5% | 502.53 (505.62) | 452.72 (455.24) | -9.9% |

The other tuned token counts, M = 32, 128, 512, 2048, are also faster (-5.7% to -15.7%). The smallest margin
over all 26 tuned shapes is -4.5% (TP4 M4096); on medians, -4.9%. Raw rounds: 6 per arm, alternating v1 / v2 in
blocks of 3, one GPU, branch head `d35a140b1`, upstream `8b114774f`.

**What moved the numbers.** Each step below is a same-node A/B of the whole MoE against v1, at v1's launch
choices unless noted.

| Step | Effect |
|---|---|
| Port of the study kernels, top-8 | Decode M1-256 -21% to +0.6%; prefill (M >= 512) +5..+41% |
| Prefill gate at 2 waves/SIMD | Left free, the compiler used 264 registers (1 wave/SIMD). TP4 M1024 gate 252 → 161 µs; prefill +3..+5% |
| Formed copies in the prefill downs | TP4 M1024 MoE +11% → +5% vs v1. Slower in the decode downs (+1 point at M4-16), so those still stage |
| Decode gate forms its book after the item's loads | About 1 point at decode M |
| Levels and scales once per K tile in the prefill downs | TP4 M512-4096: +2.4..+5.3% → -1.4..+0.8% |
| 768-column TP4 prefill-down items | About 1 more point at TP4 prefill. At TP8 it measured slower (+0.6..+0.7%) and was reverted |
| Retune for 256/top-8 | e.g. TP4 M4 picks route9 (27-28 µs vs 35) |
| No barrier wait on the gate's own loads | TP8 M512 gate 86 → 84 µs |
| No tasks for empty slots | TP8 M512-1024 -12..-22%, TP4 M512-1024 -12..-13% |

**Measured and dropped:**
- DBC: a wave's d words loaded once per item and broadcast by permlane.
  - Slower at M2-64 (+1..+3 points).
  - Kept as a per-shape choice it was never selected once the barrier fix was in.
- A gate variant with default-policy (temporal) weight loads: never selected by the tuner.
- Formed copies in the decode downs.
- 768-column TP8 prefill-down items.

## Not measured

- Real checkpoint weights (all numbers use synthetic layers); end-to-end serving; quality on the native
  kernels.
- The FP8 shared-expert MLP that IQ2R-v2 runs outside the MoE. The table compares the MoE only, so v1's
  column includes the shared expert (fused as route 9) and v2's does not.
- Variation across GPUs and nodes. All numbers are one GPU per session; run-to-run noise was about ±1%, with
  occasional v1 outliers (TP8 M512 at 169-171 µs in two rounds against ~151 µs in the others).
- Token counts between the tuned rows (they use the next tuned row up), M > 4096, power.
- IQ2R-v2 on 257-expert / top-9 stacks for speed (correctness only).
- NTT (non-temporal triplet loads in the decode downs), a per-shape win in the study, was not ported.
