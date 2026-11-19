# GLM-5.3 IQ2R-v3: 1,024-entry codebook on the IQ2R-v2 kernels

IQ2R-v3 keeps the IQ2R-v2 scaling (E8M0 `d`, 4-bit level per 32 weights, four
pre-scaled codebook copies, FP32 gains) and grows the codebook from 512 to
1,024 entries: 10-bit indices, 2.41 instead of 2.28 bits per weight. This file
records the v2 baseline the v3 kernels are measured against and the on-chip
memory budget of a 1,024-entry book in each kernel.

## Setup

One host with 8 MI350X (gfx950, 256 CUs); everything ran on GPUs 4-7.
Image `glm53-ab:aiter-6dbf9e876-atom-17aca439a` (ROCm 7.2, PyTorch 2.11),
aiter `8f01badb8` and ATOM `e05b98f5` (this branch's base), flydsl 0.5.0.
The tuned CSV's `us` column was measured on an MI355X; the same kernels are
5-20% slower on this MI350X, so every comparison below uses times from this
host.

## v2 MoE baseline

`iq2r_glm53_moe_out` (sort through reduction) at the tuned configs, timed by
the tuner (`iq2r_glm53_tune.py --run_config aiter/configs/iq2r_glm53_tuned.csv`):
CUDA-graph replay, each call on its own copy of the layer, at least 1 GiB of
expert weights per replay, synthetic weights. 16 rounds on one GPU; µs, best
round (median). The target columns apply the v3 criteria to the median:
at most 1.07x v2 for M <= 64, 1.05x above, never above 1.10x.

| M | TP4 v2 | TP4 v3 target | TP8 v2 | TP8 v3 target |
|---|---|---|---|---|
| 1 | 18.42 (18.54) | <= 19.8 | 17.15 (17.20) | <= 18.4 |
| 2 | 20.44 (20.88) | <= 22.3 | 18.39 (18.52) | <= 19.8 |
| 4 | 31.63 (32.27) | <= 34.5 | 20.99 (21.45) | <= 23.0 |
| 8 | 55.74 (56.16) | <= 60.1 | 36.18 (36.67) | <= 39.2 |
| 16 | 83.30 (83.95) | <= 89.8 | 50.85 (51.76) | <= 55.4 |
| 32 | 124.12 (124.85) | <= 133.6 | 71.60 (74.70) | <= 79.9 |
| 64 | 170.52 (171.46) | <= 183.5 | 91.60 (92.09) | <= 98.5 |
| 128 | 193.32 (194.78) | <= 204.5 | 105.33 (107.25) | <= 112.6 |
| 256 | 206.45 (208.61) | <= 219.0 | 119.09 (120.36) | <= 126.4 |
| 512 | 264.67 (267.66) | <= 281.0 | 152.05 (153.62) | <= 161.3 |
| 1024 | 315.35 (316.43) | <= 332.2 | 195.39 (196.54) | <= 206.4 |
| 2048 | 495.04 (501.15) | <= 526.2 | 320.89 (322.91) | <= 339.1 |
| 4096 | 836.43 (842.89) | <= 885.0 | 538.66 (539.82) | <= 566.8 |

Round-to-round spread is about 1% (up to 4% at TP8 M32).

Where the time goes (torch profiler, eager, rotated weights; µs per call):

| M | TP4 sort | TP4 gate | TP4 down | TP8 sort | TP8 gate | TP8 down |
|---|---|---|---|---|---|---|
| 1 | 2.6 | 10.6 (m1) | 7.5 (route9) | 2.7 | 10.3 (m1) | 6.4 (route9) |
| 4 | 3.3 | 17.4 | 12.9 (route9) | 3.3 | 11.8 | 8.2 (route9) |
| 16 | 4.6 | 45.0 | 28.0 + 4.7 | 3.8 | 26.7 | 18.3 + 3.7 |
| 64 | 5.0 | 94.8 | 59.0 + 4.6 | 5.3 | 48.9 | 30.8 + 4.8 |
| 256 | 10.3 | 110.8 | 67.2 + 5.9 | 10.3 | 57.3 | 39.6 + 6.6 |
| 1024 | 5.5 + 37.8 | 158.8 | 108.6 | 5.7 + 37.5 | 84.6 | 82.1 |
| 4096 | 15.7 + 94.0 | 475.6 | 359.2 | 15.5 + 92.5 | 246.2 | 246.3 |

(Down above M4 is the down GEMM plus the route reduction; prefill sort is
quantize plus sort.)

## Serving baseline: native IQ2R-v2 against MXFP4, standard kernels

ATOM `benchmark_serving`, random prompts, 1024 input and 1024 output tokens
(`--random-range-ratio 1`, `--ignore-eos`), 10 x concurrency prompts after
2 x concurrency warmups. One server at a time, FP8 KV cache,
`--max-model-len 262144`, `--max-num-seqs 256`, CUDA graphs for batch 1-256,
non-expert layers PTPC FP8 in both arms. Standard kernels only: ATOM's GLM
MonoKernel (a fused decode for batch 1-2 that only supports MXFP4 experts) is
off (`ATOM_MONO_ENABLE=0`); IQ2R-v2 never uses it.

- IQ2R-v2: `GLM-5.3-IQ2Rv2-final`, native kernels, shared expert block FP8 as stored.
- MXFP4: the `GLM-5.3` FP8 checkpoint with routed and shared experts quantized to
  MXFP4 at load, **A4W4** (FP4 activations).

TP4 (GPUs 4-7), output tokens/s (mean TPOT, mean TTFT):

| Concurrency | MXFP4 | IQ2R-v2 | IQ2R-v2 / MXFP4 |
|---|---|---|---|
| 1 | 88 (11.3 ms, 0.10 s) | 80 (12.5 ms, 0.11 s) | 0.91 |
| 2 | 181 (10.9 ms, 0.16 s) | 150 (12.6 ms, 0.78 s) | 0.83 |
| 4 | 318 (11.8 ms, 0.87 s) | 288 (13.2 ms, 0.72 s) | 0.91 |
| 8 | 530 (13.1 ms, 2.04 s) | 496 (15.1 ms, 1.09 s) | 0.93 |
| 16 | 974 (15.8 ms, 0.65 s) | 910 (16.8 ms, 0.83 s) | 0.93 |
| 32 | 1627 (19.1 ms, 0.60 s) | 1510 (20.3 ms, 0.90 s) | 0.93 |
| 64 | 2504 (24.7 ms, 0.93 s) | 2372 (25.9 ms, 1.16 s) | 0.95 |
| 128 | 3859 (31.6 ms, 1.63 s) | 3650 (33.0 ms, 2.11 s) | 0.95 |
| 256 | 5784 (40.8 ms, 3.61 s) | 5495 (43.0 ms, 3.73 s) | 0.95 |

TP8 (GPUs 0-7):

| Concurrency | MXFP4 | IQ2R-v2 | IQ2R-v2 / MXFP4 |
|---|---|---|---|
| 1 | 87 (11.3 ms, 0.10 s) | 79 (12.6 ms, 0.11 s) | 0.90 |
| 2 | 171 (11.6 ms, 0.17 s) | 148 (12.6 ms, 0.95 s) | 0.87 |
| 4 | 328 (11.2 ms, 1.02 s) | 294 (12.8 ms, 0.85 s) | 0.90 |
| 8 | 605 (12.2 ms, 1.07 s) | 536 (13.9 ms, 1.08 s) | 0.89 |
| 16 | 1160 (13.2 ms, 0.64 s) | 1021 (15.0 ms, 0.74 s) | 0.88 |
| 32 | 2055 (15.2 ms, 0.42 s) | 1851 (16.8 ms, 0.51 s) | 0.90 |
| 64 | 3062 (19.3 ms, 1.67 s) | 3034 (20.4 ms, 0.77 s) | 0.99 |
| 128 | 5236 (23.4 ms, 1.07 s) | 4807 (25.4 ms, 1.29 s) | 0.92 |
| 256 | 7950 (30.3 ms, 1.97 s) | 7455 (32.0 ms, 2.37 s) | 0.94 |

- IQ2R-v2 is 5-17% behind MXFP4 at TP4 on this host and stack.
- At TP8 it is 6-13% behind (1% at concurrency 64, where the MXFP4 run had a slow
  first-token time).
- The MoE is about 37% of IQ2R-v2's TPOT at concurrency 16 (84 µs x 75 layers of
  16.8 ms), so each 1% of MoE time is about 0.4% of decode throughput.
- MXFP4 with the MonoKernel on (the ATOM default) gives 162 and 295 tokens/s
  at concurrency 1 and 2; above that the MonoKernel is not used.

## On-chip memory with a 1,024-entry codebook

A gfx950 CU has 160 KiB of LDS. The four codebook copies take 16 KiB per
expert today and 32 KiB with 1,024 entries. Per-kernel LDS and VGPRs come from
`-Rpass-analysis=kernel-resource-usage` on today's source and on a copy with
`iq2r_codebook_words` at 1,024 entries. Workgroups per CU is the smaller of
the LDS and VGPR limits.

| Kernel | Used for | Threads | LDS v2 -> 1,024 book (bytes) | VGPRs | Workgroups per CU | Waves per SIMD |
|---|---|---|---|---|---|---|
| `glm53_gate_m1_kernel` | M1 gate | 512 | 49,152 -> 65,536 | 124 | 2 -> 2 | 4 -> 4 |
| `glm53_gate_decode_nobarrier_v2_kernel` | M2-256 gate | 512 | 49,152 -> 65,536 | 110 | 2 -> 2 | 4 -> 4 |
| `glm53_gate_prefill_kernel` | M >= 512 gate | 256 | 33,536 -> 49,920 | 254 | 2 -> 2 | 2 -> 2 |
| `glm53_down_route9_v2_kernel` | M1-4 down | 512 | 134,016-136,832 -> 265,088-267,904 | 84-186 | 1 -> **does not fit** | 2 -> - |
| `glm53_down_tp4_kernel` | TP4 M8-256 down | 256 | 24,832 -> 41,216 | 163 | 3 -> 3 | 3 -> 3 |
| `glm53_down_tp8_ordered_kernel` 1 atom | TP8 M16-128 down | 256 | 20,608 -> 36,992 | 124 | 4 -> 4 | 4 -> 4 |
| `glm53_down_tp8_ordered_kernel` 2 atoms | TP8 M8, M256 down | 256 | 24,832 -> 41,216 | 108 | 4 -> **3** | 4 -> **3** |
| `glm53_down_prefill_tp4_kernel` | TP4 M >= 512 down | 256 | 50,176 -> 66,560 | 164 | 3 -> **2** | 3 -> **2** |
| `glm53_down_prefill_tp8_kernel` | TP8 M >= 512 down | 256 | 24,832 -> 41,216 | 118 | 4 -> **3** | 4 -> **3** |

- route9 keeps one 16 KiB set of copies per route (8 x 16 KiB); with 1,024
  entries that is 256 KiB, over the 160 KiB of a CU. This is the "doesn't fit"
  of the October 8 x 512 scale option.
- The TP8 2-atom ordered and prefill downs miss four workgroups per CU by 256
  bytes each (41,216 > 160 KiB / 4 = 40,960); 256 bytes is their activation
  scale array.
- TP4 prefill down needs at most 54,613 bytes for three workgroups per CU; it
  would use 66,560.
- The gate kernels are VGPR-limited at the occupancy they have today.

**Measured cost of the LDS alone.** The v2 kernels rebuilt with 16 KiB of
unused LDS per workgroup (route9 excepted), alternating with v2 on one GPU, 10
rounds each, ratio of medians:

| M | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TP4 | 0.991 | 1.010 | 1.001 | 1.002 | 1.004 | 1.014 | 1.018 | 1.017 | 1.012 | 1.020 | 1.038 | 1.040 | 1.040 |
| TP8 | 0.991 | 1.017 | 1.010 | 1.031 | 1.007 | 1.014 | 1.013 | 1.014 | 1.026 | 1.010 | 1.009 | 1.010 | 1.010 |

M1-M4 leave route9 unchanged, so they measure only the gate kernels' LDS
(noise level). The occupancy losses show where expected: TP4 prefill +4%,
TP8 M8 and M256 +3%.

**Without route9.** The best non-route9 launch at M2 and M4 (the tuner's own
candidates) is 14-21% slower than route9 (TP4 M2 1.21x, M4 1.14x; TP8 M2 1.20x,
M4 1.16x); M1 has no other down kernel. route9 has to be adapted, not dropped.

## Options for the kernels that lose occupancy (to be decided)

| Kernel | Option | Expected cost against v2 |
|---|---|---|
| route9 (M1-4) | **A.** Keep only copy 0 per route in LDS (8 x 8 KiB = 64 KiB, about 72 KB in all) and form the needed copy in registers after the gather with the existing exact byte add `b + delta_c[b & 7]` (`v_and`, `v_perm`, `v_add` per dword, about 24 VALU per atom) | Removes per-route copy formation, which would double to 32 KiB per route. The kernel is latency-bound at 2 waves per SIMD, so the VALU should mostly hide: estimated route9 +0-10%, MoE +0-4% at M1-4 |
| | **B.** Four routes per workgroup (4 x 32 KiB), two workgroups per (columns, token), deterministic sum of the two halves through an FP32 partial and an arrival counter | Twice the workgroups (M1 would use all 256 CUs; today 128) but half the waves per CU and extra partial traffic; uncertain, either sign |
| | C. Use the non-route9 downs at M2-4 | Measured +14-21%; no option at M1. Rejected |
| TP8 ordered 2-atom, TP8 prefill down | Move the 256-byte scale array out of LDS (registers, or aliased with the input tile) | Back to 4 workgroups per CU: recovers the measured ~3% (TP8 M8, M256) and ~1% (TP8 prefill) |
| TP4 prefill down | Stage the 64 x 512 input tile in two K halves (16 KiB less, 49.4 KB in all, three workgroups per CU), one more barrier per task | Recovers most of the measured 4%; or accept the 4% and keep only ~1% for the 10-bit unpack at M >= 1024 against the 1.05x target |
| Gate kernels | None needed | Occupancy unchanged |

Costs not yet measured, all common to every option: +5.7% weight bytes; copy 0
loads of 8 instead of 4 KiB and twice the copy formation per gate/up item; the
10-bit unpack. Today an atom's four 9-bit indices are a low dword plus a
nibble that shares the metadata word with the level nibble; 10 bits need a
byte of high bits per atom, so the metadata word no longer holds them and the
record layout changes (2 more bytes per lane per quad).

## Format: 1,024 entries, 10-bit indices (phase 2)

Everything else is the shipped IQ2R-v2 format (`aiter/ops/iq2r_v2.py`): E8M0 `d`
per row and 256 inputs, a 4-bit level per 32 weights selecting one of four
pre-scaled copies `E4M3(codebook * 2^(c/4))`, explicit signs, FP32 gains for gate
and up. The codebook size is a parameter (`codebook_entries`, 512 by default):

- `iq2r_v2_seed_codebook(entries=1024)` seeds the IQ2_XXS grid at `2^(j/4)`,
  j = 0..3 (256 entries each, scaled from the E4M3 grid: the recipe-v3 seed of
  `scale_compare.py --book1024`); `iq2r_v2_learn_codebook(entries=1024)` learns
  from it, still within bytes 0x08..0x78.
- `iq2r_v2_write_data`, `iq2r_v2_read` and `iq2r_v2_materialize` take
  `codebook_entries`; the reference decoder is unchanged otherwise.
- Format name `iq2r-1024-fullsign-c4e4q-e8m0d256-native-v2`; packed layout
  `glm53-packed-v3`. 512-entry checkpoints keep `glm53-packed-v2` and every
  512-entry byte is unchanged.

**Records.** An 8-weight index gains a tenth bit, four per atom and lane (an atom
is 16 rows x 128 inputs, 32 weights per lane):

- Atoms 0 and 1 of a triplet keep theirs in the atom-two record's fourth
  metadata byte, which IQ2R-v2 leaves as padding.
- Atom 2's go to one byte per lane after every two K tiles: records are stored in
  slabs of two K tiles (256 inputs, the tensor-parallel slicing unit), the two
  tiles' six-block groups (2 x 3,584 bytes), then 64 bytes per triplet.
- Down records grow 1.8% over 512 entries (7,296 instead of 7,168 bytes per
  slab), the information content; no new padding.
- Packed gate/up quads (`iq2r_glm53_pack_v2`): an atom's packed indices are
  `i0 | i1 << 10 | i2 << 20 | i3 << 30` (40 bits): bits 0-31 replace the index
  bytes, bits 32-35 the metadata nibble, bits 36-39 a nibble of the lane's
  tenth-bit word. Each six-tile K slice (one decode-gate wave's) stores its six
  2,304-byte tiles, then a 12-byte word per lane (tile t's four nibbles in bits
  16t..16t+15): 2,432 bytes per tile on average, +5.6% (the information
  content).
- In all, weights are 2.375 + 0.03 (`d`) bits per weight against 2.28: +4.3%
  bytes.

The packers infer the codebook size from the auxiliary size; pack/unpack and TP
slicing round-trip bit for bit for both sizes.

## Kernels (phase 2)

Every gate and down kernel takes the codebook size as a template parameter; the
host picks it from the packed sizes, so 512-entry checkpoints run the same
instantiations as before (their generated code is unchanged).

| Kernel | 1,024-entry change | LDS (bytes) | Workgroups per CU |
|---|---|---|---|
| M1 gate | 8 KB copy 0 per thread block, copies formed in LDS; each wave loads its six tiles and tenth bits at once and decodes with the decode gate's cross-atom pipelining | 65,536 | 2 (as v2) |
| Decode gate | copies formed in LDS; one tenth-bit load per item and wave | 65,536 | 2 (as v2) |
| Prefill gate | copies formed in LDS; one tenth-bit dword live | 49,920 | 2 (as v2) |
| route9 (M1-4) | `glm53_down_route9_wide_kernel`: copy 0 per route (8 KB, DMA); a gathered codeword becomes its class's copy in registers (`b + delta_c[b & 7]`, per-class tables in LDS) | 68,736-71,552 | 1 (as v2) |
| Decode downs (TP4 single, TP8 ordered) | copy 0 staged (8 KB instead of 32 KB of stored copies), copies 1-3 formed by each wave from the chunks it staged; TP8 activation scales in registers | 36,864-41,216 | 3 (TP4), 4 (TP8), as v2 |
| Prefill downs | TP8: scales in registers (4 workgroups per CU as v2); TP4: 2 workgroups per CU (LDS), built for that occupancy with an unrolled K loop | 40,960 (TP8), 66,560 (TP4) | 4, 2 |

Tests (`op_tests/test_iq2r_glm53.py`, `test_iq2r_glm53_unpack.py`): every MoE test
runs on 512- and 1,024-entry layers (dense reference, explicit configs, invalid
routes, chunked prefill); `test_glm53_down_decode_bit_exact` checks that every
down kernel decodes bit-exactly (one-hot activations: each output is
`BF16(gain * W[:, k])`, compared with `iq2r_v2_materialize`), for both sizes.

## Building and serving a 1,024-entry checkpoint (phase 3)

`compile` takes `--codebook-entries 1024`; everything else is the shipped recipe
(the calibration artifact, importance, encoder, 0.98 gain, shared expert FP8):

    # one process per GPU on its own MoE layers, then one run without --layers
    HIP_VISIBLE_DEVICES=$i python3 -m aiter.quantization.iq2r.glm5.compile \
        --model-dir $FP8 --output-dir $OUT --calibration-cache calibration.pt \
        --codebook-entries 1024 --layers ...
    python3 -m aiter.quantization.iq2r.glm5.compile --model-dir $FP8 \
        --output-dir $OUT --calibration-cache calibration.pt --codebook-entries 1024

Encoding takes about 9 s per expert (gate/up and down) on one MI350X, against
6 s with 512 entries. Layer files record `iq2r_codebook_entries`, format
`iq2r-1024-fullsign-c4e4q-e8m0d256-native-v2` and layout `glm53-packed-v3`;
the config's `quantization_config.iq2r_layout` is `glm53-packed-v3`.

ATOM (`Iq2rMoEMethod`) reads the codebook size from `iq2r_layout`
(`glm53-packed-v2`: 512 entries, `glm53-packed-v3`: 1,024) and sizes, slices
and checks the buffers accordingly; the AITER kernels pick their instantiation
from the packed sizes. Serve as a v2 checkpoint (the shared expert stays block
FP8: exclude `*.mlp.shared_experts*` from online quantization).

## Results (phase 2 and 3)

**Correctness.** `op_tests/test_iq2r_glm53.py` (202 passed: every MoE test on 512- and
1,024-entry layers, every down kernel bit-exact), `test_iq2r_glm53_unpack.py`,
`test_iq2r_glm5_compile.py` (17 passed, 512 and 1,024 builds) and ATOM's
`tests/test_iq2r_moe_method.py` (27 passed). On the GLM-5.3 1,024-entry checkpoint
(`/models/zai-org/GLM-5.3-IQ2Rv3-final`, built with the shipped calibration):
`pack(unpack(x)) == x` on layer 3, and the MoE on its TP4 and TP8 rank-0 shards
matches the dense reference from the unpacked bytes at M = 1..1024 (relative RMSE at
most 0.019, cosine at least 0.99983; 16 experts, random top-8 routing).

**MoE time, synthetic weights.** As the baseline: tuner graph replay, 16 rounds on one
MI350X, every round timing v2 and v3 back to back; µs median, ratio = median of the
per-round ratios. v2 rows are the shipped tuned rows, v3 the tuned 1,024-entry rows.

| M | TP4 v2 | TP4 v3 | TP4 v3/v2 | TP8 v2 | TP8 v3 | TP8 v3/v2 | target |
|---|---|---|---|---|---|---|---|
| 1 | 18.34 | 20.66 | 1.126 | 16.96 | 18.11 | 1.068 | <= 1.07 |
| 2 | 20.81 | 22.98 | 1.104 | 18.59 | 19.54 | 1.051 | <= 1.07 |
| 4 | 32.20 | 36.77 | 1.128 | 21.62 | 22.91 | 1.061 | <= 1.07 |
| 8 | 56.40 | 59.64 | 1.060 | 36.93 | 37.69 | 1.019 | <= 1.07 |
| 16 | 84.32 | 89.32 | 1.060 | 51.72 | 55.07 | 1.065 | <= 1.07 |
| 32 | 125.67 | 137.15 | 1.093 | 74.75 | 75.19 | 0.998 | <= 1.07 |
| 64 | 172.84 | 181.06 | 1.049 | 92.69 | 94.87 | 1.027 | <= 1.07 |
| 128 | 195.91 | 208.00 | 1.061 | 106.98 | 112.97 | 1.059 | <= 1.05 |
| 256 | 210.38 | 227.05 | 1.077 | 120.90 | 122.71 | 1.013 | <= 1.05 |
| 512 | 269.31 | 291.02 | 1.081 | 153.71 | 164.88 | 1.074 | <= 1.05 |
| 1024 | 318.67 | 338.49 | 1.063 | 197.10 | 210.30 | 1.065 | <= 1.05 |
| 2048 | 502.15 | 530.85 | 1.057 | 323.61 | 346.44 | 1.070 | <= 1.05 |
| 4096 | 849.79 | 897.17 | 1.056 | 545.31 | 572.44 | 1.051 | <= 1.05 |

**MoE time, real checkpoint layers** (v3/v2; same procedure on the rank-0 shard of
each checkpoint's layer, random routing; 2 rounds):

| M | TP4 layer 3 | TP4 layer 40 | TP8 layer 3 | TP8 layer 40 |
|---|---|---|---|---|
| 1 | 1.120 | 1.119 | 1.076 | 1.079 |
| 2 | 1.119 | 1.118 | 1.067 | 1.071 |
| 4 | 1.170 | 1.178 | 1.109 | 1.121 |
| 8 | 1.065 | 1.041 | 1.020 | 1.037 |
| 16 | 1.076 | 1.041 | 1.043 | 1.044 |
| 32 | 1.049 | 1.059 | 1.050 | 0.998 |
| 64 | 1.081 | 1.079 | 1.058 | 1.029 |
| 128 | 1.071 | 1.081 | 1.065 | 1.071 |
| 256 | 1.069 | 1.051 | 1.035 | 1.044 |
| 512 | 1.088 | 1.086 | 1.071 | 1.080 |
| 1024 | 1.066 | 1.067 | 1.069 | 1.060 |
| 2048 | 1.072 | 1.055 | 1.070 | 1.080 |
| 4096 | 1.059 | 1.049 | 1.050 | 1.051 |

**Serving, TP4 (GPUs 4-7) and TP8 (GPUs 0-7)**, standard kernels (`ATOM_MONO_ENABLE=0`),
the phase-1 setup; output tokens/s (mean TPOT, mean TTFT):

TP4:

| Concurrency | MXFP4 tok/s (TPOT, TTFT) | v2 tok/s (TPOT, TTFT) | v3 tok/s (TPOT, TTFT) | v2/MXFP4 | v3/MXFP4 |
|---|---|---|---|---|---|
| 1 | 88 (11.3 ms, 0.10 s) | 80 (12.5 ms, 0.11 s) | 79 (12.6 ms, 0.11 s) | 0.91 | 0.90 |
| 2 | 181 (10.9 ms, 0.16 s) | 150 (12.6 ms, 0.78 s) | 149 (12.7 ms, 0.79 s) | 0.83 | 0.82 |
| 4 | 318 (11.8 ms, 0.87 s) | 288 (13.2 ms, 0.72 s) | 279 (13.7 ms, 0.74 s) | 0.91 | 0.88 |
| 8 | 530 (13.1 ms, 2.04 s) | 496 (15.1 ms, 1.09 s) | 491 (15.2 ms, 1.10 s) | 0.93 | 0.92 |
| 16 | 974 (15.8 ms, 0.65 s) | 910 (16.8 ms, 0.83 s) | 890 (17.2 ms, 0.85 s) | 0.93 | 0.91 |
| 32 | 1627 (19.1 ms, 0.60 s) | 1510 (20.3 ms, 0.90 s) | 1465 (21.0 ms, 0.91 s) | 0.93 | 0.90 |
| 64 | 2504 (24.7 ms, 0.93 s) | 2372 (25.9 ms, 1.16 s) | 2305 (26.6 ms, 1.17 s) | 0.95 | 0.92 |
| 128 | 3859 (31.6 ms, 1.63 s) | 3650 (33.0 ms, 2.11 s) | 3550 (34.0 ms, 2.08 s) | 0.95 | 0.92 |
| 256 | 5784 (40.8 ms, 3.61 s) | 5495 (43.0 ms, 3.73 s) | 5265 (44.9 ms, 3.80 s) | 0.95 | 0.91 |

TP8:

| Concurrency | MXFP4 tok/s (TPOT, TTFT) | v2 tok/s (TPOT, TTFT) | v3 tok/s (TPOT, TTFT) | v2/MXFP4 | v3/MXFP4 |
|---|---|---|---|---|---|
| 1 | 87 (11.3 ms, 0.10 s) | 79 (12.6 ms, 0.11 s) | 79 (12.6 ms, 0.11 s) | 0.90 | 0.90 |
| 2 | 171 (11.6 ms, 0.17 s) | 148 (12.6 ms, 0.95 s) | 146 (12.8 ms, 0.95 s) | 0.87 | 0.86 |
| 4 | 328 (11.2 ms, 1.02 s) | 294 (12.8 ms, 0.85 s) | 289 (13.0 ms, 0.88 s) | 0.90 | 0.88 |
| 8 | 605 (12.2 ms, 1.07 s) | 536 (13.9 ms, 1.08 s) | 530 (14.0 ms, 1.16 s) | 0.89 | 0.88 |
| 16 | 1160 (13.2 ms, 0.64 s) | 1021 (15.0 ms, 0.74 s) | 1012 (15.2 ms, 0.65 s) | 0.88 | 0.87 |
| 32 | 2055 (15.2 ms, 0.42 s) | 1851 (16.8 ms, 0.51 s) | 1832 (17.0 ms, 0.49 s) | 0.90 | 0.89 |
| 64 | 3062 (19.3 ms, 1.67 s) | 3034 (20.4 ms, 0.77 s) | 2954 (20.9 ms, 0.75 s) | 0.99 | 0.96 |
| 128 | 5236 (23.4 ms, 1.07 s) | 4807 (25.4 ms, 1.29 s) | 4743 (25.7 ms, 1.31 s) | 0.92 | 0.91 |
| 256 | 7950 (30.3 ms, 1.97 s) | 7455 (32.0 ms, 2.37 s) | 7285 (32.8 ms, 2.41 s) | 0.94 | 0.92 |

**Against the targets.**

- Met: TP8 decode (M <= 64: 1.00-1.07), TP4 M8-16 and M64, TP8 M256.
- Missed by up to 3 points: TP4 M32 (1.09), M128-256 (1.06-1.08); prefill at both
  widths (1.05-1.08); TP8 M128 (1.06).
- Missed by more: TP4 M1-4 (1.10-1.13 synthetic, 1.12-1.18 real) and TP8 M4 on real
  weights (1.11-1.12).
- Serving: v3 is 0.96-1.00 of v2 (TP4 0.96-0.99, TP8 0.97-1.00) and 0.86-0.96 of
  MXFP4 A4W4; the stretch goal (v3 >= MXFP4) is not met.

**Why.**

- Bytes: +4.3% weights (+5.6% gate/up, +1.8% down), the floor for every memory-bound
  point.
- route9 (M1-4): eight routes' 32 KB copies do not fit, so each gathered codeword is
  formed in registers; its compute phase is 72% longer (in-kernel timestamps:
  3,526 -> 6,079 cycles), about 1.5 µs of TP4 M1.
- Prefill gate: ALU-bound decode; the 10-bit unpack adds 13% ALU (218 -> 246 per 32
  MFMAs), about +10% gate time.
- M1 gate: loading phase about 20% longer than v2's (8 KB copy 0, more bytes); the
  consumption-order loads recovered 0.8 µs.

**Tried and dropped (measured):** a 16-bit tenth-bit word per tile and a 32-bit word
per slab (extra loads); forming the decode downs' copies from registers (VGPRs,
1.17-1.27); route9 with two waves per route (own or shared copy 0; 1.07-1.33); the
M1 gate loading all tiles before decoding; a merged high-index byte per tile; the
prefill gate at Unroll 1. Kept: copy-0 DMA plus per-wave LDS formation in the decode
downs, TP8 scales in registers, the TP4 prefill down at two workgroups per CU, the
single2 down (TP4 M8-16), and consumption-order loads in the M1 gate.

**Open options** (not built): route9 with four routes per workgroup (four full 32 KB
books, cross-workgroup top-8 sum) for M1-4; the redline-style split-K decode down
with a separate top-k reduction.

## Shared expert fused into the MoE launches

**Why.** Serving ablations at TP4 (one part of the MoE block switched off per run,
`ATOM_MONO_ENABLE=0`) showed that the gap to MXFP4 was not the routed kernels:

| Mean TPOT (ms) | c=1 | c=64 |
|---|---|---|
| MXFP4 | 11.31 | 25.31 |
| MXFP4, MoE block removed | 7.54 | 12.00 |
| v2 | 12.49 | 26.85 |
| v2, MoE block removed | 7.55 | 12.07 |
| v2, routed experts only | 10.45 | 19.78 |
| v2, FP8 shared expert only | 11.03 | 16.82 |

The routed IQ2R block alone (2.90 ms per step at c=1, 7.71 ms at c=64) is cheaper
than MXFP4's whole MoE block (3.77 / 13.31 ms). The FP8 shared expert, a separate
six-kernel MLP on a second stream, cost 3.48 ms on its own at c=1 (46 µs per layer)
and the second stream hid only 1.4 ms of it. Serving it as PTPC FP8 (12.30 ms) or
MXFP4 (12.43 ms) instead changed nothing: the cost is the separate launches, not the
format. v1 avoided it by quantizing the shared expert to 2 bits as expert 256 inside
the IQ2R kernels, which costs accuracy (AIME 49 vs 54, SWE 3 vs 19-26 in the redline
study).

**Design.** The shared expert stays 8-bit and runs as route slot 8 of every token
(weight 1) inside the gate and down launches, for batches of up to
`FUSED_SHARED_TOKENS` (256) tokens:

- Weights: MXFP8 (E4M3 bytes, one E8M0 scale per 32 inputs), repacked at load from
  the checkpoint's block FP8 (`iq2r_glm53_pack_shared`) in MFMA operand order, so a
  lane loads 32 bytes and one scale byte per 16 x 128 tile; nothing to decode.
  9.4 MB per layer per TP4 rank (about 0.7 GB per GPU for 75 layers; the FP8 copy
  stays loaded for prefill).
- Gate: extra work items in the M1 and decode gate launches, one per 16 x MA
  tokens x 16 gate/up rows over the whole K (eight K768 slices, one per wave; about
  the bytes of a routed item). They write SiLU(gate) * up in BF16 to the workspace
  (`shared_act`). MA = 1, 2 or 4 by batch size, so weights are re-read once per 16,
  32 or 64 tokens.
- Down: route9 (M1-4) spreads the shared expert's column blocks x K tiles over its
  eight waves after their routes and adds the result in its top-8 sum; the decode
  downs (M2-256) run extra items writing the shared route rows, which the route
  reduction adds with weight 1. Both quantize the BF16 activation to MXFP8 while
  building the MFMA operand (block maxima exchanged between lane groups), with the
  rounding of the routed path.
- Routing: the decode sort places the shared routes after the routed ones in token
  order; they get no tasks.
- No workgroup waits on another. A split-K first version (partials plus an arrival
  counter) was 3-5x slower: an agent-scope `__threadfence()` on gfx950 compiles to
  `buffer_wbl2 sc1` + `buffer_inv sc1`, a writeback and invalidate of the XCD's L2.
- Larger batches use the prefill kernels and the separate shared MLP, as before.
- ATOM: `DeepseekV2MoE.process_weights_after_loading` dequantizes the shared expert
  and attaches it to `Iq2rMoEMethod`; the MoE dispatch op takes the fused path for up
  to 256 tokens. `ATOM_IQ2R_FUSE_SHARED=0` keeps the separate MLP.
- Tuning: the tuned CSV gained a `shared` key; rows with `shared=1` are tuned with
  the shared expert in the launch (decode kernels only) and are looked up first.

**Cost in the MoE call** (TP4, 1,024 entries, tuner graph replay, tuned rows): M1
24.1 µs (+2.3 over routed only), M4 41.1 (+4.2), M8 60.0 (+0.9), M16 92.1 (+2.3),
M32 140.5 (+6), M64 195.4 (+15), M256 247.0 (+23). The separate MLP cost about
27 µs per layer net at c=1 and about 93 µs at c=64.

**Correctness.** `test_glm53_moe_fused_shared_expert` (M = 1..1024, TP4/TP8, 512 and
1,024 entries): relative RMSE <= 0.021, cosine >= 0.99978 against the dense
reference with the shared expert, and repeated runs give identical output.
`op_tests/test_iq2r_glm53.py`: 266 passed. ATOM `tests/test_iq2r_moe_method.py`:
29 passed. Serving sanity prompts answer correctly.

**Serving, TP4 (GPUs 4-7)**, standard kernels, one server at a time, MXFP4 measured
in the same session; output tokens/s (mean TPOT, mean TTFT):

| Concurrency | MXFP4 | v3 fused, tuned | v2 fused | v3 separate shared | v3 fused / MXFP4 | v3 separate / MXFP4 |
|---|---|---|---|---|---|---|
| 1 | 85 (11.7 ms, 0.10 s) | 91 (10.9 ms, 0.10 s) | 92 (10.8 ms, 0.10 s) | 79 (12.6 ms, 0.11 s) | 1.07 | 0.93 |
| 2 | 180 (11.0 ms, 0.20 s) | 167 (11.2 ms, 0.78 s) | 169 (11.1 ms, 0.82 s) | 149 (12.7 ms, 0.79 s) | 0.93 | 0.83 |
| 4 | 315 (11.8 ms, 0.88 s) | 313 (12.1 ms, 0.71 s) | 321 (11.8 ms, 0.70 s) | 279 (13.7 ms, 0.74 s) | 0.99 | 0.88 |
| 8 | 524 (13.3 ms, 2.05 s) | 539 (13.8 ms, 1.08 s) | 515 (14.4 ms, 1.13 s) | 491 (15.2 ms, 1.10 s) | 1.03 | 0.94 |
| 16 | 964 (15.8 ms, 0.78 s) | 961 (15.8 ms, 0.85 s) | 936 (16.3 ms, 0.83 s) | 890 (17.2 ms, 0.85 s) | 1.00 | 0.92 |
| 32 | 1589 (19.4 ms, 0.74 s) | 1558 (19.7 ms, 0.91 s) | 1589 (19.3 ms, 0.87 s) | 1465 (21.0 ms, 0.91 s) | 0.98 | 0.92 |
| 64 | 2492 (24.8 ms, 0.94 s) | 2456 (24.8 ms, 1.26 s) | 2488 (24.6 ms, 1.14 s) | 2305 (26.6 ms, 1.17 s) | 0.99 | 0.92 |
| 128 | 3838 (31.8 ms, 1.64 s) | 3707 (32.5 ms, 2.08 s) | 3807 (31.7 ms, 2.03 s) | 3550 (34.0 ms, 2.08 s) | 0.97 | 0.93 |
| 256 | 5708 (41.3 ms, 3.63 s) | 5450 (43.3 ms, 3.80 s) | 5534 (42.4 ms, 3.94 s) | 5265 (44.9 ms, 3.80 s) | 0.95 | 0.92 |

(v2 fused used the untuned shared rows. "v3 separate shared" is the phase-3 run.)

- v3 TPOT is at or below MXFP4's up to c=64 except c=4 (+0.3 ms); c=128-256 trail
  by 0.7-2.0 ms, where decode batches carry prefill chunks of more than 256 tokens
  (separate shared MLP) and M=256 decode adds 23 µs per layer.
- c=2: median TTFT equals MXFP4's (193 vs 180 ms); the mean carries one request
  that waits about 6 s the first time a 2,048-token prefill batch forms. It is
  intermittent, IQ2R-only (present before the fusion too) and leaves no compiled
  artifact; not resolved.
- TP8 serving was not rerun (GPUs 0-3 were not available); the kernels are tested
  and timed at TP8 (fused shared expert +2.4 to +6 µs per call at M <= 64).
