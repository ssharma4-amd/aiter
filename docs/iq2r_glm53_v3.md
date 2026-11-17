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
