# IQ2R-v2 GLM-5.3 MoE kernels on gfx950: performance findings

This is the record of the IQ2R-v2 kernel study (2026-11-03 to 11-04, MI355X, gfx950). It covers what was tried,
what each change measured, the fastest configuration found, why the gate is still behind v1, and what is left.
Appendix A collects the lessons from the v1 kernels that apply to v2. Appendix B summarizes the 11-bit-index (FP11)
kernel study, which was stopped.

Read this before starting new v2 kernel work. Most ideas in sections 7, 13 and 15 have been measured already.

## Contents

1. [Summary](#1-summary)
2. [Code on this branch](#2-code-on-this-branch)
3. [Format and shapes](#3-format-and-shapes)
4. [How everything was measured](#4-how-everything-was-measured)
5. [v1 reference numbers](#5-v1-reference-numbers)
6. [Down kernels](#6-down-kernels)
7. [Gate kernels](#7-gate-kernels)
8. [Best configuration vs v1](#8-best-configuration-vs-v1)
9. [Format ablations (weight error)](#9-format-ablations-weight-error)
10. [Macro catalog](#10-macro-catalog)
11. [gfx950 microarchitecture facts](#11-gfx950-microarchitecture-facts)
12. [Correctness](#12-correctness)
13. [Dead ends](#13-dead-ends)
14. [Porting to the one-book format](#14-porting-to-the-one-book-format)
15. [Open items](#15-open-items)
16. [Tools](#16-tools)
- [Appendix A. Lessons from the v1 kernels](#appendix-a-lessons-from-the-v1-kernels)
- [Appendix B. FP11 (11-bit index) study](#appendix-b-fp11-11-bit-index-study)

---

## 1. Summary

The goal was to match or beat the merged v1 kernels (aiter#5728) per stage (gate+up, down) at every M, for both
TP4 (I=512) and TP8 (I=256).

**Down: done for decode.**
- M ≤ 4 uses route9 "gen7", which builds the codebook class copies inside the kernel from one 4 KB base book. It
  is bit-exact against the staged-copy build. It needs every codebook byte in 0x08..0x78 (section 12).
- M1/M2 down: I=512 6.48 / 7.24 µs vs v1 7.61 / 8.06; I=256 5.41 / 5.52 µs vs v1 5.96 / 6.30.
- M8–256: faster than v1 at I=512 (−2..−9% with the committed defaults). At I=256, M8–64 is faster and M128–512 is
  +1..+2%. NTT=1 makes I=256 M8–128 5–17% faster, but it must be chosen per launch (not implemented).
- Prefill (M ≥ 512) is between −1.5% and +4.4% vs v1.

**Gate: close, not done.**
- Best build (the committed default): OPSEL + GG=2 + ABUF + PFA + NT=2.
- Same-session gate µs at M16/64/256: I=512 39.24 / 80.20 / 103.51 vs v1 38.51 / 79.26 / 97.04 (+1.9 / +1.2 /
  +6.7%). I=256 23.32 / 46.63 / 56.15 vs v1 24.30 / 45.64 / 54.52 (−4.0 / +2.2 / +3.0%).
- The remaining gap is fully accounted for (section 7.4):
  - per-K-step `d` reloads: ~3.7% at I=512 M256;
  - per-expert book load + copy formation: ~2–4%.
  Removing both (a timing-only build) gives 97.87 µs, i.e. v1.
- With OPSEL + ABUF the gate inner loop is 110 VALU per K256 quad, the same as v1. It started at 139.

**What closes the gate gap (section 15):**
1. Port to the one-book format (`0d7a0963a`): the gate stages 16 KB instead of 32 KB. Estimated 1.5–2%, not measured.
2. DBC: one `d` load per item plus a permlane broadcast. It is written but was never built (default 0).
3. Re-time v1 with NT=2. NT=2 also changes the v1 code path and v1 was never timed with it, so the real gap may
   differ.

**Not covered by any measurement here:**
- the 256-expert / top-8 runtime of the v2 checkpoint;
- the separate FP8 shared expert;
- end-to-end serving.

---

## 2. Code on this branch

Branch `glm-5.3-iq2r-v2-kernel-exp`. History:

| Commit | Content |
|---|---|
| `03947b68d` | v2 format (two-level MFMA-native scales) + first v2 kernels |
| `a5a477367` | prefill gate `d` prefetch (the commit title lists earlier work that is already in 03947b68d) |
| `881960d84` | this study: 70 `GLM53_X_*` experiment macros in `csrc/kernels/iq2r/iq2r_gemm_gfx950.cu` (4631 lines) |
| next | this document and the study tools (`docs/iq2r_v2_kernel_study/`) |

- The branch is on the **two-book** v2 format: separate gate and up codebooks, i.e. 32 KB of gate class copies per
  expert. The v2 format was later changed to one book shared by gate and up (`0d7a0963a` on `glm-5.3-iq2r-v2`).
  Section 14 lists what the port needs.
- `vbench.py` builds two-book synthetic weights (`books = (seed, seed*1.0625)`). Change it for the one-book format.

**The defaults build the fastest variants found:**

| Stage | Defaults |
|---|---|
| Route9 down (M ≤ 4) | R9GEN=3, GPERM=1, R9X=1, R9PIPE=1, R9AL=1, R9SS=1, R9LAT=1, R9PRO=1, R9RFL=1, R9SR=1, R9ADDR=1, RB=1, R9T=4 |
| Down M ≥ 8 | LVM=1, S8KO=1, T8KO=1, A32=1, T8LV=1, S8OCC=3, NTT=0 |
| Gate | OPSEL=1, UNR=1, GG=2, CHUNK=1, ABUF=1, PFA=1, NT=2, DBC=0 |

Every other `GLM53_X_*` macro is 0 (probe or rejected). Section 10 lists all of them.

**Rebuilding.**
- The study drivers set macros by editing the `#define`s in the file with `sed`, then delete
  `aiter/jit/module_iq2r_moe.so` so the JIT rebuilds. The JIT otherwise loads a stale .so silently.
- Do not pass these macros with `-D`. Two `#ifndef` guards wrap whole groups:
  - `#ifndef GLM53_X_DPRE` (lines 1025–1042) also covers BLK, NOFORM, LDSB, ABUF, PF2;
  - `#ifndef GLM53_X_DBC` (lines 1043–1079) also covers the 31 down/route9 macros after it.
  A `-D` of the first macro drops the defaults of the rest.
- RB=0 no longer compiles (`static_assert` in `glm53_gather_atom`), and R9X needs RB.
- Three macros change v1 (`Classes==0`) code too: NT, NTT and R9T. All others are gated on `Classes>0`.

Macro blocks: NTT line 419, NT 532, lines 578–604 and 1010–1079. `iq2r_v2_form_copies` is at 701, route9 at 2206, the
nobarrier gate at 3639.

---

## 3. Format and shapes

**Decode.** `w = 2^(d-127) · 2^(l//4) · E4M3(book[idx] · 2^((l%4)/4)) · sign`, then two FP32 output gains (even and
odd rows).

- `d`: one E8M0 byte per 16-row band per atom per K256, stored in the auxiliary `d` table.
- `l`: a 4-bit level per atom (32 weights), in the high nibble of the record metadata byte.
  - Bits 4–5 are the class `c`, which selects the codebook copy.
  - Bits 6–7 are the exponent offset `e`, added to `d`.
  - The packer orders classes s = (0, 2, 1, 3).
- Half of the class bytes a lane needs sit in the partner lane 16 away. One `permlane16_swap` fetches them
  (`iq2r_levels`).
- MFMA scale byte = `d + e`, for all four atoms in one add: `d_word + ((metadata >> 6) & 0x03030303)`
  (`iq2r_scale_bytes`). It cannot carry because `d ≤ 254 − e`.
- Indices are 9-bit, as in v1: 512 entries × 8 B = 4 KB per copy.
- Records (quads for gate/up, triplets for down) are byte-identical in layout to v1: 11,067,392 B per expert.

**v1 for comparison.** One E8M0 base per 16-row band, constant over the item, plus a 4-bit delta. v1 loads its base
once per item and keeps it in a register, so it has no per-K `d` stream. That stream is the main structural cost v2
adds (section 7.6).

**Auxiliary per expert.**

| | v1 | v2 two-book (this branch) | v2 one-book (`0d7a0963a`) |
|---|---|---|---|
| Gate codebook bytes staged per expert | 4 KB | 2 books × 4 classes × 4 KB = 32 KB | 16 KB |
| Gate CTA LDS (nobarrier) | 36,864 B | 65,536 B | ~49 KB |
| Down codebook bytes | 4 KB | 16 KB | 16 KB |

- Copy offset = class × 4 KB + book × 16 KB.
- One-book aux layout: `[C*4096 copies][d table][8 B gains]`, with
  `aux_bytes = classes*4096 + (n//(16*atoms))*(k//256)*64 + 8` (gate atoms 4, down atoms 3).
- Size cost of v2 two-book vs v1 (75 MoE layers × 257 experts, formula): canonical 216.40 vs 213.49 GB (+2.9 GB).
  Packed per GPU: TP4 53.53 vs 51.96 GB, TP8 27.24 vs 26.06 GB.

**Shapes.**
- hidden 6144, moe_intermediate 2048.
- TP4 → I=512 (16 n-tiles); TP8 → I=256 (8 n-tiles).
- The kernels assume 257 experts / top-9: v1 runs the shared expert as routed expert 256. The v2 checkpoint has 256
  routed experts and keeps the shared expert in block FP8, so it needs 256 / top-8 (section 15).

**Kernel picked per M (tuned CSV, both formats).**

| M | I=512 gate / down | I=256 gate / down |
|---|---|---|
| 1 | decode2 / route9 | decode2 / route9 |
| 2 | nobarrier4 / route9 | nobarrier4 / route9 |
| 4 | nobarrier2 / single8 | nobarrier4 / route9 |
| 8 | nobarrier2 / single8 | nobarrier4 / ordered8 |
| 16 | nobarrier2 / single4 | nobarrier2 / single4 |
| 32–256 | nobarrier2 / single8 | nobarrier2 / ordered4 |
| 512 | prefill2 / prefill | nobarrier2 / ordered4 |
| 1024–4096 | prefill / prefill | prefill4 / prefill |

"decode" at M1 is `glm53_gate_m1_kernel`. "nobarrier" is `glm53_gate_decode_nobarrier_kernel`. Down "single" is
`glm53_down_tp4_kernel` with MAtoms=1, and "ordered" is `glm53_down_tp8_ordered_kernel`.

---

## 4. How everything was measured

- **Harness: `vbench.py`.**
  - Per-stage CUDA-graph replay, cold: each replay reads ≥ 1 GiB of rotated layer copies, so experts come from HBM,
    not the 256 MB MALL. This is the tuner's method.
  - REPS=3, min.
  - Routes are seeded per M (`torch.manual_seed(1000+tokens)`). Unseeded routing gave numbers that could not be
    compared.
  - Weights are synthetic (`v2_synthetic_weights`, cached in `cache/`).
- **GPUs.** I=512 on one GPU and I=256 on another, kept fixed. One GPU was ~5% slower than the other; never compare
  across GPUs.
- **Same list, same session.** Compare only the same M list run in the same session. Within a list, noise is about
  ±1%. Between sessions v1 drifted ±5% (I=512 M256: 97.0 / 103.0 / 106.9 µs). That is why every important row has a
  same-session v1 run (`ref_v1`, `l3_v1`).
- **Timing-only probes** (STG4, STG0, NOCOPY, NOD, NOFORM, R9W, R9ND, …) produce wrong output on purpose. They
  measure what removing one cost would buy.
- **Phase timing.** GTS=1 (gate) and TS=1 (route9) write `s_memtime` stamps past the output. Read them with `tsg.py`
  and `ts.py`.
- **Counters.** rocprofv3 PMC via `prof.sh`. Use small counter groups: one group with TA counters produced no CSV.
- Everything ran in one container on one host, through `optest.sh` / `env.sh`.

---

## 5. v1 reference numbers

v1 = `aiter` main with aiter#5728 merged (`e10bd0a92`). Two same-session runs, gate / down µs. These are the
references for the sweeps in sections 6–7.

| I | M | ref_v1 gate | ref_v1 down | ref2_v1 gate | ref2_v1 down |
|---|---|---|---|---|---|
| 512 | 1 | 8.55 | 7.61 | 8.62 | 7.61 |
| 512 | 2 | 11.51 | 8.06 | 11.45 | 8.07 |
| 512 | 4 | 13.73 | 12.16 | 13.88 | 12.43 |
| 512 | 8 | 23.71 | 19.25 | 23.86 | 19.62 |
| 512 | 16 | 36.91 | 27.58 | 38.51 | 28.77 |
| 512 | 32 | 55.94 | 40.49 | 56.18 | 39.21 |
| 512 | 64 | 76.92 | 57.61 | 77.74 | 56.54 |
| 512 | 128 | 87.75 | 64.72 | 89.71 | 63.42 |
| 512 | 256 | 94.93 | 73.33 | 97.06 | 72.92 |
| 512 | 512 | 136.30 | 95.41 | 139.12 | 96.10 |
| 512 | 1024 | 147.16 | 121.55 | 149.37 | 122.17 |
| 512 | 2048 | 208.14 | 222.06 | 210.19 | 217.19 |
| 512 | 4096 | 330.35 | 388.63 | 330.36 | 383.65 |
| 256 | 1 | 8.29 | 5.96 | 8.29 | 6.02 |
| 256 | 2 | 8.61 | 6.30 | 8.60 | 6.27 |
| 256 | 4 | 9.64 | 7.94 | 9.53 | 7.93 |
| 256 | 8 | 14.19 | 11.96 | 14.34 | 12.05 |
| 256 | 16 | 24.03 | 17.36 | 23.51 | 17.37 |
| 256 | 32 | 33.10 | 24.34 | 33.06 | 24.24 |
| 256 | 64 | 44.93 | 32.12 | 44.86 | 32.15 |
| 256 | 128 | 49.12 | 37.31 | 49.33 | 38.14 |
| 256 | 256 | 53.19 | 39.16 | 53.75 | 39.37 |
| 256 | 512 | 75.52 | 45.06 | 75.31 | 45.41 |
| 256 | 1024 | 98.15 | 88.84 | 98.26 | 88.87 |
| 256 | 2048 | 121.53 | 165.69 | 121.10 | 164.98 |
| 256 | 4096 | 186.96 | 298.43 | 186.74 | 299.14 |

**Gate reference for the same-list attribution runs (`l3_v1`):** I=512 M16/64/256 = 38.51 / 79.26 / 97.04;
I=256 = 24.30 / 45.64 / 54.52.

**Starting point (original v2 kernels, before this study), v2/v1 complete MoE.**
- I=512: M1 1.79×, M2 1.66×, M4 1.17×, M8–256 1.22–1.29×, M512–4096 1.07–1.09×.
- I=256: M1 1.51×, M2 1.28×, M4 1.60× (route9), M8–512 1.14–1.26×.
- Original v2 down:
  - I=512 M1–256: 18.63, 20.55, 15.34, 24.83, 33.64, 48.90, 71.54, 79.02, 87.55.
  - I=256 M1–512: 10.11, 11.05, 19.53, 12.07, 17.46, 26.24, 36.68, 40.59, 43.41, 49.16.
- Original v2 gate I=512 M128/M256: 132.2 / 136.9 vs v1 93.5 / 102.6.

---

## 6. Down kernels

### 6.1 Route9 (M ≤ 4)

**Structure.**
- One CTA per (48-column triplet, token).
- 576 threads = 9 waves, one wave per route (expert); `__launch_bounds__(576,1)`.
- Each route's bf16 value goes to LDS, then an ordered FP32 combine of the 9 values. The reduction is fused.
- At M1 only 128 of 256 CUs have work.
- v2 needs a 16 KB book (4 class copies) per route, i.e. 144 KB LDS, so 1 CTA per CU. The original v2 kernel
  avoided this by reading the copies straight from L2, which is why it was 2.4× slower than v1.

**Cost breakdown at I=512 M1** (from timing-only probes and per-wave timestamps):
- Fixed cost + loads: ≈ 5.17 µs.
- Staging the extra 12 KB per route: ≈ +1.2 µs. Staging fills a CU at ~37 B/clk.
- Decode: ≈ +2.3 µs. It is VALU-issue-bound, and SIMD0 holds 3 of the 9 waves (0, 4, 8), so SIMD0 sets the time.
- Staging and decode did not overlap.
- Within a SIMD the oldest wave's loads are served first:
  - routes 0–3 have data at ~2.1 µs, 4–7 at 3.1–3.4 µs, route 8 at ~3.8 µs;
  - a lone route's decode takes ~1.5 µs.

**Steps that worked, in order** (µs; I=512 M1 / M2, I=256 M1 / M2 / M4):

| Step | Mechanism | Result |
|---|---|---|
| v1 | — | 7.69 / 8.09, 5.92 / 6.31 / 7.88 |
| original v2 | copies read from L2 | 18.63 / 20.55, 10.11 / 11.05 / 19.53 |
| r1/r2 | LDS DMA staging of the 16 KB book per route; R9T=4: 2 triplets per CTA at M ≥ 4 | 8.46 / 9.12, 6.21 / 6.54 / 9.19 |
| rf (R9RFL) | `readfirstlane` on expert id and row | 8.23 / 8.84, 6.18 / 6.56 / 8.98 |
| R9PRO, R9SR, R9ADDR, RB | branch-free prologue; scalar route; 32-bit triplet offset; one byte address `copy_offset \| index<<3` | 7.88 / 8.58 at I=512 |
| R9LAT, R9AL, R9SS | gain + route weights loaded with the prologue scalars; activation row DMA'd to LDS; scale row as scalar loads + `v_perm` | 7.41 / 8.30, 5.68 / 6.13 |
| R9PIPE | gather (LDS reads) of unit+1 before signs + MFMA of unit | 7.34 / 8.25, 5.59 / 6.13 / 8.34 |
| R9X=1 | route-8 split: wave 8 only stages; waves 0–3 decode its tiles after their own | 7.28 / 8.19, 5.82 / 6.28 / 8.23 |
| **gen7: R9GEN=3 + GPERM=1** | each route loads only its 4 KB base book and forms the other 3 copies in LDS with integer ops | **6.48 / 7.24, 5.41 / 5.52 / 7.41** |

- R9RFL matters a lot for its size. The aux pointer that builds the DMA buffer resource was not provably uniform.
  The compiler then wrapped the DMA in waterfall loops: ~160 extra instructions (82 `v_readfirstlane`, `v_cmp_eq_u64`,
  `s_and_saveexec`). `__builtin_amdgcn_readfirstlane` on expert and row removes them.
- R9X helps I=512 and I=256 M4 but costs 0.1–0.2 µs at I=256 M1/M2. It wants a per-shape choice.

**gen7 copy formation (`iq2r_v2_form_copies`, GPERM path).** For a normal E4M3 byte `b`, `E4M3(b · 2^(c/4))` is `b`
plus a delta that depends only on the 3 mantissa bits. Four bytes are done at once:

```cpp
m  = w & 0x07070707u;
r1 = w + 0x02020202u;                                            // class 2: x 2^(2/4)
r2 = w + __builtin_amdgcn_perm(0x04040404u, 0x05040403u, m);    // class 1: x 2^(1/4)
r3 = w + __builtin_amdgcn_perm(0x06060606u, 0x06060605u, m);    // class 3: x 2^(3/4)
```

- That is about 6 instructions per dword. The fp32 multiply + `cvt` path needs about 14, and fp8 `cvt` is half-rate.
- Exact for bytes 0x08..0x78 only: normal, no overflow into the exponent field past 0x7f.
- GenEarly (R9GEN=3):
  - every wave loads its 4 KB base (`gen_base[4]`, uint4 × 4 per lane) before any weight load;
  - routes 4–8 form their copies before issuing their weight loads, since those loads queue behind routes 0–3's
    anyway;
  - routes 0–3 form after.
- **Variants that lost:**

  | Build | Change | Result |
  |---|---|---|
  | gen1, gen4, gen5 | HW `cvt` path | ~100 VALU + 12 `ds_write` per lane per copy set on the critical path, more than the staging it saves. gen4: 7.75 / 8.44 at I=512 |
  | gen8 | GEN=2 + GPERM | 6.67 / 7.56 |
  | gen9 | GEN=3 without R9X | 6.99 / 7.57 |

**Route9 ISA, v2 vs v1 (rf build).** 1246 vs 1037 instructions, 685 vs 578 VALU. The v2 extras were v_or +54, v_perm
+28, and_or +28, readfirstlane +16, s_nop +36.

### 6.2 Single / ordered (M ≥ 8)

| Macro | Change | Result |
|---|---|---|
| S8KO | K-outer `glm53_down_tp4_triplet`: the 3 atoms share each K tile's activation fragment, level swap and scale bytes. The per-atom `((k0+k1)+k2)+k3` order is kept | I=512 M256 75.00 vs v1 76.03 |
| A32 | triplet source offset as 32-bit (`glm53_ws_source_base`); tp8 also takes `readfirstlane(threadIdx.y)` as wave | I=512 M256 71.21 |
| T8KO | tp8-ordered K-outer: K0 MFMAs of all 3 atoms, then the K1 MFMAs accumulate onto them | I=256 M256 72.97 → ~parity |
| T8LV | levels + scale bytes once per K tile, passed in (`given_levels` / `given_scales`) | I=256 M256 40.80 (v1 40.58) |
| LVM | mask + OR before the permlane swap: one mask instead of two | small; kept |

- After these, I=512 M4–256 beat v1: 10.98, 18.10, 26.64, 39.28, 56.45, 63.55, 71.21.
- The ordered4 ISA before T8LV had 603 VALU vs v1's 536. The extras were the gain multiply, perm / permlane / mov
  and and_or.

### 6.3 Full down sweeps vs v1

Builds:
- `cur_v2`: gen7 down, NT=0, NTT=0. These are the committed down defaults; NT only affects gate loads.
- `ntf`: NT=2.
- `nttf`: NT=2 + NTT=1 (nontemporal triplet loads).

Δ is vs `ref_v1`. Same session.

**I=512**

| M | ref_v1 | cur_v2 | Δ | ntf | Δ | nttf | Δ |
|---|---|---|---|---|---|---|---|
| 1 | 7.61 | 6.55 | −13.9% | 6.51 | −14.5% | 6.55 | −13.9% |
| 2 | 8.06 | 7.27 | −9.8% | 7.23 | −10.3% | 7.20 | −10.7% |
| 4 | 12.16 | 10.95 | −10.0% | 11.17 | −8.1% | 10.46 | −14.0% |
| 8 | 19.25 | 17.92 | −6.9% | 17.79 | −7.6% | 16.70 | −13.2% |
| 16 | 27.58 | 27.04 | −2.0% | 26.95 | −2.3% | 25.14 | −8.8% |
| 32 | 40.49 | 38.69 | −4.4% | 38.25 | −5.5% | 35.38 | −12.6% |
| 64 | 57.61 | 52.73 | −8.5% | 54.52 | −5.4% | 51.69 | −10.3% |
| 128 | 64.72 | 60.86 | −6.0% | 63.98 | −1.1% | 61.75 | −4.6% |
| 256 | 73.33 | 68.14 | −7.1% | 71.00 | −3.2% | 71.47 | −2.5% |
| 512 | 95.41 | 97.56 | +2.3% | 98.76 | +3.5% | 99.90 | +4.7% |
| 1024 | 121.55 | 124.32 | +2.3% | 124.59 | +2.5% | 126.22 | +3.8% |
| 2048 | 222.06 | 218.63 | −1.5% | 219.15 | −1.3% | 214.01 | −3.6% |
| 4096 | 388.63 | 393.46 | +1.2% | 392.02 | +0.9% | 386.06 | −0.7% |

**I=256**

| M | ref_v1 | cur_v2 | Δ | ntf | Δ | nttf | Δ |
|---|---|---|---|---|---|---|---|
| 1 | 5.96 | 5.42 | −9.1% | 5.39 | −9.6% | 5.42 | −9.1% |
| 2 | 6.30 | 5.54 | −12.1% | 5.55 | −11.9% | 5.53 | −12.2% |
| 4 | 7.94 | 7.36 | −7.3% | 7.34 | −7.6% | 7.34 | −7.6% |
| 8 | 11.96 | 11.35 | −5.1% | 11.15 | −6.8% | 10.83 | −9.4% |
| 16 | 17.36 | 14.95 | −13.9% | 15.06 | −13.2% | 14.38 | −17.2% |
| 32 | 24.34 | 23.81 | −2.2% | 23.99 | −1.4% | 22.89 | −6.0% |
| 64 | 32.12 | 31.72 | −1.2% | 32.07 | −0.2% | 30.57 | −4.8% |
| 128 | 37.31 | 38.00 | +1.8% | 37.86 | +1.5% | 35.20 | −5.7% |
| 256 | 39.16 | 39.67 | +1.3% | 39.63 | +1.2% | 39.95 | +2.0% |
| 512 | 45.06 | 45.63 | +1.3% | 45.85 | +1.8% | 49.32 | +9.5% |
| 1024 | 88.84 | 92.74 | +4.4% | 92.96 | +4.6% | 96.72 | +8.9% |
| 2048 | 165.69 | 168.79 | +1.9% | 168.93 | +2.0% | 168.72 | +1.8% |
| 4096 | 298.43 | 306.35 | +2.7% | 301.71 | +1.1% | 305.29 | +2.3% |

**Verdict.**
- NTT=1 helps I=512 M4–64 and I=256 M8–128. It hurts I=256 M512/1024 (+8..+10%).
- It needs a per-launch flag in the tuned CSV. Until then the default is NTT=0.
- Prefill down (M ≥ 512) was not worked on in this study.

---

## 7. Gate kernels

### 7.1 Early steps

| Step | Mechanism | Result | Verdict |
|---|---|---|---|
| M1 fix | `glm53_gate_m1_kernel` stages the book by async LDS DMA alongside the first quad | I=512 M1 13.60 → 9.27 µs (v1 8.71) | kept |
| s4 | nobarrier gate: gains in SGPRs + one `d` word per K256. 132 → 128 VGPRs, so 2 CTAs/CU again | I=512 M128 132.2 → 104.3, M256 136.9 → 111.8; I=256 M32 47.13 → 40.13 | kept |
| s7/s8 | copy offset OR-ed into the existing address op; hoisted `d` pointer; VALU per K128 156 → 142 (v1 124) | neutral (±3%) | kept |
| contiguous item runs per CTA | reuse the book across consecutive items | slower for v1 and v2; breaks activation L2 sharing | dropped |

- Lowering VALU did not speed up v1 either. So the loop is not purely VALU-bound.

**x-series** (timing-only, gate µs; I=512 M2 / M128, I=256 M32 / M512):

| Build | I=512 M2 | M128 | I=256 M32 | M512 |
|---|---|---|---|---|
| v1 | 11.72 | 93.50 | 33.46 | 76.23 |
| s8 | 12.91 | 102.16 | 38.75 | 84.92 |
| x1: 8 KB book | 12.07 | 94.81 | 36.48 | 79.99 |
| x2: copy offset forced 0 | 12.24 | 95.66 | 36.45 | 81.47 |
| x3: no `d` loads | 12.82 | 100.59 | 37.98 | 84.38 |

- x6, in-kernel copy generation with fp32 multiply + `cvt`: correct but slower than loading 32 KB.
- x8w, activation prefetch one step ahead: v1 got 3–6% faster at M ≤ 128 (I=512). v2 spilled at 131 VGPRs and got
  worse.

**a-series** (gate µs, I=512): carried levels (a3) was a small, consistent win and became unconditional. The other
two options:
- epilogue prefetch (a4): small win at I=512, neutral at I=256;
- grouping items by expert (a5): hurt at medium and large M.

| M | v1 | a0 base | a3 carried | a4 prefetch | a5 grouping | a2 all |
|---|---|---|---|---|---|---|
| 16 | 40.78 | 41.38 | 41.05 | 41.40 | 44.44 | 45.83 |
| 64 | 84.73 | 89.24 | 86.34 | 87.33 | 89.74 | 90.29 |
| 128 | 94.81 | 100.41 | 96.68 | 98.64 | 98.87 | 99.25 |
| 256 | 103.01 | 111.91 | 109.66 | 109.98 | 115.60 | 107.34 |

**y-series.**
- y1 simulated a 16 KB book: parity only at I=512 M ≤ 16.
- Copy select was worth 1–5% and `d` loads 1–3%.
- y2, a double-buffered book, was worse. So the cost is bandwidth / LDS traffic, not latency.

### 7.2 Same-list attribution and variants

Every row below is a rebuild with the named macros, timed at M16 / M64 / M256 on fixed GPUs.
- Rows marked MIS are timing-only.
- `n_*` rows repeat the attribution on top of NT=2.
- The trailing number in parentheses is the row's Δ vs `l3_v1` at M256.

| Variant | Mechanism | I=512 | I=256 |
|---|---|---|---|
| l3_v1 | v1 reference | 38.51 / 79.26 / 97.04 | 24.30 / 45.64 / 54.52 |
| base3 | gate as in `cur_v2` | 43.10 / 89.56 / 115.51 (+19.0%) | 26.08 / 50.13 / 59.99 |
| stg4 (MIS) | stage only 4 KB | 39.95 / 83.76 / 107.21 | 25.12 / 48.60 / 56.40 |
| stg0 (MIS) | always stage expert 0's aux (L2-hot) | 41.52 / 87.66 / 110.52 | 26.03 / 49.67 / 58.95 |
| nocd2 (MIS) | no copy select, no `d` | 40.04 / 82.64 / 107.92 | 25.06 / 47.69 / 56.82 |
| stg4cd (MIS) | all three | 38.99 / 80.59 / 104.05 | 23.49 / 45.34 / 54.85 |
| pf | touch next item's book into L2 | 56.81 / 117.47 / 140.23 | 33.05 / 64.28 / 75.94 |
| nt2 | NT=2 (aux operand `nt` on quad loads) | 40.11 / 85.44 / 109.98 | 24.85 / 48.58 / 57.93 |
| nt3 / nt18 / nt16 | sc0+nt / nt+sc1 / sc1 | 109.80 / 109.16 / 114.70 at M256 | no better than nt2 |
| n_all (MIS) | NT=2 + 4 KB book + no copy select + no `d` | 36.86 / 75.23 / 97.87 | 22.65 / 43.31 / 53.97 |
| geb | next item's book DMA in the epilogue | 40.69 / 84.57 / 108.93 | 24.73 / 47.98 / 57.94 |
| xq | previous item preloads this item's first quad; waves 4–7 stage its book | 54.15 / 112.12 / 133.83 | 32.29 / 61.56 / 72.93 |
| pal | partials alias the book buffer → two book buffers fit | 42.04 / 87.43 / 111.87 | 25.69 / 50.65 / 59.32 |
| **opsel** | scaled MFMA `op_sel` picks the atom's byte of the scale word; activation scale not replicated ×0x01010101 | 39.58 / 82.96 / 104.99 | 25.01 / 48.87 / 57.67 |
| unr2 / unr6 | unroll the 6-iteration K loop by 2 / 6 | 108.98 / 130.22 at M256 | worse |
| ggo | OPSEL + GG=1: load copy 0 (8 KB for 2 books), form copies 1–3 in kernel; `launch_bounds(512,4)` | 39.54 / 82.79 / 106.97 | 23.52 / 47.50 / 57.81 |
| **gg2o** | GG=2: form right after the base load, so copy 0 is not live across other loads | 39.23 / 81.59 / 102.67 (+5.8%) | 24.09 / 47.26 / 56.61 |
| gef | form the next expert's book in the epilogue | 40.21 / 85.01 / 110.51 | 25.68 / 49.62 / 60.87 |
| dpre | preload the item's 3 `d` words | 40.33 / 85.57 / 105.36 | 25.25 / 49.34 / 57.53 |
| chunk2 | CTA takes 2 adjacent n-tiles of a task, sharing one staging | 42.29 / 83.16 / 104.96 | 23.37 / 46.56 / 58.83 |
| gg3 | load the base after gather + first quad | 39.55 / 80.93 / 103.64 | 23.73 / 46.67 / 56.11 |
| gef3 | waves 4–7 load the next expert's copy 0 under partial stores, form after the barrier | 39.33 / 82.37 / 105.07 | 25.21 / 47.73 / 57.12 |
| blk | contiguous run of the XCD's items per CTA | 38.67 / 82.53 / 112.58 | 24.08 / 46.01 / 57.87 |
| p1 (MIS) | book never staged | 38.66 / 78.03 / 100.01 | 23.11 / 45.49 / 54.73 |
| p2 (MIS) | + no `d` | 37.75 / 77.14 / 97.63 | 22.63 / 44.62 / 54.27 |
| p3 (MIS) | + no copy select | 36.31 / 73.93 / 95.56 | 22.04 / 42.59 / 52.97 |
| ldsb6 | waves 4–7 DMA the next expert's copies 0 into an 8 KB LDS stage during the K loop and form in the epilogue (spill fixed) | 39.11 / 81.50 / 103.52 | 23.75 / 46.51 / 56.61 |
| **abuf** | K loop loads activations, scales and `d` through buffer descriptors with the K offset in an SGPR; quad offset scalar; no per-iteration 64-bit address math | **39.24 / 80.20 / 103.51 (+6.7%)** | **23.32 / 46.63 / 56.15 (+3.0%)** |
| pf2 | two quads in flight | 40.64 / 85.74 / 109.23 | 25.26 / 48.48 / 58.69 |
| nod3 (MIS) | abuf + no `d` | 39.22 / 78.69 / 99.66 | 23.03 / 45.92 / 55.07 |
| nof3 (MIS) | abuf + no book | 37.76 / 77.22 / 100.20 | 22.80 / 45.04 / 55.32 |

Δ vs `l3_v1` at M16 / M64 / M256:

| Variant | I=512 | I=256 |
|---|---|---|
| base3 | +11.9 / +13.0 / +19.0% | +7.3 / +9.8 / +10.0% |
| n_base (NT=2) | +5.6 / +7.2 / +13.6% | +3.7 / +7.5 / +6.3% |
| opsel | +2.8 / +4.7 / +8.2% | +2.9 / +7.1 / +5.8% |
| gg2o | +1.9 / +2.9 / +5.8% | −0.9 / +3.5 / +3.8% |
| **abuf** | **+1.9 / +1.2 / +6.7%** | **−4.0 / +2.2 / +3.0%** |
| n_all (MIS) | −4.3 / −5.1 / +0.9% | −6.8 / −5.1 / −1.0% |
| p3 (MIS) | −5.7 / −6.7 / −1.5% | −9.3 / −6.7 / −2.8% |

- The rmse vs the dense reference is unchanged across all OPSEL / GG / ABUF builds: I=512 0.0115 / 0.0126 / 0.0142;
  I=256 0.0047 / 0.0127 / 0.0140.
- gg2o and abuf are within noise of each other at M256 (102.67 vs 103.51, different sessions). abuf is the default
  because it was the best at M64 and at I=256 and it frees VGPRs (126 → 114).

### 7.3 Full gate sweeps vs v1 (before OPSEL / GG / ABUF)

Builds: `cur_v2` (PFA=1, GG=0, NT=0) and `ntf` (NT=2). Δ vs `ref_v1`. There is **no full-M sweep of the final abuf
build**; only M16 / M64 / M256 were run.

| M | I=512 ref_v1 | cur_v2 | ntf | I=256 ref_v1 | cur_v2 | ntf |
|---|---|---|---|---|---|---|
| 1 | 8.55 | 9.25 (+8.2%) | 8.86 (+3.6%) | 8.29 | 8.85 (+6.8%) | 8.60 (+3.7%) |
| 2 | 11.51 | 12.88 (+11.9%) | 12.36 (+7.4%) | 8.61 | 9.51 (+10.5%) | 9.15 (+6.3%) |
| 4 | 13.73 | 15.19 (+10.6%) | 14.10 (+2.7%) | 9.64 | 10.56 (+9.5%) | 9.98 (+3.5%) |
| 8 | 23.71 | 26.30 (+10.9%) | 24.73 (+4.3%) | 14.19 | 15.52 (+9.4%) | 14.64 (+3.2%) |
| 16 | 36.91 | 41.79 (+13.2%) | 40.10 (+8.6%) | 24.03 | 25.88 (+7.7%) | 24.72 (+2.9%) |
| 32 | 55.94 | 62.95 (+12.5%) | 60.61 (+8.3%) | 33.10 | 37.93 (+14.6%) | 35.73 (+7.9%) |
| 64 | 76.92 | 87.21 (+13.4%) | 84.23 (+9.5%) | 44.93 | 49.80 (+10.8%) | 47.33 (+5.3%) |
| 128 | 87.75 | 99.66 (+13.6%) | 95.52 (+8.9%) | 49.12 | 54.42 (+10.8%) | 51.85 (+5.6%) |
| 256 | 94.93 | 113.02 (+19.1%) | 105.45 (+11.1%) | 53.19 | 58.67 (+10.3%) | 56.70 (+6.6%) |
| 512 | 136.30 | 150.10 (+10.1%) | 147.99 (+8.6%) | 75.52 | 81.88 (+8.4%) | 85.57 (+13.3%) |
| 1024 | 147.16 | 157.40 (+7.0%) | 159.31 (+8.3%) | 98.15 | 106.27 (+8.3%) | 105.75 (+7.7%) |
| 2048 | 208.14 | 221.70 (+6.5%) | 228.30 (+9.7%) | 121.53 | 128.16 (+5.5%) | 130.08 (+7.0%) |
| 4096 | 330.35 | 347.93 (+5.3%) | 366.61 (+11.0%) | 186.96 | 198.72 (+6.3%) | 205.13 (+9.7%) |

- NT=2 helps the decode gate (up to 8 points at I=512) but hurts the prefill gate and I=256 M512. It also wants a
  per-launch choice.
- The prefill gate (`glm53_gate_prefill_kernel`, M ≥ 512) is +5..+10% without NT and was not worked on beyond PFA.
  PFA stages the book by async DMA instead of a VGPR copy loop; it is bit-exact. PFA prefill gate µs:
  I=512 M512–4096 150.00 / 158.42 / 221.83 / 352.83; I=256 M1024–4096 107.15 / 129.14 / 199.38.

### 7.4 Where the remaining gate time goes (I=512 M256)

**Phase cycles per item (GTS=1, `s_memtime`, setup / K loop / epilogue, ±1000).**

| Build | setup | K loop | epilogue | note |
|---|---|---|---|---|
| v1 | 2598 | 14234 | 3539 | total 20371 |
| v2 base | 3748 | 15421 | 4023 | total 23192 |
| stg4 (MIS) | 2566 | 15315 | 3983 | |
| geb | 2502 | 15031 | 5303 | |
| xq4 | 1195 | 18485 | 5988 | |
| pal | 882 | 17359 | 4775 | ≈ 23016 |
| p0 (gg2o state) M256 | 3378 | 13952 | 3851 | |
| ldsb M256 | 822 | 17377 | 4212 | |

- **Moving latency does not remove it.** The two co-resident CTAs per CU share VALU and LDS (each ~60% busy).
  Taking book staging out of setup (ldsb: setup 3378 → 822) made the K loop slower by the same amount
  (13952 → 17377).
- Staging costs ~1200 cycles per item wherever it is placed, unless its volume is cut.
- With OPSEL, the v2 K loop is at or below v1's cycles. The rest of the gap is:
  - setup (+~800 cycles) and epilogue (+~400);
  - 9 vs 8.5 items per CTA.

**Attribution.** Over a stripped v2 (p3, 95.56 µs) that beats v1 (97.04), the gap was:
- formation ≈ 3.2 µs;
- `d` loads ≈ 2.4 µs;
- copy-offset VALU ≈ 2.1 µs.

After ABUF:
- without `d` reloads: 103.5 → 99.7 (~3.7%);
- without book load and formation: M16 → 37.8, M64 → 77.2 (~2–4%);
- without both: 97.87 ≈ v1.

**Gate inner loop, VALU per K256 quad.**
- v1 110.
- v2 before OPSEL / ABUF: 139. The PALIAS ISA suggested +29, but the real figure was +13.
- v2 with ABUF: **110**. VGPRs 126 → 114.
- Opcode diff v1 → v2 (ABUF):
  - v_add_u32 5 → 20, v_and_or 35 → 47, v_perm 0 → 8, v_and 14 → 4, lshl_or 0 → 3, mov 9 → 12;
  - permlane16_swap 0 → 1, global_load_dword 0 → 1 (the `d` word).

**rocprof counters, nobarrier, I=512 M256, v1 vs v2 (before ABUF).**

| Counter | v1 | v2 |
|---|---|---|
| VALU | 2.92e7 | 2.92e7 |
| VMEM_RD | 1.35e6 | 1.62e6 |
| SALU | 2.75e6 | 3.86e6 |
| TCC_HIT | 2.39e6 | 3.44e6 |
| TCC_MISS | 3.68e6 | 3.86e6 |
| SQ_WAIT_INST_ANY | 4.80e7 | 5.56e7 |
| LDS bank conflicts | 1.55e7 | 1.70e7 |
| TCP_TCC_READ_REQ | 5.86e6 | 7.09e6 (+21%) |

- The extra TCP→TCC reads are the per-item book (32 KB) and the per-K `d` words.

### 7.5 Why `d` costs ~3.7%, and DBC

- In the K loop every lane group loads the `d` word for its (row band, K256). The same word is requested by four
  lane groups, so only 16 of 64 requests are unique.
- It is a separate load stream: 6 loads per wave per item. v1 has no such stream; its base is per item.
- **DBC (written, never built; `GLM53_X_DBC`, default 0).** Lanes `16c+r` load column `c`, row `r`. That is 48 unique
  words in one load per item, then `permlane16_swap` + `permlane32_swap` broadcast each column to all four rows:

  ```cpp
  const uint32_t x = iq2r_scheduled_buffer_load1(d_rsrc, BookWords*8 + min(lane,47)*4, (n_tile*24 + swave*3)*64, 0);
  const auto pairs = __builtin_amdgcn_permlane16_swap(x, x, false, true);
  const auto even  = __builtin_amdgcn_permlane32_swap(pairs[0], pairs[0], false, true);
  const auto odd   = __builtin_amdgcn_permlane32_swap(pairs[1], pairs[1], false, true);
  d_cols[0] = even[0]; d_cols[1] = odd[0]; d_cols[2] = even[1];
  // K loop: bases = iteration < 2 ? d_cols[0] : iteration < 4 ? d_cols[1] : d_cols[2];
  ```

  It costs 3 VGPRs across the K loop. The 114-VGPR ABUF build has room under the 128 limit.
- **Alternative: interleave `d` into the records.** That is a format change and needs agreement first.

---

## 8. Best configuration vs v1

**Gate (committed defaults = abuf build), same session.**

| | I=512 M16 | M64 | M256 | I=256 M16 | M64 | M256 |
|---|---|---|---|---|---|---|
| v1 (`l3_v1`) | 38.51 | 79.26 | 97.04 | 24.30 | 45.64 | 54.52 |
| v2 abuf | 39.24 | 80.20 | 103.51 | 23.32 | 46.63 | 56.15 |
| Δ | +1.9% | +1.2% | +6.7% | −4.0% | +2.2% | +3.0% |
| v2 at the study start | | | 136.9 | | | 74.87 |

- The M1 gate (`gate_m1`, GG=2) and the prefill gate were not re-timed after GG / ABUF.
- The last full-M numbers are the `ntf` column of section 7.3.

**Down, best per M.** The "committed" column is NTT=0 (`cur_v2` / `ntf`). The "best" column picks NTT per M, which
the tuned CSV cannot express yet.

| M | I=512 v1 | committed | best | Δ best | I=256 v1 | committed | best | Δ best |
|---|---|---|---|---|---|---|---|---|
| 1 | 7.61 | 6.48–6.55 | 6.48 | −14.8% | 5.96 | 5.39–5.42 | 5.39 | −9.6% |
| 2 | 8.06 | 7.23–7.27 | 7.20 | −10.7% | 6.30 | 5.52–5.55 | 5.52 | −12.4% |
| 4 | 12.16 | 10.95 | 10.46 | −14.0% | 7.94 | 7.34–7.41 | 7.34 | −7.6% |
| 8 | 19.25 | 17.79 | 16.70 | −13.2% | 11.96 | 11.15 | 10.83 | −9.4% |
| 16 | 27.58 | 26.95 | 25.14 | −8.8% | 17.36 | 14.95 | 14.38 | −17.2% |
| 32 | 40.49 | 38.25 | 35.38 | −12.6% | 24.34 | 23.81 | 22.89 | −6.0% |
| 64 | 57.61 | 52.73 | 51.69 | −10.3% | 32.12 | 31.72 | 30.57 | −4.8% |
| 128 | 64.72 | 60.86 | 60.86 | −6.0% | 37.31 | 37.86 | 35.20 | −5.7% |
| 256 | 73.33 | 68.14 | 68.14 | −7.1% | 39.16 | 39.63 | 39.63 | +1.2% |
| 512 | 95.41 | 97.56 | 97.56 | +2.3% | 45.06 | 45.63 | 45.63 | +1.3% |
| 1024 | 121.55 | 124.32 | 124.32 | +2.3% | 88.84 | 92.74 | 92.74 | +4.4% |
| 2048 | 222.06 | 218.63 | 214.01 | −3.6% | 165.69 | 168.79 | 168.72 | +1.8% |
| 4096 | 388.63 | 392.02 | 386.06 | −0.7% | 298.43 | 301.71 | 301.71 | +1.1% |

- **Re-check after the commit.** The committed 881960d84 defaults were rebuilt in 8.7 s on a warm ccache. Route9
  down reproduced: I=512 M1 6.50 µs (6.48 logged), M2 7.14 (7.24).
- **No complete-MoE number exists** (gate + down + sort + reduce) for the final configuration. Do not quote one.

---

## 9. Format ablations (weight error)

These ablations measure weight error, not kernel time. They were run to decide how much format change the kernels
may assume.

**Setup.** 36 matrices: 12 real GLM-5.3 experts from layers 10 / 40 / 70, gate / up / down each. The metric is
importance-weighted relative error after the best output gain. Scripts are in
`docs/iq2r_v2_kernel_study/class_ablation/`.

**Classes per codebook (copies per book).**

| Classes | Error | vs C4 |
|---|---|---|
| C8 | 0.0622 | −2.2% |
| C4 (format) | 0.0636 | — |
| C2 | 0.0671 | +5.6% |
| C1 (pure power of two) | 0.0741 | +16.6%, worse on every matrix |

**Book sharing (error vs C4 with own books).**

| Variant | Error vs C4 |
|---|---|
| b4: one book shared by gate and up | +0.10% gate, +0.11% up |
| i4: 4 independent books (no class copies) | −3.8% |
| bi4: shared + independent | −3.4% |
| bi2h | +3.48% |
| b2 | +5.4% |
| i2q | +5.0% |
| bi2q | +5.1% |
| c2q | +7.7% |

- Every 2-book variant is ≥ +3.5%.
- The shared gate/up book is the basis of `0d7a0963a`. A full-model check found 0.09% loss.

**Rejected format ideas.**
- Fold the class into the index: +0.25 bpw.
- Duplicate classes per lane: +2.8% bytes.
- Class as a per-row gain, or scaling the activations: invalid, because the class varies along K.

**Real codebook byte range.** The 36 learned books span bytes 0x21..0x4f (smallest value 0.14). That is inside the
GPERM range 0x08..0x78.

---

## 10. Macro catalog

Status key:
- **K** = kept, on in the defaults.
- **R** = tried and rejected.
- **P** = timing probe; its output is wrong on purpose.
- **U** = written, never built.

µs: I=512 unless noted.

**Load path**

| Macro | Default | What | Status |
|---|---|---|---|
| NTT | 0 | triplet loads via `__builtin_nontemporal_load` (v1 too) | per-shape win; needs per-launch choice |
| NT | 2 | aux / cache-policy operand of the gate quad's buffer loads (v1 too) | K: M256 110.0 vs 114.5 |
| A32 | 1 | 32-bit triplet source offset | K |

**Levels, copies, signs, scales**

| Macro | Default | What | Status |
|---|---|---|---|
| LVM | 1 | mask + OR before the permlane swap | K |
| GPERM | 1 | integer copy formation (bytes 0x08..0x78); 0 = fp32 multiply + `cvt` | K |
| RB | 1 | route bits + copy offset + index as one byte address | K (0 no longer compiles) |
| SGN | 0 | `__umul24` / asm `v_mul_u32_u24` sign spread | R |
| OPSEL | 1 | MFMA scale `op_sel` picks the atom's byte | K |
| NOCOPY | 0 | copy offsets forced to 0 | P |
| PRB, DND, S8ND, R9ND | 0 | no signs / no LDS read / no decode | P |

**Down M ≥ 8**

| Macro | Default | What | Status |
|---|---|---|---|
| S8OCC | 3 | occupancy of the v2 single down | K (v1 value) |
| S8W | 0 | stage only part of the book | P |
| S8IL | 1 | decode + MFMA interleaved per K tile | superseded by S8KO |
| S8NB | 0 | drop the `m < active_m` check | R |
| S8KO | 1 | K-outer triplet (atoms share activation, levels, scales) | K |
| T8KO | 1 | tp8-ordered K-outer | K |
| T8LV | 1 | levels + scales once per K tile | K |

**Route9 (M ≤ 4)**

| Macro | Default | What | Status |
|---|---|---|---|
| R9T | 4 | 2 triplets per CTA when tokens ≥ R9T (v1 too) | K |
| R9W | 0 | stage only part of the book | P |
| R9ROT | 0 | rotated DMA chunk order | R |
| R9VG | 0 | stage through VGPRs | R |
| R9SLP | 0 | `s_sleep` × route before staging | R |
| R9RFL | 1 | `readfirstlane` expert / row (no waterfall) | K |
| R9DROP | 0 | drop one route | P |
| R9IDLE | 0 | 12 waves, 3 idle | R |
| R9BAR | 0 | extra `__syncthreads` around decode | R |
| R9HELP | 0 | 3 helper waves decode the last K tiles of routes 0 / 4 / 8 | R |
| R9EB | 0 | extra `s_barrier` after staging | R |
| R9SR | 1 | `route = readfirstlane(threadIdx.y)` | K |
| R9ADDR | 1 | direct 32-bit triplet offset | K |
| R9STG, R9ORD, R9LATE | 0 | book DMA after data loads, staggered / SIMD-ordered / unstaggered | R |
| R9PRO | 1 | branch-free validity in the prologue | K |
| R9S8 | 0 | 8-wave CTA, route 8's units round-robin | R |
| R9X | 1 | route-8 split (wave 8 stages; waves 0–3 decode its tiles) | K (hurts I=256 M1/M2 a little) |
| R9GEN | 3 | in-kernel copies; 3 = prologue load, routes 4–8 form before their data loads | K (gen7) |
| R9AL | 1 | activation row → LDS by one DMA per route | K |
| R9AX | 0 | reuse K-tile-0 activation / scale | P |
| R9SS | 1 | scale row as scalar loads + `v_perm` | K |
| R9LAT | 1 | gain + route weights with the prologue scalars | K |
| R9GRP | 0 | issue loads one SIMD group at a time | R |
| R9PIPE | 1 | gather of unit+1 before signs + MFMA of unit | K |
| R9SB | 1 | `iq2r_scale_bytes` on the non-pipe path | moot with PIPE |
| R12 | 0 | `glm53_down_route12_kernel` (12 waves) | R |
| TS | 0 | per-wave timestamps | P |

**Gate**

| Macro | Default | What | Status |
|---|---|---|---|
| GG | 2 | form the gate book in-kernel from the copy-0 bases (gate_m1 + nobarrier) | K |
| PFA | 1 | prefill gate book by async DMA | K (bit-exact) |
| CHUNK | 1 | adjacent n-tiles per CTA | K at 1 |
| BLK | 0 | contiguous XCD item runs per CTA | R |
| XQ | 0 | previous item preloads this item's first quad + book | R |
| GTS | 0 | per-phase cycle stamps | P |
| NOFORM, STG4, STG0, NOD | 0 | no book / base only / fake data; 4 KB stage; expert-0 aux; `d` = 0x70707070 | P |
| ABUF | 1 | buffer-descriptor K loop with SGPR offsets | K |
| DBC | 0 | one `d` load per item + permlane broadcast | U |
| PF2 | 0 | two quads in flight | R |
| DPRE | 0 | preload the item's `d` words | R |
| LDSB | 0 | next expert's base to an 8 KB LDS stage during the K loop | R |
| PF | 0 | L2 touch of the next book | R |
| UNR | 1 | K-loop unroll | K at 1 |
| ORDER | 0 | `sched_barrier` after the `d` select | R |
| GEF | 0 | form the next expert's book in the epilogue | R |
| GEB | 0 | next expert's book DMA in the epilogue | R |
| PALIAS | 0 | partials alias the book → double buffer | R |
| `GLM53_GATE_BOOKS` | 2 | books in the nobarrier gate (1 was a timing probe here; it is the format on the one-book branch) | — |
| `GLM53_GATE_DOUBLE` | 0 | two LDS book buffers | R |

---

## 11. gfx950 microarchitecture facts

These were measured or observed in this study and in the FP11 study (Appendix B).

**Instruction rates** (`ubench/rate.hip`)
- `v_lshlrev_b64`, `v_and_or_b32` and `v_perm_b32` issue at the same rate.
- `v_permlane16_swap` costs 2× a normal VALU op.
- fp8 `cvt` (`cvt_pk_f32_fp8`, `cvt_pk_fp8_f32`) is half-rate.
- `v_mul_lo_u32` is quarter-rate. It does the sign spread (96–100 per route9 kernel, v1 and v2 alike). Replacing it
  lost: LLVM folds `(n*C)<<7` into it, and inline asm adds `s_nop`s.
- MFMA 16x16x128 f8f6f4 = 32 cycles.
- Wave64 VALU issue ≈ 4–5 cycles at 1 wave/SIMD.
- A dependent VALU → DS address costs +6–7 cycles.

**Scaled MFMA (`v_mfma_scale_f32_16x16x128_f8f6f4`)**
- One E8M0 byte per lane for A and for B; lane = 16·g + r.
- `op_sel` selects which byte of the scale register is used. So a packed word of 4 atom scales needs no
  `×0x01010101` replicate or extract (OPSEL).
- `op_sel` per atom:

  | Atom | Modifiers |
  |---|---|
  | 0 | `op_sel_hi:[0,0,0]` |
  | 1 | `op_sel:[0,1,0] op_sel_hi:[0,0,0]` |
  | 2 | `op_sel_hi:[0,1,0]` |
  | 3 | `op_sel:[0,1,0] op_sel_hi:[0,1,0]` |

**Occupancy**
- The nobarrier gate (512 threads, 8 waves) needs ≤ 128 VGPRs for 2 CTAs/CU. Above 128 it drops and is slower.
- The second `__launch_bounds__` argument is the minimum waves per EU (SIMD), not blocks per CU. `(512,2)` does
  nothing; `(512,4)` gives 2 CTAs/CU.
- Co-resident CTAs are throughput-coupled (section 7.4). Moving latency between phases does not remove it.

**Memory and LDS**
- The I=512 gate at M256 reads ~4.6 TB/s, close to memory-bound.
- Contiguous item runs hurt DRAM/TLB locality and activation L2 sharing.
- A whole CU fills LDS by DMA at ~37 B/clk.
- LDS: random b64 gathers ~72 B/clk per CU; linear b128 ~245 B/clk.
- Within a SIMD, the oldest wave's loads are served first (section 6.1).
- vmcnt completes in order and counts stores. Scratch spills force `vmcnt(0)`.
- `__threadfence` is a whole-L2 writeback.

**Codegen**
- A non-uniform pointer feeding a buffer resource creates waterfall loops. Use `__builtin_amdgcn_readfirstlane` on
  the pointer's inputs.
- Hoisted lane-derived invariants spill under VGPR pressure. Force rematerialization with
  `int l = lane; asm volatile("" : "+v"(l));`.
- VOP3 literal selectors cost a VGPR. `v_perm` selectors in SGPRs saved 4 VGPRs.
- In-order vmcnt makes the MFMAs wait for every outstanding load. With about one quad in flight per wave, VALU between
  quad arrival and the next issue costs bandwidth.
- In v1 and v2 the next quad's atom-0 LDS lookups issue at atom 3. So a quad prefetch must land within the same
  iteration.
- **Inline asm** is invisible to the hazard recognizer, so every asm → asm dependent pair gets an `s_nop 0`. A
  `"+v"` anchor on a scale register put an `s_nop` before every MFMA (worth 6–11 µs in the FP11 stream gate).
- **LLVM 22 (ROCm 7.2) machine-sinks scaled MFMAs into lane-dependent branches** (llvm/llvm-project#178627): the
  output turns NaN. Keep every MFMA's use and every store branch-free. v1 uses out-of-bounds buffer offsets for
  invalid rows (`glm53_row_offset`).

---

## 12. Correctness

- **`vbench.py CHECK=1`** compares each run with a dense reference decode and prints `rmse=`.
  - **Known:** I=256 M1 reports MISMATCH: rmse 0.0317 against a 0.03 threshold, cosine 0.99952.
  - It appears in every build, including the original v2 kernels with the same seeded routes. So it is a
    borderline test case, not a regression. Not investigated.
- **Bit-exact checks: `DUMP=prefix` + `cmpdump.py A B`.** `torch.equal` on int16 views, over I 512/256 × M 1, 2, 4.
  - gen7 vs x1 (staged copies): bit-exact on all 6 shapes.
  - PFA on vs off: bit-exact.
  - **Not done: OPSEL, GG=2 and ABUF.** Only their rmse is unchanged. Run cmpdump on them before porting.
- **Op tests:** v2 67 passed / 7 skipped; v1 main 90 / 10; compile tests 10.
- **GPERM / GG precondition: every codebook byte must be in 0x08..0x78** (normal E4M3, < 2^8).
  - The learned books span 0x21..0x4f, so real checkpoints pass today.
  - But the encoder only clamps to [0, pin] (`aiter/ops/iq2r_v2.py`). It guarantees the upper bound, not the lower.
  - Before formation can ship, either clamp the codebook to ≥ 2^-6 at encode time or check at load time and fall
    back to staged copies.

---

## 13. Dead ends

These were measured. Do not retry them without a new reason.

**Gate**
- Contiguous item runs per CTA (s-series, BLK): about 2× slower or worse at M256.
- Grouping items by expert: hurts at medium and large M.
- CHUNK=2 / 4: worse at small M (CHUNK=4: M2 27.20, M256 116.59).
- Moving book staging elsewhere: GEB, XQ (+40%), GEF 1/2/3, LDSB, PALIAS double buffer, y2 double buffer, GG=3.
  All of them move the cost without removing it, because co-resident CTAs are throughput-coupled.
- L2 touch prefetch (PF): +30–45%. Two quads in flight (PF2): +5%.
- DPRE: neutral to worse.
- UNR=2 / 6: worse; 6 is +28–34%.
- NT=3 / 16 / 18: no better than NT=2.
- GG=1 without `launch_bounds(…,4)`: 133 µs at M256.
- In-kernel copies via fp32 multiply + `cvt` (x6, gen1): slower than loading.
- Activation prefetch (x8w) for v2: spills.

**Route9**
- T=2 always, rotated chunk order, VGPR staging, `s_sleep` stagger, route12, R9BAR / HELP / EB / IDLE / STG / GRP,
  SGN 1/2.
- Replacing `v_mul_lo_u32` in the sign spread.
- gen1 / gen4 / gen5 (HW `cvt` copy formation).

**Formats.** C1 / C2, every 2-book variant, class folded into the index, duplicated classes (section 9).

**Process lesson.** Once the masked probe (n_all) showed that removing the book and `d` reaches v1, the right next
step was to raise the format options (shared book, `d` placement). Instead, many book-placement variants were tried,
and they can only move cost around. When a probe isolates a cost that only a format change removes, escalate it.

---

## 14. Porting to the one-book format

The v2 branch (`glm-5.3-iq2r-v2`) moved on after `a5a477367`:
- `0d7a0963a`: one codebook for gate and up (the only kernel change, 93 lines);
- `5749aaf01`: shared expert kept in FP8 (Python / docs / tests);
- `9f3c93b83`: encoder speedup (Python).

**A trial 3-way merge** (base `a5a477367`, ours = one-book tip, theirs = this branch) gives **7 conflict hunks:**
- `iq2r_v2_bases` vs `iq2r_v2_bases_at`;
- 4 `iq2r_stage_codebook_async` call sites (down tp4 single, prefill tp4, prefill tp8, tp8-ordered). The one-book
  branch keeps the run-time `waves` argument; this branch uses `<Words,Waves>`;
- nobarrier `GateBooks` / `BookWords`;
- nobarrier `lane_book` / `d_lane`.

**Breaks that merge cleanly but do not compile:**
- **`iq2r_levels`.** The one-book branch removes its second argument and `iq2r_book_bits`. This branch passes it from
  `glm53_decode_packed_atom` (`book_bits`), route9 (`rb_bits`, and `Bits8` for route 8), the carried decode and the
  nobarrier gate.
  - Route9 RB and R9X reuse that argument to carry route bits. So keep a generic `or_bits` parameter rather than a
    book selector.
- **Template arguments.** `iq2r_codebook_words<Classes,Books>` (8 uses) becomes `<Classes>`;
  `iq2r_v2_bases<Classes,Books>` becomes `<Classes>`.
- **Gate formation helpers** (`iq2r_v2_gate_base_load` / `gate_form` / `gate_base_dma`) are hard-coded to 2 books:
  8 × 1 KB chunks with a `(c>>2)*1024` book stride. One book needs 4 chunks and no stride. Every `GateBooks==2 &&`
  guard (GG, NOFORM, LDSB, GEF) must become generic, or be dropped along with the rejected variants.
- **Aux offsets.** Gate `d` and gain offsets follow `BookWords*8`. Re-check `d_rsrc` / `d_off` and the DBC load
  offset.

**Suggested order.** One commit per step. v1 output must stay bit-exact. v2 output must match the staged-copy dumps.
1. Shared helpers:
   - `iq2r_stage_codebook_async<Words,Waves>`;
   - `iq2r_gains` / `iq2r_row_gain`, `iq2r_v2_bases_at`;
   - OR-joined `glm53_issue_codebook`;
   - `glm53_sign_pair` / `glm53_sign_word`;
   - LVM;
   - `iq2r_buffer_rsrc` / `iq2r_buffer_load_u8`.
2. Down M ≥ 8: S8KO (`glm53_down_tp4_triplet`), A32, T8KO, T8LV.
3. Route9 gen7:
   - `Triplets` template + R9T=4 (changes v1 too; re-time v1 M4);
   - LDS-staged per-route books with RB;
   - R9GEN=3 + GPERM (the GPERM path of `iq2r_v2_form_copies` only);
   - R9X, R9PIPE (`glm53_gather_atom` / `glm53_sign_atom`);
   - R9AL, R9SS, R9LAT, R9PRO, R9RFL, R9SR, R9ADDR.
4. Gate:
   - carried levels + OPSEL (`glm53_cross_k_decode_mfma_carried`);
   - ABUF;
   - GG=2 with 4 chunks;
   - PFA;
   - NT=2 (changes v1 too).
5. DBC, once it has been timed.
6. The encoder guarantee for formation (section 12).

**Re-time on one book.** The one-book gate stages 16 KB, or forms from one 4 KB base. The ranking of GG=2 vs plain
DMA staging (and LDSB, GEF) may change, so re-time them before deleting the alternatives.

**Leave on this branch only:**
- All P and R macros.
- `glm53_down_route9_helper`, `glm53_down_route12_kernel`, `iq2r_v2_gate_base_dma`.
- The GPERM=0 `cvt` path, GG=1/3, R9GEN=1/2, the non-PIPE route9 path, the old `glm53_down_tp4_atom` path.

---

## 15. Open items

In priority order:

1. **Port to one book** (section 14), then cmpdump-verify OPSEL / GG=2 / ABUF.
2. **Build and time DBC.** Expected to recover ~3.7% at I=512 M256.
3. **Run full M1–4096 sweeps of the final build** vs a same-session v1. Also time v1 with NT=2 and R9T=4, which this
   study never did.
4. **Per-launch NT / NTT / R9X (and CHUNK) choice in the tuned CSV**, plus v2 rows that need a format key.
5. **Prefill gate and prefill down.** They are still +5..+10% and −1..+4%. They were not worked on.
6. **GPERM / GG precondition:** an encoder clamp ≥ 2^-6, or a load-time check with a staged-copy fallback.
7. **256 routed experts / top-8 runtime for the v2 checkpoint.**
   - The kernels and sort assume 257 / top-9.
   - The FP8 shared expert needs its own path in ATOM, launched first on a side stream with proven device overlap
     (Appendix A.2).
   - Then re-time v1 with 8 rows per token.
8. **I=256 M1 MISMATCH** (rmse 0.0317 vs 0.03).
9. **Format options that need agreement before any work:**
   - `d` interleaved into the records (removes the `d` stream);
   - bi4 (−3.4% weight error, 16 KB);
   - lane-grouped classes;
   - down sign repack to the gate's bit-plane layout (lossless).

**Hardware needed:** 1 GPU for kernel tuning, 4 for a TP4 end-to-end check, 8 for TP8.

---

## 16. Tools

Everything used in this study is in `docs/iq2r_v2_kernel_study/`:
- the scripts are kept exactly as they ran;
- `archive.tar.xz` holds every timing log, the rocprof CSVs, the four kernel snapshots and the pre-commit patches.

**Path assumptions.**
- The scripts run in one container. The study directory is mounted as `D=/workspace/experiments/iq2r-v2-20261103`
  and the aiter trees as `/workspace/splits/<tree>`.
- Some scripts also use the host path of the same tree. Edit `optest.sh`, `env.sh`, `vab.sh`, `vabl.sh`,
  `tsgrun.sh`, `mkvar.py` and `mky.py` for a new host.
- `env.sh`:
  - unsets `AITER_*`, `ATOM_*` and `PYTHONSAFEPATH` (it breaks CK JIT builds);
  - sets `HIP_VISIBLE_DEVICES=${GPU}`, `GPU_ARCHS=gfx950`, `PYTHONPATH=$TREE:$TREE/op_tests`, `CCACHE_DIR=/ccache`;
  - runs `ulimit -c 0`.
- `vbench.py` caches synthetic weights in `$D/cache/` (~0.8 GB per file). They regenerate.

**Drivers**

| Script | Use |
|---|---|
| `optest.sh LOG CMD` | run CMD in the container with `env.sh`; writes `LOG.log`. Env: `GPU`, `TREE` |
| `vbench.py FMT INTER M,M,… [CSV \| gate:grid:down:grid:chunks]` | per-stage cold graph timing. Env: `STAGES=gate,down`, `CHECK=1`, `REPS=3`, `DUMP=prefix`, `TREE` |
| `vab.sh NAME TREE "K=V …" GPU_A GPU_B [MS512] [MS256]` | set `GLM53_X_K` (fails on an unknown macro), remove the .so, build, time I=512 on GPU_A and I=256 on GPU_B |
| `vabl.sh` | `vab.sh` with a `flock` around timing, so two trees build in parallel but never time together |
| `ab.sh TAG TREE [MS512] [MS256]` | rebuild, then v1 + v2 gate at both shapes |
| `r9run.sh NAME R9T [R9W R9ROT R9VG R9ND R9SLP]` | route9 down at M 1, 2, 4 (`r9run.sh gen7 4` = the gen7 run) |
| `drun.sh NAME X_KEY=VAL …` | any down macro, all decode M |
| `stg.sh NAME STG ORD` | R9STG / R9ORD then r9run |
| `tsgrun.sh NAME "K=V …" [GPU] [FMT] [INTER] [MS]` + `tsg.py` | GTS=1 build, per-item gate phase cycles |
| `ts.py INTER M` | route9 per-wave timestamps (TS=1 build) |
| `prof.sh NAME FMT INTER M "PMC …" [REGEX]` | rocprofv3 counters → `prof/NAME/` |
| `isa.sh` / `isa2.sh NAME` | `hipcc -S` with the JIT flags → `NAME.s` / `.occ` / `.waits` |
| `xrun.sh` + `mkvar.py`, `arun.sh`, `yrun.sh` + `mky.py` | x / a / y series. They rewrite the kernel from the s8 / a6 snapshots in `archive.tar.xz`; only for reproducing those rows |

**Analysis**

| Script | Use |
|---|---|
| `cmp.py A B`, `cmpn.py BASE A …`, `tab.sh`, `gtab.sh STAGE LOG …` | log tables and ratios |
| `cmpdump.py A B` | bit-exact compare of `dumps/{A,B}_{512,256}_{1,2,4}.pt` |
| `occ.py`, `waits.py`, `kres.py` | VGPR / AGPR / LDS / occupancy, waits, kernel metadata from `.s` |
| `loopdump.py`, `loopmix.py`, `loopops.py`, `fnops.py`, `icount.py`, `ihist.py A.s B.s`, `kext.py` | inner-loop and whole-kernel instruction mixes and diffs |
| `pavg.py`, `profsum.py` | rocprof CSV summaries |

**Microbenchmarks and research**
- `ubench/rate.hip`, `ubench/mul.cpp`: VALU issue rates.
- `regen/`: fp8 `cvt` vs torch rounding (`cvt.hip`), the per-mantissa delta table (`dtab.py`), copy-formation
  variants (`fm.hip`: V3 is GPERM), throughput (`tp.hip`).
- `class_ablation/`: `fetch.py LAYER E,E,…` range-downloads FP8 experts from the public `zai-org/GLM-5.3` repo (needs
  an HF token). `ablation.py` / `ablation2.py` are the class and book ablations; `bookrange.py` gives the byte range.
  The `res*.jsonl` and `run*.log` files are the results. The importance files (`imp_L*.pt`) and expert tensors are
  not included.

**Typical session**

```bash
# time one gate variant against the defaults at both shapes
./vab.sh dbc aiter-v2-exp "DBC=1" 3 5 16,64,256 16,64,256
./gtab.sh gate l3_v1 abuf dbc
# route9 down + bit-exact check against the gen7 dumps
GPU=3 TREE=/workspace/splits/aiter-v2-exp DUMP=$D/dumps/new STAGES=down ./optest.sh new_512 'python3 $D/vbench.py v2 512 1,2,4'
python3 cmpdump.py gen7 new
```

---

## Appendix A. Lessons from the v1 kernels

These come from the v1 campaign (September–October 2026, experiments E001–E453) that produced aiter#5728. Only
lessons that apply to v2 are kept. E-numbers identify the experiments.

### A.1 What moved v1 the most

- **Packed, OR-only decode was the biggest lever.** All of these transforms are exact and offline:
  - the codebook sign folded into record signs;
  - 9-bit indices split into low32 + high4 words;
  - signs transposed into planes;
  - gate/up interleaved into quads.

  Results:
  - E261 / E262 at TP8 M4096: gate VALU −22%, down −15%.
  - E280 (packing + short sign lifetimes + four independent codebook reads): gate VGPRs 129 → 102, which restored
    2 CTAs/CU.
  - Static sign planes (E327–E339): gate vector instructions −16..−19%.
  - Native single-copy decode (E453) is bit-exact but 1–5% slower than packed.
- **Occupancy decides.** Every change that crossed 256 VGPRs or dropped from 2 to 1 CTA/CU lost: E107, E117, E122,
  E174, E350, E355, E364, E390, E417, E419.
  - Above ~4 waves/SIMD more occupancy bought little: E427 got +28% occupancy for +1.4%.
  - Below 2 waves/SIMD everything lost.
  - E422 (minimum 4 waves) took the gate from 49.23 to 36.19 µs even though aggregate waits rose.
- **The private codebook per route wave** (route9, E110 / E111) cut fused down 8.61 → 5.99 µs at M1, with
  LDS waits −56%. It loses at M ≥ 8 because grouped expert reuse is lost, so route9 stops at M4.
- **Remove waste before adding pipelines.** E094 removed prefetch from inactive waves: down −20..−28%.
- **Read the ISA before hand-pipelining.** The compiler already hoists later records and activations and uses
  graduated waits (E105). It already overlapped all nine payload reads (E373).

### A.2 What failed

- **Deeper prefetch rings and double buffering** (E107, E122, E385, E452 three-stage A, register double-buffered
  weights −25%) lost residency or gained nothing.
- **E436, the KPF=2 register ring,** corrupted output: it overwrote the undecoded current tile with tile i+2 (rel RMSE
  ~1.15). Read the assembly whenever a carried LDS result or ring is introduced. E407 was a similar unsafe carry.
- **Fusions.** Every fusion or temp-buffer decode lost: E237–E257, E296, E300, E346, E348. So did:
  - gate + SwiGLU fusion, −19.7%;
  - router + route / task / quant fusion, −22..−33%;
  - atomic down / reduction, −19..−37%.
- **DWORD-plane codebooks** doubled bank conflicts (E404).
- **Shared expert.** The separate shared-expert GEMM was exposed (E021: 76.6 ms per rank at C64, zero overlap). An
  alternate stream mattered a lot: disabling ATOM's dual stream lost 31–36%. Launching the shared expert before
  routing was best; later was 3–10% slower (E032).
  - Fusing it as expert 256 (E039) beat two streams by 30–37% at M ≤ 64 but only 5.6% at M256.
  - v2 keeps the shared expert outside IQ2R again, so these lessons apply directly.

### A.3 Measurement lessons

1. **Time cold.** Rotate ≥ 1 GiB per graph replay (4× the 256 MB MALL). A per-layer TP4 expert set is 0.69 GB.
   Hot-cache microbenchmarks hide traffic savings and inflate reuse.
2. **Cover hot and spread routing with one policy.** Many wins were hot-vs-spread trade-offs (E330, E402, E413).
3. **Use graph replay for decode-sized calls.** Graph-disabled runs are only good for kernel-duration attribution.
4. **Treat isolated gains under 3% as noise** for serving. Tuner-level wins up to 4.9% did not show in serving.
5. **Measure the complete MoE boundary,** not one kernel. In E060 down was 142–160 µs of a 486 µs boundary.
6. **Only the stock `atom.benchmarks.benchmark_serving` counts** for serving numbers. Repeated-prompt workloads
   flattered IQ2R by up to 30 points (E147).
7. **Assert which kernel ran.** Silent fallbacks happened (E146, E175, E225). Set `AITER_LOG_TUNED_CONFIG` to log the
   CSV row picked.
8. **Check exactness across graph-input changes.** In E072 / E075, a per-layer tensor in a shared workspace made every
   graph replay use the last layer's weights. Qualify by alternating layers through one workspace in eager and graph
   mode.
9. **Profile rank 0 only** with rocprof; all-rank profiling serializes ranks.
10. **One GPU, one job.** Overlapping jobs and orphaned profiler workers invalidated timings twice.
11. **Rebuild the JIT** after any signature or template change. A stale `.so` gives silent wrong results or ABI
    failures.

### A.4 Where v1 time went

| M | Bound by |
|---|---|
| 1 | latency (one row, 9 experts) |
| 2–16 | gate near the bandwidth limit; down latency-bound (a load-only down is 1.6–1.9× faster) |
| 32–256 | waits, task padding, occupancy (E086: waits 55–62% of wave cycles) |
| 512–4096 | per-K-tile wait / barrier at 2 waves/SIMD in the prefill kernels |

- **Rows per expert stay small.** Under spread routing the mean rows per active expert only goes from 1 to 8 over
  C1–C256, so decode stays low-intensity.
- **MTP.** The IQ2R margin at concurrency C with MTP equals the non-MTP margin at 4×C. So the M=4..64 decode kernels
  decide MTP results.
- **End-to-end, 2026-10-07** (same code, same node, ISL/OSL 1024, FP8 KV), IQ2R vs MXFP4 output tok/s:
  - TP4 +2.7..+16.4%;
  - TP8 +3.0..+16.2%.

  The weakest points were C2–C8 (M=2–8) and TP4 C256, which match the M2 / M8 / M256 kernel holes.

### A.5 Open v1 ideas that carry over

- Remove the route-output round trip: 27 MiB written and read at M256 vs a 3 MiB final output.
- Make the prefill gate not bound by per-K-tile waits at 2 waves/SIMD. v1 is 750 vs MXFP4 607 µs at M4096.
- Time MTP row counts (M5 / 6 / 12 / 24 / 48), which the CSV rounds up to the next row.
- Re-tune from scratch under the cold protocol rather than reusing v1's CSV rows.

---

## Appendix B. FP11 (11-bit index) study

This study (2026-10-23) built 11-bit-index variants of the v1 kernels, with 2048-entry codebooks of 16 KB, to see
whether a bigger codebook was affordable. **It was stopped: FP11 is not viable.** The code is not on this branch; it
exists as an uncommitted diff (about +2600 / −360 lines) on top of the v1 kernels, kept with the study notes.

**Format.**
- Every kernel is templated on `Bits`. The host detects the bit width from the aux size.
- Gate quad 2560 B (9-bit: 2304); down triplet 2048 B (1792); codebook 16 KB (4 KB).
- Bytes per expert are +13% (TP4) / +14% (TP8).
- LDS:
  - route9 needs ~148.5 KB, so 1 CTA/CU;
  - decode gates 49 KB;
  - the 64-row stream gate 155.5 KB.
- A 9-bit checkpoint can be widened by aliasing codebook entries. FP11 kernels must then match 9-bit bit for bit, and
  the tuner enforced this.

**Result: FP11 is slower than 9-bit at every M.**

| TP4 whole MoE (µs) | M1 | M4 | M16 | M64 | M256 | M512 | M1024 | M4096 |
|---|---|---|---|---|---|---|---|---|
| 9-bit | 19.84 | 30.19 | 70.56 | 143.87 | 189.11 | 260.51 | 301.02 | 751.49 |
| FP11 | 22.2 | 35.55 | 77.36 | 166.6 | 219.96 | 271.83 | 311.13 | 770.33 |
| MXFP4 | 29.67 | 35.65 | 92.97 | 192.08 | 224.46 | 240.02 | 286.41 | 605.14 |

- FP11 vs 9-bit: +7..+20% at M ≤ 256, +2.5..+11% above. TP8 is similar.
- The decode gate is byte-bound, so the +13% bytes shows up almost 1:1.

**Gate kernel lessons that apply to any wide-codebook or prefill gate.**
- **Persistent streaming prefill gate** (64-row items, `Ahead`-deep load pipeline, whole-buffer descriptors), best
  steps:

  | Step | M512 / M4096 µs |
  |---|---|
  | first build | 285 / 738 |
  | descriptors | 199 / 551 |
  | gather one step ahead | 181 / 484 |
  | step fence | 175 / 470 |
  | 16-bundle software pipeline | 158 / 427 |
  | asm anchor removed | **151.92 / 416.00** |

  The old prefill gate measured 169 / 401 µs in the same harness.
- **128-row gate** (MFMAs accumulate in place in AGPRs): 360.26 µs at M4096 vs MXFP4 276. It is the best FP11 gate
  at M ≥ 2048 but worse at M ≤ 1024.
  - Item-switch gaps of 20–23K cycles come from `vmcnt(0)` waits and `v_div_scale` in the exact SiLU division.
  - A scheduling model says LPT item ordering would give ~5%.
- **Two waves/SIMD** for the stream gate did not help. Forcing the old prefill gate to 1 CTA/CU cost 36–39%.
- **Fusing gate + down in one kernel** with `__threadfence` cost 30–590 µs (whole-L2 writeback). Agent-scope atomics
  still left a tail.
- **Prefill down** is store / latency-structure bound (probes: no stores 75 µs vs 216; no MFMA 207; no decode 196).
  A weight-stationary "run" kernel won only at M ≥ 2048.

**Untested state.** The last edit of the 128-row gate (transposed MFMA operands, register-only epilogue) was never
built. The last known-good version is its `v6-m128` snapshot.
