# IQ2R-v2 kernels: handoff to the recipe agent (2026-11-06)

**Short version:**
- The branch fast-forwards onto `glm-5.3-iq2r-v2`, and both op tests pass.
- v2 is faster than v1 at every tuned M at TP4 and TP8 (worst -4.5%).
- 256 experts / top-8 works as ATOM already calls it.
- **One checkpoint issue (section 2):** in `/models/zai-org/GLM-5.3-IQ2Rv2-final`, 9 of 38,400 expert matrices
  (all `down_proj`) have codebook entries below 0x08. On those, two of the new down kernels decode a few
  codeword components slightly differently from the stored copies. Fix those 9 matrices before the quality
  evals.

## 1. Branch state

- **Branch:** `glm-5.3-iq2r-v2-kernels` on `fork` (ssharma4-amd/aiter), pushed. Working tree clean, apart from
  an untracked local session note, `FLEET_SESSIONS.md`, which git ignores via `.git/info/exclude`.
- **Head:** the commit that adds this file (see the reply). Before it: `e604136a6`.
- **Base:** `b5c7638cd`. No rebase or merge onto anything after it.
  - `glm-5.3-iq2r-v2` and `glm-5.3-iq2r-v2-recipe` both still point at `b5c7638cd`, so merging this branch is a
    fast-forward.
  - Upstream `main` moved, but has no IQ2R, tuner or JIT-core changes since `796ff6a35`.
- **Last kernel or Python code change:** `d35a140b1`, which is what was tested and timed. Later commits change
  only docs, a test docstring, and one kernel comment that was added and then removed.

Commits (oldest first; all "WIP" ones are meant to be squashed for the PR):

| Commit | Content |
|---|---|
| `9e8af89ea` | Port the study's fastest kernels to the one-codebook format; native 256-expert / top-8 |
| `dfec56c49` | v1 nobarrier gate uses the sign-pair helper (fix after the port) |
| `7403f9e37` | Codebook range: learner clamp + pack-time check; tuner times 256/top-8 |
| `3ad49ec8c` | Prefill gate at 2 waves/SIMD; tests for top-8 and the codebook range |
| `a6e627cd3` | Gate DBC and late book formation as switches (to time) |
| `26d0d7d61` | Down kernels form copies in LDS (to time) |
| `d4e3cce9f` | Late gate book formation kept; DBC dropped; only prefill downs form copies |
| `801ece042` | Prefill downs: levels/scales once per K tile; prefill gate forms copies |
| `a9a93cab2` | TP4 prefill down on 768-column items; interim tuned rows |
| `8ee3bf5e4` | Fix the TP4 prefill down template argument |
| `c85d3b40a` | Tuned-config test covers the 256/top-8 rows |
| `f4f63e200` | Gate barrier no longer waits on its own loads; `nobarrier_d` gate (later dropped) |
| `b07220cc6` | `nobarrier_t` gate replaces `nobarrier_d` (later dropped) |
| `e39e4950d` | TP8 prefill down on 768-column items (reverted next) |
| `b9e979987` | Revert of `e39e4950d` (slower at TP8 M1024/2048) |
| `7a2eb8cfb` | Top-8 sorts emit no tasks for the empty ninth slots |
| `d35a140b1` | Drop `nobarrier_t`; final tuned rows for 256/top-8 |
| `4950983cb` | Kernel doc |
| `d05453579` | Doc and kernel comment on the task lists and work split |
| `e604136a6` | Doc trimmed to the task-list note; kernel comment removed again |

Changed files vs `b5c7638cd`:
- `aiter/configs/iq2r_glm53_tuned.csv`, `aiter/configs/iq2r_glm53_untuned.csv`
- `aiter/iq2r_glm53.py`, `aiter/ops/iq2r_v2.py`
- `csrc/include/iq2r.h` (header comment only; the gate-kernel enum is back to 0/1/2)
- `csrc/kernels/iq2r/iq2r_gemm_gfx950.cu`, `csrc/kernels/iq2r/iq2r_moe_aux_gfx950.cu`,
  `csrc/kernels/iq2r/iq2r_glm53_tune.py`
- `docs/iq2r_glm53.md`, `docs/iq2r_v2_kernels.md`, `docs/iq2r_v2_kernels_handoff.md`
- `op_tests/test_iq2r_glm53.py`, `op_tests/test_iq2r_glm53_v2.py`

## 2. Format and checkpoint compatibility

- **Packed weight layout, auxiliary byte format, `iq2r_glm53_v2_aux_bytes`, slice helpers:** unchanged. The
  packed auxiliary still stores all 4 codebook copies (16 KB), the d table and the 2 gains, at the same offsets
  and size.
- **`iq2r_glm53_pack_v2`:** output bytes unchanged. It now calls `iq2r_v2_check_codebook` first and raises if
  any codebook byte is outside 0x08..0x78.
- **Encoder / learner output bytes:**
  - `iq2r_v2_learn_codebook` now clamps codebook entries to >= 2^-6 (byte 0x08); before, it clamped at 0.
  - Output bytes change only for matrices whose learned book had entries below 2^-6. Every other matrix encodes
    byte-identically. Nothing else in the encoder changed.
- **How the 0x08..0x78 range was resolved:** in the encoder, plus a pack-time check. The kernels were not
  changed to handle the full E4M3 range, since that would have cost formation VALU.
  - Why it matters: several kernels (the M1 and decode gates, the prefill gate, the route9 M<=4 down, the
    prefill downs) load only copy 0 and form copies 1-3 in LDS as `b + delta_c[b & 7]`.
  - That formula equals the stored `E4M3(codebook * 2^(c/4))` exactly for bytes 0x08..0x78, and is not exact
    below.
  - The decode downs (M8-512 `single`/`ordered`/`packed`) still read the stored copies.
- **Can a checkpoint built at `b5c7638cd` (`/models/zai-org/GLM-5.3-IQ2Rv2-final`) run unchanged? No, not
  exactly. It loads and runs, but 9 matrices decode slightly differently.** Checked on this machine:
  - I read every packed auxiliary in the checkpoint (read-only, CPU-only container; the shards are root-only),
    150 shards.
  - 38,400 expert matrices; copy-0 bytes span 0x00..0x4f.
  - 9 matrices have entries below 0x08. Formed copies 1-3 differ from the stored copies in 42 bytes in total.

  | Matrix | Expert | Entries < 0x08 (values) | Codewords |
  |---|---|---|---|
  | layer 3 down | 2 | 3 (0x01, 0x06, 0x07) | 158, 172, 230 |
  | layer 3 down | 91 | 1 (0x06) | 121 |
  | layer 3 down | 200 | 1 (0x07) | 167 |
  | layer 4 down | 140 | 1 (0x07) | 318 |
  | layer 4 down | 166 | 1 (0x07) | 240 |
  | layer 5 down | 157 | 1 (0x03) | 128 |
  | layer 8 down | 86 | 3 (0x00, 0x06, 0x07) | 13, 62, 157 |
  | layer 9 down | 15 | 2 (0x05, 0x06) | 183, 190 |
  | layer 77 down | 134 | 1 (0x02) | 466 |

  - **Effect:**
    - For these experts, the route9 (M <= 4) and prefill downs decode the listed codeword components with an
      absolute error of roughly 0.002-0.016 in codebook units; codebook entries go up to about 7.5.
    - Every other path, expert and layer is exact.
    - I expect the quality effect to be negligible, but it is not bit-exact against the reference decode, and I
      did not measure quality.
  - **Why nothing catches it:** ATOM loads `glm53-packed-v2` tensors directly and never calls
    `iq2r_glm53_pack_v2`, so the new check does not run on this checkpoint.
  - **Fix: pick one before the evals.**
    - (a) Re-encode those 9 down matrices with the current learner. They come out in range, and the rest of the
      checkpoint stays byte-identical.
    - (b) Patch the packed auxiliary in place: set copy-0 bytes below 0x08 to 0x08 and rewrite copies 1-3 from
      it. This moves 14 tiny codebook components to 2^-6, about 0.016.
    - Option (a) is cleaner. Either way, rerun the scan above, which should then report 0 rows; the script is
      in the private repo (section 7).

## 3. 256 experts and top-8 routing

- **How:** `iq2r_glm53_moe_out` accepts `[256, ...]` stacks with `[tokens, 8]` ids and weights directly; no
  host-side padding. Internally:
  - Routes keep 9 slots per token (`token*9 + k`).
  - The sort and quantize kernels (`TopK` template) read the 8 ids and treat slot 8 as empty: no tasks,
    `scatter = -1`.
  - The reductions read `[tokens, 8]` weights.
  - The M<=4 down runs 8 route waves (`glm53_down_route9_v2_kernel<..., 8>`).
  - The launchers infer top-k from the routing tensor's size, and the stack size from the weights.
- **API:** no signature change.
  - `iq2r_glm53_moe_out(x, w13, w13_aux, w2, w2_aux, topk_weights, topk_ids, out, workspace)` as before.
  - Routing must be `[tokens, 9]` with a 257-expert stack, or `[tokens, 8]` with a 256-expert stack. Anything
    else raises `ValueError`.
  - `iq2r_glm53_config(tokens, intermediate, experts=257)` gained the `experts` argument; ATOM does not call it.
  - ATOM-v2 (`76b69cb38`, `atom/model_ops/moe.py`) already passes routed-only `topk_ids`/`topk_weights` and
    the packed `[256, ...]` tensors, so it needs no change. I did not run ATOM.
- **v1 path (257 experts, top-9):**
  - Unchanged in behaviour: route9 and the nobarrier gate are upstream's kernel bodies; the other kernels take
    v2 paths only for v2 auxiliaries.
  - The one shared change on the v1 path is the study's sign-pair helper in the decode gates, which produces
    the same bits.
  - Tested: `op_tests/test_iq2r_glm53.py` passes.
  - v1 speed in this branch matched upstream within noise. In one same-session run with v1's tuned CSV, this
    branch's v1 rows were within ±1% of upstream's.

## 4. Correctness

- **Test command** (from the aiter tree, with `PYTHONPATH=<tree>:<tree>/op_tests`):
  `python3 -m pytest -q op_tests/test_iq2r_glm53_v2.py op_tests/test_iq2r_glm53.py`
  - **Result: 230 passed, 24 skipped.** The skips are existing explicit-config cases that a given M or width
    cannot run.
  - Where:
    - Fleet session `dev-307a3146`, node `smci355-ccs-aus-m08-05` (MI355X, gfx950, austin site), image
      `registry.crsuse2.internal/curated/rocm-pytorch@sha256:4449f856653602317e4101a76fce599c7fcd58ccec2e539951fce5f73083179e`
      (ROCm 7.2.4, PyTorch 2.10), GPU 1.
    - Commit `d35a140b1`; later commits do not change code.
    - Earlier commits passed the same suite on other Fleet nodes.
  - Not run on this machine (no local GPU use).
- **Bit-exact (OPSEL, GG=2, ABUF and the rest of the port):** 188 of 188 runs identical. Setup:
  - Reference: `b5c7638cd` (staged copies, no OPSEL/GG/ABUF).
  - The ported kernels and the reference ran the same synthetic 257-expert IQ2R-v2 layer, with codebooks of
    random bytes over 0x08..0x78.
  - Cases: every valid explicit config, TP4 and TP8, M = 1, 2, 3, 4, 5, 8, 16, 48, 64, 100, 256, 512, 1024,
    2048, 4096.
  - Compared with `torch.equal`: the gate stage's FP8 output and scales (valid routes) and the final output.
  - Top-8 on a 256-expert stack vs top-9 with route 8 dropped: also 188/188 identical.
  - Rerun on the final code (`d35a140b1`) on `dev-307a3146`.
- **Real GLM-5.3 weights:**
  - Kernel tests and timing used synthetic weights only.
  - The only real-weight work is the codebook scan in section 2, which reads the auxiliary bytes and runs no
    kernel.
  - No real-layer MoE output was checked against a reference.

## 5. Speed

Timing setup:
- **Method:** the tuner's.
  - `iq2r_glm53_tune.py`'s `_run_lookup` through `finaltime.py`, using each tree's own tuned CSV.
  - CUDA-graph replay.
  - Every call reads its own copy of the layer, at least 1 GiB of expert weights per replay, so weights come
    from HBM (cold).
  - 50 warmup + 500 timed calls; synthetic weights.
- **Hardware and runs:** one MI355X (`smci355-ccs-aus-m08-05`, GPU 7). 6 rounds per arm, in blocks of 3,
  alternating v1, v2, v1, v2. Table = best of 6, with the median in brackets.
- **v1 arm:** upstream `main` (`8b114774f`), 257 experts / top-9, its tuned CSV.
- **v2 arm:** this branch at `d35a140b1`, 256 experts / top-8.

Whole MoE, `iq2r_glm53_moe_out`, µs:

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

- **Speed target:** met. No point misses.
  - The other tuned M (32, 128, 512, 2048) are -5.7% to -15.7%.
  - Worst over all 26 shapes: -4.5% best-of, -4.9% on medians.
- **Caveat: the v2 MoE excludes the shared expert, which v1 runs as route 9.** v2's FP8 shared-expert MLP
  runs outside this call and was not timed, so this is not a whole-layer comparison.
- **What was kept** (each measured in same-node A/Bs; details in `docs/iq2r_v2_kernels.md`, "What moved the
  numbers"):

  | Change | Effect |
  |---|---|
  | No tasks for empty top-8 slots | TP8 M512-1024 -12..-22%, TP4 M512-1024 -12..-13%. Before, the skipped tasks unbalanced the gate's work split |
  | Prefill gate at 2 waves/SIMD | Was 264 registers (1 wave/SIMD). TP4 M1024 gate 252 → 161 µs |
  | Levels/scales once per K tile in the prefill downs | TP4 M512-4096 +2.4..+5.3% → about parity |
  | Ported study kernels | Codebook copies formed in LDS from the 4 KB base (gen7 route9, GG=2 gates), OPSEL, ABUF, K-outer downs |
  | Route9 with 8 waves at M4, from the retune | TP4 M4 about 35 → 27 µs |
  | 768-column TP4 prefill-down items | About 1 point |
  | Gate book formed after the item's loads, no barrier wait on its own loads | About 1 point at decode M; TP8 M512 gate 86 → 84 µs |

- **DBC (gate d words loaded once per item):**
  - Built and timed: +1..+3% at M2-64, never faster elsewhere once the barrier fix was in.
  - Dropped. It bought nothing.
- **Codebook copies formed in-kernel:**
  - Prefill downs: TP4 M1024 MoE +11% → +5% vs v1 at that point.
  - Prefill gate: neutral, kept for less traffic.
  - Decode downs: about 1 point slower at M4-16, so those still stage the 16 KB.
  - Decode gates and route9 use formation, as in the study.

## 6. Dispatch configs

- **`aiter/configs/iq2r_glm53_tuned.csv`:** 26 rows added for `gfx950, cu_num 256, model_dim 6144, expert
  256, topk 8`. Token 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, at `inter_dim` 512 (TP4) and
  256 (TP8).
  - Existing 257 / top-9 rows unchanged.
  - The same 26 keys were added to `iq2r_glm53_untuned.csv`.
- **Tuned on MI355X** (cu_num 256): Fleet session `dev-34b52025`, node `smci355-ccs-aus-m14-33`, GPU 7, tree
  `7a2eb8cfb`.
  The final code (`d35a140b1`) differs only by removing a gate variant that no row used.
- **Lookups:**
  - `iq2r_glm53_moe_out` looks up `(gfx, cu_num, inter_dim, experts)` with `experts =` the stack's first
    dimension, so 256-expert stacks hit these rows.
  - `test_glm53_tuned_config` checks every row and the round-up for both expert counts.
  - Other gfx / CU-count combinations have no 256/top-8 rows and fall back to `_default_config`. Not timed.

## 7. Code cleanliness for the upstream PR

- **`GLM53_X_*` switches and the study's dead variants:** all gone. That includes route12, the route9 helper
  waves, LDSB, GEF, PF and XQ. The experiment-only gate variants `nobarrier_d` / `nobarrier_t` were removed
  again.
- **Experimental code left:**
  - None as switches.
  - The 9-route split path (`Routes == 9` in `glm53_down_route9_v2_kernel`) stays because 257-expert v2
    stacks still run.
  - The TP8 prefill down keeps 384-column items; that kernel is still templated on `Groups`.
- **History:** many WIP commits. They need squashing for the PR as the plan says.
- **Lint:** black (26.10.1) and ruff 0.16.0 pass on all changed Python files.
- **Kernel doc for the PR:** `docs/iq2r_v2_kernels.md` covers what runs, why, correctness, measurements and
  what was not measured. `docs/iq2r_glm53.md` is updated (256/top-8, dispatch keys).
- **Internal notes and raw logs:** `ssharma4-amd/glm53-iq2r-v2-private`, `findings/kernels/`:
  - `README.md`;
  - `raw/final/` (final table rounds);
  - `raw/ab/` (all A/B rounds);
  - `raw/bitexact_final.txt`;
  - `fleet_sessions.md`;
  - `tools/` (finaltime, bitexact, stageprof, abrun).
  - The checkpoint scan scripts (`ckpt_books.py`, `ckpt_bad.py`) are in `tools/`. Run them in a CPU-only
    container with the checkpoint mounted read-only.

## 8. Open items and resources

**Priority order:**
1. **Fix the 9 out-of-range down matrices** in `GLM-5.3-IQ2Rv2-final` (section 2) before the quality evals,
   then rescan.
   - Also consider a load-time check in ATOM, or an aiter helper on the packed auxiliary: compare formed vs
     stored copies, or range-check copy 0. Packed checkpoints skip `iq2r_glm53_pack_v2`.
2. **Real-weight check:** compare one real layer's `iq2r_glm53_moe_out` at TP4 and TP8, M = 1, 4, 64, 1024,
   against the dense decode (`iq2r_v2_materialize`). Not done.
3. **End-to-end:** serving, AIME-60 / SWE-29, throughput vs v1 and MXFP4. Not done. Includes the FP8 shared
   expert's cost, which no number here covers.
4. **Cross-GPU and cross-node variation of the timings:** not measured. All final numbers come from one GPU.
5. **Optional speed work:** NTT (non-temporal triplet loads in the decode downs), a per-shape win in the study,
   not ported. Route layout with stride = top-k instead of 9 slots, to save scratch memory; no time to gain
   expected.

**Not measured:** real weights in any kernel test, serving, quality, the FP8 shared expert, power, token
counts between tuned rows, M > 4096, IQ2R-v2 speed on 257-expert stacks.

**Fleet:**
- All sessions are released or ended (`dev-307a3146` released; a Fleet-created duplicate `dev-d81b661e`
  released). Nothing running.
- 6 of 7 sessions were lost, mostly to Fleet "GPU ownership conflict" errors that also hit sessions where GPU
  0 was never used. Details in the private `fleet_sessions.md`. Worth reporting to the Fleet operators.
- The `crusoe-recovered` site is disconnected. The austin hosts mount neither `/models` nor `/user-models`.
  **Nothing was written to `/user-models/shubham/`.**

**Needs from ATOM or the recipe side:**
- The fix for the 9 matrices (or a decision to accept them).
- Keep passing routed-only `[tokens, 8]` routing with `[256, ...]` stacks; nothing else changes for ATOM.
- If you rebuild any layers, rerun the codebook scan.
