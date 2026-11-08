# IQ2R-v2 SWE-bench gap: checkpoint or native path?

Branch `glm-5.3-iq2r-v2-debug` (from `glm-5.3-iq2r-v2` at `abc34e947`). Status: all four steps done.

## Answer

**Neither: the new checkpoint is not worse, and the native path does not differ from the reference on the
checkpoint's own bytes.**

- **Checkpoint (step 2):**
  - Decoded from its own bytes, `GLM-5.3-IQ2Rv2-final` matches `GLM-5.3-IQ2Rv2u-calv2` matrix for matrix
    against the FP8 source.
  - Over all 38,400 routed-expert matrices, the importance-weighted relative MSE ratio (final / calv2) has mean
    0.9999 (gate/up) and 0.9996 (down). final is worse on 48.5% / 38.6% of matrices.
  - No layer deviates (per-layer ratio 0.9988-1.0002), including the re-encoded 3/4/5/8/9/77.
- **Packing (step 1):** the packed bytes unpack and pack back byte for byte on all 75 layers, so the packed
  format loses nothing.
- **Native kernels (step 3):**
  - On the checkpoint's own packed bytes (TP8 rank 0 and TP4 rank 1 slices, layers 5/20/40/60/75, real routing,
    M = 1..4096), `iq2r_glm53_moe_out` matches the dense reference from the unpacked bytes in all 90 cases:
    relative RMSE <= 0.025, cosine >= 0.99970 (limits 3e-2 / 0.9995).
  - Against FP32 activations the native path adds at most 0.002 over the ideal MXFP8 reference.

**What is left as an explanation of the SWE-29 gap (19-20 vs 26):**
- Run-to-run noise: one run each on 29 tasks.
- Something outside the MoE kernels in the new stack's IQ2R-v2 path, e.g. how ATOM runs the FP8 shared expert
  next to the MoE.
- The activation precision of the native path versus the old decode-to-FP8 path.
  - The native MoE quantizes activations to MXFP8, which alone is 3-4.3% relative RMSE per MoE layer against
    FP32 activations (step 3b).
  - How the old path quantizes activations for its FP8 MoE was not measured. Note that IQ2R v1 native uses the
    same MXFP8 activations.

The decisive test is the recipe agent's SWE-29 on the canonical copy of this checkpoint
(`/models/zai-org/GLM-5.3-IQ2Rv2-final-canonical`) with the old stack:
- about 26 means the native/new-stack path costs the points;
- about 19-20 means the checkpoint or noise does.

## 1. Unpacking the checkpoint

- **New function:** `iq2r_glm53_unpack_v2` (`aiter/iq2r_glm53.py`) is the inverse of `iq2r_glm53_pack_v2`. It
  undoes, in reverse order:
  - the gate quad regroup (padding N blocks come back as zeros, as `iq2r_v2_write_data` writes them);
  - the sign-bit transpose;
  - the 9-bit index packing;
  - the d-table relayout.
- **Auxiliary:** it keeps codebook copy 0 (the codebook) and the gains; copies 1-3 are recomputed by packing.
- **Test:** `op_tests/test_iq2r_glm53_v2_unpack.py` (CPU only, synthetic canonical buffers from the format's own
  writers, I = 256, 512, 2048) checks:
  - `unpack(pack(c)) == c` byte for byte;
  - `pack(unpack(p)) == p`;
  - unpacking a TP slice of a packed layer decodes to the same slice of the dense weights.
  - Result: 4 passed.
- **Proof on the real checkpoint:** for every layer file of `/models/zai-org/GLM-5.3-IQ2Rv2-final` (layers 3-77,
  gate/up and down, all 256 experts), `iq2r_glm53_pack_v2(iq2r_glm53_unpack_v2(x)) == x` byte for byte for all
  four tensors (data and auxiliary of gate/up and down). **75 of 75 layers equal.**
- **Codebooks:** every codebook byte is in 0x08..0x78 (after the re-encode of layers 3/4/5/8/9/77), so the stored
  copies are exactly the ones the kernels form.

## 2. The two checkpoints against the FP8 source

**Method:**
- For every routed expert of all 75 layers, decode with `iq2r_v2_materialize`, gains applied as served:
  - gate/up as the interleaved matrix (gate0, up0, …) [4096, 6144];
  - down [6144, 2048].
- final is decoded from its unpacked canonical store (step 4, byte-equivalent per step 1); calv2 from its
  canonical store.
- Compare against the block-FP8 source dequantized with its 128x128 scales.
- **Importance-weighted relative MSE:** `sum_k imp_k sum_n (Q - W)^2 / sum_k imp_k sum_n W^2`, where `imp` is
  `calibration_v2.pt` via `load_glm5_importance` (normalized to mean 1; the same importance for both stores).
- **Plain relative MSE:** the same without `imp`.
- **Also recorded:** gains, codebook byte range and distinct bytes, codewords used, d range, mean level.

**Distribution over all 19,200 matrices per projection and store:**

| | gate/up final | gate/up calv2 | down final | down calv2 |
|---|---|---|---|---|
| weighted rel MSE, mean | 0.07898 | 0.07899 | 0.06690 | 0.06692 |
| median | 0.08029 | 0.08029 | 0.07113 | 0.07117 |
| p99 | 0.08079 | 0.08080 | 0.07996 | 0.07997 |
| max | 0.08111 | 0.08124 | 0.08193 | 0.08186 |
| plain rel MSE, mean | 0.08309 | 0.08309 | 0.09922 | 0.09924 |
| plain rel MSE, max | 0.12879 | 0.12802 | 2.12947 | 2.15886 |
| gains, mean ± sd | 1.0348 ± 0.0009 | 1.0348 ± 0.0009 | 1.0414 ± 0.0078 | 1.0414 ± 0.0079 |
| gains, range | 1.0315-1.0571 | 1.0314-1.0562 | 0.8964-1.1022 | 0.8964-1.1022 |
| codebook bytes | 0x11-0x4f | 0x0c-0x4f | 0x08-0x4f | 0x01-0x4f |
| codewords used (min) | 512 | 512 | 512 | 512 |

**Paired, final / calv2 per matrix:**
- weighted: gate/up mean 0.99993 (p99 1.0034, max 1.0063); down mean 0.99956 (p99 1.0031, max 1.047);
- plain: gate/up mean 1.00000; down mean 0.99984.

**Per-layer means** (weighted rel MSE, final vs calv2), selected layers; all 75 are in the private repo's
`per_layer.csv`:

| Layer | gate/up final | gate/up calv2 | ratio | down final | down calv2 | ratio |
|---|---|---|---|---|---|---|
| 3 | 0.04695 | 0.04695 | 1.0000 | 0.01613 | 0.01614 | 0.9988 |
| 4 | 0.06125 | 0.06125 | 1.0000 | 0.02245 | 0.02246 | 0.9997 |
| 5 | 0.06497 | 0.06498 | 0.9999 | 0.02345 | 0.02345 | 0.9999 |
| 8 | 0.07566 | 0.07567 | 0.9999 | 0.05883 | 0.05884 | 0.9998 |
| 9 | 0.07714 | 0.07712 | 1.0002 | 0.06422 | 0.06424 | 0.9996 |
| 20 | 0.07979 | 0.07982 | 0.9997 | 0.06577 | 0.06580 | 0.9996 |
| 40 | 0.08039 | 0.08041 | 0.9998 | 0.07393 | 0.07395 | 0.9997 |
| 60 | 0.08035 | 0.08036 | 0.9999 | 0.07116 | 0.07120 | 0.9994 |
| 75 | 0.07983 | 0.07984 | 0.9999 | 0.06802 | 0.06807 | 0.9992 |
| 77 | 0.07950 | 0.07950 | 0.9999 | 0.06122 | 0.06124 | 0.9998 |

Per-layer ratio range over all 75 layers: gate/up 0.9994-1.0002, down 0.9988-0.9999.

**Flags checked; nothing systematic:**
- **Re-encoded layers 3, 4, 5, 8, 9, 77:** ratios 0.9988-1.0002, the same as every other layer.
- **Down vs gate/up:** both equal; down is slightly in final's favour.
- **Low-token fallback** (fewer than 256 routed calibration tokens; weight-energy importance): 3 experts, layer
  32 e30, layer 43 e65, layer 46 e139. final is 0.2-0.3% better than calv2 on them.
- **Gain outliers:** the extremes (layer 4 e96 down 0.896; layer 77 e44 down 1.102) are identical in both stores
  to 3 decimals, so they come from the weights, not the recipe.
- **Codebooks:** final's bytes are all in 0x08..0x4f. calv2 has 12 down codebooks with bytes below 0x08, which
  its decode-to-FP8 path decodes exactly.
- **Worst 50 by weighted error, by plain error, and by ratio:** see the private repo. The worst ratio is one
  matrix, layer 77 e207 down: 0.0514 vs 0.0491.

## 3. Native kernels on the checkpoint's own bytes

**Where it ran:** this machine, GPU 7 only, container `glm53-ab:aiter-6dbf9e876-atom-17aca439a`, aiter tree
`glm-5.3-iq2r-v2-debug`. Fleet could not be used: its API was down, and then its artifact service failed to
complete uploads.

**Inputs:**
- Per layer 5, 20, 40, 60, 75: the checkpoint's packed tensors, sliced with the aiter slice helpers to TP8 rank 0
  (intermediate 256) and TP4 rank 1 (intermediate 512).
- The first M tokens of `glm53-acts-e4` with their real top-8 routing (256 experts), M = 1, 2, 4, 8, 16, 64, 256,
  1024, 4096.

**References:**
- (a) the op test's dense reference (MXFP8 activations, BF16 SwiGLU, BF16 route outputs, FP32 sum) built from
  the slice's unpacked bytes via `iq2r_v2_materialize`;
- (b) the same weights with FP32 activations and FP32 SwiGLU.

**Result vs (a):** all 90 (layer, TP, M) cases pass; relative RMSE / cosine per case:

| Layer | TP | M=1 | 2 | 4 | 8 | 16 | 64 | 256 | 1024 | 4096 |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | TP8 r0 | 0.0166/0.99987 | 0.0150/0.99989 | 0.0143/0.99990 | 0.0129/0.99992 | 0.0120/0.99993 | 0.0106/0.99994 | 0.0118/0.99993 | 0.0110/0.99994 | 0.0124/0.99992 |
| 5 | TP4 r1 | 0.0178/0.99985 | 0.0154/0.99988 | 0.0149/0.99989 | 0.0141/0.99990 | 0.0133/0.99991 | 0.0119/0.99993 | 0.0123/0.99992 | 0.0110/0.99994 | 0.0127/0.99992 |
| 20 | TP8 r0 | 0.0127/0.99992 | 0.0118/0.99993 | 0.0136/0.99991 | 0.0149/0.99989 | 0.0132/0.99991 | 0.0132/0.99991 | 0.0136/0.99991 | 0.0126/0.99992 | 0.0133/0.99991 |
| 20 | TP4 r1 | 0.0134/0.99991 | 0.0138/0.99990 | 0.0120/0.99993 | 0.0132/0.99991 | 0.0142/0.99990 | 0.0130/0.99992 | 0.0128/0.99992 | 0.0131/0.99991 | 0.0132/0.99991 |
| 40 | TP8 r0 | 0.0169/0.99986 | 0.0134/0.99991 | 0.0119/0.99993 | 0.0117/0.99993 | 0.0129/0.99992 | 0.0162/0.99987 | 0.0138/0.99990 | 0.0137/0.99991 | 0.0140/0.99990 |
| 40 | TP4 r1 | 0.0068/0.99998 | 0.0085/0.99996 | 0.0103/0.99995 | 0.0189/0.99982 | 0.0246/0.99970 | 0.0174/0.99985 | 0.0150/0.99989 | 0.0143/0.99990 | 0.0141/0.99990 |
| 60 | TP8 r0 | 0.0152/0.99989 | 0.0140/0.99990 | 0.0116/0.99993 | 0.0105/0.99994 | 0.0103/0.99995 | 0.0111/0.99994 | 0.0138/0.99990 | 0.0139/0.99990 | 0.0141/0.99990 |
| 60 | TP4 r1 | 0.0134/0.99991 | 0.0125/0.99992 | 0.0109/0.99994 | 0.0145/0.99990 | 0.0137/0.99991 | 0.0124/0.99992 | 0.0138/0.99990 | 0.0136/0.99991 | 0.0139/0.99990 |
| 75 | TP8 r0 | 0.0063/0.99998 | 0.0073/0.99997 | 0.0093/0.99996 | 0.0138/0.99991 | 0.0136/0.99991 | 0.0149/0.99989 | 0.0144/0.99990 | 0.0139/0.99990 | 0.0138/0.99991 |
| 75 | TP4 r1 | 0.0062/0.99998 | 0.0148/0.99989 | 0.0128/0.99992 | 0.0126/0.99992 | 0.0122/0.99993 | 0.0128/0.99992 | 0.0136/0.99991 | 0.0139/0.99990 | 0.0141/0.99990 |

Worst case: layer 40 TP4 M16, 0.0246 / 0.99970. All outputs are finite.

**Result vs (b) (activation precision):**
- Native vs FP32 activations: 0.029-0.043 relative RMSE at M4096.
- MXFP8 reference vs FP32 activations: the same, 0.029-0.043.
- The native kernels add at most 0.002 (mean 0.0003) over the MXFP8 reference. The rest is MXFP8 activation
  quantization, which the IQ2R v1 native path also has.

## 4. The new checkpoint in the calv2 layout

`/models/zai-org/GLM-5.3-IQ2Rv2-final-canonical`: **done.**
- 600 files `layer-LL-eEEE.safetensors` (32 experts each), tensors `{e}.gate_up.data/.aux` and
  `{e}.down.data/.aux`, 201 GB, root-owned, plus `manifest.json`.
- The same file names as `GLM-5.3-IQ2Rv2u-calv2`, and the same tensor names, shapes and dtypes in a sample of
  18 files.
- No shared-expert files: as in calv2, the shared expert stays FP8.
- Written from the unpacked buffers that passed the step-1 proof.

## Not measured

- **Served or end-to-end quality:** SWE-29 on the canonical copy is the recipe agent's run.
- **The old decode-to-FP8 path's activation quantization, and its error against FP32 activations:** step 3(b)
  measures only the native path's. Comparing the two would show whether the new stack's MoE is numerically
  further from FP32 than the old one.
- **The FP8 shared expert and everything outside the MoE in the new stack.**
- **Step 3 on layers other than 5/20/40/60/75, other TP ranks, and tokens beyond the first 4096 captured.**
