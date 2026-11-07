# IQ2R-v2 SWE-bench gap: checkpoint or native path?

Branch `glm-5.3-iq2r-v2-debug` (from `glm-5.3-iq2r-v2` at `abc34e947`). Status: steps 1, 2 and 4 done. Step 3
(native kernels on the checkpoint's own bytes) is waiting for a free GPU.

## Answer so far

**The new checkpoint is not worse.**
- Decoded from its own bytes, `GLM-5.3-IQ2Rv2-final` matches `GLM-5.3-IQ2Rv2u-calv2` matrix for matrix against
  the FP8 source.
- Over all 38,400 routed-expert matrices, the importance-weighted relative MSE ratio (final / calv2) has mean
  0.9999 (gate/up) and 0.9996 (down).
- final is worse on 48.5% / 38.6% of matrices, i.e. a coin flip, slightly in final's favour. No layer deviates:
  the per-layer ratio is 0.9988-1.0002.

**So the SWE-29 gap (19-20 vs 26) comes either from the native kernel/packing path or from run-to-run noise.**
- Packing is ruled out: the checkpoint's packed bytes unpack to canonical buffers that pack back byte for byte
  on every layer (step 1), so no information is lost or changed by the packed format.
- That leaves the native kernels on these bytes (step 3, pending), or noise. The recipe agent's rerun of the
  26/29 checkpoint on the old stack will show how much of 26 was luck.
- The canonical copy of the new checkpoint (step 4) is ready, so the old stack can serve it: this separates
  "checkpoint" from "native path" directly.

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

**Pending: needs one GPU.** The local GPUs were busy at the time, and the private checkpoint bytes cannot be
moved to Fleet.

The script (`findings/debug/tools/native_check.py` in the private repo) runs:
- `iq2r_glm53_moe_out` on the checkpoint's packed tensors, sliced with the aiter slice helpers to TP8 rank 0
  and TP4 rank 1;
- on the first M captured tokens of `glm53-acts-e4` layers 5, 20, 40, 60 and 75, with their real top-8 routing,
  M = 1, 2, 4, 8, 16, 64, 256, 1024, 4096.

It compares against:
- (a) the op test's dense reference (MXFP8 activations, BF16 SwiGLU, BF16 route outputs) built from the unpacked
  bytes, with limits relative RMSE < 3e-2, cosine > 0.9995;
- (b) an FP32-activation reference, to measure what the native path's activation quantization adds beyond the
  weights. (a) alone cannot show that, because it quantizes activations the same way.

## 4. The new checkpoint in the calv2 layout

`/models/zai-org/GLM-5.3-IQ2Rv2-final-canonical`: **done.**
- 600 files `layer-LL-eEEE.safetensors` (32 experts each), tensors `{e}.gate_up.data/.aux` and
  `{e}.down.data/.aux`, 201 GB, root-owned, plus `manifest.json`.
- The same file names as `GLM-5.3-IQ2Rv2u-calv2`, and the same tensor names, shapes and dtypes in a sample of
  18 files.
- No shared-expert files: as in calv2, the shared expert stays FP8.
- Written from the unpacked buffers that passed the step-1 proof.

## Not measured (yet)

- Step 3: the native kernels on the checkpoint's own bytes.
- Any served or end-to-end quality (SWE-29 on the canonical copy is the recipe agent's).
- Activation-level differences between the old decode-to-FP8 path and the native path, other than step 3(b).
