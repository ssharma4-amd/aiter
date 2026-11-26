# GLM-5.3-Flash IQ2R on MI350P

MI350P is gfx950 (CDNA4, the MI350X/MI355X ISA and code objects) with 128 CUs and 144 GiB per GPU. aiter
tells it apart from MI355X (256 CUs) by the CU count: tuned rows are keyed by `(gfx, cu_num)`, so the
`gfx950,128` rows of `aiter/configs/iq2r_glm53_tuned.csv` are MI350P's launch choices and the 256-CU
rows stay MI355X's. The kernels are the same; their grids scale with the runtime CU count.

## Ranks wider than the kernels: TP1 and TP2

The IQ2R kernels run 512-wide (TP4) or 256-wide (TP8) intermediate shards. One MI350P holds
GLM-5.3-Flash IQ2R v3 (120 GB) only at TP1, intermediate 2048, so a wider rank runs as stacked 512-wide
kernel shards:

* weights `[shards, experts, bytes]`, shard `s` being the slice TP4 rank `s` of the rank's range would
  load (`iq2r_glm53_slice_layer`; every shard keeps the expert's codebook);
* `iq2r_glm53_moe_out` sorts and quantizes the routes once; the M1 gate, decode gate, decode down and
  route9 run every shard in one launch (`Glm53SharedArgs::shards`: the shard is part of the work-item
  index and only moves base pointers; route9 shards on grid z); the reduce (or
  `iq2r_glm53_shard_sum_out` after route9) sums the shards, each rounded to BF16, in FP32 in shard order;
* prefill runs one shard per launch.

A TP1 rank's output is bit-identical to four TP4 ranks' outputs summed in FP32, and a one-shard rank
(TP4/TP8) computes exactly as before. `IQ2RGlm53Workspace.allocate(tokens, 2048, ...)` sizes the
stacked buffers; `iq2r_glm53_pack_shared` stacks the shared expert's shards. ATOM loads TP1/TP2 ranks
the same way (`Iq2rMoEMethod`).

## Tuned rows

`csrc/kernels/iq2r/iq2r_glm53_tune.py` tunes wide ranks too. The MI350P rows cover GLM-5.3-Flash with the
1,024-entry codebook at inter_dim 2048, 1024, 512 and 256, M 1..4096, and with the fused shared expert
M 1..256. On MI350P the decode down kernels beat route9 at M2 and M4 for TP1.

## Tests and benchmarks

```bash
pytest op_tests/test_iq2r_glm53_wide.py     # TP1/TP2 vs the dense reference, TP1 == 4 x TP4, shared expert
pytest op_tests/test_iq2r_glm53.py           # TP4/TP8, both models
python3 op_tests/bench_iq2r_glm53_tp1.py     # TP1 and one TP4 rank, CUDA graph replay, cold weights
python3 op_tests/bench_glm5next_mxfp4_moe.py # the MXFP4 MoE as ATOM runs it (A4W4, A8W4), same timing
```

## Measured on one MI350P

One GLM-5.3-Flash MoE layer at TP1, synthetic weights, routed experts only, CUDA graph replay with the
experts read from HBM (us). IQ2R v3 with the tuned rows; MXFP4 with aiter's default kernels (there are no
128-CU MXFP4 rows for these shapes).

| M | IQ2R v3 | MXFP4 A4W4 | MXFP4 A8W4 |
|---|---|---|---|
| 1 | 55.8 | 55.6 | 55.7 |
| 2 | 74.5 | 88.4 | 89.3 |
| 4 | 125.3 | 152.7 | 153.0 |
| 8 | 213.0 | 276.1 | 279.4 |
| 16 | 369.1 | 466.6 | 464.4 |
| 32 | 608.2 | 745.9 | 743.1 |
| 64 | 835.6 | 1064.2 | 1046.6 |
| 128 | 1001.0 | 1257.5 | 1226.2 |
| 256 | 1065.2 | 1346.3 | 1271.5 |

MXFP4 does not serve GLM-5.3-Flash on one MI350P: its routed experts alone are ~162 GB.

GLM-5.3-Flash IQ2R v3 serving on one MI350P at TP1 (ATOM `benchmark_serving`, 1024 in / 1024 out,
FP8 KV, with the `gfx950,128` IQ2R, BF16 GEMM and FP8 block-scale GEMM rows): 75.2 output tok/s at
concurrency 1 (TPOT 13.1 ms), 250.7 at 4, 645.3 at 16, 1344.0 at 64. See ATOM's
`recipes/GLM-5.3-Flash-IQ2R-MI350P.md`.

GLM-5.3-Flash's BF16 (`glm53_flash_bf16_tuned_gemm.csv`) and FP8 block-scale
(`a8w8_blockscale_bpreshuffle_tuned_gemm_glm5_3_flash.csv`) rows for 128 CUs cover the TP1 decode shapes
(M up to 64). The FP8 block-scale module compiles the kernels its tables name: after adding rows, delete
`aiter/jit/module_gemm_a8w8_blockscale_bpreshuffle.so` so it rebuilds.

