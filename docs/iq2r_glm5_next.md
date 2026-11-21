# GLM-5.3-Flash IQ2R MoE on gfx950

GLM-5.3-Flash (`glm5_next`: 45 text layers, 42 of them MoE, hidden 4096, 288 routed experts, top-8, one shared
expert, SwiGLU clamped at 10) runs its routed experts from the same 2-bit IQ2R-v2 weights and kernels as GLM-5.3
(`docs/iq2r_glm53.md`), at TP4 (`inter_dim` 512) and TP8 (`inter_dim` 256) on MI355X/MI350X (gfx950). This page
covers what differs; format, recipe, dispatch and kernel design are GLM-5.3's.

- 1,024-entry codebook with LDLQ on gate/up (layer Hessians and the decision bucket), as GLM-5.3's recipe; 512
  entries also build and run.
- The shared expert stays 8-bit: block FP8 in the checkpoint and, for batches of up to 256 tokens, MXFP8 inside
  the IQ2R launches.

## What differs from GLM-5.3

The kernels are instantiated per model shape (`Glm5MoeShape`, `csrc/include/iq2r_glm5_shape.h`;
`IQ2RGlm5Shape` `GLM53` / `GLM5_NEXT` in `aiter/iq2r_glm53.py`). GLM-5.3's instantiations compile to the same
instructions as before.

| | GLM-5.3 | GLM-5.3-Flash |
|---|---|---|
| Hidden size (gate/up K, down N) | 6144 | 4096 |
| Routed experts | 256 | 288 |
| SwiGLU | `silu(gate) * up` | `silu(min(gate, 10)) * clamp(up, -10, 10)`, shared expert included |
| Down rows stored | 6144 | 4608 (4096 + 512 zero rows) |
| Decode-gate K slice per wave | 6 K tiles | 4 K tiles |
| Packed layouts (512 / 1,024 entries) | `glm53-packed-v2` / `-v3` | `glm5next-packed-v2` / `-v3` |
| Decode sort block | 256 threads | 512 threads |

- **Down rows.** The down kernels tile N in 384- and 768-column steps and the per-token down in 48-row
  triplets; 4096 is not a multiple of 768, so the packed down matrices carry 512 zero rows (4608 = 6 x 768).
  The padding rows are computed and not stored: route outputs are `[routes, 4608]` and the reductions read the
  first 4096 columns. The fused shared expert's MXFP8 down weights are zero-padded the same way. The padding adds
  12.5% to the down weights, about 4% of the expert bytes.
- **K slices.** Each decode-gate wave sums K/8 = 512 inputs (4 K tiles, against 6 for GLM-5.3). A 1,024-entry
  gate/up quad therefore stores 4-tile slices followed by an 8-byte tenth-bit word per lane, so the packed gate/up
  layout depends on the model; the canonical IQ2R-v2 records do not.
- **Dispatch.** The launchers pick the shape from the expert count of the packed stacks (256 or 288), the routing
  kernels from the hidden size. Tuned rows are keyed by `model_dim` and `expert`.

## Building the checkpoint

The same two tools, on the block-FP8 GLM-5.3-Flash checkpoint (`model_type` `glm5_next`, text model under
`model.language_model.`). `calibrate` loads it with Transformers' `glm5_next` support (tested with 5.16.1) through
its multimodal model class, with the block-FP8 weights dequantized to BF16 at load (the KDA forget-gate
projections, stored in BF16, keep their weights); only the text model runs. The model needs about 210 GB per GPU on
4 GPUs.

    python3 -m aiter.quantization.iq2r.glm5.calibrate --model-dir $FP8 \
        --text-file corpus.jsonl --output calibration.pt \
        --hessian-dir hessians --decision-text-file decisions.jsonl

    # one process per GPU, each on its own MoE layers (3..44) ...
    SLICES=(3-13 14-24 25-34 35-44)
    for i in 0 1 2 3; do
      HIP_VISIBLE_DEVICES=$i python3 -m aiter.quantization.iq2r.glm5.compile \
          --model-dir $FP8 --output-dir $OUT --calibration-cache calibration.pt \
          --codebook-entries 1024 --ldlq --hessian-dir hessians \
          --layers ${SLICES[$i]} &
    done; wait
    # ... then one run over all layers writes the shards, index and config
    python3 -m aiter.quantization.iq2r.glm5.compile --model-dir $FP8 \
        --output-dir $OUT --calibration-cache calibration.pt \
        --codebook-entries 1024 --ldlq --hessian-dir hessians

Measured on 4 MI350X (PyTorch 2.11 on ROCm, Transformers 5.16.1, no `kernels` package):

- Calibration: about 50 minutes (importance over 266K corpus tokens in about 18 minutes, then the `H_all` and
  decision passes). Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: the PyTorch KDA path of a
  16,384-token row needs 12-27 GB at once, and with 210 GB of weights per GPU fragmentation otherwise exhausts the
  rest.
- Compile: about 29 minutes per MoE layer per GPU (288 experts: encode, LDLQ, checks), about 5.3 hours for the
  42 layers on 4 GPUs.
- Checkpoint: 120 GB, against 328 GB for the FP8 checkpoint. The shared experts, the MTP layer (45) and the vision
  tower keep their checkpoint weights; `iq2r_modules` names ATOM's module paths (`model.layers.N.mlp.experts*`).

The calibration and decision corpora of `docs/iq2r_glm53.md` apply as they are: GLM-5.3-Flash has the same
tokenizer and renders text-only conversations identically with its chat template (its template adds image,
video and audio placeholders). The capture recorded 160,731 scored positions over the calibration corpus and
2,360 decision positions over the decision corpus (1,500 of them in SWE-agent text). Every routed expert was
reached by at least 33 corpus tokens; 19 of the 12,096 fell under the 256-token threshold and use their weight
energy as importance.

## Correctness

`op_tests/test_iq2r_glm53.py` runs every MoE test on GLM-5.3-Flash too: TP4 and TP8, 512- and 1,024-entry
codebooks, against the dense reference with the clamped SwiGLU, with and without the fused shared expert, every
explicit config, invalid routes and chunked prefill, and the bit-exact decode of every down kernel.
`test_iq2r_glm53_unpack.py` covers both shapes' pack/unpack and TP slicing, and
`test_iq2r_glm5_next_compile.py` the glm5_next checkpoint tools.

## MoE timing

`iq2r_glm53_moe_out` at the tuned configs (CUDA-graph replay, cold experts, synthetic 1,024-entry weights,
shared expert fused up to M256), one MI350X, µs. GLM-5.3's rows of `iq2r_glm53_tuned.csv` were measured on an
MI355X, which is 5-20% faster on these kernels.

| M | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TP4, shared fused | 22 | 24 | 35 | 50 | 73 | 111 | 146 | 176 | 191 | | | | |
| TP4, routed only | 18 | 21 | 32 | 49 | 70 | 108 | 142 | 165 | 175 | 234 | 284 | 375 | 641 |
| TP8, shared fused | 19 | 20 | 23 | 33 | 46 | 62 | 82 | 97 | 109 | | | | |
| TP8, routed only | 16 | 18 | 21 | 32 | 46 | 62 | 80 | 92 | 97 | 126 | 196 | 252 | 415 |

## Serving against MXFP4

ATOM at TP4 on 4 MI350X, FP8 KV cache, `--max-num-seqs 16`, CUDA graphs. IQ2R: the 1,024-entry LDLQ checkpoint
on the native kernels, shared expert in the IQ2R launches up to 256 tokens. MXFP4: the FP8 checkpoint with the
routed and shared experts quantized to MXFP4 at load (`*expert*`), with MXFP8 activations (A8W4) or FP4
activations (A4W4, ATOM's default). Non-expert layers are the checkpoint's block FP8 in every arm.

`benchmark_serving`, random 1024-token prompts and 512 output tokens (`--ignore-eos`), 4 x concurrency prompts
(at least 16), the second of two repetitions; output tokens/s (mean TPOT, mean TTFT):

| Concurrency | MXFP4 A4W4 | MXFP4 A8W4 | IQ2R | IQ2R / A4W4 | IQ2R / A8W4 |
|---|---|---|---|---|---|
| 1 | 105 (9.4 ms, 0.08 s) | 98 (10.1 ms, 0.09 s) | 100 (9.9 ms, 0.08 s) | 0.95 | 1.02 |
| 4 | 391 (10.0 ms, 0.13 s) | 368 (10.6 ms, 0.16 s) | 380 (10.3 ms, 0.14 s) | 0.97 | 1.03 |
| 16 | 1337 (11.7 ms, 0.16 s) | 1253 (12.5 ms, 0.17 s) | 1333 (11.8 ms, 0.13 s) | 1.00 | 1.06 |
| 64 (queued at 16) | 1367 (11.6 ms) | 1262 (12.5 ms) | 1342 (11.8 ms) | 0.98 | 1.06 |

IQ2R is level with A4W4 at 16 concurrent requests and 3-5% behind at 1-4; it is 2-6% ahead of A8W4.

Memory per GPU from ATOM's budget at 0.9 utilization: weights plus peak activations 32.8 GB (MXFP4 46.7 GB A8W4,
48.1 GB A4W4), which leaves 7-8% more KV cache (521,546 against 486,840 and 483,326 FP8 blocks). The packed
checkpoint is 120 GB.

## Quality against MXFP4

Paired on one day, TP4, against MXFP4 on the same FP8 checkpoint: SWE-bench and Terminal-Bench on the servers
above (Terminal-Bench with `--max-model-len 262144`), AIME with a BF16 KV cache and `--max-num-seqs 64`.

| | IQ2R, 1,024 entries, LDLQ + decision corpus | MXFP4 A4W4 | MXFP4 A8W4 |
|---|---|---|---|
| AIME-60 (T 1.0, top-p 0.95, 2 passes, 24K thinking + 8K answer) | 52, 53 | 51, 55 | 53, 53 |
| AIME median output tokens | 12.8K, 10.4K | 10.0K, 11.2K | 9.6K, 7.4K |
| AIME answers forced at the budget | 21, 21 | 17, 20 | 15, 17 |
| SWE-bench Lite, the 15 split tasks of `docs/iq2r_glm53.md` (2 runs) | 8, 6 | 8, 4 | 8, 10 |
| Terminal-Bench 2.0, 3 hard tasks (5 attempts each) | 8/15 | 5/15 | 10/15 |

- AIME-60 and the SWE-bench settings are those of `docs/iq2r_glm53.md` (mini-swe-agent 2.4.5, temperature 0.95,
  thinking capped at 16,384 tokens plus 4,096 answer tokens, 150 steps, no network); the 15 tasks were chosen
  on GLM-5.3, not on this model.
- AIME is level within its run-to-run spread (about +-3); IQ2R thinks longer than MXFP4 A8W4 (median output
  about 1.35x) and is forced to answer at the budget more often; A4W4 sits between them.
- SWE-bench: IQ2R 14 of 30, A4W4 12, A8W4 18. Against A8W4, in the first run each arm solved 3 tasks the other
  did not, in the second IQ2R 2 and A8W4 6.
- Terminal-Bench 2.0: the three tasks of `docs/iq2r_glm53.md` (cobol-modernization, write-compressor,
  torch-tensor-parallelism) with Harbor 0.25.0 and the terminus-2 agent, temperature 0.95, the same thinking budget,
  9 concurrent trials and the standard timeouts. Per task IQ2R 5, 0, 3; A4W4 5, 0, 0 (one torch-tensor-parallelism
  trial lost its sandbox's tmux session); A8W4 4, 1, 5. Most failures are the 900 s agent time limit.
- These are small, hard-case subsets: 15-task SWE runs move by a few tasks and the Terminal-Bench subset by about
  +-4 of 15 between runs (GLM-5.3's MXFP4 A8W4 scored 13 and 9 in two identical runs).
- No degenerate output in any arm (no `</think>` collapses, no runaway generations); 1-3 agent runs per arm
  and repetition ended at the context limit.

## Not measured

- The full SWE-bench Lite and Terminal-Bench 2.0, GSM8K and the teacher-forced KL against FP8 on this recipe.
- MI355X; the 512-entry recipe on GLM-5.3-Flash; MTP; longer contexts than the agent runs used.
