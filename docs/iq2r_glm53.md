# GLM-5.3 IQ2R MoE on gfx950

`aiter/iq2r_glm53.py` runs the GLM-5.3 routed experts (256 routed + the
shared expert fused as expert 256, top-9, hidden 6144) from 2-bit IQ2R
weights on MI355X, at TP4 (`inter_dim` 512) and TP8 (`inter_dim` 256).

IQ2R was developed by the MK1 team at AMD. Its learned codebooks are seeded
from the IQ2_XXS grid of ggml/llama.cpp (MIT).

## Layout

The compiler interleaves the gate and up rows of each expert
(gate0, up0, gate1, up1, ...) before encoding. `iq2r_glm53_pack` then turns
the per-expert IQ2R records into one packed stack per layer: it folds the
codebook signs into the records and regroups the gate/up records into quads
of four 16-column blocks. The packed stack is the only copy kept on the
GPU. A checkpoint stored in this layout can be sliced to any supported TP
rank with contiguous slices (`iq2r_glm53_slice_gate`, and the generic
`iq2r_slice_*` helpers for down and aux).

## Building the checkpoint

Two tools turn the block-FP8 GLM-5.3 checkpoint into the packed checkpoint
ATOM serves. They need `safetensors` (`pip install safetensors`, or the
`iq2r` extra).

1. `aiter.quantization.iq2r.glm5.calibrate` runs the FP8 model in Hugging Face
   Transformers (5.16 or newer, plus `kernels` for the FP8 matmul) over a
   text corpus, prefill only. For every routed and shared expert projection
   it records the importance of each input channel k,
   `sum_t r_t^2 x_t[k]^2 / sum_t r_t^2`, where x_t is the projection input
   and r_t the token's routing weight (1 for the shared expert). The
   published checkpoint used 32,768 tokens of the WikiText-2 train split:

       python3 - <<'EOF'
       import json
       from datasets import load_dataset
       rows = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
       texts = [row["text"].strip() for row in rows if row["text"].strip()]
       with open("wikitext2-train.jsonl", "w") as f:
           f.writelines(json.dumps({"text": t}, ensure_ascii=False) + "\n" for t in texts)
       EOF
       # 23,767 rows, sha256 df1912f6b3c51b485ff7181cdde30f9ccd303c02031fea262179071ee1cbc1d4
       python3 -m aiter.quantization.iq2r.glm5.calibrate --model-dir $FP8 \
           --text-file wikitext2-train.jsonl --output calibration.pt \
           --device-map balanced

   The defaults (64 sequences of 512 tokens) are that recipe. Spread over 4
   MI355X GPUs, loading takes about 30 minutes and the capture about 10. An
   expert that no corpus token reaches gets one forced pass over its layer's
   inputs, which the artifact records; any still unobserved is an error. On
   a host without Hugging Face Hub access, `--fp8-kernel-dir` loads a local
   build of `kernels-community/finegrained-fp8`.
2. `aiter.quantization.iq2r.glm5.compile` writes the packed checkpoint
   (`iq2r_layout` = `glm53-packed-v1`). For each MoE layer (3-77) it encodes
   the 256 routed experts and the shared expert, fused as expert 256, on the
   GPU, weighting every error term by the calibrated importance. It packs the
   layer with `iq2r_glm53_pack` and writes
   `iq2r-layer-NNNN-{gate-up,down}.safetensors`. The layers can be split
   across GPUs:

       SLICES=(3-12 13-22 23-32 33-42 43-51 52-60 61-69 70-77)
       for i in 0 1 2 3 4 5 6 7; do
         python3 -m aiter.quantization.iq2r.glm5.compile --model-dir $FP8 \
             --output-dir $OUT --calibration-cache calibration.pt \
             --device cuda:$i --layers ${SLICES[$i]} --resume &
       done; wait
       python3 -m aiter.quantization.iq2r.glm5.compile --model-dir $FP8 \
           --output-dir $OUT --calibration-cache calibration.pt --resume

   The last run covers every layer. Instead of re-encoding the existing layer
   files it checks their tensor keys, shapes, dtypes and format metadata,
   and that they were encoded with the same calibration (its sha256),
   `--iterations` and `--sample-vectors`. It then copies the remaining FP8
   tensors into new shards and writes the index, config and tokenizer. The
   config lists
   `iq2r_modules` patterns for the routed and shared experts of each compiled
   layer, so the experts of the MTP layer (78) stay FP8.

The result is about 235 GB and loads at any supported TP width with no
load-time relayout.

## IQ2R-v2

IQ2R-v2 (`aiter/ops/iq2r_v2.py`) keeps IQ2R's records, learned codebooks
and kernels and changes only how a 32-weight group is scaled:

| | IQ2R | IQ2R-v2 |
|---|---|---|
| per row and 256 inputs | - | E8M0 `d` |
| per 16-row N block | E8M0 base exponent | - |
| per 32 weights (4 bits) | exponent delta | level `l`: `m[l % 4] * 2^(l / 4)`, `m = 2^(j/4)` |
| codewords | E4M3 codebook | `E4M3(codebook * m[c])`, four pre-scaled copies |
| per projection | - | FP32 output gain (gate and up separately) |
| bits per weight | ~2.25 | ~2.28 |

The power-of-two parts (`d` and `2^(l / 4)`) form the scaled-MFMA E8M0
operand, the mantissa `m[c]` selects a codebook copy in LDS, and the gain is
applied once per output, so the kernels run the same MFMA stream. The
copies' E4M3 rounding is part of the format: the encoder, the reference
decoder (`iq2r_v2_materialize`) and the kernels agree bit for bit. As in
IQ2R, gate and up share one codebook (learned on the interleaved matrix); on
six GLM-5.3 experts, separate gate and up codebooks lowered the weighted
weight error by only 0.09%, for twice the codebook memory.

In a lane group `g`, the metadata high nibble holds the exponent of block
`g`, whose MFMA scale that lane supplies, and the class of block `s(g)`,
`s = (0, 2, 1, 3)`. The classes a lane needs for its own codewords are then
in its own nibble or its neighbour group's, one `v_permlane16_swap` away.

The kernels recognize the format from the packed auxiliary size, so
`iq2r_glm53_moe_out` and ATOM's call are unchanged. A packed IQ2R-v2
auxiliary holds the codebook copies, the `d` table laid out per lane
(`[N/64][K/256][16][4]` for gate/up, `[N/48][K/256][16][4]` for down) and
the two gains. `iq2r_glm53_pack_v2` builds it; `iq2r_glm53_v2_slice_gate_aux`
and `iq2r_glm53_v2_slice_down_aux` take TP shards.

The compiler builds it with `--format iq2r-v2`:

    python3 -m aiter.quantization.iq2r.glm5.compile --model-dir $FP8 \
        --output-dir $OUT --calibration-cache calibration.pt \
        --format iq2r-v2 --device cuda:$i --layers ... --resume

Each step of the recipe:

1. Importance: the calibration artifact's router-weighted second moment
   (above), from the calibration corpus below rather than WikiText alone.
2. Codebook (`iq2r_v2_learn_codebook`): per matrix (one for the interleaved
   gate/up matrix, one for down), seeded with the IQ2_XXS
   grid at 1x and sqrt(2)x, importance-weighted k-means on 65,536 sampled
   8-vectors alternating with the encoder (`--iterations` rounds). Codewords
   stay below the seed maximum and a round is kept only if the matrix's
   weighted cost drops.
3. Encoder (`iq2r_v2_encode`), after the IQ2_XXS quantizer of ggml/llama.cpp
   as adapted by antirez: each weight's error is weighted by
   `importance * sqrt(sigma^2 + w^2)` (sigma^2 the mean square of its
   256-weight block); a group's starting scale comes from 13 candidate
   least-squares fits; then `d` (three exponents) and every group's level
   (all 16) are searched exhaustively with codeword re-assignment.
4. Gain: each matrix's output gain makes `<W, Wq> / <W, W>` equal
   `--target-gain` (0.98; a slight shrink lowered KL in the study).

### Calibration corpus

The evaluated IQ2R-v2 checkpoint was calibrated on 40 windows of 8,192
tokens (321,818 tokens), chosen to match how the model is used for agentic
coding and reasoning. Agent and math text dominate because calibrating on one
source raised the expert error on the others by 12-29% in a pilot.

| Share | Windows | Source |
|---:|---:|---|
| 61% | 24 | GLM-5.3 mini-swe-agent trajectories on SWE-bench Verified instances that are not in SWE-bench Lite, rendered with the serving chat template and bash tool |
| 20% | 8 | GLM-5.3 (FP8) reasoning traces on MATH-500 level 4-5 problems, chat template |
| 6% | 2 + 1 | multi-turn chat built from other MATH problems (final answers, no thinking), plus one window of general chat prompts |
| 5% | 2 | WikiText-2 train |
| 8% | 3 | Python standard-library source |

No SWE-bench Lite or AIME text is included, and the windows exclude the
sequences used to measure KL (64-gram overlap with them is 0.2%).

The corpus is not published with aiter. Within AMD it is the text JSONL
`iq2r_v2_calib_corpus.jsonl` (one `{"kind", "text"}` object per window,
sha256
`1d2595d86c63046090db291e6b3e0650bde09352cbe8e237221c17006e5e3bcc`, 1.2 MB),
kept in the private repository `ssharma4-amd/glm53-iq2r-calib` and next to the
evaluated checkpoint (`GLM-5.3-IQ2Rv2-Brw`); ask the IQ2R maintainers for
access. It feeds
the calibration tool directly:

    python3 -m aiter.quantization.iq2r.glm5.calibrate --model-dir $FP8 \
        --text-file iq2r_v2_calib_corpus.jsonl --output calibration.pt \
        --sequence-length 8192 --max-tokens 327680 --device-map balanced

Any corpus with the same mix of agent trajectories, long reasoning, code,
chat and prose can stand in for it.

On GLM-5.3 (12 routed matrices, three layers) this matches the weight error
of the evaluated IQ2R-v2 checkpoint (importance-weighted relative MSE 0.0737
for both). That checkpoint, served at 2.32 bits per weight, scored 23/29 on a
SWE-bench Lite subset where MXFP4 scored 22/29 under the same settings.

`op_tests/test_iq2r_glm53_v2.py` checks the IQ2R-v2 MoE against a dense
reference at both TP widths, checks that TP slicing commutes with packing,
and (`python3 op_tests/test_iq2r_glm53_v2.py`) times IQ2R against IQ2R-v2.

## Dispatch

`iq2r_glm53_moe_out` looks up the launch choice for the token count in
`aiter/configs/iq2r_glm53_tuned.csv` (keyed on gfx, cu_num, token and
inter_dim). Other token counts use the next tuned row up that can run
them, else `_default_config`.
`AITER_LOG_TUNED_CONFIG=1` logs the choice once per shape. Each row names:

- the gate kernel (`decode`, `nobarrier`, `prefill`) and its grid;
- the down kernel (`route9`, `single`, `packed`, `ordered`, `prefill`) and its grid;
- the number of prefill down chunks.

`AITER_CONFIG_IQ2R_GLM53` points the lookup at a different CSV. To tune
new shapes, add them to `aiter/configs/iq2r_glm53_untuned.csv` and run
`python3 csrc/kernels/iq2r/iq2r_glm53_tune.py`, which adds them to
`iq2r_glm53_tuned.csv`. The tuner uses the common aiter tuner flags:
`--all` retunes every shape, `--run_config [CSV]` times the launch choices
a CSV makes, and `-o2 FILE` keeps the timing of every candidate.

The tuner checks each candidate against a dense reference, then times it
by CUDA graph replay, as serving runs it; eager timing adds host overhead
that hides the best small-M launches. Each call in the graph reads its own
copy of the layer weights and its own routing, with at least 1 GiB of
expert weights per replay, so the weights come from HBM as they do when a
serving step walks the MoE layers. The decode kernels accept up to 1024
tokens so the CSV can choose them for MTP verify batches, which are
(1 + speculative tokens) x concurrency.

`python3 op_tests/test_iq2r_glm53.py` times the tuned and default configs
over a token sweep (`-t`, `-i` select the token counts and widths), prints
a summary table, then runs the correctness tests.

## Results

ATOM serving on MI355X, `benchmark_serving`, ISL 1024, OSL 1024, 10 × C
prompts at concurrency C = 1..256, FP8 KV cache, CUDA graphs, no IQ2R
environment variables. Both arms quantize the non-expert layers online to
PTPC FP8. MXFP4 is stock ATOM with the routed experts quantized online to
MXFP4. IQ2R loads the packed IQ2R experts, about 52.0 GB per GPU at TP4 and
26.1 GB at TP8.

IQ2R output throughput is 7-21% above MXFP4 at TP4 (C = 1: 92.0 vs
82.6 tokens/s; C = 256: 5143.5 vs 4732.0) and 9-20% above at TP8 (C = 1:
90.2 vs 79.5; C = 256: 6637.5 vs 5808.4). The MXFP4 runs used an older
upstream ATOM base, so part of each difference may come from upstream
changes.

## Tests

`op_tests/test_iq2r_glm53.py` checks the packed MoE against a dense Torch
reference at both TP widths. The other `op_tests/test_iq2r_*.py` files cover
the format, the host and device encoders and the two checkpoint tools; the
calibration test skips without Transformers.
