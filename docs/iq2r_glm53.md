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

The kernels recognize the format from the packed auxiliary size. A packed IQ2R-v2
auxiliary holds the codebook copies, the `d` table laid out per lane
(`[N/64][K/256][16][4]` for gate/up, `[N/48][K/256][16][4]` for down) and
the two gains. `iq2r_glm53_pack_v2` builds it; `iq2r_glm53_v2_slice_gate_aux`
and `iq2r_glm53_v2_slice_down_aux` take TP shards.

### Recipe

The recipe is fixed: every step below is the code's default, and the
commands under Build reproduce it. The method does not depend on GLM-5.3
and is not tuned to one benchmark; the compiler itself targets GLM-5.3's
shapes.

1. **Only the routed experts are 2-bit; the shared expert stays FP8.**
   - What: the 256 routed experts of each MoE layer are encoded. The shared
     expert keeps its block-FP8 weights in the checkpoint shards, and
     `iq2r_modules` lists only the routed experts, so it runs as an ordinary
     FP8 MLP next to the MoE instead of being fused as expert 256.
   - Why: each token picks 8 of the 256 routed experts, so a routed expert
     sees about 1 token in 32, and its output is scaled by the routing
     weight. Every token passes through the shared expert, with weight 1, so
     the shared expert's quantization error lands on every token in every
     layer. It is also small: keeping it FP8 costs about 2 GB on GLM-5.3
     (75 layers x 37.7 M weights x ~0.7 byte that 2 bits would save).
   - Measured: with everything else the same, the shared expert at 2 bits
     scored 49 and 50 of 60 on AIME-60, against 54 and 54 in FP8.
   - Serving runs the MoE on the 256 routed experts with top-8 routing:
     `iq2r_glm53_moe_out` takes `[256, ...]` stacks with `[tokens, 8]` ids and
     weights (see [iq2r_v2_kernels.md](iq2r_v2_kernels.md)).
2. **Calibration: router-weighted statistics, one conversation per sequence.**
   - What: `calibrate.py --per-row` runs the FP8 model over the calibration
     corpus below, each row (one whole conversation, rendered with the chat
     template) its own sequence of up to 16,384 tokens. For every routed
     expert it records `sum r^2 x^2 / sum r^2` per input channel: x is the
     MoE input for gate/up and `silu(gate) * up` for down, r the token's
     routing weight to that expert.
   - Why per row: as in serving, each token attends only to its own
     conversation, and long agent sessions keep their full context. Packing
     unrelated texts into short windows would give activations the model
     never produces in use.
   - Why router-weighted: an expert's output is multiplied by r, so a
     token's error in that expert reaches the layer output scaled by r, and
     its squared error by r^2. Tokens the router barely sends to the expert
     should barely shape its importance.
3. **Importance: normalized, with a floor and a fallback.**
   - What: that moment divided by its mean, floored at 1e-6. An expert that
     fewer than 256 corpus tokens reach (3 of 19,200 on GLM-5.3) uses its
     weight's column energy `sum_n W[n, k]^2` instead.
   - Why: dividing by the mean keeps the cost on the same scale for every
     matrix; the floor keeps every channel in the cost. With under 256
     tokens the moment is mostly noise from a few tokens, and column energy
     is a data-free stand-in.
4. **Codebook: learned per matrix, from scratch.**
   - What (`iq2r_v2_learn_codebook`): one 512-entry codebook per expert for
     the interleaved gate/up matrix and one for down, seeded with the
     IQ2_XXS grid at 1x and sqrt(2)x, then 4 rounds of importance-weighted
     k-means on 65,536 sampled 8-vectors, alternating with the encoder.
     Codewords stay below the seed maximum, and a round is kept only if the
     weighted cost of the whole matrix drops.
   - Why: each matrix's weights have their own distribution, and a codebook
     fitted to it lowers the error a fixed grid leaves. Starting from the
     public IQ2_XXS grid, not from an earlier study's codebooks, keeps the
     recipe independent of any previous checkpoint. Judging each round on
     the whole matrix stops k-means from overfitting its sample. Gate and up
     share a codebook because separate ones lowered the weighted error by
     only 0.09% for twice the memory.
5. **Encoder: weighted error, exhaustive scale search.**
   - What (`iq2r_v2_encode`, after the IQ2_XXS quantizer of ggml/llama.cpp
     as adapted by antirez): each weight's squared error is weighted by
     `importance * sqrt(sigma^2 + w^2)` (sigma^2 the mean square of its
     256-weight block). A group's starting scale comes from 13 candidate
     least-squares fits; then `d` (three exponents) and every group's level
     (all 16 of c4e4q) are searched exhaustively, re-assigning codewords
     for each choice.
   - Why: importance puts the error where activations are large;
     `sqrt(sigma^2 + w^2)` protects the large weights of a block more than
     its small ones. The search space is small (3 x 16 choices per group),
     so searching it exactly costs little and finds the best scales the
     format can store.
6. **Gain: a slight output shrink.**
   - What: each matrix's output is scaled by one FP32 gain so that
     `<W, Wq> / <W, W> = 0.98` (separate gains for the gate and the up
     rows), applied once per output in the kernel.
   - Why: quantization leaves Wq slightly too large or too small as a
     whole, which one number per matrix corrects. Aiming a little below 1
     follows least squares: when the reconstruction carries noise, scaling
     it slightly down lowers the expected output error. 0.98 was chosen by
     KL divergence on GLM-5.3.

The layer files record the calibration hash and every setting above, so
`--resume` only keeps layers built the same way.

### Calibration corpus

A general mix of 164 conversations, 266,396 tokens. Every reply is
generated by GLM-5.3 (FP8) and rendered with the serving chat template, so
the captured activations are the ones the model produces in use. The mix
covers agent work, chat, reasoning, code, other languages, long context and
plain prose, so no single use dominates the importance; an earlier
agent-heavy corpus (61% SWE-bench trajectories) was replaced by it.

| Share | Rows | Tokens | Source |
|---:|---:|---:|---|
| 24.5% | 10 | 65,251 | agent sessions: 5 mini-swe-agent trajectories on SWE-bench Verified instances not in SWE-bench Lite, 5 Terminal-Bench sessions (each truncated to 8,192 tokens) |
| 19.7% | 48 | 52,364 | chat: 24 topics x 2 turns, half with thinking on, half off |
| 18.5% | 16 | 49,324 | reasoning, thinking on: 8 MATH-500 level 4-5 problems, 8 SuperGPQA questions |
| 12.3% | 32 | 32,784 | code: explain, review, test or refactor Python standard-library functions; write programs |
| 9.6% | 40 | 25,548 | multilingual: 5 tasks in each of zh, es, fr, de, ja, ko, hi, ar |
| 9.4% | 2 | 25,037 | long context: questions on a long WikiText document and on a full source file |
| 6.0% | 16 | 16,088 | prose: WikiText-2 train windows, no reply |

It contains no SWE-bench Lite, AIME or GPQA text. The corpus is not
published with aiter. Within AMD it is `iq2r_v2_calib_corpus_v2.jsonl` (one
`{"kind", "id", "text"}` object per row, 1.1 MB, sha256
`1aab8c266a7a062ece43b5ae2b2a511e9344459f93370faf53a0bcffeb891bc8`) in the
private repository `ssharma4-amd/glm53-iq2r-calib`; ask the IQ2R
maintainers for access. Its calibration artifact records
`corpus_sha256 = 1a1bc46c4e157c8835001ab83636bf009c4d5bccaedb1a06b85f7fe48ad81f8a`
(the hash `calibrate.py` takes over the row texts). Any corpus with a
similar mix can stand in for it.

### Build

    python3 -m aiter.quantization.iq2r.glm5.calibrate --model-dir $FP8 \
        --text-file iq2r_v2_calib_corpus_v2.jsonl --output calibration.pt \
        --per-row --sequence-length 16384 --max-tokens 1000000 \
        --device-map balanced

    # one process per GPU, each on its own MoE layers (3..77) ...
    python3 -m aiter.quantization.iq2r.glm5.compile --model-dir $FP8 \
        --output-dir $OUT --calibration-cache calibration.pt \
        --format iq2r-v2 --device cuda:$i --layers ... --resume
    # ... then one run over all layers writes the shards, index and config
    python3 -m aiter.quantization.iq2r.glm5.compile --model-dir $FP8 \
        --output-dir $OUT --calibration-cache calibration.pt \
        --format iq2r-v2 --resume

Calibration takes about 25 minutes on 8 MI350X; encoding takes about 6 s
per expert (gate/up and down) on one GPU.

### Results

On GLM-5.3, a checkpoint built with this recipe, one run each. It was built
with the same encoder functions and corpus, from statistics captured by an
equivalent script (per-expert cosine to `calibrate.py`'s: median 0.9998),
and served by decoding the experts to FP8 at load (TP8, FP8 KV cache):

| | AIME-60, 2 passes | SWE-bench Lite subset (29) |
|---|---:|---:|
| IQ2R-v2 (this recipe) | 54, 53 | 26 |
| IQ2R-v2, earlier build, shared expert FP8 | 54, 54 | - |
| IQ2R-v2, earlier build, shared expert 2-bit | 49, 50 | 3 |
| MXFP4 (4 AIME runs) | 55, 55, 55, 54 | 22 |

The earlier build used the agent-heavy corpus and codebooks refined from a
previous study instead of learned from scratch; its two rows differ only in
the shared expert. The SWE-bench
numbers come from one run each and move by 2-3 between runs. On
SWE-bench the 2-bit model takes more steps than MXFP4 (median 88 against 76
per task) and about 1.4x the output tokens.

`op_tests/test_iq2r_glm53_v2.py` checks the IQ2R-v2 MoE against a dense
reference at both TP widths, checks that TP slicing commutes with packing,
and (`python3 op_tests/test_iq2r_glm53_v2.py`) times IQ2R against IQ2R-v2.

## Dispatch

`iq2r_glm53_moe_out` looks up the launch choice for the token count in
`aiter/configs/iq2r_glm53_tuned.csv` (keyed on gfx, cu_num, token, inter_dim
and the expert count: rows with 257 experts and top-9 for stacks with the
fused shared expert, rows with 256 experts and top-8 for IQ2R-v2 stacks
without it). Other token counts use the next tuned row up that can run
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
