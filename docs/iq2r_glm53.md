# GLM-5.3 IQ2R-v2 MoE on gfx950

`aiter/iq2r_glm53.py` runs the GLM-5.3 routed experts (256 experts, top-8, hidden 6144) from 2-bit IQ2R-v2
weights on MI355X, at TP4 (`inter_dim` 512) and TP8 (`inter_dim` 256). The shared expert stays FP8 and runs as an
ordinary MLP next to the MoE.

IQ2R was developed by the MK1 team at AMD. Its learned codebooks are seeded from the IQ2_XXS grid of
ggml/llama.cpp (MIT).

## Format

`aiter/ops/iq2r_v2.py`. Each group of 8 weights is a 9-bit index into a learned 512-entry E4M3 codebook (one
per expert for gate/up, learned on the interleaved gate0, up0, gate1, ... rows, and one for down) plus 8
explicit sign bits. The scales:

| | |
|---|---|
| per row and 256 inputs | E8M0 `d` |
| per 32 weights (4 bits) | level `l`: `m[l % 4] * 2^(l / 4)`, `m = 2^(j/4)` |
| codewords | `E4M3(codebook * m[c])`, four pre-scaled copies |
| per projection | FP32 output gain (gate and up separately) |
| bits per weight | ~2.28 |

The power-of-two parts (`d` and `2^(l / 4)`) form the scaled-MFMA E8M0
operand, the mantissa `m[c]` selects a codebook copy in LDS, and the gain is
applied once per output, so there is no per-weight arithmetic. The
copies' E4M3 rounding is part of the format: the encoder, the reference
decoder (`iq2r_v2_materialize`) and the kernels agree bit for bit. Gate and
up share one codebook; on six GLM-5.3 experts, separate gate and up
codebooks lowered the weighted weight error by only 0.09%, for twice the
codebook memory.

In a lane group `g`, the metadata high nibble holds the exponent of block
`g`, whose MFMA scale that lane supplies, and the class of block `s(g)`,
`s = (0, 2, 1, 3)`. The classes a lane needs for its own codewords are then
in its own nibble or its neighbour group's, one `v_permlane16_swap` away.

`iq2r_glm53_pack_v2` turns the per-expert records into one packed stack per
layer (the four 9-bit indices of an atom pack as low32 + high4, the gate/up
record signs transpose into bit planes, and the gate/up records regroup into
quads of four 16-column blocks). The packed auxiliary holds the codebook copies, the `d` table laid out per lane
(`[N/64][K/256][16][4]` for gate/up, `[N/48][K/256][16][4]` for down) and
the two gains. Every transform is per record, so a packed checkpoint is cut
to any supported TP rank with contiguous slices (`iq2r_glm53_slice_gate`,
`iq2r_glm53_v2_slice_gate_aux`, `iq2r_slice_input_data`,
`iq2r_glm53_v2_slice_down_aux`); `iq2r_glm53_unpack_v2` inverts the packing.

## Building the checkpoint

Two tools turn the block-FP8 GLM-5.3 checkpoint into the packed checkpoint
(`iq2r_layout` = `glm53-packed-v2`) ATOM serves. They need `safetensors`
(`pip install safetensors`, or the `iq2r` extra), and calibration needs
Hugging Face Transformers (5.16 or newer, plus `kernels` for the FP8 matmul).

    python3 -m aiter.quantization.iq2r.glm5.calibrate --model-dir $FP8 \
        --text-file corpus.jsonl --output calibration.pt

    # one process per GPU, each on its own MoE layers (3..77) ...
    SLICES=(3-12 13-22 23-32 33-42 43-51 52-60 61-69 70-77)
    for i in 0 1 2 3 4 5 6 7; do
      HIP_VISIBLE_DEVICES=$i python3 -m aiter.quantization.iq2r.glm5.compile \
          --model-dir $FP8 --output-dir $OUT --calibration-cache calibration.pt \
          --layers ${SLICES[$i]} &
    done; wait
    # ... then one run over all layers writes the shards, index and config
    python3 -m aiter.quantization.iq2r.glm5.compile --model-dir $FP8 \
        --output-dir $OUT --calibration-cache calibration.pt

Calibration runs the FP8 model over the corpus (one conversation per
sequence, up to 16,384 tokens each and 1M in total) and takes about 25
minutes on 8 MI350X. An expert that no corpus token reaches gets one forced
pass over its layer's inputs, which the artifact records. On a host without
Hugging Face Hub access, `--fp8-kernel-dir` loads a local build of
`kernels-community/finegrained-fp8`.

Encoding takes about 6 s per expert (gate/up and down) on one GPU. Layer
files already in `$OUT` that were built from the same calibration and recipe
are kept, so the last run only checks them, copies the remaining FP8
tensors (the shared expert and MTP layer 78 included) into new shards and
writes the index, config and tokenizer. The result is about 241 GB.

## Recipe

The recipe is fixed: every step below is the code's default, and the
commands under "Building the checkpoint" reproduce it. The method does not depend on GLM-5.3
and is not tuned to one benchmark; the compiler itself targets GLM-5.3's
shapes.

1. **Only the routed experts are 2-bit; the shared expert stays FP8.**
   - What: the 256 routed experts of each MoE layer are encoded. The shared
     expert keeps its block-FP8 weights in the checkpoint shards, and
     `iq2r_modules` lists only the routed experts, so it runs as an ordinary
     FP8 MLP next to the MoE.
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
     weights (see Kernels).
2. **Calibration: router-weighted statistics, one conversation per sequence.**
   - What: `calibrate.py` runs the FP8 model over the calibration
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
     public IQ2_XXS grid keeps the recipe independent of any previous
     checkpoint. Judging each round on
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

The layer files record the calibration hash and every setting above, so a
compile run only keeps existing layer files built the same way.

### Calibration corpus

A general mix of 164 conversations, 266,396 tokens. Every reply is
generated by GLM-5.3 (FP8) and rendered with the serving chat template, so
the captured activations are the ones the model produces in use. The mix
covers agent work, chat, reasoning, code, other languages, long context and
plain prose, so no single use dominates the importance.

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
published with aiter. It is `iq2r_v2_calib_corpus_v2.jsonl` (one
`{"kind", "id", "text"}` object per row, 1.1 MB, sha256
`1aab8c266a7a062ece43b5ae2b2a511e9344459f93370faf53a0bcffeb891bc8`), and its
calibration artifact records
`corpus_sha256 = 1a1bc46c4e157c8835001ab83636bf009c4d5bccaedb1a06b85f7fe48ad81f8a`
(the hash `calibrate.py` takes over the row texts). Any corpus with a
similar mix can stand in for it; the next section describes how this one
was built.

#### Building a comparable corpus

File format. `--text-file` takes one row per nonempty line. A line starting
with `{` is JSON and its `text` (else `prompt`, else `content`) field is the
row; any other line is the row as is. Each row is tokenized exactly as
written, with no special tokens added, and fed as its own sequence of up to
16,384 tokens.

Render with the chat template first. `calibrate.py` does not apply a chat
template. For conversations, write each whole conversation already rendered
with the checkpoint's `chat_template.jinja` (role tags and `<think>` blocks
included), for example
`tokenizer.apply_chat_template(messages, tokenize=False)`. Raw prompt and
reply text gives different activations from the ones the model sees in
serving. Plain prose rows need no template.

Generation. Every reply comes from the FP8 checkpoint served behind an
OpenAI-compatible endpoint, sampled at temperature 1.0 and top_p 0.95. Prompts
were drawn with a fixed seed. The GLM-5.3 template always thinks:

- thinking-on rows keep the reasoning as `<think>...</think>` before the
  answer;
- thinking-off rows keep only the answer, as earlier turns appear in a
  multi-turn chat.

| Kind | Prompts | Thinking | Max reply tokens |
|---|---|---|---|
| agent | 5 mini-swe-agent trajectories on SWE-bench Verified instances not in Lite; 5 Terminal-Bench (terminus-2) sessions; each rendered whole and cut to its first 8,192 tokens | as the agent runs | per the agent harness |
| reasoning | 8 MATH-500 level 4-5 problems; 8 SuperGPQA (train, difficulty "hard") multiple-choice questions | on | 6,000 |
| code | 24 Python standard-library functions or classes of 15-60 lines, each with one of explain / review / write tests / refactor; 8 write-a-program tasks in Python, Rust, SQL, C++, bash, TypeScript and Go | alternating | 1,500 |
| chat | 24 everyday topics (advice, explanations, writing), each twice (thinking on and off), with one follow-up turn | on / off | 700 per turn |
| multilingual | 5 everyday tasks in each of zh, es, fr, de, ja, ko, hi, ar | alternating | 700 |
| long context | ~60K characters of WikiText-2 train with questions on its topics; the source of Python's `argparse.py` (~55K characters) with a question on its architecture | off | 1,500 |
| prose | 16 windows of WikiText-2 train, first 1,024 tokens each, no template | - | - |

Thinking-off requests add 3,000 tokens to the budget for the reasoning they
discard. Keep the rows away from any benchmark you will evaluate on; this
corpus leaves out SWE-bench Lite, AIME and GPQA.

Size. About 160 rows and 250-300K tokens leaves most of the 19,200 routed
experts with at least 256 routed tokens. Experts below that fall back to
weight energy, so a much smaller corpus moves more experts onto the
fallback. `calibrate.py` stops at 1M tokens.

## Dispatch

`iq2r_glm53_moe_out` looks up the launch choice for the token count in
`aiter/configs/iq2r_glm53_tuned.csv` (keyed on gfx, cu_num, token and
inter_dim). Other token counts use the next tuned row up that can run
them, else `_default_config`.
`AITER_LOG_TUNED_CONFIG=1` logs the choice once per shape. Each row names:

- the gate kernel (`nobarrier` for decode, `prefill`) and its grid;
- the down kernel (`route9`, `single`, `ordered`, `prefill`) and its grid;
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

## Kernels

`iq2r_glm53_moe_out` runs sort/quantize, gate+up GEMM with SwiGLU and MXFP8 requantization, down GEMM and the
weighted top-8 reduction.

**256 experts, top-8.** The checkpoint keeps the shared expert in FP8 outside the MoE, so the MoE gets
`[256, ...]` stacks with `[tokens, 8]` ids and weights.

- Routes keep nine slots per token (`token * 9 + k`). With top-8, slot 8 is empty: the sort kernels
  (`glm53_route_id`) read `[tokens, 8]` ids and treat slot 8, negative ids and ids >= 256 as invalid.
- The sorts group routes by expert (0-255, then the invalid routes) and build the gate and down task lists from
  the groups. Invalid routes get no tasks and `scatter = -1`, so the reductions skip them.
- The reductions read `[tokens, 8]` weights; the route9 down runs 8 route waves (2 per SIMD).

| Stage | Kernel | Notes |
|---|---|---|
| M1 gate | `glm53_gate_m1_kernel` | Loads copy 0 of the book (4 KB) and forms copies 1-3 in LDS; non-temporal weight loads |
| Decode gate | `glm53_gate_decode_nobarrier_v2_kernel` | Per-XCD item ranges; the book is formed after the item's loads are issued; buffer-descriptor K loop with scalar offsets; per-atom MFMA scale `op_sel`; non-temporal weight loads; 2 workgroups per CU |
| Prefill gate | `glm53_gate_prefill_kernel` | Book formed from copy 0; 2 waves per SIMD |
| M1-M4 down | `glm53_down_route9_v2_kernel` | Each route wave forms its own book in LDS; activation row by DMA; 2 triplets per CTA at M4 |
| Decode down | `glm53_down_tp4_kernel`, `glm53_down_tp8_ordered_kernel` | K tiles outermost; levels and scale bytes once per K tile; 16 KB of copies staged by DMA |
| Prefill down | `glm53_down_prefill_tp4/tp8_kernel` | Book formed from copy 0; levels and scale bytes once per K tile; TP4 items cover 768 columns |

**Copy formation.** The packed auxiliary stores the four 4 KB codebook copies `E4M3(codebook * 2^(c/4))`. Most
kernels read only copy 0 and form copies 1-3 in LDS with integer byte adds (`iq2r_v2_copy_words`): for a
normal E4M3 byte `b`, copy c's byte is `b + delta_c[b & 7]`. This is exact only for bytes 0x08..0x78 (2^-6 to
256): `iq2r_v2_learn_codebook` keeps its entries in that range, and `iq2r_glm53_pack_v2` rejects any codebook
byte outside it (`iq2r_v2_check_codebook`). Every codebook byte of the GLM-5.3 IQ2R-v2 checkpoint is in
0x08..0x4f.


## Correctness

- **Tests** (`op_tests/test_iq2r_glm53.py`, `op_tests/test_iq2r_glm53_unpack.py`):
  - the MoE against a dense reference, TP4 and TP8, M = 1..4096, with the tuned and with explicit launch
    choices;
  - ids -1 and 256 contribute nothing;
  - the copy-formation formula on every byte in 0x08..0x78 (and that 0x07 breaks it);
  - the learner's range on zero and tiny weights, and the packer's rejection of out-of-range books;
  - `iq2r_glm53_unpack_v2` inverts `iq2r_glm53_pack_v2`, also on TP slices.
- **On the GLM-5.3 checkpoint's own bytes:** `pack(unpack(x)) == x` on all 75 MoE layers. On layers 5, 20, 40, 60
  and 75 (TP8 rank 0 and TP4 rank 1 slices, captured layer inputs with their real top-8 routing, M = 1..4096),
  the MoE output matches the dense reference built from the unpacked bytes in all 90 cases: relative RMSE at
  most 0.025, cosine at least 0.9997.

## Measurements

One MI355X (gfx950, 256 CUs), ROCm 7.2.4, PyTorch 2.10. Whole MoE (`iq2r_glm53_moe_out`), the tuner's method:
CUDA-graph replay; every call reads its own copy of the layer, at least 1 GiB of expert weights per replay, so
experts come from HBM; synthetic weights.

Against the IQ2R (v1) kernels this format replaces (257 experts with the fused shared expert, top-9, their tuned
CSV); both on the same GPU, alternating, 6 rounds each; µs, best round (median in brackets):

| M | TP4 IQ2R | TP4 IQ2R-v2 | Δ | TP8 IQ2R | TP8 IQ2R-v2 | Δ |
|---|---|---|---|---|---|---|
| 1 | 22.69 (22.70) | 17.73 (17.77) | -21.9% | 19.07 (19.10) | 16.37 (16.41) | -14.2% |
| 2 | 25.78 (25.81) | 20.00 (20.02) | -22.4% | 19.95 (20.06) | 17.84 (17.93) | -10.6% |
| 4 | 36.65 (36.71) | 26.73 (26.84) | -27.1% | 23.17 (23.51) | 20.09 (20.37) | -13.3% |
| 8 | 52.33 (52.40) | 47.73 (47.92) | -8.8% | 35.30 (35.60) | 32.63 (32.84) | -7.6% |
| 16 | 76.07 (76.17) | 71.60 (72.11) | -5.9% | 51.13 (51.27) | 45.87 (46.32) | -10.3% |
| 32 | 109.09 (109.34) | 104.07 (104.38) | -4.6% | 68.21 (68.59) | 63.07 (63.27) | -7.5% |
| 64 | 146.15 (146.70) | 139.21 (139.50) | -4.7% | 84.93 (85.43) | 79.44 (79.80) | -6.5% |
| 128 | 169.05 (170.38) | 158.97 (159.37) | -6.0% | 101.06 (101.75) | 91.60 (91.87) | -9.4% |
| 256 | 188.75 (189.99) | 174.79 (175.14) | -7.4% | 110.63 (112.11) | 102.88 (103.41) | -7.0% |
| 512 | 257.62 (259.57) | 223.26 (223.55) | -13.3% | 148.23 (149.11) | 128.32 (130.03) | -13.4% |
| 1024 | 293.53 (295.45) | 261.55 (262.06) | -10.9% | 211.57 (212.79) | 166.76 (168.62) | -21.2% |
| 2048 | 448.61 (449.08) | 419.34 (419.44) | -6.5% | 304.70 (305.14) | 277.91 (278.84) | -8.8% |
| 4096 | 747.83 (748.96) | 715.96 (716.82) | -4.3% | 509.62 (510.52) | 462.57 (464.70) | -9.2% |

IQ2R-v2 is faster at every tuned token count; the smallest margin is -4.3% (TP4 M4096). The IQ2R column
includes the shared expert (fused as route 9); IQ2R-v2 runs it as a separate FP8 MLP, not in this table.

## Quality: two recipes and what they scored

Two IQ2R-v2 recipes were evaluated on GLM-5.3. Both encode only the routed
experts (the shared expert stays FP8), use router-weighted importance, the
`importance * sqrt(sigma^2 + w^2)` encoder and a 0.98 output gain.

| | **Shipped recipe** (this code) | **v2-B-rw** (alternative recipe, not in this code) |
|---|---|---|
| Scales | E8M0 `d` per row and 256 inputs + 4-bit level per 32 (2-bit exponent, 2-bit class `2^(c/4)`) | FP16 `d` per 256 + 4-bit `ls` per 32, scale `d * (2*ls + 1) / 8` |
| Codebooks | one per expert for gate/up, one for down, learned from the IQ2_XXS grid | separate gate, up and down, refined from earlier experiments' codebooks |
| Calibration corpus | corpus v2 (general mix, above) | v1: 61% SWE-bench agent trajectories |
| Bits per weight | ~2.28 | ~2.32 |
| Runs on | the native gfx950 kernels | only by decoding the experts to FP8 at load: FP16 `d` and the odd multipliers are not MFMA scales |

Results (TP8 unless noted, FP8 KV cache, 16 concurrent sequences except two of
the MXFP4 29-task runs at 64; one run per number unless several are listed):

| Measure | Shipped recipe | v2-B-rw | MXFP4 |
|---|---|---|---|
| **SWE-bench Lite, 150 tasks** (Lite rounds 0-2, TP4, 24 agents) | **107** native (38, 38, 31); ~107 with the 15 disagreeing tasks rerun twice | **115** decode-to-FP8 (39, 41, 35) | **110** A8W4 (39, 36, 35); ~111.7 with the 15 tasks rerun; round 0 A4W4 38 |
| SWE-bench Lite, 29-task subset | native 19, 20; decode-to-FP8 19, 24 (this checkpoint), 26, 21 (an earlier build of the same recipe) | 23 | 22, 22, 22 (A4W4); 21 (newer serving stack) |
| AIME-60, sampled, 24K thinking budget | native 55, 53; decode-to-FP8 54, 53 | 55 | 55, 55, 55, 54 |
| KL vs FP8 (teacher-forced, 4 SWE agent + 1 math 8K windows) | 0.090 | not measured (v2-B, same encoder: 0.084) | 0.034 |

What the numbers say:
- **Shipped vs MXFP4: about 4-5 tasks behind on SWE-bench.** On the 150 tasks,
  6 were solved only by the shipped model and 9 only by MXFP4. Those 15 tasks
  were run twice more on both: shipped 7 and 5, MXFP4 11 and 12. Over the three
  runs MXFP4 did better on 11 of the 15, the shipped model on 1 (sign test p
  ~0.006). With the 135 tasks both agreed on held fixed, the 150-task estimate
  is about 107 for the shipped model and 111.7 for MXFP4 A8W4 (~3 points of
  resolve rate). AIME is level.
- **Shipped vs v2-B-rw: v2-B-rw is better on SWE-bench.** 9 tasks solved only
  by v2-B-rw against 1 only by the shipped model (sign test p ~0.02). The
  native kernels and packing were ruled out: the checkpoint's packed bytes
  unpack and repack exactly, the kernels match the dense reference on its own
  bytes (see Correctness), and the same weights score the same
  through the decode-to-FP8 path. The difference is in the recipe; the most
  likely cause for SWE-bench specifically is the calibration corpus (v2-B-rw's
  was agent-heavy). Not isolated yet.
- **29 tasks are too few to compare recipes:** the same checkpoint scored 19
  and 24 in two runs.
- **Behaviour:** the 2-bit models take more agent steps than MXFP4 (median 88
  against 76 per task) and about 1.4x the output tokens; no collapses; about
  0.7% of replies switch to another script (none for MXFP4).
- **The shared expert must stay FP8:** at 2 bits it cost AIME 54, 54 -> 49, 50
  and SWE-bench 29-task 3.
- **Speed:** see Measurements above.

## Not measured

- Speed on real checkpoint weights; end-to-end serving throughput.
- The FP8 shared-expert MLP that runs outside the MoE.
- Variation across GPUs and nodes; token counts between tuned rows; M > 4096; power.
