# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""GLM-5.3-Flash routed MoE with MXFP4 experts, timed as ATOM runs it at TP1.

One layer, full width: hidden 4096, 288 routed experts, top-8, moe_intermediate
2048, SiLU(gate) * up with GLM's clamp (swiglu_limit 10), BF16 in/out, MXFP4
weights (fp4x2 + E8M0 per 32, QuantType.per_1x32). Synthetic random weights.

Mirrors ATOM's ``Mxfp4MoEMethod`` (atom/model_ops/moe.py) on gfx950 with
``ATOM_USE_TRITON_MOE`` unset (off on gfx95x unless ATOM_USE_TRITON_GEMM=1), no
EP, so weight prep is branch C (``moe_shuffle_weight`` / ``moe_shuffle_scale``)
and ``apply`` calls ``aiter.fused_moe`` directly:

* ``a4w4``: default. GGUU (``ATOM_MOE_GU_ITLV=0``), gate_mode separated; aiter
  quantizes the activations to MXFP4.
* ``a8w4``: ``ATOM_GLM5_MXFP4_FP8_ACT=1``. ``mxfp4_fp8_activations`` forces
  the GUGU interleave and passes ``quant_dtype_a=dtypes.fp8`` (MXFP8 rows).

Two timings per M: ``us`` from ``run_perftest`` (rotating copies of every
tensor argument, weights included; the summed GPU kernel time of one fused_moe
call: sort, quant, both stages), and ``graph_us``, a CUDA graph of calls with
their own routing as op_tests/bench_iq2r_glm53_tp1.py times the IQ2R MoE.

    python3 op_tests/bench_glm5next_mxfp4_moe.py [--tokens 1 8 64] \
        [--variant a4w4 a8w4] [--csv mxfp4_moe_bench.csv]
"""

import os

os.environ.setdefault("AITER_LOG_TUNED_CONFIG", "1")

import argparse  # noqa: E402
import functools  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402

import pandas as pd  # noqa: E402
import torch  # noqa: E402

import aiter  # noqa: E402
import aiter.fused_moe as aiter_fused_moe  # noqa: E402
from aiter import ActivationType, QuantType, dtypes  # noqa: E402
from aiter.fused_moe import fused_moe  # noqa: E402
from aiter.jit.utils.chip_info import get_cu_num, get_gfx  # noqa: E402
from aiter.ops.flydsl.moe_common import GateMode  # noqa: E402
from aiter.ops.quant import per_1x32_f4_quant  # noqa: E402
from aiter.ops.shuffle import moe_shuffle_scale, moe_shuffle_weight  # noqa: E402
from aiter.test_common import run_perftest  # noqa: E402

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "csrc", "kernels", "iq2r")
)
from iq2r_glm53_tune import time_us  # noqa: E402

READ_BYTES = 1 << 30  # expert bytes one graph replay reads at least
from aiter.utility.fp4_utils import e8m0_to_f32, mxfp4_to_f32  # noqa: E402

HIDDEN = 4096
EXPERTS = 288
TOPK = 8
INTER = 2048
SWIGLU_LIMIT = 10.0
# As in op_tests/test_iq2r_glm53.py: hidden std 8, weight std 0.02, so the
# gate/up outputs (std ~10) often pass the clamp at 10.
HIDDEN_SCALE = 8.0
WEIGHT_STD = 0.02

VARIANTS = ("a4w4", "a8w4")
DEFAULT_TOKENS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


# ----------------------------------------------------------------- weights ---
def _quantize_experts(rows: int, cols: int, seed: int, chunk: int = 16):
    """Random BF16 (E, rows, cols) weights -> MXFP4 (E, rows, cols//2) fp4x2 and
    (E, rows, cols//32) uint8 E8M0 scales, the checkpoint layout ATOM loads."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.empty((EXPERTS, rows, cols // 2), dtype=torch.uint8, device="cuda")
    s = torch.empty((EXPERTS, rows, cols // 32), dtype=torch.uint8, device="cuda")
    for e0 in range(0, EXPERTS, chunk):
        e1 = min(e0 + chunk, EXPERTS)
        w = (
            torch.randn((e1 - e0, rows, cols), generator=gen, device="cuda")
            * WEIGHT_STD
        ).to(torch.bfloat16)
        wq, ws = per_1x32_f4_quant(w, quant_dtype=dtypes.fp4x2)
        q[e0:e1] = wq.view(torch.uint8).view(e1 - e0, rows, cols // 2)
        s[e0:e1] = ws.view(torch.uint8).view(e1 - e0, rows, cols // 32)
        del w, wq, ws
    return q.view(dtypes.fp4x2), s


def _dequant(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """One expert's MXFP4 (rows, cols//2) + E8M0 (rows, cols//32) -> fp32."""
    vals = mxfp4_to_f32(q.view(torch.uint8).clone())
    return vals * e8m0_to_f32(s.clone()).repeat_interleave(32, dim=-1)


def _atom_prep(w13, w13_s, w2, w2_s, guinterleave: bool):
    """Mxfp4MoEMethod._process_weight_layout_after_loading, branch C (FlyDSL
    fused_moe), as on gfx950 with ATOM_USE_TRITON_MOE off.

    ATOM assigns the shuffled tensors through ``Parameter.data`` and sets only
    ``is_shuffled`` on the parameter: the ``gate_up_interleave`` attribute that
    shuffle_weight puts on its result does not survive, so it is left off here
    too (``.view`` gives a fresh tensor object without it)."""
    w1 = moe_shuffle_weight(
        w13, experts_cnt=EXPERTS, is_guinterleave=guinterleave, gate_up=True
    )
    w2s = moe_shuffle_weight(
        w2, experts_cnt=EXPERTS, is_guinterleave=guinterleave, gate_up=False
    )
    w1 = w1.view(w1.dtype)
    w2s = w2s.view(w2s.dtype)
    w1.is_shuffled = True
    w2s.is_shuffled = True
    s1 = moe_shuffle_scale(
        w13_s.reshape(-1, w13_s.shape[-1]),
        EXPERTS,
        is_guinterleave=guinterleave,
        gate_up=True,
    )
    s2 = moe_shuffle_scale(
        w2_s.reshape(-1, w2_s.shape[-1]),
        EXPERTS,
        is_guinterleave=guinterleave,
        gate_up=False,
    )
    return w1, w2s, s1, s2


# ----------------------------------------------------------------- routing ---
def _routing(tokens: int, seed: int):
    """Like op_tests/test_iq2r_glm53.py::_routing: random top-8 of 288 experts,
    softmax weights (fp32), BF16 hidden."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    scores = torch.rand((tokens, EXPERTS), generator=gen, device="cuda")
    topk_ids = scores.topk(TOPK, dim=-1).indices.to(torch.int32).contiguous()
    topk_weights = torch.softmax(
        torch.randn((tokens, TOPK), generator=gen, device="cuda"), dim=-1
    ).contiguous()
    hidden = (
        torch.randn((tokens, HIDDEN), generator=gen, device="cuda") * HIDDEN_SCALE
    ).to(torch.bfloat16)
    return hidden, topk_weights, topk_ids


# --------------------------------------------------------------- reference ---
def _fq_mxfp4(x: torch.Tensor) -> torch.Tensor:
    q, s = per_1x32_f4_quant(
        x.to(torch.bfloat16).contiguous(), quant_dtype=dtypes.fp4x2
    )
    q = q.view(torch.uint8).view(x.shape[0], -1)
    s = s.view(torch.uint8).view(x.shape[0], -1)[:, : x.shape[1] // 32]
    return _dequant(q, s)


def _fq_mxfp8(x: torch.Tensor) -> torch.Tensor:
    blocks = x.float().unflatten(-1, (-1, 32))
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    exponent = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
    exponent = torch.where(amax > 0, exponent, torch.full_like(exponent, -127))
    scale = torch.exp2(exponent)
    return ((blocks / scale).to(torch.float8_e4m3fn).float() * scale).flatten(-2)


def _reference(raw, hidden, topk_weights, topk_ids, act_fq=None):
    """Dense fp32 MoE over the dequantized MXFP4 weights (GGUU rows), GLM's
    clamped SwiGLU. ``act_fq`` fake-quantizes the stage-1 input and the stage-2
    input (the variant's activation format); None keeps them fp32."""
    w13, w13_s, w2, w2_s = raw
    x = hidden.float()
    if act_fq is not None:
        x = act_fq(x)
    out = torch.zeros_like(x)
    for e in topk_ids.unique().tolist():
        rows, slot = (topk_ids == e).nonzero(as_tuple=True)
        h = x[rows] @ _dequant(w13[e], w13_s[e]).T
        gate, up = h[:, :INTER], h[:, INTER:]
        a = torch.nn.functional.silu(gate.clamp(max=SWIGLU_LIMIT)) * up.clamp(
            -SWIGLU_LIMIT, SWIGLU_LIMIT
        )
        if act_fq is not None:
            a = act_fq(a.to(torch.bfloat16).float())
        y = a @ _dequant(w2[e], w2_s[e]).T
        out.index_add_(0, rows, y * topk_weights[rows, slot, None])
    return out


def _metrics(actual: torch.Tensor, expected: torch.Tensor):
    a, b = actual.float().flatten(), expected.float().flatten()
    rrmse = ((a - b).square().mean().sqrt() / b.square().mean().sqrt()).item()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    return rrmse, cos


# ------------------------------------------------------- dispatch capture ---
class _LogCapture(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records = []

    def emit(self, record):
        msg = record.getMessage()
        if "fused_moe" in msg or "run_1stage" in msg or "tuned" in msg.lower():
            self.records.append(msg)


_captured_meta = []
_orig_get_2stage_cfgs = aiter_fused_moe.get_2stage_cfgs


@functools.wraps(_orig_get_2stage_cfgs)
def _recording_get_2stage_cfgs(*args, **kwargs):
    meta = _orig_get_2stage_cfgs(*args, **kwargs)
    _captured_meta.append(meta)
    return meta


aiter_fused_moe.get_2stage_cfgs = _recording_get_2stage_cfgs


def _describe_stage(fn) -> str:
    if fn is None:
        return "-"
    if isinstance(fn, functools.partial):
        name = getattr(fn.func, "__name__", repr(fn.func))
        kws = [
            f"{k}={v}"
            for k, v in (fn.keywords or {}).items()
            if isinstance(v, str) and v
        ]
        return f"{name}({', '.join(kws)})" if kws else name
    return getattr(fn, "__name__", repr(fn))


def _describe_meta(meta) -> str:
    return (
        f"stage1={_describe_stage(meta.stage1)} | "
        f"stage2={_describe_stage(meta.stage2)} | block_m={meta.block_m} "
        f"ksplit={meta.ksplit} 1stage={meta.run_1stage} "
        f"fuse_quant={meta.fuse_quant or '-'} flat={int(meta.flat)}"
    )


def _gpu_kernels(fn, *args) -> str:
    """GPU kernels one call launches, in order, with their times (us)."""
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        fn(*args)
        torch.cuda.synchronize()
    names = []
    for evt in prof.events():
        if evt.device_type == torch.autograd.DeviceType.CUDA:
            names.append(f"{evt.name[:90]}:{evt.device_time:.1f}")
    return " ; ".join(names)


# -------------------------------------------------------------------- main ---
def _make_call(variant: str):
    gate_mode = (
        GateMode.INTERLEAVE.value if variant == "a8w4" else GateMode.SEPARATED.value
    )
    extra = {"gate_mode": gate_mode, "swiglu_limit": SWIGLU_LIMIT}
    if variant == "a8w4":
        extra["quant_dtype_a"] = dtypes.fp8

    def call(x, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale, ntok):
        # run_perftest's deep copies may drop python attributes; ATOM's
        # parameters carry is_shuffled=True.
        w1.is_shuffled = True
        w2.is_shuffled = True
        # Mxfp4MoEMethod.apply, TP path (fused_experts is None, no Triton).
        return fused_moe(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            expert_mask=None,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=None,
            a2_scale=None,
            doweight_stage1=False,
            bias1=None,
            bias2=None,
            num_local_tokens=ntok,
            **extra,
        )

    return call


def _graph_us(call, m, w1, w2s, s1, s2, ntok, iters):
    """Per-call time of a CUDA graph of calls with their own routing and input,
    reading at least READ_BYTES of expert weights per replay (as the IQ2R
    tuner and op_tests/bench_iq2r_glm53_tp1.py time)."""
    expert_bytes = sum(t.nbytes for t in (w1, w2s, s1, s2)) // EXPERTS
    calls, read = [], 0
    while read < READ_BYTES or len(calls) < 2:
        hidden, topk_weights, topk_ids = _routing(m, 0x9000 + len(calls))
        calls.append(
            lambda h=hidden, w=topk_weights, i=topk_ids: call(
                h, w1, w2s, w, i, s1, s2, ntok
            )
        )
        read += topk_ids.unique().numel() * expert_bytes
    return time_us(calls, 20, max(iters, 200))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tokens", type=int, nargs="+", default=list(DEFAULT_TOKENS))
    parser.add_argument(
        "--variant", nargs="+", choices=VARIANTS, default=list(VARIANTS)
    )
    parser.add_argument("--csv", default=None)
    parser.add_argument(
        "--check-tokens",
        type=int,
        nargs="*",
        default=None,
        help="M values to check against the dense reference (default: all)",
    )
    parser.add_argument(
        "--no-num-local-tokens",
        action="store_true",
        help="pass num_local_tokens=None (ATOM passes the scheduled-token "
        "count, an int32 [1] GPU tensor, in a normal forward)",
    )
    parser.add_argument("--iters", type=int, default=101)
    args = parser.parse_args()
    check_tokens = set(args.tokens if args.check_tokens is None else args.check_tokens)

    capture = _LogCapture()
    aiter.logger.addHandler(capture)
    aiter.logger.setLevel(logging.INFO)

    torch.manual_seed(0)
    print(
        f"gfx={get_gfx()} cu_num={get_cu_num()} torch={torch.__version__} "
        f"AITER_LOG_TUNED_CONFIG={os.environ.get('AITER_LOG_TUNED_CONFIG')}"
    )
    print("quantizing synthetic weights ...", flush=True)
    w13, w13_s = _quantize_experts(2 * INTER, HIDDEN, seed=0x6151)
    w2, w2_s = _quantize_experts(HIDDEN, INTER, seed=0x6152)
    raw = (w13, w13_s, w2, w2_s)

    rows = []
    for variant in args.variant:
        guinterleave = variant == "a8w4"  # ATOM: ATOM_MOE_GU_ITLV or fp8_activations
        w1, w2s, s1, s2 = _atom_prep(w13, w13_s, w2, w2_s, guinterleave)
        call = _make_call(variant)
        act_fq = _fq_mxfp8 if variant == "a8w4" else _fq_mxfp4
        for m in args.tokens:
            hidden, topk_weights, topk_ids = _routing(m, 0x5300 + m)
            ntok = (
                None
                if args.no_num_local_tokens
                else torch.tensor([m], dtype=torch.int32, device="cuda")
            )
            call_args = (hidden, w1, w2s, topk_weights, topk_ids, s1, s2, ntok)
            row = {"M": m, "variant": variant}
            try:
                _captured_meta.clear()
                capture.records.clear()
                out = call(*call_args)
                torch.cuda.synchronize()
                meta = _captured_meta[-1] if _captured_meta else None
                row["kernels"] = _describe_meta(meta) if meta else "?"
                logs = [r for r in capture.records if "[fused_moe] using" in r]
                row["tuned"] = (
                    "default"
                    if logs and all(" default " in r for r in logs)
                    else ("tuned" if logs else "?")
                )
                row["dispatch_log"] = logs[-1] if logs else ""
                row["finite"] = bool(torch.isfinite(out).all().item())
                if m in check_tokens:
                    dense = _reference(raw, hidden, topk_weights, topk_ids)
                    aq = _reference(raw, hidden, topk_weights, topk_ids, act_fq)
                    row["rrmse_dense"], row["cos_dense"] = _metrics(out, dense)
                    row["rrmse_actq"], row["cos_actq"] = _metrics(out, aq)
                row["gpu_kernels"] = _gpu_kernels(call, *call_args)
                _, us = run_perftest(call, *call_args, num_iters=args.iters)
                row["us"] = us
                row["graph_us"] = _graph_us(call, m, w1, w2s, s1, s2, ntok, args.iters)
            except Exception as exc:  # report and keep going
                row["error"] = f"{type(exc).__name__}: {exc}"[:500]
                print(f"[{variant} M={m}] FAILED: {row['error']}", flush=True)
            rows.append(row)
            print(
                f"{variant} M={m:4d} us={row.get('us', float('nan')):9.2f} "
                f"graph_us={row.get('graph_us', float('nan')):9.2f} "
                f"rrmse_dense={row.get('rrmse_dense', float('nan')):.4f} "
                f"cos_dense={row.get('cos_dense', float('nan')):.5f} "
                f"rrmse_actq={row.get('rrmse_actq', float('nan')):.4f} "
                f"cos_actq={row.get('cos_actq', float('nan')):.5f} "
                f"{row.get('tuned', '')}",
                flush=True,
            )
        del w1, w2s, s1, s2
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 200)
    cols = [
        c
        for c in (
            "M",
            "variant",
            "us",
            "graph_us",
            "rrmse_dense",
            "cos_dense",
            "rrmse_actq",
            "cos_actq",
            "tuned",
            "error",
        )
        if c in df.columns
    ]
    print("\nGLM-5.3-Flash routed MoE, MXFP4 weights (synthetic), one layer, TP1")
    print(df[cols].to_string(index=False))
    print("\nkernels (get_2stage_cfgs metadata):")
    for _, r in df.drop_duplicates(["variant", "kernels"]).iterrows():
        if isinstance(r.get("kernels"), str):
            print(f"  {r['variant']} from M={r['M']}: {r['kernels']}")
    print("\nGPU kernels per call (name:us), first M of each kernel set:")
    if "gpu_kernels" in df.columns:
        for _, r in df.drop_duplicates(["variant", "kernels"]).iterrows():
            print(f"  {r['variant']} M={r['M']}: {r.get('gpu_kernels')}")
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        df.to_csv(args.csv, index=False)
        print(f"\nsaved {args.csv}")


if __name__ == "__main__":
    main()
