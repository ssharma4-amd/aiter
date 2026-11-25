# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Time the GLM-5.3-Flash IQ2R MoE of one TP1 rank (intermediate 2048, run as
four 512-wide shards) and of one TP4 rank, per token count.

Timing follows csrc/kernels/iq2r/iq2r_glm53_tune.py: CUDA graph replays, each
call with its own routing and input, so the routed experts come from HBM as in
serving (a replay reads more expert bytes than the 256 MB MALL holds). Every
case is checked against the dense reference first.

    python3 op_tests/bench_iq2r_glm53_tp1.py [--tokens 1 2 4 8 16] [--csv out.csv]
"""

import argparse
import os
import sys

import pandas as pd
import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "csrc", "kernels", "iq2r")
)
from iq2r_glm53_tune import (  # noqa: E402
    dense_mismatch,
    dense_reference,
    routing,
    shared_reference,
    synthetic_shared,
    synthetic_weights,
    time_us,
)

from aiter.iq2r_glm53 import (  # noqa: E402
    GLM5_NEXT,
    IQ2RGlm53Workspace,
    iq2r_glm53_config,
    iq2r_glm53_moe_out,
    iq2r_glm53_slice_layer,
)
from aiter.jit.utils.chip_info import get_cu_num, get_gfx  # noqa: E402

FULL = 2048
ENTRIES = 1024
STACKS = ("gate", "gate_aux", "down", "down_aux")
READ_BYTES = 1 << 30  # expert bytes one graph replay reads at least


def _expert_bytes(stacks) -> int:
    return sum(t.nbytes for t in stacks) // GLM5_NEXT.experts


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256]
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    torch.manual_seed(0)
    packed, gate_dense, down_dense, pick = synthetic_weights(
        FULL, "cuda", ENTRIES, GLM5_NEXT
    )
    ranks = {
        width: tuple(
            iq2r_glm53_slice_layer(name, t, 0, width, FULL, ENTRIES, GLM5_NEXT)
            for name, t in zip(STACKS, packed)
        )
        for width in (FULL, 512)
    }
    del packed
    shared, exact = synthetic_shared(FULL, "cuda", GLM5_NEXT)
    # The TP4 rank 0 reference: the first 512 intermediate columns of each expert.
    tp4_dense = (
        [g[:1024] for g in gate_dense],  # gate/up rows interleave
        [d[:, :512] for d in down_dense],
    )
    limit = GLM5_NEXT.swiglu_limit
    rows = []
    for tokens in args.tokens:
        for case, width, with_shared in (
            ("tp1", FULL, False),
            ("tp1+shared", FULL, True),
            ("tp4-rank", 512, False),
        ):
            stacks = ranks[width]
            workspace = IQ2RGlm53Workspace.allocate(
                tokens, width, device="cuda", shape=GLM5_NEXT
            )
            # Calls per replay: enough routed expert bytes to flush the MALL.
            per_call = min(tokens * 8, GLM5_NEXT.experts) * _expert_bytes(stacks)
            calls = max(2, -(-READ_BYTES // per_call))
            inputs = []
            for _ in range(calls):
                weights, ids = routing(tokens, "cuda", GLM5_NEXT.experts)
                hidden = (
                    torch.randn((tokens, GLM5_NEXT.hidden), device="cuda") * 8.0
                ).to(torch.bfloat16)
                inputs.append((hidden, weights, ids, torch.empty_like(hidden)))
            part = shared if with_shared else None
            config = iq2r_glm53_config(tokens, width, ENTRIES, with_shared, GLM5_NEXT)

            def call(hidden, weights, ids, out, workspace=workspace):
                iq2r_glm53_moe_out(
                    hidden, *stacks, weights, ids, out, workspace, shared=part
                )

            hidden, weights, ids, out = inputs[0]
            call(*inputs[0])
            gate_ref, down_ref = (
                (gate_dense, down_dense) if width == FULL else tp4_dense
            )
            expected = dense_reference(
                hidden, weights, ids, gate_ref, down_ref, pick, limit
            )
            if with_shared:
                expected += shared_reference(hidden, *exact, limit)
            problem = dense_mismatch(out, expected)
            us = time_us(
                [lambda i=i: call(*i) for i in inputs], args.warmup, args.iters
            )
            unique = sum(ids.unique().numel() for _, _, ids, _ in inputs) / calls
            read = unique * _expert_bytes(stacks)
            rows.append(
                {
                    "gfx": get_gfx(),
                    "cu_num": get_cu_num(),
                    "M": tokens,
                    "case": case,
                    "inter": width,
                    "config": f"{config.gate_kernel}/{config.gate_grid} "
                    f"{config.down_kernel}/{config.down_grid}/{config.down_chunks}",
                    "us": round(us, 2),
                    "expert_GB/s": round(read / us / 1e3, 1),
                    "check": problem or "ok",
                }
            )
            print(rows[-1], flush=True)
            del workspace, inputs
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    if args.csv:
        table.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
