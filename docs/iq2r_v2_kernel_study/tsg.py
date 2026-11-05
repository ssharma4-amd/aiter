"""Nobarrier gate per-wave phase cycles (needs GLM53_X_GTS=1). usage: tsg.py FMT INTER M[,M..]"""
import os, sys
import numpy as np
import torch
fmt, inter = sys.argv[1], int(sys.argv[2])
ms = [int(x) for x in sys.argv[3].split(",")]
sys.argv = [sys.argv[0], fmt, str(inter), "1", "x:0:route9:0:1"]
import aiter.iq2r_glm53 as g
from aiter.iq2r_glm53 import IQ2RGlm53Config
src = open("/workspace/experiments/iq2r-v2-20261103/vbench.py").read()
src = src[: src.index("fmt, inter = sys.argv[1]")]
exec(src)
_make = v2_synthetic_weights if fmt == "v2" else tune.synthetic_weights
def cached_weights(intermediate_size, device):
    return torch.load(f"/workspace/experiments/iq2r-v2-20261103/cache/{fmt}_{intermediate_size}.pt", map_location=device)
tune.synthetic_weights = cached_weights
tuner = tune.IQ2RGlm53Tuner()
for tokens in ms:
    torch.manual_seed(1000 + tokens)
    s = tuner._shape(tokens, inter)
    cfg = IQ2RGlm53Config("nobarrier", 2, "single", 8, 1)
    layers = [c[0] for c in s.calls]
    stages(s, cfg, layers[0], tokens)["quant"]()
    torch.cuda.synchronize()
    ws = s.workspace
    routes = tokens * g.TOPK
    gather = ws.gather_indices[:routes]
    quant, quant_s = ws.route_input_fp8[:tokens], ws.route_input_scales[:tokens]
    gtasks = ws.gate_tasks[: g._task_capacity(routes, 16)]
    grid = 2 * 256
    extra = (grid * 8 * 8 * 4 + inter - 1) // inter + 1
    big = torch.zeros((routes + extra, inter), dtype=torch.float8_e4m3fn, device="cuda")
    inter_t = big[:routes]
    inter_s = ws.intermediate_scales[:routes]
    recs = []
    for rep in range(30):
        for layer in layers:
            gu, gua, dn, dna = layer
            g.iq2r_glm53_gate_out(quant, quant_s, gu, gua, gtasks, ws.gate_task_count, gather, inter_t, inter_s,
                                  g.GATE_KERNELS["nobarrier"], 2)
        if rep >= 5:
            torch.cuda.synchronize()
            recs.append(big[routes:].view(torch.uint8).reshape(-1)[: grid * 8 * 8 * 4].view(torch.int32).cpu().numpy().reshape(grid, 8, 8).astype(np.int64) & 0xFFFFFFFF)
    R = np.stack(recs)  # reps, cta, wave, 8
    items = R[..., 3]
    cyc = lambda i: R[..., i].sum(axis=(1, 2)) / np.maximum(items.sum(axis=(1, 2)), 1)
    print(f"{fmt} I={inter} M={tokens}: items/CTA {items[0].mean(axis=1).mean():.2f} (max {items[0].max()}) new/items {R[0,...,4].sum()/max(items[0].sum(),1):.2f} | "
          f"cycles per item (median over reps): setup {np.median(cyc(0)):.0f} kloop {np.median(cyc(1)):.0f} epi {np.median(cyc(2)):.0f}", flush=True)
    # per-CTA total busy cycles (wave 0)
    tot = (R[..., 0] + R[..., 1] + R[..., 2])[:, :, 0]
    print(f"   CTA busy cycles wave0: med {np.median(tot):.0f} max(med over reps) {np.median(tot.max(axis=1)):.0f}")
    del s
    torch.cuda.empty_cache()
