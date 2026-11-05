"""Per-stage graph timing (cold, tuner layer rotation) of the GLM-5.3 IQ2R MoE, v1 or v2.

usage: vbench.py FORMAT INTER M,M,... [CSV | gate:grid:down:grid:chunks]
  FORMAT v1 | v2 ; CSV default: the tree's aiter/configs/iq2r_glm53_tuned.csv

Routes are seeded per M. REPS (3): min over repeated timings. STAGES=gate,down
times only those stages (no full). Every stage is timed alone by CUDA graph replay over the rotated layer copies
(>= 1 GiB of expert weights per replay, the tuner's method). Prints quant / gate
/ down / reduce / sum / full in us.
"""
import importlib.util
import os
import sys

import pandas as pd
import torch

import aiter.iq2r_glm53 as g
from aiter.iq2r_glm53 import IQ2RGlm53Config

TREE = os.environ.get("TREE", "/workspace/splits/aiter-iq2r-v2")
spec = importlib.util.spec_from_file_location(
    "tune", f"{TREE}/csrc/kernels/iq2r/iq2r_glm53_tune.py")
tune = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tune)


def v2_synthetic_weights(intermediate_size: int, device: str):
    from aiter.ops import iq2r_v2 as v2
    levels = v2.IQ2R_V2_LEVELS_C4
    seed = v2.iq2r_v2_seed_codebook(device)
    books = (seed, (seed * 1.0625).to(torch.float8_e4m3fn).float())
    H = g.HIDDEN

    def il(a, b):
        return torch.stack((a, b), dim=1).reshape(-1, *a.shape[1:])

    def gate_up():
        gate = torch.randn((intermediate_size, H), device=device) * 0.02
        up = torch.randn((intermediate_size, H), device=device) * 0.02
        imp = torch.ones(H, device=device)
        eg = v2.iq2r_v2_encode(gate, imp, books[0], levels)
        eu = v2.iq2r_v2_encode(up, imp, books[1], levels)
        gains = v2.iq2r_v2_gains(il(gate, up), il(eg.reconstruction, eu.reconstruction),
                                 0.98, interleaved=True)
        data = v2.iq2r_v2_write_data(il(eg.indices, eu.indices), il(eg.negative, eu.negative),
                                     il(eg.levels, eu.levels), levels)
        aux = v2.iq2r_v2_write_auxiliary(torch.stack(books), il(eg.d, eu.d), gains)
        dense = v2.iq2r_v2_materialize(data, aux, 2 * intermediate_size, H, levels, books=2)
        return data.reshape(-1), aux.reshape(-1), dense

    def down():
        w = torch.randn((H, intermediate_size), device=device) * 0.02
        enc = v2.iq2r_v2_encode(w, torch.ones(intermediate_size, device=device), books[0], levels)
        gains = v2.iq2r_v2_gains(w, enc.reconstruction, 0.98)
        data = v2.iq2r_v2_write_data(enc.indices, enc.negative, enc.levels, levels)
        aux = v2.iq2r_v2_write_auxiliary(books[0][None], enc.d, gains)
        dense = v2.iq2r_v2_materialize(data, aux, H, intermediate_size, levels)
        return data.reshape(-1), aux.reshape(-1), dense

    pick = torch.arange(g.EXPERTS, device=device) % 3
    gs = [gate_up() for _ in range(3)]
    ds = [down() for _ in range(3)]
    st = lambda parts, i: torch.stack([p[i] for p in parts])[pick].contiguous()
    packed = g.iq2r_glm53_pack_v2(st(gs, 0), st(gs, 1), st(ds, 0), st(ds, 1),
                                  intermediate_size=intermediate_size, levels=levels)
    return packed, [p[2] for p in gs], [p[2] for p in ds], pick


def stages(s, config, layer, tokens):
    gu, gua, dn, dna = layer
    ws = s.workspace
    weights, idsk = s.weights, s.ids
    routes = tokens * g.TOPK
    ids = idsk.reshape(-1)
    gather, scatter = ws.gather_indices[:routes], ws.scatter_indices[:routes]
    count = ws.task_count
    inter, inter_s = ws.intermediate_fp8[:routes], ws.intermediate_scales[:routes]
    qrows = g.TOPK if tokens == 1 else tokens
    quant, quant_s = ws.route_input_fp8[:qrows], ws.route_input_scales[:qrows]
    out = s.output
    st = {}
    if tokens == 1:
        tasks = ws.tasks[:routes]
        st["quant"] = lambda: g.iq2r_glm53_m1_route_quant_out(
            s.hidden, ids, scatter, tasks, count, quant, quant_s)
        st["gate"] = lambda: g.iq2r_glm53_gate_m1_out(
            quant, quant_s, gu, gua, tasks, count, inter, inter_s)
        tasks = ws.tasks[: g._task_capacity(routes, 32)]
    elif config.gate_kernel != "prefill" and tokens <= g.MAX_DECODE_TOKENS:
        tasks = ws.tasks[: g._task_capacity(routes, 32)]
        gtasks = ws.gate_tasks[: g._task_capacity(routes, 16)]
        st["quant"] = lambda: g.iq2r_glm53_sort_quant_out(
            s.hidden, ids, gather, scatter, tasks, count, quant, quant_s, gtasks,
            ws.gate_task_count)
        st["gate"] = lambda: g.iq2r_glm53_gate_out(
            quant, quant_s, gu, gua, gtasks, ws.gate_task_count, gather, inter, inter_s,
            g.GATE_KERNELS[config.gate_kernel], config.gate_grid)
    else:
        tasks = ws.tasks[: g._task_capacity(routes, 64)]
        st["quant"] = lambda: g.iq2r_glm53_prefill_sort_quant_out(
            s.hidden, ids, gather, scatter, tasks, count, quant, quant_s, ws.sort_scratch)
        st["gate"] = lambda: g.iq2r_glm53_gate_out(
            quant, quant_s, gu, gua, tasks, count, gather, inter, inter_s,
            g.GATE_KERNELS["prefill"], config.gate_grid)
        st["down"] = lambda: g.iq2r_glm53_down_reduce_out(
            inter, inter_s, dn, dna, tasks, count, ws.route_output[:routes], scatter,
            weights, out, config.down_chunks)
        return st
    if config.down_kernel == "route9":
        st["down"] = lambda: g.iq2r_glm53_down_route9_out(
            inter, inter_s, dn, dna, ids, scatter, weights, out)
        return st
    ro = ws.route_output[:routes]
    st["down"] = lambda: g.iq2r_glm53_down_out(
        inter, inter_s, dn, dna, tasks, count, ro, g.DOWN_KERNELS[config.down_kernel],
        config.down_grid)
    st["reduce"] = lambda: g.iq2r_glm53_route_reduce_out(ro, weights, scatter, out)
    return st


fmt, inter = sys.argv[1], int(sys.argv[2])
ms = [int(x) for x in sys.argv[3].split(",")]
fixed = sys.argv[4].split(":") if len(sys.argv) > 4 and ":" in sys.argv[4] else None
csv = sys.argv[4] if len(sys.argv) > 4 and not fixed else f"{TREE}/aiter/configs/iq2r_glm53_tuned.csv"
table = None if fixed else pd.read_csv(csv)
if table is not None:
    table = table[(table.inter_dim == inter) & (table.cu_num == torch.cuda.get_device_properties(0).multi_processor_count)]
_make = v2_synthetic_weights if fmt == "v2" else tune.synthetic_weights


def cached_weights(intermediate_size, device):
    path = f"/workspace/experiments/iq2r-v2-20261103/cache/{fmt}_{intermediate_size}.pt"
    if os.path.exists(path):
        return torch.load(path, map_location=device)
    out = _make(intermediate_size, device)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(out, path)
    return out


tune.synthetic_weights = cached_weights
tuner = tune.IQ2RGlm53Tuner()
check = os.environ.get("CHECK", "1") == "1"
reps = int(os.environ.get("REPS", "3"))  # min over repeated graph timings
only = [x for x in os.environ.get("STAGES", "").split(",") if x]
for tokens in ms:
    if fixed:
        config = IQ2RGlm53Config(fixed[0], int(fixed[1]), fixed[2], int(fixed[3]), int(fixed[4]))
    else:
        row = table[table.token == tokens].iloc[0]
        config = IQ2RGlm53Config(row.gate_kernel, int(row.gate_grid), row.down_kernel,
                                 int(row.down_grid), int(row.down_chunks))
    torch.manual_seed(1000 + tokens)  # same routes in every run
    s = tuner._shape(tokens, inter)
    layers = [c[0] for c in s.calls]
    per_layer = [stages(s, config, layer, tokens) for layer in layers]
    tuner._run(s, config, layers[0], s.weights, s.ids)
    torch.cuda.synchronize()
    bad = tune.dense_mismatch(s.output, s.expected) if check else None
    if os.environ.get("DUMP"):
        torch.save(s.output.cpu(), f"{os.environ['DUMP']}_{inter}_{tokens}.pt")
    if check:
        o, e = s.output.float(), s.expected
        bad = (bad or "") + f" rmse={((o - e).square().mean() / e.square().mean()).sqrt().item():.4f}"
    res = {}
    for name in per_layer[0]:
        if only and name not in only:
            continue
        res[name] = min(tune.time_us([st[name] for st in per_layer], 20, 400) for _ in range(reps))
    full = min(tune.time_us(
        [lambda c=c: tuner._run(s, config, c[0], c[1], c[2]) for c in s.calls], 50, 500)
        for _ in range(reps)) if not only else 0.0
    msg = " ".join(f"{k} {v:7.2f}" for k, v in res.items())
    print(f"{fmt} I={inter} M={tokens:5d} {config.gate_kernel}{config.gate_grid}/"
          f"{config.down_kernel}{config.down_grid}c{config.down_chunks} | {msg} | sum "
          f"{sum(res.values()):7.2f} full {full:7.2f} | copies {len(layers)}"
          f"{' MISMATCH ' + bad if bad and 'relative' in bad else (bad or '')}", flush=True)
    del s, per_layer
    torch.cuda.empty_cache()
