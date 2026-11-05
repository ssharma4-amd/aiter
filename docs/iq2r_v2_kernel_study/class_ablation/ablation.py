"""Class ablation: encode real GLM-5.3 experts with IQ2R-v2 level sets that differ only in classes.

usage: ablation.py LEVELSET [LAYERS]   LEVELSET in c4, c2, c1, c8 ; LAYERS default 10,40,70
Per matrix (gate, up, down; own learned codebook each, as the recipe), prints
  rel   = sum imp (W - Wq)^2 / sum imp W^2            (raw reconstruction)
  relg  = same after the best least-squares output gain (removes scale bias)
  cost  = the encoder's own weighted cost
  hist  = how often each of the 16 levels is used
"""
import json
import os
import sys
import time

import torch
from safetensors import safe_open

from aiter.ops import iq2r_v2 as v2

HERE = os.path.dirname(os.path.abspath(__file__))
SETS = {
    "c4": v2.IQ2R_V2_LEVELS_C4,
    "c8": v2.IQ2R_V2_LEVELS_C8,
    "c2": v2.IQ2RV2Levels("c2e8h", (1.0, 2.0 ** 0.5)),
    "c1": v2.IQ2RV2Levels("c1e16", (1.0,)),
}
name = sys.argv[1]
levels = SETS[name]
layers = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "10,40,70").split(",")]
dev = "cuda"


def dequant(f, key):
    w = f.get_tensor(key + ".weight").to(dev).float()
    s = f.get_tensor(key + ".weight_scale_inv").to(dev).float()
    return w * s.repeat_interleave(128, 0).repeat_interleave(128, 1)[: w.shape[0], : w.shape[1]]


def metrics(w, q, imp):
    iw = imp[None, :].double()
    w, q = w.double(), q.double()
    den = (iw * w * w).sum()
    rel = float((iw * (w - q).square()).sum() / den)
    g = float((iw * w * q).sum() / (iw * q * q).sum())
    relg = float((iw * (w - g * q).square()).sum() / den)
    return rel, relg, g


out = open(f"{HERE}/res_{name}.jsonl", "a")
for layer in layers:
    imp = torch.load(f"{HERE}/imp_L{layer}.pt")
    with safe_open(f"{HERE}/experts_L{layer}.safetensors", "pt") as f:
        for i, e in enumerate(imp["experts"]):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                t0 = time.time()
                w = dequant(f, f"model.layers.{layer}.mlp.experts.{e}.{proj}")
                im = (imp["down"] if proj == "down_proj" else imp["gate_up"])[i].to(dev).float()
                book = v2.iq2r_v2_learn_codebook(w, im, levels)
                enc = v2.iq2r_v2_encode(w, im, book, levels)
                rel, relg, g = metrics(w, enc.reconstruction, im)
                hist = torch.bincount(enc.levels.reshape(-1).long(), minlength=16).tolist()
                row = dict(set=name, layer=layer, expert=e, proj=proj, rel=rel, relg=relg,
                           gain=g, cost=enc.cost, hist=hist, sec=round(time.time() - t0, 1))
                print(json.dumps(row), flush=True)
                out.write(json.dumps(row) + "\n")
                out.flush()
