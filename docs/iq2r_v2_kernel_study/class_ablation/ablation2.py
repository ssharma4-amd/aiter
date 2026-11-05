"""Staging ablation: options that shrink the codebook bytes each kernel stages per expert.

usage: ablation2.py VARIANT [LAYERS]
  b4    one codebook shared by gate and up, C4 copies          (gate kernel 16 KB)
  b2    shared gate/up codebook, C2 copies m=(1, sqrt2)        (gate kernel  8 KB)
  i4    4 independently learned books, class = book selector   (same bytes as C4)
  i2h   2 independent books, levels m=(1, sqrt2)               (half of C4)
  i2q   2 independent books, levels m=(1, 2^0.25)              (half of C4)
  c2q   C2 copies m=(1, 2^0.25) (c2 in ablation.py is m=(1, sqrt2))
  bi2h  shared gate/up + 2 independent books m=(1, sqrt2)      (gate kernel  8 KB)
  bi2q  shared gate/up + 2 independent books m=(1, 2^0.25)     (gate kernel  8 KB)
Independent books start from the learned copies (E4M3(book * m[c])) and are refined
per class with the same Lloyd step and whole-matrix acceptance as the copies, so
they can only improve on them. Metrics as ablation.py; results in res2_VARIANT.jsonl.
"""
import json
import os
import sys
import time

import torch
from safetensors import safe_open

from aiter.ops import iq2r_v2 as v2

HERE = os.path.dirname(os.path.abspath(__file__))
H = v2.IQ2RV2Levels("c2e8h", (1.0, 2.0 ** 0.5))
Q = v2.IQ2RV2Levels("c2e8q", (1.0, 2.0 ** 0.25))
VARIANTS = {  # name: (levels, shared gate/up book, independent books)
    "b4": (v2.IQ2R_V2_LEVELS_C4, True, False),
    "b2": (H, True, False),
    "i4": (v2.IQ2R_V2_LEVELS_C4, False, True),
    "i2h": (H, False, True),
    "i2q": (Q, False, True),
    "c2q": (Q, False, False),
    "bi2h": (H, True, True),
    "bi2q": (Q, True, True),
    "bi4": (v2.IQ2R_V2_LEVELS_C4, True, True),
}
name = sys.argv[1]
levels, shared, independent = VARIANTS[name]
layers = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "10,40,70").split(",")]
dev = "cuda"

# _encode_blocks takes ready-made per-class books [C, 512, 8] as well as one codebook.
_copies = v2.iq2r_v2_codebook_copies
v2.iq2r_v2_codebook_copies = lambda book, lv: book if book.dim() == 3 else _copies(book, lv)


@torch.no_grad()
def refine_books(weight, importance, books, rounds=4, inner=4, seed=0, chunk=4096):
    """Per-class Lloyd refinement of independent books [C, 512, 8] (iq2r_v2_learn_codebook's step)."""
    classes = levels.classes
    k = weight.shape[1]
    columns = k // 256
    caps = float(v2.iq2r_v2_seed_codebook(dev).max()) * torch.tensor(levels.multipliers, device=dev)
    blocks = weight.float().reshape(-1, 256)
    block_importance = importance.float().reshape(columns, 256)
    g = torch.Generator(device="cpu").manual_seed(seed)
    count = min(blocks.shape[0], max(1, 65536 * classes // 32))  # same vectors per entry as copies
    pick = torch.randperm(blocks.shape[0], generator=g)[:count].to(dev)
    sample, sample_imp = blocks[pick], block_importance[pick % columns]
    vectors = sample.abs().reshape(-1, 8)
    weights = v2._cost_weights(sample, sample_imp).reshape(-1, 8)

    def cost_of(b):
        return v2.iq2r_v2_encode(weight, importance, b, levels, chunk_blocks=chunk).cost

    best = cost_of(books)
    for _ in range(rounds):
        enc = v2._encode_blocks(sample, sample_imp, books, levels, chunk)
        level = enc.levels.long()
        cls = (level % classes).repeat_interleave(4, dim=1).reshape(-1)
        t = torch.exp2(enc.d.float()[:, None] - 127 + (level // classes).float())
        t = t.repeat_interleave(4, dim=1).reshape(-1)
        index = enc.indices.reshape(-1).clone()
        cand = books
        for step in range(inner):
            if step:
                for c in range(classes):
                    sel = cls == c
                    a, b = v2._codeword_scores(vectors[sel], weights[sel], cand[c])
                    index[sel] = torch.argmin(t[sel, None] * a - 2 * b, dim=-1)
            key = cls * 512 + index
            flat = cand.reshape(-1, 8)
            num = torch.zeros_like(flat).index_add_(0, key, weights * t[:, None] * vectors)
            den = torch.zeros_like(flat).index_add_(0, key, weights * t[:, None].square())
            occ = den > 0
            cand = torch.where(occ, num / torch.where(occ, den, 1.0), flat).reshape(books.shape)
        cand = v2._e4m3(torch.minimum(cand.clamp_min(0.0), caps[:, None, None]))
        c_cost = cost_of(cand)
        if c_cost >= best:
            break
        books, best = cand, c_cost
    return books


def learn(weight, importance):
    book = v2.iq2r_v2_learn_codebook(weight, importance, levels)
    if not independent:
        return book
    return refine_books(weight, importance, v2.iq2r_v2_codebook_copies(book, levels))


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


out = open(f"{HERE}/res2_{name}.jsonl", "a")


def report(layer, e, proj, w, im, book, t0):
    enc = v2.iq2r_v2_encode(w, im, book, levels)
    rel, relg, g = metrics(w, enc.reconstruction, im)
    hist = torch.bincount(enc.levels.reshape(-1).long(), minlength=16).tolist()
    row = dict(set=name, layer=layer, expert=e, proj=proj, rel=rel, relg=relg, gain=g,
               cost=enc.cost, hist=hist, sec=round(time.time() - t0, 1))
    print(json.dumps(row), flush=True)
    out.write(json.dumps(row) + "\n")
    out.flush()


for layer in layers:
    imp = torch.load(f"{HERE}/imp_L{layer}.pt")
    with safe_open(f"{HERE}/experts_L{layer}.safetensors", "pt") as f:
        for i, e in enumerate(imp["experts"]):
            pre = f"model.layers.{layer}.mlp.experts.{e}."
            gu_imp = imp["gate_up"][i].to(dev).float()
            gate, up = dequant(f, pre + "gate_proj"), dequant(f, pre + "up_proj")
            if shared:
                t0 = time.time()
                book = learn(torch.cat((gate, up)), gu_imp)
                report(layer, e, "gate_proj", gate, gu_imp, book, t0)
                report(layer, e, "up_proj", up, gu_imp, book, t0)
                continue
            for proj, w in (("gate_proj", gate), ("up_proj", up)):
                t0 = time.time()
                report(layer, e, proj, w, gu_imp, learn(w, gu_imp), t0)
            t0 = time.time()
            down = dequant(f, pre + "down_proj")
            im = imp["down"][i].to(dev).float()
            report(layer, e, "down_proj", down, im, learn(down, im), t0)
