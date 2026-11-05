"""Byte range of learned c4 codebooks on the 36 ablation matrices: entries below 2^-6 (bytes < 0x08) and the max."""
import json, os, sys, torch
from safetensors import safe_open
from aiter.ops import iq2r_v2 as v2
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(f"{HERE}/ablation.py").read().split("def metrics")[0].split("name = sys.argv[1]")[0])
dev = "cuda"
def dequant(f, key):
    w = f.get_tensor(key + ".weight").to(dev).float()
    s = f.get_tensor(key + ".weight_scale_inv").to(dev).float()
    return w * s.repeat_interleave(128, 0).repeat_interleave(128, 1)[: w.shape[0], : w.shape[1]]
levels = v2.IQ2R_V2_LEVELS_C4
for layer in (10, 40, 70):
    imp = torch.load(f"{HERE}/imp_L{layer}.pt")
    with safe_open(f"{HERE}/experts_L{layer}.safetensors", "pt") as f:
        for i, e in enumerate(imp["experts"]):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                w = dequant(f, f"model.layers.{layer}.mlp.experts.{e}.{proj}")
                im = (imp["down"] if proj == "down_proj" else imp["gate_up"])[i].to(dev).float()
                book = v2.iq2r_v2_learn_codebook(w, im, levels)
                b = book.to(torch.float8_e4m3fn).view(torch.uint8)
                small = int((b < 8).sum()); zero = int((b == 0).sum())
                print(json.dumps(dict(layer=layer, expert=e, proj=proj, below_min_normal=small, zeros=zero,
                                      min_byte=int(b.min()), max_byte=int(b.max()), min_value=float(book.min()))), flush=True)
