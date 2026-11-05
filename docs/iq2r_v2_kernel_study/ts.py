"""Route9 per-wave timestamps (needs GLM53_X_TS=1 build). usage: ts.py INTER M [copies]"""
import os, sys
import numpy as np
import torch
sys.argv = [sys.argv[0], "v2", sys.argv[1], sys.argv[2], "x:0:route9:0:1"] + sys.argv[3:]
import aiter.iq2r_glm53 as g
from aiter.iq2r_glm53 import IQ2RGlm53Config
src = open("/workspace/experiments/iq2r-v2-20261103/vbench.py").read()
src = src[: src.index("fmt, inter = sys.argv[1]")]
exec(src)
fmt, inter, tokens = "v2", int(sys.argv[2]), int(sys.argv[3])
_make = v2_synthetic_weights
def cached_weights(intermediate_size, device):
    path = f"/workspace/experiments/iq2r-v2-20261103/cache/{fmt}_{intermediate_size}.pt"
    return torch.load(path, map_location=device)
tune.synthetic_weights = cached_weights
tuner = tune.IQ2RGlm53Tuner()
torch.manual_seed(1000 + tokens)
s = tuner._shape(tokens, inter)
cfg = IQ2RGlm53Config("decode", 2, "route9", 4, 1) if tokens == 1 else IQ2RGlm53Config("nobarrier", 4, "route9", 4, 1)
layers = [c[0] for c in s.calls]
tuner._run(s, cfg, layers[0], s.weights, s.ids)
torch.cuda.synchronize()
ws = s.workspace
routes = tokens * g.TOPK
ids = s.ids.reshape(-1)
scatter = ws.scatter_indices[:routes]
inter_t, inter_s = ws.intermediate_fp8[:routes], ws.intermediate_scales[:routes]
words = tokens * 128 * 9 * 16
big = torch.zeros(tokens * 6144 + 2 * words, dtype=torch.bfloat16, device="cuda")
out = big[: tokens * 6144].view(tokens, 6144)
recs = []
for rep in range(40):
    for li, layer in enumerate(layers):
        gu, gua, dn, dna = layer
        g.iq2r_glm53_down_route9_out(inter_t, inter_s, dn, dna, ids, scatter, s.weights, out)
        if rep >= 5 and li == len(layers) - 1:
            torch.cuda.synchronize()
            recs.append(big[tokens * 6144:].view(torch.int32).cpu().numpy()[:words].reshape(tokens, 128, 9, 16).astype(np.int64) & 0xFFFFFFFF)
torch.cuda.synchronize()
R = np.stack(recs)  # reps, tokens, cta, route, 16
mt = lambda a, b: ((R[..., b] - R[..., a]) & 0xFFFFFFFF)
rt0, rt1 = R[..., 8], R[..., 9]
span = (rt1.max(axis=(1, 2, 3)) - rt0.min(axis=(1, 2, 3))) * 10  # ns
wave = mt(0, 6)
freq = (wave.sum() / (((rt1 - rt0) & 0xFFFFFFFF).sum() * 10))  # cycles per ns
print(f"I={inter} M={tokens} reps={len(recs)} memtime GHz~{freq:.3f} kernel span(rt) med {np.median(span):.0f} ns")
c2ns = lambda x: x / freq
start = (rt0 - rt0.min(axis=(1, 2, 3), keepdims=True)) * 10
cta_start = start.min(axis=3)
print(f"  CTA start skew ns: med {np.median(cta_start):.0f} p90 {np.percentile(cta_start,90):.0f} max {np.median(cta_start.max(axis=(1,2))):.0f}")
wave_in_cta = start - start.min(axis=3, keepdims=True)
print(f"  wave start within CTA ns: med {np.median(wave_in_cta):.0f} max(med over reps) {np.median(wave_in_cta.max(axis=(1,2,3))):.0f}")
names = ["scalar(0-1)", "issue(1-2)", "wait(2-3)", "decode(3-4)", "sync(4-5)", "combine(5-6)", "total(0-6)"]
pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (0, 6)]
print("  phase ns (median over waves) by route:" )
print("   route " + " ".join(f"{n:>13s}" for n in names))
for r in range(9):
    vals = [np.median(c2ns(mt(a, b)[..., r, ])) for a, b in pairs]
    print(f"   {r:5d} " + " ".join(f"{v:13.0f}" for v in vals))
print("   base arrival / formation end (ts12, ts13 - ts0) ns by route:", [(int(np.median(c2ns(mt(0, 12)[..., r]))), int(np.median(c2ns(mt(0, 13)[..., r])))) for r in range(9)])
print("   helper flag-wait end (ts7 - ts0) ns by route 0-3:", [int(np.median(c2ns(mt(0, 7)[..., r]))) for r in range(4)])
vals = [np.median(c2ns(mt(a, b))) for a, b in pairs]
print("     all " + " ".join(f"{v:13.0f}" for v in vals))
end = (rt1 - rt0.min(axis=(1, 2, 3), keepdims=True)) * 10
print(f"  CTA end ns: med {np.median(end.max(axis=3)):.0f} p90 {np.percentile(end.max(axis=3),90):.0f}")
xcc = R[..., 11] & 0xF
cu = (R[..., 10] >> 8) & 0xF
se = (R[..., 10] >> 13) & 0x7
print("  CTAs per (xcc,se,cu) max:", np.max(np.unique(np.stack([xcc[-1,:,:,0], se[-1,:,:,0], cu[-1,:,:,0]], -1).reshape(-1, 3), axis=0, return_counts=True)[1]))
