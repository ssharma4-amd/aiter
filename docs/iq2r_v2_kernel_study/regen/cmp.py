import torch, math
h = torch.frombuffer(bytearray(open("cvt.bin","rb").read()), dtype=torch.uint8).reshape(4,256)
base = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
for c in range(4):
    m = torch.tensor(2.0 ** (c / 4.0), dtype=torch.float32)
    ref = (base * m).to(torch.float8_e4m3fn).view(torch.uint8)
    bad = (ref != h[c]).nonzero().flatten().tolist()
    nanref = torch.isnan((base*m).to(torch.float8_e4m3fn).float())
    print(f"c={c} m={m.item():.6f} mismatches={len(bad)}", [(b, hex(ref[b].item()), hex(h[c][b].item()), base[b].item()) for b in bad[:12]])
