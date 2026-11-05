import sys, torch
a, b = sys.argv[1], sys.argv[2]
for inter in (512, 256):
    for m in (1, 2, 4):
        x = torch.load(f"{a}_{inter}_{m}.pt"); y = torch.load(f"{b}_{inter}_{m}.pt")
        print(inter, m, "bit-exact" if torch.equal(x.view(torch.int16), y.view(torch.int16)) else f"DIFF {(x != y).sum().item()} of {x.numel()}")
