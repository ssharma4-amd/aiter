import torch
mul=[torch.tensor(2**(c/4),dtype=torch.float32) for c in range(4)]
allb=torch.arange(0,128,dtype=torch.uint8)
v=allb.view(torch.float8_e4m3fn).float()
for c in (1,2,3):
    t=[]
    for m in range(8):
        b=(7<<3)|m
        r=(v[b]*mul[c]).to(torch.float8_e4m3fn).view(torch.uint8).item()
        t.append(r-b)
    lo=t[0]|t[1]<<8|t[2]<<16|t[3]<<24; hi=t[4]|t[5]<<8|t[6]<<16|t[7]<<24
    r=(v*mul[c]).to(torch.float8_e4m3fn).view(torch.uint8).int()
    bad=[hex(b) for b in range(128) if r[b].item()!=b+t[b&7]]
    print(c,t,hex(lo),hex(hi),"bad",bad)
