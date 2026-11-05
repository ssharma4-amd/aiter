"""Range-download selected GLM-5.3 expert tensors (FP8 + block scales) from HF.

usage: fetch.py LAYER E,E,...  -> experts_L{LAYER}.safetensors
"""
import json
import struct
import sys
import urllib.request

REPO = "https://huggingface.co/zai-org/GLM-5.3/resolve/main/"
TOKEN = open("/home/ssharma4/.cache/huggingface/token").read().strip()


def get(name, start, stop):
    req = urllib.request.Request(REPO + name, headers={
        "Authorization": f"Bearer {TOKEN}", "Range": f"bytes={start}-{stop - 1}"})
    with urllib.request.urlopen(req) as r:
        data = r.read()
    assert len(data) == stop - start, (name, start, stop, len(data))
    return data


layer, experts = int(sys.argv[1]), [int(x) for x in sys.argv[2].split(",")]
weight_map = json.load(open("index.json"))["weight_map"]
headers = {}
out_header, blobs, offset = {}, [], 0
for e in experts:
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for suffix in ("weight", "weight_scale_inv"):
            key = f"model.layers.{layer}.mlp.experts.{e}.{proj}.{suffix}"
            shard = weight_map[key]
            if shard not in headers:
                n = struct.unpack("<Q", get(shard, 0, 8))[0]
                headers[shard] = (8 + n, json.loads(get(shard, 8, 8 + n)))
            base, header = headers[shard]
            info = header[key]
            a, b = info["data_offsets"]
            blob = get(shard, base + a, base + b)
            out_header[key] = {"dtype": info["dtype"], "shape": info["shape"],
                               "data_offsets": [offset, offset + len(blob)]}
            blobs.append(blob)
            offset += len(blob)
            print(key, info["dtype"], info["shape"], flush=True)
raw = json.dumps(out_header).encode()
raw += b" " * (-len(raw) % 8)
with open(f"experts_L{layer}.safetensors", "wb") as f:
    f.write(struct.pack("<Q", len(raw)))
    f.write(raw)
    for blob in blobs:
        f.write(blob)
