"""Export a checkpoint to plain files the browser can load.

Writes docs/model/weights.bin (all tensors as float32, back to back) and
docs/model/config.json (model size, vocabulary and where each tensor lives).
Also writes tests/fixture.json so the JavaScript model can be checked
against PyTorch.
"""
import json
import os

import numpy as np
import torch

from model import GPT, GPTConfig

CKPT = os.path.join("out", "ckpt.pt")
OUT_DIR = os.path.join("docs", "model")

ckpt = torch.load(CKPT, map_location="cpu")
chars = ckpt["chars"]
model = GPT(GPTConfig(**ckpt["config"]))
model.load_state_dict(ckpt["model"])
model.eval()

tensors, blobs, offset = {}, [], 0
for name, value in model.state_dict().items():
    arr = value.numpy().astype("<f4")
    tensors[name] = {"shape": list(arr.shape), "offset": offset}
    blobs.append(arr.tobytes())
    offset += arr.size

os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "weights.bin"), "wb") as f:
    f.write(b"".join(blobs))
with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
    json.dump({"config": ckpt["config"], "chars": chars, "tensors": tensors}, f)
print(f"exported {offset:,} floats ({offset * 4 / 1e6:.1f} MB) to {OUT_DIR}")

# reference output for tests/parity.mjs
prompt = "Alice was beginning to get very tired"
stoi = {ch: i for i, ch in enumerate(chars)}
ids = [stoi[c] for c in prompt]
with torch.no_grad():
    logits, _ = model(torch.tensor([ids]))
with open(os.path.join("tests", "fixture.json"), "w") as f:
    json.dump({"prompt": prompt, "ids": ids,
               "logits": [round(x, 5) for x in logits[0, -1].tolist()]}, f)
print("wrote tests/fixture.json")
