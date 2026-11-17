"""Generate text from a trained checkpoint, e.g.
    python src/sample.py --prompt "Alice was"
"""
import argparse
import os

import torch

from model import GPT, GPTConfig

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default=os.path.join("out", "ckpt.pt"))
p.add_argument("--prompt", default="\n")
p.add_argument("--chars", type=int, default=400)
p.add_argument("--temperature", type=float, default=0.8)
p.add_argument("--top-k", type=int, default=20)
args = p.parse_args()

ckpt = torch.load(args.ckpt, map_location="cpu")
chars = ckpt["chars"]
stoi = {ch: i for i, ch in enumerate(chars)}
model = GPT(GPTConfig(**ckpt["config"]))
model.load_state_dict(ckpt["model"])
model.eval()

ids = [stoi[c] for c in args.prompt if c in stoi] or [stoi["\n"]]
out = model.generate(torch.tensor([ids]), args.chars, args.temperature, args.top_k)
print("".join(chars[i] for i in out[0].tolist()))
