"""Train the tiny GPT on data/input.txt (character level)."""
import argparse
import math
import os
import time

import torch

from model import GPT, GPTConfig

p = argparse.ArgumentParser()
p.add_argument("--iters", type=int, default=3000)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--eval-every", type=int, default=250)
p.add_argument("--out", default=os.path.join("out", "ckpt.pt"))
args = p.parse_args()

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"

with open(os.path.join("data", "input.txt"), encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]

cfg = GPTConfig(vocab_size=len(chars))
model = GPT(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"device={device}  vocab={len(chars)}  params={n_params/1e6:.2f}M")


def get_batch(source):
    ix = torch.randint(len(source) - cfg.block_size - 1, (args.batch,))
    x = torch.stack([source[i:i + cfg.block_size] for i in ix])
    y = torch.stack([source[i + 1:i + 1 + cfg.block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(n=20):
    model.eval()
    out = {}
    for name, source in (("train", train_data), ("val", val_data)):
        losses = [model(*get_batch(source))[1].item() for _ in range(n)]
        out[name] = sum(losses) / n
    model.train()
    return out


def lr_at(step, warmup=100):
    if step < warmup:
        return args.lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, args.iters - warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
t0 = time.time()
for step in range(args.iters + 1):
    if step % args.eval_every == 0 or step == args.iters:
        losses = estimate_loss()
        print(f"step {step:5d}  train {losses['train']:.3f}  val {losses['val']:.3f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        torch.save({"model": model.state_dict(), "config": cfg.__dict__, "chars": chars}, args.out)
    for group in opt.param_groups:
        group["lr"] = lr_at(step)
    _, loss = model(*get_batch(train_data))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

print(f"done, saved {args.out}")
