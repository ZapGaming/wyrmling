"""Train Wyrmling (Mamba-2). Byte-level, chat-focused.

  python train.py --config tiny  --data_dir data/tiny      # smoke test (CPU ok)
  python train.py --config w300m --data_dir data/wyrmling  # the prototype (GPU)
  python train.py --config w1b   --data_dir data/wyrmling  # the flagship (weeks)

GPU speed needs the Mamba kernels: `pip install mamba-ssm causal-conv1d` (CUDA). Without
them HF falls back to a slow but correct pure-PyTorch path (fine for CPU smoke tests).
bf16 on Ampere/170HX, auto fp16+GradScaler on Turing. Cosine LR+warmup, grad-accum, clip,
ckpt/resume, bits/byte eval, SQLite run DB. No fabricated metrics — prints what it measures.
"""
import argparse
import math
import os
import time

import torch

from data import ByteData
from metrics import RunDB
from wyrmling import PRESETS, build, n_params


def get_lr(it, warmup, max_iters, lr, min_lr):
    if it < warmup:
        return lr * (it + 1) / warmup
    if it > max_iters:
        return min_lr
    r = (it - warmup) / max(1, (max_iters - warmup))
    return min_lr + 0.5 * (1 + math.cos(math.pi * r)) * (lr - min_lr)


@torch.no_grad()
def estimate_loss(model, data, eval_iters, ctx):
    model.eval()
    out = {}
    for split in ("train", "val"):
        if split == "val" and data.val is None:
            continue
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, _ = data.get_batch(split)
            with ctx:
                losses[k] = model(input_ids=x, labels=x).loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(PRESETS), default="tiny")
    ap.add_argument("--data_dir", default="data/tiny")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--block_size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--max_iters", type=int, default=3000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min_lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_interval", type=int, default=250)
    ap.add_argument("--eval_iters", type=int, default=50)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = "bfloat16" if use_bf16 else ("float16" if torch.cuda.is_available() else "float32")
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = (torch.amp.autocast(device_type=device.type, dtype=ptdtype)
           if device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False))
    scaler = torch.amp.GradScaler(device=device.type, enabled=(dtype == "float16"))
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True

    print(f"config={a.config}  params={n_params(a.config)/1e6:.1f}M  device={device}  dtype={dtype}")
    data = ByteData(a.data_dir, a.block_size, a.batch_size, device)
    model = build(a.config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay, betas=(0.9, 0.95))

    start_iter = 0
    ckpt_path = os.path.join(a.out_dir, f"wyrmling_{a.config}.pt")
    if a.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); start_iter = ck["iter"]
        print(f"resumed from iter {start_iter}")

    db = RunDB(os.path.join(a.out_dir, "runs.db"))
    db.start(a.config, n_params(a.config) / 1e6,
             {"block_size": a.block_size, "batch_size": a.batch_size,
              "grad_accum": a.grad_accum, "lr": a.lr, "max_iters": a.max_iters, "arch": "mamba2"})

    ln2 = math.log(2)
    t0 = time.time()
    running, steps = 0.0, 0
    for it in range(start_iter, a.max_iters):
        lr = get_lr(it, a.warmup, a.max_iters, a.lr, a.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        for _ in range(a.grad_accum):
            x, _ = data.get_batch("train")
            with ctx:
                loss = model(input_ids=x, labels=x).loss / a.grad_accum
            scaler.scale(loss).backward()
            running += loss.item() * a.grad_accum
            steps += 1
        if a.grad_clip:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        scaler.step(opt); scaler.update()

        if it % a.eval_interval == 0 or it == a.max_iters - 1:
            m = estimate_loss(model, data, a.eval_iters, ctx)
            tr = m.get("train", float("nan")); va = m.get("val", float("nan"))
            dt = time.time() - t0
            print(f"iter {it:6d} | train {tr:.3f} ({tr/ln2:.3f} b/byte) | "
                  f"val {va:.3f} ({va/ln2:.3f} b/byte) | lr {lr:.2e} | {dt:.0f}s")
            db.log(it, tr, va, lr, dt)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "iter": it, "config": a.config}, ckpt_path)
        elif it % 50 == 0:
            print(f"iter {it:6d} | loss {running/max(steps,1):.3f} | lr {lr:.2e}")
            running, steps = 0.0, 0

    print("done. sample:")
    model.eval()
    prompt = torch.tensor(bytearray("The ", "utf-8"), dtype=torch.long, device=device).unsqueeze(0)
    out = model.generate(prompt, max_new_tokens=160, do_sample=True, temperature=0.9,
                         top_k=40, eos_token_id=4, pad_token_id=0)
    print(bytes(out[0].to(torch.uint8).cpu()).decode(errors="backslashreplace"))


if __name__ == "__main__":
    main()
