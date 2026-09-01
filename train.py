"""Train Wyrmling (Mamba-2). Byte-level, chat-focused. HEAVY file logging.

Everything important is appended (flushed) to <out_dir>/status.txt so progress is
visible even if stdout is buffered or the process is killed. Any crash writes its
full traceback to status.txt. Use --max_minutes to cap wall-clock.

  python -u train.py --config s100m --data_dir data/gutenberg --max_minutes 45 --out_dir out
"""
import argparse
import math
import os
import time
import traceback

import torch

from data import ByteData
from metrics import RunDB
from wyrmling import PRESETS, build, n_params

_T0 = time.time()
_STATUS = None


def log(msg):
    """Append a flushed, timestamped line to status.txt AND print it."""
    line = f"[{time.time()-_T0:7.1f}s] {msg}"
    print(line, flush=True)
    if _STATUS:
        with open(_STATUS, "a", encoding="utf-8") as f:
            f.write(line + "\n")


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


def run(a):
    ln2 = math.log(2)
    log(f"BOOT config={a.config} data={a.data_dir} block={a.block_size} batch={a.batch_size} "
        f"max_iters={a.max_iters} max_minutes={a.max_minutes}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        log(f"CUDA ok: {torch.cuda.get_device_name(0)} "
            f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    else:
        log("WARNING: CUDA NOT available — running on CPU")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = "bfloat16" if use_bf16 else ("float16" if torch.cuda.is_available() else "float32")
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = (torch.amp.autocast(device_type=device.type, dtype=ptdtype)
           if device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False))
    scaler = torch.amp.GradScaler(device=device.type, enabled=(dtype == "float16"))
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    log(f"dtype={dtype} params={n_params(a.config)/1e6:.1f}M")

    data = ByteData(a.data_dir, a.block_size, a.batch_size, device)
    log(f"DATA_LOADED train_bytes={len(data.train)} val_bytes={0 if data.val is None else len(data.val)}")

    model = build(a.config).to(device)
    log("MODEL_BUILT")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay, betas=(0.9, 0.95))

    start_iter = 0
    ckpt_path = os.path.join(a.out_dir, f"wyrmling_{a.config}.pt")
    if a.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); start_iter = ck["iter"]
        log(f"RESUMED from iter {start_iter}")

    db = RunDB(os.path.join(a.out_dir, "runs.db"))
    db.start(a.config, n_params(a.config) / 1e6, {"block": a.block_size, "batch": a.batch_size, "lr": a.lr})
    log("OPT_READY entering loop")

    running, steps = 0.0, 0
    for it in range(start_iter, a.max_iters):
        if a.max_minutes and (time.time() - _T0) / 60.0 >= a.max_minutes:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "iter": it,
                        "config": a.config}, ckpt_path)
            log(f"TIME_LIMIT at iter {it}; saved {ckpt_path}")
            break
        lr = get_lr(it, a.warmup, a.max_iters, a.lr, a.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        last = 0.0
        for _ in range(a.grad_accum):
            x, _y = data.get_batch("train")
            with ctx:
                loss = model(input_ids=x, labels=x).loss / a.grad_accum
            scaler.scale(loss).backward()
            last += loss.item() * a.grad_accum
            running += loss.item() * a.grad_accum
            steps += 1
        if a.grad_clip:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        scaler.step(opt); scaler.update()

        # heavy per-iter logging (every iter for first 30, then every 5th)
        if it < 30 or it % 5 == 0:
            log(f"iter {it} loss {last:.4f} bpb {last/ln2:.3f} lr {lr:.2e}")

        if it % a.eval_interval == 0 or it == a.max_iters - 1:
            m = estimate_loss(model, data, a.eval_iters, ctx)
            tr = m.get("train", float("nan")); va = m.get("val", float("nan"))
            log(f"EVAL iter {it} train_bpb {tr/ln2:.3f} val_bpb {va/ln2:.3f}")
            db.log(it, tr, va, lr, time.time() - _T0)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "iter": it,
                        "config": a.config}, ckpt_path)
            log(f"CKPT saved iter {it}")

    log("SAMPLING")
    model.eval()
    prompt = torch.tensor(bytearray("The ", "utf-8"), dtype=torch.long, device=device).unsqueeze(0)
    out = model.generate(prompt, max_new_tokens=160, do_sample=True, temperature=0.9,
                         top_k=40, eos_token_id=4, pad_token_id=0)
    sample = bytes(out[0].to(torch.uint8).cpu()).decode(errors="backslashreplace")
    log("SAMPLE " + sample.replace("\n", " ")[:200])
    log(f"TRAIN_DONE ckpt={ckpt_path}")


def main():
    global _STATUS
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(PRESETS), default="tiny")
    ap.add_argument("--data_dir", default="data/tiny")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--max_iters", type=int, default=100000)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min_lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_interval", type=int, default=25)
    ap.add_argument("--eval_iters", type=int, default=3)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_minutes", type=float, default=0.0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    _STATUS = os.path.join(a.out_dir, "status.txt")
    open(_STATUS, "w", encoding="utf-8").close()  # truncate at start
    try:
        run(a)
    except Exception:
        log("FATAL\n" + traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
