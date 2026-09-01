"""Talk to a trained Wyrmling (Mamba-2) checkpoint (byte-level, chat-formatted).

  python chat.py --ckpt out/wyrmling_w300m.pt
"""
import argparse

import torch

from data import ASSISTANT, EOT, USER
from wyrmling import build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max_new", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=device)
    model = build(ck["config"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"Wyrmling [{ck['config']}] ready. Ctrl-C to exit.\n")

    history = []
    while True:
        try:
            user = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not user:
            continue
        history += [USER] + list(user.encode("utf-8", "replace")) + [ASSISTANT]
        idx = torch.tensor(history, dtype=torch.long, device=device).unsqueeze(0)
        out = model.generate(idx, max_new_tokens=a.max_new, do_sample=True,
                             temperature=a.temperature, top_k=a.top_k,
                             eos_token_id=EOT, pad_token_id=0)
        full = out[0].tolist()
        reply_ids = [b for b in full[len(history):] if b not in (USER, ASSISTANT, EOT)]
        print(f"wyrm › {bytes(reply_ids).decode('utf-8','replace')}\n")
        history = full[:4096] if len(full) <= 4096 else full[-4096:]


if __name__ == "__main__":
    main()
