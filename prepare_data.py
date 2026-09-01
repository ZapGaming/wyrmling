"""Build train.bin / val.bin (uint8 byte streams) for Wyrmling.

Two modes:

  # 1) Smoke test — no deps beyond `requests`. Downloads tiny-shakespeare.
  python prepare_data.py --tiny --out data/tiny

  # 2) Real CHAT-FOCUSED mix — needs `datasets`. Streams a byte budget from a
  #    curated NON-CODE mix (web + synthetic knowledge + books + dialogue).
  python prepare_data.py --mix chat --gb 20 --out data/wyrmling

The chat mix is deliberately code-free (this is a chat model). Dialogue sources
are wrapped in the reserved-control-byte chat format; web/books are raw bytes.
Edit MIX below to retune the blend. Nothing here is measured — it just packs data.
"""
import argparse
import os
import sys

import numpy as np

from data import encode_conversation, encode_text

# Chat-focused blend. Weights are byte-budget fractions (must sum ~1.0).
# NO CODE. Knowledge comes from synthetic textbooks (Cosmopedia) + edu web;
# warmth/persona from the dialogue sets. Adjust freely.
MIX = {
    "chat": [
        # (hf_dataset, config, split, text_field, weight, is_chat)
        ("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", "train", "text", 0.45, False),
        ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2",     "train", "text", 0.25, False),
        ("pg19",                         None,               "train", "text", 0.10, False),
        ("Anthropic/hh-rlhf",            None,               "train", None,   0.08, "hh"),
        ("HuggingFaceH4/ultrachat_200k", None,       "train_sft", "messages", 0.07, "messages"),
        ("facebook/empathetic_dialogues", None,             "train", None,   0.05, "empathetic"),
    ],
}


def _write_bin(byts: bytes, out_path: str):
    arr = np.frombuffer(byts, dtype=np.uint8)
    arr.tofile(out_path)
    print(f"  wrote {out_path}  ({len(arr)/1e6:.1f} MB)")


def _get(url):
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "wyrmling/1.0"})
    return urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")


def build_tiny(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    b = bytes(encode_text(_get(url)))
    n = int(0.9 * len(b))
    _write_bin(b[:n], os.path.join(out_dir, "train.bin"))
    _write_bin(b[n:], os.path.join(out_dir, "val.bin"))
    print("tiny-shakespeare ready — smoke-test data.")


# Public-domain, dialogue-rich books (good for a chat-focused byte model). ~10-20 MB total.
GUTENBERG = [1342, 84, 1661, 2701, 98, 1400, 74, 345, 2542, 158]


def build_gutenberg(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    parts = []
    for bid in GUTENBERG:
        try:
            txt = _get(f"https://www.gutenberg.org/cache/epub/{bid}/pg{bid}.txt")
            parts.append(txt)
            print(f"  book {bid}: {len(txt)/1e6:.1f} MB")
        except Exception as e:
            print(f"  book {bid}: skip ({str(e)[:40]})")
    b = bytes(encode_text("\n\n".join(parts)))
    n = int(0.95 * len(b))
    _write_bin(b[:n], os.path.join(out_dir, "train.bin"))
    _write_bin(b[n:], os.path.join(out_dir, "val.bin"))
    print(f"gutenberg corpus ready — {len(b)/1e6:.1f} MB total.")


def _chat_bytes_from(row, kind):
    """Turn one dataset row into chat-formatted bytes (assistant-warm framing)."""
    if kind == "messages":  # ultrachat: [{"role","content"}]
        ids, _ = encode_conversation(row["messages"])
        return bytes(ids)
    if kind == "hh":  # anthropic hh: "chosen" transcript "\n\nHuman: .. \n\nAssistant: .."
        txt = row.get("chosen", "")
        msgs = []
        for chunk in txt.split("\n\nHuman: ")[1:]:
            if "\n\nAssistant: " in chunk:
                u, a = chunk.split("\n\nAssistant: ", 1)
                msgs.append({"role": "user", "content": u.strip()})
                msgs.append({"role": "assistant", "content": a.strip()})
        ids, _ = encode_conversation(msgs) if msgs else ([], [])
        return bytes(ids)
    if kind == "empathetic":
        msgs = [{"role": "user", "content": row.get("context", "")},
                {"role": "assistant", "content": row.get("utterance", "")}]
        ids, _ = encode_conversation(msgs)
        return bytes(ids)
    return b""


def build_mix(mix_name, gb, out_dir):
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("pip install datasets  (needed for the real mix; use --tiny to smoke-test)")
    os.makedirs(out_dir, exist_ok=True)
    budget = int(gb * 1e9)
    specs = MIX[mix_name]
    train_path = os.path.join(out_dir, "train.bin")
    written = 0
    with open(train_path, "wb") as fout:
        for ds, cfg, split, field, weight, is_chat in specs:
            target = int(budget * weight)
            got = 0
            print(f"[{ds}:{cfg or ''}] target {target/1e9:.2f} GB")
            stream = load_dataset(ds, cfg, split=split, streaming=True)
            for row in stream:
                if is_chat:
                    b = _chat_bytes_from(row, is_chat)
                else:
                    b = bytes(encode_text(row[field])) + b"\n"
                if not b:
                    continue
                fout.write(b)
                got += len(b); written += len(b)
                if got >= target:
                    break
            print(f"  +{got/1e9:.2f} GB")
    print(f"train.bin = {written/1e9:.2f} GB. (Hold out a val slice separately if desired.)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true", help="tiny-shakespeare smoke data")
    ap.add_argument("--gutenberg", action="store_true", help="~15 MB public-domain books")
    ap.add_argument("--mix", choices=list(MIX), help="real chat-focused mix (needs datasets)")
    ap.add_argument("--gb", type=float, default=20.0, help="byte budget in GB for the mix")
    ap.add_argument("--out", default="data/wyrmling")
    a = ap.parse_args()
    if a.tiny:
        build_tiny(a.out)
    elif a.gutenberg:
        build_gutenberg(a.out)
    elif a.mix:
        build_mix(a.mix, a.gb, a.out)
    else:
        ap.error("pass --tiny, --gutenberg, or --mix chat")
