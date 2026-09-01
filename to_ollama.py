"""Package a trained Wyrmling checkpoint into the user's Ollama Models folder.

RELIABLE: exports HF format (config.json + safetensors + tokenizer) into the
destination as a folder-with-config. BEST-EFFORT: converts to GGUF via llama.cpp
(byte-level tokenizer is the one fiddly bit). Never hard-crashes the caller.
"""
import os
import shutil
import subprocess
import sys
import traceback

import torch

from wyrmling import build

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out")
CKPT = os.path.join(OUT, "wyrmling_s100m.pt")
HF = os.path.join(OUT, "wyrmling-hf")
DEST = r"C:\Users\zapm1\Downloads\Ollama Models\wyrmling-100m"


def export_hf():
    ck = torch.load(CKPT, map_location="cpu")
    m = build(ck["config"])
    m.load_state_dict(ck["model"])
    m.save_pretrained(HF)


def write_byte_tokenizer(path):
    # minimal char/byte-level tokenizer (256 tokens) so the converter has a vocab
    from tokenizers import Tokenizer, models, pre_tokenizers
    vocab = {chr(i): i for i in range(256)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token=chr(0)))
    tok.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    tok.save(os.path.join(path, "tokenizer.json"))


def try_gguf():
    llama = os.path.join(ROOT, "llama.cpp")
    if not os.path.isdir(llama):
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/ggerganov/llama.cpp", llama], check=True)
    write_byte_tokenizer(HF)
    gguf = os.path.join(OUT, "wyrmling-100m-f16.gguf")
    subprocess.run([sys.executable, os.path.join(llama, "convert_hf_to_gguf.py"), HF,
                    "--outfile", gguf, "--outtype", "f16"], check=True)
    return gguf


def main():
    if not os.path.exists(CKPT):
        print("!! No checkpoint at", CKPT, "- train first.")
        return
    print("Exporting HF format...")
    export_hf()
    os.makedirs(DEST, exist_ok=True)

    hfdest = os.path.join(DEST, "hf")
    if os.path.isdir(hfdest):
        shutil.rmtree(hfdest)
    shutil.copytree(HF, hfdest)
    print("HF model placed ->", hfdest)

    gguf_ok = False
    try:
        print("Converting to GGUF via llama.cpp (first run clones it, ~1 min)...")
        g = try_gguf()
        shutil.copy(g, DEST)
        with open(os.path.join(DEST, "Modelfile"), "w") as f:
            f.write("FROM ./" + os.path.basename(g) + "\n")
        gguf_ok = True
        print("GGUF placed ->", DEST)
    except Exception:
        print("GGUF step failed (byte-level tokenizer needs manual wiring) - HF model is still there.")
        traceback.print_exc()

    with open(os.path.join(DEST, "README.txt"), "w", encoding="utf-8") as f:
        f.write("Wyrmling 100M - Mamba-2, byte-level, trained locally on your 4060.\n\n")
        if gguf_ok:
            f.write("GGUF ready. In this folder run:\n  ollama create wyrmling -f Modelfile\n  ollama run wyrmling\n")
        else:
            f.write("HF model is in hf\\. GGUF conversion pending the byte-tokenizer step.\n")
    print("DONE ->", DEST)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
