# 🐉 Wyrmling

**A byte-level, chat-focused small LLM on the Mamba-2 architecture — one GGUF that runs everywhere.**

A *wyrmling* is a baby dragon. Architecture: **Mamba-2** — a pure selective **state-space model**
(not a transformer). Chosen because it's *both* genuinely unique *and* natively supported by
llama.cpp, so a trained Wyrmling converts to **one `.gguf`** that runs seamlessly on Android, iOS,
desktop and WASM. (Pivoted from BDH, which isn't GGUF-native — see `ARCHITECTURE.md`.)

**Status: pre-launch. Stack built + tested; no weights trained. No number ships that isn't ours.**

Chat model, **not** code — the model is task-agnostic, so that's enforced purely in the data mix.
Byte-level (`vocab=256`, no tokenizer). Separate project from Faultward/ring0.

## Sizes (verified)
`tiny` 3.8M (smoke) · **`w300m` 303.9M (prototype)** · **`w1b` 1034.5M (flagship)**.

## Quickstart (smoke test — CPU ok)
```bash
pip install -r requirements.txt
python prepare_data.py --tiny --out data/tiny
python train.py --config tiny --data_dir data/tiny --max_iters 500
python serve.py --ckpt out/wyrmling_tiny.pt      # OpenAI API on :8080
```

## Layout
| File | What |
|---|---|
| `wyrmling.py` | Mamba-2 presets (`build`, `n_params`) via HF `Mamba2ForCausalLM` |
| `data.py` / `prepare_data.py` | byte dataset + chat control-byte framing; code-free `--mix chat` |
| `train.py` + `metrics.py` | training loop (HF API) + SQLite run DB |
| `serve.py` / `chat.py` | OpenAI-compatible local server / REPL |
| `android/` | Compose app exposing the OpenAI API on-device (llama.cpp GGUF engine seam) |
| `ARCHITECTURE.md` / `GGUF.md` / `MOBILE.md` | the pivot decision / conversion recipe / mobile |

## Deliverable
Train → `save_pretrained` → attach byte tokenizer → `convert_hf_to_gguf.py` → `.gguf` →
llama.cpp everywhere. GPU speed needs `pip install mamba-ssm causal-conv1d`. Recipe in `GGUF.md`.
