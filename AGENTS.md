# Wyrmling — project guide

Byte-level, **chat-focused** small LLM. Architecture: **Mamba-2** (pure selective SSM — pivoted
FROM BDH). **Separate project from `ring0`/Faultward.** Arch decision: `ARCHITECTURE.md`. GGUF
path: `GGUF.md`. Mobile: `MOBILE.md`. Roadmap/datasets (still valid): `../ring0/BDH-1B-PLAN.md`.

## Why Mamba-2 (the pivot)
Resolves "unique architecture" + "seamless GGUF everywhere": Mamba-2 is non-transformer (unique)
AND natively supported by llama.cpp + `convert_hf_to_gguf.py`. BDH is NOT in llama.cpp → dropped.
The SSM's constant-size state IS the high-TPS decode (O(1)/token) — native to HF `generate`.

## Identity / rules
- **Chat model, NOT code.** Enforced only in the DATA (`prepare_data.py` MIX is code-free).
- **Pre-launch. Software stack built + tested; weights NOT trained.** Never publish a number that
  didn't come from our own run.
- Byte-level, `vocab_size=256`, chat framing via control bytes 0x02/0x03/0x04 (`data.py`).

## Presets (verified param counts, byte vocab, tied embeds)
`tiny`=3.8M (smoke) · **`w300m`=303.9M (prototype)** · **`w1b`=1034.5M (flagship)**.
Config in `wyrmling.py` (`_cfg`): head_dim=64, expand=2, state=128, n_groups=1. Constraint:
`hidden·expand == num_heads·head_dim`.

## Files & what's TESTED (this session, CPU)
- `wyrmling.py` — Mamba-2 presets (`build`, `n_params`) via HF `Mamba2ForCausalLM`. Param counts
  verified; tiny forward+backward+grads pass; HF `generate` works; `save_pretrained` emits HF format
  (`architectures:["Mamba2ForCausalLM"]`) → feeds the GGUF converter. ✅
- `train.py`+`metrics.py` — HF API (`model(input_ids=x, labels=x).loss`), bf16/fp16 auto, cosine LR,
  ckpt/resume, bits/byte eval, SQLite run DB. Multi-iter run = GPU (CPU too slow here / naive Mamba path).
- `serve.py` — OpenAI-compatible local server (stdlib http; `/v1/models`, `/v1/chat/completions`
  stream+non-stream). Server framework tested over HTTP; now calls HF `generate`.
- `data.py`/`prepare_data.py` — byte dataset + chat control-byte framing; code-free `--mix chat`.
- `chat.py` — REPL via HF generate.
- Fast GPU kernels: `pip install mamba-ssm causal-conv1d` (CUDA). Without them HF uses a correct
  slow pure-PyTorch fallback (the fast-path warning is expected/benign).

## Android app (`android/`) — on-device OpenAI API
Kotlin/Compose (Zome's Gradle: AGP 8.5.2 / Kotlin 1.9.24 / Gradle 8.9). Embedded NanoHTTPD on
`0.0.0.0:8080` exposes `/v1`. Pluggable `WyrmEngine`: `MockEngine` wired; **`LlamaCppEngine`** is the
GGUF/JNI seam (renamed from ExecuTorch — Mamba-2 is GGUF-native so llama.cpp is the runtime).
CI `.github/workflows/android.yml` builds a debug APK — **NOT yet compiled (CI-pending)**.

## Deliverable = GGUF (seamless)
Train → `save_pretrained` → attach byte tokenizer (the one integration detail, `GGUF.md`) →
`convert_hf_to_gguf.py` → `.gguf` runs on every llama.cpp platform incl. the app. Verify on `tiny` first.

## Don't
- Don't add code to the data mix. Don't `pkill -f python3`/broad patterns (starves/OOMs the box —
  stop servers by port: `kill $(lsof -ti tcp:<port>)`). Don't publish unmeasured numbers.
