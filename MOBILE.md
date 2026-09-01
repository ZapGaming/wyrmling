# Wyrmling on mobile — one GGUF, everywhere

**Status: pre-launch. App UI + on-device OpenAI API built and tested (Python side); no trained
weights yet, so on-device inference runs a labelled demo engine until the `.gguf` exists.**

## The path is now trivial (that's the point of the Mamba-2 pivot)
Wyrmling is **Mamba-2**, a natively-supported llama.cpp architecture. So there is no per-platform
runtime work: train → `convert_hf_to_gguf.py` → **one `.gguf`** that runs on Android, iOS, desktop
(CPU/CUDA/Metal/Vulkan) and WASM via llama.cpp. Full recipe in `GGUF.md`.

- Constant-size SSM state = O(1)/token, O(1) memory decode → excellent on phones.
- Quantized: 300M ≈ ~150–200 MB, 1B ≈ ~500–600 MB (Q4_K_M). Phone-sized.
- Byte-level (vocab 256): one integration detail — attach a byte tokenizer for conversion (`GGUF.md`).

## The app (`android/`)
Compose chat app that **exposes an OpenAI-compatible API on-device** (NanoHTTPD, `0.0.0.0:8080`):
any OpenAI SDK on the phone/LAN hits `http://<ip>:8080/v1`. Inference goes through the pluggable
`WyrmEngine` — `MockEngine` now, **`LlamaCppEngine` (GGUF via JNI)** once the model is trained +
converted. Same API shape as desktop `serve.py`, so clients are identical everywhere.

## Sequence
Train 300M → 1B (HF) → attach byte tokenizer → `.gguf` (verify on tiny first) → drop into
`LlamaCppEngine` → ship APK via CI. No performance numbers until measured on our own runs.
