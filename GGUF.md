# Wyrmling → GGUF (seamless cross-platform)

The whole reason Wyrmling is **Mamba-2**: it's a supported llama.cpp architecture, so a trained
model converts to ONE `.gguf` that runs on every llama.cpp platform — Android, iOS, desktop
(CPU/CUDA/Metal/Vulkan), and WASM — with no per-platform work.

**Status: recipe documented + HF-export half tested (tiny model). Full conversion is verified on
the tiny model as the first export step — not yet run end-to-end.**

## Recipe
```bash
# 1) export the trained checkpoint to HF format  (tested — emits config.json + model.safetensors)
python -c "from wyrmling import build; import torch; \
  ck=torch.load('out/wyrmling_w300m.pt',map_location='cpu'); \
  m=build(ck['config']); m.load_state_dict(ck['model']); m.save_pretrained('wyrmling-w300m-hf')"

# 2) attach a tokenizer  (THE one integration detail — see below)

# 3) convert with llama.cpp (Mamba-2 is natively supported)
git clone https://github.com/ggerganov/llama.cpp
python llama.cpp/convert_hf_to_gguf.py wyrmling-w300m-hf \
  --outfile wyrmling-w300m-f16.gguf --outtype f16

# 4) quantize for mobile
llama.cpp/llama-quantize wyrmling-w300m-f16.gguf wyrmling-w300m-Q4_K_M.gguf Q4_K_M

# 5) run ANYWHERE
llama.cpp/llama-cli -m wyrmling-w300m-Q4_K_M.gguf -p "hello"
```

## The one integration detail: the byte-level tokenizer
We're byte-level (`vocab_size=256`, no tokenizer files). `convert_hf_to_gguf.py` needs tokenizer
metadata, so before step 3 we must attach one of:
- **(A) a minimal byte-level tokenizer** — a `tokenizer.json` whose 256 tokens are the raw bytes
  (`<0x00>`…`<0xFF>`), mapped to llama.cpp's byte vocab. Keeps the tokenizer-free property. Preferred.
- **(B) a small standard BPE** (e.g. a 8–16k tokenizer trained on our corpus) — maximally seamless
  for the converter, better token-efficiency, at the cost of the pure byte-level elegance.

Nail this on the **tiny** model first (convert → run one prompt) so the pipeline is proven before
the 300M/1B runs. This is the only non-boilerplate step; the architecture itself is already GGUF-native.

## Then: the Android app
The app's engine seam points at a **llama.cpp GGUF runtime** (JNI / `llama.android`) — load the
`.gguf`, done. Same OpenAI API the app already exposes on-device. No ExecuTorch, no per-platform port.
