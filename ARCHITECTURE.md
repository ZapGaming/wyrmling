# Wyrmling architecture decision — BDH → Mamba-2

**Chosen: Mamba-2 (pure selective state-space model). Byte-level, chat-focused.**

## Why the pivot
Two goals were in tension until this decision:
1. **Seamless GGUF / cross-platform** — one model file that runs everywhere.
2. **A unique, non-standard architecture** — not a vanilla transformer.

The resolving fact: **llama.cpp only runs architectures it has a C++ graph for.** BDH (the
previous plan) is *not* one — it would need a from-scratch port, so it can never be "seamless GGUF."
But several **non-transformer** architectures ARE natively supported by both llama.cpp and HF's
`convert_hf_to_gguf.py`:

| Arch | Type | Unique? | GGUF-native? | Notes |
|---|---|---|---|---|
| **Mamba-2** | selective SSM | ✅ very | ✅ | **chosen** — clean HF training, constant-state decode |
| RWKV-7 | linear-attn RNN | ✅ very | ✅ | strong alt; more bespoke training stack |
| Jamba | Mamba+MoE+attn hybrid | ✅ | ✅ | heavier, MoE complexity |
| FalconMamba | pure SSM | ✅ | ✅ | similar to Mamba |
| ~~BDH~~ | Hebbian/graph | ✅✅ | ❌ | not in llama.cpp — dropped for seamlessness |

Mamba-2 wins for a from-scratch chat model: cleanest HF path (`Mamba2ForCausalLM` → train → convert),
and the SSM's **constant-size recurrent state IS the high-TPS decode** (O(1)/token, O(1) memory) —
native to HF `generate`, and ideal for mobile.

## Chosen presets (verified param counts, byte vocab=256, tied embeddings)
| Preset | hidden | layers | heads | params |
|---|---:|---:|---:|---:|
| `tiny` (smoke) | 256 | 8 | 8 | 3.8M |
| **`w300m` (prototype)** | 1024 | 46 | 32 | **303.9M** |
| **`w1b` (flagship)** | 2048 | 40 | 64 | **1034.5M** |

Config: `head_dim=64, expand=2, state_size=128, n_groups=1, chunk_size=128`. Constraint enforced by
Mamba2: `hidden·expand == num_heads·head_dim`.

## What carries over from the BDH plan (`../ring0/BDH-1B-PLAN.md`, now arch-superseded)
- **Roadmap unchanged**: 300M prototype → 1B flagship.
- **Datasets unchanged**: code-free chat mix (SmolLM-corpus / FineWeb-Edu / Cosmopedia / dialogue).
- **Chat framing unchanged**: byte control bytes 0x02/0x03/0x04 (`data.py`).
- Dropped: BDH param math, shared-weight recurrence, the BDH-specific add-on menu.

## Speed
Mamba-2 fast kernels need `pip install mamba-ssm causal-conv1d` (CUDA) on the 170HX/GPU box.
Without them HF uses a correct but slow pure-PyTorch fallback (fine for CPU smoke tests).

## Serving / deliverable
Train (HF) → `save_pretrained` → attach byte tokenizer → `convert_hf_to_gguf.py` → `.gguf` →
llama.cpp everywhere (incl. the Android app's engine). Full path in `GGUF.md`. No numbers until measured.
