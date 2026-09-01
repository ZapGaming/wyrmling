# Wyrmling — byte-level chat LLM on **Mamba-2** (pure selective state-space, NOT a transformer).
#
# Why Mamba-2 (pivot from BDH): it is BOTH non-standard/unique AND natively supported by
# llama.cpp + HF's convert_hf_to_gguf.py. So a trained Wyrmling converts to ONE .gguf that
# runs seamlessly on every llama.cpp platform (Android, iOS, desktop, WASM). BDH is not in
# llama.cpp, so it could never be "seamless GGUF" without a from-scratch C++ port.
#
# The SSM's constant-size recurrent state IS the high-TPS decode (O(1)/token, O(1) memory) —
# native to HF generate. Byte-level (vocab=256) keeps our tokenizer-free chat framing.
import warnings

import torch
from transformers import Mamba2Config, Mamba2ForCausalLM

warnings.filterwarnings("ignore")

VOCAB = 256          # byte-level, no tokenizer
EOS_BYTE = 4         # EOT control byte (see data.py) — generation stop token
PAD_BYTE = 0


def _cfg(hidden, layers, head_dim=64, expand=2, state=128, groups=1):
    return Mamba2Config(
        vocab_size=VOCAB,
        hidden_size=hidden,
        num_hidden_layers=layers,
        state_size=state,
        expand=expand,
        head_dim=head_dim,
        num_heads=hidden * expand // head_dim,   # constraint: hidden*expand == num_heads*head_dim
        n_groups=groups,
        chunk_size=128,
        tie_word_embeddings=True,
        bos_token_id=1,
        eos_token_id=EOS_BYTE,
        pad_token_id=PAD_BYTE,
    )


# Sizes verified via meta-device param count (byte vocab, tied embeddings):
PRESETS = {
    "tiny":  _cfg(256, 8),     # smoke test (~small), CPU-runnable
    "s10m":  _cfg(384, 10),    # ~10.0M — quick "does it learn" test model
    "s100m": _cfg(1024, 16),   # ~105.9M — bigger GPU test (RTX 4060)
    "w300m": _cfg(1024, 46),   # prototype  (~304M)
    "w1b":   _cfg(2048, 40),   # flagship   (~1.03B)
}


def build(preset: str) -> Mamba2ForCausalLM:
    """Instantiate a Wyrmling (Mamba-2) model for a named preset."""
    return Mamba2ForCausalLM(PRESETS[preset])


def n_params(preset: str) -> int:
    """Param count without allocating (meta device)."""
    with torch.device("meta"):
        m = Mamba2ForCausalLM(PRESETS[preset])
    return sum(p.numel() for p in m.parameters())
