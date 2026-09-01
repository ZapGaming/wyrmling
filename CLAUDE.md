# Wyrmling — local Claude Code session brief

You are running **locally on the user's Windows laptop** (real terminal — no sandbox/bridge limits).
Goal for this session: **train the ~100M Mamba-2 model on the local RTX 4060, then convert it to a
GGUF and drop it in the user's Ollama Models folder.** Everything below is already in this repo.

## What Wyrmling is
Byte-level (`vocab=256`, no tokenizer), **chat-focused** small LLM on **Mamba-2** (HF
`Mamba2ForCausalLM`). Chosen because Mamba-2 is llama.cpp-native → converts to one `.gguf` that
runs everywhere. Presets in `wyrmling.py`: `tiny` 3.8M, `s10m` 10M, `s100m` 106M, `w300m` 304M, `w1b` 1.03B.

## Local environment (already verified)
- Windows, Python 3.13, **torch 2.6.0+cu124**, **transformers 5.3.0**, git — all present.
- **RTX 4060 Laptop, 8 GB.** bf16 supported (Ada).
- `mamba-ssm`/`causal-conv1d` NOT installed → HF uses the **naive Mamba path** (correct, just slower;
  the "fast path not available" warning is expected/benign). Optionally try
  `pip install mamba-ssm causal-conv1d` for speed, but it's hard to build on Windows — skip if it fights.

## Run the training (this works — it's just a normal local process here)
```bat
cd C:\Users\zapm1\wyrmling-run
git pull
if not exist data\gutenberg\train.bin  python prepare_data.py --gutenberg --out data\gutenberg
python -u train.py --config s100m --data_dir data\gutenberg --block_size 160 --batch_size 10 ^
  --max_minutes 45 --eval_interval 25 --eval_iters 3 --warmup 50 --out_dir out
```
- `train.py` logs every iter to console AND flushes to `out\status.txt` (watch either). `bpb`=bits/byte,
  starts ~8.0 (random), should drop. Checkpoints to `out\wyrmling_s100m.pt`. `--max_minutes` caps wall-clock.
- If 8 GB OOMs, lower `--block_size`/`--batch_size`. If too slow, use `--config s10m` first to prove learning.

## Then package for Ollama
```bat
python -u to_ollama.py
```
Exports HF format + attempts GGUF, copies to `C:\Users\zapm1\Downloads\Ollama Models\wyrmling-100m\`.

## The ONE hard part — solve it properly here
`to_ollama.py`'s GGUF step will likely FAIL, because a **byte-level (256-vocab) model needs a tokenizer
that `convert_hf_to_gguf.py` accepts**, and the placeholder tokenizer in `to_ollama.py` is a guess.
This is the real work for this session (you have a terminal to iterate, unlike the remote attempt):
1. Clone `github.com/ggerganov/llama.cpp`, look at how it tokenizes Mamba/byte models.
2. Build a proper byte-level `tokenizer.json` (256 raw-byte tokens) OR set the right
   `--vocab-type`/tokenizer-model metadata so llama.cpp does raw-byte I/O.
3. Convert → `llama-quantize` to Q4_K_M/Q8_0 → verify with `llama-cli -m ... -p "hi"` that it runs.
4. Write the Ollama `Modelfile` (`FROM ./wyrmling-100m-*.gguf`) and confirm `ollama create` + `ollama run`.

## History / gotchas (so you don't repeat them)
- Remote attempts failed ONLY because the Zo device bridge kills background processes and times out —
  NOT a code bug. The model forward/backward/generate/export are all tested-working. Locally none of that applies.
- Repo is public: `github.com/ZapGaming/wyrmling`. Keep `git pull`ing / pushing as you go.
- `train.py` wraps everything in try/except → any crash's full traceback lands in `out\status.txt`.
