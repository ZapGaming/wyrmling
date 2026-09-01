@echo off
setlocal
cd /d %~dp0
title Wyrmling Trainer
echo ============================================================
echo    WYRMLING  -  Mamba-2 byte-level LLM  -  trains on your GPU
echo    Keep this window OPEN. Live progress shows below.
echo ============================================================
echo.

echo [1/4] Getting latest code...
set BASE=https://raw.githubusercontent.com/ZapGaming/wyrmling/master
for %%F in (train.py wyrmling.py data.py metrics.py prepare_data.py to_ollama.py) do (
  powershell -NoProfile -Command "try { iwr %BASE%/%%F -OutFile %%F } catch { Write-Host '  (kept local %%F)' }"
)
echo.

echo [2/4] Preparing training data...
if exist data\gutenberg\train.bin goto data_ok
python prepare_data.py --gutenberg --out data\gutenberg
:data_ok
echo   data ready.
echo.

echo [3/4] Training on GPU (up to 45 min). Edit --max_minutes in this .bat to change.
echo    Live loss below (bpb = bits-per-byte, lower is better; random start ~8.0):
echo.
python -u train.py --config s100m --data_dir data\gutenberg --block_size 160 --batch_size 10 --max_minutes 45 --eval_interval 25 --eval_iters 3 --warmup 50 --out_dir out
echo.

echo [4/4] Packaging for Ollama...
python -u to_ollama.py
echo.

echo ============================================================
echo    FINISHED. Output folder:
echo    C:\Users\zapm1\Downloads\Ollama Models\wyrmling-100m
echo ============================================================
pause
