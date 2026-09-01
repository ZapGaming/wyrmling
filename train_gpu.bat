@echo off
cd /d %~dp0
if not exist out md out
python -u train.py --config s100m --data_dir data\gutenberg --block_size 160 --batch_size 10 --max_minutes 45 --eval_interval 25 --eval_iters 3 --warmup 50 --lr 1e-3 --out_dir out > out\run.log 2>&1
