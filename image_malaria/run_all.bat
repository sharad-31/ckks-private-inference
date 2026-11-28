@echo off
REM ============================================================
REM Full pipeline: ~3-4 hours with 8 workers.
REM Put the Parasitized and Uninfected folders in data\raw\malaria first.
REM ============================================================

REM --- Step 10: prepare data ---
python 10_prepare_data.py

REM --- Step 11: train plaintext LR + CNN ---
python 11_train.py

REM ============================================================
REM  C2: Per-patient encrypted inference (WITHOUT batching)
REM ============================================================

REM CNN - both presets
python 12_ckks_inference.py --model cnn --preset precise  --workers 8
python 12_ckks_inference.py --model cnn --preset standard --workers 8

REM LR exact (no sigmoid polynomial) - both presets
python 12_ckks_inference.py --model lr --poly 0 --preset precise  --workers 8
python 12_ckks_inference.py --model lr --poly 0 --preset standard --workers 8

REM LR polynomial sigmoid approximations
python 12_ckks_inference.py --model lr --poly 3 --preset standard --workers 8
python 12_ckks_inference.py --model lr --poly 5 --preset precise  --workers 8

REM ============================================================
REM  C3: SIMD batching (WITH batching)
REM ============================================================

REM CNN batched - standard
python 15_ckks_batch.py --model cnn --preset standard --workers 2

REM LR batched - exact (no poly) - standard and precise
python 15_ckks_batch.py --model lr --poly 0 --preset standard --workers 2
python 15_ckks_batch.py --model lr --poly 0 --preset precise  --workers 2

REM LR batched - poly3 standard
python 15_ckks_batch.py --model lr --poly 3 --preset standard --workers 2

REM CNN batched - deep (more levels, for decision)
python 15_ckks_batch.py --model cnn --preset deep --workers 2

REM CNN batched + encrypted threshold decision
python 15_ckks_batch.py --model cnn --preset deep --decision --workers 2
python 15_ckks_batch.py --model cnn --preset deep --decision --scale-pct 99 --workers 2
python 15_ckks_batch.py --model cnn --preset deep --decision --scale-pct 95 --workers 2

REM ============================================================
REM  Encrypted threshold decision (per-patient, step 14)
REM ============================================================
python 14_ckks_threshold.py --model cnn --preset deep --workers 8 --split test
python 14_ckks_threshold.py --model lr  --preset deep --workers 8 --split test

REM --- Step 13: summary ---
python 13_make_summary.py

echo.
echo === ALL DONE ===
