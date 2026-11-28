@echo off
REM ============================================================
REM  FULL PIPELINE — Tabular (CNN + LR) + Image (all runs)
REM  Run from: C:\minor_project\ckks-private-inference
REM  Estimated time: ~5-6 hours total
REM ============================================================

echo ============================================================
echo  BLOCK 1: TABULAR — Extract + Train CNN + CKKS CNN
echo ============================================================

cd /d C:\minor_project\ckks-private-inference\tabular_mimic

python 10_extract_timeseries.py --dataset mimic4 --grid 12x10
python 10_extract_timeseries.py --dataset mimic4 --grid 12x5
python 10_extract_timeseries.py --dataset mimic3 --grid 12x10
python 10_extract_timeseries.py --dataset mimic3 --grid 12x5

python 11_train_cnn.py --dataset mimic4 --grid 12x10
python 11_train_cnn.py --dataset mimic4 --grid 12x5
python 11_train_cnn.py --dataset mimic3 --grid 12x10
python 11_train_cnn.py --dataset mimic3 --grid 12x5

python 12_ckks_cnn_inference.py --dataset mimic4 --grid 12x10 --preset precise  --split all --workers 8
python 12_ckks_cnn_inference.py --dataset mimic4 --grid 12x10 --preset standard --split all --workers 8
python 12_ckks_cnn_inference.py --dataset mimic4 --grid 12x5  --preset precise  --split all --workers 8
python 12_ckks_cnn_inference.py --dataset mimic4 --grid 12x5  --preset standard --split all --workers 8
python 12_ckks_cnn_inference.py --dataset mimic3 --grid 12x10 --preset precise  --split all --workers 8
python 12_ckks_cnn_inference.py --dataset mimic3 --grid 12x10 --preset standard --split all --workers 8
python 12_ckks_cnn_inference.py --dataset mimic3 --grid 12x5  --preset precise  --split all --workers 8
python 12_ckks_cnn_inference.py --dataset mimic3 --grid 12x5  --preset standard --split all --workers 8

echo ============================================================
echo  BLOCK 2: TABULAR — Train LR + CKKS LR
echo ============================================================

python 11b_train_tabular_lr.py --dataset mimic4 --grid 12x10
python 11b_train_tabular_lr.py --dataset mimic4 --grid 12x5
python 11b_train_tabular_lr.py --dataset mimic3 --grid 12x10
python 11b_train_tabular_lr.py --dataset mimic3 --grid 12x5

python 12b_ckks_tabular_lr.py --dataset mimic4 --grid 12x10 --preset precise  --split all --workers 8
python 12b_ckks_tabular_lr.py --dataset mimic4 --grid 12x10 --preset standard --split all --workers 8
python 12b_ckks_tabular_lr.py --dataset mimic4 --grid 12x5  --preset precise  --split all --workers 8
python 12b_ckks_tabular_lr.py --dataset mimic4 --grid 12x5  --preset standard --split all --workers 8
python 12b_ckks_tabular_lr.py --dataset mimic3 --grid 12x10 --preset precise  --split all --workers 8
python 12b_ckks_tabular_lr.py --dataset mimic3 --grid 12x10 --preset standard --split all --workers 8
python 12b_ckks_tabular_lr.py --dataset mimic3 --grid 12x5  --preset precise  --split all --workers 8
python 12b_ckks_tabular_lr.py --dataset mimic3 --grid 12x5  --preset standard --split all --workers 8

python 13_make_summary.py

echo ============================================================
echo  BLOCK 3: IMAGE — Full pipeline
echo ============================================================

cd /d C:\minor_project\ckks-private-inference\image_malaria

python 10_prepare_data.py
python 11_train.py

python 12_ckks_inference.py --model cnn --preset precise  --split all --workers 8
python 12_ckks_inference.py --model cnn --preset standard --split all --workers 8
python 12_ckks_inference.py --model lr --poly 0 --preset precise  --split all --workers 8
python 12_ckks_inference.py --model lr --poly 0 --preset standard --split all --workers 8
python 12_ckks_inference.py --model lr --poly 3 --preset standard --split all --workers 8
python 12_ckks_inference.py --model lr --poly 5 --preset precise  --split all --workers 8

python 15_ckks_batch.py --model cnn --preset standard --split all --workers 2
python 15_ckks_batch.py --model cnn --preset deep    --split all --workers 2
python 15_ckks_batch.py --model lr --poly 0 --preset standard --split all --workers 2
python 15_ckks_batch.py --model lr --poly 0 --preset precise  --split all --workers 2
python 15_ckks_batch.py --model lr --poly 3 --preset standard --split all --workers 2
python 15_ckks_batch.py --model cnn --preset deep --decision --split all --workers 2
python 15_ckks_batch.py --model cnn --preset deep --decision --scale-pct 99 --split all --workers 2
python 15_ckks_batch.py --model cnn --preset deep --decision --scale-pct 95 --split all --workers 2

python 14_ckks_threshold.py --model cnn --preset deep --split test --workers 8
python 14_ckks_threshold.py --model lr  --preset deep --split test --workers 8

python 13_make_summary.py

echo.
echo ============================================================
echo  ALL DONE
echo ============================================================
pause
