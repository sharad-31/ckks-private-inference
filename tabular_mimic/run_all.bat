@echo off
REM full pipeline for both datasets and both grids, then one combined summary
setlocal enabledelayedexpansion
for %%D in (mimic4 mimic3) do (
  for %%N in (10 5) do (
    set MIMIC_DATASET=%%D
    set N_VITALS=%%N
    python 10_extract_timeseries.py
    python 11_train_cnn.py
    python 12_ckks_cnn_inference.py --preset precise  --workers 8
    python 12_ckks_cnn_inference.py --preset standard --workers 8
  )
)
python 13_make_summary.py
