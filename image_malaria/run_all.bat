@echo off
REM full pipeline (about 2-3 hours with 8 workers). Put the Parasitized and Uninfected folders in data\raw\malaria first.
python 10_prepare_data.py
python 11_train.py
python 12_ckks_inference.py --model cnn --preset precise  --workers 8
python 12_ckks_inference.py --model cnn --preset standard --workers 8
python 12_ckks_inference.py --model lr --poly 0 --preset precise  --workers 8
python 12_ckks_inference.py --model lr --poly 3 --preset standard --workers 8
python 12_ckks_inference.py --model lr --poly 5 --preset precise  --workers 8
python 15_ckks_batch.py --model cnn --preset standard --workers 2
python 15_ckks_batch.py --model lr --poly 3 --preset standard --workers 2
python 15_ckks_batch.py --model cnn --preset deep --decision --workers 2
python 14_ckks_threshold.py --model cnn --preset deep --workers 8 --split test
python 14_ckks_threshold.py --model lr  --preset deep --workers 8 --split test
python 13_make_summary.py
