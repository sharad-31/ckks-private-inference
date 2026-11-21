# CKKS private inference on images - NIH Malaria cell images (parasitized vs uninfected)

Same pipeline as the MIMIC project, for an image dataset: plaintext LR (pixels) and CNN, encrypted inference with
CKKS (TenSEAL), SIMD batching and encrypted threshold decision, one summary.

## Dataset
NIH Malaria Cell Images: 27,558 segmented red-blood-cell images (13,779 parasitized, 13,779 uninfected).
- Kaggle: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria  (`pip install kagglehub`,
  `kagglehub.dataset_download('iarunava/cell-images-for-detecting-malaria')`)
- NIH: https://lhncbc.nlm.nih.gov/LHC-downloads/dataset.html (Malaria datasets section, cell images zip)

Put the two class folders anywhere under `data/raw/malaria/` (nested folders such as `cell_images/cell_images/...`
are found automatically):
```
data/raw/malaria/.../Parasitized/*.png
data/raw/malaria/.../Uninfected/*.png
```

## Steps
| step | script | what it does |
|---|---|---|
| 10 | `10_prepare_data.py` | reads images, resizes to 28x28 grayscale, **patient-level split 70/15/15** (patient/slide id = file name before `_IMG`, no patient in two splits), `reports/cohort_*.json`, `reports/samples_*.png` |
| 11 | `11_train.py` | LR on 784 pixels (C chosen on validation) and CNN `Conv 4ch 7x7 stride 3 -> square -> FC`; saves weights, whole-dataset eval file, `reports/train_*.json` |
| 12 | `12_ckks_inference.py` | encrypted inference patient by patient on the **whole dataset**: `--model cnn|lr`, `--poly 0|3|5` (LR sigmoid polynomial), `--preset standard|precise`, `--split all|test|...`, `--n` debug, resumable chunks |
| 14 | `14_ckks_threshold.py` | novelty: threshold decision under encryption (sign approximation), client only gets the class; `--preset deep|deeper` |
| 15 | `15_ckks_batch.py` | novelty: SIMD batching, 4096 patients per ciphertext slot-vector, no rotations; `--decision` adds the encrypted threshold |
| 13 | `13_make_summary.py` | `reports/SUMMARY.md` and `summary.csv` with every result |

`run_all.bat` / `run_all.sh` run everything (about 2-3 hours with 8 workers).

## Notes
- CNN geometry: 28x28 image, 7x7 kernel, stride 3 -> 8x8 = 64 windows x 64 (49 padded) = 4096 slots = one `standard` ciphertext. Multiplicative depth 3 (conv, square, weighted sum).
- LR: depth 1 (dot product); polynomial sigmoid degree 3 needs 3 levels (`standard`), degree 5 needs 4 levels (`precise`).
- TenSEAL reads the image transposed; `common.calibrate` finds the window order automatically at start of every worker, so the CNN weights always line up.
- AUROC over `--split all` includes training images; report model quality from the test split (`per_split -> test` in the json files, "Test-split rows" in the summary).
- Batching memory: a batch holds 28x28 = 784 ciphertexts; use `--workers 2` (the deep presets are larger still).
- Resume: if a run stops, start the same command again. Delete `reports/cache_*` for a fresh run.

## Notes added after the first full run (malaria)
- LR with a sigmoid polynomial: step 12 / 15 now also report `class_flips_calibrated` / `acc_enc_calibrated` = decision taken
  with the threshold moved into polynomial space (`poly(t_logit)`), which is what a deployed client would use.
  `class_flips` (raw threshold, exact-sigmoid reference) measures the approximation error of the polynomial itself.
- Encrypted threshold: `--scale-pct 99` (steps 14 and 15 --decision) scales the decision variable with the 99th percentile
  of |logit - t| instead of the maximum. More patients saturate to +-1 (less score leakage), but patients with |x| > 1.7
  can overflow the polynomial and get a wrong sign: the json reports `n_overflow`, `class_flips`, `saturated_fraction`.
- Other CNN sizes: `$env:CHANNELS="8"` (or KERNEL / STRIDE / IMG) writes results under a different tag, nothing is overwritten.

## GPU
- **Step 11 (training)** uses the GPU automatically if PyTorch has CUDA (`python check_gpu.py` tells you). On Windows
  `pip install torch` installs the CPU-only build: run `nvidia-smi`, then take the install command for your CUDA version from
  https://pytorch.org/get-started/locally/ (Windows, Pip) - it looks like
  `pip install torch --index-url https://download.pytorch.org/whl/cu128` - after `pip uninstall torch -y`.
  `DEVICE=cpu` forces the CPU.
- **Steps 12, 14, 15 (encryption)** use TenSEAL = Microsoft SEAL, which is CPU-only: there is no GPU switch for it. The
  speed levers there are `--workers` (physical cores) and step 15 (SIMD batching: whole dataset in about 2 minutes).
  GPU CKKS needs another library (e.g. Liberate.FHE, Linux + CUDA build only) and a rewrite of these steps.
