# CKKS private inference - hybrid CNN on MIMIC-III / MIMIC-IV

Model: time-series grid (12 time bins x N vitals) -> Conv2D (full-width kernel) -> square -> FC,
plus a static-feature logistic branch. Final logit is computed **under CKKS** (TenSEAL) and compared with plaintext.

## Switch dataset / grid (config.py, top 2 lines)
| setting | values |
|---|---|
| `DATASET`  | `"mimic4"` or `"mimic3"` |
| `N_VITALS` | `10` -> 12x10 grid, `5` -> 12x5 grid |

(or set env vars `MIMIC_DATASET`, `N_VITALS`; `run_all.*` does that in a loop).
Every output file name contains `<dataset>_<grid>_<target>`, so runs never overwrite each other.

## Pipeline
| step | script | output |
|---|---|---|
| 10 | `10_extract_timeseries.py` | cohort, 12xN grid (`data/processed/ts_*.npz`), built-in static features, `reports/cohort_*.json` |
| 11 | `11_train_cnn.py` | model + weights json + test set (`models/`), `reports/train_*.json` (static-LR baseline vs CNN) |
| 12 | `12_ckks_cnn_inference.py --preset precise|standard --workers 8 [--split all|test|val|train] [--chunk 2000]` | **runs on the WHOLE cohort by default** (`--split all`): `reports/ckks_*_all.json` (overall + per-split AUROC, accuracy, signed/abs logit error, flips), `reports/predictions_*.csv` (stay_id, split, y, plain logit, encrypted logit) |
| 13 | `13_make_summary.py` | `reports/SUMMARY.md`, `reports/summary.csv` (all runs in one table) |

Quick start: copy MIMIC csv files into `data/raw/mimic4/` (and/or `mimic3/`), `pip install -r requirements.txt`, then `run_all.bat` (Windows) or `./run_all.sh`.

## Notes
- Cohort: adults, first ICU stay per patient, ICU LOS >= 48h; target `los7` (LOS > 7 days) or `mortality` (`TARGET` in config). Split 70/15/15 by patient, seed 42.
- MIMIC-III age = year(intime) - year(dob) (ages > 89 are date-shifted, capped at 90). MIMIC-III chartevents rows with `error = 1` are dropped.
- Static features: built-in set (age, sex, admission type, first care unit, first-24h means of 5 vitals). To use your own 38 LR features, set `STATIC_CSV` in config.py (csv with a `stay_id` column).
- Vital itemids are in `config.VITAL_IDS` (CareVue + MetaVision for MIMIC-III). Verify them once against `d_items`.
- CKKS: `standard` = N 8192 (fast, ~1e-3 logit error), `precise` = N 16384 (~1e-5). Depth used: 3.

## Full-dataset encrypted inference (step 12)
- Default is `--split all`: every patient of the cohort is encrypted (MIMIC-IV 35,794 stays, MIMIC-III 22,536 stays), not only the test split. Use `--split test` for the held-out patients only; `--n 200` is a quick debug run.
- AUROC over `all` includes the patients the CNN was trained on, so report the **test** numbers from the `per_split` block for model quality and the **all** numbers for encryption fidelity (error, flips).
- Long runs are chunked (`--chunk`, default 2000 patients). If a run stops, start the same command again: finished chunks in `reports/cache_*` are reused. Delete that folder for a fresh run. Each chunk is encrypted with a fresh key, so flip counts can differ slightly between runs.
- Measured wall time (8 workers, per patient): precise 0.13-0.18 s, standard 0.04-0.05 s -> one MIMIC-IV grid takes about 86 min (precise) + 30 min (standard); MIMIC-III about 66 + 20 min. `run_all` (2 datasets x 2 grids x 2 presets) is roughly 6-7 hours.
