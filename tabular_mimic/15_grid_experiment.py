"""Step 15 - does a bigger CNN grid (more time bins, more variables) improve the result?

For every combination of --bins and --vars it runs  10_extract_timeseries.py  and  11_train_cnn.py  (plaintext training, a few
minutes on a GPU) and prints one table. Nothing encrypted happens here: run step 12 only for the grid you decide to keep.

  grid          = time bins x variables       12 x 5 (basic) is the original CNN input, 12 x 10 (extended) adds sbp, dbp and the
                  3 GCS parts, 12 x 12 (full) adds lactate and creatinine from the labs
  slots needed  = (bins - kernel_h + 1) * kernel_h * variables      what the CKKS circuit has to pack into one ciphertext:
                  <= 4096 -> preset 'standard' works, <= 8192 -> only 'precise', more -> does not fit
  The kernel always covers 12 hours (kernel_h = bins / 4), so the receptive field stays the same when the bins get finer.

Usage (Windows: no '# comments' on the command line):
    python 15_grid_experiment.py
    python 15_grid_experiment.py --target mortality
    python 15_grid_experiment.py --bins 12 24 --vars basic extended full --channels 4 8
    python 15_grid_experiment.py --skip-extract          reuse the time-series files of an earlier run

Afterwards, for the best grid  (example: 24 bins, extended):
    python 12_ckks_cnn_inference.py --ts-name ts_g24_extended --models models_cnn_g24_extended --reports reports_cnn_g24_extended --n 100
    python 12_ckks_cnn_inference.py --ts-name ts_g24_extended --models models_cnn_g24_extended --reports reports_cnn_g24_extended --workers 4
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/processed")
ap.add_argument("--bins", type=int, nargs="+", default=[12], help="default: 12 bins only (12 x 5, 12 x 10, 12 x 12)")
ap.add_argument("--vars", choices=["basic", "extended", "full"], nargs="+", default=["basic", "extended", "full"])
ap.add_argument("--channels", type=int, nargs="+", default=[4])
ap.add_argument("--target", choices=["los7", "mortality"], default="los7")
ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
ap.add_argument("--skip-extract", action="store_true", help="do not run step 10 again if the time-series file exists")
ap.add_argument("--out", default="reports_grid")
args = ap.parse_args()
Path(args.out).mkdir(exist_ok=True)
N_VARS = {"basic": 5, "extended": 10, "full": 12}


LOGS = Path(args.out) / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
# Windows decodes a child's output with cp1252 by default and breaks on the progress-bar characters: force UTF-8
CHILD_ENV = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")


def run(cmd, label, log_name):
    print(f"  {label} ...", flush=True)
    r = subprocess.run([sys.executable] + cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=CHILD_ENV)
    out, err = r.stdout or "", r.stderr or ""
    (LOGS / f"{log_name}.log").write_text(out + "\n----- stderr -----\n" + err, encoding="utf-8")      # full output of the step
    if r.returncode != 0:
        print("    FAILED (full log: " + str(LOGS / f"{log_name}.log") + "):\n      " + "\n      ".join((err or out).strip().splitlines()[-6:]))
    return r.returncode == 0


rows = []
for bins in args.bins:
    for var in args.vars:
        name = f"ts_g{bins}_{var}"
        kh = max(3, bins // 4)
        L, V = bins - kh + 1, N_VARS[var]
        slots = L * kh * V
        preset = "standard" if slots <= 4096 else ("precise" if slots <= 8192 else "does not fit")
        print(f"\nGrid {bins} bins x {V} variables, kernel {kh}, slots needed {slots} ({preset})")
        if not (args.skip_extract and (Path(args.data) / f"{name}.npz").exists()):
            if not run(["10_extract_timeseries.py", "--bins", str(bins), "--vars", var, "--name", name, "--data", args.data],
                       "extracting the time series", f"{name}_extract"):
                continue
        for ch in args.channels:
            models = f"models_cnn_g{bins}_{var}" + (f"_c{ch}" if len(args.channels) > 1 else "")
            if not run(["11_train_cnn.py", "--ts-name", name, "--models", models, "--target", args.target, "--channels", str(ch),
                        "--kernel-h", str(kh), "--device", args.device, "--data", args.data], f"training the CNN ({ch} channels)",
                       f"{models}_train"):
                continue
            m = json.loads((Path(models) / "metrics_cnn.json").read_text())
            rows.append(dict(grid=f"{bins} x {V}", kernel_h=kh, channels=ch, slots_needed=slots, ckks_preset=preset,
                             val_auroc=m["val_auroc_cnn"], val_accuracy=m["val_accuracy_cnn"], val_auprc=m["val_auprc_cnn"],
                             lr_static_auroc=m["val_auroc_lr_static"], best_epoch=m["best_epoch"], models_dir=models, ts_name=name))

if rows:
    tbl = pd.DataFrame(rows)
    tbl.to_csv(Path(args.out) / "grid_results.csv", index=False)
    print(f"\nTarget: {args.target}.  'lr_static_auroc' = logistic regression on the static features only (reference, same for all rows).")
    print(tbl.drop(columns=["models_dir", "ts_name"]).round(4).to_string(index=False))
    best = tbl.sort_values("val_auroc", ascending=False).iloc[0]
    print(f"\nBest validation AUROC: grid {best.grid}, kernel {best.kernel_h}, {best.channels} channels -> {best.val_auroc:.4f} "
          f"(accuracy {best.val_accuracy:.4f}, CKKS preset: {best.ckks_preset})")
    print(f"Saved {Path(args.out) / 'grid_results.csv'}")
else:
    print("\nNo grid finished: read the FAILED messages above.")
