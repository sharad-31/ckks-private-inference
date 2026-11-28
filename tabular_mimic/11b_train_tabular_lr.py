"""Step 11b: train Logistic Regression on FLATTENED grid + static features.

This gives an LR that sees the SAME input as the CNN (time-series grid + static),
so the LR vs CNN comparison under CKKS is a fair model-type comparison.

Usage:
  python 11b_train_tabular_lr.py --dataset mimic4 --grid 12x10
  python 11b_train_tabular_lr.py --dataset mimic3 --grid 12x5
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, f1_score, average_precision_score,
                             accuracy_score, precision_score, recall_score)

# Add project root to sys.path
ROOT_REPO = Path(__file__).resolve().parent.parent
if str(ROOT_REPO) not in sys.path:
    sys.path.insert(0, str(ROOT_REPO))

import config as C
from common.utils import set_seed


def load():
    ts_file = C.PROC / f"ts_{C.TAG}.npz"
    if not ts_file.exists():
        raise FileNotFoundError(
            f"Processed time-series file not found: {ts_file}\n"
            f"Please run Step 10 first: python 10_extract_timeseries.py --dataset {C.DATASET} --grid {C.N_BINS}x{C.N_VITALS}"
        )
    z = np.load(ts_file, allow_pickle=True)
    X, y, sid, split = z["X"], z["y"], z["stay_id"], z["split"]
    if not C.STATIC_CSV.exists():
        raise FileNotFoundError(f"Static features CSV not found: {C.STATIC_CSV}")
    st = pd.read_csv(C.STATIC_CSV).set_index("stay_id")
    st = st.drop(columns=[c for c in C.STATIC_DROP if c in st.columns]).select_dtypes("number")
    keep = np.isin(sid, st.index)
    print(f"stays with static features: {keep.sum()}/{len(keep)}")
    X, y, sid, split = X[keep], y[keep], sid[keep], split[keep]
    S = st.loc[sid].values.astype(np.float32)
    tr = split == "train"
    mu_s, sd_s = np.nanmean(S[tr], 0), np.nanstd(S[tr], 0) + 1e-6
    S = np.nan_to_num((S - mu_s) / sd_s)
    return X, S, y, sid, split, list(st.columns)


def best_threshold(y, logit):
    p = 1 / (1 + np.exp(-logit))
    ths = np.linspace(0.05, 0.95, 91)
    return float(ths[np.argmax([f1_score(y, p >= t) for t in ths])])


def full_metrics(y, logit, threshold):
    """Return dict of acc, precision, recall, F1, AUROC, AUPRC at the given threshold."""
    prob = 1 / (1 + np.exp(-logit))
    pred = (prob >= threshold).astype(int)
    return dict(
        auroc=float(roc_auc_score(y, logit)),
        auprc=float(average_precision_score(y, logit)),
        acc=float(accuracy_score(y, pred)),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        f1=float(f1_score(y, pred, zero_division=0)),
    )


def main():
    ap = argparse.ArgumentParser(description="Train LR on flattened grid + static (fair LR vs CNN)")
    ap.add_argument("--dataset", choices=["mimic3", "mimic4"], default=C.DATASET)
    ap.add_argument("--vitals", type=int, choices=[5, 10], default=C.N_VITALS)
    ap.add_argument("--grid", default=None, help="Grid shorthand e.g. 12x10 or 12x5")
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()
    if args.grid:
        args.vitals = int(args.grid.split("x")[1])

    # Apply configuration
    C.DATASET = args.dataset
    C.N_VITALS = args.vitals
    C.GRID = (C.N_BINS, C.N_VITALS)
    C.KERNEL_W = C.N_VITALS
    C.TAG = f"{C.DATASET}_{C.N_BINS}x{C.N_VITALS}_{C.TARGET}"
    C.STATIC_CSV = C.PROC / f"static_builtin_{C.DATASET}.csv"
    set_seed(args.seed)

    X, S, y, sid, split, scols = load()
    d = {k: split == k for k in ("train", "val", "test")}

    # Flatten grid and concatenate with static features
    Xf = X.reshape(len(X), -1)                               # (n, 12*N_VITALS)
    F = np.hstack([Xf, S]).astype(np.float32)                # (n, 12*N_VITALS + n_static)
    n_grid = Xf.shape[1]
    n_static = S.shape[1]
    n_feat = F.shape[1]
    print(f"grid {C.GRID} | grid features {n_grid} | static features {n_static} | total {n_feat}")
    print(f"train/val/test {d['train'].sum()}/{d['val'].sum()}/{d['test'].sum()} | pos {y.mean():.3f}")

    # Tune regularisation on validation
    best = (-1, None, None)
    for Cc in (0.001, 0.01, 0.1, 1.0, 10.0):
        clf = LogisticRegression(C=Cc, max_iter=3000, solver="lbfgs").fit(F[d["train"]], y[d["train"]])
        auc = roc_auc_score(y[d["val"]], clf.decision_function(F[d["val"]]))
        print(f"  LR C={Cc:<6} val AUROC {auc:.4f}")
        if auc > best[0]:
            best = (auc, Cc, clf)
    lr_val, lr_C, clf = best
    lr_logit = clf.decision_function(F)                       # logits for entire dataset
    thr = best_threshold(y[d["val"]], lr_logit[d["val"]])

    # Full metrics on test split
    test_m = full_metrics(y[d["test"]], lr_logit[d["test"]], thr)
    val_m  = full_metrics(y[d["val"]],  lr_logit[d["val"]],  thr)
    print(f"LR-full  val AUROC {lr_val:.4f}  test AUROC {test_m['auroc']:.4f}  "
          f"test acc {test_m['acc']:.4f}  F1 {test_m['f1']:.4f}  (C={lr_C}, thr {thr:.2f})")

    # Save weights for CKKS (simple dot product: z = w·x + b)
    w = clf.coef_[0].tolist()                                 # length = n_feat
    b = float(clf.intercept_[0])
    json.dump(dict(
        w=w, b=b, threshold=thr, C_reg=lr_C,
        n_features=n_feat, n_grid=n_grid, n_static=n_static,
        grid=list(C.GRID), static_columns=scols,
    ), open(C.MODELS / f"lr_full_{C.TAG}.json", "w"))

    # Save evaluation predictions (same layout as CNN eval file)
    np.savez(C.MODELS / f"eval_lr_{C.TAG}.npz",
             F=F, y=y, logit=lr_logit, split=split, stay_id=sid)

    # Save training report
    report = dict(
        tag=C.TAG, dataset=C.DATASET, grid=list(C.GRID), target=C.TARGET,
        model="lr_full", n_features=n_feat, n_grid=n_grid, n_static=n_static,
        n_train=int(d["train"].sum()), n_val=int(d["val"].sum()), n_test=int(d["test"].sum()),
        positive_rate=float(y.mean()), C_reg=lr_C, threshold=thr,
        val=val_m, test=test_m,
    )
    json.dump(report, open(C.REPORT / f"train_lr_{C.TAG}.json", "w"), indent=2)
    print(f"saved lr_full_{C.TAG}.json, eval_lr_{C.TAG}.npz, train_lr_{C.TAG}.json")


if __name__ == "__main__":
    main()
