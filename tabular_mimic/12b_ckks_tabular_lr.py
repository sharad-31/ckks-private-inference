"""Step 12b: CKKS encrypted inference for the full-feature LR (flattened grid + static).

The LR is a simple dot product: z = w·x + b, depth 1.
Both standard (3 levels) and precise (4 levels) have enough headroom.

Usage:
  python 12b_ckks_tabular_lr.py --dataset mimic4 --grid 12x10 --preset precise
  python 12b_ckks_tabular_lr.py --dataset mimic4 --grid 12x10 --preset standard
"""
import argparse
import json
import sys
import time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from sklearn.metrics import roc_auc_score

# Add project root to sys.path
ROOT_REPO = Path(__file__).resolve().parent.parent
if str(ROOT_REPO) not in sys.path:
    sys.path.insert(0, str(ROOT_REPO))

import config as C
from common.ckks_context import make_ckks_context
from common.metrics import compute_split_metrics

import tenseal as ts

W_GLOBAL = None
CTX_GLOBAL = None


def init_worker(preset, weights):
    global W_GLOBAL, CTX_GLOBAL
    W_GLOBAL = weights
    CTX_GLOBAL = make_ckks_context(preset, galois=True)    # dot() uses internal rotations


def infer(x):
    """Encrypt feature vector, compute dot product + bias, decrypt logit."""
    enc_x = ts.ckks_vector(CTX_GLOBAL, x.tolist())
    z = enc_x.dot(W_GLOBAL["w"]) + W_GLOBAL["b"]
    return z.decrypt()[0]


def main():
    ap = argparse.ArgumentParser(description="CKKS encrypted LR inference (tabular)")
    ap.add_argument("--dataset", choices=["mimic3", "mimic4"], default=C.DATASET)
    ap.add_argument("--vitals", type=int, choices=[5, 10], default=C.N_VITALS)
    ap.add_argument("--grid", default=None, help="Grid shorthand e.g. 12x10 or 12x5")
    ap.add_argument("--preset", default="precise", choices=list(C.PRESETS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    ap.add_argument("--n", type=int, default=0, help="debug: only first n patients (0 = all)")
    ap.add_argument("--chunk", type=int, default=2000, help="patients per saved chunk (resume point)")
    a = ap.parse_args()
    if a.grid:
        a.vitals = int(a.grid.split("x")[1])

    # Apply configuration
    C.DATASET = a.dataset
    C.N_VITALS = a.vitals
    C.GRID = (C.N_BINS, C.N_VITALS)
    C.KERNEL_W = C.N_VITALS
    C.TAG = f"{C.DATASET}_{C.N_BINS}x{C.N_VITALS}_{C.TARGET}"

    # Load LR weights
    wf = C.MODELS / f"lr_full_{C.TAG}.json"
    if not wf.exists():
        raise FileNotFoundError(
            f"LR weights not found: {wf}\n"
            f"Run step 11b first: python 11b_train_tabular_lr.py --dataset {C.DATASET} --grid {C.N_BINS}x{C.N_VITALS}"
        )
    W = json.load(open(wf))

    # Load evaluation data
    ef = C.MODELS / f"eval_lr_{C.TAG}.npz"
    if not ef.exists():
        raise FileNotFoundError(f"Evaluation file not found: {ef}\nRun step 11b first.")
    t = np.load(ef, allow_pickle=True)
    F, y, lp, split, sid = t["F"], t["y"], t["logit"], t["split"], t["stay_id"]

    # Subset
    idx = np.arange(len(y)) if a.split == "all" else np.where(split == a.split)[0]
    if a.n:
        idx = idx[:a.n]
    F, y, lp, split, sid = F[idx], y[idx], lp[idx], split[idx], sid[idx]

    slots = C.PRESETS[a.preset]["poly"] // 2
    n_feat = F.shape[1]
    print(f"LR-full | {C.TAG} | features {n_feat} | slots {slots} | "
          f"preset {a.preset} | split {a.split}: {len(y)} patients")
    assert n_feat <= slots, f"Feature dim {n_feat} exceeds slot count {slots}"

    # Chunked + resumable encrypted inference
    cache = C.REPORT / f"cache_lr_{C.TAG}_{a.preset}_{a.split}_{len(y)}"
    cache.mkdir(exist_ok=True)
    t0 = time.time()
    done_now = 0
    with Pool(a.workers, initializer=init_worker, initargs=(a.preset, W)) as p:
        for ci, s0 in enumerate(range(0, len(y), a.chunk)):
            f = cache / f"chunk_{ci:05d}.npy"
            if f.exists():
                continue
            sl = slice(s0, s0 + a.chunk)
            np.save(f, np.array(p.map(infer, list(F[sl]), chunksize=16)))
            done_now += len(F[sl])
            el = time.time() - t0
            print(f"  {min(s0 + a.chunk, len(y))}/{len(y)} patients | {el/60:.1f} min elapsed", flush=True)
    dt = time.time() - t0
    n_chunks = -(-len(y) // a.chunk)
    le = np.concatenate([np.load(cache / f"chunk_{i:05d}.npy") for i in range(n_chunks)])

    # Metrics
    thr = W["threshold"]
    metrics_summary = compute_split_metrics(y, lp, le, split, threshold=thr)
    overall = metrics_summary["overall"]

    res = dict(
        tag=C.TAG, dataset=C.DATASET, grid=list(C.GRID), model="lr_full",
        preset=a.preset, split=a.split, threshold=thr,
        n=len(y), n_features=n_feat,
        auroc_plain=overall["auroc_plain"],
        auroc_enc=overall["auroc_enc"],
        acc_plain=overall["acc_plain"],
        acc_enc=overall["acc_enc"],
        logit_err_mean=overall["logit_err_mean"],
        logit_err_max=overall["logit_err_max"],
        err_signed_mean=overall["err_signed_mean"],
        class_flips=overall["class_flips"],
        sec_per_patient_wall=float(dt / max(1, done_now)) if done_now else float(dt / len(y)),
        total_sec=float(dt),
        workers=a.workers,
        per_split=metrics_summary["per_split"],
    )
    out_json = C.REPORT / f"ckks_lr_{C.TAG}_{a.preset}_{a.split}.json"
    json.dump(res, open(out_json, "w"), indent=2)
    print(f"\nwrote {out_json}")
    print(f"  AUROC  plain {overall['auroc_plain']:.4f} -> enc {overall['auroc_enc']:.4f}")
    print(f"  acc    plain {overall['acc_plain']:.4f} -> enc {overall['acc_enc']:.4f}")
    print(f"  logit error  mean {overall['logit_err_mean']:.2e}  max {overall['logit_err_max']:.2e}")
    print(f"  flips  {overall['class_flips']}/{len(y)} ({overall['flip_rate']*100:.3f}%)")

    # Save per-stay predictions
    import pandas as pd
    pp = 1.0 / (1.0 + np.exp(-lp))
    pe = 1.0 / (1.0 + np.exp(-le))
    pred = pd.DataFrame(dict(
        stay_id=sid, split=split, y=y,
        prob_plain=pp, prob_enc=pe,
        logit_plain=lp, logit_enc=le,
        flip=(pp >= thr) != (pe >= thr),
    ))
    out_csv = C.REPORT / f"predictions_lr_{C.TAG}_{a.preset}_{a.split}.csv"
    pred.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
