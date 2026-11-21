"""Step 12: encrypted inference of the hybrid CNN with CKKS (TenSEAL).
usage:  python 12_ckks_cnn_inference.py --preset precise --workers 8 [--n 200]
"""
import argparse
import json
import sys
import time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import tenseal as ts

# Add project root to sys.path
ROOT_REPO = Path(__file__).resolve().parent.parent
if str(ROOT_REPO) not in sys.path:
    sys.path.insert(0, str(ROOT_REPO))

import config as C
from common.ckks_context import make_ckks_context
from common.metrics import evaluate_classification_fidelity, compute_split_metrics

W = None
CTX = None


def init_worker(preset, weights):
    global W, CTX
    W = weights
    CTX = make_ckks_context(preset, galois=True)


def infer(args):
    x, s = args                                   # x: (12, N_VITALS), s: (static_dim,)
    kh, kw = W["kernel"]
    # --- client: im2col (all sliding windows in one vector) + encrypt ---
    # TenSEAL reads the matrix transposed, so kernel dims are passed as (kw, kh);
    # the kernel itself is passed as the normal (kh, kw) matrix.
    enc_x, wn = ts.im2col_encoding(CTX, x.tolist(), kw, kh, 1)
    assert wn == W["n_windows"], f"unexpected window count {wn}"
    enc_s = ts.ckks_vector(CTX, s.tolist())
    # --- server: only ciphertext ops ---
    acc = None
    for c in range(W["channels"]):
        y = enc_x.conv2d_im2col(W["conv_w"][c], wn) + W["conv_b"][c]
        y = y.square()
        y = y.dot(W["fc_conv_w"][c])
        acc = y if acc is None else acc + y
    out = acc + enc_s.dot(W["static_w"]) + W["bias"]
    # --- client: decrypt logit (sigmoid applied in clear, exact) ---
    return out.decrypt()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="precise", choices=list(C.PRESETS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--split", default="all", choices=["all", "train", "val", "test"],
                    help="all = the WHOLE cohort (default)")
    ap.add_argument("--n", type=int, default=0, help="debug: only first n patients (0 = no limit)")
    ap.add_argument("--chunk", type=int, default=2000, help="patients per saved chunk (resume point)")
    a = ap.parse_args()

    W_ = json.load(open(C.MODELS / f"cnn_{C.TAG}.json"))
    assert tuple(W_["grid"]) == C.GRID, "model trained with a different grid - rerun 10 & 11"
    t = np.load(C.MODELS / f"eval_{C.TAG}.npz", allow_pickle=True)
    X, S, y, lp, split, sid = (t[k] for k in ("X", "S", "y", "logit", "split", "stay_id"))
    idx = np.arange(len(y)) if a.split == "all" else np.where(split == a.split)[0]
    if a.n:
        idx = idx[:a.n]
    X, S, y, lp, split, sid = X[idx], S[idx], y[idx], lp[idx], split[idx], sid[idx]

    slots = C.PRESETS[a.preset]["poly"] // 2
    need = C.N_WINDOWS * C.KERNEL_H * C.KERNEL_W
    print(f"grid {C.GRID} | windows {C.N_WINDOWS} x {C.KERNEL_H*C.KERNEL_W} = "
          f"{need} slots used of {slots} | preset {a.preset} | split {a.split}: {len(y)} patients")
    assert need <= slots, "grid too big for this preset"

    # chunked + resumable: finished chunks are kept in reports/cache_*, rerun continues
    cache = C.REPORT / f"cache_{C.TAG}_{a.preset}_{a.split}_{len(y)}"
    cache.mkdir(exist_ok=True)
    t0 = time.time()
    done_now = 0
    with Pool(a.workers, initializer=init_worker, initargs=(a.preset, W_)) as p:
        for ci, s0 in enumerate(range(0, len(y), a.chunk)):
            f = cache / f"chunk_{ci:05d}.npy"
            if f.exists():
                continue
            sl = slice(s0, s0 + a.chunk)
            np.save(f, np.array(p.map(infer, list(zip(X[sl], S[sl])), chunksize=8)))
            done_now += len(X[sl])
            el = time.time() - t0
            print(f"  {min(s0 + a.chunk, len(y))}/{len(y)} patients | {el/60:.1f} min elapsed", flush=True)
    dt = time.time() - t0
    le = np.concatenate([np.load(cache / f"chunk_{i:05d}.npy") for i in range(-(-len(y) // a.chunk))])

    thr = W_["threshold"]
    metrics_summary = compute_split_metrics(y, lp, le, split, threshold=thr)
    overall = metrics_summary["overall"]

    res = dict(
        tag=C.TAG,
        dataset=C.DATASET,
        grid=list(C.GRID),
        preset=a.preset,
        split=a.split,
        threshold=thr,
        n=len(y),
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
        per_split=metrics_summary["per_split"]
    )
    out_json = C.REPORT / f"ckks_{C.TAG}_{a.preset}_{a.split}.json"
    json.dump(res, open(out_json, "w"), indent=2)
    print(f"\nwrote {out_json}")
    print(f"  AUROC  plain {overall['auroc_plain']:.4f} -> enc {overall['auroc_enc']:.4f}")
    print(f"  acc    plain {overall['acc_plain']:.4f} -> enc {overall['acc_enc']:.4f}")
    print(f"  logit error  mean {overall['logit_err_mean']:.2e}  max {overall['logit_err_max']:.2e}")
    print(f"  flips  {overall['class_flips']}/{len(y)} ({overall['flip_rate']*100:.3f}%)")

    # save per-stay predictions csv
    pp = 1.0 / (1.0 + np.exp(-lp))
    pe = 1.0 / (1.0 + np.exp(-le))
    pred = pd.DataFrame(dict(
        stay_id=sid, split=split, y=y,
        prob_plain=pp, prob_enc=pe,
        logit_plain=lp, logit_enc=le,
        flip=(pp >= thr) != (pe >= thr)
    ))
    out_csv = C.REPORT / f"predictions_{C.TAG}_{a.preset}_{a.split}.csv"
    pred.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
