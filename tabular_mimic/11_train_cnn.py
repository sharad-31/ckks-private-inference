"""Step 11: train hybrid CNN (grid conv -> square -> FC  +  static LR branch)."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from sklearn.linear_model import LogisticRegression

# Add project root to sys.path
ROOT_REPO = Path(__file__).resolve().parent.parent
if str(ROOT_REPO) not in sys.path:
    sys.path.insert(0, str(ROOT_REPO))

import config as C
from common.utils import set_seed

dev = "cuda" if torch.cuda.is_available() else "cpu"


class HybridCNN(nn.Module):
    def __init__(self, n_static):
        super().__init__()
        self.conv = nn.Conv2d(1, C.CHANNELS, (C.KERNEL_H, C.KERNEL_W))
        self.fc_conv = nn.Linear(C.CHANNELS * C.N_WINDOWS, 1)   # = FC per channel, summed
        self.fc_static = nn.Linear(n_static, 1)

    def forward(self, x, s):
        h = self.conv(x.unsqueeze(1))          # (B, ch, 10, 1)
        h = (h * h).flatten(1)                 # square activation (CKKS friendly)
        return (self.fc_conv(h) + self.fc_static(s)).squeeze(1)


def load():
    ts_file = C.PROC / f"ts_{C.TAG}.npz"
    if not ts_file.exists():
        raise FileNotFoundError(
            f"Processed time-series file not found: {ts_file}\n"
            f"Please run Step 10 first: python 10_extract_timeseries.py --dataset {C.DATASET} --vitals {C.N_VITALS}"
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
    mu, sd = np.nanmean(S[tr], 0), np.nanstd(S[tr], 0) + 1e-6
    S = np.nan_to_num((S - mu) / sd)
    return X, S, y, sid, split, list(st.columns)


def predict(model, X, S):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), 2048):
            out.append(model(torch.tensor(X[i:i+2048]).to(dev),
                             torch.tensor(S[i:i+2048]).to(dev)).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description="Train plaintext hybrid CNN and static LR baseline on MIMIC")
    ap.add_argument("--dataset", choices=["mimic3", "mimic4"], default=C.DATASET, help="MIMIC dataset")
    ap.add_argument("--vitals", type=int, choices=[5, 10], default=C.N_VITALS, help="Vitals grid width")
    ap.add_argument("--grid", default=None, help="Grid shorthand e.g. 12x10 or 12x5 (overrides --vitals)")
    ap.add_argument("--epochs", type=int, default=C.EPOCHS, help="Maximum epochs")
    ap.add_argument("--batch", type=int, default=C.BATCH, help="Batch size")
    ap.add_argument("--lr", type=float, default=C.LR, help="Learning rate")
    ap.add_argument("--seed", type=int, default=C.SEED, help="Random seed")
    args = ap.parse_args()
    if args.grid:                                       # e.g. "12x10" -> vitals=10
        args.vitals = int(args.grid.split("x")[1])

    # Apply configuration
    C.DATASET = args.dataset
    C.N_VITALS = args.vitals
    C.GRID = (C.N_BINS, C.N_VITALS)
    C.KERNEL_W = C.N_VITALS
    C.TAG = f"{C.DATASET}_{C.N_BINS}x{C.N_VITALS}_{C.TARGET}"
    C.EPOCHS = args.epochs
    C.BATCH = args.batch
    C.LR = args.lr
    C.STATIC_CSV = C.PROC / f"static_builtin_{C.DATASET}.csv"
    set_seed(args.seed)

    X, S, y, sid, split, scols = load()
    d = {k: split == k for k in ("train", "val", "test")}
    print(f"grid {C.GRID} | static {S.shape[1]} | train/val/test "
          f"{d['train'].sum()}/{d['val'].sum()}/{d['test'].sum()} | pos {y.mean():.3f}")

    base = LogisticRegression(max_iter=2000).fit(S[d["train"]], y[d["train"]])
    lr_val = roc_auc_score(y[d["val"]], base.decision_function(S[d["val"]]))
    lr_test = roc_auc_score(y[d["test"]], base.decision_function(S[d["test"]]))
    print("static-only LR  val AUROC %.4f  test AUROC %.4f" % (lr_val, lr_test))

    model = HybridCNN(S.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=C.LR, weight_decay=C.WD)
    pw = torch.tensor((1 - y[d["train"]].mean()) / y[d["train"]].mean(), device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    Xt, St, yt = (torch.tensor(a[d["train"]]) for a in (X, S, y.astype(np.float32)))

    best, best_state, bad = 0, None, 0
    for ep in range(C.EPOCHS):
        model.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), C.BATCH):
            b = perm[i:i + C.BATCH]
            opt.zero_grad()
            loss = lossf(model(Xt[b].to(dev), St[b].to(dev)), yt[b].to(dev))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        auc = roc_auc_score(y[d["val"]], predict(model, X[d["val"]], S[d["val"]]))
        if auc > best:
            best, bad = auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep % 5 == 0:
            print(f"  epoch {ep:3d}  val AUROC {auc:.4f}")
        if bad >= C.PATIENCE:
            break
    model.load_state_dict(best_state)

    lv = predict(model, X[d["val"]], S[d["val"]]); pv = 1 / (1 + np.exp(-lv))
    ths = np.linspace(0.05, 0.95, 91)
    thr = float(ths[np.argmax([f1_score(y[d["val"]], pv >= t) for t in ths])])
    lt = predict(model, X[d["test"]], S[d["test"]])
    cnn_test = roc_auc_score(y[d["test"]], lt)
    print(f"CNN  val AUROC {best:.4f}  test AUROC {cnn_test:.4f}  threshold {thr:.2f}")
    
    # Save training report json
    json.dump(dict(tag=C.TAG, dataset=C.DATASET, grid=list(C.GRID), target=C.TARGET,
                   n_static=int(S.shape[1]), n_train=int(d["train"].sum()),
                   n_test=int(d["test"].sum()), positive_rate=float(y.mean()),
                   lr_val_auroc=float(lr_val), lr_test_auroc=float(lr_test),
                   cnn_val_auroc=float(best), cnn_test_auroc=float(cnn_test),
                   cnn_test_auprc=float(average_precision_score(y[d["test"]], lt)),
                   threshold=thr, epochs_run=ep + 1),
              open(C.REPORT / f"train_{C.TAG}.json", "w"), indent=2)

    sd = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
    w = dict(
        conv_w=sd["conv.weight"][:, 0].tolist(),            # (ch, 3, N_VITALS)
        conv_b=sd["conv.bias"].tolist(),
        fc_conv_w=sd["fc_conv.weight"].reshape(C.CHANNELS, C.N_WINDOWS).tolist(),
        static_w=sd["fc_static.weight"][0].tolist(),
        bias=float(sd["fc_conv.bias"] + sd["fc_static.bias"]),
        threshold=thr,
        grid=list(C.GRID),
        channels=C.CHANNELS,
        kernel=[C.KERNEL_H, C.KERNEL_W],
        n_windows=C.N_WINDOWS,
        static_columns=scols
    )
    json.dump(w, open(C.MODELS / f"cnn_{C.TAG}.json", "w"), indent=2)

    # Save evaluation predictions npz for step 12
    all_logits = predict(model, X, S)
    np.savez(C.MODELS / f"eval_{C.TAG}.npz", X=X, S=S, y=y, logit=all_logits, split=split, stay_id=sid)
    print(f"saved model weights and evaluation set for {C.TAG}")


if __name__ == "__main__":
    main()
