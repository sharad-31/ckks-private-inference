"""Step 11: train the plaintext models.
 LR  : logistic regression on the IMG*IMG pixels
 CNN : Conv2d(1, CHANNELS, KERNEL, stride) -> square -> FC   (all CKKS friendly)
"""
import json, os, numpy as np, torch, torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
import config as C

torch.manual_seed(C.SEED); np.random.seed(C.SEED)
dev = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")      # DEVICE=cpu forces CPU
if dev.startswith("cuda") and not torch.cuda.is_available():
    raise SystemExit("CUDA requested but this PyTorch build has no CUDA. See README 'GPU' (install the CUDA build) or set DEVICE=cpu")
if dev.startswith("cuda"):
    torch.backends.cudnn.benchmark = True
    print(f"training on GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: training on CPU (on Windows 'pip install torch' gives the CPU-only build - see README 'GPU')")


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, C.CHANNELS, C.KERNEL, stride=C.STRIDE)
        self.fc = nn.Linear(C.CHANNELS * C.N_WINDOWS, 1)

    def forward(self, x):
        h = self.conv(x.unsqueeze(1))          # (B, ch, SIDE, SIDE)
        h = (h * h).flatten(1)                 # square activation; order = channel, window row, window col
        return self.fc(h).squeeze(1)


def predict(model, X):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), 2048):
            out.append(model(torch.tensor(X[i:i + 2048]).to(dev)).cpu().numpy())
    return np.concatenate(out)


def best_threshold(y, logit):
    p = 1 / (1 + np.exp(-logit)); ths = np.linspace(0.05, 0.95, 91)
    return float(ths[np.argmax([f1_score(y, p >= t) for t in ths])])


def main():
    z = np.load(C.PROC / f"images_{C.TAG}.npz", allow_pickle=True)
    X, y, split = z["X"], z["y"], z["split"]
    d = {k: split == k for k in ("train", "val", "test")}
    print(f"{C.TAG}: images {X.shape} | train/val/test {d['train'].sum()}/{d['val'].sum()}/{d['test'].sum()} "
          f"| positives {y.mean():.3f} | CNN windows {C.N_WINDOWS} x {C.KERNEL}x{C.KERNEL}")
    Xf = X.reshape(len(X), -1)

    # ---------------- logistic regression ----------------
    best = (-1, None, None)
    for Cc in (0.001, 0.01, 0.1, 1.0, 10.0):
        clf = LogisticRegression(C=Cc, max_iter=3000).fit(Xf[d["train"]], y[d["train"]])
        auc = roc_auc_score(y[d["val"]], clf.decision_function(Xf[d["val"]]))
        print(f"  LR C={Cc:<6} val AUROC {auc:.4f}")
        if auc > best[0]: best = (auc, Cc, clf)
    lr_val, lr_C, clf = best
    lr_logit = clf.decision_function(Xf)
    lr_thr = best_threshold(y[d["val"]], lr_logit[d["val"]])
    lr_test = roc_auc_score(y[d["test"]], lr_logit[d["test"]])
    lr_acc = float(((lr_logit[d["test"]] >= np.log(lr_thr / (1 - lr_thr))) == y[d["test"]]).mean())
    print(f"LR   val AUROC {lr_val:.4f} test AUROC {lr_test:.4f} test acc {lr_acc:.4f} (C={lr_C}, thr {lr_thr:.2f})")
    json.dump(dict(w=clf.coef_[0].tolist(), b=float(clf.intercept_[0]), threshold=lr_thr, C=lr_C),
              open(C.MODELS / f"lr_{C.TAG}.json", "w"))

    # ---------------- CNN ----------------
    model = CNN().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=C.LR, weight_decay=C.WD)
    lossf = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X[d["train"]]).to(dev)                  # whole training set lives on the device
    yt = torch.tensor(y[d["train"]].astype(np.float32)).to(dev)
    best_auc, best_state, bad = 0, None, 0
    for ep in range(C.EPOCHS):
        model.train(); perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), C.BATCH):
            b = perm[i:i + C.BATCH]
            opt.zero_grad()
            loss = lossf(model(Xt[b]), yt[b])
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        auc = roc_auc_score(y[d["val"]], predict(model, X[d["val"]]))
        if auc > best_auc:
            best_auc, bad = auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep % 5 == 0: print(f"  epoch {ep:3d}  val AUROC {auc:.4f}")
        if bad >= C.PATIENCE: break
    model.load_state_dict(best_state)
    cnn_logit = predict(model, X)
    thr = best_threshold(y[d["val"]], cnn_logit[d["val"]])
    cnn_test = roc_auc_score(y[d["test"]], cnn_logit[d["test"]])
    cnn_acc = float(((cnn_logit[d["test"]] >= np.log(thr / (1 - thr))) == y[d["test"]]).mean())
    print(f"CNN  val AUROC {best_auc:.4f} test AUROC {cnn_test:.4f} test acc {cnn_acc:.4f} (thr {thr:.2f})")

    sd = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
    json.dump(dict(conv_w=sd["conv.weight"][:, 0].tolist(), conv_b=sd["conv.bias"].tolist(),
                   fc_w=sd["fc.weight"].reshape(C.CHANNELS, C.N_WINDOWS).tolist(), bias=float(sd["fc.bias"][0]),
                   threshold=thr, kernel=[C.KERNEL, C.KERNEL], stride=C.STRIDE, channels=C.CHANNELS,
                   n_windows=C.N_WINDOWS, img=C.IMG), open(C.MODELS / f"cnn_{C.TAG}.json", "w"))
    np.savez(C.MODELS / f"eval_{C.TAG}.npz", X=X, y=y, split=split, sid=np.arange(len(y)),
             logit_cnn=cnn_logit, logit_lr=lr_logit)
    json.dump(dict(tag=C.TAG, dataset=C.DATASET, img=C.IMG, n_train=int(d["train"].sum()), n_test=int(d["test"].sum()),
                   positive_rate=float(y.mean()), lr_val_auroc=float(lr_val), lr_test_auroc=float(lr_test),
                   lr_test_acc=lr_acc, lr_threshold=lr_thr, cnn_val_auroc=float(best_auc), cnn_test_auroc=float(cnn_test),
                   cnn_test_acc=cnn_acc, cnn_test_auprc=float(average_precision_score(y[d["test"]], cnn_logit[d["test"]])),
                   cnn_threshold=thr, cnn_params=int(sum(p.numel() for p in model.parameters())), epochs_run=ep + 1),
              open(C.REPORT / f"train_{C.TAG}.json", "w"), indent=2)
    print("saved models + whole-dataset eval file")


if __name__ == "__main__":
    main()
