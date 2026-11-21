"""Step 12: encrypted inference on the WHOLE dataset (patient by patient).
  python 12_ckks_inference.py --model cnn --preset precise --workers 8
  python 12_ckks_inference.py --model lr --poly 0 --preset standard      (exact sigmoid on the client)
  python 12_ckks_inference.py --model lr --poly 3 --preset standard      (sigmoid polynomial degree 3, 3 levels)
  python 12_ckks_inference.py --model lr --poly 5 --preset precise       (degree 5, 4 levels)
Options: --split all|train|val|test (default all), --n 200 (debug), --chunk 2000 (resume point).
"""
import argparse, json, time, numpy as np
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score
import config as C
import common as K

G = {}


def init_worker(model, preset, W, poly_coefs):
    ctx = K.make_ctx(preset, galois=True)
    G.update(model=model, ctx=ctx, W=W, a=poly_coefs)
    if model == "cnn":
        G["st"] = K.cnn_init(ctx, W)


def infer(x):
    ctx, W, a = G["ctx"], G["W"], G["a"]
    if G["model"] == "cnn":
        z = K.cnn_forward(ctx, G["st"], x)
    else:
        z = K.lr_forward(ctx, W["w"], W["b"], x)
        if a:
            z = K.poly_sigmoid(z, a)
    return z.decrypt()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cnn", choices=["cnn", "lr"])
    ap.add_argument("--preset", default="precise", choices=["standard", "precise"])
    ap.add_argument("--poly", type=int, default=0, choices=[0, 3, 5], help="LR only: sigmoid polynomial degree (0 = none)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=2000)
    a = ap.parse_args()

    W, d, t_logit, scale, zmax = K.load_eval(a.model, a.split, a.n)
    need = 3 if a.model == "cnn" else 1 + {0: 0, 3: 2, 5: 3}[a.poly]
    assert K.levels(a.preset) >= need, f"{a.model} poly={a.poly} needs {need} levels; preset {a.preset} has {K.levels(a.preset)}"
    coefs = K.fit_sigmoid_poly(a.poly, zmax) if (a.model == "lr" and a.poly) else None
    X, y, lp, split, sid = d["X"], d["y"], d["lp"], d["split"], d["sid"]
    if a.model == "cnn":
        slots = C.PRESETS[a.preset]["poly"] // 2
        assert C.N_WINDOWS * C.next_pow2(C.KERNEL ** 2) <= slots, "image does not fit one ciphertext - use --preset precise or a smaller IMG"
    print(f"{C.TAG} | {a.model}{' poly' + str(a.poly) if coefs else ''} | preset {a.preset} | split {a.split}: {len(y)} images")

    name = f"{a.model}{('_poly' + str(a.poly)) if coefs else ''}_{a.preset}_{a.split}_{len(y)}"
    cache = C.REPORT / f"cache_{C.TAG}_{name}"; cache.mkdir(exist_ok=True)
    t0 = time.time(); done = 0
    with Pool(a.workers, initializer=init_worker, initargs=(a.model, a.preset, W, coefs)) as p:
        for ci, s0 in enumerate(range(0, len(y), a.chunk)):
            f = cache / f"chunk_{ci:05d}.npy"
            if f.exists(): continue
            sl = slice(s0, s0 + a.chunk)
            np.save(f, np.array(p.map(infer, list(X[sl]), chunksize=4)))
            done += len(X[sl]); print(f"  {min(s0 + a.chunk, len(y))}/{len(y)} | {(time.time() - t0) / 60:.1f} min", flush=True)
    dt = time.time() - t0
    out = np.concatenate([np.load(cache / f"chunk_{i:05d}.npy") for i in range(-(-len(y) // a.chunk))])

    cls_plain = lp >= t_logit
    if coefs:                                   # encrypted output is a probability; compare with exact sigmoid
        thr = W["threshold"]; ref = 1 / (1 + np.exp(-lp)); cls_enc = out >= thr
    else:
        ref = lp; cls_enc = out >= t_logit
    cls_cal = (out >= K.poly_sigmoid_plain(t_logit, coefs)) if coefs else cls_enc   # threshold moved into polynomial space
    def stats(m):
        return dict(n=int(m.sum()), auroc_plain=float(roc_auc_score(y[m], lp[m])), auroc_enc=float(roc_auc_score(y[m], out[m])),
                    acc_plain=float((cls_plain[m] == y[m]).mean()), acc_enc=float((cls_enc[m] == y[m]).mean()),
                    err_mean=float(np.abs(out[m] - ref[m]).mean()), err_max=float(np.abs(out[m] - ref[m]).max()),
                    err_signed_mean=float((out[m] - ref[m]).mean()), class_flips=int((cls_plain[m] != cls_enc[m]).sum()),
                    acc_enc_calibrated=float((cls_cal[m] == y[m]).mean()), class_flips_calibrated=int((cls_plain[m] != cls_cal[m]).sum()))
    res = dict(tag=C.TAG, dataset=C.DATASET, model=a.model, poly=a.poly if coefs else 0, preset=a.preset, split=a.split,
               sec_per_patient_wall=(dt / done if done else None), total_sec=dt, workers=a.workers,
               poly_interval=zmax if coefs else None, poly_coefs=coefs, **stats(np.ones(len(y), bool)))
    if a.split == "all":
        res["per_split"] = {k: stats(split == k) for k in ("train", "val", "test") if (split == k).any()}
    print(json.dumps(res, indent=2))
    json.dump(res, open(C.REPORT / f"ckks_{C.TAG}_{name}.json", "w"), indent=2)
    import pandas as pd
    pd.DataFrame(dict(idx=sid, split=split, y=y, plain_logit=lp, enc_out=out)).to_csv(C.REPORT / f"predictions_{C.TAG}_{name}.csv", index=False)


if __name__ == "__main__":
    main()
