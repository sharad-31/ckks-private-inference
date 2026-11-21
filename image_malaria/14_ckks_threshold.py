"""Step 14 (novelty 1): ENCRYPTED THRESHOLD DECISION, patient by patient (LR or CNN).
The server also evaluates  logit >= threshold  under encryption (sign approximation);
the client decrypts ONE number ~ +-1 (the class) and never sees the risk score.
  python 14_ckks_threshold.py --model cnn --preset deep --workers 4 [--split test] [--n 200]
"""
import argparse, json, time, numpy as np
from multiprocessing import Pool
import config as C
import common as K

G = {}


def init_worker(model, preset, W, F, iters):
    ctx = K.make_ctx(preset, galois=True)
    G.update(model=model, ctx=ctx, W=W, F=F, iters=iters)
    if model == "cnn":
        G["st"] = K.cnn_init(ctx, W, fc=F["fc"], bias=F["bias"])


def decide(x):
    ctx = G["ctx"]
    z = K.cnn_forward(ctx, G["st"], x) if G["model"] == "cnn" else K.lr_forward(ctx, G["F"]["w"], G["F"]["b"], x)
    return K.sign_poly(z, G["iters"]).decrypt()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cnn", choices=["cnn", "lr"])
    ap.add_argument("--preset", default="deep", choices=["deep", "deeper"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--scale-pct", type=float, default=100.0, help="percentile of |logit-t| (train) used for the scale; 100 = max (safe), 99 = more saturation but rare outliers can overflow")
    a = ap.parse_args()
    iters = C.PRESETS[a.preset]["sign_iters"]
    W, d, t_logit, scale, _ = K.load_eval(a.model, a.split, a.n, a.scale_pct)
    F = K.fold(W, a.model, t_logit, scale)
    X, y, lp, split, sid = d["X"], d["y"], d["lp"], d["split"], d["sid"]
    xs = (lp - t_logit) / scale
    print(f"{C.TAG} | {a.model} | preset {a.preset} ({iters} sign iterations) | split {a.split}: {len(y)} images | threshold logit {t_logit:.3f}, scale {scale:.2f}")

    name = f"{a.model}_{a.preset}_{a.split}_{len(y)}" + (f"_p{a.scale_pct:g}" if a.scale_pct < 100 else "")
    cache = C.REPORT / f"cache_thr_{C.TAG}_{name}"; cache.mkdir(exist_ok=True)
    t0 = time.time(); done = 0
    with Pool(a.workers, initializer=init_worker, initargs=(a.model, a.preset, W, F, iters)) as p:
        for ci, s0 in enumerate(range(0, len(y), a.chunk)):
            f = cache / f"chunk_{ci:05d}.npy"
            if f.exists(): continue
            sl = slice(s0, s0 + a.chunk)
            np.save(f, np.array(p.map(decide, list(X[sl]), chunksize=2)))
            done += len(X[sl]); print(f"  {min(s0 + a.chunk, len(y))}/{len(y)} | {(time.time() - t0) / 60:.1f} min", flush=True)
    dt = time.time() - t0
    out = np.concatenate([np.load(cache / f"chunk_{i:05d}.npy") for i in range(-(-len(y) // a.chunk))])

    cp, ce = xs >= 0, out >= 0
    def stats(m):
        return dict(n=int(m.sum()), class_flips=int((cp[m] != ce[m]).sum()), acc_plain=float((cp[m] == y[m]).mean()),
                    acc_enc=float((ce[m] == y[m]).mean()), saturated_fraction=float((np.abs(out[m]) >= 0.9).mean()))
    res = dict(tag=C.TAG, dataset=C.DATASET, model=a.model, preset=a.preset, split=a.split, sign_iters=iters,
               threshold_logit=t_logit, scale=scale, scale_pct=a.scale_pct, n_overflow=int((np.abs(xs) > 1.7).sum()), saturation_input=K.saturation_input(iters), workers=a.workers,
               sec_per_patient_wall=(dt / done if done else None), total_sec=dt,
               corr_out_vs_logit=float(np.corrcoef(out, lp)[0, 1]), **stats(np.ones(len(y), bool)))
    if a.split == "all":
        res["per_split"] = {k: stats(split == k) for k in ("train", "val", "test") if (split == k).any()}
    print(json.dumps(res, indent=2))
    json.dump(res, open(C.REPORT / f"thr_{C.TAG}_{name}.json", "w"), indent=2)
    import pandas as pd
    pd.DataFrame(dict(idx=sid, split=split, y=y, x_plain=xs, out_enc=out)).to_csv(C.REPORT / f"thr_predictions_{C.TAG}_{name}.csv", index=False)


if __name__ == "__main__":
    main()
