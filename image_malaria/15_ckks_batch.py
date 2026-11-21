"""Step 15 (novelty 2): SIMD BATCHING (LR or CNN).
One ciphertext holds ONE pixel of B patients (feature-major layout), B up to N/2 slots (4096 / 8192).
A whole batch is scored with ciphertext x scalar and a few squarings only: no rotations, no Galois keys,
and every pixel is encrypted once and shared by all sliding windows (no im2col duplication).
  python 15_ckks_batch.py --model cnn --preset standard --workers 2
  python 15_ckks_batch.py --model lr --poly 3 --preset standard
  python 15_ckks_batch.py --model cnn --preset deep --decision      (threshold decision under encryption too)
Memory: a batch holds IMG*IMG ciphertexts; use --workers 2-4.
"""
import argparse, json, time, numpy as np, tenseal as ts
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score
import config as C
import common as K

G = {}


def init_worker(model, preset, W, F, decision, iters, coefs):
    G.update(model=model, ctx=K.make_ctx(preset, galois=False), W=W, F=F, decision=decision, iters=iters, a=coefs)


def score_batch(Xb):
    ctx, W, F, model = G["ctx"], G["W"], G["F"], G["model"]
    B = len(Xb)
    t0 = time.time()
    pix = [[ts.ckks_vector(ctx, Xb[:, r, c].tolist()) for c in range(C.IMG)] for r in range(C.IMG)]
    ct_bytes = len(pix[0][0].serialize()) * C.IMG * C.IMG
    t1 = time.time()
    if model == "lr":
        w = F["w"] if G["decision"] else W["w"]; b = F["b"] if G["decision"] else W["b"]
        z = None
        for r in range(C.IMG):
            for c in range(C.IMG):
                term = pix[r][c] * w[r * C.IMG + c]
                z = term if z is None else z + term
        z = z + b
        if G["a"]:
            z = K.poly_sigmoid(z, G["a"])
    else:
        fc = F["fc"] if G["decision"] else W["fc_w"]; bias = F["bias"] if G["decision"] else W["bias"]
        S, Kk = C.STRIDE, C.KERNEL
        z = None
        for ch in range(W["channels"]):
            ker = W["conv_w"][ch]
            for a in range(C.SIDE):
                for b_ in range(C.SIDE):
                    s = None
                    for i in range(Kk):
                        for j in range(Kk):
                            term = pix[a * S + i][b_ * S + j] * ker[i][j]
                            s = term if s is None else s + term
                    v = (s + W["conv_b"][ch]).square() * fc[ch][a * C.SIDE + b_]
                    z = v if z is None else z + v
        z = z + bias
    if G["decision"]:
        z = K.sign_poly(z, G["iters"])
    t2 = time.time()
    out = np.array(z.decrypt())[:B]
    t3 = time.time()
    return out, dict(B=B, enc=t1 - t0, eval=t2 - t1, dec=t3 - t2, ct_bytes=ct_bytes, out_bytes=len(z.serialize()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cnn", choices=["cnn", "lr"])
    ap.add_argument("--preset", default="standard", choices=list(C.PRESETS))
    ap.add_argument("--poly", type=int, default=0, choices=[0, 3, 5])
    ap.add_argument("--decision", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--batch", type=int, default=0, help="patients per ciphertext (0 = all slots)")
    ap.add_argument("--scale-pct", type=float, default=100.0, help="decision only: percentile of |logit-t| used as scale (100 = max)")
    a = ap.parse_args()
    iters = C.PRESETS[a.preset].get("sign_iters", 0)
    if a.decision:
        assert iters > 0, "--decision needs --preset deep or deeper"; assert a.poly == 0
    W, d, t_logit, scale, zmax = K.load_eval(a.model, a.split, a.n, a.scale_pct)
    need = (3 if a.model == "cnn" else 1 + {0: 0, 3: 2, 5: 3}[a.poly]) + (2 * iters if a.decision else 0)
    assert K.levels(a.preset) >= need, f"needs {need} levels, preset {a.preset} has {K.levels(a.preset)}"
    coefs = K.fit_sigmoid_poly(a.poly, zmax) if (a.model == "lr" and a.poly) else None
    F = K.fold(W, a.model, t_logit, scale)
    X, y, lp, split, sid = d["X"], d["y"], d["lp"], d["split"], d["sid"]
    slots = C.PRESETS[a.preset]["poly"] // 2
    B = min(a.batch or slots, slots)
    batches = [X[i:i + B] for i in range(0, len(y), B)]
    print(f"{C.TAG} | {a.model}{' poly' + str(a.poly) if coefs else ''} | preset {a.preset} | split {a.split}: {len(y)} images | "
          f"{len(batches)} batches of <= {B} | decision under encryption: {a.decision}")

    t0 = time.time()
    with Pool(a.workers, initializer=init_worker, initargs=(a.model, a.preset, W, F, a.decision, iters, coefs)) as p:
        outs = p.map(score_batch, batches, chunksize=1)
    dt = time.time() - t0
    out = np.concatenate([o[0] for o in outs]); info = [o[1] for o in outs]
    tot = lambda k: float(sum(i[k] for i in info))
    res = dict(tag=C.TAG, dataset=C.DATASET, model=a.model, poly=a.poly, preset=a.preset, split=a.split, decision=a.decision,
               batch_size=B, n_batches=len(batches), workers=a.workers, total_sec_wall=dt, sec_per_patient_wall=dt / len(y),
               cpu_ms_per_patient_encrypt=1000 * tot("enc") / len(y), cpu_ms_per_patient_eval=1000 * tot("eval") / len(y),
               upload_ciphertexts_per_batch=C.IMG * C.IMG, upload_MB_per_patient=tot("ct_bytes") / 1e6 / len(y),
               download_MB_per_patient=tot("out_bytes") / 1e6 / len(y))
    cp = lp >= t_logit
    if a.decision:
        xs = (lp - t_logit) / scale; ce = out >= 0
        def stats(m):
            return dict(n=int(m.sum()), class_flips=int((cp[m] != ce[m]).sum()), acc_plain=float((cp[m] == y[m]).mean()),
                        acc_enc=float((ce[m] == y[m]).mean()), saturated_fraction=float((np.abs(out[m]) >= 0.9).mean()))
        res.update(scale=scale, scale_pct=a.scale_pct, n_overflow=int((np.abs(xs) > 1.7).sum()), sign_iters=iters, saturation_input=K.saturation_input(iters), corr_out_vs_logit=float(np.corrcoef(out, lp)[0, 1]),
                   **stats(np.ones(len(y), bool)))
        cols = dict(x_plain=xs, out_enc=out)
    else:
        if coefs: ref, ce = 1 / (1 + np.exp(-lp)), out >= W["threshold"]
        else:     ref, ce = lp, out >= t_logit
        cal = (out >= K.poly_sigmoid_plain(t_logit, coefs)) if coefs else ce
        def stats(m):
            return dict(n=int(m.sum()), auroc_plain=float(roc_auc_score(y[m], lp[m])), auroc_enc=float(roc_auc_score(y[m], out[m])),
                        acc_plain=float((cp[m] == y[m]).mean()), acc_enc=float((ce[m] == y[m]).mean()),
                        err_mean=float(np.abs(out[m] - ref[m]).mean()), err_max=float(np.abs(out[m] - ref[m]).max()),
                        err_signed_mean=float((out[m] - ref[m]).mean()), class_flips=int((cp[m] != ce[m]).sum()),
                        acc_enc_calibrated=float((cal[m] == y[m]).mean()), class_flips_calibrated=int((cp[m] != cal[m]).sum()))
        res.update(**stats(np.ones(len(y), bool))); cols = dict(plain_logit=lp, enc_out=out)
    if a.split == "all":
        res["per_split"] = {k: stats(split == k) for k in ("train", "val", "test") if (split == k).any()}
    print(json.dumps(res, indent=2))
    tag = f"{C.TAG}_{a.model}{('_poly' + str(a.poly)) if a.poly else ''}_{a.preset}_{a.split}{'_decision' if a.decision else ''}{('_p' + format(a.scale_pct, 'g')) if (a.decision and a.scale_pct < 100) else ''}"
    json.dump(res, open(C.REPORT / f"batch_{tag}.json", "w"), indent=2)
    import pandas as pd
    pd.DataFrame(dict(idx=sid, split=split, y=y, **cols)).to_csv(C.REPORT / f"batch_predictions_{tag}.csv", index=False)


if __name__ == "__main__":
    main()
