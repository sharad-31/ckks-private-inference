"""
Encrypted inference procedures using CKKS homomorphic encryption (TenSEAL).
Implements im2col convolution, square activations, polynomial sigmoid approximations,
and iterative sign-polynomial thresholding.
"""
import json
from pathlib import Path
import numpy as np
import tenseal as ts


# -----------------------------------------------------------------------------
# im2col 2D Convolution Calibration & Execution
# -----------------------------------------------------------------------------
def calibrate_im2col(ctx, img_h=28, img_w=28, kernel_h=7, kernel_w=7, stride=3, n_windows=64):
    """
    Calibrate TenSEAL im2col window ordering and kernel orientation.
    TenSEAL encodes matrices in row/col major order that may differ from PyTorch;
    this calibration computes the 1-to-1 permutation vector once.
    """
    side_h = (img_h - kernel_h) // stride + 1
    side_w = (img_w - kernel_w) // stride + 1
    expected_windows = side_h * side_w
    assert expected_windows == n_windows, f"Expected {expected_windows} windows, got {n_windows}"

    for seed in range(5):
        rng = np.random.RandomState(seed)
        img = rng.randn(img_h, img_w)
        ker = rng.randn(kernel_h, kernel_w)

        # Plaintext windowed conv
        plain = []
        for r in range(side_h):
            for c in range(side_w):
                patch = img[r * stride : r * stride + kernel_h, c * stride : c * stride + kernel_w]
                plain.append((patch * ker).sum())
        plain = np.array(plain)

        # TenSEAL reads matrix transposed: pass kernel_w, kernel_h
        enc, wn = ts.im2col_encoding(ctx, img.tolist(), kernel_w, kernel_h, stride)
        assert wn == n_windows, f"Unexpected window count from TenSEAL: {wn}"

        for kt in (False, True):
            k_eval = ker.T if kt else ker
            dec = np.array(enc.conv2d_im2col(k_eval.tolist(), wn).decrypt())[:wn]
            perm = np.array([int(np.argmin(np.abs(dec - p))) for p in plain])
            if len(set(perm)) == wn and np.abs(dec[perm] - plain).max() < 5e-3:
                return perm, kt

    raise RuntimeError("Could not calibrate TenSEAL im2col window ordering.")


def im2col_cnn_init(ctx, conv_w, conv_b, fc_w, bias, img_h=28, img_w=28, kernel_h=7, kernel_w=7, stride=3, n_windows=64):
    """
    Initializes calibrated weights for homomorphic CNN evaluation.
    Reorders FC weights according to im2col slot mapping.
    """
    perm, kt = calibrate_im2col(ctx, img_h, img_w, kernel_h, kernel_w, stride, n_windows)
    calibrated_ker = [(np.array(k).T if kt else np.array(k)).tolist() for k in conv_w]
    
    calibrated_fcs = []
    for row in fc_w:
        arr = np.array(row)
        out = np.empty_like(arr)
        out[perm] = arr  # enc slot perm[i] <- plain window i
        calibrated_fcs.append(out.tolist())

    return {
        "ker": calibrated_ker,
        "conv_b": conv_b,
        "fc": calibrated_fcs,
        "bias": bias,
        "kernel_h": kernel_h,
        "kernel_w": kernel_w,
        "stride": stride,
        "n_windows": n_windows
    }


def im2col_cnn_forward(ctx, state, x_image, static_vec=None, static_w=None):
    """
    Evaluates depth-three CNN under CKKS homomorphic encryption:
    Input (im2col ciphertext) -> Conv2D (ct-pt mul) + Bias -> Square activation (ct-ct mul)
    -> FC readout (ct-pt dot product) [+ Static branch dot product] -> Output logit ciphertext.
    """
    kw, kh = state["kernel_w"], state["kernel_h"]
    stride = state["stride"]
    wn = state["n_windows"]

    # Client-side: im2col encoding + encryption
    enc_x, count = ts.im2col_encoding(ctx, x_image.tolist(), kw, kh, stride)
    assert count == wn, f"Window count mismatch: {count} != {wn}"

    # Server-side evaluation
    acc = None
    for c in range(len(state["ker"])):
        y = enc_x.conv2d_im2col(state["ker"][c], wn) + state["conv_b"][c]
        y = y.square()
        y = y.dot(state["fc"][c])
        acc = y if acc is None else acc + y

    # Optional static branch (used in tabular hybrid CNN)
    if static_vec is not None and static_w is not None:
        enc_s = ts.ckks_vector(ctx, static_vec.tolist())
        acc = acc + enc_s.dot(static_w)

    return acc + state["bias"]


def ckks_lr_forward(ctx, w, b, x_vec):
    """
    Evaluates linear/logistic regression under CKKS:
    z = (x · w) + b
    """
    enc_x = ts.ckks_vector(ctx, np.asarray(x_vec).reshape(-1).tolist())
    return enc_x.dot(w) + b


# -----------------------------------------------------------------------------
# Polynomial Sigmoid Approximations (deg 3, deg 5)
# -----------------------------------------------------------------------------
def fit_sigmoid_poly(deg, B):
    """
    Fits least-squares odd polynomial approximation to logistic sigmoid on [-B, B]:
    sigma(z) ≈ 0.5 + a1*z + a3*z^3 (+ a5*z^5)
    """
    u = np.linspace(-1, 1, 4001)
    A = np.stack([u ** p for p in range(1, deg + 1, 2)], axis=1)
    target = 1.0 / (1.0 + np.exp(-B * u)) - 0.5
    c, *_ = np.linalg.lstsq(A, target, rcond=None)
    return [float(c[i] / (B ** p)) for i, p in enumerate(range(1, deg + 1, 2))]


def poly_sigmoid_plain(z, coeffs):
    """Plaintext evaluation of polynomial sigmoid."""
    return 0.5 + sum(ai * (z ** (2 * i + 1)) for i, ai in enumerate(coeffs))


def poly_sigmoid(z_enc, coeffs):
    """
    Ciphertext evaluation of polynomial sigmoid:
    2 multiplicative levels for degree 3, 3 levels for degree 5.
    """
    z2 = z_enc.square()
    t = z_enc * coeffs[0] + z2 * (z_enc * coeffs[1])
    if len(coeffs) == 3:  # degree 5
        t = t + z2.square() * (z_enc * coeffs[2])
    return t + 0.5


# -----------------------------------------------------------------------------
# Iterative Sign-Polynomial Encrypted Thresholding
# -----------------------------------------------------------------------------
def fold_threshold(W_or_w, model_or_b=None, t_logit=0.0, scale=1.0, fc_w=None, bias=None, is_lr=False):
    """
    Folds decision threshold (logit - t)/scale into the last linear layer weights
    so no additional multiplicative level is required for threshold centering.

    Supports both:
      fold_threshold(W, model, t_logit, scale)
    and:
      fold_threshold(w=w, b=b, t_logit=..., scale=..., is_lr=True)
    """
    if isinstance(W_or_w, dict):
        W = W_or_w
        model = str(model_or_b) if model_or_b is not None else ("lr" if "w" in W else "cnn")
        if model == "lr":
            return dict(w=[float(v / scale) for v in W["w"]], b=float((W["b"] - t_logit) / scale))
        return dict(fc=[[float(v / scale) for v in row] for row in W["fc_w"]], bias=float((W["bias"] - t_logit) / scale))
    
    if is_lr:
        return {
            "w": [float(v / scale) for v in W_or_w],
            "b": float((model_or_b - t_logit) / scale)
        }
    return {
        "fc": [[float(v / scale) for v in row] for row in fc_w],
        "bias": float((bias - t_logit) / scale)
    }


def sign_poly_plain(x, iters):
    """Plaintext iterative evaluation of f(x) = 1.5x - 0.5x^3."""
    for _ in range(iters):
        x = 1.5 * x - 0.5 * (x ** 3)
    return x


def sign_poly(x_enc, iters):
    """
    Homomorphic iterative evaluation of sign polynomial:
    f(x) = 1.5x - 0.5x^3 requires 2 multiplicative levels per iteration.
    """
    for _ in range(iters):
        x_enc = x_enc.square() * (x_enc * -0.5) + x_enc * 1.5
    return x_enc


def saturation_input(iters, level=0.9):
    """Calculates minimum positive input x such that sign_poly_plain(xs, iters) >= level."""
    xs = np.linspace(1e-4, 1.0, 100000)
    return float(xs[np.argmax(sign_poly_plain(xs, iters) >= level)])


# -----------------------------------------------------------------------------
# Evaluation Data Loading Helper
# -----------------------------------------------------------------------------
def load_eval_data(models_dir, tag, model, split="all", n=0, scale_pct=100.0):
    """
    Loads saved evaluation data, model weights, and threshold for inference experiments.
    """
    models_path = Path(models_dir)
    W = json.load(open(models_path / f"{model}_{tag}.json"))
    t = np.load(models_path / f"eval_{tag}.npz", allow_pickle=True)
    X, y, sp, sid, lp = t["X"], t["y"], t["split"], t["sid"], t[f"logit_{model}"]
    thr = W["threshold"]
    t_logit = float(np.log(thr / (1 - thr)))
    dev = np.abs(lp[sp == "train"] - t_logit)
    scale = float(1.05 * (dev.max() if scale_pct >= 100 else np.percentile(dev, scale_pct)))
    zmax = float(1.05 * np.abs(lp[sp == "train"]).max())
    idx = np.arange(len(y)) if split == "all" else np.where(sp == split)[0]
    if n:
        idx = np.sort(np.random.RandomState(0).permutation(idx)[:n])
    return W, dict(X=X[idx], y=y[idx], lp=lp[idx], split=sp[idx], sid=sid[idx]), t_logit, scale, zmax
