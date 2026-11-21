"""
toy_demo.py - runs WITHOUT any dataset.

Builds a tiny square-activation CNN with random weights, evaluates it on random 11x11
"images" first in plaintext and then under CKKS (TenSEAL), and reports, for the
standard and the precise preset:
    * the logit error |encrypted - plaintext|
    * the number of decisions that flip at a threshold placed at the median logit
It also runs the feature-major SIMD batching variant (one ciphertext per pixel,
one image per slot, no rotations / Galois keys) and compares it with plaintext.

    python demo/toy_demo.py            # default: 20 images
    python demo/toy_demo.py --n 40

The numbers are only a smoke test of the pipeline logic (random weights, random data);
they are NOT results of the study.
"""
import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np

# Add project root to sys.path
ROOT_REPO = Path(__file__).resolve().parent.parent
if str(ROOT_REPO) not in sys.path:
    sys.path.insert(0, str(ROOT_REPO))

# small geometry so the demo is fast: 11x11 image, 3x3 kernel, stride 2 -> 5x5 = 25 windows
os.environ.update(IMG="11", KERNEL="3", STRIDE="2", CHANNELS="4")
sys.path.insert(0, str(ROOT_REPO / "image_malaria"))
import config as C            # noqa: E402
import tenseal as ts          # noqa: E402
from common.ckks_context import make_ckks_context
from common.encrypted_inference import im2col_cnn_init, im2col_cnn_forward
from common.metrics import evaluate_classification_fidelity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="number of random images")
    args = ap.parse_args()

    rng = np.random.RandomState(0)
    K, S, SIDE, NW, CH = C.KERNEL, C.STRIDE, C.SIDE, C.N_WINDOWS, C.CHANNELS
    W = dict(
        conv_w=(rng.randn(CH, K, K) * 0.3).tolist(),
        conv_b=(rng.randn(CH) * 0.1).tolist(),
        fc_w=(rng.randn(CH, NW) * 0.1).tolist(),
        bias=0.05
    )
    X = rng.randn(args.n, C.IMG, C.IMG) * 0.8

    def plain_logit(x):
        z = W["bias"]
        for c in range(CH):
            ker = np.array(W["conv_w"][c])
            h = np.array([
                (x[a*S:a*S+K, b*S:b*S+K] * ker).sum() + W["conv_b"][c]
                for a in range(SIDE) for b in range(SIDE)
            ])
            z += (h ** 2) @ np.array(W["fc_w"][c])          # square activation + linear read-out
        return z

    zp = np.array([plain_logit(x) for x in X])
    thr = float(np.median(zp))
    print(f"{args.n} random images, {C.IMG}x{C.IMG}, {NW} windows per channel, threshold = median plaintext logit {thr:.4f}\n")

    print("Per-patient CKKS inference (im2col + conv -> square -> dot; client decrypts)")
    for preset in ("standard", "precise"):
        ctx = make_ckks_context(preset, galois=True)
        st = im2col_cnn_init(
            ctx, W["conv_w"], W["conv_b"], W["fc_w"], W["bias"],
            img_h=C.IMG, img_w=C.IMG, kernel_h=K, kernel_w=K, stride=S, n_windows=NW
        )
        t0 = time.time()
        ze = np.array([im2col_cnn_forward(ctx, st, x).decrypt()[0] for x in X])
        dt = (time.time() - t0) / len(X) * 1000
        
        # Using unified fidelity metrics
        metrics = evaluate_classification_fidelity(np.ones(len(X)), zp, ze, threshold=0.5)
        flips = int(((zp >= thr) != (ze >= thr)).sum())
        print(f"  {preset:9s} max |err| = {metrics['logit_err_max']:.2e}   "
              f"mean |err| = {metrics['logit_err_mean']:.2e}   flips = {flips}/{len(X)}   {dt:.0f} ms/image")

    print("\nSIMD batching (one ciphertext per pixel, one image per slot, no rotations)")
    ctx = make_ckks_context("standard", galois=False)
    t0 = time.time()
    pix = [[ts.ckks_vector(ctx, X[:, r, c].tolist()) for c in range(C.IMG)] for r in range(C.IMG)]   # client
    acc = None
    for c in range(CH):                                                                                 # server
        ker = np.array(W["conv_w"][c])
        for a in range(SIDE):
            for b in range(SIDE):
                y = None
                for u in range(K):
                    for v in range(K):
                        t = pix[a*S+u][b*S+v] * float(ker[u, v])
                        y = t if y is None else y + t
                y = (y + W["conv_b"][c]).square() * W["fc_w"][c][a*SIDE+b]
                acc = y if acc is None else acc + y
    ze = np.array((acc + W["bias"]).decrypt())[:len(X)]
    dt = (time.time() - t0) / len(X) * 1000
    metrics = evaluate_classification_fidelity(np.ones(len(X)), zp, ze, threshold=0.5)
    flips = int(((zp >= thr) != (ze >= thr)).sum())
    print(f"  standard  max |err| = {metrics['logit_err_max']:.2e}   "
          f"mean |err| = {metrics['logit_err_mean']:.2e}   flips = {flips}/{len(X)}   {dt:.0f} ms/image (batch of {len(X)})")
    print(f"  ciphertexts uploaded: {C.IMG * C.IMG} (one per pixel position); batch of only {len(X)} images, so ms/image is not representative")


if __name__ == "__main__":
    main()
