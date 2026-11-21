"""
Common CKKS and evaluation utilities shared across tabular and image pipelines.
"""
import sys
from pathlib import Path
from .ckks_context import (
    PRESETS,
    make_ckks_context,
    get_preset_slots,
    get_preset_levels
)
from .encrypted_inference import (
    calibrate_im2col,
    im2col_cnn_init,
    im2col_cnn_forward,
    ckks_lr_forward,
    fit_sigmoid_poly,
    poly_sigmoid_plain,
    poly_sigmoid,
    sign_poly_plain,
    sign_poly,
    saturation_input,
    fold_threshold,
    load_eval_data
)
from .metrics import evaluate_classification_fidelity, compute_split_metrics
from .utils import run_chunked_inference, next_pow2, set_seed

# Aliases for direct compatibility with scripts
make_ctx = make_ckks_context
levels = get_preset_levels
calibrate = calibrate_im2col
cnn_forward = im2col_cnn_forward
lr_forward = ckks_lr_forward
fold = fold_threshold


def cnn_init(ctx, W, fc=None, bias=None, img_h=28, img_w=28, kernel_h=7, kernel_w=7, stride=3, n_windows=64):
    fc_weights = fc if fc is not None else W.get("fc_w", W.get("fc"))
    final_bias = W.get("bias") if bias is None else bias
    return im2col_cnn_init(
        ctx,
        conv_w=W.get("conv_w", W.get("ker")),
        conv_b=W["conv_b"],
        fc_w=fc_weights,
        bias=final_bias,
        img_h=img_h,
        img_w=img_w,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        stride=stride,
        n_windows=n_windows
    )


def load_eval(model, split="all", n=0, scale_pct=100.0, models_dir=None, tag=None):
    """Loads model and evaluation npz for encrypted inference."""
    if models_dir is None or tag is None:
        try:
            import config as C
            models_dir = models_dir or getattr(C, "MODELS", None)
            tag = tag or getattr(C, "TAG", None)
        except ImportError:
            pass
    if models_dir is None or tag is None:
        raise ValueError("models_dir and tag must be specified or available in config")
    return load_eval_data(models_dir, tag, model, split=split, n=n, scale_pct=scale_pct)


__all__ = [
    "PRESETS",
    "make_ckks_context",
    "make_ctx",
    "get_preset_slots",
    "get_preset_levels",
    "levels",
    "calibrate_im2col",
    "calibrate",
    "im2col_cnn_init",
    "cnn_init",
    "im2col_cnn_forward",
    "cnn_forward",
    "ckks_lr_forward",
    "lr_forward",
    "fit_sigmoid_poly",
    "poly_sigmoid_plain",
    "poly_sigmoid",
    "sign_poly_plain",
    "sign_poly",
    "saturation_input",
    "fold_threshold",
    "fold",
    "load_eval_data",
    "load_eval",
    "evaluate_classification_fidelity",
    "compute_split_metrics",
    "run_chunked_inference",
    "next_pow2",
    "set_seed"
]
