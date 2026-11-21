"""
Evaluation and fidelity metrics for plaintext vs CKKS encrypted inference.
Computes AUROC, accuracy, logit approximation error, and class-flip counts.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score


def evaluate_classification_fidelity(y_true, logit_plain, logit_enc, threshold=0.5):
    """
    Evaluates fidelity of encrypted inference against plaintext predictions.

    Args:
        y_true (np.ndarray): Binary ground truth labels (0 or 1).
        logit_plain (np.ndarray): Plaintext model logits.
        logit_enc (np.ndarray): Decrypted CKKS model logits.
        threshold (float): Decision threshold applied to probabilities.

    Returns:
        dict: Summary of fidelity and predictive performance metrics.
    """
    y_true = np.asarray(y_true).astype(int)
    lp = np.asarray(logit_plain).astype(float)
    le = np.asarray(logit_enc).astype(float)

    # Probabilities via standard sigmoid
    prob_p = 1.0 / (1.0 + np.exp(-lp))
    prob_e = 1.0 / (1.0 + np.exp(-le))

    # Binary decisions
    dec_p = (prob_p >= threshold).astype(int)
    dec_e = (prob_e >= threshold).astype(int)

    # Logit approximation errors
    abs_err = np.abs(le - lp)
    signed_err = le - lp
    flips = int(np.sum(dec_p != dec_e))
    n = len(y_true)

    res = {
        "n": n,
        "auroc_plain": float(roc_auc_score(y_true, lp)) if len(np.unique(y_true)) > 1 else float("nan"),
        "auroc_enc": float(roc_auc_score(y_true, le)) if len(np.unique(y_true)) > 1 else float("nan"),
        "acc_plain": float(accuracy_score(y_true, dec_p)),
        "acc_enc": float(accuracy_score(y_true, dec_e)),
        "logit_err_mean": float(np.mean(abs_err)),
        "logit_err_max": float(np.max(abs_err)) if n > 0 else 0.0,
        "err_signed_mean": float(np.mean(signed_err)),
        "class_flips": flips,
        "flip_rate": float(flips / n) if n > 0 else 0.0
    }
    return res


def compute_split_metrics(y_true, logit_plain, logit_enc, split_labels, threshold=0.5):
    """
    Computes fidelity metrics overall and per data split ('train', 'val', 'test').
    """
    overall = evaluate_classification_fidelity(y_true, logit_plain, logit_enc, threshold)
    per_split = {}
    
    unique_splits = np.unique(split_labels)
    for sp in unique_splits:
        mask = (split_labels == sp)
        if np.sum(mask) > 0:
            per_split[str(sp)] = evaluate_classification_fidelity(
                y_true[mask],
                logit_plain[mask],
                logit_enc[mask],
                threshold
            )
            
    return {
        "overall": overall,
        "per_split": per_split
    }
