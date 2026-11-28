"""
Generate all 8 publication figures (300 DPI PNGs) directly from the
tabular_mimic/reports/ and image_malaria/reports/ experimental outputs.

Usage:
    python generate_figures.py
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
TAB_REP = os.path.join(ROOT, "tabular_mimic", "reports")
IMG_REP = os.path.join(ROOT, "image_malaria", "reports")
IMG_MOD = os.path.join(ROOT, "image_malaria", "models")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_GRAY  = "#B8BEC7"
C_SLATE = "#7B90AC"
C_NAVY  = "#1C4973"
C_TEAL  = "#34789C"
C_SAND  = "#D9A373"
C_RUST  = "#B2471B"
C_RED   = "#B83B3B"
C_LBLUE = "#C8D5E6"
C_MBLUE = "#6E9EC9"


def make_fig1():
    """Fig. 1: Client-server workflow for LR and CNN under CKKS."""
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")

    zones = [
        (1.5, 38.5, 97, 16.5, "#EDF3EB", "#6D8C6A", "CLIENT (holds secret key)", 10.5),
        (1.5, 16.5, 97, 19.5, "#FAF0E1", "#9C7E54", "SERVER (plaintext weights, no secret key)", 10.5),
        (1.5, 1.5,  97, 13.0, "#EDF3EB", "#6D8C6A", "CLIENT (secret key)", 10.5),
    ]
    for x, y, w, h, fc, ec, title, fs in zones:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.8",
                              facecolor=fc, edgecolor=ec, linestyle="--", linewidth=1.2, zorder=1)
        ax.add_patch(rect)
        ax.text(x + 1.5, y + h - 2.0, title, fontsize=fs, fontweight="bold", color="#1A1A1A", va="center", zorder=3)

    def draw_box(x, y, w, h, text, fc="#DCE6F2", ec="#2B3A4A", fs=9.3):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.8",
                             facecolor=fc, edgecolor=ec, linewidth=1.1, zorder=2)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color="#111111", linespacing=1.25, zorder=3)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.55",
                                    color="#333333", lw=1.3), zorder=4)

    draw_box(3.5,  40.2, 19.5, 10.0, "Input\n(grid + static / image)", fc="#FFFFFF")
    draw_box(27.0, 40.2, 20.5, 10.0, "Normalise with\ntrain-only statistics")
    draw_box(51.5, 40.2, 22.0, 10.0, "Encode: im2col or\nfeature-major (SIMD)")
    draw_box(77.5, 40.2, 18.5, 10.0, "CKKS\nencrypt")

    arrow(23.2, 45.2, 26.8, 45.2)
    arrow(47.7, 45.2, 51.3, 45.2)
    arrow(73.7, 45.2, 77.3, 45.2)

    draw_box(4.0,  18.2, 27.5, 11.0,
             "Model A: LR\nEncrypted dot product, depth 1\n(image LR may use poly sigmoid)",
             fc="#F4D7B5", ec="#4A3820")
    draw_box(35.0, 18.2, 32.5, 11.0,
             "Model B: CNN\nConv (plain kernels) \u2192 square \u2192 dot\nwith readout (+ static branch); depth 3",
             fc="#F4D7B5", ec="#4A3820")

    opt_box = FancyBboxPatch((70.0, 18.2), 26.5, 11.0, boxstyle="round,pad=0.2,rounding_size=0.8",
                             facecolor="#FFF9F0", edgecolor="#B89B72", linestyle=":", linewidth=1.1, zorder=2)
    ax.add_patch(opt_box)
    ax.text(83.25, 23.7,
            "Optional (image only, deep\npreset): sign polynomial after\nmodel = encrypted threshold",
            ha="center", va="center", fontsize=8.8, style="italic", color="#3A2D18", linespacing=1.25, zorder=3)

    ax.plot([86.75, 86.75, 17.75], [40.0, 31.3, 31.3], color="#333333", lw=1.3, zorder=4)
    arrow(17.75, 31.3, 17.75, 29.4)
    arrow(51.25, 31.3, 51.25, 29.4)

    draw_box(4.0,  2.7, 27.5, 7.6, "Decrypt logit\n(or saturated output)")
    draw_box(35.5, 2.7, 26.0, 7.6, "Apply sigmoid\nin the clear")
    draw_box(65.5, 2.7, 30.5, 7.6, "Compare with validation\nthreshold \u2192 class")

    arrow(17.75, 18.3, 17.75, 10.5)
    arrow(51.25, 18.3, 51.25, 10.5)
    arrow(31.7,  6.5,  35.3,  6.5)
    arrow(61.7,  6.5,  65.3,  6.5)

    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT_DIR, "fig_1.png"), bbox_inches="tight")
    plt.close(fig)


def make_fig2():
    """Fig. 2: Per-patient im2col vs feature-major SIMD batch layout."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6))

    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 6)
    ax1.axis("off")
    ax1.set_title("(a) One image per ciphertext (im2col)", fontsize=11, pad=8)
    ax1.text(5.0, 4.7, "64 windows \u00d7 64 slots = 4,096 slots", ha="center", va="center", fontsize=9.8, fontweight="bold")

    colors = ["#4F81BD", "#D9964B", "#5E9B76", "#8B74A8", "#4F81BD", "#D9964B", "#5E9B76", "#8B74A8"]
    x0, w_box = 0.6, 1.05
    for idx, col in enumerate(colors):
        r = FancyBboxPatch((x0 + idx * (w_box + 0.05), 3.0), w_box, 1.0,
                           boxstyle="square,pad=0", facecolor=col, edgecolor="#222222", linewidth=1.0)
        ax1.add_patch(r)
        lbl = f"w{idx}" if idx < 7 else "w63"
        ax1.text(x0 + idx * (w_box + 0.05) + w_box / 2, 3.5, lbl, ha="center", va="center",
                 color="white", fontsize=8.5, fontweight="bold")

    ax1.annotate("", xy=(x0, 2.75), xytext=(x0 + w_box, 2.75),
                 arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.0))
    ax1.text(5.0, 2.25, "Each window: 49 pixel values padded to 64 slots", ha="center", va="center", fontsize=9.2)
    ax1.text(5.0, 1.05, "1 ciphertext = 1 image\nRotations and Galois keys required",
             ha="center", va="center", fontsize=9.2, style="italic", color="#333333",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#F4F6F9", edgecolor="#CCCCCC", lw=0.8))

    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 6)
    ax2.axis("off")
    ax2.set_title("(b) Feature-major SIMD batch", fontsize=11, pad=8)
    ax2.text(5.4, 5.28, "pixel positions (784 ciphertexts per batch)", ha="center", va="center", fontsize=9.5, fontweight="bold")

    gx0, gy0, ncols, nrows, cw, ch = 1.7, 1.55, 9, 6, 0.82, 0.50
    for r in range(nrows):
        for c in range(ncols):
            is_hi = (c == 3)
            fc = "#F2D7A0" if is_hi else "#F2F2F2"
            ec = "#A67C37" if is_hi else "#A8A8A8"
            rect = mpatches.Rectangle((gx0 + c * cw, gy0 + r * ch), cw, ch,
                                      facecolor=fc, edgecolor=ec, linewidth=1.2 if is_hi else 0.7, zorder=2 if is_hi else 1)
            ax2.add_patch(rect)

    hi_rect = mpatches.Rectangle((gx0 + 3 * cw, gy0), cw, nrows * ch,
                                 facecolor="none", edgecolor="#8C5A14", linewidth=1.6, zorder=3)
    ax2.add_patch(hi_rect)
    ax2.text(gx0 + 3.5 * cw, gy0 + nrows * ch + 0.16, "ct\u2096", ha="center", va="bottom",
             fontsize=8.5, fontweight="bold", color="#8C5A14")
    ax2.text(0.85, gy0 + (nrows * ch) / 2, "images\n(slots)", ha="center", va="center", fontsize=9.3)
    ax2.text(5.2, 0.8, "1 ciphertext = 1 pixel across 4,096 (or 8,192) images; no rotations",
             ha="center", va="center", fontsize=9.2, style="italic", color="#333333",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF8EC", edgecolor="#DECBA8", lw=0.8))

    fig.tight_layout(pad=0.6)
    fig.savefig(os.path.join(OUT_DIR, "fig_2.png"), bbox_inches="tight")
    plt.close(fig)


def make_fig3():
    """Fig. 3: Tabular test AUROC & accuracy vs always-negative baseline."""
    configs = [("mimic3", "12x10", "MIMIC-III\n12\u00d710"),
               ("mimic3", "12x5",  "MIMIC-III\n12\u00d75"),
               ("mimic4", "12x10", "MIMIC-IV\n12\u00d710"),
               ("mimic4", "12x5",  "MIMIC-IV\n12\u00d75")]

    stat_auc, lr_auc, cnn_auc, lr_acc, cnn_acc, neg_base = [], [], [], [], [], []
    for ds, grid, _ in configs:
        tr_cnn = json.load(open(os.path.join(TAB_REP, f"train_{ds}_{grid}_los7.json")))
        tr_lr  = json.load(open(os.path.join(TAB_REP, f"train_lr_{ds}_{grid}_los7.json")))
        ck_cnn = json.load(open(os.path.join(TAB_REP, f"ckks_{ds}_{grid}_los7_precise_all.json")))
        stat_auc.append(tr_cnn["lr_test_auroc"])
        lr_auc.append(tr_lr["test"]["auroc"])
        cnn_auc.append(tr_cnn["cnn_test_auroc"])
        lr_acc.append(tr_lr["test"]["acc"])
        cnn_acc.append(ck_cnn["per_split"]["test"]["acc_plain"])
        neg_base.append(0.7432 if ds == "mimic3" else 0.7724)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.0))
    x = np.arange(len(configs))
    w = 0.25

    b1 = ax1.bar(x - w, stat_auc, w, label="Static-only LR", color=C_GRAY, edgecolor="#222222", lw=0.7, zorder=3)
    b2 = ax1.bar(x,     lr_auc,   w, label="LR (grid + static)", color=C_SLATE, edgecolor="#222222", lw=0.7, zorder=3)
    b3 = ax1.bar(x + w, cnn_auc,  w, label="Hybrid CNN", color=C_NAVY, edgecolor="#222222", lw=0.7, zorder=3)
    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.004, f"{h:.3f}",
                     ha="center", va="bottom", fontsize=7.8, rotation=90)

    ax1.set_ylim(0.50, 0.855)
    ax1.set_ylabel("Test AUROC")
    ax1.set_title("(a) Ranking quality (y-axis from 0.5)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([c[2] for c in configs])
    ax1.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9, edgecolor="none")

    w2 = 0.30
    b4 = ax2.bar(x - w2 / 2, lr_acc,  w2, label="LR (grid + static)", color=C_SLATE, edgecolor="#222222", lw=0.7, zorder=3)
    b5 = ax2.bar(x + w2 / 2, cnn_acc, w2, label="Hybrid CNN", color=C_NAVY, edgecolor="#222222", lw=0.7, zorder=3)
    for i, nb in enumerate(neg_base):
        ax2.plot([i - 0.38, i + 0.38], [nb, nb], color="#E02424", linestyle="--", lw=1.8,
                 label="Always-negative" if i == 0 else None, zorder=4)
    for bars in (b4, b5):
        for bar in bars:
            h = bar.get_height()
            if h >= 0.74:
                ax2.text(bar.get_x() + bar.get_width() / 2, h - 0.015, f"{h:.3f}",
                         ha="center", va="top", fontsize=7.8, rotation=90, color="white", fontweight="bold", zorder=5)
            else:
                ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.006, f"{h:.3f}",
                         ha="center", va="bottom", fontsize=7.8, rotation=90, zorder=5)

    ax2.set_ylim(0.40, 0.90)
    ax2.set_ylabel("Test accuracy")
    ax2.set_title("(b) Accuracy vs always-negative baseline")
    ax2.set_xticks(x)
    ax2.set_xticklabels([c[2] for c in configs])
    ax2.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
    ax2.legend(loc="upper center", ncol=3, frameon=True, facecolor="white", framealpha=0.9, edgecolor="none", fontsize=8.2)

    fig.tight_layout(pad=1.0)
    fig.savefig(os.path.join(OUT_DIR, "fig_3.png"), bbox_inches="tight")
    plt.close(fig)


def make_fig4():
    """Fig. 4: Test-split class flips across all tasks and presets."""
    tasks = [
        ("MIMIC-III\n12\u00d710",
         os.path.join(TAB_REP, "ckks_lr_mimic3_12x10_los7_standard_all.json"),
         os.path.join(TAB_REP, "ckks_mimic3_12x10_los7_standard_all.json"),
         os.path.join(TAB_REP, "ckks_lr_mimic3_12x10_los7_precise_all.json"),
         os.path.join(TAB_REP, "ckks_mimic3_12x10_los7_precise_all.json")),
        ("MIMIC-III\n12\u00d75",
         os.path.join(TAB_REP, "ckks_lr_mimic3_12x5_los7_standard_all.json"),
         os.path.join(TAB_REP, "ckks_mimic3_12x5_los7_standard_all.json"),
         os.path.join(TAB_REP, "ckks_lr_mimic3_12x5_los7_precise_all.json"),
         os.path.join(TAB_REP, "ckks_mimic3_12x5_los7_precise_all.json")),
        ("MIMIC-IV\n12\u00d710",
         os.path.join(TAB_REP, "ckks_lr_mimic4_12x10_los7_standard_all.json"),
         os.path.join(TAB_REP, "ckks_mimic4_12x10_los7_standard_all.json"),
         os.path.join(TAB_REP, "ckks_lr_mimic4_12x10_los7_precise_all.json"),
         os.path.join(TAB_REP, "ckks_mimic4_12x10_los7_precise_all.json")),
        ("MIMIC-IV\n12\u00d75",
         os.path.join(TAB_REP, "ckks_lr_mimic4_12x5_los7_standard_all.json"),
         os.path.join(TAB_REP, "ckks_mimic4_12x5_los7_standard_all.json"),
         os.path.join(TAB_REP, "ckks_lr_mimic4_12x5_los7_precise_all.json"),
         os.path.join(TAB_REP, "ckks_mimic4_12x5_los7_precise_all.json")),
        ("Image\n28\u00d728",
         os.path.join(IMG_REP, "ckks_malaria_28_lr_standard_all_27558.json"),
         os.path.join(IMG_REP, "ckks_malaria_28_cnn_standard_all_27558.json"),
         os.path.join(IMG_REP, "ckks_malaria_28_lr_precise_all_27558.json"),
         os.path.join(IMG_REP, "ckks_malaria_28_cnn_precise_all_27558.json")),
    ]

    vals = [[json.load(open(f))["per_split"]["test"]["class_flips"] for _, *fs in tasks for f in [fs[idx]]] for idx in range(4)]

    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    x = np.arange(len(tasks))
    w = 0.19
    for dx, v, lbl, col in [
        (-1.5 * w, vals[0], "LR, standard",  C_SLATE),
        (-0.5 * w, vals[1], "CNN, standard", C_NAVY),
        ( 0.5 * w, vals[2], "LR, precise",   C_LBLUE),
        ( 1.5 * w, vals[3], "CNN, precise",  C_MBLUE),
    ]:
        bars = ax.bar(x + dx, v, w, label=lbl, color=col, edgecolor="#222222", lw=0.7, zorder=3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.35, f"{int(h)}", ha="center", va="bottom", fontsize=8.5, zorder=4)

    ax.set_ylim(0, max(max(v) for v in vals) + 4)
    ax.set_ylabel("Test-split class flips")
    ax.set_xticks(x)
    ax.set_xticklabels([t[0] for t in tasks])
    ax.grid(axis="y", linestyle=":", alpha=0.55, zorder=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=4, frameon=False)

    fig.tight_layout(pad=0.5)
    fig.savefig(os.path.join(OUT_DIR, "fig_4.png"), bbox_inches="tight")
    plt.close(fig)


def make_fig5():
    """Fig. 5: Distribution of |encrypted - plaintext logit|."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.8))
    bins = np.logspace(-8, -1.3, 55)

    for lbl, path, col, ls in [
        ("LR, precise",   os.path.join(TAB_REP, "predictions_lr_mimic4_12x10_los7_precise_all.csv"),  "#9EB0C4", "-"),
        ("CNN, precise",  os.path.join(TAB_REP, "predictions_mimic4_12x10_los7_precise_all.csv"),     C_NAVY,    "-"),
        ("LR, standard",  os.path.join(TAB_REP, "predictions_lr_mimic4_12x10_los7_standard_all.csv"), C_SAND,    "--"),
        ("CNN, standard", os.path.join(TAB_REP, "predictions_mimic4_12x10_los7_standard_all.csv"),    C_RUST,    "--"),
    ]:
        df = pd.read_csv(path)
        err = np.clip(np.abs(df["logit_enc"].values - df["logit_plain"].values), 1e-8, None)
        ax1.hist(err, bins=bins, histtype="step", label=lbl, color=col, linestyle=ls, linewidth=2.0, zorder=3)
        ax1.hist(err, bins=bins, histtype="stepfilled", color=col, alpha=0.08, zorder=2)

    ax1.set_xscale("log")
    ax1.set_xlim(5e-9, 5e-2)
    ax1.set_xlabel("|encrypted \u2212 plaintext logit|")
    ax1.set_ylabel("Stays")
    ax1.set_title("(a) MIMIC-IV 12\u00d710 (all 35,794 stays)")
    ax1.grid(True, which="major", linestyle=":", alpha=0.5, zorder=0)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9, edgecolor="none")

    for lbl, path, col, ls in [
        ("CNN, precise",  os.path.join(IMG_REP, "predictions_malaria_28_cnn_precise_all_27558.csv"),  C_NAVY, "-"),
        ("CNN, standard", os.path.join(IMG_REP, "predictions_malaria_28_cnn_standard_all_27558.csv"), C_RUST, "--"),
    ]:
        df = pd.read_csv(path)
        err = np.clip(np.abs(df["enc_out"].values - df["plain_logit"].values), 1e-8, None)
        ax2.hist(err, bins=bins, histtype="step", label=lbl, color=col, linestyle=ls, linewidth=2.0, zorder=3)
        ax2.hist(err, bins=bins, histtype="stepfilled", color=col, alpha=0.10, zorder=2)

    ax2.set_xscale("log")
    ax2.set_xlim(5e-9, 5e-2)
    ax2.set_xlabel("|encrypted \u2212 plaintext logit|")
    ax2.set_ylabel("Images")
    ax2.set_title("(b) Image CNN (all 27,558 images)")
    ax2.grid(True, which="major", linestyle=":", alpha=0.5, zorder=0)
    ax2.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9, edgecolor="none")

    fig.tight_layout(pad=0.9)
    fig.savefig(os.path.join(OUT_DIR, "fig_5.png"), bbox_inches="tight")
    plt.close(fig)


def make_fig6():
    """Fig. 6: Image test metrics & confusion matrices."""
    lr_w  = json.load(open(os.path.join(IMG_MOD, "lr_malaria_28.json")))
    cnn_w = json.load(open(os.path.join(IMG_MOD, "cnn_malaria_28.json")))
    df_cnn = pd.read_csv(os.path.join(IMG_REP, "predictions_malaria_28_cnn_precise_all_27558.csv"))
    df_lr  = pd.read_csv(os.path.join(IMG_REP, "predictions_malaria_28_lr_precise_all_27558.csv"))
    df_cnn = df_cnn[df_cnn["split"] == "test"]
    df_lr  = df_lr[df_lr["split"] == "test"]

    def get_m(df, thr):
        y = df["y"].values
        p = 1.0 / (1.0 + np.exp(-df["plain_logit"].values))
        pred = (p >= thr).astype(int)
        return [
            roc_auc_score(y, df["plain_logit"].values),
            average_precision_score(y, p),
            accuracy_score(y, pred),
            precision_score(y, pred),
            recall_score(y, pred),
            f1_score(y, pred),
        ], confusion_matrix(y, pred)

    m_lr, cm_lr   = get_m(df_lr,  lr_w["threshold"])
    m_cnn, cm_cnn = get_m(df_cnn, cnn_w["threshold"])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.6, 3.6), gridspec_kw={"width_ratios": [1.35, 1.0, 1.0]})
    names = ["AUROC", "AUPRC", "Acc.", "Prec.", "Rec.", "F1"]
    x, w = np.arange(len(names)), 0.35
    b1 = ax1.bar(x - w / 2, m_lr,  w, label="LR",  color=C_SLATE, edgecolor="#222222", lw=0.7, zorder=3)
    b2 = ax1.bar(x + w / 2, m_cnn, w, label="CNN", color=C_NAVY,  edgecolor="#222222", lw=0.7, zorder=3)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.008, f"{h:.2f}",
                     ha="center", va="bottom", fontsize=7.8, rotation=90)
    ax1.set_ylim(0.50, 1.06)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_title("(a) Test metrics (y from 0.5)")
    ax1.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
    ax1.legend(loc="upper left", ncol=2, frameon=True, facecolor="white", framealpha=0.9, edgecolor="none")

    def plot_cm(ax, cm, title):
        cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
        ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100, aspect="equal")
        for r in range(2):
            for c in range(2):
                ax.text(c, r, f"{cm[r, c]}\n({cm_pct[r, c]:.1f}%)",
                        ha="center", va="center", color="white" if cm_pct[r, c] > 55 else "#111111", fontsize=9.2)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Uninf.", "Parasit."]); ax.set_yticklabels(["Uninf.", "Parasit."])
        ax.set_xlabel("Predicted"); ax.set_title(title)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color("#222222"); sp.set_linewidth(0.9)

    plot_cm(ax2, cm_cnn, "(b) CNN confusion")
    ax2.set_ylabel("True")
    plot_cm(ax3, cm_lr, "(c) LR confusion")

    fig.tight_layout(pad=0.9)
    fig.savefig(os.path.join(OUT_DIR, "fig_6.png"), bbox_inches="tight")
    plt.close(fig)


def make_fig7():
    """Fig. 7: Wall-clock cost for tabular and image inference."""
    configs = [("mimic3", "12x10", "MIMIC-III\n12\u00d710"),
               ("mimic3", "12x5",  "MIMIC-III\n12\u00d75"),
               ("mimic4", "12x10", "MIMIC-IV\n12\u00d710"),
               ("mimic4", "12x5",  "MIMIC-IV\n12\u00d75")]

    lr_s  = [json.load(open(os.path.join(TAB_REP, f"ckks_lr_{ds}_{g}_los7_standard_all.json")))["sec_per_patient_wall"] * 1000 for ds, g, _ in configs]
    cnn_s = [json.load(open(os.path.join(TAB_REP, f"ckks_{ds}_{g}_los7_standard_all.json")))["sec_per_patient_wall"] * 1000 for ds, g, _ in configs]
    lr_p  = [json.load(open(os.path.join(TAB_REP, f"ckks_lr_{ds}_{g}_los7_precise_all.json")))["sec_per_patient_wall"] * 1000 for ds, g, _ in configs]
    cnn_p = [json.load(open(os.path.join(TAB_REP, f"ckks_{ds}_{g}_los7_precise_all.json")))["sec_per_patient_wall"] * 1000 for ds, g, _ in configs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.8), gridspec_kw={"width_ratios": [1.3, 1.0]})
    x, w = np.arange(len(configs)), 0.20
    for dx, v, lbl, col in [
        (-1.5 * w, lr_s,  "LR, standard",  C_SLATE),
        (-0.5 * w, cnn_s, "CNN, standard", C_NAVY),
        ( 0.5 * w, lr_p,  "LR, precise",   C_SAND),
        ( 1.5 * w, cnn_p, "CNN, precise",  C_RUST),
    ]:
        bars = ax1.bar(x + dx, v, w, label=lbl, color=col, edgecolor="#222222", lw=0.7, zorder=3)
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h * 1.05, f"{int(round(h))}", ha="center", va="bottom", fontsize=8.0)

    ax1.set_yscale("log"); ax1.set_ylim(5, 500)
    ax1.set_ylabel("Wall-clock ms / patient (8 workers)"); ax1.set_title("(a) Tabular, per patient")
    ax1.set_xticks(x); ax1.set_xticklabels([c[2] for c in configs])
    ax1.grid(axis="y", which="major", linestyle=":", alpha=0.55, zorder=0)
    ax1.legend(loc="upper left", ncol=2, frameon=True, facecolor="white", framealpha=0.9, edgecolor="none", fontsize=8.3)

    # Panel (b): Controlled 2-worker comparison (per-patient vs SIMD batch)
    lr_pat  = json.load(open(os.path.join(IMG_REP, "ckks_malaria_28_lr_standard_test_4079.json")))["sec_per_patient_wall"] * 1000
    cnn_pat = json.load(open(os.path.join(IMG_REP, "ckks_malaria_28_cnn_standard_test_4079.json")))["sec_per_patient_wall"] * 1000
    lr_bat  = json.load(open(os.path.join(IMG_REP, "batch_malaria_28_lr_standard_all.json")))["sec_per_patient_wall"] * 1000
    cnn_bat = json.load(open(os.path.join(IMG_REP, "batch_malaria_28_cnn_standard_all.json")))["sec_per_patient_wall"] * 1000

    x2, w2 = np.arange(2), 0.34
    b_pat = ax2.bar(x2 - w2 / 2, [lr_pat, cnn_pat], w2, label="Per patient (2 workers)", color=C_SAND, edgecolor="#222222", lw=0.7, zorder=3)
    b_bat = ax2.bar(x2 + w2 / 2, [lr_bat, cnn_bat], w2, label="SIMD batch (2 workers)",  color=C_TEAL, edgecolor="#222222", lw=0.7, zorder=3)
    for bar in b_pat:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h * 1.07, f"{h:.1f}", ha="center", va="bottom", fontsize=8.8)
    for idx, bar in enumerate(b_bat):
        h = bar.get_height()
        spd = (lr_pat / lr_bat) if idx == 0 else (cnn_pat / cnn_bat)
        ax2.text(bar.get_x() + bar.get_width() / 2, h * 1.07, f"{h:.1f}\n({spd:.1f}\u00d7)",
                 ha="center", va="bottom", fontsize=8.4, color="#14455E", fontweight="bold")

    ax2.set_yscale("log"); ax2.set_ylim(0.6, 220)
    ax2.set_ylabel("Wall-clock ms / image (2 workers)"); ax2.set_title("(b) Image, standard preset (controlled)")
    ax2.set_xticks(x2); ax2.set_xticklabels(["LR", "CNN"])
    ax2.grid(axis="y", which="major", linestyle=":", alpha=0.55, zorder=0)
    ax2.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9, edgecolor="none", fontsize=8.3)

    fig.tight_layout(pad=0.9)
    fig.savefig(os.path.join(OUT_DIR, "fig_7.png"), bbox_inches="tight")
    plt.close(fig)


def make_fig8():
    """Fig. 8: Batched encrypted threshold on 4,079 test images."""
    files = [
        ("max (100%)", os.path.join(IMG_REP, "batch_malaria_28_cnn_deep_all_decision.json")),
        ("99th pct",   os.path.join(IMG_REP, "batch_malaria_28_cnn_deep_all_decision_p99.json")),
        ("95th pct",   os.path.join(IMG_REP, "batch_malaria_28_cnn_deep_all_decision_p95.json")),
    ]
    sat_pct, flip_cnt, flip_pct = [], [], []
    for _, path in files:
        d = json.load(open(path))["per_split"]["test"]
        sat_pct.append(d["saturated_fraction"] * 100)
        flip_cnt.append(d["class_flips"])
        flip_pct.append(d["class_flips"] / d["n"] * 100)

    fig, ax1 = plt.subplots(figsize=(8.4, 3.6))
    ax2 = ax1.twinx()
    x, w = np.arange(len(files)), 0.34
    b1 = ax1.bar(x - w / 2, sat_pct,  w, label="Saturated (score hidden), % (left axis)", color=C_TEAL, edgecolor="#222222", lw=0.7, zorder=3)
    b2 = ax2.bar(x + w / 2, flip_pct, w, label="Class flips, % of test images (right axis)", color=C_RED, edgecolor="#222222", lw=0.7, zorder=3)

    for bar in b1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.8, color="#14455E", fontweight="bold")
    for idx, bar in enumerate(b2):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{flip_cnt[idx]} ({h:.2f}%)", ha="center", va="bottom", fontsize=8.8, color="#8E2424", fontweight="bold")

    ax1.set_ylim(0, 28); ax2.set_ylim(0, 0.58)
    ax1.set_ylabel("Saturated outputs (%)", color="#14455E")
    ax2.set_ylabel("Test-split class flips (%)", color="#8E2424")
    ax2.spines["top"].set_visible(False)
    ax1.set_xlabel("Scale s"); ax1.set_xticks(x); ax1.set_xticklabels([f[0] for f in files])
    ax1.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", frameon=True, facecolor="white", framealpha=0.95, edgecolor="none")

    fig.tight_layout(pad=0.6)
    fig.savefig(os.path.join(OUT_DIR, "fig_8.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    for fn in (make_fig1, make_fig2, make_fig3, make_fig4, make_fig5, make_fig6, make_fig7, make_fig8):
        fn()
    print("Generated fig_1.png .. fig_8.png in", OUT_DIR)
