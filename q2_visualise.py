#!/usr/bin/env python3
"""
Results Visualisation for Corporate Bankruptcy Forecasting
Generates charts suitable for GitHub README and portfolio presentation.

Usage:
    python q2_visualise.py
    python q2_visualise.py --results q2_local_results.csv --outdir charts
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "figure.dpi":       150,
})

# ── Palette ───────────────────────────────────────────────────────────────────
COLORS = {
    "Baseline (no scaling, no PCA)":  "#1E2761",
    "StandardScaler only (no PCA)":   "#1A7A4A",
    "PCA only (scaled internally)":   "#C07A1A",
    "StandardScaler + PCA":           "#A0303A",
}
SHORT_NAMES = {
    "Baseline (no scaling, no PCA)":  "Baseline",
    "StandardScaler only (no PCA)":   "Scaled",
    "PCA only (scaled internally)":   "PCA Only",
    "StandardScaler + PCA":           "Scaled+PCA",
}


def load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["short_name"] = df["Experiment"].map(SHORT_NAMES)
    df["color"]      = df["Experiment"].map(COLORS)
    return df


# ── Chart 1: Metric comparison bar chart ─────────────────────────────────────
def plot_metric_comparison(df: pd.DataFrame, outdir: str):
    metrics = {
        "PR-AUC":    "TestPRAUC",
        "Recall":    "TestRecall",
        "Precision": "TestPrec",
        "F2 Score":  "TestF2",
        "MCC":       "TestMCC",
        "AUC-ROC":   "TestAUC",
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Test Metrics by Experiment\n(Corporate Bankruptcy Forecasting — Temporal Split)",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, (label, col) in zip(axes.flatten(), metrics.items()):
        bars = ax.bar(
            df["short_name"], df[col],
            color=df["color"], edgecolor="white", linewidth=0.8, width=0.55,
        )
        ax.set_title(label, fontweight="bold", fontsize=11)
        ax.set_ylim(0, min(df[col].max() * 1.25, 1.05))
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=20)

        for bar, val in zip(bars, df[col]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(outdir, "01_metric_comparison.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Chart 2: Confusion matrices ───────────────────────────────────────────────
def plot_confusion_matrices(df: pd.DataFrame, outdir: str):
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Confusion Matrices — Test Set\n(Corporate Bankruptcy Forecasting — Temporal Split)",
                 fontsize=13, fontweight="bold")

    for ax, (_, row) in zip(axes, df.iterrows()):
        tp, tn = int(row["Test_TP"]), int(row["Test_TN"])
        fp, fn = int(row["Test_FP"]), int(row["Test_FN"])
        total  = tp + tn + fp + fn

        # Standard layout:
        # row 0 = Actual Healthy,     col 0 = Predicted Healthy,    col 1 = Predicted Distressed
        # row 1 = Actual Distressed,  col 0 = Predicted Healthy,    col 1 = Predicted Distressed
        # [0,0]=TN  [0,1]=FP
        # [1,0]=FN  [1,1]=TP

        cell_data = [
            [(tn, "TN", "#EEF3FC", "#1A5276"),  (fp, "FP", "#FDECEA", "#922B21")],
            [(fn, "FN", "#FFF3E0", "#784212"),  (tp, "TP", "#E8F5E9", "#1A7A4A")],
        ]
        row_labels = ["Actual\nHealthy", "Actual\nDistressed"]
        col_labels = ["Predicted\nHealthy", "Predicted\nDistressed"]

        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)

        for i in range(2):
            for j in range(2):
                val, label, bg, tc = cell_data[i][j]
                pct = val / total * 100
                # i=0 → top row (y=1), i=1 → bottom row (y=0)
                y_pos = 1 - i
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, y_pos - 0.5), 1, 1,
                    color=bg, zorder=0,
                ))
                ax.text(j, y_pos,
                        f"{label}\n{val:,}\n({pct:.2f}%)",
                        ha="center", va="center",
                        fontsize=9, fontweight="bold", color=tc)

        ax.set_xticks([0, 1])
        ax.set_yticks([1, 0])   # y=1 → row 0 (Healthy), y=0 → row 1 (Distressed)
        ax.set_xticklabels(col_labels, fontsize=9)
        ax.set_yticklabels(row_labels, fontsize=9)
        ax.set_title(row["short_name"], fontweight="bold",
                     color=row["color"], fontsize=11)
        ax.grid(False)
        ax.spines[:].set_visible(False)

    plt.tight_layout()
    path = os.path.join(outdir, "02_confusion_matrices.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Chart 3: Scaling vs PCA effect ───────────────────────────────────────────
def plot_scaling_pca_effect(df: pd.DataFrame, outdir: str):
    metrics = ["PR-AUC", "F2 Score", "MCC", "Recall", "Precision"]
    cols    = ["TestPRAUC", "TestF2", "TestMCC", "TestRecall", "TestPrec"]

    baseline = df[df["short_name"] == "Baseline"].iloc[0]
    scaled   = df[df["short_name"] == "Scaled"].iloc[0]
    pca      = df[df["short_name"] == "Scaled+PCA"].iloc[0]

    x      = np.arange(len(metrics))
    width  = 0.28

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Effect of Scaling and PCA on Test Metrics",
                 fontsize=13, fontweight="bold")

    b1 = ax.bar(x - width, [baseline[c] for c in cols],
                width, label="Baseline", color=COLORS["Baseline (no scaling, no PCA)"],
                edgecolor="white")
    b2 = ax.bar(x,         [scaled[c]   for c in cols],
                width, label="Scaled",   color=COLORS["StandardScaler only (no PCA)"],
                edgecolor="white")
    b3 = ax.bar(x + width, [pca[c]      for c in cols],
                width, label="Scaled+PCA", color=COLORS["StandardScaler + PCA"],
                edgecolor="white")

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7.5,
                    fontweight="bold", rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10)
    ax.set_title("Scaling has minimal effect; PCA significantly degrades performance",
                 fontsize=10, style="italic", color="gray")

    plt.tight_layout()
    path = os.path.join(outdir, "03_scaling_pca_effect.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Chart 4: Training time ────────────────────────────────────────────────────
def plot_training_time(df: pd.DataFrame, outdir: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle("Training Time by Experiment", fontsize=13, fontweight="bold")

    bars = ax.barh(df["short_name"][::-1], df["TrainTime_s"][::-1],
                   color=df["color"][::-1], edgecolor="white", height=0.5)

    for bar, val in zip(bars, df["TrainTime_s"][::-1]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}s", va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Training Time (seconds)")
    ax.set_xlim(0, df["TrainTime_s"].max() * 1.2)
    ax.grid(axis="y", alpha=0)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(outdir, "04_training_time.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Chart 5: Summary dashboard ────────────────────────────────────────────────
def plot_dashboard(df: pd.DataFrame, outdir: str):
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#F8FAFF")
    gs  = GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.4)

    fig.suptitle(
        "Corporate Bankruptcy Forecasting — Results Dashboard\n"
        "Random Forest · 130 Features · Temporal Split (Train: 2006–2017 | Test: 2018–2020)",
        fontsize=13, fontweight="bold", y=0.98,
    )

    # ── Top row: 4 stat cards ─────────────────────────────────────────────────
    best = df[df["short_name"] == "Baseline"].iloc[0]
    stats = [
        ("Best Recall",    f"{best['TestRecall']:.1%}",  "Baseline",  "#1E2761"),
        ("Best MCC",       f"{best['TestMCC']:.3f}",     "Baseline",  "#1E2761"),
        ("Best PR-AUC",    f"{best['TestPRAUC']:.3f}",   "Baseline",  "#1E2761"),
        ("PCA PR-AUC Drop",f"−{best['TestPRAUC'] - df[df['short_name']=='Scaled+PCA'].iloc[0]['TestPRAUC']:.3f}",
         "vs Scaled+PCA", "#A0303A"),
    ]
    for i, (title, value, sub, color) in enumerate(stats):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor("white")
        ax.text(0.5, 0.65, value, transform=ax.transAxes,
                ha="center", va="center", fontsize=22, fontweight="bold", color=color)
        ax.text(0.5, 0.25, title, transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="#444")
        ax.text(0.5, 0.08, sub, transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="#888")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#D0D9F0")
            spine.set_linewidth(1.5)
            spine.set_visible(True)

    # ── Middle left: metric bars ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :2])
    metrics = ["TestPRAUC", "TestF2", "TestMCC", "TestRecall", "TestPrec"]
    mlabels = ["PR-AUC", "F2", "MCC", "Recall", "Precision"]
    x = np.arange(len(metrics))
    w = 0.2
    for idx, (_, row) in enumerate(df.iterrows()):
        ax2.bar(x + (idx - 1.5) * w,
                [row[m] for m in metrics],
                w, color=row["color"], label=row["short_name"],
                edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(mlabels)
    ax2.set_ylim(0, 1.1)
    ax2.set_title("Test Metrics by Experiment", fontweight="bold")
    ax2.legend(fontsize=8, loc="upper right")

    # ── Middle right: training time ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 2:])
    ax3.barh(df["short_name"][::-1], df["TrainTime_s"][::-1],
             color=df["color"][::-1], edgecolor="white", height=0.5)
    for i, (_, row) in enumerate(df[::-1].iterrows()):
        ax3.text(row["TrainTime_s"] + 1, i, f"{row['TrainTime_s']:.1f}s",
                 va="center", fontsize=9, fontweight="bold")
    ax3.set_xlabel("Seconds")
    ax3.set_title("Training Time", fontweight="bold")
    ax3.set_xlim(0, df["TrainTime_s"].max() * 1.25)
    ax3.grid(axis="y", alpha=0)

    # ── Bottom: confusion matrices ────────────────────────────────────────────
    for idx, (_, row) in enumerate(df.iterrows()):
        ax = fig.add_subplot(gs[2, idx])
        tp, tn = int(row["Test_TP"]), int(row["Test_TN"])
        fp, fn = int(row["Test_FP"]), int(row["Test_FN"])
        total  = tp + tn + fp + fn

        # Standard layout:
        # [0,0]=TN [0,1]=FP  → row 0 = Actual Healthy   (y=1)
        # [1,0]=FN [1,1]=TP  → row 1 = Actual Distressed (y=0)
        cell_data = [
            [(tn, "TN", 0.0),  (fp, "FP", 5.0)],
            [(fn, "FN", 10.0), (tp, "TP", 80.0)],
        ]

        # Draw colored background rectangles
        bg_colors = [["#EEF3FC", "#FDECEA"], ["#FFF3E0", "#E8F5E9"]]
        text_cols = [["#1A5276", "#922B21"], ["#784212", "#1A7A4A"]]

        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)

        for i in range(2):
            for j in range(2):
                val, label, _ = cell_data[i][j]
                y_pos = 1 - i   # i=0 → y=1 (top), i=1 → y=0 (bottom)
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, y_pos - 0.5), 1, 1,
                    color=bg_colors[i][j], zorder=0,
                ))
                ax.text(j, y_pos,
                        f"{label}\n{val:,}",
                        ha="center", va="center",
                        fontsize=8, fontweight="bold",
                        color=text_cols[i][j])

        ax.set_xticks([0, 1])
        ax.set_yticks([1, 0])
        ax.set_xticklabels(["Pred H", "Pred D"], fontsize=8)
        ax.set_yticklabels(
            ["Act\nHealthy", "Act\nDistressed"] if idx == 0 else ["", ""],
            fontsize=8
        )
        ax.set_title(row["short_name"], fontweight="bold",
                     color=row["color"], fontsize=9)
        ax.grid(False)
        ax.spines[:].set_visible(False)

    path = os.path.join(outdir, "05_dashboard.png")
    plt.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="q2_local_results.csv")
    parser.add_argument("--outdir",  default="charts")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"\n  Loading results from: {args.results}")
    df = load_results(args.results)

    print(f"  Generating charts into: {args.outdir}/\n")
    plot_metric_comparison(df, args.outdir)
    plot_confusion_matrices(df, args.outdir)
    plot_scaling_pca_effect(df, args.outdir)
    plot_training_time(df, args.outdir)
    plot_dashboard(df, args.outdir)

    print(f"\n  Done — {len(os.listdir(args.outdir))} charts saved to '{args.outdir}/'")
    print("  Use 05_dashboard.png as the hero image in your GitHub README.")


if __name__ == "__main__":
    main()
