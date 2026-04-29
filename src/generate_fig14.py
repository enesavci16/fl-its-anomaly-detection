"""
generate_fig14.py
==================
Convergence curve figure (fig14) uretir.
convergence_summary.json'dan okur, mean +- std band olarak gosterir.

Calistirma:
    source ~/fl-env/bin/activate
    cd ~/fl-its
    python src/generate_fig14.py

Cikti:
    results/fig14_convergence_curve.png
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS_DIR  = os.path.join(REPO_ROOT, "results")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "convergence_summary.json")
OUT_PATH     = os.path.join(RESULTS_DIR, "fig14_convergence_curve.png")

# ── Stil (IEEE uyumlu) ────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})

# Paul Tol renk paleti (colorblind-safe)
COLOR_S2 = "#4477AA"   # mavi  — IID
COLOR_S3 = "#EE6677"   # kirmizi — Non-IID
COLOR_S1 = "#228833"   # yesil — S1 baseline referans cizgisi

# ── Veri Yukle ────────────────────────────────────────────────────────────────

with open(SUMMARY_PATH) as f:
    summary = json.load(f)

N_ROUNDS = len(summary["s2"])
rounds   = list(range(1, N_ROUNDS + 1))

s2_mean = [summary["s2"][f"round_{r}"]["mean"] for r in rounds]
s2_std  = [summary["s2"][f"round_{r}"]["std"]  for r in rounds]
s3_mean = [summary["s3"][f"round_{r}"]["mean"] for r in rounds]
s3_std  = [summary["s3"][f"round_{r}"]["std"]  for r in rounds]

s2_mean = np.array(s2_mean)
s2_std  = np.array(s2_std)
s3_mean = np.array(s3_mean)
s3_std  = np.array(s3_std)

S1_F1 = 0.7715   # centralised baseline

# ── Ciz ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(5.5, 3.8))

# S1 baseline referans cizgisi
ax.axhline(S1_F1, color=COLOR_S1, linewidth=1.4,
           linestyle="--", alpha=0.85, zorder=1)
ax.text(0.02, S1_F1 + 0.006,
        f"S1 Centralised ({S1_F1:.3f})",
        transform=ax.get_yaxis_transform(),
        color=COLOR_S1, fontsize=9, va="bottom", ha="left")

# S2 IID — mean cizgisi + std bandi
ax.fill_between(rounds,
                s2_mean - s2_std,
                s2_mean + s2_std,
                color=COLOR_S2, alpha=0.18, zorder=2)
ax.plot(rounds, s2_mean,
        color=COLOR_S2, linewidth=2.0,
        marker="o", markersize=6,
        label=f"S2 IID FL  (final: {s2_mean[-1]:.3f} $\\pm$ {s2_std[-1]:.3f})",
        zorder=3)

# S3 Non-IID — mean cizgisi + std bandi
ax.fill_between(rounds,
                s3_mean - s3_std,
                s3_mean + s3_std,
                color=COLOR_S3, alpha=0.18, zorder=2)
ax.plot(rounds, s3_mean,
        color=COLOR_S3, linewidth=2.0,
        marker="s", markersize=6,
        label=f"S3 Non-IID FL  (final: {s3_mean[-1]:.3f} $\\pm$ {s3_std[-1]:.3f})",
        zorder=3)

# Eksen ayarlari
ax.set_xlabel("Communication Round")
ax.set_ylabel("Average F1 Score")
ax.set_title("FL Convergence: F1 Score per Round\n"
             "(mean $\\pm$ std over 5 seeds)")
ax.set_xticks(rounds)
ax.set_xlim(0.7, N_ROUNDS + 0.5)
ax.set_ylim(0.45, 0.85)
ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
ax.spines[["top", "right"]].set_visible(False)

# Legend
ax.legend(loc="lower right", framealpha=0.9,
          edgecolor="#cccccc", fontsize=9)

# Shaded band aciklamasi
band_patch = mpatches.Patch(color="gray", alpha=0.25,
                             label="$\\pm$1 std (5 seeds)")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles + [band_patch],
          labels=labels + ["$\\pm$1 std (5 seeds)"],
          loc="lower right", framealpha=0.9,
          edgecolor="#cccccc", fontsize=9)

plt.tight_layout()
plt.savefig(OUT_PATH)
plt.close()

print(f"fig14 kaydedildi: {OUT_PATH}")
