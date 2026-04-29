"""
generate_fig08.py
==================
S1 Isolation Forest confusion matrix - English labels.
Hardcoded values from baseline experiment:
  TP=184, FP=2, FN=107, TN=9907

Calistirma:
    source ~/fl-env/bin/activate
    cd ~/fl-its
    python src/generate_fig08.py

Cikti:
    results/fig08_s1_confusion_matrix.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
OUT_PATH    = os.path.join(RESULTS_DIR, "fig08_s1_confusion_matrix.png")

# ── Stil ──────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      11,
    "figure.dpi":     300,
    "savefig.dpi":    300,
    "savefig.bbox":   "tight",
})

# ── Confusion Matrix Degerleri ────────────────────────────────────────────────

TN, FP = 9907, 2
FN, TP = 107,  184

cm = np.array([[TN, FP],
               [FN, TP]])

labels = np.array([["TN\n9907", "FP\n2"],
                   ["FN\n107",  "TP\n184"]])

# ── Ciz ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(4.0, 3.5))

# Renk haritasi — koyu = yuksek deger
im = ax.imshow(cm, interpolation="nearest",
               cmap=plt.cm.Blues)

# Deger etiketleri
thresh = cm.max() / 2.0
for i in range(2):
    for j in range(2):
        ax.text(j, i, labels[i, j],
                ha="center", va="center",
                fontsize=13, fontweight="bold",
                color="white" if cm[i, j] > thresh else "black")

# Eksenler
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Normal", "Anomaly"], fontsize=11)
ax.set_yticklabels(["Normal", "Anomaly"], fontsize=11)
ax.set_xlabel("Predicted Label", fontsize=11)
ax.set_ylabel("True Label", fontsize=11)
ax.set_title(f"S1 — Centralised IF (Test F1 = 0.771)", fontsize=11)

plt.tight_layout()
plt.savefig(OUT_PATH)
plt.close()

print(f"fig08 kaydedildi: {OUT_PATH}")
print(f"  TP={TP}, FP={FP}, FN={FN}, TN={TN}")
print(f"  Precision={TP/(TP+FP):.4f}, Recall={TP/(TP+FN):.4f}")
