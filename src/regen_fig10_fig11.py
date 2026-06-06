#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os, shutil

RESULTS = '/home/enes/fl-its/results'
ASYU    = '/home/enes/fl-its/asyu_figures'
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(ASYU,    exist_ok=True)

BLUE  = '#0077BB'
RED   = '#CC3311'
GREEN = '#009988'
GRAY  = '#BBBBBB'

plt.rcParams.update({'font.family': 'serif', 'font.size': 11})

# ── FIG 10 ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4.5))
scenarios = ['S1\nCentralized', 'S2\nIID FL', 'S3\nNon-IID FL']
f1_values = [0.7715, 0.7407, 0.5527]
colors    = [BLUE, GREEN, RED]

bars = ax.bar(scenarios, f1_values, width=0.45,
              color=colors, edgecolor='white', linewidth=0.8, zorder=3)

for bar, val in zip(bars, f1_values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.008,
            f'{val:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.annotate('', xy=(1, f1_values[1] + 0.01),
            xytext=(0, f1_values[0] + 0.01),
            arrowprops=dict(arrowstyle='->', color='#333333', lw=1.2))
ax.text(0.5, max(f1_values[0], f1_values[1]) + 0.038,
        '-9.3%', ha='center', fontsize=9, color='#333333')

ax.annotate('', xy=(2, f1_values[2] + 0.01),
            xytext=(0, f1_values[0] + 0.01),
            arrowprops=dict(arrowstyle='->', color='#333333', lw=1.2))
ax.text(1.5, f1_values[0] + 0.038,
        '-27.5%', ha='center', fontsize=9, color='#333333')

ax.axhline(y=f1_values[0], color=BLUE, linestyle='--',
           linewidth=1.0, alpha=0.5, zorder=2)
ax.set_ylim(0.4, 0.95)
ax.set_ylabel('Average F1 Score', fontsize=12)
ax.set_title('Ablation Study: Anomaly Detection F1 Score\n'
             'Across Three Scenarios (PeMS04, 4 Sensors)',
             fontsize=12, fontweight='bold', pad=8)
ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{RESULTS}/fig10.png', dpi=300, bbox_inches='tight')
plt.close()
print('fig10 kaydedildi ✓')

# ── FIG 11 ──────────────────────────────────────────────────────────
sensor_labels = ['S278\n(P25)', 'S240\n(P50)', 'S71\n(P75)', 'S298\n(P95)']

s2_local  = [0.7797, 0.7130, 0.7903, 0.6935]
s2_global = [0.7717, 0.8030, 0.6038, 0.6325]
s2_delta  = [g - l for g, l in zip(s2_global, s2_local)]

s3_local  = [0.5000, 0.4576, 0.5574, 0.7210]
s3_global = [0.5217, 0.4217, 0.5885, 0.7456]
s3_delta  = [g - l for g, l in zip(s3_global, s3_local)]

x = np.arange(len(sensor_labels))
w = 0.32

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
fig.suptitle('Client Drift Analysis: Local vs. Federated F1 Score per Sensor',
             fontsize=13, fontweight='bold', y=1.01)

for ax, local, glob_, delta, title in [
        (axes[0], s2_local, s2_global, s2_delta, '(a) S2 — IID FL'),
        (axes[1], s3_local, s3_global, s3_delta, '(b) S3 — Non-IID FL')]:

    ax.bar(x - w/2, local, width=w, color=GRAY,
           edgecolor='white', linewidth=0.7,
           label='Local (pre-agg.)', zorder=3)
    ax.bar(x + w/2, glob_, width=w,
           color=[GREEN if d >= 0 else RED for d in delta],
           edgecolor='white', linewidth=0.7,
           label='Global (federated)', zorder=3)

    for i, (xl, xg, d) in enumerate(zip(x - w/2, x + w/2, delta)):
        sign  = '+' if d >= 0 else ''
        color = GREEN if d >= 0 else RED
        ypos  = max(local[i], glob_[i]) + 0.015
        ax.text((xl + xg)/2, ypos,
                f'{sign}{d:.3f}',
                ha='center', va='bottom', fontsize=8.5,
                fontweight='bold', color=color)

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sensor_labels, fontsize=9.5)
    ax.set_xlabel('Sensor ID', fontsize=10)
    ax.set_ylim(0.30, 0.98)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[0].set_ylabel('F1 Score', fontsize=11)

legend_patches = [
    mpatches.Patch(facecolor=GRAY,  edgecolor='white', label='Local (pre-agg.)'),
    mpatches.Patch(facecolor=GREEN, edgecolor='white', label='Global — positive drift'),
    mpatches.Patch(facecolor=RED,   edgecolor='white', label='Global — negative drift'),
]
fig.legend(handles=legend_patches, loc='lower center',
           ncol=3, fontsize=9, framealpha=0.9,
           bbox_to_anchor=(0.5, -0.08))

plt.tight_layout()
plt.savefig(f'{RESULTS}/fig11.png', dpi=300, bbox_inches='tight')
plt.close()
print('fig11 kaydedildi ✓')

# ── asyu_figures'a kopyala ───────────────────────────────────────────
shutil.copy(f'{RESULTS}/fig10.png', f'{ASYU}/fig10_ablation_f1.png')
shutil.copy(f'{RESULTS}/fig11.png', f'{ASYU}/fig11_client_drift.png')
print('asyu_figures kopyalandı ✓')
