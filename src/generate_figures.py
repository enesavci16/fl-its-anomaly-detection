"""
FL + ITS Ablasyon Çalışması — IEEE Kalite Görselleri
=====================================================
ASYU 2026 Bildirisi için fig10–fig13 üretimi.

Kullanım:
    python3 src/generate_figures.py [--results-dir RESULTS_DIR]

Veri kaynağı önceliği:
    1. results/ klasöründeki CSV/TXT dosyaları (öncelikli)
    2. Hardcoded fallback değerleri (dosya bulunamazsa)

Çıktı:
    results/fig10_ablation_f1.png
    results/fig11_client_drift.png
    results/fig12_comm_cost.png
    results/fig13_s1_metrics.png
"""

import os
import sys
import csv
import json
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ── Argüman ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="ASYU 2026 figure generator")
parser.add_argument("--results-dir", default="results",
                    help="results/ klasörü yolu (default: results/)")
args, _ = parser.parse_known_args()
RESULTS_DIR = args.results_dir
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── IEEE Görsel Ayarları ────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
})

# ── Renk Paleti (Paul Tol "Bright" — colorblind-safe) ──────────────────────
C_S1  = "#4477AA"
C_S2  = "#228833"
C_S3  = "#EE6677"
C_LOC = "#BBBBBB"


# ══════════════════════════════════════════════════════════════════════════
# VERİ YÜKLEME
# ══════════════════════════════════════════════════════════════════════════

def load_s1_metrics(results_dir):
    """s1_baseline_metrics.csv okur. Bulamazsa hardcoded fallback döner."""
    path = os.path.join(results_dir, "s1_baseline_metrics.csv")
    if os.path.exists(path):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            row = next(reader)
            # Kolon adlarını lowercase + strip
            row = {k.strip().lower(): v.strip() for k, v in row.items()}
            f1   = float(row.get("f1",        row.get("f1_score",   "0.7715")))
            prec = float(row.get("precision",                        "0.9892"))
            rec  = float(row.get("recall",                           "0.6323"))
        print(f"[✓] S1 metrikleri dosyadan yüklendi: {path}")
    else:
        print(f"[!] {path} bulunamadı — hardcoded fallback kullanılıyor.")
        f1, prec, rec = 0.7715, 0.9892, 0.6323
    return {"f1": f1, "precision": prec, "recall": rec}


def parse_fl_results(path):
    """
    s2_iid_results.txt veya s3_noniid_results.txt'i ayrıştırır.

    Beklenen format (her satır bir sensör):
        sensor_id,local_f1,global_f1
        278,0.7797,0.7717
        ...
    Ya da serbest metin içinde "Local F1: 0.xxxx" / "Global F1: 0.xxxx" satırları.
    İkisini de dener.
    """
    results = []
    if not os.path.exists(path):
        return None

    with open(path) as f:
        content = f.read()

    # ── Yöntem 1: CSV-like (header + rows) ──
    try:
        lines = [l.strip() for l in content.strip().splitlines() if l.strip()]
        # İlk satır header mı?
        if "sensor" in lines[0].lower() or "local" in lines[0].lower():
            reader = csv.DictReader(lines[0:1] + lines[1:], delimiter=",")
            for row in csv.DictReader(lines):
                row = {k.strip().lower(): v.strip() for k, v in row.items()}
                sid     = row.get("sensor_id", row.get("sensor", "?"))
                loc_f1  = float(row.get("local_f1",  row.get("local",  "0")))
                glo_f1  = float(row.get("global_f1", row.get("global", "0")))
                results.append((sid, loc_f1, glo_f1))
            if results:
                return results
    except Exception:
        pass

    # ── Yöntem 2: Serbest metin — "Sensor 278" + "Local F1:" + "Global F1:" ──
    import re
    sensor_blocks = re.split(r"(?=Sensor\s+\d+)", content, flags=re.IGNORECASE)
    for block in sensor_blocks:
        sid_m = re.search(r"Sensor\s+(\d+)", block, re.IGNORECASE)
        loc_m = re.search(r"Local\s+F1\s*[=:]\s*([\d.]+)", block, re.IGNORECASE)
        glo_m = re.search(r"Global\s+F1\s*[=:]\s*([\d.]+)", block, re.IGNORECASE)
        if sid_m and loc_m and glo_m:
            results.append((sid_m.group(1),
                            float(loc_m.group(1)),
                            float(glo_m.group(1))))
    if results:
        return results

    return None  # Ayrıştırılamadı


def load_fl_metrics(results_dir):
    """S2/S3 per-sensör metriklerini multiseed_results.json seed-42'den yükler.
    Bu, Table IV ile aynı kaynaktan beslenmeyi sağlar (tek otoriter kaynak)."""
    sensor_labels = ["S278 (P25)", "S240 (P50)", "S71 (P75)", "S298 (P95)"]
    SENSOR_IDS = ["278", "240", "71", "298"]

    ms_path = os.path.join(results_dir, "multiseed_results.json")
    try:
        with open(ms_path) as f:
            ms = json.load(f)
        seed_key = "42"  # seed=42, Table IV ile aynı
        s2_42 = ms["s2"][seed_key]["sensor"]
        s3_42 = ms["s3"][seed_key]["sensor"]
        s2_local  = [s2_42[sid]["local_f1"]  for sid in SENSOR_IDS]
        s2_global = [s2_42[sid]["global_f1"] for sid in SENSOR_IDS]
        s3_local  = [s3_42[sid]["local_f1"]  for sid in SENSOR_IDS]
        s3_global = [s3_42[sid]["global_f1"] for sid in SENSOR_IDS]
        print(f"[✓] Per-sensör metrikler {ms_path} (seed=42) dosyasından yüklendi.")
    except Exception as e:
        print(f"[!] multiseed_results.json yüklenemedi ({e}) — hardcoded fallback.")
        s2_local  = [0.7500, 0.7009, 0.7563, 0.7097]
        s2_global = [0.8033, 0.7130, 0.7368, 0.7097]
        s3_local  = [0.5000, 0.5778, 0.5574, 0.4898]
        s3_global = [0.5217, 0.4444, 0.5574, 0.6872]

    return {
        "labels":    sensor_labels,
        "s2_local":  s2_local,
        "s2_global": s2_global,
        "s3_local":  s3_local,
        "s3_global": s3_global,
    }


# ── Veri yükle ──────────────────────────────────────────────────────────────
s1 = load_s1_metrics(RESULTS_DIR)
fl = load_fl_metrics(RESULTS_DIR)

s2_avg_global = float(np.mean(fl["s2_global"]))
s3_avg_global = float(np.mean(fl["s3_global"]))

print(f"\n── Yüklenen değerler ──────────────────────────────")
print(f"S1  F1 = {s1['f1']:.4f}  P = {s1['precision']:.4f}  R = {s1['recall']:.4f}")
print(f"S2  avg global F1 = {s2_avg_global:.4f}")
print(f"S3  avg global F1 = {s3_avg_global:.4f}")
print(f"───────────────────────────────────────────────────\n")


# ══════════════════════════════════════════════════════════════════════════
# FIG10 — Ana Ablasyon Bar Chart (S1 / S2 / S3 ortalama F1)
# ══════════════════════════════════════════════════════════════════════════
def make_fig10():
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    scenarios = ["S1\nCentralized", "S2\nIID FL", "S3\nNon-IID FL"]
    avg_f1s   = [s1["f1"], s2_avg_global, s3_avg_global]
    colors    = [C_S1, C_S2, C_S3]
    x         = np.arange(len(scenarios))

    bars = ax.bar(x, avg_f1s, width=0.50, color=colors,
                  edgecolor="black", linewidth=0.6, zorder=3)

    for bar, val in zip(bars, avg_f1s):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.013,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold")

    # S1 referans çizgisi
    ax.axhline(s1["f1"], color=C_S1, linestyle=":", linewidth=1.0, alpha=0.7, zorder=2)

    # Yüzde delta okları
    for col_idx, avg_val in [(1, s2_avg_global), (2, s3_avg_global)]:
        delta_pct = (avg_val - s1["f1"]) / s1["f1"] * 100
        ax.annotate("", xy=(col_idx, avg_val), xytext=(col_idx, s1["f1"]),
                    arrowprops=dict(arrowstyle="<->", color="gray", lw=0.8))
        ax.text(col_idx + 0.28, (avg_val + s1["f1"]) / 2,
                f"{delta_pct:+.1f}%", fontsize=7.5,
                color="dimgray", va="center")

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Average F1 Score")
    ax.set_ylim(0.40, 0.90)
    ax.set_title("Ablation Study: Anomaly Detection F1 Score\nAcross Three Scenarios (PeMS04, 4 Sensors)")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "fig10_ablation_f1.png")
    plt.savefig(out)
    plt.close()
    print(f"[✓] fig10 kaydedildi: {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════
# FIG11 — Client Drift: Local vs Global F1 (S2 + S3 yan yana)
# ══════════════════════════════════════════════════════════════════════════
def make_fig11():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)
    x  = np.arange(len(fl["labels"]))
    bw = 0.32

    for ax, local_f1, global_f1, title, color_glo in zip(
        axes,
        [fl["s2_local"],  fl["s3_local"]],
        [fl["s2_global"], fl["s3_global"]],
        ["(a) S2 — IID FL", "(b) S3 — Non-IID FL"],
        [C_S2, C_S3],
    ):
        ax.bar(x - bw/2, local_f1,  width=bw, label="Local (pre-agg.)",
               color=C_LOC, edgecolor="black", linewidth=0.6, zorder=3)
        ax.bar(x + bw/2, global_f1, width=bw, label="Global (federated)",
               color=color_glo, edgecolor="black", linewidth=0.6, zorder=3)

        for i, (loc, glo) in enumerate(zip(local_f1, global_f1)):
            diff = glo - loc
            sign = "+" if diff >= 0 else ""
            fc   = "#228833" if diff >= 0 else "#CC3311"
            ax.text(x[i], max(loc, glo) + 0.025,
                    f"{sign}{diff:.3f}",
                    ha="center", va="bottom",
                    fontsize=7, color=fc, fontweight="bold")

        ax.axhline(np.mean(local_f1),  color=C_LOC,    linestyle=":",  lw=1.0, alpha=0.8)
        ax.axhline(np.mean(global_f1), color=color_glo, linestyle="--", lw=1.0, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(fl["labels"])
        ax.set_title(title)
        ax.set_xlabel("Sensor ID")
        ax.set_ylim(0.30, 0.98)
        ax.set_axisbelow(True)
        ax.legend(loc="lower right", framealpha=0.85)

    axes[0].set_ylabel("F1 Score")
    fig.suptitle("Client Drift Analysis: Local vs. Federated F1 Score per Sensor",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "fig11_client_drift.png")
    plt.savefig(out)
    plt.close()
    print(f"[✓] fig11 kaydedildi: {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════
# FIG12 — İletişim Maliyeti (Merkezi vs FL)
# ══════════════════════════════════════════════════════════════════════════
def make_fig12():
    import matplotlib.ticker as ticker
    comm_s1_raw = 21 * 1024
    comm_fl_c2s = 1.33
    comm_fl_s2c = 0.45
    total_fl    = comm_fl_c2s + comm_fl_s2c
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    x  = np.arange(3)
    bw = 0.45
    ax.bar(0, comm_s1_raw, width=bw, color="#CC3311", edgecolor="black",
           linewidth=0.6, hatch="////", label="Raw sensor data (S1 only)", zorder=3)
    ax.bar([1, 2], [comm_fl_c2s, comm_fl_c2s], width=bw, color="#4477AA",
           edgecolor="black", linewidth=0.6, label="Offset: client->server", zorder=3)
    ax.bar([1, 2], [comm_fl_s2c, comm_fl_s2c], width=bw,
           bottom=[comm_fl_c2s, comm_fl_c2s], color="#BBCC33",
           edgecolor="black", linewidth=0.6, label="Threshold: server->client", zorder=3)
    ax.text(-0.3, comm_s1_raw * 1.4, "~21 MB (est.)",
            ha="left", va="bottom", fontsize=8, fontweight="bold", color="#CC3311")
    for xi in [1, 2]:
        ax.text(xi, total_fl * 3.5, "1.33+0.45=1.78 KB",
                ha="center", va="bottom", fontsize=8)
    ax.annotate("~11800x reduction vs S1",
                xy=(0.5, total_fl * 10), xytext=(0.3, 400),
                fontsize=8.5, color="#228833", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#228833", lw=1.0), ha="left")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(["S1 Centralized", "S2 IID FL", "S3 Non-IID FL"])
    ax.set_ylabel("Data Transmitted (KB)")
    ax.set_ylim(0.1, comm_s1_raw * 60)
    ax.set_title("Communication Cost (4 Clients x 3 FL Rounds, PeMS04)")
    ax.legend(loc="upper right", framealpha=0.90, fontsize=7.5)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_axisbelow(True)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "fig12_comm_cost.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[*] fig12 kaydedildi: {out}")
    return out

def make_fig13():
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    metrics = ["Precision", "Recall", "F1"]
    vals    = [s1["precision"], s1["recall"], s1["f1"]]
    colors  = [C_S1, C_S2, C_S3]
    x       = np.arange(len(metrics))

    bars = ax.bar(x, vals, width=0.45, color=colors,
                  edgecolor="black", linewidth=0.6, zorder=3)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.013,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.15)
    ax.set_title("S1 Centralized Baseline:\nClassification Metrics (PeMS04, Test Set)")
    ax.set_axisbelow(True)

    ax.text(0.5, 0.08,
            "High Precision, Low Recall:\nConservative IF threshold behavior",
            transform=ax.transAxes, ha="center", fontsize=7.5,
            style="italic", color="dimgray",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "fig13_s1_metrics.png")
    plt.savefig(out)
    plt.close()
    print(f"[✓] fig13 kaydedildi: {out}")
    return out


def make_fig14():
    """FL stability (convergence) eğrisi — convergence_summary.json'dan okunur."""
    cv_path = os.path.join(RESULTS_DIR, "convergence_summary.json")
    try:
        with open(cv_path) as f:
            cv = json.load(f)
        s2_means = [cv["s2"][f"round_{r}"]["mean"] for r in [1, 2, 3]]
        s2_stds  = [cv["s2"][f"round_{r}"]["std"]  for r in [1, 2, 3]]
        s3_means = [cv["s3"][f"round_{r}"]["mean"] for r in [1, 2, 3]]
        s3_stds  = [cv["s3"][f"round_{r}"]["std"]  for r in [1, 2, 3]]
        print(f"[✓] Convergence verisi {cv_path} dosyasından yüklendi.")
    except Exception as e:
        print(f"[!] convergence_summary.json yüklenemedi ({e}) — fallback.")
        s2_means = [0.7422, 0.7266, 0.7126]
        s2_stds  = [0.0258, 0.0166, 0.0112]
        s3_means = [0.5865, 0.6050, 0.5890]
        s3_stds  = [0.0361, 0.0688, 0.0227]

    rounds = [1, 2, 3]
    s1_f1  = 0.7715

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.errorbar(rounds, s2_means, yerr=s2_stds,
                fmt="s-", color=C_S2, linewidth=1.6, markersize=6,
                capsize=4,
                label=f"S2 IID FL (final: {s2_means[2]:.3f}±{s2_stds[2]:.3f})",
                zorder=4)
    ax.errorbar(rounds, s3_means, yerr=s3_stds,
                fmt="^-", color=C_S3, linewidth=1.6, markersize=6,
                capsize=4,
                label=f"S3 Non-IID FL (final: {s3_means[2]:.3f}±{s3_stds[2]:.3f})",
                zorder=4)
    ax.axhline(y=s1_f1, color=C_S1, linestyle="--", linewidth=1.2,
               label="S1 Centralised (0.772)", zorder=3)

    ax.set_xticks(rounds)
    ax.set_xlabel("Communication Round", fontsize=11)
    ax.set_ylabel("Average F1 Score", fontsize=11)
    ax.set_ylim(0.45, 0.90)
    ax.set_title("FL Stability: Avg F1 per Round\n(mean ± std over 5 seeds)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5, framealpha=0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "fig14_convergence_curve.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] fig14 kaydedildi: {out}")
    return out

# ══════════════════════════════════════════════════════════════════════════
# ÇALIŞTIR
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    outputs = []
    outputs.append(make_fig10())
    outputs.append(make_fig11())
    outputs.append(make_fig12())
    outputs.append(make_fig13())

    print("\n── Özet ────────────────────────────────────────────")
    for o in outputs:
        print(f"  {o}")
    print("────────────────────────────────────────────────────")
    print("✓ Tüm IEEE görsel dosyaları oluşturuldu.")
