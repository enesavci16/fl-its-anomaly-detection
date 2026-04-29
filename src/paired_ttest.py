"""
paired_ttest.py
================
S1/S2 ve S2/S3 arasindaki F1 farklarinin istatistiksel anlamliligini test eder.
Multi-seed sonuclarindan okur.

Calistirma:
    source ~/fl-env/bin/activate
    cd ~/fl-its
    python src/paired_ttest.py

Cikti:
    results/ttest_results.json
"""

import json
import os
import numpy as np
from scipy import stats

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS_DIR  = os.path.join(REPO_ROOT, "results")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "multiseed_results.json")
OUT_PATH     = os.path.join(RESULTS_DIR, "ttest_results.json")

# ── Veri Yukle ────────────────────────────────────────────────────────────────

with open(SUMMARY_PATH) as f:
    data = json.load(f)

SEEDS = [42, 0, 1, 2, 3]

s2_f1s = [data["s2"][seed]["avg_global_f1"] for seed in [str(s) for s in SEEDS]]
s3_f1s = [data["s3"][seed]["avg_global_f1"] for seed in [str(s) for s in SEEDS]]

# S1 tek run — multi-seed yok, sabit deger
S1_F1 = 0.7715

# ── Testler ───────────────────────────────────────────────────────────────────

def run_ttest(a, b, label):
    """Paired t-test: a ve b ayni seed'lerden gelen paired olcumler."""
    t_stat, p_val = stats.ttest_rel(a, b)
    mean_diff = np.mean(np.array(a) - np.array(b))
    significant = p_val < 0.05
    print(f"\n  {label}")
    print(f"    Grup A: {[round(x,4) for x in a]}")
    print(f"    Grup B: {[round(x,4) for x in b]}")
    print(f"    Fark (A-B): mean={mean_diff:+.4f}")
    print(f"    t={t_stat:.4f}, p={p_val:.4f} "
          f"{'*** ANLAMLI (p<0.05)' if significant else '(p>=0.05, anlamli degil)'}")
    return {
        "t_statistic": round(float(t_stat), 4),
        "p_value":     round(float(p_val), 4),
        "mean_diff":   round(float(mean_diff), 4),
        "significant": bool(significant),
        "group_a":     [round(x, 4) for x in a],
        "group_b":     [round(x, 4) for x in b],
    }

# ── Ana Akis ──────────────────────────────────────────────────────────────────

print("=" * 60)
print("Paired t-test: FL Senaryo Karsilastirmasi")
print(f"Seeds: {SEEDS}  |  alpha=0.05")
print("=" * 60)

print(f"\n  S1 Centralised F1 = {S1_F1} (single run, sabit)")
print(f"  S2 IID FL F1s     = {[round(x,4) for x in s2_f1s]}")
print(f"  S3 Non-IID FL F1s = {[round(x,4) for x in s3_f1s]}")

results = {}

# Test 1: S2 vs S3
results["S2_vs_S3"] = run_ttest(
    s2_f1s, s3_f1s,
    "Test 1: S2 (IID FL) vs S3 (Non-IID FL)"
)

# Test 2: S2 vs S1 (S1 sabit, one-sample t-test)
t_stat, p_val = stats.ttest_1samp(s2_f1s, S1_F1)
mean_diff = np.mean(s2_f1s) - S1_F1
significant = p_val < 0.05
print(f"\n  Test 2: S2 vs S1 Centralised (one-sample t-test, mu={S1_F1})")
print(f"    S2 F1s: {[round(x,4) for x in s2_f1s]}")
print(f"    Fark (S2-S1): mean={mean_diff:+.4f}")
print(f"    t={t_stat:.4f}, p={p_val:.4f} "
      f"{'*** ANLAMLI (p<0.05)' if significant else '(p>=0.05, anlamli degil)'}")
results["S2_vs_S1"] = {
    "test_type":   "one_sample_ttest",
    "mu":          S1_F1,
    "t_statistic": round(float(t_stat), 4),
    "p_value":     round(float(p_val), 4),
    "mean_diff":   round(float(mean_diff), 4),
    "significant": bool(significant),
    "s2_f1s":      [round(x, 4) for x in s2_f1s],
}

# Test 3: S3 vs S1
t_stat, p_val = stats.ttest_1samp(s3_f1s, S1_F1)
mean_diff = np.mean(s3_f1s) - S1_F1
significant = p_val < 0.05
print(f"\n  Test 3: S3 vs S1 Centralised (one-sample t-test, mu={S1_F1})")
print(f"    S3 F1s: {[round(x,4) for x in s3_f1s]}")
print(f"    Fark (S3-S1): mean={mean_diff:+.4f}")
print(f"    t={t_stat:.4f}, p={p_val:.4f} "
      f"{'*** ANLAMLI (p<0.05)' if significant else '(p>=0.05, anlamli degil)'}")
results["S3_vs_S1"] = {
    "test_type":   "one_sample_ttest",
    "mu":          S1_F1,
    "t_statistic": round(float(t_stat), 4),
    "p_value":     round(float(p_val), 4),
    "mean_diff":   round(float(mean_diff), 4),
    "significant": bool(significant),
    "s3_f1s":      [round(x, 4) for x in s3_f1s],
}

# Kaydet
with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 60)
print("OZET")
print("=" * 60)
for test_name, res in results.items():
    sig = "ANLAMLI ***" if res["significant"] else "anlamli degil"
    print(f"  {test_name:<15}: p={res['p_value']:.4f}  [{sig}]  "
          f"mean_diff={res['mean_diff']:+.4f}")
print(f"\n  Sonuclar kaydedildi: {OUT_PATH}")
print("=" * 60)
