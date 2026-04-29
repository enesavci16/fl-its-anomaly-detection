"""
lof_baseline.py
================
Local Outlier Factor (LOF) ile merkezi baseline karsilastirmasi.
S1 (Isolation Forest) ile ayni veri ve split kullanilir.

Calistirma:
    source ~/fl-env/bin/activate
    cd ~/fl-its
    python src/lof_baseline.py

Cikti:
    results/lof_baseline_results.json
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score

# ── Sabitler ──────────────────────────────────────────────────────────────────

SENSOR_IDS    = [278, 240, 71, 298]
SENSOR_FILE_IDS = {278: "278", 240: "240", 71: "071", 298: "298"}
FEATURES      = ["flow", "speed", "occupancy"]
CONTAMINATION = 0.00826   # S1 ile ayni

REPO_ROOT   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR    = os.path.join(REPO_ROOT, "data", "processed")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Veri Yukleme ──────────────────────────────────────────────────────────────

def load_all_pooled():
    """
    Tum sensorlerin train/test verisini birlestir (S1 ile ayni).
    """
    X_trains, X_tests, y_tests = [], [], []

    for sid in SENSOR_IDS:
        fid  = SENSOR_FILE_IDS[sid]
        path = os.path.join(DATA_DIR, f"labels_sensor_{fid}.csv")
        df   = pd.read_csv(path)

        train = df[df["split"] == "train"]
        test  = df[df["split"] == "test"]

        X_trains.append(train[FEATURES].values)
        X_tests.append(test[FEATURES].values)
        y_tests.append(test["is_anomaly"].values)

        print(f"  Sensor {sid}: train={len(train)}, test={len(test)}, "
              f"anomaly={int(test['is_anomaly'].sum())}")

    X_train = np.vstack(X_trains)
    X_test  = np.vstack(X_tests)
    y_test  = np.concatenate(y_tests)

    return X_train, X_test, y_test


# ── Model Degerlendirme ────────────────────────────────────────────────────────

def evaluate(y_true, y_pred):
    return {
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
    }


def run_if(X_train, X_test, y_test):
    """S1 ile ayni Isolation Forest — referans."""
    model = IsolationForest(
        n_estimators=100,
        max_samples="auto",
        contamination=CONTAMINATION,
        random_state=42,
    )
    model.fit(X_train)
    preds = (model.predict(X_test) == -1).astype(int)
    return evaluate(y_test, preds)


def run_lof(X_train, X_test, y_test):
    """
    LOF novelty detection modu:
    novelty=True ile fit(train) -> predict(test) yapilir.
    contamination S1 ile ayni tutulur.
    """
    model = LocalOutlierFactor(
        n_neighbors=20,
        contamination=CONTAMINATION,
        novelty=True,       # test setinde predict() icin gerekli
    )
    model.fit(X_train)
    # LOF: -1 anomali, 1 normal → 0/1'e cevir
    preds = (model.predict(X_test) == -1).astype(int)
    return evaluate(y_test, preds)


# ── Ana Akis ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Non-FL Baseline Karsilastirmasi: IF vs LOF")
    print(f"Contamination: {CONTAMINATION}  |  Merkezi (pooled) egitim")
    print("=" * 60)

    print("\n[1/2] Veri yukleniyor (pooled)...")
    X_train, X_test, y_test = load_all_pooled()
    print(f"\n  Toplam: train={len(X_train)}, test={len(X_test)}, "
          f"anomaly={int(y_test.sum())} ({100*y_test.mean():.2f}%)")

    print("\n[2/2] Modeller egitiliyor ve degerlendirilyor...")

    if_results  = run_if(X_train, X_test, y_test)
    lof_results = run_lof(X_train, X_test, y_test)

    # Sonuclari kaydet
    results = {
        "S1_IsolationForest": if_results,
        "S1_LOF":             lof_results,
        "contamination":      CONTAMINATION,
        "n_train":            int(len(X_train)),
        "n_test":             int(len(X_test)),
        "n_anomaly_test":     int(y_test.sum()),
    }

    out_path = os.path.join(RESULTS_DIR, "lof_baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Terminal ozeti
    print("\n" + "=" * 60)
    print("SONUCLAR")
    print("=" * 60)
    print(f"  {'Model':<25} {'F1':>6}  {'Precision':>10}  {'Recall':>8}")
    print(f"  {'-'*23:<25} {'-'*6:>6}  {'-'*10:>10}  {'-'*8:>8}")
    for name, res in [("S1 Isolation Forest", if_results),
                      ("S1 LOF (n_neighbors=20)", lof_results)]:
        print(f"  {name:<25} {res['f1']:>6.4f}  "
              f"{res['precision']:>10.4f}  {res['recall']:>8.4f}")
    print("=" * 60)

    delta = lof_results["f1"] - if_results["f1"]
    winner = "IF" if delta < 0 else "LOF"
    print(f"\n  F1 farki (LOF - IF): {delta:+.4f} → {winner} daha iyi")
    print(f"  Sonuclar kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
