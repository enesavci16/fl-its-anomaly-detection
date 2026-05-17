"""
fedavg_multiseed.py  (v2)
==========================
Multi-seed evaluation of S2 (IID FL) and S3 (Non-IID FL) scenarios.

Calistirma:
    source ~/fl-env/bin/activate
    cd ~/fl-its
    python src/fedavg_multiseed.py

Cikti:
    results/multiseed_results.json   -- ham seed-bazli sonuclar
    results/multiseed_summary.json   -- mean +- std ozet
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score

# ── Sabitler ──────────────────────────────────────────────────────────────────

SENSOR_IDS    = [278, 240, 71, 298]
SEEDS         = [42, 0, 1, 2, 3]
N_ROUNDS      = 3
CONTAMINATION = 0.00826
FEATURES = ["flow", "occupancy", "speed"]

IID_DIR    = "/home/enes/fl-its-data/data/splits/iid"
NONIID_DIR = "/home/enes/fl-its-data/data/splits/non_iid"

REPO_ROOT   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Veri Yukleme ──────────────────────────────────────────────────────────────

def load_client_data(client_idx: int, scenario: str):
    """
    IID veya Non-IID client verisini yukle.
    scenario: 'iid' veya 'non_iid'
    """
    base = IID_DIR if scenario == "iid" else NONIID_DIR
    path = os.path.join(base, f"client_{client_idx}.csv")
    df   = pd.read_csv(path)

    train = df[df["split"] == "train"]
    test  = df[df["split"] == "test"]

    X_train = train[FEATURES].values
    X_test  = test[FEATURES].values
    y_test  = test["is_anomaly"].values

    return X_train, X_test, y_test


def load_all_clients(scenario: str):
    """4 client icin veri yukle."""
    data = {}
    for i in range(4):
        X_train, X_test, y_test = load_client_data(i, scenario)
        data[i] = (X_train, X_test, y_test)
        sid = SENSOR_IDS[i]
        print(f"  Client {i} (Sensor {sid}): "
              f"train={len(X_train)}, test={len(X_test)}, "
              f"anomaly={int(y_test.sum())}")
    return data


# ── FedAvg Cekirdegi ──────────────────────────────────────────────────────────

def train_local_if(X_train: np.ndarray, seed: int) -> IsolationForest:
    """Yerel Isolation Forest egit."""
    model = IsolationForest(
        n_estimators=100,
        max_samples="auto",
        contamination=CONTAMINATION,
        random_state=seed
    )
    model.fit(X_train)
    return model


def fedavg_aggregate(offsets: list, n_samples: list) -> float:
    """Agirlikli ortalama offset_ (Federated Threshold Calibration)."""
    total = sum(n_samples)
    return sum(o * n for o, n in zip(offsets, n_samples)) / total


def evaluate(model: IsolationForest, X_test: np.ndarray,
             y_test: np.ndarray, global_offset: float = None) -> float:
    """F1 skoru hesapla. global_offset verilirse gecici olarak uygula."""
    orig = model.offset_
    if global_offset is not None:
        model.offset_ = global_offset

    preds = model.predict(X_test)
    preds_binary = (preds == -1).astype(int)
    f1 = f1_score(y_test, preds_binary, zero_division=0)

    model.offset_ = orig
    return f1


# ── Senaryo Calistiricilar ────────────────────────────────────────────────────

def run_fedavg(data: dict, seed: int) -> dict:
    """
    3 round FedAvg calistir.
    data: {client_idx: (X_train, X_test, y_test)}
    Her round: yerel egit → aggregate → global offset guncelle.
    Son round sonrasi local ve global F1 raporla.
    """
    global_offset = None
    local_models  = [None] * 4

    for round_num in range(1, N_ROUNDS + 1):
        offsets   = []
        n_samples = []

        for i in range(4):
            X_train = data[i][0]
            model   = train_local_if(X_train, seed=seed * 10 + round_num + i)

            if global_offset is not None:
                model.offset_ = global_offset

            offsets.append(model.offset_)
            n_samples.append(len(X_train))
            local_models[i] = model

        global_offset = fedavg_aggregate(offsets, n_samples)

    # Son round sonrasi degerlendirme
    sensor_results = {}
    for i in range(4):
        X_test, y_test = data[i][1], data[i][2]
        local_f1  = evaluate(local_models[i], X_test, y_test)
        global_f1 = evaluate(local_models[i], X_test, y_test,
                             global_offset=global_offset)
        sid = SENSOR_IDS[i]
        sensor_results[str(sid)] = {
            "local_f1":  round(float(local_f1), 4),
            "global_f1": round(float(global_f1), 4),
            "delta":     round(float(global_f1 - local_f1), 4),
        }

    avg_global = float(np.mean([v["global_f1"]
                                for v in sensor_results.values()]))
    return {"sensor": sensor_results, "avg_global_f1": round(avg_global, 4)}


# ── Ozet Istatistikler ────────────────────────────────────────────────────────

def compute_summary(all_results: dict) -> dict:
    """Seed'ler uzerinden mean +- std hesapla."""
    summary = {}
    for scenario in ["s2", "s3"]:
        avg_f1s = [all_results[scenario][seed]["avg_global_f1"]
                   for seed in SEEDS]
        summary[scenario] = {
            "mean_f1": round(float(np.mean(avg_f1s)), 4),
            "std_f1":  round(float(np.std(avg_f1s, ddof=1)), 4),
            "min_f1":  round(float(np.min(avg_f1s)), 4),
            "max_f1":  round(float(np.max(avg_f1s)), 4),
            "per_seed": {str(s): round(float(f), 4)
                         for s, f in zip(SEEDS, avg_f1s)},
        }
    return summary


# ── Ana Akis ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("FedAvg Multi-Seed Evaluation (S2 IID + S3 Non-IID)")
    print(f"Seeds: {SEEDS}  |  Rounds: {N_ROUNDS}  |  Clients: 4")
    print("=" * 60)

    # Veri yukle
    print("\n[1/3] IID verisi yukleniyor...")
    iid_data = load_all_clients("iid")

    print("\n[2/3] Non-IID verisi yukleniyor...")
    noniid_data = load_all_clients("non_iid")

    # Deneyleri calistir
    print("\n[3/3] Deneyler calistiriliyor...")
    all_results = {"s2": {}, "s3": {}}

    for seed in SEEDS:
        print(f"\n  Seed {seed}:")

        print(f"    S2 (IID)     ", end="", flush=True)
        s2 = run_fedavg(iid_data, seed)
        all_results["s2"][seed] = s2
        print(f"-> avg F1 = {s2['avg_global_f1']:.4f}")

        print(f"    S3 (Non-IID) ", end="", flush=True)
        s3 = run_fedavg(noniid_data, seed)
        all_results["s3"][seed] = s3
        print(f"-> avg F1 = {s3['avg_global_f1']:.4f}")

    # Kaydet
    raw_path = os.path.join(RESULTS_DIR, "multiseed_results.json")
    with open(raw_path, "w") as f:
        json.dump(all_results, f, indent=2)

    summary = compute_summary(all_results)
    summary_path = os.path.join(RESULTS_DIR, "multiseed_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Terminal ozeti
    print("\n" + "=" * 60)
    print("OZET")
    print("=" * 60)
    for scenario, label in [("s2", "S2 IID FL    "),
                             ("s3", "S3 Non-IID FL")]:
        s = summary[scenario]
        print(f"  {label}: F1 = {s['mean_f1']:.4f} +- {s['std_f1']:.4f}"
              f"  [{s['min_f1']:.4f} - {s['max_f1']:.4f}]")

    print(f"\n  Sonuclar kaydedildi:")
    print(f"    {raw_path}")
    print(f"    {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
