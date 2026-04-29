"""
fedavg_multiseed.py
====================
Multi-seed evaluation of S2 (IID FL) and S3 (Non-IID FL) scenarios.

Çalıştırma:
    source ~/fl-env/bin/activate
    cd ~/fl-its-anomaly-detection
    python src/fedavg_multiseed.py

Çıktı:
    results/multiseed_results.json   — ham seed-bazlı sonuçlar
    results/multiseed_summary.json   — mean ± std özet

NOT: Bu script FedAvg matematiğini in-process çalıştırır (TCP soketi olmadan).
     Mininet/socket altyapısı S2/S3 single-run'da zaten doğrulandı.
     Multi-seed istatistiksel kararlılık ölçümü için in-process yeterlidir.
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score

# ── Sabitler ──────────────────────────────────────────────────────────────────


SENSOR_IDS   = [278, 240, 71, 298]
SEEDS        = [42, 0, 1, 2, 3]
N_ROUNDS     = 3
CONTAMINATION = 0.00826
FEATURES     = ["flow", "speed", "occupancy"]

IID_DIR    = "/home/enes/fl-its-data/data/splits/iid"
NONIID_DIR = "/home/enes/fl-its-data/data/splits/non_iid"

REPO_ROOT   = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Repo kök dizini (script src/ altında çalışıyorsa bir üst dizin)
REPO_ROOT   = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR    = os.path.join(REPO_ROOT, "data", "processed")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Veri Yükleme ──────────────────────────────────────────────────────────────

def load_client_data(client_idx: int, scenario: str):
    """
    IID veya Non-IID client verisini yükle.
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
    """4 client için veri yükle."""
    data = {}
    for i in range(4):
        X_train, X_test, y_test = load_client_data(i, scenario)
        data[i] = (X_train, X_test, y_test)
        print(f"  Client {i}: train={len(X_train)}, test={len(X_test)}")
    return data


def load_all_sensors():
    """Tüm sensörler için veri sözlüğü döndürür."""
    data = {}
    for sid in SENSOR_IDS:
        data[sid] = load_sensor_data(sid)
        print(f"  Sensör {sid}: "
              f"train={data[sid][0].shape[0]}, "
              f"val={data[sid][2].shape[0]}, "
              f"test={data[sid][4].shape[0]}")
    return data


# ── FedAvg Çekirdeği ──────────────────────────────────────────────────────────

def train_local_if(X_train: np.ndarray, seed: int) -> IsolationForest:
    """Yerel Isolation Forest eğit."""
    model = IsolationForest(
        n_estimators=100,
        max_samples="auto",
        contamination=CONTAMINATION,
        random_state=seed
    )
    model.fit(X_train)
    return model


def fedavg_aggregate(offsets: list, n_samples: list) -> float:
    """
    Federated Threshold Calibration:
    Ağırlıklı ortalama offset_ (sample count ile).
    """
    total = sum(n_samples)
    global_offset = sum(o * n for o, n in zip(offsets, n_samples)) / total
    return global_offset


def evaluate(model: IsolationForest, X_test: np.ndarray,
             y_test: np.ndarray, global_offset: float = None) -> dict:
    """
    Modeli test setinde değerlendir.
    global_offset verilirse, modelin offset_'ini geçici olarak günceller.
    """
    if global_offset is not None:
        original_offset = model.offset_
        model.offset_ = global_offset

    # IsolationForest: -1 → anomali (1), 1 → normal (0)
    preds = model.predict(X_test)
    preds_binary = (preds == -1).astype(int)

    result = {
        "f1":        f1_score(y_test, preds_binary, zero_division=0),
        "precision": precision_score(y_test, preds_binary, zero_division=0),
        "recall":    recall_score(y_test, preds_binary, zero_division=0),
    }

    if global_offset is not None:
        model.offset_ = original_offset  # geri yükle

    return result


# ── Senaryo Çalıştırıcılar ────────────────────────────────────────────────────

def run_s2_iid(data: dict, seed: int) -> dict:
    """
    S2: IID FL
    Tüm train verisini karıştır, 4 parçaya böl, her parçayı bir client'a ata.
    3 round FedAvg çalıştır.
    """
    # Tüm train verilerini birleştir
    X_trains = [data[sid][0] for sid in SENSOR_IDS]
    X_all    = np.vstack(X_trains)

    # Seed ile karıştır
    rng  = np.random.default_rng(seed)
    idx  = rng.permutation(len(X_all))
    X_all_shuffled = X_all[idx]

    # 4 eşit parçaya böl (iloc-style güvenli split)
    n     = len(X_all_shuffled)
    parts = []
    for i in range(4):
        start = (i * n) // 4
        end   = ((i + 1) * n) // 4
        parts.append(X_all_shuffled[start:end])

    # Her client kendi test seti üzerinde değerlendirilir (kendi sensörü)
    test_data = [(data[sid][4], data[sid][5]) for sid in SENSOR_IDS]

    sensor_results = {}
    global_offset  = None  # ilk round'da None, sonra fedavg değeri

    for round_num in range(1, N_ROUNDS + 1):
        offsets   = []
        n_samples = []
        local_models = []

        for i, sid in enumerate(SENSOR_IDS):
            model = train_local_if(parts[i], seed=seed + round_num + i)
            if global_offset is not None:
                model.offset_ = global_offset  # global threshold uygula
            offsets.append(model.offset_)
            n_samples.append(len(parts[i]))
            local_models.append(model)

        global_offset = fedavg_aggregate(offsets, n_samples)

    # Final round sonrası değerlendirme
    for i, sid in enumerate(SENSOR_IDS):
        X_test, y_test = test_data[i]
        local_res  = evaluate(local_models[i], X_test, y_test)
        global_res = evaluate(local_models[i], X_test, y_test,
                              global_offset=global_offset)
        sensor_results[str(sid)] = {
            "local_f1":  local_res["f1"],
            "global_f1": global_res["f1"],
            "delta":     global_res["f1"] - local_res["f1"],
        }

    avg_global = np.mean([v["global_f1"] for v in sensor_results.values()])
    return {"sensor": sensor_results, "avg_global_f1": avg_global}


def run_s3_noniid(data: dict, seed: int) -> dict:
    """
    S3: Non-IID FL
    Her client kendi sensörünün train verisini kullanır.
    3 round FedAvg çalıştır.
    """
    test_data = [(data[sid][4], data[sid][5]) for sid in SENSOR_IDS]

    sensor_results = {}
    global_offset  = None

    for round_num in range(1, N_ROUNDS + 1):
        offsets   = []
        n_samples = []
        local_models = []

        for i, sid in enumerate(SENSOR_IDS):
            X_train = data[sid][0]
            model   = train_local_if(X_train, seed=seed + round_num + i)
            if global_offset is not None:
                model.offset_ = global_offset
            offsets.append(model.offset_)
            n_samples.append(len(X_train))
            local_models.append(model)

        global_offset = fedavg_aggregate(offsets, n_samples)

    # Final değerlendirme
    for i, sid in enumerate(SENSOR_IDS):
        X_test, y_test = test_data[i]
        local_res  = evaluate(local_models[i], X_test, y_test)
        global_res = evaluate(local_models[i], X_test, y_test,
                              global_offset=global_offset)
        sensor_results[str(sid)] = {
            "local_f1":  local_res["f1"],
            "global_f1": global_res["f1"],
            "delta":     global_res["f1"] - local_res["f1"],
        }

    avg_global = np.mean([v["global_f1"] for v in sensor_results.values()])
    return {"sensor": sensor_results, "avg_global_f1": avg_global}


# ── Özet İstatistikler ────────────────────────────────────────────────────────

def compute_summary(all_results: dict) -> dict:
    """
    Seed'ler üzerinden mean ± std hesapla.
    all_results: {"s2": {seed: {...}}, "s3": {seed: {...}}}
    """
    summary = {}
    for scenario in ["s2", "s3"]:
        avg_f1s = [
            all_results[scenario][seed]["avg_global_f1"]
            for seed in SEEDS
        ]
        summary[scenario] = {
            "mean_f1": float(np.mean(avg_f1s)),
            "std_f1":  float(np.std(avg_f1s, ddof=1)),
            "min_f1":  float(np.min(avg_f1s)),
            "max_f1":  float(np.max(avg_f1s)),
            "per_seed": {str(s): float(f) for s, f in zip(SEEDS, avg_f1s)},
        }
    return summary


# ── Ana Akış ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("FedAvg Multi-Seed Evaluation (S2 IID + S3 Non-IID)")
    print(f"Seeds: {SEEDS}  |  Rounds: {N_ROUNDS}  |  Clients: {len(SENSOR_IDS)}")
    print("=" * 60)

    print("\n[1/3] Veri yükleniyor...")
    data = load_all_sensors()

    all_results = {"s2": {}, "s3": {}}

    print("\n[2/3] Deneyler çalıştırılıyor...")
    for seed in SEEDS:
        print(f"\n  Seed {seed}:")

        print(f"    S2 (IID)     ", end="", flush=True)
        s2_res = run_s2_iid(data, seed)
        all_results["s2"][seed] = s2_res
        print(f"→ avg F1 = {s2_res['avg_global_f1']:.4f}")

        print(f"    S3 (Non-IID) ", end="", flush=True)
        s3_res = run_s3_noniid(data, seed)
        all_results["s3"][seed] = s3_res
        print(f"→ avg F1 = {s3_res['avg_global_f1']:.4f}")

    print("\n[3/3] Sonuçlar kaydediliyor...")

    # Ham sonuçlar (seed bazlı)
    raw_path = os.path.join(RESULTS_DIR, "multiseed_results.json")
    with open(raw_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  → {raw_path}")

    # Özet (mean ± std)
    summary = compute_summary(all_results)
    summary_path = os.path.join(RESULTS_DIR, "multiseed_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  → {summary_path}")

    # Terminale özet yazdır
    print("\n" + "=" * 60)
    print("ÖZET")
    print("=" * 60)
    for scenario, label in [("s2", "S2 IID FL   "), ("s3", "S3 Non-IID FL")]:
        s = summary[scenario]
        print(f"  {label}: F1 = {s['mean_f1']:.4f} ± {s['std_f1']:.4f}  "
              f"[{s['min_f1']:.4f} – {s['max_f1']:.4f}]")
    print("=" * 60)
    print("\nBitti. Şimdi generate_figures.py ile fig14 üretebilirsin.")


if __name__ == "__main__":
    main()
