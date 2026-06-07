"""
convergence_logger.py
======================
Her FL round sonrasi F1 skoru loglar.
S2 (IID) ve S3 (Non-IID) icin convergence curve verisi uretir.
5 seed uzerinden mean +- std hesaplar.

Calistirma:
    source ~/fl-env/bin/activate
    cd ~/fl-its
    python src/convergence_logger.py

Cikti:
    results/convergence_curves.json  -- round x seed x scenario F1 degerleri
    results/convergence_summary.json -- her round icin mean +- std
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
FEATURES      = ["flow", "occupancy", "speed"]

IID_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "splits", "iid")
NONIID_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "splits", "non_iid")

REPO_ROOT   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Veri Yukleme ──────────────────────────────────────────────────────────────

def load_all_clients(scenario: str):
    base = IID_DIR if scenario == "iid" else NONIID_DIR
    data = {}
    for i in range(4):
        path  = os.path.join(base, f"client_{i}.csv")
        df    = pd.read_csv(path)
        train = df[df["split"] == "train"]
        test  = df[df["split"] == "test"]
        data[i] = (
            train[FEATURES].values,
            test[FEATURES].values,
            test["is_anomaly"].values,
        )
    return data


# ── FedAvg Yardimcilari ───────────────────────────────────────────────────────

def train_local_if(X_train: np.ndarray, seed: int) -> IsolationForest:
    model = IsolationForest(
        n_estimators=100,
        max_samples="auto",
        contamination=CONTAMINATION,
        random_state=seed,
    )
    model.fit(X_train)
    return model


def fedavg_aggregate(offsets: list, n_samples: list) -> float:
    total = sum(n_samples)
    return sum(o * n for o, n in zip(offsets, n_samples)) / total


def evaluate_global(models: list, data: dict,
                    global_offset: float) -> float:
    """4 client'in global F1 ortalamasini hesapla."""
    f1s = []
    for i in range(4):
        _, X_test, y_test = data[i]
        orig = models[i].offset_
        models[i].offset_ = global_offset
        preds  = models[i].predict(X_test)
        binary = (preds == -1).astype(int)
        f1s.append(f1_score(y_test, binary, zero_division=0))
        models[i].offset_ = orig
    return float(np.mean(f1s))


# ── Convergence Kayit ────────────────────────────────────────────────────────

def run_convergence(data: dict, seed: int) -> dict:
    """
    N_ROUNDS boyunca FedAvg calistir (stability semantigi).
    Her round: 4 client taze IF egitir -> threshold aggregate -> avg F1 kaydet.
    Threshold-only calibration tek turda yakinsadiginden, sonraki turlar
    ayni veri uzerinde farkli random seed ile kararliligi gosterir.
    Donus: {"round_1": f1, "round_2": f1, "round_3": f1}
    """
    round_f1s = {}

    for round_num in range(1, N_ROUNDS + 1):
        offsets   = []
        n_samples = []
        models    = []

        for i in range(4):
            X_train = data[i][0]
            model   = train_local_if(
                X_train, seed=seed * 10 + round_num + i
            )
            offsets.append(model.offset_)
            n_samples.append(len(X_train))
            models.append(model)

        global_offset = fedavg_aggregate(offsets, n_samples)
        avg_f1 = evaluate_global(models, data, global_offset)
        round_f1s[f"round_{round_num}"] = round(avg_f1, 4)

    return round_f1s


# ── Ozet Hesapla ──────────────────────────────────────────────────────────────

def compute_convergence_summary(all_curves: dict) -> dict:
    """
    Her round icin 5 seed'in mean +- std'sini hesapla.
    all_curves: {"s2": {seed: {"round_1": f1, ...}}, "s3": {...}}
    """
    summary = {}
    for scenario in ["s2", "s3"]:
        summary[scenario] = {}
        for r in range(1, N_ROUNDS + 1):
            key  = f"round_{r}"
            vals = [all_curves[scenario][seed][key] for seed in SEEDS]
            summary[scenario][key] = {
                "mean": round(float(np.mean(vals)), 4),
                "std":  round(float(np.std(vals, ddof=1)), 4),
            }
    return summary


# ── Ana Akis ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("FL Convergence Logger (S2 IID + S3 Non-IID)")
    print(f"Seeds: {SEEDS}  |  Rounds: {N_ROUNDS}  |  Clients: 4")
    print("=" * 60)

    print("\n[1/3] Veriler yukleniyor...")
    iid_data    = load_all_clients("iid")
    noniid_data = load_all_clients("non_iid")
    print("  IID ve Non-IID veri yuklendi.")

    print("\n[2/3] Convergence deneyleri calistiriliyor...")
    all_curves = {"s2": {}, "s3": {}}

    for seed in SEEDS:
        print(f"\n  Seed {seed}:")

        s2_curve = run_convergence(iid_data, seed)
        all_curves["s2"][seed] = s2_curve
        s2_vals = [f"{s2_curve[f'round_{r}']:.4f}"
                   for r in range(1, N_ROUNDS + 1)]
        print(f"    S2 (IID)     rounds: {' -> '.join(s2_vals)}")

        s3_curve = run_convergence(noniid_data, seed)
        all_curves["s3"][seed] = s3_curve
        s3_vals = [f"{s3_curve[f'round_{r}']:.4f}"
                   for r in range(1, N_ROUNDS + 1)]
        print(f"    S3 (Non-IID) rounds: {' -> '.join(s3_vals)}")

    print("\n[3/3] Sonuclar kaydediliyor...")
    curves_path = os.path.join(RESULTS_DIR, "convergence_curves.json")
    with open(curves_path, "w") as f:
        json.dump(all_curves, f, indent=2)
    print(f"  -> {curves_path}")

    summary = compute_convergence_summary(all_curves)
    summary_path = os.path.join(RESULTS_DIR, "convergence_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  -> {summary_path}")

    # Terminal ozeti
    print("\n" + "=" * 60)
    print("CONVERGENCE OZETI (mean +- std per round)")
    print("=" * 60)
    print(f"  {'Round':<8} {'S2 IID':<20} {'S3 Non-IID':<20}")
    print(f"  {'-'*6:<8} {'-'*18:<20} {'-'*18:<20}")
    for r in range(1, N_ROUNDS + 1):
        key = f"round_{r}"
        s2  = summary["s2"][key]
        s3  = summary["s3"][key]
        print(f"  {r:<8} "
              f"{s2['mean']:.4f} +- {s2['std']:.4f}     "
              f"{s3['mean']:.4f} +- {s3['std']:.4f}")
    print("=" * 60)
    print("\nBitti. generate_figures.py ile fig14 uretebilirsin.")


if __name__ == "__main__":
    main()
