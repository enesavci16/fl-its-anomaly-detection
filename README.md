# FL + ITS Anomaly Detection — Federated Learning with Mininet/SDN

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![Ubuntu 22.04](https://img.shields.io/badge/ubuntu-22.04_LTS-orange.svg)](https://ubuntu.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Conference: ASYU 2026](https://img.shields.io/badge/conference-ASYU_2026-red.svg)](https://asyu.org.tr/)

> **Paper:** "Privacy-Aware Traffic Anomaly Detection via Federated Learning on PeMS04: A Mininet/SDN-Based Ablation Study"  
> **Conference:** ASYU 2026 — Signal Processing and Communications Applications Conference (İstanbul, September 2026)  
> **Authors:** Enes Avcı, Murtaza Çiçioğlu — Bursa Uludağ University, Dept. of Computer Engineering

---

## Overview

This repository implements a federated learning (FL) pipeline for traffic anomaly detection in Intelligent Transportation Systems (ITS), emulated over a real software-defined network using Mininet and os-ken SDN controller. The core contribution is a **three-scenario ablation study**:

| Scenario | Description | F1 Score |
|----------|-------------|----------|
| **S1** — Centralized baseline | All sensor data trained on a single server | **0.7715** |
| **S2** — IID Federated Learning | FedAvg over uniformly distributed data | **0.7028** (−8.9%) |
| **S3** — Non-IID Federated Learning | FedAvg over heterogeneous real sensor splits | **0.5843** (−24.3%) |

**Key design decision:** Instead of merging Isolation Forest trees via FedAvg (infeasible due to sklearn pickle instability), we implement **Federated Threshold Calibration** — each client keeps its own trees; only the weighted average of `offset_` is shared and aggregated.

---

## Repository Structure

```
fl-its-anomaly-detection/
│
├── data/
│   ├── raw/                        # PeMS04 raw .npz (not committed — download separately)
│   ├── processed/
│   │   ├── sensor_278.csv          # Sensor P25 — low-activity profile
│   │   ├── sensor_240.csv          # Sensor P50 — median profile
│   │   ├── sensor_071.csv          # Sensor P75 — high-activity profile
│   │   └── sensor_298.csv          # Sensor P95 — peak-activity profile
│   └── splits/
│       ├── iid/                    # IID train/val/test splits (4 × sensor)
│       └── non_iid/                # Non-IID splits preserving real sensor distributions
│
├── src/
│   ├── its_topo.py                 # Mininet topology: 4 sensor hosts + TMC + OpenFlow switch
│   ├── fedavg_server.py            # TMC aggregator — collects offset_, runs FedAvg
│   ├── fedavg_client.py            # Sensor client — trains local IF, sends offset_
│   ├── fedavg_utils.py             # Federated Threshold Calibration logic
│   ├── fedavg_multiseed.py         # Multi-seed runner (5 seeds → mean ± std)
│   ├── convergence_logger.py       # Logs F1 per FL round (for fig14)
│   ├── lof_baseline.py             # LOF baseline comparison
│   ├── paired_ttest.py             # Statistical significance tests (S1 vs S2 vs S3)
│   ├── generate_fig08.py           # Confusion matrix figure
│   └── generate_figures.py         # All ablation figures (fig10–fig14)
│
├── notebooks/
│   ├── 01_pems04_exploration.ipynb # Data exploration (307 sensors, feature distributions)
│   ├── 02_baseline_and_labeling.ipynb  # S1 centralized baseline + anomaly labeling
│   └── 03_ablation_figures.ipynb   # Figure generation pipeline
│
├── results/
│   ├── fig01–fig14.png             # All paper figures (300 DPI, IEEE serif font)
│   └── metrics/                    # JSON result files per scenario/seed
│
├── paper/
│   └── main.tex                    # IEEE IEEEtran LaTeX source (ASYU 2026 submission)
│
└── README.md                       # This file
```

---

## System Requirements

> ⚠️ **This environment combination is verified and frozen.** Do NOT use Ubuntu 24.04 or Docker — see [Why These Exact Versions](#why-these-exact-versions).

| Component | Required Version | Notes |
|-----------|-----------------|-------|
| **Host OS** | Windows 10/11 or Linux | VirtualBox host |
| **Hypervisor** | VirtualBox 7.0+ | VMware also works |
| **Guest OS** | **Ubuntu 22.04 LTS** | CRITICAL — not 24.04 |
| **Python** | 3.10.x | Ships with Ubuntu 22.04 |
| **Mininet** | 2.3.0 | via apt |
| **Open vSwitch** | 2.17.x | via apt |
| **os-ken** | 2.3.1 | via apt (Ryu fork) |
| **scikit-learn** | 1.3.2 | via pip/venv |
| **Flower (flwr)** | 1.7.0 | via pip/venv |

**VM resource recommendations:**

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 4 GB | 8 GB |
| CPU cores | 2 | 4 |
| Disk | 20 GB | 25 GB |
| Network | NAT | NAT |

---

## Part 1 — Environment Setup (First Time Only)

### 1.1 Install System Packages

```bash
# Enable universe repository if not already enabled
sudo add-apt-repository universe
sudo apt update

# Install Mininet, OVS, os-ken, and build tools — verified working combination
sudo apt install -y \
    mininet \
    openvswitch-switch \
    python3-os-ken \
    python3-pip \
    python3-venv \
    curl git nano wget
```

```bash
# Start Open vSwitch (required before any mn command)
sudo service openvswitch-switch start
```

### 1.2 Verify Installation

```bash
mn --version              # Expected: 2.3.0
ovs-vsctl --version       # Expected: ovs-vsctl (Open vSwitch) 2.17.x
osken-manager --version   # Expected: ...os-ken 2.3.1
python3 --version         # Expected: Python 3.10.12

# Verify os-ken import
python3 -c "from os_ken.base import app_manager; print('os-ken OK')"
```

### 1.3 Prepare os-ken Simple Switch (One-Time Only)

The `python3-os-ken` apt package does not include example applications. Download and patch the learning switch manually:

```bash
mkdir -p ~/osken-apps
cd ~/osken-apps

# Download from faucetsdn/ryu mirror (actively maintained)
wget https://raw.githubusercontent.com/faucetsdn/ryu/master/ryu/app/simple_switch_13.py

# Patch: ryu → os_ken namespace
sed -i 's/from ryu\./from os_ken./g' simple_switch_13.py
sed -i 's/import ryu\./import os_ken./g' simple_switch_13.py
sed -i 's/app_manager\.RyuApp/app_manager.OSKenApp/g' simple_switch_13.py
```

Verify the switch works:

```bash
# Terminal 1 — start controller
osken-manager ~/osken-apps/simple_switch_13.py

# Terminal 2 — start test topology
sudo mn --controller=remote,ip=127.0.0.1,port=6653 \
        --switch=ovs,protocols=OpenFlow13 \
        --topo=single,3
mininet> pingall
# Expected: *** Results: 0% dropped (6/6 received)
```

### 1.4 Clone the Repository

```bash
cd ~
git clone https://github.com/enesavci16/fl-its-anomaly-detection.git
cd fl-its-anomaly-detection
```

### 1.5 Create Python Virtual Environment

```bash
python3 -m venv ~/fl-env
source ~/fl-env/bin/activate

pip install --upgrade pip
pip install \
    numpy==1.26.4 \
    pandas==2.0.3 \
    scikit-learn==1.3.2 \
    matplotlib==3.7.3 \
    seaborn==0.12.2 \
    flwr==1.7.0 \
    scipy==1.11.4 \
    jupyter
```

> **Always activate the venv before running any project script:**
> ```bash
> source ~/fl-env/bin/activate
> ```

---

## Part 2 — Dataset Preparation

### 2.1 Download PeMS04

PeMS04 is publicly available via the ASTGCN repository:

```bash
cd ~/fl-its-anomaly-detection

# Option A: wget
wget -O data/raw/pems04.npz \
    "https://github.com/Davidham3/ASTGCN/raw/master/datasets/PEMS04/PEMS04.npz"

# Option B: if the above URL changes, also available at:
# https://github.com/guoshnBJTU/ASTGNN/tree/main/data
```

Verify the download:

```bash
source ~/fl-env/bin/activate
python3 - <<'EOF'
import numpy as np
data = np.load('data/raw/pems04.npz')
print("Keys:", list(data.keys()))
arr = data['data']
print("Shape:", arr.shape)  # Expected: (16992, 307, 3)
print("dtype:", arr.dtype)  # Expected: float32 or float64
EOF
```

Expected output: `Shape: (16992, 307, 3)` — 59 days × 307 sensors × 3 features (flow, speed, occupancy).

### 2.2 Run Preprocessing Notebooks

Open Jupyter and run the notebooks **in order**:

```bash
source ~/fl-env/bin/activate
cd ~/fl-its-anomaly-detection
jupyter notebook
```

Run in this order:

| Notebook | What it does | Key output |
|----------|-------------|------------|
| `01_pems04_exploration.ipynb` | Data exploration, sensor selection | `fig01–fig06.png` |
| `02_baseline_and_labeling.ipynb` | Z-score anomaly labeling, S1 centralized baseline | `sensor_XXX.csv`, `labels_sensor_XXX.csv`, `zscore_stats.json` |

After running notebook 02, verify the processed files exist:

```bash
ls data/processed/
# Expected: sensor_278.csv  sensor_240.csv  sensor_071.csv  sensor_298.csv
# Expected: labels_sensor_278.csv  labels_sensor_240.csv  ...  zscore_stats.json

ls data/splits/iid/
# Expected: sensor_278_iid.csv  sensor_240_iid.csv  ...

ls data/splits/non_iid/
# Expected: sensor_278_non_iid.csv  sensor_240_non_iid.csv  ...
```

---

## Part 3 — Running the Experiments

### Overview: How the FL Simulation Works

```
[Mininet Network]
┌────────────────────────────────────────────────────┐
│  s1 (OpenFlow Switch) ── os-ken controller (6653)  │
│   │                                                │
│   ├── h1 (sensor_278, P25)  → runs fedavg_client  │
│   ├── h2 (sensor_240, P50)  → runs fedavg_client  │
│   ├── h3 (sensor_071, P75)  → runs fedavg_client  │
│   ├── h4 (sensor_298, P95)  → runs fedavg_client  │
│   └── h_tmc (TMC server)    → runs fedavg_server  │
└────────────────────────────────────────────────────┘
        ↑ TCP sockets over OpenFlow-managed network
```

Each FL round:
1. **Server** broadcasts current global `offset_` to all clients (port 9000)
2. **Each client** trains a local Isolation Forest on its sensor data, then sends its local `offset_` + sample count back to the server
3. **Server** computes weighted average: `global_offset = Σ(n_i × offset_i) / Σ(n_i)` and broadcasts updated `offset_`
4. Repeat for 3 rounds

---

### Scenario S1 — Centralized Baseline (No Mininet Required)

S1 runs entirely in Python without Mininet. All sensor data is pooled on a single server.

```bash
source ~/fl-env/bin/activate
cd ~/fl-its-anomaly-detection

python3 - <<'EOF'
# Quick S1 baseline verification
from src.fedavg_utils import run_centralized_baseline
metrics = run_centralized_baseline(data_dir='data/processed', labels_dir='data/processed')
print(f"S1 — Test F1: {metrics['f1']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
# Expected: F1=0.7715 | Precision=0.9892 | Recall=0.6323
EOF
```

Or use the notebook `02_baseline_and_labeling.ipynb` which already has S1 results.

---

### Scenario S2 — IID Federated Learning (Mininet Required)

Open **5 terminals** in the VM (or use `xterm` inside Mininet — see option B below).

#### Option A — Five Separate Terminals

**Terminal 1 — Start os-ken SDN Controller:**

```bash
osken-manager ~/osken-apps/simple_switch_13.py
```
> Leave this running. You should see: `loading app simple_switch_13.py`

**Terminal 2 — Start Mininet Topology:**

```bash
sudo mn --custom ~/fl-its-anomaly-detection/src/its_topo.py \
        --topo its_topo \
        --controller remote,ip=127.0.0.1,port=6653 \
        --switch ovs,protocols=OpenFlow13
```

Verify connectivity inside Mininet:

```
mininet> pingall
# Expected: *** Results: 0% dropped (20/20 received)
```

**Terminal 3 — Start FL Server on TMC host:**

```bash
# Inside Mininet CLI:
mininet> h_tmc python3 /home/enes/fl-its-anomaly-detection/src/fedavg_server.py \
              --rounds 3 \
              --n_clients 4 \
              --mode iid \
              --results_dir /home/enes/fl-its-anomaly-detection/results/ &
```

Or from a separate terminal using `mnexec`:

```bash
# Get TMC PID from Mininet, then:
sudo mnexec -a <tmc_pid> python3 src/fedavg_server.py --rounds 3 --n_clients 4 --mode iid
```

**Terminals 4–7 — Start FL Clients (one per sensor):**

```bash
# Inside Mininet CLI, run all 4 clients simultaneously:
mininet> h1 python3 /home/enes/fl-its-anomaly-detection/src/fedavg_client.py \
              --sensor_id 278 --split iid --server_ip 10.0.0.5 --server_port 9000 &
mininet> h2 python3 /home/enes/fl-its-anomaly-detection/src/fedavg_client.py \
              --sensor_id 240 --split iid --server_ip 10.0.0.5 --server_port 9000 &
mininet> h3 python3 /home/enes/fl-its-anomaly-detection/src/fedavg_client.py \
              --sensor_id 071 --split iid --server_ip 10.0.0.5 --server_port 9000 &
mininet> h4 python3 /home/enes/fl-its-anomaly-detection/src/fedavg_client.py \
              --sensor_id 298 --split iid --server_ip 10.0.0.5 --server_port 9000 &
```

#### Option B — Automated Launch via its_topo.py (Recommended)

`its_topo.py` can launch all processes automatically using `xterm`:

```bash
# Terminal 1 — Controller (always first)
osken-manager ~/osken-apps/simple_switch_13.py

# Terminal 2 — Full automated experiment
source ~/fl-env/bin/activate
sudo -E env PATH=$PATH ~/fl-env/bin/python3 \
    src/its_topo.py --mode iid --rounds 3
```

This automatically:
- Starts Mininet topology
- Opens xterm for each host
- Launches server on h_tmc and clients on h1–h4
- Logs results to `results/s2_iid_results.json`

**Expected S2 output:**

```
[Round 1/3] Global offset: -0.4821 | Avg F1: 0.6934
[Round 2/3] Global offset: -0.4876 | Avg F1: 0.7012
[Round 3/3] Global offset: -0.4901 | Avg F1: 0.7028
S2 IID FL completed — Mean F1: 0.7028
```

---

### Scenario S3 — Non-IID Federated Learning

Identical to S2, but change `--mode iid` to `--mode non_iid`:

```bash
# Option B (automated):
sudo -E env PATH=$PATH ~/fl-env/bin/python3 \
    src/its_topo.py --mode non_iid --rounds 3
```

Or inside Mininet CLI, replace `--split iid` with `--split non_iid` for all client commands.

**Expected S3 output:**

```
[Round 1/3] Global offset: -0.5134 | Avg F1: 0.5621
[Round 2/3] Global offset: -0.5287 | Avg F1: 0.5793
[Round 3/3] Global offset: -0.5311 | Avg F1: 0.5843
S3 Non-IID FL completed — Mean F1: 0.5843
```

---

### Multi-Seed Runs (Statistical Robustness)

Run all three scenarios across 5 random seeds to obtain mean ± std:

```bash
source ~/fl-env/bin/activate
python3 src/fedavg_multiseed.py --seeds 0 1 2 3 4 --modes s1 s2 s3
```

Expected results:

| Scenario | Mean F1 | Std |
|----------|---------|-----|
| S1 Centralized | 0.7715 | ±0.000 (deterministic) |
| S2 IID FL | 0.724 | ±0.029 |
| S3 Non-IID FL | 0.594 | ±0.018 |

---

### LOF Baseline Comparison

```bash
source ~/fl-env/bin/activate
python3 src/lof_baseline.py
# Expected: LOF F1=0.160 vs IF F1=0.772 — confirms Isolation Forest model selection
```

---

### Statistical Significance Tests

Run after multi-seed experiments:

```bash
source ~/fl-env/bin/activate
python3 src/paired_ttest.py --results_dir results/metrics/

# Expected output:
# S2 vs S3: t=13.281, p=0.0002 *** (significant)
# S1 vs S2: t=-3.664, p=0.022  *   (significant)
# S1 vs S3: t=-21.938, p<0.0001 *** (significant)
```

---

## Part 4 — Generating Paper Figures

```bash
source ~/fl-env/bin/activate
cd ~/fl-its-anomaly-detection

python3 src/generate_figures.py
# Generates fig10–fig14 in results/ at 300 DPI, IEEE serif font

# Or via notebook:
jupyter notebook notebooks/03_ablation_figures.ipynb
```

| Figure | Description |
|--------|-------------|
| `fig10.png` | Ablation F1 bar chart — S1 / S2 / S3 with delta arrows |
| `fig11.png` | Client drift — Local vs Global F1 per sensor (S2 and S3 side-by-side) |
| `fig12.png` | Communication cost comparison (centralized vs FL) |
| `fig13.png` | S1 Precision / Recall / F1 breakdown |
| `fig14.png` | Convergence curve — F1 per FL round with ±1 std bands |

---

## Part 5 — Reproducing from Scratch (Checklist)

For a reviewer or advisor starting from zero:

- [ ] Ubuntu 22.04 LTS VM running in VirtualBox
- [ ] `sudo apt install -y mininet openvswitch-switch python3-os-ken python3-pip python3-venv`
- [ ] `sudo service openvswitch-switch start`
- [ ] os-ken simple_switch_13.py downloaded and patched (Section 1.3)
- [ ] `git clone https://github.com/enesavci16/fl-its-anomaly-detection`
- [ ] venv created and activated, packages installed (Section 1.5)
- [ ] PeMS04 downloaded to `data/raw/pems04.npz` (Section 2.1)
- [ ] Notebooks 01 and 02 executed (Section 2.2)
- [ ] `pingall` test: 0% dropped (Section 1.3)
- [ ] S1 baseline: F1 ≈ 0.7715
- [ ] S2 IID FL: F1 ≈ 0.7028
- [ ] S3 Non-IID FL: F1 ≈ 0.5843
- [ ] Figures generated in `results/`

**Total estimated setup time:** 45–60 minutes (first time), including VM provisioning.  
**Experiment runtime:** S1 ~2 min, S2 ~8 min, S3 ~8 min, multi-seed ~45 min.

---

## Key Design Decisions

### Why Federated Threshold Calibration (not FedAvg on trees)?

Standard FedAvg aggregates model parameters by weighted averaging. Isolation Forest trees cannot be merged this way because:

1. **sklearn pickle instability:** After serialization/deserialization, internal attributes (`_max_features`, `_max_samples`, `_decision_path_lengths`) are lost, making the trees non-functional.
2. **Scale incompatibility:** Trees trained on different data distributions have structurally incompatible depth profiles; averaging their parameters is mathematically undefined.

**Our solution:** Each client keeps its own trees. Only `offset_` (the anomaly threshold scalar, shape `(1,)`) is transmitted and aggregated:

```
global_offset = Σ(n_i × offset_i) / Σ(n_i)
```

**Known limitation:** Weighted averaging of percentile-based thresholds is an approximation — `weighted_avg(percentile(A), percentile(B)) ≠ percentile(concat(A,B))`. The approximation error grows under Non-IID conditions and is one of the three components explaining the S3 F1 drop (alongside client drift and information loss from not sharing raw data).

### Why Isolation Forest?

- Unsupervised: no ground-truth anomaly labels required during training
- Linear time complexity: O(n) — fits embedded sensor constraints
- Significantly outperforms LOF on this dataset: F1=0.772 vs F1=0.160

### Why os-ken instead of Ryu?

Ubuntu 22.04 (Jammy) dropped `python3-ryu` from its repositories. `os-ken` is the OpenStack-maintained fork of Ryu with near-identical API — only the import namespace and app base class name differ. Ryu itself is unmaintained since 2020.

### Why Ubuntu 22.04 (not 24.04)?

Ubuntu 24.04 ships Python 3.12 + enforces PEP 668 (no `pip install` outside venv), and has `eventlet` compatibility issues with the os-ken 2.3.1 async loop. The 22.04 + Python 3.10 combination is experimentally verified (April 11, 2026 — VirtualBox snapshot: `H0-osken-mininet-working`).

---

## Privacy Claim Precision

This system is **raw-data-locality-preserving**: sensor data never leaves the local node. However, it does **not** provide formal differential privacy guarantees. The transmitted `offset_` scalar can in theory leak distributional information. Full differential privacy (DP-FedAvg) is deferred to the journal extension.

---

## Citation

```bibtex
@inproceedings{avci2026flits,
  title     = {Privacy-Aware Traffic Anomaly Detection via Federated Learning on PeMS04:
               A Mininet/SDN-Based Ablation Study},
  author    = {Avcı, Enes and Çiçioğlu, Murtaza},
  booktitle = {2026 Signal Processing and Communications Applications Conference (ASYU)},
  year      = {2026},
  address   = {İstanbul, Turkey},
  publisher = {IEEE}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.  
Dataset (PeMS04): © California Department of Transportation (Caltrans) — open access for research use.
