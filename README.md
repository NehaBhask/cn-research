# Privacy-Preserving Topology-Aware Federated Learning
### with Dynamic User Join/Leave Handling

A federated learning framework for ISP networks that clusters participants by **network topology similarity** while preserving privacy via **differential privacy (DP)**. Includes a dynamic membership handler that tracks when end-user devices join or leave routers and decides when re-clustering is warranted.

---

## Overview

In a standard federated setup, all ISPs average their model weights together (FedAvg), ignoring the fact that ISPs with structurally similar networks (star, mesh, ring, etc.) may benefit more from aggregating *with each other* than with topologically dissimilar peers.

This project proposes **topology-aware clustered FedAvg**:

1. Each ISP computes a **Laplacian spectrum fingerprint** of its private graph
2. Gaussian DP noise is added before sharing — the raw graph is **never sent**
3. A central server **clusters ISPs by fingerprint similarity** (spectral clustering)
4. FedAvg is performed **within each cluster**, not globally
5. When users join/leave ISP networks, **fingerprint drift** is measured and re-clustering is triggered only when structurally necessary

---

## Project Structure

```
.
├── main.py                        # Entry point — runs all three experiments
├── requirements.txt
├── core/
│   ├── topology_embedding.py      # Laplacian fingerprinting + DP noise
│   ├── clustered_aggregator.py    # Clustered FedAvg + FedAvg baseline
│   └── graph_dynamics.py          # Dynamic join/leave + drift detection
├── data/
│   ├── isp_simulator.py           # Synthetic ISP graph generator
│   ├── geant_loader.py            # Real GEANT backbone network loader
│   ├── geant.txt                  # GEANT topology (SNDlib native format)
│   └── SNDlib_files/              # Supporting SNDlib documentation
└── results/                       # Output plots (auto-created)
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** `networkx`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`

---

## Running

```bash
python main.py
```

This runs all three experiments on both synthetic topologies and the real GEANT dataset, printing results to stdout and saving plots to `results/`.

---

## Experiments

### Experiment 1 — Clustering Quality
Measures how well Laplacian fingerprinting recovers true ISP topology groupings.

- **Silhouette score**: cluster tightness and separation (–1 worst, +1 best)
- **Adjusted Rand Index (ARI)**: agreement with ground-truth topology labels (1.0 = perfect)

### Experiment 2 — Privacy vs Clustering Accuracy Tradeoff
Sweeps the DP privacy budget `epsilon` from 0.1 (strong privacy, high noise) to 10.0 (weak privacy, low noise) and measures the degradation in clustering quality.

| Epsilon | Privacy | Expected quality |
|---------|---------|-----------------|
| 0.1     | Strong  | Low (noisy)     |
| 1.0     | Balanced| Moderate        |
| 5.0+    | Weak    | High (less noise)|

Output plots: `results/privacy_tradeoff_synthetic.png`, `results/privacy_tradeoff_geant.png`

### Experiment 3 — Dynamic User Join/Leave
Simulates end-user devices connecting to and disconnecting from ISP routers, tracking fingerprint drift after each event.

- **Part A**: Single user sessions — drift should stay *below* the adaptive threshold (no re-cluster)
- **Part B**: Burst of users — drift grows with burst size, eventually crossing threshold and triggering re-cluster

Output plots: `results/drift_distribution_synthetic.png`, `results/drift_distribution_geant.png`

---

## Core Concepts

### Topology Fingerprinting (`core/topology_embedding.py`)

Each ISP's private graph `G_i` is represented as a **Laplacian spectrum fingerprint** — the `k` smallest eigenvalues of the normalised graph Laplacian, optionally concatenated with coarse graph statistics (mean degree, density, clustering coefficient, etc.).

This is provably non-invertible: knowing the fingerprint does not allow reconstruction of the original graph.

### Differential Privacy

Before sharing, Gaussian noise calibrated to the **L2 sensitivity** of the fingerprint is added:

```
sigma = sensitivity × sqrt(2 × ln(1.25 / delta)) / epsilon
```

with `delta = 1e-5`. Sensitivity is estimated empirically by measuring how much the fingerprint changes under single-edge additions/removals.

### Adaptive Re-clustering Threshold

The drift threshold used to decide whether to notify the server scales with graph size `n` and burst size `m`:

```
τ(n, m) = 1.5 / (√n × (1 + ln m))
```

Grounded in eigenvalue perturbation theory (Weyl's inequality): a single edge addition shifts eigenvalues by O(1/√n). When `m=1`, this reduces to `τ = 1.5/√n`.

### Clustered Aggregation (`core/clustered_aggregator.py`)

The central server:
1. Receives noisy fingerprints from all ISPs
2. Builds a cosine similarity matrix and runs **spectral clustering**
3. Performs **weighted FedAvg within each cluster** (weighted by dataset size)
4. Returns one aggregated model per cluster

A `FedAvgAggregator` baseline (all ISPs averaged together) is also provided for comparison.

---

## GEANT Dataset

The `data/geant.txt` file contains the real **GEANT** European research backbone network in SNDlib native format: 22 PoP nodes (AT, BE, CH, CZ, DE, ES, FR, GR, HR, HU, IE, IL, IT, LU, NL, NY, PL, PT, SE, SI, SK, UK) and 36 physical links.

The loader (`data/geant_loader.py`) partitions the network into ISP subgraphs for federation experiments. It falls back to a hardcoded topology if the file is missing.

---

## Key Design Decisions

- **The raw graph `G_i` never leaves the ISP.** Only the noisy fingerprint is shared.
- **Even the drift notification reveals only a scalar value** — the magnitude of change, not the change itself.
- **No GNN model is used.** The contribution is purely the clustering strategy; the "model weights" being aggregated are generic and could represent any FL model.
- **Clustering is re-run lazily** — only when drift exceeds the adaptive threshold, reducing communication overhead.

---

## Extending the Code

**Add a new topology type:**
Edit `data/isp_simulator.py` → `build_topology()` and add a new branch.

**Use your own graph:**
```python
from core.topology_embedding import compute_fingerprint, add_dp_noise
import networkx as nx

G = nx.read_edgelist("my_network.txt")
fp = compute_fingerprint(G, k=20)
noisy_fp = add_dp_noise(fp, epsilon=1.0)
```

**Integrate with a real FL framework:**
Replace the dummy `{"w": np.random.randn(4)}` weights in the experiments with actual model state dicts. The aggregator's `aggregate()` method accepts any dict of numpy arrays.

---


## License

MIT
