"""
topology_embedding.py
---------------------
Converts each ISP's private graph into a privacy-safe fingerprint.

Core idea:
  The Laplacian spectrum (eigenvalues of the normalized Laplacian)
  captures the structural shape of a graph without revealing actual
  node identities or edge connections.

  An ISP shares ONLY this fingerprint — the actual graph G_i
  never leaves the ISP.

Privacy layer:
  Gaussian noise (calibrated to privacy budget epsilon) is added
  to the fingerprint before sharing, giving differential privacy.

  Even with the noisy fingerprint, the server cannot reconstruct
  the original graph. It can only measure structural similarity
  between ISPs to group them into clusters.
"""

import numpy as np
import networkx as nx
from typing import Optional


def compute_laplacian_spectrum(G: nx.Graph, k: int = 20) -> np.ndarray:
    """
    Compute the normalized Laplacian spectrum of graph G.

    Returns a fixed-length vector of k smallest eigenvalues,
    padded with zeros if G has fewer than k nodes.
    """
    L = nx.normalized_laplacian_matrix(G).toarray().astype(float)
    eigenvalues = np.sort(np.linalg.eigvalsh(L))   # real, ascending

    if len(eigenvalues) >= k:
        return eigenvalues[:k].astype(np.float32)
    else:
        return np.pad(eigenvalues, (0, k - len(eigenvalues))).astype(np.float32)


def compute_graph_stats(G: nx.Graph) -> np.ndarray:
    """
    Compute coarse graph statistics — safe to share, not reversible.
    Appended to the spectrum to improve clustering discriminability.
    """
    degrees = [d for _, d in G.degree()]
    return np.array([
        np.mean(degrees),
        np.std(degrees),
        nx.density(G),
        float(G.number_of_nodes()) / 20.0,   # normalized scale
        float(nx.average_clustering(G)),
    ], dtype=np.float32)


def compute_fingerprint(G: nx.Graph, k: int = 20,
                        include_stats: bool = True) -> np.ndarray:
    """
    Full topology fingerprint = Laplacian spectrum + graph stats.
    This is what gets shared with the central server (after DP noise).
    """
    spectrum = compute_laplacian_spectrum(G, k=k)
    if not include_stats:
        return spectrum
    return np.concatenate([spectrum, compute_graph_stats(G)])


def add_dp_noise(fingerprint: np.ndarray,
                 sensitivity: float = 1.0,
                 epsilon: float = 1.0) -> np.ndarray:
    """
    Add calibrated Gaussian noise to the fingerprint for differential privacy.

    Gaussian mechanism: noise ~ N(0, sigma^2) where
        sigma = sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon

    We use a simplified version with delta=1e-5.

    Lower epsilon = stronger privacy = more noise = less accurate clustering.
    Higher epsilon = weaker privacy = less noise = better clustering.

    Args:
        fingerprint:  Raw topology fingerprint
        sensitivity:  L2 sensitivity of the fingerprint (default 1.0)
        epsilon:      Privacy budget (typical: 0.1 to 10.0)

    Returns:
        Noisy fingerprint — safe to share with central server
    """
    delta = 1e-5
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    noise = np.random.normal(0, sigma, size=fingerprint.shape).astype(np.float32)
    return fingerprint + noise


def assign_fingerprints(clients: list, k: int = 20,
                        epsilon: float = 1.0) -> list:
    """
    Compute and store noisy fingerprints for all ISP clients.
    Called once at the start of federated training.

    Returns list of noisy fingerprints (one per client) —
    these are what the server receives.
    """
    noisy_fingerprints = []
    for client in clients:
        raw_fp   = compute_fingerprint(client.graph, k=k)
        noisy_fp = add_dp_noise(raw_fp, epsilon=epsilon)
        client.fingerprint = noisy_fp
        noisy_fingerprints.append(noisy_fp)
        print(f"  ISP {client.client_id} ({client.topo_type:6s}) | "
              f"spectrum[:4]: [{', '.join(f'{v:.3f}' for v in raw_fp[:4])}...] "
              f"| eps={epsilon}")
    return noisy_fingerprints


def compute_adaptive_threshold(n: int, m: int = 1, alpha: float = 1.5) -> float:
    """
    Compute the adaptive drift threshold τ(n, m).

    For single-user events (m=1):
        τ(n, 1) = α / √n
        This is grounded in eigenvalue perturbation theory — adding one
        edge to a graph of n nodes shifts eigenvalues by O(1/√n)
        (Weyl's inequality). α=1.5 is calibrated so the threshold sits
        safely above the observed single-user drift range on synthetic
        graphs (0.076–0.088 on n=11–15 nodes).

    For burst events (m > 1):
        τ(n, m) = (α / √n) × 1 / (1 + ln(m))
        The logarithmic decay reflects the diminishing marginal
        structural impact of each additional edge: the first few users
        cause the largest eigenvalue shifts; subsequent users have
        progressively smaller incremental impact (analogous to entropy
        scaling in information theory). When m=1, ln(1)=0 so the
        formula reduces exactly to the single-user threshold.

    Args:
        n:     Number of routers (nodes) in the ISP graph
        m:     Number of users joining simultaneously (default 1)
        alpha: Calibration constant (default 1.5, empirically derived)

    Returns:
        τ: drift threshold above which re-clustering is triggered
    """
    import math
    if n <= 0:
        n = 1
    if m <= 0:
        m = 1
    return alpha / (math.sqrt(n) * (1.0 + math.log(m)))


def fingerprint_drift(fp_old: np.ndarray, fp_new: np.ndarray,
                      n_nodes: int = None, m_users: int = 1) -> dict:
    """
    Measure how much a fingerprint has changed after a graph update.
    Used to decide whether to trigger re-clustering at the server.

    Uses adaptive threshold τ(n, m) = (1.5/√n) × 1/(1+ln m):
      - Scales with graph size n: small graphs need higher thresholds
        because a single node addition is a larger relative change
      - Scales with burst size m: larger bursts lower the threshold
        so significant structural changes are always detected
      - When m=1 (single user): reduces to τ = 1.5/√n
      - Falls back to fixed τ=0.10 if n_nodes not provided
        (for backward compatibility)

    Args:
        fp_old:   Fingerprint before the event
        fp_new:   Fingerprint after the event
        n_nodes:  Number of routers in the ISP graph (enables adaptive τ)
        m_users:  Number of users joining simultaneously (default 1)

    Returns:
        drift_score:      0 = identical, 1 = completely different
        cosine_sim:       cosine similarity between old and new
        threshold_used:   the τ value actually applied
        should_recluster: True if drift exceeds threshold
    """
    old_n = fp_old / (np.linalg.norm(fp_old) + 1e-8)
    new_n = fp_new / (np.linalg.norm(fp_new) + 1e-8)
    cos_sim = float(np.dot(old_n, new_n))
    drift   = 1.0 - cos_sim

    # Use adaptive threshold if graph size is known, else fall back to 0.10
    if n_nodes is not None:
        tau = compute_adaptive_threshold(n_nodes, m_users)
    else:
        tau = 0.10  # backward-compatible fallback

    return {
        "drift_score":      round(drift, 6),
        "cosine_sim":       round(cos_sim, 6),
        "threshold_used":   round(tau, 6),
        "should_recluster": drift > tau,
    }


if __name__ == "__main__":
    print("=== Topology Embedding Test ===\n")
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from data.isp_simulator import build_topology

    names = ["star", "mesh", "ring", "tree"]
    graphs = {n: build_topology(n, 10, seed=i) for i, n in enumerate(names)}

    print("Spectra (first 5 eigenvalues):")
    fps = {}
    for name, G in graphs.items():
        fp = compute_fingerprint(G, k=10, include_stats=False)
        fps[name] = fp
        print(f"  {name:6s}: {fp[:5]}")

    print("\nPairwise cosine similarity (should: star~star=1, star~mesh<1):")
    keys = list(fps.keys())
    header = f"{'':8}" + "".join(f"{k:>10}" for k in keys)
    print(header)
    for i, ki in enumerate(keys):
        row = f"{ki:8}"
        for j, kj in enumerate(keys):
            d = fingerprint_drift(fps[ki], fps[kj])
            row += f"{d['cosine_sim']:>10.3f}"
        print(row)

    print("\nDP noise effect (epsilon=0.1 vs epsilon=5.0):")
    fp_raw   = fps["star"]
    fp_noisy_strong = add_dp_noise(fp_raw, epsilon=0.1)
    fp_noisy_weak   = add_dp_noise(fp_raw, epsilon=5.0)
    d_strong = fingerprint_drift(fp_raw, fp_noisy_strong)
    d_weak   = fingerprint_drift(fp_raw, fp_noisy_weak)
    print(f"  eps=0.1 (strong privacy): drift={d_strong['drift_score']:.4f}")
    print(f"  eps=5.0 (weak  privacy): drift={d_weak['drift_score']:.4f}")