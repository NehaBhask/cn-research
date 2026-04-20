"""
clustered_aggregator.py
-----------------------
Clusters ISPs by topology fingerprint similarity and performs
weighted FedAvg within each cluster.

In this focused version there is NO GNN model — the "model weights"
being aggregated are intentionally generic (could be any FL model).
The contribution here is purely the clustering strategy.

Three methods compared:
  1. No federation  — each ISP isolated
  2. FedAvg         — all ISPs averaged together (standard baseline)
  3. Ours (Clustered) — topology-aware clustering then per-cluster FedAvg
"""

import numpy as np
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, adjusted_rand_score
from collections import defaultdict
from typing import Optional


class TopologyAwareAggregator:
    """
    Central server aggregation with topology-aware clustering.

    The server:
      1. Receives noisy fingerprints from all ISPs
      2. Computes pairwise cosine similarity
      3. Spectral-clusters ISPs into K groups
      4. Aggregates model weights within each cluster
      5. Evaluates clustering quality
    """

    def __init__(self, num_clusters: int = 3,
                 clustering_method: str = "spectral"):
        self.num_clusters       = num_clusters
        self.clustering_method  = clustering_method
        self.cluster_assignments: Optional[np.ndarray] = None
        self.similarity_matrix:   Optional[np.ndarray] = None
        self.cluster_centroids:   dict[int, np.ndarray] = {}
        self.round_number: int = 0

    def cluster(self, fingerprints: list) -> np.ndarray:
        """
        Cluster ISPs based on topology fingerprint similarity.

        Args:
            fingerprints: List of N noisy fingerprint vectors

        Returns:
            cluster_ids: np.ndarray of shape (N,)
        """
        N = len(fingerprints)
        if N <= self.num_clusters:
            return np.arange(N)

        fp_matrix = normalize(np.stack(fingerprints), norm="l2")
        self.similarity_matrix = fp_matrix @ fp_matrix.T
        affinity = (self.similarity_matrix + 1.0) / 2.0  # shift to [0,1]

        if self.clustering_method == "spectral":
            model = SpectralClustering(
                n_clusters=self.num_clusters, affinity="precomputed",
                assign_labels="kmeans", random_state=42, n_init=10)
        else:
            model = KMeans(n_clusters=self.num_clusters,
                           random_state=42, n_init=10)

        ids = model.fit_predict(affinity if self.clustering_method == "spectral"
                                else fp_matrix)
        self.cluster_assignments = ids

        # Compute centroids
        self.cluster_centroids = {}
        for i, fp in enumerate(fingerprints):
            cid = int(ids[i])
            self.cluster_centroids.setdefault(cid, []).append(fp)
        self.cluster_centroids = {
            k: np.mean(v, axis=0) for k, v in self.cluster_centroids.items()
        }
        return ids

    def aggregate(self, client_weights: list, fingerprints: list,
                  dataset_sizes: Optional[list] = None) -> dict:
        """
        Cluster then do weighted FedAvg within each cluster.

        Returns:
            cluster_models: dict mapping cluster_id -> aggregated weights
        """
        self.round_number += 1
        N = len(client_weights)
        if dataset_sizes is None:
            dataset_sizes = [1] * N

        ids = self.cluster(fingerprints)

        clusters = defaultdict(list)
        for i, cid in enumerate(ids):
            clusters[int(cid)].append(i)

        print(f"\n  [Round {self.round_number}] Clusters:")
        for cid, members in sorted(clusters.items()):
            print(f"    Cluster {cid}: ISPs {members}")

        cluster_models = {}
        for cid, members in clusters.items():
            sizes  = np.array([dataset_sizes[i] for i in members], dtype=float)
            weights = sizes / sizes.sum()
            agg = {}
            for key in client_weights[0].keys():
                agg[key] = sum(
                    weights[j] * client_weights[members[j]][key]
                    for j in range(len(members))
                )
            cluster_models[cid] = agg
        return cluster_models

    def get_model_for_client(self, client_idx: int,
                             cluster_models: dict) -> dict:
        cid = int(self.cluster_assignments[client_idx])
        return cluster_models[cid]

    def clustering_quality(self, fingerprints: list,
                           true_topo_labels: Optional[list] = None) -> dict:
        """
        Evaluate how well the clustering separates topology types.

        Metrics:
          silhouette_score: how tight and well-separated the clusters are
                            (-1 worst, +1 best)
          ARI:              compares predicted clusters to true topology
                            labels if available (1.0 = perfect match)
        """
        if self.cluster_assignments is None or len(set(self.cluster_assignments)) < 2:
            return {}

        fp_matrix = normalize(np.stack(fingerprints), norm="l2")
        sil = silhouette_score(fp_matrix, self.cluster_assignments)

        result = {"silhouette_score": round(float(sil), 4)}

        if true_topo_labels is not None:
            ari = adjusted_rand_score(true_topo_labels, self.cluster_assignments)
            result["adjusted_rand_index"] = round(float(ari), 4)

        return result

    def find_best_cluster_for_new_fingerprint(
            self, new_fp: np.ndarray,
            similarity_threshold: float = 0.75) -> dict:
        """
        Given a new ISP's fingerprint, find which existing cluster it
        belongs to. Used when a new ISP joins the federation.
        """
        if not self.cluster_centroids:
            return {"cluster_id": 0, "similarity": 0.0, "action": "first_client"}

        new_n = new_fp / (np.linalg.norm(new_fp) + 1e-8)
        best_cid, best_sim = -1, -1.0

        for cid, centroid in self.cluster_centroids.items():
            c_n   = centroid / (np.linalg.norm(centroid) + 1e-8)
            sim   = float(np.dot(new_n, c_n))
            if sim > best_sim:
                best_sim, best_cid = sim, cid

        action = "assigned" if best_sim >= similarity_threshold else "new_cluster"
        return {"cluster_id": best_cid, "similarity": round(best_sim, 4),
                "action": action}


class FedAvgAggregator:
    """Baseline: average ALL ISPs into one global model."""

    def aggregate(self, client_weights: list,
                  dataset_sizes: Optional[list] = None) -> dict:
        N = len(client_weights)
        sizes   = np.array(dataset_sizes or [1] * N, dtype=float)
        weights = sizes / sizes.sum()
        agg = {}
        for key in client_weights[0].keys():
            agg[key] = sum(weights[i] * client_weights[i][key] for i in range(N))
        return agg


if __name__ == "__main__":
    print("=== Clustered Aggregator Test ===\n")
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from data.isp_simulator import build_topology
    from core.topology_embedding import compute_fingerprint, add_dp_noise
    import numpy as np

    topos = ["star","star","mesh","mesh","ring","ring","tree","random"]
    graphs = [build_topology(t, 10, seed=i) for i, t in enumerate(topos)]
    fps    = [add_dp_noise(compute_fingerprint(G, k=15), epsilon=2.0)
              for G in graphs]

    dummy_weights = [{"w": np.random.randn(4, 4)} for _ in range(8)]

    agg = TopologyAwareAggregator(num_clusters=3)
    cluster_models = agg.aggregate(dummy_weights, fps)

    quality = agg.clustering_quality(fps, true_topo_labels=topos)
    print(f"\nClustering quality: {quality}")
    print("Expected: star ISPs together, mesh/ring similar, tree separate")