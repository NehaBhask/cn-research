"""
main.py
-------
Focused experiments for:
  Privacy-Preserving Topology-Aware Federated Learning
  with Dynamic User Join/Leave Handling

Three experiments:

  Experiment 1: Clustering quality
    - How well does Laplacian fingerprinting group ISPs by topology?
    - Compare: no noise vs DP noise at various epsilon values
    - Metric: silhouette score, adjusted rand index vs true topology

  Experiment 2: Privacy vs clustering accuracy tradeoff
    - Sweep epsilon from 0.1 (strong privacy) to 10.0 (weak privacy)
    - Show how clustering quality degrades as privacy increases
    - This is the core privacy analysis for the paper

  Experiment 3: Dynamic user join/leave
    - Users join and leave routers inside each ISP's network
    - Track fingerprint drift per event
    - Show when re-clustering is triggered vs not triggered
    - Show that privacy is maintained throughout (only noisy fps shared)
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.isp_simulator      import create_federation
from core.topology_embedding  import (assign_fingerprints, compute_fingerprint,
                                       add_dp_noise, fingerprint_drift)
from core.clustered_aggregator import TopologyAwareAggregator
from core.graph_dynamics       import GraphDynamicsHandler

np.random.seed(42)
os.makedirs("results", exist_ok=True)


# ── Experiment 1: Clustering quality ──────────────────────────────────────────
def experiment_clustering_quality(clients, k=20):
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Clustering quality")
    print("=" * 60)

    # Ground truth topology labels (what the clustering SHOULD recover)
    true_labels = [c.topo_type for c in clients]

    # Raw fingerprints (no noise — best case)
    raw_fps = [compute_fingerprint(c.graph, k=k) for c in clients]

    agg = TopologyAwareAggregator(num_clusters=3)
    dummy_weights = [{"w": np.random.randn(4)} for _ in clients]
    agg.aggregate(dummy_weights, raw_fps)

    quality = agg.clustering_quality(raw_fps, true_topo_labels=true_labels)
    print(f"\n  No noise (epsilon=inf):")
    print(f"    Silhouette score:      {quality.get('silhouette_score', 'N/A')}")
    print(f"    Adjusted Rand Index:   {quality.get('adjusted_rand_index', 'N/A')}")
    print(f"    (ARI=1.0 means perfect recovery of topology groupings)")

    print(f"\n  Cluster assignments vs true topology:")
    print(f"  {'ISP':>4} {'Topology':>10} {'Cluster':>8}")
    print(f"  {'-'*26}")
    for i, c in enumerate(clients):
        print(f"  {i:>4} {c.topo_type:>10} {agg.cluster_assignments[i]:>8}")

    return quality


# ── Experiment 2: Privacy vs clustering accuracy tradeoff ─────────────────────
def experiment_privacy_tradeoff(clients, k=20, prefix="synthetic"):
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Privacy (epsilon) vs clustering quality")
    print("=" * 60)
    print("  Lower epsilon = stronger privacy = more noise = lower clustering quality")
    print("  Higher epsilon = weaker privacy  = less noise = better clustering quality")

    true_labels = [c.topo_type for c in clients]
    raw_fps     = [compute_fingerprint(c.graph, k=k) for c in clients]
    epsilons    = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    results     = []

    print(f"\n  {'Epsilon':>10} {'Silhouette':>12} {'ARI':>8} {'Privacy'}")
    print(f"  {'-'*48}")

    for eps in epsilons:
        sil_scores, ari_scores = [], []

        # Run 10 trials (noise is random — average over multiple seeds)
        import io, contextlib
        for trial in range(10):
            noisy_fps = [add_dp_noise(fp, epsilon=eps) for fp in raw_fps]
            agg = TopologyAwareAggregator(num_clusters=3)
            dummy_weights = [{"w": np.random.randn(4)} for _ in clients]
            with contextlib.redirect_stdout(io.StringIO()):  # suppress trial prints
                agg.aggregate(dummy_weights, noisy_fps)
            q = agg.clustering_quality(noisy_fps, true_topo_labels=true_labels)
            if q:
                sil_scores.append(q.get("silhouette_score", 0))
                ari_scores.append(q.get("adjusted_rand_index", 0))

        avg_sil = np.mean(sil_scores) if sil_scores else 0
        avg_ari = np.mean(ari_scores) if ari_scores else 0
        privacy_label = ("strong" if eps <= 0.5 else
                         "moderate" if eps <= 2.0 else "weak")

        results.append({"epsilon": eps, "silhouette": avg_sil, "ari": avg_ari})
        print(f"  {eps:>10.1f} {avg_sil:>12.4f} {avg_ari:>8.4f}   {privacy_label}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    eps_vals = [r["epsilon"] for r in results]
    ax1.plot(eps_vals, [r["silhouette"] for r in results], "-o", color="#534AB7", lw=2)
    ax1.set_xlabel("Privacy budget (epsilon)")
    ax1.set_ylabel("Silhouette score")
    ax1.set_title("Clustering quality vs privacy budget")
    ax1.set_xscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5, label="eps=1 (balanced)")
    ax1.legend()

    ax2.plot(eps_vals, [r["ari"] for r in results], "-s", color="#1D9E75", lw=2)
    ax2.set_xlabel("Privacy budget (epsilon)")
    ax2.set_ylabel("Adjusted Rand Index")
    ax2.set_title("Topology recovery vs privacy budget")
    ax2.set_xscale("log")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"results/privacy_tradeoff_{prefix}.png", dpi=150)
    print(f"\n  Plot saved to results/privacy_tradeoff_{prefix}.png")
    return results


# ── Experiment 3: Dynamic user join/leave ─────────────────────────────────────
def experiment_user_dynamics(clients, k=20, prefix="synthetic"):
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Dynamic user join/leave inside ISP networks")
    print("=" * 60)

    # Part A: single user sessions (expected: low drift, no re-cluster)
    print("\n  Part A: Single user join/leave sessions")
    print("  (adaptive threshold τ=1.5/√n — no re-cluster expected for single users)")

    all_drifts_join  = []
    all_drifts_leave = []
    recluster_counts = {"join": 0, "leave": 0}
    total_events     = {"join": 0, "leave": 0}

    for client in clients:
        import math
        tau = 1.5 / math.sqrt(client.num_nodes)
        topo_label = client.topo_type if client.topo_type not in ("unknown", "", None) else f"partition"
        print(f"\n  ISP {client.client_id} ({topo_label}, "
              f"{client.num_nodes} nodes, τ={tau:.4f}):")

        handler = GraphDynamicsHandler(epsilon=1.0)
        G = client.graph.copy()

        # Simulate 4 single-user sessions on this ISP
        for session in range(4):
            routers = list(G.nodes())
            router  = int(routers[session % len(routers)])

            G_joined, new_node = handler.user_joins(G, router_id=router)
            join_res = handler.process_event("join", G, G_joined, router, new_node, k)
            all_drifts_join.append(join_res["drift_score"])
            total_events["join"] += 1
            if join_res["should_recluster"]:
                recluster_counts["join"] += 1

            G_left, _ = handler.user_leaves(G_joined, node_id=new_node)
            leave_res = handler.process_event("leave", G_joined, G_left,
                                               router, new_node, k)
            all_drifts_leave.append(leave_res["drift_score"])
            total_events["leave"] += 1
            if leave_res["should_recluster"]:
                recluster_counts["leave"] += 1

        print(handler.get_summary())

    # Part B: burst of users joining (expected: drift crosses threshold)
    print("\n  " + "-"*54)
    print("  Part B: Burst of users joining simultaneously")
    print("  (expected: large drift -> re-cluster triggered)")
    print("  " + "-"*54)

    burst_results = []
    # Track per-burst-size results for summary table
    burst_summary = {}   # {n_users: {"drifts": [], "taus": [], "reclusters": []}}
    burst_sizes   = [1, 3, 5, 8, 10]
    for n in burst_sizes:
        burst_summary[n] = {"drifts": [], "taus": [], "reclusters": []}

    for client in clients[:3]:   # show on 3 ISPs for brevity
        topo_label = client.topo_type if client.topo_type not in ("unknown", "", None) else f"partition"
        print(f"\n  ISP {client.client_id} ({topo_label}):")
        from core.topology_embedding import compute_fingerprint

        fp_original = compute_fingerprint(client.graph, k=k)
        G_burst = client.graph.copy()
        joined_nodes = []
        handler_burst = GraphDynamicsHandler(epsilon=1.0)
        n_original = client.graph.number_of_nodes()  # reference graph size

        for n_users in burst_sizes:
            # Add one more user to reach n_users total
            while len(joined_nodes) < n_users:
                router = int(list(G_burst.nodes())[len(joined_nodes) % G_burst.number_of_nodes()])
                G_burst, new_node = handler_burst.user_joins(G_burst, router_id=router)
                joined_nodes.append((new_node, router))

            fp_current = compute_fingerprint(G_burst, k=k)
            from core.topology_embedding import fingerprint_drift, compute_adaptive_threshold
            # Use adaptive threshold with burst size m=n_users
            drift = fingerprint_drift(fp_original, fp_current,
                                      n_nodes=n_original, m_users=n_users)
            tau = compute_adaptive_threshold(n_original, n_users)
            recluster = drift["should_recluster"]
            burst_results.append(drift["drift_score"])

            # Store for summary
            burst_summary[n_users]["drifts"].append(drift["drift_score"])
            burst_summary[n_users]["taus"].append(tau)
            burst_summary[n_users]["reclusters"].append(recluster)

            flag = " <-- RE-CLUSTER TRIGGERED" if recluster else ""
            print(f"    {n_users:2d} users joined | drift={drift['drift_score']:.4f} | "
                  f"τ={tau:.4f} | re-cluster={recluster}{flag}")

        # Now all users leave one by one
        print(f"    Users leaving:")
        G_leave = G_burst.copy()
        for i, (node_id, router) in enumerate(joined_nodes):
            if node_id in G_leave.nodes():
                G_leave, _ = handler_burst.user_leaves(G_leave, node_id=node_id)
        fp_after_all_leave = compute_fingerprint(G_leave, k=k)
        drift_after = fingerprint_drift(fp_original, fp_after_all_leave)
        print(f"    All {len(joined_nodes)} users left | "
              f"drift from original={drift_after['drift_score']:.6f} | "
              f"Graph restored: {G_leave.number_of_nodes()} == {client.graph.number_of_nodes()} nodes")

    # ── Part B Summary Table ───────────────────────────────────────────────
    print(f"\n  {'='*64}")
    print(f"  Part B summary — burst users (averaged over 3 topologies):")
    print(f"  {'Users (m)':>10} {'Avg τ':>10} {'Avg Drift':>12} {'Re-clusters':>13} {'Triggered?':>11}")
    print(f"  {'-'*64}")
    for n_users in burst_sizes:
        data       = burst_summary[n_users]
        avg_tau    = np.mean(data["taus"])
        avg_drift  = np.mean(data["drifts"])
        n_recl     = sum(data["reclusters"])
        total_isp  = len(data["reclusters"])
        triggered  = "YES ✓" if n_recl > 0 else "no"
        print(f"  {n_users:>10} {avg_tau:>10.4f} {avg_drift:>12.4f} "
              f"{n_recl:>5}/{total_isp:<7} {triggered:>11}")

    # Summary across all ISPs
    print(f"\n  {'='*56}")
    print(f"  Part A summary — single user sessions, all {len(clients)} ISPs:")
    print(f"  {'Metric':<40} {'Join':>8} {'Leave':>8}")
    print(f"  {'-'*58}")
    print(f"  {'Total events':<40} {total_events['join']:>8} {total_events['leave']:>8}")
    print(f"  {'Re-clusters triggered':<40} {recluster_counts['join']:>8} {recluster_counts['leave']:>8}")
    print(f"  {'Re-cluster rate':<40} "
          f"{recluster_counts['join']/total_events['join']:.1%}".rjust(9) +
          f"{recluster_counts['leave']/total_events['leave']:.1%}".rjust(9))
    print(f"  {'Mean drift score':<40} {np.mean(all_drifts_join):>8.4f} {np.mean(all_drifts_leave):>8.4f}")
    print(f"  {'Max drift score':<40} {np.max(all_drifts_join):>8.4f} {np.max(all_drifts_leave):>8.4f}")

    # Plot drift distribution
    # Plot: single user drift distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.hist(all_drifts_join,  bins=12, alpha=0.65,
             label="User joins",  color="#534AB7")
    ax1.hist(all_drifts_leave, bins=12, alpha=0.65,
             label="User leaves", color="#1D9E75")
    ax1.set_xlabel("Fingerprint drift score")
    ax1.set_ylabel("Number of events")
    ax1.set_title("Part A: Single user — drift stays below adaptive τ=1.5/√n")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot: burst drift vs number of users
    burst_x = [1, 3, 5, 8, 10]
    colors  = ["#534AB7", "#1D9E75", "#D85A30"]
    markers = ["-o", "-s", "-^"]
    for idx, client in enumerate(clients[:3]):
        # Use topology type if available, else fall back to "Partition N"
        topo = getattr(client, "topo_type", None)
        if topo and topo not in ("unknown", ""):
            label = f"ISP {client.client_id} ({topo})"
        else:
            label = f"Partition {client.client_id}"
        slice_start = idx * len(burst_x)
        slice_end   = slice_start + len(burst_x)
        if len(burst_results) >= slice_end:
            ax2.plot(burst_x, burst_results[slice_start:slice_end],
                     markers[idx], color=colors[idx], lw=2, label=label)
    # Adaptive threshold curve: τ(n,m) = 1.5 / (√n × (1 + ln m))
    avg_n = np.mean([c.graph.number_of_nodes() for c in clients[:3]])
    from core.topology_embedding import compute_adaptive_threshold
    adaptive_taus = [compute_adaptive_threshold(avg_n, m) for m in burst_x]
    ax2.plot(burst_x, adaptive_taus, "--", color="#D85A30", lw=2,
             label=f"Adaptive τ(n≈{avg_n:.0f},m)")
    ax2.set_xlabel("Number of users joined simultaneously")
    ax2.set_ylabel("Fingerprint drift score")
    ax2.set_title("Part B: Burst of users — drift crosses adaptive threshold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"results/drift_distribution_{prefix}.png", dpi=150)
    print(f"\n  Plot saved to results/drift_distribution_{prefix}.png")

    return {
        "all_drifts_join":   all_drifts_join,
        "all_drifts_leave":  all_drifts_leave,
        "recluster_counts":  recluster_counts,
        "total_events":      total_events,
        "burst_results":     burst_results,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "#" * 60)
    print("  Privacy-Preserving Topology-Aware Federated Learning")
    print("  with Dynamic User Join/Leave Handling")
    print("#" * 60)

    print("\n[1/4] Creating ISP federation...")
    clients = create_federation(num_clients=8)

    print("\n[2/4] Computing topology fingerprints...")
    assign_fingerprints(clients, k=20, epsilon=1.0)

    print("\n[3/4] Running experiments...")
    q1 = experiment_clustering_quality(clients, k=20)
    q2 = experiment_privacy_tradeoff(clients, k=20)
    q3 = experiment_user_dynamics(clients, k=20)

    
    # ── GEANT real-dataset validation ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("GEANT REAL DATASET VALIDATION")
    print("=" * 60)
    print("  (22 nodes, 36 links — real European backbone)")

    try:
        from data.geant_loader import create_geant_clients
        import io, contextlib

        geant_path = "data/geant.txt"
        geant_clients = create_geant_clients(geant_path, num_partitions=4)

        print("\n  [GEANT] Computing fingerprints...")
        assign_fingerprints(geant_clients, k=20, epsilon=1.0)

        print("\n  [GEANT] Experiment 1 — Clustering quality:")
        q_geant = experiment_clustering_quality(geant_clients, k=20)

        print("\n  [GEANT] Experiment 2 — Privacy tradeoff:")
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            q2_geant = experiment_privacy_tradeoff(geant_clients, k=20, prefix="geant")
        # Print just the table, not the per-trial noise
        lines = captured.getvalue().split("\n")
        for line in lines:
            if any(x in line for x in ["Epsilon","---","strong","moderate",
                                        "weak","Plot","Lower","Higher"]):
                print(" ", line)

        print("\n  [GEANT] Experiment 3 — User join/leave dynamics:")
        experiment_user_dynamics(geant_clients, k=20, prefix="geant")

    except Exception as e:
        print(f"  GEANT experiments failed: {e}")
        import traceback
        traceback.print_exc()
    print("\n[4/4] Final summary for paper:")
    print(f"\n  Clustering (no noise): "
          f"Silhouette={q1.get('silhouette_score','N/A')}, "
          f"ARI={q1.get('adjusted_rand_index','N/A')}")
    print(f"  Privacy tradeoff: eps=1.0 gives "
          f"Silhouette={next(r['silhouette'] for r in q2 if r['epsilon']==1.0):.4f}")
    print(f"  Dynamic events: "
          f"{q3['total_events']['join']+q3['total_events']['leave']} total, "
          f"{q3['recluster_counts']['join']+q3['recluster_counts']['leave']} "
          f"re-clusters triggered")
    print(f"\n  Results saved to ./results/")


if __name__ == "__main__":
    main()