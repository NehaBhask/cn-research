"""
graph_dynamics.py
-----------------
The core novel contribution of this project.

Handles end-user devices (phones, laptops, IoT) dynamically
joining and leaving routers inside an ISP's network graph.

Each join/leave event:
  1. Updates the ISP's graph G_i (add or remove a node + edge)
  2. Recomputes the topology fingerprint
  3. Measures fingerprint drift from the previous state
  4. Decides whether to notify the server for re-clustering

Privacy is preserved throughout:
  - The actual graph G_i is never sent to the server
  - Only the (noisy) fingerprint is shared
  - Even the drift notification reveals only a scalar value

Two key questions answered per event:
  Q1: How much did the topology change? (drift score)
  Q2: Is the change big enough to need re-clustering? (threshold check)
"""

import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DynamicsEvent:
    """Records a single user join or leave event."""
    event_type:    str        # "join" or "leave"
    node_id:       int        # the device node
    router_id:     int        # which router it connected/disconnected from
    graph_before:  int        # number of nodes before
    graph_after:   int        # number of nodes after
    drift_score:   float      # fingerprint drift (0=no change, 1=full change)
    reclustered:   bool       # whether the server was asked to re-cluster
    epsilon_used:  float      # DP privacy budget used for this update


class GraphDynamicsHandler:
    """
    Manages dynamic user membership inside an ISP's network topology.

    An ISP uses this locally — the server never sees this object.
    When a re-cluster is needed, only the updated noisy fingerprint
    is sent to the server.
    """

    def __init__(self,
                 epsilon: float = 1.0):
        """
        Args:
            epsilon: DP privacy budget for fingerprint sharing.
                     Lower = stronger privacy, more noise.

        Note: drift threshold is now computed adaptively per event
        using τ(n, m) = 1.5/√n × 1/(1+ln m), so no fixed threshold
        is stored here.
        """
        self.epsilon = epsilon
        self.events: list[DynamicsEvent] = []

    # ──────────────────────────────────────────────────────────────────
    # Graph update operations
    # ──────────────────────────────────────────────────────────────────

    def user_joins(self, G: nx.Graph, router_id: int,
                   new_node_id: int = None) -> tuple:
        """
        A new device connects to router_id in graph G.

        Returns:
            G_new:    Updated graph (original G is not modified)
            node_id:  ID assigned to the new device node
        """
        G_new = G.copy()
        if new_node_id is None:
            new_node_id = max(G_new.nodes()) + 1 if G_new.nodes() else 0

        # Validate router exists
        if router_id not in G_new.nodes():
            router_id = list(G_new.nodes())[0]

        G_new.add_node(new_node_id)
        G_new.add_edge(new_node_id, router_id)

        print(f"    [JOIN]  Device {new_node_id} connected to Router {router_id} "
              f"| Graph: {G.number_of_nodes()} -> {G_new.number_of_nodes()} nodes")
        return G_new, new_node_id

    def user_leaves(self, G: nx.Graph, node_id: int) -> tuple:
        """
        Device node_id disconnects from the network.

        Returns:
            G_new:       Updated graph
            router_ids:  Which routers this device was connected to
        """
        if node_id not in G.nodes():
            print(f"    [LEAVE] Node {node_id} not found in graph — skipping")
            return G.copy(), []

        G_new      = G.copy()
        router_ids = list(G_new.neighbors(node_id))
        G_new.remove_node(node_id)

        print(f"    [LEAVE] Device {node_id} disconnected from {router_ids} "
              f"| Graph: {G.number_of_nodes()} -> {G_new.number_of_nodes()} nodes")
        return G_new, router_ids

    # ──────────────────────────────────────────────────────────────────
    # Drift detection and re-cluster decision
    # ──────────────────────────────────────────────────────────────────

    def process_event(self, event_type: str, G_old: nx.Graph,
                      G_new: nx.Graph, router_id: int,
                      node_id: int, k: int = 20) -> dict:
        """
        After a join or leave, compute drift and decide on re-clustering.

        This is the complete privacy-preserving pipeline:
          1. Compute raw fingerprints for old and new graph (private)
          2. Measure drift between them (private)
          3. If drift > threshold, prepare noisy fingerprint for server
          4. Return decision

        Args:
            event_type: "join" or "leave"
            G_old:      Graph before the event
            G_new:      Graph after the event
            router_id:  Router involved in the event
            node_id:    Device node involved
            k:          Fingerprint dimension

        Returns:
            result dict with all relevant information
        """
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from core.topology_embedding import (
            compute_fingerprint, add_dp_noise, fingerprint_drift
        )

        # Compute fingerprints (stays local — never sent raw)
        fp_old  = compute_fingerprint(G_old, k=k)
        fp_new  = compute_fingerprint(G_new, k=k)

        # Measure structural drift using adaptive threshold τ(n, m=1)
        # n_nodes = size of graph BEFORE the event (the reference graph)
        drift   = fingerprint_drift(fp_old, fp_new,
                                    n_nodes=G_old.number_of_nodes(),
                                    m_users=1)

        # Only send updated fingerprint to server if drift is significant
        should_recluster = drift["should_recluster"]
        noisy_fp_for_server = None
        if should_recluster:
            noisy_fp_for_server = add_dp_noise(fp_new, epsilon=self.epsilon)

        result = {
            "event_type":            event_type,
            "node_id":               node_id,
            "router_id":             router_id,
            "nodes_before":          G_old.number_of_nodes(),
            "nodes_after":           G_new.number_of_nodes(),
            "drift_score":           drift["drift_score"],
            "cosine_similarity":     drift["cosine_sim"],
            "should_recluster":      should_recluster,
            "noisy_fp_for_server":   noisy_fp_for_server,
            "privacy_epsilon":       self.epsilon,
        }

        # Log the event
        self.events.append(DynamicsEvent(
            event_type   = event_type,
            node_id      = node_id,
            router_id    = router_id,
            graph_before = G_old.number_of_nodes(),
            graph_after  = G_new.number_of_nodes(),
            drift_score  = drift["drift_score"],
            reclustered  = should_recluster,
            epsilon_used = self.epsilon,
        ))

        action = "RE-CLUSTER triggered" if should_recluster else "no re-cluster needed"
        print(f"      drift={drift['drift_score']:.4f} | {action}")

        return result

    # ──────────────────────────────────────────────────────────────────
    # Batch simulation
    # ──────────────────────────────────────────────────────────────────

    def simulate_user_session(self, G: nx.Graph, k: int = 20) -> dict:
        """
        Simulate a complete user session:
          1. User joins a random router
          2. User leaves (same session ends)

        Returns dict with both join and leave results.
        """
        routers = list(G.nodes())
        router  = int(np.random.choice(routers))

        print(f"\n    -- Session start: joining via Router {router} --")
        G_joined, new_node = self.user_joins(G, router_id=router)
        join_result = self.process_event("join", G, G_joined, router, new_node, k)

        print(f"\n    -- Session end: leaving --")
        G_left, _ = self.user_leaves(G_joined, node_id=new_node)
        leave_result = self.process_event("leave", G_joined, G_left, router, new_node, k)

        return {"join": join_result, "leave": leave_result,
                "graph_restored": G_left.number_of_nodes() == G.number_of_nodes()}

    def get_summary(self) -> str:
        """Print a summary table of all events."""
        if not self.events:
            return "No events recorded."

        lines = [
            f"\n  {'Event':<8} {'Node':>6} {'Router':>8} "
            f"{'Before':>8} {'After':>7} {'Drift':>8} {'Re-cluster':>12}",
            "  " + "-" * 62,
        ]
        for e in self.events:
            lines.append(
                f"  {e.event_type:<8} {e.node_id:>6} {e.router_id:>8} "
                f"{e.graph_before:>8} {e.graph_after:>7} "
                f"{e.drift_score:>8.4f} {str(e.reclustered):>12}"
            )

        recluster_count = sum(1 for e in self.events if e.reclustered)
        lines.append(f"\n  Total events: {len(self.events)} | "
                     f"Re-clusters triggered: {recluster_count}")
        return "\n".join(lines)


if __name__ == "__main__":
    print("=== Graph Dynamics Test ===\n")
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from data.isp_simulator import build_topology

    handler = GraphDynamicsHandler(drift_threshold=0.10, epsilon=1.0)

    # ── Test on star topology ────────────────────────────────────────
    print("-- Star topology (8 nodes) --")
    G_star = build_topology("star", 8, seed=1)

    # Single user session
    result = handler.simulate_user_session(G_star, k=15)
    print(f"  Graph restored after leave: {result['graph_restored']}")
    print(f"  Join drift:  {result['join']['drift_score']:.4f}")
    print(f"  Leave drift: {result['leave']['drift_score']:.4f}")

    # ── Multiple users join (stress test) ───────────────────────────
    print("\n-- 6 users join a mesh topology one by one --")
    G_mesh = build_topology("mesh", 9, seed=2)
    G_current = G_mesh.copy()

    from core.topology_embedding import compute_fingerprint
    fp_original = compute_fingerprint(G_mesh, k=15)

    for i in range(6):
        router = i % G_current.number_of_nodes()
        G_current, new_node = handler.user_joins(G_current, router_id=router)

        from core.topology_embedding import compute_fingerprint, fingerprint_drift
        fp_current = compute_fingerprint(G_current, k=15)
        drift = fingerprint_drift(fp_original, fp_current)
        print(f"  After {i+1} joins: nodes={G_current.number_of_nodes()}, "
              f"drift={drift['drift_score']:.4f}, "
              f"re-cluster={drift['should_recluster']}")

    print(handler.get_summary())