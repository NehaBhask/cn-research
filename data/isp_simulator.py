"""
isp_simulator.py
----------------
Simulates ISP network graphs with different topologies.
No traffic data, no congestion labels — just the graph structure.

Each ISP has a private graph G_i representing routers (nodes)
connected by cables (edges). The topology type determines how
congestion propagates — but that is NOT what we model here.

What this file provides:
  - Graph creation (star, mesh, ring, tree, random)
  - A list of ISPClient objects for the federation
"""

import numpy as np
import networkx as nx


TOPOLOGY_TYPES = ["star", "mesh", "ring", "tree", "random"]


def build_topology(topo_type: str, num_nodes: int, seed: int = 0) -> nx.Graph:
    """Build a NetworkX graph for the given topology type."""
    if topo_type == "star":
        G = nx.star_graph(num_nodes - 1)

    elif topo_type == "mesh":
        side = int(np.ceil(np.sqrt(num_nodes)))
        G = nx.grid_2d_graph(side, side)
        G = nx.convert_node_labels_to_integers(G)
        G = G.subgraph(list(G.nodes)[:num_nodes]).copy()

    elif topo_type == "ring":
        G = nx.cycle_graph(num_nodes)

    elif topo_type == "tree":
        G = nx.balanced_tree(r=2, h=int(np.log2(max(num_nodes, 2))))
        G = nx.convert_node_labels_to_integers(G)

    elif topo_type == "random":
        G = nx.erdos_renyi_graph(num_nodes, p=0.35, seed=seed)
        if not nx.is_connected(G):
            comps = list(nx.connected_components(G))
            for i in range(len(comps) - 1):
                u = list(comps[i])[0]
                v = list(comps[i + 1])[0]
                G.add_edge(u, v)
    else:
        raise ValueError(f"Unknown topology: {topo_type}")

    return G


class ISPClient:
    """
    One ISP in the federation.
    Holds only the private graph topology — no traffic data.
    """

    def __init__(self, client_id: int, topo_type: str,
                 num_nodes: int = 12, seed: int = None):
        self.client_id  = client_id
        self.topo_type  = topo_type
        _seed = seed if seed is not None else client_id * 13
        self.graph: nx.Graph = build_topology(topo_type, num_nodes, seed=_seed)
        self.num_nodes  = self.graph.number_of_nodes()

        # Filled later by topology_embedding.py
        self.fingerprint = None

    def __repr__(self):
        return (f"ISPClient(id={self.client_id}, topo={self.topo_type}, "
                f"nodes={self.num_nodes})")


def create_federation(num_clients: int = 8) -> list:
    """Create a diverse set of ISP clients with different topologies."""
    topo_cycle = (TOPOLOGY_TYPES * 4)[:num_clients]
    clients = []
    for i in range(num_clients):
        c = ISPClient(
            client_id=i,
            topo_type=topo_cycle[i],
            num_nodes=np.random.default_rng(i).integers(8, 16),
            seed=i * 13,
        )
        clients.append(c)
        print(f"  {c}")
    return clients


if __name__ == "__main__":
    print("=== ISP Federation ===")
    clients = create_federation(8)