"""
geant_loader.py
---------------
Loads the GEANT backbone network from SNDlib native format.

The file you have is the complete GEANT network — save it as:
  data/geant.txt

It contains three sections:
  NODES  — 22 European PoP routers with coordinates
  LINKS  — 36 physical fiber connections between routers
  DEMANDS — traffic volumes between all city pairs

Usage:
  python data/geant_loader.py          # test the loader
  from data.geant_loader import load_geant, create_geant_clients
"""

import os
import re
import networkx as nx
import numpy as np


GEANT_NODES = [
    "at","be","ch","cz","de","es","fr","gr",
    "hr","hu","ie","il","it","lu","nl","ny",
    "pl","pt","se","si","sk","uk"
]


def _short(full_id: str) -> str:
    """Convert 'at1.at' -> 'at', 'ny1.ny' -> 'ny'"""
    return full_id.strip().split(".")[-1].lower()


def load_geant(path: str = None) -> nx.Graph:
    """
    Load GEANT topology from SNDlib native format file.

    Save your file as data/geant.txt and pass that path.
    Falls back to hardcoded topology if file not found.

    Args:
        path: Path to your geant.txt (or folder containing it)

    Returns:
        G: NetworkX graph with 22 nodes and 36 edges
    """
    # Resolve path
    filepath = None
    if path:
        if os.path.isfile(path):
            filepath = path
        elif os.path.isdir(path):
            # Search folder for the native format file
            for f in sorted(os.listdir(path)):
                fp = os.path.join(path, f)
                if os.path.isfile(fp):
                    try:
                        with open(fp, "r", errors="ignore") as fh:
                            content = fh.read(500)
                        if "NODES" in content and "LINKS" in content:
                            filepath = fp
                            print(f"  Found topology file: {f}")
                            break
                    except Exception:
                        pass

    if filepath is None:
        print("  File not found — using hardcoded GEANT topology.")
        return _hardcoded()

    try:
        return _parse_native(filepath)
    except Exception as e:
        print(f"  Parse error: {e} — using hardcoded.")
        return _hardcoded()


def _parse_native(filepath: str) -> nx.Graph:
    """
    Parse SNDlib native format file.

    NODES section lines:  node_id ( longitude latitude )
    LINKS section lines:  link_id ( source target ) 0.00 0.00 0.00 0.00 ( cap cost )
    DEMANDS section lines: demand_id ( source target ) 1 value UNLIMITED

    Key distinction: LINKS lines have 0.00 immediately after the ) of (src tgt).
    DEMANDS lines have an integer (1) after the ). We use this to parse only links.
    """
    with open(filepath, "r", errors="ignore") as f:
        text = f.read()

    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    node_idx = {}
    G = nx.Graph()

    # ── Parse NODES ───────────────────────────────────────────────────
    # Each node line: node_id ( x y )
    node_pattern = re.compile(
        r"^\s*(\w[\w.]*?)\s*\(\s*[-\d.]+\s+[-\d.]+\s*\)",
        re.MULTILINE
    )
    for m in node_pattern.finditer(text):
        full_id = m.group(1)
        # Skip section headers like NODES, LINKS, DEMANDS, META
        if full_id.upper() in ("NODES","LINKS","DEMANDS","META",
                               "ADMISSIBLE_PATHS","ADMISSIBLE"):
            continue
        short = _short(full_id)
        if short and short not in node_idx:
            idx = len(node_idx)
            node_idx[short] = idx
            G.add_node(idx, label=short, full_id=full_id)

    # ── Parse LINKS ───────────────────────────────────────────────────
    # Link lines: link_id ( src tgt ) 0.00 0.00 0.00 0.00 ( cap cost )
    # The "0.00" right after ) distinguishes links from demands (which have "1")
    link_pattern = re.compile(
        r"^\s*\S+\s+\(\s*(\S+)\s+(\S+)\s*\)\s+0\.00",
        re.MULTILINE
    )
    links_added = 0
    for m in link_pattern.finditer(text):
        src = _short(m.group(1))
        tgt = _short(m.group(2))
        s   = node_idx.get(src)
        t   = node_idx.get(tgt)
        if s is not None and t is not None and s != t:
            G.add_edge(s, t)
            links_added += 1

    if links_added == 0:
        print("  Warning: no links parsed — using hardcoded fallback.")
        return _hardcoded()

    # Ensure connectivity
    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        for i in range(len(comps) - 1):
            u = list(comps[i])[0]
            v = list(comps[i + 1])[0]
            G.add_edge(u, v)

    print(f"  Loaded GEANT from file: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")
    return G


def _parse_demands(filepath: str) -> np.ndarray:
    """
    Parse DEMANDS section to get traffic volumes.
    Returns a (22, 22) matrix of Mbps values.
    """
    with open(filepath, "r", errors="ignore") as f:
        text = f.read()

    node_list = GEANT_NODES
    node_idx  = {n: i for i, n in enumerate(node_list)}
    n         = len(node_list)
    T         = np.zeros((n, n))

    demands_block = re.search(r"DEMANDS\s*\((.*?)\)", text, re.DOTALL)
    if not demands_block:
        return T

    # Format: demand_id ( source target ) routing_unit demand_value max_path
    demand_pattern = re.compile(
        r"\w[\w.]*\s*\(\s*(\w+\.\w+)\s+(\w+\.\w+)\s*\)\s+\d+\s+([\d.]+)"
    )
    for m in demand_pattern.finditer(demands_block.group(1)):
        src = _short(m.group(1))
        tgt = _short(m.group(2))
        val = float(m.group(3))
        s   = node_idx.get(src)
        t   = node_idx.get(tgt)
        if s is not None and t is not None:
            T[s][t] = val

    return T


def _hardcoded() -> nx.Graph:
    """Exact GEANT topology hardcoded from SNDlib (22 nodes, 36 edges)."""
    node_idx = {n: i for i, n in enumerate(GEANT_NODES)}
    edges = [
        ("at","ch"),("at","de"),("at","hu"),("at","ny"),("at","si"),
        ("be","fr"),("be","lu"),("be","nl"),("ch","fr"),("ch","it"),
        ("cz","de"),("cz","pl"),("cz","sk"),("de","fr"),("de","gr"),
        ("de","ie"),("de","it"),("de","nl"),("de","se"),("es","fr"),
        ("es","it"),("es","pt"),("fr","lu"),("fr","uk"),("gr","it"),
        ("hr","hu"),("hr","si"),("hu","sk"),("ie","uk"),("il","it"),
        ("il","nl"),("nl","uk"),("ny","uk"),("pl","se"),("pt","uk"),
        ("se","uk"),
    ]
    G = nx.Graph()
    for i, name in enumerate(GEANT_NODES):
        G.add_node(i, label=name)
    for a, b in edges:
        G.add_edge(node_idx[a], node_idx[b])
    print(f"  Loaded hardcoded GEANT: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")
    return G


def create_geant_clients(path: str = None, num_partitions: int = 4) -> list:
    """
    Load GEANT and split into ISP client subgraphs for the federation.

    Args:
        path:            Path to geant.txt, or folder containing it
        num_partitions:  Number of ISP partitions (clients) to create
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from data.isp_simulator import ISPClient

    G_full = load_geant(path)
    nodes  = list(G_full.nodes())
    n      = len(nodes)
    psize  = max(1, n // num_partitions)

    clients = []
    for i in range(num_partitions):
        start  = i * psize
        end    = start + psize if i < num_partitions - 1 else n
        pnodes = nodes[start:end]

        G_sub = G_full.subgraph(pnodes).copy()
        G_sub = nx.convert_node_labels_to_integers(G_sub)

        if G_sub.number_of_nodes() > 1 and not nx.is_connected(G_sub):
            comps = list(nx.connected_components(G_sub))
            for j in range(len(comps) - 1):
                u = list(comps[j])[0]
                v = list(comps[j + 1])[0]
                G_sub.add_edge(u, v)

        client             = ISPClient.__new__(ISPClient)
        client.client_id   = i
        client.topo_type   = f"geant_p{i}"
        client.graph       = G_sub
        client.num_nodes   = G_sub.number_of_nodes()
        client.fingerprint = None
        clients.append(client)

        labels = [G_full.nodes[n].get("label","?") for n in pnodes[:G_sub.number_of_nodes()]]
        print(f"  Partition {i}: {G_sub.number_of_nodes()} nodes "
              f"({', '.join(labels[:5])}{'...' if len(labels)>5 else ''}), "
              f"{G_sub.number_of_edges()} edges")

    return clients


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from core.topology_embedding import compute_fingerprint, fingerprint_drift

    print("=== GEANT Loader Test ===\n")

    # Search common locations for your file
    search = ["data/geant.txt","data/geant","geant.txt","geant","."]
    path = next((p for p in search if os.path.exists(p)), None)
    print(f"Using: {path or 'hardcoded fallback'}\n")

    G = load_geant(path)

    print(f"\nGraph properties:")
    print(f"  Nodes     : {G.number_of_nodes()}")
    print(f"  Edges     : {G.number_of_edges()}")
    print(f"  Avg degree: {sum(d for _,d in G.degree())/G.number_of_nodes():.2f}")
    print(f"  Density   : {nx.density(G):.4f}")
    print(f"  Connected : {nx.is_connected(G)}")

    labels = [G.nodes[n].get("label", str(n)) for n in sorted(G.nodes())]
    print(f"\n22 PoP nodes: {', '.join(labels)}")

    fp = compute_fingerprint(G, k=20)
    print(f"\nFingerprint[:6]: {fp[:6].round(4)}")

    print("\nCreating 4-ISP federation from GEANT:")
    clients = create_geant_clients(path, num_partitions=4)

    print("\nPairwise fingerprint similarity:")
    fps = [compute_fingerprint(c.graph, k=20) for c in clients]
    header = f"  {'':14}" + "".join(f"Part{i:>8}" for i in range(len(fps)))
    print(header)
    for i in range(len(fps)):
        row = f"  Partition {i}:   "
        for j in range(len(fps)):
            sim = fingerprint_drift(fps[i], fps[j])["cosine_sim"]
            row += f"{sim:>10.3f}"
        print(row)

    print("\nDone. To use in main.py:")
    print("  from data.geant_loader import create_geant_clients")
    print("  clients = create_geant_clients('data/geant.txt', num_partitions=4)")

