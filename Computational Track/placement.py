import math
import random
from typing import List, Tuple, Dict, Callable
import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment


def build_interaction_graph(program: List[Tuple]) -> nx.Graph:
    """Builds a weighted logical interaction graph from 2Q gates."""
    g = nx.Graph()
    for op in program:
        if op[0] == '2Q':
            u, v = op[1], op[2]
            w = g[u][v]['weight'] + 1 if g.has_edge(u, v) else 1
            g.add_edge(u, v, weight=w)
    return g


def get_active_logical(program: List[Tuple]) -> List[int]:
    return sorted(list({q for gate in program for q in gate[1:]}))


# Strategy Generators

def generate_all_hub_placements(program: List[Tuple], hardware_graph: nx.Graph, dist_matrix: Dict) -> List[Dict]:
    active_logical = get_active_logical(program)
    if not active_logical:
        return []

    int_graph = build_interaction_graph(program)
    if int_graph.nodes:
        l_hub = max(active_logical, key=lambda l: int_graph.degree(l, weight='weight') if l in int_graph else 0)
        logical_order = list(nx.bfs_tree(int_graph, source=l_hub).nodes)
    else:
        l_hub = active_logical[0]
        logical_order = [l_hub]

    for l in active_logical:
        if l not in logical_order:
            logical_order.append(l)

    placements = []
    for p_hub in sorted(hardware_graph.nodes):
        phys_by_proximity = sorted(
            hardware_graph.nodes,
            key=lambda p: (dist_matrix[p_hub][p], -hardware_graph.degree[p])
        )
        placements.append({l: p for l, p in zip(logical_order, phys_by_proximity)})
    return placements


def generate_linear_path_placements(program: List[Tuple], hardware_graph: nx.Graph, dist_matrix: Dict) -> List[Dict]:
    active_logical = get_active_logical(program)
    if not active_logical:
        return []

    int_graph = build_interaction_graph(program)
    endpoints = [node for node in int_graph.nodes if int_graph.degree[node] == 1]
    start_logical = endpoints[0] if endpoints else active_logical[0]

    logical_chain = list(nx.dfs_preorder_nodes(int_graph, source=start_logical))
    for l in active_logical:
        if l not in logical_chain:
            logical_chain.append(l)

    path_placements = []
    for start_phys in sorted(hardware_graph.nodes):
        phys_path = []
        visited = set()

        def dfs_physical(curr):
            visited.add(curr)
            phys_path.append(curr)
            if len(phys_path) == len(active_logical):
                return True
            for neighbor in hardware_graph.neighbors(curr):
                if neighbor not in visited and dfs_physical(neighbor):
                    return True
            return False

        dfs_physical(start_phys)
        if len(phys_path) < len(active_logical):
            for p in hardware_graph.nodes:
                if p not in visited:
                    phys_path.append(p)
                    if len(phys_path) == len(active_logical):
                        break

        path_placements.append({l: p for l, p in zip(logical_chain, phys_path)})
    return path_placements


def generate_spectral_placements(program: List[Tuple], hardware_graph: nx.Graph, dist_matrix: Dict) -> List[Dict]:
    active_logical = get_active_logical(program)
    if len(active_logical) <= 2:
        return []

    int_graph = build_interaction_graph(program)
    hw_nodes = list(hardware_graph.nodes)

    try:
        logical_pos = nx.spectral_layout(int_graph, weight='weight', dim=2)
    except Exception:
        logical_pos = nx.spring_layout(int_graph, weight='weight', dim=2, seed=42)

    try:
        hw_pos = nx.spectral_layout(hardware_graph, dim=2)
    except Exception:
        hw_pos = nx.spring_layout(hardware_graph, dim=2, seed=42)

    cost_matrix = np.zeros((len(active_logical), len(hw_nodes)))
    for i, l in enumerate(active_logical):
        l_coord = logical_pos.get(l, np.zeros(2))
        for j, p in enumerate(hw_nodes):
            cost_matrix[i, j] = np.linalg.norm(l_coord - hw_pos[p])

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return [{active_logical[r]: hw_nodes[c] for r, c in zip(row_ind, col_ind)}]


def compute_circuit_dag_layers(program: list[tuple]) -> list[int]:
    """Computes the topological layer (depth) of each gate in the circuit."""
    qubit_layer = {}
    gate_layers = []
    for op in program:
        if op[0] == '2Q':
            u, v = op[1], op[2]
            layer = max(qubit_layer.get(u, 0), qubit_layer.get(v, 0))
            gate_layers.append(layer)
            qubit_layer[u] = layer + 1
            qubit_layer[v] = layer + 1
        else:
            q = op[1]
            layer = qubit_layer.get(q, 0)
            gate_layers.append(layer)
            qubit_layer[q] = layer + 1
    return gate_layers


def generate_qap_sa_placements(
    program: list[tuple], 
    hardware_graph: nx.Graph, 
    dist_matrix: dict = None, 
    num_seeds: int = 5,
    gamma: float = 0.8
) -> list[dict]:
    """
    QAP placement using DAG Layer-Decayed Weights (w = gamma^layer).
    """
    active_logical = sorted(list({q for gate in program for q in gate[1:]}))
    hw_nodes = sorted(list(hardware_graph.nodes))
    k = len(active_logical)
    if k == 0:
        return []

    # Use provided dist_matrix or compute if missing
    if dist_matrix is None:
        dist_matrix = dict(nx.all_pairs_shortest_path_length(hardware_graph))

    # Compute gate layers
    gate_layers = compute_circuit_dag_layers(program)

    # Build DAG-weighted interaction graph
    int_graph = nx.Graph()
    for gate_idx, gate in enumerate(program):
        if gate[0] == '2Q':
            u, v = gate[1], gate[2]
            layer = gate_layers[gate_idx]
            w = gamma ** layer
            if int_graph.has_edge(u, v):
                int_graph[u][v]['weight'] += w
            else:
                int_graph.add_edge(u, v, weight=w)

    def compute_qap_cost(mapping: dict) -> float:
        return sum(
            data.get('weight', 1.0) * dist_matrix[mapping[u]][mapping[v]]
            for u, v, data in int_graph.edges(data=True)
            if u in mapping and v in mapping
        )

    def run_2opt_steepest_descent(mapping: dict, current_cost: float) -> dict:
        curr_mapping = dict(mapping)
        curr_cost = current_cost

        while True:
            best_delta = 0.0
            best_move = None

            # 1. Logical-Logical 2-Swaps
            for i in range(k):
                for j in range(i + 1, k):
                    l1, l2 = active_logical[i], active_logical[j]
                    curr_mapping[l1], curr_mapping[l2] = curr_mapping[l2], curr_mapping[l1]
                    new_cost = compute_qap_cost(curr_mapping)
                    delta = new_cost - curr_cost

                    if delta < best_delta:
                        best_delta = delta
                        best_move = ('swap', l1, l2)

                    curr_mapping[l1], curr_mapping[l2] = curr_mapping[l2], curr_mapping[l1]

            # 2. Logical-Empty Moves
            unassigned_phys = [p for p in hw_nodes if p not in curr_mapping.values()]
            for l in active_logical:
                old_p = curr_mapping[l]
                for empty_p in unassigned_phys:
                    curr_mapping[l] = empty_p
                    new_cost = compute_qap_cost(curr_mapping)
                    delta = new_cost - curr_cost

                    if delta < best_delta:
                        best_delta = delta
                        best_move = ('move', l, empty_p)

                    curr_mapping[l] = old_p

            if best_move and best_delta < -1e-5:
                if best_move[0] == 'swap':
                    _, l1, l2 = best_move
                    curr_mapping[l1], curr_mapping[l2] = curr_mapping[l2], curr_mapping[l1]
                else:
                    _, l, new_p = best_move
                    curr_mapping[l] = new_p
                curr_cost += best_delta
            else:
                break

        return curr_mapping

    placements = []

    for seed in range(num_seeds):
        random.seed(42 + seed)
        initial_phys = random.sample(hw_nodes, k)
        curr_mapping = {l: p for l, p in zip(active_logical, initial_phys)}
        curr_cost = compute_qap_cost(curr_mapping)

        # Stage 1: SA Exploration
        temp = 15.0
        cooling = 0.94
        for _ in range(150):
            l1, l2 = random.sample(active_logical, 2)
            curr_mapping[l1], curr_mapping[l2] = curr_mapping[l2], curr_mapping[l1]
            new_cost = compute_qap_cost(curr_mapping)

            delta = new_cost - curr_cost
            if delta < 0 or math.exp(-delta / temp) > random.random():
                curr_cost = new_cost
            else:
                curr_mapping[l1], curr_mapping[l2] = curr_mapping[l2], curr_mapping[l1]

            temp *= cooling

        # Stage 2: 2-Opt Polish
        polished_mapping = run_2opt_steepest_descent(curr_mapping, curr_cost)
        placements.append(polished_mapping)

    return placements

def generate_mwst_placements(program: List[Tuple], hardware_graph: nx.Graph, dist_matrix: Dict) -> List[Dict]:
    """
    Extracts Maximum Weight Spanning Tree (MWST) from interaction graph 
    and maps outward from physical anchors.
    """
    active_logical = get_active_logical(program)
    if len(active_logical) <= 2:
        return []

    int_graph = build_interaction_graph(program)
    if not int_graph.edges:
        return []

    mwst = nx.maximum_spanning_tree(int_graph, weight='weight')
    root_logical = max(active_logical, key=lambda l: (mwst.degree(l) if l in mwst else 0))
    logical_tree_order = list(nx.bfs_tree(mwst, source=root_logical).nodes)
    for l in active_logical:
        if l not in logical_tree_order:
            logical_tree_order.append(l)

    placements = []
    for p_anchor in sorted(hardware_graph.nodes):
        phys_by_proximity = sorted(
            hardware_graph.nodes,
            key=lambda p: (dist_matrix[p_anchor][p], -hardware_graph.degree[p])
        )
        placements.append({l: p for l, p in zip(logical_tree_order, phys_by_proximity)})

    return placements


def generate_compact_subgrid_placements(
    program: List[Tuple], hardware_graph: nx.Graph, dist_matrix: Dict, num_clusters: int = 5
) -> List[Dict]:
    """
    Constrains layout to compact contiguous subgraphs on target hardware 
    using spectral matching.
    """
    active_logical = get_active_logical(program)
    k = len(active_logical)
    if k == 0 or k > len(hardware_graph):
        return []

    int_graph = build_interaction_graph(program)
    placements = []
    seeds = sorted(hardware_graph.nodes, key=lambda p: hardware_graph.degree[p], reverse=True)[:num_clusters]

    for seed in seeds:
        subgrid_phys = sorted(
            hardware_graph.nodes, 
            key=lambda p: (dist_matrix[seed][p], -hardware_graph.degree[p])
        )[:k]
        sub_hw = hardware_graph.subgraph(subgrid_phys)

        try:
            logical_pos = nx.spectral_layout(int_graph, weight='weight', dim=2)
        except Exception:
            logical_pos = nx.spring_layout(int_graph, weight='weight', dim=2, seed=42)

        try:
            hw_pos = nx.spectral_layout(sub_hw, dim=2)
        except Exception:
            hw_pos = nx.spring_layout(sub_hw, dim=2, seed=42)

        cost_matrix = np.zeros((k, k))
        for i, l in enumerate(active_logical):
            l_coord = logical_pos.get(l, np.zeros(2))
            for j, p in enumerate(subgrid_phys):
                p_coord = hw_pos[p]
                cost_matrix[i, j] = np.linalg.norm(l_coord - p_coord)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        placements.append({active_logical[r]: subgrid_phys[c] for r, c in zip(row_ind, col_ind)})

    return placements


# Strategy Registry for Quick Pipeline Configuration
PLACEMENT_STRATEGIES: List[Callable] = [
    generate_all_hub_placements,
    generate_linear_path_placements,
    generate_spectral_placements,
    generate_qap_sa_placements,
    generate_mwst_placements,
    generate_compact_subgrid_placements,
]


def collect_all_placements(program: List[Tuple], hardware_graph: nx.Graph, dist_matrix: Dict) -> List[Dict]:
    raw_placements = []
    for strategy in PLACEMENT_STRATEGIES:
        raw_placements.extend(strategy(program, hardware_graph, dist_matrix))

    # Deduplicate
    placements = []
    seen = set()
    for p in raw_placements:
        frozen = tuple(sorted(p.items()))
        if frozen not in seen:
            seen.add(frozen)
            placements.append(p)
    return placements