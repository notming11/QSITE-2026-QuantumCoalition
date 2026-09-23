import concurrent.futures
import multiprocessing
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict
import networkx as nx
import math

from sabre_core import sabre_layout, SabreParams
from placement import collect_all_placements, get_active_logical


@dataclass
class SearchSpace:
    w_options: List[float]
    ext_options: List[int]
    decay_options: List[float]
    bwd_w_multiplier: float = 1.25  # Scales forward W for stronger backward lookahead
    bwd_w_max: float = 0.8          # Caps maximum backward W to avoid lookahead thrashing
    noise_scale: float = 0.05       # 5% cost perturbation scale
    num_stochastic_seeds: int = 3   # Number of stochastic trials per config


DEFAULT_SEARCH_SPACE = SearchSpace(
    w_options=[0.35, 0.4, 0.45, 0.5],
    ext_options=[10, 15],
    decay_options=[0.01],
    bwd_w_multiplier=1.25,
    bwd_w_max=0.8,
    noise_scale=0.05,
    num_stochastic_seeds=6
)

def compute_circuit_lower_bound(program: list[tuple], active_qubits: list[int]) -> float:
    """Computes the absolute mathematical minimum score possible for a circuit."""
    num_2q = sum(1 for op in program if op[0] == '2Q')
    if num_2q == 0:
        return 0.0

    # 1. Critical path depth (pure DAG dependency bound)
    qubit_depth = {}
    for op in program:
        if op[0] == '2Q':
            u, v = op[1], op[2]
            d = max(qubit_depth.get(u, 0), qubit_depth.get(v, 0)) + 1
            qubit_depth[u] = d
            qubit_depth[v] = d
    critical_path = max(qubit_depth.values()) if qubit_depth else 0

    # 2. Maximum parallelism bound
    max_pairs = max(1, len(active_qubits) // 2)
    capacity_bound = math.ceil(num_2q / max_pairs)

    min_depth = max(critical_path, capacity_bound)
    return 0.5 * min_depth  # 0 SWAPs + 0.5 * min_depth

def _evaluate_single_config(args) -> Tuple[float, Dict, List[Tuple]]:
    (
        program,
        hardware_graph,
        dist_matrix,
        placement,
        W,
        ext,
        decay,
        max_iterations,
        bwd_w_multiplier,
        bwd_w_max,
        noise_scale,
        seed,
    ) = args

    # Seed local worker process RNG
    random.seed(seed)

    # 1. Forward Pass Parameters with Noise
    fwd_params = SabreParams(W=W, ext_size=ext, decay_delta=decay, noise_scale=noise_scale)

    # 2. Asymmetric Backward Pass Parameters with Noise
    bwd_W = min(bwd_w_max, W * bwd_w_multiplier)
    bwd_params = SabreParams(W=bwd_W, ext_size=ext, decay_delta=0.0, noise_scale=noise_scale)
    # print(placement)
    refined_placement, routed, score = sabre_layout(
        program=program,
        hardware_graph=hardware_graph,
        dist_matrix=dist_matrix,
        initial_placement=placement,
        fwd_params=fwd_params,
        bwd_params=bwd_params,
        max_iterations=max_iterations
    )
    return score, refined_placement, routed

def cancel_redundant_swaps(routed_program: List[Tuple]) -> List[Tuple]:
    """Removes consecutive identical SWAPs on the same physical qubits."""
    cleaned = []
    for op in routed_program:
        if op[0] == 'SWAP' and cleaned:
            last_op = cleaned[-1]
            if last_op[0] == 'SWAP' and tuple(sorted(op[1:])) == tuple(sorted(last_op[1:])):
                cleaned.pop()  # Cancel out mutual SWAPs
                continue
        cleaned.append(op)
    return cleaned

def solve(
    program: List[Tuple],
    hardware_graph: nx.Graph,
    search_space: SearchSpace = DEFAULT_SEARCH_SPACE,
    parallel: bool = True
) -> Tuple[Dict[int, int], List[Tuple]]:
    active_logical = get_active_logical(program)
    
    # Precompute hardware distance matrix ONCE globally
    dist_matrix = dict(nx.all_pairs_shortest_path_length(hardware_graph))

    # 1. Gather Candidate Placements
    placements = collect_all_placements(program, hardware_graph, dist_matrix)

    # 2. Build Task Cross-Product Search Grid with Stochastic Seeds
    tasks = [
        (
            program,
            hardware_graph,
            dist_matrix,
            placement,
            W,
            ext,
            decay,
            7,  # max_iterations
            search_space.bwd_w_multiplier,
            search_space.bwd_w_max,
            search_space.noise_scale,
            1000 + seed_idx,
        )
        for placement in placements
        for W in search_space.w_options
        for ext in search_space.ext_options
        for decay in search_space.decay_options
        for seed_idx in range(search_space.num_stochastic_seeds)
    ]

    best_overall_score = float('inf')
    best_placement = None
    best_routed_program = None

    # 3. Execution
    if parallel:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(_evaluate_single_config, tasks, chunksize=16)
            
            min_possible_score = compute_circuit_lower_bound(program, active_logical)
            
            for score, refined_placement, routed in results:
                if score < best_overall_score:
                    best_overall_score = score
                    best_placement = {l: refined_placement[l] for l in active_logical}
                    best_routed_program = routed
                    
                if best_overall_score <= min_possible_score:
                    return best_placement, cancel_redundant_swaps(best_routed_program)
    else:
        for task in tasks:
            score, refined_placement, routed = _evaluate_single_config(task)
            if score < best_overall_score:
                best_overall_score = score
                best_placement = {l: refined_placement[l] for l in active_logical}
                best_routed_program = routed

    return best_placement, cancel_redundant_swaps(best_routed_program)