from networkx.algorithms import non_randomness
from dataclasses import dataclass
from typing import List, Tuple, Dict
import networkx as nx
import random

@dataclass(frozen=True)
class SabreParams:
    W: float
    ext_size: int
    decay_delta: float
    noise_scale: float = 0


def estimate_score(routed_program: List[Tuple]) -> float:
    """Computes fast score estimate (SWAPs + 0.5 * depth)."""
    swaps = sum(1 for op in routed_program if op[0] == 'SWAP')
    qubit_free_layer: Dict[int, int] = {}
    for op in routed_program:
        if op[0] in ('2Q', 'SWAP'):
            p1, p2 = op[1], op[2]
            start_layer = max(qubit_free_layer.get(p1, 0), qubit_free_layer.get(p2, 0))
            qubit_free_layer[p1] = start_layer + 1
            qubit_free_layer[p2] = start_layer + 1
    depth = max(qubit_free_layer.values()) if qubit_free_layer else 0
    return swaps + 0.5 * depth


def single_sabre_pass(
    program: List[Tuple],
    hardware_graph: nx.Graph,
    dist_matrix: Dict[int, Dict[int, int]],
    initial_placement: Dict[int, int],
    params: SabreParams
) -> Tuple[List[Tuple], Dict[int, int]]:
    """Runs a single SABRE routing pass with tie-breaking for equal cost candidate swaps."""
    hw_nodes = sorted(list(hardware_graph.nodes))
    num_nodes = len(hw_nodes)

    # Precompute physical node degrees for fast tie-breaker evaluation
    node_degrees = dict(hardware_graph.degree())

    # 1-to-1 Bijection tracking
    inactive_logical = [l for l in range(num_nodes) if l not in initial_placement]
    unassigned_physical = [p for p in hw_nodes if p not in initial_placement.values()]

    logical_to_phys = dict(initial_placement)
    for l, p in zip(inactive_logical, unassigned_physical):
        logical_to_phys[l] = p

    phys_to_logical = {p: l for l, p in logical_to_phys.items()}
    decay = {p: 1.0 for p in hw_nodes}

    routed_program = []
    num_gates = len(program)
    current_idx = 0

    while current_idx < num_gates:
        gate = program[current_idx]
        kind = gate[0]

        if kind == '1Q':
            l = gate[1]
            routed_program.append(('1Q', logical_to_phys[l]))
            current_idx += 1
            # decay = {p: 1.0 for p in hw_nodes}
            continue

        l1, l2 = gate[1], gate[2]
        p1, p2 = logical_to_phys[l1], logical_to_phys[l2]

        if hardware_graph.has_edge(p1, p2):
            routed_program.append(('2Q', p1, p2))
            current_idx += 1
            decay = {p: 1.0 for p in hw_nodes}
            continue

        extended_layer = program[current_idx + 1 : current_idx + 1 + params.ext_size]

        candidate_swaps = set()
        for p in (p1, p2):
            for neighbor in hardware_graph.neighbors(p):
                candidate_swaps.add(tuple(sorted((p, neighbor))))

        best_cost = float('inf')
        best_swap = None
        best_tie_score = (-float('inf'), -float('inf'))

        for cand_p1, cand_p2 in candidate_swaps:
            l_a, l_b = phys_to_logical[cand_p1], phys_to_logical[cand_p2]
            logical_to_phys[l_a], logical_to_phys[l_b] = cand_p2, cand_p1

            curr_q1, curr_q2 = logical_to_phys[l1], logical_to_phys[l2]
            f_cost = dist_matrix[curr_q1][curr_q2]

            e_cost = 0.0
            e_count = 0
            for e_gate in extended_layer:
                if e_gate[0] == '2Q':
                    q1, q2 = logical_to_phys[e_gate[1]], logical_to_phys[e_gate[2]]
                    e_cost += dist_matrix[q1][q2]
                    e_count += 1

            if e_count > 0:
                e_cost /= e_count

            cost = max(decay[cand_p1], decay[cand_p2]) * (f_cost + params.W * e_cost)

            if params.noise_scale > 0.0:
                cost *= (1.0 + random.uniform(-params.noise_scale, params.noise_scale))

            # Tie-breaker metric: (Higher degree sum, Lower decay sum)
            degree_sum = node_degrees[cand_p1] + node_degrees[cand_p2]
            decay_sum = decay[cand_p1] + decay[cand_p2]
            tie_score = (degree_sum, -decay_sum)

            # Revert SWAP
            logical_to_phys[l_a], logical_to_phys[l_b] = cand_p1, cand_p2

            # Evaluate best cost with tie-breaking
            if cost < best_cost - 1e-9:
                best_cost = cost
                best_swap = (cand_p1, cand_p2)
                best_tie_score = tie_score
            elif abs(cost - best_cost) <= 1e-9:
                if tie_score > best_tie_score:
                    best_cost = cost
                    best_swap = (cand_p1, cand_p2)
                    best_tie_score = tie_score

        if best_swap:
            sw_p1, sw_p2 = best_swap
            routed_program.append(('SWAP', sw_p1, sw_p2))

            l_a, l_b = phys_to_logical[sw_p1], phys_to_logical[sw_p2]
            logical_to_phys[l_a], logical_to_phys[l_b] = sw_p2, sw_p1
            phys_to_logical[sw_p1], phys_to_logical[sw_p2] = l_b, l_a

            decay[sw_p1] += params.decay_delta
            decay[sw_p2] += params.decay_delta

    return routed_program, logical_to_phys


def sabre_layout(
    program: List[Tuple],
    hardware_graph: nx.Graph,
    dist_matrix: Dict[int, Dict[int, int]],
    initial_placement: Dict[int, int],
    fwd_params: SabreParams,
    bwd_params: SabreParams,
    max_iterations: int = 5
) -> Tuple[Dict[int, int], List[Tuple], float]:
    """
    Executes bidirectional SABRE layout with best-so-far tracking across iterations.
    """
    reversed_program = program[::-1]
    curr_initial = dict(initial_placement)

    best_score = float('inf')
    best_initial_placement = dict(initial_placement)
    best_routed_program = []

    for _ in range(max_iterations):
        # 1. Forward Pass
        fwd_routed, fwd_final_mapping = single_sabre_pass(
            program, hardware_graph, dist_matrix, curr_initial, fwd_params
        )
        score = estimate_score(fwd_routed)

        # Track global best across all iterations
        if score < best_score:
            best_score = score
            best_initial_placement = dict(curr_initial)
            best_routed_program = fwd_routed

        # 2. Backward Pass (refines initial placement for next iteration)
        _, bwd_final_mapping = single_sabre_pass(
            reversed_program, hardware_graph, dist_matrix, fwd_final_mapping, bwd_params
        )

        # 3. Early convergence check
        if curr_initial == bwd_final_mapping:
            break
        curr_initial = bwd_final_mapping

    return best_initial_placement, best_routed_program, best_score