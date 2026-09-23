# High-Performance Quantum Circuit Router: Bidirectional SABRE & QAP Placement

An advanced, competition-grade quantum circuit routing engine that maps logical quantum circuits onto constrained physical QPUs with minimal SWAP overhead and circuit depth.

---

## Technical Highlights & Hackathon Innovations

Most standard quantum compilers rely on basic greedy heuristic routing or static initial placement. This submission introduces a multi-stage, hardware-aware optimization pipeline combining **graph-theoretic initial placement**, **bidirectional SABRE layout refinement**, and **theoretical lower-bound early stopping**.

```text
Logical Circuit ──► Multi-Strategy Placement ──► Asymmetric Bidirectional SABRE ──► Redundant SWAP Cancellation ──► Optimized Physical Circuit
                             │                                 │
                             ▼                                 ▼
                     (QAP-SA, Spectral,                (Parallel Search Grid &
                      MWST, Subgrids)                 Lower-Bound Early Stopping)
```

### 1. Multi-Strategy Initial Placement Suite (`placement.py`)
Finding an optimal initial physical mapping is critical for minimizing SWAP overhead. Our placement engine evaluates 6 distinct structural placement strategies:
* **DAG Layer-Decayed QAP with Simulated Annealing & 2-Opt Polish**: Formulates initial placement as a Quadratic Assignment Problem (QAP). Gate interaction weights decay exponentially by DAG topological layer ($w = \gamma^{\text{layer}}$), prioritizing early-stage interactions. Placements are explored via Simulated Annealing and polished with steepest-descent 2-Opt local search.
* **Spectral Embedding Matching**: Computes 2D spectral graph layouts of both logical interaction topology and hardware coupling graphs, solving a linear sum assignment problem to align logical qubits to physical nodes.
* **Compact Subgrid Clustering**: Constrains logical circuits to contiguous, highly connected physical subgraphs on large QPUs using central seed nodes and spectral alignment.
* **Maximum Weight Spanning Tree (MWST)**: Extracts maximum interaction trees and maps outwards from high-degree physical anchors.
* **BFS Hub & DFS Linear Path Placements**: Structural traversals for star-like and chain-like gate dependency profiles.

### 2. Enhanced Bidirectional SABRE Core (`sabre_core.py`)
Our routing kernel extends the standard SABRE algorithm with several crucial stability and optimization mechanisms:
* **Asymmetric Bidirectional Refinement**: Runs iterative forward and backward routing passes. The forward pass evaluates routing cost, while the backward pass refines the initial layout mapping for subsequent iterations.
* **Structural Tie-Breaking Metric**: When multiple candidate SWAPs yield identical lookahead costs, tie-breaking selects physical edges based on maximum node degree sum ($\sum \text{deg}(p)$) and minimum node decay sum to favor highly connected hardware regions.
* **Anti-Oscillation Decay Memory**: Maintains decay factors on physical nodes involved in SWAPs to prevent infinite thrashing loops, preserving decay memory across single-qubit operations.
* **Stochastic Cost Perturbation**: Applies controlled noise ($\pm 5\%$) to lookahead cost calculations, enabling multi-seed exploration to break out of greedy local minima.

### 3. Parallel Search Engine & Theoretical Lower-Bound Early Pruning (`solver.py`)

* **Theoretical Lower-Bound Early Exit**: Before executing search iterations, `solver.py` computes the absolute mathematical lower bound for circuit score:

$$\text{Score}_{\text{min}} = 0.5 \times \max\left(\text{CriticalPath}_{\text{DAG}}, \left\lceil \frac{N_{2Q}}{\lfloor N_{\text{active}}/2 \rfloor} \right\rceil\right)$$

  If any worker process achieves this theoretical minimum score ($0 \text{ SWAPs}$ and optimal depth), the parallel executor terminates immediately, saving compute time.
* **Parallel Cross-Product Search Grid**: Executes multi-core grid searches over parameter tuples $(W, E, \text{Decay}, \text{Seeds})$ and candidate initial placements using `concurrent.futures.ProcessPoolExecutor`.
* **Redundant SWAP Cancellation**: A post-processing cleanup pass identifies and purges adjacent inverse SWAP operations (`SWAP u v` followed by `SWAP u v`) that preserve dependency ordering.

---

## Core Module Architecture

```text
.
├── solver.py          # Primary entry point, parallel search grid, theoretical lower-bound calculator
├── placement.py       # Multi-strategy initial placement generators (QAP-SA, Spectral, MWST, Subgrids)
├── sabre_core.py      # Core routing engine (single pass, bidirectional SABRE, tie-breaking, score estimation)
└── starter.ipynb      # Main evaluation and benchmarking notebook
```

## Module Descriptions
### ```solver.py```

- ```solver(program, hardware_graph, [search_space, parallel])```: Main routing API function. COmputes distance matrices, gather candidate placements, build parallel task grids, montiors lower bounds, and returne sthe optimal layout and routed circuit.
- ```compute_circuit_lower_bound()```: Calculates pure DAG critical path and hardware capacity limits.
- ```cancel reduncant_swaps()```: cleans adjacent back-to-back SWAP gates.

### ```placement.py```
- ```generate_all_hub_placements()```: Fix a qubit to be a hub and create placement by shortest distant.
- ```generate_linear_path_placement()```: Form a mapping that represent a chain on physical qubit.
- ```generate_qap_sa_placements()```: Implements layer-decayed QAP with Simulated Annealing and 2-Opt local optimization.
- ```generate_spectral_placements()```: Aligns logical and physical spectral graph embeddings via the Hungarian algorithm (```linear_sum_assignment```)
- ```generate_compact_subgrid_placements()```: Retricts routing to compact high-density hardware clusters.
- ```collect_all_placements()```: Runs all strategy generators and deduplicates initial mapping candidates.

### ```sabre_core.py```
- ```sabre_layout()```: Coordinates forward/backward passes and tracks the global minimum score across iterations.
-```single_sabre_pass()```: Executes a single routing pass with lookahead, decay memory, stochastic noice, and degree-based tie-breaking.
-```estimate_score()```: Fast evaluator for circuit quality using $Score=SWAPs+0.5\times Depth$

## Performance Benchmark
| Benchmark | Logical Qubits | 2Q Gates | SWAPs | Depth | Final Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `ghz_star` | 8 | 7 | 2 | 9 | **6.5** |
| `chain_trotter` | 10 | 9 | 0 | 9 | **4.5** |
| `ladder_trotter` | 12 | 16 | 3 | 7 | **6.5** |
| `qaoa_random` | 10 | 18 | 6 | 11 | **11.5** |
| `dense_random` | 14 | 40 | 27 | 21 | **37.5** |
| `vqe_layers` | 16 | 45 | 0 | 6 | **3.0** |
| **Total Standard Score** | **—** | **—** | **38** | **63** | **69.5** |

## Detailed Algorithm Mechanics

### 1. SABRE Lookahead Cost Function & Tie-Breaking

The core SABRE routing engine determines which physical `SWAP` to insert by evaluating a multi-factor cost heuristic for every candidate swap edge $(p_1, p_2)$ adjacent to unresolved $2\text{Q}$ gate physical locations.

#### The Heuristic Cost Formula

For a candidate `SWAP` between physical nodes $p_1$ and $p_2$, the cost metric is defined as:

$$\text{Cost}(p_1, p_2) = \max\Big(\text{decay}[p_1], \text{decay}[p_2]\Big) \times \left( D(q_1, q_2) + W \times \bar{D}_{E} \right) \times (1 + \epsilon)$$

Where:

1. **Immediate Front Distance (\$D(q_1, q_2)\$)**:
   The shortest-path distance on the hardware graph between the physical locations $q_1 = \pi(l_1)$ and $q_2 = \pi(l_2)$ of the current unresolved $2\text{Q}$ gate $(l_1, l_2)$, assuming the candidate swap is executed.

2. **Extended Lookahead Window ($\bar{D}_E$)**: The average shortest-path hardware distance of the next $E$ upcoming 2Q gates in the circuit's topological DAG dependency front ($E =$ ext\_size):

$$\bar{D}_E = \frac{1}{\vert{}E_{2\text{Q}}\vert{}} \sum_{g=(u,v) \in E_{2\text{Q}}} D\big(\pi(u), \pi(v)\big)$$

   The parameter $W \in [0.0, 1.0]$ controls the weight of future dependencies versus immediate progress.

3. **Anti-Oscillation Decay Memory (`decay[p]`)**: To prevent infinite thrashing loops where two physical qubits repeatedly swap back and forth without executing gates, every physical node $p$ tracks a decay factor initialized to $1.0$.

   * When a swap is executed on $(p_1, p_2)$, their decay values increase:

$$\text{decay}[p_1] \leftarrow \text{decay}[p_1] + \delta, \quad \text{decay}[p_2] \leftarrow \text{decay}[p_2] + \delta$$

   * Decay factors reset back to $1.0$ **only** when a 2Q gate is successfully executed.
   * *Note on 1Q Gates*: Single-qubit gates preserve decay memory without resetting it, ensuring stability in interleaved circuits.

4. **Stochastic Cost Perturbation ($\epsilon$)**: A minor random cost variation $\epsilon \sim U$(-noise\_scale, noise\_scale) enables multi-seed stochastic exploration.

#### Structural Tie-Breaking Metric

When two candidate swaps produce identical heuristic costs ($\vert{}\text{Cost}_A - \text{Cost}_B\vert{} \le 10^{-9}$), SABRE evaluates a secondary structural tie-breaker:

$$\text{TieScore}(p_1, p_2) = \Big(\text{deg}(p_1) + \text{deg}(p_2),\; -\big(\text{decay}[p_1] + \text{decay}[p_2]\big)\Big)$$

- **Primary Criterion**: Prefers physical edges with a higher combined hardware degree sum $\text{deg}(p_1) + \text{deg}(p_2)$ to push active qubits toward highly connected hardware central hubs.
- **Secondary Criterion**: Prefers edges with lower cumulative decay factors.

---

### 2. Advanced Initial Placement Initialization Strategies

A poor initial logical-to-physical mapping forces SABRE to insert excess `SWAP` gates in early circuit layers. To address this, `placement.py` constructs diverse candidate mappings using graph theory and continuous optimization.

---

#### A. DAG Layer-Decayed QAP with Simulated Annealing & 2-Opt Polish

Initial placement is formulated as a **Quadratic Assignment Problem (QAP)**, where logical qubits $l \in V_{\text{logical}}$ must be mapped to physical nodes $p \in V_{\text{hardware}}$ to minimize total weighted distance.

##### Layer-Decayed Interaction Weights
Instead of treating all $2\text{Q}$ gates equally, gate interaction weights decay exponentially based on their topological depth ($\text{layer} \in \mathbb{Z}_{\ge 0}$) in the circuit's Directed Acyclic Graph (DAG):

$$w(u, v) = \sum_{g=(u,v)} \gamma^{\text{layer}(g)}, \quad \text{where } \gamma = 0.8$$

Gates occurring at layer $0$ carry full weight ($1.0$), while distant future gates carry exponentially smaller weight ($\gamma^k$), prioritizing physical proximity for gates executed early in the circuit.

##### Two-Stage Optimization Solver
1. **Stage 1: Simulated Annealing (SA) Exploration**
   Starting from a random initial placement $\pi$, the algorithm explores candidate logical-logical 2-swaps $\pi'$.
   - The cost function is the weighted QAP distance:
     $$\text{Cost}_{\text{QAP}}(\pi) = \sum_{(u,v) \in E_{\text{logical}}} w(u,v) \cdot D\big(\pi(u), \pi(v)\big)$$
   - Swaps causing cost increase $\Delta > 0$ are accepted with Boltzmann probability $P(\text{accept}) = \exp(-\Delta / T)$, cooling from $T_0 = 15.0$ at rate $\alpha = 0.94$.

2. **Stage 2: Steepest-Descent 2-Opt Polish**
   After annealing, a deterministic 2-Opt local search refines the solution by exhaustively testing all pairwise swaps ($\pi(l_1) \leftrightarrow \pi(l_2)$) and logical-to-empty physical moves ($\pi(l) \to p_{\text{empty}}$) until reaching a strict local optimum.

---

#### B. Spectral Embedding Matching

Spectral placement converts discrete graph connectivity into continuous geometric embeddings using the **Graph Laplacian**.
```
Logical Interaction Graph ──► Spectral Laplacian Eigenvectors ──► 2D Continuous Embedding ─┐
                                                                                           ├─► Hungarian Algorithm (LAP) ──► Discrete Initial Layout
Hardware Coupling Graph   ──► Spectral Laplacian Eigenvectors ──► 2D Continuous Embedding ─┘
```
1. **Spectral Layout Computation**:
   Construct the unweighted Laplacian matrix $L = D - A$ for both the logical interaction graph and the hardware coupling graph.
   The 2D spatial positions $\mathbf{x}_l \in \mathbb{R}^2$ (for logical qubit $l$) and $\mathbf{y}_p \in \mathbb{R}^2$ (for physical node $p$) are derived from the eigenvectors corresponding to the smallest non-zero eigenvalues of $L$ (the Fiedler vectors).

2. **Euclidean Distance Cost Matrix**:
   A pairwise Euclidean distance matrix $C \in \mathbb{R}^{\vert{}V_L\vert{} \times \vert{}V_H\vert{}}$ is constructed:
   $$C_{i, j} = \Vert{}\mathbf{x}_i - \mathbf{y}_j\Vert{}_2$$

3. **Linear Assignment Solving**:
   The optimal discrete mapping $\pi$ minimizing continuous geometric distortion is solved exactly in $O(N^3)$ time using the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`):
   $$\min_{\pi} \sum_{i=1}^{k} C_{i, \pi(i)}$$

---

#### C. Compact Subgrid Clustering

When running smaller circuits (e.g., $8$–$12$ qubits) on larger QPUs (e.g., $20$+ qubits), placing qubits sparsely across the entire processor causes unnecessary routing overhead across long hardware paths.

1. **Subgrid Extraction**:
   Selects the top high-degree physical nodes as central "seeds." For each seed $p_{\text{seed}}$, the algorithm selects the $k = \vert{}V_{\text{active}}\vert{}$ closest physical nodes based on hardware distance $D(p_{\text{seed}}, p)$ and node degree, extracting a compact, contiguous physical subgraph $G_{\text{sub}}$.

2. **Localized Matching**:
   Applies localized spectral layout alignment between the logical interaction graph and the extracted sub-hardware graph $G_{\text{sub}}$.

3. **Benefit**:
   Forces logical qubits to remain localized within tightly coupled physical regions, reducing the maximum path length required during routing.