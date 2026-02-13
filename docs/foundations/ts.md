# Tabu Search (TS)

## Overview

Tabu Search is a memory-guided local search metaheuristic. It explores the neighborhood of the current best solution, accepting the best neighbor even if it worsens fitness (to escape local optima), while maintaining a **tabu list** of recently visited moves to prevent cycling.

In this project, TS is used for **focused refinement** after GA exploration — taking a good solution and making it better through systematic local changes.

## Algorithm

```
TABU SEARCH
────────────
Input:  initial_genome, iterations, neighbors_per_iter, tabu_size
Output: best genome found

1. Set current = initial_genome, best = initial_genome
2. Initialize empty tabu list (bounded deque)
3. For each iteration:
   a. GENERATE neighbors_per_iter neighbors from current genome
   b. EVALUATE all neighbors (batch evaluation via Rust+Metal)
   c. FILTER: remove neighbors whose moves are in tabu list
      (unless they pass aspiration criteria: better than global best)
   d. SELECT best non-tabu neighbor as new current
   e. ADD the selected move to tabu list
   f. UPDATE global best if current is better
   g. Check early stopping (patience-based)
4. Return best genome found
```

## Key Concepts

### Neighborhood
A "neighbor" is a genome that differs from the current solution by a single small change (a "move"):
- Swap one neuron's connection from bit A to bit B
- Add/remove one neuron from a cluster
- Change one cluster's bits_per_neuron by ±1

### Tabu List
A bounded FIFO queue of recent moves. Moves in the tabu list are forbidden for `tabu_size` iterations, preventing the search from cycling back to recently visited solutions.

### Aspiration Criteria
A tabu move is allowed if it produces a solution better than the current global best. This prevents the tabu list from blocking genuinely good moves.

### Progressive Threshold
Accuracy threshold increases linearly over iterations, progressively filtering out low-accuracy solutions. This focuses the search on higher-quality regions as it progresses.

## Cooperative Multi-Start Variant

Standard TS generates all neighbors from a single best genome, which can stagnate at local optima. The **cooperative multi-start** variant maintains a reference set of diverse high-quality solutions.

### How It Works

When `diversity_sources_pct > 0`:

1. **Reference set**: Top N% of the neighbor cache (ranked by fitness) form an equal reference set
2. **Equal generation**: Each source generates an equal share of neighbors, capped at `neighbors_per_iter`
3. **No fitness weighting**: All sources in the reference set are treated equally

```
Example: population=50, diversity_sources_pct=0.2

Reference set: top 20% of 50 = 10 genomes
Neighbors per source: 50 / 10 = 5
Total neighbors: 10 x 5 = 50 (same as neighbors_per_iter)
```

This is based on the **Cooperative Multi-Start Tabu Search** framework (Crainic, Toulouse, & Gendreau, 1997) and draws from Glover's **Scatter Search** (1998) concept of maintaining diverse reference sets.

### Configuration

```python
TSConfig(
    diversity_sources_pct=0.0,  # 0.0 = classic single-source (default)
                                 # 0.2 = top 20% as reference set
)
```

### When to Use

- `0.0` (default): When the search space is smooth and single-path refinement works well
- `0.1-0.3`: When TS stagnates at local optima and broader exploration is needed
- Higher values diminish returns as sources become less elite

## Configuration

```python
TSConfig(
    iterations=100,           # Maximum iterations
    neighbors_per_iter=50,    # Neighbors generated per iteration
    tabu_size=10,             # Size of tabu list (moves remembered)
    total_neighbors_size=50,  # Cache size for top genomes
    mutation_rate=0.1,        # Magnitude of mutations
    patience=10,              # Early stop patience
    check_interval=10,        # Check improvement every N iterations
    min_improvement_pct=0.5,  # Minimum improvement to reset patience
    diversity_sources_pct=0.0,# Cooperative multi-start (0 = disabled)
)
```

## In This Project

TS serves as the **refinement phase** following each GA exploration:
- Phase 3: TS refinement of neuron counts (from Phase 2 GA best)
- Phase 5: TS refinement of bit counts (from Phase 4 GA best)
- Phase 7: TS refinement of connections (from Phase 6 GA best)

TS is seeded with GA's final population, providing a warm start. The `total_neighbors_size` cache preserves the best genomes found across all iterations for seeding the next phase.

## References

- Glover, F. (1986). "Future Paths for Integer Programming and Links to Artificial Intelligence." *Computers & Operations Research*, 13(5), 533-549.
- Glover, F., & Laguna, M. (1997). *Tabu Search.* Kluwer Academic Publishers.
- Crainic, T. G., Toulouse, M., & Gendreau, M. (1997). "Toward a taxonomy of parallel tabu search heuristics." *INFORMS Journal on Computing*, 9(1), 61-72.
- Glover, F. (1998). "A Template for Scatter Search and Path Relinking." *LNCS* 1363, 1-51.
- Gendreau, M., & Potvin, J.-Y. (2005). "Tabu Search." In *Search Methodologies*. Springer.
