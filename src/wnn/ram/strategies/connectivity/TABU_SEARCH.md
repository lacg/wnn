# Tabu Search (TS) for Connectivity Optimization

Based on Garcia (2003) thesis on global optimization methods for WNN connectivity patterns.

## Overview

Tabu Search is a local search metaheuristic that explores neighborhoods while using memory (tabu list) to avoid cycling. It achieved the **best results** in Garcia's thesis: 17.27% error reduction with only 5 iterations.

Key characteristic: **Always moves to the best neighbor**, even if it's worse than current solution. This allows escaping local minima.

## Connectivity Map Encoding

Same as GA - see [GENETIC_ALGORITHM.md](./GENETIC_ALGORITHM.md) for details.

```
connections: Tensor[num_neurons, n_bits_per_neuron]
# or flattened 1D for tiered architectures
```

## Algorithm Steps

### 1. Initialization

```python
current = initial_connections.clone()
current_error = evaluate(current)
best = current.clone()
best_error = current_error
tabu_list = deque(maxlen=tabu_size)  # Stores recent moves
```

### 2. Neighbor Generation

Generate N neighbors by mutating current solution:

```python
def generate_neighbor(current, mutation_rate):
    neighbor = current.clone()
    for neuron_idx in range(num_neurons):
        for conn_idx in range(n_bits_per_neuron):
            if random.random() < mutation_rate:
                old_value = neighbor[neuron_idx, conn_idx]
                new_value = random.randint(0, total_input_bits - 1)
                neighbor[neuron_idx, conn_idx] = new_value
                move = (neuron_idx, old_value, new_value)
    return neighbor, move
```

### 3. Tabu Filtering

A move is **tabu** if it reverses a recent move:

```python
def is_tabu(move, tabu_list):
    # move = (neuron_idx, old_value, new_value)
    for tabu_move in tabu_list:
        # Check if this move reverses a tabu move
        if (tabu_move[0] == move[0] and      # same neuron
            tabu_move[2] == move[1]):         # new→old reversal
            return True
    return False
```

**Example**:
```
tabu_list = [(2, 5, 11)]  # Changed neuron 2's connection from 5 to 11

# This move is TABU (reverses the change):
move = (2, 11, 5)  # Changing neuron 2's connection from 11 back to 5

# This move is NOT tabu:
move = (2, 11, 8)  # Changing neuron 2's connection from 11 to 8 (different target)
```

### 4. Best Neighbor Selection

```python
neighbors = []
for _ in range(neighbors_per_iter):
    neighbor, move = generate_neighbor(current, mutation_rate)
    if not is_tabu(move, tabu_list):
        error = evaluate(neighbor)
        neighbors.append((neighbor, error, move))

# Sort by error (ascending) and pick best
neighbors.sort(key=lambda x: x[1])
best_neighbor, best_error, best_move = neighbors[0]
```

### 5. Move Acceptance

**Key TS characteristic**: Always accept the best non-tabu neighbor, even if worse:

```python
current = best_neighbor
current_error = best_neighbor_error
tabu_list.append(best_move)  # Add move to tabu list

if current_error < global_best_error:
    global_best = current.clone()
    global_best_error = current_error
```

This differs from Simulated Annealing which probabilistically accepts worse solutions.

### 6. Iteration Loop

```
For iteration in 1..max_iterations:
    1. Generate N neighbor candidates with mutations
    2. Filter out tabu moves
    3. Batch-evaluate all non-tabu neighbors
    4. Select best neighbor (even if worse than current)
    5. Move to best neighbor, add move to tabu list
    6. Update global best if improved
    7. Check early stopping (every 5 iterations)
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `iterations` | 5 | Number of iterations (Garcia found 5 sufficient) |
| `neighbors_per_iter` | 30 | Neighbors generated per iteration |
| `tabu_size` | 5 | Size of tabu list (recent moves to avoid) |
| `mutation_rate` | 0.001 | Per-connection mutation probability |
| `early_stop_patience` | 1 | Checks without improvement before stop |
| `early_stop_threshold_pct` | 0.02 | Minimum PPL improvement % to reset patience |

## Why Tabu Search Works Well

1. **Parallel Exploration**: Tests 30 neighbors per iteration vs SA's 1
2. **Deterministic Best Selection**: Always picks best neighbor (no randomness in selection)
3. **Memory Prevents Cycling**: Tabu list avoids revisiting recent solutions
4. **Escapes Local Minima**: Accepts worse solutions to explore new regions

## Early Stopping

Same as GA - every 5 iterations, check PPL improvement:

```python
if ppl_improvement < threshold_pct:
    patience_counter += 1
    if patience_counter > patience:
        STOP
```

## Overfitting Control

TS supports diversity mode when validation diverges from training:

| Mode | Neighbors | Mutation | Tabu Size | Trigger |
|------|-----------|----------|-----------|---------|
| Normal | 30 | 0.001 | 5 | val/train ratio stable |
| Mild Diversity | 60 (2x) | 0.002 (2x) | 10 (2x) | ratio increase > 0% |
| Severe Diversity | 90 (3x) | 0.003 (3x) | 15 (3x) | ratio increase > 1% |

## Complexity

- **Time**: O(iterations × neighbors × evaluation_cost)
- **Space**: O(neighbors × connectivity_size + tabu_size)

With batch evaluation, all 30 neighbors are evaluated in parallel per iteration.

## Comparison with GA

| Aspect | GA | Tabu Search |
|--------|----|----|
| Search Type | Population-based | Single-solution |
| Iterations | 50 generations | 5 iterations |
| Parallelism | 30 individuals | 30 neighbors |
| Selection | Tournament (probabilistic) | Best (deterministic) |
| Memory | Elitism (best individuals) | Tabu list (recent moves) |
| Exploration | Crossover + mutation | Mutation only |

**When to use which**:
- GA: Larger search spaces, more exploration needed
- TS: Faster convergence, refinement of good solutions

**Best practice**: Use GA first for broad exploration, then TS for local refinement. This is why `--strategy GA,TS` is the default.

## Usage

```python
from wnn.ram.strategies.connectivity import (
    TabuSearchStrategy,
    TabuSearchConfig,
)

config = TabuSearchConfig(
    iterations=1000,
    neighbors_per_iter=30,
    tabu_size=5,
    mutation_rate=0.001,
    early_stop_patience=5,
)

ts = TabuSearchStrategy(config=config, verbose=True)

result = ts.optimize(
    connections=ga_optimized_connections,  # Start from GA result
    evaluate_fn=lambda conn: compute_perplexity(conn),
    total_input_bits=context_size * bits_per_token,
    num_neurons=15,
    n_bits_per_neuron=20,
    batch_evaluate_fn=rust_batch_eval,
)

print(f"Improved by {result.improvement_percent:.2f}%")
```

## Combined GA+TS Pipeline

```python
# Phase 1: GA for broad exploration
ga_result = ga.optimize(connections, ...)

# Phase 2: TS for local refinement
ts_result = ts.optimize(ga_result.optimized_connections, ...)

# Total improvement
total_improvement = (initial - ts_result.final_error) / initial * 100
```

## References

- Garcia, L.M. (2003). "Global Optimization Methods for Choosing Connectivity Patterns of Weightless Neural Networks"
- Glover, F. (1989). "Tabu Search - Part I". ORSA Journal on Computing.
