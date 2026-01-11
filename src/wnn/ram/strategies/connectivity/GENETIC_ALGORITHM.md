# Genetic Algorithm (GA) for Connectivity Optimization

Based on Garcia (2003) thesis on global optimization methods for WNN connectivity patterns.

## Overview

GA is a population-based metaheuristic that evolves connectivity patterns through selection, crossover, and mutation. It's good for exploring large search spaces and achieved 89% memory reduction while maintaining accuracy.

## Connectivity Map Encoding

### Representation

```
connections: Tensor[num_neurons, n_bits_per_neuron]
```

Each neuron has `n_bits_per_neuron` connections. Each connection value is an index (0 to `total_input_bits - 1`) indicating which input bit that connection observes.

**Example** (3 neurons, 4 bits/neuron, 16 input bits):
```
connections = [
    [2, 5, 11, 14],   # Neuron 0 observes input bits 2, 5, 11, 14
    [0, 3, 8, 12],    # Neuron 1 observes input bits 0, 3, 8, 12
    [1, 6, 9, 15],    # Neuron 2 observes input bits 1, 6, 9, 15
]
```

### For Tiered Architectures

When neurons have different `n_bits_per_neuron` (variable fan-in), connections are stored as a **flattened 1D tensor**:

```
connections_flat: Tensor[total_connections]

# Example: 2 neurons with 4 bits, 1 neuron with 8 bits
# Total = 2*4 + 1*8 = 16 connections
connections_flat = [2, 5, 11, 14, 0, 3, 8, 12, 1, 4, 6, 7, 9, 10, 13, 15]
#                   |-- neuron 0 --|  |-- neuron 1 --|  |------ neuron 2 ------|
```

## Algorithm Steps

### 1. Initialization

```
Population = [individual_0, individual_1, ..., individual_N]
```

- `individual_0` = original connectivity (unchanged)
- `individual_1..N` = mutations of original (10x mutation rate)

Each individual is a `(connectivity, fitness)` tuple. Fitness starts as `None` (unevaluated).

### 2. Evaluation (with Caching)

```python
for each individual with fitness=None:
    fitness = evaluate_fn(connectivity)  # e.g., cross-entropy loss
```

**Key optimization**: Elite individuals retain cached fitness across generations - no re-evaluation needed.

### 3. Selection (Tournament)

```python
def tournament_select(population, tournament_size=3):
    candidates = random.sample(population, tournament_size)
    return min(candidates, key=lambda x: x.fitness)
```

Pick 3 random individuals, return the one with lowest error.

### 4. Crossover (Single-Point, Neuron-Level)

```python
def crossover(parent1, parent2, num_neurons, neuron_offsets=None):
    if random.random() > crossover_rate:
        return parent1.clone()

    crossover_point = random.randint(1, num_neurons - 1)
    child = parent1.clone()

    if parent1.dim() == 1 and neuron_offsets:
        # Tiered: use neuron boundary from offsets
        conn_boundary = neuron_offsets[crossover_point]
        child[conn_boundary:] = parent2[conn_boundary:]
    else:
        # Uniform 2D: crossover at row
        child[crossover_point:] = parent2[crossover_point:]
    return child
```

**Example (Uniform 2D)**:
```
Parent 1: [[2,5,11,14], [0,3,8,12], [1,6,9,15]]
Parent 2: [[4,7,10,13], [2,5,8,11], [0,3,6,9]]
                        ^ crossover point = 1

Child:    [[2,5,11,14], [2,5,8,11], [0,3,6,9]]
           from P1       from P2
```

**Example (Tiered 1D with variable bits/neuron)**:
```
Architecture: Tier 0 = 500 clusters × 3 neurons × 20 bits = 1500 neurons
              Tier 1 = rest clusters × 3 neurons × 8 bits

neuron_offsets = [0, 20, 40, ..., 29980, 30000, 30008, 30016, ...]
                 |---- tier0 (20 bits/neuron) ----||-- tier1 (8 bits) --|
                 neuron 0,1,2...                   neuron 1500,1501...

Parent 1 (flat): [c0-c19 | c20-c39 | ... | c30000-c30007 | ...]
                  N0        N1             N1500

crossover_point = 1500 → conn_boundary = neuron_offsets[1500] = 30000

Child: [c0...c29999 | C30000...C_end]
       from P1         from P2
```

This ensures crossover respects neuron boundaries even when neurons have different bits.

### 5. Mutation (Per-Connection)

```python
def mutate(connectivity, mutation_rate=0.01):
    for neuron_idx in range(num_neurons):
        for conn_idx in range(n_bits_per_neuron):
            if random.random() < mutation_rate:
                connectivity[neuron_idx, conn_idx] = random.randint(0, total_input_bits - 1)
    return connectivity
```

Each connection has 1% chance of being reassigned to a random input bit.

### 6. Elitism

The top 2 individuals (by fitness) are copied unchanged to the next generation, preserving their cached fitness.

### 7. Generation Loop

```
For generation in 1..max_generations:
    1. Sort population by fitness
    2. Copy elite individuals to new population (with cached fitness)
    3. Fill remaining slots via:
       - Tournament selection of 2 parents
       - Crossover to create child
       - Mutation of child
       - Child has fitness=None (needs evaluation)
    4. Evaluate only new individuals (fitness=None)
    5. Update global best if improved
    6. Check early stopping (every 5 generations)
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `population_size` | 30 | Number of individuals per generation |
| `generations` | 50 | Maximum generations to run |
| `mutation_rate` | 0.01 | Per-connection mutation probability |
| `crossover_rate` | 0.7 | Probability of crossover vs cloning |
| `elitism` | 2 | Number of best individuals preserved |
| `early_stop_patience` | 1 | Checks without improvement before stop |
| `early_stop_threshold_pct` | 0.02 | Minimum PPL improvement % to reset patience |

## Early Stopping

Every 5 generations, check if PPL improved by at least `threshold_pct`:

```python
ppl_improvement = (exp(prev_best) - exp(current_best)) / exp(prev_best) * 100

if ppl_improvement >= threshold_pct:
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter > patience:
        STOP
```

## Overfitting Control

GA supports diversity mode when validation loss increases relative to training:

| Mode | Population | Mutation Rate | Trigger |
|------|------------|---------------|---------|
| Normal | 30 | 0.01 | val/train ratio stable |
| Mild Diversity | 45 (1.5x) | 0.015 (1.5x) | ratio increase > 0% |
| Severe Diversity | 60 (2x) | 0.02 (2x) | ratio increase > 1% |

## Complexity

- **Time**: O(generations × population × evaluation_cost)
- **Space**: O(population × connectivity_size)

With batch evaluation (Rust accelerator), all individuals are evaluated in parallel, reducing wall-clock time significantly.

## Usage

```python
from wnn.ram.strategies.connectivity import (
    GeneticAlgorithmStrategy,
    GeneticAlgorithmConfig,
)

config = GeneticAlgorithmConfig(
    population_size=30,
    generations=1000,
    mutation_rate=0.01,
    early_stop_patience=5,
)

ga = GeneticAlgorithmStrategy(config=config, verbose=True)

result = ga.optimize(
    connections=initial_connections,
    evaluate_fn=lambda conn: compute_perplexity(conn),
    total_input_bits=context_size * bits_per_token,
    num_neurons=15,
    n_bits_per_neuron=20,
    batch_evaluate_fn=rust_batch_eval,  # Optional: massive speedup
)

print(f"Improved by {result.improvement_percent:.2f}%")
optimized = result.optimized_connections
```

## References

- Garcia, L.M. (2003). "Global Optimization Methods for Choosing Connectivity Patterns of Weightless Neural Networks"
