# Connectivity Optimization

## Why Connectivity Matters

In RAM-based neural networks, the **connectivity map** — which input bits each neuron observes — is the primary generalization mechanism. Unlike weighted neural networks where gradient descent adjusts continuous parameters, RAM WNNs learn by:

1. **Connectivity optimization**: Finding which input bits each neuron should observe (analogous to learning which features matter)
2. **Memory writes**: Storing input-output mappings at the computed addresses (analogous to final weight values)

Random connectivity provides a baseline, but optimized connectivity dramatically improves performance. The connectivity map determines what **features** each neuron detects, and the right features lead to generalization.

## The Optimization Problem

Given:
- **N** neurons across **K** output clusters
- **B** total input bits (e.g., context_size x bits_per_token = 64 for 4-gram GPT-2)
- Each neuron observes **b** bits (its "bits_per_neuron" parameter)

Find: The assignment of **b** input bit indices to each neuron that minimizes cross-entropy loss on the training data.

This is a combinatorial optimization problem. For 200 neurons each choosing 20 bits from 64 available, the search space is C(64,20)^200 — astronomically large. Exact search is infeasible, so we use **metaheuristic** optimization.

## Three Metaheuristics

The project implements three complementary optimization algorithms, following the approach of Garcia (2003):

| Algorithm | Strategy | Strength | Weakness |
|-----------|----------|----------|----------|
| [Genetic Algorithm](ga.md) | Population-based search | Explores diverse solutions | Can be slow to converge |
| [Tabu Search](ts.md) | Memory-guided local search | Fast local refinement | Can miss distant optima |
| [Simulated Annealing](sa.md) | Temperature-based random walk | Escapes local optima | Requires careful tuning |

### How They Work Together

In practice, the algorithms are used in a **phased pipeline**:

```
Phase 1: Grid Search     → Find promising (neurons, bits) configurations
Phase 2: GA Exploration  → Population search for neuron counts
Phase 3: TS Refinement   → Local refinement of neuron counts
Phase 4: GA Exploration  → Population search for bit counts
Phase 5: TS Refinement   → Local refinement of bit counts
Phase 6: GA Exploration  → Population search for connections
Phase 7: TS Refinement   → Local refinement of connections
```

Each phase seeds from the previous phase's best genome. GA provides broad exploration, TS provides focused refinement.

## Genome Representation

A "genome" in this project represents a complete architecture configuration:

```python
ClusterGenome:
    neurons_per_cluster: [200, 200, ..., 200]  # per output cluster
    bits_per_cluster:    [20,  20,  ..., 20]    # per output cluster
    connections:         [[5,12,3,...], ...]     # per neuron: which input bits
```

Different phases mutate different dimensions:
- **Neuron phases**: Mutate `neurons_per_cluster` (how many neurons per output)
- **Bit phases**: Mutate `bits_per_cluster` (how many input bits per neuron)
- **Connection phases**: Mutate `connections` (which specific input bits)

## Evaluation Pipeline

Each genome evaluation consists of:

1. **Training**: Write memory cells from training data (Rust, CPU parallel)
2. **Forward pass**: Read memory cells on evaluation data (Rust, CPU)
3. **Reconstruction + CE**: Compute cross-entropy loss (Metal GPU or CPU fallback)

The Rust+Metal accelerator evaluates 50 genomes in parallel, making population-based optimization feasible.

## Key Results

From Garcia (2003), connectivity optimization improved recognition rates by 15-30% on pattern recognition tasks compared to random connectivity. In our language modeling experiments, optimized connectivity reduces cross-entropy by 5-15% compared to random initialization.

## References

- Garcia, L. A. C. (2003). *Metodos de Otimizacao Global para Determinacao das Conexoes de Redes Neurais sem Peso.* MSc Thesis, Federal University of Pernambuco (UFPE), Brazil.
- Aleksander, I., & Morton, H. (1990). *An Introduction to Neural Computing.* Chapman & Hall.
