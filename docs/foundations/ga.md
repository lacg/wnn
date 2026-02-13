# Genetic Algorithm (GA)

## Overview

The Genetic Algorithm is a population-based metaheuristic inspired by natural selection. It maintains a population of candidate solutions (genomes), evolves them through selection, crossover, and mutation, and converges toward high-fitness solutions.

In this project, GA is used for **broad exploration** of the connectivity search space — finding promising regions before Tabu Search refines them.

## Algorithm

```
GENETIC ALGORITHM
─────────────────
Input:  population_size, generations, mutation_rate, elite_pct
Output: best genome found

1. Initialize population of random genomes
2. Evaluate fitness of all genomes (batch evaluation via Rust+Metal)
3. For each generation:
   a. SELECT parents via tournament selection
   b. CROSSOVER pairs of parents to create offspring
   c. MUTATE offspring with probability mutation_rate
   d. EVALUATE fitness of all offspring
   e. ELITE PRESERVATION: keep top elite_pct% unchanged
   f. Replace population with elites + best offspring
   g. Check early stopping (patience-based)
4. Return best genome found
```

## Key Operations

### Selection
**Tournament selection** with configurable tournament size. Two random genomes compete; the one with better fitness (lower CE) is selected as a parent. This balances exploration (randomness) with exploitation (fitness pressure).

### Crossover
**Uniform crossover** per cluster: for each output cluster, randomly inherit the (neurons, bits, connections) configuration from either parent. This preserves cluster-level coherence while mixing architectural choices.

### Mutation
Per-cluster mutations with probability `mutation_rate`:
- **Neuron mutation**: Add or remove neurons (±1 to ±3)
- **Bit mutation**: Change bits per neuron (±1 to ±2)
- **Connection mutation**: Swap individual input bit assignments

### Elite Preservation
The top `elite_pct`% of the population (by fitness ranking) are copied unchanged into the next generation. This ensures the best solutions are never lost.

Dual elite selection: 10% by CE (best loss) + 10% by accuracy (best classification). This maintains diversity between solutions optimized for different objectives.

## Configuration

```python
GAConfig(
    population_size=50,      # Number of genomes per generation
    generations=100,         # Maximum generations
    mutation_rate=0.1,       # Per-cluster mutation probability
    elite_pct=0.2,           # Fraction preserved unchanged
    patience=10,             # Early stop after N checks without improvement
    check_interval=10,       # Check improvement every N generations
    min_improvement_pct=1.0, # Minimum % improvement to reset patience
)
```

## Fitness Evaluation

Genomes are ranked using a configurable fitness calculator:

- **CE**: Pure cross-entropy ranking (lower = better)
- **HARMONIC_RANK**: Weighted harmonic mean of CE rank and accuracy rank, balancing both objectives

See the [Fitness Calculator](../BITWISE_OPTIMIZATION.md) documentation for details.

## In This Project

GA serves as the **exploration phase** in each optimization dimension:
- Phase 2: GA over neuron counts per cluster
- Phase 4: GA over bit counts per cluster
- Phase 6: GA over connection assignments

Population size of 50 genomes, evaluated in parallel via the Rust+Metal accelerator, provides good diversity without excessive computation.

## References

- Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems.* University of Michigan Press.
- Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning.* Addison-Wesley.
