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

### Elitism

**Elitism** is the practice of preserving the best individuals unchanged across generations, guaranteeing monotonic improvement of the population's best fitness. Without elitism, the best solution found can be lost through crossover or mutation — Rudolph (1994) proved that GAs without elitism cannot guarantee convergence to the global optimum.

In our implementation, `elitism_pct × 2` of the population (default 20%) is preserved each generation. The elites are selected using the **unified fitness calculator** — the same ranking function (e.g., weighted harmonic mean of CE rank and accuracy rank) used for all other selection decisions. This ensures that elite preservation and offspring selection optimize for the same objective.

The remaining slots are filled with offspring generated via tournament selection, crossover, and mutation. Elites and offspring are then combined, and the process repeats.

**Elite survival tracking**: The GA tracks how many initial elites survive to the final population and computes an "elite win rate" (fraction of generations where the best elite outperformed the best new offspring). High elite win rates can indicate insufficient diversity or premature convergence.

## Configuration

```python
GAConfig(
    population_size=50,      # Number of genomes per generation
    generations=100,         # Maximum generations
    crossover_rate=0.7,      # Crossover probability per offspring
    tournament_size=3,       # Tournament selection size
    elitism_pct=0.1,         # Fraction preserved (×2 internally → 20% kept)
    patience=10,             # Early stop after N checks without improvement
    check_interval=10,       # Check improvement every N generations
    min_improvement_pct=0.05,# Minimum % improvement to reset patience
)
```

## Fitness Evaluation

Genomes are ranked using a configurable [fitness calculator](fitness.md). The default is **Harmonic Rank** — the weighted harmonic mean of each genome's CE rank and accuracy rank within the population. This naturally balances both objectives and penalizes genomes that are good at one metric but poor at the other.

The fitness calculator is used for:
- **Elitism**: Selecting which genomes are preserved unchanged
- **Tournament selection**: Comparing two candidates during parent selection
- **Best tracking**: Determining the overall best genome across all generations

See the [Fitness Calculators](fitness.md) page for all available types, formulas, and trade-offs.

## In This Project

GA serves as the **exploration phase** in each optimization dimension:
- Phase 2: GA over neuron counts per cluster
- Phase 4: GA over bit counts per cluster
- Phase 6: GA over connection assignments

Population size of 50 genomes, evaluated in parallel via the Rust+Metal accelerator, provides good diversity without excessive computation.

## References

- Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems.* University of Michigan Press.
- Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning.* Addison-Wesley.
- De Jong, K. A. (1975). *An Analysis of the Behavior of a Class of Genetic Adaptive Systems.* PhD Dissertation, University of Michigan.
  - First systematic study of GA parameters including elitism (called "preselection" in his terminology).
- Rudolph, G. (1994). "Convergence Analysis of Canonical Genetic Algorithms." *IEEE Transactions on Neural Networks*, 5(1), 96-101.
  - Proves that GAs with elitism converge to the global optimum with probability 1; without elitism, convergence is not guaranteed.
