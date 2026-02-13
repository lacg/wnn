# Simulated Annealing (SA)

## Overview

Simulated Annealing is a probabilistic metaheuristic inspired by the annealing process in metallurgy. It explores the solution space by accepting both improving and worsening moves, with the probability of accepting worse moves decreasing over time according to a **temperature schedule**.

In the context of RAM connectivity optimization, SA provides an alternative to TS for escaping local optima, with the advantage of requiring fewer configuration parameters.

## Algorithm

```
SIMULATED ANNEALING
────────────────────
Input:  initial_genome, max_iterations, initial_temp, cooling_rate
Output: best genome found

1. Set current = initial_genome, best = initial_genome
2. Set temperature = initial_temp
3. For each iteration:
   a. GENERATE a random neighbor of current genome
   b. EVALUATE neighbor fitness
   c. Compute delta = neighbor_fitness - current_fitness
   d. If delta < 0 (improvement):
      - Accept neighbor as new current
   e. Else (worsening):
      - Accept with probability exp(-delta / temperature)
   f. UPDATE best if current is better than best
   g. COOL: temperature *= cooling_rate
   h. Check stopping criteria
4. Return best genome found
```

## Key Concepts

### Temperature Schedule

The temperature controls the exploration-exploitation trade-off:

- **High temperature** (early): Accept most moves, including worsening ones. The search explores broadly.
- **Low temperature** (late): Accept only improving moves. The search converges to a local optimum.

Common cooling schedules:
- **Geometric**: T(k+1) = alpha * T(k), where alpha is typically 0.95-0.99
- **Linear**: T(k) = T_0 - k * delta
- **Adaptive**: Adjust based on acceptance rate

### Acceptance Probability

For a worsening move with fitness increase delta > 0:

```
P(accept) = exp(-delta / T)
```

At high T, P ≈ 1 (accept almost anything). At low T, P ≈ 0 (reject worsening moves).

### Comparison with TS

| Aspect | Tabu Search | Simulated Annealing |
|--------|-------------|---------------------|
| **Escape mechanism** | Tabu list prevents cycling | Temperature allows uphill moves |
| **Memory** | Explicit (tabu list) | None (stateless) |
| **Neighbors per step** | Many (batch evaluated) | Typically one |
| **Determinism** | Mostly deterministic | Stochastic |
| **Tuning** | tabu_size, neighbors | temperature, cooling_rate |

## In This Project

SA is implemented in the generic optimization framework (`generic_strategies.py`) but is currently less used than the GA+TS pipeline. It serves as:

1. A reference implementation for comparison against GA and TS
2. An alternative when TS stagnates and a different search strategy is needed
3. A simpler algorithm for quick prototyping of new optimization dimensions

## Configuration

SA shares the base `OptimizationConfig` parameters (patience, check_interval, etc.) with GA and TS, plus SA-specific parameters for temperature scheduling.

## References

- Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). "Optimization by Simulated Annealing." *Science*, 220(4598), 671-680.
- Cerny, V. (1985). "Thermodynamical approach to the traveling salesman problem." *Journal of Optimization Theory and Applications*, 45(1), 41-51.
- Garcia, L. A. C. (2003). *Metodos de Otimizacao Global para Determinacao das Conexoes de Redes Neurais sem Peso.* MSc Thesis, UFPE, Brazil.
