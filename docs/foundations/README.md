# Foundations

This directory contains foundational documentation for the Weightless Neural Network (WNN) research project. It covers the core concepts, algorithms, and optimization techniques used throughout the codebase.

## Contents

| Document | Description |
|----------|-------------|
| [RAM Neurons](ram.md) | What RAM neurons are and why partial connectivity enables generalization |
| [Connectivity Optimization](optimization.md) | Overview of the metaheuristic search for optimal neuron connectivity |
| [Genetic Algorithm](ga.md) | GA-based population search for connectivity patterns |
| [Tabu Search](ts.md) | TS-based local search with memory, including cooperative multi-start variant |
| [Simulated Annealing](sa.md) | SA-based probabilistic search with temperature scheduling |

## Reading Order

For newcomers to the project:

1. Start with **RAM Neurons** to understand the fundamental building block
2. Read **Connectivity Optimization** for why connectivity matters and how the three algorithms relate
3. Dive into **GA**, **TS**, or **SA** based on which algorithm you're working with

## Key References

- Aleksander, I., & Morton, H. (1990). *An Introduction to Neural Computing.* Chapman & Hall.
- Garcia, L. A. C. (2003). *Metodos de Otimizacao Global para Determinacao das Conexoes de Redes Neurais sem Peso.* MSc Thesis, UFPE, Brazil.
- Ludermir, T. B., Carvalho, A., Braga, A. P., & Souto, M. (1999). "Weightless neural models: a review of current and past works." *Neural Computing Surveys*, 2.
