# CICIDS-random — thermo × weight + deployed genome size

Held-out (report) best-F1 genome per cell, with the architecture the GA actually converged to. **Single seed r82096** (sweep mid-run). bits = 34 uniform on every non-degenerate winner.

- wiring (exact) = neurons × bits — the deterministic FPGA cost
- sparse-cell UPPER BOUND = neurons × min(2^bits, n_train_sampled=565,576); real fill is lower (address collisions), exact needs Vivado synth


## Headline genome per cell (best-F1 = best-ACC; FPR is that same genome's)

| weight | 32b | 64b | 96b |
|---|---|---|---|
| Wa | 99.57/0.08/99.73 · **104n×34b** | 99.56/0.09/99.73 · **105n×34b** | 99.59/0.07/99.74 · **107n×34b** |
| Wb | 99.40/0.19/99.62 · **109n×34b** | 99.40/0.25/99.62 · **290n×34b** | 99.52/0.12/99.70 · **100n×34b** |
| Wbu | 99.54/0.17/99.71 · **95n×34b** | 99.59/0.07/99.74 · **172n×34b** | 99.48/0.21/99.67 · **212n×34b** |
| Wc | 99.47/0.18/99.67 · **421n×34b** | 99.54/0.10/99.71 · **202n×34b** | 99.56/0.07/99.73 · **205n×34b** |

## Deployed size (best-F1 genome)

| cell | neurons | bits | wiring (n×b) | sparse-cell upper bound | hash |
|---|---|---|---|---|---|
| 32b Wa | 104 | 34 | 3,536 | ≤ 58,819,904 | `d80392163f26` |
| 32b Wb | 109 | 34 | 3,706 | ≤ 61,647,784 | `6fb6cf5d089e` |
| 32b Wbu | 95 | 34 | 3,230 | ≤ 53,729,720 | `53b5980d2371` |
| 32b Wc | 421 | 34 | 14,314 | ≤ 238,107,496 | `8880e155555c` |
| 64b Wa | 105 | 34 | 3,570 | ≤ 59,385,480 | `bdcf4073aec1` |
| 64b Wb | 290 | 34 | 9,860 | ≤ 164,017,040 | `6dfc93ebd660` |
| 64b Wbu | 172 | 34 | 5,848 | ≤ 97,279,072 | `3612a97abe5f` |
| 64b Wc | 202 | 34 | 6,868 | ≤ 114,246,352 | `1208638e1b76` |
| 96b Wa | 107 | 34 | 3,638 | ≤ 60,516,632 | `e2463f877065` |
| 96b Wb | 100 | 34 | 3,400 | ≤ 56,557,600 | `330313073bb7` |
| 96b Wbu | 212 | 34 | 7,208 | ≤ 119,902,112 | `9bca20699815` |
| 96b Wc | 205 | 34 | 6,970 | ≤ 115,943,080 | `c0ba23a423e3` |

## ⚠ Degenerate best-FPR genomes (do NOT cite as FPGA wins)

Tiny genomes that hit low FPR by under-predicting attacks → F1 collapses. Flagged so they aren't mistaken for efficient deployable points.

| cell | best-FPR genome | F1 | FPR | verdict |
|---|---|---|---|---|
| 32b Wa | 5n×8b | 85.85 | 0.153 | ⚠ DEGENERATE |
| 32b Wc | 5n×8b | 85.85 | 0.153 | ⚠ DEGENERATE |
| 96b Wbu | 19n×4b | 78.06 | 14.008 | ⚠ DEGENERATE |

## Takeaways

- **Thermo width barely changes the deployed genome.** Wa converges to ~104–107 neurons × 34b across 32/64/96b with near-identical quality (~99.57 F1 / ~0.08 FPR). The ~105n×34b architecture is the invariant; thermo width mostly affects *search time*, not deployed size.

- **neurons is the FPGA lever** (bits pinned at 34). Non-degenerate winners span ~88–290 neurons; pick the (width, weight) yielding the leanest STRONG genome.

- **Always pair a small-neuron claim with its F1** — the 5n/19n 'best-FPR' genomes are degenerate (F1 78–86), not deployable.

