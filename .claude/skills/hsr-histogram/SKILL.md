---
name: hsr-histogram
description: Run the per-genome-shape × HSR timing histogram analysis. Use when the user asks for "hsr histogram", "hsr update", "which hsr is winning", or wants the per-shape side-by-side timing breakdown across HSR=1/2/3/5/7/8/10.
---

# HSR histogram

Generate and display the genome-shape × HSR timing comparison from per-genome
`eval_time_ms` data.

## Steps

1. Run the histogram script. By default it auto-detects the active HSR cohort:
   ```bash
   python3 scripts/hsr_histogram_analysis.py --min-samples 1
   ```
   Use `--min-samples 3` once each cell has more data (typically when cohort
   has n≥4 per HSR).
2. To target a specific dataset cohort, pass `--cohort PREFIX`. List options:
   ```bash
   python3 scripts/hsr_histogram_analysis.py --list
   ```
3. Show the user:
   - **Cohort timing data** header (total rows, per-HSR flow count)
   - **Genome shape × HSR table** (mean ± std in ms, ★ marks winner per shape)
   - **Δ vs winner table** (ms slower than fastest HSR for each shape)
   - **Aggregate verdict** (winner-count distribution across shapes)
4. Brief insights:
   - Which HSR ratios are leading and which shapes they dominate
   - Whether the pattern matches the theoretical expectation:
     low HSR (1-2) wins → pure-path beats hybrid; high HSR (8-10) wins → hybrid
     beats pure-path. Middle ratios (3-5) win for intermediate balance points.
   - Caveat about small n: cells with n=1 std=0 are not reliable yet.

## HSR semantics (important — don't confuse direction)

HSR is the **ceiling on CPU-vs-GPU speed imbalance that still enables hybrid mode**
(see `adaptive.rs:4678`). It does NOT bias which primary path is chosen; that's
the separate B11 logic (`use_gpu_batched`). HSR only decides whether to recruit
CPU concurrently with GPU.

- **HSR=1**: hybrid only when paths are nearly tied → in practice "never hybrid"
- **HSR=10**: hybrid even when one path is 10× faster → "always hybrid"

If a low HSR wins, it means **pure-path beats hybrid** for that shape (bandwidth
contention costs > parallelism gain). If a high HSR wins, hybrid is paying off
despite contention.

## When to mention the HSR-as-function approach

The user is interested in HSR(neurons, bits) → optimal HSR rather than a single
global default. The histogram script's per-shape "winner" output is exactly the
training data for that function. If the user asks about fitting it, point to:
- Current per-shape winners table
- Possible interpolation/regression approach over (n, b, HSR) → time
