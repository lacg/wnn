# weightless

RAM (weightless) neural networks for scikit-learn — WiSARD with a six-mode cell ladder,
order-independent training, and a Rust core that scores on the CPU everywhere and on the
Metal GPU on Apple silicon.

```python
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from weightless import ThermometerEncoder, WiSARDClassifier, CellMode

clf = make_pipeline(
	ThermometerEncoder(n_bits=8),
	WiSARDClassifier(neurons_per_class=50, bits_per_neuron=16, cell_mode=CellMode.QUAD_WEIGHTED),
)
cross_val_score(clf, X, y, cv=5)
```

## The cell ladder

| mode | cell states | read |
|---|---|---|
| `BINARY` | FALSE / TRUE | classical WiSARD: TRUE fires; other classes never write |
| `TERNARY` | FALSE / untrained / TRUE | untrained reads `empty_value` |
| `PLN` | as TERNARY | untrained fires a fair coin (seeded) |
| `QUAD_BINARY` | 4-state nudging | WEAK_TRUE / TRUE fire (ablation) |
| `QUAD_WEIGHTED` | 4-state nudging | graded 0 / .25 / .75 / 1 — **default** |
| `QSR` | 4-state nudging | coin with p = graded weight (seeded) |

`PLN` equals `TERNARY` and `QSR` equals `QUAD_WEIGHTED` in expectation; `predict_proba`
returns that expected read, `predict` the seeded sample.

## Coming from wisardpkg

`WiSARDClassifier(cell_mode=CellMode.BINARY)` is the classical discriminator: one-shot,
own-class-only writes, TRUE-only reads. Everything above it on the ladder adds graduated
confidence from the negatives.

## Install (development)

```
cd packages/weightless
maturin develop --release
pytest
```

Training is order-independent by construction (`partial_fit` accumulates exactly), the
classifier consumes bit matrices only (the encoder makes them), and `export_keys()` returns
the trained memory as sorted keys — the deployment format.
