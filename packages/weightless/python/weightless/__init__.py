"""weightless — RAM (weightless) neural networks for scikit-learn.

	from weightless import WiSARDClassifier, ThermometerEncoder, CellMode
	clf = make_pipeline(ThermometerEncoder(n_bits=8), WiSARDClassifier(neurons_per_class=50, bits_per_neuron=16))
	cross_val_score(clf, X, y, cv=5)

The classifier consumes BITS (a 0/1 matrix); the encoder makes them. Cell modes
are the six-rung ladder in `CellMode`. Scoring runs on the CPU everywhere and on
the Metal GPU on Apple silicon (`backends()` says which is live here).
"""

from ._cell_mode import CellMode
from ._core import ABI_VERSION as _CORE_ABI, backends
from .classifier import WiSARDClassifier
from .encoder import ThermometerEncoder, ThermometerMethod

EXPECTED_ABI = 1
if _CORE_ABI != EXPECTED_ABI:
	raise ImportError(
		f"weightless._core ABI {_CORE_ABI} != expected {EXPECTED_ABI}: "
		"the compiled extension and the Python package are from different builds — reinstall the wheel"
	)

__version__ = "0.1.0"
__all__ = ["CellMode", "WiSARDClassifier", "ThermometerEncoder", "ThermometerMethod", "backends"]
