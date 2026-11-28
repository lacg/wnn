"""CTRL-84 (ABI 71): one trained MLP baseline head — the value object that moves between the
PyTorch fitter (mlp_fitter.py), the Rust controller (WnnController.set_mlp_head) and disk.

The head replaces ONLY the WNN's output-layer lookup: it maps the controller's pre-thermometer
feature vector to the per-bank raw decode in [0, 1]; the Rust controller does the rest (CTBR
alphabet, delta law, inner rate loop). Layout of `params` (mlp_head.rs): per layer, the row-major
weights W[out][in] then the biases, input -> hidden... -> output — exactly torch.nn.Linear's
(weight.flatten(), bias) order, so export is a concatenation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np


@dataclass
class MlpPayload:
	in_dim: int
	hidden: tuple
	activation: str
	shift: np.ndarray
	scale: np.ndarray
	params: np.ndarray
	out_dim: int = 4
	meta: dict = field(default_factory=dict)

	def __post_init__(self):
		self.hidden = tuple(int(h) for h in self.hidden)
		self.shift = np.ascontiguousarray(self.shift, dtype=np.float32)
		self.scale = np.ascontiguousarray(self.scale, dtype=np.float32)
		self.params = np.ascontiguousarray(self.params, dtype=np.float32).ravel()
		need = param_count(self.in_dim, self.hidden, self.out_dim)
		if self.params.size != need or self.shift.size != self.in_dim or self.scale.size != self.in_dim:
			raise ValueError(f"MlpPayload: {self.params.size} params / {self.shift.size} shift / {self.scale.size} scale "
			                 f"do not fit in_dim {self.in_dim} hidden {self.hidden} out_dim {self.out_dim} (need {need} params)")

	def attach(self, controller) -> None:
		"""Install this head on a Rust WnnController (replaces its output-layer lookup)."""
		controller.set_mlp_head(self.in_dim, list(self.hidden), self.activation,
		                        self.shift.tolist(), self.scale.tolist(), self.params.tolist())

	def rust_tuple(self) -> tuple:
		"""The dagger_train_mlp fit-callback return form."""
		return (list(self.hidden), self.activation, self.shift, self.scale, self.params)

	@classmethod
	def from_rust(cls, in_dim: int, out_dim: int, exported: tuple, meta: dict | None = None) -> "MlpPayload":
		"""From WnnController.mlp_head_export()."""
		hidden, act, shift, scale, params = exported
		return cls(in_dim=in_dim, hidden=tuple(hidden), activation=act, out_dim=out_dim,
		           shift=np.asarray(shift, dtype=np.float32), scale=np.asarray(scale, dtype=np.float32),
		           params=np.asarray(params, dtype=np.float32), meta=dict(meta or {}))

	def shape_key(self) -> tuple:
		return (self.in_dim, self.hidden, self.out_dim, self.activation)

	def cost(self) -> dict:
		"""Per-decision arithmetic and footprint (matches mlp_head.rs MlpHead::cost)."""
		widths = (self.in_dim, *self.hidden, self.out_dim)
		macs = sum(a * b for a, b in zip(widths[:-1], widths[1:]))
		n = int(self.params.size)
		return dict(params=n, macs=macs + self.in_dim, adds=macs + sum(widths[1:]), activations=sum(self.hidden),
		            bytes_fp32=4 * (n + 2 * self.in_dim), bytes_int8=n + 4 * 2 * self.in_dim)

	def save(self, path: str) -> None:
		np.savez(path, in_dim=self.in_dim, hidden=np.array(self.hidden), activation=self.activation,
		         out_dim=self.out_dim, shift=self.shift, scale=self.scale, params=self.params,
		         meta=json.dumps(self.meta))

	@classmethod
	def load(cls, path: str) -> "MlpPayload":
		z = np.load(path, allow_pickle=False)
		return cls(in_dim=int(z["in_dim"]), hidden=tuple(int(h) for h in z["hidden"]), activation=str(z["activation"]),
		           out_dim=int(z["out_dim"]), shift=z["shift"], scale=z["scale"], params=z["params"],
		           meta=json.loads(str(z["meta"])))


def param_count(in_dim: int, hidden, out_dim: int) -> int:
	widths = (in_dim, *hidden, out_dim)
	return sum(a * b + b for a, b in zip(widths[:-1], widths[1:]))
