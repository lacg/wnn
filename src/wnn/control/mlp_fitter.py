"""CTRL-84 (ABI 71): the MLP baseline's gradient step — the DAgger fit callback.

Rust (ram_controller.dagger_train_mlp) owns the DAgger loop: rollouts with the MLP flying
(Rust forward), teacher labels, gate, record, eval, best-checkpoint. After each round it calls
this object with the round's NEW records; the fitter appends them to the AGGREGATED dataset
(DAgger's data aggregation, kept across rounds AND folds) and fits the network on all of it,
then hands the head back. Only the gradient step lives here (PyTorch, MPS by default) — no
rollout, scorer or label logic.

Standardisation (shift = mean, scale = 1/std of each feature) is fixed at the FIRST fit (fold 0,
round 0 records) and kept: refitting it later would silently change what the warm-started
first layer reads. A constant feature (std < 1e-9, e.g. the tgt_* inputs the WNN drops) gets
scale 0 — the head cannot read it, as the WNN cannot.

Loss: mean squared error between the LINEAR output and the raw-decode target (the value whose
thermometer the WNN writes); deploy clamps the output to [0, 1] (mlp_head.rs).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mlp_payload import MlpPayload


@dataclass
class MlpFitConfig:
	hidden: tuple = (64, 64)
	activation: str = "relu"
	lr: float = 1e-3
	epochs: int = 4               # passes over the aggregated dataset per DAgger round
	weight_decay: float = 0.0
	batch_size: int = 1024
	seed: int = 0
	device: str = "auto"          # "auto" = mps if available else cpu


class MlpFitter:
	def __init__(self, cfg: MlpFitConfig, in_dim: int, out_dim: int, neutral: float):
		import torch
		self.torch = torch
		self.cfg = cfg
		self.in_dim, self.out_dim = int(in_dim), int(out_dim)
		self.device = self._device(cfg.device)
		torch.manual_seed(int(cfg.seed))
		self.model = self._build(neutral).to(self.device)
		self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
		self.shift = np.zeros(self.in_dim, dtype=np.float32)
		self.scale = np.ones(self.in_dim, dtype=np.float32)
		self._norm_fixed = False
		self._feats: list = []
		self._targets: list = []
		self.n_records = 0
		self.n_fits = 0
		self.fit_log: list = []

	def _device(self, want: str):
		t = self.torch
		if want == "auto":
			return t.device("mps" if t.backends.mps.is_available() else "cpu")
		return t.device(want)

	def _build(self, neutral: float):
		"""Hidden layers: torch's default init. Output layer: zero weights, bias = the decode
		NEUTRAL, so the untrained head is the empty WNN's policy (hold: zero delta)."""
		nn = self.torch.nn
		act = {"relu": nn.ReLU, "tanh": nn.Tanh}[self.cfg.activation]
		layers, w = [], self.in_dim
		for h in self.cfg.hidden:
			layers += [nn.Linear(w, int(h)), act()]
			w = int(h)
		out = nn.Linear(w, self.out_dim)
		with self.torch.no_grad():
			out.weight.zero_()
			out.bias.fill_(float(neutral))
		layers.append(out)
		return nn.Sequential(*layers)

	def payload(self, meta: dict | None = None) -> MlpPayload:
		chunks = []
		for m in self.model:
			if isinstance(m, self.torch.nn.Linear):
				chunks.append(m.weight.detach().float().cpu().numpy().ravel())
				chunks.append(m.bias.detach().float().cpu().numpy().ravel())
		return MlpPayload(in_dim=self.in_dim, hidden=tuple(self.cfg.hidden), activation=self.cfg.activation,
		                  shift=self.shift, scale=self.scale, params=np.concatenate(chunks), out_dim=self.out_dim,
		                  meta=dict(meta or {}))

	def load_payload(self, p: MlpPayload) -> None:
		"""Warm-start the network from a head (the one the best-checkpoint rule kept)."""
		if p.shape_key() != self.payload().shape_key():
			raise ValueError(f"load_payload: shape {p.shape_key()} != fitter {self.payload().shape_key()}")
		off = 0
		with self.torch.no_grad():
			for m in self.model:
				if isinstance(m, self.torch.nn.Linear):
					for t in (m.weight, m.bias):
						k = t.numel()
						t.copy_(self.torch.from_numpy(p.params[off:off + k].copy()).view_as(t))
						off += k
		self.shift, self.scale = p.shift.copy(), p.scale.copy()

	def _fix_norm(self, x: np.ndarray) -> None:
		mu = x.mean(axis=0, dtype=np.float64)
		sd = x.std(axis=0, dtype=np.float64)
		self.shift = mu.astype(np.float32)
		self.scale = np.where(sd > 1e-9, 1.0 / np.maximum(sd, 1e-30), 0.0).astype(np.float32)
		self._norm_fixed = True

	def __call__(self, feats: bytes, targets: bytes, n: int, in_dim: int, out_dim: int, round_idx: int):
		"""dagger_train_mlp's fit callback: aggregate, fit, return the new head (or None)."""
		if (in_dim, out_dim) != (self.in_dim, self.out_dim):
			raise ValueError(f"fit callback dims ({in_dim}, {out_dim}) != fitter ({self.in_dim}, {self.out_dim})")
		if n > 0:
			self._feats.append(np.frombuffer(feats, dtype="<f4").reshape(n, in_dim).copy())
			self._targets.append(np.frombuffer(targets, dtype="<f4").reshape(n, out_dim).copy())
			self.n_records += n
		if self.n_records == 0:
			return None
		x = np.concatenate(self._feats)
		y = np.concatenate(self._targets)
		self._feats, self._targets = [x], [y]
		if not self._norm_fixed:
			self._fix_norm(x)
		loss = self._fit(x, y)
		self.n_fits += 1
		self.fit_log.append(dict(round=int(round_idx), n=int(self.n_records), loss=float(loss)))
		return self.payload().rust_tuple()

	def _fit(self, x: np.ndarray, y: np.ndarray) -> float:
		t = self.torch
		z = (x - self.shift) * self.scale
		X = t.from_numpy(z.astype(np.float32)).to(self.device)
		Y = t.from_numpy(y.astype(np.float32)).to(self.device)
		g = t.Generator().manual_seed(int(self.cfg.seed) * 100003 + self.n_fits)
		bs = int(self.cfg.batch_size)
		self.model.train()
		last = float("nan")
		for _ in range(int(self.cfg.epochs)):
			perm = t.randperm(X.shape[0], generator=g).to(self.device)
			tot = t.zeros((), device=self.device)
			for i in range(0, X.shape[0], bs):
				idx = perm[i:i + bs]
				self.opt.zero_grad(set_to_none=True)
				loss = t.nn.functional.mse_loss(self.model(X[idx]), Y[idx])
				loss.backward()
				self.opt.step()
				tot = tot + loss.detach() * idx.numel()
			last = float(tot) / max(X.shape[0], 1)
		return last
