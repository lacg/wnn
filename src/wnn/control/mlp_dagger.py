"""CTRL-84 (ABI 71): train an MLP baseline head by DAgger, under the WNN's own training config.

The WNN's per-genome training (ControllerEvaluator, K = 5 folds ACCUMULATED into one memory:
fold k+1 continues from fold k) becomes, for the MLP: ONE controller carrying the head, trained
fold after fold by ram_controller.dagger_train_mlp with the SAME packed config
(evaluator.packed_train_config: teacher, rounds, episodes per round, tilt curriculum, gate,
weather, plant, anchor) and the SAME fold seeds, the fitter aggregating every round's records
across folds. Each fold ends on the head the best-checkpoint rule kept (as the WNN's cells do),
and the next fold's fitter warm-starts from it.
"""

from __future__ import annotations

from .mlp_fitter import MlpFitConfig, MlpFitter
from .mlp_payload import MlpPayload


class MlpDaggerTrainer:
	def __init__(self, evaluator, fit_cfg: MlpFitConfig, num_rounds: int | None = None,
	             episodes_per_round: int | None = None):
		self.ev = evaluator
		self.fit_cfg = fit_cfg
		self.num_rounds = num_rounds
		self.episodes_per_round = episodes_per_round

	def _cfg(self):
		cfg = self.ev.packed_train_config()
		if self.num_rounds is not None:
			cfg.num_rounds = int(self.num_rounds)        # smoke / dry-run shrink only
		if self.episodes_per_round is not None:
			cfg.episodes_per_round = int(self.episodes_per_round)
		return cfg

	def train(self, fold_seeds: list, log=print) -> tuple:
		"""Returns (payload, per-fold stats dicts, fitter)."""
		from wnn.control import _accel as ra
		from .evaluator import random_connectivity
		spec = self.ev.spec
		sc, oc = random_connectivity(spec, seed=0)  # never read: the head replaces the lookup
		c = self.ev.new_train_controller(spec, sc, oc)
		nf, nm = int(c.num_features()), int(spec.num_motors)
		fitter = MlpFitter(self.fit_cfg, nf, nm, float(c.neutral_decode))
		fitter.payload().attach(c)
		cfg = self._cfg()
		rg = self.ev.rg_config
		target = list(rg.target_rpy) if getattr(rg, "target_rpy", None) is not None else [0.0, 0.0, 0.0]
		stats = []
		for k, seed in enumerate(fold_seeds):
			ts = ra.dagger_train_mlp(c, cfg, fitter, target, int(seed))
			kept = MlpPayload.from_rust(nf, nm, c.mlp_head_export())
			fitter.load_payload(kept)
			st = dict(fold=k, seed=int(seed), iter_fitness=list(ts.iter_fitness),
			          iter_mean_err_deg=list(ts.iter_mean_err_deg), iter_stable_rate=list(ts.iter_stable_rate),
			          iter_records=list(ts.iter_cells_written), iter_n_trained=list(ts.iter_n_trained),
			          records_total=fitter.n_records)
			stats.append(st)
			log(f"[ctrl84] fold {k + 1}/{len(fold_seeds)} seed {seed}: rounds {cfg.num_rounds} x {cfg.episodes_per_round} eps | "
			    f"round eval err {['%.2f' % e for e in st['iter_mean_err_deg']]} stable {['%.2f' % s for s in st['iter_stable_rate']]} | "
			    f"records {fitter.n_records} | last fit loss {fitter.fit_log[-1]['loss'] if fitter.fit_log else float('nan'):.3e}")
		meta = dict(fit=vars(self.fit_cfg) | {"hidden": list(self.fit_cfg.hidden)}, folds=[int(s) for s in fold_seeds],
		            rounds=int(cfg.num_rounds), episodes_per_round=int(cfg.episodes_per_round), records=fitter.n_records)
		return MlpPayload.from_rust(nf, nm, c.mlp_head_export(), meta=meta), stats, fitter
