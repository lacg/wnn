"""
Registered-but-inert flow params (IDS-54, 08/11/2026).

A key in KNOWN_PARAMS that nothing reads is worse than an unknown key: the
unknown-key warning is silenced, so a flow that SETS it believes it ran with it.
The audit (docs/ids_audit_controller_bug_classes_2026-11-08.md §4) found six.
Each is now either wired (mutation_rate -> GAConfig/TSConfig) or handled here:

  - accepted at its DEFAULT with a one-line note (thousands of queued/completed
    flows carry the default — they must keep ingesting);
  - REFUSED loudly (ValueError -> the flow fails at ingestion) at any other
    value, because the run would silently differ from its config.

Plus `warn_inapplicable_params`: a key that has no effect for the flow's
architecture (an `ids_*` key on a controller flow, an LM tier/stage key on an
IDS flow, ...) is logged — the run is valid, the key is a no-op.
"""

from typing import Callable, Iterable

# key -> (accepted default, why any other value is refused)
_INERT_AT_DEFAULT = {
	"decode_mode": (
		"argmax",
		"no decoder reads it — multiclass flows already report every decode mode",
	),
	"ids_arch_type": (
		"tiered",
		"IDS genomes are always tiered clusters; 'bitwise' was never implemented",
	),
}

# key -> why the recorded value is not applied (any value accepted, noted)
_DEPRECATED_RECORDS = {
	"fitness_ce_anchor_normalized": "the CE anchor is derived automatically per task "
	"(NORMALIZED_CE_ANCHOR x the task's base-rate entropy)",
}

# Handled HERE, so validate_flow_params does not also call them UNKNOWN.
INERT_KEYS = frozenset(_INERT_AT_DEFAULT)

_LM_KEYS = frozenset({
	"tier_config", "tier0_only", "optimize_tier0_only", "num_stages", "stage_k", "stage_mode",
	"stage_context_size", "stage_cluster_type", "stage_min_bits", "stage_max_bits",
	"stage_min_neurons", "stage_max_neurons", "stage_neurons_grid", "stage_bits_grid",
	"tiered_neurons_grid", "tiered_bits_grid", "bigram_lambda", "unigram_lambda",
	"label_smoothing", "reweight_rounds", "reweight_max_boost",
})
_LM_ARCHITECTURES = frozenset({"tiered", "bitwise", "multi_stage"})


def enforce_inert_params(params: dict, log: Callable[[str], None], prefix: str = "[PARAMS]") -> None:
	"""Note inert keys at their default; raise on any other value."""
	refusals = []
	for key, (default, why) in _INERT_AT_DEFAULT.items():
		if key not in params or params[key] is None:
			continue
		if params[key] == default:
			log(f"{prefix} note: '{key}'={params[key]!r} is the default and has no effect ({why})")
		else:
			refusals.append(f"'{key}'={params[key]!r} (only {default!r} is accepted: {why})")
	for key, why in _DEPRECATED_RECORDS.items():
		if key in params:
			log(f"{prefix} note: '{key}'={params[key]!r} is DEPRECATED and not applied ({why})")
	if refusals:
		raise ValueError(f"{prefix} REFUSED inert flow params (IDS-54): " + "; ".join(refusals))


def warn_inapplicable_params(params: dict, log: Callable[[str], None], prefix: str = "[PARAMS]") -> list:
	"""Log every key that has no effect for this flow's architecture."""
	arch = params.get("architecture_type", "tiered")
	noop = sorted(k for k in params if not _applies(k, arch))
	for key in noop:
		log(f"{prefix} ⚠️  '{key}' has no effect for architecture_type={arch!r} — ignored")
	return noop


def _applies(key: str, arch: str) -> bool:
	if key.startswith("ids_"):
		return arch == "ids"
	if key.startswith("controller_"):
		return arch == "controller"
	if key in _LM_KEYS:
		return arch in _LM_ARCHITECTURES
	return True


def enforce_phase_order(params: dict, phase_types: Iterable[str]) -> None:
	"""phase_order is applied when the flow is CREATED (it orders the experiment
	rows; the worker runs rows in order). A non-default value is refused unless
	the rows actually follow it — otherwise the run would silently differ."""
	order = params.get("phase_order")
	if order in (None, "neurons_first"):
		return
	searched = [p for p in phase_types if p and p != "grid_search"]
	if order == "bits_first" and searched and "bits" in searched[0]:
		return
	raise ValueError(
		f"[PARAMS] REFUSED phase_order={order!r} (IDS-54): the worker runs the experiment rows "
		f"in order and they do not follow it (first searched phase: {searched[:1]})"
	)


def refuse_inert_experiment_knobs(params: dict, experiment_kind: str) -> None:
	"""random_search exists only in the GA loop: on a TS / adaptation experiment
	it would be silently ignored (IDS-51/54)."""
	if params.get("random_search") and experiment_kind in ("ts", "adaptation"):
		raise ValueError(
			f"[PARAMS] REFUSED random_search=True on a {experiment_kind} experiment: only the GA "
			"implements the random-search baseline"
		)
