"""CTRL-58 (Luiz 18/11/2026): the staged REFINEMENT fitness — configuration.

WHY (fitness audit, logs/controller/ctrl48/fitness_audit/): once the population flies
every TRAIN episode, the base ZRank fitness stops discriminating. stable has MAD = 0 in
every generation (effective weight 0.013 — a tripwire only), the viability gate
(stable >= 0.70, err <= 8°) never binds, pos sits at its noise floor, and magnitude
patience watches only err + stable, so NEURONS stops at gen 8/15 and MEMORY at 6/120.

WHAT. A one-way latch (RefineLatch) switches the GA, mid-run, to a refinement fitness
(FitnessCalculatorControllerRefine): ZRank over steady / reward / near-fail tail / pos,
behind a TIGHTENED gate (stable, err, steady, pos) that is the regression guard; stable
leaves the weighted sum and lives only in the gate. The latch fires on TRAIN-POOL
metrics only — never val, never test — or at the start of a named stage.

Everything here is DEFAULT OFF: without --refine-fitness / --refine-select no object
below is built and every run is bit-identical to the pre-CTRL-58 code.
"""

from dataclasses import dataclass
from typing import Optional

REFINE_STAGES = ("neurons", "bits", "connections", "memory", "none")


@dataclass(frozen=True)
class RefineConfig:
	"""Every CTRL-58 knob, read once from the CLI (refine_config_from_args)."""
	# Trigger: >= trigger_frac of the TRAIN pool meets all four bounds for
	# trigger_gens consecutive generations, OR trigger_stage starts.
	trigger_frac: float = 0.80
	trigger_stable: float = 1.00       # fraction
	trigger_err: float = 1.5           # degrees
	trigger_steady: float = 1.0        # degrees
	trigger_pos: float = 0.40          # metres
	trigger_gens: int = 2
	trigger_stage: str = "memory"      # "none" = the population condition only
	# Tightened gate (the regression guard). Infeasible genomes rank behind every
	# feasible one, ordered by summed normalised violation (Deb's rules).
	gate_stable: float = 0.99
	gate_err: float = 1.5
	gate_steady: float = 1.0
	gate_pos: float = 0.40
	# Weighted sum (ZRank). stable is NOT here: it lives in the gate only.
	w_steady: float = 0.50
	w_reward: float = 0.25
	w_tail: float = 0.15
	w_pos: float = 0.10
	# Amendment 19/11 (ABI 60): the mid-window err column (0.3-1.6 s). 0 = column absent =
	# the pre-amendment calculator, bit-identical.
	w_mid: float = 0.0
	# Magnitude patience after the switch also watches steady (floor, degrees).
	patience_steady_floor: float = 0.15
	# The two master switches (independent).
	search: bool = False               # --refine-fitness
	select: bool = False               # --refine-select


def refine_config_from_args(args) -> Optional[RefineConfig]:
	"""The run's RefineConfig, or None when neither switch is set (the default)."""
	search = bool(getattr(args, "refine_fitness", False))
	select = bool(getattr(args, "refine_select", False))
	if not (search or select):
		return None
	return RefineConfig(
		trigger_frac=float(args.refine_trigger_frac), trigger_stable=float(args.refine_trigger_stable),
		trigger_err=float(args.refine_trigger_err), trigger_steady=float(args.refine_trigger_steady),
		trigger_pos=float(args.refine_trigger_pos), trigger_gens=int(args.refine_trigger_gens),
		trigger_stage=str(args.refine_trigger_stage),
		gate_stable=float(args.refine_gate_stable), gate_err=float(args.refine_gate_err),
		gate_steady=float(args.refine_gate_steady), gate_pos=float(args.refine_gate_pos),
		w_steady=float(args.refine_w_steady), w_reward=float(args.refine_w_reward),
		w_tail=float(args.refine_w_tail), w_pos=float(args.refine_w_pos),
		w_mid=float(getattr(args, "refine_w_mid", 0.0)),
		patience_steady_floor=float(args.refine_patience_steady_floor),
		search=search, select=select)


def validate_refine_args(args) -> None:
	"""Refuse an impossible or unsupported CTRL-58 configuration loudly, at launch."""
	cfg = refine_config_from_args(args)
	if cfg is None:
		return
	_validate_bounds(cfg)
	_validate_weights(cfg)
	if cfg.search:
		_refuse_search_combos(args, cfg)
	if cfg.select and getattr(args, "select_tail", "off") != "p99":
		raise SystemExit("--refine-select ranks on the refinement score + the p99 tail: pass "
		                 "--select-tail p99 with it (the tail rides the existing rank-sum path).")


def _validate_bounds(cfg: RefineConfig) -> None:
	if not (0.0 < cfg.trigger_frac <= 1.0):
		raise SystemExit(f"--refine-trigger-frac must be in (0, 1], got {cfg.trigger_frac}")
	for flag, v in (("--refine-trigger-stable", cfg.trigger_stable), ("--refine-gate-stable", cfg.gate_stable)):
		if not (0.0 < v <= 1.0):
			raise SystemExit(f"{flag} is a stable FRACTION in (0, 1], got {v}")
	for flag, v in (("--refine-trigger-err", cfg.trigger_err), ("--refine-trigger-steady", cfg.trigger_steady),
	                ("--refine-trigger-pos", cfg.trigger_pos), ("--refine-gate-err", cfg.gate_err),
	                ("--refine-gate-steady", cfg.gate_steady), ("--refine-gate-pos", cfg.gate_pos),
	                ("--refine-patience-steady-floor", cfg.patience_steady_floor)):
		if not (v > 0.0):
			raise SystemExit(f"{flag} must be > 0, got {v}")
	if cfg.trigger_gens < 1:
		raise SystemExit(f"--refine-trigger-gens must be >= 1, got {cfg.trigger_gens}")
	if cfg.trigger_stage not in REFINE_STAGES:
		raise SystemExit(f"--refine-trigger-stage must be one of {REFINE_STAGES}, got {cfg.trigger_stage!r}")


def _validate_weights(cfg: RefineConfig) -> None:
	ws = (cfg.w_steady, cfg.w_reward, cfg.w_tail, cfg.w_pos, cfg.w_mid)
	if any(w < 0.0 for w in ws) or sum(ws) <= 0.0:
		raise SystemExit(f"--refine-w-* must be >= 0 with a positive sum, got {ws}")


def _refuse_search_combos(args, cfg: RefineConfig) -> None:
	"""--refine-fitness paths that are NOT wired (refused, never silently ignored). ABI 59
	removed two refusals: --refine-w-tail > 0 (the scorers now export the train-pool tail)
	and --resume-from-emergency (the latch state is checkpointed and restored)."""
	if getattr(args, "strategy", "ga") == "ts":
		raise SystemExit("--refine-fitness is wired into the GA loop only; --strategy ts is refused.")
	for flag in ("axis_curriculum", "difficulty_curriculum", "difficulty_adaptive"):
		if getattr(args, flag, False):
			raise SystemExit(f"--refine-fitness is not wired into --{flag.replace('_', '-')} (refused).")


def refine_token(args) -> str:
	"""Provenance refine= value; "" when both switches are off (the line is then unchanged)."""
	cfg = refine_config_from_args(args)
	if cfg is None:
		return ""
	mode = "+".join(m for m, on in (("search", cfg.search), ("select", cfg.select)) if on)
	mid = f",mid{cfg.w_mid:g}" if cfg.w_mid > 0.0 else ""  # absent at 0: the pre-amendment token
	return (f"{mode}:trig={cfg.trigger_frac:g},s{cfg.trigger_stable:g},e{cfg.trigger_err:g},"
	        f"d{cfg.trigger_steady:g},p{cfg.trigger_pos:g},k{cfg.trigger_gens},{cfg.trigger_stage}"
	        f":gate={cfg.gate_stable:g},{cfg.gate_err:g},{cfg.gate_steady:g},{cfg.gate_pos:g}"
	        f":w=steady{cfg.w_steady:g},rw{cfg.w_reward:g},tail{cfg.w_tail:g},pos{cfg.w_pos:g}{mid}"
	        f":pfloor={cfg.patience_steady_floor:g}")


def refine_weights_record(args) -> Optional[dict]:
	"""The checkpoint/winner fitness_weights entry; None when off (payload unchanged)."""
	cfg = refine_config_from_args(args)
	if cfg is None:
		return None
	from dataclasses import asdict
	rec = asdict(cfg)
	if cfg.w_mid == 0.0:
		rec.pop("w_mid")  # absent at 0: the pre-amendment payload, byte for byte
	return rec


def mid_armed_from_args(args) -> bool:
	"""ABI 60: whether the run's evaluators report the mid-window column — only when the
	refinement SEARCH fitness weights it (--refine-fitness with --refine-w-mid > 0)."""
	cfg = refine_config_from_args(args)
	return cfg is not None and cfg.search and cfg.w_mid > 0.0
