"""Stage-select val units (`--val-units N`, pre-registered in docs/ctrl_select_tail_prereg.md).

WHAT. Stage-select flies every headline candidate on N val UNITS; unit i is the seed
`seeds.val + i` (seeds.val from the seed registry, wnn.seeds.resolve_seed_set). N=5 is
the default and is exactly the pre-flag list, so every existing run re-ranks
identically; N=20 extends the SAME list (its first 5 units are today's 5).

DISJOINTNESS (asserted at launch and in --recalc-headline, never assumed). A val unit
must not be a report unit (--report-seeds / --report-seed) nor the train or test seed.
Checked on two levels:
  1. the raw seeds;
  2. the episode-pool seeds the scorer actually samples from — the raw seed (K=1) and
     wnn.control.evaluator.fold_pool_seed(seed, k) for k < --num-eval-folds — over the
     val units, the report units and the train/test seeds. A unit's initial conditions
     AND its weather / motor-wear draw both derive from that pool seed, so (2) is the
     statement that no val episode is a report or training episode. fold_pool_seed is
     a bijection on 32-bit seeds for a fixed k (odd multiplier mod 2^32), so (2) also
     catches a val seed that only collides after the uint32 wrap.
"""

DEFAULT_VAL_UNITS = 5


def val_units_count(args) -> int:
	return int(getattr(args, "val_units", DEFAULT_VAL_UNITS) or DEFAULT_VAL_UNITS)


def validate_val_units(args) -> None:
	"""Refuse N < 1 loudly, at launch."""
	n = int(getattr(args, "val_units", DEFAULT_VAL_UNITS))
	if n < 1:
		raise SystemExit(f"--val-units {n}: stage-select needs at least one val unit")


def val_units_suffix(args) -> str:
	"""Provenance select= suffix: '' at the default (line unchanged), else ',val_units=N'."""
	n = val_units_count(args)
	return "" if n == DEFAULT_VAL_UNITS else f",val_units={n}"


def val_unit_seeds(val_seed: int, n: int) -> list:
	"""Unit i = val_seed + i — the pre-flag derivation, extended to n units."""
	return [int(val_seed) + i for i in range(int(n))]


def _report_units(args) -> list:
	rs = getattr(args, "report_seeds", None)
	if rs:
		return [int(s) for s in rs]
	one = getattr(args, "report_seed", None)
	return [] if one is None else [int(one)]


def _pools(seeds, folds: int) -> set:
	"""Every seed an episode can be drawn from: the raw seed (K=1) and its K fold pools."""
	from wnn.control.evaluator import fold_pool_seed
	pools = {int(s) & 0xFFFFFFFF for s in seeds}
	return pools | {fold_pool_seed(int(s), k) for s in seeds for k in range(folds)}


def _refuse_overlap(what: str, val: set, other: set, other_name: str) -> None:
	both = sorted(val & other)
	if both:
		raise SystemExit(f"--val-units: {len(both)} val {what} also {other_name} "
		                 f"({both[:5]}{' ...' if len(both) > 5 else ''}) — val must be disjoint")


def checked_val_unit_seeds(args, seeds) -> list:
	"""The stage-select val units of this run, after the disjointness checks above."""
	units = val_unit_seeds(seeds.val, val_units_count(args))
	if len(set(units)) != len(units):
		raise SystemExit(f"--val-units: duplicate val units {units}")
	report, train_test = _report_units(args), [int(seeds.train), int(seeds.test)]
	_refuse_overlap("seeds are", set(units), set(report), "report units")
	_refuse_overlap("seeds are", set(units), set(train_test), "the train/test seed")
	folds = max(1, int(getattr(args, "num_eval_folds", 5)))
	vp = _pools(units, folds)
	_refuse_overlap("pool seeds are", vp, _pools(report, folds), "report pool seeds")
	_refuse_overlap("pool seeds are", vp, _pools(train_test, folds), "train/test pool seeds")
	return units


def val_units_line(args, seeds) -> str:
	"""One greppable launch line stating the units and that they were checked."""
	units = checked_val_unit_seeds(args, seeds)
	folds = max(1, int(getattr(args, "num_eval_folds", 5)))
	return (f"[val-units] {len(units)} stage-select val units {units[0]}..{units[-1]} — "
	        f"disjoint from {len(_report_units(args))} report units and train={seeds.train}/"
	        f"test={seeds.test} (raw seeds + K={folds} fold pool seeds) — checked")
