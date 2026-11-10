"""Tail-aware stage-select (`--select-tail near_fail`; docs/ctrl_dres_failures.md idea #3).

WHY. Stage-select ranks the union of every stage's top-K on 5 val seeds with the run's
own fitness calculator, which reads per-genome MEANS (stable%, err, ...). A genome that
loses 1 episode in 1000 is invisible there: on 5 x 100 val episodes its stable% moves
by 0-0.2 pp, below the noise between candidates. The rf_c200_dres held-out fails were
exactly that (3 of 3500 on one genome, 0 on its sibling), and each was preceded by
near-misses: per-episode mean error above ~4 deg (docs §"Population check").

WHAT. near_fail adds a TAIL term: per candidate, the count of val episodes that
diverged or whose per-episode mean attitude error exceeds X deg (--select-tail-deg,
default 4.0; the same per-episode error the 5 deg stable bound is applied to). The
count comes from the GPU scorer's per-episode rows (score_with_episodes, the
ControllerEvaluator.per_episode_sink added 14/09/2026) — nothing is recomputed here.

HOW IT COMBINES: an added rank term (Borda), NOT lexicographic.
    combined = rank(fit) + rank(tail)      (average ranks, lower = better)
    winner   = argmin combined; ties -> lower fit, then candidate order.
  * Only RANKS can be added: the fit value's scale depends on --fit-aggregation
    (harmonic / arithmetic / z-score / desirability), a count has none of those.
  * Lexicographic (fewest near-fails first) would let ONE near-miss on 500 val
    episodes — a Poisson count whose mean is ~1 for the worst genomes measured —
    override any fitness gap. The rank sum lets a candidate with a clearly better fit
    survive a single near-miss, while among near-tied fits (stages land within
    ~0.05 deg of each other) the clean candidate wins.
  * Invariant: when every candidate has the same tail count (the common case — most
    have 0), rank(tail) is a constant and the winner is EXACTLY today's winner.

Selection only: the tail never enters the GA fitness, and is never measured on the
report seeds (the val draws are the only ones the sink is armed for).
"""

import math

SELECT_TAIL_MODES = ("off", "near_fail")
# The scorer's stable bound (metal_controller.rs / cpu_score.rs stable_thresh): an
# episode is a FAIL above it. A tail bound at or above it would count fails only,
# which stable% already ranks.
STABLE_BOUND_DEG = 5.0
# Per-episode row layout of score_with_episodes (metal_controller.rs):
# [genome_idx, episode_idx, stable, diverged, mean_err_rad, ...]
_ROW_GENOME, _ROW_DIVERGED, _ROW_ERR = 0, 3, 4


def select_tail_mode(args) -> str:
	return getattr(args, "select_tail", "off") or "off"


def validate_select_tail(args) -> None:
	"""Refuse an unknown mode or a bound outside (0, 5) deg — loudly, at launch."""
	mode = select_tail_mode(args)
	if mode not in SELECT_TAIL_MODES:
		raise SystemExit(f"--select-tail {mode!r}: choices are {SELECT_TAIL_MODES}")
	deg = float(getattr(args, "select_tail_deg", 4.0))
	if not (math.isfinite(deg) and 0.0 < deg < STABLE_BOUND_DEG):
		raise SystemExit(f"--select-tail-deg {deg:g} must be in (0, {STABLE_BOUND_DEG:g}) deg: at or "
		                 "above the stable bound it counts only fails, which stable% already ranks.")


def select_tail_token(args) -> str:
	"""Provenance select= value: 'off' or 'near_fail:<deg>'."""
	mode = select_tail_mode(args)
	return "off" if mode == "off" else f"{mode}:{float(getattr(args, 'select_tail_deg', 4.0)):g}"


def new_episode_sink(args):
	"""A fresh per-episode sink for one val scoring call; None when the tail is off
	(None leaves the scorer call byte-identical to the pre-tail one)."""
	return [] if select_tail_mode(args) != "off" else None


def tail_count(sink: list, deg: float, label: str) -> int:
	"""Episodes of genome 0 that diverged or whose mean error exceeds `deg`. Refuses an
	empty sink: the per-episode rows exist only on the GPU scorer, and a silent 0 would
	read as a perfectly clean candidate."""
	rows = [r for entry in sink for r in entry["episodes"] if int(r[_ROW_GENOME]) == 0]
	if not rows:
		raise RuntimeError(
			f"--select-tail: no per-episode rows for {label} — the per-episode export is "
			"GPU-scorer only (score_controllers_metal); unset WNN_CONTROLLER_GPU_EVAL=0 or "
			"run with --select-tail off.")
	bound = math.radians(deg)
	return sum(1 for r in rows if r[_ROW_DIVERGED] != 0.0 or not (r[_ROW_ERR] <= bound))


def average_ranks(values: list) -> list:
	"""1-based ranks, ascending (lower value = rank 1), ties sharing their mean rank."""
	order = sorted(range(len(values)), key=lambda i: values[i])
	ranks = [0.0] * len(values)
	i = 0
	while i < len(order):
		j = i
		while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
			j += 1
		for k in range(i, j + 1):
			ranks[order[k]] = (i + j) / 2.0 + 1.0
		i = j + 1
	return ranks


def tail_rank_winner(labels: list, fits: dict, tails: dict) -> tuple:
	"""(winner, {label: combined}) under combined = rank(fit) + rank(tail); see the
	module note. Ties go to the lower fit, then to the earlier candidate."""
	r_fit = average_ranks([fits[l] for l in labels])
	r_tail = average_ranks([tails[l] for l in labels])
	combined = {l: r_fit[i] + r_tail[i] for i, l in enumerate(labels)}
	winner = min(labels, key=lambda l: (combined[l], fits[l], labels.index(l)))
	return winner, combined
