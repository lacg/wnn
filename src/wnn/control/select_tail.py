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

p99 (10/11/2026, pre-registered in docs/ctrl_select_tail_prereg.md) replaces the
count with a PERCENTILE: per candidate, the 99th percentile of its per-episode mean
attitude error (the same row field near_fail and the 5 deg stable bound read) over
ALL its val episodes POOLED (val units x episodes), a diverged episode counting as
+inf. It is combined exactly like near_fail (the rank sum above).
  * Method: numpy.percentile(errs, 99, method="inverted_cdf") on the pooled array —
    Hyndman & Fan type 1, the empirical quantile: the smallest observed error e with
    F(e) >= 0.99, i.e. sorted[ceil(0.99 n) - 1]. On 2000 episodes (20 units x 100)
    that is the 21st-largest error, on 500 the 6th-largest.
  * Why not numpy's default "linear": it interpolates b - (b - a) * t, and with a
    diverged (+inf) neighbour that is inf - inf = NaN, so a candidate with diverged
    episodes would rank by NaN. An order statistic is inf-safe and always an episode
    that was actually flown.
  * A NON-finite error on a non-diverged row also counts as +inf (near_fail counts it
    as a near-miss too): an unmeasurable episode must not read as a clean one.

Selection only: the tail never enters the GA fitness, and is never measured on the
test seeds (the val draws are the only ones the sink is armed for).
"""

import math

import numpy as np

SELECT_TAIL_MODES = ("off", "near_fail", "p99")
# The pre-registered percentile (docs/ctrl_select_tail_prereg.md) and its numpy method.
P99_Q, P99_METHOD = 99.0, "inverted_cdf"
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
	"""Provenance select= value: 'off', 'near_fail:<deg>' or 'p99', plus ',val_units=N'
	when --val-units is not the default 5 (so a default run's line is unchanged)."""
	from wnn.control.val_units import val_units_suffix
	mode = select_tail_mode(args)
	tok = mode if mode in ("off", "p99") else f"{mode}:{float(getattr(args, 'select_tail_deg', 4.0)):g}"
	return tok + val_units_suffix(args)


def new_episode_sink(args):
	"""A fresh per-episode sink for one val scoring call; None when the tail is off
	(None leaves the scorer call byte-identical to the pre-tail one)."""
	return [] if select_tail_mode(args) != "off" else None


def genome0_rows(sink: list, label: str) -> list:
	"""The per-episode rows of genome 0 in one val call's sink. Refuses an empty sink:
	the rows exist only on the GPU scorer, and a silent nothing would read as a
	perfectly clean candidate."""
	rows = [r for entry in sink for r in entry["episodes"] if int(r[_ROW_GENOME]) == 0]
	if not rows:
		raise RuntimeError(
			f"--select-tail: no per-episode rows for {label} — the per-episode export is "
			"GPU-scorer only (score_controllers_metal); unset WNN_CONTROLLER_GPU_EVAL=0 or "
			"run with --select-tail off.")
	return rows


def near_fail_count(rows: list, deg: float) -> int:
	"""Episodes that diverged or whose mean error exceeds `deg` (non-finite counts)."""
	bound = math.radians(deg)
	return sum(1 for r in rows if r[_ROW_DIVERGED] != 0.0 or not (r[_ROW_ERR] <= bound))


def tail_count(sink: list, deg: float, label: str) -> int:
	"""near_fail over one val call's sink (genome 0; an empty sink is refused)."""
	return near_fail_count(genome0_rows(sink, label), deg)


def episode_errors_deg(rows: list) -> np.ndarray:
	"""Per-episode mean attitude error (deg); diverged or non-finite -> +inf."""
	errs = np.array([float(r[_ROW_ERR]) for r in rows], dtype=np.float64)
	dead = np.array([float(r[_ROW_DIVERGED]) != 0.0 for r in rows], dtype=bool)
	errs = np.degrees(errs)
	errs[dead | ~np.isfinite(errs)] = np.inf
	return errs


def p99_deg(rows: list) -> float:
	"""The pre-registered tail: 99th percentile (inverted_cdf) of the pooled errors."""
	if not rows:
		raise RuntimeError("--select-tail p99: no per-episode rows to take a percentile of")
	return float(np.percentile(episode_errors_deg(rows), P99_Q, method=P99_METHOD))


def tail_value(args, rows: list) -> float:
	"""The candidate's tail term over ALL its pooled val rows (lower = better)."""
	if select_tail_mode(args) == "p99":
		return p99_deg(rows)
	return near_fail_count(rows, float(getattr(args, "select_tail_deg", 4.0)))


def tail_cell(args, value) -> str:
	"""The stage-table cell for one candidate's tail term."""
	if select_tail_mode(args) == "p99":
		return "p99=%.2f°" % value
	return "tail=%d" % value


def tail_footer(args) -> str:
	"""The stage-table footer naming the tail rule the headline was picked under."""
	if select_tail_mode(args) == "p99":
		tail = (f"tail = p{P99_Q:g} (numpy {P99_METHOD}) of per-episode err over ALL pooled val "
		        "episodes, diverged = +inf")
	else:
		tail = (f"tail = val episodes diverged or err > "
		        f"{float(getattr(args, 'select_tail_deg', 4.0)):g}°")
	return (f"  (--select-tail {select_tail_token(args)}: {tail}; "
	        "HEADLINE = min R = rank(fit) + rank(tail), ties -> lower fit)")


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
