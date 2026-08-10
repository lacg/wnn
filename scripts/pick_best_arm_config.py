"""Pick the winning configuration from a completed arm's markers, or report none.

WHY THIS EXISTS (10/08/2026). A chained arm often needs an input that is only known
once the PREVIOUS arm lands — E1 needs the outer-q arm's winning quantile, E2 needs
E1's winning encoder. Hardcoding it at arm time pre-judges the result; hand-arming
between chains means a human has to be awake at the moment a 6-hour arm finishes.
This resolves the input from the markers instead, applying the SAME pre-registered
rule a human would: a candidate wins only if it beats the control on EVERY seed on
headline steady WITHOUT losing stable.

The rule is deliberately strict and the "no winner" answer is a first-class outcome,
printed as an empty string. A caller that gets no winner must fall back to the control
configuration — never to "the least-bad candidate". Best-of-N on a refuted arm is how
a failed lever gets promoted into a paper.

Usage:
    python scripts/pick_best_arm_config.py <markdir> <config-field> <control-value>
        [--stable-tolerance 0.0]

Prints the winning config value on stdout (empty if none), and the full comparison
table on stderr so the chain log carries the reasoning, not just the verdict.
Exit 0 always when the markers parse — "no winner" is a result, not an error.
"""
import argparse
import json
import re
import sys
from pathlib import Path

# " [stage-select] HEADLINE held-out: stable=99.6% err=1.23° steady=0.69°"
_TRIPLE = re.compile(r"stable=([\d.]+)%\s+err=([\d.]+)\S*\s+steady=([\d.]+)")


def _headline(marker: dict) -> tuple[float, float, float] | None:
	"""(stable%, err°, steady°) from a marker's headline block, or None.

	Reads `headline_holdout` — the stage-selected genome's own report-seed triple.
	Deliberately NOT the per-stage multiseed rows: stage selection routinely picks
	different stages on different seeds, so a fixed-stage read compares two different
	objects (the outer-q controls headlined NEURONS#0 and MEMORY#1 respectively).
	"""
	m = _TRIPLE.search(marker.get("headline_holdout") or "")
	return (float(m.group(1)), float(m.group(2)), float(m.group(3))) if m else None


def _load(markdir: Path, field: str) -> dict[tuple[str, int], tuple[float, float, float]]:
	"""Map (config-value, seed) -> headline triple for every complete marker."""
	out: dict[tuple[str, int], tuple[float, float, float]] = {}
	for path in sorted(markdir.glob("*.json")):
		try:
			d = json.loads(path.read_text())
		except (json.JSONDecodeError, OSError) as e:
			print(f"  SKIP {path.name}: unreadable ({e})", file=sys.stderr)
			continue
		if d.get("rc") != 0:
			print(f"  SKIP {path.name}: rc={d.get('rc')}", file=sys.stderr)
			continue
		triple = _headline(d)
		if triple is None or field not in d or "seed" not in d:
			print(f"  SKIP {path.name}: no headline triple or missing {field}/seed",
			      file=sys.stderr)
			continue
		out[(str(d[field]), int(d["seed"]))] = triple
	return out


def _beats(cand: tuple[float, float, float], ctrl: tuple[float, float, float],
           stable_tol: float) -> bool:
	"""Pre-registered bar: lower steady AND stable not worse (within tolerance)."""
	return cand[2] < ctrl[2] and cand[0] >= ctrl[0] - stable_tol


def main() -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("markdir")
	ap.add_argument("field", help="marker field holding the config value, e.g. outer_q")
	ap.add_argument("control", help="the control value of that field, e.g. none")
	ap.add_argument("--stable-tolerance", type=float, default=0.0,
	                help="percentage points of stable a candidate may give up (default 0)")
	args = ap.parse_args()

	markdir = Path(args.markdir)
	if not markdir.is_dir():
		print(f"NO MARKER DIR {markdir} — no winner", file=sys.stderr)
		print("")
		return 0

	table = _load(markdir, args.field)
	seeds = sorted({s for _, s in table})
	ctrl_seeds = [s for s in seeds if (args.control, s) in table]
	if not ctrl_seeds:
		print(f"NO CONTROL cells ({args.field}={args.control}) — cannot judge, no winner",
		      file=sys.stderr)
		print("")
		return 0

	candidates = sorted({v for v, _ in table} - {args.control})
	print(f"  control {args.field}={args.control} on seeds {ctrl_seeds}:", file=sys.stderr)
	for s in ctrl_seeds:
		st, er, sd = table[(args.control, s)]
		print(f"    s{s}  {st:.1f}% / {er:.2f}° / {sd:.2f}°", file=sys.stderr)

	winner, best_margin = "", 0.0
	for cand in candidates:
		# A candidate must be measured on EVERY seed the control was measured on.
		# Judging on a subset is how a lever passes by only being tried where it works.
		missing = [s for s in ctrl_seeds if (cand, s) not in table]
		if missing:
			print(f"  {args.field}={cand}: INCOMPLETE (missing seeds {missing}) — not eligible",
			      file=sys.stderr)
			continue
		margins, passed = [], True
		for s in ctrl_seeds:
			c, k = table[(cand, s)], table[(args.control, s)]
			ok = _beats(c, k, args.stable_tolerance)
			margins.append(k[2] - c[2])
			print(f"  {args.field}={cand} s{s}: {c[0]:.1f}% / {c[1]:.2f}° / {c[2]:.2f}°"
			      f"  vs ctrl steady {k[2]:.2f}°  Δ{k[2] - c[2]:+.2f}  {'PASS' if ok else 'FAIL'}",
			      file=sys.stderr)
			passed = passed and ok
		mean_margin = sum(margins) / len(margins)
		if passed and mean_margin > best_margin:
			winner, best_margin = cand, mean_margin
		print(f"  {args.field}={cand}: {'BEATS' if passed else 'does NOT beat'} the control "
		      f"on all {len(ctrl_seeds)} seeds (mean Δsteady {mean_margin:+.2f}°)",
		      file=sys.stderr)

	if winner:
		print(f"WINNER {args.field}={winner} (mean Δsteady {best_margin:+.2f}°)", file=sys.stderr)
	else:
		print(f"NO WINNER — every candidate failed the bar. Caller must fall back to "
		      f"{args.field}={args.control}, NOT to the least-bad candidate.", file=sys.stderr)
	print(winner)
	return 0


if __name__ == "__main__":
	sys.exit(main())
