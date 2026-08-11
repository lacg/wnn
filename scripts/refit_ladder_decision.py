"""Apply the pre-registered refit-ladder rule to the E1 markers and print the verdict.

THE RULE (pre-registered 10/08/2026, docs/l4_teacher_screen_results.md "E1b"). The
decision metric is HEADLINE steady — the programme's standing comparison surface and
the genome that actually ships. Decide on the 95% CI of the paired Δ, NOT a win count:

    CI entirely below 0                  -> PROMOTE   (real improvement)
    CI entirely above 0                  -> REFUTE    (real harm)
    CI spans 0, half-width <= 0.15 deg   -> STOP      (genuine null)
    CI spans 0, half-width  > 0.15 deg   -> ESCALATE  (next rung: 5 -> 7 -> 10)

WHY THIS EXISTS AS A SCRIPT. The n=7 rung has to be armed BEFORE n=5 finishes (the box
runs one controller at a time and a chain waits on a PID), but arming it
unconditionally would pre-commit to escalation regardless of what n=5 says — the same
optional-stopping error the rule was written to avoid. So the rung gates itself on this
script's verdict instead.

The unanimity bar this replaced ("beat the control on EVERY seed") got STRICTER as N
grew, so more evidence made a real effect harder to demonstrate. See the doc.

Usage:
    python scripts/refit_ladder_decision.py [markdir]
Prints the per-seed table and stats to stderr, and one verdict word to stdout:
PROMOTE | REFUTE | STOP | ESCALATE | INSUFFICIENT
"""
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pick_best_arm_config import _headline  # single source of truth for the triple regex

NULL_HALF_WIDTH = 0.15   # deg; "genuine null" band, pre-registered
# t(0.975) keyed by DEGREES OF FREEDOM (= n-1), not by n. Keyed by df deliberately:
# the first version of this table was keyed by n and then indexed with n-1, which
# silently used the t for df=n-2 and inflated every interval (n=4 read 0.73 instead
# of 0.54). The verdict happened to survive; the number would have gone into a paper.
T95_BY_DF = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
             7: 2.365, 8: 2.306, 9: 2.262}


def _pairs(markdir: Path) -> dict[int, dict[str, float]]:
	"""seed -> {refit_level: headline steady}, from complete (rc=0) markers only."""
	out: dict[int, dict[str, float]] = {}
	for path in sorted(markdir.glob("*.json")):
		try:
			d = json.loads(path.read_text())
		except (json.JSONDecodeError, OSError):
			continue
		if d.get("rc") != 0 or "refit" not in d or "seed" not in d:
			continue
		# The 2x2 degenerated to c30-only, but guard anyway: mixing encoders would
		# pair a refit cell against a control fitted on a different ladder.
		if d.get("enc") not in (None, "c30"):
			continue
		triple = _headline(d)
		if triple is not None:
			out.setdefault(int(d["seed"]), {})[str(d["refit"])] = triple[2]
	return out


def main() -> int:
	markdir = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/e1_coverage_markers")
	pairs = _pairs(markdir)
	complete = {s: v for s, v in pairs.items() if "on" in v and "off" in v}

	print(f"  paired seeds: {len(complete)} (of {len(pairs)} seeds with any marker)",
	      file=sys.stderr)
	deltas = []
	for seed in sorted(complete):
		off, on = complete[seed]["off"], complete[seed]["on"]
		d = on - off                      # positive = refit WORSE
		deltas.append(d)
		print(f"    s{seed}  ctrl {off:.2f}  refit {on:.2f}   Δ {d:+.2f}", file=sys.stderr)

	if len(deltas) < 3:
		print(f"  INSUFFICIENT: {len(deltas)} pairs, need >= 3 for a CI", file=sys.stderr)
		print("INSUFFICIENT")
		return 0

	n = len(deltas)
	mean, sd = st.mean(deltas), st.stdev(deltas)
	half = T95_BY_DF.get(n - 1, 1.96) * sd / n ** 0.5
	lo, hi = mean - half, mean + half
	print(f"  n={n}  mean Δ {mean:+.3f}  SD {sd:.3f}  95% CI [{lo:+.2f}, {hi:+.2f}] "
	      f"(half-width {half:.2f})", file=sys.stderr)

	if hi < 0:
		verdict = "PROMOTE"
	elif lo > 0:
		verdict = "REFUTE"
	elif half <= NULL_HALF_WIDTH:
		verdict = "STOP"
	else:
		verdict = "ESCALATE"
	print(f"  VERDICT: {verdict}", file=sys.stderr)
	print(verdict)
	return 0


if __name__ == "__main__":
	sys.exit(main())
