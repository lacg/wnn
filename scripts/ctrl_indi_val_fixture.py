"""INDI spec §10 (docs/ctrl_indi_teacher_spec.md): write the VAL-unit fixture the
`rival_probe_tests::val_sweep` cargo test scores.

One text file per val unit, holding the EXACT arguments the shipped samplers hand
`score_classical_baseline` for the MPCOF>rate(h<action_repeat>) rival row — built by
the same calls `classical_baseline.teacher_metrics` makes, so nothing is re-derived
here and nothing is re-sampled in Rust. One `key v v v ...` line per argument.

Units: the run's stage-select val units through `checked_val_unit_seeds` (disjointness
from the test units and the train/test pool seeds asserted). Any seed in
99990101..99990400 (the report-only units) is refused.

The first unit also carries `expect_est_row`: what the INSTALLED wheel returns for the
est-fed row, so the cargo test can prove its build and this fixture reproduce the
shipped rival (one unit, a few seconds of CPU; nothing else is scored here).

Usage:
  python scripts/ctrl_indi_val_fixture.py --run-log logs/controller/ctrl48/<tag>.out --out /tmp/ctrl_indi_probe/fixture
"""
import argparse
import inspect
import shlex
from pathlib import Path

TEST_ONLY = range(99990101, 99990401)
MPCOF = 4


def _run_args(run_log: str):
	from wnn.control import phased_ga as P
	text = open(run_log, errors="replace").read()
	argv = shlex.split([l for l in text.splitlines() if l.startswith("[argv]")][0][len("[argv]"):])
	a = P.build_arg_parser().parse_args(argv)
	ec = P.episode_config_from_args(a)
	P._resolve_ctbr(a, ec)
	return a, ec


def _val_units(a) -> list:
	from wnn.control.val_units import checked_val_unit_seeds
	from wnn.seeds import resolve_seed_set
	s = resolve_seed_set(base=a.base_seed, run_index=0, train=a.train_seed, test=a.test_seed, val=a.val_seed)
	units = checked_val_unit_seeds(a, s)
	bad = [u for u in units if u in TEST_ONLY]
	if bad:
		raise SystemExit(f"val units {bad} fall in the report-only range — refused")
	return units


def _fields(a, ec, unit: int) -> dict:
	"""Every keyword of the rival row, as classical_baseline.teacher_metrics builds it."""
	from wnn.control import classical_baseline as CB
	draw = CB.HoldoutDraw(seed=unit, episodes=a.test_episodes, steps=ec.steps_per_episode,
	                      eval_folds=a.num_eval_folds)
	q0, w0, f = CB._episode_fields(ec, draw)
	f = {**f, **ec.airframe_kwargs(), **ec.motor_lag_kwargs(), **CB.TeacherFeed(use_estimator=True).fields(),
	     **CB._stage1_fields(ec, draw), **CB._abi32_fields(ec), "ctbr_outer_hold": int(a.action_repeat)}
	return {"init_qs": list(q0), "init_omegas": list(w0), "steps": draw.steps, "stable_deg": draw.stable_deg, **f}


def _with_defaults(fields: dict, scorer) -> dict:
	"""Fill the arguments the row leaves at the scorer's own default, read from the
	wheel's signature (never copied by hand)."""
	out = dict(fields)
	for name, p in inspect.signature(scorer).parameters.items():
		if name in out or name == "teacher_id":
			continue
		if p.default is inspect.Parameter.empty or p.default is Ellipsis:
			raise SystemExit(f"argument {name} is neither supplied by the row nor a plain default")
		out[name] = p.default
	return out


def _line(key: str, v) -> str:
	vals = v if isinstance(v, (list, tuple)) else [v]
	if key == "dist_seed":
		return f"{key} {int(vals[0])}"
	return key + " " + " ".join(repr(float(x)) for x in vals)


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--run-log", required=True)
	ap.add_argument("--out", required=True)
	ap.add_argument("--units", choices=["val", "test"], default="val",
	                help="val (default): the run's stage-select val units. test: the run's --test-seeds — "
	                     "REPORT-ONLY, for a one-shot confirmation Luiz approved (spec §12); never for tuning.")
	args = ap.parse_args()
	from wnn.control._accel import score_classical_baseline
	a, ec = _run_args(args.run_log)
	units = _val_units(a) if args.units == "val" else [int(x) for x in a.test_seeds]
	out = Path(args.out)
	out.mkdir(parents=True, exist_ok=True)
	for i, unit in enumerate(units):
		f = _with_defaults(_fields(a, ec, unit), score_classical_baseline)
		lines = [_line("unit_seed", unit)] + [_line(k, v) for k, v in f.items()]
		if i == 0:
			call = {k: v for k, v in f.items() if k not in ("init_qs", "init_omegas", "steps", "stable_deg")}
			r = score_classical_baseline(MPCOF, f["init_qs"], f["init_omegas"], f["steps"], f["stable_deg"], **call)
			lines.append(_line("expect_est_row", [r[0], r[1], r[2], r[4]]))
			print(f"[fixture] unit {unit} installed-wheel est-fed row: stable {r[0]:.4f} err {r[1]:.4f} steady {r[2]:.4f} pos {r[4]:.4f}")
		(out / f"unit_{unit}.txt").write_text("\n".join(lines) + "\n")
	print(f"[fixture] {len(units)} {args.units.upper()} units {units[0]}..{units[-1]} x {a.test_episodes} episodes -> {out}")


if __name__ == "__main__":
	main()
