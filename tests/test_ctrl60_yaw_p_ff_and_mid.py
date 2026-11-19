"""ABI 60 (Luiz 19/11/2026): --yaw-p-ff (docs/ctrl59_yaw_pterm_spec.md) and the mid-window
scorer row / --refine-w-mid (docs/ctrl58_a2_prereg.md, Amendment 19/11) — the Python side.

Pinned:
  * every source file that mentions `yaw_trim_ff` also mentions `yaw_p_ff`, and every Python
    file forwards it at the SAME number of sites (the "--fit-weight-alt was a no-op at 4 of 5
    sites" class of bug);
  * the flag parses (absent = None, bare = 1.0, a value), is refused without --yaw-trim-ff /
    --obs-yaw-err or with a bad scale, and the provenance token gains `+pff<scale>`;
  * an ABI 59 ControllerSpec dict (no yaw_p_ff) loads as None, and yaw_p_ff_kw of a spec
    object pickled before the field existed is None;
  * --refine-w-mid: default 0 = the pre-amendment token, payload and calculator; > 0 adds the
    cost column (changes the ranking when only mean_mid_error_deg differs), refuses a missing
    value, stays out of the stage-select calculator, and is in the params registry.
Needs the installed ram_controller at ABI 60 (wnn.control imports the facade; the ranking
test calls fitness_combine). No GPU, no sim.
"""
import re
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from wnn.control.evaluator import ControllerSpec, yaw_p_ff_kw
from wnn.control.refine_config import (RefineConfig, mid_armed_from_args, refine_token,
                                      refine_weights_record, validate_refine_args)
from wnn.control.refine_fitness import FitnessCalculatorControllerRefine
from wnn.ram.metrics import ControllerMetrics

SRC = Path(__file__).resolve().parents[1] / "src" / "wnn"


def _args(*argv):
	from wnn.control.phased_ga import build_arg_parser
	return build_arg_parser().parse_args(list(argv))


# ---- --yaw-p-ff: forwarding sites ------------------------------------------------------

def _sources():
	files = list((SRC / "control").glob("*.py")) + [SRC / "ram" / "experiments" / "params.py"]
	files += [p for p in (SRC / "ram" / "strategies" / "accelerator" / "controller").rglob("*")
	          if p.suffix in (".rs", ".metal") and p.name != "ctrl58_yaw_trim_ff_tests.rs"]
	return files


def test_every_file_that_mentions_yaw_trim_ff_mentions_yaw_p_ff():
	missing = [str(p.relative_to(SRC)) for p in _sources()
	           if "yaw_trim_ff" in p.read_text() and "yaw_p_ff" not in p.read_text()]
	assert not missing, f"yaw_trim_ff is forwarded but yaw_p_ff is not: {missing}"


def test_python_forwarding_site_counts_match():
	"""`yaw_trim_ff=<...>` keyword sites vs `yaw_p_ff=<...>` keyword sites, file by file."""
	seen = 0
	for p in (SRC / "control").glob("*.py"):
		text = p.read_text()
		trim = len(re.findall(r"\byaw_trim_ff=", text))
		pff = len(re.findall(r"\byaw_p_ff=", text))
		assert trim == pff, f"{p.name}: {trim} yaw_trim_ff= sites vs {pff} yaw_p_ff= sites"
		seen += trim
	assert seen >= 10, f"vacuous: only {seen} forwarding sites found"


def test_metal_kernel_carries_the_p_term_at_every_yff_site():
	shader = (SRC / "ram/strategies/accelerator/controller/shaders/controller_rollout.metal").read_text()
	lines = shader.splitlines()
	sites = [i for i, l in enumerate(lines) if "P.yff_ct" in l]
	assert sites, "vacuous: no yaw FF decode site"
	for i in sites:
		near = "\n".join(lines[max(0, i - 8):i + 8])
		assert "P.ypf_on" in near, f"line {i + 1}: a yaw-FF decode site without the P-term next to it"


# ---- --yaw-p-ff: the flag ---------------------------------------------------------------

BASE = ["--output-mode", "ctbr", "--obs-cascade-dhat-residual", "--obs-yaw-err", "--yaw-trim-ff"]


def test_flag_parses_absent_bare_and_valued():
	assert _args().yaw_p_ff is None
	assert _args("--yaw-p-ff").yaw_p_ff == 1.0
	assert _args("--yaw-p-ff", "0.5").yaw_p_ff == 0.5
	# A bare flag followed by another option must not swallow it.
	a = _args("--yaw-p-ff", "--yaw-trim-ff")
	assert a.yaw_p_ff == 1.0 and a.yaw_trim_ff


def test_flag_requirements_and_scale_range():
	from wnn.control.phased_ga import resolve_block10_args
	resolve_block10_args(_args(*BASE, "--yaw-p-ff"))
	resolve_block10_args(_args(*BASE, "--yaw-p-ff", "2"))
	resolve_block10_args(_args(*BASE))                       # off: never refuses
	no_ff = [a for a in BASE if a != "--yaw-trim-ff"]
	with pytest.raises(SystemExit, match="--yaw-trim-ff"):
		resolve_block10_args(_args(*no_ff, "--yaw-p-ff"))
	no_err = [a for a in BASE if a != "--obs-yaw-err"]
	with pytest.raises(SystemExit, match="--obs-yaw-err"):
		resolve_block10_args(_args(*no_err, "--yaw-p-ff"))
	for bad in ("0", "-1", "2.5", "nan", "inf"):
		with pytest.raises(SystemExit, match="SCALE"):
			resolve_block10_args(_args(*BASE, "--yaw-p-ff", bad))
	# L2y / L2y' stay refused through the trim FF's own rule.
	with pytest.raises(SystemExit, match="alternatives"):
		resolve_block10_args(_args(*BASE, "--yaw-p-ff", "--obs-yaw-trim-err-p"))


def test_provenance_token():
	from wnn.control.phased_ga import block10_tokens
	assert block10_tokens(_args(*BASE))["yawref"] == "ff"
	assert block10_tokens(_args(*BASE, "--yaw-p-ff"))["yawref"] == "ff+pff1"
	assert block10_tokens(_args(*BASE, "--yaw-p-ff", "0.5"))["yawref"] == "ff+pff0.5"
	assert block10_tokens(_args(*BASE, "--obs-yaw-from-att-est", "--yaw-p-ff"))["yawref"] == "mahony+ff+pff1"
	assert block10_tokens(_args())["yawref"] == ""


def test_make_spec_and_grid_search_forward_the_scale():
	import inspect
	from wnn.control import controller_grid_search, phased_ga
	sig = inspect.signature(phased_ga._make_spec)
	assert sig.parameters["yaw_p_ff"].default is None
	assert 'yaw_p_ff=getattr(args, "yaw_p_ff", None)' in inspect.getsource(controller_grid_search)


def test_abi59_spec_loads_with_the_p_term_off():
	from wnn.control.checkpoint_io import _spec_from_plain, _spec_to_plain
	fields = {f for f in ControllerSpec.__dataclass_fields__}
	assert "yaw_p_ff" in fields and ControllerSpec.__dataclass_fields__["yaw_p_ff"].default is None
	spec = ControllerSpec(yaw_trim_ff=True, obs_yaw_err=True)
	old = _spec_to_plain(spec)
	old.pop("yaw_p_ff")                                      # an ABI 59 winner's spec dict
	loaded = _spec_from_plain(old)
	assert loaded.yaw_p_ff is None and yaw_p_ff_kw(loaded) is None and loaded == spec
	# A spec OBJECT from before the field existed (legacy pickle path): no attribute at all.
	legacy = type("LegacySpec", (), {"yaw_trim_ff": True})()
	assert yaw_p_ff_kw(legacy) is None
	assert yaw_p_ff_kw(replace(spec, yaw_p_ff=0.5)) == 0.5
	assert _spec_from_plain(_spec_to_plain(replace(spec, yaw_p_ff=0.5))).yaw_p_ff == 0.5


def test_params_registry_and_facade_abi():
	from wnn.control import _accel
	from wnn.ram.experiments.params import KNOWN_PARAMS
	assert "controller_yaw_p_ff" in KNOWN_PARAMS and "controller_refine_w_mid" in KNOWN_PARAMS
	assert _accel.EXPECTED_ABI == 60 and _accel.ABI_VERSION == 60


# ---- --refine-w-mid ---------------------------------------------------------------------

def _m(steady=0.5, mid=None, err=1.0, pos=0.3, reward=-1.0):
	return ControllerMetrics(reward=reward, stable_rate=1.0, mean_attitude_error_deg=err,
	                         mean_steady_error_deg=steady, mean_position_error_m=pos,
	                         mean_mid_error_deg=mid)


def _calc(cfg, with_tail=True):
	return FitnessCalculatorControllerRefine(cfg, zrank_clamp=3.0, zrank_mad_floor=False,
	                                         zrank_pos_floor=0.0, pos_metric="pos", with_tail=with_tail)


def test_w_mid_default_is_the_pre_amendment_run():
	a = _args("--refine-fitness", "--refine-w-tail", "0")
	assert a.refine_w_mid == 0.0 and not mid_armed_from_args(a) and not mid_armed_from_args(_args())
	tok = refine_token(a)
	assert "mid" not in tok and tok.endswith(":pfloor=0.15") and ",pos0.1:" in tok
	rec = refine_weights_record(a)
	assert "w_mid" not in rec and rec == {k: v for k, v in asdict(RefineConfig(w_tail=0.0, search=True)).items() if k != "w_mid"}
	cfg = RefineConfig(w_tail=0.0)
	pool = [_m(0.5), _m(0.6, err=1.2, reward=-1.2), _m(0.7, err=0.9, reward=-0.9)]
	calc = _calc(cfg)
	assert calc.weight_mid == 0.0 and "mid=" not in calc.name
	# The column is absent: values in mean_mid_error_deg cannot move the fitness.
	with_mid = [replace(m, mean_mid_error_deg=0.1 * (i + 1)) for i, m in enumerate(pool)]
	assert calc.fitness(pool) == calc.fitness(with_mid)


def test_w_mid_token_record_and_arming():
	a = _args("--refine-fitness", "--refine-w-steady", "0.40", "--refine-w-mid", "0.25", "--refine-w-reward", "0.10")
	validate_refine_args(a)
	assert mid_armed_from_args(a)
	assert ":w=steady0.4,rw0.1,tail0.15,pos0.1,mid0.25:" in refine_token(a)
	assert refine_weights_record(a)["w_mid"] == 0.25
	# --refine-select alone never arms the search column.
	assert not mid_armed_from_args(_args("--refine-select", "--select-tail", "p99", "--refine-w-mid", "0.25"))
	with pytest.raises(SystemExit, match="refine-w"):
		validate_refine_args(_args("--refine-fitness", "--refine-w-mid", "-0.1"))


def test_w_mid_changes_the_ranking_when_only_mid_differs():
	cfg = RefineConfig(w_tail=0.0, w_steady=0.40, w_reward=0.10, w_mid=0.25)
	same = dict(steady=0.5, err=1.0, pos=0.3, reward=-1.0)
	pool = [_m(mid=0.9, **same), _m(mid=0.5, **same), _m(mid=0.7, **same)]
	f = _calc(cfg).fitness(pool)
	order = sorted(range(3), key=lambda i: f[i])
	rev = sorted(range(3), key=lambda i: -f[i])
	by_mid = [1, 2, 0]                                        # lowest mid first
	assert by_mid in (order, rev), f"fitness {f} does not order the pool by mid"
	assert len(set(f)) == 3
	# With the weight at 0 the same pool is a three-way tie.
	assert len(set(_calc(replace(cfg, w_mid=0.0)).fitness(pool))) == 1
	assert "mid=0.25" in _calc(cfg).name


def test_w_mid_refuses_a_missing_value_and_stays_out_of_stage_select():
	cfg = RefineConfig(w_tail=0.0, w_mid=0.25)
	with pytest.raises(ValueError, match="mean_mid_error_deg"):
		_calc(cfg).fitness([_m(mid=0.5), _m(mid=None, err=1.2)])
	with pytest.raises(ValueError, match="mean_mid_error_deg"):
		_calc(cfg).fitness([_m(mid=0.5), _m(mid=float("nan"), err=1.2)])
	select = _calc(cfg, with_tail=False)
	assert select.weight_mid == 0.0
	select.fitness([_m(mid=None), _m(mid=None, err=1.2, steady=0.6)])   # no mid needed, no refusal


def test_heldout_row_prints_mid_only_when_present():
	from wnn.control.phased_ga import _heldout_row_str
	off = _heldout_row_str(_m(steady=0.5))
	on = _heldout_row_str(_m(steady=0.5, mid=0.62))
	assert "mid=" not in off
	assert on == off.replace("  steady=0.50°", "  steady=0.50°  mid=0.62°") and "mid=0.62°" in on
