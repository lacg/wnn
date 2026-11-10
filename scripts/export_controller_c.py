#!/usr/bin/env python3
"""Emit a controller winner as a self-contained C header for MCU deployment
and benchmarking.

TWO KINDS OF EXPORT (07/11/2026).
  * CASCADE (default): everything the on-board C (mcu/wnn_cascade.c) needs to run
    the winner exactly as the simulator does — the lookup tables PLUS the
    thermometer thresholds, every feature-block flag, the integral/estimator/
    action-repeat/delta-law constants, the CTBR bank alphabet, the firmware rate
    loop's gains (rate-I normalisation) and, for CTRL-38, the d-hat observer's
    plant [arm, k_drag, Ixx, Iyy, Izz], gain and motor lag. The C side can only
    compute what it ports, so a winner it cannot run is REFUSED with every reason
    listed — never exported partially. Parity: mcu/parity/test_cascade_parity.py.
  * --tables-only: the lookup tables alone, for the instruction-count benches
    (bench.c / bench_inc.c feed synthetic inputs). The header says so
    (no WNN_EXPORT_CASCADE), and wnn_cascade.h refuses to compile against it.

THRESHOLDS ARE NOT IN THE CHECKPOINT. Every scorer refits them from the run's
TRAIN seed (phased_ga._report_thresholds) — the ladder the cells were written
under. The exporter does the same from the run's own argv (checkpoint meta, or
--run-argv), so the shipped ladder is the scored one. Runs whose ladder cannot be
reproduced that way (--threshold-refit-from-student, --threshold-outer-quantile)
are refused.

REPRESENTATION. What ships is a membership test, not a key->value table: per
neuron, a SORTED array of TRUE addresses, binary-searched at run time.

THE ON-SET IS output_values == TRUE, NOT output_universe (fixed 02/09/2026).
A populated cell is NOT necessarily TRUE: BINARY stores both, and
`cell_to_weight` (neuron_memory.rs:113) reads TRUE(1)=1.0 with FALSE, EMPTY and
every stray value alike at 0.0 — a stored FALSE weighs EXACTLY what a miss
weighs, so FALSE cells need not be stored at all.

CONNECTIVITY TYPE. wnn_conn is uint8_t only while every index fits [0, 255];
otherwise int16_t (a 28-feature x 8-bit cascade input is 224 bits, a 32-feature
one 256). The old exporter always wrote uint8_t, which silently truncated any
index >= 256 and any -1 (unconnected) slot.

Emits (both kinds):
  wnn_conn[N*B]   input-bit index feeding each neuron's address bit (MSB first)
  wnn_keys[]      all neurons' sorted ON-set addresses, concatenated
  wnn_off[N+1]    start offset per neuron into wnn_keys

Usage:
  PYTHONPATH=src/wnn python scripts/export_controller_c.py \
      --winner <winner.yaml.gz | stageN.yaml.gz> --out mcu/wnn_model.h \
      [--run-argv <run.out | argv.txt>] [--tables-only]
"""

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# neuron_memory.rs: FALSE=0, TRUE=1, EMPTY=2. BINARY weighs TRUE 1.0, all else 0.0.
TRUE_CELL = 1
# The C keys are uint32 and the address is built in a uint32.
MAX_C_BITS = 32


@dataclass
class ExportModel:
	"""Everything the header carries. Built from a checkpoint (main) or directly
	(mcu/parity builds synthetic CTRL-38 models through the same renderer)."""
	source: str
	spec: object            # ControllerSpec, the GENOME's (spec_from_arch)
	onset: dict             # true_onset_genome()
	thresholds: list        # f64 as fitted; the controller reads them as f32
	motor_lag_s: float      # the plant lag the scorer arms the observer with
	est_gains: tuple        # (kp, ki) of the attitude estimator (wheel defaults)
	dhat_l_gain: float      # CTBR_MPC_L_GAIN read from the wheel (0 when d-hat off)


def parse_args():
	ap = argparse.ArgumentParser(description="Export a controller winner as a C header")
	ap.add_argument("--winner", required=True)
	ap.add_argument("--out", required=True)
	ap.add_argument("--run-argv", default=None,
	                help="the run's phased_ga argv when the checkpoint meta lacks it: the run's "
	                     ".out log (its '[argv]' line) or a file holding the argv")
	ap.add_argument("--tables-only", action="store_true",
	                help="lookup tables only, for the instruction-count benches (NOT runnable)")
	return ap.parse_args()


# ------------------------------------------------------------------ on-set --

def true_onset(winner: str) -> dict:
	"""Load a winner checkpoint and build its TRUE-only on-set.

	Shared with scripts/count_true_keys.py so the deployability column and the
	shipped header can never disagree on what a key is.
	"""
	from wnn.control.checkpoint_io import load_controller_checkpoint
	ckpt = load_controller_checkpoint(winner, skip_population=True)
	return true_onset_genome(ckpt["best_genome"], ckpt["spec"])


def _genome_spec(genome, base_spec):
	# The neuron count is the GENOME's, not motors x levels of the base spec
	# (19/09/2026): the NEURONS GA reshapes the output layer and the evaluator
	# derives levels_per_motor from the genome (spec_from_arch). Same derivation.
	from wnn.control.evaluator import spec_from_arch
	return spec_from_arch(genome, base_spec)


def _feature_geometry(spec) -> tuple:
	# num_features / bits_per_feature are METHODS on ControllerSpec, not fields.
	nf = spec.num_features() if callable(spec.num_features) else spec.num_features
	bpf = spec.bits_per_feature() if callable(spec.bits_per_feature) else spec.bits_per_feature
	return int(nf), int(bpf)


def _onsets_by_neuron(cells) -> tuple:
	"""TRUE-only (02/09/2026): output_universe is every POPULATED cell, the value
	lives in output_values; a stored FALSE weighs what a miss weighs."""
	universe, values = cells.output_universe, cells.output_values
	if len(values) != len(universe):
		raise SystemExit(f"output_values {len(values)} != output_universe {len(universe)}")
	onsets, n_true, n_false = defaultdict(list), 0, 0
	for (n, addr), v in zip(universe, values):
		if int(v) == TRUE_CELL:
			onsets[int(n)].append(int(addr))
			n_true += 1
		else:
			n_false += 1
	return onsets, n_true, n_false


def _concat_onsets(onsets, neurons: int) -> tuple:
	keys, off = [], [0]
	for n in range(neurons):
		keys.extend(sorted(set(onsets.get(n, []))))
		off.append(len(keys))
	return keys, off


def true_onset_genome(genome, base_spec) -> dict:
	"""TRUE-only on-set of an already-loaded genome (e.g. a stage-select
	headline pulled from a stage checkpoint). Same geometry rules as true_onset."""
	spec = _genome_spec(genome, base_spec)
	bits, neurons = int(spec.output_bits_per_neuron), int(genome.output_neurons)
	motors, levels = int(spec.num_motors), int(spec.levels_per_motor)
	if neurons != motors * levels:
		raise SystemExit(f"output_neurons {neurons} is not a multiple of num_motors {motors}")
	_, out_conn = genome.to_connections()
	if len(out_conn) != neurons * bits:
		raise SystemExit(f"connection count {len(out_conn)} != {neurons}x{bits}")
	onsets, n_true, n_false = _onsets_by_neuron(genome.cells)
	keys, off = _concat_onsets(onsets, neurons)
	nf, bpf = _feature_geometry(spec)
	return dict(neurons=neurons, bits=bits, motors=motors, levels=levels,
	            features=nf, bits_per_feature=bpf, in_bits=nf * bpf,
	            conn=[int(c) for c in out_conn], keys=keys, off=off, n_true=n_true, n_false=n_false)


# ---------------------------------------------------------------- refusals --

def _layer_refusals(spec, onset) -> list:
	why = []
	if int(spec.state_neurons) != 0:
		why.append(f"state_neurons={spec.state_neurons}: the recurrent state layer is not ported (sn=0 only)")
	if bool(getattr(spec, "output_full_window", False)):
		why.append("output_full_window: the K-frame output window is not ported (current frame only)")
	if onset["bits"] > MAX_C_BITS:
		why.append(f"output_bits_per_neuron={onset['bits']} > {MAX_C_BITS} (uint32 keys/addresses)")
	if max(onset["conn"], default=0) >= onset["in_bits"]:
		why.append(f"a connection indexes input bit {max(onset['conn'])} >= {onset['in_bits']}")
	return why


def _decode_refusals(spec, onset) -> list:
	why = []
	if spec.ctbr is None:
		why.append("output_mode=pwm (motor-level): the on-board C ports the CASCADE (ctbr) only")
	if str(spec.memory_mode).upper() != "BINARY":
		why.append(f"memory_mode={spec.memory_mode}: only BINARY (TRUE on-set) is ported")
	if spec.resolved_output_decode() != "antagonist":
		why.append(f"output_decode={spec.resolved_output_decode()}: only the antagonist decode is ported")
	if onset["motors"] != 4 or onset["levels"] % 2:
		why.append(f"{onset['motors']} banks x {onset['levels']} levels: need 4 banks, even levels")
	return why


def _feature_refusals(spec) -> list:
	why = []
	for flag in ("obs_pwm", "dhat_ff", "decouple_outputs"):
		if bool(getattr(spec, flag, False)):
			why.append(f"{flag}: not ported (CTBR refuses it in Rust as well)")
	if spec.dhat_b is not None:
		why.append("dhat_b (the motor-level d-hat observer): not ported")
	if bool(getattr(spec, "obs_att_bias_pred", False)):
		why.append("obs_att_bias_pred (CTRL-46 Mahony-bias predictor): not ported to the on-board C yet")
	return why


def cascade_refusals(spec, onset) -> list:
	"""Every reason the on-board C cannot reproduce this winner (empty = exportable)."""
	return _decode_refusals(spec, onset) + _layer_refusals(spec, onset) + _feature_refusals(spec)


def refuse(why: list, what: str):
	raise SystemExit(f"REFUSING to export {what} — the on-board C cannot compute it:\n  - "
	                 + "\n  - ".join(why) + "\n(use --tables-only for a bench-only header)")


# ------------------------------------------------- run reconstruction ------

def run_argv(meta: dict, argv_file) -> list:
	"""The run's phased_ga argv: the checkpoint meta's, else --run-argv's."""
	if argv_file is None:
		if "argv" not in meta:
			raise SystemExit("the checkpoint carries no argv: pass --run-argv <run .out log>")
		return [str(a) for a in meta["argv"]]
	text = Path(argv_file).read_text(errors="replace")
	m = re.search(r"^\[argv\] (.*)$", text, re.MULTILINE)
	return (m.group(1) if m else text).split()


def threshold_refusals(args) -> list:
	why = []
	if getattr(args, "threshold_refit_from_student", False):
		why.append("--threshold-refit-from-student: the ladder came from a student rollout")
	if getattr(args, "threshold_outer_quantile", None) is not None:
		why.append("--threshold-outer-quantile: grid and memory stages fit different ladders")
	return why


def train_seed_thresholds(argv: list, spec) -> tuple:
	"""(thresholds, motor_lag_s) exactly as the held-out scorer fits them."""
	from wnn.control.phased_ga import build_arg_parser, episode_config_from_args, _report_thresholds
	from wnn.seeds import resolve_seed_set
	args = build_arg_parser().parse_args(argv)
	why = threshold_refusals(args)
	if why:
		refuse(why, "this run's thermometer")
	ec = episode_config_from_args(args)
	base = args.base_seed if args.base_seed is not None else args.seed
	seeds = resolve_seed_set(base=base, run_index=0, train=args.train_seed,
	                         test=args.test_seed, val=args.val_seed)
	thr = _report_thresholds(args, ec, spec, seeds.train, seeds.train, True)
	return [float(t) for t in thr], float(getattr(ec, "motor_lag_s", 0.0))


def wheel_constants(spec) -> tuple:
	"""((est_kp, est_ki), dhat_l_gain) as the INSTALLED wheel defines them — never
	a hand copy: the estimator gains are WnnController's constructor defaults (no
	Python caller overrides them) and the d-hat gain is CTBR_MPC_L_GAIN, which the
	CtbrSpec provenance line prints."""
	from wnn.control import _accel as ra
	sig = ra.WnnController.__text_signature__ or ""
	kp, ki = (re.search(rf"\b{k}=([0-9.eE+-]+)", sig) for k in ("est_kp", "est_ki"))
	if kp is None or ki is None:
		raise SystemExit(f"cannot read est_kp/est_ki defaults from the wheel signature: {sig!r}")
	lgain = 0.0
	plant = dhat_plant(spec.ctbr)
	if plant is not None:
		# The dres-only spec prints no gain: ask the same CtbrSpec with the observer
		# emitted (CTRL-48 a3 runs the a2 observer, ctbr.rs cascade_obs).
		probe = dict(spec.ctbr, obs_dhat_plant=list(plant))
		m = re.search(r"obs=dhat\(l=([0-9.eE+-]+),", repr(ra.CtbrSpec(**probe)))
		if m is None:
			raise SystemExit("cannot read the d-hat gain from the wheel's CtbrSpec provenance")
		lgain = float(m.group(1))
	return (float(kp.group(1)), float(ki.group(1))), lgain


def model_from_checkpoint(path: str, argv_file) -> ExportModel:
	from wnn.control.checkpoint_io import load_controller_checkpoint
	return model_from_ckpt(load_controller_checkpoint(path, skip_population=True), path, argv_file)


def model_from_ckpt(ckpt: dict, path: str, argv_file) -> ExportModel:
	"""An already-loaded checkpoint (stage checkpoints are hundreds of MB)."""
	spec = _genome_spec(ckpt["best_genome"], ckpt["spec"])
	onset = true_onset_genome(ckpt["best_genome"], ckpt["spec"])
	why = cascade_refusals(spec, onset)
	if why:
		refuse(why, path)
	thr, lag = train_seed_thresholds(run_argv(ckpt.get("meta", {}), argv_file), spec)
	est, lgain = wheel_constants(spec)
	return ExportModel(source=path, spec=spec, onset=onset, thresholds=thr,
	                   motor_lag_s=lag, est_gains=est, dhat_l_gain=lgain)


# --------------------------------------------------------------- rendering --

def f32_lit(x: float) -> str:
	"""Exact C float literal of x rounded to f32 (what pyo3 hands Rust's f32)."""
	return f"({float(np.float32(x)).hex()}f)"


def f64_lit(x: float) -> str:
	return f"({float(x).hex()})"


def init_list(values, lit) -> str:
	return "{" + ", ".join(lit(v) for v in values) + "}"


def emit_array(name: str, ctype: str, values, per_line: int = 16) -> str:
	body = []
	for i in range(0, len(values), per_line):
		body.append("\t" + ", ".join(str(v) for v in values[i:i + per_line]) + ",")
	return (f"static const {ctype} {name}[{len(values)}] = {{\n"
	        + "\n".join(body) + "\n};\n")


def conn_ctype(conn: list) -> str:
	lo, hi = min(conn, default=0), max(conn, default=0)
	if 0 <= lo and hi <= 255:
		return "uint8_t"
	if -32768 <= lo and hi <= 32767:
		return "int16_t"
	raise SystemExit(f"connection index range [{lo}, {hi}] does not fit int16_t")


def render_geometry(o: dict) -> list:
	return [
		f"#define WNN_NEURONS   {o['neurons']}",
		f"#define WNN_BITS      {o['bits']}",
		f"#define WNN_MOTORS    {o['motors']}",
		f"#define WNN_LEVELS    {o['levels']}",
		f"#define WNN_INPUT_BITS {o['in_bits']}",
		# The benches derive their thermometer walk from these rather than 15x8.
		f"#define WNN_FEATURES  {o['features']}",
		f"#define WNN_BITS_PER_FEATURE {o['bits_per_feature']}",
		f"#define WNN_NUM_KEYS  {len(o['keys'])}",
		f"typedef {conn_ctype(o['conn'])} wnn_conn_t;",
	]


FEATURE_FLAGS = (
	("TILT_P", "obs_tilt_p"), ("TILT_I", "obs_tilt_i"), ("PERAXIS_P", "obs_peraxis_p"),
	("PERAXIS_I", "obs_peraxis_i"), ("PERAXIS_YAW", "obs_peraxis_yaw"),
	("YAW_ERR", "obs_yaw_err"), ("YAW_ERR_I", "obs_yaw_err_i"),
	("COLLECTIVE_CMD", "obs_collective_cmd"), ("ALT_ERR", "obs_alt_err"), ("VZ", "obs_vz"),
	("POS_ERR_XY", "obs_pos_err_xy"), ("VEL_XY", "obs_vel_xy"), ("ATT_EST", "obs_att_estimator"),
)


def render_feature_flags(spec) -> list:
	ctbr = spec.ctbr
	lines = [f"#define WNN_OBS_{c} {int(bool(getattr(spec, a, False)))}" for c, a in FEATURE_FLAGS]
	lines.append(f"#define WNN_OBS_RATE_I {int(bool(ctbr.get('obs_rate_i', False)))}")
	lines.append(f"#define WNN_OBS_DHAT {int(ctbr.get('obs_dhat_plant') is not None)}")
	lines.append(f"#define WNN_OBS_DRES {int(ctbr.get('obs_dhat_residual_plant') is not None)}")
	return lines


def dhat_plant(ctbr):
	"""The observer's plant [arm, k_drag, Ixx, Iyy, Izz] when a2 (d-hat) or a3 (the
	CTRL-48 residual) runs it, else None — ctbr.rs cascade_obs: a2's or a3's."""
	c = ctbr or {}
	return c.get("obs_dhat_plant") or c.get("obs_dhat_residual_plant")


def render_controller_consts(m: ExportModel) -> list:
	s = m.spec
	return [
		f"#define WNN_DT {f32_lit(s.dt)}",
		f"#define WNN_INTEGRAL_LEAK {f32_lit(s.integral_leak)}",
		f"#define WNN_INTEGRAL_SCALE {f32_lit(s.integral_scale)}",
		f"#define WNN_EST_KP {f32_lit(m.est_gains[0])}",
		f"#define WNN_EST_KI {f32_lit(m.est_gains[1])}",
		f"#define WNN_ACTION_REPEAT {int(s.action_repeat)}",
		f"#define WNN_DELTA_CONTROL {int(bool(s.delta_control))}",
		f"#define WNN_DELTA_MAX {f32_lit(s.delta_max)}",
		f"#define WNN_DELTA_LEAK {f32_lit(s.delta_leak)}",
		"#define WNN_NEUTRAL (0.5f)  /* antagonist decode: neutral_decode_for */",
	]


def render_ctbr_consts(m: ExportModel) -> list:
	c = m.spec.ctbr
	plant = dhat_plant(c) or [0.0] * 5
	return [
		f"#define WNN_CTBR_RANGE_INIT {init_list(c['range'], f32_lit)}",
		f"#define WNN_CTBR_GAMMA_INIT {init_list(c['gamma'], f32_lit)}",
		f"#define WNN_RATE_GAINS_INIT {init_list(c['rate_gains'], f64_lit)}",
		f"#define WNN_RATE_OUT_LIMIT_N {f64_lit(c['out_limit_n'])}",
		f"#define WNN_RATE_HOVER_N {f64_lit(c['hover_n'])}",
		f"#define WNN_K_THRUST {f64_lit(c['k_thrust'])}",
		f"#define WNN_MAIN_LOOP_HZ {int(c['main_loop_hz'])}u",
		f"#define WNN_ATTITUDE_HZ {f64_lit(c['attitude_hz'])}",
		f"#define WNN_RATE_LPF_HZ {f64_lit(c['lpf_hz'])}",
		f"#define WNN_DHAT_PLANT_INIT {init_list(plant, f64_lit)}  /* arm, k_drag, Ixx, Iyy, Izz */",
		f"#define WNN_DHAT_L_GAIN {f64_lit(m.dhat_l_gain)}",
		f"#define WNN_MOTOR_LAG_S {f32_lit(m.motor_lag_s)}",
	]


def render_cascade_macros(m: ExportModel) -> list:
	return (["", "/* ---- cascade controller (mcu/wnn_cascade.c) ---- */", "#define WNN_EXPORT_CASCADE 1"]
	        + render_feature_flags(m.spec) + render_controller_consts(m) + render_ctbr_consts(m))


def render_tables(o: dict, thresholds) -> list:
	out = [
		emit_array("wnn_conn", "wnn_conn_t", o["conn"], 24),
		emit_array("wnn_keys", "uint32_t", o["keys"], 12),
		emit_array("wnn_off", "uint32_t", o["off"], 16),
	]
	if thresholds is not None:
		out.append(emit_array("wnn_thresholds", "float", [f32_lit(t) for t in thresholds], 4))
	return out


def render_header(onset: dict, source: str, model) -> str:
	"""`model` None => tables only. Two guarded sections: macros (wnn_cascade.h
	includes those alone) and the tables (one translation unit owns them)."""
	macros = render_geometry(onset) + (render_cascade_macros(model) if model else [])
	return "\n".join([
		"/* GENERATED by scripts/export_controller_c.py -- do not edit. */",
		f"/* source: {source} */",
		"#ifndef WNN_MODEL_H_MACROS", "#define WNN_MODEL_H_MACROS", "#include <stdint.h>", "",
		*macros, "#endif /* WNN_MODEL_H_MACROS */", "",
		"#if !defined(WNN_MODEL_MACROS_ONLY) && !defined(WNN_MODEL_H_TABLES)",
		"#define WNN_MODEL_H_TABLES",
		*render_tables(onset, model.thresholds if model else None),
		"#endif /* WNN_MODEL_H_TABLES */",
	]) + "\n"


def check_threshold_count(m: ExportModel):
	o = m.onset
	if len(m.thresholds) != o["features"] * o["bits_per_feature"]:
		raise SystemExit(f"thresholds {len(m.thresholds)} != {o['features']}x{o['bits_per_feature']}")


def write_header(out: str, text: str):
	p = Path(out)
	p.parent.mkdir(parents=True, exist_ok=True)
	p.write_text(text)


def report(out: str, o: dict, model):
	kb = (len(o["keys"]) * 4 + len(o["conn"])) / 1024
	kind = "CASCADE (runnable)" if model else "TABLES ONLY (bench)"
	print(f"wrote {out}  [{kind}] neurons={o['neurons']} bits={o['bits']} in_bits={o['in_bits']} "
	      f"features={o['features']}x{o['bits_per_feature']} keys={len(o['keys'])}  (~{kb:.0f} KB of .rodata)")
	print(f"  on-set filter: TRUE={o['n_true']} dropped FALSE/other={o['n_false']} "
	      f"({o['n_false'] / max(1, o['n_true'] + o['n_false']):.1%} of populated cells)")
	if model:
		c = model.spec.ctbr
		print(f"  blocks: rate_i={bool(c.get('obs_rate_i'))} dhat={c.get('obs_dhat_plant') is not None} "
		      f"dres={c.get('obs_dhat_residual_plant') is not None} "
		      f"motor_lag={model.motor_lag_s}s action_repeat={model.spec.action_repeat}")


def main():
	args = parse_args()
	if args.tables_only:
		onset, model = true_onset(args.winner), None
		if min(onset["conn"], default=0) < 0:
			refuse(["an unconnected (-1) address bit: the benches index inbits[] with it"], args.winner)
	else:
		model = model_from_checkpoint(args.winner, args.run_argv)
		check_threshold_count(model)
		onset = model.onset
	write_header(args.out, render_header(onset, args.winner, model))
	report(args.out, onset, model)


if __name__ == "__main__":
	main()
