#!/usr/bin/env python3
"""MCU cascade parity: the exported C controller (mcu/wnn_cascade.c, host build)
against the Rust WnnController (installed ram_controller wheel, read-only), step
for step, on the SAME recorded input stream.

WHAT IS ASSERTED, every physical tick:
  * the feature vector, bit for bit (f32);
  * the decision/hold cadence (action_repeat);
  * the held CTBR setpoint [coll dev, p, q, r] (Rust ctbr_held_setpoint);
  * the motor command the inner rate loop flies (Rust step() return vs the C
    reference rate stage, mcu/wnn_ratefw.c);
and on every DECISION tick:
  * the thermometer frame (Rust last_output_layer_input),
  * every output neuron's address (last_output_addresses),
  * every neuron's fire bit (get_last_output_cells == TRUE).
The first mismatch FAILS with its episode/step/quantity/index and both values.

THE CASES.
  flags_off   the trained C-S2-d cascade winner (logs/controller/ct40/full), both
              CTRL-38 blocks off — the production thresholds are refit from the
              run's own argv by the exporter.
  rate_i      CTRL-38 --obs-rate-i on   } NO trained winner with the blocks exists
  dhat        CTRL-38 --obs-cascade-dhat } yet (the CTRL-38 A/B has not run and the
  both        both blocks on            } pop-6 smokes saved no checkpoints), so these
              are SYNTHETIC genomes built through the Python API: the winner's own
              spec + the block flag(s), 64 neurons x 12 bits with >= 4 of every
              neuron's bits wired to the cascade columns, random TRUE on-sets (half
              of each neuron's 4096 addresses), thresholds from the PRODUCTION fit
              (the exporter's train-seed refit on the run argv + the block flag(s),
              which samples the cascade columns on the training cascade).
  dhat/f32    informational only: the same d-hat model built -DWNN_DHAT_F32 (the
              Metal kernel's f32 observer). Reported, never asserted.

THE INPUT STREAM. The Rust flags-off winner flies the run's own plant (airframe,
specific-force IMU, rotor drag, motor lag, translation, the L4C weather incl. the
per-episode motor asymmetry) closed loop; the per-tick inputs are recorded and then
replayed open-loop into a FRESH Rust controller and the C. Replay determinism is
itself checked: the fresh Rust controller must reproduce the recorded commands.

Run:
  PYTHONPATH=src/wnn python mcu/parity/test_cascade_parity.py [--episodes 3 --steps 2000]
"""

import argparse
import ctypes
import dataclasses
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import export_controller_c as exp  # noqa: E402

DEFAULT_WINNER = REPO / "logs/controller/ct40/full/C-S2-d_full_s2_winner.yaml.gz"
SHIM = REPO / "mcu/parity/wnn_parity_shim.c"
CFLAGS = ["-O2", "-std=c11", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC"]
F32P = ctypes.POINTER(ctypes.c_float)


class Mismatch(AssertionError):
	pass


@dataclass
class Flight:
	"""One recorded episode: per-episode reset inputs + per-tick inputs."""
	q0: np.ndarray
	yaw0: float
	anchor: float
	ins: dict           # name -> (T, k) float32
	pwm: np.ndarray     # (T, 4) the recording controller's commands


@dataclass
class Case:
	name: str
	model: object       # exp.ExportModel
	genome: object      # evaluator.ControllerGenome (the Rust controller's build input)
	defines: tuple = ()
	strict: bool = True       # False: count mismatches, never fail (informational)
	expect_fail: bool = False  # a MUTANT: the comparator must catch it
	cflags: tuple = ()         # appended after CFLAGS (a later flag wins)


# ------------------------------------------------------------------ C side --

class CShim:
	def __init__(self, header: Path, case: "Case", workdir: Path, tag: str):
		lib = workdir / f"shim_{tag}.dylib"
		cmd = ["cc", *CFLAGS, *case.cflags, f"-I{REPO / 'mcu'}", f'-DWNN_MODEL_HEADER="{header}"',
		       *[f"-D{d}" for d in case.defines], str(SHIM), "-o", str(lib), "-lm"]
		subprocess.run(cmd, check=True)
		self.lib = ctypes.CDLL(str(lib))
		self.dims = self._dims()
		self.lib.shim_init()

	def _dims(self) -> dict:
		d = (ctypes.c_int * 8)()
		self.lib.shim_dims(d)
		keys = ("features", "in_bits", "neurons", "bits", "levels", "ar", "rate_i", "dhat")
		return dict(zip(keys, list(d)))

	def reset(self, f: Flight):
		q0 = np.ascontiguousarray(f.q0, dtype=np.float32)
		self.lib.shim_reset(ctypes.c_float(f.yaw0), q0.ctypes.data_as(F32P), ctypes.c_float(f.anchor))

	def tick(self, row: dict) -> tuple:
		sp, pwm = np.zeros(4, np.float32), np.zeros(4, np.float32)
		args = [row[k].ctypes.data_as(F32P) for k in ("gyro", "accel", "target", "vert", "horiz")]
		dec = self.lib.shim_tick(*args, sp.ctypes.data_as(F32P), pwm.ctypes.data_as(F32P))
		return bool(dec), sp, pwm

	def dbg(self) -> tuple:
		d = self.dims
		out = (np.zeros(d["features"], np.float32), np.zeros(d["in_bits"], np.uint8),
		       np.zeros(d["neurons"], np.uint32), np.zeros(d["neurons"], np.uint8))
		self.lib.shim_dbg(*[a.ctypes.data_as(ctypes.c_void_p) for a in out])
		return out


# --------------------------------------------------------------- Rust side --

def rust_controller(genome, motor_lag_s: float):
	from wnn.control.evaluator import build_controller
	c = build_controller(genome)
	c.set_observer_motor_lag(float(motor_lag_s))
	return c


def rust_reset(c, f: Flight):
	c.reset(float(f.yaw0))
	c.seed_att_estimator([float(v) for v in f.q0])
	c.set_collective_anchor(float(f.anchor))


def rust_step(c, row: dict):
	c.set_vertical_obs(*[float(v) for v in row["vert"]])
	c.set_horizontal_obs(*[float(v) for v in row["horiz"]])
	return np.array(c.step(*[row[k].tolist() for k in ("gyro", "accel", "target")]), np.float32)


# ---------------------------------------------------------------- recording --

def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
	cr, sr, cp, sp = math.cos(roll / 2), math.sin(roll / 2), math.cos(pitch / 2), math.sin(pitch / 2)
	cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
	return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
	                 cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy], np.float32)


def make_sim(ec):
	from wnn.control import _accel as ra
	af = ec.airframe
	sim = ra.AttitudeSim(dt=ec.dt, arm_length=af.arm_length, k_thrust=af.k_thrust,
	                     k_drag=af.k_drag, inertia=list(af.inertia), gravity=af.gravity)
	sim.set_imu_specific_force(bool(ec.imu_specific_force))
	sim.set_rotor_drag(float(ec.rotor_drag))
	return sim


def start_episode(sim, ec, rng) -> tuple:
	"""cpu_score's order: reset -> translation -> vertical/horizontal state -> weather."""
	from wnn.control.training import apply_disturbance
	tilt = math.radians(5.0)
	q0 = quat_from_euler(*rng.uniform(-tilt, tilt, 3))
	sim.reset(q=q0.tolist(), omega=rng.uniform(-0.3, 0.3, 3).tolist())
	sim.set_motor_lag(float(ec.motor_lag_s))
	sim.set_translation(float(ec.airframe.mass) * float(rng.uniform(0.9, 1.1)))
	sim.set_vertical_state(float(rng.uniform(-0.1, 0.1)), 0.0)
	sim.set_horizontal_state(*[float(v) for v in rng.uniform(-0.5, 0.5, 2)], 0.0, 0.0)
	apply_disturbance(sim, ec.disturbance, rng)
	yaw0 = math.atan2(2 * (q0[0] * q0[3] + q0[1] * q0[2]), 1 - 2 * (q0[2] ** 2 + q0[3] ** 2))
	return q0, float(np.float32(yaw0)), float(np.float32(sim.hover_pwm()))


def observe(sim, ec, anchor: float) -> dict:
	gyro, accel = sim.read_imu()
	x, y = sim.position_xy()
	vx, vy = sim.velocity_xy()
	vert = [anchor, float(ec.target_altitude) - sim.altitude, sim.vertical_velocity]
	vals = dict(gyro=gyro, accel=accel, target=(0.0, 0.0, 0.0), vert=vert, horiz=(-x, -y, vx, vy))
	return {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in vals.items()}


def record_flight(ctl, sim, ec, rng, steps: int) -> Flight:
	q0, yaw0, anchor = start_episode(sim, ec, rng)
	f = Flight(q0=q0, yaw0=yaw0, anchor=anchor, ins={}, pwm=None)
	rust_reset(ctl, f)
	rows, pwms = [], []
	for _ in range(steps):
		if sim.is_unstable():
			break
		row = observe(sim, ec, anchor)
		pwm = rust_step(ctl, row)
		sim.step(pwm.tolist())
		rows.append(row)
		pwms.append(pwm)
	f.ins = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
	f.pwm = np.stack(pwms)
	return f


def record_flights(genome, ec, lag: float, episodes: int, steps: int, seed: int) -> list:
	ctl, sim, rng = rust_controller(genome, lag), make_sim(ec), np.random.default_rng(seed)
	return [record_flight(ctl, sim, ec, rng, steps) for _ in range(episodes)]


# --------------------------------------------------------------- comparison --

def bits_equal(a, b) -> np.ndarray:
	a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
	return a.view(np.uint32) == b.view(np.uint32)


class Comparator:
	"""Strict: raise on the first mismatch. Lenient: count mismatches per quantity."""

	def __init__(self, strict: bool):
		self.strict, self.counts, self.first = strict, {}, {}

	def check(self, ok, what: str, where: str, rust, c):
		ok = np.asarray(ok)
		if ok.all():
			return
		i = int(np.flatnonzero(~ok.ravel())[0])
		msg = (f"{where}: {what}[{i}] rust={np.asarray(rust).ravel()[i]!r} "
		       f"c={np.asarray(c).ravel()[i]!r} ({int((~ok).sum())} of {ok.size} differ)")
		if self.strict:
			raise Mismatch(msg)
		self.counts[what] = self.counts.get(what, 0) + 1
		self.first.setdefault(what, msg)


def compare_tick(cmp: Comparator, rctl, shim: CShim, t: int, where: str, r_pwm, c_out):
	dec, sp, pwm = c_out
	feats, frame, addr, fired = shim.dbg()
	r_feats = np.array(rctl.get_last_feature_vector(), np.float32)
	cmp.check(bits_equal(r_feats, feats), "feature", where, r_feats, feats)
	cmp.check(np.array(dec == (t % shim.dims["ar"] == 0)), "decision", where, [t % shim.dims["ar"] == 0], [dec])
	r_sp = np.array(rctl.ctbr_held_setpoint, np.float64)
	cmp.check(r_sp == sp.astype(np.float64), "setpoint", where, r_sp, sp)
	cmp.check(bits_equal(r_pwm, pwm), "pwm", where, r_pwm, pwm)
	if dec:
		compare_decision(cmp, rctl, where, frame, addr, fired)
	return feats, frame, fired, sp


def compare_decision(cmp, rctl, where: str, frame, addr, fired):
	r_frame = np.array(rctl.last_output_layer_input(), np.uint8)
	cmp.check(r_frame == frame, "frame_bit", where, r_frame, frame)
	r_addr = np.array([a for _, a in rctl.last_output_addresses()], np.uint64)
	cmp.check(r_addr == addr.astype(np.uint64), "address", where, r_addr, addr)
	r_fired = (np.frombuffer(bytes(rctl.get_last_output_cells()), np.uint8) == exp.TRUE_CELL).astype(np.uint8)
	cmp.check(r_fired == fired, "fire_bit", where, r_fired, fired)


# ---------------------------------------------------------------- telemetry --

class Telemetry:
	"""Evidence the run exercised what it claims: tick counts, cascade-column
	ranges and frame activity, fire rate, non-neutral setpoints."""

	def __init__(self, n_cas: int, bpf: int):
		self.n_cas, self.bpf = n_cas, bpf
		self.ticks = self.decisions = 0
		self.cas_vals, self.cas_frames, self.fired, self.sps = [], set(), [], []

	def add(self, dec: bool, feats, frame, fired, sp):
		self.ticks += 1
		if self.n_cas:
			self.cas_vals.append(feats[-self.n_cas:].copy())
		if not dec:
			return
		self.decisions += 1
		if self.n_cas:
			self.cas_frames.add(frame[-self.n_cas * self.bpf:].tobytes())
		self.fired.append(float(fired.mean()))
		self.sps.append(sp.copy())

	def summary(self) -> str:
		sps = np.array(self.sps)
		nz = ", ".join(f"{(sps[:, m] != 0).mean():.0%}" for m in range(4))
		s = (f"ticks={self.ticks} decisions={self.decisions} fire_rate={np.mean(self.fired):.3f} "
		     f"setpoint!=0 per bank [T,p,q,r]=[{nz}]")
		if self.n_cas:
			v = np.array(self.cas_vals)
			rng = " ".join(f"[{v[:, i].min():+.3g},{v[:, i].max():+.3g}]" for i in range(self.n_cas))
			s += f"\n      cascade columns range {rng}; distinct cascade frame patterns={len(self.cas_frames)}"
		return s


# ------------------------------------------------------------------- replay --

def replay(case: Case, flights: list, workdir: Path, check_recorded: bool) -> tuple:
	header = workdir / f"{case.name.replace('/', '_')}.h"
	exp.write_header(str(header), exp.render_header(case.model.onset, case.model.source, case.model))
	shim = CShim(header, case, workdir, case.name.replace("/", "_"))
	rctl, cmp = rust_controller(case.genome, case.model.motor_lag_s), Comparator(case.strict)
	tel = Telemetry(_n_cas(case.model.spec), case.model.onset["bits_per_feature"])
	for e, f in enumerate(flights):
		replay_flight(case, f, e, shim, rctl, cmp, tel, check_recorded)
	return tel, cmp


def _n_cas(spec) -> int:
	from wnn.control.evaluator import ctbr_cascade_feature_count
	return ctbr_cascade_feature_count(spec.ctbr)


def replay_flight(case, f: Flight, e: int, shim, rctl, cmp, tel, check_recorded: bool):
	rust_reset(rctl, f)
	shim.reset(f)
	for t in range(len(f.pwm)):
		row = {k: np.ascontiguousarray(v[t]) for k, v in f.ins.items()}
		r_pwm = rust_step(rctl, row)
		where = f"[{case.name}] episode {e} step {t}"
		if check_recorded and not bits_equal(r_pwm, f.pwm[t]).all():
			raise Mismatch(f"{where}: the fresh Rust controller does not reproduce the recording "
			               f"({r_pwm} vs {f.pwm[t]}) — replay is not deterministic")
		c_out = shim.tick(row)
		tel.add(c_out[0], *compare_tick(cmp, rctl, shim, t, where, r_pwm, c_out))


# ---------------------------------------------------------- case building --

def load_winner(path: Path) -> tuple:
	from wnn.control.checkpoint_io import load_controller_checkpoint
	from wnn.control.evaluator import controller_genome_from_arch
	ckpt = load_controller_checkpoint(str(path), skip_population=True)
	model = exp.model_from_ckpt(ckpt, str(path), None)
	genome = controller_genome_from_arch(ckpt["best_genome"], ckpt["spec"], model.thresholds)
	return model, genome, [str(a) for a in ckpt["meta"]["argv"]]


def run_ec(argv: list):
	from wnn.control.phased_ga import build_arg_parser, episode_config_from_args
	return episode_config_from_args(build_arg_parser().parse_args(argv))


def synthetic_conn(n_neurons: int, bits: int, in_bits: int, n_cas_bits: int, rng) -> list:
	"""Each neuron: >= 4 of its bits on the cascade columns (the LAST n_cas_bits)."""
	cas, rest = np.arange(in_bits - n_cas_bits, in_bits), np.arange(in_bits - n_cas_bits)
	conn = []
	for _ in range(n_neurons):
		row = np.concatenate([rng.choice(cas, 4, replace=False), rng.choice(rest, bits - 4, replace=False)])
		conn.extend(int(v) for v in rng.permutation(row))
	return conn


def synthetic_cells(n_neurons: int, bits: int, rng) -> tuple:
	"""TRUE on half of every neuron's addresses: (keys, off, triples)."""
	keys, off, triples = [], [0], []
	for n in range(n_neurons):
		on = np.sort(rng.choice(1 << bits, (1 << bits) // 2, replace=False))
		keys.extend(int(a) for a in on)
		off.append(len(keys))
		triples.extend((n, int(a), exp.TRUE_CELL) for a in on)
	return keys, off, triples


def synthetic_spec(base_spec, ec, rate_i: bool, dhat: bool):
	ctbr = dict(base_spec.ctbr)
	if rate_i:
		ctbr["obs_rate_i"] = True
	if dhat:
		af = ec.airframe
		ctbr["obs_dhat_plant"] = [float(af.arm_length), float(af.k_drag), *(float(x) for x in af.inertia)]
	return dataclasses.replace(base_spec, ctbr=ctbr, levels_per_motor=16, output_bits_per_neuron=12,
	                           state_neurons=0, state_bits_per_neuron=0)


def synthetic_case(name: str, base_spec, argv: list, ec, rate_i: bool, dhat: bool, seed: int) -> Case:
	from wnn.control.evaluator import ControllerGenome
	spec, rng = synthetic_spec(base_spec, ec, rate_i, dhat), np.random.default_rng(seed)
	flags = (["--obs-rate-i"] if rate_i else []) + (["--obs-cascade-dhat"] if dhat else [])
	thr, lag = exp.train_seed_thresholds(argv + flags, spec)
	nf, bpf = exp._feature_geometry(spec)
	n_neur, bits = 4 * spec.levels_per_motor, spec.output_bits_per_neuron
	conn = synthetic_conn(n_neur, bits, nf * bpf, _n_cas(spec) * bpf, rng)
	keys, off, triples = synthetic_cells(n_neur, bits, rng)
	onset = dict(neurons=n_neur, bits=bits, motors=4, levels=spec.levels_per_motor, features=nf,
	             bits_per_feature=bpf, in_bits=nf * bpf, conn=conn, keys=keys, off=off,
	             n_true=len(keys), n_false=0)
	est, lgain = exp.wheel_constants(spec)
	model = exp.ExportModel(source=f"synthetic:{name}", spec=spec, onset=onset, thresholds=thr,
	                        motor_lag_s=lag, est_gains=est, dhat_l_gain=lgain)
	assert not exp.cascade_refusals(spec, onset), exp.cascade_refusals(spec, onset)
	genome = ControllerGenome(spec=spec, thresholds=thr, state_connections=[],
	                          output_connections=conn, output_cells=triples)
	return Case(name=name, model=model, genome=genome)


def build_cases(model, genome, argv: list, ec) -> list:
	base = model.spec
	cases = [Case(name="flags_off", model=model, genome=genome)]
	for i, (name, ri, dh) in enumerate((("rate_i", True, False), ("dhat", False, True), ("both", True, True))):
		cases.append(synthetic_case(name, base, argv, ec, ri, dh, seed=1000 + i))
	return cases + control_cases(cases)


def control_cases(cases: list) -> list:
	"""dhat/f32 + flags_off/fp-contract: what breaks parity, measured (informational).
	dhat/mutant-lgain: a 1.5x observer gain in the C header ONLY — the comparator must
	catch it, or every PASS above is vacuous."""
	off, dhat = cases[0], cases[2]
	bad = dataclasses.replace(dhat.model, dhat_l_gain=dhat.model.dhat_l_gain * 1.5)
	return [
		dataclasses.replace(dhat, name="dhat/f32", defines=("WNN_DHAT_F32",), strict=False),
		dataclasses.replace(off, name="flags_off/fp-contract", cflags=("-ffp-contract=fast",), strict=False),
		dataclasses.replace(dhat, name="dhat/mutant-lgain", model=bad, expect_fail=True),
	]


# --------------------------------------------------------------------- main --

def run(winner: Path, episodes: int, steps: int, seed: int) -> int:
	model, genome, argv = load_winner(winner)
	ec = run_ec(argv)
	flights = record_flights(genome, ec, model.motor_lag_s, episodes, steps, seed)
	print(f"recorded {len(flights)} episodes, {sum(len(f.pwm) for f in flights)} ticks "
	      f"(motor lag {model.motor_lag_s}s, airframe {ec.airframe.name}, run weather)")
	failed = 0
	with tempfile.TemporaryDirectory(prefix="wnn_parity_") as td:
		for case in build_cases(model, genome, argv, ec):
			failed += report_case(case, flights, Path(td))
	print("PARITY " + ("FAILED" if failed else "PASSED") + f" ({failed} strict case(s) failed)")
	return 1 if failed else 0


def report_case(case: Case, flights: list, workdir: Path) -> int:
	try:
		tel, cmp = replay(case, flights, workdir, check_recorded=(case.name == "flags_off"))
	except Mismatch as m:
		print(f"  {'PASS (mutant caught)' if case.expect_fail else 'FAIL'} {case.name}: {m}")
		return 0 if case.expect_fail else 1
	if case.expect_fail:
		print(f"  FAIL {case.name}: the mutant was NOT detected — the comparison is vacuous")
		return 1
	status = "PASS (bit-identical)" if case.strict else "INFO (not asserted)"
	print(f"  {status} {case.name} [{case.model.source}]\n      {tel.summary()}")
	for what, n in cmp.counts.items():
		print(f"      {what}: {n} tick(s) differ; first: {cmp.first[what]}")
	return 0


def test_cascade_parity():
	assert run(DEFAULT_WINNER, episodes=2, steps=1000, seed=7) == 0


def main():
	ap = argparse.ArgumentParser(description="MCU cascade parity (C vs Rust)")
	ap.add_argument("--winner", type=Path, default=DEFAULT_WINNER)
	ap.add_argument("--episodes", type=int, default=3)
	ap.add_argument("--steps", type=int, default=2000)
	ap.add_argument("--seed", type=int, default=7)
	a = ap.parse_args()
	sys.exit(run(a.winner, a.episodes, a.steps, a.seed))


if __name__ == "__main__":
	main()
