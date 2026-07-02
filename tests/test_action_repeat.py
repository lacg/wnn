"""Action-repeat (arm R) targeted tests — spec .claude/plans/action_repeat_spec.md.

Four parts:
  (i)   N=1 anchor: a default-constructed controller is bit-identical to an
        explicit action_repeat=1 controller (the hold branch never fires).
  (ii)  N=5 CPU semantics: step() returns the IDENTICAL pwm for the 4 hold steps
        after each decision, then may change at the next decision; the feature
        accumulators (get_last_feature_vector) still advance on hold steps.
  (iii) N=5 cpu≡gpu: the same controllers scored via the CPU eval path
        (run_episode) and via score_controllers_metal agree — tight for the
        untrained (non-chaotic) controller, statistical for the seeded one.
  (iv)  Trainer-consistency probe (the trap's direct check): train at N=5 via
        bptt_train_window on a recorded PID trajectory with a FROZEN state layer
        (fully-seeded reservoir + protect_learned → state MSBs cannot flip), then
        deploy step() over the same sensor stream — every output address visited
        at a decision step MUST be in the trained cell set. A ring/decision
        misalignment between the trainer re-forward and deploy fails this.

Run:  PYTHONPATH=src python tests/test_action_repeat.py
"""

from __future__ import annotations

import math
import sys

import numpy as np

from wnn.control._accel import AttitudeSim, WnnController, score_controllers_metal
from wnn.control.evaluator import ControllerSpec, ControllerGenome, build_controller
from wnn.control.genome import FiniteStateGenome
from wnn.control.pid import AttitudePID, AttitudePIDConfig
from wnn.control.training import (
	EpisodeConfig, run_episode, make_wnn_action_fn, _sample_initial_state,
)


NM, LV, BPF, K = 4, 8, 8, 4
SN, SBPN, OBPN = 4, 12, 12
NUM_OUT = NM * LV


def _feature_thresholds(nf: int) -> list:
	"""Per-feature thermometer thresholds over each feature's PLAUSIBLE dynamic
	range (gyro rad/s, accel m/s² with z near +g, targets ~0) so the encoded
	bits actually flip during a rollout — a uniform (-2,2) span leaves accel-z
	bits constant and starves the address diversity these tests rely on."""
	ranges = {0: (-1.5, 1.5), 1: (-1.5, 1.5), 2: (-1.5, 1.5),      # gyro
	          3: (-4.0, 4.0), 4: (-4.0, 4.0), 5: (8.0, 9.9),        # accel
	          6: (-0.1, 0.1), 7: (-0.1, 0.1), 8: (-0.1, 0.1)}       # target
	th: list = []
	for f in range(nf):
		lo, hi = ranges.get(f, (-1.0, 3.0))  # extras (e.g. tilt_i): generic span
		th += [float(x) for x in np.linspace(lo, hi, BPF)]
	return th


def _mk_controller(seed: int, action_repeat: int | None, seed_cells: bool,
                   obs_tilt_i: bool = False) -> WnnController:
	"""Small controller with sensor-region connections + optional dense output cells."""
	nf = 9 + int(obs_tilt_i)
	frame_bits = nf * BPF
	sensor_total = K * frame_bits
	rng = np.random.default_rng(seed)
	th = _feature_thresholds(nf)
	sc = [int(x) for x in rng.integers(0, sensor_total, size=SN * SBPN)]
	# Output layer sees [current frame | state MSBs]; include the state tail and
	# FORCE one state bit per neuron so K-window (state-input) divergence is
	# visible in the output addresses (the trainer-consistency probe's sensor).
	oc = [int(x) for x in rng.integers(0, frame_bits + SN, size=NUM_OUT * OBPN)]
	for n in range(NUM_OUT):
		oc[n * OBPN] = frame_bits + (n % SN)
	kwargs = dict(
		num_motors=NM, levels_per_motor=LV, bits_per_feature=BPF, input_window_k=K,
		state_neurons=SN, state_bits_per_neuron=SBPN, output_bits_per_neuron=OBPN,
		thresholds=th, state_connections=sc, output_connections=oc,
		obs_tilt_i=obs_tilt_i,
	)
	if action_repeat is not None:
		kwargs["action_repeat"] = action_repeat
	c = WnnController(**kwargs)
	if seed_cells:
		# Dense pseudo-random output cells so the decode VARIES with the input.
		vrng = np.random.default_rng(seed + 1)
		for n in range(NUM_OUT):
			for addr in range(1 << OBPN):
				c.write_output_cell(n, addr, int(vrng.integers(0, 4)))
	return c


def _rollout_pwms(c: WnnController, steps: int, tilt_deg: float = 6.0):
	"""Closed-loop rollout; returns (pwms per step, feature vectors per step)."""
	sim = AttitudeSim()
	ang = math.radians(tilt_deg)
	sim.reset(q=[math.cos(ang / 2), math.sin(ang / 2) * 0.7, math.sin(ang / 2) * 0.7, 0.0],
	          omega=[0.05, -0.04, 0.02])
	c.reset()
	pwms, feats = [], []
	for _ in range(steps):
		if sim.is_unstable():
			break
		gyro, accel = sim.read_imu()
		pwm = c.step(list(gyro), list(accel), [0.0, 0.0, 0.0])
		pwms.append(list(pwm))
		feats.append(list(c.get_last_feature_vector()))
		sim.step(list(pwm))
	return pwms, feats


def test_i_n1_anchor() -> bool:
	"""Default ctor ≡ explicit action_repeat=1, step-for-step bit-identical."""
	a = _mk_controller(3, None, seed_cells=True)
	b = _mk_controller(3, 1, seed_cells=True)
	pa, _ = _rollout_pwms(a, 200)
	pb, _ = _rollout_pwms(b, 200)
	same = pa == pb
	# No-hold sanity: the pwm is not permanently frozen (address bits flip as
	# the sim evolves → decode changes; a stuck hold would freeze it forever).
	changes = sum(1 for t in range(1, len(pa)) if pa[t] != pa[t - 1])
	print(f"  [i] N=1 anchor: steps={len(pa)} bit-identical={same} "
	      f"consecutive-changes={changes}/{len(pa) - 1}")
	return same and changes > 0


def test_ii_n5_hold_semantics() -> bool:
	"""N=5: pwm frozen within each 5-block; accumulators advance on holds."""
	c = _mk_controller(5, 5, seed_cells=True, obs_tilt_i=True)
	pwms, feats = _rollout_pwms(c, 200)
	n = len(pwms)
	ok_hold = True
	for t in range(n):
		if t % 5 != 0 and pwms[t] != pwms[t - 1]:
			ok_hold = False
			print(f"  [ii] HOLD VIOLATION at t={t}: {pwms[t]} != {pwms[t-1]}")
			break
	# Decisions must actually change the pwm at least once (dense random cells).
	dec_changes = sum(1 for t in range(5, n, 5) if pwms[t] != pwms[t - 1])
	# Feature vector must advance on hold steps (sensors evolve + tilt_i integrates).
	feat_moves = sum(1 for t in range(1, n) if t % 5 != 0 and feats[t] != feats[t - 1])
	holds = sum(1 for t in range(1, n) if t % 5 != 0)
	print(f"  [ii] N=5: steps={n} hold-blocks-frozen={ok_hold} "
	      f"decision-changes={dec_changes}/{(n - 1) // 5} hold-feature-advances={feat_moves}/{holds}")
	return ok_hold and dec_changes > 0 and feat_moves == holds


def _cpu_score(controller, ep_seeds, ec):
	sim = AttitudeSim()
	rewards, errs, stable = [], [], 0
	for s in ep_seeds:
		controller.reset()
		rng = np.random.default_rng(s)
		res = run_episode(make_wnn_action_fn(controller), sim, ec, rng=rng)
		rewards.append(res.cumulative_reward)
		errs.append(res.mean_attitude_error_rad)
		if (not res.diverged) and res.mean_attitude_error_rad <= math.radians(5.0):
			stable += 1
	n = len(ep_seeds)
	return float(np.mean(rewards)), float(np.mean(errs)), stable / n


def test_iii_n5_cpu_gpu_parity() -> bool:
	"""N=5: CPU eval path vs score_controllers_metal on identical ICs."""
	E, STEPS, TILT = 8, 600, 10.0
	spec = ControllerSpec(num_motors=NM, levels_per_motor=LV, bits_per_feature=BPF,
		input_window_k=K, state_neurons=SN, state_bits_per_neuron=SBPN,
		output_bits_per_neuron=OBPN, delta_control=False, action_repeat=5)
	nf = spec.num_features()
	th = list(np.linspace(-2.0, 2.0, nf * BPF).astype(float))
	ec = EpisodeConfig(dt=0.001, steps_per_episode=STEPS,
		max_initial_tilt_rad=math.radians(TILT), max_initial_yaw_rad=math.radians(TILT),
		max_initial_body_rate=0.5, max_initial_yaw_rate=0.3)

	controllers = []
	for s, seed_cells in ((11, False), (12, True)):
		g = FiniteStateGenome.random(spec, seed=s)
		c = build_controller(ControllerGenome(spec=spec, thresholds=th,
			state_connections=g.state_connections, output_connections=g.output_connections))
		if seed_cells:
			vrng = np.random.default_rng(s)
			for n in range(NUM_OUT):
				for addr in range(1 << OBPN):
					c.write_output_cell(n, addr, int(vrng.integers(0, 4)))
		controllers.append(c)

	rng = np.random.default_rng(321)
	ep_seeds = [int(rng.integers(0, 2**31)) for _ in range(E)]
	q0, omega0 = [], []
	for s in ep_seeds:
		r = np.random.default_rng(s)
		q, om = _sample_initial_state(r, ec.max_initial_tilt_rad, ec.max_initial_yaw_rad,
			ec.max_initial_body_rate, ec.max_initial_yaw_rate)
		q0 += [float(x) for x in q]
		omega0 += [float(x) for x in om]

	cpu = [_cpu_score(c, ep_seeds, ec) for c in controllers]
	gpu = score_controllers_metal(controllers, q0, omega0, E, STEPS)

	ok = True
	for lab, (cr, ce, cs), g in zip(("untrained", "seeded"), cpu, gpu):
		gr, ge, gs = g[0], g[1], g[2]
		rew_rel = abs(cr - gr) / max(abs(cr), 1.0)
		err_abs = abs(math.degrees(ce) - math.degrees(ge))
		stbl_abs = abs(cs - gs)
		print(f"  [iii] {lab:<10} reward {cr:>10.2f}/{gr:>10.2f} (rel {rew_rel*100:.3f}%)  "
		      f"err° {math.degrees(ce):.3f}/{math.degrees(ge):.3f} (Δ{err_abs:.3f}°)  "
		      f"stable {cs:.2f}/{gs:.2f} (Δ{stbl_abs:.2f})")
		if lab == "untrained":
			ok = ok and rew_rel < 0.02 and err_abs < 0.5 and stbl_abs <= 0.01
		else:
			# Chaotic closed loop → statistical parity (mirrors test_controller_gpu_parity).
			ok = ok and err_abs < 3.0 and stbl_abs <= 0.25
	return ok


def test_iv_trainer_consistency() -> bool:
	"""Train at N=5 with a frozen state layer, then check EVERY deploy-time
	decision-step output address is in the trained cell set."""
	N, STEPS = 5, 300
	c = _mk_controller(7, N, seed_cells=False)
	# Freeze the state layer's READ behavior: fully-seeded reservoir (no EMPTY
	# cell anywhere) + protect_learned=True in bptt → a commit can never flip a
	# cell's QSR MSB (same-side nudges only), so the state trajectory — hence
	# every address — is identical between the trainer re-forward and deploy.
	c.seed_state_reservoir(42)

	# Record one PID-driven episode (raw sensor stream — what the dagger records).
	# Aggressive IC so the sensors keep moving → many DISTINCT addresses (a
	# near-hover stream collapses to one address per neuron and the probe has
	# no discriminating power).
	sim = AttitudeSim()
	pid = AttitudePID(AttitudePIDConfig())
	ang = math.radians(20.0)
	q0 = [math.cos(ang / 2), math.sin(ang / 2) * 0.8, math.sin(ang / 2) * 0.6, 0.0]
	sim.reset(q=list(q0), omega=[1.0, -0.8, 0.5])
	pid.reset()
	gyros, accels, targets, pwms = [], [], [], []
	tgt = (0.0, 0.0, 0.0)
	for _ in range(STEPS):
		gyro, accel = sim.read_imu()
		q = sim.quaternion
		pwm = pid.step(q, gyro, tgt)
		gyros.append([float(x) for x in gyro])
		accels.append([float(x) for x in accel])
		targets.append(list(tgt))
		pwms.append([float(x) for x in pwm])
		sim.step(list(pwm))

	sw, ow = c.bptt_train_window(gyros, accels, targets, pwms,
		topk_per_neuron=2, reset_state=True, protect_learned=True)
	_state_cells, output_cells = c.export_cells()
	trained = {(n, a) for (n, a, _v) in output_cells}
	# The mask records ceil(STEPS/N) decisions; every record commits ≥1 write on
	# first touch (output memory starts EMPTY → protect never skips a new addr).
	print(f"  [iv] bptt at N={N}: state_writes={sw} output_writes={ow} "
	      f"trained-output-cells={len(trained)}")
	if ow == 0:
		print("  [iv] FAIL: trainer wrote no output cells")
		return False

	# Deploy: replay the SAME sensor stream through step(); collect the output
	# addresses read at each decision step.
	c.reset()
	missing, visited = 0, 0
	for t in range(STEPS):
		c.step(gyros[t], accels[t], targets[t])
		if t % N == 0:
			for (n, addr) in c.last_output_addresses():
				visited += 1
				if (n, addr) not in trained:
					missing += 1
	print(f"  [iv] deploy decision addresses: visited={visited} missing-from-trained={missing}")
	# Negative control — the MODE-MISMATCH the mask prevents: deploy the SAME
	# trained cells on an N=1 clone (every-step ring pushes) and sample the
	# addresses at the former decision steps. The window contents differ from
	# the N=5 training re-forward, so SOME addresses must be missing from the
	# trained set — proving this probe has teeth.
	c1 = _mk_controller(7, 1, seed_cells=False)
	c1.seed_state_reservoir(42)
	sc_cells, oc_cells = c.export_cells()
	c1.restore_cells(sc_cells, oc_cells)
	c1.reset()
	mis_missing, mis_visited = 0, 0
	for t in range(STEPS):
		c1.step(gyros[t], accels[t], targets[t])
		if t % N == 0:
			for (n, addr) in c1.last_output_addresses():
				mis_visited += 1
				if (n, addr) not in trained:
					mis_missing += 1
	print(f"  [iv] N=1-deploy control: visited={mis_visited} missing={mis_missing} (should be >0)")
	return missing == 0 and visited > 0 and mis_missing > 0


def main() -> int:
	results = []
	for name, fn in (("(i) N=1 anchor", test_i_n1_anchor),
	                 ("(ii) N=5 hold semantics", test_ii_n5_hold_semantics),
	                 ("(iii) N=5 cpu≡gpu", test_iii_n5_cpu_gpu_parity),
	                 ("(iv) trainer consistency", test_iv_trainer_consistency)):
		print(f"[{name}]")
		ok = fn()
		results.append((name, ok))
		print(f"  → {'PASS' if ok else 'FAIL'}")
	print("\nSUMMARY:")
	for name, ok in results:
		print(f"  {'PASS' if ok else 'FAIL'}  {name}")
	return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
	sys.exit(main())
