"""E5 residual-hybrid @L2 PROOF (Phase-0 go/no-go).

Does `hybrid = compose_residual(PD, learned_WNN_residual)` clear the memoryless-PD
ceiling (84 @L2) toward PID+ (99.8)? The WNN learns clamp(PID+ − PD) via
residual-DAGGER under L2 (PID+ builds its integral against the armed tau_bias; the
WNN learns that integral action PD lacks). Prints held-out (seeds disjoint from
training) for PD-alone / PID+-alone / the trained hybrid.

Success bar: hybrid_stable > PD_stable (residual adds value) — ideally → PID+.
"""
import math
import sys

import numpy as np

from wnn.control.training import (EpisodeConfig, DisturbanceConfig, make_pid_action_fn,
    make_residual_action_fn, sample_ics_flat)
from wnn.control.evaluator import ControllerSpec, fit_thresholds_from_pid_rollouts, random_connectivity
from wnn.control.dagger import (DaggerConfig, train_dagger, eval_closed_loop_reset,
    _pd_config, _pid_plus_config, _residual_baseline_config, make_expert)
from wnn.control.pid import AttitudePID


def _gains_of(cfg):
    """[kp_rp, ki_rp, kd_rp, iclamp_rp, kp_yaw, ki_yaw, kd_yaw, iclamp_yaw, hover, authority]."""
    return [cfg.roll.kp, cfg.roll.ki, cfg.roll.kd, cfg.roll.i_clamp,
            cfg.yaw.kp, cfg.yaw.ki, cfg.yaw.kd, cfg.yaw.i_clamp,
            cfg.hover_throttle, cfg.max_axis_authority]


def _dist_args(ec):
    """Mirror evaluator._score_population_gpu: EpisodeConfig.disturbance → GPU args."""
    dist = getattr(ec, "disturbance", None)
    if dist is None:
        return {}
    asym = dist.resolved_motor_asym(np.random.default_rng(int(dist.seed)))
    return dict(dist_enabled=True,
        dist_tau_bias=[float(x) for x in dist.tau_bias],
        dist_gust_sigma=float(dist.gust_sigma), dist_gust_tau_c=float(dist.gust_tau_c),
        dist_motor_asym=[float(x) for x in asym],
        dist_gyro_sigma=float(dist.gyro_sigma), dist_gyro_bias_walk=float(dist.gyro_bias_walk),
        dist_accel_sigma=float(dist.accel_sigma), dist_seed=int(dist.seed))


def score_gpu(ctrl, ec, num_eps, seed, gains, scale, clamp):
    """Held-out score via the COLLAPSED Rust path (score_controllers_metal composes
    PID_base + clamped WNN residual in-kernel). scale=0 ⇒ pure-PID ruler. Same ICs
    (sample_ics_flat) the Python eval_closed_loop_reset draws → interchangeable."""
    from wnn.control._accel import score_controllers_metal
    q0, omega0 = sample_ics_flat(seed, num_eps, ec)
    rows = score_controllers_metal([ctrl], q0, omega0, num_eps, ec.steps_per_episode,
        residual_enabled=True, residual_scale=scale, residual_clamp=clamp, pid_gains=gains,
        **_dist_args(ec))
    r = rows[0]
    return dict(stable=r[2] * 100.0, err=math.degrees(r[1]), rise=r[6] * 1000.0,
                settle=r[7] * 1000.0, itae=r[9])


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 20260609
    baseline = sys.argv[2] if len(sys.argv) > 2 else "pd"   # "pd" (84) | "stock_pid" (97)
    STEPS = 2000
    HELDOUT_SEED = seed + 9_000_000            # disjoint from train (seed) + DAGGER eval (seed+7M)
    # residual_clamp is a searched param (learn-the-clamp): argv[3] overrides the
    # per-motor authority bound. Retrain-per-value — the clamp shapes the DAGGER
    # LABEL clamp(PID+ − baseline), not just inference.
    SCALE = 1.0
    CLAMP = float(sys.argv[3]) if len(sys.argv) > 3 else 0.4

    # Residual WNN: signed output (delta_control off) + integral observations so it
    # can key the residual on the accumulated bias the PD baseline can't see.
    # CAPACITY probe (Task #1 close-the-LQR-gap): argv[6]=state_neurons (default 16),
    # argv[7]=state_bits_per_neuron (default 32). More NEURONS = more partial-connectivity
    # perspectives (the ensemble generalization lever); bits scale per-neuron address space.
    STATE_NEURONS = int(sys.argv[6]) if len(sys.argv) > 6 else 16
    STATE_BITS = int(sys.argv[7]) if len(sys.argv) > 7 else 32
    spec = ControllerSpec(num_motors=4, levels_per_motor=16, bits_per_feature=8,
        input_window_k=4, state_neurons=STATE_NEURONS, state_bits_per_neuron=STATE_BITS,
        output_bits_per_neuron=32,
        delta_control=False, obs_tilt_p=True, obs_tilt_i=True,
        obs_peraxis_p=True, obs_peraxis_i=True)

    # Disturbance level is a searched param (argv[4]) — OFF/L1/L2/L3. Lighter regimes
    # let the controllers settle inside the 2° band, so rise/settle discriminate
    # ("faster reaction?"); L2's ~3.75° floor pins them at the sentinel.
    level = sys.argv[4] if len(sys.argv) > 4 else "L2"
    expert = sys.argv[5] if len(sys.argv) > 5 else "pid_plus"   # pid_plus | lqr | mpc
    dist = None if level == "OFF" else DisturbanceConfig.preset(level, seed=911)
    ecL2 = EpisodeConfig(dt=0.001, steps_per_episode=STEPS,
        max_initial_tilt_rad=math.radians(5.0), max_initial_yaw_rad=math.radians(5.0),
        max_initial_body_rate=0.5, max_initial_yaw_rate=0.3,
        disturbance=dist)

    print(f"[e5-proof] seed={seed}  baseline={baseline}  steps={STEPS}  dist={level}  scale={SCALE} clamp={CLAMP}"
          f"  spec={STATE_NEURONS}n×{STATE_BITS}b  expert={expert}", flush=True)

    def score(tag, action_fn, reset_fn):
        _, m = eval_closed_loop_reset(action_fn, reset_fn, ecL2, 20, HELDOUT_SEED)
        print(f"[e5-proof] {tag:22s} stable={m['stable_rate']*100:5.1f}%  err={m['mean_attitude_error_deg']:.2f}°"
              f"  rise={m['mean_rise_time_s']*1000:6.1f}ms  settle2°={m['mean_settle_time_abs2deg_s']*1000:6.1f}ms"
              f"  settle5%={m['mean_settle_time_rel5pct_s']*1000:6.1f}ms  ITAE={m['mean_itae']:.3f}", flush=True)
        return m["stable_rate"] * 100

    # Rulers @L2 (held-out): the chosen baseline (its own floor) + the PID+ ceiling.
    base_ruler = "pd (ruler 84)" if baseline == "pd" else "stock_pid (ruler 97)"
    bl = AttitudePID(_residual_baseline_config(baseline))
    base_s = score(f"BASE {base_ruler}", make_pid_action_fn(bl), bl.reset)
    # Ceiling ruler = the DAGGER expert the WNN imitates (PID+ | LQR | MPC).
    pp = make_expert(expert); pp_s = score(f"EXPERT ({expert})", make_pid_action_fn(pp), pp.reset)

    # Train the residual hybrid under L2.
    print(f"[e5-proof] fitting thresholds + connectivity...", flush=True)
    thr = fit_thresholds_from_pid_rollouts(spec, num_episodes=10, seed=seed)
    sc, oc = random_connectivity(spec, seed=seed)
    cfg = DaggerConfig(residual=True, residual_baseline=baseline, residual_expert=expert,
        residual_scale=SCALE, residual_clamp=CLAMP,
        num_iterations=8, episodes_per_iter=20, steps_per_episode=STEPS, eval_episodes=20,
        episode_config=ecL2, seed=seed, progress=True)
    print(f"[e5-proof] residual-DAGGER @L2 (8 iters × 20 eps)...", flush=True)
    ctrl, stats = train_dagger(spec, thr, sc, oc, cfg)

    # Held-out hybrid — baseline rebuilt from the SAME config DAGGER trained against.
    base = AttitudePID(_residual_baseline_config(baseline))
    hy_fn = make_residual_action_fn(make_pid_action_fn(base), ctrl, SCALE, CLAMP, 4)
    def hy_reset():
        ctrl.reset(); base.reset()
    hy_s = score("HYBRID (base+residual)", hy_fn, hy_reset)

    # ---- COLLAPSED RUST PATH: score the SAME three held-out via score_controllers_metal
    #      (Phase 2 composes PID_base + clamped WNN residual in-kernel). PD/PID+ rulers
    #      use scale=0 (WNN ignored → pure baseline); HYBRID uses the trained residual.
    #      L2 disturbance realization differs (GPU channel-15 vs CPU per-episode rng) so
    #      expect STATISTICAL, not bit-exact, agreement — the FINDINGS should reproduce.
    # The GPU scorer only knows PID mixing (not the LQR K-matrix / MPC QP), so the
    # GPU EXPERT ruler is meaningful only for pid_plus. For lqr/mpc the Python expert
    # ruler above stands; the GPU HYBRID score is the real test (it scores the composed
    # PD+WNN regardless of which teacher the WNN imitated).
    base_g = _gains_of(_residual_baseline_config(baseline))
    g_base = score_gpu(ctrl, ecL2, 20, HELDOUT_SEED, base_g, 0.0, CLAMP)
    g_hy = score_gpu(ctrl, ecL2, 20, HELDOUT_SEED, base_g, SCALE, CLAMP)
    g_pp = score_gpu(ctrl, ecL2, 20, HELDOUT_SEED, _gains_of(_pid_plus_config()), 0.0, CLAMP) if expert == "pid_plus" else None
    print("\n[e5-proof] ----- collapsed Rust path (score_controllers_metal) -----", flush=True)
    rows = [("BASE (gpu)", g_base), ("HYBRID (gpu)", g_hy)]
    if g_pp is not None:
        rows.insert(1, ("EXPERT (gpu)", g_pp))
    for tag, gm in rows:
        print(f"[e5-proof] {tag:22s} stable={gm['stable']:5.1f}%  err={gm['err']:.2f}°"
              f"  rise={gm['rise']:6.1f}ms  settle2°={gm['settle']:6.1f}ms  ITAE={gm['itae']:.3f}", flush=True)

    print("\n[e5-proof] ===== VERDICT =====", flush=True)
    print(f"[e5-proof] seed={seed} baseline={baseline} expert={expert}  [python] BASE {base_s:.1f} | HYBRID {hy_s:.1f} | EXPERT {pp_s:.1f}", flush=True)
    pp_rust = f"{g_pp['stable']:.1f}" if g_pp is not None else "n/a"
    print(f"[e5-proof] seed={seed} baseline={baseline} expert={expert}  [rust]   BASE {g_base['stable']:.1f} | HYBRID {g_hy['stable']:.1f} | EXPERT {pp_rust}", flush=True)
    verdict = ("BEATS BASE — residual adds value ✅" if hy_s > base_s + 2 else
               "≈ BASE (no lift)" if hy_s >= base_s - 2 else "BELOW BASE ❌")
    gpu_repro = ("REPRODUCES ✅" if g_hy['stable'] > g_base['stable'] + 2 else "does NOT reproduce ❌")
    print(f"[e5-proof] python: {verdict}  |  rust path: {gpu_repro}"
          f"  (in-search best-iter stable={max(stats['iter_stable_rate'])*100:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
