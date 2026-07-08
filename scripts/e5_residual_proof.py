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

from wnn.control.training import (EpisodeConfig, DisturbanceConfig, make_pid_action_fn,
    make_residual_action_fn)
from wnn.control.evaluator import ControllerSpec, fit_thresholds_from_pid_rollouts, random_connectivity
from wnn.control.dagger import (DaggerConfig, train_dagger, eval_closed_loop_reset,
    _pd_config, _pid_plus_config, _residual_baseline_config)
from wnn.control.pid import AttitudePID


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 20260609
    baseline = sys.argv[2] if len(sys.argv) > 2 else "pd"   # "pd" (84) | "stock_pid" (97)
    STEPS = 2000
    HELDOUT_SEED = seed + 9_000_000            # disjoint from train (seed) + DAGGER eval (seed+7M)
    SCALE, CLAMP = 1.0, 0.4                     # generous authority for the proof (learn-the-clamp later)

    # Residual WNN: signed output (delta_control off) + integral observations so it
    # can key the residual on the accumulated bias the PD baseline can't see.
    spec = ControllerSpec(num_motors=4, levels_per_motor=16, bits_per_feature=8,
        input_window_k=4, state_neurons=16, state_bits_per_neuron=32, output_bits_per_neuron=32,
        delta_control=False, obs_tilt_p=True, obs_tilt_i=True,
        obs_peraxis_p=True, obs_peraxis_i=True)

    ecL2 = EpisodeConfig(dt=0.001, steps_per_episode=STEPS,
        max_initial_tilt_rad=math.radians(5.0), max_initial_yaw_rad=math.radians(5.0),
        max_initial_body_rate=0.5, max_initial_yaw_rate=0.3,
        disturbance=DisturbanceConfig.preset("L2", seed=911))

    print(f"[e5-proof] seed={seed}  baseline={baseline}  steps={STEPS}  L2 armed  scale={SCALE} clamp={CLAMP}", flush=True)

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
    pp = AttitudePID(_pid_plus_config()); pp_s = score("PID+ (ceiling 99.8)", make_pid_action_fn(pp), pp.reset)

    # Train the residual hybrid under L2.
    print(f"[e5-proof] fitting thresholds + connectivity...", flush=True)
    thr = fit_thresholds_from_pid_rollouts(spec, num_episodes=10, seed=seed)
    sc, oc = random_connectivity(spec, seed=seed)
    cfg = DaggerConfig(residual=True, residual_baseline=baseline, residual_scale=SCALE, residual_clamp=CLAMP,
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

    print("\n[e5-proof] ===== VERDICT =====", flush=True)
    print(f"[e5-proof] seed={seed} baseline={baseline}  BASE {base_s:.1f}  |  HYBRID {hy_s:.1f}  |  PID+ {pp_s:.1f}", flush=True)
    verdict = ("BEATS BASE — residual adds value ✅" if hy_s > base_s + 2 else
               "≈ BASE (no lift)" if hy_s >= base_s - 2 else "BELOW BASE ❌")
    print(f"[e5-proof] {verdict}  (in-search best-iter stable={max(stats['iter_stable_rate'])*100:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
