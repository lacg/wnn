"""E5 residual hybrid — residual-DAGGER mechanics + the floor guarantee.

The hybrid is `action = compose_residual(baseline(err), wnn)`: the analytic PD/
stock-PID baseline carries the bulk action, the WNN learns only the clamped
residual (the integral action PD lacks). The load-bearing property is that an
UNTRAINED WNN (EMPTY cells → 0.5) yields residual 0, so the untrained hybrid is
IDENTICALLY the analytic baseline — training can only help. This test asserts:

  1. compose_residual / residual_train_target are exact inverses under the clamp.
  2. an untrained hybrid matches the PD baseline closed-loop (the floor).
  3. residual-DAGGER runs end-to-end (composes + trains + evals the hybrid).
"""
import math
import sys

from wnn.control.training import compose_residual, residual_train_target, make_pid_action_fn, EpisodeConfig
from wnn.control.evaluator import ControllerSpec, fit_thresholds_from_pid_rollouts, random_connectivity
from wnn.control.dagger import (DaggerConfig, train_dagger, eval_closed_loop_reset,
    _pd_config, _pid_plus_config, _eval_closed_loop_residual)
from wnn.control.pid import AttitudePID
from wnn.control._accel import WnnController

SCALE, CLAMP, NM = 1.0, 0.2, 4


def test_compose_inverts_target():
    """residual_train_target then compose_residual round-trips the clamped residual."""
    base = (0.5, 0.5, 0.5, 0.5)
    expert = (0.9, 0.1, 0.55, 0.5)            # residuals +0.4→clamp+0.2, −0.4→−0.2, +0.05, 0
    tgt = residual_train_target(expert, base, SCALE, CLAMP, NM)
    got = compose_residual(base, tgt, SCALE, CLAMP, NM)
    exp = (0.7, 0.3, 0.55, 0.5)               # base + clamp(expert−base)
    assert all(abs(a - b) < 1e-9 for a, b in zip(got, exp)), (got, exp)


def test_untrained_hybrid_is_baseline():
    """Floor: an untrained WNN residual is 0, so the hybrid == the PD baseline."""
    base = (0.4, 0.6, 0.5, 0.55)
    neutral = (0.5, 0.5, 0.5, 0.5)            # EMPTY-cell decode
    assert compose_residual(base, neutral, SCALE, CLAMP, NM) == base


def _spec():
    return ControllerSpec(num_motors=4, levels_per_motor=16, bits_per_feature=8,
        input_window_k=4, state_neurons=4, state_bits_per_neuron=24, output_bits_per_neuron=24,
        delta_control=False, obs_tilt_p=True, obs_tilt_i=True)


def test_residual_dagger_runs_and_floors():
    spec, seed = _spec(), 0
    thr = fit_thresholds_from_pid_rollouts(spec, num_episodes=6, seed=seed)
    sc, oc = random_connectivity(spec, seed=seed)
    ec = EpisodeConfig(dt=0.001, steps_per_episode=800, max_initial_tilt_rad=math.radians(15.0),
        max_initial_yaw_rad=math.radians(15.0), max_initial_body_rate=0.5, max_initial_yaw_rate=0.3)

    pd = AttitudePID(_pd_config())
    _, pd_m = eval_closed_loop_reset(make_pid_action_fn(pd), pd.reset, ec, 12, seed + 7_000_000)

    un = WnnController(num_motors=4, levels_per_motor=16, bits_per_feature=8, input_window_k=4,
        state_neurons=4, state_bits_per_neuron=24, output_bits_per_neuron=24, thresholds=thr,
        state_connections=sc, output_connections=oc, delta_control=False,
        obs_tilt_p=True, obs_tilt_i=True)
    cfg = DaggerConfig(residual=True, residual_baseline="pd", residual_scale=SCALE, residual_clamp=CLAMP,
        num_iterations=2, episodes_per_iter=4, steps_per_episode=800, eval_episodes=12,
        episode_config=ec, seed=seed, progress=False)
    _, un_m = _eval_closed_loop_residual(un, AttitudePID(_pd_config()), cfg, NM)

    # Untrained hybrid tracks the PD baseline closed-loop (the floor, within noise).
    assert abs(un_m["stable_rate"] - pd_m["stable_rate"]) < 0.10, (un_m["stable_rate"], pd_m["stable_rate"])

    ctrl, stats = train_dagger(spec, thr, sc, oc, cfg)
    assert stats["train_steps"] > 0 and stats["iter_cells_written"][0] > 0
    assert len(stats["iter_fitness"]) == 2


if __name__ == "__main__":
    test_compose_inverts_target(); print("compose/target inverse: PASS")
    test_untrained_hybrid_is_baseline(); print("floor (untrained==baseline): PASS")
    test_residual_dagger_runs_and_floors(); print("residual-DAGGER e2e: PASS")
    print("ALL PASS")
