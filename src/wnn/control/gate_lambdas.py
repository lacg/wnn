"""CTRL-17 G4 (26/09/2026): the λ_alt / λ_pos the DAgger GATE and the per-round
CHECKPOINT rank on.

Before G4 the trainer's gate and checkpoint scored attitude only: λ_pos was packed
but never read, λ_alt was not packed at all. Episodes that needed more tilt (a
larger |x0|) or more collective were systematically gated OUT, and the checkpoint
kept the round that learned the least translation.

RESOLUTION RULE (the documented default). The `--reward-lambda-*` flags are the
REWARD-SHAPING λ and default to UNSET (None):

* **explicit** (`--reward-lambda-alt X`, including an explicit 0): the gate uses X,
  and so does the scorer's reward, exactly as before. Every banked recipe passes
  `--reward-lambda-alt 0` explicitly, so its gate stays attitude-only on altitude.
* **unset**: the scorer's reward term stays 0 (the rank weights already carry the
  channel into fitness, scale-free), and the gate λ is DERIVED from the run's rank
  weights so that the gate weighs each channel the way the fitness does:

      λ_ch = (w_ch / w_att) · E[att_err²] / E[ch_err²]

  with w_att = the attitude rank weights (err_sq + stable + steady), and the two
  expectations taken over the episode's INITIAL-condition draws in the rollouts the
  gate actually scores: attitude U(−θ, θ) on roll and pitch (E = 2θ²/3, θ = the
  trainer's full tilt), altitude U(−s, s) (E = s²/3), horizontal U(−s, s)² radial
  (E = 2s²/3). So λ_alt = (w_alt/w_att)·2θ²/s_alt² and λ_pos = (w_pos/w_att)·θ²/s_xy².
  A channel with rank weight 0 derives λ = 0 (the attitude-only gate, bit-identical).

This is a normalization, not a tuned value: it makes a start at the channel's
typical offset cost the gate what a start at the typical tilt does, times the ratio
the run's own fitness puts between them. Sweeping λ (docs/scope_c_stage2_lambda_pos_
sweep.md) overrides it through the explicit flag.
"""

import math


def reward_lambda(args, name: str) -> float:
	"""The reward-shaping λ for the SCORER: the explicit flag, or 0.0 when unset."""
	v = getattr(args, name, None)
	return 0.0 if v is None else float(v)


def _attitude_weight(args) -> float:
	return (float(getattr(args, "fit_weight_err_sq", 0.0))
	        + float(getattr(args, "fit_weight_stable", 0.0))
	        + float(getattr(args, "fit_weight_steady", 0.0)))


def _derive(weight: float, w_att: float, attitude_e2: float, channel_e2: float,
            flag: str) -> float:
	"""(w/w_att) · E[att²]/E[ch²]; 0 for an unweighted channel; loud when undefined."""
	if weight <= 0.0:
		return 0.0
	if w_att <= 0.0 or channel_e2 <= 0.0:
		raise SystemExit(
			f"cannot derive the gate λ for {flag}: the attitude rank weight "
			f"({w_att}) and the channel's start spread (E={channel_e2}) must both be > 0. "
			f"Pass {flag} explicitly.")
	return (weight / w_att) * attitude_e2 / channel_e2


def resolve_gate_lambdas(args, ec, tilt_rad: float) -> tuple[float, float, str]:
	"""(λ_alt, λ_pos, provenance) for the DAgger gate + checkpoint of this run.
	(0, 0) when translation is off (the gate is attitude-only by construction)."""
	if not getattr(ec, "translation", False):
		return 0.0, 0.0, "translation off"
	w_att = _attitude_weight(args)
	att_e2 = 2.0 * tilt_rad * tilt_rad / 3.0
	s_alt = float(getattr(ec, "max_initial_alt_offset_m", 0.0))
	s_xy = float(getattr(ec, "max_initial_xy_offset_m", 0.0))
	src = []
	lam_alt = getattr(args, "reward_lambda_alt", None)
	if lam_alt is None:
		lam_alt = _derive(float(getattr(args, "fit_weight_alt", 0.0)), w_att, att_e2,
		                  s_alt * s_alt / 3.0, "--reward-lambda-alt")
		src.append("alt derived")
	else:
		src.append("alt explicit")
	lam_pos = getattr(args, "reward_lambda_pos", None)
	if lam_pos is None:
		lam_pos = 0.0 if s_xy <= 0.0 else _derive(
			float(getattr(args, "fit_weight_pos", 0.0)), w_att, att_e2,
			2.0 * s_xy * s_xy / 3.0, "--reward-lambda-pos")
		src.append("pos derived")
	else:
		src.append("pos explicit")
	return float(lam_alt), float(lam_pos), ", ".join(src)


def gate_lambda_line(args, ec, tilt_rad: float) -> str:
	"""The one-line provenance a run header / marker can grep."""
	a, p, src = resolve_gate_lambdas(args, ec, tilt_rad)
	return (f"[GATE-λ] DAgger gate + checkpoint rank on λ_alt={a:.6g} λ_pos={p:.6g} "
	        f"({src}; tilt {math.degrees(tilt_rad):.1f}°)")
