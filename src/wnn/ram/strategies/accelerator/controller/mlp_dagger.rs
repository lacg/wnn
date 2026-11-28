//! CTRL-84 (ABI 71): DAgger with an MLP student — the WNN's trainer loop with ONE stage swapped.
//!
//! `dagger_train_inplace_with` (dagger_train.rs) runs, per round: roll out
//! `episodes_per_round` episodes with the STUDENT flying (teacher labels every visited state),
//! gate them, write the survivors into the RAM tables, closed-loop eval, best-checkpoint. This
//! loop is the same, stage for stage, with the SAME helpers (same RNG stream, same episode ICs,
//! teacher bank, tilt curriculum, gate, eval, checkpoint rule). Only the write differs: the
//! gated trajectories are REPLAYED through the same window preparation the WNN's walk consumes
//! (`bptt_prepare_checked` over `traj_windows`) and emitted as records
//! (pre-thermometer feature vector, per-bank raw-decode target) — the target is exactly the
//! value whose thermometer the WNN writes (`output_target_at`). The records go to a FIT callback
//! (Python: PyTorch on the AGGREGATED dataset — DAgger's data aggregation), which returns the new
//! head; the next round's rollouts fly it through `WnnController::step` (the Rust forward).
//!
//! sn=0 only (the MLP replaces the memoryless output layer; set_mlp_head_core refuses sn>0), so
//! the WNN loop's state-split / GPU-walk branches have no counterpart here.

use crate::controller::WnnController;
use crate::dagger_train::{
	episode_passes_gate_rs, eval_closed_loop_rs, rollout_and_label_rs, sim_default, traj_windows,
	validate_train_regime, AirframeRs, RewardGatedConfigPacked, TeacherBank, TrainStats, TrajectoryRs,
};
use crate::mlp_head::{MlpAct, MlpHead};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rand::rngs::SmallRng;
use rand::SeedableRng;

/// One round's NEW records: `feats` [n * in_dim], `targets` [n * out_dim].
#[derive(Default)]
pub(crate) struct MlpRecords
{
	pub(crate) feats: Vec<f32>,
	pub(crate) targets: Vec<f32>,
	pub(crate) n: usize,
}

/// Replay one trajectory exactly as train_on_trajectory_rs feeds the walk, recording
/// (features, targets) per decision record instead of writing cells.
pub(crate) fn record_trajectory(controller: &mut WnnController, traj: &TrajectoryRs, cfg: &RewardGatedConfigPacked, out: &mut MlpRecords)
{
	for w in traj_windows(traj, cfg)
	{
		let plan = controller.bptt_prepare_checked(
			&w.g,
			&w.a,
			&w.tg,
			&w.pp,
			w.first,
			w.ig.as_deref(),
			traj.init_yaw,
			w.sp.as_deref(),
			w.lb.as_deref(),
			w.ro.as_ref(),
		);
		if let Some(p) = plan
		{
			out.feats.extend_from_slice(&p.feats);
			controller.mlp_targets_into(&p, &w.pp, w.lb.as_deref(), &mut out.targets);
			out.n += p.n_rec;
		}
	}
}

/// The loop. `fit(records, round)` returns the new head (None = keep the current one).
pub(crate) fn dagger_train_mlp_with(
	controller: &mut WnnController,
	cfg: &RewardGatedConfigPacked,
	target: [f32; 3],
	seed: u64,
	fit: &mut dyn FnMut(&MlpRecords, usize) -> Result<Option<MlpHead>, String>,
) -> Result<TrainStats, String>
{
	validate_train_regime(controller, cfg)?;
	if controller.mlp_head_ref().is_none()
	{
		return Err("dagger_train_mlp: the controller has no MLP head (install the initial head first)".into());
	}
	let mut rng = SmallRng::seed_from_u64(seed);
	let af = AirframeRs::from_cfg(cfg);
	let mut teachers = TeacherBank::new(af);
	let mut sim = sim_default(cfg);
	controller.set_observer_actuator_core(sim.actuator());
	let mut stats = TrainStats::default();
	let mut history_scores: Vec<f64> = Vec::new();
	let mut best_fit = f64::NEG_INFINITY;
	let mut best_head: Option<MlpHead> = None;
	for it in 0..cfg.num_rounds
	{
		let tilt_rad = cfg.round_tilt_rad(it);
		let mut trajs: Vec<TrajectoryRs> = Vec::with_capacity(cfg.episodes_per_round);
		for ep in 0..cfg.episodes_per_round
		{
			let teacher = teachers.get_mut(cfg.teacher_id_for(it, ep));
			trajs.push(rollout_and_label_rs(controller, teacher, &mut sim, cfg, tilt_rad, &mut rng, target));
		}
		let round_scores: Vec<f64> = trajs.iter().map(|t| t.cumulative_reward).collect();
		let mean_ep_reward = round_scores.iter().sum::<f64>() / round_scores.len().max(1) as f64;
		let mut rec = MlpRecords::default();
		let mut n_trained = 0usize;
		for traj in &trajs
		{
			if episode_passes_gate_rs(traj.cumulative_reward, &round_scores, &history_scores, cfg)
			{
				record_trajectory(controller, traj, cfg, &mut rec);
				n_trained += 1;
				stats.train_steps += traj.steps;
			}
		}
		if let Some(h) = fit(&rec, it)?
		{
			controller.set_mlp_head_core(h)?;
		}
		history_scores.extend_from_slice(&round_scores);
		let (n_motors, lvls) = (controller.num_motors(), controller.levels_per_motor());
		let (fit_v, mean_err_rad, stable_rate, mean_jerk, mean_mono) =
			eval_closed_loop_rs(controller, &mut sim, cfg, &mut rng, target, cfg.full_tilt_deg.to_radians(), n_motors, lvls);
		stats.iter_fitness.push(fit_v);
		stats.iter_mean_err_deg.push(mean_err_rad.to_degrees());
		stats.iter_stable_rate.push(stable_rate);
		stats.iter_tilt_deg.push(tilt_rad.to_degrees());
		stats.iter_n_trained.push(n_trained);
		// MLP: "cells written" carries the round's NEW records (documented on the PyO3 entry).
		stats.iter_cells_written.push(rec.n);
		stats.iter_mean_episode_reward.push(mean_ep_reward);
		stats.iter_motor_jerk_mean.push(mean_jerk);
		stats.iter_mono_violations.push(mean_mono);
		if cfg.keep_best_checkpoint && fit_v > best_fit
		{
			best_fit = fit_v;
			best_head = controller.mlp_head_ref().cloned();
		}
	}
	if let Some(h) = best_head
	{
		controller.set_mlp_head_core(h)?;
	}
	Ok(stats)
}

fn f32_bytes<'py>(py: Python<'py>, v: &[f32]) -> Bound<'py, PyBytes>
{
	let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
	PyBytes::new(py, &b)
}

fn bytes_f32(b: &[u8], what: &str) -> Result<Vec<f32>, String>
{
	if b.len() % 4 != 0
	{
		return Err(format!("dagger_train_mlp: {what} is {} bytes, not a whole number of float32", b.len()));
	}
	Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

/// CTRL-84 (ABI 71): DAgger-train the controller's MLP head in place (one fold; call once per
/// fold seed, as dagger_train_batch_inplace does for the WNN). The controller must carry the
/// INITIAL head (WnnController.set_mlp_head). After each round's gated rollouts are recorded,
/// `fit_callback(feats, targets, n, in_dim, out_dim, round)` is called with the round's NEW
/// records as little-endian float32 bytes (feats [n, in_dim], targets [n, out_dim] — the
/// per-bank raw-decode targets in [0, 1]); it returns None (keep the head) or
/// (hidden: list[int], activation: str, shift: bytes, scale: bytes, params: bytes) — the new
/// head, which the next round flies. Returns TrainStats with iter_cells_written = the round's
/// record count (no cells are written).
#[pyfunction]
#[pyo3(signature = (controller, cfg, fit_callback, target_rpy = [0.0, 0.0, 0.0], seed = 0))]
pub fn dagger_train_mlp(
	py: Python<'_>,
	controller: &mut WnnController,
	cfg: RewardGatedConfigPacked,
	fit_callback: PyObject,
	target_rpy: [f32; 3],
	seed: u64,
) -> PyResult<TrainStats>
{
	let in_dim = controller.num_features();
	let out_dim = controller.num_motors();
	let mut py_err: Option<PyErr> = None;
	let mut fit = |rec: &MlpRecords, round: usize| -> Result<Option<MlpHead>, String> {
		let args = (f32_bytes(py, &rec.feats), f32_bytes(py, &rec.targets), rec.n, in_dim, out_dim, round);
		let r = match fit_callback.call1(py, args)
		{
			Ok(r) => r,
			Err(e) =>
			{
				let msg = format!("dagger_train_mlp: fit_callback raised: {e}");
				py_err = Some(e);
				return Err(msg);
			}
		};
		if r.is_none(py)
		{
			return Ok(None);
		}
		let r = r.bind(py);
		let (hidden, act, shift, scale, params): (Vec<usize>, String, Bound<'_, PyBytes>, Bound<'_, PyBytes>, Bound<'_, PyBytes>) =
			r.extract().map_err(|e| format!("dagger_train_mlp: fit_callback must return None or (hidden, act, shift: bytes, scale: bytes, params: bytes): {e}"))?;
		MlpHead::new(
			in_dim,
			hidden,
			out_dim,
			MlpAct::parse(&act)?,
			bytes_f32(shift.as_bytes(), "shift")?,
			bytes_f32(scale.as_bytes(), "scale")?,
			bytes_f32(params.as_bytes(), "params")?,
		)
		.map(Some)
	};
	let res = dagger_train_mlp_with(controller, &cfg, target_rpy, seed, &mut fit);
	match (res, py_err)
	{
		(Ok(s), _) => Ok(s),
		(Err(_), Some(e)) => Err(e),
		(Err(m), None) => Err(pyo3::exceptions::PyValueError::new_err(m)),
	}
}
