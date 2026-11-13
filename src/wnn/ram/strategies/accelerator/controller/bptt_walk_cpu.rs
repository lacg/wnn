//! ABI 35: `bptt_train_window` split into PREPARE (memory-independent, shared with the
//! GPU walk) and the CPU WALK (the reference the GPU walk is parity-gated against).
//!
//! A child module of `controller` so it can reach WnnController's private fields
//! without widening them. Semantics are the pre-ABI-35 window walk, verbatim, with
//! two exact speedups (item B):
//!   * the solver reads a window-lived SolverView instead of cloning every neuron's
//!     DashMap and hashing it into a HashSet on every record;
//!   * the 4 per-motor solves and the (b) state solve run concurrently (pure reads of
//!     memory as of the record; they meet only in the vote/aggregate, whose integer
//!     sums do not depend on evaluation order).

use ram_core::cell_mode::{CellMode, UnwrittenLean};
use super::*;
use crate::bptt_plan::{BpttL4, BpttWindowPlan, NO_STATE_OVERRIDE};
use crate::controller_training::solve_qsr_reachable_view;
use crate::solver_view::SolverView;
use rayon::prelude::*;

impl WnnController
{
	/// Window geometry shared by both walks: (frame_bits, sensor_window,
	/// state_input_len, out_input_len, num_out).
	pub(crate) fn bptt_geometry(&self) -> (usize, usize, usize, usize, usize)
	{
		let frame_bits = self.num_features * self.bits_per_feature;
		let sensor_window = self.input_window_k * frame_bits;
		(
			frame_bits,
			sensor_window,
			sensor_window + self.state_neurons,
			self.output_input_len(),
			self.num_motors * self.levels_per_motor,
		)
	}

	/// bptt_train_window's entry contract for ONE window (yaw anchor + the fail-LOUD
	/// replay guards), then prepare. Shared by the CPU path and the GPU round planner,
	/// so both see exactly the same windows.
	#[allow(clippy::too_many_arguments)]
	pub(crate) fn bptt_prepare_checked(
		&mut self,
		gyros: &[[f32; 3]],
		accels: &[[f32; 3]],
		targets: &[[f32; 3]],
		pid_pwms: &[[f32; 4]],
		reset_state: bool,
		state_integral_targets: Option<&[[f32; 3]]>,
		init_yaw: f32,
		student_pwms: Option<&[[f32; 4]]>,
		label_base: Option<&[[f32; 4]]>,
		replay_obs: Option<&ReplayObs>,
	) -> Option<BpttWindowPlan>
	{
		// Yaw-anchor: single-window bptt ⇒ pending_init_yaws holds just this traj's yaw.
		self.pending_init_yaws = vec![init_yaw];
		let w = gyros.len();
		if w == 0
		{
			return None;
		}
		// DOB Fix A guard: an observer-on replay without the applied stream would
		// silently reproduce the frozen-accumulator divergence — the exact class
		// of completes-cleanly-measures-nothing trap this project keeps hitting.
		// Fail LOUDLY instead.
		if self.dhat_b.is_some()
		{
			let ok = student_pwms.map(|s| s.len() == w).unwrap_or(false);
			assert!(
				ok,
				"bptt_train_window: obs_dhat is on but student_pwms is missing or \
				 misaligned (got {:?}, need {} steps) — the replay observer would \
				 diverge from deploy (Fix A, 12/08/2026)",
				student_pwms.map(|s| s.len()),
				w
			);
		}
		// True-delta label guard: same fail-LOUD shape as Fix A. A missing
		// baseline would silently measure the delta from nothing.
		if self.dagger_label_delta
		{
			assert!(
				self.state_neurons == 0,
				"dagger_label_delta is sn=0 only — the split trainer labels absolute"
			);
			let ok = label_base.map(|b| b.len() == w).unwrap_or(false);
			assert!(
				ok,
				"bptt_train_window: dagger_label_delta is on but label_base is missing or \
				 misaligned (got {:?}, need {} steps)",
				label_base.map(|b| b.len()),
				w
			);
		}
		self.replay_obs_guard(replay_obs, w, "bptt_train_window");
		self.bptt_prepare_window(
			gyros,
			accels,
			targets,
			pid_pwms,
			reset_state,
			state_integral_targets,
			student_pwms,
			label_base,
			replay_obs,
		)
	}

	// ---- accessors for the GPU walk host (bptt_walk_gpu.rs) ----
	pub(crate) fn bptt_layer(&self, state: bool) -> &SparseLayerMemory
	{
		if state
		{
			&self.state_memory
		}
		else
		{
			&self.output_memory
		}
	}
	pub(crate) fn bptt_conns(&self) -> (&[i64], &[i64])
	{
		(&self.state_connections, &self.output_connections)
	}
	pub(crate) fn bptt_prev_state(&self) -> &[u8]
	{
		&self.prev_state
	}
	pub(crate) fn bptt_set_prev_state(&mut self, v: &[u8])
	{
		self.prev_state.copy_from_slice(v);
	}
	/// (num_motors, levels, obpn, sn, sbpn, memory_mode, output_full_window)
	pub(crate) fn bptt_dims(&self) -> (usize, usize, usize, usize, usize, u8, bool)
	{
		(
			self.num_motors,
			self.levels_per_motor,
			self.output_bits_per_neuron,
			self.state_neurons,
			self.state_bits_per_neuron,
			self.memory_mode,
			self.output_full_window,
		)
	}

	/// The memory-independent half of the old forward roll: features, frames, the
	/// K-frame ring, action-repeat holds, observers — plus the teacher target bits and
	/// the Option-A state targets. Mutates the controller's feature state exactly as the
	/// old roll did (reset, ring, accumulators). Returns None on an empty window or a
	/// cancel (the old function returned (0, 0) at both points).
	#[allow(clippy::too_many_arguments)]
	pub(crate) fn bptt_prepare_window(
		&mut self,
		gyros: &[[f32; 3]],
		accels: &[[f32; 3]],
		targets: &[[f32; 3]],
		pid_pwms: &[[f32; 4]],
		reset_state: bool,
		state_integral_targets: Option<&[[f32; 3]]>,
		student_pwms: Option<&[[f32; 4]]>,
		label_base: Option<&[[f32; 4]]>,
		replay_obs: Option<&ReplayObs>,
	) -> Option<BpttWindowPlan>
	{
		let w = gyros.len();
		if w == 0
		{
			return None;
		}
		// Option A gate: env flag + targets present + length matches the window.
		let use_integral_target = state_integral_targets.map(|v| v.len() == w).unwrap_or(false)
			&& std::env::var("WNN_STATE_INTEGRAL_TARGET")
				.map(|s| s == "1")
				.unwrap_or(false);
		let bpf = self.bits_per_feature;
		let (frame_bits, sensor_window, _sil, _oil, num_out) = self.bptt_geometry();
		if reset_state
		{
			let iy = self.pending_init_yaws.first().copied().unwrap_or(0.0);
			self.reset(iy);
		}
		let mut plan = BpttWindowPlan {
			reset: reset_state,
			n_rec: 0,
			rec_step: Vec::with_capacity(w),
			sensor: Vec::with_capacity(w * sensor_window),
			frame: Vec::with_capacity(w * frame_bits),
			out_target: Vec::new(),
			state_target: None,
		};
		for t in 0..w
		{
			if ram_core::cancel::check_cancel()
			{
				return None;
			}
			self.apply_replay_obs(replay_obs, t);
			let feats = self.compute_features(gyros[t], accels[t], targets[t]);
			// DOB Fix A: record applied[t] for the NEXT step's observer input.
			if let Some(sp) = student_pwms
			{
				self.observe_applied(sp[t]);
			}
			// Action-repeat hold: tick the accumulators only — no ring push, no record.
			if self.action_repeat > 1
			{
				let hold = self.step_counter % self.action_repeat != 0;
				self.step_counter += 1;
				if hold
				{
					continue;
				}
			}
			let mut frame = vec![false; frame_bits];
			for f in 0..self.num_features
			{
				let v = feats[f];
				let row = f * bpf;
				for b in 0..bpf
				{
					frame[row + b] = v >= self.thresholds[row + b];
				}
			}
			if self.input_history.len() == self.input_window_k
			{
				self.input_history.pop_front();
			}
			self.input_history.push_back(frame.clone());
			let mut sensor = vec![false; sensor_window];
			let pad = self.input_window_k - self.input_history.len();
			for (i, fr) in self.input_history.iter().enumerate()
			{
				let slot = (pad + i) * frame_bits;
				sensor[slot..slot + frame_bits].copy_from_slice(fr);
			}
			plan.sensor.extend_from_slice(&sensor);
			plan.frame.extend_from_slice(&frame);
			plan.rec_step.push(t);
		}
		plan.n_rec = plan.rec_step.len();
		// Teacher targets, evaluated AFTER the roll — where the old backward pass read them.
		let levels = self.levels_per_motor;
		plan.out_target = Vec::with_capacity(plan.n_rec * num_out);
		for d in 0..plan.n_rec
		{
			let t = plan.rec_step[d];
			for m in 0..self.num_motors
			{
				let p = self.output_target_at(m, t, pid_pwms, label_base);
				for i in 0..levels
				{
					plan.out_target.push(self.otb(p, i));
				}
			}
		}
		if use_integral_target
		{
			plan.state_target = Some(self.integral_state_targets(&plan, state_integral_targets.unwrap()));
		}
		Some(plan)
	}

	/// Option A: per (record, state neuron) the thermometer level of the PID integral —
	/// 3 axes split across state_neurons/3 neurons; the remainder falls back to d_s.
	fn integral_state_targets(&self, plan: &BpttWindowPlan, integ: &[[f32; 3]]) -> Vec<u8>
	{
		let sn = self.state_neurons;
		let npa = sn / 3;
		let (t_cell, f_cell) = (CellMode::of(self.memory_mode).true_cell(), CellMode::of(self.memory_mode).false_cell());
		let mut out = vec![NO_STATE_OVERRIDE; plan.n_rec * sn];
		for d in 0..plan.n_rec
		{
			let ig = &integ[plan.rec_step[d]];
			for n in 0..sn
			{
				if npa > 0 && n < 3 * npa
				{
					let (axis, level) = (n / npa, n % npa);
					let norm = ((ig[axis] + 1.0) * 0.5).clamp(0.0, 1.0);
					out[d * sn + n] = if norm * (npa as f32) > (level as f32) { t_cell } else { f_cell };
				}
			}
		}
		out
	}

	/// Recurrent state roll over a prepared window: per record the state-layer input
	/// (sensor window + prev-state fire bits) and the output-layer input. Reads the
	/// state memory as of the window start (the backward commits come after).
	fn bptt_state_roll(&mut self, plan: &BpttWindowPlan) -> (Vec<Vec<bool>>, Vec<Vec<bool>>)
	{
		let (frame_bits, sensor_window, state_input_len, _oil, _no) = self.bptt_geometry();
		let mut rs: Vec<Vec<bool>> = Vec::with_capacity(plan.n_rec);
		let mut ro: Vec<Vec<bool>> = Vec::with_capacity(plan.n_rec);
		for d in 0..plan.n_rec
		{
			let mut in_state = vec![false; state_input_len];
			in_state[..sensor_window].copy_from_slice(&plan.sensor[d * sensor_window..(d + 1) * sensor_window]);
			for (n, &v) in self.prev_state.iter().enumerate()
			{
				in_state[sensor_window + n] = CellMode::of(self.memory_mode).fire_bit(v);
			}
			let mut new_state = vec![0u8; self.state_neurons];
			for n in 0..self.state_neurons
			{
				let cs = n * self.state_bits_per_neuron;
				let addr = compute_address_sparse(
					&in_state,
					&self.state_connections[cs..cs + self.state_bits_per_neuron],
					self.state_bits_per_neuron,
				);
				new_state[n] = self.state_memory.read_cell(n, addr);
			}
			let frame = &plan.frame[d * frame_bits..(d + 1) * frame_bits];
			ro.push(self.build_output_input(&in_state, frame, &new_state));
			rs.push(in_state);
			self.prev_state = new_state;
		}
		(rs, ro)
	}

	/// L4 walk order (sn=0 only); with both flags off this is (0..n_rec).rev().
	fn bptt_walk_order(&self, plan: &BpttWindowPlan, l4: &BpttL4) -> Vec<usize>
	{
		let n_rec = plan.n_rec;
		let att_ok = l4.att_errs.map(|v| v.len() == l4.window_len).unwrap_or(false);
		let active = (l4.write_priority_err || l4.write_err_floor_deg > 0.0) && self.state_neurons == 0 && att_ok;
		if !active
		{
			return (0..n_rec).rev().collect();
		}
		let ae = l4.att_errs.unwrap();
		let rs = &plan.rec_step;
		let floor_rad = l4.write_err_floor_deg.to_radians();
		let mut idx: Vec<usize> = (0..n_rec)
			.filter(|&d| l4.write_err_floor_deg <= 0.0 || ae[rs[d]] >= floor_rad)
			.collect();
		if l4.write_priority_err
		{
			idx.sort_by(|&a, &b| {
				ae[rs[a]]
					.partial_cmp(&ae[rs[b]])
					.unwrap_or(std::cmp::Ordering::Equal)
					.then(b.cmp(&a))
			});
		}
		else
		{
			idx.reverse();
		}
		idx
	}

	/// The CPU window walk — the parity reference for the GPU walk.
	pub(crate) fn bptt_walk_cpu(
		&mut self,
		plan: &BpttWindowPlan,
		topk_per_neuron: usize,
		protect_learned: bool,
		l4: &BpttL4,
	) -> (usize, usize)
	{
		let (frame_bits, sensor_window, state_input_len, out_input_len, num_out) = self.bptt_geometry();
		// A reset window starts from prev_state = 0. prepare() already did that when the
		// walk follows its own prepare; this makes a LIST of prepared plans walkable too.
		if plan.reset
		{
			self.prev_state.iter_mut().for_each(|v| *v = 0);
		}
		let (rec_state_input, rec_out_input) = self.bptt_state_roll(plan);
		let state_bits_in = self.state_neurons;
		let levels = self.levels_per_motor;
		let obpn = self.output_bits_per_neuron;
		let mode = self.memory_mode;
		let default_cell = CellMode::of(mode).default_cell(UnwrittenLean::True);
		// Views exist only where a solve runs (sn>0); sn=0 is the direct-write fast path.
		let mut views: Option<(SolverView, SolverView)> = if state_bits_in > 0
		{
			Some((
				SolverView::from_memory(&self.output_memory, default_cell),
				SolverView::from_memory(&self.state_memory, default_cell),
			))
		}
		else
		{
			None
		};
		let walk_order = self.bptt_walk_order(plan, l4);
		// CTRL-46: the open vote round (taken for the walk, restored at the end). Under
		// the vote the memory is frozen, so the views are never updated either.
		let mut vote = self.vote_tally.take();
		let mut s_writes = 0usize;
		let mut o_writes = 0usize;
		let mut d_next: Option<Vec<bool>> = None;
		for &d in &walk_order
		{
			if ram_core::cancel::check_cancel()
			{
				self.vote_tally = vote;
				return (s_writes, o_writes);
			}
			let tgt = &plan.out_target[d * num_out..(d + 1) * num_out];
			// (a)+(b) solves: pure reads of memory as of this record → concurrent.
			let d_s: Vec<bool> = if let Some((ov, sv)) = views.as_ref()
			{
				let solve_motor = |m: usize| -> Option<Vec<bool>> {
					solve_qsr_reachable_view(
						ov,
						m * levels,
						&self.output_connections[m * levels * obpn..(m + 1) * levels * obpn],
						levels,
						obpn,
						out_input_len,
						&rec_out_input[d],
						&tgt[m * levels..(m + 1) * levels],
						frame_bits,
						topk_per_neuron,
						mode,
					)
				};
				let solve_state = |dn: &Vec<bool>| -> Option<Vec<bool>> {
					let target_sides: Vec<bool> = (0..self.state_neurons).map(|n| dn[n]).collect();
					solve_qsr_reachable_view(
						sv,
						0,
						&self.state_connections,
						self.state_neurons,
						self.state_bits_per_neuron,
						state_input_len,
						&rec_state_input[d + 1],
						&target_sides,
						sensor_window,
						topk_per_neuron,
						mode,
					)
				};
				let (motor_sols, state_sol): (Vec<Option<Vec<bool>>>, Option<Option<Vec<bool>>>) = rayon::join(
					|| (0..self.num_motors).into_par_iter().map(solve_motor).collect(),
					|| d_next.as_ref().map(solve_state),
				);
				let mut vote = vec![0i32; state_bits_in];
				for sol in motor_sols.iter().flatten()
				{
					for i in 0..state_bits_in
					{
						vote[i] += if sol[frame_bits + i] { 1 } else { -1 };
					}
				}
				let d_out: Vec<bool> = (0..state_bits_in)
					.map(|i| match vote[i].cmp(&0)
					{
						std::cmp::Ordering::Greater => true,
						std::cmp::Ordering::Less => false,
						std::cmp::Ordering::Equal => rec_out_input[d][frame_bits + i],
					})
					.collect();
				match state_sol
				{
					None => d_out,
					Some(solved) =>
					{
						let d_trans: Vec<bool> = match solved
						{
							Some(sol) => (0..state_bits_in).map(|i| sol[sensor_window + i]).collect(),
							None => d_out.clone(),
						};
						(0..state_bits_in)
							.map(|i| if d_out[i] == d_trans[i] { d_out[i] } else { rec_state_input[d][sensor_window + i] })
							.collect()
					}
				}
			}
			else
			{
				Vec::new()
			};

			// (c) Commit STATE layer toward the desired state at the recorded address.
			let t_cell = CellMode::of(mode).true_cell();
			let f_cell = CellMode::of(mode).false_cell();
			for n in 0..self.state_neurons
			{
				let ov = plan.state_target.as_ref().map(|v| v[d * self.state_neurons + n]).unwrap_or(NO_STATE_OVERRIDE);
				let target_val: u8 = if ov != NO_STATE_OVERRIDE
				{
					ov
				}
				else if d_s[n]
				{
					t_cell
				}
				else
				{
					f_cell
				};
				let cs = n * self.state_bits_per_neuron;
				let addr = compute_address_sparse(
					&rec_state_input[d],
					&self.state_connections[cs..cs + self.state_bits_per_neuron],
					self.state_bits_per_neuron,
				);
				// CTRL-46: under the vote rule the visit goes to the round's tally and the
				// memory stays frozen (no write, no view update).
				if let Some(t) = vote.as_mut()
				{
					t.visit_state_value(n, addr, target_val, mode);
					continue;
				}
				let cur = self.state_memory.read_cell(n, addr);
				if protect_learned
					&& cur != ram_core::neuron_memory::EMPTY_U8
					&& CellMode::of(mode).fire_bit(cur) != CellMode::of(mode).fire_bit(target_val)
				{
					continue;
				}
				let nv = CellMode::of(mode).nudge_toward_value(cur, target_val);
				if nv != cur
				{
					self.state_memory.write_cell(n, addr, nv, true);
					if let Some((_, sv)) = views.as_mut()
					{
						sv.apply_write(n, addr, nv);
					}
					s_writes += 1;
				}
			}

			// (d) Commit OUTPUT layer toward the teacher's PWM at the recorded out address.
			for n in 0..num_out
			{
				let target_true = tgt[n];
				let cs = n * obpn;
				let addr = compute_address_sparse(&rec_out_input[d], &self.output_connections[cs..cs + obpn], obpn);
				if let Some(t) = vote.as_mut()
				{
					t.visit(false, n, addr, target_true);
					continue;
				}
				let cur = self.output_memory.read_cell(n, addr);
				if protect_learned
					&& cur != ram_core::neuron_memory::EMPTY_U8
					&& CellMode::of(mode).fire_bit(cur) != target_true
				{
					continue;
				}
				let nv = CellMode::of(mode).nudge_step(cur, target_true);
				if nv != cur
				{
					self.output_memory.write_cell(n, addr, nv, true);
					if let Some((ov, _)) = views.as_mut()
					{
						ov.apply_write(n, addr, nv);
					}
					o_writes += 1;
				}
			}
			d_next = Some(d_s);
		}
		self.vote_tally = vote;
		(s_writes, o_writes)
	}

	/// CTRL-46: open a DAgger round under `WriteRule::Vote(share)` — every walk until
	/// `vote_commit_round` records its visits in the tally and writes nothing.
	pub(crate) fn vote_begin_round(&mut self, share: ram_core::vote_share::VoteShare) -> Result<(), String>
	{
		if self.vote_tally.is_some()
		{
			return Err("vote_begin_round: a round is already open (missing vote_commit_round)".into());
		}
		self.vote_tally = Some(crate::vote_tally::VoteTally::for_mode(self.memory_mode, share)?);
		Ok(())
	}

	/// CTRL-46: commit the open round's tally (ram_core vote_net_to_cell) and close it.
	/// Returns (state cells changed, output cells changed); (0, 0) with no round open.
	pub(crate) fn vote_commit_round(&mut self) -> (usize, usize)
	{
		match self.vote_tally.take()
		{
			Some(t) => t.commit(&self.state_memory, &self.output_memory),
			None => (0, 0),
		}
	}

	/// The open round's tally (the GPU walk host adds its visits here).
	pub(crate) fn vote_tally_mut(&mut self) -> Option<&mut crate::vote_tally::VoteTally>
	{
		self.vote_tally.as_mut()
	}

	#[cfg(test)]
	pub(crate) fn vote_tally_ref(&self) -> Option<&crate::vote_tally::VoteTally>
	{
		self.vote_tally.as_ref()
	}
}
