//! Gated evaluation (base RAM + gating model).
//!
//! Split out of adaptive.rs (D3, 11/06/2026).

use super::*;

/// Evaluate a SINGLE genome WITH gating, returning both gated and non-gated metrics.
///
/// This function:
/// 1. Trains base RAM on training data
/// 2. Trains gating model on training data (target gate = true only for target cluster)
/// 3. Evaluates WITHOUT gating → (ce, acc)
/// 4. Evaluates WITH gating → (gated_ce, gated_acc)
///
/// # Returns
/// (ce, accuracy, gated_ce, gated_accuracy)
#[allow(clippy::too_many_arguments)]
pub fn evaluate_genome_with_gating(
	bits_flat: &[usize],
	neurons_flat: &[usize],
	connections_flat: &[i64],
	num_clusters: usize,
	train_input_bits: &ram_core::packed_bits::PackedBits,
	train_targets: &[i64],
	train_negatives: &[i64],
	num_train: usize,
	num_negatives: usize,
	eval_input_bits: &ram_core::packed_bits::PackedBits,
	eval_targets: &[i64],
	num_eval: usize,
	total_input_bits: usize,
	settings: ram_core::neuron_memory::EvalSettings,
	neurons_per_gate: usize,
	bits_per_gate_neuron: usize,
	vote_threshold_frac: f32,
	gating_seed: u64,
) -> (f64, f64, f64, f64)
{
	let empty_value = settings.empty_value;
	use crate::gating::RAMGating;

	let epsilon = 1e-10f64;
	let memory_mode = settings.memory_mode;

	// ========================================================================
	// Step 1: Train base RAM (same as existing evaluation)
	// ========================================================================

	// Build config groups for this genome
	let bits_per_cluster: Vec<usize> = bits_flat.to_vec();
	let neurons_per_cluster: Vec<usize> = neurons_flat.to_vec();
	let groups = build_groups(&bits_per_cluster, &neurons_per_cluster);

	// Create hybrid memory for each config group
	let group_memories: Vec<GroupMemory> = groups
		.iter()
		.map(|g| GroupMemory::new(g.total_neurons(), g.bits, memory_mode))
		.collect();

	// Reorganize connections for coalescing if needed
	let connections: Vec<i64> = if use_coalesced_groups()
	{
		reorganize_connections_for_coalescing(
			connections_flat,
			&bits_per_cluster,
			&neurons_per_cluster,
			&groups,
		)
	}
	else
	{
		connections_flat.to_vec()
	};

	// Build cluster-to-group mapping
	let mut cluster_to_group: Vec<(usize, usize)> = vec![(0, 0); num_clusters];
	for (group_idx, group) in groups.iter().enumerate()
	{
		for (local_idx, &cluster_id) in group.cluster_ids.iter().enumerate()
		{
			cluster_to_group[cluster_id] = (group_idx, local_idx);
		}
	}

	// Train: iterate over training examples (parallel)
	let use_nudge = memory_mode != ram_core::neuron_memory::TERNARY;
	(0..num_train).into_par_iter().for_each(|ex_idx| {
		// Cooperative SIGTERM cancellation (added 31/05/2026): once the flag
		// is set, every remaining example is a no-op. The par_iter still
		// drains its work-stealing queue (rayon can't be cancelled mid-flight),
		// but each example becomes a single atomic load + early return —
		// total drain time is well under one second for any realistic
		// num_train, including the 46M flow.
		if ram_core::cancel::check_cancel()
		{
			return;
		}
		let input_bits = train_input_bits.packed_row(ex_idx);

		let true_cluster = train_targets[ex_idx] as usize;

		// Train positive example
		{
			let (group_idx, local_cluster) = cluster_to_group[true_cluster];
			let group = &groups[group_idx];
			let memory = &group_memories[group_idx];

			let actual_neurons = if let Some(ref an) = group.actual_neurons
			{
				an[local_cluster] as usize
			}
			else
			{
				group.neurons
			};

			let neuron_base = local_cluster * group.neurons;
			let conn_base = group.conn_offset + local_cluster * group.neurons * group.bits;

			for n in 0..actual_neurons
			{
				let conn_start = conn_base + n * group.bits;
				let address = ram_core::neuron_memory::compute_address_packed_bytes(
					input_bits,
					&connections[conn_start..],
					group.bits,
				);
				if use_nudge
				{
					memory.nudge(neuron_base + n, address, true);
				}
				else
				{
					memory.write(neuron_base + n, address, TRUE, false);
				}
			}
		}

		// Train negative examples.
		//
		// Single-cluster (binary IDS) encodes the FALSE direction via
		// train_targets[ex_idx] == 0 + nudge_direction in the positive branch
		// above; the negative loop is multi-class only (K > 1 with explicit
		// per-example negative cluster IDs in train_negatives). Skip when
		// there's only one cluster — defense against callers that mis-form
		// the train_negatives buffer (e.g., the FPGA export wrapper bug at
		// lib.rs:7662-7667 that pre-dated this guard, where row-indices got
		// mis-interpreted as cluster-ids and panicked at the indexing below).
		if cluster_to_group.len() == 1
		{
			// Inside a rayon closure (per-example), `return` exits this
			// closure invocation cleanly — equivalent to `continue` in a
			// regular for-loop.
			return;
		}
		let neg_start = ex_idx * num_negatives;
		for k in 0..num_negatives
		{
			let false_cluster = train_negatives[neg_start + k] as usize;
			if false_cluster == true_cluster
			{
				continue;
			}

			let (group_idx, local_cluster) = cluster_to_group[false_cluster];
			let group = &groups[group_idx];
			let memory = &group_memories[group_idx];

			let actual_neurons = if let Some(ref an) = group.actual_neurons
			{
				an[local_cluster] as usize
			}
			else
			{
				group.neurons
			};

			let neuron_base = local_cluster * group.neurons;
			let conn_base = group.conn_offset + local_cluster * group.neurons * group.bits;

			for n in 0..actual_neurons
			{
				let conn_start = conn_base + n * group.bits;
				let address = ram_core::neuron_memory::compute_address_packed_bytes(
					input_bits,
					&connections[conn_start..],
					group.bits,
				);
				if use_nudge
				{
					memory.nudge(neuron_base + n, address, false);
				}
				else
				{
					memory.write(neuron_base + n, address, FALSE, false);
				}
			}
		}
	});

	// ========================================================================
	// Step 2: Train gating model (parallel)
	// ========================================================================

	let gating = RAMGating::new(
		num_clusters,
		neurons_per_gate,
		bits_per_gate_neuron,
		total_input_bits,
		vote_threshold_frac,
		Some(gating_seed),
	);

	// Build target gates in parallel: for each example, target_gate[target] = true
	let target_gates_flat: Vec<bool> = (0..num_train)
		.into_par_iter()
		.flat_map(|ex_idx| {
			let target = train_targets[ex_idx] as usize;
			let mut gates = vec![false; num_clusters];
			if target < num_clusters
			{
				gates[target] = true;
			}
			gates
		})
		.collect();

	// Train gating using parallel batch training
	gating.train_batch(train_input_bits, &target_gates_flat, num_train, false);

	// ========================================================================
	// Step 3: Evaluate WITHOUT gating - pre-compute all scores
	// ========================================================================

	let all_scores: Vec<Vec<f64>> = (0..num_eval)
		.into_par_iter()
		.map(|ex_idx| {
			// Cancellation poll: skip the per-example work and return all-zero
			// scores. The aggregate below will still complete (rayon doesn't
			// support mid-iter cancel) but every remaining example becomes a
			// fast no-op.
			if ram_core::cancel::check_cancel()
			{
				return vec![0.0f64; num_clusters];
			}
			let input_bits = eval_input_bits.packed_row(ex_idx);

			let mut scores = vec![0.0f64; num_clusters];

			for (group_idx, group) in groups.iter().enumerate()
			{
				let memory = &group_memories[group_idx];

				for (local_cluster, &cluster_id) in group.cluster_ids.iter().enumerate()
				{
					let actual_neurons = if let Some(ref an) = group.actual_neurons
					{
						an[local_cluster] as usize
					}
					else
					{
						group.neurons
					};

					let neuron_base = local_cluster * group.neurons;
					let conn_base = group.conn_offset + local_cluster * group.neurons * group.bits;

					let mut sum = 0.0f32;
					for n in 0..actual_neurons
					{
						let conn_start = conn_base + n * group.bits;
						let address = ram_core::neuron_memory::compute_address_packed_bytes(
							input_bits,
							&connections[conn_start..],
							group.bits,
						);
						let cell = memory.read(neuron_base + n, address);
						sum += cell_to_weight(cell, memory_mode, empty_value);
					}

					scores[cluster_id] = (sum / actual_neurons as f32) as f64;
				}
			}

			scores
		})
		.collect();

	// Compute CE and accuracy without gating
	let (total_ce, total_correct): (f64, u64) = all_scores
		.par_iter()
		.enumerate()
		.map(|(ex_idx, scores)| {
			let target_idx = eval_targets[ex_idx] as usize;

			// Find prediction (argmax) for accuracy
			let predicted = scores
				.iter()
				.enumerate()
				.max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
				.map(|(idx, _)| idx)
				.unwrap_or(0);
			let correct: u64 = if predicted == target_idx { 1 } else { 0 };

			// Softmax and cross-entropy
			let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
			let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
			let sum_exp: f64 = exp_scores.iter().sum();

			let target_prob = exp_scores[target_idx] / sum_exp;
			let ce = -(target_prob + epsilon).ln();

			(ce, correct)
		})
		.reduce(|| (0.0, 0), |(ce1, c1), (ce2, c2)| (ce1 + ce2, c1 + c2));

	let ce = total_ce / num_eval as f64;
	let accuracy = total_correct as f64 / num_eval as f64;

	// ========================================================================
	// Step 4: Evaluate WITH gating
	// ========================================================================

	let (total_gated_ce, total_gated_correct): (f64, u64) = (0..num_eval)
		.into_par_iter()
		.map(|ex_idx| {
			let input_bits = eval_input_bits.packed_row(ex_idx);
			let target_idx = eval_targets[ex_idx] as usize;

			// Get scores for this example
			let scores = &all_scores[ex_idx];

			// Compute gates for this input
			let gates = gating.forward_single(input_bits);

			// Apply gates to scores (multiply)
			let gated_scores: Vec<f64> = scores
				.iter()
				.zip(gates.iter())
				.map(|(&s, &g)| s * g as f64)
				.collect();

			// Check if any gates are open
			let any_open: bool = gates.iter().any(|&g| g > 0.0);

			if !any_open
			{
				// No gates open - use original scores (fallback)
				let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
				let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
				let sum_exp: f64 = exp_scores.iter().sum();
				let target_prob = exp_scores[target_idx] / sum_exp;
				let ce = -(target_prob + epsilon).ln();

				let predicted = scores
					.iter()
					.enumerate()
					.max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
					.map(|(idx, _)| idx)
					.unwrap_or(0);
				let correct: u64 = if predicted == target_idx { 1 } else { 0 };

				(ce, correct)
			}
			else
			{
				// Gated evaluation
				let predicted = gated_scores
					.iter()
					.enumerate()
					.max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
					.map(|(idx, _)| idx)
					.unwrap_or(0);
				let correct: u64 = if predicted == target_idx { 1 } else { 0 };

				// Softmax on gated scores
				let max_score = gated_scores
					.iter()
					.cloned()
					.fold(f64::NEG_INFINITY, f64::max);
				let exp_scores: Vec<f64> = gated_scores
					.iter()
					.map(|&s| (s - max_score).exp())
					.collect();
				let sum_exp: f64 = exp_scores.iter().sum();

				let target_prob = exp_scores[target_idx] / sum_exp;
				let ce = -(target_prob + epsilon).ln();

				(ce, correct)
			}
		})
		.reduce(|| (0.0, 0), |(ce1, c1), (ce2, c2)| (ce1 + ce2, c1 + c2));

	let gated_ce = total_gated_ce / num_eval as f64;
	let gated_accuracy = total_gated_correct as f64 / num_eval as f64;

	(ce, accuracy, gated_ce, gated_accuracy)
}
