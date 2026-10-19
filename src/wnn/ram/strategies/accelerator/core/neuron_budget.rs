//! Whole-genome neuron budget (flow param `max_neurons_total`, 19/10/2026).
//!
//! `max_neurons` caps each CLUSTER, so a K-cluster genome can reach K x max
//! (1m <= 500, 2m <= 1000 at max 500). The budget caps the SUM. An over-budget
//! genome shrinks every cluster in proportion to its size (largest-remainder
//! apportionment, deterministic, no RNG), never below `min_neurons` (a cluster
//! already below it is left alone), and drops neurons from the END of each
//! cluster — the same neurons the neurons mutation drops. Within budget the
//! genome is returned unchanged, so the cluster split stays free: the GA can
//! still grow one cluster at another's expense.

/// Per-cluster counts that fit `total`. Unchanged when the sum already fits.
/// If even the floors exceed `total` the floors are returned (callers validate
/// total >= clusters x min_neurons up front).
pub fn budget_counts(neurons: &[usize], total: usize, min_neurons: usize) -> Vec<usize>
{
	let sum: usize = neurons.iter().sum();
	if sum <= total
	{
		return neurons.to_vec();
	}
	let floors: Vec<usize> = neurons.iter().map(|&n| n.min(min_neurons)).collect();
	let share = |n: usize| (n as u128 * total as u128, sum as u128);
	let mut out: Vec<usize> = neurons
		.iter()
		.zip(&floors)
		.map(|(&n, &f)| {
			let (num, den) = share(n);
			((num / den) as usize).max(f)
		})
		.collect();
	let used: usize = out.iter().sum();
	if used < total
	{
		hand_out_remainders(&mut out, neurons, total - used, |n| share(n));
	}
	else
	{
		take_back(&mut out, &floors, used - total);
	}
	out
}

/// Give `left` single neurons to the clusters with the largest fractional
/// share (ties: lowest index), never above the cluster's own count.
fn hand_out_remainders(out: &mut [usize], neurons: &[usize], mut left: usize, share: impl Fn(usize) -> (u128, u128))
{
	let mut order: Vec<usize> = (0..out.len()).collect();
	order.sort_by_key(|&c| {
		let (num, den) = share(neurons[c]);
		(std::cmp::Reverse(num % den), c)
	});
	for &c in order.iter().cycle().take(order.len() * 2)
	{
		if left == 0
		{
			break;
		}
		if out[c] < neurons[c]
		{
			out[c] += 1;
			left -= 1;
		}
	}
}

/// Remove `excess` neurons one at a time from the largest cluster still above
/// its floor (ties: lowest index).
fn take_back(out: &mut [usize], floors: &[usize], mut excess: usize)
{
	while excess > 0
	{
		let pick = (0..out.len()).filter(|&c| out[c] > floors[c]).max_by_key(|&c| (out[c], std::cmp::Reverse(c)));
		match pick
		{
			Some(c) =>
			{
				out[c] -= 1;
				excess -= 1;
			}
			None => return,
		}
	}
}

/// Fit a genome (bits per neuron, neurons per cluster, connections) to `total`,
/// truncating each cluster from its end. Unchanged when it already fits.
pub fn fit_genome(bits: &[usize], neurons: &[usize], conns: &[i64], total: usize, min_neurons: usize) -> (Vec<usize>, Vec<usize>, Vec<i64>)
{
	let target = budget_counts(neurons, total, min_neurons);
	if target == neurons
	{
		return (bits.to_vec(), neurons.to_vec(), conns.to_vec());
	}
	let has_conns = !conns.is_empty();
	let (mut nb, mut nc) = (Vec::new(), Vec::new());
	let (mut n_idx, mut c_idx) = (0usize, 0usize);
	for (c, &count) in neurons.iter().enumerate()
	{
		for local in 0..count
		{
			let b = bits[n_idx];
			if local < target[c]
			{
				nb.push(b);
				if has_conns
				{
					nc.extend_from_slice(&conns[c_idx..c_idx + b]);
				}
			}
			n_idx += 1;
			c_idx += b;
		}
	}
	(nb, target, nc)
}

#[cfg(test)]
#[path = "neuron_budget_tests.rs"]
mod tests;
