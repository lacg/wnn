//! CTRL-84 (ABI 71): the MLP baseline's policy head — a dense network in place of the WNN's
//! output-layer lookup, everything else of the controller unchanged.
//!
//! WHERE IT SITS. `WnnController::step` builds the feature vector (`compute_features`, the ONE
//! source of truth for the observation), then the WNN thresholds it, addresses the output
//! neurons and decodes each bank's cells to a raw decode in [0, 1] (`raw_bank_decode`). With a
//! head set, that raw decode is the head's output instead: the downstream decode (CTBR bank
//! alphabet, delta law, leak, clamp, the inner firmware rate loop), the plant, the scorers and
//! the DAgger trainer are the SAME code. So the MLP and the WNN differ ONLY in the function
//! features -> per-bank raw decode.
//!
//! INPUT: the pre-thermometer feature vector (`last_feature_vector`), standardised
//! `z_i = (x_i - shift_i) * scale_i` (shift/scale fitted on the training records; a constant
//! feature gets scale 0, i.e. no input). OUTPUT: `num_motors` raw decodes, the linear output
//! clamped to [0, 1] — exactly the range the WNN's decode spans.
//!
//! ARITHMETIC (the CPU/GPU parity contract): f32 throughout; each neuron is
//! `acc = b; for i in 0..n_in { acc = fma(w[i], z[i], acc) }` in ascending `i`, then the
//! activation. `f32::mul_add` and Metal's `fma` are both the correctly rounded fused op, so
//! with ReLU the CPU and the kernel (controller_rollout.metal `mlp_forward`) agree BIT FOR BIT
//! on the same input; tanh goes through each side's own tanh and agrees to ~1e-6.
//!
//! PARAMETER LAYOUT (`params`, flat f32): per layer l (inputs n_l -> outputs n_{l+1}) the
//! row-major weights W_l[o][i] (o outer) then the biases b_l[o]; layers in order
//! input -> hidden_0 -> ... -> output. The GPU blob is `[shift | scale | params]`.

/// The hidden-layer activation (the output layer is linear, then clamped to [0, 1]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MlpAct
{
	Relu,
	Tanh,
}

impl MlpAct
{
	pub(crate) fn parse(s: &str) -> Result<Self, String>
	{
		match s
		{
			"relu" => Ok(Self::Relu),
			"tanh" => Ok(Self::Tanh),
			_ => Err(format!("mlp activation {s:?}: expected \"relu\" or \"tanh\"")),
		}
	}

	pub(crate) fn name(self) -> &'static str
	{
		match self
		{
			Self::Relu => "relu",
			Self::Tanh => "tanh",
		}
	}

	/// The kernel's `P.mlp_act` code.
	pub(crate) fn code(self) -> u32
	{
		match self
		{
			Self::Relu => 0,
			Self::Tanh => 1,
		}
	}

	#[inline]
	fn apply(self, v: f32) -> f32
	{
		match self
		{
			Self::Relu => v.max(0.0),
			Self::Tanh => v.tanh(),
		}
	}
}

/// The widest hidden layer the Metal kernel's thread-private buffers hold.
pub(crate) const MLP_MAX_WIDTH: usize = 1024;
/// Hidden layers the kernel unrolls (1 or 2). The CPU path takes the same bound so a head
/// that scores on the CPU always scores on the GPU.
pub(crate) const MLP_MAX_HIDDEN_LAYERS: usize = 2;

#[derive(Clone, Debug)]
pub(crate) struct MlpHead
{
	in_dim: usize,
	hidden: Vec<usize>,
	out_dim: usize,
	act: MlpAct,
	shift: Vec<f32>,
	scale: Vec<f32>,
	params: Vec<f32>,
}

/// Number of f32 parameters (weights + biases) of a dense net with these widths.
pub(crate) fn param_count(in_dim: usize, hidden: &[usize], out_dim: usize) -> usize
{
	layer_dims(in_dim, hidden, out_dim).iter().map(|&(i, o)| i * o + o).sum()
}

fn layer_dims(in_dim: usize, hidden: &[usize], out_dim: usize) -> Vec<(usize, usize)>
{
	let mut widths = Vec::with_capacity(hidden.len() + 2);
	widths.push(in_dim);
	widths.extend_from_slice(hidden);
	widths.push(out_dim);
	widths.windows(2).map(|w| (w[0], w[1])).collect()
}

impl MlpHead
{
	pub(crate) fn new(
		in_dim: usize,
		hidden: Vec<usize>,
		out_dim: usize,
		act: MlpAct,
		shift: Vec<f32>,
		scale: Vec<f32>,
		params: Vec<f32>,
	) -> Result<Self, String>
	{
		if in_dim == 0 || out_dim == 0
		{
			return Err(format!("mlp head: in_dim {in_dim} / out_dim {out_dim} must be > 0"));
		}
		if hidden.is_empty() || hidden.len() > MLP_MAX_HIDDEN_LAYERS
		{
			return Err(format!(
				"mlp head: {} hidden layers — 1..={MLP_MAX_HIDDEN_LAYERS} supported (the Metal kernel unrolls two)",
				hidden.len()
			));
		}
		if let Some(&w) = hidden.iter().find(|&&w| w == 0 || w > MLP_MAX_WIDTH)
		{
			return Err(format!("mlp head: hidden width {w} outside 1..={MLP_MAX_WIDTH}"));
		}
		if in_dim > MLP_MAX_WIDTH
		{
			return Err(format!("mlp head: in_dim {in_dim} > {MLP_MAX_WIDTH}"));
		}
		if shift.len() != in_dim || scale.len() != in_dim
		{
			return Err(format!(
				"mlp head: shift/scale lengths {}/{} != in_dim {in_dim}",
				shift.len(),
				scale.len()
			));
		}
		let need = param_count(in_dim, &hidden, out_dim);
		if params.len() != need
		{
			return Err(format!("mlp head: {} params, the widths need {need}", params.len()));
		}
		if !(shift.iter().chain(&scale).chain(&params).all(|v| v.is_finite()))
		{
			return Err("mlp head: non-finite shift/scale/params".into());
		}
		Ok(Self { in_dim, hidden, out_dim, act, shift, scale, params })
	}

	pub(crate) fn in_dim(&self) -> usize
	{
		self.in_dim
	}
	pub(crate) fn out_dim(&self) -> usize
	{
		self.out_dim
	}
	pub(crate) fn hidden(&self) -> &[usize]
	{
		&self.hidden
	}
	pub(crate) fn act(&self) -> MlpAct
	{
		self.act
	}
	pub(crate) fn params(&self) -> &[f32]
	{
		&self.params
	}
	pub(crate) fn shift(&self) -> &[f32]
	{
		&self.shift
	}
	pub(crate) fn scale(&self) -> &[f32]
	{
		&self.scale
	}

	/// The forward pass: `out[o]` = clamp(linear output o, 0, 1). `x.len() == in_dim`,
	/// `out.len() == out_dim` (caller-checked by WnnController::set_mlp_head_core).
	pub(crate) fn forward(&self, x: &[f32], out: &mut [f32])
	{
		let mut cur: Vec<f32> = (0..self.in_dim).map(|i| (x[i] - self.shift[i]) * self.scale[i]).collect();
		let dims = layer_dims(self.in_dim, &self.hidden, self.out_dim);
		let last = dims.len() - 1;
		let mut base = 0usize;
		for (l, &(n_in, n_out)) in dims.iter().enumerate()
		{
			let w = &self.params[base..base + n_in * n_out];
			let b = &self.params[base + n_in * n_out..base + n_in * n_out + n_out];
			let mut next = vec![0.0f32; n_out];
			for (o, slot) in next.iter_mut().enumerate()
			{
				let row = &w[o * n_in..(o + 1) * n_in];
				let mut acc = b[o];
				for i in 0..n_in
				{
					acc = row[i].mul_add(cur[i], acc);
				}
				*slot = if l == last { acc.clamp(0.0, 1.0) } else { self.act.apply(acc) };
			}
			base += n_in * n_out + n_out;
			cur = next;
		}
		out[..self.out_dim].copy_from_slice(&cur);
	}

	/// The kernel blob: `[shift | scale | params]`.
	pub(crate) fn gpu_blob(&self) -> Vec<f32>
	{
		let mut v = Vec::with_capacity(2 * self.in_dim + self.params.len());
		v.extend_from_slice(&self.shift);
		v.extend_from_slice(&self.scale);
		v.extend_from_slice(&self.params);
		v
	}

	/// Same widths/activation (the GPU population must be uniform in shape).
	pub(crate) fn same_shape(&self, o: &MlpHead) -> bool
	{
		self.in_dim == o.in_dim && self.hidden == o.hidden && self.out_dim == o.out_dim && self.act == o.act
	}

	/// Per-decision arithmetic: (multiply-accumulates, activations, f32 parameters read).
	pub(crate) fn cost(&self) -> (usize, usize, usize)
	{
		let dims = layer_dims(self.in_dim, &self.hidden, self.out_dim);
		let macs = dims.iter().map(|&(i, o)| i * o).sum::<usize>() + self.in_dim; // + the standardisation multiply
		let acts = self.hidden.iter().sum::<usize>();
		(macs, acts, self.params.len() + 2 * self.in_dim)
	}
}

#[cfg(test)]
mod tests
{
	use super::*;

	fn head(act: MlpAct, seed: u64) -> MlpHead
	{
		let (i, h, o) = (5usize, vec![7usize, 3], 4usize);
		let n = param_count(i, &h, o);
		let mut s = seed;
		let mut r = || {
			s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
			((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
		};
		let params: Vec<f32> = (0..n).map(|_| r()).collect();
		let shift: Vec<f32> = (0..i).map(|_| r()).collect();
		let scale: Vec<f32> = (0..i).map(|_| 1.0 + r()).collect();
		MlpHead::new(i, h, o, act, shift, scale, params).expect("head")
	}

	/// The forward equals a plain f64 evaluation of the same layout (catches a layout or
	/// index slip; f32 vs f64 rounding only).
	#[test]
	fn mlp_forward_matches_reference_layout()
	{
		for act in [MlpAct::Relu, MlpAct::Tanh]
		{
			let h = head(act, 7);
			let x = [0.3f32, -1.2, 0.0, 2.5, -0.4];
			let mut out = [0.0f32; 4];
			h.forward(&x, &mut out);
			let mut cur: Vec<f64> = (0..5).map(|i| ((x[i] - h.shift[i]) * h.scale[i]) as f64).collect();
			let dims = layer_dims(5, &[7, 3], 4);
			let mut base = 0;
			for (l, &(ni, no)) in dims.iter().enumerate()
			{
				let mut nx = vec![0.0f64; no];
				for o in 0..no
				{
					let mut a = h.params[base + ni * no + o] as f64;
					for i in 0..ni
					{
						a += h.params[base + o * ni + i] as f64 * cur[i];
					}
					nx[o] = if l == 2 { a.clamp(0.0, 1.0) } else if act == MlpAct::Relu { a.max(0.0) } else { a.tanh() };
				}
				base += ni * no + no;
				cur = nx;
			}
			for o in 0..4
			{
				assert!((out[o] as f64 - cur[o]).abs() < 1e-5, "{act:?} out {o}: {} vs {}", out[o], cur[o]);
			}
		}
	}

	#[test]
	fn mlp_forward_is_deterministic_and_clamped()
	{
		let h = head(MlpAct::Relu, 11);
		let x = [10.0f32, -10.0, 4.0, -3.0, 9.0];
		let (mut a, mut b) = ([0.0f32; 4], [0.0f32; 4]);
		h.forward(&x, &mut a);
		h.clone().forward(&x, &mut b);
		assert_eq!(a.map(f32::to_bits), b.map(f32::to_bits));
		assert!(a.iter().all(|v| (0.0..=1.0).contains(v)));
	}

	#[test]
	fn mlp_refusals_and_cost()
	{
		assert!(MlpHead::new(3, vec![], 4, MlpAct::Relu, vec![0.0; 3], vec![1.0; 3], vec![]).is_err());
		assert!(MlpHead::new(3, vec![2, 2, 2], 4, MlpAct::Relu, vec![0.0; 3], vec![1.0; 3], vec![0.0; 100]).is_err());
		assert!(MlpHead::new(3, vec![2], 4, MlpAct::Relu, vec![0.0; 3], vec![1.0; 3], vec![0.0; 5]).is_err());
		let n = param_count(24, &[64, 64], 4);
		assert_eq!(n, 24 * 64 + 64 + 64 * 64 + 64 + 64 * 4 + 4);
		let h = MlpHead::new(24, vec![64, 64], 4, MlpAct::Relu, vec![0.0; 24], vec![1.0; 24], vec![0.0; n]).unwrap();
		assert_eq!(h.cost(), (24 * 64 + 64 * 64 + 64 * 4 + 24, 128, n + 48));
		assert!(MlpAct::parse("gelu").is_err());
	}
}
