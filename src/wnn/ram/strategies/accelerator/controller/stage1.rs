// Stage1Cfg — the scope C stage 1 vertical channel as ONE config object.
//
// Exists so the scorers gain a single parameter instead of eight more
// positional ones (house rule: typed config over long parameter lists —
// rollout_one already takes 30+). Absent (None) ⇒ the attitude-only rollout,
// bit-identical to every result flown before 13/08/2026.
//
// The per-episode vectors are the CPU twin of the GPU's interleaved buffer 30
// and of training.sample_vertical_ics_flat — the single Python-side source of
// vertical draw order. Both scorers must read the SAME draws or CPU and GPU
// are not interchangeable, which is the invariant sample_ics_flat already
// carries for attitude.
//
// Mass is here because it is a randomized PLANT parameter (Luiz, 12/08:
// a controller observes that it is sinking, not its own mass). It is never a
// feature — see docs/scope_c_full_controller_spec.md.

pub struct Stage1Cfg {
	/// The altitude every episode is asked to hold (m). Episodes start at
	/// `init_z[ep]` relative to it, so a non-zero start IS the error to correct.
	pub target_altitude: f32,
	/// Reward weight on altitude error. 0.0 ⇒ the attitude-only reward,
	/// bit-identically (compute_reward_stage1 short-circuits).
	pub lambda_alt: f32,
	/// Per-episode draws, each `num_episodes` long.
	pub init_z: Vec<f32>,
	pub init_vz: Vec<f32>,
	pub mass: Vec<f32>,
	pub collective_frac: Vec<f32>,
}

impl Stage1Cfg {
	/// Refuse a malformed config loudly rather than index out of bounds deep in
	/// a rayon worker — and refuse a non-positive mass, which would make the
	/// vertical dynamics divide by ~0 and emit NaN altitudes.
	pub fn validate(&self, num_episodes: usize) -> Result<(), String> {
		for (name, len) in [
			("init_z", self.init_z.len()), ("init_vz", self.init_vz.len()),
			("mass", self.mass.len()), ("collective_frac", self.collective_frac.len()),
		] {
			if len != num_episodes {
				return Err(format!(
					"Stage1Cfg.{name} has {len} entries but there are {num_episodes} episodes \
					 — the per-episode vertical draws must line up with the attitude ICs"));
			}
		}
		if let Some(bad) = self.mass.iter().find(|m| !m.is_finite() || **m <= 0.0) {
			return Err(format!("Stage1Cfg.mass must be finite and > 0 kg, found {bad}"));
		}
		if !self.target_altitude.is_finite() || !self.lambda_alt.is_finite() {
			return Err("Stage1Cfg: target_altitude and lambda_alt must be finite".into());
		}
		Ok(())
	}

	/// Flatten to the GPU's interleaved layout: 4 floats per episode,
	/// [z0, vz0, mass, collective_frac] — buffer 30's contract.
	pub fn to_gpu_blob(&self) -> Vec<f32> {
		let mut out = Vec::with_capacity(self.init_z.len() * 4);
		for ep in 0..self.init_z.len() {
			out.push(self.init_z[ep]);
			out.push(self.init_vz[ep]);
			out.push(self.mass[ep]);
			out.push(self.collective_frac[ep]);
		}
		out
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn cfg(n: usize) -> Stage1Cfg {
		Stage1Cfg {
			target_altitude: 0.0, lambda_alt: 0.0,
			init_z: vec![0.1; n], init_vz: vec![-0.2; n],
			mass: vec![0.0393; n], collective_frac: vec![0.05; n],
		}
	}

	#[test]
	fn validate_catches_length_and_mass_errors() {
		assert!(cfg(4).validate(4).is_ok());
		assert!(cfg(4).validate(5).is_err(), "a length mismatch must be refused");
		let mut bad = cfg(3);
		bad.mass[1] = 0.0;
		assert!(bad.validate(3).is_err(), "zero mass would divide the vertical dynamics by ~0");
		let mut nan = cfg(3);
		nan.lambda_alt = f32::NAN;
		assert!(nan.validate(3).is_err());
	}

	/// The GPU blob layout IS buffer 30's contract — interleaved 4/episode in
	/// the order the shader indexes (e*4 + {0,1,2,3}).
	#[test]
	fn gpu_blob_is_interleaved_in_shader_order() {
		let c = Stage1Cfg {
			target_altitude: 0.0, lambda_alt: 0.0,
			init_z: vec![1.0, 5.0], init_vz: vec![2.0, 6.0],
			mass: vec![3.0, 7.0], collective_frac: vec![4.0, 8.0],
		};
		assert_eq!(c.to_gpu_blob(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
	}
}
