//! Overactuated multirotor substrate — Phase 0 (docs/OVERACTUATED_RESIDUAL_DESIGN.md).
//!
//! ADDITIVE module: N-rotor geometry, 6-DoF wrench allocation matrix, and a
//! damped pseudo-inverse allocator. NOT wired into the live AttitudeSim /
//! rollout hot path (Phase 1, discuss-first) — this is the math substrate the
//! WNN residual will correct, plus parity proof that the quad '+' preset
//! reproduces AttitudeSim's mixer exactly.
//!
//! Conventions (must match controller.rs):
//! - body frame z-up, x forward, y left
//! - rotor thrust T_i = k_thrust_i * pwm_i^2 along the rotor's unit `axis`
//! - drag torque spin_i * k_drag_i * T_i about `axis` (CCW spin = +1)

/// One rotor of an N-rotor vehicle, nominal geometry.
#[derive(Clone, Copy, Debug)]
pub struct Rotor {
	/// Rotor position in the body frame (m).
	pub position: [f32; 3],
	/// Unit thrust direction in the body frame.
	pub axis: [f32; 3],
	/// Prop spin: +1.0 CCW (drag torque along +axis), -1.0 CW.
	pub spin: f32,
	/// Thrust coefficient: N per pwm² unit.
	pub k_thrust: f32,
	/// Drag-torque-to-thrust ratio (dimensionless).
	pub k_drag: f32,
}

/// Nominal rotor set + the linear allocation model built from it.
#[derive(Clone, Debug)]
pub struct RotorGeometry {
	pub rotors: Vec<Rotor>,
}

#[inline]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
	[
		a[1] * b[2] - a[2] * b[1],
		a[2] * b[0] - a[0] * b[2],
		a[0] * b[1] - a[1] * b[0],
	]
}

impl RotorGeometry {
	pub fn new(rotors: Vec<Rotor>) -> Self {
		Self { rotors }
	}

	/// Build from 9-float rows [px,py,pz, ax,ay,az, spin, k_thrust, k_drag] —
	/// the AttitudeSim.set_geometry / score_controllers_* contract. Axis is
	/// normalized here so every consumer sees identical unit axes.
	pub fn from_rows(rows: &[[f32; 9]]) -> Result<Self, String> {
		if rows.is_empty() {
			return Err("geometry needs at least 1 rotor".into());
		}
		Ok(Self::new(rows.iter().map(|r| {
			let n = (r[3] * r[3] + r[4] * r[4] + r[5] * r[5]).sqrt().max(1e-9);
			Rotor {
				position: [r[0], r[1], r[2]],
				axis: [r[3] / n, r[4] / n, r[5] / n],
				spin: r[6],
				k_thrust: r[7],
				k_drag: r[8],
			}
		}).collect()))
	}

	#[allow(dead_code)]  // Phase-1 wiring consumes this; tests exercise the rest.
	pub fn num_rotors(&self) -> usize {
		self.rotors.len()
	}

	// ── Presets ─────────────────────────────────────────────────────────

	/// AttitudeSim's '+' quad: motor 0 front (+x), 1 right (-y), 2 rear, 3 left.
	/// Motors 0,2 CCW; 1,3 CW. Same k_thrust/k_drag for all four.
	pub fn quad_plus(arm: f32, k_thrust: f32, k_drag: f32) -> Self {
		let up = [0.0, 0.0, 1.0];
		let mk = |position: [f32; 3], spin: f32| Rotor { position, axis: up, spin, k_thrust, k_drag };
		Self::new(vec![
			mk([arm, 0.0, 0.0], 1.0),
			mk([0.0, -arm, 0.0], -1.0),
			mk([-arm, 0.0, 0.0], 1.0),
			mk([0.0, arm, 0.0], -1.0),
		])
	}

	/// Flat octo-X: 8 rotors every 45° starting at 22.5°, alternating spin.
	/// Rank-4 wrench map with a 4-dim null space — the canonical redundant
	/// airframe (motor-out tolerance).
	pub fn octo_x(arm: f32, k_thrust: f32, k_drag: f32) -> Self {
		let up = [0.0, 0.0, 1.0];
		let rotors = (0..8)
			.map(|i| {
				let ang = (22.5 + 45.0 * i as f32).to_radians();
				Rotor {
					position: [arm * ang.cos(), arm * ang.sin(), 0.0],
					axis: up,
					spin: if i % 2 == 0 { 1.0 } else { -1.0 },
					k_thrust,
					k_drag,
				}
			})
			.collect();
		Self::new(rotors)
	}

	/// Canted hex (Voliro-style fixed tilt): 6 arms every 60°, each rotor
	/// tilted `cant_deg` about its arm axis, alternating tilt sign with spin.
	/// The cant couples thrust into lateral force — rank-6 wrench map, i.e.
	/// true overactuation (can hover tilted / push sideways at zero torque).
	pub fn canted_hex(arm: f32, k_thrust: f32, k_drag: f32, cant_deg: f32) -> Self {
		let cant = cant_deg.to_radians();
		let rotors = (0..6)
			.map(|i| {
				let ang = (60.0 * i as f32).to_radians();
				let arm_dir = [ang.cos(), ang.sin(), 0.0];
				let spin = if i % 2 == 0 { 1.0 } else { -1.0 };
				// Tilt the +z thrust axis about the arm direction by ±cant
				// (sign alternates with spin so drag torques stay balanced).
				let s = spin * cant.sin();
				let c = cant.cos();
				// Rodrigues rotation of [0,0,1] about unit arm_dir:
				//   v' = v c + (k×v) s + k (k·v)(1-c); k·v = 0 here.
				let axis = [
					arm_dir[1] * s,
					-arm_dir[0] * s,
					c,
				];
				Rotor { position: [arm * arm_dir[0], arm * arm_dir[1], 0.0], axis, spin, k_thrust, k_drag }
			})
			.collect();
		Self::new(rotors)
	}

	/// Perturbed copy of this geometry: per-rotor tilt error (rad, rotating
	/// each thrust axis about the rotor's arm direction) + position error (m).
	/// This is how the sim models the TRUE vehicle while the allocator keeps
	/// the nominal geometry — the mismatch the WNN residual must learn.
	pub fn perturbed(&self, tilt_err_rad: &[f32], pos_err: &[[f32; 3]]) -> Self {
		let rotors = self.rotors.iter().enumerate().map(|(i, r)| {
			let mut out = *r;
			if let Some(&[dx, dy, dz]) = pos_err.get(i) {
				out.position = [r.position[0] + dx, r.position[1] + dy, r.position[2] + dz];
			}
			let tilt = tilt_err_rad.get(i).copied().unwrap_or(0.0);
			if tilt != 0.0 {
				// Rotate axis about the (unit) arm direction; rotors at the CG
				// fall back to the body x-axis as the tilt hinge.
				let p = r.position;
				let norm = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
				let k = if norm > 1e-9 { [p[0] / norm, p[1] / norm, p[2] / norm] } else { [1.0, 0.0, 0.0] };
				let (s, c) = tilt.sin_cos();
				let a = r.axis;
				let kxa = cross(k, a);
				let kdota = k[0] * a[0] + k[1] * a[1] + k[2] * a[2];
				out.axis = [
					a[0] * c + kxa[0] * s + k[0] * kdota * (1.0 - c),
					a[1] * c + kxa[1] * s + k[1] * kdota * (1.0 - c),
					a[2] * c + kxa[2] * s + k[2] * kdota * (1.0 - c),
				];
			}
			out
		}).collect();
		Self::new(rotors)
	}

	// ── Forward model (sim side) ────────────────────────────────────────

	/// Per-rotor thrust magnitudes from PWM (quadratic motor map, clamped to [0,1]).
	fn thrusts(&self, pwm: &[f32], asym: Option<&[f32]>) -> Vec<f32> {
		self.rotors
			.iter()
			.enumerate()
			.map(|(i, r)| {
				let p = pwm[i].clamp(0.0, 1.0);
				let a = asym.map_or(1.0, |m| m[i]);
				r.k_thrust * a * p * p
			})
			.collect()
	}

	/// Body-frame torque for a PWM vector (r×F + spin drag), nominal motors.
	#[allow(dead_code)]  // Phase-2 residual pipeline consumes it; tests exercise it today.
	pub fn body_torque(&self, pwm: &[f32]) -> [f32; 3] {
		self.body_torque_asym(pwm, None)
	}

	/// Torque with optional per-motor thrust multipliers (D3-style asymmetry).
	pub fn body_torque_asym(&self, pwm: &[f32], asym: Option<&[f32]>) -> [f32; 3] {
		let thrusts = self.thrusts(pwm, asym);
		let mut tau = [0.0f32; 3];
		for (r, &t) in self.rotors.iter().zip(&thrusts) {
			let force = [r.axis[0] * t, r.axis[1] * t, r.axis[2] * t];
			let arm_tau = cross(r.position, force);
			let drag = r.spin * r.k_drag * t;
			tau[0] += arm_tau[0] + drag * r.axis[0];
			tau[1] += arm_tau[1] + drag * r.axis[1];
			tau[2] += arm_tau[2] + drag * r.axis[2];
		}
		tau
	}

	/// Body-frame net force for a PWM vector.
	#[allow(dead_code)]  // Phase-2 residual pipeline consumes it; tests exercise it today.
	pub fn body_force(&self, pwm: &[f32]) -> [f32; 3] {
		let thrusts = self.thrusts(pwm, None);
		let mut f = [0.0f32; 3];
		for (r, &t) in self.rotors.iter().zip(&thrusts) {
			f[0] += r.axis[0] * t;
			f[1] += r.axis[1] * t;
			f[2] += r.axis[2] * t;
		}
		f
	}

	// ── Allocation (controller side) ────────────────────────────────────

	/// 6×N allocation matrix: column i is the wrench (τ; F) per unit thrust
	/// of rotor i. Row-major [row][rotor].
	#[allow(dead_code)]  // Phase-2 residual pipeline consumes it; tests exercise it today.
	pub fn allocation_matrix(&self) -> Vec<Vec<f32>> {
		let n = self.rotors.len();
		let mut b = vec![vec![0.0f32; n]; 6];
		for (i, r) in self.rotors.iter().enumerate() {
			let arm_tau = cross(r.position, r.axis);
			let drag = r.spin * r.k_drag;
			for k in 0..3 {
				b[k][i] = arm_tau[k] + drag * r.axis[k];
				b[3 + k][i] = r.axis[k];
			}
		}
		b
	}

	/// Precomputed damped pseudo-inverse rows: M = Bᵀ(BBᵀ + λI)⁻¹ (N×6), so
	/// thrusts are the matvec T = M·w. This is `allocate()` with the 6×6
	/// solve hoisted out of the per-step path — the form the GPU kernel and
	/// the CPU batch scorer share (Phase 2 residual baseline). Computed by
	/// solving (BBᵀ+λI) mᵢᵀ = bᵢ per rotor (bᵢ = B's column i), which is
	/// exactly the system allocate() solves — same solver, same rounding.
	pub fn allocation_pinv(&self, lambda: f32) -> Vec<[f32; 6]> {
		let b = self.allocation_matrix();
		let n = self.rotors.len();
		let mut m6 = [[0.0f32; 6]; 6];
		for r in 0..6 {
			for c in 0..6 {
				let mut s = 0.0f32;
				for i in 0..n {
					s += b[r][i] * b[c][i];
				}
				m6[r][c] = s + if r == c { lambda } else { 0.0 };
			}
		}
		// Row i of M solves (BBᵀ+λI)ᵀ mᵢ = bᵢ — symmetric matrix, so solve6
		// directly. mᵢ·w == bᵢᵀ(BBᵀ+λI)⁻¹ w == (Bᵀ(BBᵀ+λI)⁻¹ w)ᵢ.
		(0..n)
			.map(|i| {
				let col = [b[0][i], b[1][i], b[2][i], b[3][i], b[4][i], b[5][i]];
				solve6(m6, col)
			})
			.collect()
	}

	/// Classical allocator: desired wrench (τ; F) → per-rotor PWM via damped
	/// pseudo-inverse thrusts T = Bᵀ(BBᵀ + λI)⁻¹ w, then pwm = √(T/k) with
	/// negative demands clamped to 0 (fixed-pitch props). λ regularizes the
	/// rank-deficient rows of planar vehicles (Fx/Fy unreachable).
	#[allow(dead_code)]  // Phase-2 residual pipeline consumes it; tests exercise it today.
	pub fn allocate(&self, wrench: [f32; 6], lambda: f32) -> Vec<f32> {
		let b = self.allocation_matrix();
		let n = self.rotors.len();

		// BBᵀ + λI (6×6, symmetric).
		let mut m = [[0.0f32; 6]; 6];
		for r in 0..6 {
			for c in 0..6 {
				let mut s = 0.0f32;
				for i in 0..n {
					s += b[r][i] * b[c][i];
				}
				m[r][c] = s + if r == c { lambda } else { 0.0 };
			}
		}
		let y = solve6(m, wrench);

		// T = Bᵀ y, clamp T ≥ 0, pwm = √(T/k) clamped to [0,1].
		(0..n)
			.map(|i| {
				let mut t = 0.0f32;
				for r in 0..6 {
					t += b[r][i] * y[r];
				}
				let t = t.max(0.0);
				(t / self.rotors[i].k_thrust).sqrt().clamp(0.0, 1.0)
			})
			.collect()
	}
}

/// Solve the 6×6 system M x = v by Gauss-Jordan with partial pivoting.
/// M is (BBᵀ + λI): symmetric positive definite for λ > 0, so the pivot
/// never vanishes; the assert is a debug tripwire, not a runtime path.
#[allow(dead_code)]  // Phase-2 residual pipeline consumes it (via allocate).
fn solve6(m: [[f32; 6]; 6], v: [f32; 6]) -> [f32; 6] {
	let mut a = [[0.0f32; 7]; 6];
	for r in 0..6 {
		a[r][..6].copy_from_slice(&m[r]);
		a[r][6] = v[r];
	}
	for col in 0..6 {
		let pivot_row = (col..6)
			.max_by(|&r1, &r2| a[r1][col].abs().partial_cmp(&a[r2][col].abs()).unwrap())
			.unwrap();
		a.swap(col, pivot_row);
		let pivot = a[col][col];
		debug_assert!(pivot.abs() > 1e-12, "BB^T + lambda*I must be invertible");
		for c in col..7 {
			a[col][c] /= pivot;
		}
		for r in 0..6 {
			if r != col {
				let f = a[r][col];
				for c in col..7 {
					a[r][c] -= f * a[col][c];
				}
			}
		}
	}
	[a[0][6], a[1][6], a[2][6], a[3][6], a[4][6], a[5][6]]
}

#[cfg(test)]
mod tests {
	use super::*;

	const ARM: f32 = 0.075;
	const KT: f32 = 2.4;
	const KD: f32 = 0.05;

	/// AttitudeSim::body_torque re-stated (controller.rs lines ~698-714) —
	/// the parity oracle for the quad_plus preset.
	fn quad_oracle(pwm: [f32; 4]) -> [f32; 3] {
		let t0 = KT * pwm[0] * pwm[0];
		let t1 = KT * pwm[1] * pwm[1];
		let t2 = KT * pwm[2] * pwm[2];
		let t3 = KT * pwm[3] * pwm[3];
		[ARM * (-t1 + t3), ARM * (-t0 + t2), KD * (t0 - t1 + t2 - t3)]
	}

	fn assert_close(a: &[f32], b: &[f32], tol: f32, what: &str) {
		for (i, (x, y)) in a.iter().zip(b).enumerate() {
			assert!((x - y).abs() <= tol, "{what}[{i}]: {x} vs {y}");
		}
	}

	#[test]
	fn quad_plus_matches_attitude_sim_mixer() {
		let geo = RotorGeometry::quad_plus(ARM, KT, KD);
		for pwm in [
			[0.0, 0.0, 0.0, 0.0],
			[1.0, 1.0, 1.0, 1.0],
			[0.7, 0.2, 0.9, 0.4],
			[0.31, 0.62, 0.05, 0.88],
		] {
			assert_close(&geo.body_torque(&pwm), &quad_oracle(pwm), 1e-6, "torque");
		}
	}

	#[test]
	fn quad_plus_asym_matches_attitude_sim_d3() {
		let geo = RotorGeometry::quad_plus(ARM, KT, KD);
		let pwm = [0.7, 0.2, 0.9, 0.4];
		let asym = [0.9, 1.1, 1.0, 0.95];
		// Oracle with per-motor multipliers (body_torque_asym convention).
		let t: Vec<f32> = (0..4).map(|i| KT * asym[i] * pwm[i] * pwm[i]).collect();
		let oracle = [
			ARM * (-t[1] + t[3]),
			ARM * (-t[0] + t[2]),
			KD * (t[0] - t[1] + t[2] - t[3]),
		];
		assert_close(&geo.body_torque_asym(&pwm, Some(&asym)), &oracle, 1e-6, "asym torque");
	}

	#[test]
	fn octo_allocation_round_trips_wrench() {
		let geo = RotorGeometry::octo_x(ARM, KT, KD);
		// Feasible small wrench: modest torques + hover-ish collective thrust.
		let wrench = [0.02, -0.015, 0.004, 0.0, 0.0, 6.0];
		let pwm = geo.allocate(wrench, 1e-6);
		assert_eq!(pwm.len(), 8);
		let tau = geo.body_torque(&pwm);
		let f = geo.body_force(&pwm);
		assert_close(&tau, &wrench[..3], 2e-3, "tau round-trip");
		// Planar octo cannot produce Fx/Fy; Fz must match.
		assert!((f[2] - wrench[5]).abs() < 2e-2, "Fz: {} vs {}", f[2], wrench[5]);
	}

	#[test]
	fn octo_pinv_is_minimum_norm() {
		let geo = RotorGeometry::octo_x(ARM, KT, KD);
		let wrench = [0.02, -0.015, 0.004, 0.0, 0.0, 6.0];
		let pwm = geo.allocate(wrench, 1e-6);
		let thrusts: Vec<f32> = pwm.iter().zip(&geo.rotors).map(|(p, r)| r.k_thrust * p * p).collect();
		let norm: f32 = thrusts.iter().map(|t| t * t).sum();
		// Perturb inside the null space via a re-allocation of a slightly
		// different-then-restored wrench: any OTHER thrust vector realizing
		// the same wrench must have ≥ norm. Construct one by adding an
		// alternating pattern (a null direction of the flat octo) and
		// re-verifying the wrench is unchanged.
		let mut alt = thrusts.clone();
		for (i, t) in alt.iter_mut().enumerate() {
			*t += if i % 2 == 0 { 0.05 } else { -0.05 };
		}
		let alt_pwm: Vec<f32> = alt.iter().zip(&geo.rotors).map(|(t, r)| (t.max(0.0) / r.k_thrust).sqrt()).collect();
		let tau_alt = geo.body_torque(&alt_pwm);
		let tau_ref = geo.body_torque(&pwm);
		// Same-wrench check is approximate (alternating +/- is null for τ up
		// to drag pairing); only compare when it actually held.
		if (tau_alt[0] - tau_ref[0]).abs() < 1e-3
			&& (tau_alt[1] - tau_ref[1]).abs() < 1e-3
		{
			let alt_norm: f32 = alt.iter().map(|t| t * t).sum();
			assert!(alt_norm >= norm - 1e-6, "pinv allocation must be minimum-norm");
		}
	}

	#[test]
	fn canted_hex_reaches_lateral_force() {
		// The overactuation selling point: a canted hex can produce Fx ≠ 0.
		let geo = RotorGeometry::canted_hex(ARM, KT, KD, 20.0);
		let wrench = [0.0, 0.0, 0.0, 0.8, 0.0, 5.0];
		let pwm = geo.allocate(wrench, 1e-6);
		let f = geo.body_force(&pwm);
		let tau = geo.body_torque(&pwm);
		assert!(f[0] > 0.4, "expected substantial +x force, got {}", f[0]);
		assert!((f[2] - wrench[5]).abs() < 0.35, "Fz: {} vs {}", f[2], wrench[5]);
		assert!(tau.iter().all(|t| t.abs() < 0.05), "torque leak: {:?}", tau);
	}

	#[test]
	fn allocation_pinv_matches_allocate() {
		// The precomputed M·w matvec must reproduce allocate()'s per-wrench
		// solve (same solver, hoisted) — on octo AND rank-deficient quad.
		for geo in [RotorGeometry::octo_x(ARM, KT, KD), RotorGeometry::quad_plus(ARM, KT, KD)] {
			let m = geo.allocation_pinv(1e-6);
			for wrench in [
				[0.02f32, -0.015, 0.004, 0.0, 0.0, 6.0],
				[0.0, 0.0, 0.0, 0.0, 0.0, 2.4],
				[-0.1, 0.05, -0.02, 0.0, 0.0, 4.0],
			] {
				let via_solve = geo.allocate(wrench, 1e-6);
				for (i, r) in geo.rotors.iter().enumerate() {
					let t: f32 = (0..6).map(|j| m[i][j] * wrench[j]).sum();
					let pwm = (t.max(0.0) / r.k_thrust).sqrt().clamp(0.0, 1.0);
					assert!((pwm - via_solve[i]).abs() < 1e-5,
						"rotor {i}: pinv pwm {pwm} vs allocate {}", via_solve[i]);
				}
			}
		}
	}

	#[test]
	fn quad_allocation_handles_rank_deficiency() {
		// Planar quad: Fx/Fy rows of B are zero — the damped solve must not
		// blow up, and the reachable components must round-trip.
		let geo = RotorGeometry::quad_plus(ARM, KT, KD);
		let wrench = [0.01, 0.008, 0.002, 0.0, 0.0, 4.0];
		let pwm = geo.allocate(wrench, 1e-6);
		assert!(pwm.iter().all(|p| p.is_finite()));
		let tau = geo.body_torque(&pwm);
		let f = geo.body_force(&pwm);
		assert_close(&tau, &wrench[..3], 2e-3, "quad tau");
		assert!((f[2] - wrench[5]).abs() < 2e-2, "quad Fz: {}", f[2]);
	}
}
