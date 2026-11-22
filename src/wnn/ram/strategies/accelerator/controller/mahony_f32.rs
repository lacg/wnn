// MahonyF32 — the WNN's on-board Mahony attitude estimator in SINGLE precision
// (ABI 61, 22/11/2026, Luiz plan "A + B").
//
// WHY f32. The Metal score kernel has no double, so its Mahony filter and the ABI-55
// heading it feeds always ran in f32 while the CPU ran f64 (estimator.rs) — one of the
// CTRL-54 CPU/GPU differences. On the Cortex-M4F (single-precision FPU) the f64 filter is
// all libgcc soft-double: ~11k instructions per tick, plus the f64 atan2 of the heading.
// From ABI 61 the WNN's estimator is THIS f32 filter on every side:
//   * CPU (this file)                      — AttEstimator, compute_features' heading;
//   * Metal (controller_rollout.metal mahony_update / mahony_heading / atan2_det,
//     evaluated under `#pragma METAL fp math_mode(safe)` + `fp contract(off)` so the
//     fast-math rollout library cannot approximate a divide / sqrt or fuse a mul-add);
//   * the MCU C (mcu/wnn_cascade.c, WNN_EST_F32).
// Every operation below is an IEEE f32 +, −, ×, ÷, sqrt or compare, in ONE fixed order
// shared by the three twins, so the quaternion, the "up" vector and the heading agree BIT
// FOR BIT (pinned: ctrl61_f32_estimator_tests.rs; the C: mcu/parity). Keep the order when
// editing — a reassociated sum is a different function.
//
// atan2 is NOT a libm call here: Rust/Apple libm, Metal and newlib each round their own
// atan2 differently (last-ulp), so the heading uses atan2_det — octant reduction, one
// tan(π/8) reduction and an odd Taylor series to t^15 (truncation < 1.8e-8 rad for
// |t| <= tan(π/8)), built from the IEEE basic ops only.
//
// The classical rivals keep the f64 MahonyFilter (estimator.rs, twin of the Python
// reference): only the WNN's own estimate changed.

/// f32 π, π/2, π/4 and 2π (the values Metal's M_PI_F family round to).
const PI_F: f32 = 3.1415927;
const PI_2_F: f32 = 1.5707964;
const PI_4_F: f32 = 0.7853982;
const TWO_PI_F: f32 = 6.2831855;
/// tan(π/8): above it the argument is reduced through atan(a) = π/4 + atan((a−1)/(a+1)).
const TAN_PI_8_F: f32 = 0.41421357;
/// Odd Taylor coefficients of atan: (−1)^k / (2k+1), k = 1..7 (f32-rounded).
const ATAN_C3: f32 = -0.33333334;
const ATAN_C5: f32 = 0.2;
const ATAN_C7: f32 = -0.14285715;
const ATAN_C9: f32 = 0.11111111;
const ATAN_C11: f32 = -0.09090909;
const ATAN_C13: f32 = 0.07692308;
const ATAN_C15: f32 = -0.06666667;

#[derive(Clone)]
pub struct MahonyF32
{
	dt: f32,
	kp: f32,
	ki: f32,
	q: [f32; 4],
	integral: [f32; 3],
}

impl MahonyF32
{
	pub fn new(dt: f32, kp: f32, ki: f32) -> Self
	{
		Self {
			dt,
			kp,
			ki,
			q: [1.0, 0.0, 0.0, 0.0],
			integral: [0.0; 3],
		}
	}

	/// Warm-start from q0 VERBATIM (the kernel's `mq = q`; the first update renormalizes).
	/// A zero or non-finite q0 falls back to identity.
	pub fn reset(&mut self, q0: [f32; 4])
	{
		let ok = q0.iter().all(|v| v.is_finite()) && q0.iter().any(|v| *v != 0.0);
		self.q = if ok { q0 } else { [1.0, 0.0, 0.0, 0.0] };
		self.integral = [0.0; 3];
	}

	/// One fusion step (gyro rad/s, specific-force accel m/s²) → q (w, x, y, z).
	/// Op-for-op twin of controller_rollout.metal mahony_update.
	pub fn update(&mut self, gyro: [f32; 3], accel: [f32; 3]) -> [f32; 4]
	{
		let (w, x, y, z) = (self.q[0], self.q[1], self.q[2], self.q[3]);
		let (mut gx, mut gy, mut gz) = (gyro[0], gyro[1], gyro[2]);
		let (mut ax, mut ay, mut az) = (accel[0], accel[1], accel[2]);
		let nrm = (ax * ax + ay * ay + az * az).sqrt();
		if nrm > 1e-9
		{
			ax /= nrm;
			ay /= nrm;
			az /= nrm;
			let vx = 2.0 * (x * z - w * y);
			let vy = 2.0 * (w * x + y * z);
			let vz = w * w - x * x - y * y + z * z;
			let ex = ay * vz - az * vy;
			let ey = az * vx - ax * vz;
			let ez = ax * vy - ay * vx;
			if self.ki > 0.0
			{
				self.integral[0] += self.ki * ex * self.dt;
				self.integral[1] += self.ki * ey * self.dt;
				self.integral[2] += self.ki * ez * self.dt;
			}
			gx += self.kp * ex + self.integral[0];
			gy += self.kp * ey + self.integral[1];
			gz += self.kp * ez + self.integral[2];
		}
		let hdt = 0.5 * self.dt;
		let dw = (-x * gx - y * gy - z * gz) * hdt;
		let dx = (w * gx + y * gz - z * gy) * hdt;
		let dy = (w * gy - x * gz + z * gx) * hdt;
		let dz = (w * gz + x * gy - y * gx) * hdt;
		let (w, x, y, z) = (w + dw, x + dx, y + dy, z + dz);
		let n = (w * w + x * x + y * y + z * z).sqrt();
		let n = if n > 0.0 { n } else { 1.0 };
		self.q = [w / n, x / n, y / n, z / n];
		self.q
	}
}

/// atan on [0, 1] (see the file header): Taylor to t^15 in Horner form, after one
/// tan(π/8) reduction. Twin of controller_rollout.metal atan01_det.
fn atan01_det(a: f32) -> f32
{
	let (base, t) = if a > TAN_PI_8_F { (PI_4_F, (a - 1.0) / (a + 1.0)) } else { (0.0, a) };
	let s = t * t;
	let p = ((((((ATAN_C15 * s + ATAN_C13) * s + ATAN_C11) * s + ATAN_C9) * s + ATAN_C7) * s + ATAN_C5) * s)
		+ ATAN_C3;
	base + (t + (t * s) * p)
}

/// Deterministic f32 atan2(y, x) from IEEE basic ops only (bit-identical on CPU, Metal
/// and the M4). atan2(±0, ±0) = 0; atan2(±0, x < 0) = +π (the heading wrap folds ±π).
pub fn atan2_det(y: f32, x: f32) -> f32
{
	let (ax, ay) = (x.abs(), y.abs());
	if ax == 0.0 && ay == 0.0
	{
		return 0.0;
	}
	let swap = ay > ax;
	let a = if swap { ax / ay } else { ay / ax };
	let mut r = atan01_det(a);
	if swap
	{
		r = PI_2_F - r;
	}
	if x < 0.0
	{
		r = PI_F - r;
	}
	if y < 0.0
	{
		r = -r;
	}
	r
}

/// Wrap into (−π, π], f32 (twin of the kernel's heading_wrap).
pub fn wrap_angle_f32(a: f32) -> f32
{
	let mut x = a;
	while x > PI_F
	{
		x -= TWO_PI_F;
	}
	while x <= -PI_F
	{
		x += TWO_PI_F;
	}
	x
}

/// ZYX yaw of q = (w, x, y, z) through atan2_det.
pub fn yaw_det(q: [f32; 4]) -> f32
{
	let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
	let siny_cosp = 2.0 * (w * z + x * y);
	let cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
	atan2_det(siny_cosp, cosy_cosp)
}

/// ABI 55 heading, f32 from ABI 61: the estimate's yaw unwrapped onto the running
/// heading h (never jumps at ±π). Twin of the kernel's mahony_heading.
pub fn mahony_heading_f32(h: f32, q: [f32; 4]) -> f32
{
	h + wrap_angle_f32(yaw_det(q) - h)
}

#[cfg(test)]
mod tests
{
	use super::*;

	/// atan2_det against f64 atan2 over a dense grid incl. axes and octant edges:
	/// within 3e-7 rad (≈ 1.25 f32 ulps at π; measured 2.64e-7).
	#[test]
	fn atan2_det_accuracy()
	{
		let mut worst = 0.0f64;
		for i in -200..=200
		{
			for j in -200..=200
			{
				let (y, x) = (i as f32 * 0.013, j as f32 * 0.011);
				if x == 0.0 && y == 0.0
				{
					continue;
				}
				let got = atan2_det(y, x) as f64;
				let want = (y as f64).atan2(x as f64);
				let mut d = (got - want).abs();
				if (d - 2.0 * std::f64::consts::PI).abs() < 1e-5
				{
					d = 0.0; // ±π branch on y = ±0, x < 0
				}
				worst = worst.max(d);
			}
		}
		assert!(worst < 3.0e-7, "atan2_det worst error {worst:e} rad");
		assert_eq!(atan2_det(0.0, 0.0), 0.0);
		assert_eq!(atan2_det(0.0, 1.0), 0.0);
		assert_eq!(atan2_det(1.0, 0.0), PI_2_F);
		assert_eq!(atan2_det(-1.0, 0.0), -PI_2_F);
	}

	/// The f32 filter tracks the f64 reference filter (estimator.rs) on the golden
	/// sinusoid stream: same algorithm, different precision — within 1e-5 per component.
	#[test]
	fn tracks_the_f64_filter()
	{
		let mut f64f = crate::estimator::MahonyFilter::new(0.001, 2.0, 0.1);
		let mut f32f = MahonyF32::new(0.001, 2.0, 0.1);
		let q0 = [0.9f32, 0.1, -0.15, 0.2];
		let n = q0.iter().map(|v| v * v).sum::<f32>().sqrt();
		let q0n = q0.map(|v| v / n);
		f64f.reset(Some(q0n));
		f32f.reset(q0n);
		let mut worst = 0.0f32;
		for t in 1..=2000usize
		{
			let tf = t as f64;
			let gyro = [
				(0.8 * (0.013 * tf).sin()) as f32,
				(-0.5 * (0.007 * tf + 1.0).sin()) as f32,
				(0.3 * (0.019 * tf + 2.0).sin()) as f32,
			];
			let accel = [(9.81 * 0.2 * (0.011 * tf).sin()) as f32, (9.81 * 0.2 * (0.017 * tf).cos()) as f32, 9.81];
			let a = f64f.update(gyro, accel);
			let b = f32f.update(gyro, accel);
			for k in 0..4
			{
				worst = worst.max((a[k] - b[k]).abs());
			}
		}
		assert!(worst < 1e-5, "f32 filter drifted from the f64 reference: {worst:e}");
	}

	/// Heading unwraps across ±π (no jump) and equals the yaw on the first turn.
	#[test]
	fn heading_unwraps()
	{
		let mut h = 3.0f32;
		for k in 0..40
		{
			let yaw = 3.0f32 + 0.01 * k as f32; // crosses π
			let q = [(yaw * 0.5).cos(), 0.0, 0.0, (yaw * 0.5).sin()];
			let nh = mahony_heading_f32(h, q);
			assert!((nh - h).abs() < 0.02, "jump at k={k}: {h} -> {nh}");
			h = nh;
		}
		assert!(h > PI_F, "heading must continue past π, got {h}");
	}
}
