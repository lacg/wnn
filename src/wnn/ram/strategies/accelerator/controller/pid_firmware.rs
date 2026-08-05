//! The Crazyflie flight firmware's cascaded attitude PID — Rust twin of
//! `wnn/control/pid_firmware.py`.
//!
//! WHY THIS EXISTS ALONGSIDE `AttitudePidRs`. That one is a SINGLE loop in radians
//! emitting PWM directly, with gains hand-tuned against the retired synthetic plant and
//! no citation. This is the controller the real vehicle flies: a CASCADE
//! (angle -> rate setpoint -> per-motor force) at 500 Hz whose gains come from
//! `bitcraze/crazyflie-firmware platform_defaults_cf21bl.h`. They are not two tunings of
//! one controller — firmware kp 6.0 is deg->deg/s inside a cascade, the legacy kp 1.2 is
//! rad->pwm in a single loop.
//!
//! SI ONLY. This module never sees a degree or an actuator count: the gain conversion
//! happens once, on the Python side, in `_SiGains.from_firmware`, and arrives here as
//! rad / rad-per-second / NEWTONS. Keeping the boundary in exactly one place is
//! deliberate (Luiz, 05/08/2026) — mixed units inside a control loop is how a stray
//! factor hides for months.
//!
//! Full derivation of the counts->newtons->pwm mapping, with the firmware file each line
//! came from, is in docs/disturbance_param_sources.md "THE UNIT MAPPING".

/// Two-pole low-pass, a line-for-line port of the firmware's `filter.c`
/// (`lpf2pSetCutoffFreq` + `lpf2pApply`).
///
/// The firmware ships `ATTITUDE_RATE_LPF_ENABLE false`, and on hardware that is safe:
/// the rate PID differentiates a GYRO the sensor stack has already low-passed. Our sim
/// hands the controller the exact instantaneous body rate, so the same unfiltered
/// derivative goes unstable — measured, not assumed: with rate kd live the loop
/// limit-cycles between the output rails (tail swing 2.20 deg), with kd zeroed it is
/// stable (0.11 deg), and kd is the ONLY term that does it. So we enable the facility the
/// firmware itself provides, at the firmware's own default cutoff (30 Hz).
#[derive(Clone, Copy, Debug)]
pub struct Lpf2p {
	b0: f64,
	b1: f64,
	b2: f64,
	a1: f64,
	a2: f64,
	d1: f64,
	d2: f64,
}

impl Lpf2p {
	pub fn new(sample_freq: f64, cutoff_freq: f64) -> Self {
		let fr = sample_freq / cutoff_freq;
		let ohm = (std::f64::consts::PI / fr).tan();
		let cos45 = (std::f64::consts::PI / 4.0).cos();
		let c = 1.0 + 2.0 * cos45 * ohm + ohm * ohm;
		let b0 = ohm * ohm / c;
		Lpf2p {
			b0,
			b1: 2.0 * b0,
			b2: b0,
			a1: 2.0 * (ohm * ohm - 1.0) / c,
			a2: (1.0 - 2.0 * cos45 * ohm + ohm * ohm) / c,
			d1: 0.0,
			d2: 0.0,
		}
	}

	pub fn reset(&mut self) {
		self.d1 = 0.0;
		self.d2 = 0.0;
	}

	/// b0, b1, b2, a1, a2 — so a caller that must hand the SAME filter to another
	/// implementation (the Metal kernel) cannot hardcode a drifting copy.
	pub fn coeffs(&self) -> [f64; 5] {
		[self.b0, self.b1, self.b2, self.a1, self.a2]
	}

	pub fn apply(&mut self, sample: f64) -> f64 {
		let mut d0 = sample - self.d1 * self.a1 - self.d2 * self.a2;
		if !d0.is_finite() {
			d0 = sample;
		}
		let out = d0 * self.b0 + self.d1 * self.b1 + self.d2 * self.b2;
		self.d2 = self.d1;
		self.d1 = d0;
		out
	}
}

/// One axis of one loop, already in SI. `i_limit`/`out_limit` of 0.0 mean "no clamp",
/// matching `pid.c`'s `if (pid->iLimit != 0)` / `if (pid->outputLimit != 0)`.
#[derive(Clone, Copy, Debug)]
pub struct SiAxis {
	pub kp: f64,
	pub ki: f64,
	pub kd: f64,
	pub i_limit: f64,
}

#[derive(Clone, Copy, Debug)]
struct PidCh {
	g: SiAxis,
	dt: f64,
	out_limit: f64,
	filt: Option<Lpf2p>,
	integ: f64,
	prev_measured: f64,
	first: bool,
}

impl PidCh {
	fn new(g: SiAxis, dt: f64, out_limit: f64, filt: Option<Lpf2p>) -> Self {
		PidCh { g, dt, out_limit, filt, integ: 0.0, prev_measured: 0.0, first: true }
	}

	fn reset(&mut self) {
		self.integ = 0.0;
		self.prev_measured = 0.0;
		self.first = true;
		if let Some(f) = self.filt.as_mut() {
			f.reset();
		}
	}

	/// `pid.c` pidUpdate: integrate, clamp, derive on the MEASUREMENT, clamp output.
	/// `wrap` mirrors pidUpdate's shouldWrap — true for yaw only.
	fn update(&mut self, setpoint: f64, measured: f64, wrap: bool) -> f64 {
		let mut error = setpoint - measured;
		if wrap {
			error = wrap_pi(error);
		}
		self.integ += error * self.dt;
		if self.g.i_limit != 0.0 {
			self.integ = clamp(self.integ, -self.g.i_limit, self.g.i_limit);
		}
		let mut deriv = if self.first {
			0.0
		} else {
			-(measured - self.prev_measured) / self.dt
		};
		if let Some(f) = self.filt.as_mut() {
			deriv = f.apply(deriv);
		}
		self.prev_measured = measured;
		self.first = false;
		let out = self.g.kp * error + self.g.ki * self.integ + self.g.kd * deriv;
		if self.out_limit != 0.0 {
			clamp(out, -self.out_limit, self.out_limit)
		} else {
			out
		}
	}
}

/// The cascade. Construct with SI gains; `step_rs` takes the sim's state and returns the
/// 4 PWMs `AttitudeSim` expects, in its '+' motor order.
#[derive(Clone, Copy, Debug)]
pub struct AttitudePidFirmwareRs {
	att: [PidCh; 3],
	rate: [PidCh; 3],
	hover_thrust_n: f64,
	k_thrust: f64,
	decimation: u32,
	tick: u32,
	held: [f64; 3],
}

impl AttitudePidFirmwareRs {
	/// `att`/`rate` are per-axis (roll, pitch, yaw) SI gains; `rate_out_limit_n` is the
	/// firmware's int16 saturation expressed in newtons; `hover_thrust_n` is m*g/4;
	/// `rate_lpf_cutoff_hz` of 0.0 disables the rate-D filter (firmware's own default).
	pub fn new_si(
		att: [SiAxis; 3],
		rate: [SiAxis; 3],
		rate_out_limit_n: f64,
		hover_thrust_n: f64,
		k_thrust: f64,
		main_loop_hz: u32,
		attitude_hz: u32,
		rate_lpf_cutoff_hz: f64,
	) -> Self {
		let dt = 1.0 / attitude_hz as f64;
		let filt = |c: f64| {
			if c > 0.0 { Some(Lpf2p::new(attitude_hz as f64, c)) } else { None }
		};
		// attFiltEnable is false in firmware and the attitude D-term is stable
		// unfiltered (roll/pitch kd = 0 anyway), so only the rate loop is filtered.
		AttitudePidFirmwareRs {
			att: [
				PidCh::new(att[0], dt, 0.0, None),
				PidCh::new(att[1], dt, 0.0, None),
				PidCh::new(att[2], dt, 0.0, None),
			],
			rate: [
				PidCh::new(rate[0], dt, rate_out_limit_n, filt(rate_lpf_cutoff_hz)),
				PidCh::new(rate[1], dt, rate_out_limit_n, filt(rate_lpf_cutoff_hz)),
				PidCh::new(rate[2], dt, rate_out_limit_n, filt(rate_lpf_cutoff_hz)),
			],
			hover_thrust_n,
			k_thrust,
			decimation: (main_loop_hz / attitude_hz).max(1),
			tick: 0,
			held: [0.0; 3],
		}
	}

	/// Build from the flat SI arrays the packed config carries, laid out as
	/// [roll, pitch, yaw] x [kp, ki, kd, i_limit]. Returns None when no cascade gains
	/// were supplied (`attitude_hz <= 0`), which is the signal to keep the legacy PID.
	#[allow(clippy::too_many_arguments)]
	pub fn from_si_arrays(
		att: [f64; 12],
		rate: [f64; 12],
		out_limit_n: f64,
		hover_n: f64,
		k_thrust: f64,
		main_loop_hz: u32,
		attitude_hz: f64,
		lpf_hz: f64,
	) -> Option<Self> {
		if attitude_hz <= 0.0 {
			return None;
		}
		let axes = |a: [f64; 12]| {
			let mut out = [SiAxis { kp: 0.0, ki: 0.0, kd: 0.0, i_limit: 0.0 }; 3];
			for i in 0..3 {
				out[i] = SiAxis {
					kp: a[i * 4],
					ki: a[i * 4 + 1],
					kd: a[i * 4 + 2],
					i_limit: a[i * 4 + 3],
				};
			}
			out
		};
		Some(Self::new_si(
			axes(att), axes(rate), out_limit_n, hover_n, k_thrust,
			main_loop_hz, attitude_hz as u32, lpf_hz,
		))
	}

	/// The ATTITUDE loop's integral accumulators, for the Option-A integral state target.
	/// The attitude integral is the cascade's analogue of the single-loop PID's I-term
	/// (both accumulate ANGLE error); the rate loop's integral accumulates a rate error
	/// and is not the same quantity, so it is deliberately not exposed here.
	pub fn integrals_f32(&self) -> [f32; 3] {
		[
			self.att[0].integ as f32,
			self.att[1].integ as f32,
			self.att[2].integ as f32,
		]
	}

	/// Matching clamp magnitudes for normalizing those integrals. A zero i_limit means
	/// "unclamped" in pid.c, so report 1.0 rather than 0.0 and avoid a divide-by-zero.
	pub fn i_clamps_f32(&self) -> [f32; 3] {
		let one = |v: f64| if v == 0.0 { 1.0f32 } else { v as f32 };
		[
			one(self.att[0].g.i_limit),
			one(self.att[1].g.i_limit),
			one(self.att[2].g.i_limit),
		]
	}

	/// The rate loop's filter coefficients, or None when filtering is off. Lets the GPU
	/// be handed exactly the filter the CPU runs.
	pub fn rate_lpf_coeffs(&self) -> Option<[f64; 5]> {
		self.rate[0].filt.map(|f| f.coeffs())
	}

	pub fn reset(&mut self) {
		for c in self.att.iter_mut().chain(self.rate.iter_mut()) {
			c.reset();
		}
		self.tick = 0;
		self.held = [0.0; 3];
	}

	/// One 1 kHz sim tick. The cascade updates every `decimation` ticks and its output
	/// is HELD in between (stabilizer_types.h RATE_DO_EXECUTE).
	pub fn step_rs(&mut self, q: [f64; 4], gyro: [f64; 3], target_rpy: [f64; 3]) -> [f64; 4] {
		if self.tick % self.decimation == 0 {
			self.held = self.update_cascade(q, gyro, target_rpy);
		}
		self.tick = self.tick.wrapping_add(1);
		self.mix(self.held)
	}

	/// f32 entry point for the sim's hot path, mirroring `AttitudePidRs::step_pub`.
	pub fn step_f32(&mut self, q: [f32; 4], gyro: [f32; 3], t: [f32; 3]) -> [f32; 4] {
		let p = self.step_rs(
			[q[0] as f64, q[1] as f64, q[2] as f64, q[3] as f64],
			[gyro[0] as f64, gyro[1] as f64, gyro[2] as f64],
			[t[0] as f64, t[1] as f64, t[2] as f64],
		);
		[p[0] as f32, p[1] as f32, p[2] as f32, p[3] as f32]
	}

	/// Angle (rad) -> rate setpoint (rad/s) -> per-motor force offset (N), per axis.
	fn update_cascade(
		&mut self,
		q: [f64; 4],
		gyro: [f64; 3],
		target_rpy: [f64; 3],
	) -> [f64; 3] {
		let rpy = quat_to_euler(q);
		let mut out = [0.0f64; 3];
		for i in 0..3 {
			// attitude_pid_controller.c hands the attitude PID the ACTUAL angle as its
			// measurement, so its D-term differentiates the angle — matters for yaw
			// alone (kd 0.35). Only yaw wraps.
			let is_yaw = i == 2;
			let rate_sp = self.att[i].update(target_rpy[i], rpy[i], is_yaw);
			out[i] = self.rate[i].update(rate_sp, gyro[i], false);
		}
		out
	}

	/// Per-axis force offset (N) -> per-motor thrust (N) -> PWM, '+' motor order.
	///
	/// Sim convention (`AttitudeSim::body_torque`): M0 front(+x), M1 right(-y),
	/// M2 rear(-x), M3 left(+y); roll = L*(-t1+t3), pitch = L*(-t0+t2),
	/// yaw = k*(t0-t1+t2-t3). Firmware halves roll/pitch and does NOT halve yaw
	/// (power_distribution_quadrotor.c). thrust = k_thrust*pwm^2, so pwm = sqrt(t/kt).
	fn mix(&self, axis_n: [f64; 3]) -> [f64; 4] {
		let (r, p, y) = (axis_n[0] / 2.0, axis_n[1] / 2.0, axis_n[2]);
		let (h, kt) = (self.hover_thrust_n, self.k_thrust);
		// +roll raises the LEFT motor (M3) and drops the RIGHT (M1); +pitch raises the
		// REAR (M2) and drops the FRONT (M0) — matching body_torque's signs.
		let t = [h - p + y, h - r - y, h + p + y, h + r - y];
		let mut pwm = [0.0f64; 4];
		for i in 0..4 {
			pwm[i] = (clamp(t[i], 0.0, kt) / kt).sqrt();
		}
		pwm
	}
}

#[inline]
fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
	if v < lo {
		lo
	} else if v > hi {
		hi
	} else {
		v
	}
}

#[inline]
fn wrap_pi(mut a: f64) -> f64 {
	while a > std::f64::consts::PI {
		a -= 2.0 * std::f64::consts::PI;
	}
	while a <= -std::f64::consts::PI {
		a += 2.0 * std::f64::consts::PI;
	}
	a
}

/// Body-to-world unit quaternion -> (roll, pitch, yaw) rad. Same form as the Python
/// twin's and `AttitudePidRs`', so the controllers cannot disagree about what attitude
/// they were handed.
fn quat_to_euler(q: [f64; 4]) -> [f64; 3] {
	let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
	let roll = (2.0 * (w * x + y * z)).atan2(1.0 - 2.0 * (x * x + y * y));
	let sinp = 2.0 * (w * y - z * x);
	let pitch = if sinp >= 1.0 {
		std::f64::consts::FRAC_PI_2
	} else if sinp <= -1.0 {
		-std::f64::consts::FRAC_PI_2
	} else {
		sinp.asin()
	};
	let yaw = (2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y * y + z * z));
	[roll, pitch, yaw]
}

#[cfg(test)]
mod tests {
	use super::*;

	/// SI gains for cf21_brushless, generated by wnn.control.pid_firmware
	/// (`_SiGains.from_firmware`) — the Python side owns the unit conversion.
	fn cf21bl() -> AttitudePidFirmwareRs {
		let att = [
			SiAxis { kp: 6.0, ki: 3.0, kd: 0.0, i_limit: 0.3490658503988659 },
			SiAxis { kp: 6.0, ki: 3.0, kd: 0.0, i_limit: 0.3490658503988659 },
			SiAxis { kp: 6.0, ki: 1.0, kd: 0.35, i_limit: 6.283185307179586 },
		];
		let rate = [
			SiAxis {
				kp: 0.03497110216713654,
				ki: 0.06994220433427308,
				kd: 0.00043713877708920673,
				i_limit: 0.5811946409141117,
			},
			SiAxis {
				kp: 0.03497110216713654,
				ki: 0.06994220433427308,
				kd: 0.00043713877708920673,
				i_limit: 0.5811946409141117,
			},
			SiAxis {
				kp: 0.02098266130028192,
				ki: 0.002920087030955901,
				kd: 0.0,
				i_limit: 2.909463863074547,
			},
		];
		AttitudePidFirmwareRs::new_si(
			att, rate, 0.09999847409781033, 0.09638325, 0.2, 1000, 500, 30.0,
		)
	}

	#[test]
	fn lpf2p_matches_firmware_coefficients() {
		let f = Lpf2p::new(500.0, 30.0);
		// DC gain must be exactly 1: sum(b) / (1 + sum(a)).
		let dc = (f.b0 + f.b1 + f.b2) / (1.0 + f.a1 + f.a2);
		assert!((dc - 1.0).abs() < 1e-12, "DC gain {} != 1", dc);
		// Coefficients as computed by filter.c's lpf2pSetCutoffFreq, cross-checked
		// against the Python twin at full precision.
		assert!((f.b0 - 0.027859766117136024).abs() < 1e-15, "b0 {:.17}", f.b0);
		assert!((f.b1 - 0.05571953223427205).abs() < 1e-15, "b1 {:.17}", f.b1);
		assert!((f.b2 - 0.027859766117136024).abs() < 1e-15, "b2 {:.17}", f.b2);
		assert!((f.a1 - (-1.475480443592646)).abs() < 1e-15, "a1 {:.17}", f.a1);
		assert!((f.a2 - 0.5869195080611903).abs() < 1e-15, "a2 {:.17}", f.a2);
	}

	#[test]
	fn lpf2p_unit_step_settles_to_one() {
		let mut f = Lpf2p::new(500.0, 30.0);
		let mut last = 0.0;
		for _ in 0..400 {
			last = f.apply(1.0);
		}
		assert!((last - 1.0).abs() < 1e-9, "step settled to {}", last);
	}

	/// Golden trajectory from the validated Python reference. Any drift here means the
	/// two implementations have diverged — the whole point of keeping this test.
	#[test]
	fn matches_python_reference_golden_trajectory() {
		let mut pid = cf21bl();
		let q = [
			0.9985736925158005,
			0.04314138300279472,
			0.026909052982958412,
			0.016288172258609745,
		];
		let gyro = [0.10, -0.05, 0.02];
		let expect: [[f64; 4]; 6] = [
			[0.69357463426104093, 0.74885607070686344, 0.65925424456891057, 0.6717273651512311],
			[0.69357463426104093, 0.74885607070686344, 0.65925424456891057, 0.6717273651512311],
			[0.693651072532246, 0.74904183162408855, 0.65915253573442278, 0.67154110722053029],
			[0.693651072532246, 0.74904183162408855, 0.65915253573442278, 0.67154110722053029],
			[0.69372758009631286, 0.74922767004288093, 0.65905072630606443, 0.67135466276125522],
			[0.69372758009631286, 0.74922767004288093, 0.65905072630606443, 0.67135466276125522],
		];
		for (k, want) in expect.iter().enumerate() {
			let got = pid.step_rs(q, gyro, [0.0; 3]);
			for i in 0..4 {
				assert!(
					(got[i] - want[i]).abs() < 1e-12,
					"step {} motor {}: got {:.17} want {:.17}",
					k, i, got[i], want[i]
				);
			}
		}
	}

	/// The 500 Hz cascade must hold its output across the intervening 1 kHz tick.
	#[test]
	fn output_is_held_across_the_decimated_tick() {
		let mut pid = cf21bl();
		let q = [0.99904822, 0.04361939, 0.0, 0.0];
		let a = pid.step_rs(q, [0.0; 3], [0.0; 3]);
		let b = pid.step_rs(q, [0.5, 0.0, 0.0], [0.0; 3]);
		assert_eq!(a, b, "output changed on a held tick");
	}

	/// Level and at rest, every motor must sit exactly at the hover PWM.
	#[test]
	fn level_at_rest_is_hover_on_all_motors() {
		let mut pid = cf21bl();
		let out = pid.step_rs([1.0, 0.0, 0.0, 0.0], [0.0; 3], [0.0; 3]);
		let hover = (0.09638325f64 / 0.2).sqrt();
		for (i, v) in out.iter().enumerate() {
			assert!((v - hover).abs() < 1e-12, "motor {} = {} != hover {}", i, v, hover);
		}
	}

	/// A positive error on each axis must produce a CORRECTIVE (negative) torque on that
	/// axis and essentially none on the others — this is what catches a mixer sign flip.
	#[test]
	fn each_axis_produces_corrective_torque_without_cross_coupling() {
		let (kt, l, kd) = (0.2f64, 0.07071067811865477f64, 0.00569278844371417f64);
		for axis in 0..3 {
			let mut pid = cf21bl();
			let half = (5.0f64).to_radians() / 2.0;
			let mut q = [half.cos(), 0.0, 0.0, 0.0];
			q[axis + 1] = half.sin();
			let pwm = pid.step_rs(q, [0.0; 3], [0.0; 3]);
			let t: Vec<f64> = pwm.iter().map(|p| kt * p * p).collect();
			let tau = [
				l * (-t[1] + t[3]),
				l * (-t[0] + t[2]),
				kd * (t[0] - t[1] + t[2] - t[3]),
			];
			assert!(tau[axis] < 0.0, "axis {} torque {} not corrective", axis, tau[axis]);
			for other in 0..3 {
				if other != axis {
					assert!(
						tau[other].abs() < 1e-15,
						"axis {} leaked {} onto axis {}",
						axis, tau[other], other
					);
				}
			}
		}
	}
}
