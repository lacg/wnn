#include <metal_stdlib>
using namespace metal;

// =============================================================================
// controller_rollout.metal — GPU-batched closed-loop drone-attitude eval.
//
// One thread = one (genome g, episode e) rollout, run for P.steps timesteps.
// Faithful f32 port of controller.rs: AttitudeSim (RK4 rigid-body dynamics) +
// WnnController::step (thermometer encode + K-window + sparse 2-layer lookup +
// Strategy-5 QSR decode) + run_episode reward. Cell lookups reuse the existing
// SparseGpuExport sorted-array + binary-search pattern (sparse_forward.metal).
//
// Only ABSOLUTE-PWM mode is implemented (delta_control=false) — the locked C1/C2
// substrate. Memory cells default to QSR EMPTY=2 (→ weight 0.75) when an address
// was never trained, matching read_cell.
//
// Parity note: closed-loop dynamics are chaotic, so GPU↔CPU parity is STATISTICAL
// (aggregate mean_err / stable_rate / reward), not bit-exact per-step over 2000
// steps — tiny f32 op-order/transcendental differences compound through feedback.
// =============================================================================

constant uint  NUM_FEATURES    = 9u;    // base raw sensors (gyro+accel+target)
constant uint  MAX_FEATURES    = 21u;   // 9 base + 8 H2 extras (tilt+per-axis ×p/i) + 4 pwm-accumulator

// Compile-time maxima for thread-private arrays (host asserts runtime <= these).
// MAX_STATE_NEURONS bumped 32→64 (09/07/2026): the NEURONS GA grows state_neurons to
// 4×grid_max (=64 for grid_max 16). At 32 the thread-private prev_state/new_state arrays
// OVERFLOWED for sn>32 → UB → zero/garbage metrics + adjacent-thread corruption (the
// whole-batch-zeros / 0-viable stall). The host now also guards sn>MAX → CPU fallback.
#define MAX_STATE_NEURONS 64
#define MAX_WINDOW        8
#define MAX_INTEGRALS     5     // tilt_i + 3×peraxis_i + yaw_err_i
#define MAX_ROTORS        8     // overactuated Phase 1 (octo-X is the widest preset)

// ---- Overactuated Phase 1 (docs/OVERACTUATED_RESIDUAL_DESIGN.md) ----------
// N-rotor geometry via FUNCTION CONSTANTS: one kernel source, the compiler
// specializes per airframe at pipeline creation. When the constants are not
// set (all existing Rust pipelines), USE_GEOMETRY=false dead-strips the
// generic path and the quad torque block below stays bit-identical.
constant bool FC_HAS_GEOMETRY [[function_constant(0)]];
constant uint FC_NUM_ROTORS   [[function_constant(1)]];
constant bool USE_GEOMETRY = is_function_constant_defined(FC_HAS_GEOMETRY) ? FC_HAS_GEOMETRY : false;
constant uint NUM_ROTORS   = is_function_constant_defined(FC_NUM_ROTORS)   ? FC_NUM_ROTORS   : 4u;

// One rotor of the geometry buffer — 48 B, all-float, mirrors the Rust-side
// RotorGpu upload. k_thrust is the EFFECTIVE coefficient (nominal x per-rotor
// asym baked at upload; tilt/position error arrive as a perturbed table), so
// the kernel needs no per-rotor disturbance fields.
struct RotorGpu {
	float px, py, pz;      // position, body frame (m)
	float ax, ay, az;      // unit thrust axis, body frame
	float spin;            // +1 CCW / -1 CW (drag torque sign)
	float k_thrust;        // N per pwm^2 (effective)
	float k_drag;          // drag-torque/thrust ratio
	float _pad0, _pad1, _pad2;
};

struct Params {
	uint num_genomes;
	uint num_episodes;
	uint steps;
	uint num_motors;
	uint levels;
	uint n_state;
	uint sbpn;             // state_bits_per_neuron
	uint obpn;             // output_bits_per_neuron
	uint bpf;              // bits_per_feature
	uint window;           // input_window_k
	uint frame_bits;       // NUM_FEATURES * bpf
	uint sensor_total;     // window * frame_bits
	uint state_bits_in;    // 2 * n_state
	// sim params
	float dt;
	float arm_length;
	float k_thrust;
	float k_drag;
	float inertia0;
	float inertia1;
	float inertia2;
	float gravity;
	float target0;
	float target1;
	float target2;
	// Delta-control mode (layout must match Rust RolloutParams exactly).
	uint  delta_control;   // 0 = absolute PWM, 1 = per-step delta + leaky accumulator
	float delta_max;
	float delta_leak;
	// H2 observation-feature config (layout must match Rust RolloutParams exactly).
	uint  num_features;    // 9 base + enabled extras (drives frame stride/loops)
	uint  obs_tilt_p;
	uint  obs_tilt_i;
	uint  obs_peraxis_p;
	uint  obs_peraxis_i;
	uint  obs_peraxis_yaw; // per-axis carries yaw (1) or only roll+pitch (0)
	uint  obs_yaw_err;     // yaw-anchor: clean scalar (target_yaw − anchored heading)
	uint  obs_yaw_err_i;   // yaw-anchor: leaky integral of the yaw error
	uint  obs_pwm;         // expose the raw throttle accumulator (num_motors feats)
	float integral_leak;
	float integral_scale;
	uint  decouple_outputs; // H3: 4 banks are controls [T,τr,τp,τy] → mix to motors
	// Action-repeat N (arm R): decide every Nth step, HOLD the PWM in between.
	// APPENDED at the END — layout must match the Rust RolloutParams exactly.
	uint  action_repeat;
	// --- W2 disturbances (layout must match Rust RolloutParams exactly). ---
	// dist_enabled=0 ⇒ the pre-W2 clean rollout (no extra float ops in the
	// loop). Field semantics mirror controller.rs `Disturbance`.
	uint  dist_enabled;
	float dist_tau_bias0;     // D1 constant torque bias (N·m, body frame)
	float dist_tau_bias1;
	float dist_tau_bias2;
	float dist_gust_sigma;    // D2 OU gust
	float dist_gust_tau_c;
	float dist_motor_asym0;   // D3 per-motor k_thrust multipliers
	float dist_motor_asym1;
	float dist_motor_asym2;
	float dist_motor_asym3;
	float dist_gyro_sigma;    // D4 sensor noise
	float dist_gyro_bias_walk;
	float dist_accel_sigma;
	uint  dist_seed;          // base seed, PRE-FOLDED to u32 (dist_seed32)
	uint  dist_ep_offset;     // global index of episode 0 in this dispatch
	                          // (score() chunks episodes; per-episode seeds
	                          // must not depend on the chunk size)
	// --- E5 residual hybrid (APPENDED at END; layout must match Rust RolloutParams).
	//     residual_enabled=0 ⇒ pure-WNN (pre-E5). When 1, the WNN output is a signed
	//     residual composed on an analytic PID baseline (compose_residual). ---
	uint  residual_enabled;
	float pid_kp_rp;
	float pid_ki_rp;
	float pid_kd_rp;
	float pid_iclamp_rp;
	float pid_kp_yaw;
	float pid_ki_yaw;
	float pid_kd_yaw;
	float pid_iclamp_yaw;
	float pid_hover;
	float pid_authority;
	float residual_scale;
	float residual_clamp;
	// --- Overactuated Phase 2 (APPENDED at END; layout must match Rust
	//     RolloutParams). alloc_baseline=1 ⇒ the residual composes on the
	//     allocator-LQR baseline from buffer(28) instead of the quad PID. ---
	uint  alloc_baseline;
	// Residual neutral anchor = the untrained-cell decode value (host sets it
	// from the controller's mode-derived neutral — QUAD 0.75 / TERNARY 0.5 /
	// BINARY 0.5). ABI 11; also the delta-control neutral since ABI 12.
	float residual_neutral;
	// Memory mode (ABI 12): 0 TERNARY / 1-2 QUAD / 3 BINARY (antagonist-pair
	// output decode). APPENDED at the END — layout must match Rust RolloutParams.
	uint  memory_mode;
};

// ---- W2 disturbance counter-RNG — bit-for-bit twin of controller.rs --------
// dist_hash_u32 / dist_uniform / dist_gauss / channel ids. The canonical
// channel-id registry lives in controller.rs (DIST_CH_*); do NOT renumber here.
constant uint  DIST_CH_GUST      = 0u;
constant uint  DIST_CH_GYRO      = 1u;
constant uint  DIST_CH_GYRO_BIAS = 2u;
constant uint  DIST_CH_ACCEL     = 3u;
// QSR/PLN stochastic-decode coin channel (twin of controller.rs DIST_CH_MEM_COIN):
// axis=motor, k=level, step=physical step t → a fresh per-timestep firing coin.
constant uint  DIST_CH_MEM_COIN  = 4u;
constant uint  DIST_CH_EP_SEED   = 15u;
constant float DIST_TWO_PI       = 6.2831855f;   // same f32 literal as Rust

inline uint dist_hash(uint seed, uint step, uint axis, uint channel, uint k) {
	uint rng = seed
		+ step * 2654435761u
		+ axis * 1000003u
		+ channel * 97780813u
		+ k * 668265263u;
	if (rng == 0u) rng = 1u;
	rng ^= rng << 13;
	rng ^= rng >> 17;
	rng ^= rng << 5;
	return rng;
}

inline float dist_uniform(uint seed, uint step, uint axis, uint channel, uint k) {
	return float(dist_hash(seed, step, axis, channel, k) >> 8) / 16777216.0f;
}

// Box-Muller over sub-draws k=0 (radius, ln-clamped like Rust) and k=1 (angle).
inline float dist_gauss(uint seed, uint step, uint axis, uint channel) {
	float u1 = fmax(dist_uniform(seed, step, axis, channel, 0u), 1e-7f);
	float u2 = dist_uniform(seed, step, axis, channel, 1u);
	return sqrt(-2.0f * log(u1)) * cos(DIST_TWO_PI * u2);
}

// Mirror of controller.rs decoded_to_delta: map a Strategy-5 decode in [0,1] to a
// per-step PWM delta in [-delta_max, +delta_max], piecewise-linear about the
// mode-derived neutral `n` (P.residual_neutral; QUAD 0.75 — ABI 12).
inline float decoded_to_delta(float decoded, float delta_max, float n) {
	if (decoded >= n) return (decoded - n) / (1.0f - n) * delta_max;
	else              return (decoded - n) / n * delta_max;
}

// ---- quaternion / vector helpers (mirror controller.rs) ---------------------

inline float4 q_mul(float4 a, float4 b) {
	return float4(
		a.x*b.x - a.y*b.y - a.z*b.z - a.w*b.w,
		a.x*b.y + a.y*b.x + a.z*b.w - a.w*b.z,
		a.x*b.z - a.y*b.w + a.z*b.x + a.w*b.y,
		a.x*b.w + a.y*b.z - a.z*b.y + a.w*b.x);
}
inline float4 q_norm(float4 q) {
	float n = sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w);
	return n > 0.0f ? q / n : float4(1.0f, 0.0f, 0.0f, 0.0f);
}
// rotate WORLD vector to BODY: q* . v . q  (q stored as (w,x,y,z) in .xyzw)
inline float3 rotate_world_to_body(float4 q, float3 v) {
	float4 conj = float4(q.x, -q.y, -q.z, -q.w);
	float4 vq   = float4(0.0f, v.x, v.y, v.z);
	float4 t    = q_mul(conj, vq);
	float4 r    = q_mul(t, q);
	return float3(r.y, r.z, r.w);
}
// Yaw-anchor: ZYX (Tait-Bryan) yaw angle (rad) from a (w,x,y,z) quaternion —
// the bit-for-bit twin of controller.rs `yaw_from_quat`. q stored (w,x,y,z) in
// .xyzw ⇒ w=q.x, x=q.y, y=q.z, z=q.w. Seeds the GPU yaw heading from q0.
inline float yaw_from_quat(float4 q) {
	return atan2(2.0f * (q.x * q.w + q.y * q.z),
	             1.0f - 2.0f * (q.z * q.z + q.w * q.w));
}

// ---- E5 residual baseline: analytic PID (mirror AttitudePidRs::step_rs, f32) ---
// wrap to (-π, π]; twin of controller.rs wrap_angle_f64.
inline float pid_wrap(float a) {
	while (a > M_PI_F)  a -= 2.0f * M_PI_F;
	while (a <= -M_PI_F) a += 2.0f * M_PI_F;
	return a;
}
// One PID cycle from (q, gyro, target) → out_pwm[4]; mutates the per-thread
// integral state pid_i[3] (roll,pitch,yaw). q stored (w,x,y,z) in .xyzw.
inline void pid_step(float4 q, thread const float* gyro, constant Params& P,
                     thread float* pid_i, thread float* out_pwm) {
	// quat → euler (mirror quat_to_euler_f64)
	float w = q.x, x = q.y, y = q.z, z = q.w;
	float roll  = atan2(2.0f * (w*x + y*z), 1.0f - 2.0f * (x*x + y*y));
	float sinp  = 2.0f * (w*y - z*x);
	float pitch = (sinp >= 1.0f) ? (M_PI_F * 0.5f)
	            : (sinp <= -1.0f) ? (-M_PI_F * 0.5f) : asin(sinp);
	float yaw   = atan2(2.0f * (w*z + x*y), 1.0f - 2.0f * (y*y + z*z));

	float e_roll  = pid_wrap(P.target0 - roll);
	float e_pitch = pid_wrap(P.target1 - pitch);
	float e_yaw   = pid_wrap(P.target2 - yaw);

	pid_i[0] = clamp(pid_i[0] + e_roll  * P.dt, -P.pid_iclamp_rp,  P.pid_iclamp_rp);
	pid_i[1] = clamp(pid_i[1] + e_pitch * P.dt, -P.pid_iclamp_rp,  P.pid_iclamp_rp);
	pid_i[2] = clamp(pid_i[2] + e_yaw   * P.dt, -P.pid_iclamp_yaw, P.pid_iclamp_yaw);

	float a = P.pid_authority;
	float u_roll  = clamp(P.pid_kp_rp*e_roll  + P.pid_ki_rp*pid_i[0] - P.pid_kd_rp*gyro[0], -a, a);
	float u_pitch = clamp(P.pid_kp_rp*e_pitch + P.pid_ki_rp*pid_i[1] - P.pid_kd_rp*gyro[1], -a, a);
	float u_yaw   = clamp(P.pid_kp_yaw*e_yaw  + P.pid_ki_yaw*pid_i[2] - P.pid_kd_yaw*gyro[2], -a, a);

	float base = P.pid_hover;
	out_pwm[0] = clamp(base - u_pitch + u_yaw, 0.0f, 1.0f);  // M0 front
	out_pwm[1] = clamp(base - u_roll  - u_yaw, 0.0f, 1.0f);  // M1 right
	out_pwm[2] = clamp(base + u_pitch + u_yaw, 0.0f, 1.0f);  // M2 rear
	out_pwm[3] = clamp(base + u_roll  - u_yaw, 0.0f, 1.0f);  // M3 left
}

// ---- Overactuated Phase 2: allocator-LQR baseline (mirror AllocBaseline::pwm) ---
// tab layout (optimal.rs to_gpu_blob): [k1, k2x, k2y, k2z, tau_max, f_hover]
// then per-rotor [m0..m5, k_thrust] (7-float stride). τ = clamp(k1·e − k2·rate,
// ±τ_max) per axis → T = M·(τ; 0,0,F_hover) → pwm = √(max(T,0)/k) in [0,1].
// Memoryless (no integral state). Writes NUM_ROTORS PWMs.
inline void alloc_step(float4 q, thread const float* gyro, constant Params& P,
                       device const float* tab, thread float* out_pwm) {
	float w = q.x, x = q.y, y = q.z, z = q.w;
	float roll  = atan2(2.0f * (w*x + y*z), 1.0f - 2.0f * (x*x + y*y));
	float sinp  = 2.0f * (w*y - z*x);
	float pitch = (sinp >= 1.0f) ? (M_PI_F * 0.5f)
	            : (sinp <= -1.0f) ? (-M_PI_F * 0.5f) : asin(sinp);
	float yaw   = atan2(2.0f * (w*z + x*y), 1.0f - 2.0f * (y*y + z*z));

	float k1 = tab[0], tmax = tab[4];
	float e0 = pid_wrap(P.target0 - roll);
	float e1 = pid_wrap(P.target1 - pitch);
	float e2 = pid_wrap(P.target2 - yaw);
	float wrench[6] = {
		clamp(k1 * e0 - tab[1] * gyro[0], -tmax, tmax),
		clamp(k1 * e1 - tab[2] * gyro[1], -tmax, tmax),
		clamp(k1 * e2 - tab[3] * gyro[2], -tmax, tmax),
		0.0f, 0.0f, tab[5],
	};
	for (uint i = 0u; i < NUM_ROTORS; i++) {
		device const float* row = tab + 6u + i * 7u;
		float thrust = 0.0f;
		for (uint j = 0u; j < 6u; j++) thrust += row[j] * wrench[j];
		out_pwm[i] = clamp(sqrt(max(thrust, 0.0f) / row[6]), 0.0f, 1.0f);
	}
}

// ---- sparse cell lookup (sorted keys + binary search; mirrors sparse_forward) -
inline uint bsearch_cell(device const ulong* keys, device const uchar* vals,
                         uint start, uint count, ulong addr) {
	if (count == 0u) return WNN_CELL_EMPTY;
	uint lo = 0u, hi = count;
	while (lo < hi) {
		uint mid = lo + (hi - lo) / 2u;
		ulong k = keys[start + mid];
		if (k == addr)      return (uint)vals[start + mid];
		else if (k < addr)  lo = mid + 1u;
		else                hi = mid;
	}
	return WNN_CELL_EMPTY;
}

// derivatives: (dω, dq) from Euler's eqn + quaternion kinematics
inline void derivatives(float3 omega, float4 q, float3 torque, constant Params& P,
                        thread float3& d_omega, thread float4& d_q) {
	float3 I  = float3(P.inertia0, P.inertia1, P.inertia2);
	float3 Iw = omega * I;
	float3 cor = cross(omega, Iw);
	float3 net = torque - cor;
	d_omega = net / I;
	float4 omega_q = float4(0.0f, omega.x, omega.y, omega.z);
	d_q = q_mul(q, omega_q) * 0.5f;
}

// Forward-only parameter view (field names match both Params and TrainParams), so
// forward_state / out_neuron_addr don't depend on which kernel calls them. Each
// kernel builds one on the stack from its own params struct (once per thread).
struct FwdParams {
	uint num_features, window, n_state, sbpn, obpn, bpf, frame_bits, sensor_total, num_motors;
	uint obs_tilt_p, obs_tilt_i, obs_peraxis_p, obs_peraxis_i, obs_peraxis_yaw, obs_pwm;
	uint obs_yaw_err, obs_yaw_err_i;   // yaw-anchor clean scalar channel
	float integral_leak, integral_scale, target0, target1, target2, dt;
	uint memory_mode;                  // ABI 12: cell decode/fire-bit semantics
};

// TERNARY untrained-cell decode weight — the controller's fixed PLN convention
// (cell_mode::TERNARY_EMPTY_VALUE twin).
constant float CTRL_TERNARY_EMPTY = 0.5f;

// 1-bit recurrent fire-bit per state cell — cell_mode::cell_fire_bit twin.
// QUAD: the QSR MSB (fired side). TERNARY/BINARY: cell == TRUE(1); the
// unwritten sparse read (EMPTY=2) does NOT fire.
inline bool ctrl_fire_bit(uint cell, uint memory_mode) {
	if (memory_mode == WNN_MODE_TERNARY || memory_mode == WNN_MODE_BINARY)
		return cell == 1u;
	return ((cell >> 1) & 1u) != 0u;
}

// =============================================================================
// derive_features — section (1) of the per-step forward: append the enabled H2
// derived features to the base sensors and tick the PHYSICAL-TIME accumulators
// (integ[], yaw_heading). Extracted from forward_state (02/07/2026, arm R) so
// action-repeat HOLD steps can tick the accumulators WITHOUT pushing the ring
// or forwarding the state layer — the exact twin of controller.rs
// compute_features() being "called exactly once per timestep". MUST mirror it.
// =============================================================================
inline void derive_features(
	thread float* sensors,            // [num_features]; [0..NUM_FEATURES) prefilled by caller
	thread const FwdParams& P,
	thread float* integ, thread float& yaw_heading,
	thread const float* pwm_acc)      // obs_pwm feature source (frozen in train, evolving in score)
{
	// H2 derived features — MUST mirror controller.rs compute_features() exactly.
	if (P.num_features > NUM_FEATURES) {
		float ax = sensors[3], ay = sensors[4], az = sensors[5];
		float tilt      = atan2(sqrt(ax*ax + ay*ay), az);
		float roll_est  = atan2(ay, az);
		float pitch_est = atan2(-ax, sqrt(ay*ay + az*az));
		// Yaw-anchored: integrate gyro-z (sensors[2]) with the real dt (heading was
		// seeded to init_yaw at episode reset) → absolute yaw. Else legacy gyro-z sum
		// (constant dt absorbed). Mirrors controller.rs compute_features bit-for-bit.
		bool yaw_anchored = (P.obs_yaw_err != 0u) || (P.obs_yaw_err_i != 0u);
		yaw_heading += yaw_anchored ? (sensors[2] * P.dt) : sensors[2];
		float roll_err  = P.target0 - roll_est;
		float pitch_err = P.target1 - pitch_est;
		float yaw_err   = P.target2 - yaw_heading;
		uint fi = NUM_FEATURES, ii = 0u;
		if (P.obs_tilt_p != 0u) { sensors[fi++] = tilt; }
		if (P.obs_tilt_i != 0u) {
			integ[ii] = P.integral_leak * integ[ii] + tilt;
			sensors[fi++] = integ[ii] * P.integral_scale; ii++;
		}
		if (P.obs_peraxis_p != 0u) {
			sensors[fi++] = roll_err; sensors[fi++] = pitch_err;
			if (P.obs_peraxis_yaw != 0u) { sensors[fi++] = yaw_err; }  // drop yaw when ref unobservable
		}
		if (P.obs_peraxis_i != 0u) {
			// roll/pitch always; yaw only when its dead-reckoned reference is enabled.
			uint npa = (P.obs_peraxis_yaw != 0u) ? 3u : 2u;
			float errs[3] = {roll_err, pitch_err, yaw_err};
			for (uint k = 0u; k < npa; k++) {
				integ[ii] = P.integral_leak * integ[ii] + errs[k];
				sensors[fi++] = integ[ii] * P.integral_scale; ii++;
			}
		}
		if (P.obs_pwm != 0u) {
			for (uint m = 0u; m < P.num_motors; m++) sensors[fi++] = pwm_acc[m];
		}
		// Yaw-anchor clean scalar channel (canonical order LAST): proportional yaw
		// error + its leaky integral. yaw_heading is the absolute anchored estimate.
		if (P.obs_yaw_err != 0u) { sensors[fi++] = yaw_err; }
		if (P.obs_yaw_err_i != 0u) {
			integ[ii] = P.integral_leak * integ[ii] + yaw_err;
			sensors[fi++] = integ[ii] * P.integral_scale; ii++;
		}
	}
}

// =============================================================================
// forward_state — shared per-step forward THROUGH the state layer. Given base
// sensors[0..NUM_FEATURES) prefilled by the caller (sim-derived in scoring,
// RECORDED gyro/accel/target in training), it: (1) appends the enabled H2 derived
// features via derive_features (single source), (2) pushes the frame into the
// K-window ring, (3) forwards the state layer over frozen cells → new_state[].
// Updates the recurrent per-step state (ring/filled, integ[], yaw_heading) in
// place. The OUTPUT-layer address is computed per-neuron by out_neuron_addr()
// below (so neither kernel has to materialize all num_out addresses). One source
// for both kernels = the train forward cannot drift from the score forward (the
// drift class behind the torque bug). Action-repeat kernels call this ONLY at
// decision steps (hold steps call derive_features alone).
// =============================================================================
inline void forward_state(
	thread float* sensors,            // [num_features]; [0..NUM_FEATURES) prefilled by caller
	thread const FwdParams& P,
	thread float* ring, thread uint& filled,
	thread float* integ, thread float& yaw_heading,
	thread const float* pwm_acc,      // obs_pwm feature source (frozen in train, evolving in score)
	thread const uchar* prev_state,
	device const int* state_conns, ulong conn_state_g,
	device const ulong* state_keys, device const uchar* state_vals,
	device const uint* state_off, device const uint* state_cnt, uint g_state_base,
	device const float* thresholds,
	thread uchar* new_state)          // OUT [n_state]
{
	// (1) H2 derived features + physical-time accumulator tick (single source).
	derive_features(sensors, P, integ, yaw_heading, pwm_acc);

	// (2) push current frame into the ring (drop oldest if full); stride = num_features
	uint nf = P.num_features;
	if (filled == P.window) {
		for (uint s = 1u; s < P.window; s++)
			for (uint f = 0u; f < nf; f++)
				ring[(s-1u)*nf + f] = ring[s*nf + f];
		filled = P.window - 1u;
	}
	for (uint f = 0u; f < nf; f++) ring[filled*nf + f] = sensors[f];
	filled += 1u;
	uint pad = P.window - filled;   // leading zero-padded window slots

	// (3) state layer forward (address per neuron via on-the-fly bits; frozen cells)
	for (uint n = 0u; n < P.n_state; n++) {
		ulong addr = 0ul;
		uint cbase = (uint)(conn_state_g) + n * P.sbpn;
		for (uint i = 0u; i < P.sbpn; i++) {
			int c = state_conns[cbase + i];
			bool bit = false;
			uint cu = (uint)c;
			if (cu < P.sensor_total) {
				uint slot = cu / P.frame_bits;
				uint within = cu % P.frame_bits;
				uint feat = within / P.bpf;
				uint b = within % P.bpf;
				if (slot >= pad) {
					uint hist = slot - pad;
					bit = ring[hist*P.num_features + feat] >= thresholds[feat*P.bpf + b];
				}
			} else {
				// 1-bit MSB-only recurrence: prefix_factor=1 since the 08/06/2026
				// migration (genome.py + EVERY CPU path use one bit per state neuron,
				// the QSR MSB). cu-sensor_total is the DIRECT state-neuron index. The
				// old idx>>1/idx&1 (2-bit) was a legacy-convention bug that silently
				// misread recurrent connections on GPU scoring (masked in parity by
				// untrained EMPTY cells); surfaced by the train parity test 2026-06-20.
				uint nn = cu - P.sensor_total;
				uchar v = prev_state[nn];
				bit = ctrl_fire_bit((uint)v, P.memory_mode);
			}
			if (bit) addr |= (1ul << (ulong)(P.sbpn - 1u - i));
		}
		uint gn = g_state_base + n;
		new_state[n] = (uchar)bsearch_cell(state_keys, state_vals,
		                                   state_off[gn], state_cnt[gn], addr);
	}
}

// Output-layer (Mealy) address for ONE output neuron n — current frame (sensors)
// + new_state bits. Shared by score (read+decode) and train (nudge); MSB-first.
inline ulong out_neuron_addr(
	uint n, thread const float* sensors, thread const uchar* new_state,
	device const int* output_conns, ulong conn_out_g, device const float* thresholds,
	thread const FwdParams& P)
{
	ulong addr = 0ul;
	uint cbase = (uint)(conn_out_g) + n * P.obpn;
	for (uint i = 0u; i < P.obpn; i++) {
		int c = output_conns[cbase + i];
		bool bit = false;
		uint cu = (uint)c;
		if (cu < P.frame_bits) {
			uint feat = cu / P.bpf, b = cu % P.bpf;
			bit = sensors[feat] >= thresholds[feat*P.bpf + b];
		} else {
			// 1-bit fire-bit (see forward_state): direct new_state-neuron index.
			uint nn = cu - P.frame_bits;
			uchar v = new_state[nn];
			bit = ctrl_fire_bit((uint)v, P.memory_mode);
		}
		if (bit) addr |= (1ul << (ulong)(P.obpn - 1u - i));
	}
	return addr;
}

kernel void controller_rollout(
	device const int*   state_conns  [[buffer(0)]],
	device const int*   output_conns [[buffer(1)]],
	device const ulong* state_keys   [[buffer(2)]],
	device const uchar* state_vals   [[buffer(3)]],
	device const uint*  state_off    [[buffer(4)]],
	device const uint*  state_cnt    [[buffer(5)]],
	device const ulong* out_keys     [[buffer(6)]],
	device const uchar* out_vals     [[buffer(7)]],
	device const uint*  out_off      [[buffer(8)]],
	device const uint*  out_cnt      [[buffer(9)]],
	device const float* thresholds   [[buffer(10)]],
	device const float* q0           [[buffer(11)]],
	device const float* omega0       [[buffer(12)]],
	constant Params&    P            [[buffer(13)]],
	device float*       out_reward   [[buffer(14)]],
	device float*       out_sumerr   [[buffer(15)]],
	device uint*        out_steps    [[buffer(16)]],
	device uint*        out_diverged [[buffer(17)]],
	device float*       out_jerk     [[buffer(18)]],
	device float*       out_mono     [[buffer(19)]],
	device float*       out_steady   [[buffer(20)]],
	// Transient-speed metrics (how FAST it corrects) — see run_episode parity.
	device float*       out_rise     [[buffer(21)]],  // rise time (s): |err| → 10% of initial
	device float*       out_settleab [[buffer(22)]],  // settle time (s), absolute ±2° band
	device float*       out_settlere [[buffer(23)]],  // settle time (s), relative ±5%-of-initial band
	device float*       out_itae     [[buffer(24)]],  // Σ t·|err|·dt (time-weighted; primary)
	device float*       out_iae      [[buffer(25)]],  // Σ |err|·dt
	device float*       out_ise      [[buffer(26)]],  // Σ err²·dt
	// Overactuated Phase 1: rotor table, read ONLY when USE_GEOMETRY (the
	// specialized pipeline dead-strips it otherwise → binding stays optional).
	device const RotorGpu* rotors    [[buffer(27)]],
	// Overactuated Phase 2: allocator-LQR baseline blob (optimal.rs
	// to_gpu_blob), read ONLY when P.alloc_baseline (host binds a 1-float pad
	// otherwise — runtime flag, not a function constant).
	device const float*    alloc_tab [[buffer(28)]],
	// Allocation-effort metric (Phase 3, Luiz 12/07): per-episode mean over
	// steps of Σ_m pwm_m² — the thrust-effort proxy the Σu² fitness term
	// ranks (misallocation costs EFFORT, not attitude error, on the
	// attitude-only sim; min-norm pinv is the optimum reference).
	device float*          out_effort [[buffer(29)]],
	uint2 tid [[thread_position_in_grid]])
{
	uint g = tid.x, e = tid.y;
	if (g >= P.num_genomes || e >= P.num_episodes) return;

	float4 q     = q_norm(float4(q0[e*4+0], q0[e*4+1], q0[e*4+2], q0[e*4+3]));
	float3 omega = float3(omega0[e*3+0], omega0[e*3+1], omega0[e*3+2]);
	float4 tgt   = q_norm(float4(1.0f, 0.0f, 0.0f, 0.0f)); // target RPY=0 → identity

	uchar prev_state[MAX_STATE_NEURONS];
	for (uint n = 0u; n < P.n_state; n++) prev_state[n] = 0u;   // reset()

	// K-window ring of observation frames (oldest-first in [0,filled)); stride =
	// P.num_features (≤ MAX_FEATURES). Holds base sensors + enabled H2 extras.
	float ring[MAX_WINDOW * MAX_FEATURES];
	uint filled = 0u;
	// H2 per-thread (per-episode) state: leaky-integral accumulators + gyro-z
	// dead-reckoned yaw heading. Zeroed at thread start = reset() at episode start.
	float integ[MAX_INTEGRALS] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	// Yaw-anchored ⇒ seed heading to the episode's true initial yaw from q0 (in-shader,
	// GPU-derived, mirrors controller.rs yaw_from_quat). Else legacy 0.0 (dead-reckon).
	float yaw_heading = (P.obs_yaw_err != 0u || P.obs_yaw_err_i != 0u) ? yaw_from_quat(q) : 0.0f;

	uint   g_state_base = g * P.n_state;
	uint   g_out_base   = g * (P.num_motors * P.levels);
	ulong  conn_state_g = (ulong)g * (ulong)P.n_state * (ulong)P.sbpn;
	ulong  conn_out_g   = (ulong)g * (ulong)(P.num_motors * P.levels) * (ulong)P.obpn;

	float cum_reward = 0.0f, sum_err = 0.0f;
	uint  steps = 0u, diverged = 0u;
	// Steady-state window (I-pressure metric): mean attitude error over the LAST
	// 20% of FULL steps. Isolates the residual offset from the transient so the
	// GA can rank an integrator's benefit (mirrors run_episode tail accumulation).
	uint  tail_start = (uint)ceil((float)P.steps * 0.80f);
	float tail_sum_err = 0.0f; uint tail_cnt = 0u;
	// Jerk: mean over steps of |Δpwm| (matches run_episode mean_pwm_jerk = mean
	// of sqrt(Σ_m (Δpwm_m)²)). Mono: thermometer-monotonicity violations on the
	// LAST emitted output thermometer (matches get_last_output_cells semantics).
	float prev_pwm[MAX_ROTORS]; bool has_prev = false;
	float sum_jerk = 0.0f; uint jerk_count = 0u;
	float sum_effort = 0.0f;   // Σ_t Σ_m pwm² (allocation-effort proxy)
	float mono_last = 0.0f;
	// Transient-speed tracking (mirrors run_episode). initial_err/band_rel set on
	// t==0's post-step err; sentinels default to FULL intended duration so a
	// "never rose / never settled" episode scores worst-case.
	float full_dur     = (float)P.steps * P.dt;
	float init_err     = -1.0f;            // <0 ⇒ not yet captured
	float band_rel     = 0.0f;
	const float band_abs = 0.0349065850f;  // radians(2.0)
	float rise_s       = full_dur; bool rise_done = false;
	float last_exc_abs = 0.0f, last_exc_rel = 0.0f;
	float itae = 0.0f, iae = 0.0f, ise = 0.0f;
	// Delta-control accumulator (persists across steps). Neutral = hover 0.5 per motor,
	// OR (decouple) T(bank0)→0.5, torque banks→0. Mirrors WnnController.pwm init/reset.
	// Init ALL MAX_ROTORS slots (compile-time count, unrolled): slots 4..7 ARE
	// read when an N>4 geometry is active (delta accumulate + hold re-emit) —
	// the old `< 4u` bound left them uninitialized there. Quad values unchanged.
	float pwm_acc[MAX_ROTORS];
	for (uint m = 0u; m < MAX_ROTORS; m++) pwm_acc[m] = (P.decouple_outputs != 0u && m >= 1u) ? 0.0f : 0.5f;
	// Action-repeat: the motor PWM emitted at the last DECISION step, returned
	// verbatim on hold steps. Hover-init like pwm_acc (never read before t=0's
	// decision). Mirrors WnnController.last_pwm.
	float last_pwm[MAX_ROTORS];
	for (uint m = 0u; m < MAX_ROTORS; m++) last_pwm[m] = (P.decouple_outputs != 0u && m >= 1u) ? 0.0f : 0.5f;

	// W2 disturbances: per-thread (this thread owns ONE episode rollout, so the
	// OU gust + gyro-bias state is episode-scoped, zeroed here = sim.reset()).
	// Per-episode seed via the channel-15 derivation on the GLOBAL episode
	// index (dist_ep_offset + e), so chunked dispatches draw identical noise.
	bool dist_on = (P.dist_enabled != 0u);
	uint ep_seed = dist_on
		? dist_hash(P.dist_seed, P.dist_ep_offset + e, 0u, DIST_CH_EP_SEED, 0u) : 0u;
	// QSR/PLN decode coin seed — ALWAYS derived (independent of dist_on): the exact
	// twin of CPU disturbance_episode_seed(dist_seed, ep). P.dist_seed is the pre-
	// folded (dist_seed32) base seed; hashing it on the GLOBAL episode index via
	// channel 15 gives the per-episode coin seed used DIRECTLY as the counter-hash
	// seed for the per-timestep decode coin (channel DIST_CH_MEM_COIN) below.
	uint coin_seed = dist_hash(P.dist_seed, P.dist_ep_offset + e, 0u, DIST_CH_EP_SEED, 0u);
	float gust[3] = {0.0f, 0.0f, 0.0f};
	float gyro_bias[3] = {0.0f, 0.0f, 0.0f};
	// E5 residual hybrid: per-episode PID integral state (roll,pitch,yaw), zeroed
	// here = AttitudePidRs::reset() at episode start. Unused when residual disabled.
	float pid_i[3] = {0.0f, 0.0f, 0.0f};

	// Forward-only param view for the shared forward_state / out_neuron_addr.
	FwdParams F = { P.num_features, P.window, P.n_state, P.sbpn, P.obpn, P.bpf,
	                P.frame_bits, P.sensor_total, P.num_motors,
	                P.obs_tilt_p, P.obs_tilt_i, P.obs_peraxis_p, P.obs_peraxis_i, P.obs_peraxis_yaw, P.obs_pwm,
	                P.obs_yaw_err, P.obs_yaw_err_i,
	                P.integral_leak, P.integral_scale, P.target0, P.target1, P.target2, P.dt,
	                P.memory_mode };

	for (uint t = 0u; t < P.steps; t++) {
		// is_unstable() check (top of run_episode loop)
		bool bad = false;
		if (!isfinite(omega.x) || !isfinite(omega.y) || !isfinite(omega.z)) bad = true;
		if (fabs(omega.x) > 50.0f || fabs(omega.y) > 50.0f || fabs(omega.z) > 50.0f) bad = true;
		if (!isfinite(q.x) || !isfinite(q.y) || !isfinite(q.z) || !isfinite(q.w)) bad = true;
		if (bad) { diverged = 1u; break; }

		// read_imu: gyro = omega; accel = -(gravity rotated to body)
		float3 grav_world = float3(0.0f, 0.0f, -P.gravity);
		float3 grav_body  = rotate_world_to_body(q, grav_world);
		float sensors[MAX_FEATURES];
		sensors[0] = omega.x; sensors[1] = omega.y; sensors[2] = omega.z;
		sensors[3] = -grav_body.x; sensors[4] = -grav_body.y; sensors[5] = -grav_body.z;
		sensors[6] = P.target0; sensors[7] = P.target1; sensors[8] = P.target2;
		// W2 D4 sensor noise — mirror controller.rs read_imu: gyro += bias
		// (+ white σ_g·ξ), accel += white σ_a·ξ; draws at step counter t.
		if (dist_on) {
			for (uint a3 = 0u; a3 < 3u; a3++) {
				sensors[a3] += gyro_bias[a3];
				if (P.dist_gyro_sigma > 0.0f)
					sensors[a3] += P.dist_gyro_sigma * dist_gauss(ep_seed, t, a3, DIST_CH_GYRO);
				if (P.dist_accel_sigma > 0.0f)
					sensors[3u + a3] += P.dist_accel_sigma * dist_gauss(ep_seed, t, a3, DIST_CH_ACCEL);
			}
		}

		// Action-repeat: hold steps tick ONLY the physical-time accumulators
		// (derive_features) and re-emit the last DECISION step's PWM — no ring
		// push, no forward, no decode; prev_state and mono_last stay the decision
		// step's (mirrors controller.rs step()'s hold branch). t=0 is a decision.
		float pwm[MAX_ROTORS];
		bool hold = (P.action_repeat > 1u) && (t % P.action_repeat != 0u);
		if (hold) {
			derive_features(sensors, F, integ, yaw_heading, pwm_acc);
			for (uint m = 0u; m < P.num_motors; m++) pwm[m] = last_pwm[m];
		} else {
			// H2 features + K-window ring + state-layer forward, via the shared
			// forward_state (single source with the training kernel).
			uchar new_state[MAX_STATE_NEURONS];
			forward_state(sensors, F, ring, filled, integ, yaw_heading, pwm_acc, prev_state,
			              state_conns, conn_state_g, state_keys, state_vals, state_off, state_cnt,
			              g_state_base, thresholds, new_state);

			// ---- output layer Strategy-5 decode, per motor (reads the cell at each
			//      neuron's shared Mealy address) --------------------------------
			float mono_step = 0.0f;
			// BINARY antagonist halves (E | I): each half is its own thermometer
			// run for mono; decode = 0.5 + (ΣE−ΣI)/levels (cell_mode.rs twin).
			uint bin_half = (P.memory_mode == WNN_MODE_BINARY) ? (P.levels / 2u) : 0u;
			for (uint m = 0u; m < P.num_motors; m++) {
				float sum = 0.0f;      // QUAD/TERNARY weight sum, or BINARY ΣE
				float sum_i = 0.0f;    // BINARY ΣI (inhibitory half)
				bool seen_one = false, prev_zero = false;
				for (uint l = 0u; l < P.levels; l++) {
					uint n = m * P.levels + l;
					uint gn = g_out_base + n;
					ulong addr = out_neuron_addr(n, sensors, new_state, output_conns, conn_out_g, thresholds, F);
					uint cell = bsearch_cell(out_keys, out_vals, out_off[gn], out_cnt[gn], addr);
					uint qv = cell & 3u;
					if (bin_half != 0u && l == bin_half) { seen_one = false; prev_zero = false; }
					// Thermometer "on" = mode fire-bit; a 1→0→1 gap is a monotonicity
					// violation (mirrors controller::monotonicity_violations_core).
					if (ctrl_fire_bit(qv, P.memory_mode)) { if (prev_zero && seen_one) mono_step += 1.0f; seen_one = true; prev_zero = false; }
					else { prev_zero = true; }
					float w = wnn_cell_weight(qv, P.memory_mode, CTRL_TERNARY_EMPTY);
					// QSR/PLN per-timestep stochastic decode (Part 5): the deterministic
					// weight IS the fire probability, so draw a fresh coin per physical
					// step t — u = dist_uniform(coin_seed, t, motor m, DIST_CH_MEM_COIN,
					// level l) — the exact CPU twin (decode_motor_cells_coin). QSR/PLN
					// never take the BINARY antagonist branch (bin_half==0 for them), so
					// this only ever feeds `sum`.
					if (P.memory_mode == WNN_MODE_QSR || P.memory_mode == WNN_MODE_PLN) {
						float u = dist_uniform(coin_seed, t, m, DIST_CH_MEM_COIN, l);
						w = (u < w) ? 1.0f : 0.0f;
					}
					if (bin_half != 0u && l >= bin_half) sum_i += w; else sum += w;
				}
				float decoded = (bin_half != 0u)
					? clamp(0.5f + (sum - sum_i) / (float)P.levels, 0.0f, 1.0f)
					: clamp(sum / (float)P.levels, 0.0f, 1.0f);
				if (P.decouple_outputs != 0u) {
					// H3: bank 0 = thrust T (neutral 0.5, [0,1]); banks 1..3 = torques
					// (neutral 0, [-1,1]). Accumulate per CONTROL; mix to motors after loop.
					bool  is_torque = (m >= 1u);
					float neutral = is_torque ? 0.0f : 0.5f;
					float lo = is_torque ? -1.0f : 0.0f;
					if (P.delta_control != 0u) {
						float delta = decoded_to_delta(decoded, P.delta_max, P.residual_neutral);
						pwm_acc[m] = clamp(neutral + P.delta_leak * (pwm_acc[m] - neutral) + delta, lo, 1.0f);
					} else {
						pwm_acc[m] = is_torque ? (decoded - 0.5f) * 2.0f : decoded;
					}
					pwm[m] = pwm_acc[m];   // control value; mixed to motors after the loop
				} else if (P.delta_control != 0u) {
					// Delta mode: decode→delta, leaky-accumulate the throttle (mirror
					// controller.rs step(): pwm = 0.5 + leak*(pwm-0.5) + delta).
					float delta  = decoded_to_delta(decoded, P.delta_max, P.residual_neutral);
					float leaked = 0.5f + P.delta_leak * (pwm_acc[m] - 0.5f);
					pwm_acc[m]   = clamp(leaked + delta, 0.0f, 1.0f);
					pwm[m]       = pwm_acc[m];
				} else {
					pwm[m] = decoded;   // absolute PWM
				}
			}
			if (P.decouple_outputs != 0u) {
				// Fixed control-allocation mix [T,τr,τp,τy]→motors (mirror controller.rs
				// mix_controls_to_motors). Signs: roll=−th1+th3, pitch=−th0+th2, yaw=Σ±.
				float T = pwm[0], tr = pwm[1], tp = pwm[2], ty = pwm[3];
				pwm[0] = clamp(T - tp + ty, 0.0f, 1.0f);  // front
				pwm[1] = clamp(T - tr - ty, 0.0f, 1.0f);  // left
				pwm[2] = clamp(T + tp + ty, 0.0f, 1.0f);  // back
				pwm[3] = clamp(T + tr - ty, 0.0f, 1.0f);  // right
			}
			mono_last = mono_step;   // keep the LAST DECISION step's violation count
			for (uint n = 0u; n < P.n_state; n++) prev_state[n] = new_state[n];
			for (uint m = 0u; m < P.num_motors; m++) last_pwm[m] = pwm[m];
		}
		// ---- Residual hybrid: pwm = clamp(base + clamp((wnn-0.5)·scale)) ----------
		// When enabled, the WNN output above is a SIGNED residual on an analytic
		// baseline (mirrors compose_residual / make_residual_action_fn):
		//   E5 (quad):       PID from pid_step — integral state advances every
		//                    physical step from the SAME gyro + q the WNN saw.
		//   Phase 2 (N-rotor): allocator-LQR from alloc_step (memoryless) —
		//                    the AllocBaseline twin, buffer(28).
		// (action_repeat=1 in the E5 config → every step is a decision; the hold
		// path holds the whole composed command via last_pwm.)
		if (P.residual_enabled != 0u && !hold) {
			float base[MAX_ROTORS];
			if (P.alloc_baseline != 0u) {
				alloc_step(q, sensors, P, alloc_tab, base);
			} else {
				pid_step(q, sensors, P, pid_i, base);   // quad-only: writes base[0..3]
			}
			for (uint m = 0u; m < P.num_motors; m++) {
				float r = clamp((pwm[m] - P.residual_neutral) * P.residual_scale,
				                -P.residual_clamp, P.residual_clamp);
				pwm[m] = clamp(base[m] + r, 0.0f, 1.0f);
			}
			for (uint m = 0u; m < P.num_motors; m++) last_pwm[m] = pwm[m];
		}
		// ---- COMMON tail (hold + decision): jerk, prev_pwm, sim, reward ------
		// Jerk: accumulate |Δpwm| once we have a previous step to diff against.
		// Hold steps re-emit last_pwm ⇒ Δ=0 (real smoothing, kept in the mean).
		if (has_prev) {
			float dj = 0.0f;
			for (uint m = 0u; m < P.num_motors; m++) { float d = pwm[m] - prev_pwm[m]; dj += d * d; }
			sum_jerk += sqrt(dj); jerk_count += 1u;
		}
		for (uint m = 0u; m < P.num_motors; m++) prev_pwm[m] = pwm[m];
		has_prev = true;
		// Allocation-effort (holds included). Two definitions, selected by run
		// config (constant within a cohort):
		//   alloc_baseline=0: raw Σ_m pwm² of the applied command.
		//   alloc_baseline=1 (+ geometry): EXCESS thrust-effort vs the
		//     pseudo-inverse optimum FOR THE SAME REALIZED WRENCH —
		//     T_i = k_i·p² (TRUE table), w = Σ B_i·T_i with
		//     B_i = [r_i×a_i + spin·kd·a_i ; a_i], T* = M_nominal·w,
		//     excess = ΣT² − ΣT*² ≥ 0. Invariant to collective shedding
		//     (the raw metric let the GA dump Fz the attitude-only sim never
		//     charges for — measured: pwm 0.5→0.365, fake '47% saving').
		{
			float se = 0.0f;
			if (P.alloc_baseline != 0u && USE_GEOMETRY) {
				float w6[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
				float sum_t2 = 0.0f;
				for (uint i = 0u; i < NUM_ROTORS; i++) {
					RotorGpu r = rotors[i];
					float pcl = clamp(pwm[i], 0.0f, 1.0f);
					float t = r.k_thrust * pcl * pcl;
					sum_t2 += t * t;
					float3 ax = float3(r.ax, r.ay, r.az);
					float3 tau_col = cross(float3(r.px, r.py, r.pz), ax) + r.spin * r.k_drag * ax;
					w6[0] += tau_col.x * t; w6[1] += tau_col.y * t; w6[2] += tau_col.z * t;
					w6[3] += ax.x * t;      w6[4] += ax.y * t;      w6[5] += ax.z * t;
				}
				float sum_topt2 = 0.0f;
				for (uint i = 0u; i < NUM_ROTORS; i++) {
					device const float* mrow = alloc_tab + 6u + i * 7u;
					float topt = 0.0f;
					for (uint j = 0u; j < 6u; j++) topt += mrow[j] * w6[j];
					topt = max(topt, 0.0f);   // fixed-pitch: the allocator clamps T ≥ 0
					sum_topt2 += topt * topt;
				}
				se = max(sum_t2 - sum_topt2, 0.0f);
			} else {
				for (uint m = 0u; m < P.num_motors; m++) se += pwm[m] * pwm[m];
			}
			sum_effort += se;
		}

		// ---- sim.step (RK4) --------------------------------------------------
		float3 torque;
		if (USE_GEOMETRY) {
			// Overactuated generic path: r x F + spin drag, accumulation order
			// mirrors overactuated.rs body_torque_asym exactly (parity gate).
			// Per-rotor asym is baked into rotors[i].k_thrust at upload.
			float3 tq = float3(0.0f);
			for (uint i = 0u; i < NUM_ROTORS; i++) {
				RotorGpu r = rotors[i];
				float p = clamp(pwm[i], 0.0f, 1.0f);
				float t = r.k_thrust * p * p;
				float3 ax = float3(r.ax, r.ay, r.az);
				float3 arm_tau = cross(float3(r.px, r.py, r.pz), ax * t);
				float drag = r.spin * r.k_drag * t;
				tq += arm_tau + drag * ax;
			}
			torque = tq;
		} else {
			float p0 = clamp(pwm[0],0.0f,1.0f), p1 = clamp(pwm[1],0.0f,1.0f);
			float p2 = clamp(pwm[2],0.0f,1.0f), p3 = clamp(pwm[3],0.0f,1.0f);
			// W2 D3: per-motor k_thrust multipliers (mirror body_torque_asym's
			// (k·asym)·p·p op order); clean path keeps the exact legacy expression.
			float th0, th1, th2, th3;
			if (dist_on) {
				th0 = P.k_thrust*P.dist_motor_asym0*p0*p0; th1 = P.k_thrust*P.dist_motor_asym1*p1*p1;
				th2 = P.k_thrust*P.dist_motor_asym2*p2*p2; th3 = P.k_thrust*P.dist_motor_asym3*p3*p3;
			} else {
				th0 = P.k_thrust*p0*p0; th1 = P.k_thrust*p1*p1;
				th2 = P.k_thrust*p2*p2; th3 = P.k_thrust*p3*p3;
			}
			torque = float3(P.arm_length*(-th1+th3),
			                P.arm_length*(-th0+th2),
			                P.k_drag*(th0-th1+th2-th3));
		}
		// W2 D1+D2: constant bias + OU gust, held constant over the RK4 step
		// (same add order as controller.rs: (base + bias) + gust).
		if (dist_on) {
			torque.x = torque.x + P.dist_tau_bias0 + gust[0];
			torque.y = torque.y + P.dist_tau_bias1 + gust[1];
			torque.z = torque.z + P.dist_tau_bias2 + gust[2];
		}
		float dt = P.dt;
		float3 k1o, k2o, k3o, k4o; float4 k1q, k2q, k3q, k4q;
		derivatives(omega, q, torque, P, k1o, k1q);
		derivatives(omega + k1o*(dt*0.5f), q + k1q*(dt*0.5f), torque, P, k2o, k2q);
		derivatives(omega + k2o*(dt*0.5f), q + k2q*(dt*0.5f), torque, P, k3o, k3q);
		derivatives(omega + k3o*dt,        q + k3q*dt,        torque, P, k4o, k4q);
		omega = omega + (k1o + k2o*2.0f + k3o*2.0f + k4o) * (dt/6.0f);
		q     = q_norm(q + (k1q + k2q*2.0f + k3q*2.0f + k4q) * (dt/6.0f));

		// W2 post-step state updates (mirror controller.rs step(): once per
		// physical step, AFTER the gust was used): OU innovation + bias walk.
		if (dist_on) {
			float sqrt_dt = sqrt(dt);
			for (uint a3 = 0u; a3 < 3u; a3++) {
				if (P.dist_gust_sigma > 0.0f) {
					float xi = dist_gauss(ep_seed, t, a3, DIST_CH_GUST);
					gust[a3] += -gust[a3] / P.dist_gust_tau_c * dt
						+ P.dist_gust_sigma * sqrt_dt * xi;
				}
				if (P.dist_gyro_bias_walk > 0.0f) {
					float xi = dist_gauss(ep_seed, t, a3, DIST_CH_GYRO_BIAS);
					gyro_bias[a3] += P.dist_gyro_bias_walk * sqrt_dt * xi;
				}
			}
		}

		// attitude_error (post-step) + reward (lambdas 0 in the eval path)
		float dot = q.x*tgt.x + q.y*tgt.y + q.z*tgt.z + q.w*tgt.w;
		float dot_abs = min(fabs(dot), 1.0f);
		float err = 2.0f * acos(dot_abs);
		cum_reward += -(err * err);
		sum_err += err;
		if (t >= tail_start) { tail_sum_err += err; tail_cnt += 1u; }

		// --- Transient-speed metrics (single pass) ---
		float t_s = (float)t * P.dt;
		if (init_err < 0.0f) { init_err = err; band_rel = 0.05f * init_err; }
		iae  += err * P.dt;
		ise  += err * err * P.dt;
		itae += t_s * err * P.dt;
		if (!rise_done && err <= 0.10f * init_err) { rise_s = t_s; rise_done = true; }
		if (err >= band_abs) last_exc_abs = t_s + P.dt;
		if (err >= band_rel) last_exc_rel = t_s + P.dt;

		steps = t + 1u;
	}

	uint idx = g * P.num_episodes + e;
	out_reward[idx]   = cum_reward;
	out_sumerr[idx]   = sum_err;
	out_steps[idx]    = steps;
	out_diverged[idx] = diverged;
	out_jerk[idx]     = jerk_count > 0u ? (sum_jerk / (float)jerk_count) : 0.0f;
	out_effort[idx]   = steps > 0u ? (sum_effort / (float)steps) : 0.0f;
	out_mono[idx]     = mono_last;
	// Diverged before reaching the tail window → no settled samples; fall back to
	// the whole-episode mean (already a failing episode, just keep it finite).
	out_steady[idx]   = tail_cnt > 0u ? (tail_sum_err / (float)tail_cnt)
	                                  : (steps > 0u ? sum_err / (float)steps : 0.0f);
	// Transient times: a diverged episode never settles → worst-case sentinels so
	// its late error blow-up can't fool last-excursion into a "fast" settle.
	if (diverged != 0u) {
		out_rise[idx]     = full_dur;
		out_settleab[idx] = full_dur;
		out_settlere[idx] = full_dur;
	} else {
		out_rise[idx]     = rise_s;
		out_settleab[idx] = min(last_exc_abs, full_dur);
		out_settlere[idx] = min(last_exc_rel, full_dur);
	}
	out_itae[idx] = itae;
	out_iae[idx]  = iae;
	out_ise[idx]  = ise;
}

// =============================================================================
// controller_train — GPU port of WnnController::split_retrain_output (the LIVE
// production output trainer under WNN_STATE_SPLIT=1). ONE THREAD = ONE GENOME:
// the thread replays that genome's recorded gated trajectories (gyro/accel/target
// + PID target PWM) IN ORDER through the shared forward (forward_state +
// out_neuron_addr, frozen state cells), and NUDGES the output cells toward the PID
// target via the marker-FSM table (find_or_claim_slot + slot_nudge from
// marker_slots.metal). Thread-per-genome ⇒ a genome's output table has a single
// writer ⇒ the clamped nudge order matches the CPU exactly ⇒ BIT-EXACT parity
// with split_retrain_output (slot_nudge == nudge_toward = clamp(cur±1,0,3)).
// Occupancy is num_genomes; P2 escalates to (genome,episode)+OI for the rest.
//
// ⚠ HOST CONTRACT: out_values must be pre-initialized to EMPTY=2 (QSR WEAK_TRUE).
// This is NOT the IDS QUAD baseline (WEAK_FALSE=1); the controller's untrained cell
// is the NEUTRAL_DECODE=0.75 HOVER sentinel (controller.rs:40) — an untrained motor
// bank must hover, not bleed throttle. The CPU nudges a fresh cell from read_cell=2,
// so slot_nudge(2,…)=clamp(2±1,0,3) matches exactly. Init to 1 breaks parity AND hover.
//
// Layout: per genome g there are ep_count[g] episodes starting at ep_base[g] in
// the flat episode arrays; episode j has step_count[j] steps starting at
// step_base[j] in the flat per-step arrays (gyros/accels/targets ×3, pid_pwms ×4).
// The output marker table is per (genome, out-neuron): region
// [slot_off[g*num_out+n] .. +slot_cap[g*num_out+n]) in the flat markers/keys/values.
// =============================================================================
struct TrainParams {
	uint num_genomes;
	uint n_state;
	uint sbpn;
	uint obpn;
	uint num_motors;
	uint levels;
	uint bpf;
	uint window;
	uint frame_bits;
	uint sensor_total;
	uint num_features;
	uint obs_tilt_p;
	uint obs_tilt_i;
	uint obs_peraxis_p;
	uint obs_peraxis_i;
	uint obs_peraxis_yaw;
	uint obs_pwm;
	uint obs_yaw_err;        // yaw-anchor: clean scalar channel
	uint obs_yaw_err_i;      // yaw-anchor: leaky integral
	float integral_leak;
	float integral_scale;
	float dt;                // yaw-anchor: gyro-z integration step
	uint  decouple_outputs;
	uint  delta_control;     // mirror: split_retrain_output uses output_decode_target regardless
	uint  selective;         // selective_output: skip output nudge where state is all-zero
	float target0;
	float target1;
	float target2;
	// Action-repeat N (arm R): the trainer re-forward must apply the SAME
	// decision mask as deploy, or trained addresses diverge from deploy's (the
	// +55pp mode-mismatch class). APPENDED at the END — layout must match the
	// Rust TrainParams exactly.
	uint  action_repeat;
	// Memory mode (ABI 12): decode/fire-bit/nudge semantics. APPENDED at END.
	uint  memory_mode;
};

// Mirror of WnnController::output_decode_target (controller.rs): map a per-motor
// ABSOLUTE commit target into the [0,1] raw-decode target. Absolute + decouple +
// torque bank (m>=1) inverts the (raw-0.5)*2 decode → raw = τ/2+0.5; else clamp.
inline float odt_train(uint motor, float target, constant TrainParams& P) {
	if (P.delta_control == 0u && P.decouple_outputs != 0u && motor >= 1u)
		return clamp(target * 0.5f + 0.5f, 0.0f, 1.0f);
	return clamp(target, 0.0f, 1.0f);
}

// Thermometer target bit per output level — cell_mode::output_target_bit twin.
// QUAD/TERNARY: cumulative thermometer. BINARY: antagonist E/I halves (net>0
// lights the first net·levels excitatory cells; net<0 the inhibitory ones).
inline bool ctrl_output_target_bit(float d, uint level_idx, uint levels, uint memory_mode) {
	if (memory_mode == WNN_MODE_BINARY) {
		uint half_l = levels / 2u;
		float net = d - 0.5f;
		if (level_idx < half_l) return net > 0.0f && (uint)(net * (float)levels) > level_idx;
		return net < 0.0f && (uint)(-net * (float)levels) > (level_idx - half_l);
	}
	return (uint)(d * (float)levels) > level_idx;
}

// TERNARY/BINARY training write — cell_mode::nudge_cell twin (last-write-wins
// direct set; the controller's DAGGER must be able to CORRECT earlier labels,
// unlike the IDS worker's TRUE-wins/one-shot batch semantics). Single writer
// per genome ⇒ deterministic, matches the CPU sequence exactly.
inline void slot_set_direct(device atomic_uint* slot_values, uint slot, bool target_true) {
	atomic_store_explicit(&slot_values[slot], target_true ? 1u : 0u, memory_order_relaxed);
}

kernel void controller_train(
	device const int*   state_conns   [[buffer(0)]],
	device const int*   output_conns  [[buffer(1)]],
	device const ulong* state_keys    [[buffer(2)]],
	device const uchar* state_vals    [[buffer(3)]],
	device const uint*  state_off     [[buffer(4)]],
	device const uint*  state_cnt     [[buffer(5)]],
	device const float* thresholds    [[buffer(6)]],
	device const uint*  ep_base       [[buffer(7)]],   // [num_genomes] first episode idx
	device const uint*  ep_count      [[buffer(8)]],   // [num_genomes]
	device const uint*  step_base     [[buffer(9)]],   // [num_episodes] first step idx
	device const uint*  step_count    [[buffer(10)]],  // [num_episodes]
	device const float* gyros         [[buffer(11)]],  // [total_steps*3]
	device const float* accels        [[buffer(12)]],  // [total_steps*3]
	device const float* targets       [[buffer(13)]],  // [total_steps*3]
	device const float* pid_pwms      [[buffer(14)]],  // [total_steps*4]
	device atomic_uint* out_markers   [[buffer(15)]],
	device ulong*       out_keys      [[buffer(16)]],
	device atomic_uint* out_values    [[buffer(17)]],
	device const uint*  slot_off      [[buffer(18)]],  // [num_genomes*num_out]
	device const uint*  slot_cap      [[buffer(19)]],  // [num_genomes*num_out]
	constant TrainParams& P           [[buffer(20)]],
	device uint*        out_writes     [[buffer(21)]],  // [num_genomes] cells touched (diagnostic)
	device const float* init_q        [[buffer(22)]],  // yaw-anchor: per-episode q0 [num_episodes*4]
	uint gid [[thread_position_in_grid]])
{
	uint g = gid;
	if (g >= P.num_genomes) return;

	uint num_out = P.num_motors * P.levels;
	uint g_state_base = g * P.n_state;
	ulong conn_state_g = (ulong)g * (ulong)P.n_state * (ulong)P.sbpn;
	ulong conn_out_g   = (ulong)g * (ulong)num_out * (ulong)P.obpn;
	uint  g_out_slot_base = g * num_out;

	uchar prev_state[MAX_STATE_NEURONS];
	uchar new_state[MAX_STATE_NEURONS];
	float ring[MAX_WINDOW * MAX_FEATURES];
	float integ[MAX_INTEGRALS];
	float pwm_acc[MAX_ROTORS];   // obs_pwm reads [0..num_motors) — N>4 safe
	uint  writes = 0u;

	// Forward-only param view for the shared forward_state / out_neuron_addr.
	FwdParams F = { P.num_features, P.window, P.n_state, P.sbpn, P.obpn, P.bpf,
	                P.frame_bits, P.sensor_total, P.num_motors,
	                P.obs_tilt_p, P.obs_tilt_i, P.obs_peraxis_p, P.obs_peraxis_i, P.obs_peraxis_yaw, P.obs_pwm,
	                P.obs_yaw_err, P.obs_yaw_err_i,
	                P.integral_leak, P.integral_scale, P.target0, P.target1, P.target2, P.dt,
	                P.memory_mode };

	uint E = ep_count[g];
	for (uint ej = 0u; ej < E; ej++) {
		uint ep = ep_base[g] + ej;
		// reset(): hover state, empty window, zero integrals/yaw, hover accumulator.
		for (uint n = 0u; n < P.n_state; n++) prev_state[n] = 0u;
		uint filled = 0u;
		for (uint k = 0u; k < MAX_INTEGRALS; k++) integ[k] = 0.0f;
		// Yaw-anchored ⇒ seed heading to this episode's true initial yaw from init_q
		// (train/record have no live q0; they replay traces). Same yaw_from_quat as score.
		float yaw_heading = (P.obs_yaw_err != 0u || P.obs_yaw_err_i != 0u)
			? yaw_from_quat(q_norm(float4(init_q[ep*4+0], init_q[ep*4+1], init_q[ep*4+2], init_q[ep*4+3])))
			: 0.0f;
		for (uint m = 0u; m < MAX_ROTORS; m++) pwm_acc[m] = (P.decouple_outputs != 0u && m >= 1u) ? 0.0f : 0.5f;

		uint T = step_count[ep];
		uint sbase = step_base[ep];
		for (uint t = 0u; t < T; t++) {
			uint s3 = (sbase + t) * 3u;
			uint s4 = (sbase + t) * 4u;
			float sensors[MAX_FEATURES];
			sensors[0] = gyros[s3+0]; sensors[1] = gyros[s3+1]; sensors[2] = gyros[s3+2];
			sensors[3] = accels[s3+0]; sensors[4] = accels[s3+1]; sensors[5] = accels[s3+2];
			sensors[6] = targets[s3+0]; sensors[7] = targets[s3+1]; sensors[8] = targets[s3+2];

			// Action-repeat hold: accumulators tick; no ring push / forward /
			// nudge (deploy reads NO addresses on hold steps). prev_state unchanged.
			// Mirrors CPU split_retrain_output's hold branch.
			if (P.action_repeat > 1u && (t % P.action_repeat != 0u)) {
				derive_features(sensors, F, integ, yaw_heading, pwm_acc);
				continue;
			}

			forward_state(sensors, F, ring, filled, integ, yaw_heading,
			              pwm_acc, prev_state, state_conns, conn_state_g, state_keys, state_vals,
			              state_off, state_cnt, g_state_base, thresholds, new_state);

			// selective_output: skip nudges where the recurrent state is all-zero
			// (preserve the hover-hold default), but still advance prev_state.
			bool state_active = false;
			for (uint n = 0u; n < P.n_state; n++) if (ctrl_fire_bit((uint)new_state[n], P.memory_mode)) { state_active = true; break; }
			if (P.selective != 0u && !state_active) {
				for (uint n = 0u; n < P.n_state; n++) prev_state[n] = new_state[n];
				continue;
			}

			for (uint n = 0u; n < num_out; n++) {
				uint motor = n / P.levels;
				uint level_idx = n % P.levels;
				ulong addr = out_neuron_addr(n, sensors, new_state, output_conns, conn_out_g, thresholds, F);
				float p = odt_train(motor, pid_pwms[s4 + motor], P);
				bool target_true = ctrl_output_target_bit(p, level_idx, P.levels, P.memory_mode);
				uint gn = g_out_slot_base + n;
				uint slot = find_or_claim_slot(out_markers, out_keys, slot_off[gn], slot_cap[gn], addr);
				if (slot != 0xFFFFFFFFu) {
					// QUAD: clamped ±1 lattice nudge. TERNARY/BINARY: direct set
					// (cell_mode::nudge_cell twin — see slot_set_direct).
					if (P.memory_mode == WNN_MODE_TERNARY || P.memory_mode == WNN_MODE_BINARY)
						slot_set_direct(out_values, slot, target_true);
					else
						slot_nudge(out_values, slot, target_true);
					writes += 1u;
				}
			}
			for (uint n = 0u; n < P.n_state; n++) prev_state[n] = new_state[n];
		}
	}
	out_writes[g] = writes;
}

// =============================================================================
// controller_record — GPU port of WnnController::split_record (P2). ONE THREAD =
// ONE (genome, episode): replays the recorded trajectory through the shared
// forward (forward_state, frozen state cells, now 1-bit-correct) and emits, per
// step, the records that scan_conflicts + the separator search consume:
//   rec_out_ins   : the output-layer INPUT bit vector  (frame_bits + state MSBs),
//                   packed — scan_conflicts buckets by this.
//   rec_state_ins : the state-layer INPUT bit vector    (sensor_total + state MSBs),
//                   packed — discriminative_walk / detect_accumulator read this.
//   rec_pwm       : the PID target PWM [4].
// Episodes are independent (each resets), so (genome,episode) parallel is safe;
// records are indexed by GLOBAL step = step_base[ep]+t (unique per step), so each
// record region has a single writer (plain |= into a zero-inited buffer). The
// in_state/in_out materialization mirrors split_record's own explicit loops (the
// CPU builds these separately from the address path too); parity guards equality.
// =============================================================================
inline void set_packed_bit(device uint* buf, uint base_word, uint pos) {
	buf[base_word + (pos >> 5)] |= (1u << (pos & 31u));
}

kernel void controller_record(
	device const int*   state_conns   [[buffer(0)]],
	device const ulong* state_keys    [[buffer(1)]],
	device const uchar* state_vals    [[buffer(2)]],
	device const uint*  state_off     [[buffer(3)]],
	device const uint*  state_cnt     [[buffer(4)]],
	device const float* thresholds    [[buffer(5)]],
	device const uint*  ep_base       [[buffer(6)]],
	device const uint*  ep_count      [[buffer(7)]],
	device const uint*  step_base     [[buffer(8)]],
	device const uint*  step_count    [[buffer(9)]],
	device const float* gyros         [[buffer(10)]],
	device const float* accels        [[buffer(11)]],
	device const float* targets       [[buffer(12)]],
	device const float* pid_pwms      [[buffer(13)]],
	device uint*        rec_out_ins   [[buffer(14)]],  // [total_records * out_words]
	device uint*        rec_state_ins [[buffer(15)]],  // [total_records * state_words]
	device float*       rec_pwm       [[buffer(16)]],  // [total_records * 4]
	constant TrainParams& P           [[buffer(17)]],
	device const float* init_q        [[buffer(18)]],  // yaw-anchor: per-episode q0 [num_episodes*4]
	// Action-repeat: first RECORD index per episode (decision-space prefix sums,
	// host-computed). Records exist only at decision steps; at N=1 this equals
	// step_base and the layout is bit-identical to the pre-repeat kernel.
	device const uint*  rec_base      [[buffer(19)]],  // [num_episodes]
	uint2 tid [[thread_position_in_grid]])
{
	uint g = tid.x, ej = tid.y;
	if (g >= P.num_genomes || ej >= ep_count[g]) return;
	uint ep = ep_base[g] + ej;

	uint g_state_base   = g * P.n_state;
	ulong conn_state_g  = (ulong)g * (ulong)P.n_state * (ulong)P.sbpn;
	uint out_input_len   = P.frame_bits + P.n_state;
	uint state_input_len = P.sensor_total + P.n_state;
	uint out_words   = (out_input_len + 31u) / 32u;
	uint state_words = (state_input_len + 31u) / 32u;

	FwdParams F = { P.num_features, P.window, P.n_state, P.sbpn, P.obpn, P.bpf,
	                P.frame_bits, P.sensor_total, P.num_motors,
	                P.obs_tilt_p, P.obs_tilt_i, P.obs_peraxis_p, P.obs_peraxis_i, P.obs_peraxis_yaw, P.obs_pwm,
	                P.obs_yaw_err, P.obs_yaw_err_i,
	                P.integral_leak, P.integral_scale, P.target0, P.target1, P.target2, P.dt,
	                P.memory_mode };

	uchar prev_state[MAX_STATE_NEURONS];
	uchar new_state[MAX_STATE_NEURONS];
	float ring[MAX_WINDOW * MAX_FEATURES];
	float integ[MAX_INTEGRALS];
	float pwm_acc[MAX_ROTORS];   // obs_pwm reads [0..num_motors) — N>4 safe
	for (uint n = 0u; n < P.n_state; n++) prev_state[n] = 0u;
	uint filled = 0u;
	for (uint k = 0u; k < MAX_INTEGRALS; k++) integ[k] = 0.0f;
	// Yaw-anchored ⇒ seed heading to this episode's true initial yaw from init_q.
	float yaw_heading = (P.obs_yaw_err != 0u || P.obs_yaw_err_i != 0u)
		? yaw_from_quat(q_norm(float4(init_q[ep*4+0], init_q[ep*4+1], init_q[ep*4+2], init_q[ep*4+3])))
		: 0.0f;
	for (uint m = 0u; m < MAX_ROTORS; m++) pwm_acc[m] = (P.decouple_outputs != 0u && m >= 1u) ? 0.0f : 0.5f;

	uint T = step_count[ep];
	uint sbase = step_base[ep];
	uint rbase = rec_base[ep];
	uint dec = 0u;   // decision (record) index within this episode
	for (uint t = 0u; t < T; t++) {
		uint s3 = (sbase + t) * 3u;
		uint s4 = (sbase + t) * 4u;
		float sensors[MAX_FEATURES];
		sensors[0] = gyros[s3+0]; sensors[1] = gyros[s3+1]; sensors[2] = gyros[s3+2];
		sensors[3] = accels[s3+0]; sensors[4] = accels[s3+1]; sensors[5] = accels[s3+2];
		sensors[6] = targets[s3+0]; sensors[7] = targets[s3+1]; sensors[8] = targets[s3+2];

		// Action-repeat hold: accumulators tick; no ring push / forward / record.
		// Mirrors CPU split_record's hold branch (records = decision steps only).
		if (P.action_repeat > 1u && (t % P.action_repeat != 0u)) {
			derive_features(sensors, F, integ, yaw_heading, pwm_acc);
			continue;
		}

		// prev_state (pre-update) is what in_state records; capture before forward.
		forward_state(sensors, F, ring, filled, integ, yaw_heading, pwm_acc, prev_state,
		              state_conns, conn_state_g, state_keys, state_vals,
		              state_off, state_cnt, g_state_base, thresholds, new_state);

		uint rec = rbase + dec;                     // global RECORD index (decision-space)
		dec += 1u;
		uint ob = rec * out_words;                  // base word for this record's out_ins
		uint sb = rec * state_words;                // base word for this record's state_ins

		// in_out = [ current frame (sensors thresholded) | new_state MSBs ]
		for (uint within = 0u; within < P.frame_bits; within++) {
			uint feat = within / P.bpf, b = within % P.bpf;
			if (sensors[feat] >= thresholds[feat*P.bpf + b]) set_packed_bit(rec_out_ins, ob, within);
		}
		for (uint n = 0u; n < P.n_state; n++)
			if (ctrl_fire_bit((uint)new_state[n], P.memory_mode)) set_packed_bit(rec_out_ins, ob, P.frame_bits + n);

		// in_state = [ K-window ring (thresholded, zero-padded front) | prev_state MSBs ]
		uint pad = P.window - filled;               // forward_state pushed this step's frame
		for (uint slot = 0u; slot < P.window; slot++) {
			if (slot < pad) continue;               // leading zero padding
			uint hist = slot - pad;
			for (uint within = 0u; within < P.frame_bits; within++) {
				uint feat = within / P.bpf, b = within % P.bpf;
				if (ring[hist*P.num_features + feat] >= thresholds[feat*P.bpf + b])
					set_packed_bit(rec_state_ins, sb, slot*P.frame_bits + within);
			}
		}
		for (uint n = 0u; n < P.n_state; n++)
			if (ctrl_fire_bit((uint)prev_state[n], P.memory_mode)) set_packed_bit(rec_state_ins, sb, P.sensor_total + n);

		rec_pwm[rec*4+0] = pid_pwms[s4+0]; rec_pwm[rec*4+1] = pid_pwms[s4+1];
		rec_pwm[rec*4+2] = pid_pwms[s4+2]; rec_pwm[rec*4+3] = pid_pwms[s4+3];

		for (uint n = 0u; n < P.n_state; n++) prev_state[n] = new_state[n];
	}
}

// =============================================================================
// controller_scan (P2b) — GPU half of scan_conflicts_coarse. ONE thread = one
// record. Computes the record's COARSE bucket key (bit-exact port of
// controller_split::coarse_key: k evenly-spaced thermometer bits per feature +
// the FULL state tail) and hash-claims a bucket SLOT for it (MarkerHashTable-style
// CAS, multi-word key). Writes slot_of[record] = slot. Records that share a coarse
// key land in the SAME slot → that slot IS the bucket. The host then groups
// records by slot (ascending record index = bit-exact instance order), computes
// PWM spread, filters >tau, and runs the adaptive-k loop — the cheap O(N) tail,
// matching scan_conflicts on the CPU. Slot PLACEMENT is internal (any
// deterministic hash works); only key-EQUALITY governs grouping, so this matches
// the CPU HashMap<coarse_key> grouping exactly. Multi-word keys handle the general
// case (num_features*k + n_state can exceed 64 bits).
// =============================================================================
constant uint MAX_KEY_WORDS = 16u;   // up to 1024 key bits (num_features*k + n_state)

inline uint get_packed_bit(device const uint* buf, uint base_word, uint pos) {
	return (buf[base_word + (pos >> 5u)] >> (pos & 31u)) & 1u;
}

inline bool keys_equal(device const ulong* a, thread const ulong* b, uint words) {
	for (uint w = 0u; w < words; w++) if (a[w] != b[w]) return false;
	return true;
}

// Multi-word find-or-claim. Slot keys stored as `key_words` consecutive ulongs
// per slot. Hash seed folds all words (FNV-1a) → slot placement; full-key compare
// on FINAL slots disambiguates collisions. Returns slot, or 0xFFFFFFFF on full.
inline uint find_or_claim_slot_multi(
	device atomic_uint* markers, device ulong* keys,
	uint key_words, uint cap, thread const ulong* key)
{
	ulong seed = 14695981039346656037ul;
	for (uint w = 0u; w < key_words; w++) { seed ^= key[w]; seed *= 1099511628211ul; }
	uint mask = cap - 1u;
	uint idx = slot_hash(seed, mask);
	for (uint probe = 0u; probe < cap; probe++) {
		uint slot = idx;
		uint m = atomic_load_explicit(&markers[slot], memory_order_relaxed);
		if (m == MARKER_FINAL) {
			if (keys_equal(keys + (ulong)slot * key_words, key, key_words)) return slot;
		} else if (m == MARKER_EMPTY) {
			uint expected = MARKER_EMPTY;
			bool won = false;
			for (uint retry = 0u; retry < 4u; retry++) {
				if (atomic_compare_exchange_weak_explicit(
					&markers[slot], &expected, MARKER_CLAIMED,
					memory_order_relaxed, memory_order_relaxed)) { won = true; break; }
				if (expected != MARKER_EMPTY) break;
			}
			if (won) {
				for (uint w = 0u; w < key_words; w++) keys[(ulong)slot * key_words + w] = key[w];
				atomic_store_explicit(&markers[slot], MARKER_FINAL, memory_order_relaxed);
				return slot;
			}
			if (expected == MARKER_FINAL && keys_equal(keys + (ulong)slot * key_words, key, key_words)) return slot;
			m = expected;
		}
		if (m != MARKER_EMPTY && m != MARKER_FINAL) {
			uint resolved = m;
			for (uint w = 0u; w < 64u; w++) {
				resolved = atomic_load_explicit(&markers[slot], memory_order_relaxed);
				if (resolved == MARKER_FINAL || resolved == MARKER_EMPTY) break;
			}
			if (resolved == MARKER_FINAL && keys_equal(keys + (ulong)slot * key_words, key, key_words)) return slot;
		}
		idx = (idx + 1u) & mask;
	}
	return 0xFFFFFFFFu;
}

struct ScanParams {
	uint num_records;
	uint out_input_len;   // frame_bits + n_state (bit length of one out_ins record)
	uint out_words;       // (out_input_len + 31) / 32
	uint frame_bits;      // num_features * bpf
	uint bpf;
	uint num_features;
	uint k;               // coarseness: bits sampled per feature
	uint key_words;       // ceil((num_features*k + n_state) / 64)
	uint slot_capacity;   // pow2
};

kernel void controller_scan(
	device const uint*   rec_out_ins  [[buffer(0)]],   // [num_records * out_words] packed
	device atomic_uint*  slot_markers [[buffer(1)]],   // [slot_capacity]
	device ulong*        slot_keys    [[buffer(2)]],   // [slot_capacity * key_words]
	device uint*         slot_of      [[buffer(3)]],   // [num_records] OUT
	constant ScanParams& P            [[buffer(4)]],
	uint i [[thread_position_in_grid]])
{
	if (i >= P.num_records) return;
	uint ob = i * P.out_words;
	thread ulong key[MAX_KEY_WORDS];
	for (uint w = 0u; w < MAX_KEY_WORDS; w++) key[w] = 0ul;
	uint kb = 0u;   // key-bit cursor (matches coarse_key push order)

	// k evenly-spaced thermometer bits per feature (bit-exact port of coarse_key).
	for (uint f = 0u; f < P.num_features; f++) {
		uint base = f * P.bpf;
		for (uint j = 0u; j < P.k; j++) {
			uint idx = (P.k >= P.bpf) ? j : (((j + 1u) * P.bpf) / (P.k + 1u));
			idx = min(idx, P.bpf - 1u);
			if (get_packed_bit(rec_out_ins, ob, base + idx) != 0u) key[kb >> 6u] |= (1ul << (kb & 63u));
			kb++;
		}
	}
	// full state tail: out_in[frame_bits .. out_input_len]
	for (uint s = P.frame_bits; s < P.out_input_len; s++) {
		if (get_packed_bit(rec_out_ins, ob, s) != 0u) key[kb >> 6u] |= (1ul << (kb & 63u));
		kb++;
	}

	slot_of[i] = find_or_claim_slot_multi(slot_markers, slot_keys, P.key_words, P.slot_capacity, key);
}

// =============================================================================
// controller_sep_walk (P3, Type-1) — GPU half of discriminative_walk. ONE thread
// = one (conflict, candidate-bit). For that bit it scans lags 1..=max_lag and
// counts how well "bit at (step-lag)" partitions the conflict's HIGH/LOW-PWM
// instances (separation_score purity numerator = correct classifications). Emits
// this bit's best (correct, lag, high_on); the HOST computes gain = correct/n in
// f32 (the exact CPU formula) and reduces over bits in candidate order with the
// CPU's `gain>best || (gain==best && lag<best.lag)` rule → the exact global
// Separator. **No float on the GPU**: since n is fixed within a conflict,
// argmax(correct/n) == argmax(correct), so we argmax on the INTEGER `correct`
// (exact) and leave the one float division to the host — this dodges Metal's
// fast-math reciprocal-approx division, which would flip the argmax on ULP ties.
// Lag ascends → strict `correct >` keeps the SMALLEST lag on ties (matches CPU).
// out_correct init 0xFFFFFFFF = "no valid lag"; host treats that (and gain≤0) as none.
// =============================================================================
struct SepParams {
	uint num_conflicts;
	uint num_bits;        // candidate_bits length B
	uint state_words;     // (state_in_len + 31)/32 — stride into PACKED state_ins
};

kernel void controller_sep_walk(
	device const uint*  conf_inst_base  [[buffer(0)]],   // [C]
	device const uint*  conf_inst_count [[buffer(1)]],   // [C]
	device const uint*  conf_inst       [[buffer(2)]],   // [total_inst] record indices
	device const uchar* labels          [[buffer(3)]],   // [total_inst] HIGH/LOW per instance
	device const uint*  ep_of           [[buffer(4)]],   // [num_records]
	device const uint*  step_of         [[buffer(5)]],   // [num_records]
	device const uint*  ep_start        [[buffer(6)]],   // [num_episodes]
	device const uint*  state_ins       [[buffer(7)]],   // [num_records * state_words] PACKED (resident)
	device const uint*  candidate_bits  [[buffer(8)]],   // [B]
	device const uint*  conf_max_lag    [[buffer(9)]],   // [C] per-conflict max lag
	device uint*        out_correct     [[buffer(10)]],  // [C*B] OUT (0xFFFFFFFF = none)
	device uint*        out_lag         [[buffer(11)]],  // [C*B] OUT
	device uint*        out_high        [[buffer(12)]],  // [C*B] OUT
	constant SepParams& P               [[buffer(13)]],
	uint2 tid [[thread_position_in_grid]])
{
	uint c = tid.x, bi = tid.y;
	if (c >= P.num_conflicts || bi >= P.num_bits) return;
	uint oidx = c * P.num_bits + bi;
	out_correct[oidx] = 0xFFFFFFFFu; out_lag[oidx] = 0u; out_high[oidx] = 0u;

	uint base = conf_inst_base[c];
	uint n = conf_inst_count[c];
	if (n == 0u) return;
	uint b = candidate_bits[bi];
	uint max_lag = conf_max_lag[c];

	bool found = false; uint best_correct = 0u; uint best_lag = 0u; uint best_high = 0u;
	for (uint lag = 1u; lag <= max_lag; lag++) {
		// require every instance to have history at this lag
		bool ok = true;
		for (uint j = 0u; j < n && ok; j++) {
			if (step_of[conf_inst[base + j]] < lag) ok = false;
		}
		if (!ok) continue;

		uint cnt[2][2] = {{0u, 0u}, {0u, 0u}};
		uint agree = 0u;
		for (uint j = 0u; j < n; j++) {
			uint i = conf_inst[base + j];
			uint rec = ep_start[ep_of[i]] + (step_of[i] - lag);
			uint feat = get_packed_bit(state_ins, rec * P.state_words, b);
			uint lab = labels[base + j] != 0u ? 1u : 0u;
			cnt[feat][lab] += 1u;
			if (feat == lab) agree += 1u;
		}
		uint correct = max(cnt[0][0], cnt[0][1]) + max(cnt[1][0], cnt[1][1]);
		uint high_on = (agree * 2u >= n) ? 1u : 0u;
		// lag ascends → strict > keeps min lag on ties (mirrors CPU `lag<best.lag`).
		if (!found || correct > best_correct) {
			found = true; best_correct = correct; best_lag = lag; best_high = high_on;
		}
	}
	if (found) {
		out_correct[oidx] = best_correct; out_lag[oidx] = best_lag; out_high[oidx] = best_high;
	}
}

// =============================================================================
// controller_sep_counts (P3, Type-2 shared primitive) — GPU half of the
// accumulator searches. ONE thread = one (conflict, candidate-bit). Computes the
// per-instance WINDOW COUNT: how many times the bit is set over the lookback
// lag 0..=max_lag (each instance counts only the lags it has history for, matching
// detect_accumulator). Counts are INTEGER → stored exactly as f32. The host then
// runs the exact CPU pearson + argmax for BOTH the increment (detect_accumulator)
// and bidirectional (detect_accumulator_bidir) searches — keeping Metal fast-math
// division/sqrt off the bit-exact correlation path. Layout: counts[block_base[c]
// + bi*n_c + j], block_base = prefix sum of B*n_c over conflicts.
// =============================================================================
struct CountParams {
	uint num_conflicts;
	uint num_bits;        // candidate_bits length B
	uint state_words;     // (state_in_len + 31)/32 — stride into PACKED state_ins
};

kernel void controller_sep_counts(
	device const uint*  conf_inst_base  [[buffer(0)]],   // [C]
	device const uint*  conf_inst_count [[buffer(1)]],   // [C]
	device const uint*  conf_inst       [[buffer(2)]],   // [total_inst]
	device const uint*  ep_of           [[buffer(3)]],   // [num_records]
	device const uint*  step_of         [[buffer(4)]],   // [num_records]
	device const uint*  ep_start        [[buffer(5)]],   // [num_episodes]
	device const uint*  state_ins       [[buffer(6)]],   // [num_records * state_words] PACKED (resident)
	device const uint*  candidate_bits  [[buffer(7)]],   // [B]
	device const uint*  conf_max_lag    [[buffer(8)]],   // [C]
	device const uint*  block_base      [[buffer(9)]],   // [C] prefix sum of B*n_c
	device float*       counts          [[buffer(10)]],  // OUT [sum(B*n_c)]
	constant CountParams& P             [[buffer(11)]],
	uint2 tid [[thread_position_in_grid]])
{
	uint c = tid.x, bi = tid.y;
	if (c >= P.num_conflicts || bi >= P.num_bits) return;
	uint base = conf_inst_base[c];
	uint n = conf_inst_count[c];
	uint b = candidate_bits[bi];
	uint max_lag = conf_max_lag[c];
	uint out_base = block_base[c] + bi * n;
	for (uint j = 0u; j < n; j++) {
		uint i = conf_inst[base + j];
		uint cnt = 0u;
		for (uint lag = 0u; lag <= max_lag; lag++) {
			if (step_of[i] >= lag) {
				uint rec = ep_start[ep_of[i]] + (step_of[i] - lag);
				if (get_packed_bit(state_ins, rec * P.state_words, b) != 0u) cnt += 1u;
			}
		}
		counts[out_base + j] = float(cnt);
	}
}

// =============================================================================
// controller_plant_table (P4 planting) — GPU half of the SPARSE truth-table write
// shared by split_plant_latch AND split_install_counter. ONE thread = one record.
// Computes the selected state neuron's address for that record
// (compute_address_sparse over the record's packed state-layer input), masks OUT
// the `num_rel` relevant bit positions → the visited BASE, then for each of the
// 2^num_rel combos writes combo_vals[combo] into the neuron's MarkerHashTable.
// combo bit j scatters to relevant position rel_pos[j]; the host precomputes
// combo_vals to encode the exact CPU `on` logic (latch / counter level-0 / level-k).
// No dedup needed: cell value is a pure function of the FULL address (relevant bits
// live AT rel_pos[j]), so records sharing a base write IDENTICAL values to IDENTICAL
// addresses — concurrent direct-sets converge.
// =============================================================================
struct PlantParams {
	uint num_records;
	uint sbpn;
	uint state_words;     // (state_input_len + 31) / 32
	uint slot_cap;        // pow2 marker-table capacity (one neuron)
	uint num_rel;         // number of relevant bits (latch=2, counter Lk=3)
};

kernel void controller_plant_table(
	device const uint*  state_ins  [[buffer(0)]],   // [num_records * state_words] packed
	device const int*   conns      [[buffer(1)]],   // [sbpn] selected neuron's connections
	device const uint*  rel_pos    [[buffer(2)]],   // [num_rel] relevant connection positions
	device const uchar* combo_vals [[buffer(3)]],   // [1<<num_rel] mode-native cell per combo (plant_cell)
	device atomic_uint* markers    [[buffer(4)]],   // [slot_cap]
	device ulong*       keys       [[buffer(5)]],   // [slot_cap]
	device atomic_uint* values     [[buffer(6)]],   // [slot_cap]
	constant PlantParams& P        [[buffer(7)]],
	uint r [[thread_position_in_grid]])
{
	if (r >= P.num_records) return;
	uint sb = r * P.state_words;
	ulong addr = 0ul;
	for (uint i = 0u; i < P.sbpn; i++) {
		int ci = conns[i];
		if (ci >= 0 && get_packed_bit(state_ins, sb, (uint)ci) != 0u) addr |= (1ul << (P.sbpn - 1u - i));
	}
	// Relevant masks; clear them from addr → base.
	ulong rmask[8];
	ulong clear = 0ul;
	for (uint j = 0u; j < P.num_rel; j++) { rmask[j] = 1ul << (P.sbpn - 1u - rel_pos[j]); clear |= rmask[j]; }
	ulong base = addr & ~clear;
	uint ncombo = 1u << P.num_rel;
	for (uint combo = 0u; combo < ncombo; combo++) {
		ulong a = base;
		for (uint j = 0u; j < P.num_rel; j++) if (((combo >> j) & 1u) != 0u) a |= rmask[j];
		uint slot = find_or_claim_slot(markers, keys, 0u, P.slot_cap, a);
		if (slot != 0xFFFFFFFFu) atomic_store_explicit(&values[slot], (uint)combo_vals[combo], memory_order_relaxed);
	}
}

// =============================================================================
// controller_plant_bidir (P4, Type-2 bidirectional planting) — GPU half of
// split_install_counter_bidir's DENSE truth-table write. ONE thread = one address
// a in 0..2^sbpn. The bidir on-logic depends only on five fixed address bits
// (up/dn/lower/self/upper at MSB positions sbpn-1..sbpn-5), so the table is
// identical for every chain neuron and each thread writes a DISTINCT cell — no hash
// table, no contention. The host structural-verify guarantees the wiring; this just
// materializes on(a). out_vals[a] = on ? on_val : off_val — the host passes the
// mode-native planted cells (cell_mode::plant_cell: QUAD 3/1, TERNARY/BINARY 1/0)
// so the kernel stays cell-semantics-agnostic like controller_plant_table.
// =============================================================================
struct BidirPlantParams { uint sbpn; uint on_val; uint off_val; };

kernel void controller_plant_bidir(
	device uchar*           out_vals [[buffer(0)]],   // [1<<sbpn]
	constant BidirPlantParams& P     [[buffer(1)]],
	uint a [[thread_position_in_grid]])
{
	uint n = 1u << P.sbpn;
	if (a >= n) return;
	uint up_b    = (a >> (P.sbpn - 1u)) & 1u;
	uint dn_b    = (a >> (P.sbpn - 2u)) & 1u;
	uint lower_b = (a >> (P.sbpn - 3u)) & 1u;
	uint self_b  = (a >> (P.sbpn - 4u)) & 1u;
	uint upper_b = (a >> (P.sbpn - 5u)) & 1u;
	bool on;
	if (dn_b == 1u && self_b == 1u && upper_b == 0u) on = false;   // decrement (top unwinds)
	else if (self_b == 1u) on = true;                              // hold
	else on = (up_b == 1u && lower_b == 1u);                       // increment
	out_vals[a] = uchar(on ? P.on_val : P.off_val);
}

// =============================================================================
// mht_lookup (P5b foundation) — read-only MarkerHashTable probe. The resident-cell
// forward variant reads state/output cells from a persistent MHT instead of the
// sorted-array bsearch (bsearch_cell). Open addressing: hash the address, probe
// forward; an EMPTY slot means "absent" (find_or_claim would have claimed it) →
// return WNN_CELL_EMPTY (=2, the controller hover default, matching bsearch_cell's
// miss). Assumes the read phase has no concurrent writes (markers ∈ {EMPTY,FINAL}).
// Parity-gated vs bsearch_cell on the same cell set (controller_mht_probe).
// =============================================================================
inline uint mht_lookup(
	device const uint*  markers, device const ulong* keys, device const uint* values,
	uint slot_off, uint slot_cap, ulong addr)
{
	uint mask = slot_cap - 1u;
	uint idx = slot_hash(addr, mask);
	for (uint probe = 0u; probe < slot_cap; probe++) {
		uint slot = slot_off + idx;
		uint m = markers[slot];
		if (m == MARKER_EMPTY) return WNN_CELL_EMPTY;          // open-addressing miss
		if (m == MARKER_FINAL && keys[slot] == addr) return values[slot] & 0xFFu;
		idx = (idx + 1u) & mask;
	}
	return WNN_CELL_EMPTY;
}

struct MhtParams {
	uint slot_cap;
	uint num_cells;
	uint sorted_count;
	uint num_q;
};

// Populate the MHT from a cell set (thread = one cell): claim slot + store value.
kernel void controller_mht_populate(
	device const ulong* cell_addrs [[buffer(0)]],
	device const uchar* cell_vals  [[buffer(1)]],
	device atomic_uint* markers    [[buffer(2)]],
	device ulong*       keys        [[buffer(3)]],
	device atomic_uint* values     [[buffer(4)]],
	constant MhtParams& P          [[buffer(5)]],
	uint i [[thread_position_in_grid]])
{
	if (i >= P.num_cells) return;
	uint slot = find_or_claim_slot(markers, keys, 0u, P.slot_cap, cell_addrs[i]);
	if (slot != 0xFFFFFFFFu) atomic_store_explicit(&values[slot], (uint)cell_vals[i], memory_order_relaxed);
}

// Probe both paths (thread = one query addr) → host compares for parity.
kernel void controller_mht_probe(
	device const ulong* sorted_keys [[buffer(0)]],
	device const uchar* sorted_vals [[buffer(1)]],
	device const uint*  markers     [[buffer(2)]],
	device const ulong* keys        [[buffer(3)]],
	device const uint*  values      [[buffer(4)]],
	device const ulong* queries     [[buffer(5)]],
	device uint*        out_bsearch [[buffer(6)]],
	device uint*        out_mht     [[buffer(7)]],
	constant MhtParams& P           [[buffer(8)]],
	uint q [[thread_position_in_grid]])
{
	if (q >= P.num_q) return;
	ulong addr = queries[q];
	out_bsearch[q] = bsearch_cell(sorted_keys, sorted_vals, 0u, P.sorted_count, addr);
	out_mht[q] = mht_lookup(markers, keys, values, 0u, P.slot_cap, addr);
}
