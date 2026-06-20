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
#define MAX_STATE_NEURONS 32
#define MAX_WINDOW        8
#define MAX_INTEGRALS     4     // tilt_i + 3×peraxis_i

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
	uint  obs_pwm;         // expose the raw throttle accumulator (num_motors feats)
	float integral_leak;
	float integral_scale;
	uint  decouple_outputs; // H3: 4 banks are controls [T,τr,τp,τy] → mix to motors
};

// Mirror of controller.rs decoded_to_delta: map a Strategy-5 decode in [0,1] to a
// per-step PWM delta in [-delta_max, +delta_max], piecewise-linear about NEUTRAL=0.75.
inline float decoded_to_delta(float decoded, float delta_max) {
	const float n = 0.75f;
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
	uint obs_tilt_p, obs_tilt_i, obs_peraxis_p, obs_peraxis_i, obs_pwm;
	float integral_leak, integral_scale, target0, target1, target2;
};

// =============================================================================
// forward_state — shared per-step forward THROUGH the state layer. Given base
// sensors[0..NUM_FEATURES) prefilled by the caller (sim-derived in scoring,
// RECORDED gyro/accel/target in training), it: (1) appends the enabled H2 derived
// features, (2) pushes the frame into the K-window ring, (3) forwards the state
// layer over frozen cells → new_state[]. Updates the recurrent per-step state
// (ring/filled, integ[], yaw_heading) in place. The OUTPUT-layer address is
// computed per-neuron by out_neuron_addr() below (so neither kernel has to
// materialize all num_out addresses). One source for both kernels = the train
// forward cannot drift from the score forward (the drift class behind the torque bug).
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
	// (1) H2 derived features — MUST mirror controller.rs compute_features() exactly.
	if (P.num_features > NUM_FEATURES) {
		float ax = sensors[3], ay = sensors[4], az = sensors[5];
		float tilt      = atan2(sqrt(ax*ax + ay*ay), az);
		float roll_est  = atan2(ay, az);
		float pitch_est = atan2(-ax, sqrt(ay*ay + az*az));
		yaw_heading += sensors[2];
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
			sensors[fi++] = roll_err; sensors[fi++] = pitch_err; sensors[fi++] = yaw_err;
		}
		if (P.obs_peraxis_i != 0u) {
			float errs[3] = {roll_err, pitch_err, yaw_err};
			for (uint k = 0u; k < 3u; k++) {
				integ[ii] = P.integral_leak * integ[ii] + errs[k];
				sensors[fi++] = integ[ii] * P.integral_scale; ii++;
			}
		}
		if (P.obs_pwm != 0u) {
			for (uint m = 0u; m < P.num_motors; m++) sensors[fi++] = pwm_acc[m];
		}
	}

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
				uint idx = cu - P.sensor_total;
				uint nn = idx >> 1, which = idx & 1u;
				uchar v = prev_state[nn];
				bit = which == 0u ? (((v >> 1) & 1u) != 0u) : ((v & 1u) != 0u);
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
			uint idx = cu - P.frame_bits;
			uint nn = idx >> 1, which = idx & 1u;
			uchar v = new_state[nn];
			bit = which == 0u ? (((v >> 1) & 1u) != 0u) : ((v & 1u) != 0u);
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
	float integ[MAX_INTEGRALS] = {0.0f, 0.0f, 0.0f, 0.0f};
	float yaw_heading = 0.0f;

	uint   g_state_base = g * P.n_state;
	uint   g_out_base   = g * (P.num_motors * P.levels);
	ulong  conn_state_g = (ulong)g * (ulong)P.n_state * (ulong)P.sbpn;
	ulong  conn_out_g   = (ulong)g * (ulong)(P.num_motors * P.levels) * (ulong)P.obpn;

	float cum_reward = 0.0f, sum_err = 0.0f;
	uint  steps = 0u, diverged = 0u;
	// Jerk: mean over steps of |Δpwm| (matches run_episode mean_pwm_jerk = mean
	// of sqrt(Σ_m (Δpwm_m)²)). Mono: thermometer-monotonicity violations on the
	// LAST emitted output thermometer (matches get_last_output_cells semantics).
	float prev_pwm[4]; bool has_prev = false;
	float sum_jerk = 0.0f; uint jerk_count = 0u;
	float mono_last = 0.0f;
	// Delta-control accumulator (persists across steps). Neutral = hover 0.5 per motor,
	// OR (decouple) T(bank0)→0.5, torque banks→0. Mirrors WnnController.pwm init/reset.
	float pwm_acc[4];
	for (uint m = 0u; m < 4u; m++) pwm_acc[m] = (P.decouple_outputs != 0u && m >= 1u) ? 0.0f : 0.5f;

	// Forward-only param view for the shared forward_state / out_neuron_addr.
	FwdParams F = { P.num_features, P.window, P.n_state, P.sbpn, P.obpn, P.bpf,
	                P.frame_bits, P.sensor_total, P.num_motors,
	                P.obs_tilt_p, P.obs_tilt_i, P.obs_peraxis_p, P.obs_peraxis_i, P.obs_pwm,
	                P.integral_leak, P.integral_scale, P.target0, P.target1, P.target2 };

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

		// H2 features + K-window ring + state-layer forward, via the shared
		// forward_state (single source with the training kernel).
		uchar new_state[MAX_STATE_NEURONS];
		forward_state(sensors, F, ring, filled, integ, yaw_heading, pwm_acc, prev_state,
		              state_conns, conn_state_g, state_keys, state_vals, state_off, state_cnt,
		              g_state_base, thresholds, new_state);

		// ---- output layer Strategy-5 decode, per motor (reads the cell at each
		//      neuron's shared Mealy address) ------------------------------------
		float pwm[4];
		float mono_step = 0.0f;
		for (uint m = 0u; m < P.num_motors; m++) {
			float sum = 0.0f;
			bool seen_one = false, prev_zero = false;
			for (uint l = 0u; l < P.levels; l++) {
				uint n = m * P.levels + l;
				uint gn = g_out_base + n;
				ulong addr = out_neuron_addr(n, sensors, new_state, output_conns, conn_out_g, thresholds, F);
				uint cell = bsearch_cell(out_keys, out_vals, out_off[gn], out_cnt[gn], addr);
				uint qv = cell & 3u;
				// Thermometer "on" = QSR MSB set (TRUE/WEAK_TRUE); a 1→0→1 gap is a
				// monotonicity violation (mirrors controller::monotonicity_violations).
				if (qv >= 2u) { if (prev_zero && seen_one) mono_step += 1.0f; seen_one = true; prev_zero = false; }
				else { prev_zero = true; }
				sum += WNN_QUAD_WEIGHTS[qv];
			}
			float decoded = clamp(sum / (float)P.levels, 0.0f, 1.0f);
			if (P.decouple_outputs != 0u) {
				// H3: bank 0 = thrust T (neutral 0.5, [0,1]); banks 1..3 = torques
				// (neutral 0, [-1,1]). Accumulate per CONTROL; mix to motors after loop.
				bool  is_torque = (m >= 1u);
				float neutral = is_torque ? 0.0f : 0.5f;
				float lo = is_torque ? -1.0f : 0.0f;
				if (P.delta_control != 0u) {
					float delta = decoded_to_delta(decoded, P.delta_max);
					pwm_acc[m] = clamp(neutral + P.delta_leak * (pwm_acc[m] - neutral) + delta, lo, 1.0f);
				} else {
					pwm_acc[m] = is_torque ? (decoded - 0.5f) * 2.0f : decoded;
				}
				pwm[m] = pwm_acc[m];   // control value; mixed to motors after the loop
			} else if (P.delta_control != 0u) {
				// Delta mode: decode→delta, leaky-accumulate the throttle (mirror
				// controller.rs step(): pwm = 0.5 + leak*(pwm-0.5) + delta).
				float delta  = decoded_to_delta(decoded, P.delta_max);
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
		mono_last = mono_step;   // keep the LAST step's thermometer-violation count
		// Jerk: accumulate |Δpwm| once we have a previous step to diff against.
		if (has_prev) {
			float dj = 0.0f;
			for (uint m = 0u; m < P.num_motors; m++) { float d = pwm[m] - prev_pwm[m]; dj += d * d; }
			sum_jerk += sqrt(dj); jerk_count += 1u;
		}
		for (uint m = 0u; m < P.num_motors; m++) prev_pwm[m] = pwm[m];
		has_prev = true;
		for (uint n = 0u; n < P.n_state; n++) prev_state[n] = new_state[n];

		// ---- sim.step (RK4) --------------------------------------------------
		float p0 = clamp(pwm[0],0.0f,1.0f), p1 = clamp(pwm[1],0.0f,1.0f);
		float p2 = clamp(pwm[2],0.0f,1.0f), p3 = clamp(pwm[3],0.0f,1.0f);
		float th0 = P.k_thrust*p0*p0, th1 = P.k_thrust*p1*p1;
		float th2 = P.k_thrust*p2*p2, th3 = P.k_thrust*p3*p3;
		float3 torque = float3(P.arm_length*(-th1+th3),
		                       P.arm_length*(-th0+th2),
		                       P.k_drag*(th0-th1+th2-th3));
		float dt = P.dt;
		float3 k1o, k2o, k3o, k4o; float4 k1q, k2q, k3q, k4q;
		derivatives(omega, q, torque, P, k1o, k1q);
		derivatives(omega + k1o*(dt*0.5f), q + k1q*(dt*0.5f), torque, P, k2o, k2q);
		derivatives(omega + k2o*(dt*0.5f), q + k2q*(dt*0.5f), torque, P, k3o, k3q);
		derivatives(omega + k3o*dt,        q + k3q*dt,        torque, P, k4o, k4q);
		omega = omega + (k1o + k2o*2.0f + k3o*2.0f + k4o) * (dt/6.0f);
		q     = q_norm(q + (k1q + k2q*2.0f + k3q*2.0f + k4q) * (dt/6.0f));

		// attitude_error (post-step) + reward (lambdas 0 in the eval path)
		float dot = q.x*tgt.x + q.y*tgt.y + q.z*tgt.z + q.w*tgt.w;
		float dot_abs = min(fabs(dot), 1.0f);
		float err = 2.0f * acos(dot_abs);
		cum_reward += -(err * err);
		sum_err += err;
		steps = t + 1u;
	}

	uint idx = g * P.num_episodes + e;
	out_reward[idx]   = cum_reward;
	out_sumerr[idx]   = sum_err;
	out_steps[idx]    = steps;
	out_diverged[idx] = diverged;
	out_jerk[idx]     = jerk_count > 0u ? (sum_jerk / (float)jerk_count) : 0.0f;
	out_mono[idx]     = mono_last;
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
	uint obs_pwm;
	float integral_leak;
	float integral_scale;
	uint  decouple_outputs;
	uint  delta_control;     // mirror: split_retrain_output uses output_decode_target regardless
	uint  selective;         // selective_output: skip output nudge where state is all-zero
	float target0;
	float target1;
	float target2;
};

// Mirror of WnnController::output_decode_target (controller.rs): map a per-motor
// ABSOLUTE commit target into the [0,1] raw-decode target. Absolute + decouple +
// torque bank (m>=1) inverts the (raw-0.5)*2 decode → raw = τ/2+0.5; else clamp.
inline float odt_train(uint motor, float target, constant TrainParams& P) {
	if (P.delta_control == 0u && P.decouple_outputs != 0u && motor >= 1u)
		return clamp(target * 0.5f + 0.5f, 0.0f, 1.0f);
	return clamp(target, 0.0f, 1.0f);
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
	float pwm_acc[4];
	uint  writes = 0u;

	// Forward-only param view for the shared forward_state / out_neuron_addr.
	FwdParams F = { P.num_features, P.window, P.n_state, P.sbpn, P.obpn, P.bpf,
	                P.frame_bits, P.sensor_total, P.num_motors,
	                P.obs_tilt_p, P.obs_tilt_i, P.obs_peraxis_p, P.obs_peraxis_i, P.obs_pwm,
	                P.integral_leak, P.integral_scale, P.target0, P.target1, P.target2 };

	uint E = ep_count[g];
	for (uint ej = 0u; ej < E; ej++) {
		uint ep = ep_base[g] + ej;
		// reset(): hover state, empty window, zero integrals/yaw, hover accumulator.
		for (uint n = 0u; n < P.n_state; n++) prev_state[n] = 0u;
		uint filled = 0u;
		for (uint k = 0u; k < MAX_INTEGRALS; k++) integ[k] = 0.0f;
		float yaw_heading = 0.0f;
		for (uint m = 0u; m < 4u; m++) pwm_acc[m] = (P.decouple_outputs != 0u && m >= 1u) ? 0.0f : 0.5f;

		uint T = step_count[ep];
		uint sbase = step_base[ep];
		for (uint t = 0u; t < T; t++) {
			uint s3 = (sbase + t) * 3u;
			uint s4 = (sbase + t) * 4u;
			float sensors[MAX_FEATURES];
			sensors[0] = gyros[s3+0]; sensors[1] = gyros[s3+1]; sensors[2] = gyros[s3+2];
			sensors[3] = accels[s3+0]; sensors[4] = accels[s3+1]; sensors[5] = accels[s3+2];
			sensors[6] = targets[s3+0]; sensors[7] = targets[s3+1]; sensors[8] = targets[s3+2];

			forward_state(sensors, F, ring, filled, integ, yaw_heading,
			              pwm_acc, prev_state, state_conns, conn_state_g, state_keys, state_vals,
			              state_off, state_cnt, g_state_base, thresholds, new_state);

			// selective_output: skip nudges where the recurrent state is all-zero
			// (preserve the hover-hold default), but still advance prev_state.
			bool state_active = false;
			for (uint n = 0u; n < P.n_state; n++) if (((new_state[n] >> 1) & 1u) != 0u) { state_active = true; break; }
			if (P.selective != 0u && !state_active) {
				for (uint n = 0u; n < P.n_state; n++) prev_state[n] = new_state[n];
				continue;
			}

			for (uint n = 0u; n < num_out; n++) {
				uint motor = n / P.levels;
				uint level_idx = n % P.levels;
				ulong addr = out_neuron_addr(n, sensors, new_state, output_conns, conn_out_g, thresholds, F);
				float p = odt_train(motor, pid_pwms[s4 + motor], P);
				bool target_true = (uint)(p * (float)P.levels) > level_idx;
				uint gn = g_out_slot_base + n;
				uint slot = find_or_claim_slot(out_markers, out_keys, slot_off[gn], slot_cap[gn], addr);
				if (slot != 0xFFFFFFFFu) { slot_nudge(out_values, slot, target_true); writes += 1u; }
			}
			for (uint n = 0u; n < P.n_state; n++) prev_state[n] = new_state[n];
		}
	}
	out_writes[g] = writes;
}
