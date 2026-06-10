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

constant uint  CELL_EMPTY      = 2u;            // QSR EMPTY
constant float QSR_W[4]        = {0.0f, 0.25f, 0.75f, 1.0f};
constant uint  NUM_FEATURES    = 9u;

// Compile-time maxima for thread-private arrays (host asserts runtime <= these).
#define MAX_STATE_NEURONS 32
#define MAX_WINDOW        8
#define MAX_FRAME_BITS    160   // NUM_FEATURES * bits_per_feature (e.g. 9*16)

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
};

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
	if (count == 0u) return CELL_EMPTY;
	uint lo = 0u, hi = count;
	while (lo < hi) {
		uint mid = lo + (hi - lo) / 2u;
		ulong k = keys[start + mid];
		if (k == addr)      return (uint)vals[start + mid];
		else if (k < addr)  lo = mid + 1u;
		else                hi = mid;
	}
	return CELL_EMPTY;
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

	// K-window ring of raw sensor frames (oldest-first in [0,filled)).
	float ring[MAX_WINDOW * NUM_FEATURES];
	uint filled = 0u;

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
		float sensors[NUM_FEATURES];
		sensors[0] = omega.x; sensors[1] = omega.y; sensors[2] = omega.z;
		sensors[3] = -grav_body.x; sensors[4] = -grav_body.y; sensors[5] = -grav_body.z;
		sensors[6] = P.target0; sensors[7] = P.target1; sensors[8] = P.target2;

		// push current frame into the ring (drop oldest if full)
		if (filled == P.window) {
			for (uint s = 1u; s < P.window; s++)
				for (uint f = 0u; f < NUM_FEATURES; f++)
					ring[(s-1u)*NUM_FEATURES + f] = ring[s*NUM_FEATURES + f];
			filled = P.window - 1u;
		}
		for (uint f = 0u; f < NUM_FEATURES; f++) ring[filled*NUM_FEATURES + f] = sensors[f];
		filled += 1u;
		uint pad = P.window - filled;   // leading zero-padded window slots

		// ---- state layer forward (address per neuron via on-the-fly bits) ----
		uchar new_state[MAX_STATE_NEURONS];
		for (uint n = 0u; n < P.n_state; n++) {
			ulong addr = 0ul;
			uint cbase = (uint)(conn_state_g) + n * P.sbpn;
			for (uint i = 0u; i < P.sbpn; i++) {
				int c = state_conns[cbase + i];
				bool bit = false;
				uint cu = (uint)c;
				if (cu < P.sensor_total) {
					uint slot = cu / P.frame_bits;     // 0=oldest window slot
					uint within = cu % P.frame_bits;
					uint feat = within / P.bpf;
					uint b = within % P.bpf;
					if (slot >= pad) {                 // else padding → bit stays false
						uint hist = slot - pad;
						bit = ring[hist*NUM_FEATURES + feat] >= thresholds[feat*P.bpf + b];
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

		// ---- output layer (Mealy) + Strategy-5 decode, per motor -------------
		float pwm[4];
		float mono_step = 0.0f;
		for (uint m = 0u; m < P.num_motors; m++) {
			float sum = 0.0f;
			bool seen_one = false, prev_zero = false;
			for (uint l = 0u; l < P.levels; l++) {
				uint n = m * P.levels + l;
				ulong addr = 0ul;
				uint cbase = (uint)(conn_out_g) + n * P.obpn;
				for (uint i = 0u; i < P.obpn; i++) {
					int c = output_conns[cbase + i];
					bool bit = false;
					uint cu = (uint)c;
					if (cu < P.frame_bits) {            // current frame = latest sensors
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
				uint gn = g_out_base + n;
				uint cell = bsearch_cell(out_keys, out_vals, out_off[gn], out_cnt[gn], addr);
				uint qv = cell & 3u;
				// Thermometer "on" = QSR MSB set (TRUE/WEAK_TRUE); a 1→0→1 gap is a
				// monotonicity violation (mirrors controller::monotonicity_violations).
				if (qv >= 2u) { if (prev_zero && seen_one) mono_step += 1.0f; seen_one = true; prev_zero = false; }
				else { prev_zero = true; }
				sum += QSR_W[qv];
			}
			pwm[m] = clamp(sum / (float)P.levels, 0.0f, 1.0f);   // absolute PWM
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
