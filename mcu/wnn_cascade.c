/* On-board cascade WNN controller. Every function names the Rust it mirrors
 * (ram_controller @ ABI 41; the CTRL-48 residual @ ABI 45; the block-10 / lever
 * set of the Y+F+L base @ ABI 60); keep the operation ORDER when editing — the parity
 * test (mcu/parity/) asserts bit equality, and a reassociated sum is a different
 * function. See wnn_cascade.h for the firmware contract. */
#include "wnn_cascade.h"
#include WNN_MODEL_HEADER

#include <math.h>
#include <string.h>

static const float wnn_ctbr_range[4] = WNN_CTBR_RANGE_INIT;
static const float wnn_ctbr_gamma[4] = WNN_CTBR_GAMMA_INIT;
static const double wnn_rate_gains[12] = WNN_RATE_GAINS_INIT;
#if WNN_OBS_OBSERVER
static const double wnn_dhat_plant[5] = WNN_DHAT_PLANT_INIT;
#endif
#if WNN_OBS_TILT_REF_ERR
static const float wnn_tilt_ref_shape[4] = WNN_TILT_REF_SHAPE_INIT;
#endif

/* att_bias_pred.rs ABP_GRAVITY: the nominal g the predictor scales the velocity
 * difference by (a Rust source constant, not a run parameter). */
#ifndef WNN_ABP_GRAVITY
#define WNN_ABP_GRAVITY 9.81f
#endif

/* ------------------------------------------------------------ helpers ---- */

/* f32::clamp / f64::clamp (NaN passes through, as in Rust). */
static float clampf_rs(float x, float lo, float hi)
{
	if (x < lo) x = lo;
	if (x > hi) x = hi;
	return x;
}

/* ---------------------------------------- Mahony estimator (estimator.rs) */

/* MahonyFilter::reset(Some(q0)). */
static void mahony_reset(wnn_mahony_t *e, const float q0[4])
{
	double q[4] = {q0[0], q0[1], q0[2], q0[3]};
	double n = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
	n = (n > 0.0) ? n : 1.0;
	for (int i = 0; i < 4; i++) e->q[i] = q[i] / n;
	for (int i = 0; i < 3; i++) e->integral[i] = 0.0;
}

/* The accelerometer correction term of MahonyFilter::update (adds into g). */
static void mahony_correct(wnn_mahony_t *e, const float accel[3], double g[3])
{
	double w = e->q[0], x = e->q[1], y = e->q[2], z = e->q[3];
	double ax = accel[0], ay = accel[1], az = accel[2];
	double norm = sqrt(ax * ax + ay * ay + az * az);
	if (!(norm > 1e-9)) return;
	ax /= norm; ay /= norm; az /= norm;
	double vx = 2.0 * (x * z - w * y), vy = 2.0 * (w * x + y * z);
	double vz = w * w - x * x - y * y + z * z;
	double err[3] = {ay * vz - az * vy, az * vx - ax * vz, ax * vy - ay * vx};
	for (int i = 0; i < 3; i++)
	{
		if (e->ki > 0.0) e->integral[i] += e->ki * err[i] * e->dt;
	}
	for (int i = 0; i < 3; i++) g[i] += e->kp * err[i] + e->integral[i];
}

/* MahonyFilter::update -> f32 quaternion. */
static void mahony_update(wnn_mahony_t *e, const float gyro[3], const float accel[3], float qo[4])
{
	double g[3] = {gyro[0], gyro[1], gyro[2]};
	mahony_correct(e, accel, g);
	double w = e->q[0], x = e->q[1], y = e->q[2], z = e->q[3], h = 0.5 * e->dt;
	double dw = (-x * g[0] - y * g[1] - z * g[2]) * h;
	double dx = (w * g[0] + y * g[2] - z * g[1]) * h;
	double dy = (w * g[1] - x * g[2] + z * g[0]) * h;
	double dz = (w * g[2] + x * g[1] - y * g[0]) * h;
	w += dw; x += dx; y += dy; z += dz;
	double n = sqrt(w * w + x * x + y * y + z * z);
	n = (n > 0.0) ? n : 1.0;
	e->q[0] = w / n; e->q[1] = x / n; e->q[2] = y / n; e->q[3] = z / n;
	for (int i = 0; i < 4; i++) qo[i] = (float)e->q[i];
}

/* att_estimator.rs up_from_quat (f32). */
static void up_from_quat(const float q[4], float up[3])
{
	float w = q[0], x = q[1], y = q[2], z = q[3];
	up[0] = 2.0f * (x * z - w * y);
	up[1] = 2.0f * (w * x + y * z);
	up[2] = w * w - x * x - y * y + z * z;
}

/* ------------------------------------- d-hat observer (cascade_dhat.rs) -- */

/* CascadeDhat::new(torque_gains(..), k_thrust, dt, 0, out_limit) then
 * set_motor_lag(WNN_MOTOR_LAG_S, dt) — the host's set_observer_motor_lag. The law
 * itself is cascade_dhat.c, shared with the MPCOF port (mcu/mpcof_cascade.c). */
#if WNN_OBS_OBSERVER
/* cascade_dhat.rs torque_gains(arm, k_drag, inertia): f32 plant, f64 gains. */
static void plant_torque_gains(double c[3])
{
	double arm = (float)wnn_dhat_plant[0], kd = (float)wnn_dhat_plant[1];
	double in[3] = {(float)wnn_dhat_plant[2], (float)wnn_dhat_plant[3], (float)wnn_dhat_plant[4]};
	c[0] = arm / in[0];
	c[1] = arm / in[1];
	c[2] = 4.0 * kd / in[2];
}
#endif

static void dhat_init(cdhat_t *d)
{
#if WNN_OBS_OBSERVER
	double c[3];
	plant_torque_gains(c);
	double clamp[3] = {c[0] * WNN_RATE_OUT_LIMIT_N, c[1] * WNN_RATE_OUT_LIMIT_N, c[2] * WNN_RATE_OUT_LIMIT_N};
	float dt = WNN_DT, t = WNN_MOTOR_LAG_S;
	double alpha = (double)(4.0f * dt / fmaxf(t, 4.0f * dt));
	cdhat_init(d, c, clamp, WNN_K_THRUST, (double)dt, alpha, WNN_DHAT_L_GAIN);
#else
	cdhat_reset(d);
#endif
}

/* CTRL-48 a3 scale (cascade_obs.rs CasResidual::new from ctbr.rs residual_scale):
 * per axis inv_a = 1/(c*ki*L), w = 1 (L = i_limit, 1 when 0); ki = 0 => 1/clamp, 0. */
static void resid_init(wnn_ctrl_t *ctl)
{
#if WNN_OBS_DRES
	double c[3];
	plant_torque_gains(c);
	for (int a = 0; a < 3; a++)
	{
		double ki = wnn_rate_gains[a * 4 + 1], lim = wnn_rate_gains[a * 4 + 3];
		double l = (lim > 0.0) ? lim : 1.0;
		int has_i = (ki != 0.0);
		ctl->resid_inv_a[a] = (float)(has_i ? 1.0 / (c[a] * ki * l) : 1.0 / (c[a] * WNN_RATE_OUT_LIMIT_N));
		ctl->resid_w[a] = has_i ? 1.0f : 0.0f;
	}
#else
	(void)ctl;
#endif
}

/* ---------------------- Mahony heading (controller.rs mahony_heading) -- */
#if WNN_OBS_YAW_FROM_ATT_EST

/* wrap_angle_f64: into (-pi, pi]. */
static double wrap_angle(double a)
{
	const double pi = 3.14159265358979311600;
	double x = a;
	while (x > pi) x -= 2.0 * pi;
	while (x <= -pi) x += 2.0 * pi;
	return x;
}

/* ABI 55: the yaw of quat_to_euler_f64(q), unwrapped onto the running heading h. */
static float mahony_heading(float h, const float q[4])
{
	double w = q[0], x = q[1], y = q[2], z = q[3];
	double siny_cosp = 2.0 * (w * z + x * y);
	double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
	double yaw = atan2(siny_cosp, cosy_cosp);
	double hd = h;
	return (float)(hd + wrap_angle(yaw - hd));
}
#endif

/* ------------------ Mahony-bias predictor (att_bias_pred.rs, CTRL-46) ---- */
#if WNN_OBS_ATT_BIAS_PRED

/* rot_t_xy: body x, y of R^T (wx, wy, 0), q = (w, x, y, z), f32. */
static void rot_t_xy(const float q[4], float wx, float wy, float out[2])
{
	float w = q[0], x = q[1], y = q[2], z = q[3];
	float r00 = 1.0f - 2.0f * (y * y + z * z);
	float r01 = 2.0f * (x * y - w * z);
	float r10 = 2.0f * (x * y + w * z);
	float r11 = 1.0f - 2.0f * (x * x + z * z);
	out[0] = r00 * wx + r10 * wy;
	out[1] = r01 * wx + r11 * wy;
}

/* AttBiasPredictor::step(v_world, q_est) -> [abp_r, abp_p] (f32; kp, ki, dt are the
 * estimator's own: enable_att_bias_pred). */
static void abp_step(wnn_abp_t *p, float vx, float vy, const float q[4], float out[2])
{
	float ux = 0.0f, uy = 0.0f, d[2];
	if (p->have_prev)
	{
		ux = (vx - p->v_prev[0]) / WNN_ABP_GRAVITY;
		uy = (vy - p->v_prev[1]) / WNN_ABP_GRAVITY;
	}
	p->v_prev[0] = vx;
	p->v_prev[1] = vy;
	p->have_prev = 1;
	rot_t_xy(q, ux, uy, d);
	for (int a = 0; a < 2; a++)
	{
		float innov = d[a] - WNN_DT * p->eps[a];
		p->eps[a] += WNN_EST_KP * innov + WNN_DT * p->j[a];
		p->j[a] += WNN_EST_KI * innov;
	}
	out[0] = p->eps[1];
	out[1] = -p->eps[0];
}
#endif

/* -------------------- tilt-reference error (tilt_ref_err.rs, block 10 L1) */

#if WNN_OBS_TILT_REF_ERR
/* TiltRefErr::effective_target: (target + tilt_ref(horiz)) + abp, roll and pitch. */
static void tilt_ref_target(const float target[3], const float h[4], const float abp[2], float out[2])
{
	const float *s = wnn_tilt_ref_shape; /* b_xy, omega_n, zeta, max_tilt */
	float w2 = s[1] * s[1];
	float c2 = 2.0f * s[2] * s[1];
	float pitch = clampf_rs((w2 * h[0] - c2 * h[2]) / s[0], -s[3], s[3]);
	float roll = clampf_rs(-(w2 * h[1] - c2 * h[3]) / s[0], -s[3], s[3]);
	out[0] = (target[0] + roll) + abp[0];
	out[1] = (target[1] + pitch) + abp[1];
}
#endif

/* --------------------------- features (controller.rs compute_features) -- */

typedef struct
{
	float tilt, roll_err, pitch_err, yaw_err;
} wnn_att_t;

/* Yaw heading: anchored = the Mahony yaw (ABI 55) or dt-integrated gyro-z; else the
 * legacy raw gyro-z sum. q = this tick's estimate (read only under ABI 55). */
static void advance_heading(wnn_ctrl_t *c, const wnn_tick_in_t *in, const float q[4])
{
	(void)q;
	if (WNN_OBS_YAW_ERR || WNN_OBS_YAW_ERR_I)
	{
#if WNN_OBS_YAW_FROM_ATT_EST
		c->yaw_heading = mahony_heading(c->yaw_heading, q);
#else
		c->yaw_heading += in->gyro[2] * WNN_DT;
#endif
	}
	else
		c->yaw_heading += in->gyro[2];
}

/* The roll / pitch targets the errors are measured against: the target, or (L1) the
 * tilt-reference target. The bias predictor ticks here once per physical tick; its
 * value does not depend on WHERE in compute_features Rust ticks it (same inputs). */
static void attitude_targets(wnn_ctrl_t *c, const wnn_tick_in_t *in, const float q[4], float tgt[2])
{
	tgt[0] = in->target[0];
	tgt[1] = in->target[1];
#if WNN_OBS_ATT_BIAS_PRED
	abp_step(&c->abp, in->horiz[2], in->horiz[3], q, c->abp_last);
#if WNN_OBS_TILT_REF_ERR
	tilt_ref_target(in->target, in->horiz, c->abp_last, tgt);
#endif
#else
	(void)c; (void)q;
#endif
}

/* Derived attitude errors: the estimator's (or the accel's) "up", atan2 forms. */
static wnn_att_t attitude_errors(wnn_ctrl_t *c, const wnn_tick_in_t *in)
{
	float src[3] = {in->accel[0], in->accel[1], in->accel[2]};
	float q[4] = {1.0f, 0.0f, 0.0f, 0.0f};
#if WNN_OBS_ATT_EST
	mahony_update(&c->est, in->gyro, in->accel, q);
	up_from_quat(q, src);
#endif
	float ax = src[0], ay = src[1], az = src[2];
	float roll_est = atan2f(ay, az), pitch_est = atan2f(-ax, sqrtf(ay * ay + az * az));
	advance_heading(c, in, q);
	float tgt[2];
	attitude_targets(c, in, q, tgt);
	wnn_att_t a;
	a.tilt = atan2f(sqrtf(ax * ax + ay * ay), az);
	a.roll_err = tgt[0] - roll_est;
	a.pitch_err = tgt[1] - pitch_est;
	a.yaw_err = in->target[2] - c->yaw_heading;
	return a;
}

/* integral_acc[k] = leak * integral_acc[k] + e; push integral * scale. The yaw
 * integrals take WNN_YAW_INTEGRAL_LEAK (ARM 3; == WNN_INTEGRAL_LEAK unless set). */
static int push_integral(wnn_ctrl_t *c, int fi, int *k, float e, float leak)
{
	c->integ[*k] = leak * c->integ[*k] + e;
	c->features[fi] = c->integ[*k] * WNN_INTEGRAL_SCALE;
	(*k)++;
	return fi + 1;
}

/* tilt / per-axis / yaw-anchor blocks, canonical order (obs_pwm and dhat_b are
 * refused under CTBR, so they never appear here). */
static int attitude_block(wnn_ctrl_t *c, int fi, const wnn_att_t *a)
{
	int k = 0;
	float errs[3] = {a->roll_err, a->pitch_err, a->yaw_err};
	int n_axes = WNN_OBS_PERAXIS_YAW ? 3 : 2;
	if (WNN_OBS_TILT_P) c->features[fi++] = a->tilt;
	const float leaks[3] = {WNN_INTEGRAL_LEAK, WNN_INTEGRAL_LEAK, WNN_YAW_INTEGRAL_LEAK};
	if (WNN_OBS_TILT_I) fi = push_integral(c, fi, &k, a->tilt, WNN_INTEGRAL_LEAK);
	for (int i = 0; WNN_OBS_PERAXIS_P && i < n_axes; i++) c->features[fi++] = errs[i];
	for (int i = 0; WNN_OBS_PERAXIS_I && i < n_axes; i++) fi = push_integral(c, fi, &k, errs[i], leaks[i]);
	if (WNN_OBS_YAW_ERR) c->features[fi++] = a->yaw_err;
	if (WNN_OBS_YAW_ERR_I) fi = push_integral(c, fi, &k, a->yaw_err, WNN_YAW_INTEGRAL_LEAK);
	return fi;
}

/* Stage-1 vertical then stage-2 horizontal pass-throughs. */
static int translation_block(wnn_ctrl_t *c, int fi, const wnn_tick_in_t *in)
{
	if (WNN_OBS_COLLECTIVE_CMD) c->features[fi++] = in->vert[0];
	if (WNN_OBS_ALT_ERR) c->features[fi++] = in->vert[1];
	if (WNN_OBS_VZ) c->features[fi++] = in->vert[2];
	if (WNN_OBS_POS_ERR_XY) { c->features[fi++] = in->horiz[0]; c->features[fi++] = in->horiz[1]; }
	if (WNN_OBS_VEL_XY) { c->features[fi++] = in->horiz[2]; c->features[fi++] = in->horiz[3]; }
	return fi;
}

/* cascade_obs.rs CascadeObs::live: v = [rate I / i_limit x3, d-hat x3] (f32). The
 * observer ticks every tick whenever a2 or a3 is on. */
static void cascade_values(wnn_ctrl_t *c, const wnn_tick_in_t *in, float v[6])
{
	for (int i = 0; i < 6; i++) v[i] = 0.0f;
	for (int i = 0; (WNN_OBS_RATE_I || WNN_OBS_DRES) && i < 3; i++)
	{
		double lim = wnn_rate_gains[i * 4 + 3], integ = in->rate_integ[i];
		v[i] = (float)((lim > 0.0) ? integ / lim : integ);
	}
#if WNN_OBS_OBSERVER
	cdhat_observe(&c->dhat, in->gyro, in->applied_prev);
	for (int a = 0; a < 3; a++) v[3 + a] = (float)c->dhat.dhat[a];
#else
	(void)c; (void)in;
#endif
}

/* CTRL-38/48 (cascade_obs.rs append): rate-I, then d-hat, then the residual
 * r = d-hat * inv_a + w * (I/L) (CasResidual::of, f32). */
static int cascade_block(wnn_ctrl_t *c, int fi, const wnn_tick_in_t *in)
{
	float v[6];
	cascade_values(c, in, v);
	for (int i = 0; WNN_OBS_RATE_I && i < 3; i++) c->features[fi++] = v[i];
	for (int a = 0; WNN_OBS_DHAT && a < 3; a++) c->features[fi++] = v[3 + a];
	for (int a = 0; WNN_OBS_DRES && a < 3; a++)
	{
		float r = v[3 + a] * c->resid_inv_a[a] + c->resid_w[a] * v[a];
		c->features[fi++] = r;
		if (a == 2) c->cresid_z = r;
	}
	return fi;
}

/* CTRL-46 (canonical order LAST, after the cascade blocks): [abp_r, abp_p]. */
static int att_bias_block(wnn_ctrl_t *c, int fi)
{
	for (int i = 0; WNN_OBS_ATT_BIAS_PRED && i < 2; i++) c->features[fi++] = c->abp_last[i];
	return fi;
}

/* compute_features: 9 raw sensors, then the enabled blocks. */
static void compute_features(wnn_ctrl_t *c, const wnn_tick_in_t *in)
{
	int fi = 0;
	for (int i = 0; i < 3; i++) c->features[fi++] = in->gyro[i];
	for (int i = 0; i < 3; i++) c->features[fi++] = in->accel[i];
	for (int i = 0; i < 3; i++) c->features[fi++] = in->target[i];
	if (WNN_FEATURES == 9) return;
	wnn_att_t a = attitude_errors(c, in);
	fi = attitude_block(c, fi, &a);
	fi = translation_block(c, fi, in);
	fi = cascade_block(c, fi, in);
	(void)att_bias_block(c, fi);
}

/* ------------------------------------------ encode + output layer -------- */

/* Thermometer: bit = feature >= threshold. */
static void encode_frame(wnn_ctrl_t *c)
{
	for (int f = 0; f < WNN_FEATURES; f++)
		for (int b = 0; b < WNN_BITS_PER_FEATURE; b++)
		{
			int i = f * WNN_BITS_PER_FEATURE + b;
			c->frame[i] = (uint8_t)(c->features[f] >= wnn_thresholds[i]);
		}
}

#ifdef WNN_KEY_FORMAT_PEF64
/* pef64 header: the same membership test over the partitioned Elias-Fano store. */
#include "wnn_pef.h"
static const wnn_pef_t wnn_pef_model = WNN_PEF_MODEL;
#define NEURON_HAS(n, addr) wnn_pef_contains(&wnn_pef_model, (uint32_t)(n), (addr))
#else
/* Membership in neuron n's sorted TRUE on-set. */
static int key_present(uint32_t addr, uint32_t lo, uint32_t end)
{
	uint32_t hi = end;
	while (lo < hi)
	{
		uint32_t m = (lo + hi) >> 1;
		if (wnn_keys[m] < addr) lo = m + 1; else hi = m;
	}
	return (lo < end) && (wnn_keys[lo] == addr);
}
#define NEURON_HAS(n, addr) key_present((addr), wnn_off[(n)], wnn_off[(n) + 1])
#endif

/* neuron_memory.rs address_over: MSB first, a negative connection reads 0. */
static uint32_t neuron_address(const uint8_t *frame, int n)
{
	uint32_t addr = 0;
	for (int b = 0; b < WNN_BITS; b++)
	{
		int conn = (int)wnn_conn[n * WNN_BITS + b];
		uint32_t bit = (conn >= 0) ? frame[conn] : 0u;
		addr |= bit << (WNN_BITS - 1 - b);
	}
	return addr;
}

static void forward_output(wnn_ctrl_t *c)
{
	for (int n = 0; n < WNN_NEURONS; n++)
	{
		c->addr[n] = neuron_address(c->frame, n);
		c->fired[n] = (uint8_t)NEURON_HAS(n, c->addr[n]);
	}
}

/* ------------------------------------------------------- decode -------- */

/* cell_mode.rs decode_motor_cells, BINARY antagonist: 0.5 + (sum E - sum I)/L. */
static float bank_decode(const wnn_ctrl_t *c, int m)
{
	const uint8_t *f = &c->fired[m * WNN_LEVELS];
	float sum_e = 0.0f, sum_i = 0.0f;
	for (int l = 0; l < WNN_LEVELS / 2; l++) sum_e += f[l] ? 1.0f : 0.0f;
	for (int l = WNN_LEVELS / 2; l < WNN_LEVELS; l++) sum_i += f[l] ? 1.0f : 0.0f;
	return clampf_rs(0.5f + (sum_e - sum_i) / (float)WNN_LEVELS, 0.0f, 1.0f);
}

/* controller.rs shape_gamma + decoded_to_delta. */
static float decoded_to_delta(float decoded, float delta_max, float n, float gamma)
{
	float t = (decoded >= n) ? (decoded - n) / (1.0f - n) : (decoded - n) / n;
	if (gamma != 1.0f) t = copysignf(powf(fabsf(t), gamma), t);
	return t * delta_max;
}

/* controller.rs delta_max_bank: the yaw bank (3) may carry its own (ABI 57). */
static float delta_max_bank(int m)
{
	return (m == 3) ? WNN_DELTA_MAX_YAW : WNN_DELTA_MAX;
}

/* controller.rs decode_ctbr (absolute, or the leaky delta law), then (ABI 58) the
 * yaw-rate feed-forward on the HELD setpoint only. */
static void decode_ctbr(wnn_ctrl_t *c)
{
	if (WNN_DELTA_CONTROL) memcpy(c->setpoint_prev, c->setpoint, sizeof c->setpoint);
	for (int m = 0; m < 4; m++)
	{
		float d = bank_decode(c, m), r = c->range[m], v;
		c->decoded[m] = d;
		if (WNN_DELTA_CONTROL)
		{
			float delta = decoded_to_delta(d, delta_max_bank(m) * r, WNN_NEUTRAL, c->gamma[m]);
			float leaked = 0.0f + WNN_DELTA_LEAK * (c->setpoint_prev[m] - 0.0f);
			v = clampf_rs(leaked + delta, -r, r);
		}
		else
			v = decoded_to_delta(d, r, WNN_NEUTRAL, c->gamma[m]);
		c->setpoint[m] = v;
		c->held[m] = v;
	}
#if WNN_YAW_TRIM_FF
	float r3 = c->range[3];
	c->held[3] = clampf_rs(c->held[3] + WNN_YAW_TRIM_FF_CT * c->cresid_z, -r3, r3);
#endif
}

/* --------------------------------------------------------- public ------ */

void wnn_init(wnn_ctrl_t *c)
{
	static const float id[4] = {1.0f, 0.0f, 0.0f, 0.0f};
	memset(c, 0, sizeof *c);
	c->est.dt = (double)WNN_DT;
	c->est.kp = (double)WNN_EST_KP;
	c->est.ki = (double)WNN_EST_KI;
	dhat_init(&c->dhat);
	resid_init(c);
	/* Runtime copies: a const gamma would let the compiler fold powf(x, 2) -> x*x. */
	memcpy(c->range, wnn_ctbr_range, sizeof c->range);
	memcpy(c->gamma, wnn_ctbr_gamma, sizeof c->gamma);
	wnn_reset(c, 0.0f, id);
}

/* WnnController::reset(init_yaw) + seed_att_estimator(q0). */
void wnn_reset(wnn_ctrl_t *c, float init_yaw, const float q0[4])
{
	mahony_reset(&c->est, q0);
	cdhat_reset(&c->dhat);
	memset(&c->abp, 0, sizeof c->abp);
	memset(c->abp_last, 0, sizeof c->abp_last);
	c->cresid_z = 0.0f;
	memset(c->integ, 0, sizeof c->integ);
	memset(c->setpoint, 0, sizeof c->setpoint);
	memset(c->held, 0, sizeof c->held);
	memset(c->setpoint_prev, 0, sizeof c->setpoint_prev);
	c->yaw_heading = (WNN_OBS_YAW_ERR || WNN_OBS_YAW_ERR_I) ? init_yaw : 0.0f;
	c->step_counter = 0;
}

/* WnnController::step, minus the inner rate loop (the firmware's). */
int wnn_tick(wnn_ctrl_t *c, const wnn_tick_in_t *in, float sp[4])
{
	compute_features(c, in);
	int decide = 1;
	if (WNN_ACTION_REPEAT > 1)
	{
		decide = (c->step_counter % WNN_ACTION_REPEAT) == 0;
		c->step_counter++;
	}
	if (decide)
	{
		encode_frame(c);
		forward_output(c);
		decode_ctbr(c);
	}
	memcpy(sp, c->held, sizeof c->held);
	return decide;
}
