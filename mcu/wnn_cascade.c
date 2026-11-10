/* On-board cascade WNN controller. Every function names the Rust it mirrors
 * (ram_controller @ ABI 41; the CTRL-48 residual @ ABI 45); keep the operation ORDER when editing — the parity
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

/* --------------------------- features (controller.rs compute_features) -- */

typedef struct
{
	float tilt, roll_err, pitch_err, yaw_err;
} wnn_att_t;

/* Derived attitude errors: the estimator's (or the accel's) "up", atan2 forms. */
static wnn_att_t attitude_errors(wnn_ctrl_t *c, const wnn_tick_in_t *in)
{
	float src[3] = {in->accel[0], in->accel[1], in->accel[2]};
#if WNN_OBS_ATT_EST
	float q[4];
	mahony_update(&c->est, in->gyro, in->accel, q);
	up_from_quat(q, src);
#endif
	float ax = src[0], ay = src[1], az = src[2];
	float roll_est = atan2f(ay, az), pitch_est = atan2f(-ax, sqrtf(ay * ay + az * az));
	c->yaw_heading += (WNN_OBS_YAW_ERR || WNN_OBS_YAW_ERR_I) ? in->gyro[2] * WNN_DT : in->gyro[2];
	wnn_att_t a;
	a.tilt = atan2f(sqrtf(ax * ax + ay * ay), az);
	a.roll_err = in->target[0] - roll_est;
	a.pitch_err = in->target[1] - pitch_est;
	a.yaw_err = in->target[2] - c->yaw_heading;
	return a;
}

/* integral_acc[k] = leak * integral_acc[k] + e; push integral * scale. */
static int push_integral(wnn_ctrl_t *c, int fi, int *k, float e)
{
	c->integ[*k] = WNN_INTEGRAL_LEAK * c->integ[*k] + e;
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
	if (WNN_OBS_TILT_I) fi = push_integral(c, fi, &k, a->tilt);
	for (int i = 0; WNN_OBS_PERAXIS_P && i < n_axes; i++) c->features[fi++] = errs[i];
	for (int i = 0; WNN_OBS_PERAXIS_I && i < n_axes; i++) fi = push_integral(c, fi, &k, errs[i]);
	if (WNN_OBS_YAW_ERR) c->features[fi++] = a->yaw_err;
	if (WNN_OBS_YAW_ERR_I) fi = push_integral(c, fi, &k, a->yaw_err);
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
		c->features[fi++] = v[3 + a] * c->resid_inv_a[a] + c->resid_w[a] * v[a];
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
	(void)cascade_block(c, fi, in);
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
		c->fired[n] = (uint8_t)key_present(c->addr[n], wnn_off[n], wnn_off[n + 1]);
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

/* controller.rs decode_ctbr (absolute, or the leaky delta law). */
static void decode_ctbr(wnn_ctrl_t *c)
{
	if (WNN_DELTA_CONTROL) memcpy(c->setpoint_prev, c->setpoint, sizeof c->setpoint);
	for (int m = 0; m < 4; m++)
	{
		float d = bank_decode(c, m), r = c->range[m], v;
		c->decoded[m] = d;
		if (WNN_DELTA_CONTROL)
		{
			float delta = decoded_to_delta(d, WNN_DELTA_MAX * r, WNN_NEUTRAL, c->gamma[m]);
			float leaked = 0.0f + WNN_DELTA_LEAK * (c->setpoint_prev[m] - 0.0f);
			v = clampf_rs(leaked + delta, -r, r);
		}
		else
			v = decoded_to_delta(d, r, WNN_NEUTRAL, c->gamma[m]);
		c->setpoint[m] = v;
	}
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
	memset(c->integ, 0, sizeof c->integ);
	memset(c->setpoint, 0, sizeof c->setpoint);
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
	memcpy(sp, c->setpoint, sizeof c->setpoint);
	return decide;
}
