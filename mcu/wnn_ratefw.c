/* See wnn_ratefw.h. Keep the Rust operation order (pid_firmware.rs). */
#include "wnn_ratefw.h"

#include <math.h>
#include <string.h>

/* std::f64::consts::PI (math.h M_PI is not ISO C). */
#define WNN_PI 3.14159265358979323846264338327950288

/* pid_firmware.rs clamp(). */
static double clamp_d(double v, double lo, double hi)
{
	if (v < lo) return lo;
	if (v > hi) return hi;
	return v;
}

/* Lpf2p::new(sample_freq, cutoff). */
static void lpf2p_init(wnn_lpf2p_t *f, double sample_freq, double cutoff)
{
	double fr = sample_freq / cutoff;
	double ohm = tan(WNN_PI / fr);
	double cos45 = cos(WNN_PI / 4.0);
	double c = 1.0 + 2.0 * cos45 * ohm + ohm * ohm;
	f->b0 = ohm * ohm / c;
	f->b1 = 2.0 * f->b0;
	f->b2 = f->b0;
	f->a1 = 2.0 * (ohm * ohm - 1.0) / c;
	f->a2 = (1.0 - 2.0 * cos45 * ohm + ohm * ohm) / c;
	f->d1 = f->d2 = 0.0;
}

/* Lpf2p::apply. */
static double lpf2p_apply(wnn_lpf2p_t *f, double sample)
{
	double d0 = sample - f->d1 * f->a1 - f->d2 * f->a2;
	if (!isfinite(d0)) d0 = sample;
	double out = d0 * f->b0 + f->d1 * f->b1 + f->d2 * f->b2;
	f->d2 = f->d1;
	f->d1 = d0;
	return out;
}

/* PidCh::update (wrap = false: the rate loop never wraps). */
static double pidch_update(wnn_pidch_t *p, int has_filt, double dt, double out_limit,
                           double setpoint, double measured)
{
	double error = setpoint - measured;
	p->integ += error * dt;
	if (p->i_limit != 0.0) p->integ = clamp_d(p->integ, -p->i_limit, p->i_limit);
	double deriv = p->first ? 0.0 : -(measured - p->prev_measured) / dt;
	if (has_filt) deriv = lpf2p_apply(&p->filt, deriv);
	p->prev_measured = measured;
	p->first = 0;
	double out = p->kp * error + p->ki * p->integ + p->kd * deriv;
	return (out_limit != 0.0) ? clamp_d(out, -out_limit, out_limit) : out;
}

void wnn_ratefw_init(wnn_ratefw_t *r, const double gains[12], double out_limit_n,
                     double hover_n, double k_thrust, uint32_t main_loop_hz,
                     double attitude_hz, double lpf_hz)
{
	uint32_t att_hz = (uint32_t)attitude_hz;   /* Rust: attitude_hz as u32 */
	memset(r, 0, sizeof *r);
	r->dt = 1.0 / (double)att_hz;
	r->has_filt = lpf_hz > 0.0;
	for (int i = 0; i < 3; i++)
	{
		wnn_pidch_t *p = &r->ax[i];
		p->kp = gains[i * 4]; p->ki = gains[i * 4 + 1];
		p->kd = gains[i * 4 + 2]; p->i_limit = gains[i * 4 + 3];
		if (r->has_filt) lpf2p_init(&p->filt, (double)att_hz, lpf_hz);
	}
	r->out_limit = out_limit_n;
	r->hover_n = hover_n;
	r->k_thrust = k_thrust;
	r->decimation = (main_loop_hz / att_hz) > 1u ? (main_loop_hz / att_hz) : 1u;
	wnn_ratefw_reset(r);
}

void wnn_ratefw_reset(wnn_ratefw_t *r)
{
	for (int i = 0; i < 3; i++)
	{
		wnn_pidch_t *p = &r->ax[i];
		p->integ = 0.0;
		p->prev_measured = 0.0;
		p->first = 1;
		p->filt.d1 = p->filt.d2 = 0.0;
		r->held[i] = 0.0;
	}
	r->tick = 0;
}

double wnn_ratefw_hover_pwm(const wnn_ratefw_t *r)
{
	return sqrt(clamp_d(r->hover_n, 0.0, r->k_thrust) / r->k_thrust);
}

void wnn_ratefw_set_anchor(wnn_ratefw_t *r, float anchor_pwm)
{
	float a = anchor_pwm;
	if (a < 0.0f) a = 0.0f;
	if (a > 1.0f) a = 1.0f;
	r->transport = (double)a - wnn_ratefw_hover_pwm(r);
}

void wnn_ratefw_integ(const wnn_ratefw_t *r, double out[3])
{
	for (int i = 0; i < 3; i++) out[i] = r->ax[i].integ;
}

/* RateStageFw::mix: per-axis force (N) -> '+' mixer -> pwm. */
static void ratefw_mix(const wnn_ratefw_t *r, double pwm[4])
{
	double ro = r->held[0] / 2.0, pi = r->held[1] / 2.0, ya = r->held[2];
	double h = r->hover_n, kt = r->k_thrust;
	double t[4] = {h - pi + ya, h - ro - ya, h + pi + ya, h + ro - ya};
	for (int i = 0; i < 4; i++) pwm[i] = sqrt(clamp_d(t[i], 0.0, kt) / kt);
}

/* RateStageFw::step(rate_sp, gyro, coll_dev). */
static void ratefw_step(wnn_ratefw_t *r, const double sp[3], const double gyro[3],
                        double coll_dev, double pwm[4])
{
	if (r->tick % r->decimation == 0)
		for (int i = 0; i < 3; i++)
			r->held[i] = pidch_update(&r->ax[i], r->has_filt, r->dt, r->out_limit, sp[i], gyro[i]);
	r->tick += 1u;
	ratefw_mix(r, pwm);
	if (coll_dev == 0.0) return;
	for (int i = 0; i < 4; i++) pwm[i] = clamp_d(pwm[i] + coll_dev, 0.0, 1.0);
}

void wnn_ratefw_fly(wnn_ratefw_t *r, const float sp[4], const float gyro[3], float pwm[4])
{
	double h[4] = {sp[0], sp[1], sp[2], sp[3]};
	double g[3] = {gyro[0], gyro[1], gyro[2]};
	double p[4];
	ratefw_step(r, &h[1], g, r->transport + h[0], p);
	for (int i = 0; i < 4; i++) pwm[i] = (float)p[i];
}
