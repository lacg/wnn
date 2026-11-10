/* See cascade_dhat.h. Keep the Rust operation ORDER (cascade_dhat.rs): the parity
 * tests (mcu/parity/, mcu/ctrl51/) assert bit equality with -ffp-contract=off. */
#include "cascade_dhat.h"

static cdhat_real clamp_r(cdhat_real x, cdhat_real lo, cdhat_real hi)
{
	if (x < lo) x = lo;
	if (x > hi) x = hi;
	return x;
}

void cdhat_init(cdhat_t *d, const double c[3], const double clamp[3], double k_thrust,
                double dt, double alpha, double l_gain)
{
	for (int a = 0; a < 3; a++)
	{
		d->c[a] = (cdhat_real)c[a];
		d->clamp[a] = (cdhat_real)clamp[a];
	}
	d->k_thrust = (cdhat_real)k_thrust;
	d->dt = (cdhat_real)dt;
	d->alpha = (cdhat_real)alpha;
	d->l_gain = (cdhat_real)l_gain;
	cdhat_reset(d);
}

void cdhat_reset(cdhat_t *d)
{
	for (int i = 0; i < 4; i++) d->lag_pwm[i] = 0;
	for (int i = 0; i < 3; i++) d->dhat[i] = 0;
	for (int i = 0; i < 3; i++) d->last_gyro[i] = 0;
	d->primed = d->seeded = d->have_gyro = 0;
}

/* CascadeDhat::lagged_force. */
static void lagged_force(const cdhat_t *d, cdhat_real f[3])
{
	cdhat_real t[4];
	for (int m = 0; m < 4; m++) t[m] = d->k_thrust * d->lag_pwm[m] * d->lag_pwm[m];
	f[0] = t[3] - t[1];
	f[1] = t[2] - t[0];
	f[2] = (t[0] - t[1] + t[2] - t[3]) * (cdhat_real)0.25;
}

/* CascadeDhat::track_lag: 0 on the priming call, else 1 with the lagged force. */
static int track_lag(cdhat_t *d, const float applied[4], cdhat_real ua[3])
{
	if (!d->primed) { d->primed = 1; return 0; }
	cdhat_real cmd[4];
	for (int m = 0; m < 4; m++) cmd[m] = clamp_r((cdhat_real)applied[m], 0, 1);
	for (int m = 0; m < 4; m++)
	{
		if (!d->seeded) d->lag_pwm[m] = cmd[m];
		else d->lag_pwm[m] += d->alpha * (cmd[m] - d->lag_pwm[m]);
	}
	d->seeded = 1;
	lagged_force(d, ua);
	return 1;
}

void cdhat_observe(cdhat_t *d, const float gyro[3], const float applied[4])
{
	cdhat_real g[3] = {(cdhat_real)gyro[0], (cdhat_real)gyro[1], (cdhat_real)gyro[2]};
	cdhat_real ua[3];
	int have_ua = track_lag(d, applied, ua);
	for (int a = 0; have_ua && d->have_gyro && a < 3; a++)
	{
		cdhat_real rate_dot = (g[a] - d->last_gyro[a]) / d->dt;
		cdhat_real residual = rate_dot - d->c[a] * ua[a] - d->dhat[a];
		d->dhat[a] = clamp_r(d->dhat[a] + d->l_gain * residual, -d->clamp[a], d->clamp[a]);
	}
	for (int a = 0; a < 3; a++) d->last_gyro[a] = g[a];
	d->have_gyro = 1;
}

int cdhat_measured_force(const cdhat_t *d, cdhat_real f[3])
{
	if (!d->seeded) return 0;
	lagged_force(d, f);
	return 1;
}
