/* Firmware RATE stage — bit-exact C port of ram_controller pid_firmware.rs
 * RateStageFw (+ Lpf2p) and ctbr.rs CtbrInner::fly/set_anchor (ABI 41).
 *
 * ON A REAL VEHICLE THIS IS NOT NEEDED: the Crazyflie firmware's own rate PID
 * (attitude_pid_controller.c, 500 Hz) is the inner loop; the WNN only hands it the
 * CTBR setpoint and reads its integrator (pid.integ) for the CTRL-38 rate-I input.
 * This port exists so the host parity test (and a firmware without a rate loop)
 * can close the cascade exactly as the simulator does. All arithmetic is f64, in
 * the Rust operation order. */
#ifndef WNN_RATEFW_H
#define WNN_RATEFW_H

#include <stdint.h>

typedef struct
{
	double b0, b1, b2, a1, a2, d1, d2;
} wnn_lpf2p_t;

typedef struct
{
	double kp, ki, kd, i_limit;
	double integ, prev_measured;
	int first;
	wnn_lpf2p_t filt;
} wnn_pidch_t;

typedef struct
{
	wnn_pidch_t ax[3];
	int has_filt;
	double dt, out_limit, hover_n, k_thrust, transport;
	uint32_t decimation, tick;
	double held[3];
} wnn_ratefw_t;

/* RateStageFw::from_si_arrays: gains [roll,pitch,yaw] x [kp,ki,kd,i_limit] (SI). */
void wnn_ratefw_init(wnn_ratefw_t *r, const double gains[12], double out_limit_n,
                     double hover_n, double k_thrust, uint32_t main_loop_hz,
                     double attitude_hz, double lpf_hz);
/* RateStageFw::reset (per flight). The anchor transport survives, as in Rust. */
void wnn_ratefw_reset(wnn_ratefw_t *r);
/* CtbrInner::set_anchor: transport = clamp(anchor, 0, 1) - hover_pwm. */
void wnn_ratefw_set_anchor(wnn_ratefw_t *r, float anchor_pwm);
/* The integrators as they stand (pid.integ) — the WNN's rate-I input. */
void wnn_ratefw_integ(const wnn_ratefw_t *r, double out[3]);
/* CtbrInner::fly: one physical tick on the held setpoint [coll dev, p, q, r]. */
void wnn_ratefw_fly(wnn_ratefw_t *r, const float sp[4], const float gyro[3], float pwm[4]);
/* RateStageFw::hover_pwm. */
double wnn_ratefw_hover_pwm(const wnn_ratefw_t *r);

#endif /* WNN_RATEFW_H */
