/* On-board CASCADE (output_mode=ctbr) WNN controller — the C twin of the Rust
 * WnnController::step for an exported winner (scripts/export_controller_c.py).
 *
 * WHAT RUNS HERE. The WNN is the OUTER loop: every physical tick it computes the
 * observation features, and on every action_repeat-th tick it thermometer-encodes
 * them, reads the output layer (sorted TRUE on-sets, binary search), and decodes the
 * four banks into a CTBR setpoint [collective deviation (pwm), p, q, r (rad/s)]. The
 * setpoint is HELD in between. The INNER loop is the firmware's own rate PID: it is
 * NOT part of this file (wnn_ratefw.c is a bit-exact port of the simulator's rate
 * stage for host testing and for vehicles whose firmware lacks one).
 *
 * WHAT THE FIRMWARE MUST HAND IN, every physical tick (wnn_tick_in_t):
 *   gyro, accel          the IMU read of THIS tick (rad/s, m/s^2 specific force)
 *   target               target attitude RPY (rad)
 *   vert                 [collective_cmd, alt_err = target - z, vz]   (stage 1)
 *   horiz                [err_x, err_y, vx, vy] (err = target - position, world) (stage 2)
 *   rate_integ           the rate PID's integrator I per axis AS IT STANDS BEFORE this
 *                        tick's rate update (pid.integ, SI units: rad) — CTRL-38 rate-I
 *   applied_prev         the motor command the plant ACTUALLY received last tick
 *                        (post rate loop, post any trim) — CTRL-38 d-hat input
 * Inputs a model does not use are ignored (the header's WNN_OBS_* flags say which).
 *
 * BIT PARITY. Every operation mirrors the Rust source in order and precision
 * (f32 features, f64 Mahony estimator / rate-I / d-hat observer). Build with
 * -ffp-contract=off: Rust never fuses a*b+c, so a fused C build is NOT the same
 * function. Transcendentals (atan2f, powf, sqrt) come from the platform libm, so
 * bit parity is a HOST claim (mcu/parity/); an MCU libm may differ in the last ulp.
 */
#ifndef WNN_CASCADE_H
#define WNN_CASCADE_H

#include <stdint.h>

#ifndef WNN_MODEL_HEADER
#define WNN_MODEL_HEADER "wnn_model.h"
#endif
#define WNN_MODEL_MACROS_ONLY
#include WNN_MODEL_HEADER
#undef WNN_MODEL_MACROS_ONLY

#ifndef WNN_EXPORT_CASCADE
#error "wnn_cascade: the model header is not a full cascade export (re-export without --tables-only)"
#endif

/* d-hat observer precision. Default f64 = the CPU reference (cascade_dhat.rs), bit
 * exact. -DWNN_DHAT_F32 = the Metal kernel's f32 twin (cheaper on an M4F, which has
 * no double-precision FPU; NOT bit-identical to the CPU reference). */
#ifdef WNN_DHAT_F32
typedef float wnn_obs_real;
#else
typedef double wnn_obs_real;
#endif

typedef struct
{
	float gyro[3];
	float accel[3];
	float target[3];
	float vert[3];
	float horiz[4];
	double rate_integ[3];
	float applied_prev[4];
} wnn_tick_in_t;

typedef struct
{
	double dt, kp, ki;
	double q[4];
	double integral[3];
} wnn_mahony_t;

typedef struct
{
	wnn_obs_real c[3], clamp[3];
	wnn_obs_real k_thrust, dt, alpha, l_gain;
	wnn_obs_real lag_pwm[4], dhat[3], last_gyro[3];
	int primed, seeded, have_gyro;
} wnn_dhat_t;

#define WNN_MAX_INTEGRALS 5

typedef struct
{
	wnn_mahony_t est;
	wnn_dhat_t dhat;
	float integ[WNN_MAX_INTEGRALS];
	float yaw_heading;
	float range[4], gamma[4];
	float setpoint[4];       /* held CTBR setpoint = Rust's self.pwm (bank deviations) */
	float setpoint_prev[4];  /* Rust's self.pwm_prev (delta law only) */
	uint32_t step_counter;
	/* Telemetry of the last tick (the parity test reads these). */
	float features[WNN_FEATURES];
	uint8_t frame[WNN_INPUT_BITS];
	uint32_t addr[WNN_NEURONS];
	uint8_t fired[WNN_NEURONS];
	float decoded[4];
} wnn_ctrl_t;

/* Once at boot. */
void wnn_init(wnn_ctrl_t *c);
/* Every flight (episode) start: init_yaw seeds the yaw heading (used when the model
 * reads yaw_err[_i]); q0 = (w,x,y,z) warm-starts the attitude estimator. */
void wnn_reset(wnn_ctrl_t *c, float init_yaw, const float q0[4]);
/* One physical tick. Writes the HELD setpoint [coll dev, p, q, r] to sp and returns
 * 1 on a decision tick (the setpoint may have changed), 0 on a hold tick. */
int wnn_tick(wnn_ctrl_t *c, const wnn_tick_in_t *in, float sp[4]);

#endif /* WNN_CASCADE_H */
