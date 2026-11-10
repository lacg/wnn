/* The cascade's offset-free OBSERVER — C twin of ram_controller cascade_dhat.rs
 * (CascadeDhat, ABI 41+). ONE definition for both on-board controllers, as in Rust:
 *   - the WNN cascade's CTRL-38 d-hat / CTRL-48 residual inputs (wnn_cascade.c);
 *   - the MPCOF teacher's measured actuator + d-hat (mpcof_cascade.c).
 *
 * THE LAW (cascade_dhat.rs): ua = the rotors' differential thrust from the MEASURED
 * actuator (the pwm applied last tick, clamped to [0,1], through the plant's
 * first-order lag with the prime/seed protocol, t = kt*pwm^2, '+' mixer axis forces);
 * d-hat <- clamp(d-hat + l*(w' - c*ua - d-hat), +-c*out_limit), w' the finite
 * difference of consecutive gyro reads.
 *
 * Precision: double (bit-exact with the Rust CPU reference) unless WNN_DHAT_F32 or
 * MPCOF_F32 is defined (the f32 twin: cheaper on an M4F, NOT bit-identical). */
#ifndef CASCADE_DHAT_H
#define CASCADE_DHAT_H

#if defined(WNN_DHAT_F32) || defined(MPCOF_F32)
typedef float cdhat_real;
#else
typedef double cdhat_real;
#endif

typedef struct
{
	cdhat_real c[3], clamp[3];
	cdhat_real k_thrust, dt, alpha, l_gain;
	cdhat_real lag_pwm[4], dhat[3], last_gyro[3];
	int primed, seeded, have_gyro;
} cdhat_t;

/* CascadeDhat::new + set_motor_lag: c = torque gains (rad/s^2 per N), clamp =
 * c*out_limit, alpha = actuator_lag::motor_lag_alpha, l = CTBR_MPC_L_GAIN. */
void cdhat_init(cdhat_t *d, const double c[3], const double clamp[3], double k_thrust,
                double dt, double alpha, double l_gain);
/* CascadeDhat::reset — per flight. */
void cdhat_reset(cdhat_t *d);
/* CascadeDhat::observe(gyro_t, applied_{t-1}); the first call after reset primes. */
void cdhat_observe(cdhat_t *d, const float gyro[3], const float applied[4]);
/* CascadeDhat::measured_force: 1 and f = lagged axis forces once seeded, else 0. */
int cdhat_measured_force(const cdhat_t *d, cdhat_real f[3]);

#endif /* CASCADE_DHAT_H */
