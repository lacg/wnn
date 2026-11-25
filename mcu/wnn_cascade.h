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
 *                        (post rate loop, post any trim) — CTRL-38 d-hat input and
 *                        the CTRL-48 residual (which also reads rate_integ)
 *   rate_err             the rate PID's tracking error per axis (setpoint - measured,
 *                        rad/s) AS IT STANDS BEFORE this tick's rate update — the
 *                        error its P-term used at its last update (pid.error; 0 after
 *                        reset) — ABI 62 / CTRL-69 rate-error input
 * Inputs a model does not use are ignored (the header's WNN_OBS_* flags say which).
 *
 * BIT PARITY. Every operation mirrors the Rust source in order and precision
 * (f32 features, f64 rate-I / d-hat observer; the Mahony estimator f32 from ABI 61,
 * f64 before — WNN_EST_F32 below). Build with
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

/* CTRL-48 a3 (--obs-cascade-dhat-residual). A header exported before 10/11/2026 has
 * no WNN_OBS_DRES: such an exporter wrote a dres winner as runnable WITHOUT the three
 * residual columns (left at 0) — re-export it. The observer runs for a2 or a3. */
#ifndef WNN_OBS_DRES
#define WNN_OBS_DRES 0
#endif
#define WNN_OBS_OBSERVER (WNN_OBS_DHAT || WNN_OBS_DRES)

/* ABI 62 (CTRL-69). Emitted by the exporter ONLY when not at the default, so every older
 * header is byte-identical and takes these (= the ABI-61 blocks):
 *   WNN_OBS_RATE_I_AXES   rate-I's emitted axes, bitmask bit 0 roll, 1 pitch, 2 yaw (7 = all)
 *   WNN_OBS_RATE_ERR      the model-free rate-error block: e = rate_err (the firmware rate
 *                         PID's last-update error) * WNN_RATE_ERR_INV_INIT[a] (1/bank range),
 *                         one-pole low-passed y += WNN_RATE_ERR_ALPHA * (x - y) every tick
 *                         (rate_err_lpf.rs; both constants exported from the wheel), after
 *                         rate-I and before d-hat; WNN_OBS_RATE_ERR_AXES its emitted axes */
#ifndef WNN_OBS_RATE_I_AXES
#define WNN_OBS_RATE_I_AXES 7
#endif
#ifndef WNN_OBS_RATE_ERR
#define WNN_OBS_RATE_ERR 0
#endif
#ifndef WNN_OBS_RATE_ERR_AXES
#define WNN_OBS_RATE_ERR_AXES 7
#endif

/* CTRL-48 block 10 / levers (ABI 46-58), ported 22/11/2026 so the Y+F+L base runs on
 * board. The exporter emits each macro ONLY when the winner uses it, so every older
 * header stays byte-identical and takes these defaults (= the block off):
 *   WNN_OBS_ATT_BIAS_PRED   CTRL-46 Mahony-bias predictor, 2 features at the tail
 *                           (att_bias_pred.rs; kp/ki/dt = the estimator's)
 *   WNN_OBS_TILT_REF_ERR    block 10 L1: roll/pitch error vs (target + the position
 *                           loop's tilt reference + predicted bias) (tilt_ref_err.rs);
 *                           WNN_TILT_REF_SHAPE_INIT = {b_xy, omega_n, zeta, max_tilt}
 *   WNN_OBS_YAW_FROM_ATT_EST ABI 55: anchored yaw heading = the Mahony estimate's yaw,
 *                           unwrapped onto the running heading (controller.rs mahony_heading)
 *   WNN_YAW_TRIM_FF_CT      ABI 58: the held yaw setpoint += cT * cresid_z (clamped);
 *                           the bank value (what the delta law leaks) is untouched
 *   WNN_DELTA_MAX_YAW       ABI 57: the yaw bank's own delta_max
 *   WNN_DELTA_LEAK_YAW      ABI 63 (CTRL-74): the yaw bank's own output leak
 *   WNN_YAW_INTEGRAL_LEAK   ARM 3: the yaw integrals' own leak
 * A refusal in scripts/export_controller_c.py is lifted only for what is ported here. */
#ifndef WNN_OBS_ATT_BIAS_PRED
#define WNN_OBS_ATT_BIAS_PRED 0
#endif
#ifndef WNN_OBS_TILT_REF_ERR
#define WNN_OBS_TILT_REF_ERR 0
#endif
#ifndef WNN_OBS_YAW_FROM_ATT_EST
#define WNN_OBS_YAW_FROM_ATT_EST 0
#endif
#ifndef WNN_YAW_TRIM_FF
#define WNN_YAW_TRIM_FF 0
#endif
#ifndef WNN_DELTA_MAX_YAW
#define WNN_DELTA_MAX_YAW WNN_DELTA_MAX
#endif
#ifndef WNN_DELTA_LEAK_YAW
#define WNN_DELTA_LEAK_YAW WNN_DELTA_LEAK
#endif
#ifndef WNN_YAW_INTEGRAL_LEAK
#define WNN_YAW_INTEGRAL_LEAK WNN_INTEGRAL_LEAK
#endif
#if WNN_OBS_TILT_REF_ERR && !WNN_OBS_ATT_BIAS_PRED
#error "wnn_cascade: obs_tilt_ref_err needs obs_att_bias_pred (Rust refuses it too)"
#endif
#if (WNN_OBS_ATT_BIAS_PRED || WNN_OBS_YAW_FROM_ATT_EST) && !WNN_OBS_ATT_EST
#error "wnn_cascade: the bias predictor / Mahony heading need obs_att_estimator"
#endif
#if WNN_YAW_TRIM_FF && !WNN_OBS_DRES
#error "wnn_cascade: yaw_trim_ff needs the CTRL-48 residual block (cresid_z)"
#endif

/* d-hat observer (mcu/cascade_dhat.[ch], shared with the MPCOF port). Precision:
 * default f64 = the CPU reference (cascade_dhat.rs), bit exact. -DWNN_DHAT_F32 = the
 * Metal kernel's f32 twin (cheaper on an M4F, which has no double-precision FPU; NOT
 * bit-identical to the CPU reference). */
#include "cascade_dhat.h"
typedef cdhat_real wnn_obs_real;
typedef cdhat_t wnn_dhat_t;

typedef struct
{
	float gyro[3];
	float accel[3];
	float target[3];
	float vert[3];
	float horiz[4];
	double rate_integ[3];
	float applied_prev[4];
	double rate_err[3];
} wnn_tick_in_t;

/* Mahony estimator precision. WNN_EST_F32 1 (emitted by the exporter when the wheel's
 * ram_controller.ATT_EST_PRECISION is "f32", ABI >= 61): the f32 filter + heading of
 * mahony_f32.rs, bit-identical to Rust AND to the Metal kernel (no libm: atan2 is the
 * deterministic atan2_det, sqrt is the IEEE sqrt). Absent / 0: the ABI <= 60 f64
 * estimator (estimator.rs MahonyFilter + the f64 heading), so older headers keep
 * their parity. */
#ifndef WNN_EST_F32
#define WNN_EST_F32 0
#endif
#if WNN_EST_F32
typedef float wnn_est_real;
#else
typedef double wnn_est_real;
#endif

typedef struct
{
	wnn_est_real dt, kp, ki;
	wnn_est_real q[4];
	wnn_est_real integral[3];
} wnn_mahony_t;

#define WNN_MAX_INTEGRALS 5

/* att_bias_pred.rs AttBiasPredictor (f32): eps/j = the linearised Mahony error
 * dynamics (body x, y), v_prev = last horizontal velocity. */
typedef struct
{
	float eps[2], j[2], v_prev[2];
	int have_prev;
} wnn_abp_t;

typedef struct
{
	wnn_mahony_t est;
	wnn_dhat_t dhat;
	wnn_abp_t abp;
	float abp_last[2];       /* [abp_r, abp_p] of this tick (rad) */
	float cresid_z;          /* this tick's yaw residual (the yaw-trim FF input) */
	float rerr_y[3];         /* ABI 62: the rate-error low-pass state (unused unless WNN_OBS_RATE_ERR) */
	float integ[WNN_MAX_INTEGRALS];
	float yaw_heading;
	float range[4], gamma[4];
	float resid_inv_a[3], resid_w[3]; /* CTRL-48 a3 scale (unused unless WNN_OBS_DRES) */
	float setpoint[4];       /* bank values = Rust's self.pwm (bank deviations) */
	float setpoint_prev[4];  /* Rust's self.pwm_prev (delta law only) */
	float held[4];           /* the HELD setpoint flown = setpoint + the yaw-trim FF */
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
