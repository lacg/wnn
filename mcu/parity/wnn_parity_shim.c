/* Host parity shim: the exported controller (wnn_cascade.c) closed over the
 * reference firmware rate loop (wnn_ratefw.c) exactly as WnnController::step
 * closes over its CtbrInner — features read the integrator BEFORE this tick's
 * rate update, the d-hat observer reads the command applied LAST tick, and the
 * held setpoint is flown on every tick. Flat C ABI for ctypes
 * (mcu/parity/test_cascade_parity.py). One translation unit: build with
 *   cc -O2 -std=c11 -ffp-contract=off -shared -fPIC -I mcu \
 *      -DWNN_MODEL_HEADER='"<abs path to header>"' mcu/parity/wnn_parity_shim.c
 */
#include "../wnn_cascade.c"
#include "../wnn_ratefw.c"

static wnn_ctrl_t g_ctl;
static wnn_ratefw_t g_rate;
static float g_applied[4];

/* features, input bits, neurons, bits, levels, action_repeat, rate_i, dhat */
void shim_dims(int out[8])
{
	int d[8] = {WNN_FEATURES, WNN_INPUT_BITS, WNN_NEURONS, WNN_BITS, WNN_LEVELS,
	            WNN_ACTION_REPEAT, WNN_OBS_RATE_I, WNN_OBS_DHAT};
	memcpy(out, d, sizeof d);
}

void shim_init(void)
{
	wnn_init(&g_ctl);
	wnn_ratefw_init(&g_rate, wnn_rate_gains, WNN_RATE_OUT_LIMIT_N, WNN_RATE_HOVER_N,
	                WNN_K_THRUST, WNN_MAIN_LOOP_HZ, WNN_ATTITUDE_HZ, WNN_RATE_LPF_HZ);
}

/* reset(init_yaw) + seed_att_estimator(q0) + set_collective_anchor(anchor).
 * pwm_applied resets to 0.5 as in Rust (the observer's priming call ignores it). */
void shim_reset(float init_yaw, const float q0[4], float anchor)
{
	wnn_reset(&g_ctl, init_yaw, q0);
	wnn_ratefw_reset(&g_rate);
	wnn_ratefw_set_anchor(&g_rate, anchor);
	for (int m = 0; m < 4; m++) g_applied[m] = 0.5f;
}

static void fill_in(wnn_tick_in_t *in, const float *gyro, const float *accel,
                    const float *target, const float *vert, const float *horiz)
{
	memcpy(in->gyro, gyro, sizeof in->gyro);
	memcpy(in->accel, accel, sizeof in->accel);
	memcpy(in->target, target, sizeof in->target);
	memcpy(in->vert, vert, sizeof in->vert);
	memcpy(in->horiz, horiz, sizeof in->horiz);
	wnn_ratefw_integ(&g_rate, in->rate_integ);
	memcpy(in->applied_prev, g_applied, sizeof g_applied);
}

/* One physical tick; returns 1 on a decision tick. */
int shim_tick(const float *gyro, const float *accel, const float *target, const float *vert,
              const float *horiz, float sp[4], float pwm[4])
{
	wnn_tick_in_t in;
	fill_in(&in, gyro, accel, target, vert, horiz);
	int decided = wnn_tick(&g_ctl, &in, sp);
	wnn_ratefw_fly(&g_rate, sp, gyro, pwm);
	memcpy(g_applied, pwm, sizeof g_applied);
	return decided;
}

void shim_dbg(float *features, uint8_t *frame, uint32_t *addr, uint8_t *fired)
{
	memcpy(features, g_ctl.features, sizeof g_ctl.features);
	memcpy(frame, g_ctl.frame, sizeof g_ctl.frame);
	memcpy(addr, g_ctl.addr, sizeof g_ctl.addr);
	memcpy(fired, g_ctl.fired, sizeof g_ctl.fired);
}
