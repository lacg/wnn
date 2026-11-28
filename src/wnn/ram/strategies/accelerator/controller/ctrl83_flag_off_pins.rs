//! CTRL-83 (ABI 66): cf21 FLAG-OFF BYTE-IDENTITY PINS.
//!
//! The CTRL-83 options — the design-model override (D1), the nominal-mass anchor (D2, on
//! the trainer and the WNN scorers) and the payload step (D3) — are default-OFF. Off, and
//! spelled out IN RANGE (design lag = the plant's 0.20, a payload of ×1.0 at 0.25 s), the
//! cf21 lineage must not move by one bit: the trainer, the CPU and Metal scorer rows and
//! the PID / mpcof rival rows all reproduce the digests CTRL-86 captured on the ABI-64
//! build (ctrl86_flag_off_pins::ABI64_CF21 — ABI 65 is itself pinned to them), on the same
//! l2b-shaped regime. A digest moving means the cf21 lineage changed: do not re-pin
//! without a lineage decision from Luiz.

use crate::ctrl86_flag_off_pins::{
	cpu_rows_on, digests, gpu_rows_on, l2b_cfg, real_life_plant, rival_kw, rows_digest, stage2_with, trained_on,
	Spelling, ABI64_CF21, LAG,
};
use crate::design_model::DesignModel;
use crate::payload_step::PayloadStep;
use crate::stress_opts::{StressKwargs, StressOpts};

#[test]
fn ctrl83_cf21_flags_off_bit_identical_to_abi64()
{
	let cfg = l2b_cfg(4, Spelling::Legacy);
	assert!(!cfg.anchor_nominal_mass, "the D2 trainer flag defaults OFF");
	let off = digests(Spelling::Legacy);
	eprintln!("[ctrl83-pin] off {off:#018x?}");
	assert_eq!(off, ABI64_CF21, "the cf21 lineage moved with the CTRL-83 options OFF");
}

/// Every CTRL-83 stress spelled out at the nominal plant is the same plant: equal digests.
#[test]
fn ctrl83_cf21_in_range_spelled_out_bit_identical_to_abi64()
{
	let (c, d_train) = trained_on(&l2b_cfg(4, Spelling::Legacy));
	assert_eq!(d_train, ABI64_CF21[0]);
	let plant = real_life_plant(Spelling::Legacy);
	let mut s1 = stage2_with(plant);
	s1.stress = StressOpts {
		design: DesignModel::new(Some(LAG), plant.actuator(LAG)).unwrap(),
		payload: PayloadStep::from_kwargs(Some(0.25), 1.0, crate::abi38_pins::DT, 500).unwrap(),
	};
	assert_eq!(rows_digest(&cpu_rows_on(&c, &s1, 500)), ABI64_CF21[2], "CPU scorer, in-range stress");
	if let Some(g) = gpu_rows_on(&c, &s1, 500)
	{
		assert_eq!(rows_digest(&g), ABI64_CF21[3], "Metal scorer, in-range stress");
	}
	let kw = StressKwargs { model_motor_lag_s: Some(LAG), payload_step_s: Some(0.25), payload_mass_scale: 1.0 };
	let s0 = stage2_with(plant);
	for (teacher, pin) in [(0u8, ABI64_CF21[4]), (4, ABI64_CF21[5])]
	{
		let r = rival_kw(teacher, &s0, LAG, kw).unwrap();
		assert_eq!(rows_digest(&[r.0, r.1, r.2, r.3, r.4, r.5, r.6, r.7]), pin, "rival {teacher}, in-range stress");
	}
}
