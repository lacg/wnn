# Model-free controller block (N0, NR) — pre-registration

Luiz 22/11/2026: "What can we do to not have MPCOF running as part of the WNN?" → stop R/H (CTRL-38 re-test;
R s1 completed and valid, H s1 killed 7 min in, not a result) and fly model-free arms only.

## What MPCOF contributes on board today (Y+F+L)
- `--obs-cascade-dhat-residual`: 3 inputs computed by MPCOF's offset-free disturbance observer (airframe model:
  kt, 0.2 s motor lag, torque gains, observer gain L).
- `--yaw-trim-ff` (F): yaw-rate feed-forward cT·cresid_z, cT taken from MPCOF's gain row; requires the residual.
- Training-time only (unchanged here): the DAgger teacher is MPCOF (`--teacher mpcof`) — learning from it offline,
  nothing of it runs on the drone.

## Arms (installed ram_controller ABI 60, no code change; driver logs/controller/ctrl48/ab_n0_nr.sh)
- **N0** = Y+F+L − `--obs-cascade-dhat-residual` − `--yaw-trim-ff`: the model-free baseline (Y = `--delta-max-yaw 0.5`
  and L = dense yaw ladder stay; neither uses a model).
- **NR** = N0 + `--obs-rate-i`: the firmware rate loop's own integrator (I / i_limit) as model-free disturbance memory.
- Seeds s1-s5, seed-major interleaved (N0 sN, NR sN); paired vs Y+F+L `..._a59_..._ffF_yld_full_sN` (not re-flown).

## Questions
1. How much does removing MPCOF's observer cost (N0 vs Y+F+L)? Expected: worse on err/steady (a45 observer-free
   arm on an older base: err 1.92° vs 1.54°).
2. Does a model-free disturbance memory recover it (NR vs N0, and NR vs Y+F+L)?

## Readout
TEST, 35 CF2.1 units, paired per seed: stable / err / steady / pos / fails. Luiz decides adoption; a model-free arm
that is close to Y+F+L may be preferred for the paper even if slightly worse (the model-free claim).
