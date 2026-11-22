# CTRL-38 re-test: rate-I (R) and d-hat (H) on base Y+F+L — pre-registration

Luiz 22/11/2026 ~08:00 EST: "Rate-I + d-hat re-test". Both inputs were judged on 08/11 on the BROKEN
encoding (10-episode ladders, docs/ctrl44*) and marked "re-test on the new base" (Plane CTRL-38, 17/11).

## Arms (one setting each, no code change, installed ram_controller ABI 60)
- **R** = Y+F+L + `--obs-rate-i`: +3 inputs = the inner rate PID integrator per axis (I / i_limit).
- **H** = Y+F+L + `--obs-cascade-dhat`: +3 inputs = d̂ from the MPCOF offset-free observer (rad/s²).
- Base Y+F+L rows = `..._a59_..._ffF_yld_full_sN` (not re-flown; ABI 60 vs 59 passed the flag-off determinism gate).
- Seeds s1-s5, seed-major interleaved (R s1, H s1, R s2, …). Driver `logs/controller/ctrl48/ab_ri_dh.sh`.

## Hypothesis
The remaining gap to MPCOF is err/steady (Y+F+L 0.812°/0.202° vs MPCOF 0.691°/0.165°). MPCOF sees the
disturbance through its observer; giving the WNN the integrator state (R) or the observer's d̂ (H) should
lower err and/or steady.

## Readout and adoption rule (Luiz's standing rule)
TEST, 35 CF2.1 units, paired per seed vs Y+F+L: stable / err / steady / pos / fails. ADOPT = consistently
better in the expected direction (err and/or steady) and not worse elsewhere on the whole vector (stable
floored from fails, pos, fails); no magnitude bar. Flat or worse → not adopt. Luiz decides.
