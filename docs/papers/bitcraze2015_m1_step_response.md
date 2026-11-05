# Bitcraze 2015 — Crazyflie 2.0 M1 motor step response (manufacturer plot)

Bitcraze AB (2015). "Step response on M1 with full battery. 0%->100%->0% PWM" (plot title, verbatim).
https://www.bitcraze.io/wp-content/uploads/2015/02/M1-step-response.png (archived copy cited by
Eschmann et al. 2023/2024). Local copy: `papers/bitcraze2015_m1_step_response.png` (513×408 PNG).

**Relevance.** The manufacturer measurement both Eschmann papers lean on. **Regime:** brushed CF2.0
7×16 mm coreless motor, LARGE-SIGNAL 0→100→0% PWM step, full battery, RPM vs seconds. The fall is an
unpowered coast (brushed driver, no active braking), so rise and fall are very different.

## What the plot shows (axes verbatim: "RPM" 0–25000, "Seconds" 0–4.5)
Peak ≈ 25,800 rpm; rise from ~1,000 rpm at ≈0.58 s; drop starts at ≈2.58 s; long tail to ~500 rpm by 4.5 s.

## Our digitisation (DERIVATION, not printed values)
Pixel-calibrated to the gridlines (1 s = 97 px ⇒ ~10 ms/px; 5000 rpm = 67.25 px):
- Rising edge τ (63.2% of 985→25,855 rpm, i.e. 16,700 rpm): t ≈ 0.649 s − 0.585 s ⇒ **τ_rise ≈ 0.064 s**
  (± ~0.01 s pixel resolution).
- Falling edge τ (36.8% of 25,446 rpm ≈ 9,400 rpm, final ≈ 0–600 rpm): ≈ 2.82–2.84 s − 2.578 s ⇒
  **τ_fall ≈ 0.23–0.26 s** (decay is not exponential — long tail).
- Arithmetic mean ≈ 0.15 s, which matches Eschmann 2023's "0.15 s ... supported by the manufacturers
  measurements", and does NOT match Eschmann 2024's "≈0.073 s (rising and falling edge averaged)".
  We cannot reconcile 0.073 s with this plot; report both, do not average.
- Greiff 2017 (Lund MSc, Table A.2) fits the same asymmetry with a sign-dependent rotor inertia
  J+ = 0.031, J− = 0.13 (units not printed) — J−/J+ ≈ 4.2 (our ratio), consistent with τ_fall/τ_rise ≈ 3.9 here.

## Absent
- No numeric τ printed; no small-signal (around-hover) step; no brushless data; no thrust trace.
