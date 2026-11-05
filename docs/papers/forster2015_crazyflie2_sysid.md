# Förster 2015 — System Identification of the Crazyflie 2.0 Nano Quadrocopter

Förster, J. (2015). *System Identification of the Crazyflie 2.0 Nano Quadrocopter.*
Bachelor thesis, ETH Zurich (IDSC; supervisors M. Hamer, R. D'Andrea), 147 pp.
DOI: 10.3929/ethz-b-000214143 — https://www.research-collection.ethz.ch/handle/20.500.11850/214143
Local PDF: `papers/forster2015_crazyflie2_sysid.pdf` (fetched from the GitHub mirror
MomsFriendlyRobotCompany/thundercougarfalconbird/docs/dynamics; ETH server rate-limited.
pdfinfo CreationDate 31/08/2015, 147 pages, which matches the ETH record).

**Relevance.** This is the primary measured source for the Crazyflie 2.0 rotor-drag coefficient that later
papers quote (e.g. -9.1785e-7 kg/rad). We use it to set the ABI-32 plant's linear rotor drag k_d (1/s).
**Regime:** a real Crazyflie 2.0 (brushed, 28.0 g) clamped to a load cell, with
a fan + PMMA tube + honeycomb wind generator. Measured wind speed was about 2–4 m/s (Fig 4.12 axes: |v| 2–4 m/s;
the carton prototype gave 2.25–4.25 m/s). The supply was a constant 3.7 V. This is NOT free flight.
Parasitic (frame) drag is removed by design: the baseline is averaged with the wind already on and motors off (§4.4.3).
So K_aero is the increment caused by the rotors only.

## Verbatim extraction

Model (eq 4.2, citing [13]): `fa = Kaero θ̇Σ R−1 ẋ`, with `Kaero = diag(κ⊥, κ⊥, κk)`,
`θ̇Σ = Σ_{i=1..4} |θ̇i|` "the sum of the angular velocities of all rotors".

Simple model, eq (4.8) and Appendix overview:
`Kaero = diag(−9.1785·10−7, −9.1785·10−7, −10.311·10−7) kg·rad−1`

Complex model, eq (4.9). Model: `fa = Kaero,1 θ̇Σ v + Kaero,2 (v).∧2 + Kaero,3 θ̇Σ² + Kaero,4 v + Kaero,5 θ̇Σ + Kaero,6`

| term | x, y | z | printed unit |
|---|---|---|---|
| Kaero,1 | −8.9158·10−7 | −3.1574·10−7 | kg·rad−1 |
| Kaero,2 | −6.8572·10−5 | 5.2068·10−5 | kg·rad−1 (as printed) |
| Kaero,3 | 1.1254·10−10 | −2.3662·10−10 | kg·rad−1 (as printed) |
| Kaero,4 | −5.4233·10−4 | −2.2935·10−3 | kg·rad−1 (as printed) |
| Kaero,5 | −1.8443·10−6 | 4.8857·10−7 | kg·rad−1 (as printed) |
| Kaero,6 | 6.4061·10−3 | 3.0985·10−3 | kg·rad−1 (as printed) |

The thesis prints the same unit, kg·rad−1, for every term. Dimensionally that is only right for Kaero,1.
Cross-coupling model, eq (4.11): `Kaero,full = [[−10.2506, −0.3177, −0.4332],[−0.3177, −10.2506, −0.4332],[−7.7050, −7.7050, −7.5530]]·10−7 kg·rad−1`.
Sum of squared errors: complex 0.0022072 N², simple 0.0035997 N², cross-coupling 0.0019127 N² (§4.6).
Observed trends (§4.5): "κk is decreasing whereas κ⊥ increasing for increasing rotor angular velocities. In addition both drag
coefficients decrease for increasing relative wind speeds."

Supporting maps (Appendix overview):
- Mass: `m = 28.0 g`
- `fi = 2.130295·10−11·cmd²i + 1.032633·10−6·cmdi + 5.484560·10−4` (thrust per rotor [N], cmd ∈ [0, 65535])
- `θi = 0.04076521·cmdi + 380.8359` (rad/s; second, better-fit series of eq 3.4. First series: `0.03950236·cmd + 420.9420`)
- Drag experiment commands: 0, 10000, 20000, 30000, **37300**, 40000, 50000. The wind angles were 0°, 48.4° and 77.9°, and the fan voltages 5–12 V.
- Analysis code: `Theta_sigma = Thetas*4`. Θ̇Σ is computed as 4 × the single measured rotor speed.

## Derivation (ours, not in the thesis)

Hover thrust per rotor: m·g/4 = 0.028·9.81/4 = 0.06867 N. Solving eq 3.3 gives cmd = 37287, which is essentially the
37300 test command. From eq 3.4, θi = 1900.8 rad/s (18,152 RPM), so Θ̇Σ = 7603 rad/s.
Linear horizontal drag force per unit speed = |κ⊥|·Θ̇Σ = 9.1785e-7 × 7603 = 6.98e-3 N/(m/s).
**k_d = |κ⊥|·Θ̇Σ / m = 6.98e-3 / 0.028 = 0.249 1/s.**
- Complex model, ∂fa/∂v at v=0: (8.9158e-7·7603 + 5.4233e-4)/0.028 = 0.261 1/s.
- Cross-coupling κ1 = 10.2506e-7: 0.278 1/s.
- Mass sensitivity: we re-solve hover Θ̇Σ with the same maps at each mass. The motor maps were measured at 28 g, so at heavier masses this is an extrapolation.
  The results are 0.027 kg → 0.253, 0.030 → 0.242, 0.034 → 0.229, 0.037 → 0.221 1/s. Roughly k_d ∝ 1/√m.

## Absent / caveats
- No free-flight validation of the drag coefficients ("Another experiment to verify the drag coefficients was not designed", §4.6).
- No measurement below about 2 m/s wind. Hover-speed behavior (|v| < 1 m/s) is extrapolated.
- No uncertainty or confidence interval on κ⊥. Only the SSE and a figure of the trend.
- No Crazyflie 2.1 or brushless data. No quadratic parasitic-drag coefficient (removed by the baseline subtraction).
- The thesis gives no hover rotor speed or hover command directly. Hover is our derivation from the maps.

## Motor transfer function (§3.4) — added 05/11/2026 (motor-lag question)

**Regime:** one CF2.0 brushed motor on the load cell, command→thrust, sinusoidal excitation
`ue[n] = A·cos(Ωl n) + B` (eq 3.6), "commanding it to the specified motor(s) with a frequency of 500 Hz";
B fixed to 30000, A = 20000 (Table 3.1: series 1 N=4000, l∈[0..2000], motor #1, 43 measurements;
series 2 N=8000, l∈[0..4000], motor #3, 40 measurements). Frequencies above 12.5 Hz ignored.
Weighted LS fit of a frequency-response estimate (method of D'Andrea et al. [5]).

Verbatim results (eqs 3.15–3.17):

    H1(z) = 6.0705967·10−8 / (1 − 0.9745210·z−1)
    H2(z) = 7.2345374·10−8 / (1 − 0.9695404·z−1)
    "it is recommended to use the second transfer function: H(z) = H2(z)."

Caveat printed by the author: latency is not compensated ("currently the latency of the signals is
completely omitted"); radio ping RTT mean 57.1030 ms (Table 3.2); load-cell latency "was not determined".
Fig 3.16: l = 12 fits "very well", l = 25 shows "a distinct shift".

**Derivation (ours, not printed):** sample period Ts = 1/500 Hz = 0.002 s (consistent with the
12.5 Hz cut: l=200, N=8000 → 200/8000·500 = 12.5 Hz). First-order τ = −Ts / ln(pole):
H1 → τ = 0.0775 s; **H2 → τ = 0.0647 s**. In our T = 4τ convention: H1 → T = 0.31 s, H2 → T = 0.26 s.
Cross-check: Eschmann et al. 2024 (arXiv:2404.07837) quote "in [2] [Förster] the time constant has been
estimated at 0.065 s" — matches our H2 derivation.

Absent: no time-domain step test, no rise/fall asymmetry, no rpm measurement in this section, no
uncertainty on the poles; the fit is command→thrust, not command→rpm.
