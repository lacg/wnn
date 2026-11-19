# Cobo-Briesewitz, Wahba, Hönig 2026 — Learned Incremental Nonlinear Dynamic Inversion for Quadrotors with and without Slung Payloads

arXiv:2503.09441v2 (14 Apr 2026); PDF header: "8th Annual Conference on Learning for Dynamics and Control" (PMLR 2026).
Local PDF: `papers/cobobriesewitz2026_learned_indi.pdf`. Read 19/11/2026 (abstract, network and timing sections only).

**Relevance.** A measured MLP inference time on the stock Crazyflie 2.1 MCU, in a hybrid (NN residual + INDI) framing.
**Regime:** real Crazyflie 2.1, 34.7 g, "STM32 microprocessor (168 MHz …".

## Verbatim extraction
- "a multi-layer perceptron (MLP) with 19 inputs, 6 outputs, 2 hidden layers with 24 dimensions each, and Leaky-ReLU
  activation"; output "residual force and torque".
- "We run all neural networks locally on the quadrotor hardware. For the MLP described in section 4.3, the inference
  time is 203 microseconds on the Crazyflie hardware."
- Abstract: "We further propose a hybrid approach that integrates learning-based predictions with INDI".

**Derivation (ours):** 19·24 + 24·24 + 24·6 = 1,176 weight multiplies → 203 µs / 1,176 ≈ 0.17 µs per MAC.

## Absent
- Tracking-error tables not extracted this run (unverified). No flash/RAM, no energy.
