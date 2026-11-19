# Eschmann, Albani, Loianno 2024 — RLtools: A Fast, Portable Deep Reinforcement Learning Library for Continuous Control

JMLR 25 (2024) 1–19 (per jmlr.org search hit). arXiv:2306.03530. Local PDF: `papers/eschmann2024_rltools.pdf`. Read 19/11/2026.

**Relevance.** THE concrete "small MLP on the same MCU" number set. **Regime:** inference benchmark only (no flight
result in this paper), float32 MLP.

## Verbatim extraction
Table 5 (on-device inference parameters for Table 1): input 13, policy structure [64, 64], output 4, ReLU.

Table 1 — inference times:

| Platform | RLtools: Generic | DSP Library | RLtools: Optimized |
|---|---|---|---|
| Crazyflie | 743 us (1.3 kHz) | 478 us (2.1 kHz) | 293 us (3.4 kHz) |
| Pixhawk 6C | 133 us (7.5 kHz) | 93 us (10.8 kHz) | 53 us (18.8 kHz) |
| Teensy 4.1 | 64 us (15.5 kHz) | 45 us (22.3 kHz) | 41 us (24.3 kHz) |
| ESP32 (Xtensa) | 4282 us (234 Hz) | 279 us (3.6 kHz) | 333 us (3 kHz) |
| ESP32-C3 (RISC-V) | 8716 us (115 Hz) | 6950 us (144 Hz) | 6645 us (150 Hz) |

Appendix E: Crazyflie "main processor is a STM32F405 microcontroller using the ARM Cortex-M4 architecture, featuring
192 KB of Random Access Memory (RAM) and running at 168 MHz."

**Derivation (ours):** 13·64 + 64·64 + 64·4 = 5,184 weight multiplies (+132 biases = 5,316 parameters ≈ 21.3 kB as
float32). 293 µs / 5,184 ≈ 0.057 µs per multiply-accumulate on the F405 (optimized); 743 / 5,184 ≈ 0.143 µs (generic).

## Absent
- No flash/RAM footprint of the deployed policy; no energy; no fixed-point/quantized variant.
