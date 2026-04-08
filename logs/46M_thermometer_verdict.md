# 46M Thermometer Encoding Sweep Verdict

**Sweep tag:** THERMO46M
**Hypothesis tested:** Is the 32-bit address-tap saturation observed in
the Pareto sweep bounded by the 8-bit thermometer encoding rather than
the address-tap width itself?

**Method:** Two architectures × 4 thermometer widths (8/16/32/64). The
8-bit baselines come from yesterday's Pareto sweep (flows 1185 and 1188);
the 16/32/64-bit runs are from today's THERMO46M sweep (flows 1189-1200).

## Results by architecture

### 96n × 32b

| Thermometer | Seeds | F1 | FPR | Acc |
|---|---:|---:|---:|---:|
| 8-bit | 1 | 84.36% | 1.61% | 97.59% |
| 16-bit | 2 | 84.13±0.08% | 1.17±0.22% | 97.53±0.02% |
| 32-bit | 2 | 84.53±0.25% | 1.50±0.27% | 97.63±0.06% |
| 64-bit | 2 | 84.59±0.13% | 1.69±0.17% | 97.64±0.03% |

### 500n × 34b

| Thermometer | Seeds | F1 | FPR | Acc |
|---|---:|---:|---:|---:|
| 8-bit | 1 | 84.73% | 1.91% | 97.68% |
| 16-bit | 2 | 84.43±0.21% | 1.51±0.05% | 97.60±0.05% |
| 32-bit | 2 | 84.46±0.41% | 1.37±0.18% | 97.61±0.09% |
| 64-bit | 2 | 84.87±0.03% | 1.75±0.02% | 97.70±0.01% |

## Verdict

- **96n × 32b:** FLAT (within ±1pp of 8b baseline)
- **500n × 34b:** FLAT (within ±1pp of 8b baseline)

### ✅ ADDRESS-TAP SATURATION CONFIRMED

Neither architecture shows meaningful F1 lift across thermometer widths.
The 32-bit address-tap width is genuinely the discrimination ceiling on
this dataset, regardless of input encoding richness. The thermometer
encoding caps effective discrimination at the input layer, and increasing
its width doesn't help downstream discrimination.

