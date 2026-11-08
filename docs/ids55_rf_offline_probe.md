# IDS-55 offline probe: do receptive-field (family) neurons help IDS? (08/11/2026, wnn-specialist agent)

## Method

- Production loader and encoders; train_only lists for cicids/ciciot, legacy for unsw.
- Fit on a 400k TRAIN subsample, score on a 100k VAL subsample; TEST never touched.
- 1000 random neurons per scheme per seed, 3 seeds, neuron sample rate 0.25.
- Widths: 34 bits (unsw, cicids), 96 bits (ciciot).
- QUAD: order-independent vote binning. BINARY: two 250-neuron WiSARD discriminators.
- Script: scripts/ids_receptive_field_probe.py; raw JSON in docs/ids55_probe/.

## Verdict (measured)

**unswr, cicids, unswt: no gain expected; pure-family neurons HURT.**
- Spread neurons already hit a seen address for 96.6–99.7% of val rows, so there is no coverage gap to close.
- Pure family drops per-neuron balanced accuracy by 2–11 pp. It costs QUAD 1–8 pp F1 (cicids −8.4) and BINARY 1–25 pp.
- Family + 20–40% cross is within ±0.6 pp of spread on unsw, and 0.5–1.1 pp below it on cicids.
- Why it differs from CTRL-45: the controller's addresses were sparsely covered; IDS traffic is heavily duplicated, and same-family features are redundant.

**unswt BINARY: an FPR LEAD.** The one-shot TRUE write drives FPR, and family neurons curb it:

| dedupe | arm | ΔF1 | FPR |
|---|---|---|---|
| off | pure family | −1.1 pp | 20.1% → 4.7% |
| off | family + 20% | +0.8 pp | 20.1% → 13.4% |
| on | pure family | +0.85 pp | 25.6% → 19.0% |

QUAD does not show it.

**ciciot (96-bit neurons): the only coverage-limited dataset.** Spread covers 66–76%, family + 20% covers 77–88%.

| change with family + 20% | dedupe off | dedupe on |
|---|---|---|
| AUC | +0.37 pp | +0.68 pp |
| QUAD F1 | +0.16 pp | +0.51 pp |
| QUAD FPR | −0.5 pp | −9.3 pp |
| BINARY F1 | +0.26 pp | +0.38 pp |
| BINARY FPR | +2.6 pp | +3.2 pp |

**Memory everywhere:** pure family writes 3–10× fewer addresses (cicids 1158 → 118, unswr 1712 → 323, ciciot 33.9k → 6.9k); family + 20% writes 1.3–2.7× fewer. On unsw/cicids this comes at roughly equal F1, which matters for FPGA LUT cost.

**Weak families:** cicids Flow-IAT timing and fwd_size; unswr legacy ct_*_ltm context and handshake; unswt handshake/timing; ciciot rate. Strong: unswr ttl_state (99.2% balanced accuracy from 13 addresses), cicids header_window, ciciot timing.

## Caveats

- Random neurons, not GA-optimized.
- A numpy harness, not the Rust substrate.
- Some families have fewer bits than the neuron width, so those rows are partly cross.
