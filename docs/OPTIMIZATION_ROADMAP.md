# Optimization Roadmap

Ideas for improving the phased architecture search, based on observations from current runs.

---

## Pass 1 Results (2026-01-19)

**Configuration:** patience=2, CE percentile=75%, default bits=8, neurons=5

| Stage | CE | PPL | Accuracy | vs Baseline |
|-------|-----|------|----------|-------------|
| Initial (baseline) | 10.3443 | 31,081 | N/A | - |
| Phase 1a: GA Neurons | 10.2747 | 28,991 | 0.02% | **+0.67%** |
| Phase 1b: TS Neurons | 10.2716 | 28,901 | 0.01% | +0.70% |
| Phase 2a: GA Bits | 10.2725 | 28,925 | 0.01% | +0.70% |
| Phase 2b: TS Bits | 10.2704 | 28,865 | 0.01% | **+0.72%** |
| Phase 3a: GA Connections | 10.2725 | 28,925 | 0.01% | +0.70% |
| Phase 3b: TS Connections | 10.2711 | 28,885 | 0.00% | +0.71% |

**Total improvement: +0.71% CE (PPL: 31,081 → 28,885)**

### Key Findings

1. **Phase 1 (Neurons) did most of the work** - +0.70% of the +0.71% total came from neurons
2. **Phases 2-3 barely improved** - Bits and connections added only +0.01-0.02%
3. **Accuracy collapsed from 0.04% → 0.00%** - CE optimization destroyed accuracy
4. **All phases hit WARNING** then early-stopped - adaptive adjustments never triggered
5. **Runtime:** ~18.5 hours total

### What This Tells Us

- **Patience too low:** Only 10 iterations before stopping (5 iter × patience 2)
- **CE elites dominate:** Final genome has 0% accuracy because we pick by CE only
- **Phase ordering may matter:** Maybe optimize bits first, then neurons?
- **Need dual tracking:** Report both best-CE and best-Acc genomes

---

## Observations

1. **Early stop triggers before adaptive mechanisms kick in** (iter 10 with patience 2)
2. **Progressive threshold barely ramps** before we bail out
3. **CE elites have poor accuracy** (0.01-0.02%) and aren't improving
4. **Adaptive levels (WARNING/CRITICAL) never trigger** - no parameter adjustments happen
5. **TS was only passing 7 genomes** to next GA (fixed, now 50)

---

## TODO: CE Percentile Filter

- [ ] **Try 50% instead of 75%** for stronger CE pressure
- [ ] **Variable by phase**: 50% for phases 1-2 (neurons/bits), 75% for phase 3 (connections)
- [ ] Compare results between 50% and 75% runs

---

## TODO: Patience & Early Stopping

- [ ] **Increase patience** from 2 to 3-4 to give threshold more ramp time
- [ ] **Delay early stop** until iteration N (e.g., min 20 iterations before early stop can trigger)
- [ ] **Consult adaptive level** before stopping - don't stop if just entered WARNING and haven't tried adjustments

---

## TODO: Progressive Threshold

- [ ] **Higher starting threshold** (0.02% instead of 0.01%)
- [ ] **Faster ramp curve** to reach meaningful levels sooner
- [ ] **Steeper ramp** in early iterations, flatten later

---

## TODO: Adaptive Level Thresholds

Current thresholds (improvement delta):
- HEALTHY: < -1% (big improvement)
- NEUTRAL: -1% to 0% (small improvement)
- WARNING: 0% to 3% (stalled/mild regression)
- CRITICAL: >= 3% (significant regression)

Ideas:
- [ ] **Lower WARNING threshold** (trigger sooner, e.g., after 0.5% stagnation)
- [ ] **Lower CRITICAL threshold** (trigger adjustments earlier)
- [ ] **Track consecutive NEUTRAL** - if NEUTRAL for N checks, escalate to WARNING
- [ ] **Actually use adaptive adjustments** - verify pop/mutation increases happen at WARNING

---

## TODO: Elite CE vs Accuracy Balance

Current: 5 CE elites + 5 Acc elites = 10 total

Problem: CE elites have very poor accuracy (0.01%) and dominate

Ideas:
- [ ] **Require minimum accuracy for CE elites** (e.g., must be > 0.02%)
- [ ] **Weighted elite selection** - CE score penalized by low accuracy
- [ ] **More Acc elites** - try 4 CE + 6 Acc or 3 CE + 7 Acc
- [ ] **Track elite diversity** - warn if CE elites are too similar

---

## Priority Order (Suggested)

1. **Patience/Early Stop** - quick win, just change config
2. **Progressive Threshold** - start higher, ramp faster
3. **CE Percentile** - try 50% next run
4. **Adaptive Thresholds** - needs code changes
5. **Elite Balance** - needs more investigation

---

## Notes

- **Pass 1 complete:** +0.71% improvement, but accuracy collapsed to 0%
- **Next:** Pass 2 with patience=4, seed from Pass 1 results
- See `results_pass1_20260119_075732.json` for full results
