# S5p substitute test — pre-registered readout (CTRL-48, prereg docs/ctrl_select_tail_prereg.md Amendments 2+3)

Generated 14/11/2026 ~02:20 UTC from experiments/ctrl_select_tail/*.{S5p,P,B}.log and runs.log END lines. 44 runs
(30 block-4/5 + 14 block-8). Every pick scored on 35 test units x 100 episodes (fixed scorer). For the 30 old runs the P pick
is the committed .P.log; for block 8 it is the run's own published headline (END line, 2-decimal err/steady).

## Rule (Amendment 2): S5p replaces P only if BOTH hold
1. total test fails S5p <= P: **S5p 88 vs P 71 (episodes 154,000 each) -> FAIL**
2. mean err and steady, S5p not worse than P by > 0.05°: Δerr +0.002°, Δsteady −0.013° -> PASS

**Verdict: S5p does NOT replace P. P stays the selection rule.** 30/44 picks differ.

## Breakdown (descriptive)
| subset | fails S5p | fails P | Δerr (S5p−P) | Δsteady (S5p−P) |
|---|---|---|---|---|
| 30 block-4/5 runs | 75 | 57 | +0.000 | −0.019 |
| 14 block-8 runs   | 13 | 14 | +0.007 | +0.001 |
| all 44            | 88 | 71 | +0.002 | −0.013 |

- On the 30 old runs S5p (75 fails) is no better than the baseline B (76, docs/ctrl_select_tail_result.md): the p99 tail term
  at 5 val units does not see the tail. P's gain therefore comes with the 20 val units (S20 was dropped, so whether 20 units
  alone would suffice is untested).
- B re-picks of the 14 block-8 runs (out-of-sample P-vs-B replication): fails **P 14 vs B 19**, Δerr(P−B) +0.000,
  Δsteady(P−B) +0.019°, 10/14 picks differ. Same direction as the primary test (57 vs 76).
- Worst single S5p moves: mlab s2 6 -> 13, mlab s3 1 -> 5, tvxy045 s3 0 -> 3 (fails).
