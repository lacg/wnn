# CTRL-48 block 10b — pre-registration (APPROVED by Luiz 15/11/2026 ~12:20 EST)

Written BEFORE any block-10b run. Luiz 15/11: fly L1+L2y′; L3 as an A/B on the fitness position column (whole-episode
`pos` vs hold-window `pos_hold`); position floor ON in every arm. Supersedes the stopped 2×2 (L2y defective; L3 stopped
pending the floor, which then did not change its pick — docs/ctrl48_block10_2x2_s1_diagnosis.md). Driver
logs/controller/ctrl48/ab_block10b.sh (NOT launched until Luiz approves).

## Arms (each = that seed's flown `both_L1` [argv] + flags; rule P kept; seeds s1-s10, SEED-MAJOR)
| arm | extra flags | role / hypothesis |
|---|---|---|
| C  | `--zrank-pos-floor auto` | control: L1 re-flown with the floor (the floor changes picks, so arm-1 runs are not the control) |
| Y  | C + `--obs-yaw-trim-err-p` | L2y′: trim-referenced proportional yaw error (integral unchanged) cuts err/steady |
| P  | C + `--alt-integral pole3` | L3, ranked on whole-episode pos: altitude integrator cuts pos / hold droop |
| H  | P + `--fit-pos-metric hold` | L3, ranked on pos_hold: same lever, fitness no longer rewards transient pos → keeps attitude |

40 runs × ~60 min ≈ 40 h. Test = 35 CF2.1 units × 100 episodes. P/H tables carry the pole3 reference rows (56ab6fb1);
C/Y carry the alt-off rows.

## Decision rule (paired by seed), applied in order
1. **Safety gate (every arm vs C):** pooled fails ≤ 17/35000 AND not significantly worse than C (exact one-sided
   conditional binomial, p < 0.05).
2. **Y vs C:** adopt L2y′ if mean steady better by > 0.03° AND mean err not worse by > 0.03°.
3. **P vs C and H vs C:** an L3 arm is adoptable if mean pos better by > 0.010 m AND mean steady and err each not worse
   by > 0.03°.
4. **Which L3 ranking (only if P or H is adoptable):** H if it is adoptable and its mean steady is better than P's by
   > 0.03° with pos not worse than P's by > 0.010 m; otherwise P if adoptable; otherwise H.
5. **Next base:** C's recipe plus every adopted lever. If Y and an L3 arm are both adopted, their combination is flown
   as its own block before it becomes the base (no untested combination is adopted).

## Interim looks (as 2×2 Amendment 1): n = 3, 5, 7
Per arm vs C, 99 % paired t intervals (t(2)=9.92, t(4)=4.60, t(6)=3.71). Early ADOPT: target interval entirely past
its threshold, guards' upper bounds < +0.03°, fails within ceiling and not worse than C. Early NOT-ADOPT: target
interval entirely on the bad side, or a gate failure already certain. Luiz decides on stopping; undecided at n=7 →
n=10 with the point-estimate rule.

## Expected (stated in advance; not part of the rule)
- C ≈ L1 (floor changed 1 of 10 picks offline; L1 s2 improves).
- Y: small steady/err gain (pre-check: yaw inputs carry 0.73-0.77 of the yaw label vs 0.19-0.39 today).
- P: pos ≈ 0.31 m, dz_hold ≈ 0.04 m, steady worse than C by ~0.1-0.2° (s1: +0.22°).
- H: pos_hold and dz_hold like P, attitude closer to C than P is.

## Reported, descriptive only
Compact pair tables per arm (A = C · B = arm), pos_hold / dz_hold, H − P on steady and pos_hold, gaps vs the
equal-info COMPARATOR row. Nothing above may change after the first block-10b END line.
