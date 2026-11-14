# IDS58: pre-registration of the vote + b64 combination on unswr

**LUIZ 14/11/2026: decisions 1, 2, 3, 5 APPROVED** (primary = combined vs A1; VAL gate before creating flows; vote-path ABI-29 certification for the secondary; headline rule as proposed: if 64 bits do not earn their cells over the 34-bit vote, IDS56U stays the unswr headline and combined is the max-accuracy row, else combined is the headline). **Decision 4 changed:** IDS58 runs NOW — the cicids validation was stopped (7549 paused) to close the unswr config first.

Written 13/11/2026 (experiment-design agent) for Luiz, who approved the idea on 13/11/2026 ("yes, let's do this
first"). Basis: the IDS56U readout (vote θ=0.02 unweighted, ADOPT) and the IDS-57 A1 readout (max_bits 64, ADOPT,
Pareto move), both in `docs/ids_results.md`; their pre-registrations `docs/ids56_unweighted_vote_prereg.md` and
`docs/ids57_a1_prereg.md`.

**Verdict of the review: GO-WITH-CHANGES.** The arm is worth flying and the brief's design is sound. Six changes are
applied in this file (§11 lists them). The two that matter most:
1. **The primary contrast is combined vs A1**, judged by the IDS56U rule. Combined vs IDS56U is a secondary, label-only
   contrast judged by the A1 price rule.
2. **A cheap VAL-only gate runs before the flows are released** (§5). It measures whether the vote still changes any
   cell at 60-64-bit addresses. It takes minutes, and it can only withdraw the arm.

**Status:** nothing was queued, paused, resumed or cancelled. The worker and `logs/ids/ids_sequencer.sh` were not
touched. No TEST number of an unreported genome was read. The queue script was run with `--dry-run` only (Appendix A).

Vocabulary (`docs/terminology_train_val_test.md`): **train** fits and drives the search (5-fold CV). **VAL** tunes
(threshold calibration, gates). **TEST** is the report. Every number in this file names its partition.

## 1. Question, hypothesis, falsifier

- **Question.** Both adopted changes were measured against the same lattice 34-bit controls and never flown together.
  Is "vote θ=0.02 + 64 bits" the unswr configuration, or is one of the two changes alone enough?
- **Primary hypothesis (H1).** The vote still adds F1 on top of 64 bits, without an FPR cost beyond the IDS56U
  tolerance.
- **Falsifier.** REJECT under §6.1, in which case A1 (lattice, 64 bits) stays the unswr accuracy configuration.
- **The competing hypothesis is substitution, and it is specific.**
  - Lattice writes TRUE on any own-class visit. With θ=0.02, a cell commits TRUE only if the own class holds more
    than 2% of the visits.
  - So the vote differs from lattice only at a cell where one own-class visit shares the address with at least about
    49 other-class visits (`core/vote_share.rs`).
  - Widening an address from 34 to 60-64 bits splits exactly those dense mixed cells.
  - Both changes may therefore fix the same failure (lattice's "TRUE wins" at mixed cells), by different means. If so,
    the effects are sub-additive and the combination ≈ A1.
- **Expected (arithmetic, not a result).** Under exact additivity the combination would sit at TEST F1 95.30 / FPR
  0.288 (§2). Substitution predicts F1 ≈ 94.93. Both are guesses; §5 measures the mechanism before flying.

## 2. Arms, existing data, seeds

One new arm, **C** (combined). Per seed it clones the lattice control and overrides exactly four keys:
`ids_write_rule=vote`, `ids_vote_share=0.02`, `balance_classes=false`, `max_bits=64`. No new controls are flown.

Name: `IDS58-unswr-WbCTRL-desir-BINARY-2m-max250-xo_blend_cluster-rate0.25-rklegacy-nb02-b64-encoff-r<seed>`.

The existing arms on the same seeds (DB read 13/11/2026). Source: `test_summaries.threshold_metadata`, GA phase,
`genome_type=best_f1`, `test_point=final`, mode `margin_val_cal` (τ fitted on VAL, scored on the 10% TEST). N=1 per
cell. Macro-F1 and FPR are in %, at stored precision.

```
seed   | L lattice 34b (ABI 28) | U vote 34b (ABI 28)    | A1 lattice 64b (ABI 29) | C vote 64b (to fly)
       | flow  F1      FPR      | flow  F1      FPR      | flow  F1      FPR       |
-------+------------------------+------------------------+-------------------------+--------------------
r20421 | 7509  94.1179 0.2008   | 7543  94.6996 0.2093   | 7546  94.9343 0.2579    | clone 7509 + 4 keys
r20422 | 7513  94.1143 0.2251   | 7544  94.4650 0.2507   | 7547  94.9316 0.2480    | clone 7513 + 4 keys
r20423 | 7517  94.3271 0.2028   | 7545  94.4990 0.2467   | 7548  94.9367 0.2789    | clone 7517 + 4 keys
-------+------------------------+------------------------+-------------------------+--------------------
mean   |       94.186±0.122     |       94.555±0.127     |       94.934±0.003      |
FPR    |       0.210±0.013      |       0.236±0.023      |       0.262±0.016       |
cells  |       0.666M           |       0.646M           |       1.965M            | expect ~2M
TEST support: 152,390 benign / 6,036 attack. 0.01 pp FPR = 15 benign rows.
```

Two numbers in this table bear on the design:
- **A1 is almost deterministic: TEST F1 94.934 ± 0.003 over three seeds.** The primary contrast therefore has a
  near-zero-noise control. The noise of a paired difference is the noise of C alone.
- **A1 vs U, read now, unregistered and descriptive only.** Paired by seed, dF1 is +0.235 / +0.467 / +0.438 (mean
  +0.380 ± 0.126), and dFPR is +0.049 / −0.003 / +0.032 (mean +0.026). Under the A1 rule's own price (mean ≥ +0.40 for
  about 3x cells), lattice-at-64 does **not** clearly pay its cells over vote-at-34: this would be AMBIGUOUS, not
  ADOPT. Both arms were selected on TEST, so this is a remark, not a finding. It is the reason §6.2 exists.

**Seeds.** r20421-23 only. These are the clean seeds of both parent arms: θ was tuned on r20419-20 genomes, and the
64-bit cap on random genomes. The extension seeds r20419-20 are used only under §6.3.

**Pairing.** Same seed means same folds and the same starting point. The grid changes with the cap and the GA diverges
from generation 1, so the pairs are matched starting points, not within-genome differences.

## 3. Which contrast is primary, and why

**Primary: C vs A1** (does the vote still add at 64 bits?). Four reasons:
1. **A1 is the incumbent.** It is the current unswr accuracy configuration, and later arms (the 100-bit follow-up)
   take A1 as their control. A new configuration has to beat the incumbent.
2. **The increment is free in cells.** C and A1 differ only in the writer, not the address width, so the IDS56U rule
   (sized for a no-cost change on unswr noise) fits as it stands. No new price has to be invented.
3. **The interaction risk lives here.** If the two changes substitute, this is the contrast that shows it.
4. **It is a same-build pair.** C and A1 both run on worker ABI 29. The other contrast is cross-build on a code path
   the existing equivalence record does not cover (§7).

**Secondary: C vs U** (do 64 bits still add on top of the vote?). This contrast does not decide which configuration
is used. It decides the **label**: whether the 3x cells are still paid for once the vote is in place, which matters
for the small-model row and for any sentence about width. It uses the A1 price rule, because this increment costs
about 3x the cells.

**Descriptive only: C vs L**, plus the additivity contrast per seed: I = (C − L) − (A1 − L) − (U − L) = C − A1 − U + L.
I < 0 means sub-additive (substitution). At n=3 this is reported, not tested.

## 4. Metric and definitions

- **Source.** `test_summaries.threshold_metadata`, GA phase (`ga_neurons`), `genome_type='best_f1'`,
  `test_point='final'`, mode `margin_val_cal` (keys `macro_f1`, `fpr`). Values are used at stored precision. Never
  `iterations.best_f1`.
- **Per seed, paired by seed:** dF1 = C − comparator (macro-F1, pp) and dFPR = C − comparator (pp).
- **Primary (vs A1), IDS56U definitions.** Win: dF1 ≥ +0.05. Tie: |dF1| < +0.05. Loss: dF1 ≤ −0.05.
- **Secondary (vs U), A1 definitions.** Win: dF1 ≥ +0.15. Tie: |dF1| < +0.15. Loss: dF1 ≤ −0.15.
- **Over the cap:** a C seed whose own TEST FPR is above 0.42% (XGBoost's banked TEST FPR, the A1 cap, fixed here).
- **Readout guard.** Print `worker_abi` for every row. C must be ABI 29 on all three seeds. If not, the primary
  contrast is cross-build: stop and bring it to Luiz (§7).

## 5. Pre-flight gate (VAL only, run before release, can only withdraw)

**Warranted: yes.** It costs minutes of offline compute, against roughly 25 h of worker time (§8). It tests the
mechanism that decides whether the flight can show anything. A retrain of a unswr genome took about 0.55 s per pass in
the IDS56U probe (`/tmp/ids56r/p5_7501.err`, 1.27M rows, 400 neurons). A 64-bit genome is slower but still on the
order of seconds to minutes.

**Procedure.** Run by ids-security (not this agent), at `nice -n 10`, through the worker's own construction path
(`IdsEvaluatorFactory` → `IDSCacheWrapper`, `set_write_rule`), as probe5 did for IDS56U. No Python reimplementation.
The worker is not touched.
- **Genomes:** the GA `best_f1` finals of A1, flows 7546 / 7547 / 7548. These are lattice-at-64 genomes that are
  already reported.
- **Retrain each genome twice:** once under lattice (`balance_classes=true`), once under vote θ=0.02
  (`balance_classes=false`).
- **Record:**
  - (a) VAL macro-F1, FPR and TPR, with τ fitted on VAL;
  - (b) **vote activity**: the number and fraction of written cells whose committed value differs between the two
    rules;
  - (c) the same count for the 34-bit control genomes of 7509 / 7513 / 7517. This is the reference for how much the
    vote did where it is known to help.
- **Read VAL only.** Do not print or read the TEST columns. TEST of "vote applied to an A1 genome" is a close proxy
  for C's TEST, so reading it would be a peek.

**Withdraw rule (fixed now, before the probe runs).** Withdraw the arm, that is, do not resume the flows and record
**WITHDRAWN (gate)**, if either:
- **Inert:** the vote changes **zero** cells on all three A1 genomes. The flight could then only re-run A1's GA with a
  different trajectory.
- **Harmful:** on at least 2 of the 3 genomes, VAL dF1 ≤ 0 OR VAL dFPR > +0.15 pp (the IDS56U futility thresholds).

Otherwise the gate passes and the flows are released (§9).

**What the gate cannot do.**
- It cannot tune θ. Only θ=0.02 is flown, whatever the probe shows.
- It checks a necessary condition, not a sufficient one. The probe retrains lattice-evolved genomes, and C's GA will
  pick its own. A gate pass is not evidence for H1.
- A small but positive VAL dF1 is a pass. If the vote-activity fraction at 64 bits is under 10% of the 34-bit
  reference, write that into the readout as the prior for substitution. It does not withdraw the arm.

## 6. The decision rule

Apply it once all three C flows have a GA `best_f1` final row. **Within each contrast, apply the steps in order; the
first match decides.**

### 6.1 Primary: C vs A1 (decides whether C replaces A1)

**At n=3 (r20421-23):**
1. **REJECT** if any of these holds:
   - wins ≤ 1/3 (ties count as non-wins);
   - mean dFPR > +0.15 pp;
   - at least 2/3 C seeds are over the cap.
2. **ADOPT** if all of these hold:
   - wins = 3/3;
   - mean dF1 ≥ +0.15 pp;
   - mean dFPR ≤ +0.05 pp;
   - every seed dFPR ≤ +0.10 pp;
   - no C seed over the cap.
3. **AMBIGUOUS** otherwise. A1 stays the configuration, and §6.3 applies.

**Label on an ADOPT:**
- **Dominant:** mean dFPR ≤ +0.01 pp (about 15 benign rows, below one FPR seed-SD of 0.016).
- **Pareto move (within tolerance):** otherwise. The wording is then "higher F1 at a slightly higher false-alarm rate
  than A1", and no sentence may say C "beats" A1 on FPR.

**Inconclusive clause (applies before the verdict is published).** Recount the vote activity of §5(b) on C's own
three GA finals; these are training-side cell counts, with no partition read. If the vote changes zero cells on all
three, label the verdict **INCONCLUSIVE (vote inert at C's optimum)**, whatever the F1 numbers say, and take it to
Luiz.

### 6.2 Secondary: C vs U (label only, does not change §6.1)

Use the A1 thresholds: win ≥ +0.15, ADOPT at mean ≥ +0.40, REJECT at mean < +0.15 or wins ≤ 1/3, and the same
absolute FPR cap. Apply in order, REJECT, then ADOPT, then AMBIGUOUS. The outcome only sets one sentence:
- **ADOPT** → "64 bits still earn their cells on top of the vote."
- **AMBIGUOUS or REJECT** → "on top of the vote, the 64-bit increment does not meet its pre-set price (+0.40 pp for
  about 3x the cells)."

In the second case, the size/accuracy choice between C and U goes to Luiz with both rows side by side. Nothing is
decided automatically.

This contrast is read **only if** the vote-path equivalence record W3v exists (§7). Without it, the contrast is
printed with the label "cross-build, uncertified" and sets no sentence.

### 6.3 Extension (only on AMBIGUOUS in §6.1, only on Luiz's go)

- **What flies.** No A1 flow exists on r20419-20, so an extension needs four flows: A1 on r20419-20 (already
  pre-registered: `scripts/queue_ids57_a1.py --extension`) and C on r20419-20 (this script would need the two
  bases added; that edit comes with Luiz's go). Cost is about 4 × 6-10 h.
- **n=5 rule:**
  - **ADOPT** if wins ≥ 4/5, mean dF1 ≥ +0.15, mean dFPR ≤ +0.05, every seed dFPR ≤ +0.10, and no seed over the cap.
  - Otherwise **REJECT**. There is no further extension.
- **Disclosure.** r20419-20 are the θ-tuning seeds of IDS56U. At 64 bits the grid draws genomes of a different width,
  so the genome overlap is gone. The readout still prints the r20421-23 subset on its own line.

### 6.4 What each outcome means for the configuration

```
primary (6.1) | secondary (6.2) | unswr accuracy configuration        | small-model row      | sentence
--------------+-----------------+-------------------------------------+----------------------+------------------------------------
ADOPT         | ADOPT           | C (vote + 64b); later arms pair vs C | U (vote, 34b)        | both changes add; 64b pays its cells
ADOPT         | not ADOPT       | C, flagged to Luiz (size choice)     | U                    | vote adds at 64b; 64b price not met
REJECT        | (any)           | A1 stays                             | U                    | the vote does not add at 64b
AMBIGUOUS     | (any)           | A1 stays pending §6.3                | U                    | none (provisional)
INCONCLUSIVE  | (any)           | A1 stays                             | U                    | vote inert at the 64b optimum
WITHDRAWN     | -               | A1 stays                             | U                    | gate: vote inert or harmful on VAL
```

The small-model row is U in every case, because U was already adopted against the 34-bit lattice. IDS58 does not
reopen that decision.

**Operating characteristics of 6.1.** These are simulated from the measured SDs, so they are a model, not a
measurement. The model holds A1 fixed, gives C an unknown per-seed SD s, and assumes dFPR per seed ~ N(f, 0.02).
Script: `/tmp/ids58/sim.py`, not committed, seed 58, 200k draws. Columns are ADOPT / REJECT / AMBIGUOUS in %.

```
true dF1 | f = +0.026 (IDS56U's dFPR)              | f = +0.05
         | s=0.05        s=0.13        s=0.20      | s=0.13
---------+-----------------------------------------+--------------
 0.00    |  0 / 93 /  7   1 / 72 / 27   5 / 65 / 31 |  1 / 72 / 28
+0.10    |  4 /  7 / 89  18 / 28 / 54  18 / 36 / 47 |  9 / 28 / 63
+0.20    | 94 /  0 /  6  60 /  4 / 36  43 / 13 / 44 | 31 /  4 / 65
+0.37    | 98 /  0 /  2  96 /  0 /  4  83 /  1 / 17 | 49 /  0 / 51
```

- **False adoption under no effect** is 1-5% at n=3, depending on C's unknown noise (s=0.13 is U's measured SD). That
  is the IDS56U rule's own behaviour.
- **Power.** The full IDS56U effect (+0.37) adopts 83-98%. An effect halved by substitution (about +0.2) adopts
  43-94%, and most of the rest goes to AMBIGUOUS, not REJECT.
- **What would not adopt.** A true dFPR at the +0.05 tolerance sends half of even large effects to AMBIGUOUS, by
  design. The FPR budget on top of A1 is 0.05 pp, not a fresh 0.05 on top of lattice.
- **Do the IDS56U thresholds fit?** Yes. They were sized on lattice seed noise (F1 SD 0.12-0.19, FPR SD 0.016). The
  control here is quieter (A1 F1 SD 0.003), so the same thresholds are, if anything, slightly more powerful.
  Nothing in the data argues for changing them.

## 7. Wheel and build

- **Primary (C vs A1): same build.** Both run on worker ABI 29. The readout guard requires all three C rows at
  ABI 29. If the worker moves to another ABI before C runs, stop and ask: the A1 rows would then need a new
  equivalence record.
- **Secondary (C vs U): cross-build, on a path the record does not cover.**
  - U ran on ABI 28. The ABI 28→29 change was the write-rule consolidation (commits 29b9bf38, 97e10bce: `cell_write.rs`,
    `common.metal`, `marker_slots.metal`).
  - `docs/ids57_abi29_equivalence.md` certified bit-identity **for the lattice rule only**: its harness refuses any
    other rule.
  - So the vote path at ABI 29 is not certified against ABI 28.
- **Required before §6.2 is read: W3v.** Retrain the GA `best_f1` finals of 7543 / 7544 / 7545 (vote θ=0.02,
  unweighted, 34 bits) on ABI 29 and reproduce their DB confusion matrices exactly. Record it as an addendum to the
  equivalence doc. These genomes are already reported, so this TEST read is allowed. It costs minutes, needs the
  harness's rule check widened to `vote`, and must be done by an agent other than whoever widens it.
- If W3v fails, the secondary contrast is reported as cross-build and uncertified (§6.2). The primary is unaffected.

## 8. Cost and the 64-bit cap

**Cost (an estimate, not measured).**
- Per GA generation: lattice 34b ≈ 209 s, vote 34b ≈ 269-281 s (×1.32), lattice 64b ≈ 280 s (×1.34). If the two
  slowdowns multiply, C ≈ 370 s per generation.
- Generations so far: 60-140 (patience-driven). That gives about **6-14 h per flow**, central about 8 h, so roughly
  **19-43 h for n=3**. A1 took 4.7-6.3 h per flow; IDS56U took 4.5-10.9 h.
- **Cost alert, as in A1 (alert only, per Luiz 12/11):** if any C flow's `ga_neurons` is still running 20 h after its
  `started_at`, alert Luiz. The running flow is never paused, cancelled or truncated. If the arm is abandoned for
  cost, it is **WITHDRAWN (cost)**, not REJECT.

**The cap.**
- A1's GA neurons sit at 60-63 bits and the grid winners at 60-64, so the 64-bit cap is close to binding. C will
  probably sit there too.
- **The test stays informative at the cap.** The question is which configuration to use at cap 64, and both C and A1
  are bounded by the same cap.
- **What the cap does limit is transfer.** A verdict at 64 bits says nothing about whether the vote still adds at
  100 bits. The vote's effect should shrink as addresses widen (§1), so a 64-bit ADOPT does not carry over to the
  100-bit follow-up.
- **Order with the 100-bit follow-up.** IDS58 reads out before the 100-bit pre-registration freezes its control. That
  pre-registration then takes whichever configuration §6.4 names. If C is adopted, the 100-bit arm is
  "C with max_bits 100", and its own VAL gate (100 and 128 bits) should also be run under the vote.
- **Width record (reported, not a rule input).** Print C's per-cluster bit-width histogram beside A1's. If C sits
  well under the cap while A1 sits at it, the vote changed the GA's width preference. That is reported as a finding,
  not as a confound.

## 9. Release and partial reads

- The script creates the three flows PAUSED. They are released (`POST /api/flows/:id/resume`, never a PATCH of the
  status) only after the §5 gate passes and only on Luiz's go.
- **Order.** r20421, then r20422, then r20423, by flow id under FIFO.
  - The IDS56UC flows (7549-7566) are paused with lower ids. If both sets are released, the lower ids run first.
  - Luiz said "do this first", so release IDS58 and leave IDS56UC paused until all three C flows are admitted, or
    accept the FIFO order. That is his call.
- **Partial reads.** Each C flow's TEST rows land when it finishes. They may be seen, but they decide nothing and change
  nothing. With fewer than three finals, the state is INCOMPLETE and no ordering of C vs A1 or C vs U is stated.
- There is one arm and one seed per flow, so no interleaving across arms is needed.

## 10. Disclosures and reporting obligations

**Shared controls and selection on TEST.**
- 7509 / 7513 / 7517 have now been read against three arms (IDS56U, A1, IDS58).
- More important here, both comparators of IDS58 (A1 and U) were adopted on TEST. Their TEST values carry
  selection, which makes them harder to beat, so the bias runs against C, not for it.
- A1's three seeds agree to 0.003 pp, so a lucky draw does not explain them. U's spread is ordinary (0.13).

**The adoption chain.** IDS-55, IDS-56, IDS56U, A1 and now IDS58 each make a keep-or-drop decision on the one unswr
TEST partition. A paper number for whichever configuration wins needs:
- an n=5 report with N stated;
- a VAL concordance check (as A1 §10.6): retrain each C and A1 GA final, apply §6.1 to VAL margin_val_cal, and say
  whether the VAL verdict equals the TEST verdict.

**The readout must contain:**
1. Per seed and mean±std (N=3): macro-F1, FPR, attack-F1, TPR, cells, neurons, bit-width histogram, GA generations
   and wall hours, for C, A1, U and L.
2. dF1 and dFPR vs A1 (primary), vs U (secondary) and vs L (descriptive), plus the additivity contrast I per seed.
3. Wins / ties / losses, seeds over the cap, the verdicts and labels, and the §6.4 row that applies.
4. Vote activity (§5(b)) on the A1 genomes (gate) and on C's genomes (§6.1 inconclusive clause).
5. The Rule-7 layout:
   - all 10 stored modes × 5 genome types, Grid vs GA side by side;
   - the FPR-banded Pareto view.
   Only the §4 line decides.
6. The `worker_abi` of every row and the W3v status.
7. This file's §5 gate record: genome ids, VAL numbers, activity counts, pass or withdraw.

**Not licensed by any outcome:**
- "the vote and 64 bits are additive" (n=3, descriptive I only);
- "C matches XGBoost" (needs the comparator's seed spread);
- any statement about 100 bits.

## 11. Review (experiment-design, 13/11/2026)

**Verdict: GO-WITH-CHANGES.**

```
#  | brief                                         | final                                                         | why
---+-----------------------------------------------+---------------------------------------------------------------+------------------------------------------------
1  | primary = C vs A1 and/or C vs U (open)        | primary C vs A1 (IDS56U rule); C vs U secondary, label only;  | one decision per contrast; A1 is the incumbent;
   |                                               | C vs L descriptive + additivity I                              | the vote is cell-free so the IDS56U price fits;
   |                                               |                                                               | interaction risk sits here; same build
2  | reuse the parents' rules                      | 6.1 = IDS56U thresholds + A1's absolute FPR cap;              | each increment keeps the price it was adopted at;
   |                                               | 6.2 = A1 thresholds (+0.40 for ~3x cells)                     | simulated OC: 1-5% null adopt, 83-98% at +0.37
3  | VAL pre-check: "say if warranted"             | required gate before release: VAL dF1/dFPR + vote-activity    | θ=0.02 acts only at cells with >= ~49 other-class
   |                                               | count on A1 genomes; withdraw if inert or harmful             | visits per own visit; 64b splits such cells;
   |                                               |                                                               | minutes of compute vs ~25 h of worker time
4  | -                                             | inconclusive clause: vote inert at C's optimum                | a writer that changes no cell does not test H1
5  | ABI 28->29 "verified bit-identical"           | true for lattice only; secondary needs W3v (vote path)        | ABI 29 was the write-rule consolidation; the
   |                                               |                                                               | harness refused non-lattice rules
6  | -                                             | AMBIGUOUS -> A1 stays; extension = 4 flows, on Luiz's go      | no A1 flows exist on r20419-20
7  | cost: A1 took 4.7-6.3 h                       | 6-14 h/flow estimate; 20 h alert-only                          | vote is ×1.32 s/gen at 34b, b64 ×1.34
8  | -                                             | A1-vs-U descriptive: +0.380 < +0.40 price                      | 64b-lattice may not pay its cells over vote-34b;
   |                                               |                                                               | this is why 6.2 is reported
```

**Not verified in this review:**
- vote activity at 60-64 bits (that is the §5 gate);
- GA wall time under the vote at 64 bits;
- W3v;
- whether a full IDS flow reproduces bit for bit from its seed. A1's 0.003-pp spread suggests near-determinism at
  the cap, but that is not shown.

## 12. Decisions that are Luiz's

1. **Accept the primary choice** (C vs A1), and that the secondary sets a label, not the configuration.
2. **The §5 gate:** who runs it (ids-security proposed), and whether the flows are created now (paused) or only after
   the gate passes.
3. **W3v:** authorise widening the equivalence harness to the vote rule. Without it, §6.2 is cross-build and
   uncertified.
4. **Release order** against the paused IDS56UC set (§9).
5. **If §6.4 row 2 fires** (vote adds at 64 bits but 64 bits do not pay over the vote at 34 bits): C or U as the unswr
   headline row.

## Appendix A: queue script dry run (13/11/2026)

`python scripts/queue_ids58_vote_b64.py --dry-run`. The script asserts the pinned recipe on each lattice control. It
also asserts that each C differs from A1 in exactly the three vote keys and from U in exactly `max_bits`, so the two
paired contrasts are exact by construction. No flow was created.

```
IDS58-unswr-WbCTRL-desir-BINARY-2m-max250-xo_blend_cluster-rate0.25-rklegacy-nb02-b64-encoff-r20421
  base 7509 (lattice, 34b)  diff-vs-base {"balance_classes": [true, false], "ids_vote_share": [null, 0.02], "ids_write_rule": ["lattice", "vote"], "max_bits": [34, 64]}  (inherited 46 keys verbatim)
  vs A1 7546 (PRIMARY)      {"balance_classes": [true, false], "ids_vote_share": [null, 0.02], "ids_write_rule": ["lattice", "vote"]}
  vs IDS56U 7543 (SECONDARY) {"max_bits": [34, 64]}
  [dry] not created
IDS58-unswr-WbCTRL-desir-BINARY-2m-max250-xo_blend_cluster-rate0.25-rklegacy-nb02-b64-encoff-r20422
  base 7513 (lattice, 34b)  diff-vs-base {"balance_classes": [true, false], "ids_vote_share": [null, 0.02], "ids_write_rule": ["lattice", "vote"], "max_bits": [34, 64]}  (inherited 46 keys verbatim)
  vs A1 7547 (PRIMARY)      {"balance_classes": [true, false], "ids_vote_share": [null, 0.02], "ids_write_rule": ["lattice", "vote"]}
  vs IDS56U 7544 (SECONDARY) {"max_bits": [34, 64]}
  [dry] not created
IDS58-unswr-WbCTRL-desir-BINARY-2m-max250-xo_blend_cluster-rate0.25-rklegacy-nb02-b64-encoff-r20423
  base 7517 (lattice, 34b)  diff-vs-base {"balance_classes": [true, false], "ids_vote_share": [null, 0.02], "ids_write_rule": ["lattice", "vote"], "max_bits": [34, 64]}  (inherited 46 keys verbatim)
  vs A1 7548 (PRIMARY)      {"balance_classes": [true, false], "ids_vote_share": [null, 0.02], "ids_write_rule": ["lattice", "vote"]}
  vs IDS56U 7545 (SECONDARY) {"max_bits": [34, 64]}
  [dry] not created
```

The live run (not executed) additionally:
- requires the installed worker ABI to be exactly 29;
- requires another IDS flow to be running (7531 was running at the time of the dry run);
- creates each flow by POST /api/flows with 2 experiments and verifies the count in the DB;
- does PATCH queued, then POST /pause, then GET to confirm `paused`.

## § VAL gate result (14/11/2026, ids-security)

Run 14/11/2026 11:20-11:41 UTC on Luiz's approval of the same day, exactly as §5 fixes it. Worker ABI 29 (asserted
in the probe). Construction path: `IdsEvaluatorFactory.load` → `IdsEvaluatorFactory._evaluator` → `IDSEvaluator`
(`IDSCacheWrapper`), with the write rule taken from the flow settings (lattice as flown, `balance_classes=true`; vote
θ=0.02 with `balance_classes=false`). Training, scoring and the cell export are Rust (`evaluate_multiclass_at_thresholds`,
`export_trained`). Every evaluator was built on a VAL-as-eval view of the dataset, so the TEST partition was never
scored. `nice -n 10`, `RAYON_NUM_THREADS=4`. The worker, the flows and `logs/ids/ids_sequencer.sh` were not touched.
Scratch probe `/tmp/ids58/gate.py` (not committed); raw rows `/tmp/ids58/g_<flow>.jsonl`.

Genomes: GA (`ga_neurons`) `best_f1`, `test_point='final'` (genome hash only was read from `test_summaries`).
VAL numbers: mode `margin_val_cal` (τ fitted on VAL, scored on VAL; VAL support 152,391 benign / 6,035 attack).
Changed cells: per logical neuron, union of exported keys, absent key = miss cell (0); the share is of the lattice
retrain's written cells. In every genome, every changed cell is a lattice TRUE that the vote leaves unwritten.

```
role         | flow | seed   | bits  | neur | lattice VAL      | vote θ=0.02 VAL  | dF1 pp  | dFPR pp | changed | of written      | neurons
             |      |        |       |      | F1 %     FPR %   | F1 %     FPR %   |         |         | cells   | cells (frac)    | touched
-------------+------+--------+-------+------+------------------+------------------+---------+---------+---------+-----------------+--------
A1 64b gate  | 7546 | r20421 | 61-63 | 499  | 95.3080  0.2526  | 95.3587  0.2717  | +0.0507 | +0.0190 | 1,539   | 1,970,635 0.078% | 294
A1 64b gate  | 7547 | r20422 | 60-63 | 499  | 95.2718  0.2382  | 95.3171  0.2605  | +0.0453 | +0.0223 | 1,586   | 1,942,165 0.082% | 289
A1 64b gate  | 7548 | r20423 | 61-63 | 496  | 95.3584  0.2756  | 95.3906  0.2914  | +0.0322 | +0.0157 | 1,520   | 1,981,177 0.077% | 283
-------------+------+--------+-------+------+------------------+------------------+---------+---------+---------+-----------------+--------
L 34b ref    | 7509 | r20421 | 34    | 493  | 94.7628  0.1870  | 95.0826  0.2257  | +0.3198 | +0.0387 | 1,570   |   710,285 0.221% | 354
L 34b ref    | 7513 | r20422 | 34    | 499  | 94.7295  0.2139  | 95.1889  0.2454  | +0.4593 | +0.0315 | 1,728   |   658,328 0.263% | 382
L 34b ref    | 7517 | r20423 | 34    | 497  | 94.7204  0.1851  | 95.0338  0.2179  | +0.3134 | +0.0328 | 1,647   |   628,328 0.262% | 383
-------------+------+--------+-------+------+------------------+------------------+---------+---------+---------+-----------------+--------
mean 64b     |      |        |       |      |                  |                  | +0.0427 | +0.0190 | 1,548   | 0.079%          |
mean 34b     |      |        |       |      |                  |                  | +0.3642 | +0.0343 | 1,648   | 0.249%          |
```

VAL TPR (attack recall, lattice → vote): 7546 88.75 → 89.33, 7547 88.32 → 88.95, 7548 89.41 → 89.86 (34b reference:
7509 85.52 → 87.41, 7513 85.97 → 88.19, 7517 85.34 → 87.08).

Rule application (§5, first match decides nothing; both conditions checked):
- **Inert** (zero changed cells on all three A1 genomes): NO. The vote changes 1,539 / 1,586 / 1,520 cells.
- **Harmful** (on ≥ 2/3 genomes, VAL dF1 ≤ 0 OR VAL dFPR > +0.15 pp): NO. 0/3 genomes meet either condition
  (dF1 +0.032 to +0.051, all > 0; dFPR +0.016 to +0.022, all ≤ +0.15).

**Verdict: PASS.** The arm is not withdrawn. Release of the three paused C flows stays Luiz's go (§9). θ is unchanged
(0.02); the gate tuned nothing.

Disclosures required by §5 (descriptive, not rule inputs):
- **Activity fraction vs the 34-bit reference.** At 64 bits the vote touches 0.079% of written cells against 0.249% at
  34 bits: 31.7% of the reference. That is above the 10% line in §5, so no substitution-prior note is mandated.
  In absolute terms the vote changes about as many cells at 64 bits (mean 1,548) as at 34 bits (mean 1,648).
- **Effect size on VAL.** On these lattice-evolved genomes the vote's VAL dF1 at 64 bits (mean +0.043 pp) is about
  one eighth of the 34-bit reference (+0.364 pp), and below the §6.1 win threshold (+0.05) on 2 of 3 genomes. This is a
  necessary-condition check on genomes C's GA will not pick (§5, "what the gate cannot do"); it is not evidence for or
  against H1, but it is the prior the readout should state: a substitution-sized effect, closer to the AMBIGUOUS/REJECT
  rows of the §6 operating table than to ADOPT.
- No TEST column was produced or read. The per-flow load log prints partition supports (row counts) only.

## Note 14/11/2026 — the reporting cell stays `best_f1 · margin_val_cal` (Luiz)

After seeing seed 1 (7567), Luiz asked why `best_f1 · margin_val_cal` is the cell, not the highest one
(`best_ce · margin_fixed0` 95.06 macro-F1). Ruling: **leave `best_f1 · margin_val_cal`**, no amendment, no added
secondary. Reasons recorded:
- Choosing the highest of the 50 TEST cells (5 genome types × 10 threshold modes, spread 0.01–0.1 pp) after
  seeing them is best-of-N selection on TEST — biased upwards by construction.
- `best_f1` is chosen by the search on train folds only; `margin_val_cal` fits τ on VAL (Protocol v2), so
  TEST is touched once. `train_cal` fits τ on data the RAM has memorised.
- Luiz's view: `best_fitness` is the most principled genome (it is what the GA ranks by); `best_f1` is fine
  too. Any change of reporting cell goes into a FUTURE cohort's prereg before it runs, never retro-fitted.
- Seed-1 context (TEST, macro F1, C − A1): the sign is the same in all 15 cells
  (train_cal / val_cal / τ=0 × 5 genomes), +0.02 to +0.13 pp; the prereg cell is +0.06, best_fitness·val_cal +0.02.
- 95.06 is MACRO F1 (benign + attack averaged); its attack-class F1 is 90.49.
