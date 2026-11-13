# IDS56UC: pre-registration of the IDS56U cross-dataset validation (CICIDS2017 + CIC-IoT-2023)

Written 13/11/2026 (experiment-design agent) from Luiz's 13/11 design. The parent result is IDS56U (unswr, n=3, ADOPT),
`docs/ids_results.md` "IDS56U — pre-registered test readout (n=3)", commit 5cc00f24. The rule it inherits is
`docs/ids56_unweighted_vote_prereg.md` §3, with the class-weight audit in §1.

**APPROVED by Luiz 13/11/2026 ~13:35 UTC: the per-dataset rescaled thresholds (§4) AND the grouped release order (per dataset,
dedupe ON before OFF, control next to its arm, cicids first; ms01 after; backlog last). Flows 7549-7566 created PAUSED
(18, 2 experiments each); release order = logs/ids/ids_sequence_1311.txt, run by logs/ids/ids_sequencer.sh (one flow at a time,
after A1 7546-7548). Nothing in §4-§5 may change from here on.**

## Review verdict (experiment-design, 13/11/2026): GO-WITH-CHANGES

**Nothing was queued, paused, resumed or cancelled. The worker (IDS-57 A1, flow 7546) was not touched.** Only
`--dry-run` was executed. Its output is in the appendix.

The design is sound: the arms, the pairing, the inherited recipes, the clean-attribution diff and the seed reuse all
hold. One element **does not survive transfer**: the *absolute* thresholds of the IDS56U rule. They were calibrated
to unswr's noise (σF1 0.15 pp, σFPR 0.016 pp). cicids and ciciot have different noise scales and very different base
rates. Simulated with the measured noise (§3.2), the literal thresholds behave like this:
- **ciciot:** P(ADOPT) is 0.3–12% *whatever the true effect is*. The rule would be a noise lottery.
- **cicids:** power is about 19% for an effect of unswr's size, measured in noise units.

**Changes made (all applied below, none to the arms or flows):**
1. **Thresholds re-expressed per dataset, using the same principle unswr used.** The F1 thresholds were noise
   thresholds (≈ 1 seed-SD), so they are scaled by each dataset's σF1. The FPR thresholds were a cost tolerance, so
   they are scaled relative to each dataset's base FPR. On unswr this reproduces the IDS56U numbers exactly. The
   literal unswr thresholds are still printed beside the verdict as a *non-deciding* sensitivity line (§3).
2. **A combination rule for the 4 cells** (§5), including what to do when ON and OFF disagree. That case comes with
   a pre-registered secondary comparison, so a "translates only with dedupe off" outcome is actionable.
3. **Readout guard tightened to the seed-quad, plus a release order** (§6.3–6.4). The 6 controls and 18 arms span
   about 12 days, which makes a worker-ABI change mid-cohort likely.
4. **Minority-class F1 reported beside macro-F1** (§2). The minority class is *benign* on ciciot, not attack.
5. **Script hardening.** Per-dataset pinned recipe keys, plus an assertion that the base is in-memory and not
   streaming, which the §1 class-weight audit assumes. The bases must be paused and never started. The PAUSED hold
   is race-guarded (§6.5).

If Luiz prefers the literal thresholds anyway, the cost is stated in §3.2: on ciciot the outcome would not depend on
the effect.

## 1. Design (Luiz 13/11)

Per dataset d ∈ {cicids, ciciot} × seed s ∈ {r20419, r20420, r20421}:

| Flow | What it is | Overrides vs its base | Pairing |
|---|---|---|---|
| **C_on** | existing carry-over lattice encon flow, paused and never run (not re-flown) | n/a | control |
| **U_on** | NEW: C_on + IDS56U package | `ids_write_rule=vote`, `ids_vote_share=0.02`, `balance_classes=false` | vs C_on |
| **C_off** | NEW: C_on with dedupe off | `ids_encoding_dedupe=off` | control |
| **U_off** | NEW: C_off + IDS56U package | `ids_encoding_dedupe=off` + the 3 above | vs C_off |

- **18 new flows.** Every one clones C_on of the same dataset and seed, verbatim, so the seed is unchanged. Bases
  are cicids 7531/7533/7535 and ciciot 7537/7539/7541.
- The recipes are inherited as-is and asserted, not overridden:

| Dataset | Fitness | max_bits | Crossover | Other |
|---|---|---|---|---|
| cicids | CE20, `zscore` | 34 | `blend_neuron` | |
| ciciot | B15AC, `desirability` | 100 | `blend_cluster` | `ciciot2023_neto_subsample` |

  Shared by both: `random_3way`, BINARY 2-cluster, 250 neurons, 96 input bits, rate 0.25, `train_only` ranking,
  RF off, k-fold 5×5, 250 generations, patience 5, population 50, in-memory encoded storage.
- **Clean attribution holds.** `balance_classes` only feeds the vote tally, and lattice ignores it (IDS56U prereg §1,
  CPU and GPU). The audit assumed an in-memory dataset; both bases are `ids_encoded_storage=memory` with no
  `ids_streaming`, and the script asserts it. So U vs C differs in the writer only.
- `ids_encoding_dedupe` drops exact-duplicate feature *columns* (IDS-52). It changes the encoding and the
  connectivity pool, not the vote counts.
- ms01 (7532/7534/7536, 7538/7540/7542) stays under the carry-over's own pre-registration (`docs/ids56_readout.md`
  §P1) and plays no part in this decision. **Shared control:** C_on is also ms01's control, so the ms01 and U_on
  verdicts are positively correlated: a lucky or unlucky control seed moves both. Do not read their agreement as
  independent confirmation.

**What "U_off = the exact unswr IDS56U setting" means.** It is exact for the *write-rule package and dedupe*. It is
not exact for the recipe. The carry-over uses `train_only` ranking (unswr IDS56U: `legacy`), and fitness, crossover
and bits are per-dataset. That is intended: the question is whether the package transfers onto each dataset's recipe.

## 2. Measurement (unchanged from IDS56U)

- **Decision metric:** `test_summaries.threshold_metadata`, GA phase (`ga_neurons`), `genome_type=best_f1`,
  `test_point=final`, mode `margin_val_cal`.
  - τ is calibrated on VAL (Protocol v2, `_3way`) and scored on the report-only 10% TEST.
  - F1 = macro-F1 (pp). FPR = benign FPR (pp). Pairs are by seed.
  - This is a TEST number. Never use `iterations.best_f1` (during-search k-fold).
- **Report beside every cell:**
  - minority-class F1: attack-F1 on cicids (attack 19.7% of TEST), **benign-F1 on ciciot** (benign is the minority,
    14.0% of TEST);
  - TPR and the confusion counts;
  - all 10 stored binary-margin modes × 5 genome types in Rule-7 layout;
  - the Pareto view;
  - GA generations run per flow (patience-driven budget asymmetry is reported, not adjusted, as in IDS56U);
  - logical neurons and cells.
- Cells are `mean±std (N)`, and the FPR matrix always sits beside F1.
- **Descriptive extra, not a decision:** C_off − C_on (the dedupe effect under lattice on these datasets) and
  U_off − U_on.

TEST supports, from IDS-47 confusion matrices on the same partitions:

| Dataset | Benign | Attack | Note |
|---|---|---|---|
| cicids | 227,132 | 55,656 | 0.01 pp FPR = 23 benign rows |
| ciciot | 20,000 | 122,975 | 0.01 pp FPR = 2 benign rows |
| unswr (reference) | 152,390 | 6,036 | |

## 3. Thresholds: the same rule, re-expressed per dataset

### 3.1 Derivation (fixed now, before any C_on/U flow has run)

The noise inputs come from the same proxy IDS56U used: the 9 IDS-47 lattice GA `best_f1` finals per dataset (3 seeds
× 3 crossovers, `random_3way`, same recipe family, `margin_val_cal` TEST). Values are read from the DB on 13/11:

```
dataset | flows      | F1 mean±SD (N=9) | FPR mean±SD (N=9) | corr(F1,FPR)
--------+------------+------------------+-------------------+-------------
unswr   | 7224-7248  | 94.133±0.147     | 0.214±0.015       | -0.14
cicids  | 7218-7240  | 99.124±0.057     | 0.668±0.046       | -1.00
ciciot  | 7221-7244  | 92.517±0.248     | 8.673±1.110       | +0.06
```

The IDS56U thresholds, re-expressed in the units that justified them (IDS56U prereg §3 "Why these bounds"):

| Threshold | unswr value | Unit it was justified in | Multiplier |
|---|---|---|---|
| win (per seed ΔF1 ≥) | +0.05 | noise, σF1 | 0.34 σF1 |
| ADOPT mean ΔF1 ≥ | +0.15 | noise, "≈ 1 seed-SD" | 1.02 σF1 |
| ADOPT mean ΔFPR ≤ | +0.05 | **cost tolerance** | 23.4% of base FPR |
| ADOPT every-seed ΔFPR ≤ | +0.10 | cost ceiling | 46.7% of base FPR |
| REJECT mean ΔFPR > | +0.15 | cost | 70.1% of base FPR |

**Why the FPR thresholds are not scaled by σ.** σ-scaling reproduces the noise behaviour, but on ciciot it would
tolerate +3.5 pp FPR (a 40% rise in false alarms). That is not the policy IDS56U adopted. Relative-to-base FPR keeps
the policy, "the arm may raise false alarms by about a quarter on average". On cicids it lands within 0.01 pp of σ-scaling.

**Pre-registered thresholds (pp, rounded to 2 dp; these decide):**

```
dataset | win ΔF1 ≥ | mean ΔF1 ≥ | mean ΔFPR ≤ | every-seed ΔFPR ≤ | REJECT mean ΔFPR >
--------+-----------+------------+-------------+-------------------+-------------------
unswr   |   +0.05   |   +0.15    |   +0.05     |      +0.10        |      +0.15     (= IDS56U, identity check)
cicids  |   +0.02   |   +0.06    |   +0.16     |      +0.31        |      +0.47
ciciot  |   +0.08   |   +0.25    |   +2.03     |      +4.05        |      +6.08
```

### 3.2 Operating characteristics (simulated; a model, not a measurement)

**Model:**
- Paired Δ per seed ~ bivariate normal with SD √2·σ (arm noise = control noise; the IDS56U observed paired dF1 SD of
  0.21 ≈ √2·0.15 supports this on unswr).
- Correlation as measured above.
- Sequential: n=3, then the n=5 extension on AMBIGUOUS.

P(ADOPT), by true effect:

```
true effect (dF1, dFPR)           | unswr  literal/new | cicids literal/new | ciciot literal/new
----------------------------------+--------------------+--------------------+-------------------
null (0, 0)                       |  0.07 / 0.07       |  0.00 / 0.07       |  0.01 / 0.07
unswr-sized in noise units        |  0.92 / 0.92       |  0.20 / 0.93       |  0.00 / 0.58
  (+2.2 σF1, +1.6 σFPR)           |                    |                    |
unswr-sized, FPR relative         |  0.92 / 0.93       |  0.16 / 0.93       |  0.02 / 0.81
  (+2.2 σF1, +12% of base FPR)    |                    |                    |
F1 only (+1.5 σF1, 0)             |  0.72 / 0.72       |  0.12 / 0.72       |  0.09 / 0.69
Pareto-bad (+1 σF1, +40% FPR)     |  0.00 / 0.00       |  0.00 / 0.00       |  0.00 / 0.02
```

**Reading:**
- **ciciot:** the literal rule's ADOPT probability barely moves with the true effect (0.00–0.09 across every
  scenario), and it falls when the effect gets *larger* in noise units. The reason: the per-seed +0.10 pp ceiling is
  0.06 paired-SDs, so it fails on noise alone.
- **cicids:** the literal rule mostly rejects or extends.
- **The new thresholds:** the null false-adopt rate stays at unswr's 7% on every dataset, and power returns to unswr
  levels. On ciciot it is a little lower when FPR rises, which is intended, because there FPR is a cost.
- Each readout also prints the literal-threshold verdict as a **sensitivity line**. It does not decide. If it
  disagrees with the deciding verdict, that is reported, not resolved.

**cicids caveat (corr −1.00).** On cicids, TEST errors are almost all false positives (≈1,500 FP vs ≈65 FN; TPR
99.88%). That makes macro-F1 an almost deterministic function of FPR, and ΔF1 > 0 essentially *means* ΔFPR < 0.
- The F1 and FPR criteria are one measurement there, not two.
- The IDS56U mechanism raised minority (attack) recall on unswr at a small FPR cost. On cicids attack recall has
  almost no headroom.
- An honest prior for cicids is therefore tie or REJECT. This is stated so a REJECT there is not over-read as
  "the vote fails". The flows run anyway (Hard Rule 5).

## 4. Per-cell decision rule (applied separately to each of the 4 cells)

Cells: cicids-ON (U_on vs C_on), cicids-OFF (U_off vs C_off), ciciot-ON, ciciot-OFF.

**Definitions**, per seed, on the §2 metric, with thresholds T from the §3.1 row of the cell's dataset:
- ΔF1 = U − C (macro-F1, pp); ΔFPR = U − C (pp).
- **win** if ΔF1 ≥ T.win; **tie** if |ΔF1| < T.win; **loss** otherwise.

**At n=3 (r20419–21).** Apply in this order; the first match decides.
1. **REJECT** if wins ≤ 1/3 (ties count as non-wins), OR mean ΔFPR > T.reject.
2. **ADOPT** if wins = 3/3 AND mean ΔF1 ≥ T.meanF1 AND mean ΔFPR ≤ T.meanFPR AND every seed ΔFPR ≤ T.seedFPR.
3. **AMBIGUOUS** otherwise → extend *that cell* to n=5 with r20422–23.

**At n=5.** **ADOPT** if wins ≥ 4/5 AND mean ΔF1 ≥ T.meanF1 AND mean ΔFPR ≤ T.meanFPR AND every seed
ΔFPR ≤ T.seedFPR. Otherwise **REJECT**. There is no further extension.

**Extension cost (not in this script).** No C_on exists for r20422–23, so extending one cell means 4 new flows: its
control and its arm × 2 seeds, cloned from the r20421 member with the seed changed.
- If the carry-over's own ms01 decision also extends, its r20422–23 lattice-encon controls are shared with the ON
  cell.
- An extension is a separate queue run on Luiz's go.
- The n=5 readout also prints the n=3 subset on its own.

**No θ shopping.** Only θ=0.02 is flown. A dataset-tuned θ is a new pre-registration, not a rescue (see §6.1 on why
0.02 means something different here).

## 5. What each cell's verdict means, and how the cells combine

Per dataset d (after any extension, each cell is ADOPT or REJECT):

| ON | OFF | Verdict for d | Action |
|---|---|---|---|
| ADOPT | ADOPT | **TRANSLATES** (robust to dedupe) | nb02 becomes d's write rule on its production encoding (ON) |
| ADOPT | REJECT | **TRANSLATES under the carry-over encoding** | adopt U_on for d. Do not claim a dedupe interaction (below) |
| REJECT | ADOPT | **translates only in the unswr setting** | run the secondary comparison below before changing d's recipe |
| REJECT | REJECT | **DOES NOT TRANSLATE to d** | d keeps lattice |

**Secondary comparison (pre-registered, used only in row 3).** Apply the §4 rule, with the same d thresholds and
seeds, to **U_off vs C_on**.
- d's recipe moves to U_off only if that also ADOPTs. Otherwise nothing changes.
- This is the comparison that answers "is U_off better than what d runs today?". It changes two factors, and is
  labelled so.

**The cross-dataset claim "IDS56U translates beyond unswr":**
- **Generalizes:** both cicids and ciciot reach rows 1 or 2.
- **Dataset-specific:** exactly one dataset reaches row 1 or 2. Name it.
- **unswr-specific:** neither does. A row-3 outcome counts as dataset-specific only if the secondary comparison
  ADOPTs.

**Multiplicity.** There are 4 cells, each with a ≈7% null false-adopt rate. If they were independent,
P(≥ 1 false ADOPT) ≈ 25%. They are positively correlated (shared seeds and partition), so the true figure is
somewhat lower. Consequences:
- **"At least one cell adopted" is never the headline.** The per-dataset verdict is.
- Row 1 needs both cells, which has a much lower null rate.

**When ON and OFF disagree.**
- At n=3, a disagreement is **not** evidence that dedupe modulates the effect. An interaction is a difference of
  paired differences, with SD ≈ √2 × the paired SD. The design has no power for it, and the cells share seeds.
- Write: "adopted under X, not under Y, at n=k".
- The descriptive C_off − C_on and U_off − U_on rows help show *where* the disagreement comes from. One example:
  dedupe-off lattice might simply be a stronger control.

**Scope.** Everything here is provisional at n=3, as with IDS56U. A paper claim needs n ≥ 5 with stated N and the
clean-seed status disclosed (§6.1).

## 6. Issues reviewed

### 6.1 Seeds r20419/r20420 (θ-tuning seeds on unswr): does not matter here

- On unswr the concern was a **genome** dependency: grid connectivity is seed-determined, so seeds r20419/20 would
  have re-evaluated the exact connectivities θ was tuned on (IDS56U prereg §5 Q2).
- On cicids and ciciot, the same seed draws connectivity over a different 96-bit encoding of different features, on
  a different partition. No data and no genome is shared with the unswr tuning set.
- There is no selection dependency, so all three seeds count as clean *for this question*.
- Reusing r20419–21 is also what lets the 6 existing controls be used: about 69 h of compute saved.

**What does matter about θ.** θ=0.02 was tuned on unswr's base rate. The vote commits a cell TRUE iff the own class's
unweighted share at that address exceeds θ, so θ is read on the scale of the minority-class prevalence:

```
dataset | minority (TEST)      | prevalence | θ / prevalence
--------+----------------------+------------+---------------
unswr   | attack               |   3.8%     | 0.52
cicids  | attack               |  19.7%     | 0.10
ciciot  | benign               |  14.0%     | 0.14
```

On cicids and ciciot, θ=0.02 is therefore a 4–5× more permissive operating point than the one adopted. This
experiment tests **the adopted constant**, which is the right question for "does the recipe transfer". It does not
test the min-share mechanism at a matched operating point. A REJECT does not refute the mechanism. A base-rate-matched
θ would be a new pre-registration.

Also on ciciot, `balance_classes=true` up-weights *benign* (the minority) in the vote tally, so turning it off removes
benign weight. That is the mirror image of unswr, and it is part of what is being tested.

### 6.2 Dedupe OFF as a second factor: justified, and it costs a third of the flows

- Without OFF, the experiment would test a setting that was never adopted: the unswr adoption is encoff, and the
  carry-over recipe is encon.
- Without ON, it would not answer the production question for these datasets.
- The 2×2 is analysed as two simple effects (ON cell, OFF cell), not as an interaction (§5).
- C_off doubles as a free measurement of dedupe under lattice on cicids and ciciot (descriptive, §2).

### 6.3 Worker ABI 29: OK, with a tightened guard

**Verified in the DB, 13/11 12:00 UTC:**
- All 6 C_on flows are `paused` with `pause_requested=1`, `started_at` NULL, 2 experiments each, both `pending`, and
  zero `test_summaries` rows.
- They were created under ABI 28 but have never run, so they will run on the live build, as will all 18 new flows.
  The comparison is same-build.

**Readout guard:**
- `test_summaries.worker_abi` must be **equal across all four flows of a seed-quad** (C_on, U_on, C_off, U_off of
  one dataset and seed). Both pairs, plus the descriptive cross-rows, depend on it.
- If any quad straddles an ABI change, stop and ask Luiz whether to re-fly that quad's earlier members.

**Recommendation:** no worker swap while a seed-quad is in flight (§6.4 keeps a quad adjacent).

### 6.4 Interleaving and release order

**The problem with releasing everything at once.** Admission is FIFO by min(id) among `queued`, and the C_on controls
have ids 7531–7541, below every new flow. Resuming them all at once would therefore run:
- all 6 C_on first,
- then the 18 new flows in creation order.

That separates each U from its C by up to 6 flows, which is about 4 days on ciciot: a window for ABI drift. It also
means no complete pair exists for days.

**Recommended release:** one seed-quad at a time. POST `/api/flows/:id/resume` the C_on of (d, s) and its three new
flows together, then the next quad once that one drains.
- FIFO then runs C_on, U_on, C_off, U_off back to back.
- The first complete ON pair lands after 2 flows, and the whole quad after 4.
- Order: cicids r20419 → r20420 → r20421 → ciciot r20419 → r20420 → r20421. cicids goes first because it is about
  2.4× cheaper.
- This is the interleaving that makes partial reads valid.

**Optional for ciciot, Luiz's call:** release its 3 C_on + 3 C_off first, run a VAL-only futility probe on their GA
`best_f1` genomes under θ=0.02 unweighted (IDS56U §3 gate, with the §3.1 ciciot REJECT bound), and then release the
U flows.
- Benefit: it could save up to 6 × ~16 h if the probe is futile.
- Cost: a longer ABI-drift window, and it needs the probe script, which was a `/tmp` scratch file for IDS56U.

None of this affects the ms01 flows. If Luiz resumes them, they slot in by id and change nothing here.

### 6.5 Holding the new flows PAUSED

- A POSTed flow is inserted as `pending`. The worker admits only `queued`.
- `POST /pause` refuses `pending`, and the transition table has no `pending → paused`.
- So the script creates each flow, verifies 2 experiments, then PATCHes `queued` and immediately POSTs `/pause`,
  which flips queued → paused atomically. It finally GETs the flow and requires `status=paused`.
- **The window is race-free only while an IDS flow is running.** IDS needs 10 of 13 cores, so the scheduler cannot
  admit a second IDS flow. The script asserts a running IDS flow before the batch and again before each PATCH. In
  any case the IDS-57 flows 7547/7548 are queued with lower ids.
- **Release:** POST `/api/flows/:id/resume`, never PATCH (memory note 09/11).

### 6.6 Cost (measured lattice base; the vote may run longer)

IDS-47 lattice durations on these recipes:

| Dataset | Range | Mean |
|---|---|---|
| cicids | 3.1–11.7 h | 6.8 h |
| ciciot | 12.6–23.8 h | 16.1 h |

Vote arms ran 1.41× lattice on unswt encon, and the unswr arm on r20421 ran 2.3× its control.

**Estimates:**
- 24 flows (18 new + 6 C_on): cicids ≈ 82–100 h, ciciot ≈ 193–230 h. Total **≈ 11.5–14 days**, after IDS-57
  (7547/7548) and anything else queued below these ids.
- An extended cell adds 4 flows: ≈ 1.2 days on cicids, ≈ 2.8 days on ciciot.

## 7. Exact params

The params are C_on's verbatim, with only the diffs shown in the appendix. The script asserts these pinned keys per
base:
- **Shared:** `random_3way`, binary, BINARY, 2-cluster, RF off, `train_only`, rate 0.25, `max_neurons` 250,
  `ids_n_bits` 96, k-fold 5×5, `ga_generations` 250, `patience` 5, `population_size` 50, `ids_encoded_storage`
  memory, lattice / null / `balance_classes` true / dedupe on.
- **cicids:** `cicids2017`, `max_bits` 34, zscore, `blend_neuron`.
- **ciciot:** `ciciot2023_neto_subsample`, `max_bits` 100, desirability, `blend_cluster`.

**Names:** `IDS56UC-<carry-over recipe tag>-{nb02-encon | lattice-encoff | nb02-encoff}-r{seed}`. There are 2
experiments per flow (grid_search + ga_neurons).

## Appendix: `python scripts/queue_ids56u_crossdataset.py --dry-run` (13/11/2026 12:05 UTC)

The dry run is DB read-only and makes no API call. Each line shows the full diff against the base. Every other key
(45–49 per flow) is inherited verbatim.

```
[dry] IDS56UC-cicids-CE20-zs-BINARY-2m-max250-xo_blend_neuron-rate0.25-nb02-encon-r20419  base 7531  diff-vs-base {"balance_classes": [true, false], "ids_vote_share": [null, 0.02], "ids_write_rule": ["lattice", "vote"]}  (inherited 46 keys verbatim)
[dry] IDS56UC-cicids-CE20-zs-BINARY-2m-max250-xo_blend_neuron-rate0.25-lattice-encoff-r20419  base 7531  diff-vs-base {"ids_encoding_dedupe": ["on", "off"]}  (inherited 48 keys verbatim)
[dry] IDS56UC-cicids-CE20-zs-BINARY-2m-max250-xo_blend_neuron-rate0.25-nb02-encoff-r20419  base 7531  diff-vs-base {"balance_classes": [true, false], "ids_encoding_dedupe": ["on", "off"], "ids_vote_share": [null, 0.02], "ids_write_rule": ["lattice", "vote"]}  (inherited 45 keys verbatim)
[dry] IDS56UC-cicids-CE20-zs-BINARY-2m-max250-xo_blend_neuron-rate0.25-nb02-encon-r20420  base 7533  diff-vs-base {"balance_classes": [true, false], "ids_vote_share": [null, 0.02], "ids_write_rule": ["lattice", "vote"]}  (inherited 46 keys verbatim)
[dry] IDS56UC-cicids-CE20-zs-BINARY-2m-max250-xo_blend_neuron-rate0.25-lattice-encoff-r20420  base 7533  diff-vs-base {"ids_encoding_dedupe": ["on", "off"]}  (inherited 48 keys verbatim)
[dry] IDS56UC-cicids-CE20-zs-BINARY-2m-max250-xo_blend_neuron-rate0.25-nb02-encoff-r20420  base 7533  diff-vs-base {"balance_classes": [true, false], "ids_encoding_dedupe": ["on", "off"], "ids_vote_share": [null, 0.02], "ids_write_rule": ["lattice", "vote"]}  (inherited 45 keys verbatim)
[dry] IDS56UC-cicids-CE20-zs-BINARY-2m-max250-xo_blend_neuron-rate0.25-nb02-encon-r20421  base 7535  diff-vs-base {"balance_classes": [true, false], "ids_vote_share": [null, 0.02], "ids_write_rule": ["lattice", "vote"]}  (inherited 46 keys verbatim)
[dry] IDS56UC-cicids-CE20-zs-BINARY-2m-max250-xo_blend_neuron-rate0.25-lattice-encoff-r20421  base 7535  diff-vs-base {"ids_encoding_dedupe": ["on", "off"]}  (inherited 48 keys verbatim)
[dry] IDS56UC-cicids-CE20-zs-BINARY-2m-max250-xo_blend_neuron-rate0.25-nb02-encoff-r20421  base 7535  diff-vs-base {"balance_classes": [true, false], "ids_encoding_dedupe": ["on", "off"], "ids_vote_share": [null, 0.02], "ids_write_rule": ["lattice", "vote"]}  (inherited 45 keys verbatim)
[dry] IDS56UC-ciciot-B15AC-desir-BINARY-2m-max250-xo_blend_cluster-rate0.25-nb02-encon-r20419  base 7537  diff-vs-base {"balance_classes": [true, false], "ids_vote_share": [null, 0.02], "ids_write_rule": ["lattice", "vote"]}  (inherited 47 keys verbatim)
[dry] IDS56UC-ciciot-B15AC-desir-BINARY-2m-max250-xo_blend_cluster-rate0.25-lattice-encoff-r20419  base 7537  diff-vs-base {"ids_encoding_dedupe": ["on", "off"]}  (inherited 49 keys verbatim)
[dry] IDS56UC-ciciot-B15AC-desir-BINARY-2m-max250-xo_blend_cluster-rate0.25-nb02-encoff-r20419  base 7537  diff-vs-base {"balance_classes": [true, false], "ids_encoding_dedupe": ["on", "off"], "ids_vote_share": [null, 0.02], "ids_write_rule": ["lattice", "vote"]}  (inherited 46 keys verbatim)
[dry] IDS56UC-ciciot-B15AC-desir-BINARY-2m-max250-xo_blend_cluster-rate0.25-nb02-encon-r20420  base 7539  diff-vs-base {"balance_classes": [true, false], "ids_vote_share": [null, 0.02], "ids_write_rule": ["lattice", "vote"]}  (inherited 47 keys verbatim)
[dry] IDS56UC-ciciot-B15AC-desir-BINARY-2m-max250-xo_blend_cluster-rate0.25-lattice-encoff-r20420  base 7539  diff-vs-base {"ids_encoding_dedupe": ["on", "off"]}  (inherited 49 keys verbatim)
[dry] IDS56UC-ciciot-B15AC-desir-BINARY-2m-max250-xo_blend_cluster-rate0.25-nb02-encoff-r20420  base 7539  diff-vs-base {"balance_classes": [true, false], "ids_encoding_dedupe": ["on", "off"], "ids_vote_share": [null, 0.02], "ids_write_rule": ["lattice", "vote"]}  (inherited 46 keys verbatim)
[dry] IDS56UC-ciciot-B15AC-desir-BINARY-2m-max250-xo_blend_cluster-rate0.25-nb02-encon-r20421  base 7541  diff-vs-base {"balance_classes": [true, false], "ids_vote_share": [null, 0.02], "ids_write_rule": ["lattice", "vote"]}  (inherited 47 keys verbatim)
[dry] IDS56UC-ciciot-B15AC-desir-BINARY-2m-max250-xo_blend_cluster-rate0.25-lattice-encoff-r20421  base 7541  diff-vs-base {"ids_encoding_dedupe": ["on", "off"]}  (inherited 49 keys verbatim)
[dry] IDS56UC-ciciot-B15AC-desir-BINARY-2m-max250-xo_blend_cluster-rate0.25-nb02-encoff-r20421  base 7541  diff-vs-base {"balance_classes": [true, false], "ids_encoding_dedupe": ["on", "off"], "ids_vote_share": [null, 0.02], "ids_write_rule": ["lattice", "vote"]}  (inherited 46 keys verbatim)
```
