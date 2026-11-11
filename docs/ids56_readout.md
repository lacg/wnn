# IDS-56 readout — min-share vote (`ms01`, θ=0.1) vs `lattice` × encoding dedupe (`encoff`/`encon`)

Written 10/11/2026 (ids-security agent, read-only on the queue). Source of truth for the numbers:
`test_summaries` (DB `/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db`), `test_point='final'`,
`threshold_metadata.margin_val_cal` (threshold fit on the 10% VAL partition, scored on the 10% TEST partition —
Protocol v2). No `iterations.best_f1` anywhere. Rule-7 tables: `docs/ids_results.md` § "IDS-56".

Design (`scripts/queue_ids56_vote_share_ab.py`, worker ABI 28, commit 590cc0c5): BINARY 2-cluster, 2m, max250
(≤500 neurons), 16b Wb top20, blend_neuron, rate 0.25, rf off, feature ranking legacy, seeds r20419–r20423 paired.

- `lattice` = `ids_write_rule=lattice` (BINARY TRUE-wins: any own-class visit makes the cell TRUE).
- `ms01` = `ids_write_rule=vote, ids_vote_share=0.1`: a cell is TRUE only if the own class's class-weighted share
  of its visits is > 0.1 (ram_core `vote_share.rs`).
- `encoff` / `encon` = `ids_encoding_dedupe` off / on (IDS-52). With it on, the thermometer keeps only threshold
  columns that are non-constant and distinct on TRAIN. On unswt that is **288 → 161 input bits**. Measured on
  the encoff TRAIN encoding: 17 bits are constant, and only 161 of the 271 non-constant bits are distinct.

## 1. unswt result (flows 7481–7500, 20/20 done)

GA Neurons `best_f1`, `margin_val_cal`. F1 / FPR / Acc in %, then neurons / logical cells.

```
seed   | lattice-encoff              | ms01-encoff                 | lattice-encon               | ms01-encon
-------+-----------------------------+-----------------------------+-----------------------------+----------------------------
r20419 | 90.82/4.29/90.84 498/216.7K | 90.84/2.91/90.85 398/163.9K | 91.09/5.30/91.12 499/357.7K | 91.82/4.48/91.85 489/327.0K
r20420 | 90.87/4.95/90.89 498/210.6K | 90.72/5.27/90.75 496/218.8K | 91.32/3.66/91.34 494/352.5K | 91.83/3.22/91.85 417/282.2K
r20421 | 90.85/6.21/90.89 497/207.2K | 90.64/3.67/90.66 441/185.4K | 91.37/4.29/91.40 499/354.6K | 91.75/3.15/91.77 499/353.2K
r20422 | 90.79/4.71/90.82 500/215.7K | 90.81/4.29/90.83 499/209.4K | 91.22/3.86/91.24 499/348.4K | 91.94/4.09/91.97 496/350.1K
r20423 | 90.45/4.61/90.47 500/217.4K | 90.56/4.82/90.58 499/216.2K | 91.27/3.48/91.29 499/361.9K | 91.80/4.37/91.83 496/353.4K
-------+-----------------------------+-----------------------------+-----------------------------+----------------------------
mean   | 90.75±0.17 4.95±0.74 90.78  | 90.71±0.12 4.19±0.93 90.73  | 91.25±0.11 4.12±0.73 91.28  | 91.83±0.07 3.86±0.63 91.85
cells  | 498.6n 213.5K±4.4K          | 466.6n 198.7K±23.5K         | 498.0n 355.0K±5.1K          | 479.4n 333.2K±30.6K
```

Paired ms01 − lattice, per seed (GA):
- **encon:** ΔF1 +0.73 / +0.52 / +0.38 / +0.73 / +0.53, mean **+0.58±0.15, 5/5**. ΔFPR −0.83 / −0.44 / −1.14 / +0.23 / +0.89,
  mean −0.26 (3/5 lower). About 6% fewer cells (333K vs 355K).
- **encoff:** ΔF1 +0.02 / −0.14 / −0.21 / +0.02 / +0.10, mean **−0.04±0.13**. **3 up, 2 down** (the main-session note said 2 up
  / 3 down; the two "+0.02" seeds are ties). ΔFPR mean −0.76 (3/5 lower).
- Grid (no GA) shows the same pattern: encon +0.68±0.16 (5/5), encoff +0.14±0.16 (4/5). So the effect is present before
  the search.
- Dedupe alone (lattice-encon − lattice-encoff): +0.50 pp F1, −0.83 pp FPR, +66% cells (213.5K → 355.0K).
  This also answers the unswt half of IDS-52 for these 5 seeds (see §4 on the duplicate held flows).

Budget: every flow ran exactly 60 GA generations, so the arms are budget-matched. The vote costs wall time per
generation: lattice 0.43±0.03 h (encoff) and 0.51±0.05 h (encon); ms01 0.75±0.08 h (encoff) and 0.72±0.02 h (encon).

Low-FPR Pareto caveat (all genome types × all modes, seed means). ms01-encon has no seed-mean point below 2.9% FPR:
its best are 91.74/3.22 (GA best_acc argmax_classnorm) and 90.35/2.92. lattice-encon keeps **90.23/2.09** (grid best_f1
argmax_classnorm). So ms01 wins the F1-optimal zone, and lattice still owns the ≤2.5% FPR zone.

## 2. Mechanism (why the gain appears only with encoding on)

Probe: `/tmp/ids56/probe.py`, `/tmp/ids56/probe2.py` (scratch, not committed). Each production GA `best_f1` genome
(20 flows) was retrained through the worker's own construction path (`IdsEvaluatorFactory` → `IDSCacheWrapper`,
OI env applied). The write rule was switched with `IDSCacheWrapper.set_write_rule`, and results were read with
the Rust `evaluate_multiclass_at_thresholds` and `export_trained`. There is no Python training or scoring.
**Fidelity:** all 20 retrains reproduce the DB `margin_val_cal` F1 exactly (e.g. 7481: 0.908159 = DB).

### 2a. MEASURED — it is the write rule, not the genome
The same genome was retrained under the other rule (GA best_f1, test F1, mean of 5):

```
genomes from arm      | trained lattice | trained ms01 | Δ
----------------------+-----------------+--------------+------
lattice-encoff        | 90.75           | 90.74        | −0.01
ms01-encoff           | 90.62           | 90.71        | +0.09
lattice-encon         | 91.25           | 91.84        | +0.59
ms01-encon            | 91.09           | 91.83        | +0.74
```
Swapping only the cell rule on a fixed lattice-evolved genome reproduces the whole encon gain. The GA did not need
to co-adapt.

### 2b. MEASURED — the vote clears FEWER cells under encon, not more
Under BINARY, ms01 can only turn a lattice-TRUE cell off; it never adds one (measured: 0 cells gained on all 20).
The table shows the share of lattice-TRUE cells that θ=0.1 clears on the same genome (lattice-arm genomes). These are
the cells whose class-weighted own share is ≤ 0.1.

```
            | TRUE cells (lattice)        | cleared by θ=0.1          | cleared by θ=0.5 (majority)
            | benign   attack   total     | benign   attack   %total  | %total
encoff      | 97.3K    116.2K   213.5K    | 4.9K     1.8K     3.1%    | 21.8%
encon       | 151.3K   203.7K   355.0K    | 6.2K     2.1K     2.3%    | 19.9%
```
About 75% of the cleared cells are benign-cluster cells: benign "seen once" in an attack-dominated address. The
encon advantage is not "more contested cells for the vote to act on". The vote acts on a smaller fraction of a
1.66× larger, finer memory.

### 2c. MEASURED — address resolution
- Per neuron (lattice-arm genomes), the number of distinct, non-constant input columns among its ~33.7 connections is
  **26.2 under encoff vs 30.8 under encon**.
- Under encoff, ~22% of each neuron's address bits are copies or constants. Its effective address space is ~2^26
  instead of ~2^31, which is why it writes 40% fewer distinct cells at the same neuron count.

### 2d. MEASURED — θ sweep on fixed genomes (test F1 / FPR, val-calibrated τ, lattice-arm genomes, n=5)

```
θ        | encoff F1   FPR   | encon F1   FPR
---------+-------------------+-----------------
lattice  | 90.75     4.95    | 91.25     4.12
0.02     | 91.21     5.62    | 91.75     4.95
0.05     | 91.29     4.04    | 91.92     4.25
0.10     | 90.74     3.71    | 91.84     3.86
0.15     | 90.48     4.26    | 91.71     3.87
0.20     | 90.25     4.68    | 91.45     3.92
0.30     | 89.76     7.56    | 90.74     6.02
0.50     | 88.89     8.66    | 89.08    10.13
```
(The ms01-arm genomes give the same shape: encoff peak 91.31 at θ=0.05, encon flat 91.75–91.83 over 0.02–0.10.)

**The vote does help without dedupe, at a smaller θ.** encoff has a narrow window: +0.46 to +0.54 at θ 0.02–0.05,
back to zero at 0.10. encon has a wide plateau: +0.46 to +0.67 over θ 0.02–0.15. θ=0.1 sits on the encoff cliff and
in the middle of the encon plateau. That is the "interaction".
⚠️ This picks θ by looking at TEST. The F1 at each θ is honest, but "θ=0.05 is best" is a post-hoc read on the report
partition. It is NOT a result until θ is chosen on VAL or a pre-registered production arm runs (§5).

### 2e. MEASURED — per attack category (TEST recall at margin_val_cal; same genome, lattice → ms01; lattice-arm genomes)

```
category (TEST n)     | encoff lattice → ms01   | encon lattice → ms01
----------------------+-------------------------+-----------------------
Normal (TNR) 18,500   | 95.05 → 96.29  (+1.2)   | 95.88 → 96.14  (+0.3)
Generic       9,438   | 99.23 → 99.27           | 99.23 → 99.29
Exploits      5,545   | 93.46 → 91.78  (−1.7)   | 93.87 → 94.00  (+0.1)
Fuzzers       3,000   | 40.11 → 27.29 (−12.8)   | 41.21 → 35.87  (−5.3)
DoS           2,061   | 95.16 → 93.98  (−1.2)   | 95.02 → 95.02  ( 0.0)
Reconnaissance 1,766  | 73.35 → 88.14 (+14.8)   | 73.02 → 92.53 (+19.5)
Analysis        351   | 94.30 → 90.14  (−4.2)   | 96.30 → 90.20  (−6.1)
Backdoor        283   | 98.80 → 97.03  (−1.8)   | 99.08 → 97.60  (−1.5)
Shellcode       199   | 74.97 → 82.51  (+7.5)   | 72.46 → 84.02 (+11.6)
Worms            23   | 85.22 → 91.30  (+6.1)   | 82.61 → 88.70  (+6.1)
```
The vote moves the boundary between **Fuzzers** and **Reconnaissance / Shellcode / Worms**:
- Fuzzers is the large, benign-looking attack class. Recon, Shellcode and Worms are mid-sized to rare.
- Under encoff the trade cancels: Recon +14.8 and Fuzzers −12.8, plus small losses on Exploits/DoS/Analysis.
- Under encon the Fuzzers loss is 2.4× smaller (−5.3), the Recon gain is larger (+19.5), Exploits/DoS no longer lose,
  and TNR does not drop.
- So the gain is on the **rare/mid classes (Recon, Shellcode, Worms)**, paid for by Fuzzers and Analysis.
- Under encon the payment is smaller.

### 2f. INFERRED (not measured directly)
- The cleared cells are mostly benign TRUE cells in attack-dominated addresses. Removing them lowers the benign
  score on attack rows that share an address with a stray benign row. That is what lifts Recon/Shellcode/Worms.
- Under encoff, the same addresses also pool Fuzzers rows with benign rows, because 22% of the address bits are
  copies. Clearing them shifts the val-calibrated τ (e.g. 7481: τ 0.030 → 0.214) and pushes boundary Fuzzers rows
  to benign.
- Under encon, finer addresses (≈31 effective bits) keep more Fuzzers rows in addresses the vote does not touch. So
  the same θ costs less.
- This fits the IDS-55 harness prediction (docs/ids55_family_fix_decomposition.md): "MS 0.1 only pays once neurons
  have resolution". Harness spread BIN→MS0.1 was +0.07 on duplicate columns vs +0.97 on distinct columns;
  production shows −0.04 vs +0.58.
- Visit counts per cell (how many cells get ANY conflicting write) are not exposed by the accelerator. Only the
  share bins above are measured.

## 3. unswr (flows 7501–7520) — no result yet

At 10/11/2026 11:41 UTC: 7501 (lattice-encoff r20419) is running, GA gen 82, 4.83 h elapsed, and 7502–7520 are
queued. Nothing has completed, so there is no unswr number.
The IDS-55 harness predicts ms01 LOSES here (BIN 90.07 vs MS0.1 87.28 at 3.9% attacks). This arm is the
generalisation check.

## 4. Durations and queue (measured)

```
dataset / config                                   | n  | h/flow (mean±SD)   | flows
---------------------------------------------------+----+--------------------+--------------------------
unswt IDS-56, all arms                              | 20 | 0.60±0.14 (12.0 h) | 7481-7500
  lattice-encoff / lattice-encon                    |  5 | 0.43±0.03 / 0.51±0.05
  ms01-encoff / ms01-encon                          |  5 | 0.75±0.08 / 0.72±0.02
unswr IDS-56 lattice-encoff r20419 (running)        |  1 | >4.83 (GA gen 82)  | 7501
unswr IDS-47 BINARY 2m max250 (all xo)              |  9 | 4.08±0.90          | 7224-7248
cicids IDS-47 BINARY 2m max250 (all xo)             |  9 | 6.79±2.32          | 7218-7240
cicids IDS-47 BINARY blend_neuron                   |  3 | 5.51 (3.08/7.75/5.70)
ciciot-sub IDS-47 BINARY 2m max250 (all xo)         |  9 | 16.11±3.70         | 7221-7244
ciciot-sub IDS-47 BINARY blend_cluster              |  3 | 15.91 (12.99/15.00/19.73)
cicids IDS-52 QUAD calibfull / IDS-48c calibsub     |  1 | 4.49 / 7.19        | 7380 / 7381
ciciot-sub IDS-47 QUAD 1m                           |  3 | 16.57±3.09         | 7241/7344/7347
```

**Remaining unswr IDS-56 (19 flows + the end of 7501), estimate.**
- lattice ≈ 5 h × 9 = 45 h.
- ms01 ≈ 1.4–1.75 × lattice (the unswt ratio) ≈ 7.5 h × 10 = 75 h.
- Total **≈ 120–130 h**, so done ≈ **15–16/11/2026**. This is an inference from unswt ratios; the only unswr data
  point is 7501.

**The 79 held flows (paused, resumed in id order by `logs/ids/ids56_after_round2c.sh` once IDS-56 is terminal),
estimate from the rows above.**
- IDS-52 cicids QUAD 9 × ~5.2 h = 47 h
- IDS-48c 4 × 7.19 h = 29 h
- IDS-52 unswt 40 × 0.47 h = 19 h
- IDS-52 unswr 16 × ~4.6 h = 74 h
- IDS-55 ciciot QUAD 10 × 16.6 h = 166 h
- Total **≈ 335 h ≈ 14 days** → ≈ 30/11/2026.

⚠️ **20 of the held flows duplicate IDS-56 lattice arms exactly.** The params are identical except
`ids_receptive_fields` absent vs `"off"`, and the default is off.
- IDS52-unswt 7390–7399 = IDS-56 7481/82, 7485/86, … (r20419–23, encoff/encon).
- IDS52-unswr 7431/32/34/35/37/38/40/41/43/44 = IDS-56 7501/02, 7505/06, … (same seeds).
- lattice is deterministic: IDS-56 7481 reproduced the IDS-55 `off` arm to the last digit.
- Cancelling them saves ≈ 4.7 h (unswt) + ≈ 46 h (unswr). Keeping them buys only a determinism re-check.
  **Luiz's call. Nothing was touched.**

## 5. Proposals (for Luiz — nothing queued)

**P1. Carry-over cohort, CICIDS2017 + CIC-IoT-2023 (minimal, n=5, `_3way`, budget-matched).**
Two arms per dataset, both `encon`: `lattice` vs `ms01` (θ=0.1, as tested). The interaction says ms01 is only
expected to pay with dedupe on, and encon is the encoding we would adopt. Seeds r20419–r20423, paired.

- cicids: clone **7219** (IDS47 cicids CE20-zs BINARY 2m max250 blend_neuron, random_3way, 96b, max_bits 34).
- ciciot: clone **7223** (IDS47 ciciot-sub B15AC-desir BINARY 2m max250 blend_cluster, random_3way, 96b, max_bits 100;
  blend_cluster = Luiz's 07/11 ciciot BINARY choice).
- Overrides (both datasets): `memory_mode=BINARY`, `ids_single_cluster=false`, `ids_receptive_fields=off`,
  `neuron_sample_rate=0.25`, `max_neurons=250`, `ids_encoding_dedupe=on`, `ids_feature_ranking=train_only`
  (IDS-53: cite train_only going forward; both arms share it, so the A/B is clean), plus the rule:
  `{ids_write_rule: lattice, ids_vote_share: null}` vs `{ids_write_rule: vote, ids_vote_share: 0.1}`.
- Budget: `ga_generations=250`, `patience=5`, `population_size=50` (inherited, identical in both arms). Experiments array
  = the standard 2-phase IDS pair (Grid Search + GA Neurons).
- Report: `test_summaries` best_f1 `margin_val_cal`, paired by seed, plus the low-FPR Pareto zone and the
  per-category probe (§2e).
- Cost estimate. Measured lattice base × the measured unswt encon vote ratio of 1.41. Dedupe may shorten it, since
  ~half of the cicids/ciciot bits are redundant (IDS-52).
  - cicids: 5 × (5.5 + 7.8) h ≈ **66 h**.
  - ciciot: 5 × (15.9 + 22.4) h ≈ **192 h**.
  - Total **≈ 258 h ≈ 10.7 days** for 20 flows.
- Placement:
  - New flows get ids > 7520. FIFO puts them **after** the remaining unswr IDS-56 (~5 days) **and after** the
    79 held flows (~14 days) once the chain resumes those.
  - Start ≈ 30/11, end ≈ 10–11/12/2026.
  - Running it before the backlog means keeping the backlog paused, i.e. a change to the chain script. That is a
    Luiz decision.
- Optional third arm `lattice-encoff` (today's production encoding) adds ≈ 28 h (cicids) + ≈ 80 h (ciciot). It only re-measures dedupe,
  which IDS-52 already covers on cicids (QUAD) in the held queue.

**P2. Cheap θ confirmation on unswt (production, pre-registered, ~7.5 h).** `ms005` (θ=0.05) × {encoff, encon},
r20419–23, clone the IDS-56 unswt arms. This tests §2d without choosing θ on TEST.
Pre-register: adopt θ=0.05 over 0.1 only if it wins F1 paired ≥4/5 without an FPR loss. If encoff-θ0.05 ≈ encon-θ0.1,
the interaction reads "θ must scale with address resolution", not "the vote needs dedupe".

**P3. Per-category scoring in the worker.** The per-category numbers above came from TEST-subset caches. The
two-cluster export returns only cluster 0's score (`score_examples_binary` → `s[0]`). A per-row K-score export
(`export_trained` → all clusters) would make per-category F1 a one-call readout. This is a Rust change and needs a
worker swap, so it is not urgent.
