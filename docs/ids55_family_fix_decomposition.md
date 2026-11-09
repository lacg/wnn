# IDS-55: can families be FIXED? Lever decomposition (09/11/2026, harness only, uncommitted)

Same numpy harness, cache and protocol as `ids55_why_family_loses.md`: unswt temporal_3way TRAIN 175,341 / VAL 41,166. Each neuron writes from a 25% sample of TRAIN. Threshold (binary) or per-class bias (multiclass) is tuned on VAL half A and reported on VAL half B. TEST is never loaded. Random genomes, 500 neurons, 34 bits, 3 seeds (11/12/13) unless noted. families_v2-like map (protocol merged into context), so protocol starvation is already removed. **This is a diagnostic. Do not report it as a result.** Scripts: `/tmp/ids55_fix/{fixlib,run,run2,run3,run4,diag}.py`. "addr" = distinct addresses touched in TRAIN, used as the cell budget. Youden is per neuron (P(fire|own) − P(fire|other)) on VAL-A; "sat" = share of benign neurons firing on >90% of attacks.

Cell rules (all 1 bit/cell at inference unless noted). p = own share of the visits after normalising by class frequency.
- **BIN** = today's TRUE-wins.
- **MAJ θ** (Luiz's majority bit): TRUE if p > 0.5+θ.
- **MAJt**: ternary, 2 bits/cell.
- **QUAD**: OI nudge, `oi_bin_to_cell` mirrored, 2 bits/cell.
- **MS θ** (min-share, added here): TRUE unless own is a clear minority, i.e. p > θ. MS 0 = BIN; MS 0.5 = MAJ 0.

## Binary unswt: cal F1 / AUC / benign Youden / benign sat / addr
| arm | rule | F1 | AUC | ben yd | sat | addr |
|---|---|---|---|---|---|---|
| spread 500n | BIN | 91.00±0.22 | .9712 | .445 | .01 | 373K |
| spread 253n (= cells of L0) | BIN / MS0.1 | 91.09±0.07 / 91.32±0.18 | .9700/.9725 | .459/.695 | .01/0 | 188K |
| **L0** family 0.2 | BIN | 89.95±0.73 | .9635 | .276 | .37 | 187K |
| L1 family + MAJ 0 / 0.1 / 0.2 | MAJ | 89.12±0.10 / 89.66±0.14 / 89.87±0.17 | .959/.962/.965 | .62/.58/.55 | 0 | 187K |
| L1 family + MS 0.1 | MS | 89.80±0.20 | .9632 | .586 | .01 | 187K |
| L2 family + QUAD | QUAD | 89.11±0.07 | .9588 | .622 | 0 | 187K |
| L3 width cap log2(D)+4 / +2, more neurons (n=2; MAJ 0.2: 86.35 / 85.41) | BIN | 88.98±0.35 / 88.35±0.54 | .956/.954 | .11/.09 | .68/.73 | 185K |
| L4 distinct+stratified bits (w≤34) | BIN / MS0.1 | 90.25±0.47 / 90.68±0.13 | .966/.969 | .366/.621 | .23/.01 | 363K |
| L5 per-class homes (no L4) | BIN / MS0.1 | 90.43±0.06 / 91.13±0.14 | .965/.971 | .456/.654 | .07/0 | 198K |
| **L4+L5** | BIN / **MS0.1** | 90.17±0.22 / **91.90±0.08** (FPR 4.08) | .964/.974 | .555/.697 | .03/0 | 409K |
| spread @ = cells of L4+L5 | BIN / MS0.1 | 91.12±0.10 / 91.19±0.01 | .971/.972 | .44/.70 | .02/0 | 410K |
| **spread, distinct cols @ = cells of L4+L5 (299n)** | BIN / **MS0.1** | 91.35±0.23 / **92.32±0.12** | .973/.975 | .575/.711 | 0 | 410K |
## What is root cause and what is only resolution
1. **Saturation is real, but removing it alone does not lift F1.** MAJ, QUAD and MS all take benign sat from .37 to 0 and roughly double benign Youden (.28 → .55-.62). Family F1 stays at 89.1-89.9 and AUC does not rise.
   - Why (`diag.py`, spread): under BIN the two discriminators are complementary. Benign means "never seen benign" (benign-only AUC .925); attack means "never seen attack". The difference score reaches AUC .9694.
   - Strict majority makes both discriminators encode the same winner, so they become redundant. Each one alone gets better (.961/.957), but together they drop to .960.
   - So per-neuron saturation is a symptom. The two-discriminator difference already absorbs most of it.
2. **The family-specific root cause is address collapse from REDUNDANT THERMOMETER COLUMNS.**
   - Only 161 of 288 bits are distinct and non-constant.
   - ttl_state has 16 distinct columns and 30 joint patterns; handshake has 10 and 14.
   - So a 28-in-family-bit neuron cannot have resolution, whatever the cell rule.
3. **The second cause is the WRONG HOMES for the benign discriminator.** Measured on train folds, per-class power puts benign homes at ~91% volume+context and attack homes mainly ttl_state (35%). Size-proportional allocation puts ~50% of benign neurons in ttl/handshake/timing.
4. **L4 and L5 are the levers that fix the family.** MS 0.1 only pays once neurons have resolution: +0.7 pp after L5, +1.7 pp after L4+L5, −0.15 pp on L0. Ranking: L5 > L4 > MS cells >> MAJ ≈ QUAD (negative) >> L3 (harmful: narrow neurons over-generalise, sat .68).
5. **Nothing is family-specific.** At equal cells the full family stack (91.90) beats plain spread (91.19) by +0.7 pp. Spread with the same two family-agnostic levers (distinct columns + MS 0.1) reaches 92.32 at the same cells and 92.22 at HALF the cells. So families lose −0.4 pp at equal memory even after every fix.
## Multiclass check (4 classes)
Classes: Normal / Generic (broad) / DoS+Exploits / rest (narrow: Fuzzers, Recon, Analysis, Backdoor, Shellcode, Worms). 125 neurons per class. Macro-F1, with per-class F1 as Normal/Generic/DoS/rest.

| arm | BIN | MS0.1 (or 0.2) | MAJ 0 | QUAD |
|---|---|---|---|---|
| spread 500n | 78.45±0.06 (87.8/98.1/80.4/47.5) | 78.62±0.31 (86.4/97.9/80.2/50.1) | 75.23 (79.6/97.4/79.4/44.6) | 75.16 |
| spread @ = cells of L0 | 77.86±0.24 | 78.29±0.38 | 75.50 | 75.42 |
| L0 family | 73.97±1.01 (87.5/98.3/77.9/**32.1**) | 77.99±0.24 (85.6/98.3/80.1/48.0) | 74.28 (**72.9**/98.0/79.6/46.6) | 74.24 |
| L4+L5 | 74.98±1.01 | 79.10±0.14 [0.2] (89.3/98.3/80.9/47.8) | 77.09 (81.9/…) | – |
| spread distinct @ = cells of L4+L5 | 79.59±0.27 | **80.16±0.42** (89.7/98.4/81.7/50.9) | – | – |

- **Under BIN, multiclass breaks families far harder.** The gap is −3.9 pp vs −1.1 pp in binary. The narrow class collapses (rest 32 vs 47.5) and the benign discriminator saturates (.35).
- **Luiz's strict majority fails the every-class test.** It recovers the narrow class (+14.5) but destroys the broad one (Normal 87.5 → 72.9), because broad classes lose every contested address. QUAD behaves the same way.
- **MS 0.1 passes on unswt.** Narrow class +16 pp, DoS +2 pp, Normal −1.9 pp. It closes the family gap from −3.9 to −0.3 pp, and it also helps spread (rest +2.6).
- **Best family (79.10) is still below spread-distinct at equal cells (80.16).**
## Generalisation warning: unswr random_3way, 3.9% attacks (n=2)
**MS / MAJ / QUAD all LOSE here.**

| arm | BIN | MS 0.1 |
|---|---|---|
| spread | 90.07 | 87.28 |
| family | 88.09 | 87.28 |
| L4+L5 | 88.52 | 86.86 |

- Every non-BIN arm ends on the same plateau (FPR 1.16%, AUC .995-.997 vs .9986-.9989 under BIN). That points to a block of contested benign rows that only BIN's "seen-benign" tie-break can rank.
- The families also lose here under BIN (−0.6 pp at equal cells).
- So MS is **not** a universal fix. It must be A/B'd per dataset.
## Recommendation (minimal)
1. **Stop investing in families for accuracy.** Every lever that fixes them is family-agnostic and lifts spread more. Families can at best equal spread; their memory saving is the same address collapse that costs F1. Keep `ids_rf=off`.
2. **Lever A: distinct-column connections.** This already exists as `ids_encoding_dedupe` (IDS-52), and the d1 cache is exactly my 161 distinct bits. Spread on distinct columns gains +0.3 pp BIN / +1.0 pp with MS at equal cells (binary), and +1.7 macro-F1 (multiclass), with no inference cost.
3. **Lever B: min-share cell commit.** Implement it as a generalisation of the existing `WriteRule::Vote` in ram_core `cell_mode.rs`/`train.rs`. Per visit, the tally adds +(1−θ)·w_own for an own-class row and −θ·w_other for any other row (w = the class-frequency weight vote already uses). Commit: tally > 0 → TRUE, else the untrained cell.
   - θ = 0.5 is today's `vote`, which the harness predicts loses.
   - New param `ids_vote_share` (θ, default 0.5 = today); register it in `KNOWN_PARAMS`.
   - Inference: 1 bit/cell, read path identical to BINARY, FPGA cost unchanged. Training reuses the OI counter's `net`.
4. **Production A/B before anyone adopts it.** n=5 paired, r20419-23, rf off, 500n, BINARY lattice 2m, Protocol v2.
   - 2×2 on unswt: {`ids_write_rule` lattice vs vote with θ=0.1} × {`ids_encoding_dedupe` off/on}.
   - Plus the same 2×2 on unswr (the harness predicts MS loses there) and ciciot.
   - Plus one unswt multiclass pair (BIN vs MS 0.1, dedupe on).
   - Report from `validation_summaries` best_f1 `margin_val_cal`, at equal logical cells.
   - Adopt only where θ=0.1 wins F1 without an FPR loss. Expect it to be per-dataset.
