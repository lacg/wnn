# S&P 2027 (Montreal) Submission Plan — post-RAID-desk-reject

Locked with Luiz 11/07/2026. **Supersedes the USENIX Sec '27 cycle-1 plan** (18/08 abstract
gate consciously skipped to fit the multiclass extension).

## Venue + hard dates (verified 11/07/2026)

| Gate | Date |
|------|------|
| Abstract registration | **10/11/2026** |
| Paper deadline (Cycle 2) | **17/11/2026** |
| Notification | 05/03/2027 |
| Conference (Montreal) | 18/05/2027 |

Fallbacks if rejected: DIMVA 2027 (~Feb deadline, Europe), ACSAC 2027 (~May deadline, Honolulu).
Note: S&P uses the **IEEE conference template** — the paper is currently LNCS (`llncs.cls`);
a template port is required (Phase 3).

## Decisions locked (11/07/2026)

1. **Venue**: S&P 2027 Cycle 2 (Montreal), not USENIX c1.
2. **Scope**: binary results remain the paper's spine; **multiclass is a major new section**.
   The "reframe multiclass-first?" decision is DEFERRED until multiclass results exist.
3. **Random-search baseline** (Review C 1.1) is in scope.
4. **Traffic-replay demo** (Review A real-world concern) is in scope.
5. **46M binary cohort continues** (4299/4300/4401-4403 NOT stopped) — it drains itself
   ~18-20/07 and its results replace the bug-contaminated RAID numbers + provide the
   46M-scale headline. Multiclass can't use the compute yet anyway (implementation is the
   critical path, not compute).

## Why RAID #226 died, and what the reviews actually said

- **Desk-reject cause: hallucinated references ONLY** (Review B + admin). Two confirmed
  fabrications: `andronic2023polylut` (five fictional "Sherborne" authors; real = Andronic &
  Constantinides) and `mdpi2024enhanced` (wrong authors; real = More, Idrissi, Mahmoud, Asyhari).
- Review A (some familiarity): **weak accept** on the binary paper. Asks: multiclass, real-world/edge eval.
- Review C (knowledgeable, reject): all issues are specification/statistics — GA underspecified,
  GA-vs-random-search motivation missing, fitness-weight tuning provenance, "independent run"
  undefined, no significance tests, normal-approx CIs invalid for bounded metrics, missing
  Di Mauro et al. TNSM 2020 (prior WNN-for-IDS!), old WNN refs, Zoghi & Serpen claim
  unverifiable, "absence of temporal evaluation" too strong (INSOMNIA exists), QSR granularity
  question.

## Feedback → fix matrix

| ID | Issue | Fix | Effort | Phase |
|----|-------|-----|--------|-------|
| T0 | Hallucinated refs (desk-reject) | Full 24-entry bib audit, every field verified vs. real sources; then a claim-citation audit of every `\cite` in main.tex | 2-3h (agents) | 0 |
| C1.2 | GA underspecified | Complete GA spec: genome encoding, tournament selection on harmonic rank, crossover/mutation operators, 20% elitism + replacement, patience termination; algorithm box + hyperparameter table (all in `ArchitectureGAStrategy` code) | 3-4h | 0 |
| C1.1 | Why GA? | Equal-budget random-search baseline, UNSW-temporal, n=20-30 | 4-6h code + 2-5d compute | 1 |
| C1.3 | Fitness-weight provenance | Honest audit of Wa/Wb/Wc selection (probe cohorts vs report sets) + clean methodology statement | 3-4h | 0 |
| C2.1 | "Independent run" undefined | Define precisely: fresh GA seed + fresh connectivity init, fixed 80/20 split, K-fold within the 80%, held-out untouched during search | 1h | 0 |
| C2.2 | No significance tests | Mann-Whitney U pairwise across configs/datasets on existing DB data | 2-3h | 0 |
| C2.3 | CI normality | Bootstrap (BCa) or Wilson intervals for F1/FPR | folded in | 0 |
| C3 | Di Mauro TNSM 2020 missing | Read + cite + reposition contribution: they ran vanilla WiSARD; we do evolutionary connectivity optimization, temporal splits, QUAD cells, FPGA | 2-3h | 0 |
| Cm1 | Old WNN refs | Add DWN (Bacellar 2024), ULEEN 2023, recent survey | 1-2h | 0 |
| Cm2 | Zoghi & Serpen claim | Verify what the paper actually supports; fix or reattribute | 1h | 0 |
| Cm3 | "absence of temporal eval" | Soften + cite INSOMNIA + concept-drift NIDS literature | 1h | 0 |
| Cm4 | Why 4 states? | Prose: 2-bit = BRAM sweet spot (3-bit doubles memory for marginal granularity); optional 8-state ablation if compute allows | 1h | 0 |
| A1 | Multiclass | Full multiclass evaluation section (design → full-stack impl → cohorts) | ~3-5 weeks incl. compute | 1-2 |
| A2 | Real-world/edge | Traffic-replay demo: pcap replay → live feature extraction → WNN inference, end-to-end latency/throughput; FPGA synthesis numbers already exist | 1-2 weeks | 2 |

## Phases

### Phase 0 — 11/07 → ~20/07 (46M cohort drains itself; writing/analysis only)
1. Bib audit (3 agents running since 11/07) → corrected `references.bib`.
2. Claim-citation audit: every `\cite` in main.tex checked that the source supports the claim.
3. GA specification section + pseudocode + hyperparameter table.
4. Stats upgrade scripts (Mann-Whitney U, bootstrap/Wilson CIs) on existing cohort data.
5. Di Mauro positioning; modern WNN refs; INSOMNIA/concept-drift; soften claims; QSR prose.
6. Fitness-weight methodology audit + writeup.

### Phase 1 — ~20/07 → ~15/08 (post-drain; interleaves with controller chain)
1. 46M cohort report → `docs/ids_results.md`.
2. Random-search baseline implementation + runs.
2b. **Fresh-seed confirmation cohorts → these become the HEADLINE cohorts**
   (weights + width frozen a priori, seeds disjoint from all probe rounds):
   UNSW-temporal n=30 (~2.5h at ~5 min/run), CICIDS as compute allows. Closes
   the C1.3 disclosure — the fitness-weight audit (11/07) found probe runs
   (~10%) retained inside reported cohorts (width+weight cell chosen on
   held-out probe results). Reporting confirmation cohorts turns the
   disclosure into "selection and reporting use disjoint seeds", one sentence.
   Fallback if compute is tight: exclude probe seeds from reported aggregates
   (n=30→27, free).
2c. **Evaluation Protocol v2 (`_3way` for BINARY AND multiclass)** — decided
   11/07 with Luiz: the confirmation cohorts and all multiclass runs use the
   80/10/10 splits with the worker NO LONGER merging test+val. Thresholds
   (val_cal/Platt/beta) calibrate on val; selection peeks val; test is
   report-only. Also fixes the latent val_cal-calibrates-on-report-set issue
   in the binary results. Worker change ~3-6h, shared by both tracks.
3. **Multiclass design doc: DONE 11/07 → `docs/MULTICLASS_DESIGN.md`**
   (K clusters via existing ClusterGenome/B5; argmax + benign-margin cascade
   decode; macro-F1/benign-FPR; frozen fitness weights; UNSW-temporal n=100
   first; Neto 8-class direct comparison; full-stack change list ~1-1.5 wk).
3. **Multiclass design doc** (the critical path): decode (argmax vs benign-first cascade),
   multiclass thresholds/calibration (binary's 7 modes have no K-class analogue — new design),
   metrics (macro-F1, per-class recall, benign-FPR) through Rust → worker → dashboard,
   label mapping (UNSW 10-class, CICIDS ~15, CIC-IoT 34→8 groups), compute budget.
   Substrate: Option-B B5 K-class GPU training already exists.
4. Replay demo design (device: M4 Mac baseline; FPGA numbers from synthesis).

### Phase 2 — ~15/08 → ~30/09 (multiclass build + cohorts)
1. Multiclass full-stack implementation (1-2 weeks).
2. Screening (n≈5/dataset) → config lock → cohorts, tiered n: UNSW-temporal n=100 first,
   CIC-IoT subsample + CICIDS n=50+ as compute allows. 46M stays binary (scale demo).
3. Replay demo build + measurements (CPU-light, parallel).

### Phase 3 — October (analysis + rewrite)
1. Multiclass reports + significance tests + tables.
2. Paper restructure: contributions list, related work (Di Mauro positioning), GA spec,
   statistical protocol, multiclass section, real-world section. **Port LNCS → IEEE template.**
3. Decide: multiclass as section vs full reframe (deferred decision — judge by results).
4. Compile before every commit (`latexmk`).

### Phase 4 — 01/11 → 17/11 (freeze + submit)
1. Final full reference re-audit + adversarial internal review (simulate Review C).
2. Abstract registration **10/11**; submit **17/11**.

## Compute plan / contention

Machine shared with the controller-teacher campaign (PID-full + seed-pairs gated on the 46M
drain, then Phase-3 hybrid runs). HARD RULE: max 2 heavy runners. Sequence: 46M drain (→~20/07)
→ controller chain + RS baseline interleave (Aug) → multiclass screening/cohorts dominate Sep.

## Risks

1. **Multiclass threshold/decode design** is genuinely new research — the benign-first cascade
   candidate overlaps the planned follow-up paper; use the cascade as decode mechanism only,
   keep the deeper cascade contributions for paper #2.
2. **Compute**: multiclass n=100 × 3 datasets may not fit → tiered n is the fallback.
3. **S&P bar** (~14-19% accept): the paper must be substantially stronger — it is (clean
   methodology, 46M scale, multiclass, RS baseline, statistical rigor, verified refs).
4. **Integrity**: zero unverified citations or claims — every entry web-verified, twice
   (Phase 0 + Phase 4).
