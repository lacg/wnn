# Paper Plan: WNN for Network Intrusion Detection

Target: **RAID 2026** (deadline April 16, 2026)
Format: Springer LNCS, 20 pages excluding references
Status: Sections 1-5 drafted, flows running

## Title

**"Evolved Feature Selection in Weightless Neural Networks for FPGA-Deployable Network Intrusion Detection"**

## Key Contributions

- **C1**: Evolved connectivity as automated feature selection, GA discovers which input bits matter, intrinsic to the architecture
- **C2**: Sub-1KB models with nanosecond inference, competitive with RF/XGBoost at 50,000x smaller model size
- **C3**: Multi-protocol evaluation, temporal + random splits, 8-mode threshold calibration, 100 independent runs per config
- **C4**: Multi-dataset evaluation, UNSW-NB15 (temporal + random), CICIDS2017 (random), CIC-IoT-2023 (random, stretch)
- **C5**: FPGA synthesis + simulation, resource utilization and latency analysis for edge deployment

## Scope Decisions (revised March 21)

- **2-phase pipeline only**: grid search + GA neurons. Full 4-phase (+ GA bits + GA connections) and 10-phase (+ Lamarckian) described briefly in Architecture, detailed evaluation in future work.
- **Random splits for CICIDS2017 and CIC-IoT-2023**: matches literature standard. Temporal splits for these datasets reserved for follow-up paper.
- **UNSW-NB15 gets both splits**: temporal (our contribution) + random (FWIW comparison).
- **100 runs per config**: gold standard for statistical validity.

## Paper Structure

1. **Introduction** (1.5p) — Speed+tiny need, evolved WNN, multi-dataset evaluation
2. **Background** (1.5p) — RAM neurons, QUAD_WEIGHTED memory, thermometer encoding, datasets
3. **Related Work** (1p) — WNN IDS (FWIW), ML IDS, FPGA IDS
4. **Architecture** (2p)
   - 4.1 Single-cluster binary discriminator
   - 4.2 Two-phase optimization pipeline (grid search + GA neurons)
   - 4.3 Fitness function (weighted harmonic rank of CE, F1, FPR)
   - 4.4 GPU-accelerated optimization (Rust+Metal)
   - *Brief mention: framework supports additional phases (bits, connections, Lamarckian) for future work*
5. **Evaluation Methodology** (1.5p) — Temporal vs random splits, 8-mode threshold protocol, statistical protocol (100 runs)
6. **Experimental Results** (3p) — UNSW-NB15 (temporal + random), CICIDS2017 (random), baselines comparison, model size
7. **Discussion** (1p) — Connectivity as feature selection, threshold sensitivity, FPGA deployment analysis, limitations
8. **Conclusion** (0.5p) — Summary, future work (additional phases, temporal splits for CICIDS, multi-class)

## Introduction Framing

Flow: *practical need, our approach, evaluation*

1. **Hook**: Edge IDS needs sub-microsecond inference with tiny models
2. **WNN as solution**: RAM neurons = O(1) lookup, sub-1KB model, but connectivity is the key
3. **Our approach**: Evolutionary optimization discovers optimal network parameters (count, address width, connectivity)
4. **Evaluation**: Multi-dataset (UNSW-NB15, CICIDS2017), temporal + random splits, 8 threshold modes, 100 runs per config
5. **Result**: Competitive with RF/XGBoost at 50,000x smaller model size

Tone: positive, factual, no superfluous adjectives. Let the work speak for itself.

## Tables (7-8)

- T1: Datasets summary (UNSW-NB15, CICIDS2017, CIC-IoT-2023: features, splits, sizes, class distribution)
- T2: Architecture comparison (Our WNN vs FWIW vs BiLSTM vs RF)
- T3: UNSW-NB15 temporal results, 100-run mean±std by threshold mode
- T4: UNSW-NB15 random results, 100-run mean±std by threshold mode
- T5: CICIDS2017 random results, 100-run mean±std by threshold mode
- T6: Comparison with literature baselines (all datasets)
- T7: Model size comparison (bytes vs KB vs MB)
- T8: FPGA resource utilization (LUTs, BRAMs, Fmax, latency)

## Figures (4-5)

- F1: RAM neuron architecture diagram (done! TikZ, 3x3 input, 2 neurons, memory tables)
- F2: Threshold sensitivity box-and-whisker (100 runs x 8 modes)
- F3: FPR-vs-F1 scatter plot (Pareto frontier, color-coded by threshold)
- F4: Connectivity heatmap (which input bits GA selects, maps to feature names)
- F5: FPGA block diagram (optional, if space permits)

## Compute Plan (100 runs each, 2-phase)

| Phase | What | Flows | Time | Done by |
|---|---|---:|---:|---|
| 1 | Current UNSW queue (random + top20-8b + misc) | 46 | 2.3 days | Mar 23 |
| 2 | UNSW temporal (66 more to reach 100) | 66 | 0.5 days | Mar 24 |
| 3 | UNSW random (67 more to reach 100) | 67 | 5.6 days | Mar 29 |
| 4 | CICIDS2017 random (99 more to reach 100) | 99 | 8.2 days | Apr 7 |
| 5 | CIC-IoT-2023 random (stretch, 100) | 100 | 6.2 days | Apr 13 |

## Task Breakdown

### Phase 0: Data Collection (in progress)
- [x] 34 top20-CE4F3R3 flows completed (UNSW-NB15 temporal)
- [x] 34 baseline-mb28-kf5x5 flows completed (UNSW-NB15 temporal)
- [x] 34 random-split UNSW-NB15 flows queued (running)
- [x] CICIDS2017 HuggingFace dataset created
- [x] CIC-IoT-2023 HuggingFace dataset created
- [x] F505 CICIDS2017 temporal test flow completed (poor results, expected)
- [x] F506 CICIDS2017 random test flow completed (99.2% F1!)
- [ ] Queue 66 more UNSW temporal (after current queue clears)
- [ ] Queue 67 more UNSW random (after current queue clears)
- [ ] Queue 99 CICIDS2017 random (after UNSW done)
- [ ] Queue 100 CIC-IoT-2023 random (stretch, after CICIDS)

### Phase 1: Analysis Scripts (March 22-24)
- [ ] 1.1: Trimmed-mean extraction script from validation_summaries
- [ ] 1.2: Connectivity heatmap (which input bits selected, maps to feature names)
- [ ] 1.3: Model size computation (neurons x bits x 2 bits/cell, bytes)
- [ ] 1.4: RF/XGBoost baselines on both splits (UNSW-NB15)

### Phase 2: Paper Writing (March 21-29, parallel with compute)
- [x] 2.1: Background section (drafted)
- [x] 2.2: Related Work section (drafted)
- [x] 2.3: Architecture section (drafted, needs revision for 2-phase focus)
- [x] 2.4: Evaluation Methodology section (drafted)
- [x] 2.5: Introduction (drafted, needs headline numbers)
- [x] 2.6: RAM neuron figure (done, TikZ)
- [ ] 2.7: Revise Architecture for 2-phase focus, move 4/10-phase to future work
- [ ] 2.8: Update Introduction with final numbers

### Phase 2.5: FPGA Synthesis (March 25-April 3)
- [ ] 2.5.1: Adapt FWIW RTL generator for our genome format
- [ ] 2.5.2: Vivado synthesis targeting Zynq Z-7020 and Alveo U25
- [ ] 2.5.3: Extract resource utilization (LUTs, BRAMs, Fmax)
- [ ] 2.5.4: Functional simulation with test vectors
- [ ] 2.5.5: Write FPGA section for Discussion

### Phase 3: Results and Figures (March 29-April 7)
- [ ] 3.1: UNSW-NB15 temporal results table (100 runs)
- [ ] 3.2: UNSW-NB15 random results table (100 runs)
- [ ] 3.3: CICIDS2017 random results table (100 runs, when available)
- [ ] 3.4: Threshold sensitivity plot
- [ ] 3.5: Comparison with baselines table
- [ ] 3.6: Model size analysis
- [ ] 3.7: Write Experimental Results section
- [ ] 3.8: Write Discussion section

### Phase 4: Polish and Submit (April 7-16)
- [ ] 4.1: Conclusion + future work
- [ ] 4.2: Internal review, all numbers match claims
- [ ] 4.3: Check page limit (LNCS: 20 pages excluding refs)
- [ ] 4.4: Supplementary materials (code, HuggingFace datasets)
- [ ] 4.5: Final proofread
- [ ] 4.6: Submit by April 16

### Phase 4.5: Pareto Deployment Classes (post-RAID, in progress 2026-05-31)
- [x] 4.5.1: Measure RF and XGBoost on UNSW-NB15 temporal at top-20 16-bit thermo (`scripts/verify_unsw_temporal_baselines.py`)
- [x] 4.5.2: Verify n=30 16b-Wb cohort (flows 2897, 2911, 2925, 2978-3004)
- [x] 4.5.3: Update baseline tables in `paper_plan.md` and `ids_goals.md`
- [x] 4.5.4: Scaffold new section `docs/paper_pareto_classes.md` (Server / Edge / FPR-extreme)
- [ ] 4.5.5: Write section opener (3-5 sentences, see TODO inside `paper_pareto_classes.md`)
- [ ] 4.5.6: Extend `scripts/analyze_xds_final.py` with `--max-neurons N` filter
- [ ] 4.5.7: Inline the new section into the paper draft proper

### Phase 5: Post-Submission / Follow-up Paper
- [ ] 5.1: FPGA hardware demo on Zynq or AWS F1
- [ ] 5.2: CIC-IoT-2023 100 runs (if not done for RAID)
- [ ] 5.3: 4-phase and 10-phase evaluation (100 runs each)
- [ ] 5.4: Temporal splits for CICIDS2017 and CIC-IoT-2023
- [ ] 5.5: Multi-class hierarchical experiment
- [ ] 5.6: Leaderboard seeding evaluation
- [ ] 5.7: Prepare conference presentation

## Timeline

| Week | Dates | Compute | Paper |
|------|-------|---------|-------|
| 1 | Mar 21-24 | UNSW queue finishing | Sections 1-5 drafted, figure done |
| 2 | Mar 24-29 | UNSW temporal 100 done, random running | Add UNSW temporal results, polish sections |
| 3 | Mar 29-Apr 3 | UNSW random 100 done, queue CICIDS | Add UNSW random results, FPGA synthesis, comparison tables |
| 4 | Apr 3-7 | CICIDS running | Discussion, conclusion, internal review |
| 5 | Apr 7-13 | CICIDS 100 done, IoT (stretch) | Add CICIDS results, final polish |
| 6 | Apr 13-16 | IoT stretch | Final proofread, submit |

## FPGA Plan

### For the Paper (Approach A+B)
- Synthesis + simulation, no hardware needed
- Adapt FWIW's open-source RTL generator for our genome
- Target: Zynq Z-7020 (edge) and Alveo U25 (data center)
- Report: LUTs, BRAMs, Fmax, clock cycles per classification, throughput (Gbps)
- Functional simulation: verify WNN classification matches Python

### For the Conference (Approach C, stretch)
- Live hardware demo on Zynq Pynq-Z2 (~$200) or AWS F1 ($1.65/hr)

## Primary Competitor: FWIW (Susskind et al., FPGA 2023)

Authors: Susskind, Arora, **Bacellar, Dutra, Miranda, Lima, Franca** (UFRJ) + John (UT Austin)

| Aspect | FWIW | Our Work |
|---|---|---|
| Connectivity | Random | **Evolved (GA)** |
| Evaluation | Random split only | **Temporal + random** |
| Datasets | UNSW-NB15 only | **UNSW-NB15 + CICIDS2017 (+IoT stretch)** |
| Threshold | Single mode | **8 modes** |
| Memory | Binary | **QUAD_WEIGHTED** |
| Model size | 272B | <1KB (similar) |
| FPGA | Yes (Vivado) | **Synthesis + simulation** |
| Statistical runs | 1 | **100** |

## Comparison Baselines

UNSW-NB15 temporal RF/XGBoost numbers updated 2026-05-31 with measured values from
`scripts/verify_unsw_temporal_baselines.py` (top-20 features, 16-bit thermometer, same
preprocessing as our WNN). Prior "~87% / ~12%" entries were estimates and underestimated
FPR by ~2x; the measured FPR (25-29%) is the realistic temporal-split baseline.

| Method | Dataset | Split | F1 | FPR | Model Size | Source |
|---|---|---|---|---|---|---|
| RF | UNSW-NB15 | temporal | 86.05% | 25.41% | 138 MB | Measured, 16b thermo, 100 trees max_depth=None |
| XGBoost | UNSW-NB15 | temporal | 84.89% | 28.57% | 0.27 MB | Measured, 16b thermo, 100 trees max_depth=6 |
| BiLSTM | UNSW-NB15 | temporal | ~89% | ~8% | ~5MB | Abdalgawad 2024 |
| FWIW | UNSW-NB15 | random | 98%+ | <1% | 272B | Susskind 2023 |
| RF | CICIDS2017 | random | 99.9% | ~1% | ~50MB | Various |
| **Our WNN, server class** | UNSW-NB15 | temporal | **88.87±0.23%** | **8.77±1.14%** | ~1.3 MB | **This work, n=30 seeds, 16b-Wb, 500n34b OI** |
| **Our WNN, edge/FPGA**    | UNSW-NB15 | temporal | **87.88%**      | **6.94%**      | **~2 KB** | **This work, 5-neuron Pareto pick (flow 2994 r56926)** |
| **Our WNN, FPR-extreme**  | UNSW-NB15 | temporal | 87.70%          | **4.63%**      | ~5 KB     | This work, 16-neuron Pareto pick |
| Our WNN | UNSW-NB15 | random | ~94% | <1% | <1KB | This work (8 runs) |
| Our WNN | CICIDS2017 | random | 99.2% | <0.5% | <1KB | This work (1 run) |

### Deltas vs measured UNSW-NB15 temporal baselines

| Comparison | F1 | FPR | Size |
|---|---|---|---|
| WNN server (309n) vs RF | +2.82 pp | **−16.64 pp** | **138× smaller** |
| WNN server (309n) vs XGBoost | +3.98 pp | **−19.80 pp** | 5× larger (XGB is small, but 4× higher FPR) |
| WNN edge (5n, ~2 KB) vs RF | +1.83 pp | **−18.47 pp** | **~138,000× smaller** |
| WNN edge (5n, ~2 KB) vs XGBoost | +2.99 pp | **−21.63 pp** | **~135× smaller** |
| WNN FPR-extreme (16n) vs RF | +1.65 pp | **−20.78 pp** | ~55,000× smaller |

### Note on F1 ceiling

No configuration we have evaluated crosses 90% F1 on UNSW-NB15 temporal binary at
top-20 features (8b-Wc peak 89.50, 16b-Wb peak 89.34, 96b-Wc peak 89.69). This
appears to be a real ceiling for the temporal split, not a WNN-specific limit:
RF and XGBoost also plateau at 84-86% F1 on the same partition. The paper's
headline differentiator is therefore FPR (−16 to −21 pp vs measured baselines)
and model size, not F1.

## Risks

1. **Random split flows too slow**: 2h/flow, 100 flows = 8 days per dataset. Mitigated by parallel paper writing.
2. **Results not strong enough**: CICIDS random already at 99.2% F1. UNSW temporal at 87% matches RF/XGBoost.
3. **Oracle threshold questioned**: Report all 8 modes prominently.
4. **FPGA synthesis fails**: Discussion section can present theoretical analysis instead.
5. **CIC-IoT-2023 not ready**: Paper viable with UNSW + CICIDS only (2 datasets).

## Writing Style Rules

- No em-dashes (AI signal, not author's style). Use commas, parentheses, colons, periods.
- No superfluous adjectives ("rigorous", "comprehensive", "significant"). State what it is, not how great it is.
- Positive framing of prior work. Say what they achieved, not what they missed.
- Let the methodology and results speak for themselves.

## References

- FWIW: Susskind et al., FPGA 2023 — [GitHub](https://github.com/ZSusskind/FWIW)
- BTHOWeN: Bacellar et al., 2022 — thermometer encoding
- Existing bibliography at `llm-optimizer/references.bib` (50+ refs)
- Week 16-18 blog posts have detailed methodology and analysis
