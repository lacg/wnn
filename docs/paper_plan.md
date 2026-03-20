# Paper Plan: WNN for Network Intrusion Detection

Target: **RAID 2026** (deadline April 16, 2026)
Status: Planning phase

## Title Options

1. "RAM-Based Weightless Neural Networks with Evolved Connectivity for Network Intrusion Detection"
2. "Sub-Kilobyte Intrusion Detection: Evolutionary Optimization of Weightless Neural Networks on UNSW-NB15"
3. "Evolved Feature Selection in Weightless Neural Networks for FPGA-Deployable Network Intrusion Detection"

## Key Contributions

- **C1**: First WNN IDS evaluated on standard temporal split (prior work uses random splits → inflated 98-99%)
- **C2**: GA/TS-optimized connectivity as automated feature selection (biggest single-phase accuracy gain)
- **C3**: Competitive accuracy (~87-90% F1) with sub-1KB models (vs MB for BiLSTM/RF)
- **C4**: Rigorous 8-mode threshold evaluation protocol (identifying calibration gap)
- **C5**: Rust+Metal GPU-accelerated optimization system (practical contribution)

## Paper Structure

1. **Introduction** (1.5p) — Problem, gap, contribution, headline result
2. **Background** (1.5p) — RAM neurons, QUAD_WEIGHTED, thermometer encoding, UNSW-NB15
3. **Architecture** (2p) — Single-cluster discriminator, connectivity optimization, phased pipeline, fitness function
4. **Evaluation Methodology** (1.5p) — Standard split, threshold protocol, 34-run stats, preprocessing immunity
5. **Experimental Results** (3p) — Phase progression, main results, baselines comparison, feature selection, model size
6. **Discussion** (1p) — Connectivity as differentiator, threshold sensitivity, FPGA deployment, limitations
7. **Related Work** (1p) — WNN IDS, WNN arch, ML IDS, FPGA IDS
8. **Conclusion** (0.5p) — Summary, future work (multi-class, online learning, cross-dataset)

## Tables (7-8)

- T1: UNSW-NB15 dataset summary (features, splits, class distribution)
- T2: Architecture comparison (WNN vs FWIW vs BiLSTM)
- T3: Phase-by-phase optimization progression
- T4: Main results — 34-run mean±std by threshold mode (top20-CE4F3R3, baseline-mb28)
- T5: Comparison with literature baselines (RF, XGBoost, SVM, DNN, BiLSTM, FWIW)
- T6: Feature selection impact (all-42 vs top-20)
- T7: Model size comparison (bytes vs KB vs MB)
- T8: Random split results (if completed — enables FWIW comparison)

## Figures (4-5)

- F1: Architecture diagram (input → encoding → RAM neurons → threshold → decision)
- F2: Phase progression bar chart (staircase improvement)
- F3: Threshold sensitivity box-and-whisker (34 runs × 8 modes)
- F4: FPR-vs-F1 scatter plot (Pareto frontier, color-coded by threshold)
- F5: Connectivity heatmap (which input bits GA selects → feature importance)

## Task Breakdown

### Phase 0: Data Collection (NOW — March 22)
- [x] 34 top20-CE4F3R3 flows completed
- [x] 34 baseline-mb28-kf5x5 flows completed
- [x] 2 4P and 2 10P flows completed
- [x] 34 random-split flows queued
- [x] F434/F435 leaderboard-seeded flows queued (prioritized)
- [ ] Queue 68 more PUB-top20-CE4F3R3 (→100 total)
- [ ] Queue 62 more 10P and 62 more 4P
- [ ] Complete remaining queued flows

### Phase 1: Analysis Scripts (March 22-23)
- [ ] 1.1: Trimmed-mean extraction script from validation_summaries
- [ ] 1.2: Connectivity heatmap (which input bits selected → feature names)
- [ ] 1.3: Model size computation (neurons × bits × 2 bits/cell → bytes)
- [ ] 1.4: Phase progression tables from per-phase validation data
- [ ] 1.5: RF/XGBoost baselines on standard split (collect from Week 16)

### Phase 2: Paper Writing — Data-Independent Sections (March 23-25)
- [ ] 2.1: Introduction (needs headline numbers)
- [ ] 2.2: Background section
- [ ] 2.3: Architecture section
- [ ] 2.4: Evaluation Methodology section
- [ ] 2.5: Related Work section
- [ ] 2.6: LaTeX project setup (ACM format for RAID)

### Phase 3: Results and Figures (March 25-27)
- [ ] 3.1: Main results tables (T4, T5)
- [ ] 3.2: Architecture diagram (F1)
- [ ] 3.3: Phase progression chart (F2)
- [ ] 3.4: Threshold sensitivity plot (F3)
- [ ] 3.5: Comparison scatter plot (F4)
- [ ] 3.6: Experimental Results section
- [ ] 3.7: Discussion section

### Phase 4: Polish and Submit (March 27 — April 10)
- [ ] 4.1: Conclusion
- [ ] 4.2: Internal review — all numbers match, claims supported
- [ ] 4.3: Check page limit (RAID: 18 pages ACM format)
- [ ] 4.4: Supplementary materials (code link, HuggingFace dataset)
- [ ] 4.5: Final proofread and submission

### Phase 5: Stretch Goals
- [ ] 5.1: Complete random-split 34-run batch (FWIW comparison)
- [ ] 5.2: 4P and 10P to 64 runs each (architecture comparison)
- [ ] 5.3: Multi-class hierarchical experiment
- [ ] 5.4: FPGA synthesis estimate

## Timeline

| Week | Dates | Focus |
|------|-------|-------|
| 1 | Mar 20-23 | Data + Analysis scripts |
| 2 | Mar 24-27 | Draft Background, Architecture, Methodology, Related Work |
| 3 | Mar 28-Apr 3 | Tables, figures, Results, Discussion |
| 4 | Apr 4-10 | Introduction, polish, internal review |
| 5 | Apr 11-16 | Final proofread, submit |

## Comparison Baselines

| Method | Split | F1 | FPR | Model Size | Source |
|--------|-------|-----|-----|------------|--------|
| RF | standard | ~87% | ~15% | ~50MB | Zoghi 2024 |
| XGBoost | standard | ~87% | ~14% | ~10MB | Zoghi 2024 |
| BiLSTM | standard | ~89% | ~12% | ~5MB | Abdalgawad 2024 |
| FWIW | random | 98%+ | <1% | 272B | Susskind 2023 |
| Our WNN | standard | 87-90% | 7-14% | <1KB | This work |
| Our WNN | random | TBD | TBD | <1KB | This work |

## Risks

1. **Flows don't complete**: Paper viable with current 34+34. Extras are nice-to-have.
2. **Results not strong enough**: Key claim is competitive F1 at sub-1KB, not raw F1 supremacy.
3. **Oracle threshold questioned**: Report all 8 modes prominently, use fixed_05 as honest primary.
4. **Single dataset**: Acknowledge. Standard/random split provides two protocols. Cross-dataset = future work.

## References

- Existing bibliography at `llm-optimizer/references.bib` (50+ refs)
- Week 16 blog post has detailed methodology and literature analysis
- FWIW paper: Susskind et al. 2023 (primary comparison)
- BTHOWeN: Bacellar et al. 2022 (thermometer encoding reference)
