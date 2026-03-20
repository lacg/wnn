# Paper Plan: WNN for Network Intrusion Detection

Target: **RAID 2026** (deadline April 16, 2026)
Format: Springer LNCS, 20 pages excluding references
Status: Skeleton written, data collection in progress

## Title

**"Evolved Feature Selection in Weightless Neural Networks for FPGA-Deployable Network Intrusion Detection"**

## Key Contributions

- **C1**: Evolved connectivity as automated feature selection — GA/TS discovers which input bits matter, intrinsic to the architecture
- **C2**: Sub-1KB models with nanosecond inference — competitive with RF/XGBoost at 50,000x smaller model size
- **C3**: Rigorous multi-protocol evaluation — temporal + random splits, 8-mode threshold calibration, 100+ independent runs
- **C4**: Three-dataset evaluation — UNSW-NB15, CICIDS2017, CIC-IoT-2023 (generalizability)
- **C5**: FPGA synthesis — resource utilization and latency analysis for edge deployment

## Paper Structure

1. **Introduction** (1.5p) — Speed+tiny need → evolved WNN → rigorous evaluation
2. **Background** (1.5p) — RAM neurons, QUAD_WEIGHTED memory, thermometer encoding, datasets
3. **Related Work** (1p) — WNN IDS (FWIW), ML IDS, FPGA IDS → identifies gaps
4. **Architecture** (2.5p)
   - 4.1 Single-cluster binary discriminator
   - 4.2 Phased evolutionary optimization (grid search → GA neurons → GA bits → GA connections)
   - 4.3 Lamarckian evolution (neurogenesis, synaptogenesis, axonogenesis) — the 10-phase pipeline
   - 4.4 Fitness function (weighted harmonic rank of CE, F1, FPR)
   - 4.5 Rust+Metal GPU acceleration
5. **Evaluation Methodology** (1.5p) — Temporal vs random splits, 8-mode threshold protocol, statistical protocol (100+ runs)
6. **Experimental Results** (3p) — UNSW-NB15, CICIDS2017, phase progression, baselines comparison, model size
7. **Discussion** (1p) — Connectivity as feature selection, threshold sensitivity, FPGA deployment analysis, limitations
8. **Conclusion** (0.5p) — Summary, future work

## Introduction Framing

Flow: *practical need → our approach → rigorous evaluation*

1. **Hook**: Edge IDS needs sub-microsecond inference with tiny models — deep learning is too heavy, rules are too rigid
2. **WNN as solution**: RAM neurons = O(1) lookup, sub-1KB model, but connectivity (which bits to observe) is the key
3. **Our approach**: Evolutionary optimization discovers optimal connectivity — effectively automated feature selection intrinsic to the architecture
4. **Rigorous evaluation**: Multi-dataset (UNSW-NB15, CICIDS2017), temporal + random splits, 8 threshold modes, 100+ runs per config — strong statistics showing the methodology is sound
5. **Result**: Competitive with RF/XGBoost on temporal splits at 50,000x smaller model size, with nanosecond inference potential on FPGA

Avoid "first honest evaluation" framing — instead demonstrate rigor through the work itself.

## Tables (8-9)

- T1: Three datasets summary (features, splits, sizes, class distribution)
- T2: Architecture comparison (Our WNN vs FWIW vs BiLSTM vs RF)
- T3: Phase-by-phase optimization progression (grid → GA → Lamarckian)
- T4: UNSW-NB15 main results — 100-run mean±std by threshold mode
- T5: CICIDS2017 main results — 34-run mean±std by threshold mode
- T6: Comparison with literature baselines (all datasets, temporal + random)
- T7: Feature selection impact (all features vs top-20)
- T8: Model size comparison (bytes vs KB vs MB)
- T9: FPGA resource utilization (LUTs, BRAMs, Fmax, latency)

## Figures (5-6)

- F1: Architecture diagram (input → thermometer encoding → RAM neurons with evolved connections → threshold → decision)
- F2: Phase progression chart (staircase improvement across 4-10 phases)
- F3: Threshold sensitivity box-and-whisker (100 runs × 8 modes)
- F4: FPR-vs-F1 scatter plot (Pareto frontier, color-coded by threshold)
- F5: Connectivity heatmap (which input bits GA selects → maps to feature names)
- F6: FPGA block diagram (optional — if space permits)

## Task Breakdown

### Phase 0: Data Collection (March 20-23)
- [x] 34 top20-CE4F3R3 flows completed (UNSW-NB15 temporal)
- [x] 34 baseline-mb28-kf5x5 flows completed (UNSW-NB15 temporal)
- [x] 2 4P and 2 10P flows completed
- [x] 34 random-split UNSW-NB15 flows queued (running)
- [x] F434/F435 leaderboard-seeded flows queued
- [x] CICIDS2017 HuggingFace dataset created
- [x] CIC-IoT-2023 HuggingFace dataset created
- [x] F505 CICIDS2017 test flow queued
- [ ] Queue 68 more PUB-top20-CE4F3R3 (→100 total)
- [ ] Queue 62 more 10P and 62 more 4P
- [ ] Create 34 CICIDS2017 flows (after F505 verified)
- [ ] Complete all queued flows

### Phase 1: Analysis Scripts (March 22-23)
- [ ] 1.1: Trimmed-mean extraction script from validation_summaries
- [ ] 1.2: Connectivity heatmap (which input bits selected → feature names)
- [ ] 1.3: Model size computation (neurons × bits × 2 bits/cell → bytes)
- [ ] 1.4: Phase progression tables from per-phase validation data
- [ ] 1.5: RF/XGBoost baselines on temporal split (UNSW-NB15 + CICIDS2017)

### Phase 2: Paper Writing — Data-Independent Sections (March 23-27)
- [ ] 2.1: Background section
- [ ] 2.2: Related Work section
- [ ] 2.3: Architecture section (including Lamarckian phases)
- [ ] 2.4: Evaluation Methodology section
- [ ] 2.5: Introduction (needs headline numbers from Phase 1)

### Phase 2.5: FPGA Synthesis (March 25-30)
- [ ] 2.5.1: Adapt FWIW RTL generator for our genome format
- [ ] 2.5.2: Vivado synthesis targeting Zynq Z-7020 (proof of concept)
- [ ] 2.5.3: Vivado synthesis targeting Alveo U25 (realistic IDS)
- [ ] 2.5.4: Extract resource utilization (LUTs, BRAMs, Fmax)
- [ ] 2.5.5: Compute latency (clock cycles) and throughput (Gbps)
- [ ] 2.5.6: Functional simulation with test vectors (verify correctness)
- [ ] 2.5.7: Write FPGA section for Discussion

### Phase 3: Results and Figures (March 28 — April 3)
- [ ] 3.1: Main results tables (T4, T5, T6)
- [ ] 3.2: Architecture diagram (F1)
- [ ] 3.3: Phase progression chart (F2)
- [ ] 3.4: Threshold sensitivity plot (F3)
- [ ] 3.5: Comparison scatter plot (F4)
- [ ] 3.6: Connectivity heatmap (F5)
- [ ] 3.7: Experimental Results section
- [ ] 3.8: Discussion section

### Phase 4: Polish and Submit (April 4-16)
- [ ] 4.1: Conclusion
- [ ] 4.2: Internal review — all numbers match, claims supported
- [ ] 4.3: Check page limit (LNCS: 20 pages excluding refs)
- [ ] 4.4: Supplementary materials (code, HuggingFace datasets, FPGA RTL)
- [ ] 4.5: Final proofread
- [ ] 4.6: Submit by April 16

### Phase 5: Conference Preparation (post-submission)
- [ ] 5.1: FPGA hardware demo on Zynq or AWS F1 (Approach C)
- [ ] 5.2: CIC-IoT-2023 flows (34 runs)
- [ ] 5.3: Multi-class hierarchical experiment
- [ ] 5.4: Prepare presentation slides

## Timeline

| Week | Dates | Focus |
|------|-------|-------|
| 1 | Mar 20-23 | Data collection + analysis scripts |
| 2 | Mar 24-27 | Draft Background, Related Work, Architecture, Methodology |
| 3 | Mar 28-Apr 3 | FPGA synthesis + tables/figures + Results + Discussion |
| 4 | Apr 4-10 | Introduction finalized, polish, internal review |
| 5 | Apr 11-16 | Final proofread, submit |

## FPGA Plan

### For the Paper (Approach A+B)
- **Synthesis-only + simulation** — no hardware needed
- Adapt FWIW's open-source RTL generator for our genome
- Target two platforms: Zynq Z-7020 (edge) and Alveo U25 (data center)
- Report: LUTs, BRAMs, Fmax, clock cycles per classification, throughput (Gbps)
- Functional simulation: verify WNN classification matches Python implementation
- Compare with FWIW's published FPGA numbers

### For the Conference (Approach C — stretch)
- **Live hardware demo** on Zynq Pynq-Z2 (~$200) or AWS F1 ($1.65/hr)
- Real packet processing at wire speed
- Order Pynq-Z2 from Digikey (ships in days, no lead time)

### Key FPGA Metrics to Report
- Resource utilization vs available (% of target device)
- Latency: clock cycles per classification
- Throughput: classifications/second → Gbps equivalent
- Power consumption (from synthesis report)
- Comparison with FWIW: same accuracy class, our evolved connectivity advantage

## Primary Competitor: FWIW (Susskind et al., FPGA 2023)

Authors: Susskind, Arora, **Bacellar, Dutra, Miranda, Lima, França** (UFRJ) + John (UT Austin)

| Aspect | FWIW | Our Work |
|---|---|---|
| Connectivity | Random | **Evolved (GA/TS)** |
| Evaluation | Random split only | **Temporal + random, 3 datasets** |
| Threshold | Single mode | **8 modes** |
| Memory | Binary | **QUAD_WEIGHTED** |
| Model size | 272B | <1KB (similar) |
| FPGA | Yes (Vivado) | **Synthesis + simulation** |
| Statistical runs | 1 | **100+** |

## Comparison Baselines

| Method | Dataset | Split | F1 | FPR | Model Size | Source |
|---|---|---|---|---|---|---|
| RF | UNSW-NB15 | temporal | ~87% | ~12% | ~50MB | Zoghi 2024 |
| XGBoost | UNSW-NB15 | temporal | ~87% | ~12% | ~10MB | Zoghi 2024 |
| BiLSTM | UNSW-NB15 | temporal | ~89% | ~8% | ~5MB | Abdalgawad 2024 |
| FWIW | UNSW-NB15 | random | 98%+ | <1% | 272B | Susskind 2023 |
| RF | CICIDS2017 | random | 99.9% | ~1% | ~50MB | Various |
| XGBoost | CICIDS2017 | random | 99.95% | <1% | ~10MB | Various |
| Our WNN | UNSW-NB15 | temporal | 87-90% | 7-14% | <1KB | This work |
| Our WNN | UNSW-NB15 | random | ~94% | <1% | <1KB | This work |
| Our WNN | CICIDS2017 | temporal | TBD | TBD | <1KB | This work |

## Risks

1. **Flows don't complete**: Paper viable with current 34+34 UNSW-NB15. CICIDS2017 is nice-to-have.
2. **Results not strong enough**: Key claim is competitive F1 at sub-1KB, not raw supremacy.
3. **Oracle threshold questioned**: Report all 8 modes. Use fixed_05 as honest primary.
4. **FPGA synthesis fails**: Discussion section can present theoretical analysis instead.
5. **CIC-IoT-2023 too slow**: Park for journal extension / ACSAC backup.

## References

- FWIW: Susskind et al., FPGA 2023 — [GitHub](https://github.com/ZSusskind/FWIW)
- BTHOWeN: Bacellar et al., 2022 — thermometer encoding
- Existing bibliography at `llm-optimizer/references.bib` (50+ refs)
- Week 16-18 blog posts have detailed methodology and analysis
