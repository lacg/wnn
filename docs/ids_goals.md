# IDS Goals — UNSW-NB15 Binary Classification

## Standard Split (175K train / 82K test — what we use)

This is the harder, more realistic split with temporal separation between train/test.

### Literature Baselines

| Model                  | Accuracy | F1-Macro | FPR    | Source                        |
|------------------------|----------|----------|--------|-------------------------------|
| Random Forest          | 87.2%    | ~87%     | ~12%   | Our reproduction (Week 18)    |
| XGBoost                | 87.3%    | ~87%     | ~12%   | Our reproduction (Week 18)    |
| SVM                    | ~86%     | ~85%     | ~15%   | Kasongo & Sun 2020            |
| DNN (sequential)       | ~88%     | ~87%     | ~10%   | Zoghi & Serpen 2024           |
| BiLSTM (deep learning) | ~89%     | ~89%     | ~8%    | Abdalgawad et al. 2024        |

### Our WNN Targets

| Metric                    | Target | Stretch Goal | Basis                                    |
|---------------------------|--------|--------------|------------------------------------------|
| **Accuracy**              | ≥ 88%  | ≥ 90%        | Beat RF (87.2%) and XGBoost (87.3%)      |
| **F1-Macro**              | ≥ 87%  | ≥ 89%        | Match or beat classical ML (~87%)        |
| **FPR**                   | ≤ 12%  | ≤ 8%         | Match classical ML; stretch = BiLSTM     |
| Multi-class (10 classes)  | ≥ 85%  | ≥ 88% (F1)   | Novel benchmark (no WNN prior art)       |

Matching RF/XGBoost (~87% F1, ~12% FPR) validates the architecture.
**Beating them is publishable** — no prior WNN work reports results on the standard split.

## Random Split (90/10 deduplicated — FWIW comparison)

Much easier — inflated numbers due to data leakage between train/test:

| Model          | Accuracy | F1     | FPR  |
|----------------|----------|--------|------|
| RF / XGBoost   | 98-99%   | 97-99% | <2%  |
| Deep Learning  | 98-99%   | 98-99% | <1%  |
| FWIW WNN       | 98.5%    | —      | —    |

Target: ≥ 98.5% (match FWIW).

## Research FPR vs Production FPR

**Our FPR targets above (8–12%) are research metrics on a balanced test set.**

Production deployment is a fundamentally different regime:

- In production, 95–99% of traffic is normal
- A 10% FPR on 1M flows/day = **~100,000 false alerts/day**
- Industry guidance: <1% FPR is "good", <0.1% is "excellent"
- **Alert fatigue** is the #1 operational killer of IDS tools — teams turn off noisy detectors

This is why production IDS uses **cascading architectures**:

```
Tier 1: WNN on FPGA (5ns, inspects EVERY packet)
  ↓ flags ~5% suspicious
Tier 2: Deep model on CPU/GPU (re-examines flagged subset only)
  ↓ confirms ~0.1% as true alerts
SOC analyst
```

The WNN's value is not standalone FPR but **throughput × recall at the first tier** —
inspecting every packet at line rate rather than sampling 1 in 100.

Production FPR targets (for the cascading system):

| Tier          | FPR Target | Role                                  |
|---------------|------------|---------------------------------------|
| Tier 1 (WNN)  | ≤ 5%       | High recall, flags suspicious traffic |
| Tier 2 (Deep) | ≤ 0.1%     | High precision, confirms real threats |
| Combined      | ≤ 0.1%     | What the SOC analyst actually sees    |

## Sources

- [Zoghi & Serpen 2024 — Building an IDS on UNSW-NB15](https://onlinelibrary.wiley.com/doi/full/10.1002/cpe.8242)
- [Abdalgawad et al. 2024 — Enhanced IDS Performance](https://www.mdpi.com/1999-4893/17/2/64)
- [Kasongo & Sun 2020 — Feature Selection on UNSW-NB15](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-020-00379-6)
- [UNSW-NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- [Check Point — FPR in Cybersecurity](https://www.checkpoint.com/cyber-hub/cyber-security/what-is-a-false-positive-rate-in-cybersecurity/)
- [Fidelis Security — Reducing IDS False Positives](https://fidelissecurity.com/cybersecurity-101/network-security/reducing-false-positives-in-intrusion-detection-systems/)
