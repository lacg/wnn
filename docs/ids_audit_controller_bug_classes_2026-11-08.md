# IDS audit for the controller bug classes (08/11/2026, read-only, ids-security agent)

**Scope.** Read-only: the DB opened read-only; worker, queue and wheels untouched.
**Probes.** Job tmp `enc_probe.py`, `keys.py`.
**Overall.** No test/val leak found in results. The findings change search efficiency and record-vs-reality.

## 1. Neuron identity by POSITION (live, search noise only)

**Read from code.**
- The rate-0.25 neuron-sampling mask is keyed on the flat neuron index `cluster_neuron_starts[c] + n` (adaptive/eval_export.rs:546-566).
- The neurons mutation removes neurons from cluster ends (core/neighbor_search.rs:367-439), so a mutation in cluster 0 re-draws the 25% subset of every later cluster's neurons. Pool-shuffle crossover (ratio 0.8) does the same.
- Fitness therefore depends on position: noisier inheritance, and elites keep lucky draws.
- Memory is always retrained, so there are no stale cells and no leak.
- Rate 1.0 is unaffected, so the IDS-47 QUAD 1m arms are clean.
- Affected: every rate-0.25 flow, including the IDS-47 BINARY arms (1,519 completed flows since id 4000).
- Size of the effect not measured; a paired probe (permute neurons within a cluster) needs the worker idle.

**Dormant.**
- `crossover_ps_neurons` child2 count when dedupe is off (line 1285; discarded at 2147).
- adaptive/adaptive_eval.rs:906,948 seed by batch position, the same bug eval_hybrid fixed; it runs only in genesis phases (flows 430/431, March).

**Fix.** Key the mask on neuron identity (hash of cluster, bits, connections); ABI bump.

## 2. Settings overridden between stages (live mechanism; hit in OLD cohorts)

**Read from code.** worker.py:1398-1401: the experiment row's creation-time fitness weights/calculator beat the live flow params (`exp_data.get(...) or params`). `max_iterations` does the same (1496/1498), while `population_size` prefers params (1487). Precedence is inconsistent, and `or` treats 0.0 as unset.

**Measured.**

| Cohort | params ce/acc | ran with |
|---|---|---|
| **PUB50-ciciot (711–846), 112 done** | 0.1/0.2 | **0.2/0.1** (log 711: evaluator CE=0.1/Acc=0.2 vs grid/GA HarmonicRank ce=0.2/acc=0.1) |
| NSR/USAMP/TWOCLUST/FLIP-ciciot (1079–1089), 8 done | 0.3/0.1 | 0.1/0.4 |
| BITS/FEAT/USAMP f1dom (1123–1150) | 0.2/0.1 | 0.0/0.1 |
| PUB-ciciot-full46M (1156) | 0.1/0.2 | 0.2/0.1 |
| after 1156 (incl. IDS-47 / S&P) | — | 0 mismatches |

So PUB50-ciciot is not apples-to-apples with the other PUB50 batches, contrary to CLAUDE.md.

**Dormant.** TSConfig/AdaptationConfig (experiment.py:792,861) never receive `magnitude_aware_patience` / `random_search`.

**Fix.** Live params win for every key (`is not None`); warn on row/params disagreement; pass both flags to TS and adaptation.

## 3. Dead / duplicate encoding bits (measured)

Thresholds are fitted on train only (ids/dataset.py:186-188), so Protocol v2 holds. But the distributive dedupe (representations/thermometer.py:316) only adds +1e-10 to equal quantiles. On integer data that makes zero-width bins whose bits are exact copies.

| dataset | bits/feature | total bits | constant | duplicate | distinct useful |
|---|---|---|---|---|---|
| unswr | 64 | 1104 | 56 | 543 | 559 (51%) |
| cicids | 96 | 1920 | 15 | 913 | 1005 (52%) |
| ciciot | 96 | 1845 | 63 | 745 | 1098 (60%) |

- Exact-duplicate features: cicids `Bwd Packet Length Mean` = `Avg Bwd Segment Size`; ciciot `Srate` = `Rate`.
- IDS-47 final genomes spend 40.6–49.2% of their connections on constant or duplicate bits. That's the uniform-random expectation, since there is no connections phase.
- Effective address bits are 77–87% of nominal: a "34b" neuron is about 26–30 effective bits.
- It's a capacity handicap: conservative vs XGBoost, fair within WNN arms, but it muddies "bits" findings.

**Possible ranking-level leak (not quantified).** The top-20 feature lists were ranked on other partitions (UNSW "standard", CICIDS "temporal", CIC-IoT neto-full train) that may overlap today's random_3way test/val. → experiment-design.

**Fix.** Unique quantiles plus fewer bits for low-cardinality features; drop exact-duplicate columns and constant bits (or sample over distinct bit classes). This changes results, so it needs an encoder version in the validation-cache key.

## 4. Registered-but-ignored params

| param | status |
|---|---|
| `decode_mode` | never read |
| `mutation_rate` | never read from params |
| `max_bit_delta` | Python mutate only, Rust derives its own |
| `ids_arch_type` | copied into an unread field; 171 flows, all at the default |
| `phase_order` | read by cli.py only; 5,670 flows, all at the default |
| `fitness_ce_anchor_normalized` | deprecated, inert; 875 flows incl. IDS-47 record 0.2128, never applied |

The reverse direction is clean (no read-but-unregistered keys).

**Fix.** Wire or drop each key; have the validator warn when a key has no effect for the flow's architecture.
