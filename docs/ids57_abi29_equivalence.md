# IDS-57 A1: worker wheel equivalence record, ABI 28 -> ABI 29 (W1-W4)

Luiz, this is the record required by `docs/ids57_a1_prereg.md` §8. The A1 arm (flows 7546-7548, `max_bits=64` on
unswr) will run on worker ABI 29; its controls (7509 / 7513 / 7517) ran on ABI 28. That pairing is accepted only if
this file shows bit-identity on the arm's code path.

**Status (12/11/2026): W1 captured on ABI 28, before the swap. W4 "before" line recorded. W2 and W3 are open and
must be filled after the swap with the commands below.**

Produced by the quality-assurance agent, which did not write the ABI 29 change.

Harness: `scripts/ids57_abi_equivalence.py` (sha256 `460d41edfdde64f9f5952859b70c1d8acddcb0a33faaac2194ff2811b2cadd03`
at W1; W2/W3 must run the same file, unchanged).

## W4: the record

| | Before (W1) | After (W2/W3) |
|---|---|---|
| Captured | 12/11/2026 15:47 UTC (10:47 ET) | _to fill_ |
| By | quality-assurance agent (not the implementer) | _to fill_ |
| `ram_accelerator.ABI_VERSION` (installed) | 28 | _to fill (must be 29)_ |
| Installed binary | `/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/lib/python3.13/site-packages/ram_accelerator/ram_accelerator.abi3.so`, installed 09/11/2026 18:47 UTC | _to fill_ |
| Binary sha256 | `14734ec14a0d560158668c876d7445a59dfbc4ee33f80a7267efa9695c456292` | _to fill_ |
| Wheel archive sha256 (`direct_url.json`, `experiments/staged/ram_accelerator-2026.212.37-cp311-abi3-macosx_11_0_arm64.whl`) | `219d62898f027682919e62a68fb8446e7989bbf22f21a4fbcffbac3f25177b19` | _to fill_ |
| Worker tree git SHA (`/Users/lacg/wnn` HEAD) | `069c6011de0a2dc628b58f22af6868ac3cc7623f` | _to fill_ |
| `src/wnn` tree SHA (`git rev-parse HEAD:src/wnn`), dirty? | `fe98e3e0da9387629b8bfe569790ac23601fccdb`, clean | _to fill_ |
| `wnn.accel.EXPECTED_ABI` | 28 | _to fill_ |
| Digest file | `experiments/ids57_abi_equivalence/w1_abi28.json`, file sha256 `438e4cd17cc3476cb0233d5694a1c1b93374e301cbec9f8d4d4772de885c7b1f`, payload sha256 `df297d9d466f48baac91d8aa1d665b73f7187530b70633d96caff6414318034c` | `experiments/ids57_abi_equivalence/w2_abi29.json` _to fill_ |

Notes on "before":
- The live worker (pid 83938, started 12/11 04:23 ET) and its flow runner for 7543 (pid 64353, started 07:54 ET) have
  this same binary mapped (`lsof`), and no commit has touched `src/wnn` since 12/11 04:00 ET, so the worker's Python
  tree equals the `src/wnn` tree SHA above. The digests therefore describe the deployed pair, not only the wheel.
- The source tree's `lib.rs` still says `ABI_VERSION = 28` at this HEAD; the ABI 29 change was not in the main
  checkout when W1 ran.

## W1: reference digests on ABI 28

### What was digested

Configuration: the params of control flow 7509, read from the DB (read-only), built through
`IdsEvaluatorSettings.from_params` -> `IdsEvaluatorFactory.load` / `_optimizer` / `_test` -> `IDSEvaluator` ->
`IDSCacheWrapper` (via `wnn.accel`). The harness refuses to run unless the settings are: unsw-nb15 `random_3way`,
binary, BINARY mode, `lattice` write rule, two clusters, `neuron_sample_rate=0.25`, `balance_classes=true`,
order-independent train on, `ids_k_folds=5`, `ids_kfold_per_gen=5`, encoding dedupe off, legacy feature ranking,
receptive fields off. Process env mirrors the live flow runner: `WNN_EVAL_IN_PLACE=1`,
`WNN_ORDER_INDEPENDENT_TRAIN=1`, `WNN_HYBRID_SPEED_RATIO=8` (the worker's own prediction for these params).

Data: 1,267,407 train rows, 158,426 VAL rows, 1,104 input bits (top20 legacy features, 64 bits per feature).

Genomes: eight, with connectivity drawn from fixed numpy seeds (57,000 + ...), none taken from any flow.

Per genome, three independent Rust training passes plus the CV:
1. **cv**: `IDSEvaluator.evaluate_batch([genome])` on the optimizer evaluator, which is the call the GA makes for a
   pool: the K-fold CV mean over all 5 folds of the train partition (`evaluate_genomes_kfold_mean_hybrid`). Stored:
   ce, acc, f1, fpr, threshold, as exact hex floats. The same eight genomes are also scored as one pool
   (`cv_pooled`).
2. **val_modes**: `evaluate_multiclass_at_thresholds` with VAL in the report slot: every decode mode's tau, CE,
   metrics and confusion matrix on VAL. `margin_val_cal` tau is the val-calibrated threshold.
3. **export**: `export_trained_layer`: sha256 of the raw per-row VAL scores (float64 bytes) and sha256 over every
   exported physical unit (cluster, bits, connections, sorted keys, cells), the grouping counts and the miss cell.
4. **score_examples**: the scoring-only entry point, sha256 of the raw VAL scores again.

**No TEST row was scored or even loaded into the Rust cache in this mode.** The evaluators are built on a copy of
the dataset whose report set is the VAL partition (`val_as_report`); the K-fold fitness uses the train partition
only. Every VAL confusion matrix in the file sums to 158,426 with the VAL class counts (152,391 / 6,035), not the
TEST ones (152,390 / 6,036).

### Digests

CV columns are the 5-fold means (F1 / FPR / CE). "Key reach" is the bit length of the largest trained address and
how many stored addresses have bit 63 set, as evidence that the case actually writes past 34 bits and across the
top of the u64.

| Case | Neurons per cluster | Bits | CV F1 / FPR / CE | VAL scores sha256 (first 16) | val-cal tau | Cells sha256 (first 16) | Stored cells | Key reach (bits / bit-63 keys) |
|---|---|---|---|---|---|---|---|---|
| `b34_s0` | [250, 250] | 34 | 0.939580 / 0.009007 / 0.330346 | `34dad5219b79aa1e` | -0.001999974250793457 | `68c3c19b965bcd7b` | 672,440 | 34 / 0 |
| `b34_s1` | [250, 250] | 34 | 0.939207 / 0.008948 / 0.330644 | `fd7ebae91132ef88` | -0.001999974250793457 | `92cd3f560557a04a` | 668,530 | 34 / 0 |
| `b48_s0` | [250, 250] | 48 | 0.943254 / 0.007703 / 0.329225 | `bd7aa981d2f97442` | 0.001999974250793457 | `2ed63f7adff9c543` | 1,298,780 | 48 / 0 |
| `b48_s1` | [250, 250] | 48 | 0.943519 / 0.007754 / 0.329168 | `4aed58c7de50a182` | 0.001999974250793457 | `b755e1ad847a0918` | 1,297,930 | 48 / 0 |
| `b64_s0` | [250, 250] | 64 | 0.946313 / 0.006513 / 0.329084 | `c1b23cc21a99df69` | 0.001999974250793457 | `9c6fe8f206543364` | 2,084,915 | 64 / 975,324 |
| `b64_s1` | [250, 250] | 64 | 0.945947 / 0.007030 / 0.329343 | `7dafb698304a5131` | 0.001999974250793457 | `e24548083f17b9db` | 2,201,641 | 64 / 1,046,424 |
| `mixed_34_48_64` | [250, 250] | 34 / 48 / 64 per neuron | 0.944572 / 0.007578 / 0.329460 | `5246b9e773c0a389` | 0.001999974250793457 | `a7fa60e4e3f47599` | 1,436,186 | 64 / 363,245 |
| `mixed_4_to_64` | [250, 137] | 4 to 64 per neuron | 0.940431 / 0.008591 / 0.346366 | `48e94d11ec6a6100` | 0.00045248866081237793 | `0281657a5d0038a1` | 767,421 | 64 / 14,273 |

Other reference values: desirability CE half-anchor (derived in the wheel from the train labels) 0.034441125208701774.
The decimals above are for reading; the comparison uses the JSON, which holds exact hex floats and full hashes
(1,033 compared fields).

### Checks on the reference itself

- **Determinism: CONFIRMED.** The digest mode was run three times on ABI 28: twice with `RAYON_NUM_THREADS=4` and
  once with `RAYON_NUM_THREADS=10` (the worker's setting). All three payloads are
  `df297d9d466f48baac91d8aa1d665b73f7187530b70633d96caff6414318034c`; `--compare` reports 1,033 fields, 0 differing.
  So a W2 difference cannot be thread scheduling or run-to-run noise.
- **Pool independence:** the pooled CV of all eight genomes in one `evaluate_batch` equals the eight single-genome
  CVs field for field.
- **Three scoring entry points agree:** for every case the raw VAL scores from `export_trained` (report slot),
  `export_trained` (val slot) and `score_examples` have the same sha256.
- **The comparison discriminates:** a copy of the W1 file with one cells hash altered is reported
  `VERDICT: DIFFERENT` with the field named.
- Runtime about 115-120 s per digest run, one process at nice 10, peak RSS 13.4 GB (the dataset load; the Rust cache
  is shared between the two evaluators). The worker, its flow, the DB and the venv were not touched.

### W3-on-28 sanity: does ABI 28 reproduce its own stored matrices?

**Yes, exactly.** `--retrain-finals 7509 7513 7517` on ABI 28 rebuilt each flow's GA `best_f1` final genome from the
DB, built the worker's `(optimizer, test)` pair with `IdsEvaluatorFactory.flat`, retrained, and compared all 10
stored decode modes of `test_summaries.threshold_metadata` (confusion matrices as integers, tau / CE / metrics as
exact floats). These rows were already reported, so the TEST read is allowed. Output:
`experiments/ids57_abi_equivalence/w3_sanity_abi28.json` (17 s).

| Flow | Seed | Genome hash | Neurons per cluster | TEST confusion, `margin_val_cal`, stored | Retrained on ABI 28 | All 10 modes exact |
|---|---|---|---|---|---|---|
| 7509 | 20421 | `4a486f07ca5fa6fc` | [250, 243] | [[152084, 306], [986, 5050]] | [[152084, 306], [986, 5050]] | yes |
| 7513 | 20422 | `8ecab079c3ac7c72` | [250, 249] | [[152047, 343], [957, 5079]] | [[152047, 343], [957, 5079]] | yes |
| 7517 | 20423 | `43660139270ab4b3` | [248, 249] | [[152081, 309], [942, 5094]] | [[152081, 309], [942, 5094]] | yes |

This means W3 has a clean baseline: if a matrix differs after the swap, the cause is the swap (wheel or Python
tree), not an irreproducible stored row.

### What W1 does not cover

- **GA breeding.** W1 digests how a given genome is trained, scored and exported. It does not digest
  `search_offspring` (tournament, crossover, mutation, the offspring RNG). If ABI 29 changed breeding at widths up to
  34 bits, W1-W3 as pre-registered would not see it. If that matters for the ruling, a breeding digest has to be
  captured on ABI 28 before the swap as well.
- **Duplicate neurons.** The synthetic genomes have none (physical cells equal logical cells in every case), so the
  multiplicity collapse is exercised only through the three GA finals of W3.
- Only the flat binary two-cluster BINARY lattice path is covered, which is the arm's path. Other modes, the
  streaming path and the single-cluster path are out of scope here.

## W2: the same digests on ABI 29 (open)

Run after the swap, with the worker's new wheel installed in the venv, harness file unchanged:

```bash
cd /Users/lacg/wnn && export PYTHONPATH=/Users/lacg/wnn/src/wnn
PY=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python
$PY -c "import ram_accelerator as a; print(a.ABI_VERSION)"          # must print 29
shasum -a 256 scripts/ids57_abi_equivalence.py                      # must be 460d41ed...2cadd03
nice -n 10 $PY scripts/ids57_abi_equivalence.py --out experiments/ids57_abi_equivalence/w2_abi29.json
$PY scripts/ids57_abi_equivalence.py --compare \
  experiments/ids57_abi_equivalence/w1_abi28.json experiments/ids57_abi_equivalence/w2_abi29.json
```

Pass = `VERDICT: BIT-IDENTICAL` (exit 0), payload `df297d9d466f48baac91d8aa1d665b73f7187530b70633d96caff6414318034c`.
Any `DIFF` line = stop, per §8 (rebuild ABI 28 offline for the reference, or re-fly all three controls; Luiz decides).
About 2 minutes, 13.4 GB peak RSS, one process.

Result: _to fill (verbatim `--compare` output)._

## W3: retrain the controls' GA finals on ABI 29 (open)

```bash
nice -n 10 $PY scripts/ids57_abi_equivalence.py --retrain-finals 7509 7513 7517 \
  --out experiments/ids57_abi_equivalence/w3_abi29.json              # exit 0 = every stored mode reproduced exactly
$PY scripts/ids57_abi_equivalence.py --compare \
  experiments/ids57_abi_equivalence/w3_sanity_abi28.json experiments/ids57_abi_equivalence/w3_abi29.json
```

Pass = `all_exact: true` in the output and `VERDICT: BIT-IDENTICAL` against the ABI 28 sanity file. The §8
requirement is the confusion matrices (`all_confusions_match`); the harness also holds the floats to exact equality.
About 20 seconds.

Result: _to fill._

## Verdict

_Open until W2 and W3 are filled. W1 alone proves nothing about ABI 29._
