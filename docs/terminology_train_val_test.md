# Train / validation / test — the convention this project uses

Decided 11/11/2026 (Luiz): **one convention everywhere — the literature's.** Checked against primary sources
the same day; quotes below.

| word | role | touched | in this project |
|---|---|---|---|
| **train** | fit the model | freely | memory writes, GA search folds (K-fold on the 80% for IDS) |
| **validation** (`val`) | model selection and tuning | repeatedly | threshold calibration (`val_cal`), controller stage-select (`--val-units`), futility gates, early stopping |
| **test** | final, unbiased report | once, at the end | every published number: the IDS `_3way` test partition, the controller test seeds |

A number is **report-grade only if it comes from the test partition.** A validation number is a tuning number:
it is optimistic by construction and must never be compared against someone else's test number.

## Sources (what each says)

- **Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*, §7.2** (via chapter notes that track the
  text): the training set is used to fit the models; the "validation set [is] used to estimate prediction error
  for model selection"; the "test set [is] used for assessment of the generalization error of the final chosen
  model"; "the test set should only be brought out at the end of the data analysis".
- **Ripley, *Pattern Recognition and Neural Networks* (1996), glossary p. 354** (as reproduced by secondary
  sources): validation set = examples used to tune the parameters (architecture) of a classifier; test set =
  examples "used only to assess the performance (generalization) of a fully specified classifier".
- **scikit-learn user guide, "Cross-validation"**: "training proceeds on the training set, after which evaluation
  is done on the validation set, and when the experiment seems to be successful, final evaluation can be done on
  the test set."
- **Arp et al., "Dos and Don'ts of Machine Learning in Computer Security", USENIX Security 2022** — the paper
  security reviewers cite for evaluation pitfalls (read from the PDF):
  - "Test snooping occurs when the test set is used for experiments before the final evaluation."
  - P5, biased parameter selection: "The final parameters of a learning-based method are not entirely fixed at
    training time. Instead, they indirectly depend on the test set."
  - Fix: "using a separate validation set for model selection and parameter tuning."
- **Wikipedia, "Training, validation, and test data sets"**: validation set = "used to tune the hyperparameters";
  test set = "used to provide an unbiased evaluation of a model fit on the training data set".

Not verified from the primary text (search did not return it): Goodfellow, Bengio, Courville, *Deep Learning* §5.3.
Secondary course notes that follow it give the same roles.

## Where the opposite usage comes from

The reversed usage (test = development, validation = final) is real, which is why the confusion happens:

- Wikipedia notes the two terms are "sometimes used interchanged" in industry and academia, and that with only a
  two-way split the held-out part is sometimes called the validation set.
- Medicine uses "validation" for the final check of a finished model ("external validation"). Walston et al.,
  "Data Set Terminology of Deep Learning in Medicine" (arXiv:2404.19303, 2024) reviews the clash and recommends
  the labels training / validation (or tuning) / test.

For ML and security venues (S&P, USENIX Security, IROS) the table at the top is the convention.

## State of the code (renamed 11/11/2026, branch `test-val-terminology`)

- **IDS.** Protocol v2 calibrates on the `_3way` val partition (`val_cal`, `margin_val_cal`) and reports on the
  test partition. Unchanged — it already followed the convention.
- **Checkpoint results table.** `validation_summaries` is now **`test_summaries`** (column `validation_point` →
  `test_point`), and `combined_validations` is now `combined_test_summaries`. The dashboard renames both on
  startup (`rename_legacy_test_tables`). API routes: `/api/flows/:id/test-summaries`,
  `/api/flows/:id/combined-test-summaries`, `/api/test-summaries/check`. Worker, frontend and iOS names follow
  (`TestSummary`, `create_test_summary`, `test_cache_key.py`, `TestProgressionChart`).
- **Controller.** Stage-select ranks on the val units (`--val-units`, `seeds.val`). The published numbers come
  from the test units: `--test-seeds` (was `--report-seeds`), `--test-seed` (was `--report-seed`, merged with the
  seed registry's test seed) and `--test-episodes` (was `--report-episodes`). Log tags are `[test-seeds]` /
  `[test-seed]`.
- **Seed registry (`src/wnn/seeds.py`).** Documented roles corrected: val = selection, test = report. No seed
  value moved.
- **Legacy LM path (`worker.py`).** It used to search on wikitext test and report on wikitext validation. It now
  searches on validation and reports on test. This is the one behaviour change; the LM stack has been retired
  since 10/06/2026, and old LM numbers are not comparable with a re-run.

### What still carries the old words, on purpose

- Log files written before the cut-over print `[report-seeds]`. The four parsers that read them
  (`recalc_headlines.py`, `repair_resumed_stage_select.py`, `rescore_first_test_seed.py`, `stable_failure_ci.py`)
  accept both tags.
- Historic sections of `docs/ids_results.md` and dated notes describe what was run at the time; identifiers in
  them were renamed, prose was not rewritten.
- HuggingFace split names (`validation`, `test`) are the dataset's own and already match the convention.
