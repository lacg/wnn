# Train / validation / test — the convention this project uses

Decided 11/11/2026 (Luiz): **one convention everywhere — the literature's.** Checked against primary sources
the same day; quotes below.

| word | role | touched | in this project |
|---|---|---|---|
| **train** | fit the model | freely | memory writes, GA search folds (K-fold on the 80% for IDS) |
| **validation** (`val`) | model selection and tuning | repeatedly | threshold calibration (`val_cal`), controller stage-select (`--val-units`), futility gates, early stopping |
| **test** | final, unbiased report | once, at the end | every published number: the IDS `_3way` test partition, the controller report seeds |

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

## State of the code against this convention (11/11/2026)

- **IDS: already consistent.** Protocol v2 calibrates on the `_3way` val partition (`val_cal`, `margin_val_cal`)
  and reports on the test partition.
- **Controller: consistent in behaviour.** Stage-select ranks on the val units (`--val-units`, `seeds.val`); the
  published numbers come from `--report-seeds`, which are the test set under this convention.
- **Inconsistent: `src/wnn/seeds.py`.** Its docstring and `SeedSet` say test = "model selection / early-stopping"
  and val = "held-out final report", the reverse of both the literature and how the controller uses `seeds.val`.
  Fix planned in `.claude/plans/test_val_swap.md`.
- `test_summaries` (DB table) stores checkpoint reports, including test-partition metrics. The name predates
  Protocol v2; renaming it is optional and listed in the plan.
