# Email draft to RAID 2026 PC chair(s)

**To**: raid26.pc.chairs@gmail.com
**Subject**: RAID 2026 camera-ready — scope question on post-submission improvements (paper #[ID])

**Timing**: Send 1-3 days AFTER receiving notification (10 July 2026 or later),
assuming acceptance. Sending before notification is unusual and chairs often
won't engage with specific-paper content questions during review.

---

Dear [PC chair name(s)],

Thank you again for the acceptance of our paper "[full paper title]"
(submission #[paper ID]) to RAID 2026.

In preparing the camera-ready, we have identified and implemented several
post-submission improvements to our experimental methodology on the
CIC-IoT-2023 benchmark. Two of these are bug corrections to our existing
methodology and we believe fall within the scope of a typical
camera-ready revision; two are more substantive refinements that I want
to explicitly check with you before including in the final version.

**Bug corrections (intended to keep)**

1. *Training-order correction*: our legacy QUAD-state cell update was a
   sequential clamped random walk on cell values rather than the intended
   order-independent vote tally. We corrected this to single-pass
   accumulation. Effect on cohort: F1 mean +0.3 pp, FPR mean −1.6 pp,
   per-seed std halved (≈0.4 → ≈0.2). The bug fix improves reproducibility
   without changing the methodology's intent.

2. *Threshold-column semantic fix*: a refactor between submission and
   camera-ready had made the `empirical_cumulative` calibration column
   numerically identical to `train_cal`. We corrected it to report the
   GA-fitness-weighted operating point as originally intended.

**Methodology refinements (need your guidance)**

3. *Thermometer encoding*: the submission used 8-bit thermometer
   encoding for CIC-IoT-2023. A post-submission encoding-width sweep
   identified 96-bit as substantially better for this dataset (the
   sweep was incomplete at submission time). Adopting 96-bit increases
   the CIC-IoT-2023 GA Neurons F1 mean from ~80% (submitted) to ~93%
   (camera-ready).

4. *Architecture search-space ceiling*: the submission capped at 500
   neurons × 34 bits per neuron. We widened to 250 × 100 bits per
   neuron based on a discovery that the GA prefers higher-bits /
   lower-neurons regimes than the submitted cap allowed.

I want to confirm with you whether changes (3) and (4) are acceptable
for the camera-ready, given they materially shift the reported numbers.
We are prepared to handle this in any of the following ways, depending
on your preference:

(a) Include the refinements in the camera-ready with an explicit
    "Post-submission methodology improvements" paragraph in Section 3
    describing each change and its impact.

(b) Report both methodologies side-by-side: the submitted version as
    the main result, the post-submission version in a "post-submission
    ablation" appendix.

(c) Revert to the submitted methodology (8-bit thermometer, 500n × 34b)
    for the camera-ready CIC-IoT-2023 numbers, keeping only the bug
    fixes (1) and (2). The methodology refinements (3, 4) would then
    be deferred to a follow-up paper.

To inform our decision either way, we are also running two additional
flows at the submitted methodology (8-bit thermometer, 500n × 34b)
with the bug fixes (1, 2) applied. This will let us quantify how much
of the +11 pp F1 gain comes from the bug fixes alone vs the methodology
refinements.

Please let me know which option you would prefer (or if a different
treatment is appropriate). I will hold off on finalizing the camera-
ready CIC-IoT-2023 sections until I hear back.

Thank you for your guidance,

[Your name]
[Your affiliation]
[Your email]
