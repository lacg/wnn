# Controller campaign glossary

_Plain-language dictionary for every label used in the drone-controller investigation
(docs/controller_horizon_findings.md, docs/controller_research_roadmap.md, the chain
watcher reports, and the memory files). One or two lines each. 04/07/2026._

## The Findings (numbered results in controller_horizon_findings.md)

| Label | What it says, in one sentence |
|---|---|
| **Finding 0** | The "WNN is missing an integrator" theory was wrong: the PID's own I-term is worth ≤0.06° in our disturbance-free sim (PD-only is just as good), so there was nothing to integrate — the WNN's gap must come from elsewhere. |
| **Finding 1** | Controllers trained on short (500-step / 0.5 s) episodes look stable but never learned to *hold* attitude — past their training horizon they slowly tumble. Training on 2000-step episodes produces a real equilibrium, which itself decays past ~2.5× its own horizon. |
| **Finding 2** | A committee (average the motor commands of several diverse controllers) is approximately horizon-free: each member drifts its own way, the drifts are uncorrelated, and the average cancels them — 92% at 10,000 steps while every individual member tumbles. |
| **Finding 3** | The drift of Finding 1 is specifically a **yaw random-walk**: roll and pitch are already held PID-tight forever; what "tumbles" is unobservable yaw slowly wandering across the 5° line (the WNN reads gyro+accel, from which absolute yaw is unrecoverable — the PID reads the true orientation). |
| **Finding 3(e)** | The *registered prediction* attached to Finding 3: give the controller a yaw reference (ANCH observation) AND train at the 2000-step horizon (→ the ANCH2K arm) and you should get a horizon-free single controller. Written down before the experiment ran, on purpose. **Refuted by Finding 6.** |
| **Finding 4** | None of the four GA-search-quality levers (immigrants, curriculum, threshold shaping, action-repeat) improves the @500 ceiling; the only levers with real signal are *what the controller observes* and *how long the training episodes are*. |
| **Finding 5** | How few input bits can the controller read and still work? Raw-state controllers (s16) have no floor down to 4 bits; PID-feature controllers (pidmix_pwm) cliff below ~12 bits. FPGA-relevant: observation width is nearly free on the raw substrate. |
| **Finding 6** | The C2K result: re-training all four observation families at the 2000-step horizon generalizes ONLY the pwm family (pooled 90.5±2.7, first single-controller recipe above 90). ANCH2K refuted (bimodal 60/87 across seeds), LEAN's 93.5 was seed luck, TILT didn't reproduce. |

## The Workstreams (W0-W5, docs/controller_research_roadmap.md — the pre-paper plan)

"W" = workstream. The roadmap written 02/07 when we decided the paper is 3-6 months out.

| Label | What it is |
|---|---|
| **W0** | The pipeline that was already running when the roadmap was written: E2 seed10s → low-edge rescue → **C2K** → committee assembly. Now essentially done (C2K complete; assembly = E4 step pending). |
| **W1** | "Understand the horizon gap." Three parts: (1) the **decay-law surface** — train at horizons {500, 1000, 2000, 4000} and evaluate each far past its horizon, to learn the *law* (is immunity always ~2.5× the trained length?); (2) drift-mode analysis (done — it produced Finding 3); (3) correlated-drift probe for committees. The **W1 surface driver** currently running is part (1): the H1000/H4000 cells that fill the two missing points of the surface. |
| **W2** | Disturbances — add wind, gusts, motor asymmetry, sensor noise to the sim and re-run everything. THE priority after W0: it's where integral action finally has work to do and where WNN-vs-PID comparisons become honest. |
| **W3** | Hybrid PID + WNN residual (= experiment E5): PID stays in control, the WNN learns a small correction on top (±10% PWM). Only compelling under W2's disturbances. |
| **W4** | Task validity: our sim is rotation-only; add translation so we can do position-hold / waypoint tasks like the RL-comparison papers (Sajus). May define the paper#1/paper#2 split. |
| **W5** | Realism ladder: W5a = Crazyflie-class physical parameters (cheap, in paper #1); W5b = PX4 software-in-the-loop (paper #2 bridge); W5c = the physical-drone question. |

## The Experiments (E1-E5 — the break-90 plan v2, .claude/plans/controller_break_90_v2.md)

"E" = experiment in the plan to break the ~90% single-controller stability ceiling.

| Label | What it is | Outcome |
|---|---|---|
| **E1** | Random immigrants: inject 15% random genomes per GA generation to fight premature convergence. | Shipped; alone it's a −2.7pp tax (Finding 4). |
| **E2** | The 6-arm lever sweep (IMM/LONG/CURR/ANCH/GAMMA/REP × 2 seeds = 12 cells) testing each search-reliability lever in isolation. | Done 03/07 → Finding 4. |
| **E3** | Threshold-gamma: concentrate thermometer thresholds near zero error (denser resolution where the controller lives). | Built; as an E2 arm it lost (−2.1pp). |
| **E4** | Best-of-K / committee assembly over existing winners: re-score candidates on FRESH seeds (the truth serum), then build mean-PWM committees. No new training. | Produced the 90.5% 5-member and 95.5% 7-member committees; the C2K-pool assembly is the pending next step. |
| **E5** | The PID+WNN residual hybrid. | Not started; folded into W3. |

## The C2K pool and the arm names

**C2K** = "Committee at 2K": re-train one member of each observation family at
**2000-step** episodes (hence 2K), two recipe seeds each (s09 = 20260609,
s10 = 20260610), so the committee can be built from non-drifters. The "2K" suffix on
an arm name just means "that family, trained @2000".

| Arm | Observation family it carries |
|---|---|
| **s16** | The baseline substrate: 16 raw thermometer-encoded states (gyro, accel, target) — no derived features. |
| **PWM / PWM2K** | s16 + a pwm accumulator observation (the running motor-command average — the one hand-crafted feature that keeps winning). |
| **TILT / TILT2K** | s16 + a lumped tilt-angle feature. Dead weight @500 (−14pp); looked rehabilitated @2000 on seed09; refuted by seed10. |
| **LEAN / LEAN2K** | The input-4 lean grid recipe from Finding 5 — tiny observation budget (4 input bits), tiny memories. |
| **ANCH / ANCH2K** | s16 + a yaw-anchor observation (a yaw reference that makes absolute yaw observable — the Finding 3(e) candidate). Bimodal: seeds either work (~87-91) or crater (~55-60). |
| **LONG** | Not a new observation — the s16 recipe simply trained @2000 in E2. It became the "ruler" (reference) for all @2000 comparisons and a committee member. |
| **IMM / CURR / GAMMA / REP** | The other E2 arms: immigrants 0.15 / difficulty-adaptive curriculum / threshold-gamma / action-repeat N=5. All refuted as stability levers (REP is a 4× compute saver though). |
| **pidmix / pidmix_pwm / pidmix_pwm_tilt** | Frame-fix-era families that feed hand-computed PID terms (P/I/D errors) as observations; 15/19/21 features. High variance, never beat s16 on the mean; pidmix_pwm is the one that peaks at input-12 in Finding 5. |

## Metrics, protocol, and jargon

| Term | Meaning |
|---|---|
| **stable%** | Fraction of held-out episodes where the craft never diverged AND mean attitude error ≤5°. The headline metric. |
| **err°** | Mean attitude error over the episode. |
| **steady°** | Mean error over the LAST 20% of steps — the drift-sensitive metric (a drifter looks fine on err° but bad here). |
| **horizon** | Episode length in steps @1 kHz (500 = 0.5 s). *Training* horizon = what the GA saw; *eval* horizon = what we test at. The whole point of Findings 1-3: these differ. |
| **ruler** | The evaluation yardstick a number was measured on (e.g. "the 2000-step ruler" = held-out episodes of 2000 steps). Numbers on different rulers are not comparable. |
| **beat-me (line)** | The incumbent score a challenger must exceed to claim a win — e.g. LONG@2000 = 88.2/2.77° for the C2K pool. Campaign slang; "reference baseline" in paper language. |
| **ho-neur / ho-mem** | Held-out score measured after the NEURONS GA phase (interim) / after the MEMORY phase (final). C2K showed ho-mem ≤ ho-neur in every cell — the gap is an overfit early-warning. |
| **fresh-seed (truth serum)** | Re-scoring a winner on 4 seeds never used in training, selection, OR reporting. Mandatory before believing any number; several report-seed "winners" crater on it. |
| **pooled (n=8)** | The 2 recipe seeds × 4 held-out seeds combined — the recipe-level number. Single-cell bests (like LEAN2K_s09's 93.5) are seed lottery until pooled. |
| **seed lottery / bimodality** | When a recipe's outcome depends on the GA random seed (e.g. ANCH seeds land 60 vs 87). Wide pooled SD (±29) is the fingerprint. |
| **cell** | One (arm × seed) training run in a sweep — e.g. C2K had 8 cells. |
| **sn** | State neurons — the recurrent layer width the GA settled on. |
| **cells (count, e.g. 108K)** | Number of trained memory addresses in the winner — its RAM footprint (sparse; this is the honest size metric). |
| **wish/sat** | Capacity pressure gauges (wish_bits / saturation). 0/0 everywhere = the ceiling is NOT capacity. |
| **grid → NEURONS → MEM** | The three phased-GA stages: grid search seeds the architecture, the neurons GA grows/wires it, the memory GA refines cell contents (Lamarckian). |
| **committee / mean-PWM** | Several controllers run in parallel; their 4 motor commands are averaged each step (a few adders on FPGA). Mean beats median here. |
| **drift / yaw random-walk** | The slow wander that kills long episodes — Finding 3 showed it's ~97% unobservable yaw, walking at a rate horizon-training slows but never zeroes. |
| **decay law** | The target of W1: the function relating training horizon H to how far past H stability survives (working estimate: ~2.5×H). H1000/H4000 fill in the curve. |
| **registered prediction** | A hypothesis written into the findings doc BEFORE the experiment runs (pre-registration), so refutations stay on the record — e.g. Finding 3(e) → refuted in Finding 6. |

## Infrastructure shorthand

| Term | Meaning |
|---|---|
| **driver** | A detached shell script (PPID=1, survives CLI restarts) that runs a sweep's cells one at a time — e.g. `scripts/c2k_driver.sh`, `scripts/w1_train_driver.sh`. |
| **marker** | The `/tmp/wnn_*_done.json` file a driver writes when finished; the next driver in the chain sits in a wait-loop on it. This is how experiments queue without a scheduler. |
| **chain watcher (PHASE 1/2/3)** | The session cron that reports on whichever chain link is live, deduced from which markers exist: PHASE 1 = low-edge rescue, PHASE 2 = C2K, PHASE 3 = W1 surface. |
| **one-controller-at-a-time** | The standing rule: never two controller trainings concurrently (they'd fight for GPU/CPU and corrupt timing comparisons). |
