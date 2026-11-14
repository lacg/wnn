# The armed status-tick cron prompt (verbatim)

The half-hourly tick runs from a **session-only** cron: it lives in memory, is
never written to disk by the runtime, and dies when the CLI exits. `/clear`
survives it; a restart does not. This file is the durable copy of the prompt
that is *currently armed*, so it can be re-armed verbatim.

`controller_status_tick.md` is the durable copy of the six-line FORMAT and its
rules — the parts that do not change. This file holds the VOLATILE half: which
chain is live, which arms have landed, the current results, what counts as an
escalation today. Refresh the STATE block here whenever the programme moves, and
re-arm the cron from it.

**Currently armed:** HOURLY `13 * * * *` (re-armed 20/10/2026 06:3x EDT after a CLI restart; Luiz: 1h ticks for controller + IDS). Armed text = the short pointer prompt in the 20/10 STATE block. Older: job `7f271521`, `13,43 * * * *`.
Re-armed 15/09/2026 16:05 EDT (Luiz: the box is NOT idle while the recalc/probe run — lines 2/3 now
report those controller-side jobs with progress + ETA; STATE block refreshed for the repair sequence,
the offline-connectivity probe and the post-repair queue). Previous: `82aa3883` (10:20), `5ed39033` (13/09).
NOTE 15/09: the armed prompt is the FORMAT + rules + the CURRENT (15/09) STATE block + a pointer to this
file for the provenance history — no longer the full 75 KB history verbatim. Everything that governs a
tick every time is in the armed text; the older QUEUE blocks are consulted from this file on demand.
Re-armed 13/09/2026 16:06 EDT after a CLI update (genuine restart — the documented killer). Previous: `da2eb7c7` (04/09).
Re-armed 04/09/2026 08:50 EDT after the power outage (CLI restart). Previous: `5ad9d655` (01/09).
Re-armed 01/09/2026 23:4x UTC after `649beddb` was lost. WHAT KILLED IT: the CLI was EXITED
and UPDATED to a new version, i.e. a genuine restart — exactly the case this file already
warned about. A `/model` switch happened to coincide, and an earlier note here blamed that;
it was wrong. Measured directly 20 minutes later: switching model again (Fable 5.1 -> Opus 5)
in a LIVE session left `5ad9d655` running. So the rule is unchanged — /clear and /model
survive, a CLI exit does not. STATE block current as of 01/09 23:0x UTC (the queue, the
leaderboard, the classical bar under --translation).

To re-arm after a CLI restart, pass everything below the line to CronCreate with
`cron: "13,43 * * * *"`, `recurring: true`.

---

Controller status tick. Check directly with Bash (do NOT spawn a subagent — a handful of files).

⚠️ SAY "RUN", NEVER "CELL". "cell" means a RAM memory cell in this project.

DISCOVER the live lever rather than assuming one — a previous cron went stale by hardcoding a finished arm. Do NOT hardcode chain names:

  TZ=America/New_York date "+%d/%m/%Y %H:%M:%S %Z"
  cd /Users/lacg/wnn
  pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" | wc -l   # LOGICAL runs — count the CHILD ONLY.
  # ⚠️ THE BRACKET IS LOAD-BEARING (11/09/2026). pgrep -f matches ANY command line containing the
  # literal — including THIS tick's own shell while it sleeps 4 s, and any long-running Bash
  # tool call that quotes the pattern. Every chain preflight uses the same pgrep as its idle
  # gate, so a monitoring command holding the literal in argv makes a chain ABORT "box not
  # idle" and the queue fail closed. That is exactly how the D0 A/B aborted at 14:38 EDT 11/09.
  # Write every monitoring pattern with a bracketed last character so it never matches itself.
  # Do NOT use the broad "-m wnn.control.phased_ga" here: it also matches the /usr/bin/time wrapper,
  # so ONE healthy run reports 2 and trips the ">1 controller running" escalation every tick. The broad
  # pattern belongs in the supervisors' kill/wait (the wrapper must not be invisible to a kill), not here.
  ps -axo pid,command | grep -E "scripts/.*(chain|driver|study|probe|handoff|supervisor|wide|queue)" | grep -v grep   # (grep -v grep drops this line's own shell; pgrep patterns above need the bracket)
  # No trailing \.sh — the supervisors are probe_handoff_supervisor.sh and sweep_ladder_probe_wide.sh,
  # neither of which ends in chain/driver/study/probe/handoff + ".sh". Anchoring on .sh hid BOTH of them
  # and made a healthy handoff look like a dead one.
  ls -dt experiments/*_markers | head -3 | while read d; do echo "$d: $(ls "$d" | wc -l)"; done
  NEW=$(ls -t logs/controller/*/*.out 2>/dev/null | head -1); echo "$NEW"; grep -aE "Gen [0-9]|GRID WINNER" "$NEW" | tail -1
  grep -ac "weight_alt > 0 but" "$NEW"      # MUST be 0 — escalate if not
  grep -cE "SIGKILL controller|SIGTERM graceful PAUSE" logs/controller/mem_watchdog.log
  vm_stat  # avail = (free+inactive+speculative+purgeable)*16384/1073741824
  sqlite3 "file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro" "select sum(status='completed'), sum(status='running'), sum(status='queued') from flows;"
  pgrep -f "wnn.ram.experiments.worker" | wc -l
  sqlite3 "file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro" "select substr(name,1,instr(name||'-','-')-1) pfx,status,count(*) from flows where status in ('queued','running') group by pfx,status;"

Output EXACTLY these SIX lines and nothing else. The STAGE gets its OWN line — standing requirement (Luiz, 16/08); do NOT collapse it into line 2:

[ctl DD/MM/YYYY HH:MM:SS EDT]  lever <name> · markers <n>/<total>
  run R/T  <tag>  (what this point tests, one clause)
  <STAGE> gen G/TOTAL  <X>s/gen  elapsed <T>  patience <used>/<max>
  elite: fit <best> · stable <S>% · err <E>° · steady <D>° · alt <A>m (best so far, during-search)
  gen:   stable <S>% · err <E>° · steady <D>° · alt <A>m (this gen's leader, during-search)
  box: <k> controller · watchdog <n> kills · avail <G> GiB · IDS <done>/<run>/<queued>

PATIENCE ON THE STAGE LINE (added 31/08/2026, Luiz). Read it off the gen line's
`(patience P/M, check every Cg)` and print it as `patience P/M`. It is USED/MAX, not
remaining — verified in generic_ga.py, where `_gens_left = (M - P) * C`, so `patience 1/5`
with `check every 2g` means one failed check spent, four left, ~8 generations before the
stage early-stops. P can be FRACTIONAL under --magnitude-aware-patience (e.g. `2.5/5`);
print it verbatim, do not round. Before the first GA gen line there is no counter yet —
print `patience —`. Do NOT confuse this with the controller's own magnitude-aware tracker,
whose display counts DOWN.

TWO GENOMES, TWO LINES. The .out gen line carries both blocks split by `|`:
  best=1.8737 (=), stable=55.00%, err=13.89°, steady=17.17°, alt=1.230m | gen: stable=0.00%, err=26.38°, steady=30.48°, alt=0.407m
· `elite:` = the INCUMBENT, fields BEFORE the `|` (five fields incl. fit).
· `gen:` = THIS GENERATION'S leader, fields AFTER the `|` (four fields, never five — it has no fitness of its own).
· They are DIFFERENT GENOMES on any `(=)` generation. Metrics are frozen per genome; if the blocks disagree that is two genomes, not one re-measured. Do NOT re-merge them.
· `(=)` does NOT mean idle — the fitness is a rank/z combine over the CURRENT pool, so it is not a fixed scale. Watching `gen:` regress is how population collapse shows before `elite:` moves.
· NEITHER is published. Stage-select ranks the union of the top-3 of EVERY stage on the val seeds.
GEN-LINE SEMANTICS CHANGED 20/09/2026 (commit on generic_ga.py; runs LAUNCHED after ~19:00 EDT
20/09 print the new form, older .out files keep the old one — tell them apart by the label after `|`):
  best=-1.1817 (new), stable=99.20%, err=1.90°, steady=1.28°, alt=0.175m | offspring: stable=98.80%, err=1.99°, steady=1.28°, alt=0.102m
· First block = pool[0], rank 1 of THIS generation's pool under the current frame — the genome that
  is published if the stage ends now (`_holdout_report` scores final_population[0]). `best=` is its
  score in this frame. `(new)` = rank 1 changed IDENTITY since the previous gen; `(=)` = same genome.
· `offspring:` = the best genome GENERATED this generation (elites excluded) — "what the population
  is producing now". On a plateau it trails pool[0]; watching it regress is population collapse.
· Print the first block as `elite:` and the second as `offspring:` (was `gen:`). The tag `(=)` in the
  OLD format was a z-frame artifact (the frozen `best_fitness` is not comparable across generations):
  16/48 ABI-30 gens printed (=) while a different genome held rank 1. In the old format read the
  `gen:` block as pool[0] and ignore `(=)`.

ALTITUDE IS NOT OPTIONAL IN A TABLE (06/09/2026, Luiz: "WHY didn't we follow the rule
of showing stable, err, steady AND ALTITUDE on all results?").
Every altitude-regimen row prints FOUR columns: stable % · err ° · steady ° · alt m
(pos m too when the run carries it). This is a STANDING rule and I broke it twice in one
session — once printing `alt —` for the A/B's OFF arm (correct: no z plant), and once
DROPPING the column from the leak comparison while the marker carried it.
WHY IT MATTERS, measured: in the leak screen the 0.90 arm wins stable/err/steady and
LOSES altitude (0.626 m vs the control's 0.515 m). gate-dist cannot see alt, so the
truncated table read as a clean sweep when the run actually TRADES ~0.11 m of altitude
hold for its attitude gains. The column is where the trade lives.
· A run with NO z plant prints `alt —` plus the reason, never 0.000 and never omitted.
· RANK on gate-distance, REPORT all four. A table with three columns is not a report.

SAME-SEED COMPARISONS: COMPARE STAGES, NOT HEADLINES (05/09/2026, Luiz asked
"same seed, what's the difference?" of two rows 7 ranks apart).
hd ranks the HEADLINE, and the headline is whatever stage-select crowned — the union
rank of the top-3 of every stage on the VAL seeds. That is a draw, not a measurement.
WORKED CASE (seed 31337002, b32 n256, byte-identical recipes, only the scorer differs):
    run                  MEMORY multiseed held-out          headline stage   hd
    TAB_on s2 (CRN)      99.8±0.4 / 1.58±0.16 / 1.14±0.09   CONNECTIONS#2    0.1442 (9th)
    SL_C   s2 (rotation) 99.8±0.4 / 1.59±0.14 / 1.13±0.25   MEMORY#0         0.1129 (2nd)
Same jerk (0.0159), same mono_viol (7): the two searches converged on the SAME
controller. Seven ranks of hd is one val draw. The scorer's real effect is one stage
earlier — GRID held out at 97.0±2.3% / 1.73° under CRN vs 43.2±11.4% / 9.01° under
rotation (the rotation grid winner was a lucky-pool artifact).
RULE: whenever two runs share a seed and shape, print the PER-STAGE rows and name each
headline's stage-select genome. NEVER report a same-seed rank gap as a difference in
performance when the MEMORY rows agree — say "selection draw" and show both.

NEVER SECOND-GUESS THE HEADLINE WITH TEST SEEDS (08/10/2026, Luiz). Stage rows (`[test-seeds] <STAGE> MULTI-SEED`)
and the HEADLINE row are BOTH test-seed numbers; stage-select picked the headline on the VAL seeds
(`stage_select_candidates`, 'val ...'). Saying another stage's genome 'should have won' or calling the headline a
'selection draw' BECAUSE its report row is worse = selecting on the test seeds (the leak stage-select exists to
prevent). WORKED CASE _tw15 s2: NEURONS#2 val 99.2/1.30/0.64 (fit −0.847) beat NEURONS#0 val 99.2/1.33/0.68
(−0.675); on test seeds #2 = 96.4±3.4, #0 = 99.2±0.7. The published row is 96.4. Allowed reading: 'near-tied on
val, split ~3 pp on report = val→report noise at n=1'. Never substitute the stage row for the headline.

NEW-MARKER FORMAT (05/09/2026, Luiz: "the new marker is difficult to find").
A banked marker is the POINT of the programme — it must not read as a run-on
sentence appended to a tick. When a marker lands, print the six lines, then a
BLANK LINE, then:

  >>> NEW MARKER n/T — <tag>   (banked HH:MM EDT, rc R, DDDDD s)

followed by TWO fenced tables, never prose (BOTH STICKS, Luiz 09/10/2026):
  1. STAGE-SELECT + TEST table — the two sticks side by side, each labelled:
       VAL    = the run's 5 val seeds — the RANKING stick stage-select used on the 9 candidates (top-3 of
                every stage). They DIFFER PER RUN (derived from the run seed): read them from the .out line
                "'val' = mean over 5 disjoint val seeds A..B" and print the range in the table header. Source: marker `stage_select_candidates` ('val' fields + fit).
       TEST = the 5 test seeds 99990101-05: fresh held-out, the PUBLISHED stick. Source: marker
                `held_stage_multiseed` (each stage's pool[0]) and the .out line
                `[test-seeds] HEADLINE-<GENOME> MULTI-SEED` (the headline, mean±SD).
     One row per candidate, ALL 9, ordered by val fit (lower = better), columns:
       candidate | VAL stable % · err ° · steady ° · alt m · fit | TEST stable % · err ° · steady ° · alt m
     TEST cells exist only for each stage's pool[0] (= STAGE#0) and the HEADLINE; print — elsewhere.
     Mark `<- HEADLINE`; mark duplicates `(= NEURONS#0)` (MEMORY#0 often IS NEURONS#0).
     The val column is why the headline won; the report column is what it costs on fresh seeds.
     NEVER compare a val number to a report number, never re-pick on report (see the rule above).
     Template (_tw15 s2; its val seeds were 1029590071-75):
       candidate   VAL stable err   steady alt    fit     | TEST stable  err        steady     alt
       NEURONS#2       99.2   1.30  0.64   0.164  -0.847  |   96.4±3.4  1.73±0.44  1.18±0.58  0.224±0.054  <- HEADLINE
       NEURONS#0       99.2   1.33  0.68   0.155  -0.675  |   99.2±0.7  1.45±0.24  0.87±0.10  0.196±0.037
       MEMORY#0        99.2   1.33  0.68   0.155  -0.675  |   (= NEURONS#0)
       GRID#2          98.4   1.38  0.76   0.148  -0.259  |   —
       ...
  2. The PAIR / REPLICATION table (and the per-seed LADDER table for `_tw` markers): one row per
     comparable run, columns: VAL headline stable % · err ° | TEST headline stable % · err ° ·
     steady ° · alt m · hd · rank, with a one-clause note per row. The TEST row is the published
     result; the VAL pair is the headline's own selection score (from that run's
     `stage_select_candidates` `<- HEADLINE` entry). hd/rank are TEST-stick.
Then at most two sentences of reading, and the commit hash. Column alignment is
the requirement — a reader must be able to scan down one metric.
ALWAYS carry the caveat the study needs (the A/B's 4-flag bundle; the 90.8-98.0%
seed band; n=1 is a direction). NEVER call a row a record without checking the
leaderboard rank.

MISSING VALUES — never invent one, never print 0.000 for "not measured" (a zero altitude reads as perfect altitude hold). Use an em dash.
· No `| gen:` block → `gen: — (pre-b872ba57 run)`.
· Before the first GA gen line, read fit/stable/err/steady/alt off the GRID WINNER line and tag `elite:` `(grid winner, during-search)`; that line prints the fitness FUNCTION but no VALUE → `fit —`, and `gen: —`.
If NO chain and NO controller are running, say so plainly on lines 2-3 and name what is pending.

STATE (01/09/2026 23:0x UTC — refresh this block when the programme changes).

AS OF 14/11/2026 16:30 UTC — RIVAL RULING ADOPTED (Luiz 14/11; docs/ctrl_rival_row_ruling.md ca05d043; TEST confirmation f6e084e8). Controller tables carry THREE reference rows, in order: `oracle-fed (label source, bound) 100.0 / 0.69 / 0.16 / 0.357 / 0` · `MPCOF equal-info (COMPARATOR) 100.0 / 0.69 / 0.17 / 0.357 / 0` · `MPCOF no bias comp 100.0 / 1.35 / 0.68 / 0.341 / 0`; gaps are quoted vs the COMPARATOR; never write matches / on par / teacher parity / 100% stable / cheaper than MPCOF. This SUPERSEDES the single 'MPCOF 100.0 1.35 0.68 0.341 0' reference row of the 12:10 entry. BLOCK 10 APPROVED (Luiz): levers L1 (tilt-reference attitude error input), L2 (yaw ladder dense near zero), L3 (altitude integrator, applied to label + all rivals) + hold-window pos column — flight-dynamics agent implementing default-off on branch ctrl-block10-levers, ABI 51 wheel BUILT NOT INSTALLED (block 9 + stress test need ABI 50); install + prereg after the stress test. pos is task-defined (mean distance over the 2 s episode from a ~0.42 m start; floor 0.34-0.36 m for any controller) — make no pos claim.
AS OF 14/11/2026 15:30 UTC — TEACHER PROBE RESULT (docs/ctrl_indi_teacher_spec.md §10-11, VAL): INDI route CLOSED (Luiz). Bias-compensated MPCOF = 0.69° err / 0.17° steady / 0.355 m on VAL (shipped rival 1.35/0.68/0.339) = EQUAL-INFORMATION rival; the shipped MPCOF row lacks the bias info the student has since block 7. Luiz approved: one-shot TEST confirmation of the compensated rival (flight-dynamics agent, running), then experiment-design decides the paper's rival row; then investigations of the student's ~0.6° gap to its oracle-fed label source and of the ~0.35 m pos (task-level, MPCOF itself 0.34 m) -> docs/ctrl_gap_after_block9.md. Until the test confirmation lands, the MPCOF reference row in tables stays the shipped one, LABELLED 'shipped rival (no bias info)'. Stress test (300 units) tooling ready (docs/ctrl48_stress_test_plan.md): launch after the block-9 decision on the winning arm's 10 controllers: `bash logs/controller/ctrl48/stress_test.sh <list> 300` detached, then stress_result.py.
AS OF 14/11/2026 12:10 UTC — REPORT FORMAT (Luiz 14/11: standardize): every paired/head-to-head controller table uses the COMPACT PAIR form — one row per seed (ALL seeds of the comparison, e.g. s1-s10), columns `stable % | err ° | steady ° | pos m | fails` each as `A | B`, then `Δ (B − A): err · steady · pos · fails`; a pool line underneath (`running pools <range>: A x/N · B y/N (gate ceiling ...)`). ALWAYS a legend line ABOVE the table naming A and B (Luiz 14/11), e.g. `A = dc_ms01 · B = both`; and directly under the header a REFERENCE row for the MPCOF cascade (Luiz 14/11): `MPCOF  100.0  1.35  0.68  0.341  0` (same on every seed; docs/ctrl48_fixed_scorer_table.md). Example:
  A = dc_ms01 · B = both
  seed   stable %        err °          steady °       pos m           fails   Δ (both − dc_ms01): err · steady · pos · fails
  s4     99.9 | 100.0    1.44 | 1.40    0.94 | 0.83    0.357 | 0.358   3 | 1   −0.04 · −0.11 · +0.001 · −2
AS OF 14/11/2026 11:50 UTC — IDS58 RUNNING: VAL gate PASS (303e0c5f; vote VAL gain at 64b only +0.04 pp -> realistic AMBIGUOUS/REJECT prior), W3v vote-path certification BIT-IDENTICAL + independent re-run (f280df44). 7549 paused 11:44Z; 7567-7569 released (7567 running since 11:44:27Z, vote/0.02/bal=false/max_bits 64 confirmed in worker log). IDS BLOCK cohorts: `IDS58 n/3` + paused IDS56C/IDS56UC. Cost alert one-shot 15/11 02:47 EST (20 h). Block 9 ticks show ALL seeds s1-s10 per arm (Luiz 14/11), decision on ALL s1-s10 (prereg Amendment 2; dc_ms01 s1-s3 re-picked under P after BLOCK9 DONE, ends "BLOCK9 REPICK DONE").
AS OF 14/11/2026 11:35 UTC — IDS RE-PLAN (Luiz 14/11): CLOSE THE CONFIG ON THE FASTEST DATASET (unswr) FIRST, then port the final config to cicids/ciciot and validate once. ids_sequencer.sh STOPPED; 7549 (IDS56UC cicids nb02 encon r20419, 4.5 h in) PAUSED by POST /pause — resume later on the SAME worker ABI; all IDS56C/IDS56UC flows stay paused (their design may need re-basing on the final unswr config — ask Luiz before releasing). IDS58 (vote+b64, docs/ids58_vote_b64_prereg.md) decisions approved 1,2,3,5: primary = combined vs A1; VAL gate (ids-security, running) must PASS before the 3 flows are created; vote-path ABI-29 certification (QA, running) for the secondary; headline rule fixed (64b not earning its cells -> IDS56U headline, combined = max-accuracy row). Worker idle BY DESIGN until IDS58 flows exist — do not escalate 'worker idle with paused flows'. Then: 100-bit follow-up (prereg needed).
AS OF 13/11/2026 22:50 UTC (amended 23:10: s4-s10 = 14 runs, Luiz) — BLOCK 9 ARMED (Luiz 13/11): HEAD-TO-HEAD dc_ms01 vs both on NEW seeds s4-s10, both with rule P, ABI 50 (prereg docs/ctrl48_block9_prereg.md). logs/controller/ctrl48/ab_block9.sh (detached) waits 'S5P DONE', then 6 runs seed-major: tags C-S2-d_wposx2_a50_rf_c200_dc_ms01_P_full_s{4..10} and C-S2-d_wposx2_a50_rf_c200_dc_biasin_ms01_P_both_full_s{4..10}; END lines in runs.log; 'BLOCK9 DONE'. No same-seed control exists for s4-s10: report each END as the two arms side by side for that seed (stable/err/steady/pos/fails), and after s10 the prereg decision (gate on s4-s10 pooled, <=12/24500; both only if steady better by >0.03° and err not worse by >0.03°; ties -> dc_ms01). This SUPERSEDES the 21:25 'both = base for block 9' wording: block 9 decides the base.
AS OF 13/11/2026 21:25 UTC — BLOCK 8 DONE (docs/ctrl48_block8_result.md, 73274406): `both` (= --yaw-centred-ladder --yaw-integral-leak 0.9995 on dc_biasin_ms01, rule P) PASS, steady −0.10° at equal err/fails: 0.77/1.37/2 per 10500 = new best. Luiz 13/11: `both` is the BASE for block 9 (arms TBD with Luiz; tag prefix C-S2-d_wposx2_a50_rf_c200_dc_biasin_ms01_P_both; argv source = each seed's block-8 _P_both .out; keep rule P).
AS OF 13/11/2026 19:40 UTC — S5p ARMED (Luiz 13/11: run on the 14 block-8 runs that exist; prereg Amendment 3). logs/controller/ctrl48/select_tail_s5p_after_block8.sh (detached PID 91542) waits 'BLOCK8 DONE', then S5p x44 (30 old + 14 block-8) then B x14 (block 8, descriptive) via select_tail_recalc.sh with SKIP_RESULT=1 (the committed P-vs-B readout is never overwritten). Progress: `RECALC k/44` then `RECALC k/14` lines in runs.log; ends 'S5P DONE'. Line 2 during it: 'S5p re-pick k/58 (substitute test: can 5 units + p99 replace P?)'. Readout by the main session per Amendment 2 (S5p vs P; for block 8, P = each run's own headline).
AS OF 13/11/2026 13:40 UTC — IDS56UC QUEUED (Luiz approved prereg docs/ids56u_crossdataset_prereg.md: rescaled thresholds + grouped order). 18 new flows 7549-7566 (cicids/ciciot x {nb02-encon, lattice-encoff, nb02-encoff} x r20419-21) created PAUSED. logs/ids/ids_sequencer.sh (detached PID 53792, log logs/ids/ids_sequencer.log, order logs/ids/ids_sequence_1311.txt) REPLACES ids57_a1_before_carryover.sh + ids56_theta_then_backlog.sh (both stopped): after A1 7546-7548 it releases ONE flow at a time: cicids ON (7531,7549,7533,7552,7535,7555) -> cicids OFF (7550,7551,7553,7554,7556,7557) -> ciciot ON (7537,7558,7539,7561,7541,7564) -> ciciot OFF (7559,7560,7562,7563,7565,7566) -> ms01 (7532..7542 even) -> backlog 59. Paused flows are BY DESIGN. Escalate `!!` on any '!!' line in ids_sequencer.log or if the sequencer dies. IDS BLOCK cohorts add `IDS56C n/12 · IDS56UC n/18`. When a dataset x dedupe cell's 6 flows are terminal, the main session reads it out per prereg §4 (same worker_abi across each seed group). CONTROLLER: ileak s3 SKIPPED (Luiz 13/11; GATE-FAIL 7/7000 after s2) -> block 8 = 14 runs.
AS OF 13/11/2026 09:15 UTC — DEPLOY COMPLETE (worker ABI 29 + controller ABI 50). 7545 ended 08:5x UTC; worker_swap stopped the worker cleanly (nothing interrupted); DB backup db/wnn.db.pre-abi29-20261113 (quick_check ok); merge 03fecd97 pushed; W2 / W2b / W3 ALL BIT-IDENTICAL (docs/ids57_abi29_equivalence.md); worker relaunched AFTER the checks (PID 3161, PPID 1, env RAYON 13 / EVAL_IN_PLACE 1); 7546 (A1 s1) admitted 09:08:54 UTC on ABI 29, max_bits 64. IDS BLOCK ABI line = 29. IDS56U 3/3 DONE -> its TEST readout is due (main session, prereg docs/ids56_unweighted_vote_prereg.md, paired vs 7509/7513/7517). COST STOP: 7546 GA started ~09:09 UTC -> alert Luiz `!!` if its GA is still running at 14/11 05:09 UTC (00:09 EST); one-shot session cron armed for that. select_tail_recalc.sh + ctrl52_bitflip.py asserts now ABI 50 (main tree merged; block 8 keeps running from the worktree = same code).
AS OF 13/11/2026 05:25 UTC — 7544 DONE (4.5 h); 7545 = the LAST flow before A1. worker_swap.py --watch-flow 7545 --no-restart ARMED (detached PID 95516, log logs/ids/worker_swap_abi29.log, marker experiments/staged/swap_abi29.json): when 7545 ends it stops the worker so 7546-7548 cannot start on ABI 28. Worker DOWN after that is BY DESIGN until the main session finishes the deploy (backup, merge, worker wheel ABI 29, relaunch, W2/W2b/W3) — do not escalate 'worker down' while the marker exists and the deploy is in progress; DO escalate if the worker is still down 2 h after the marker. If 7546 got admitted before the stop, it must be restarted FROM THE BEGINNING on ABI 29 (never from an ABI-28 checkpoint).
AS OF 13/11/2026 03:40 UTC — BLOCK 8 FLYING (Luiz 12/11 22:30 EST: start now, controller-only install). ram_controller ABI 50 INSTALLED in the shared venv (swap-free; ram_accelerator md5 unchanged, worker untouched). Driver logs/controller/ctrl48/ab_block8.sh (detached PID 94226, stdout ab_block8.driver.log) runs the branch code from the worktree (PYTHONPATH=.claude/worktrees/agent-a162d4e33a11c36bb/src) until the full deploy merges it; checks ABI 50 + branch facade before EVERY run ('BLOCK8 ABORT' otherwise). 15 runs SEED-MAJOR: tags C-S2-d_wposx2_a50_rf_c200_dc_biasin_ms01_P_{ctl,ileak,ycl,yileak,both}_full_s{1,2,3}, all with P; END lines in runs.log; ends 'BLOCK8 DONE'. Smoke PASS (ABI 50 flags-off reproduced block-7 s2 exactly). Report each arm END PAIRED vs the SAME seed's block-8 _P_ctl (control re-flown on ABI 50), triple + pos + fails, FAILURE GATE per 11/11. Line 2: run k/15 + the arm's hypothesis (ileak: leaky rate integral cuts yaw offset? ycl: yaw-centred ladder? yileak: yaw-only leak? both: ladder+yaw leak?). ⚠ Until the merge, the MAIN-TREE controller facade asserts ABI 49 and will refuse ABI 50: any controller tool run with PYTHONPATH=src/wnn (select_tail_recalc.sh, ctrl52_bitflip.py) must wait for the merge or point at the worktree. Remaining deploy = worker wheel ABI 29 + merge at 7545's boundary (W2/W2b/W3), unchanged.
AS OF 13/11/2026 02:15 UTC — S5p SUBSTITUTE TEST PRE-REGISTERED (Luiz 13/11; docs/ctrl_select_tail_prereg.md Amendment 2): after block 8, re-pick S5p on the 30 old runs + 15 block-8 runs, and B on the 15 block-8 runs (~5.6 h, select_tail_recalc.sh with its ABI assert -> 50). S20 DROPPED. S5p replaces P only if total test fails S5p <= P AND mean err/steady not worse by > 0.05°.
AS OF 13/11/2026 02:00 UTC — Luiz 13/11: NO block-7 P re-pick (block-8 control measures it). S20/S5p secondaries run AFTER block 8 (more data: block-8 checkpoints can be re-picked too); scope (S5p only vs both) + any substitute-rule criteria pending Luiz, to be pre-registered BEFORE they run.
AS OF 13/11/2026 01:25 UTC — SELECT-TAIL DONE, rule P ADOPTED (pre-registered 3/3: test fails 57 vs 76, Δerr +0.001°, Δsteady +0.015°, 19/30 picks differ; docs/ctrl_select_tail_result.md, 7125992c). Luiz 13/11: BLOCK 8 ADDS P = `--val-units 20 --select-tail p99` to ALL 15 runs (control + 4 arms; selection only, the search is unchanged; ~+10 min/run). Deploy gate now = 7544-7545 terminal only (select-tail condition met). S20/S5p secondaries: pending Luiz.
AS OF 12/11/2026 15:45 UTC — IDS-57 A1 QUEUED (Luiz 12/11; prereg docs/ids57_a1_prereg.md, experiment-design GO-WITH-CHANGES): flows 7546-7548 = unswr lattice encoff r20421-23 with max_bits 64 only, paired vs 7509/7513/7517; run right after IDS56U. logs/ids/ids57_a1_before_carryover.sh (detached) keeps the carry-over 7531-7542 paused until 7543-7548 are terminal. IDS BLOCK cohorts add `IDS57 n/3`. DEPLOY GATE gains a 4th condition: W1 on file (experiments/ids57_abi_equivalence/w1_abi28.json + docs/ids57_abi29_equivalence.md, QA agent) BEFORE the worker swap; after the swap the main session runs W2/W3. All three A1 flows must run on ONE worker ABI: if 7545 ends before the deploy gate is ready, the main session pauses 7546-7548 (POST /pause) until the swap, then /resume. COST STOP (prereg): if the FIRST A1 flow's GA is still running 20 h after its start, tell Luiz with `!!` — pausing the still-queued A1 flows needs his order (not pre-authorised); never pause the running one.
AS OF 12/11/2026 14:40 UTC — DEPLOY APPROVED (Luiz 12/11, 'yes to both'): branch unify-write-rules-core (write-rule merge + yaw flags --yaw-centred-ladder / --yaw-integral-leak, both default-off) ships worker ABI 29 + controller ABI 50 together. GATE = all of: 'SELECTTAIL DONE' (the select-tail driver was RESTARTED 14:41 UTC with rules "B P" only = 60 re-picks, PID 72039; progress `RECALC k/60`; it writes + commits docs/ctrl_select_tail_result.md itself; the S20/S5p secondaries run AFTER the deploy on ABI 50 — the driver asserts ABI 49, so the main session relaunches it for "S20 S5p" with that assert updated) AND IDS56U 7543-7545 terminal AND the yaw-flag commit landed + parity green. When the gate holds the MAIN SESSION (not the tick) deploys at the next IDS flow boundary: worker_swap.py --no-restart -> backup DB -> merge -> maturin install worker + controller -> relaunch worker (env RAYON_NUM_THREADS=13 WNN_EVAL_IN_PLACE=1) -> resume/restart any interrupted flow -> verify 29/50 -> ABI-50 smoke. Tick line: say `deploy gate: <which conditions still open>`. After deploy: block 8 on ABI 50 (Luiz 12/11: 5 arms x s1-s3 = 15 runs, SEED-MAJOR) = the base control re-flown + arm1 (+--integral-leak 0.999) + arm2 (+--yaw-centred-ladder) + arm3 (+--yaw-integral-leak 0.9995) + arm4 (+--yaw-centred-ladder --yaw-integral-leak 0.9995; pre-check says arm 3 pays off mainly with arm 2), base = dc_biasin_ms01 (Luiz 12/11; arms paired vs that base re-flown on ABI 50). Block-8 driver must check ram_controller ABI before EVERY run and write 'BLOCK8 ABORT' on a mismatch or import error (hardening, Luiz 12/11).
AS OF 12/11/2026 12:30 UTC — SELECT-TAIL ARMED (Luiz chose option a): prereg amendment 1 (d008dfcd) = 30 runs (refit100 x3 dropped, cannot be re-picked). logs/controller/ctrl48/select_tail_after_block7.sh (detached) waits 'BLOCK7 (DONE|ABORT)' then runs select_tail_recalc.sh on experiments/ctrl_select_tail/runs_blocks45_a49_amended.txt, rules B P S20 S5p rule-major = 120 re-picks (B+P first, ~10 h; secondaries ~10 h more). Report progress from runs.log `RECALC k/120` lines; 'SELECTTAIL DONE' -> docs/ctrl_select_tail_result.md. The older Part-1-HELD text below is superseded.
AS OF 12/11/2026 12:10 UTC — CTRL-48 BLOCK 7 FLYING + SELECT-TAIL PREREG HELD (Luiz approved both 12/11): driver logs/controller/ctrl48/ab_block7.sh (detached PID 76135, stdout ab_block7.driver.log), ONE arm rf_c200_dc_biasin_ms01 (= dc + --obs-att-bias-pred --write-rule vote --vote-share 0.1; tests: does biasin's steady gain + ms01's fewest-fails survive the combination?) x s1-s3, argv = the block-6 dc control's own [argv] + the 2 flags; markers $L/C-S2-d_wposx2_a49_rf_c200_dc_biasin_ms01_full_s{1,2,3}.done, END lines in logs/controller/ctrl48/runs.log, ends 'BLOCK7 DONE'/'BLOCK7 ABORT'. Report each END in the block-6 table format PAIRED vs the SAME seed's block-6 C-S2-d_wposx2_a49_rf_c200_dc_full_sN END row (control NOT re-flown), triple + fails, gate per the 11/11 FAILURE GATE. PART 1 (docs/ctrl_select_tail_prereg.md, P=--val-units 20 --select-tail p99 vs B) is NOT running: the 3 refit100 runs of the 33-run prereg set cannot be recalculated (phased_ga refuses --recalc-headline with --threshold-refit-from-student), so the set needs Luiz's ruling; driver ready = logs/controller/ctrl48/select_tail_recalc.sh <run-list> (list experiments/ctrl_select_tail/runs_blocks45_a49.txt); when it runs, report 'select-tail k/N re-picks' from the latest `RECALC k/N` line in runs.log, end 'SELECTTAIL DONE' -> docs/ctrl_select_tail_result.md.
AS OF 11/11/2026 (cut-over) — VOCABULARY (Luiz 11/11, docs/terminology_train_val_test.md): val = tuning/selection (stage-select val units), test = the published numbers. The controller flags are --test-seeds / --test-seed / --test-episodes and new .out files print `[test-seeds] <STAGE> MULTI-SEED` / `[test-seeds] HEADLINE-<GENOME> MULTI-SEED`; .out files written before the cut-over print `[report-seeds]` — grep `\[(test|report)-seeds\]` to read both. Tables label the two sticks VAL and TEST (was VAL and REPORT). IDS readouts query `test_summaries` (was `validation_summaries`; column `test_point`). Never relabel a val number as a report number.
AS OF 11/11/2026 08:40 EST — FAILURE GATE (Luiz 11/11): every controller readout at n=3 applies, IN ORDER: (1) SAFETY GATE on stable_fail pooled over EVERY fixed-scorer run of the same configuration across blocks (a45 runs re-scored on ABI 47 + native a49 runs; e.g. dres = 6 runs / 21,000 episodes; Luiz 11/11): PASS only if the pooled RATE <= 0.05% (0.05% ceiling; Luiz's goal is MPCOF's 0 — the ceiling may tighten to 0) AND not significantly worse than the same-block control (exact one-sided Poisson/binomial test, p<0.05); (2) arms that pass are RANKED on steady then err; (3) the fails column is always shown. Mark gate failures `GATE-FAIL` in the verdict line. GATE TABLE FORMAT (Luiz 11/11, liked it): one row per configuration, sorted PASS first (by steady) then GATE-FAIL (by rate): `arm | runs | fails/episodes | rate | mean steady | mean err | verdict`, verdict for passers = `PASS, beats/ties/loses on steady (±x.xx°)[, better/worse err]` vs the pooled dres control (±0.03° = tie). Applied to date: block-4 quad 8 / lv32 8 / mlab 44 = GATE-FAIL, ms01 1 / biasin 4 pass; block-5 refit100 19 = GATE-FAIL, others pass; a49 dres control 2 pass. Before n=3, show the running pooled count vs the 5 ceiling.
AS OF 10/11/2026 18:30 EST — IDS56U QUEUED (Luiz 10/11, rec 2, n=3 first; prereg docs/ids56_unweighted_vote_prereg.md, experiment-design GO-WITH-CHANGES): 3 flows 7543-7545 (unswr, vote theta 0.02, balance_classes=false, encoff, r20421-23), controls = lattice 7509/7513/7517 (not re-flown). Hold script (PID see pgrep ids56_theta) now waits 7481-7545 before resuming the 59 held. FUTILITY GATE (main session, NOT the tick): when 7509/7513/7517 each land, run logs/ids/ids56u_gate/probe5.py on its GA genome; withdraw the arm if on >=2/3 VAL dF1<=0 OR VAL dFPR>+0.15pp — Luiz PRE-AUTHORIZED the cancel (10/11 19:00): if the gate fails, cancel 7543-7545 (PATCH status=cancelled) and tell him. IDS BLOCK cohorts add `IDS56U n/3`.
AS OF 10/11/2026 16:00 EST — BLOCK 6 GAINS the block-4 fixes ON dropconst (Luiz 10/11): logs/controller/ctrl48/ab_block6_extra_arms.txt adds rf_c200_dc_ms01, rf_c200_dc_lv32, rf_c200_dc_mlab, rf_c200_dc_biasin (quad stays OUT per the 11:30 entry) -> block 6 = 8 arms x 3 seeds = 24 runs, all paired vs the a49 rf_c200_dc control.
AS OF 10/11/2026 11:30 EST — quad OUT of block 6 unless s3 surprises (Luiz agreed 10/11): n=2 no steady movement (0.90/1.00 vs 0.90/0.99), more fails (6/2 vs 1/0), ~10x memory, ~2.3x run time.
AS OF 10/11/2026 07:55 EST — IDS-56 P1 CARRY-OVER QUEUED n=3 (Luiz 10/11: n=3 FIRST, n=5 only if ambiguous — standing rule): 12 flows 7531-7542 (IDS56C-cicids-* 7531-7536, IDS56C-ciciot-* 7537-7542; lattice vs ms01, both encon, r20419-21, random_3way). Luiz: carry-over runs BEFORE the held backlog — logs/ids/ids56_theta_then_backlog.sh (PID 52759) now waits 7481-7542 terminal, then resumes the 59 held. IDS BLOCK cohorts: `IDS56 n/50 (unswt 20/20, unswr n/20, theta005 n/10) · IDS56C n/12 (cicids n/6, ciciot n/6) · held 59`.
AS OF 10/11/2026 07:10 EST — IDS (Luiz 10/11): 20 duplicate IDS-52 lattice held flows CANCELLED (7390-7399, 7431/32/34/35/37/38/40/41/43/44) -> held backlog is 59. IDS-56 P2 queued: 10 flows 7521-7530 (unswt ms005 = vote theta 0.05 x encoff/encon x 5 seeds, pre-registered: adopt only if F1 paired >=4/5 vs ms01 without FPR loss). logs/ids/ids56_theta_then_backlog.sh (PPID 1) REPLACES ids56_after_round2c.sh: waits 7481-7530 terminal, then POST /resume of the 59 held. IDS BLOCK cohorts: `IDS56 n/50 (unswt 20/20, unswr n/20, theta005 n/10) · held 59`. Carry-over cohort (CICIDS + CIC-IoT, lattice vs ms01 encon) PENDING Luiz (n=3 vs n=5).
AS OF 10/11/2026 07:00 EST — BLOCK 6 ARMED (Luiz 10/11, Plane CTRL-50): logs/controller/ctrl48/ab_block6.sh (PPID 1) waits 'BLOCK5 DONE' (aborts on 'BLOCK5 ABORT'), ABI 49, tags C-S2-d_wposx2_a49_rf_c200_dc*_full_sN, seeds s1..s3 SEED-MAJOR, paired vs the same seed's a49 rf_c200_dc control (dres + --drop-constant-features, re-flown in the block): rf_c200_dc, rf_c200_dc_rateI (+ --obs-rate-i), rf_c200_dc_leak (+ --integral-leak 0.999), rf_c200_dc_rateI_leak (both); block-4 winners appended from ab_block6_extra_arms.txt ('name|flags' lines) before block 6 starts. Ends 'BLOCK6 DONE'. Table: a block-6 section.
AS OF 10/11/2026 00:15 EST — ABI 49 SUPERSEDES ABI 48 (Luiz 09/11 'fix every bug'): blocks 4+5 RELAUNCHED (block 4 PID 72605, block 5 PID 72607). Block 4 merges branch ctrl-refit-fix (dc86efd5 = everything in ABI 48 + the student-refit fix: Rust collector flies this plant, trained student, refit ladder reaches EVERY stage) and asserts controller ABI 49; ALL block-4/5 tags are C-S2-d_wposx2_a49_* (NOT a47/a48). Block 5 gains a 5th arm rf_c200_dres_refit100 (#2, --threshold-refit-from-student --threshold-refit-episodes 100) -> 15 runs. Flags-off is bit-identical to ABI 47/48.
AS OF 09/11/2026 23:30 EST — BLOCK 5 ARMED (Luiz 09/11): logs/controller/ctrl48/ab_block5.sh (PPID 1) waits 'BLOCK4 DONE' (aborts on 'BLOCK4 ABORT'), then 4 arms x s1..s3 SEED-MAJOR on ABI 48, tags C-S2-d_wposx2_a48_*, each paired vs the same seed's a48 rf_c200_dres (block 4): rf_c200_dres_oq02 (#1), rf_c200_dres_cent_oq02 (#1, centred ladder + q 0.02), rf_c200_dres_ct30 (#5, calib tilt 30), rf_c200_dres_tvxy045 (#4, training-only initial sideways velocity 0.45). Ends 'BLOCK5 DONE'. Idea #3 = offline re-rank via --recalc-headline (no GA run); idea #2 (student refit) NOT flown until its bug fix lands. Add a block-5 section to the table.
AS OF 09/11/2026 22:30 EST — BLOCK 4 NOW DEPLOYS ABI 48 (Luiz 09/11): ab_block4.sh (PID 95820) merges branch ctrl-dres-ideas (6dcf8f35 = CTRL-46 + scorer fix + --select-tail / --train-xy-offset / --train-init-vxy, all default-off, flags-off bit-identical to ABI 47) and asserts controller ABI 48; tags are C-S2-d_wposx2_a48_* (NOT a47). Block-4 WNN rows are fixed-scorer natively. The a48 rf_c200_dres control also serves BLOCK 5 (the 5 dres-failure fix ideas, docs/ctrl_dres_failures.md), which will be armed once docs/ctrl_dres_fix_offline_tests.md lands.
AS OF 09/11/2026 20:45 EST — oq02 s1 VOID (Luiz 09/11): --threshold-outer-quantile was dropped by the NEURONS/MEMORY/held-out threshold fits (fixed 8542f4c1), so rf_c200_dres_oq02 s1 flew dres's ladder (identical numbers). Files renamed *_VOID*. oq02 s2/s3 fly FIXED inside block 3; s1 is RE-FLOWN on ABI 45 by ab_block4.sh (PID 57990) right after 'BLOCK3dres DONE', before the ABI 47 deploy. Table: oq02 s1 = `void, re-fly queued` until it lands.
AS OF 09/11/2026 21:00 EST — TABLE NOW ON THE FIXED SCORER (Luiz 09/11; SUPERSEDES the 18:45 'keep old rows + one ABI 47 row' instruction): every block-2rf WNN row comes from docs/ctrl48_fixed_scorer_table.md (all 39 runs re-scored on controller ABI 47, headline genome as flown). Drop the separate 'ABI 47 re-score' row and the scorer footnote. Block-3 runs fly on ABI 45 so their runs.log END rows are PRE-FIX: print them marked `*` (pre-fix) until the main session re-scores them with scripts/ctrl_rescore_abi47.py (tick: do not run it). Block-4 rows (ABI 47) are fixed-scorer natively. Footnote: `all rows estimator-fed (onboard Mahony), 35 CF2.1 units; WNN rows fixed scorer (ABI 47), * = pre-fix; rivals CPU-scored`.
AS OF 09/11/2026 18:45 EST — GPU SCORER BUG + BLOCK 4 (Luiz 09/11): the Metal scorer indexed per-episode translation ICs chunk-local -> every GPU WNN row on ABI <= 46 saw only 25 distinct translation starts (rivals are CPU-scored, UNAFFECTED). Fixed = controller ABI 47 (branch ctrl-scorer-fix 7303ea0a). In the table: keep every runs.log WNN row as is (old scorer) and add ONE row right under rf_c200_dres: `rf_c200_dres (ABI 47 re-score)` s1 100.0 / 1.49 / 0.97 / 0.366 (0) · s2 99.9 / 1.65 / 1.11 / 0.364 (3) · s3 100.0 / 1.47 / 0.99 / 0.373 (0), and print under the table: `WNN rows = old GPU scorer (ABI <= 46) unless marked ABI 47; rivals CPU-scored`. BLOCK 4: logs/controller/ctrl48/ab_block4.sh (PPID 1) waits 'BLOCK3dres DONE', merges ctrl-scorer-fix + installs controller ABI 47 ('BLOCK4 ABORT' = escalate !!), then 6 arms x s1..s3 SEED-MAJOR tagged C-S2-d_wposx2_a47_*: rf_c200_dres (ABI 47 CONTROL) + quad (QUAD_WEIGHTED+antagonist), ms01 (--write-rule vote --vote-share 0.1), lv32 (--target-levels 32), mlab (--dagger-label-attitude estimate), biasin (--obs-att-bias-pred); ends 'BLOCK4 DONE'. Block-4 arms pair ONLY with the a47 dres control, never with a45 rows. Add them to the table as a block-4 section when they land.
AS OF 09/11/2026 14:45 EST — CONTROLLER BLOCK 3 ARMED (Luiz 09/11): logs/controller/ctrl48/ab_block3dres.sh (PPID 1) waits for 'BLOCK2rf DONE', then 3 arms × s1..s3 SEED-MAJOR, each paired vs the same seed's rf_c200_dres: rf_c200_dres_oq02 (+ --threshold-outer-quantile 0.02, GLOBAL ladder change), rf_c200_rateI_dhat (rate-I + raw d-hat, NO residual = mechanism test), rf_c200_dres_dropconst (+ --drop-constant-features). Ends 'BLOCK3dres DONE'. Tick line 1 `lever ctrl48 block 2rf|3dres — seed sN run k/K <tag>`. In the table add the 3 new arms under rf_c200_dres.
AS OF 09/11/2026 13:35 EST — CONTROLLER TABLE FORMAT (Luiz 09/11): on each END print ONE table with EVERY block-2rf arm that has run (all 10 arms + rf base as they land, not just the current seed or winners), columns s1/s2/s3 (`running`/`queued` for unfinished), cells `stable% / err° / steady° / pos m (fails)`, PLUS two teacher rows at the top: `MPCOF cascade (teacher)` (= MPCOF>rate(h10) in the code/JSON; Luiz 09/11: always call it 'MPCOF cascade') 100.0% / 1.35° / 0.68° / 0.341 m (0) and `MPCOF` 100.0% / 1.81° / 1.06° / 0.354 m (0) in every column, and (Luiz 09/11 16:30) four classical-baseline rows at the BOTTOM, same in every column (NO `[est]` suffix on any row — it only means estimator-fed, which EVERY row is; print once under the table: `all rows estimator-fed (onboard Mahony), 35 CF2.1 units, rivals re-scored on ABI 45 09/11`): `PID` 91.1% / 3.33° / 2.76° / 0.352 m (310), `LQI` 79.1% / 3.92° / 2.06° / 0.390 m (731), `LQR` 58.7% / 5.02° / 4.00° / 0.410 m (1447), `MPC` 58.3% / 5.05° / 4.03° / 0.411 m (1459) (35-unit scoring, experiments/ctrl_airframe_tail/rivals_35airframes.json, teachers are seed-independent). Source for WNN rows: grep ' END ' logs/controller/ctrl48/runs.log for C-S2-d_wposx2_a45_rf_*_full_sN and C-S2-d_wposx2_a44_rf_full_sN (base). Never state a mechanism as fact unless investigated.
AS OF 09/11/2026 10:40 EST — IDS-56 CHAIN ARMED (Luiz 09/11; 14:30 EST Luiz chose (a): unswr STAYS inside the hold, backlog resumes only after all 40 IDS56 flows, ~13/11 evening): logs/ids/ids56_after_round2c.sh (PPID 1, log logs/ids/ids56_after_round2c.log) REPLACED hold_for_ids55_round2c.sh. Sequence: round 2c (7476-7480) on ABI 27 -> worker_swap --stop-now to worker ABI 28 (wheel + experiments/staged/ids56_vote_share_abi28.patch in one step; marker experiments/staged/swap_abi28.json) -> queue 40 IDS56 flows (min-share vote θ=0.1 vs lattice × dedupe off/on; 20 unswt ~0.4 h then 20 unswr ~4 h) -> backlog (79, held_for_ids55_round2c.txt) RESUMED in id order only when IDS56 is terminal. The chain fails CLOSED: an [ids56] ABORT line = escalate `!!`, backlog stays paused. ON THE SWAP MARKER: the main session must COMMIT the applied patch (33 files incl. ABI 28 accel.py) — check `git status`. IDS BLOCK cohorts: `IDS55 r2c n/5 · IDS56 n/40 (unswt n/20, unswr n/20) · held 79`. Multiclass arm PARKED until after the paper (n=100).
AS OF 09/11/2026 08:05 EST — IDS: ROUND 2c (Luiz 09/11): unswt family 0.2 on families_v2 (protocol->context), flows 7476-7480, runs FIRST after the running flow; logs/ids/hold_for_ids55_round2c.sh (PPID 1) paused the other 79 queued flows (list logs/ids/held_for_ids55_round2c.txt) and RESUMES them in id order when 2c is terminal — paused flows BY DESIGN. Dashboard fixed + restarted (e4e8376f: PATCH queued clears pause_requested). Cross 0.4 deferred (Luiz: 0.2 first). IDS BLOCK cohorts: `IDS55 r2c n/5 (r run, q q) · held 79`. When 2c completes: per-seed table (absolute) off-500 | family v1 | family v2 | off-262 (F1/FPR/Acc · neurons/cells); harness predicted v2 gap ~-0.45 pp vs off-500.
AS OF 09/11/2026 04:35 EST — IDS: round 2a DONE (equal memory: spread 262n 90.53 F1 beats family 89.92 at 111.9K vs 111.8K cells; docs/ids55_ab_readout.md f53c18eb). Hold RELEASED: all 80 held flows queued again in id order (IDS-52/48c first, then the 10 CIC-IoT round-2b flows 7462/63/65/66/68/69/71/72/74/75). Release bug: PATCH queued left pause_requested=1 -> 7381-7406 self-paused; fixed via /pause+/resume. ESCALATE (`!!`) any flow that goes running->paused within minutes. IDS BLOCK cohorts line: `IDS52 n/N (r run, q q) · IDS48c ... · IDS55 r2b CIC-IoT n/10 (q)`. Pending Luiz: families_v2 follow-ups (cross 0.2 / 0.4 on unswt).
AS OF 08/11/2026 23:30 EST — IDS: IDS-55 unswt A/B DONE (family −0.83 pp F1 5/5, ~48% fewer cells; docs/ids55_ab_readout.md). ROUND 2 queued 7461-7475 (Luiz: "unswt now, CIC-IoT later"): 2a = unswt rf-off EQUAL MEMORY (7461 7464 7467 7470 7473, max_neurons_total 262) runs FIRST; logs/ids/hold_for_ids55_round2a.sh (PPID 1) paused the other 80 queued flows (IDS-52/48c + the 10 CIC-IoT 2b flows; list logs/ids/held_for_ids55_round2a.txt) and re-queues them in id order when 2a is terminal — paused flows are BY DESIGN, do not escalate. IDS BLOCK cohorts line: `IDS55 r2a n/5 (r run, q q) · IDS55 r2b CIC-IoT 0/10 (held) · IDS52/48c held 70`. When 2a completes: readout table per seed (absolute, side by side): round-1 off (213.5K) | round-1 family | 2a off-tot262 — F1/FPR/Acc + neurons/cells; question = family F1 per cell vs off at equal memory. A why-investigation (ids-security agent) writes docs/ids55_why_family_loses.md.
AS OF 08/11/2026 19:30 EST — RESULT FORMAT (Luiz 08/11: "show like this and not with a delta, it messes me up"): on each END print ABSOLUTE values side by side, NEVER delta columns — this OVERRIDES the cron prompt's 'PAIRED deltas'. Table: one row per arm (base rf 0.2 first), one column per seed, each cell `stable% / err° / steady° / pos m (fails)`; unfinished = `running`/`queued`. Prose may name direction ("0.2 best on err/steady"). Stable is NOT 'saturated' as a dismissal: the aim is MPCOF's 100% (PID 91.1%) — always show the fails count. IDS-55 A/B: flows 7452/7454 FAILED (GA seed projection with _rng=None, fixed 9517d9d3) and were RESTARTED from the beginning 00:25Z; IDS55 total 10. After the A/B: apply /tmp/ids55_merge_v2.patch (families_v2: handshake->timing, protocol->context) + commit.

AS OF 08/11/2026 16:45 EST — CTRL-45 RECEPTIVE FIELDS CONFIRMED n=3 (rf vs a44 base: +14.5 pp stable [+6.8,+22.1], −1.66° err, −1.13° steady, −0.07 m pos). RF IS THE NEW BASE; stable is saturated → RANK ON err/steady. Queue (logs/controller/ctrl48/runs.log): block 1b = ab_block1b.sh (rf-cross dose 0 / 0.4 × s1..s3, paired vs C-S2-d_wposx2_a44_rf_full_sN) → block 2rf = ab_block2rf.sh (waits 'BLOCK1b (rf-cross dose) DONE'): rf + ONE change × s1..s3 — dropconst, c200, c200_leak, c200_rateI, c200_dres, dropacc, b32, c200_tmix, c200_centred, c200_dhat (tags C-S2-d_wposx2_a45_rf_<arm>_full_sN), paired vs the same seed's a44 rf. The old spread block 2 (ab_block2.sh) was KILLED (Luiz: re-base on RF). Tick line 1 `lever ctrl48 block 1b|2rf — seed sN run k/K <tag>`; on each END print paired Δ vs that seed's a44 rf (err/steady first).

AS OF 08/11/2026 10:25 EST — ABI 44 DEPLOYED (b48efce8: CTRL-46 bank-segment neurogenesis, CTRL-47, CTRL-45 receptive fields). The CTRL-37/38 replication is DONE (n=3 s2/s3/s4: only leak999 helps, +2.5 pp). NEW LEVER = logs/controller/ctrl48/ab_base_rf.sh (detached): seeds s1..s3 SEED-MAJOR × {C-S2-d_wposx2_a44_base_full_sN, ..._a44_rf_full_sN (--receptive-fields axis --rf-cross 0.2)}, posx2, 35 units; results logs/controller/ctrl48/runs.log; 'BLOCK1 (base, rf) DONE' at the end. The a44 base is the pairing control for ALL new arms (ABI-45 calibration/feature flags off = bit-identical to 44). NEVER pair a44 rows with the ctrl37_38 rows (neurogenesis fix changed the baseline). ABI 45 (CTRL-48: --calib-episodes/--calib-regime/--calib-ladder, --drop-constant-features, --drop-features, --obs-cascade-dhat-residual) is being BUILT, not installed; it gets installed at a run boundary by the main session. Tick line 1 `lever ctrl48 block 1 — seed sN run k/2 <tag>`; on each END print paired Δ vs that seed's a44 base.

AS OF 07/11/2026 20:30 EST — REPLICATION QUEUED (Luiz, option A): logs/controller/ctrl37_38/replicate.sh waits for the s2 triple END, then flies seeds s3 then s4 (SEED-MAJOR) × {base C-S2-d_wposx2_full_sN, leak999, rateI, dhat, leak999_rateI, leak999_rateI_dhat} (posx2, 35 units), ~75 min each, ~15 h; markers 'REPLICATE seed sN DONE' / 'REPLICATE DONE' in logs/controller/ctrl37_38/runs.log. Tick line 1 `lever replicate — seed sN run k/6 <tag>`. When a seed block completes, print the per-arm table for that seed next to s2, and the running n=2/3 mean per arm.

AS OF 07/11/2026 18:45 EST — COMBO queued (Luiz): logs/controller/ctrl37_38/combo.sh waits for 'CTRL-37/38 posx2 DONE' then flies C-S2-d_wposx2_leak999_rateI_full_s2 (--integral-leak 0.999 --obs-rate-i, posx2, 35 units). Per-unit read (experiments/ctrl_airframe_tail/perunit_*.json, scripts/diag/ctrl_perunit_extract.py): leak999 rescues ROLL-heavy units (123/118/121 +14..+21 pp), rateI rescues YAW-heavy units (130 +47, 134 +22) — complementary; extreme-yaw units 115/132 barely move with either. When the combo ENDS: print the per-unit table for every unit <90% in any arm (control / leak999 / rateI / posx2 / posx2 arms / combo) with roll/pitch/yaw imbalance. Tick line 1 after the posx2 chain: `lever CTRL-42 combo (leak999+rateI)`, then `lever CTRL-43 triple (leak999+rateI+dhat)` (logs/controller/ctrl37_38/triple.sh, waits for the combo END; tag C-S2-d_wposx2_leak999_rateI_dhat_full_s2).

AS OF 07/11/2026 18:15 EST — CTRL-30 DECIDED (Luiz): posx2 weights (.2557/.2045/.3580/.1818) for CTRL-37/38. REPORT pos, NOT alt (Luiz 07/11: pos = mean 3-D position error, alt is one of its components) — tables are stable · err · steady · pos; alt only for runs without pos. Lever = logs/controller/ctrl37_38/chain.sh (detached): records the control-weights rateI END (pid 99722) into logs/controller/ctrl38/runs.log, then flies C-S2-d_wposx2_leak999 / _rateI / _dhat_full_s2 (posx2 weights, 35 units) -> logs/controller/ctrl37_38/runs.log. PAIR each posx2 arm vs C-S2-d_wposx2_full_s2 over 35 units: 83.1±26.3% / 3.67° / 2.20° / pos 0.431 (590/3500). Bonus control-weights set vs the 82.2% control: leak999 86.2±21.2% / 3.59° / 2.03° / pos 0.437 (483/3500); rateI pending. Tick line 1 `lever CTRL-37/38 posx2 — run k/3 <tag>`. The ctrl38/ab.sh waiter was stopped on purpose. Supersedes the 14:00 and 13:00 entries.

AS OF 07/11/2026 14:00 EST — CONTROLLER WHEEL ABI 41 INSTALLED (CTRL-38, e4e1005f): flags-off bit-identical (control re-scored over 35 units = 623/3500 MEMORY#2 under 40 and 41). Queue (all detached, each waits for the previous): CTRL-30 round A (ctrl30/chain.sh) -> CTRL-37 leak999 (ctrl30/leak999.sh) -> CTRL-38 A/B (logs/controller/ctrl38/ab.sh: C-S2-d_rateI --obs-rate-i, then C-S2-d_dhat --obs-cascade-dhat; results logs/controller/ctrl38/runs.log). CTRL-37/38 arms report over 35 CF2.1 units natively — compare to the control's 35-unit row 82.2±25.4% / 3.71° / 2.29° / alt 0.188 / pos 0.432 (623/3500). Tick line 1 for them: `lever CTRL-37 leak999` / `lever CTRL-38 A/B — run k/2 <tag>`; line 2 names the hypothesis (does a longer integral / the rate-PID integrator / the teacher's d-hat close the motor-wear tail). CTRL-39 (recorder train≠deploy bug, motor-level obs-pwm/obs-dhat lineages) awaits Luiz. Supersedes nothing; the 13:00 entry still governs round A.

AS OF 07/11/2026 13:00 EST — AIRFRAME TAIL (CTRL-36, paper-critical): each test seed = ONE Crazyflie 2.1 unit with its own ±20% per-motor wear draw (same model, cf21_brushless); the 5 test seeds are lenient units. Over 35 units (99990101-135) the C-S2-d control = 82.2±25.4% / 3.71° / 2.29° / alt 0.188 / pos 0.432 (623/3500 fails); PID[est] 91.1%, MPCOF 100%. CTRL-30 round A (logs/controller/ctrl30/chain.sh) is now READ over 35 airframes: logs/controller/ctrl30/rescore35.sh (detached) re-scores each arm via --recalc-headline as its .done lands -> 'RESCORE35 END <arm>' lines in logs/controller/ctrl30/runs.log. Then CTRL-37 = logs/controller/ctrl30/leak999.sh (detached; waits 'CTRL-30 ROUND A DONE'): C-S2-d + --integral-leak 0.999, reported over 35 airframes. A rescore runs a second phased_ga briefly — 2 controllers during a rescore is EXPECTED, not an escalation. On each round-A END print its 5-seed headline AND (once present) its RESCORE35 row next to the control's 82.2%/3.71°. CTRL-38 (cascade disturbance input) = design pending Luiz. Supersedes the 12:30 entry.

AS OF 07/11/2026 12:30 EST — CTRL-27 CLOSED: Luiz chose the CASCADE. CTRL-31 (tau re-score, logs/controller/ct40/ctrl31_tau.sh) runs first; then CTRL-30 ROUND A = logs/controller/ctrl30/chain.sh (detached, waits for 'CTRL-31 DONE'): 6 weight arms on the C-S2-d cascade recipe, s31337002, n=1, results logs/controller/ctrl30/runs.log (errx2, stabx2, steadyx2, posx2, pos0, equal; ~1.2 h each). Control = C-S2-d_full_s2 97.4%/2.86°/1.64°/alt 0.186/pos 0.409 (weights .28125/.225/.39375/.10), not re-flown. Tick line 1 `lever CTRL-31 tau re-score — k/5 <T>` then `lever CTRL-30 round A — run k/6 <tag>`; on each END print the headline next to the control. n=1 = direction only. Supersedes the 12:00 entry.

AS OF 07/11/2026 12:00 EST — CTRL-27 PAIRED n=1 on the STAGE-2 plant (lag 0.20, ABI 40, s31337002, test seeds 99990101-05). Cascade C-S2-d_full_s2 DONE: HEADLINE 97.4±3.8% / 2.86±0.24° / steady 1.64±0.11° / alt 0.186 m / pos 0.409 m (1 h 13 min). LEVER = logs/controller/ct40/full_M-S2-d.sh (pid 98195, PPID 1) -> phased_ga motor twin M-S2-d_full_s2 (log logs/controller/ct40/full/M-S2-d_full_s2.out), now in MEMORY stage-select/report scoring; END line -> logs/controller/ct40/runs.log. Then CTRL-31 (logs/controller/ct40/ctrl31_tau.sh, pid 71384) auto-starts on 'END M-S2-d_full_s2': re-scores the cascade at motor lag 0.20 (control, must reproduce 97.4%/2.86°), 0.15, 0.26, 0.30, 0.10 -> logs/controller/ct40/full/C-S2-d_tau<T>_s2.out; 'CTRL-31 DONE' in runs.log. Tick line 1 `lever ct40 CTRL-27 motor twin M-S2-d_full_s2 (<stage>)`, then `lever CTRL-31 tau re-score — k/5 <T>`. Tags use `<C|M>-<S1|S2>-<features>_<grid|full>_s<N>`. When the motor-twin HEADLINE lands: print it next to the cascade headline (stable/err/steady/alt/pos/jerk + duration); the architecture decision is Luiz's. Round 1 (CTRL-5) stays PAUSED; multi_axis_chain.sh still frozen at ABI 39.

AS OF 06/11/2026 18:40 EST — ROUND 1 PAUSED AGAIN (Luiz option b) for the ABI-40 grids. Controller wheel ABI 40 sha 154fb2423f9e299f (CTRL-34: cascade mpcof teacher syncs to the flown rate loop + --rg-gate). multi_axis_chain.sh is still frozen at ABI 39 — re-freeze before any relaunch. Anchor archived *.abi39_paused_for_abi40 (killed in population build, no gens). LEVER = logs/controller/ct40/chain.sh (detached), pop-50 GRID-ONLY, s31337002, results logs/controller/ct40/runs.log: C1dg/S1dg (cascade + delta law + gate off) -> M2g/SM2g (motor-level gate off) -> C1/C1g/C1d/S1 ablations. References: motor+law+asym M2 96.0%/2.57°, SM2 81.6%/4.02°; probe C1dg 96.4%, S1dg 79.0%. Tick line 1 `lever ct40 grids — run k/8 <tag>`.

AS OF 06/11/2026 17:15 EST — CTRL-5 ROUND 1 RELAUNCHED on the MOTOR-LEVEL path, ABI 39 (sha cd0e0407f4ac9e4e; provenance train_asym=per_episode:0.2, obs_pwm=delivered, arch=motor:teacher=mpcof) + lag law leak 0.7/dmax 0.6 on every condition. Main-session call under Luiz's never-idle rule (he had not answered). Lever = scripts/multi_axis_chain.sh --round 1 (MA_ANCHOR=arm, MA_AXIS_C_MAX_CELLS=1e9); log /private/tmp/multi_axis_chain.log; runs logs/controller/multi_axis/<tag>.out. Grid evidence: stage-2 motor+law+asym 81.6%/4.02° (passes gate). The ABI-37 symmetric-training anchor was archived as *.abi37_symtrain_paused. Cascade (CTRL-27) PARKED: WNN cascade 53-56% (_op31) / 26-35% (stage-2) at lag 0.20 regardless of 100/250 Hz or BINARY/QUAD (logs/controller/ct39/runs.log). Tick line 1 `lever multi_axis_chain (CTRL-5 round 1, ABI 39, pop 50) — run k/9 <tag> · banked n`. Supersedes the 15:40 entry.

AS OF 06/11/2026 15:40 EST — CASCADE VALIDATION (CTRL-27) on ABI 39 (sha cd0e0407f4ac9e4e; CTRL-32 per-episode training asymmetry + CTRL-33 MPCOF-over-closed-rate-loop teacher, both DONE). Lever = logs/controller/ct39/chain.sh (detached), pop-50 GRID-ONLY runs on s31337002, results in logs/controller/ct39/runs.log. Order: C1 cascade _op31 lag 0.20 -> S1 cascade stage-2 lag 0.20 -> C0 cascade _op31 lag 0 -> M2 motor+law lag 0.20 asym -> M0 motor lag 0 asym -> C1p / S1p PID-teacher ablations. References: motor-level _op31 lag 0 96.6%/1.60°, lag 0.20 0%/33.69°, +law 70.0%/4.55°; stage-2 motor+law 64.6%/5.42°. Rival bars (MPCOF>rate, L4C): _op31 lag 0.20 100%/0.562°; stage-2 lag 0.20 100%/0.895°. Tick line 1 `lever ct39 cascade validation — run k/7 <tag>`. Round 1 stays PAUSED.

AS OF 06/11/2026 10:40 EST — ROUND 1 PAUSED (Luiz 06/11: "understand why is more important than running experiments blindly"). Anchor stopped at NEURONS gen 8 (exit 143, emergency dump in logs/controller/multi_axis/ckpt/ma_anchor_s31337002 — resume with --resume-from-emergency <emergency_stage*.yaml.gz> --resume-mode same); chain killed. Gräfe τ CHECKED (paper-scout): sim τ_eff 49.5 ms vs Gräfe 50 ms = correct. LEVER NOW = logs/controller/op31_lag/run.sh (detached): the _op31 champion recipe on the ABI-31 plant (inclinometer, no drag, level ref, xy 0), ABI 37, grid only: R1 lag 0.20 -> R0 lag 0 (control) -> R2 lag 0.20 + lag law; results in logs/controller/op31_lag/runs.log. Tick line 1 `lever op31 lag isolation — run k/3 <tag>`; reference _op31 s2 ABI 31 grid 96.6%/1.96°/1.20°/0.118 m. Weight re-sweep under the new plant is PENDING the diagnosis (Luiz 06/11).

AS OF 06/11/2026 08:20 EST — IDS: IDS-47 n=3 DONE (7248, 07:06 EST); readout posted to Luiz (margin_val_cal, GA best_fitness), decision trim / n=5 / accept PENDING — the 89 paused IDS47 flows stay paused until he answers. IDS-48 RE-ARMED by the main session under Luiz's never-idle rule (worker was idle 1 h): 20 flows 7348-7367 queued 13:15Z (IDS48-calibfull/-calibsub200k-unswt-QUAD, paired n=10). Un-paused IDS47 flows (lower ids) will still run first (FIFO min(id)). IDS BLOCK cohorts line: `IDS47 31/120 (0 run, 0 q, 89 paused — decision pending) · IDS48 n/20 (...)`; ETA line per IDS48 from its own completed flows (print `ETA IDS48: — (no completed flow yet)` until one finishes). Pre-registered IDS-48 read lives in scripts/queue_ids_calib_paired.py (adopt iff dF1 and dFPR 95% CIs within ±0.10 pp).

AS OF 06/11/2026 05:15 EST — CTRL-5 ROUND 1 RELAUNCHED on ABI 37 (sha 24c868fd052e341d, obs_pwm=delivered) with the LAG LAW --delta-leak 0.7 --delta-max 0.6 on EVERY condition (25ce7f3a). Lever = scripts/multi_axis_chain.sh --round 1 (detached, MA_ANCHOR=arm, MA_AXIS_C_MAX_CELLS=1000000000 = axis C uncapped; sn8 still needs SN8_APPROVED). Log /private/tmp/multi_axis_chain.log; runs logs/controller/multi_axis/<tag>.out. Order: anchor -> L4A -> L4B -> lagoff -> T030 -> pid -> lqr -> cf2x -> sn4. Lag hunt (flight-dynamics agent, logs/controller/abi36_fix/): ABI 36 delivered-actuator obs-pwm; ABI 37 fixed 2 GPU-scorer bugs (MAX_FEATURES 21 < 26 overflow, collective_cmd hover-relative vs absolute) — EVERY earlier GPU-scored --translation marker (incl. CTRL-8) carries those bugs; new lag-0 grid ref 93.2%/2.94°/2.14°/0.233 m. Lag-0.20 anchor GRID 64.6%/5.42° (gate needs 70%) — flying anyway (never-idle rule); the anchor's full-pipeline headline is the real read. Tick line 1 `lever multi_axis_chain (CTRL-5 round 1, ABI 37, pop 50) — run k/9 <tag> · banked n`. Supersedes the 23:45 entry.

AS OF 05/11/2026 23:45 EST — ROUND 1 STOPPED (Luiz 06/11: stop, score mpcof, investigate first). Anchor grid winner at lag 0.20 was 0.0%/82.28°/111.16°/alt 1.714 m. Measured: MPCOF teacher at T=0.20 = 100% stable 1.79°/1.06° (experiments/ctrl22/rivals_abi35_ma_anchor_lag020.json, 6c4e1fdb) — teacher is fine; ABI 35 at --motor-lag-s 0 reproduces ABI 33 arm_kd078's grid bit-exactly (89.4%/3.13°/2.52°/0.159 m) — no regression. => the WNN student collapses under lag. A flight-dynamics agent is diagnosing (observability of the actuator state / label vs lag / lag-path bug). It may run ONE short controller diagnostic at a time (grid-only ~16 min or pop-6 smokes) — report those as `lever lag-collapse diagnosis (flight-dynamics agent)`. RELAUNCH after the fix with axis C UNCAPPED (Luiz 06/11): `MA_ANCHOR=arm MA_AXIS_C_MAX_CELLS=1000000000 ... detach_launch.py ... multi_axis_chain.sh --round 1` (re-freeze the sha first if the wheel changed). The chain will redo ma_anchor_s31337002 (no marker was banked).

AS OF 05/11/2026 22:05 EST — CTRL-5 ROUND 1 FLYING on ABI 35 (wheel sha 7e08dd27843f19a1, post-2a27ce01 canonical export order; chain re-frozen 584984a4). Lever = scripts/multi_axis_chain.sh --round 1 (pid 94866, PPID 1, MA_ANCHOR=arm; log /private/tmp/multi_axis_chain.log; runs logs/controller/multi_axis/<tag>.out; markers experiments/multi_axis_markers/<tag>.json). Order: ma_anchor -> axA_L4A -> axA_L4B -> axF_lagoff -> axF_T030 -> axB_pid -> axB_lqr -> axD_cf2xfw_kd025; axC_sn4/sn8 REFUSED until Luiz sets MA_AXIS_C_MAX_CELLS (decision 4; sn4 smoke peak RSS 1.5 GiB, ~391k cells/genome). Plant: airframe-bound lag (cf21 T=0.20 s Gräfe 2026; cf2x 0.26 s), rotor drag 0.78, trainer device cpu. Tick line 1 `lever multi_axis_chain (CTRL-5 round 1, pop 50) — run k/8 <tag> · banked n`; line 2 `<tag> s31337002  (CTRL-5 axis <X>: <hypothesis> vs the lagged anchor, same seed)`. On each banked marker print ONLY the TEST headline row (stable/err/steady/alt/pos/dx/dy) next to the anchor's s2 headline + that condition's rival bar; n=1 = direction only. No leaderboard commits. Supersedes the 17:40 smokes entry.

AS OF 05/11/2026 17:40 EST — CTRL-8 CLOSED at n=3 (Luiz): ADOPT position inputs; arm recipe = multi-axis anchor (experiments/ctrl8_ab_verdict.json; deviation from the n=3 stop rule disclosed). A/B supervisor + arm_s5 killed. CTRL-5 round-1 SMOKES flying: scripts/ma_smokes_round1.sh (log logs/controller/ma_smoke/smokes.log; 8 pop-6 plumbing runs: anchor, L4A, L4B, T015, pid, lqr, cf2x kd0.25, sn4 — NO T030 (option A), NO sn8). Tick line 1 `lever ma_smokes_round1 (CTRL-5 smokes, pop 6) — k/8 <tag>`; line 2 `<tag> smoke: plumbing only (rc/abi/headline/peak RSS), numbers not read`; print each END line from smokes.log. A flight-dynamics review of a lag-aware teacher (T=0.30) is running. Supersedes the 16:20 entry.

AS OF 05/11/2026 16:20 EST — n=3 INTERIM: CONTINUE (rule_v2 stop needs all three attitude CIs clear; steady CI [-1.63,-0.03] misses -0.10). Supervisor relaunched (scripts/ctrl8_ab_pop50.sh, PPID 1): arm_s5 -> twin_s5 -> arm_s6 -> twin_s6, ETA ~07/11 08:00 EST. Line 1 `lever ctrl8_ab_pop50 (CTRL-8 stage-2 A/B) — run 5/8 arm_s5 · pairs 3/5 (s2-s4)`. n=3 paired means (arm-twin): stable +6.33 pp [+3.99,+8.68] · err -0.53° [-0.84,-0.22] · steady -0.83° [-1.63,-0.03] · alt +0.016 m · pos3d -0.020 m [-0.064,+0.025] · dx -0.026 · dy -0.018. Supersedes the 14:15 supervisor-stopped entry.

AS OF 05/11/2026 16:00 EST — IDS: n=3 FIRST (Luiz 05/11). The 89 IDS-47 flows beyond n=3 (ids > 7248; list logs/ids/ids47_n4plus_paused.txt) are PAUSED via the dashboard API; the IDS-48 waiter (pid 91225) is STOPPED. After 7248 the worker goes IDLE BY DESIGN — do NOT escalate 'worker idle with paused flows'. IDS BLOCK cohorts line: `IDS47 n/120 (r run, q q, 89 paused)`; ETA line: `ETA IDS47 n=3 (flow 7248): ~06/11 04:00 EST -> readout + decision (trim / n=5 / accept), then IDS-48`. When 7248 completes: post the n=3 readout per dataset x arm (margin_val_cal F1/FPR/Acc mean±SD + h/flow) and ASK Luiz; un-pause/cancel only on his word; re-arm IDS-48 with `nohup bash logs/ids/queue_ids48_after_ids47_n3.sh >> logs/ids/queue_ids48_after_ids47_n3.log 2>&1 &` (Plane IDS-48 created, blocked by IDS-47). Controller: CTRL-5 spec draft 4 being re-based by a controller agent (docs/multi_axis_programme_spec.md).

AS OF 05/11/2026 14:15 EST — A/B SUPERVISOR STOPPED BY HAND (pid 5601 killed; twin_s4 child pid 24311 re-parented to 1 and still flying). No ab.log END line will be written for twin_s4: read logs/controller/ctrl8_ab_pop50/twin_s4.out for `[stage-select] HEADLINE held-out`. When it appears: n=3 interim check per rule_v2. Arm wins clearly -> verdict, s5/s6 CANCELLED, commit, tell Luiz, propose CTRL-5. Otherwise -> relaunch `nohup scripts/ctrl8_ab_pop50.sh` detached (skips banked winners, starts arm_s5). Line 1 while twin_s4 flies is unchanged (run 4/8); if the controller is idle with no supervisor, say `lever IDLE — n=3 interim check pending`.

AS OF 05/11/2026 08:15 EST — A/B RULE v2 (Luiz 05/11, experiments/ctrl8_ab_rule.json rule_v2): holistic — attitude block (stable/err/steady, paired 95% t-CI vs margins 0.5 pp/0.10°/0.10°) decides; pos3d only tie-breaks. When twin_s4 ENDS (n=3): print the pair table + paired mean/CI per metric; if all three attitude CIs clear their margins in the arm's favour -> STOP the A/B (kill scripts/ctrl8_ab_pop50.sh + its phased_ga child only if it is arm_s5, else let nothing new start: touch experiments/HOLD_CONTROLLER is NOT used — kill the supervisor before it launches arm_s5), commit the verdict, tell Luiz, then propose CTRL-5. Otherwise continue to n=5.

AS OF 04/11/2026 05:55 EST — CTRL-8 STAGE-2 A/B FLYING (Luiz 04/11: "yes, s3-s6 ... just do it"). scripts/ctrl8_ab_pop50.sh (PPID 1; log logs/controller/ctrl8_ab_pop50/ab.log; runs <tag>.out) flies arm_s3, twin_s3, arm_s4, twin_s4, arm_s5, twin_s5, arm_s6, twin_s6 (~9 h each, ~3 days); s2 = arm_kd078/twin_kd078. Rule PRE-REGISTERED experiments/ctrl8_ab_rule.json (n=5, adopt iff mean paired pos3d < 0 AND err <= +0.10 deg AND stable >= -0.5 pp; n=3 interim = direction only, no early stop). Tick line 1 `lever ctrl8_ab_pop50 (CTRL-8 stage-2 A/B) — run k/8 <tag> · pairs n/5`; line 2 `<tag> sN  (CTRL-8: do position inputs beat the position-blind twin, same seed, k_d 0.78)`. When each TWIN ends: one PAIR table (TEST headline arm vs twin per finished seed, stable/err/steady/alt/pos/dx/dy + Δ) + running-mean Δ line. At 5/5: verdict under the rule, tell Luiz. Supersedes the 04:15 idle lines.

AS OF 04/11/2026 04:15 EST — twin_kd078 DONE (rc 0, 04:00 EST); CONTROLLER IDLE. Same seed s31337002, k_d 0.78, TEST headline: arm_kd078 97.0±0.9/2.55±0.19/2.02±0.20/alt 0.163/pos 0.411 vs twin_kd078 (NEURONS#1) 91.4±2.2/3.04±0.14/2.64±0.17/alt 0.179/pos 0.451 — arm better on every column, n=1. Next = stage-2 A/B arm vs twin s2-s6 at pop 50, k_d 0.78 — awaiting Luiz's OK; nothing queued. Tick controller lines while idle: line 1 `lever IDLE — CTRL-8 twin done; stage-2 A/B awaiting Luiz`; line 2 `next: arm vs twin s2-s6 (pop 50, k_d 0.78) after Luiz OKs`; stage/elite/offspring `—`.

AS OF 03/11/2026 18:00 EST — DRAG SWEEP COMPLETE; DEFAULT_ROTOR_DRAG 0.3 -> 0.78 (Folk 2026, Luiz 31/10 'use 0.78 if supported'). Headlines (test seeds): kd0.78 97.0/2.55/2.02/0.163/0.411 · kd0.5 93.2/2.66/2.05/0.170/0.406 · kd0.25 83.6/3.64/3.58/0.209/0.453 (GRID#0) · kd0 86.2/3.72/4.17/0.180/0.420. twin_kd078 flying since 17:53 EST (run 5/5): line 1 `lever ctrl8_twin_kd078_pop50 (CTRL-8 position-blind twin, pop 50) — run 5/5 twin_kd078`; line 2 `twin_kd078  (CTRL-8: does arm_kd078 hold position/attitude better than the same recipe without position inputs, k_d 0.78)`. Readouts show ONLY the headline (report) row + comparators (Luiz 03/11: no VAL/candidate tables).

AS OF 02/11/2026 16:25 EST — TWIN QUEUED + RIVALS ON ABI 33 (Luiz 02/11: "yes"). scripts/ctrl8_twin_kd078_pop50.sh (pid 23883, PPID 1) waits for the sweep AND ctrl8_drag_off_pop50.sh, then flies twin_kd078 = arm_kd078's recipe at k_d 0.78 MINUS --obs-pos-err-xy --obs-vel-xy (w_pos 0.10 kept), s31337002, pop 50, ABI 33. Controller queue: arm_kd078 DONE (97.0/2.55/2.02/0.163/pos 0.411) -> arm_kd050 -> arm_kd025 -> arm_kd000 -> twin_kd078; line 1 then `run k/5`. Rivals re-scored on ABI 33 with the arm's argv: experiments/ctrl22/rivals_abi33_tilt_kd{0.78,0.5,0.25,0.0}.json (37dca57c) — quote THESE (not the abi32 kd0.8 files) next to each arm. When twin_kd078 ends: PAIR table arm_kd078 vs twin_kd078 (TEST headline stable/err/steady/alt/pos/dx/dy) + rivals at 0.78, n=1 direction caveat.

AS OF 02/11/2026 06:50 EST — SWAP CHAIN + IDS-48 ARMED. (1) ABI 23 (dup-aware scoring) swap pid 24354 fires after flow 7239. (2) logs/ids/swap_abi24_after_abi23.sh (pid 91208, log .log) waits for swap_abi23.json, COMMITS src/wnn/accel.py (ABI 23), then arms the ABI-24 swap (ids_calib_sample param, 6012c6f8; default off = bit-identical; marker experiments/staged/swap_abi24.json) for the next idle point. After swap_abi24.json appears: verify next flow not cancelled, worker ABI 24, commit src/wnn/accel.py 'worker ABI 24 deployed'. (3) IDS-48 waiter logs/ids/queue_ids48_after_ids47_n3.sh (pid 91225) queues 20 flows (IDS48-calibfull/-calibsub200k-unswt-QUAD-r<seed>, paired n=10, scripts/queue_ids_calib_paired.py) once flow 7248 is terminal AND ABI 24 is live; pre-registered read in the script header (adopt iff dF1 and dFPR 95% CIs within ±0.10 pp). cohorts line adds `IDS48 n/20` once queued. Controller: ABI 33 deployed 02/11 06:25 EST (exact MPC, 9dd6d762); drag sweep RESTARTED on it 06:25 EST (run 1 arm_kd078; old ABI-32 attempt kept as *.abi32_oldmpc_aborted). flow_runner controller-path EpisodeConfig fix 411bb38d.

AS OF 02/11/2026 06:40 EST — WORKER ABI 23 (duplicate-aware scoring, b107331a, bit-identical to ABI 22, ~2.2x per genome-fold at rate 1.0) SWAP ARMED: worker_swap pid 24354 (log /tmp/worker_swap_abi23.log, marker experiments/staged/swap_abi23.json) waits for flow 7239, installs the ABI-23 wheel + applies experiments/staged/abi23_dupscore_accel_python.patch, RAYON 13. On the first tick after the marker: check the next flow is running/queued (not cancelled), worker ABI = 23, then commit src/wnn/accel.py 'worker ABI 23 deployed (dup-aware scoring)'. ABI 22 deployed 01/11 20:36 EST (c46aaa4b). Controller: staged ABI-33 controller wheel (exact MPC, 195563e9) predates the ram_core change in b107331a — REBUILD the controller wheel before installing it (awaiting Luiz on the sweep restart).

AS OF 01/11/2026 19:45 EST — IDS-47 n=3 ETA RE-ESTIMATED: ~06/11 01:00 EST ±10 h (7237 cicids QUAD = 500n x 34b genomes, ~29.5 h; ciciot QUAD ~20 h; ~100 h after 7237). IDS BLOCK ETA line: `ETA IDS47 n=3 (flow 7248): ~06/11 01:00 EST ±10 h -> readout + rethink`. Rate probe (ids-security agent, /tmp/ids_rate_probe): QUAD rate 0.25 is NOT 4x faster on 7237's genomes (0.83-0.98x of rate 1.0 per fold); scoring dominates (83%), multiplicity collapses only TRAINING. Proposed bit-exact duplicate-aware scoring (~2x at rate 1.0, ram_core -> both wheels) awaits Luiz.

AS OF 01/11/2026 11:05 EST — IDS-34 FIX (15baa638, worker ABI 22) SWAP ARMED (Luiz 01/11: "swap as soon as possible"). worker_swap pid 71622 (log /tmp/worker_swap_abi22.log, marker experiments/staged/swap_abi22.json) waits for flow 7237 to finish, installs the ABI-22 wheel + applies experiments/staged/abi22_ids34_accel_python.patch, restarts the worker (RAYON 13). On the first tick after the marker appears: (1) check the swapped-out flow and the next flow are running/queued, NOT cancelled (the 26/09 requeue race); (2) confirm `python -c 'import ram_accelerator as a; print(a.ABI_VERSION)'` = 22; (3) `git add src/wnn/accel.py` + commit 'worker ABI 22 deployed (IDS-34)' + push. Caveat for the IDS-47 readout: flows up to 7237 ran on ABI 21 (lossy low-bit search scoring), flows from 7238 on ABI 22 — seed r20421 is split across the two; reported held-out numbers are unaffected (no validated genome at b<=12). Controller: pop-50 drag sweep run 1 restarted 07:09 EST with --eval-mem-budget auto (now the phased_ga default, c5fa11e8); no-drag control (arm_kd000) waiter pid 26229 queued after the sweep.

AS OF 31/10/2026 15:00 EDT — QUEUE SWAPPED TO POP 50 (Luiz 31/10: smoke runs check PLUMBING only; any decision — drag on/off, k_d 0.25-0.78 — is made at the full pop 50). experiments/HOLD_CONTROLLER is touched ON PURPOSE: ctrl8_resmoke_abi32.sh stops after twin_drag (arm_nodrag/twin_nodrag dropped). The pop-6 drag sweep waiter was killed. scripts/ctrl8_drag_sweep_pop50.sh (pid 52936, PPID 1; log logs/controller/ctrl8_drag_sweep_pop50/sweep.log) waits for the re-smoke to exit, REMOVES HOLD, then flies s31337002 pop 50 arm at k_d 0.78 -> 0.5 -> 0.25 (stage-2 bundle, w_pos 0.10, --att-err-ref tilt, NEURONS 15 gens / patience 3 — Luiz yes 31/10; MEMORY unchanged). ~20-25 h per run on ABI 32 (estimate). Do NOT remove HOLD by hand. Tick line 1 after the twin: `lever ctrl8_drag_sweep_pop50 (CTRL-8 rotor-drag sweep, pop 50) — run k/3 <tag>`; line 2 `CTRL-8 drag sweep (pop 50): WNN held-out attitude/position vs k_d (0.78 Folk 2026 / 0.5 / 0.25 Förster 2015)`. Supersedes the 14:20 pop-6 sweep entry.

AS OF 31/10/2026 14:20 EDT — ROTOR-DRAG WNN SWEEP ARMED (Luiz 31/10: "sweep 0.25, 0.5, 0.78 and use 0.78 if supported"). scripts/ctrl8_drag_sweep_abi32.sh (pid 19344, PPID 1; log logs/controller/ctrl8_drag_sweep_abi32/sweep.log) waits for the re-smoke, then flies arm_kd078 -> twin_kd078 -> arm_kd050 -> arm_kd025 (same re-smoke recipe; ~4.7 h each). Sources: Folk 2026 CF2.1-Brushless free flight -> k_d ~0.78 at 39.3 g; Förster 2015 rotor-only ~0.25 (docs/papers). Default stays 0.3 until the sweep reads out; switch the phased_ga default to 0.78 only if supported, and tell Luiz. Tick: once the re-smoke ends, line 1 `lever ctrl8_drag_sweep_abi32 (CTRL-8 rotor-drag sweep) — run k/4 <tag>`, line 2 `CTRL-8 drag sweep: how the WNN's held-out attitude/position change with k_d (0.78 from Folk 2026 vs 0.5/0.25)`.

AS OF 31/10/2026 08:05 EDT — ABI 32 LANDED (f99611bf; controller wheel ABI 32): real-life plant defaults under --translation (specific-force IMU, rotor drag 0.3 1/s [PLACEHOLDER coefficient, no cf21 source], WNN roll/pitch from the rivals' Mahony estimator), --att-err-ref tilt|level, pos/dx/dy on gen lines; --imu-inclinometer --no-rotor-drag reproduces ABI 31. NEVER pair ABI-31 and ABI-32 markers. CTRL-22 rivals re-scored on ABI 32 (experiments/ctrl22/stage2_rivals_op31_xy05_abi32_{tilt,level}.json). CTRL-8 RE-SMOKE FLYING: scripts/ctrl8_resmoke_abi32.sh (log logs/controller/ctrl8_resmoke_abi32/sweep.log, run logs <tag>.out): arm_drag -> twin_drag -> arm_nodrag -> twin_nodrag; s31337002, pop 6, all stages, xy 0.5, 2 s, w_pos 0.10, --att-err-ref tilt. Tick controller lines: line 1 `lever ctrl8_resmoke_abi32 (CTRL-8 re-smoke, ABI 32) — run k/4 <tag>`; line 2 `CTRL-8: on the real-life plant, does the WNN hold position better than its position-blind twin, drag on vs off`; stage line from the .out (gen lines now carry pos/dx/dy); on each END print the HEADLINE held-out stable/err/steady/alt/pos/dx/dy and the PID rival pos. Table when each arm/twin pair is done. No leaderboard commits for smoke runs.

AS OF 31/10/2026 06:10 EDT — CTRL-8 SMOKE SWEEP STOPPED after run 2 (Luiz 31/10: fix, then restart). Results kept: arm_wpos010 97.0%/1.65°/0.99°/alt 0.187/pos3d 0.425 m vs twin_wpos010 99.2%/1.30°/0.57°/alt 0.168/pos3d 0.459 m (n=1). Fix list before the re-smoke (Luiz 31/10): (1) G12 accelerometer = real specific force under translation (CPU+Metal, parity) — lineage break, re-baseline the classics (CTRL-22); (2) G5 stage-2 attitude error measured against the cascade's reference tilt, not level; (3) --skip-stages memory must SKIP MEMORY; (4) gen line + reports print position, plus per-axis dx/dy/dz (fitness keeps 3-D, Luiz); (5) desirability half-anchor for pos. Controller IDLE until the re-smoke. Tick controller lines: line 1 `lever IDLE — CTRL-8 fixes in progress (G12, G5, skip-memory, pos logging)`; line 2 `next: CTRL-8 re-smoke after the fixes land`.

AS OF 30/10/2026 23:00 EDT — CTRL-8 SMOKE + w_pos SWEEP FLYING (Luiz 30/10: "let's add [the position term] and do a sweep"; "let's do this smoke"). Lever = scripts/ctrl8_smoke_sweep.sh (log logs/controller/ctrl8_smoke/sweep.log; run logs logs/controller/ctrl8_smoke/<tag>.out). One seed s31337002, _op31 argv, pop 6, grid->neurons, MEMORY skipped, --xy-offset 0.5, 2 s episodes, 3-D position metric (D2 horizontal metric NOT approved yet). Order: arm_wpos010 -> twin_wpos010 (no position features) -> arm_wpos000 -> arm_wpos005 -> arm_wpos020 (weights x(1-w_pos)). Tick controller lines: line 1 `lever ctrl8_smoke_sweep (CTRL-8 smoke) — run k/5 <tag>`; line 2 `CTRL-8 smoke: does the WNN hold position from a 0.5 m start better than its position-blind twin (and which w_pos)`; stage line from the .out; on each END line print the NEURONS MULTI-SEED held-out stable/err/steady/alt/pos. Smoke criteria: (b) fitter spans non-degenerate for the 4 xy features; (d) arm pos < 0.385 m start mean AND < twin. Results table when the twin is done; full sweep readout at the end. No leaderboard commits for smoke runs. Design doc: docs/ctrl8_stage2_design.md (D1-D6 await Luiz).

AS OF 30/10/2026 20:45 EDT — CTRL-20 CLOSED: NOT ADOPTED (Luiz 30/10). n=5 non-inferiority: mean dhd +0.0086 FAIL, stable -0.36 pp pass, err +0.068 pass; genome differed 3/5 (s3,s4,s6), worse on all 3 (d06a493e; Plane CTRL-20 Done). Recipe stays `_op31`. CONTROLLER IS IDLE (chain exited 23:58Z); NEXT = CTRL-8 stage 2 on the decided fitness (S16noJM zscore, alt 0.0, no MAD floor) — Luiz 30/10 "start the ctrl8 now". Prep in progress (no box-heavy work until the design is approved): CTRL-17 gaps re-checked (G1-G4, G10a, G11, G13 fixed; open G5 tilt-referenced attitude error, G6 label content, G7 horizon (CTRL-22 measured 2 s/6 s), G10b pos metric is 3-D, G10d pos_* forwarding, G12 accel = ideal inclinometer). Tick controller lines while idle: line 1 `lever IDLE — CTRL-20 closed (not adopted); CTRL-8 prep` · line 2 `next: CTRL-8 stage-2 smoke after Luiz approves the pre-registered design` · stage/grid lines `—` · box + IDS lines as usual. No PAIR tables until a CTRL-8 marker exists.

AS OF 30/10/2026 20:30 EDT — IDS-47 PLAN = n=3, THEN RETHINK (Luiz 30/10: "we are doing n=3 for now and rethink if we need anything more later"). No n=5 / n=10 is assumed; the queue still holds all 120 flows (nothing cancelled) but readouts and ETAs target n=3 only. IDS BLOCK ETA line is now exactly: `ETA IDS47 n=3 (flow 7248): ~04/11 03:00 EST (range 03/11 pm - 04/11 pm) -> readout + rethink` (drop the n=5 line). At n=3: per dataset x crossover readout (incl. QUAD r20420/r20421) and ASK Luiz what, if anything, runs after.

AS OF 29/10/2026 09:40 EDT — IDS-47 TRIM DECISION DEFERRED TO n=3 (Luiz 29/10: "let's wait n=3 ... then we decide"). Nothing cancelled, nothing reordered; the queue runs as-is. n=3 = seed r20421 complete on all 3 datasets = flow 7248 (unswr blend_cluster r20421) done; ETA ~04/11 03:00 EST, range 03/11 afternoon - 04/11 afternoon (re-estimated 30/10 18:10 EDT; US clocks fall back 01/11; cicids binary has run 3-12 h, ciciot ~13-16.5 h). Luiz's leaning (NOT final): cicids -> blend_neuron only; unswr -> blend_neuron + blend_cluster up to n=5; ciciot -> torn between inherit and blend_cluster. When 7248 completes: post the n=3 readout per dataset x crossover (best_f1, GA final, margin_val_cal F1/FPR/Acc mean±SD + hours, plus the QUAD r20421 row and the old tandem QUAD refs flagged pre-ABI-20) and ASK Luiz which queued crossover flows to cancel — do not cancel without his OK. IDS BLOCK: add `IDS47 trim decision at n=3 (7248), ETA ~04/11 EST` to the ETA line.

AS OF 29/10/2026 06:20 EDT — CTRL-20 RULE AMENDED TO NON-INFERIORITY (Luiz 29/10: judge it "never worse, sometimes better" — yes). Amended after 1/5 banked (s2 = exact zero, same published genome as _op31; commit 14b032d3), before s3-s6; logged on Plane CTRL-20. RULE NOW: adopt iff mean paired hd NOT WORSE (Δhd <= 0) AND mean err not worse than +0.10° AND mean stable not worse than -0.5 pp (TEST headline, paired vs _op31, n=5). Every PAIR table adds a `same genome?` column and the running-mean line adds `published genome differs on k/n seeds`. At 5/5: verdict under THIS rule + era caveat, tell Luiz, do NOT flip the recipe default without his OK. Everything else in the 28/10 17:25 entry stands (s3 started 04:56 EDT 29/10; ~11.5 h/seed). ALSO 29/10: controller random immigrants are CLOSED — already flown 07/2026 (E2 arm IMM, --immigrants 0.15: 81.6±4.7 vs anchor 84.3±4.4, a −2.7 pp tax; docs/controller_horizon_findings.md); Luiz: tried on the controller and did not improve → not re-flown.

AS OF 28/10/2026 17:25 EDT — CTRL-20 A/B FLYING (Luiz 28/10: "A/B for CTRL-20 after the probe"). CTRL-21 DONE (both probes banked; floors = median of six robust estimates: stable 1.93 pp, steady 0.330°, err² column 1.88 reward). CTRL-22 rival bar DONE (411abc44; G6 label-content folded into CTRL-8 prep). Lever = scripts/ctrl20_mad_floor_chain.sh (log /private/tmp/ctrl20_mad_floor.log) -> seed_arm_chain `_mf31` x5 (s2-s6; _op31 recipe + --zrank-mad-floor; header must show madfloor=on), chain log /private/tmp/seed_arm_mf31.log, paired vs `_op31`. s2 started 17:22 EDT 28/10; ~10.3 h/seed -> ETA all 5 ~31/10 evening EDT. Ticks track `_mf31` n/5: line 1 `lever seed_arm_chain (CTRL-20 A/B)` + markers `_mf31 n/5`; line 2 = `run n/5  _mf31 sN  (CTRL-20: does the measured MAD floor pick better published genomes than _op31 without selling attitude)`. On each `_mf31` marker: the single PAIR table (TEST headline: mf31 vs op31 per banked seed, stable/err/steady/alt/hd/rank) + running mean delta line, commit "Leaderboard: <tag> banked (CTRL-20 n/5)", push. PRE-REGISTERED RULE (Plane CTRL-20): adopt iff mean paired hd better (lower) AND err not worse than +0.10° AND stable not worse than -0.5 pp. At 5/5: print the verdict + chain era caveat, tell Luiz, do NOT change the recipe default without his OK. Next after: CTRL-8 (stage 2) on the decided fitness — its prep (flight-dynamics observability check, smoke-plan items 1-3, G6 Rust trace export, pre-registered design) may proceed during the A/B but nothing heavy flies beside it.

AS OF 28/10/2026 16:55 EDT — CTRL-19 CLOSED, CTRL-21 PROBE RUNNING (Luiz 28/10: "Keep 0.0, close CTRL-19, and launch CTRL-21"). Alt weight STAYS 0.0 (rung 2 _alt10x31 5/5 rule did not fire; 0.0 vs 0.05 a tie inside noise; Luiz's reservation: rank fitness is blind to altitude, attitude terms are the indirect proxy). CTRL-21 probe: scripts/crn_noise_probe.py on the _op31 s31337002 MEMORY population (57 genomes, --frames 3, 5 pools x 200 ep, frame A MATCH), launched 16:53 EDT (20:53Z) detached (pid 58097 under /usr/bin/time 58095), out /private/tmp/ctrl21_probe_op31.out, JSON experiments/crn_noise_probe/PROBE_SL_C_b24n256_cf21_brushless_L4C_g10_s31337002_op31.json. NOTE it ALREADY RAN 30/09 on _op30 (PROBE_..._op30.json, never read out) — the _op31 run is the current-recipe replication. Controller lines while it runs: line 1 `lever crn_noise_probe (CTRL-21)`, line 2 `run CTRL-21 probe on _op31 s2 MEMORY pop (measures within-frame comparison noise per fitness metric = CTRL-20 MAD floors)`, then frame progress from the .out tail. No markers bank (no leaderboard commit). When the JSON lands: print SD_g(D) per metric for A-B/A-C/B-C next to the _op30 values (scratchpad probe_sum.py prints them), git add -f the JSON, commit "CTRL-21: _op31 probe banked", push, comment on Plane CTRL-21, then PROPOSE the CTRL-20 floors to Luiz — do not deploy mad-floor without his OK. Agreed order after: CTRL-20 decision -> CTRL-22/CTRL-8 stage 2 -> CTRL-5 multi-axis.

AS OF 26/10/2026 10:30 EDT — CTRL-19 RUNG 2 ARMED + IDS-47 n=5 (Luiz 26/10). Controller: after _alt31 s6 banks, scripts/ctrl19_alt10_chain.sh (pid 36620, PPID 1, log /private/tmp/ctrl19_alt10.log) launches `_alt10x31` x5 (s2-s6, alt weight 0.10, S16noJM x0.90; chain log /private/tmp/seed_arm_alt10x31.log) paired vs `_op31`, same pre-registered rule. Luiz: "0.05 is a toss up... instead of giving more seeds to 0.05, try out 0.10". When the 5th _alt31 marker banks: commit "(CTRL-19 5/5)" as usual, tell Luiz rung 2 started (do NOT propose CTRL-21 yet — it now follows rung 2). Ticks then track `_alt10x31` n/5: line 2 = `run n/5  _alt10x31 sN  (CTRL-19 rung 2: does alt weight 0.10 buy altitude without selling attitude vs _op31)`; on its markers the same single PAIR table (alt10x31 vs op31 per seed, plus the alt31 row for that seed) and commit "Leaderboard: <tag> banked (CTRL-19b n/5)". From now on every controller run prints an `[argv]` line + stores meta.argv (f4e30b5b). IDS-47: Luiz keeps the full design to the n=5 interim (no trimming, no reorder). Measured first-seed durations: cicids BINARY ~5.6-6.6 h, ciciot BINARY 16.47 h (7221, 130 gens, 92.43/7.80/96.18 vs QUAD ref ~93.0/6.9/96.5 in ~5.4 h). ETA line: `ETA IDS47 n=5: ~10-12 days (cicids ~6h, ciciot ~16.5h per BINARY flow; unswr unmeasured)`.

AS OF 25/10/2026 20:10 EDT — NEW-MARKER FORMAT SIMPLIFIED (Luiz 25/10: "I now see enough to trust the report headline, so let's just show the report headline winner in the end and when comparing against others as well"). SUPERSEDES the two-table NEW-MARKER FORMAT (05/09 + 09/10 'BOTH STICKS'): on a new marker print the `>>> NEW MARKER n/T` line, then ONE fenced PAIR table with TEST-headline columns only — run | headline genome | stable % · err ° · steady ° · alt m · hd · rank — one row per comparable run (this marker, its same-seed control, the earlier seeds of the arm + their controls), then the running-mean delta line. NO STAGE-SELECT table, NO VAL columns, NO per-stage rows (the six fixed controller lines are unchanged). At most two sentences of reading + the commit hash. Also 25/10: Luiz KEEPS QUAD neuron_sample_rate 1.0 for IDS-47 and the next round (SR1 tie, cheaper with the multiplicity layer) — no re-clone.

AS OF 25/10/2026 18:00 EDT — IDS-47 QUAD RE-CLONE (Luiz 25/10: "yes, this is the right thing to do... recloning to run k-fold=5 on them"). The 5 REUSED pre-ABI-20 tandem QUAD flows (6431 cicids r20419, 6432 ciciot r20419, 6434 unswr r20419, 6436 cicids r20420, 6437 ciciot r20420 — single-fold offspring scoring; 6431/6432 also rate 0.25) are REPLACED for the IDS-47 readout by 7343-7347 (IDS47-<cohort>-QUAD-1m-r20419/r20420), cloned from each cohort's r20421 QUAD clone with only seed changed + ids_offspring_folds="all" (logs/ids/reclone_ids47_quad5.py). Appended at the queue TAIL (FIFO, nothing reordered), so the r20419/r20420 QUAD pairs land last. IDS-47 is now 120 flows: cohorts line `IDS47 n/120`. Every IDS-47 readout pairs QUAD by the IDS47-*-QUAD-1m-r<seed> NAME, never the old tandem ids. 7219 duration in the DB (3.08h) is the post-ENOSPC resume only.

AS OF 25/10/2026 09:05 EDT — DISK-FULL INCIDENT + CTRL-19 RELAUNCH. Internal disk hit 100% at 08:14 EDT (ENOSPC): CTRL-19 s4 died in MEMORY gen 3 (no marker; dead run kept as ckpt/<tag>.enospc + <tag>.out.enospc), IDS 7219 died rc=120 and was re-queued by the stale-reaper (resumes after 7220). Fix: logs/ UNTRACKED (f46e1951) and every entry moved to /Volumes/20260401-WDBlack-SN850X-2TB/wnn/logs with per-entry symlinks (334 GB free on internal). Bugs fixed 2d2c4d99 (ladder now exits 1 + logs INCOMPLETE on a failed point; checkpoint .tmp orphans unlinked on error). CTRL-19 chain RELAUNCHED by hand 13:04Z, pid 77487 (PPID 1), same env as scripts/ctrl19_alt_weight_chain.sh, log /private/tmp/seed_arm_alt31.log: skipped s2/s3 (banked), s4 re-flying FROM SCRATCH, then s5, s6. Every tick also prints disk free: `df -h /System/Volumes/Data` on the box: line (warn under 50 GB).

AS OF 25/10/2026 00:35 EDT (supersedes the 16:05 entry) — IDS-45 COMPLETE 20/20; CAP RULE FIRED 04:03Z: DECISION max250 (rule 3: tot250 dF1 -0.30±0.13 fails rule 1; tot500 dF1 -0.04±0.16 CI does not exclude 0, fails rule 2). Readout + json: logs/ids/ids47_after_ids45.log, logs/ids/ids45_decision.json. Watcher 8801 un-paused all 115 IDS-47 flows (failed=0) and exited. AWAITS Luiz's morning validation of max250. cohorts line now: `IDS47 n/115 · tandem 396 PAUSED · IDS-6 11 PAUSED`. IDS-47 tasks from the 09:30 entry still apply: report the first BINARY flow duration per dataset (cicids 7218 first), and post the paired interim readout when seed r20423 completes on all 3 datasets (n=5).

AS OF 24/10/2026 16:05 EDT (supersedes the 09:05 entry's ASK) — IDS-47 CAP SELF-DECIDED (Luiz 24/10: "choose for me and in the morning I will validate"). Watcher 49554 KILLED, replaced by pid 8801 logs/ids/ids47_after_ids45.sh (.log): when IDS-45 is terminal it runs the PRE-REGISTERED rule logs/ids/ids45_decide.py (paired n=10 vs max250 blend_neuron, best_f1/GA final/margin_val_cal): 1) tot250 if dF1 >= -0.10 pp AND dFPR <= +0.20 pp; 2) tot500 if dF1 CI excludes 0 (>0) AND dFPR <= +0.20; 3) else max250; fewer than 10 pairs = FAIL-SAFE max250. max250 -> un-pauses all 115 IDS-47 flows; tot250/tot500 -> cancels the paused 115 and re-clones them at the cap (logs/ids/queue_ids47_xds.py --cap X --cancel-paused, same seed-major interleave). On the first tick after it fires: print the readout (`python logs/ids/ids45_decide.py`), the decision + reason, and verify IDS-47 = 115 queued (no paused/pending left); tell Luiz it awaits his morning validation. cohorts line: `IDS45 n/20 · IDS47 115 PAUSED (cap rule fires after IDS-45, watcher 8801)`. FIRST IDS-45 flow (7208 tot500): grid shows `EXCLUDED neurons` for n*2 > 500; GA genome totals <= cap but may be UNEVEN (e.g. 300/200) — the grid only tests even splits.

AS OF 24/10/2026 09:05 EDT (supersedes the IDS-47 run order below) — IDS-47 PAUSED (115); IDS-45 now 20 flows (tot500 7208-7217 + tot250 7333-7342) and runs right after IDS-46. Watcher pid 49554 (logs/ids/unpause_ids47_quad_after_ids45.sh/.log) un-pauses the 25 IDS-47 QUAD flows when IDS-45 is terminal; the 90 BINARY wait for Luiz. When IDS-45 completes: post the paired budget readout (max250/cluster vs tot250 vs tot500, blend_neuron, n=10: F1/FPR/Acc, neurons, cells, hours) and ASK Luiz which cap IDS-47's BINARY arms use (if it changes, cancel+re-clone the 90 at the new cap). cohorts line: `IDS45 n/20 · IDS47 paused (QUAD after IDS-45)`.

AS OF 24/10/2026 09:30 EDT — IDS-47 QUEUED (flows 7218-7332, 115): cross-dataset A/B BINARY lattice 2m max250 x {inherit,blend_neuron,blend_cluster} vs QUAD 1m on cicids / ciciot-subsample / unswr, seeds r20419-28 seed-major. Runs after IDS-45. The config of the 08:00 entry is UNSW-temporal only until IDS-47 reads out. IDS BLOCK `cohorts:` line: `IDS46 n/60 · IDS45 n/10 · IDS47 n/115 · …`. When IDS-47 seed r20423 completes for all 3 datasets (n=5) post a paired interim readout so Luiz can choose stop-at-5/7/10. On the first IDS-47 BINARY flow per dataset report its duration (untimed off unswt). 396 tandem + 11 IDS-6 stay PAUSED (Luiz: cancel only after IDS-47).

AS OF 24/10/2026 08:00 EDT — IDS: config FIXED BINARY-lattice-2m max250 + blend_neuron (Luiz). IDS-43 cancelled. IDS-45 queued 7208-7217 (10 flows, tot500 free split) — runs after IDS-46, before the tandem. IDS BLOCK `cohorts:` line now reads `IDS46 n/60 live · IDS45 n/10 · …`. On the FIRST IDS-45 flow start: smoke-check its log for the grid's max_neurons_total exclusion line and that GA genomes total <= 500 (first production use). When IDS-46 completes: post the final readout (same tables as ids_results.md IDS-46 CONFIG DECISION, n=10). When IDS-45 completes: paired readout vs max250 blend_neuron, then show Luiz the IDS-47 dry run.

AS OF 24/10/2026 05:35 EDT — LEVELS ENDED, CTRL-19 FLYING (Luiz 23/10 option A). LEVELS stopped at 9 markers (n512 s2-s5; s6 dropped): n256 STAYS (n512 flat vs n256, n384 worse). queue_2009c, its seed_arm_chain and the ctrl19 waiter were killed after s5 banked; CTRL-19 seed_arm_chain launched BY HAND 09:35Z (pid 48009, log /private/tmp/seed_arm_alt31.log, tag [arm_alt31]). No more `_Lop31` markers will bank — ticks now track `_alt31` n/5 (line 2 + commit format from the 21/10 entry). The _tw15-at-winner step is moot: n256 won and `_tw15` at n256 is banked. After CTRL-19 the controller queue is EMPTY — next work is AGREED (Luiz 24/10): CTRL-21 probe → CTRL-20 decision → CTRL-22/CTRL-8 stage 2 → CTRL-5 multi-axis. Nothing is queued; CTRL-21 starts when CTRL-19's 5th marker banks (box at 2 heavy runners until then) — tell Luiz then, do not auto-launch.

AS OF 21/10/2026 09:25 EDT — CTRL-19 ARMED BEHIND LEVELS (Luiz 21/10: "start with 0.05 only after LEVELS"). scripts/ctrl19_alt_weight_chain.sh (PPID 1, log /private/tmp/ctrl19_alt_weight.log, tag [ctrl19]) polls queue_2009c every 5 min; on "QUEUE COMPLETE" it runs seed_arm_chain `_alt31` x5 (s2-s6, n256, `_op31` recipe + S16noJM x0.95 + --fit-weight-alt 0.05), paired vs `_op31`; fails closed if LEVELS aborts. Markers glob `*_alt31.json` n/5. Line 2 for it: "`_alt31` sN  (CTRL-19: does alt weight 0.05 buy altitude without selling attitude vs _op31)". Read (pre-registered, Plane CTRL-19): alt better AND err not worse > +0.10° AND stable not worse > -0.5 pp. The `_tw15`-at-LEVELS-winner proposal still needs Luiz's OK and goes AFTER CTRL-19 (if n256 wins LEVELS, `_tw15` at n256 is already banked). On each `_alt31` marker: NEW-MARKER block + PAIR table vs `_op31` (4 columns + new hd) and commit "Leaderboard: <tag> banked (CTRL-19 n/5)". The IDS BLOCK rules of the 20/10 entry still apply.

AS OF 20/10/2026 06:30 EDT — LEVELS FLYING (CTRL-18, `_Lop31`, curriculum 8->30 stays). Lever = scripts/queue_2009c_obs_pwm_recipe.sh (pid 47639, PPID 1) -> seed_arm_chain. n384 s2-s6 then n512 s2-s6 = 10 markers, glob `*_Lop31.json`; banked 3/10 (s2, s3, s4). Line 2: "`_Lop31` nNNN sN  (LEVELS: does nNNN beat the n256 anchor at 8->30)". AFTER LEVELS: queue ONE `_tw15` arm at the LEVELS-winning size, same seeds (Luiz's tilt plan #2); seeds s6-s9 at n256 only if still unresolved (#3).
IDS (worker ABI 21 since 19/10, PID 48705): IDS-40 cohort (60 flows, IDS40-*) running FIFO, then IDS-44 ga_crossover_bits A/B flows 7082-7090 (IDS44-*, 3 rules x 3 seeds). When the last IDS40 flow completes: the IDS-40 readout goes to docs/ids_results.md (Rule 7) + Plane IDS-40 comment — say so on the IDS block, do not write it inside a tick.

IDS BLOCK (Luiz 20/10: "status of the controller AND ids runs"). After the six controller lines (and after any NEW-MARKER tables), print a blank line then EXACTLY these lines:
  [ids] worker <up PID|DOWN> · ABI <n> · completed <c> · running <r> · queued <q>
  running: <id> <name> · <phase> gen G · elapsed <h>h  (one line per running flow; phase/gen from the latest iterations row of its running experiment)
  cohorts: <PFX> <done>/<total> (<run> run, <queued> q) · ...   (every prefix with queued or running flows; total = all non-cancelled flows of that prefix)
  last done: <id> <name> at DD/MM HH:MM EDT · <dur>h   (most recent completed flow)
  ETA <PFX>: DD/MM HH:MM EDT   (latest_done + remaining x avg duration of that prefix's completed flows; per active prefix)
Sources: sqlite3 on the read-only db URI; ABI via `wnn/bin/python -c "import ram_accelerator as a;print(a.ABI_VERSION)"` is too slow per tick — read experiments/staged/swap_abi21.json or print the last known ABI (21). Escalate (one extra line starting `!!`) if the worker is DOWN while flows are queued, or any flow failed since the last tick.

AS OF 07/10/2026 14:30 EDT — CTRL-24 (b20) CLOSED: NOT ADOPTED, b24 stays (experiments/ctrl_b20_verdict.json; n=4 mean headline delta stable −1.45 pp [CI −2.55,−0.35] · err +0.17° · steady +0.23° · alt +0.16 m [CI +0.01,+0.30], fails every margin; keys clause moot). queue_b20 exited ('b20 DONE 4/4'). TILT-WIDTH LADDER FLYING since 13:47 EDT (_tw7 s2 first). Line 2 for a `_tw` run per the 05/10 block. LEVELS flies at b24.

AS OF 05/10/2026 10:05 EDT — TILT-WIDTH LADDER ARMED (Luiz 05/10). scripts/queue_tilt_width_sweep.sh (pid 87148, PPID 1, log /private/tmp/queue_tilt_width.log, tag [qtw]) waits for "b20 DONE 4/4" in /private/tmp/queue_b20.log, then flies `_tw7` (3->7), `_tw10` (5->10), `_tw15` (8->15) SEED-MAJOR interleaved (s2 all three, then s3, s4, s5; 12 runs ~6 days), each paired vs `_op31` (8->30) same seed; `_tc31` (5->5) is the other baseline. Rule PRE-REGISTERED experiments/ctrl_tilt_width_rule.json. Order now: `_tc31` s5 -> queue_2009c PARK -> b20 x4 -> tilt-width x12 -> tilt decision -> LEVELS. DO NOT write experiments/tilt_decision until the [qtw] log says "TILT-WIDTH DONE 12/12" (supersedes the b20-only gate below). tilt_decision accepts only coherent|wide: if a `_tw` arm wins, LEVELS needs a relaunch with its flags (Luiz's call). Line 2 for a `_tw` run: "`_twN` sN  (tilt-width ladder: curriculum E->F vs _op31 8->30)"; markers `_tw(7|10|15)\.json` n/12. On each `_tw` marker print the per-seed LADDER table (5->5, 3->7, 5->10, 8->15, 8->30 rows, same seed). If the supervisor dies after a reboot, relaunch it detached (idempotent, marker-gated).

AS OF 03/10/2026 14:30 EDT — CTRL-17 CLOSED (s6 banked 13:43 EDT; _op31 n=5 anchor 99.16/1.42°/0.79°/0.185 m). Queue on STEP 6b `_tc31` s2-s5 (CTRL-23, tilt 5->5) paired vs `_op31`; line 2 names `_tc31` sN and "tilt-coherence vs _op31". CTRL-18 stays open.

AS OF 01/10/2026 11:30 EDT — CTRL-24 (b20 + obs-pwm, Luiz 01/10) SLOTTED BEFORE LEVELS: scripts/queue_b20_before_levels.sh waits for queue_2009c's tilt PARK, then flies SL_C_b20n256_..._s2-5_op31 (ARM_BITS=20, ARM_NO_CONTROL=1), paired by hand vs b24 _op31 per experiments/ctrl_b20_rule.json (non-inferiority every column + keys <= 0.6x). At the tilt PARK: do the stress re-score, but DO NOT write experiments/tilt_decision until /private/tmp/queue_b20.log says "b20 DONE 4/4" — LEVELS starts the moment that file exists. If b20 is adopted, LEVELS needs the b20 recipe (Luiz). Supervisor must be (re)launched detached after any reboot, alongside queue_2009c.

AS OF 30/09/2026 17:55 EDT — CTRL-16 VERDICT KEEP (gate 3/4 passed, size fails: headline keys 1,886,009 vs 799,654; _pipeN30 s2 GRID#2 re-count reproduced log line 981 exactly, 580,994 keys). ABI-31 LANDED (merge 981ecbb8, smokes PASS, ABI31_LANDED touched). queue_2009c now in STEP 6a: `_op31` s2-s5 (anchor recipe --obs-pwm + PIPE_FLAGS on the ABI-31 trainer, paired vs `_op30` = the TRAINER read, NEVER pair across ABI 30/31 otherwise), then s6 (5th seed of the 64-level rung, CTRL-18), then STEP 6b `_tc31` x4 (CTRL-23), then PARK for the tilt decision. Line 2 names `_op31` sN and "trainer read vs _op30". The 13:45 block's gap task is DONE. Leaderboard h743 now = headline genome (`~` = not yet counted); 178 headlines counted 30/09.

AS OF 30/09/2026 13:45 EDT — CTRL-16 SIZE CLAUSE AMENDED (Luiz 30/09): it now counts the stage-select HEADLINE genome's TRUE keys (<tag>@headline, count_true_keys.py --headline), not the saved MEMORY best_genome. ctrl16_verdict.py prints PENDING + replace_anchor null while any headline is uncounted; the queue treats null as not-replace and proceeds to 5b/PARK (nothing irreversible before the PARK). ONE count is pending by design: _pipeN30 s2 headline GRID#2 is arch-only. DO IN THE CONTROLLER GAP (after STEP 5b's probe finishes, while PARKED, before landing ABI-31):
  PYTHONPATH=src/wnn nice -n 10 /Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python scripts/count_true_keys.py --headline experiments/sweepladder_markers/SL_C_b24n256_cf21_brushless_L4C_g10_s31337002_pipeN30.json --retrain-args experiments/ctrl16_args/ARGS_SL_C_b24n256_cf21_brushless_L4C_g10_s31337002_pipeN30.txt
  VERIFY its RESULT row on 99990101 == 98.0% / 1.75° / 1.00° / 0.301 m (log line 981); if not, the argv reconstruction is wrong — do NOT cache-trust it. Then re-run scripts/ctrl16_verdict.py and report the final CTRL-16 verdict (if REPLACE: do not touch ABI31_LANDED, escalate to Luiz).
Headline keys so far: _full30 s2 1,701,212 · s3 772,185 · s4 4,401,674 | _pipeN30 s3 772,185 · s4 1,169,649 · s5 675,789.

AS OF 26/09/2026 16:45 EDT — LEVER HANDED OVER (Luiz: adopt --obs-pwm). queue_2009b supervisor 67249 TERMed alone;
the _full30 chain (seed_arm_chain 14691, now PPID 1) keeps flying s3-s5. NEW LEVER = scripts/queue_2009c_obs_pwm_recipe.sh
(pid 10056, PPID 1, same log /private/tmp/queue_2009.log, tag [q2009c]): waits for the chain -> CTRL-16 verdict (STOPs if
_full30 replaces the anchor) -> STEP 5b CTRL-21 probe (~47 min, controller gap) -> STEP 5c PARK until experiments/ABI31_LANDED (OPTION B, Luiz 26/09: the CTRL-17 stage-2 trainer fixes, ABI 31, altitude side incl., land HERE by hand + smoke ONE) -> _op31 x5 anchor re-fly -> `_tc31` x4 tilt-coherence arm (5°/5°, vs _op31) -> STEP 6c PARK for the tilt decision (experiments/tilt_decision = coherent|wide; apply experiments/ctrl_tilt_rule.json incl. the 15°/30° stress re-score via scripts/rescore_winners.py --tilt) -> LEVELS on the chosen curriculum (_Ltc31 or _Lop31). When parked, lines 2-3 say "PARKED BY DESIGN — ABI-31 landing" and the session LANDS per docs/ctrl17_abi31_landing_runbook.md (branch stage2-trainer-fixes 9aaaef1e, wheel staged, sha in the runbook) — nothing pending on Luiz for it.
Roadmap after STOP: CTRL-17 audit -> CTRL-8 stage 2 -> multi-axis round 1 (stage 2 BEFORE multi-axis, Luiz 26/09).
IDS: 94 QSR -abi13 reruns PAUSED. IDS IS HIGHEST PRIORITY — never stop/pause/hold IDS for controller work. Open checks all DONE: ga_state PASSED (WNN-1 closed), s3 marker REPAIRED 28/09 06:20 (backup in experiments/marker_backups/). s3 banked 05:51 EDT 28/09 (headline NEURONS#0 = the _pipeN30 s3 genome, bit-identical: the counter-RNG did NOT change the GRID/NEURONS prefix, so the old 'bit-identical for s2 only' era note below is WRONG). s4 flying.

AS OF 26/09/2026 15:10 EDT — WNN-1 LANDED, HOLD RELEASED, s3 FLYING. Merge 68f408c2 (ga-template-unify: WNN-1 +
eval-batch-pack) + worker wheel ABI 14 installed in one step; smoke rc 0 (5 stages, 15-candidate stage-select). s2
_full30 banked 14:40 (marker 1/4; writer bug fixed 64059ede — markers now carry held_stage_multiseed, all 5 stages;
s2 repaired from its .out). s3 _full30 launched 15:08 on git=68f408c2 (counter-RNG; packed eval sub-batches — watch
BITS widths: expect >> 2, the "[ControllerEvaluator] N genomes -> K sub-batches (widths [...])" line). OPEN CHECKS for
the ticks: (1) once s3 NEURONS writes an in-stage checkpoint (gen >= 2), read its header
(load_checkpoint(..., skip_population=True)) and confirm extra["ga_state"] = {stopper, scaler, incumbent} — the
smoke's 2-gen stages could not exercise the cadence path; (2) IDS: worker_swap pid 45016 watches flow 6253 and
restarts the worker when it ends (log /private/tmp/worker_swap_wnn1.log) — confirm the relaunch and that new IDS
flows run on ABI 14; (3) when s3 banks, REPAIR its marker the same way as s2 (its ladder sourced the pre-fix lib).
Era note: s2 = old RNG scheme, s3-s5 = counter-RNG (pairing vs _pipeN30 unaffected; the bit-identical NEURONS prefix
holds for s2 only).

AS OF 26/09/2026 09:30 EDT — LANDING AT THE s2->s3 HOLD CHANGED (Luiz): merge branch `ga-template-unify`
(b4392f95, WNN-1, CONTAINS eval-batch-pack) — NOT eval-batch-pack alone. It needs worker wheel ABI 14 installed in
the SAME step (wnn/accel.py asserts 14; controller runs import it too). Follow the LANDING RUNBOOK in
.claude/plans/ga_template_unification.md — it includes pausing queued IDS flows first. DECIDED (Luiz 26/09 ~13:15): deploy as soon as the CONTROLLER
allows (no waiting on IDS); the in-flight IDS flow finishes on its imported code, then worker_swap.py RESTARTS the
IDS worker clean (runbook steps 6-7), and the paused IDS flows are requeued after the relaunch. Wheel 14 is PRE-STAGED at /Volumes/20260401-WDBlack-SN850X-2TB/cargo-target/wheels/abi14_wnn1_b4392f95/;
merge-tree dry run is clean. When s2 banks and the box idles, say "idle BY REQUEST — WNN-1 landing" and DO the
runbook (nothing is pending on Luiz for it).

AS OF 25/09/2026 10:16 EDT — HOLD ARMED BY REQUEST (Luiz). experiments/HOLD_CONTROLLER exists. `_full30` s31337002
(CTRL-16) flies on untouched and banks normally; the chain then PARKS before s3 (wait_while_held logs `HOLD —` to the
arm log every 10 min). When the box goes idle with the sentinel present, lines 2-3 say "idle BY REQUEST — HOLD for the
eval-batch-pack merge", NOT a failure. At that idle: (1) print the s2 NEW-MARKER block (FIVE stage rows), (2) merge branch
`eval-batch-pack` (34d71ad4: eval sub-batches packed by the SUM of cells, not N x the largest genome; parity-tested),
(3) smoke one short run and confirm a `[ControllerEvaluator] N genomes -> K sub-batches (widths [...])` line with widths
well above 2, (4) rm the sentinel. WHY: s2 ran BITS at width 2 = 199% CPU of 1600% (~4.2 h/gen) because one 3.26M-cell
genome sized the whole population. CTRL-16 verdict rule PRE-REGISTERED (325e11ec, experiments/ctrl16_rule.json): gate bar
+ mean TRUE keys <= the anchor's 808,168; apply with `python scripts/ctrl16_verdict.py` once 4/4 bank. Stage line: print
BITS/CONNECTIONS widths when visible (`dagger-batch: w/w`).

AS OF 22/09/2026 05:00 EDT — GATE FIRED: PROMOTED (04:53 EDT). experiments/promotion_gate_2009.json: mean delta
_pipeN30 − _hd30 (n=4 headline) stable −1.0 pp · err −0.08° · steady −0.15° · alt −0.04 m → 3/4 favourable, err ≤ +0.10°.
Every 95% CI straddles 0 (a pre-registered MEAN rule fired; carry the caveat: sign 3/4 fires by chance 31%; s4 was
a 1/4 loser with alt the only win). ANCHOR = `_pipeN30` and RECIPE = `--skip-stages connections,bits
--max-output-neurons 512 --max-cells 1000000000` from here: STEP 3 `_op30` ×4 (CTRL-15, --obs-pwm ONLY, vs `_pipeN30`)
flying since 04:53 EDT → STEP 4 `_pon30` s2 → STEP 5 `_full30` ×4 → STEP 6 LEVELS n=5 → STOP → CTRL-19 `_alt30`
(now pairs vs `_pipeN30` with the promoted recipe) → multi-axis round 1 (MA_ANCHOR_SUFFIX=_pipeN30 + MA_RECIPE_EXTRA,
--patience-track-pool0 ON there). Line 2 names `_pipeN30` as the control. `_pipeN30` 4/4: hd 0.1165/0.1021/0.1906/0.1224
(ranks 19/7/>40/40), all QSPI-tier (676-1170k TRUE keys), 8.6-11.6 h each vs the anchor's 4.7-6.1 h.

AS OF 20/09/2026 19:10 EDT (addendum to the 15:40 19/09 block below; queue_2009b unchanged, STEP 2 flying).
`_hd30` 4/4 CLOSED 12:46 EDT: era A/B `_hd30 − _hd29` every CI straddles 0 (MEMORY row: stable −1.05 pp
[−5.5,+3.4] · err −0.25° [−0.79,+0.29] · steady −0.30° [−1.03,+0.44] · alt −0.05 m); the systematic change
is SIZE — 573-776k TRUE keys, every `_hd30` winner is QSPI tier (none fits the 2 MB internal flash).
GEN-LINE FORMAT CHANGED (723c1ed0): runs launched after ~19:00 EDT 20/09 print `best=… (new|=)` = pool[0]
and `| offspring:` — see the GEN-LINE SEMANTICS block above; s2 `_pipeN30` keeps the old `| gen:` form.
`--patience-track-pool0` landed DEFAULT OFF (551fe127) — NOT in any recipe; adopt at the multi-axis
round-1 boundary. CTRL-19 (Todo): `_alt30` ×4 = anchor recipe + `--fit-weight-alt 0.05` vs the anchor of
the day, patience flag OFF both sides, slot AFTER queue_2009b STOP and BEFORE multi_axis_chain --round 1.
Line 2 keeps naming the queue_2009b step; when the queue STOPs, name CTRL-19 as the next lever (Luiz to arm).

AS OF 19/09/2026 15:40 EDT (supersedes the 12:10 block below — ABI-30 LINEAGE RESTART FLYING).
WHAT HAPPENED 19/09 afternoon: the DAgger trainer never received the airframe (spec §5.1.3a, memory
project_trainer_airframe_gap — every cf21 run trained on the SYNTHETIC plant, hover 0.20 vs cf21's 0.694).
Luiz: fix + re-fly everything on the fixed trainer. `_hd29` s6 was allowed to bank (15:19 EDT, 98.8/1.73/1.19/
0.258 — the ABI-29 lineage is 5/5 and CLOSED); queue_1909 was then killed at HOLD (logged in its log), the
ABI-30 controller wheel installed (airframe-in-trainer fix + axis-F actuator lag `--motor-lag-s`, settling
time T; τ=T/4), pin held (60/60 legacy-plant lines bit-identical ABI 29→30), Rust 213/213, cf21 smoke rc 0.
LEVER = scripts/queue_2009b_promote_gate.sh (pid 67249, PPID 1; log /private/tmp/queue_2009.log; the 2009
supervisor was swapped out at 16:04 EDT, its `_hd30` chain 39966 untouched). PROMOTION GATE after step 2 (Luiz
16:0x): if `_pipeN30 − _hd30` favours `_pipeN30` on >=3/4 HEADLINE columns and err is not worse by >0.10°, the
neurons-GA pipeline becomes the RECIPE and `_pipeN30` the ANCHOR for `_op30`, `_pon30`, the levels rungs and the
multi-axis round 1 (experiments/promotion_gate_2009.json records the verdict; the tick reads it and names the
anchor on line 2 from then on). Order:
STEP 1 `_hd30` ×4 (anchor, ARM_NO_CONTROL; era A/B vs `_hd29` printed after) → STEP 2 `_pipeN30` ×4 vs `_hd30`
(CTRL-7) → STEP 3 `_op30` ×4 (CTRL-15) → STEP 4 `_pon30` s2 (CTRL-10) → STEP 5 `_full30` ×4 (CTRL-16) →
STEP 6 LEVELS n=5 (CTRL-18: `_hd30` s6, then 96 levels n384 `_L30` ×5, 128 levels n512 `_L30` ×5; off-chip
QSPI tier allowed) → STOP. ~78 h + ~48 h. Then multi_axis_chain.sh --round 1 with MA_ANCHOR_SUFFIX=_hd30.
SUFFIX RULE: the lineage suffix carries the ABI (`_hd30`) — same-suffix re-flies are SKIPPED as banked and
paired_power pairs by suffix. Line 2 names the current step's tag + what it tests. Any `_L30` marker has
levels_per_motor 96 or 128 — print it. The 4 worktree branches (actuator-lag, stats-tools, multi-axis-chain,
trainer-airframe) are MERGED (e43ae0a0); scripts/multi_axis_chain.sh --dry-run is the round-1 plan.
IDS (line 6 only): worker ABI 13, 6072 ~gen 85, 150 queued incl. 94 `-abi13` QSR reruns.

AS OF 19/09/2026 12:10 EDT (superseded 15:40 — queue_1909 killed after step 1; steps 2-5 re-flown as ABI-30 lineage).
LEVER = scripts/queue_1909_close_then_full.sh (pid 180, PPID 1; log /private/tmp/queue_1909.log). Order
(Luiz: "close what is in the middle before a new chapter"): STEP 1 `_hd29` s31337006 (control for the
5th seed; ARM_NO_CONTROL, log /private/tmp/hd29_s6.log) → STEP 2 `_pipeN` s31337006 vs `_hd29` + n=5
paired verdict (CTRL-7) → STEP 3 `_op` ×4 (CTRL-15, --obs-pwm only, vs `_hd29`; secondary vs `_bd`) →
STEP 4 `_pon` s2 (CTRL-10: plant ON, vertical features OFF, vs `_hd29`) → STEP 5 `_full` ×4 (CTRL-16:
GRID→NEURONS→BITS→CONNECTIONS→MEMORY, --skip-stages "," --bits-gens 5 --bits-patience 3, vs `_hd29`,
second read vs `_pipeN`) → STOP. ~60 h total; each arm's .out is logs/controller/sweep_ladder/<tag>.out.
Line 2 names the CURRENT step's tag + what it tests; markers count = the step's own seeds. Stage order
for `_full` has FIVE stage rows (GRID/NEURONS/BITS/CONNECTIONS/MEMORY) — print all of them on its marker.
Smoke of the 5-stage path passed 12:03 EDT (scratchpad/full_smoke, rc 0). IDS: ABI-13 swap COMPLETE 12:22 EDT —
6072 finished gen 67 on the new wheel clean, the watcher un-paused all 56 (queue 1 running / 150 queued incl. the
94 `-abi13` QSR reruns / 297 older paused untouched). 6073 (ablqsr) is the first QSR flow on the fixed trainer:
expect ~1.5-2 h, not 10 h — if it runs >4 h, say so on line 6.

AS OF 19/09/2026 12:00 EDT (superseded 12:10; BOX IDLE BY DESIGN — no longer true).
EVERYTHING THE 15/09 BLOCK QUEUED HAS FLOWN AND BANKED — the repair sequence (recalc passes)
finished, the offline-connectivity probe CLOSED (rho 0.33/0.29), stage D `_pipeN` 4/4 (5d81bb44),
arm B label-only `_dl` 4/4 (035e06b2), clean control `_hd29` 4/4 (b8068f05, 08:03 EDT 19/09).
NO chain, NO supervisor, NO phased_ga, NO repair job, NO probe is running or queued. Tick lines 2/3:
"no run — box idle BY DESIGN; pending on Luiz: pick the next lever (CTRL-15 `_op`, full pipeline arm,
5th seed)". Do NOT call the IDS worker a controller job.
VERDICTS (era-clean, arm − `_hd29`, n=4, mean [95% t-CI]):
  `_pipeN` (GRID→NEURONS→MEMORY): stable +2.05 pp [−1.6,+5.7] · err −0.21° [−0.50,+0.09] · steady −0.16°
  [−0.20,−0.12] EXCLUDES 0 · alt −0.07 m [−0.10,−0.05] EXCLUDES 0 · hd −0.041; all 4 winners FIT the H743
  (325k/351k/399,425 EXACT/325k TRUE keys). CTRL-7 → DONE 19/09 ("err unresolved at n=4").
  `_dl` (label only): null on attitude (err +0.13°, steady +0.17° 0/4), alt −0.17 m, 2-3x jerk. Not adopted.
  `_hd29 − _hd` era A/B: NO systematic trainer-era shift (every CI straddles 0) — STRADDLE caveat RETIRED.
EXPORTER FIXED 19/09: export_controller_c.py / count_true_keys.py read the GENOME's neuron count
(spec_from_arch: levels = output_neurons // motors) — `_pipeN` s4 on=248 = 4×62 levels, 1176 KB packed.
PLANE: CTRL-7 Done · CTRL-9 (`_dl`) banked, still In Progress (Luiz to close) · CTRL-15 `_op` ×4 in Todo
(NOT launched) · candidate not yet a story: FULL pipeline GRID→NEURONS→BITS→CONNECTIONS→MEMORY
(--skip-stages "" — phased_ga's native order; Luiz 19/09) ×4 ≈ 32 h.
IDS SIDE (for line 6 only): worker on ABI 13 since 11:40 EDT 19/09 (QSR OI-trainer fix); flow 6072
resumed from gen 66 as the smoke; 94 QSR `-abi13` reruns queued (ids 6264-6357) behind the 56 earlier
flows. IDS is FIFO — never reorder.
SERVICES: dashboard/worker/sampler/watchdog/vite all relaunched via detach_launch.py (PPID=1) after the
11:13 EDT CLI restart reaped the nohup-launched set; post_reboot_bringup.sh now uses the helper.

AS OF 15/09/2026 16:00 EDT (supersedes the 10:15 block below).
TICK RULE (Luiz 15:5x EDT): NEVER print "box idle" while a controller-side job runs. The marker-repair
sequence (recalc_headlines.py passes, pid 66162) and the offline-connectivity validity probe
(offline_connectivity_probe.py, worktree /Users/lacg/wnn-offline, isolated wheel) take the GPU/CPU —
line 2 names them with progress + ETA, line 3 carries the current pass's changed/unchanged/refused
tally and the probe's latest milestone. "box idle" is reserved for nothing controller-side at all.
FLYING: (1) repair sequence — rescore 3rd pass done → arch-only done (14 unch / 1 changed / 1 refused)
→ MEMORY pass 53 (since ~14:55 EDT, ~10 min/marker) → CONNECTIONS 75; ETA ~14:00-18:00 EDT 16/09;
then leaderboard precedence recalc > aligned > holdout (step 5). (2) offline probe — launched 15:35
EDT (scratchpad/probe_run.sh|.log): population = _bd s2 CONNECTIONS ckpt (57), dataset A = 200 DAgger
episodes over the 8→30° curriculum (400k records, 5 folds), proxy = mean held-out H(y|a) (elite 0.7357,
pop 0.7321-0.7389, H(y) 0.905), truth = evaluate_batch ZRank fitness; then dataset B at the scorer's
5°. Verdict = Spearman: rho >= 0.7 build · <= 0.4 CLOSED · between report. Output
experiments/offline_probe/PROBE_*.json.
QUEUED: scripts/post_repair_queue.sh (pid 28596, PPID 1) → stage D `_pipeN` ×4 → arm B label-only
`_dl` ×4 → STOP (both via scripts/seed_arm_chain.sh; CTRL-7 / CTRL-9 on the Plane board).

QUEUE AS OF 15/09/2026 10:15 EDT (POST-REBOOT, macOS 27; BOX IDLE BY DESIGN — the queue DRAINED;
supersedes the 13/09 15:50 block).
Everything the 13/09 block queued has FLOWN and BANKED: no-steal window 2/2 (b24n384 s2 _win2 + k=1
twin, 14/09), arm B re-fly 4/4 (`_bd` s2-s5, ABI 29, last banked 23:20 EDT 14/09, commit 1661bf55),
mut1tap 4/4 earlier. NOTHING IS QUEUED and NO chain/supervisor exists — that is the 13/09 STEP 4
"STOP". Tick lines 2/3: "no run — box idle BY DESIGN, queue drained; pending on Luiz: pick the next
lever". Candidate levers, all UNQUEUED (Luiz's call): window-k seeds 3-5 (declined 13/09), Stage D
pipeline A/B, stage-2 horizontal translation (needs trainer audit), 3rd translation arm (plant ON,
features OFF), sn>0 x altitude (low priority), offline connectivity (design first), leak multi-seed
ladder (conditional), the paper-rows stale-altitude rerun (declined 12/09).
ARM B 4/4 HEADLINE paired verdict (n=4, t-CI, arm − _hd control): err −0.39° [−0.68,−0.10] EXCLUDES 0
(4/4); steady −0.16° [−0.43,+0.12]; alt −0.10 m [−0.44,+0.23]; stable +0.25 pp [−1.7,+2.2] — steady/
stable/alt INDETERMINATE at n=4. Two-flag bundle. Arm winners 2.6-4.0x the H743 flash — NOT deployable.
REBOOT 15/09 ~09:36 EDT (macOS 27): scripts/post_reboot_bringup.sh relaunched dashboard (CARGO_TARGET_DIR
binary), IDS worker (rayon 13, requeued flow 5938), mem sampler + watchdog, vite (https://localhost:5173);
all PPID 1. The MARKER-REPAIR SEQUENCE WAS DELIBERATELY NOT RELAUNCHED (SKIP_REPAIR=1): both recalc
passes were refusing ~all rows — DIAGNOSED 15/09 09:3x EDT: GRID#2 is the only arch-only candidate in the
GRID top-3 (checkpoint cells-carrying idx [0,1,10,11,23,24,26,34] = the 8-genome holdout sample), so it is
RE-TRAINED by the recalc, and `build_run_args` rebuilds NO training-side flag (teacher-hover guessed
'legacy' when the marker lacks the field — every derived-hover run before ABI 29 refuses; and every run
flown before ABI 28 trained under the pre-stale-altitude-fix replay, so today's wheel can never reproduce
its cells). Gate 3 is CORRECT; its inputs are wrong. The 7 gate-1 "identity None" refusals are old .out
files with no `fit =` line (alt-weight sweep). Fix = Luiz's call: (a) rebuild the training flags from the
.out/marker (hover, leak, dmax, gamma, label scale) AND accept that pre-ABI-28 rows cannot pass gate 3 on
this wheel — score GRID#2 with cells trained TODAY and label the row, or (b) drop GRID#2 from the recalc
candidate set (select over the cells-carrying 8), or (c) abandon the recalc. Do NOT relaunch the sequence
as-is (each refused marker costs 12-36 GPU-min). Recalc so far: 1 accepted-unchanged (s2_bd), 0 changed.
XCODE 27 (auto-installed 00:13 EDT 15/09) BROKE /usr/bin/git: "You have not agreed to the Xcode license".
Workaround in any shell: `export DEVELOPER_DIR=/Library/Developer/CommandLineTools` before git. Permanent
fix needs Luiz: `sudo xcodebuild -license accept`. Every commit/push in a tick must set DEVELOPER_DIR.
Tick cron re-armed 15/09 after the reboot (session-only; CLI restarts kill it).

QUEUE AS OF 13/09/2026 15:50 EDT (WINDOW-K STOPPED at s2_win4; NO-STEAL WINDOW ARMS then ARM B;
supersedes the 12:05 block's queue).
Luiz: window-k n=1 per k on seed 2 already reads as a direction (win2 79.8%/3.96° → win3 16.0%/8.26°
→ win4 flying); framed1 at k=2 takes ~85 of the anchor's 256 current-frame neurons (2:1 recency
split). Window-k supervisor 61549 + refly queue 51294 KILLED (queued runs only); the ladder child
94138 keeps flying s2_win4 (banks ~19:20 EDT). Seeds 3-5 of window-k are NOT flown.
scripts/post_wink_queue.sh (pid 65814, PPID 1, log /private/tmp/post_wink_queue.log) waits for the
win4 marker, then: STEP 1 `SL_C_b24n384_..._s31337002_win2` — k=2 framed1 at n=384 = exactly 256
current + 128 previous (Luiz's "no-steal" arm; NOTE n=384 ⇒ 96 levels/motor vs the anchor's 64);
STEP 2 `SL_C_b24n384_..._s31337002` — k=1 n=384 twin (separates neuron count from the timeline);
STEP 3 arm B re-fly ×4 via armb_refly_queue.sh (~20 h); STEP 4 STOP — box IDLE, say so.
Lever lines: STEP 1/2 "no-steal window: does adding a previous frame (256+128) match the 256n
k=1 anchor?"; STEP 3 "arm B true-delta label (--dagger-label-delta --obs-pwm) vs the _hd controls —
RE-FLY on the obs_pwm-fixed wheel". Markers: STEP 1/2 n/2 (`b24n384.*s31337002(_win2)?\.json`),
STEP 3 `_bd\.json` n/4.

DEPLOYED 13/09/2026 14:34-14:37 EDT (armb_deploy.sh, exited): ram_controller ABI 29 INSTALLED
(wheel sha 026e02aa…) + EXPECTED_ABI=29 committed (0707c70a); smoke of the arm B bundle on the
installed wheel 100%/2.53° → HOLD lifted; window-k resumed with s2_win4 on ABI 29 at 14:37 EDT
(provenance line abi=29; obs_pwm-off runs bit-identical). win3 banked 14:33 EDT: HEADLINE
16.0%/8.26°/9.00°/0.539m (MEMORY#2) vs win2 79.8%/3.96° vs _crn 99.6%/1.41° — k=3 worse at n=1.
Remaining supervisors: window-k chain 61549 (10 runs to go) and armb_refly_queue 51294 (waits for
12/12 then re-flies arm B ×4 on the fixed wheel). Wheel-swap question CLOSED for the controller
wheel (the worker wheel is untouched, still ABI 12).

UPDATE 13/09/2026 13:00 EDT — obs_pwm needed a THIRD fix (05fe0f68): fixes 1+2 alone still left the
smoke dead (CPU miss diagnostic: 1-6 hits/256 per step vs 14-25 for the control). Cause: the raw
accumulator encodes the per-episode COLLECTIVE JITTER (anchor 0.66-0.76 at t=0) and splits the memory
by episode; the feature is now pwm − bank_anchor, the replay stream carries the feature value
(ReplayObs.pwm_dev). Smoke on the v2 wheel (logs/controller/armb_smoke_fixed2/): ctrl 100%/2.48° ·
+label-delta 90%/3.18° · +obs-pwm 100%/2.57° · both 100%/2.53° — all fly. The deploy script installs
whatever wheel sits at the fixed path: now the v2 build (sha 9739f144…, ABI 29, 208 tests). Nothing
else in the 12:05 block changes.

QUEUE AS OF 13/09/2026 12:05 EDT (obs_pwm FIXED e554661b, ABI 29 wheel BUILT; deploy + arm B re-fly
ARMED; window-k flying; supersedes the 10:45 block for what is queued).
Luiz 11:3x EDT: "Fix both!" — DONE in source (207 controller tests incl. a new CPU/GPU record parity
sweep with obs_pwm ON): ReplayObs.pwm_acc stream (rollout records the accumulator; CPU bptt/split +
Metal train/record buffers 25/22 restore it; fail-loud guards) + WnnController.set_pwm_accumulator
for the fitter (teacher's previous action ⇒ real ladder). obs_pwm-off runs bit-identical.
WHEEL: built, NOT yet installed — /Volumes/…/cargo-target/wheels/ram_controller-2026.212.37-cp311-
abi3-macosx_11_0_arm64.whl (sha 40b67d8c…). It also carries today's ram_core delta (the deferred
swap): additive + the never-flown QUAD_BINARY read. Provenance line records wheel sha per run.
THREE SUPERVISORS, all PPID 1, all fail closed:
  1. window-k chain (queue_after_ab_chain.sh 61549) — run 2/12 s2_win3 flying, ETA ~15:20 EDT.
  2. armb_deploy.sh (50153, scratchpad, log /private/tmp/armb_deploy.log): HOLD is SET (12:02);
     when win3 banks the ladder parks; the deploy installs the wheel + sets EXPECTED_ABI=29 in ONE
     step, smokes the arm B bundle (logs/controller/armb_smoke/both_fixed.out), and rm's the HOLD
     ONLY if the smoke flies (stable>=50%, err<=10°). If it fails, HOLD STAYS and the box idles —
     escalate; wheel and Python stay consistent either way. Expect ~10 min idle around 15:20-15:35.
     While it runs the controller count reads 1 (the smoke) — expected. src/wnn/control/_accel.py
     will show as modified (EXPECTED_ABI 28→29) — commit it after the deploy log says installed.
  3. armb_refly_queue.sh (51294, scripts/, log /private/tmp/armb_refly_queue.log): waits for
     12/12 _win markers AND the window-k chain to exit, checks the installed ABI >= 29, archives
     the void `_bd` s2 marker/out/winner to experiments/sweepladder_markers_void_abi28/ (README),
     then runs arm_b_delta_label_chain.sh (4 seeds, ~20 h). Lever line then: "arm B true-delta
     label (--dagger-label-delta --obs-pwm) vs the _hd controls — RE-FLY on the obs_pwm-fixed wheel".
Tick: lever = window-k until its 12/12; box line counts 1 controller. Any "ABORT" in the two
armb logs is an escalation.

ARM B DIAGNOSED (13/09 11:05 EDT, smoke DONE — box back to ONE controller = window-k).
Smoke logs/controller/armb_smoke/{ctrl,obspwm,labeldelta,both}.out (same shape/recipe, tiny budget):
ctrl 100%/2.48° · +label-delta 90%/3.18° (flies) · +obs-pwm 0.0%/70.1° (DEAD) · both DEAD.
`--obs-pwm` is the killer, `--dagger-label-delta` is innocent. TWO defects on the pwm feature:
 (1) THRESHOLD FITTER (evaluator.py fit_thresholds_from_pid_rollouts): the feature controller it
     drives is untrained → its accumulator never leaves the anchor → the 4 pwm features are
     CONSTANT during calibration → quantile ladder DEGENERATE (all 8 thresholds = 0.5, span 0,
     verified). Any deploy deviation flips all 32 pwm bits → every RAM lookup misses → student
     does nothing (effort 0.9, mono_viol 2).
 (2) REPLAY (Rust, documented KNOWN GAP, test replay_parity_for_policy_state_features
     OBS_PWM_FIXED=false): training replays the recorded trajectory without restoring the
     accumulator, so pwm features are frozen at hover in TRAIN and evolve in DEPLOY. Fix = the
     same student_pwms plumbing as Fix A (obs_dhat).
Both must be fixed before --obs-pwm means anything; arm B (true-delta label) NEEDS obs_pwm by
design ("or the student is taught a function it cannot see"). Every earlier --obs-pwm driver
(c2k, bit_sweep, e5 x3, frame_fix x3, low_edge) carried at least defect (2). DECISION IS LUIZ'S:
fix (1)+(2) → controller wheel rebuild (swap-free) + Python fitter → smoke → re-fly arm B ×4
(~20 h) — or drop arm B. NOTHING queued for arm B until he rules.

QUEUE AS OF 13/09/2026 10:45 EDT (HOLD LIFTED; post-D0 queue KILLED; WINDOW-K FLYING directly;
supersedes the 03:30 block).
Luiz (10:3x EDT): "we have lots of things on queue... why is it on hold?" — the HOLD idled the whole
sequential queue behind one broken arm. FIX: killed the parked supervisors (post_d0_queue 6225,
arm_b_delta_label_chain 85345, sweep_ladder_gamma 15369 — all sleeping in wait_while_held, NO run
was computing), rm'd experiments/HOLD_CONTROLLER, launched STEP 4 directly:
  scripts/queue_after_ab_chain.sh  pid 61549 (PPID 1), log /private/tmp/queue_after_ab.log,
  nohup /private/tmp/queue_after_ab.nohup. Idempotent: probe markers + leak090 4/4 + s2_win2 SKIPPED;
  flies the remaining 11 window-k FRAMED runs, seed-major: s2 win3, win4; s3 win2,3,4; s4 win2,3,4;
  s5 win2,3,4 (~4.7 h each, ~50 h). Controls = the banked b24 n256 CRN runs (s2 = _crn).
  Lever line: "window-k FRAMED (input-window-k 2/3/4, conn-policy framed1, stride 10) vs the
  b24 n256 anchors — does a longer framed input window buy attitude without an altitude cost?"
  Markers count: `_win[234]\.json$` → n/12 (s2_win2 already banked 09/09).
ARM B (STEP 3) is OUT of the queue: s2 `_bd` VOID (trainer dead at every stage, 0.0%/57°), runs
2-4 NOT flown. Pending on Luiz/me: diagnose the --dagger-label-delta --obs-pwm trainer under a
tiny smoke (levels 6) BEFORE any re-fly; then a fresh arm B chain (arm_b_delta_label_chain.sh is
idempotent — it would skip s2 unless the void marker is moved aside). Do NOT relaunch
post_d0_queue.sh as-is (it re-enters arm B at 1/4).
Wheel-swap question (22:45 block) still open; no swap unasked.

QUEUE AS OF 13/09/2026 03:30 EDT (arm B run 1/4 BANKED VOID 03:01 EDT; box HELD before run 2/4;
supersedes the 01:05 block — which follows and still holds for the 2x2 verdict + arm B diagnosis).
`_bd` s31337002 marker: rc 0, 8424 s, HEADLINE 0.0% / 57.15° / 57.15° / 0.238 m (CONNECTIONS#0);
GRID 0.0±0.0% / 62.05±1.96°; CONNECTIONS = MEMORY 0.0±0.0% / 57.15±2.62°; effort 0.916. VOID row —
NOT banked into git/leaderboard by the tick (a dead trainer is not a result); leave it on disk for
the diagnosis. The ladder (sweep_ladder_gamma.sh, child of arm_b_delta_label_chain.sh 85345, of
post_d0_queue.sh 6225) is parked in wait_while_held before s31337003_bd — /private/tmp/
sweep_ladder_gamma.log prints "HOLD — still held" every 10 min. NOTHING FLIES until
experiments/HOLD_CONTROLLER is removed. Tick lines 2/3 while held: "no run — HOLD set after arm B
run 1/4 banked void (grid dead 0.0%/62°); pending: Luiz's arm B diagnosis, then rm the HOLD (resumes
at arm B run 2/4) or kill the queue supervisor + relaunch with arm B fixed/bypassed".

QUEUE AS OF 13/09/2026 01:05 EDT (2x2 COMPLETE 4/4 + VERDICT; arm B run 1/4 flying DEAD-ON-ARRIVAL;
HOLD SENTINEL SET by the tick; supersedes the 22:45 block).
2x2 leak×label-scale COMPLETE 4/4, banked (s2/s3/s4/s5 `_l090_ls2`, ABI 28). VERDICT on the HEADLINE
(paired, n=4, t-CI): interaction (leak cost at s=2 − at s=1) err +0.16° [−0.88,+1.19] · steady +0.23°
[−0.66,+1.11] · alt −0.24 m [−1.16,+0.68] · stable −0.8 pp [−4.6,+3.0] — ALL straddle 0; the leak's
altitude cost at s=1 +0.35 m [−0.54,+1.25] (s3's 1.545 m drives it), at s=2 +0.12 m [−0.12,+0.35];
NO evidence s=2 removes the leak's altitude cost; the two levers are independent at n=4 (a direction).
s2's control is `_crn` (the bare s2 is rotation-era) — the chain's own ctrl_tag does the same.
STEP 3 arm B (`--dagger-label-delta --obs-pwm --teacher-hover derived`, controls `_hd`) run 1/4
`SL_C_..._s31337002_bd` started 00:40 EDT under scripts/arm_b_delta_label_chain.sh (pid 85345) →
phased_ga 85433. ITS GRID IS DEAD: winner 0.0% stable / 60.19° err; GRID multi-seed held-out
0.0±0.0% / 62.05±1.96° / mono_viol 2 / effort 1.19; ALL 55 grid+expand points ~70°/0.0% (control
`_hd` s2 grid: 98.4% / 1.93°). Structural, not a draw — the two-flag bundle had NEVER flown end-to-end
(d0_smoke flew hover only; no .out carries the flag). ESCALATED at the 01:00 tick; Luiz asleep.
ACTION TAKEN: `experiments/HOLD_CONTROLLER` touched 01:02 EDT (reversible: rm) — the chain's
wait_while_held holds BEFORE run 2/4 launches, after run 1/4 banks (~05:30 EDT). Run 1/4 was NOT
killed (never kill a running run unasked). Lever line for arm B while it flies: "arm B true-delta
label (--dagger-label-delta --obs-pwm) vs the derived-hover anchors — GRID DEAD, escalated, HOLD set".
FOR LUIZ (morning): (1) decide arm B — diagnose the delta-label trainer under a smoke
(scripts/…smoke with --dagger-label-delta --obs-pwm, levels 6, tiny budget) before any re-fly; the
`_bd` s2 marker, if it banks, is a VOID row (label it); (2) rm the HOLD to resume the queue (it
would resume at arm B run 2/4 — kill the queue supervisor first if arm B is to be skipped, then
relaunch post_d0_queue.sh only once arm B is fixed or its step is bypassed); (3) the wheel-swap
decision from the 22:45 block still stands (no swap unasked).
Tick lines while HOLD is set and the box is idle: "no run — HOLD set after arm B run 1/4 (grid dead);
pending: arm B diagnosis, then rm experiments/HOLD_CONTROLLER".

QUEUE AS OF 12/09/2026 22:45 EDT (2x2 run 4/4 flying; public package + wgpu landed; WHEEL
SWAP DECISION DEFERRED to Luiz tomorrow morning; supersedes the 12:50 block).
2x2 leak×label-scale (post-D0 step 2): runs 1-3/4 banked (s2/s3/s4 _l090_ls2), run 4/4 = seed
31337005 flying since 21:42 EDT (ETA ~00:40 EDT 13/09). Interaction so far (HEADLINE, difference of
differences): s2 alt −0.13 m / err +0.26°; s3 alt −1.06 m / err −0.66°; s4 alt +0.28 m / err +0.92°
— three seeds, three signs; verdict = paired CI at n=4 (leak_x_labelscale_chain.sh prints it).
Then the queue continues on its own: STEP 3 arm B (4 runs `_bd`, controls = _hd), STEP 4
window-k 2..12 (11 runs), STEP 5 stop. Lever line for arm B: "arm B true-delta label
(--dagger-label-delta --obs-pwm) vs the derived-hover anchors".
SOURCE/WHEEL NOTE: ram_core source (public package work, 12/09) is ahead of BOTH installed research
wheels; the delta is additive + the QUAD_BINARY read (mode never flown); controller 206 + worker 116
tests pass on today's source; ABI unchanged. NO swap unless Luiz says so (decision deferred to 13/09
morning); provenance line already records git sha per run. Options if he wants one: controller =
touch experiments/HOLD_CONTROLLER → holds after the current run banks → smoke → install → rm;
worker = scripts/worker_swap.py --auto-detect-running --install-wheel … at the flow boundary.

QUEUE AS OF 12/09/2026 12:50 EDT (RERUN DECLINED — option (a); POST-D0 QUEUE RELAUNCHED;
supersedes the 10:15 block).
`_fix` banked 12:26 EDT (5d423ef2): HEADLINE 97.6%/1.75°/1.33°/0.335m; one-flag pair vs void
`_hd` s2 (same recipe, bug only): −0.6pp/−0.32°/−0.19°/−0.351m — the fixed wheel is BETTER. Luiz (12:4x EDT): the fix is good but the shift is NOT worth 150 h —
NO rerun; void rows STAND with a caveat; revisit (b) step-2-only or (c) full if later data
says so. scripts/stale_altitude_refly_chain.sh stays on disk, go-gated (SAR_GO=1), unused.
HEADLINE ONLY (Luiz, 12/09): every pair is read on the HEADLINE held-out; never lead with a
stage row. Seed-2 headline: _fix 97.6%/1.75°/1.33°/0.335m · _hd (bug only) 98.2%/2.07°/1.52°/
0.686m · _crn (bug + legacy hover) 99.6%/1.41°/0.85°/0.300m — the fix beats _hd, _crn beats
both; n=1. No straddle caveat is to be printed (Luiz removed it).
RELAUNCHED 12:49 EDT: scripts/post_d0_queue.sh (pid 6225, PPID 1, log
/private/tmp/post_d0_queue.log, nohup /private/tmp/post_d0_queue.nohup). Step 1 skipped
(4/4 _hd). Live steps, ONE controller at a time, marker-gated, HOLD sentinel honoured:
  STEP 2  2x2 leak x label-scale, LS_STAR=2 forced, derived hover, 4 runs `_l090_ls2`
          (~20 h) — leak_x_labelscale_chain.sh, log /private/tmp/leak_x_labelscale.log.
          Lever line: "2x2 leak×label-scale — does s=2 remove leak 0.90's altitude cost?"
  STEP 3  arm B true-delta label (--dagger-label-delta --obs-pwm), derived hover,
          controls = _hd, 4 runs `_bd` (~20 h) — sentinel LABEL_REBASE_LANDED present.
  STEP 4  window-k FRAMED runs 2..12 (~50 h) — queue_after_ab_chain.sh.
  STEP 5  STOP — multi-axis programme is spec only. Box IDLE; say so.
Idle window: touch experiments/HOLD_CONTROLLER (never kill the queue).

QUEUE AS OF 12/09/2026 10:15 EDT (RE-FLY `_fix` FLYING; rerun chain WRITTEN, go-gated;
supersedes the 10:xx block).
FLYING: SL_C_b24n256_cf21_brushless_L4C_g10_s31337002_fix (seed-2 anchor on the fixed
wheel, derived hover, CRN) under /private/tmp/…/scratchpad/stale_refly.sh → sweep_ladder_gamma
→ phased_ga; started 09:33 EDT, ETA ~18:40 EDT. Lever line: "stale-altitude fix re-fly —
first read of how bad the void data was". On its marker: bank (count_true_keys →
gate_distance_leaderboard → git add -f → commit/push) and report the PAIR per stage
(GRID/CONNECTIONS/MEMORY/HEADLINE, four columns): the ONE-FLAG pair is `_fix` vs `_hd`
s31337002 (both derived hover; only the fix differs); `_crn` also differs by hover.
Luiz's DECISION (12/09 09:5x EDT): IF the pair shows a material shift → rerun the PAPER
ROWS ONLY, 33 runs / ~150 h: scripts/stale_altitude_refly_chain.sh (written, dry-tested on
a scratch copy, refuses without SAR_GO=1). Rankings whose arms were all equally affected
(gatedwsweep, fitnessab, altweight, specialist1, stage1lambda, bits round 1, the round-2
grid) are NOT re-flown. The chain archives the void markers to *_markers_void_abi27/
(README in each), re-flies under the same tags, and pairs only against abi-28 markers.
LAUNCH (only after Luiz's go, box idle, `_fix` banked):
  SAR_GO=1 nohup bash scripts/stale_altitude_refly_chain.sh \
    > /private/tmp/stale_altitude_refly.nohup 2>&1 &   (start_new_session; verify PPID=1)
  then git add -f the *_void_abi27 dirs + commit. Log /private/tmp/stale_altitude_refly.log.
IF the pair shows NO material shift → nothing reruns; the void rows stand with a caveat.
The post-D0 queue must NOT be relaunched as-is. HOLD sentinel live.

QUEUE AS OF 12/09/2026 10:xx EDT (STALE-ALTITUDE-FEATURES BUG FIXED; box IDLE awaiting
Luiz's rerun-scope decision; supersedes the 02:30 block).
Controller wheel ABI 28 INSTALLED (marker-provenance merged 50a9bdba + the fix): every
replay trainer (CPU BPTT, CPU split, Metal train/record, Python fallback refuses) now
re-applies the per-step vertical/horizontal observation the rollout recorded; 206 Rust
tests pass incl. 5 new pins; smoke rc 0 (logs/controller/stale_altitude_fix/smoke.out,
provenance line abi=28 wheel_sha256=f8aafb70e1dcd641). The 148 banked altitude-regimen
runs (594 h, 14/08→12/09, incl. the 4 anchors, arm A, D0 A/B) are VOID —
docs/controller_stale_altitude_rerun_inventory.md. NOTHING FLIES until Luiz sets the
rerun scope/order; the post-D0 queue must NOT be relaunched as-is (its steps compare
against void markers). Tick lines 2/3: "no run — box idle; stale-altitude fix landed,
awaiting rerun order (inventory doc)". HOLD sentinel is now live in controller_arm_lib.

QUEUE AS OF 12/09/2026 02:30 EDT (A/B COMPLETE; box IDLE BY DESIGN; supersedes 21:26).
D0 A/B 4/4 banked 02:10 EDT: EQUIVALENT — err −0.008° CI [−0.30,+0.29], alt +0.004 m
CI [−0.23,+0.23], both straddle; derived hover CONFIRMED as default (spec §0.10,
docs/controller_d0_ab_verdict.txt, commit 62084929). The d0 chain exited; the queue was
killed 21:26 by request. NOTHING IS FLYING AND NOTHING WILL LAUNCH — that is the
requested idle window. Tick lines 2/3: "no run — box held idle for the
marker-provenance merge (spec §5); A/B complete 4/4". NOT an escalation. Do NOT relaunch.
NEXT (needs Luiz or an explicit order): merge branch marker-provenance (worktree
/Users/lacg/wnn-provenance: R9 provenance line + marker field, R11 stable_fail export +
scripts/stable_failure_ci.py, HOLD sentinel, harness fix) into
controller-grid-orchestrator-unification; smoke ONE 4-minute phased_ga; then relaunch
`bash scripts/post_d0_queue.sh` detached — idempotent, skips step 1, goes to the 2x2
(s=2 forced, derived) → arm B → window-k. ALSO pending at this window: training.py:513
rate=None refusal. OPEN DECISION: D8 (spec §0b) — the CPU replay trainer feeds stale
vertical features to EVERY altitude-regimen run; the probe is not built; Luiz may want
it before the 2x2 flies.

QUEUE AS OF 11/09/2026 21:26 EDT (queue KILLED by request; supersedes the 14:36 block).
post_d0_queue.sh (pid 38411) was killed at 21:26 EDT ON LUIZ'S ORDER to make an idle
window. The D0 A/B chain (d0_hover_ab_chain.sh, pid 40354, now PPID 1) and its runs
were NOT touched: run 3/4 s31337004_hd was flying at the kill and the chain launches
4/4 s31337005_hd by itself. After 4/4 banks (~05:00 EDT 12/09) the chain exits and
NOTHING launches — THE BOX IS IDLE BY DESIGN. Tick lines 2/3 then read:
"no run — box held idle for the marker-provenance merge (spec §5)". Do NOT relaunch.
PENDING AT THAT WINDOW (branch marker-provenance, worktree /Users/lacg/wnn-provenance,
commits 401e5a85 f6b0e512 + the HOLD sentinel): merge into
controller-grid-orchestrator-unification, smoke ONE 4-minute phased_ga, relaunch
`bash scripts/post_d0_queue.sh` detached (idempotent: step 1 complete → straight to
step 2). Also at that window: the training.py:513 rate=None refusal (Python-only).
From that merge on, an idle window is REQUESTED, never made by a kill:
`touch experiments/HOLD_CONTROLLER` → every chain banks its current run and waits
(logs "HOLD —"); `rm` it to resume. A tick that sees the box idle must check for
that file and say "idle by request (HOLD present)" if it is there.
A/B so far 2/4: s2 _hd 3 fails/500 vs _crn 2; s3 _hd 3 vs plain 5; err deltas
+0.24°/−0.01° — all CIs straddle, n=2 MDE ~2° err; nothing resolved (expected).


QUEUE AS OF 11/09/2026 14:36 EDT (D0 LANDED; supersedes the 09:50 block below).
D0 FIX DEPLOYED: merge c6adf6f9, controller wheel ram_controller-2026.212.37 ABI 27
installed at the idle window; --teacher-hover {legacy,derived} (default legacy, proven
BIT-IDENTICAL to arm A's banked smoke on the old wheel); derived smoke rc 0. Sentinel
experiments/labelscale_markers/LABEL_REBASE_LANDED.json created by the deploy (arm B
unblocked). Every chain now appends --teacher-hover ${TEACHER_HOVER:-derived}.
THE QUEUE — scripts/post_d0_queue.sh (pid 35306, PPID 1, log /private/tmp/post_d0_queue.log),
idle-gated, marker-gated, fails closed, IN ORDER:
  1. D0 A/B: scripts/d0_hover_ab_chain.sh, 4 runs tagged _hd (--teacher-hover derived) vs
     the banked legacy anchors (CRN for seed 2), ~20 h. Lever name: "D0 A/B (derived
     teacher hover vs legacy 0.5; equivalence check)". EXPECTED EFFECT ≈ 0 (probe ratio
     1.01). Read: ERR primary (MDE ~0.6°), ALT = no-regression check (a CI excluding 0 on
     alt is a BUG), steady secondary, stable descriptive. CI not tally. If both straddle
     zero → derived becomes the DEFAULT. Log /private/tmp/d0_hover_ab.log.
  2. 2x2 leak x label-scale FORCED on s=2 (LS_STAR=2), derived hover, 4 runs _l090_ls2,
     ~20 h. Lever name: "2x2 leak x label-scale (s=2 forced, derived hover)". Inputs (s=2
     arm, leak-0.90 control) were LEGACY-trained — state it; the A/B bounds that gap.
  3. ARM B true-delta label, derived hover, controls = the _hd A/B runs, 4 runs _bd, ~20 h.
     Lever name: "arm B true-delta label (2-flag bundle, derived hover)". Gated on the
     sentinel (present). Log /private/tmp/arm_b_delta_label.log.
  4. WINDOW-K FRAMED runs 2..12, derived hover, via queue_after_ab_chain.sh, ~50 h.
  5. STOP — multi-axis programme has a spec (docs/multi_axis_programme_spec.md draft 3 +
     §0 D0), no chains yet.
ARM A COMPLETE 12/12; CI read: s=8 costs +0.21 m alt / +1.2° err (CIs exclude 0), s=4
costs 0.8 pp stable, s=2 unresolved. No rung adopted.

QUEUE AS OF 11/09/2026 09:50 EDT (Luiz's order; supersedes the 08/09 22:20 block below).
MEASUREMENT FIX LANDED FIRST (commit 4b2bb46a, scripts/paired_power.py, output in
docs/controller_paired_power.txt). Three things every tick must now respect:
  · VERDICTS come from the mean paired delta and its 95% CI on the MEMORY row, NOT from
    a win/loss tally. "k of n wins" is a sign test: 3/4 fires 31% of the time on a DEAD
    lever, 67% across three rungs. Quote a tally only as a descriptive aside.
  · SIZE BEFORE FLYING. Observed paired SD at b24 n256: steady ~0.33-0.48 deg, err ~0.3,
    stable ~0.4-0.9 pp, ALTITUDE ~0.04-0.08 m. n=4 resolves ~0.6 deg on steady and
    ~0.1 m on altitude. A steady null at n=4 is INDETERMINATE, never a refutation.
  · ARM A's real result: the label scale measurably COSTS altitude at s=8 (+0.22 m, CI
    excludes 0) and stability at s=4 (+0.8 pp, CI excludes 0); steady unresolved.
    s=2 2/4, s=4 1/4, s=8 0/4 on the old tally — no rung "qualified", and that is not a
    refutation either.
THE QUEUE — scripts/post_arma_queue.sh (pid 93298, PPID 1, log
/private/tmp/post_arma_queue.log) replaced the bare 2x2 launcher (91240, killed 09:50).
It waits for arm A 12/12 + idle box, then runs IN ORDER, marker-gated, fails closed:
  1. 2x2 leak x label-scale FORCED ON s=2 (LS_STAR=2): 4 runs _l090_ls2, ~20 h. s=2 was
     FORCED, not selected — say so in every report. Lever name: "2x2 leak x label-scale
     (s=2 forced): does s=2 remove the leak's altitude cost". Read = the altitude
     INTERACTION on the MEMORY row, four columns, CI not tally. Chain log
     /private/tmp/leak_x_labelscale.log.
  2. ARM B true-delta label — HELD (11/09 10:30 EDT, D0 investigation): under
     --translation the delta label = pid_pwms − leaked_baseline (~0.694) while the mpcof
     teacher emits ~0.5 at level → every level step labelled "max descend". The queue
     SKIPS step 2 with a loud "HOLD" line unless
     experiments/labelscale_markers/LABEL_REBASE_LANDED.json exists (touched only by the
     label re-base deploy). The chain's own preflight refuses the same way. Queue was
     killed in its pure-wait phase and relaunched (new pid in the log). Do NOT create the
     sentinel by hand. Spec docs/multi_axis_programme_spec.md §0.9.
  3. WINDOW-K FRAMED runs 2..12 via scripts/queue_after_ab_chain.sh (skips its done
     steps), 11 runs ~50 h. Run 1 collapsed (72%/4.43 deg). Lever name: "window-k
     FRAMED ladder". Log /private/tmp/queue_after_ab.log.
  4. MULTI-AXIS programme: NO SCRIPT, NO SPEC. The queue STOPS after step 3 and says
     so; the box goes idle awaiting a written design.
ARM A COMPLETE 12/12 (12:26 EDT 11/09, all banked, chain exited). THE BOX IS IDLE BY
DESIGN (Luiz, Priority 0): the post-arm-A queue was STOPPED so nothing launches until
the D0 fix lands. D0 = DAgger training teachers hard-code hover 0.5 while the plant
hovers at √(m·g/4k)=0.694; probe says the label effect is immaterial (ratio 1.01) but
the fix is required (PID trainer saturates, arm B's delta label is void). Rust fix in
progress on branch hover-anchor-derived (worktree). NEXT: deploy controller wheel at
this idle window, smoke ONE, A/B 4 anchor seeds (--teacher-hover derived vs banked),
then re-arm the queue with derived ON. Tick line 2/3 while idle: "no run — box held
idle for the D0 fix (spec §0)".


QUEUE AS OF 08/09/2026 22:20 EDT (Luiz chose option b; supersedes the 20:30 block below).
1. IN FLIGHT: SL_C_b24n256_..._s31337002_win2 (window-k FRAMED k=2, run 1 of 12) under
   its ladder (pid 18075, PPID 1) — the queue SEQUENCER (queue_after_ab_chain.sh, 99439)
   was KILLED 21:58 EDT by request; the run + ladder continue and will bank the marker.
   Its GRID winner was 14.8% / 8.42 deg / 11.69 deg (below the gate) — the framed repair
   did not repair GRID; CONNECTIONS/MEMORY may recover. Window-k runs 2..12 are DEFERRED:
   relaunch = re-run scripts/queue_after_ab_chain.sh (marker-gated, skips done steps).
2. ARMED: scripts/launch_label_scale_when_idle.sh (pid 27801, PPID 1) waits for the box to
   go idle, then execs scripts/label_scale_arm_chain.sh — LABEL-SCALE ARM A, s in {2,4,8}
   with --delta-max 0.1/s (G held at 4), tags _ls2/_ls4/_ls8, seed-major rounds, 12 runs
   ~55 h, controls = the banked b24 n256 CRN runs. Lever name for ticks: "label-scale
   arm A (dead zone s x narrower, same gain)". Markers n/12 in sweepladder_markers; log
   /private/tmp/label_scale_arm.log. Step 0 smoke (pins + tiny phased_ga) runs first.
   DEPLOYED 08/09 22:05-22:20 EDT: wheel ram_controller-2026.212.37 installed (ABI 26),
   branch label-scale-and-legacy-fix MERGED (ca946474) + helper fix (970cd4d9). The
   in-flight run keeps its old .so and its already-imported modules — unaffected.
3. QUEUED behind arm A (Luiz 09/09 06:50 EDT): the 2x2 leak x label-scale — the ONE missing
   cell (leak 0.90, s=s*), 4 runs tagged _l090_ls{s*}, s* = arm A's best rung by paired
   steady wins (>=3/4 required, else the chain aborts; LS_STAR overrides). Launcher
   scripts/launch_leak_x_labelscale_when_ready.sh (pid 91240, PPID 1) waits for 12 _ls
   markers + idle box, then execs scripts/leak_x_labelscale_chain.sh. Lever name: "2x2
   leak x label-scale (does s* remove the leak's altitude cost)". Log
   /private/tmp/leak_x_labelscale.log. Read = the altitude INTERACTION, four columns.
4. THEN arm B --dagger-label-delta --obs-pwm (true-delta label), same controls.
LEAK-0.90 LADDER: COMPLETE, a TRADE (attitude 3-1 arm, altitude 0-4), NOT adopted — see the
20:30 block. Read every actuation arm in gain terms G = 2*dmax/(1-leak).

QUEUE AS OF 08/09/2026 20:30 EDT (supersedes the 06/09 block below, kept for provenance).
LEAK-0.90 LADDER COMPLETE 4/4 (banked 20:18 EDT 08/09). Paired MEMORY same-rule row, all
four columns (arm = --delta-leak 0.90, control = banked CRN b24 n256 at 0.95):
  seed      arm  stable/err/steady/alt          control                        d(hd)    d(alt)
  31337002  99.6 / 1.27 / 0.80 / 0.361 (.0932)  99.6 / 1.41 / 0.85 / 0.300 (.1029) -.0097  +.061
  31337003  99.0 / 1.67 / 1.08 / 0.531 (.1285)  99.0 / 1.65 / 1.35 / 0.351 (.1271) +.0014  +.180
  31337004  99.2 / 1.43 / 0.99 / 0.436 (.1093)  97.6 / 1.89 / 1.38 / 0.386 (.1615) -.0522  +.050
  31337005  99.6 / 1.51 / 0.95 / 0.463 (.1099)  99.6 / 1.58 / 1.11 / 0.349 (.1147) -.0048  +.114
  TALLY attitude (same-rule): arm 3 - 1 control. ALTITUDE: control 4 - 0 arm (+0.10 m mean,
  0.35 -> 0.45 m). VERDICT: a TRADE, not a promotion — the arm buys ~0.1 deg err / ~0.2 deg
  steady with ~0.1 m of altitude hold. Read in GAIN terms: G = 2*dmax/(1-leak) = 2 vs 4,
  tau 10 vs 20 ms — the accumulator bleeds the collective correction twice as fast. Also
  4/4: the arm's MEMORY row == its CONNECTIONS row (MEMORY improved nothing under 0.90).
  NOT adopted. Leak stays 0.95 in the recipe.
NOW RUNNING (queue_after_ab_chain.sh pid 99439, STEP 3): WINDOW-K FRAMED @ b24 n256, k in
{2,3,4} x seeds {2,3,4,5} = 12 runs (~55 h), tags _win2/_win3/_win4, flags --conn-policy
framed1 --output-full-window --input-window-k K --frame-stride 10 --conn-mutation-scope
window; k=1 control = the banked b24 n256 runs. Lever name: "window-k FRAMED ladder".
Markers n/12 in experiments/sweepladder_markers. Log /private/tmp/queue_after_ab.log.
STAGED, NOT DEPLOYED (memory project_live_dagger_label_dead_zone): the DAgger label fix
branch `label-scale-and-legacy-fix` (262574a7; wheel built in /Users/lacg/wnn-labelfix/
wheels, ABI 26 additive, 196/196 tests). The live label is the teacher's ABSOLUTE pwm
floored to the 1/64 grid (dead zone +-0.0156 pwm; motors 1/3 labelled neutral ~2/3 of the
hold window). ARM A chain WRITTEN, not launched: scripts/label_scale_arm_chain.sh (s=2/4/8
with dmax 0.1/s, G held at 4). DEPLOY ORDER that needs no idle window: install the WHEEL
first (old Python + new wheel is safe and bit-identical; new Python + old wheel is NOT),
then merge the Python. Whether to interrupt window-k for arm A is LUIZ'S CALL — pending.
ALTITUDE COLUMN RULE stands: four columns on every controller surface.

QUEUE AS OF 06/09/2026 08:40 EDT (supersedes the 04/09 block below, kept for
provenance). THE 04/09 QUEUE DRAINED at 08:10 EDT 06/09: translation A/B 6/6 (ON-only),
CRN re-fly of b24 (0.1029 — best altitude-regimen row on BOTH scales), leak revisit 2/2.
BANKED READS: b32 n256 replicates at 5 CRN seeds (gate-dist 0.1325±0.0191, same-rule
0.1220±0.0149, err 1.62±0.15° on both scales). CRN at b24 s2 is worth -0.094 gate-dist /
-0.133 same-rule vs rotation, both scales AGREE (not a selection draw). The LEAK SCREEN IS
UNINTERPRETABLE: both arms CRN, control rotation, scorer effect >= the gap — the chain's
own "BETTER — promote" verdict must NOT be acted on. Within-era: 0.90 beats 0.80 by 0.077.
⚠️ NEW QUESTION (06/09, verified): the bits round-2 ordering may be a SCORER-ERA
ARTIFACT. Round 2 averaged seeds 2 (rotation) and 3 (CRN) per width. CRN-era-only
MEMORY-row (same-rule) means at n=256 — CURVE COMPLETE 07/09 01:55 EDT (3/3 markers):
  b24 98.95% / 1.63° / 1.17° / 0.346 m  same-rule 0.1266  (n=4, SD 0.0253, s2 .1029 s3 .1271 s4 .1615 s5 .1147)
  b28 99.00% / 1.69° / 1.27° / 0.314 m  same-rule 0.1295  (n=2, SD 0.0090)
  b32 99.24% / 1.62° / 1.09° / 0.317 m  same-rule 0.1220  (n=5, SD 0.0149, s2 .1122 s3 .1278 s4 .1241 s5 .1036 s6 .1424)
VERDICT: NO WIDTH SEPARATES. The PAIRED same-seed b24-vs-b32 tally over the 4 shared seeds
is 2-2 (b24 wins s2/s3, b32 wins s4/s5) — a dead tie by the paired-majority standard, which
is the standard here. Mean gap 0.0046 = under a third of b32's SD and a fifth of b24's.
The ordering flipped TWICE in one day as seeds landed (b24 led at n=2, b32 at n=3, tie at
n=4) — that is what an n<=5 read of a 0.005 effect does. Round 2's "b32 interior optimum"
is DEAD; nothing replaces it on attitude. DECIDE ON DEPLOYABILITY, not on hd.
NEVER write an ordering as a bare inequality; write it in words with 'lower hd = better'.
DEPLOYABILITY IS A HARD CONSTRAINT (Luiz, 06/09 18:10 EDT): a published winner must
fit the H743's 2 MB internal flash as TRUE-only keys (docs/chip_selection.md, 'Recipe
constraint'). Measured: b24 n256 fits (1.2-1.5 MB), b32 n256 does NOT (4.0-5.1 MB, off-chip
2-2.5x). So on the bits curve b24 is the deployable width unless seeds 4+5 sink it.
  1. RUNNING scripts/crn_refly_chain.sh (pid 77643, launched 08:30 EDT) — the 0.95 LEAK
     CONTROL re-fly under CRN: SL_C_b32n64_..._g10_s31337002_crn, same ladder invocation
     as b24, ~2 h. It is the missing comparator that makes the leak screen readable.
     Lever: "leak control re-fly (CRN)". Log /private/tmp/crn_refly.log.
  2. ARMED scripts/crn_bits_curve_chain.sh (pid 86645, launched 08:38 EDT; waits for #1's
     box) — Luiz's "#1, #2": (a) b28 n256 s31337002 under CRN via crn_refly_chain.sh
     (SL_C_b28n256_..._s31337002_crn, ~5 h) — the missing cell of a 3-width x 2-seed CRN
     curve; then (b) b24 n256 seeds 31337004 and 31337005 via the ladder directly
     (SL_C_b24n256_..._s3133700{4,5}, no rotation control exists, ~4.7 h each) — balances
     b24 against b32's five seeds. Sequential, marker-gated, fails closed. Verdict prints
     CRN-era per-width means. Lever: "CRN bits curve". Log /private/tmp/crn_bits_curve.log.
     ETA: #1 ~10:30 EDT 06/09 → 2a ~15:30 → 2b ~20:15 → ~01:00 EDT 07/09; then the racing probe (~1.7 h).
  3. CONDITIONAL (Luiz: "#5 if leak survives"): when #1 lands, PAIR leak 0.90/0.80 (CRN)
     against the 0.95 CRN control on BOTH scales with all FOUR columns. If 0.90 still
     wins by more than the pool noise (~0.4°/2.5pp), a leak multi-seed ladder earns a
     slot; if not, L3's refutation is UPGRADED to "at both alphabets". NOT auto-queued —
     report and wait for the call.
  4. RACING — PROBE COMPLETE 07/09 03:42 EDT, 3/3 markers. RECOMMENDATION: DO NOT
     IMPLEMENT. The design was RIGHT about the rung and WRONG about the premise.
     EXACTNESS PASSED: identical=True, max|dreward|=0.000000 — cutting at a fold
     boundary and resuming from exported cells reproduces straight-through bit-for-bit,
     no Rust change needed, exactly as derived.
     PREDICTIVITY FAILED at BOTH stages. Spearman(rank after fold f, rank after fold 5),
     60 candidates, keep top third:
       CONNECTIONS  f1 +0.236 (10/20)  f2 +0.248 (11/20)  f3 +0.215 (9/20)  f4 -0.006 (5/20)
       GRID         f1 -0.082 ( 5/20)  f2 +0.187 ( 9/20)  f3 +0.089 (6/20)  f4 +0.124 (7/20)
     rho ~0.2 means an early fold explains ~4% of the final ordering; GRID's fold-1 rho is
     NEGATIVE. Keeping the top third retains 5-11 of the true top-20 (chance is ~6.7).
     regret 0 at cuts 1-3 is NOT reassurance — the survivors happen to include the winner
     at a rank correlation that cannot be relied on — and at cut 4, the MOST-informed cut,
     the true best is LOST at both stages (regret 0.0945 / 0.1341). Racing this rung would
     discard candidates on noise to save ~40-50% of training. CLOSED unless a different
     rung is proposed (the DAgger ROUND inside a fold is the only untested one).
     Markers experiments/racing_markers/PROBE_{smoke,stage3_s31337002,stage0_s31337002}.json.
  5. THEN: mutation-step A/B, rate 1/32 (one tap per neuron) vs the current 0.1 per tap
     x 32 taps (P(untouched)=3%: every child rewires every neuron). Unblocked now that
     the scorer effect is measured.
  ALTITUDE COLUMN RULE re-affirmed 06/09 (see ALTITUDE IS NOT OPTIONAL above).

QUEUE AS OF 04/09/2026 21:30 EDT (item 1 rewritten 21:30 — OFF arms dropped; supersedes the FOUR-chain block below, kept for
provenance). POWER OUTAGE ~07:14 EDT 04/09 killed every process; everything was
relaunched 08:50 EDT with PPID=1 (dashboard from CARGO_TARGET_DIR → worker rayon 13
→ mem sampler + watchdog → translation_ab_chain.sh + leak_revisit_chain.sh; cron
re-armed as da2eb7c7). /private/tmp logs were wiped by the reboot, so chain logs
start at 12:50Z. BITS ROUND 2 IS DONE (sentinel BITS_ROUND2_DONE.json, 11:03Z,
eleven minutes before the outage): winner b=32 n=256 γ=1, mean hd 0.1191; the
seed-3 curve is FLAT (0.125-0.135 across b24-b32) and b32's seed-3 point is the
only rotation-era one, so "b32 interior optimum" rests on seed 2 — say so.
  1. scripts/translation_ab_chain.sh — ON-ONLY since 04/09 21:25 EDT (Luiz: "bank
     run 2, drop the OFF arms, keep ON"). THE OFF ARMS WERE DROPPED after ONE
     reference point: knowing what the altitude regimen costs attitude changes no
     lever (the axis is mandatory), the A/B is a 4-flag bundle that cannot split
     plant cost from feature cost anyway, and the recoverable gap (~0.3°) is the
     smaller part of the ~0.9° to MPCOF. The remaining budget flies the ON seeds
     31337003..31337006, which ARE the paired replication the b32 n256 record
     (hd 0.1129, n=1) needs. Chain relaunched via scripts/translation_ab_on_handoff.sh
     (killed the old loop in the marker window, patched a TAB_ARMS hook in, relaunched
     — the ~2-min race-window ON s3 launch was preempted and re-flown from scratch).
     TOTAL = 6 markers (ON×5 + OFF s31337002), 2/6 banked. ~5.9 h/run → 4 × 5.9 h
     → A/B ~05/09 21:00 EDT; then the b24 CRN re-fly (~5.9 h → ~06/09 03:00); then
     the leak revisit (2 × ~2.5 h → ~06/09 08:00). A run crossing 5 h is EXPECTED.
     THE SEED-31337002 PAIR (banked 04/09, n=1, 4-flag bundle: OFF = 5 features, no z
     plant — an OFF win reads "the plant AND/OR the 3 vertical features cost attitude"):
        stage            ON (8 feat, z plant)          OFF (5 feat, no z)        gap ON−OFF
        GRID  multiseed  97.0±2.3 / 1.73±0.37 / 1.27  99.0±0.9 / 1.66±0.26 / 0.97   +0.07° err
        CONN  multiseed  97.6±1.2 / 1.65±0.25 / 0.98  99.8±0.4 / 1.29±0.08 / 0.64   +0.36° err, +0.34° steady
        MEM   multiseed  99.8±0.4 / 1.58±0.16 / 1.14  99.4±0.5 / 1.31±0.12 / 0.66   +0.27° err, +0.48° steady
        HEADLINE (val)   97.6 / 1.64 / 0.98  hd 0.1442 (7th)   99.8 / 1.33 / 0.91  hd 0.0949 (1st)
     alt: ON 0.371 m (WNN emits collective); OFF — (no z axis, NOT 0.000). The
     regimen costs ~0.3° err and ~0.5° steady at this seed — inside the ~0.4° pool
     noise on err, n=1, a DIRECTION. OFF at hd 0.0949 is an ATTITUDE-ONLY number:
     it beats PID (0.1241) and edges MPC (0.0958) on the attitude-only scale, but it
     is NOT comparable to the altitude-regimen rows and is NOT a record claim.
     Lever name for ticks: "translation A/B (ON replication, CRN, b32 n256 γ1)";
     markers n/6. Log /private/tmp/translation_ab.log · handoff log
     /private/tmp/translation_ab_handoff.log · markers experiments/translationab_markers/.
     OPEN (Luiz's call, NOT queued): a third arm — translation ON, vertical features
     OFF (legal: phased_ga refuses features without the plant, not the reverse) —
     would split plant cost from feature cost at seed 31337002 (~6 h). Skipped
     unless the gap is judged worth a lever.
  1b. scripts/crn_refly_chain.sh (QUEUED 04/09, Luiz; re-gated 21:24 EDT on the
     5 ON TAB markers, TAB_ARMS=on) — waits on them, then flies b24 n256 γ1 seed 31337002 under CRN via the ladder
     script (SL_TAG_SUFFIX=_crn → marker SL_C_b24n256_..._s31337002_crn.json),
     ~5 h. CRN IS THE FIX, NOT AN ARM (Luiz: "we are not going back") — this is
     the paired MEASUREMENT of what it changed: same shape, same seed, only the
     scorer differs; control = the rotation-era marker (hd 0.1972). b32 was
     DROPPED 14:30 EDT: the A/B's ON arm at seed 31337002 is BYTE-IDENTICAL to
     SL_C_b32n256_..._g10_s31337002 (same FEAT/weights/gate/γ and the same
     train/test/val seeds 3072558954/2504449327/1029590071), so
     TAB_on_b32n256_..._s31337002 (CRN) IS the b32 re-fly — pair it against
     hd 0.1129 (99.8/1.59/1.13). READ (04/09): its MEMORY winner came in at
     99.8±0.4% / 1.58±0.16° / 1.14±0.09° — bit-for-bit the rotation number (the
     winner was already at the ceiling); headline differs only via stage-select
     (CONNECTIONS#2, hd 0.1442). But the final POPULATION (8 genomes × 5 report
     seeds) held out at 98.8%/1.65° under CRN vs 95.9%/1.96° under rotation —
     the search converged on a genuinely better pool, n=1. That is the CRN
     effect to look for at b24 (where the rotation winner was NOT at the
     ceiling). Verdict = the leaderboard rows + the population lines. Log
     /private/tmp/crn_refly.log. Lever: "CRN re-fly".
  2. scripts/leak_revisit_chain.sh — waits on the 5 ON TAB markers AND the b24 re-fly
     marker (re-gated 04/09 so the waiters cannot race), then delta_leak
     {0.90, 0.80} × 1 seed at b32 n64 vs the BANKED rotation-era SL_C_b32n64
     control (needs a CRN re-fly of the 0.95 control before it is read, +1 run).
  bits_round2_chain.sh and sweep_ladder_gamma.sh are FINISHED — not relaunched.
  Un-queued, Luiz's call: b24 s31337002 under CRN (the clean CRN-vs-rotation read,
  ~5 h) · the leak control re-fly · mutation-step A/B (rate 1/32).
THE QUEUE — FOUR chains (five processes), ONE controller at a time, ~90h to
drain (~06/09). Every gate below waits on MARKERS, never on a process, and every
wait is PURE: nothing here can preempt a live run. A marker is a CLAIM THE RUN
FINISHED, withheld on a watchdog kill (rc 143/137), on a crash, and on a clean
exit with no MEMORY triple — so "chain gone, markers missing" means a run needs
a human, and each gate FAILS CLOSED there, leaving the box idle to be inspected
rather than stacking work on a crash.
  1. PID 3384  scripts/sweep_ladder_gamma.sh — STAGE C, the levels ladder at
     gamma=1. Phase 1 (the gamma A/B) is DONE and its gate fired. Phase 2 is
     b in {36,32} x n in {64,96,256}, NEURON-MAJOR. Log /private/tmp/sweep_ladder_gamma.log
  2. PID 50191 scripts/sweep_ladder_gamma2_supervisor.sh — waits for all six
     gamma=1 markers, then PATCHES an SL_FORCE_PHASE2_GAMMA hook into
     sweep_ladder_gamma.sh (safe only once nothing is executing that file — bash
     resumes at a byte offset) and relaunches it at gamma=2 for n in {64,96}.
     n=256 is deliberately out: the alphabet probe refuted it on footprint and
     gamma's whole claim is resolution at ZERO extra footprint.
     Log /private/tmp/sweep_ladder_gamma2.log
  3. PID 54524 scripts/bits_round2_chain.sh — BITS ROUND 2 AT THE WINNING
     ALPHABET (inserted 02/09 01:03Z; Luiz: option A). Gated on the four gamma=2
     markers. Then: picks (n*, gamma*) = the lowest-hd point among the ten
     stage-C ladder markers (SL_R2_NEURONS / SL_R2_GAMMA override; the table is
     logged); patches SL_SKIP_PHASE1 + SL_SWEEP_LABEL hooks into
     sweep_ladder_gamma.sh in the same exit window the gamma=2 supervisor uses;
     relaunches THAT script per seed so the recipe is never copied: b in
     {24,28,32,36,40} x seed 31337002 (the banked b32/b36 points at (n*,gamma*)
     are reused, never re-flown), cull top-3 / within 1.25x on hd, then seed
     31337003 on the survivors. ~6 new runs, ~23h at n*<=96. Writes
     experiments/sweepladder_markers/BITS_ROUND2_DONE.json {bits,neurons,gamma,
     means,survivors} ONLY if every survivor carries every seed. Tags are SL_C_*
     like the ladder's; the marker's "sweep":"bits-round2" is the provenance.
     ⚠️ A running sweep_ladder_gamma.sh whose .out tag has s31337003 or b in
     {24,28,40} IS round 2 — name the lever "bits round 2", not "levels ladder".
     ⚠️ If n*=256 the chain logs a warning: gamma=2 was never flown there and each
     run is ~4.5x the cost (~100h total) — SL_R2_NEURONS=96 is the override.
     Log /private/tmp/bits_round2.log
  4. PID 54525 scripts/translation_ab_chain.sh — the TRANSLATION A/B, RE-GATED
     02/09: waits on the gamma=2 markers AND the round-2 sentinel, then flies at
     the SENTINEL'S (b*, n*, gamma*) — no longer the hardcoded b32 n64 gamma=1 —
     2 arms x 5 seeds, seed-major, after PREFLIGHTING the OFF flag set against
     phased_ga's guards on the idle box. TAB_BITS/TAB_NEURONS/TAB_GAMMA override.
     Log /private/tmp/translation_ab.log
  5. PID 54526 scripts/leak_revisit_chain.sh — gated on the 10 TAB markers at
     ANY shape (glob; the old gate hardcoded b32n64 and would have waited
     forever), then delta_leak {0.90,0.80} x 1 seed at b32 n64 vs the banked
     SL_C_b32n64 control (leak 0.95). Log /private/tmp/leak_revisit.log
ETA: gamma=1 ladder ~02/09 07:00 EDT -> gamma=2 ~02/09 22:00 -> round 2 ~04/09
-> A/B ~05/09 late -> leak ~06/09.

⚠️ THE TRANSLATION A/B IS NOT A ONE-FLAG A/B AND CANNOT BE. phased_ga.py:3049
refuses --obs-collective-cmd/--obs-alt-err/--obs-vz without --translation, because
with no z axis those three channels are constant zeros — three wasted features and
a silently different address space. The arms therefore differ by a FOUR-FLAG
BUNDLE and the OFF arm is a 5-feature controller against the ON arm's 8. An OFF
win reads as "the plant AND/OR those three features cost attitude". Quote that
caveat with the numbers, always.
⚠️ ALTITUDE HAS NEVER BEEN IN THE OBJECTIVE. Every ladder and sweep run uses
--reward-lambda-alt 0 and alt RANK weight 0.0. "Altitude regimen" means the PLANT
integrates vertical translation and the OBSERVATION carries 3 vertical features —
never that anything optimized for altitude.

THE LEADERBOARD IS THE BAR — docs/controller_gate_distance_leaderboard.md,
regenerated by scripts/gate_distance_leaderboard.py (do NOT hand-edit). It ranks
ALL 174 markers with a headline held-out on one scale. USE IT BEFORE CALLING
ANYTHING A RECORD: on 01/09 the ladder's hd 0.4034 was called a programme best
and is 38th, because the chain's own STATE table lists only that chain's runs.
Gate-distance, the scale sweep_ladder_gamma.sh ranks on:
    hd = 0.5556*(err/8.0) + 0.4444*min(K*-log2(stable), 20.0),  K = log0.5/log0.70
    ALTITUDE ROWS SINCE 21/10/2026 (Luiz): the leaderboard adds an alt term —
    hd = 0.4445*(err/8.0) + 0.3555*min(K*-log2(stable), 20.0) + 0.20*(alt/1.0 m)
    (ALT_W/ALT_BAR_M in gate_distance_leaderboard.py; attitude-only rows keep the line
    above). Quote the LEADERBOARD's hd in PAIR tables. A chain's own log still prints
    the old two-term hd: don't mix the two. The GA fitness still has alt=0.0 (CTRL-19 open).
hd 1.0 is ON the gate (stable>=0.70 AND err<=8.0°). It is the GATE's geometry, not
a neutral summary — stable enters through a log, so points near 70% are worth far
more than the same pp down at 30%. RANK on hd; TEST the triple.

⚠️ EVERY CLASSICAL CONTROLLER STILL BEATS EVERY WNN RUN. Same 5 test seeds,
disturbance L4C, airframe cf21_brushless, from experiments/l4teach_markers/
baselines_L4C_cf21bl.json (scripts/compute_baselines.py, n=5 seeds each):
    ctrl    stable         err      steady    alt m      hd
    MPCOF   100.0%   0.70±0.01°     0.01°     0.076   0.0483
    LQI     100.0%   0.89±0.06°     0.44°     0.076   0.0621
    LQR     100.0%   1.05±0.08°     0.59°     0.076   0.0726
    MPC     100.0%   1.38±0.17°     1.04°     0.076   0.0958
    PID     100.0%   1.79±0.36°     1.03°     0.076   0.1241
All five hold 100% stable. The BEST WNN anywhere in the altitude regimen is
GWS_C10noJM_s31337005 at hd 0.1578 (97.4%/1.80/1.25); the ladder's best is b=32
n=64 at 0.2240. So the ranking is MPCOF < LQI < LQR < MPC < PID < every WNN —
the weakest classical is still ahead of the strongest weightless run, and PID is
the one to quote because it is the WEAKEST classical, not a hard bar.
⚠️ hd IS NOT AN ALTITUDE METRIC, AND NOT A STEADY ONE. It is stable and err
ONLY — two of the four reported columns. Ranking the altitude regimen on hd
ignores the column the regimen is named after. RANK on hd; TEST all four.
⚠️ TRANSLATION DOES NOT COST EVERY CONTROLLER EQUALLY, so never extrapolate one
to the others. Measured 01/09 by re-deriving the table with --translation:
    MPCOF +1%   PID +0%   LQI +10%   LQR +12%   MPC +27%
PID and MPCOF barely move; MPC loses more than a quarter of its accuracy. An
earlier note here reasoned "PID transfers exactly, so the rest probably do" —
that was wrong for three of the five, which is why the flag exists now.
⚠️ ALT DOES NOT SEPARATE THE CLASSICALS: all five sit at 0.076±0.041 m, identical
to 3dp, because the attitude controller never touches collective — the outer loop
flies it, so that number is the outer loop's, not the controller's. The WNN DOES
emit collective (--obs-collective-cmd, 4 motor channels), so its alt (best ladder
point 0.515 m) is a harder task than the classicals' 0.076 m. Do NOT read
"6.8x worse on altitude" as like-for-like.
Always compare ESTIMATOR-FED (teacher + Mahony on the same noisy IMU);
PID[oracle] is informational only. Both tables are banked:
experiments/l4teach_markers/baselines_L4C_cf21bl.json (attitude-only) and
..._translation.json (the altitude regimen, the one that matches every WNN run
since 17/08). A baselines file with no `translation` key predates 01/09 and is
attitude-only.

TOP OF THE ALTITUDE RECORD (from the leaderboard, regenerated 02/09):
  0.1129  99.8%/1.59/1.13  SL_C_b32n256  (1st) ← the levels ladder's 64 lvl/motor point
  0.1578  97.4%/1.80/1.25  GWS_C10noJM_s31337005     · 0.1717  98.0%/2.11/1.59  GWS_S16noJM_s31337005
  0.1736  97.2%/1.99/1.68  GWS_E50S50_s31337005      · 0.1921  97.0%/2.22/2.06  GWS_C10noJM_s31337002
  0.2078  94.2%/1.92/1.48  SL_C_b36n256              · 0.2240  95.4%/2.38/1.89  SL_C_b32n64
⚠️ SL_C_b32n256 IS THE FIRST WNN RUN TO BEAT A CLASSICAL — hd 0.1129 vs PID's
0.1241 (99.8%/1.59°/1.13° vs PID 100.0%/1.79°/1.03°): it wins err by 0.20° and
concedes 0.2pp stable and 0.10° steady, so it is a GATE-DISTANCE win, not a sweep.
It is n=1, inside the 90.8-98.0% seed band, and NEEDS ITS PAIRED REPLICATE before
it is a claim. MPCOF/LQI/LQR/MPC all remain ahead.
The gated weight sweep put NINE runs inside the gate five weeks before the ladder
did; the ladder is RECOVERING the historical alphabet, not passing it.

100% STABLE IS AN ATTITUDE-ONLY RESULT. 32 markers reach it, every one
attitude-only; the altitude ceiling over 86 markers is 98.0%. It is NOT the state
layer — 27 of the 32 are sn=0 — and NOT the teacher: lqi, lqr, mpc and mpcof all
reach it. The regime is the only clean separator, and no run has ever toggled only
that flag, which is exactly what chain 3 is for. State neurons and altitude have
NEVER been flown together (sn>0 & altitude = 0 markers): what a state layer would
do UNDER altitude is untested, not refuted — LOW PRIORITY (Luiz, 01/09).

THE LEVELS RESULT — output_neurons = num_motors x levels_per_motor, so n=32 on
this quad is 8 levels/motor, HALF the historical config every pre-sweep cohort
used. Doubling the alphabet is the first lever in this programme to move the
triple the same direction twice:
  b=36   8 lvl (n=32) 66.6%/5.94/6.53 hd 0.9190 · 16 lvl 87.0%/3.31/3.08 hd 0.4034
       · 24 lvl (n=96) 93.8%/2.38/1.63 hd 0.2450                      monotone, no knee
  b=32   8 lvl (n=32) 57.2%/6.45/5.99 hd 1.1440 · 16 lvl 95.4%/2.38/1.89 hd 0.2240
THE WIDTH ORDERING INVERTED. b=36 beat b=32 across the whole b-sweep; at 16
levels/motor b=32 wins by nearly as much the other way. The b=36 knee was measured
at 8 levels/motor and does NOT survive the alphabet change — as the mechanism note
predicted, steady is set by the OUTPUT alphabet, not the input lens.
WATCH: at b=36 n=96 the GRID ALONE reached 88.0±1.7%, better than the b=36 n=64
run's fully-searched 87.0% headline. If the grid keeps closing on the searched
result as the alphabet widens, the connectivity search is buying less each step.

THE GAMMA ARM IS REFUTED — 4 of 4 measured pairs, verdict 02/09/2026.
gamma=1.0 is --delta-gamma's default (identity, NO shaping); gamma=2.0 concentrates
resolution near zero at ZERO extra footprint, which is why it was worth a full arm.
Paired against its own gamma=1 control at the same shape and seed (31337002),
on the ladder's gate-distance scale:
    shape              gamma=1   gamma=2    delta      winner
    b=36 n= 32  ( 8lvl)  0.9190   1.5147   +0.5958    gamma=1
    b=32 n= 32  ( 8lvl)  1.1440   0.9976   -0.1463    gamma=2
    b=36 n= 64  (16lvl)  0.4034   0.9792   +0.5758    gamma=1
    b=32 n= 64  (16lvl)  0.2240   0.6067   +0.3828    gamma=1
    PAIRED MAJORITY: gamma=1 3 - 1 gamma=2.
THE MECHANISM IS DEAD, NOT MERELY BEHIND. The arm's whole claim was that finer
resolution near zero SUBSTITUTES for spending neurons. b=32 is the one width that
preferred gamma=2 — and it is the width that REVERSES hardest once the alphabet
doubles (-0.1463 at 8 lvl becomes +0.3828 at 16 lvl). So gamma=2's single win was
an artifact of the alphabet-starved regime, where nothing was inside the gate and
any change looked like progress. Phase 1's tiebreak picked gamma=1 for the right
answer by the wrong route; it is now the right answer for a measured reason.
⚠️ n=1 PER PAIR, inside the 90.8-98.0% seed band: this is a DIRECTION, not a
significance claim. It is enough to stop spending on the axis, NOT enough to write
"gamma shaping does not work" in a paper without replication.
⚠️ gamma < 1 has NO motivation — it coarsens exactly where the limit cycle lives.
The mechanism note argued gamma > 1; that is what lost. Do not "try the other side".
STOPPING RULE (Luiz's call, 02/09): the two n=96 pairs FINISH so the factorial is
complete — stopping a pre-registered 2x3 at 3-1 because the interim looks decided
is optional stopping, and a hole at the widest alphabet is exactly what a reviewer
asks about. ~5h of a ~90h queue. NOTHING FURTHER on gamma after that: a new gamma
experiment needs a NEW mechanism argument, not a re-run.
  b=36 n=96 must beat gamma=1's 0.2450 · b=32 n=96 must beat 0.3899.
⚠️ `_g20_` means 20 GENERATIONS in an SL_A tag and GAMMA 2.0 in an SL_C tag.

SEED SPREAD IS WIDER THAN ANY EFFECT MEASURED. One fixed recipe (GWS_S16noJM)
spans 90.8-98.0% stable across five base seeds. An n=1 point sits inside that band,
so a "new best" needs the PAIRED SAME-SEED comparator, never the leaderboard top.

DURING-SEARCH IS ANTI-PREDICTIVE. b=36 was WORST at gen 1 (43.0%) and best held out
(66.6%); b=36 gamma=2 led from the grid and finished 11.4pp of stable behind. The
HELD-OUT GRID is noise too. Fitness VALUES are NOT comparable across arms (zscore
vs desirability are different scales) — compare the held-out triple, never `best=`.
MEMORY USUALLY BUYS NOTHING — rejected by stage-select at 7 of 8 ladder widths, and
at b=36 n=32 its multiseed line is BIT-IDENTICAL to CONNECTIONS.
BUDGET CAPS IN THE HEADER LIE. `400c` is a CAP the run never approaches, and
--skip-stages can prune the stage entirely — the gated wsweep's "400c" never ran a
connections generation. Only the `STAGE n (...) done: gen G/T` lines say what ran.
TWO RUNS CARRY A CONTAMINATED WALL CLOCK: b=40 GATE and b=40 DESIR span a SIGSTOP
pause (~1h17m, 30/08). Exclude both from any cost-vs-width analysis.
BANKED (do NOT re-derive): gated wsweep COMPLETE — S16noJM won (94.1/2.35/2.01,
n=5, paired majority, NOT significance); no PID win. The b=48 budget confound is
DEAD (4x the generations returned a bit-identical headline).
PREEMPT IS HARD (memory feedback_sigterm_does_not_preempt_phased_ga): phased_ga
HANDLES SIGTERM and does not exit. Supervisors match `-m wnn.control.phased_ga`
(catching the /usr/bin/time wrapper, which re-parents to PID 1) and escalate
SIGTERM -> 60s -> SIGKILL, failing closed. wait_no_controller is a PURE WAIT and
never escalates. A supervisor silent for many minutes is usually a blocked WAIT,
not a dead one: check `ps` for it AND for what it waits on.
CRN FITNESS LANDED 03/09 21:05 EDT (commit 5c3e7e61, Luiz: option c — land now,
stop nothing). DIAGNOSIS: since the 30/05 K-fold ROTATION each generation's
offspring were scored on the NEXT pool while elites kept the score of the pool
they were born on (the 23/02 fix re-evaluates only at STAGE boundaries). One
100-episode score carries ~0.4°/~2.5pp of pool-to-pool noise for the SAME genome
— the size of the between-genome spread — so `best=` sat at (=) for whole stages
(178 stages measured: CONNECTIONS improved 30% of gens, MEMORY 23%) while the
population's HELD-OUT moved 50pp (grid winner 100%/1.55° in-search → 43% held-out).
FIX: `--score-crn` DEFAULT ON — every genome scored on ALL 5 pools every
generation (mean, `combine_pool_scores`), no rotation, SAME training seeds for
everyone. Fitness is now deterministic per genome, so cached elite scores are
honest. SEARCH ONLY: the held-out TEST evaluators are untouched (fold 0, same
protocol as every marker + the baselines file). Cost: ~10% on training stages,
~5x on the score-only MEMORY stage (15 min → ~75 min).
⚠️ CODE-ERA BOUNDARY: every run LAUNCHED after 03/09 21:05 EDT is CRN; b32 seed
31337003 (launched 16:49) and ALL seed-31337002 points are ROTATION-era. The
.out's startup line prints `fitness_pools=CRN(...)` or `rotation(...)` — read it,
never assume. Round 2's seed-3 b28/b24 are CRN; the paired b32-vs-b28 comparison
therefore mixes scorers on the SEARCH (the held-out protocol is identical, so the
triples remain comparable — it is the search quality that differs, which is the
intervention). The translation A/B is self-contained (both arms CRN). The LEAK
REVISIT compares to the BANKED SL_C_b32n64 control (rotation-era): its 0.95
control must be RE-FLOWN under CRN for a like-for-like (+1 run, ~2h) — not yet
queued. Test: tests/controller_score_crn.py. Memory: project_crn_fitness_landed.
CRN READOUTS SO FAR (seed 31337003 CRN vs seed 31337002 rotation — confounded by
SEED, not a clean A/B): b24 hd 0.1972 → 0.1271 (10th → 3rd of the archive),
b28 0.1378 → 0.1349; the CRN grid winner held out at 97.6% (b28) / 94.6% (b24)
vs 84.8% / 90.6% rotation-era. Direction is right, size unknown until one shape is
re-flown at the SAME seed under CRN.
NEXT LEVER (parked until CRN is measured): CONNECTIONS mutation is a JUMP, not a
step — rate 0.1 per tap × 32 taps ⇒ P(neuron untouched)=0.9^32=3%, every child
rewires every neuron. A/B rate 1/32 (one tap per neuron) once CRN has a paired read.
OPEN — THE BITS AXIS HAS NO REPLICATION (found 01/09, Luiz). Round 1 of the bits
sweep (34 SL_A markers) is ONE seed, 31337002, at n=32 = 8 levels/motor — an
alphabet-starved regime: every width but b=36 sits OUTSIDE the gate (hd 0.92 at
b36, 1.14 at b32, 1.51 at b34 — the b34 dip between its neighbours IS the n=1
noise). Round 2 (cull top-6 + seed 31337003) NEVER RAN; the levels ladder took
the top-2 widths informally but ALSO at one seed. The pre-registered relaunch
(scripts/ladder_relaunch_supervisor.sh) is DEAD — a second seed at 8 lvl would
replicate a ranking the levels result already inverted. RETHOUGHT round 2 =
a bits re-sweep AT THE WINNING ALPHABET: b in {24,28,32,36,40} at (n*, gamma*)
once the gamma=1 ladder names n* and the gamma=2 arm names gamma*, TWO seeds
from the start (31337002 reuses the banked b32/b36 points; 31337003 new),
widths-major so a stall leaves the whole curve at low res, cull top-3 / 1.25x
after the first seed. SCHEDULED as queue item 3 (Luiz, 01/09: option A — after
gamma=2, before the translation A/B, which now flies at round 2's winner). The b32-vs-b36 ordering is claimed ONLY if the paired
same-seed comparison agrees 2/2.
OPEN, NEVER FLOWN (0 markers each), behind the queue:
  · STAGE 2 = HORIZONTAL translation (--xy-offset, --obs-pos-err-xy, --obs-vel-xy,
    --fit-weight-pos RADIAL). A 4-flag bundle (phased_ga.py:3070 refuses the
    features without --translation AND --xy-offset>0). PREREQUISITE: audit the
    trainer — stage 1 hard-asserted on DAgger lacking a translation/collective
    teacher (memory project_stage1_trainer_gap); whether the teacher can command
    LATERAL is unverified.
  · WINDOW k-LADDER (stage C of sweep_ladder_chain.sh:42-45): k=1 min1 control,
    k>=2 framed1, gated on mean headline steady, 2 seeds. --input-window-k already
    DEFAULTS to 4 and no ladder run passes it — the pool is 4 frames everywhere;
    what is unswept is the SAMPLING policy (spread vs framed1). k=1 -> k>=2 moves
    pool AND policy together: pre-register it as a bundle.
  · STAGE D pipeline A/B: grid->GA-NEURONS->MEMORY vs grid->GA-CONNECTIVITY->
    MEMORY at (b*, n*, k*), 4 runs. Distinct from the translation A/B.
  · stages B-D under the winning aggregation; re-score 9 alt arms; rerun banked
    sweeps; make --fit-aggregation REQUIRED.
  · RACING / SUCCESSIVE HALVING (Luiz 03/09: add): train once, score every
    offspring on a few episodes, keep the top third, spend the full 100 only on
    contenders. Attacks the 2,400 s/gen directly — 50 full scorings per gen is
    the cost, not the optimiser.
  · OFFLINE CONNECTIVITY (Luiz 03/09: add): the trainer is supervised (DAgger vs
    a teacher), so "does this 32-tap tuple separate the teacher's actions?" is
    measurable from the collected dataset in seconds, no rollout — feature
    selection instead of black-box search. Bigger design; discuss first.
  · LEAVE-ONE-NEURON-OUT targeted mutation — NOT adopted (Luiz 03/09): it
    re-rolls only the weak neurons and never explores the neighbourhood of the
    good ones; pure exploitation of the weak slots. Would need a paired
    exploration term before it is an arm.
LOW PRIORITY (Luiz, 01/09): sn>0 x altitude (sn>0 IS well flown, attitude-only;
only the CONJUNCTION is unflown — chain 3 tests the stronger regime hypothesis);
UNSW/CICIDS MULTICLASS -> a later paper, results not worth chasing now.
Installed: ram_accelerator 12 / ram_controller 26, facades 12 / 26 — all four
agree, nothing staged.

IDS: worker PID 4994, PPID=1, ABI-12 wheel, rayon 13. ~2603 completed / 1 running /
~130 queued / 0 failed / 409 cancelled. Worker is FIFO min-id.
⚠️ THE DASHBOARD BINARY: launch ONLY from $CARGO_TARGET_DIR
(/Volumes/20260401-WDBlack-SN850X-2TB/cargo-target/release/wnn-dashboard, cwd
dashboard/). dashboard/target/release/ froze on 03/07 and predates the stale-reaper
fix; launching it requeued LIVE flows and cost 5899. The stale copy is renamed
.STALE-jul03-do-not-run. Memory: reference_dashboard_launch_cargo_target.
DIAGNOSTIC TELL for that bug: `running=0` while the worker log writes MARKER_TRAIN
lines AND a `queued` flow has a seconds-old heartbeat — BOTH halves, since
GET /api/flows returns only a PAGE and can show running=0 innocently. Check:
`select id from flows where status='queued' and last_heartbeat > datetime('now','-3 minutes');`
— MUST be empty.
THE ADDRESS FIX (29/08, memory project_bits_above_64_or_fold): bits > 64 used to
OR-FOLD connection slots i and i+64 onto one address bit; ram_core now names wide
tuples by a splitmix64 hash, <= 64 bits is IDENTITY. All reruns are queued.
BANKED: general AC/CE claim DEAD (6/18 pairs); CE20 beats production +0.951pp on
unswt-16b ONLY; unswr-quad SATURATED; cicids cell COMPLETE and NULL
(docs/ids_results.md §12). IDSZ COMPLETE · SP100 DEAD control · multiclass
baselines COMPLETE (UNSW temporal bar 0.52).
ESCALATE if the worker process count is 0 while flows are queued, or if any flow
moves to `failed`.

BLOCKED BY LUIZ, do not start: Vivado/EC2 work; buying hardware; the FPGA
flow_2747_best_fpr run.

ONLY add lines beyond the six if:
(a) a NEW controller marker landed — quote every stage's held-out block (stable%/err°/steady°, plus alt where the run prints it), name the arm's alt RANK weight + λ_alt + seed, mark the headline stage, and when both arms of a seed exist print the PAIR table.
(b) an escalation — chain dead before its markers complete, rc!=0, >1 controller running, avail below 4 GiB, a run past 5 h, any "weight_alt > 0 but ... is None", IDS worker down while flows are queued, or any IDSX/MCS flow failed.
(c) the box went IDLE — say so and name the unstarted pending items.
Otherwise stop after the six lines.
