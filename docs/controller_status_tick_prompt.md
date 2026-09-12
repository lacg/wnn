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

**Currently armed:** job `da2eb7c7`, schedule `13,43 * * * *` (off the :00/:30 marks on purpose).
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

NEW-MARKER FORMAT (05/09/2026, Luiz: "the new marker is difficult to find").
A banked marker is the POINT of the programme — it must not read as a run-on
sentence appended to a tick. When a marker lands, print the six lines, then a
BLANK LINE, then:

  >>> NEW MARKER n/T — <tag>   (banked HH:MM EDT, rc R, DDDDD s)

followed by TWO fenced tables, never prose:
  1. PER-STAGE held-out, one row per stage (GRID / CONNECTIONS / MEMORY /
     HEADLINE), columns: stable % · err ° · steady ° · alt m, each as mean±SD
     over the report seeds (HEADLINE is a single draw, no SD). Name the
     stage-select genome next to HEADLINE.
  2. The PAIR / REPLICATION table: one row per comparable run (this marker, its
     paired arm or same-seed comparator, the relevant leaderboard rows), columns:
     stable % · err ° · steady ° · hd · rank, with a one-clause note per row
     saying what it is (rotation-era, CRN-era same seed, other arm, ...).
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

QUEUE AS OF 12/09/2026 12:50 EDT (RERUN DECLINED — option (a); POST-D0 QUEUE RELAUNCHED;
supersedes the 10:15 block).
`_fix` banked 12:26 EDT (5d423ef2): HEADLINE 97.6%/1.75°/1.33°/0.335m; one-flag pair vs void
`_hd` s2 (same recipe, bug only): −0.6pp/−0.32°/−0.19°/−0.351m — the fixed wheel is BETTER. Luiz (12:4x EDT): the fix is good but the shift is NOT worth 150 h —
NO rerun; void rows STAND with a caveat; revisit (b) step-2-only or (c) full if later data
says so. scripts/stale_altitude_refly_chain.sh stays on disk, go-gated (SAR_GO=1), unused.
STRADDLE CAVEAT (HEADLINE ONLY — Luiz's rule): every new run is ABI 28 (fixed) while its
banked control is void (buggy), so "arm minus control" = arm effect + fix effect. The fix
effect on the one clean pair (`_fix` − `_hd`, headline) is err −0.32°, alt −0.35 m, stable
−0.6 pp — the FIXED wheel is BETTER. So a new arm that beats a void control by <=0.3° err
or <=0.35 m alt may be showing the wheel, not the arm; a bigger win is real. The 2x2 reads
an interaction (difference of differences) where a constant wheel offset cancels; arm B vs
`_hd` is the exposed one. Say this only when a pair straddles the wheel. If it bites,
re-flying the 3–4 controls (option b, ~14 h) is the fix. Old rows stand as the record.
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
hd 1.0 is ON the gate (stable>=0.70 AND err<=8.0°). It is the GATE's geometry, not
a neutral summary — stable enters through a log, so points near 70% are worth far
more than the same pp down at 30%. RANK on hd; REPORT the triple.

⚠️ EVERY CLASSICAL CONTROLLER STILL BEATS EVERY WNN RUN. Same 5 report seeds,
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
ignores the column the regimen is named after. RANK on hd; REPORT all four.
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
honest. SEARCH ONLY: the held-out REPORT evaluators are untouched (fold 0, same
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
