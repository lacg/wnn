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

**Currently armed:** job `89707e2c`, schedule `13,43 * * * *` (off the :00/:30 marks on purpose).
STATE block refreshed 30/08/2026 ~03:0x UTC: the controller wheel is now INSTALLED
(it was the last "built not installed" item) and the ciciot granularity reruns are
QUEUED, so both of those lines had already gone stale within the hour.

To re-arm after a CLI restart, pass everything below the line to CronCreate with
`cron: "13,43 * * * *"`, `recurring: true`.

---

Controller status tick. Check directly with Bash (do NOT spawn a subagent — a handful of files).

⚠️ SAY "RUN", NEVER "CELL". "cell" means a RAM memory cell in this project.

DISCOVER the live lever rather than assuming one — a previous cron went stale by hardcoding a finished arm. Do NOT hardcode chain names:

  TZ=America/New_York date "+%d/%m/%Y %H:%M:%S %Z"
  cd /Users/lacg/wnn
  pgrep -f "MacOS/Python -u -m wnn.control.phased_ga" | wc -l   # LOGICAL runs — count the CHILD ONLY.
  # Do NOT use the broad "-m wnn.control.phased_ga" here: it also matches the /usr/bin/time wrapper,
  # so ONE healthy run reports 2 and trips the ">1 controller running" escalation every tick. The broad
  # pattern belongs in the supervisors' kill/wait (the wrapper must not be invisible to a kill), not here.
  ps -axo pid,command | grep -E "scripts/.*(chain|driver|study|probe|handoff|supervisor|wide)" | grep -v grep
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

MISSING VALUES — never invent one, never print 0.000 for "not measured" (a zero altitude reads as perfect altitude hold). Use an em dash.
· No `| gen:` block → `gen: — (pre-b872ba57 run)`.
· Before the first GA gen line, read fit/stable/err/steady/alt off the GRID WINNER line and tag `elite:` `(grid winner, during-search)`; that line prints the fitness FUNCTION but no VALUE → `fit —`, and `gen: —`.
If NO chain and NO controller are running, say so plainly on lines 2-3 and name what is pending.

STATE (01/09/2026 23:0x UTC — refresh this block when the programme changes).

THE QUEUE — three chains, ONE controller at a time, ~63h to drain (~04/09). Every
gate below waits on MARKERS, never on a process, and every wait is PURE: nothing
here can preempt a live run. A marker is a CLAIM THE RUN FINISHED, withheld on a
watchdog kill (rc 143/137), on a crash, and on a clean exit with no MEMORY triple
— so "chain gone, markers missing" means a run needs a human, and each gate FAILS
CLOSED there, leaving the box idle to be inspected rather than stacking work on a
crash.
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
  3. PID 90455 scripts/translation_ab_chain.sh — the TRANSLATION A/B, 2 arms x 5
     seeds at b=32 n=64, seed-major. Gated on the four gamma=2 markers, then
     PREFLIGHTS the OFF flag set against phased_ga's guards on the idle box
     before committing 37h. Log /private/tmp/translation_ab.log

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
    MPCOF  100.0% / 0.69±0.01° / 0.00±0.00°   hd 0.0481
    LQI    100.0% / 0.81±0.03° / 0.36±0.03°   hd 0.0564
    LQR    100.0% / 0.93±0.04° / 0.42±0.07°   hd 0.0647
    MPC    100.0% / 1.09±0.08° / 0.65±0.12°   hd 0.0756
    PID    100.0% / 1.78±0.34° / 1.03±0.36°   hd 0.1240   (per seed err 1.24 1.73 2.06 2.25 1.64)
All five hold 100% stable. The BEST WNN anywhere in the altitude regimen is
GWS_C10noJM_s31337005 at hd 0.1578 (97.4%/1.80/1.25); the ladder's best is b=32
n=64 at 0.2240. So the ranking is MPCOF < LQI < LQR < MPC < PID < every WNN —
the weakest classical is still ahead of the strongest weightless run, and PID is
the one to quote because it is the WEAKEST classical, not a hard bar.
⚠️ ONLY PID IS MEASURED UNDER TRANSLATION. compute_baselines.py has no
--translation flag, so that table is the ATTITUDE-ONLY plant. PID transfers
exactly — the banked 1.7848/1.0342 matches the in-run PID[est] row (1.78/1.03) to
2dp on the same seeds — which is evidence the other four transfer too, but it is
not a measurement of them. Nothing measures LQR/MPC/LQI/MPCOF ALTITUDE HOLD at
all; only PID prints an alt (0.157m). Do not quote the other four as altitude-
regimen numbers without saying so.
Always compare ESTIMATOR-FED (teacher + Mahony on the same noisy IMU);
PID[oracle] is informational only.

TOP OF THE ALTITUDE RECORD (from the leaderboard, 86 markers):
  0.1578  97.4%/1.80/1.25  GWS_C10noJM_s31337005     · 0.1717  98.0%/2.11/1.59  GWS_S16noJM_s31337005
  0.1736  97.2%/1.99/1.68  GWS_E50S50_s31337005      · 0.1921  97.0%/2.22/2.06  GWS_C10noJM_s31337002
  0.2240  95.4%/2.38/1.89  SL_C_b32n64  (10th)       · 0.2450  93.8%/2.38/1.63  SL_C_b36n96  (17th)
The gated weight sweep put NINE runs inside the gate five weeks before the ladder
did; the ladder is RECOVERING the historical alphabet, not passing it.

100% STABLE IS AN ATTITUDE-ONLY RESULT. 32 markers reach it, every one
attitude-only; the altitude ceiling over 86 markers is 98.0%. It is NOT the state
layer — 27 of the 32 are sn=0 — and NOT the teacher: lqi, lqr, mpc and mpcof all
reach it. The regime is the only clean separator, and no run has ever toggled only
that flag, which is exactly what chain 3 is for. State neurons and altitude have
NEVER been flown together (sn>0 & altitude = 0 markers): what a state layer would
do UNDER altitude is untested, not refuted.

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

THE GAMMA GATE CHOSE OFF (gamma=1.0 is --delta-gamma's default = identity = NO
shaping; gamma=2.0 is ON). It went 1-1 and was settled only by the MAGNITUDE of the
summed delta, which b=36's large loss dominated — b=32 genuinely PREFERRED gamma=2
(hd 0.9976 vs 1.1440). Read it as the tiebreak rule firing on n=1 per width, not as
a refutation; chain 2 re-tests it.
  b=36  gamma=2 hd 1.5147 vs gamma=1 0.9190  -> worse
  b=32  gamma=2 hd 0.9976 vs gamma=1 1.1440  -> BETTER
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
OPEN (behind the queue): stages B-D under the winning aggregation; re-score 9 alt
arms; rerun banked sweeps; make --fit-aggregation REQUIRED. The ladder relaunch
(cull + seed 31337003) still awaits Luiz.
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
