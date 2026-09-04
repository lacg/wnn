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

**Currently armed:** job `5ad9d655`, schedule `13,43 * * * *` (off the :00/:30 marks on purpose).
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

QUEUE AS OF 04/09/2026 08:55 EDT (supersedes the FOUR-chain block below, kept for
provenance). POWER OUTAGE ~07:14 EDT 04/09 killed every process; everything was
relaunched 08:50 EDT with PPID=1 (dashboard from CARGO_TARGET_DIR → worker rayon 13
→ mem sampler + watchdog → translation_ab_chain.sh + leak_revisit_chain.sh; cron
re-armed as da2eb7c7). /private/tmp logs were wiped by the reboot, so chain logs
start at 12:50Z. BITS ROUND 2 IS DONE (sentinel BITS_ROUND2_DONE.json, 11:03Z,
eleven minutes before the outage): winner b=32 n=256 γ=1, mean hd 0.1191; the
seed-3 curve is FLAT (0.125-0.135 across b24-b32) and b32's seed-3 point is the
only rotation-era one, so "b32 interior optimum" rests on seed 2 — say so.
  1. scripts/translation_ab_chain.sh — RUNNING run 1/10, TAB_on_b32n256_..._s31337002
     (CRN; its pre-outage attempt died at 07:14 with no marker and was re-flown
     from scratch). 2 arms × 5 seeds at b32 n256 γ1, seed-major, ~5 h each
     → ~06/09 15:00 EDT. Log /private/tmp/translation_ab.log ·
     markers experiments/translationab_markers/ (0/10).
  1b. scripts/crn_refly_chain.sh (QUEUED 04/09 09:05 EDT, Luiz) — waits on the 10
     TAB markers, then flies b24 AND b32 (n256 γ1, seed 31337002) under CRN in ONE
     ladder instance, sequentially (SL_TAG_SUFFIX=_crn → markers
     SL_C_b{24,32}n256_..._s31337002_crn.json), ~5 h each. CRN IS THE FIX, NOT AN
     ARM (Luiz: "we are not going back") — these are the paired MEASUREMENT of
     what it changed: same shape, same seed, only the scorer differs; controls =
     the rotation-era markers (b24 hd 0.1972, b32 hd 0.1129). Verdict = the
     leaderboard rows. Log /private/tmp/crn_refly.log. Lever name: "CRN re-fly".
  2. scripts/leak_revisit_chain.sh — waits on the 10 TAB markers AND both re-fly
     markers (re-gated 04/09 so the waiters cannot race), then delta_leak
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
