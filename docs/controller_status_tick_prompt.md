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
  <STAGE> gen G/TOTAL  <X>s/gen  elapsed <T>
  elite: fit <best> · stable <S>% · err <E>° · steady <D>° · alt <A>m (best so far, during-search)
  gen:   stable <S>% · err <E>° · steady <D>° · alt <A>m (this gen's leader, during-search)
  box: <k> controller · watchdog <n> kills · avail <G> GiB · IDS <done>/<run>/<queued>

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

STATE (31/08/2026 18:4x UTC — refresh this block when the programme changes).

⚠️ POWER OUTAGE 31/08 ~18:28 UTC. The box rebooted; every PPID=1 process died mid-flight and was
restarted by hand at 18:36-18:37Z: dashboard (4626), IDS worker (4994, ABI 12, rayon 13), mem
sampler (5458), mem watchdog (5459, log RESET so its kill count restarts at 0), and the b48
long-budget supervisor (5618). The vite dev server (:5173) was NOT restarted — nothing needs it.

CONTROLLER (the lever): the LONG-BUDGET b=48 RUN. The WIDE PROBE IS COMPLETE (6/6 markers, all
banked and committed) and scripts/sweep_ladder_probe_wide.sh is DONE — do not expect it in `ps`.
LIVE: scripts/b48_longbudget_supervisor.sh (PID 5618, PPID=1, log /private/tmp/b48_longbudget.log)
-> SL_A_b48n32_cf21_brushless_L4C_g20_s31337002, relaunched 18:37:22Z after the outage killed its
first attempt 49 min into GRID (no marker was written, so the restart is clean and total).
THE ONE THING THIS RUN TESTS: conns-gens 20 / conns-patience 5 against the probe's 5/3. Every
other flag is byte-identical to the probe. If b=48 improves on its probe result with more budget,
the b=36 knee is a BUDGET artifact; if it does not, the knee is real. 20 gens is a CAP — magnitude-
aware patience may stop it near gen 10-12 (~6h) and that null IS the answer.
⚠️ scripts/probe_handoff_supervisor.sh IS DEAD AND MUST NOT BE RESTARTED AS-IS. Its step 1 is a
SIGTERM->SIGKILL preempt of any live controller and its step 2 re-runs the b=64 smoke, so
relaunching the whole script would kill the b48 run and then start a second controller. All it
still owes is its last three lines: wait_no_controller, then launch scripts/sweep_ladder_ab_chain.sh
for the cull + seed 31337003. That relaunch is PENDING A DECISION FROM LUIZ — until he rules, the
ladder does NOT restart on its own when b=48 finishes, and the box will go IDLE. Say so when it does.
THE PROBE RESULT — b=36 is the knee on this seed, and b=64 collapses. Gate-distance (lower =
closer to flying; hd 1.0 = ON the gate; nothing has ever got inside):
  b36 0.919 · b32 1.144 · b34 1.507 · b24 1.518 · b28 1.578 · b30 1.695 · b22 1.701 · b20 1.760 ·
  b26 1.861 · b18 2.261 · b12 2.338 · b14 2.546 · b16 3.193 · b40 1.628 · b48 2.260 · b64 11.325
Three widths above b=36 decline monotonically and the fourth collapses; b=34's earlier dip-and-
recover was ONE width, b40->b48->b64 never recovers.
PAIRS (GATE vs DESIR, headline held-out, alt RANK weight 0 and lambda_alt 0 on both arms):
  b=64   0.2%/51.57/71.64 alt 1.832m  vs   3.8%/48.18/64.30 alt 1.971m   DESIR 3-1
  b=48  32.8%/12.54/14.94 alt 0.760m  vs  24.6%/19.40/21.14 alt 1.568m   GATE 4-0
  b=40  48.0%/10.27/10.51 alt 1.835m  vs  32.6%/11.87/13.86 alt 1.037m   GATE 3-1
  b=36  66.6%/ 5.94/ 6.53 alt 1.055m  vs  37.4%/10.29/11.55 alt 1.819m   GATE 4-0
  b=34  47.2%/ 8.23/ 8.68 alt 0.647m  vs  44.2%/10.50/11.83 alt 1.212m   GATE 4-0
  b=32  57.2%/ 6.45/ 5.99 alt 0.982m  vs  45.6%/10.46/10.27 alt 2.789m   GATE 4-0
GATE wins 5/6 widths. b=64 is DESIR's ONLY column win anywhere in the probe and it is a win
between two runs that are both far outside the gate (3.8% stable is not flight) — exactly the
regime where the gated arm ranks on the incommensurable violation function, so read it as noise
in a dead width, not as a reversal.
MEMORY BUYS NOTHING — bit-identical to CONNECTIONS at most widths; at b36-desir it went BACKWARDS
40.0->36.0%; at b48 and b64 stage-select REJECTED it outright. Any during-search memory gain is
provisional.
DURING-SEARCH IS ANTI-PREDICTIVE: b=36 was WORST of b32/34/36/40 at gen 1 (43.0%) and finished
BEST held out (66.6%); b=40 was near-best at gen 1 and near-worst held out. Never read a gen line
as a forecast. The HELD-OUT GRID is noise too (b32 11.8% · b34 0.6% · b36 4.4% · b40 5.2%).
Fitness VALUES are NOT comparable across arms (zscore vs desirability are different scales) —
compare the held-out triple, never `best=`.
GRID IS NOT IDENTICAL ACROSS ARMS. The grid has ONE shape but seeds 50 genomes (seed_pop=50) and
each arm ranks those 50 by its OWN function, so the arms can diverge from stage 0. Agreement,
where it happens, is a result and not a tautology.
TWO RUNS CARRY A CONTAMINATED WALL CLOCK: b=40 GATE and b=40 DESIR span a deliberate SIGSTOP pause
(~1h17m, 10:18-11:35 EDT 30/08). phased_ga times by wall clock and cannot see the freeze. Exclude
both from any cost-vs-width analysis; every other width is clean.
BANKED (do NOT re-derive): gated wsweep COMPLETE — S16noJM won (94.1/2.35/2.01, n=5, paired
majority, NOT significance); no PID win.
PREEMPT IS HARD (memory feedback_sigterm_does_not_preempt_phased_ga): phased_ga HANDLES SIGTERM
and does not exit. Supervisors match `-m wnn.control.phased_ga` (catching the /usr/bin/time
wrapper, which re-parents to PID 1) and escalate SIGTERM -> 60s -> SIGKILL, failing closed.
wait_no_controller is a PURE WAIT and never escalates. A supervisor silent for many minutes is
usually a blocked WAIT, not a dead one: check `ps` for it AND for what it waits on.
OPEN (behind this run): the ladder relaunch (cull + seed 31337003) awaiting Luiz; stages B-D under
the winning aggregation; re-score 9 alt arms; rerun banked sweeps; make --fit-aggregation REQUIRED.
Installed: ram_accelerator 12 / ram_controller 26, facades 12 / 26 — all four agree, nothing staged.

IDS: worker RESTARTED after the outage — PID 4994, PPID=1, ABI-12 wheel, rayon 13 (18:37:04Z).
The dashboard (PID 4626) re-queued the one stale `running` flow, 5897, on startup: >180s without a
heartbeat is requeue-for-resume, never fail (worker._recover_stale_flows is the same-semantics
fallback). NOTHING was lost to the outage on the IDS side — no flow failed, count stays 0.
THE ADDRESS FIX (29/08, memory project_bits_above_64_or_fold): bits > 64 used to OR-FOLD
connection slots i and i+64 onto one address bit, so a "96-bit" neuron was a 64-bit neuron
with 32 input pairs merged. ram_core now names wide tuples by a splitmix64 hash; <= 64 bits is
IDENTITY (bit-exact, nothing to re-run). Blast radius is decided by the WINNER's width, not
the config cap.
LIVE COHORT = IDSXD (AC/CE matched pairs, desirability clones). THE ABI-12 RELEASE CHAIN IS
DONE: scripts/worker_abi12_handoff.sh completed 30/08 17:43:02Z and EXITED — do not expect it in
`ps`, its absence is success, not failure. Smoke flows 5894/5895 both `completed` on the fixed
wheel, the 14 paused ciciot-96b flows (5896-5909) were released to `queued`, and reruns
6050/6051 were restarted. Nothing is blocked on the address fix any more.
ORDERING NOTE — a flow with a HIGHER id may run while lower ids sit queued; this is correct and
self-correcting, not a bug. `admit()` (src/wnn/ram/experiments/scheduler.py) picks
`min(id)` among flows that are QUEUED AT THAT INSTANT, and it never preempts a running flow. On
30/08 the worker admitted 5911 at 17:42:05Z, forty seconds BEFORE the handoff moved 5896-5909
from `paused` to `queued` at 17:42:45-50Z — so 5911 genuinely was the lowest queued id when it
was chosen. The next admission takes min(id)=5896 and FIFO resumes. Do NOT "fix" this by
stopping the running flow.
QUEUED (worker is FIFO min-id): 5910-5967 unswr/others (<=64 caps, unaffected) · 5894/5895
(96b, restarted FROM BEGINNING — their pre-fix checkpoints were folded) · 6050/6051 IDSXD
-w64fix reruns · 6052-6071 the SP abl2big -w64fix reruns (10 cicids + 5 unswr + 5 unswt,
queued 30/08 on Luiz's call: that arm's whole question is the 250n x 100b cap and every
affected winner never tested a wide neuron).
· 6072-6089 the SP-ciciot granularity reruns (ablpln 9, ablqsr 4, abl3s 1, bin 4), queued
30/08 on Luiz's call via scripts/queue_w64fix_reruns.py. Selected by the WINNER's widest
bits_per_neuron, not the config cap — all 40 flows in those arms were configured max_bits=100
but only 18 winners actually went above 64. Emitted ROUND-ROBIN across the four arms, so
stopping the queue anywhere leaves every arm with roughly equal n.
· 6090-6116 the XDS cross-dataset OI reruns (27), queued 30/08 on Luiz's call, interleaved
across the four DATASETS: ciciot-subsample 15, unsw-temporal 7, unsw-random 4,
cicids-random 1. XDS is 27 and not the 29 previously recorded — that count included two
CANCELLED flows (XDS-ciciot-46M-96b-Wc r63432/r15385), which banked no result and so have
nothing to invalidate.
NOTHING IS LEFT UNQUEUED: all three rerun sets (6052-6071, 6072-6089, 6090-6116) are in.
FPGA claims are untouched throughout (every Vivado-synthesised design is <= 64 bits).
BANKED: general AC/CE claim DEAD (6/18 pairs); CE20 beats production +0.951pp on unswt-16b
ONLY; unswr-quad SATURATED; cicids cell COMPLETE and NULL (docs/ids_results.md §12).
IDSZ COMPLETE · SP100 DEAD control · multiclass baselines COMPLETE (UNSW temporal bar 0.52).
ESCALATE if the worker process count is 0 while flows are queued, or if any IDSXD/SP flow moves
to `failed`.

ONLY add lines beyond the six if:
(a) a NEW controller marker landed — quote every stage's held-out block (stable%/err°/steady°, plus alt where the run prints it), name the arm's alt RANK weight + λ_alt + seed, mark the headline stage, and when both arms of a seed exist print the PAIR table.
(b) an escalation — chain dead before its markers complete, rc!=0, >1 controller running, avail below 4 GiB, a run past 5 h, any "weight_alt > 0 but ... is None", IDS worker down while flows are queued, or any IDSX/MCS flow failed.
(c) the box went IDLE — say so and name the unstarted pending items.
Otherwise stop after the six lines.
