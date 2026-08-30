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

**Currently armed:** job `65ce1643`, schedule `13,43 * * * *` (off the :00/:30 marks on purpose).
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

STATE (30/08/2026 13:1x UTC — refresh this block when the programme changes).

CONTROLLER (the lever): ROUND 1 OF THE STAGE-A A/B IS COMPLETE (26/26 markers, seed 31337002)
and the box has handed over to the WIDE PROBE. The ladder chain (sweep_ladder_ab_chain.sh) is
STOPPED — do not expect it in `ps`; the handoff relaunches it after the probe.
LIVE: scripts/probe_handoff_supervisor.sh (relaunched 30/08 13:02:55Z, PPID=1, log
/private/tmp/probe_handoff.log). It smokes b=64 (/private/tmp/b64_smoke.out) and, on rc=0 ONLY,
launches scripts/sweep_ladder_probe_wide.sh — b=40/48/64 x {gate,desir}, seed 31337002, 6
markers, log /private/tmp/sweep_ladder_probe_wide.log. When those 6 land it relaunches the
ladder for the cull + seed 31337003. Fails closed at every gate. b=64 is the last honest width
(u64 keys). Markers: experiments/sweepladder_markers (git-tracked); the DIR holds 27 for 26
chain markers because b10's legacy marker predates this chain.
ROUND-1 RESULT — b=36 WINS, b=32 IS NOT THE KNEE. The chain's own cull wrote
SURVIVORS = [36 32 34 24 28 30] (of [12..36]); b=36 ranks FIRST, so the gate-distance curve was
STILL FALLING at the top of the ladder and b=34 (hd 1.507) was a one-width dip, not the descent
breaking. The banked pre-b36 curve (b32 1.144 · b34 1.507 · b24 1.518 · b28 1.578 · b30 1.695 ·
b22 1.701 · b20 1.760 · b26 1.861 · b18 2.261 · b12 2.338 · b14 2.546 · b16 3.193 · b10 8.609)
is superseded at the top: b36 sits below b32. Nothing has got INSIDE the gate (hd < 1). Later
seeds fly the six survivors only, both arms.
b=36 PAIR: GATE 66.6+/-6.0%/5.94+/-0.67deg/6.53+/-0.60deg alt 1.055m (headline CONNECTIONS#0)
vs DESIR 37.4%/10.29deg/11.55deg alt 1.819m (CONNECTIONS#1) — GATE wins all four columns.
b=34 PAIR: GATE 47.2%/8.23/8.68 alt 0.647m vs DESIR 44.2%/10.50/11.83 alt 1.212m — GATE again.
b=32 PAIR: GATE 57.2%/6.45/5.99 alt 0.982m vs DESIR 45.6%/10.46/10.27 alt 2.789m — GATE again.
So GATE (zscore + gate 0.70/8.0) beats DESIR (--fit-aggregation desirability, no gate flags) on
ALL FOUR COLUMNS in 3/3 pairs, widening with width. Two recurring patterns: the MEMORY stage
buys nothing (bit-identical to CONNECTIONS at b32-desir, b34-both, b36-gate; at b36-desir it
went BACKWARDS 40.0% -> 36.0%), and DESIR drifts further in altitude with alt weight 0 and
lambda_alt 0 on both arms. Fitness VALUES are not comparable across arms (zscore vs
desirability are different scales) — compare the held-out triple, never `best=`.
BANKED (do NOT re-derive): gated wsweep COMPLETE — S16noJM won (94.1/2.35/2.01, n=5, paired
majority, NOT significance); no PID win. 0/686 samples feasible at 32n on the OLD ladder ->
the gated arm's weights never applied in-search.
PREEMPT IS NOW HARD (30/08, memory feedback_sigterm_does_not_preempt_phased_ga): phased_ga
HANDLES SIGTERM and does not exit, so the old handoff's plain `kill` left a condemned seed-2 run
flying at 800% CPU and blocked the probe for 30 min. The supervisor now matches
`-m wnn.control.phased_ga` (catching the /usr/bin/time wrapper, which re-parents to PID 1) and
escalates SIGTERM -> 60s grace -> SIGKILL, failing closed. wait_no_controller stays a PURE WAIT
and never escalates — it is also used after the smoke and after the probe, where the in-flight
run is legitimate. A supervisor silent for many minutes is usually a blocked WAIT, not a dead
supervisor: check `ps` for it AND for what it is waiting on before concluding anything died.
OPEN (behind the probe): stages B-D under the winning aggregation; re-score 9 alt arms; rerun
banked sweeps; make --fit-aggregation REQUIRED. Controller wheel ABI 26 INSTALLED 30/08 ~02:55Z
with both Python patches; ALL staged patches applied and .claude/plans/staged/ DELETED — nothing
left to install. Installed: ram_accelerator 12 / ram_controller 26, facades 12 / 26 — all agree.

IDS: worker UP on the ABI-12 wheel (PID 82705, PPID=1, rayon 13) — swapped 30/08 02:27Z.
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
