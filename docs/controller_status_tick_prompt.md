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

**Currently armed:** schedule `13,43 * * * *` (off the :00/:30 marks on purpose).
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
  pgrep -f "MacOS/Python -u -m wnn.control.phased_ga" | wc -l
  ps -axo pid,command | grep -E "scripts/.*(chain|driver|study)\.sh" | grep -v grep
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

STATE (30/08/2026 02:4x UTC — refresh this block when the programme changes).

CONTROLLER (the lever): the STAGE-A DESIRABILITY A/B — scripts/sweep_ladder_ab_chain.sh
(PID 73478, PPID=1), log /private/tmp/sweep_ladder_ab.log, markers
experiments/sweepladder_markers (git-tracked), outdir logs/controller/sweep_ladder.
Idempotent per marker: `marker exists — skip`, so the chain can be killed at a run
boundary and relaunched to resume exactly where it stopped.
TWO ARMS per (width, seed), S16noJM weights on BOTH; only the aggregation differs:
  arm GATE  = zscore + gate 0.70/8.0 (shipped regime; tag SL_A_b{b}n32_..._s{seed})
  arm DESIR = --fit-aggregation desirability, NO gate flags (tag ..._desir_s{seed})
Widths [12..36 step 2] (b=10 TRIMMED, measured dead), seeds 31337002/31337003, WIDTH-MAJOR
with arm pairs adjacent. Round 1 = 26 runs (13 widths x 2 arms, seed 31337002); the marker
DIR holds 27 because b10's legacy gate marker predates this chain — count the CHAIN's own
progress, not `ls | wc -l`. Cull ranks on GATE-DISTANCE (desirability half-lives over the
gate pair, err .5556 @ 8 deg / stable .4444 @ 0.70), min across arms, top-6 or within 1.25x;
then seed 31337003 flies survivors only, both arms (up to 12 more runs).
GATE-DISTANCE CURVE, seed 31337002, headline held-out, min across arms (hd 1.0 = ON the
gate; nothing has got inside): b32 1.144 (BEST) · b34 1.507 · b24 1.518 · b28 1.578 · b30
1.695 · b22 1.701 · b20 1.760 · b26 1.861 · b18 2.261 · b12 2.338 · b14 2.546 · b16 3.193 ·
b10 8.609. b34 is the FIRST width to break the descent, so b36 decides whether b32 is the
knee. b12-b18 gate rows are INHERITED from the earlier chain (no `arm` field) — same config,
not this chain's work.
b=32 PAIR (both arms in): GATE 57.2%/6.45/5.99 alt 0.982m (headline CONNECTIONS#1, hd 1.144)
vs DESIR 45.6%/10.46/10.27 alt 2.789m (CONNECTIONS#0, hd 1.705) — GATE wins all four columns;
DESIR's MEMORY stage was bit-identical to its CONNECTIONS (bought nothing) and its altitude
drift is ~3x GATE's with alt weight 0 on both.
ARMED BEHIND IT (both PPID=1): scripts/probe_handoff_supervisor.sh (PID 56929, log
/private/tmp/probe_handoff.log) waits for the b=36 desir marker, then STOPS the chain before
it can start the seed-2 round, SMOKES b=64 on a tiny budget, and only on rc=0 launches
scripts/sweep_ladder_probe_wide.sh (b=40/48/64, both arms, seed 31337002 — the WIDE PROBE
Luiz approved 29/08: does the curve keep falling past b=36). It relaunches the ladder (cull +
seed 2) when the probe's 6 markers land. Fails closed at every gate. b=64 is the last honest
width (u64 keys).
BANKED (do NOT re-derive): gated wsweep COMPLETE — S16noJM won (94.1/2.35/2.01, n=5, paired
majority, NOT significance); no PID win. 0/686 samples feasible at 32n on the OLD ladder ->
the gated arm's weights never applied in-search.
OPEN (behind the A/B): stages B-D under the winning aggregation; re-score 9 alt arms; rerun
banked sweeps; make --fit-aggregation REQUIRED. Controller wheel ABI 26 is INSTALLED as of
30/08 ~02:55Z, together with BOTH staged Python patches (_accel.py EXPECTED_ABI 26 + the
widths passed to the 4 remap call sites) — all three staged/*.patch files are APPLIED, do NOT
re-apply them. It landed mid-run and that was safe: lsof showed the live controller had
ram_controller's .so ALREADY MAPPED, so that process has it cached in sys.modules and can
never see the replacement, and the next spawned run takes wheel AND patches together.
Installed now: ram_accelerator 12 / ram_controller 26, facades 12 / 26 — all four agree.

IDS: worker UP on the ABI-12 wheel (PID 82705, PPID=1, rayon 13) — swapped 30/08 02:27Z.
THE ADDRESS FIX (29/08, memory project_bits_above_64_or_fold): bits > 64 used to OR-FOLD
connection slots i and i+64 onto one address bit, so a "96-bit" neuron was a 64-bit neuron
with 32 input pairs merged. ram_core now names wide tuples by a splitmix64 hash; <= 64 bits is
IDENTITY (bit-exact, nothing to re-run). Blast radius is decided by the WINNER's width, not
the config cap.
LIVE COHORT = IDSXD (AC/CE matched pairs, desirability clones). scripts/worker_abi12_handoff.sh
(PID 81974, log /private/tmp/worker_abi12_handoff.log) is waiting on SMOKE flow 5895: on
`completed` it releases the 14 paused ciciot-96b flows (5896-5909) and restarts reruns
6050/6051. If it stops on anything else, the cohort stays paused BY DESIGN — read the log.
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
STILL NOT QUEUED, Luiz's call: the XDS cohort's 29 folded winners. FPGA claims are untouched
throughout (every Vivado-synthesised design is <= 64 bits).
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
