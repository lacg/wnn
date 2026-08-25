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

**Currently armed:** job `ef691620`, schedule `13,43 * * * *` (off the :00/:30 marks on purpose),
re-armed 25/08/2026 20:0x EDT after a CLI restart. STATE block refreshed at the
same time (the previous block still named the finished gated weight sweep and
IDSZ as live).

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

STATE (25/08/2026 20:0x EDT — refresh this block when the programme changes).

CONTROLLER (the lever): the SWEEP LADDER — scripts/sweep_ladder_chain.sh, log
/private/tmp/sweep_ladder.log, markers experiments/sweepladder_markers (n/28, git-tracked),
outdir logs/controller/sweep_ladder, ckpts logs/controller/sweep_ladder/ckpt/<tag>. Armed
25/08 20:55Z, PPID=1, idempotent per marker. Runs under the VIABILITY GATE (--gate-stable 0.70
--gate-err 8.0) with --fit-aggregation zscore --zrank-clamp 3.0 and the SWEEP-WON weights
**S16noJM**: --fit-weight-err-sq 0.3125 --fit-weight-stable 0.25 --fit-weight-steady 0.4375.
STAGE A = the BITS sweep: 14 widths [10 12 14 16 18 20 22 24 26 28 30 32 34 36] x 2 seeds
(31337002/31337003), SEED-MAJOR interleave (round 1 = one seed of every width), neurons FIXED
at 32, --skip-stages neurons,bits so the swept width cannot drift. cf21_brushless, L4C.
~3.9 h/run measured at b=10 (~1900 s/gen x 5 CONNECTIONS gens) — wider bits will be slower, so
the 6-7 day A-D estimate is optimistic. Stage A ranks on LOWEST MEAN headline steady; cull
after round 1 (top 6 by steady OR within 1.25x of best); culled widths keep their round-1 marker.
NOTE b=10 produces 0/50 viable at population build (err ~45-57°, far outside the gate) — that is
Deb's rules working, not a fault; the narrowest width is expected to be a floor point.
BANKED (do NOT re-derive): the GATED WEIGHT SWEEP is COMPLETE, 29/29, all rc=0, 98 h of box
time. **S16noJM WON** (mean 94.1% stable / 2.35° err / 2.01° steady, n=5), paired 4-1/3-2 over
C10noJM and 3-2/3-2 over STEADY40; dropping jerk+mono improves BOTH parents (the saturation
finding). NOT a significance claim — 5-seed paired majority. E50S50 is n=4 BY DESIGN, NEVER
mean-compare it against n=5. No PID win: the best run (C10noJM s31337005, 97.4/1.80/1.25) loses
every column to PID on its matched seed (100%/1.24°/0.55°). Full table docs/controller_results.md.
27 pre-24/08 legacy-regime ladder files are archived to logs/controller/sweep_ladder_pre24aug_legacy/
— NOT comparable, never merge into the curve.
OPEN (behind the ladder): re-score 9 alt arms + rerun every banked sweep under the winning
aggregation; make --fit-aggregation REQUIRED (docs/FIT_AGGREGATION_REQUIRED_SPEC.md).

IDS: worker UP (13-core budget, ABI 7). Live = **IDSX** AC/CE matched-pair cohort (1 running,
109 queued). Banked: the general AC/CE claim is DEAD (6/18 pairs, pooled -0.026pp); CE20 beats
production +0.951pp on unswt-16b ONLY; unswr-quad is SATURATED; cicids REVERSES the prediction
and reached n=2. B34 bits-matched arm: read out when it drains.
QUEUED BEHIND IDSX: **MCS** multiclass screening, flows 5827-5841 (3 arms x 5 seeds 20401-20405,
UNSW temporal_3way quad 16b top20, production Wb weights, ONLY variable = ids_classification ∈
{binary, multi, hierarchical}). PRE-REGISTERED READ: macro-F1 + benign-FPR primaries, per-class
recall table MANDATORY (QSR lesson: aggregate-F1 win with recall losses on 8/9 classes is NOT
"detects better"). SEQUENCING: these 15 run BEFORE any IDSX-winner reseed; PAUSE them first if a
reseed must jump the queue.
RF/XGB multiclass baselines (the bar): UNSW temporal_3way macro-F1 0.52 · CICIDS random_3way
0.81 RF / 0.65 XGB · CIC-IoT subsample 0.89 — CIC-IoT does NOT collapse, so the bar is
per-dataset. The 46M neto_full leg is still running (chain PID 41321, log
/private/tmp/mc_baselines_chain.log); COMMIT each docs/multiclass_baselines/*.json as it lands.
IDSZ is COMPLETE (n=5, CE20 leads). SP100 is a DEAD control (superseded code era) — do not
resurrect it.
ESCALATE if the worker process count is 0 while flows are queued, or if any IDSX/MCS flow moves
to `failed`.

ONLY add lines beyond the six if:
(a) a NEW controller marker landed — quote every stage's held-out block (stable%/err°/steady°, plus alt where the run prints it), name the arm's alt RANK weight + λ_alt + seed, mark the headline stage, and when both arms of a seed exist print the PAIR table.
(b) an escalation — chain dead before its markers complete, rc!=0, >1 controller running, avail below 4 GiB, a run past 5 h, any "weight_alt > 0 but ... is None", IDS worker down while flows are queued, or any IDSX/MCS flow failed.
(c) the box went IDLE — say so and name the unstarted pending items.
Otherwise stop after the six lines.
