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

CONTROLLER (the lever): the STAGE-A DESIRABILITY A/B — scripts/sweep_ladder_ab_chain.sh,
log /private/tmp/sweep_ladder_ab.log, markers experiments/sweepladder_markers (git-tracked),
outdir logs/controller/sweep_ladder. Armed 26/08 19:21Z, PPID=1, idempotent per marker.
TWO ARMS per (width, seed), S16noJM weights on BOTH; only the aggregation differs:
  arm GATE  = zscore + gate 0.70/8.0 (shipped ABI-24 regime; tag SL_A_b{b}n32_..._s{seed})
  arm DESIR = --fit-aggregation desirability, NO gate flags (ABI 25; tag ..._desir_s{seed})
Widths [12..36 step 2] (b=10 TRIMMED, measured dead), seeds 31337002/31337003, WIDTH-MAJOR
with arm pairs adjacent. The 4 banked markers (b12-b18 s31337002) are reused as gate-arm
points. Round-1 cull ranks on GATE-DISTANCE (desirability half-lives over the gate pair,
err .5556 @ 8 deg / stable .4444 @ 0.70) — NOT steady (Luiz 26/08). Stage A ONLY; stages
B-D relaunch after the A/B verdict is read ONCE.
DESIRABILITY (26/08, Luiz's redesign; docs/DESIRABILITY_FITNESS_SHAPES.md): one
multiplicative utility, score = weighted half-lives lost, LOWER better, ABSOLUTE scale
(fit values comparable across gens/runs — for desir-arm runs `(=)` IS a fixed scale;
gate-arm zscore stays pool-relative). Calculator prints `Desir(...)`; a desir run passing
gate flags CRASHES by design. Old ladder KILLED 26/08 ~19:10Z mid-b20 (Luiz: no need to
wait); its 5 markers stand.
BANKED (do NOT re-derive): gated wsweep COMPLETE — S16noJM won (94.1/2.35/2.01, n=5,
paired majority, NOT significance); no PID win. Old-ladder finding: 0/686 samples feasible
at 32n -> the gated arm's weights NEVER applied in-search (violation-only ranking); 128n
wsweep was 95% feasible (capacity mismatch, not wrong thresholds). Width curve so far
(gate arm, s31337002, headline held-out): b12 22.8%/19.72/19.24 · b14 24.2/28.41/42.67 ·
b16 12.4/18.20/22.11 · b18 31.0%/11.54/11.82 (leader).
OPEN (behind the A/B): stages B-D under the winning aggregation; re-score 9 alt arms;
rerun banked sweeps; make --fit-aggregation REQUIRED.

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
