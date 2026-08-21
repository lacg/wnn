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

**Currently armed:** job `ba2c26d9`, schedule `13,43 * * * *` (off the :00/:30
marks on purpose), armed 20/08/2026 10:15 EDT.

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
  sqlite3 "file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro" "select status,count(*) from flows where name like 'IDSZ-%' group by status;"

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

STATE (20/08/2026 10:15 EDT — refresh this block when the programme changes).

CONTROLLER (the lever): GATED WEIGHT SWEEP — scripts/gated_weight_sweep_chain.sh, log
/private/tmp/gated_wsweep.log, markers experiments/gatedwsweep_markers (n/30), outdir
logs/controller/gated_wsweep, ckpts logs/controller/gated_wsweep/ckpt/<tag>. The first
controller runs under the VIABILITY GATE (ABI 24, commit 2aa768a3): --fit-aggregation zscore
--zrank-clamp 3.0 --gate-stable 0.70 --gate-err 8.0. SIX ARMS x 5 seeds (31337002..31337006),
SEED-MAJOR interleave (round 1 = one seed of every arm): C10 (err .40/stable .30/jerk .20/
mono .10, CONTROL 1) · S16 (err .25/steady .35/stable .20/jerk .15/mono .05, CONTROL 2) ·
C10noJM (err .57/stable .43, matched pair) · S16noJM (err .3125/stable .25/steady .4375,
matched pair) · E50S50 · STEADY40. Flight config byte-identical to the fitness A/B: 128n,
grid-bits 24+30, mpcof, cf21_brushless, L4C, lambda_alt=0. ~3.3 h/run, 30 runs ~ 99 h. Armed
21/08 ~14:16 EDT, PPID=1. Idempotent per marker. Matched-pair prediction (pre-registered):
with viability gated, removing jerk/mono weights is at worst neutral on the primaries.
The FITNESS A/B is COMPLETE (10/10, corrected verdict zscore 5/5 stable, 4/5 err —
dominance-filtered stage-select; markers experiments/fitnessab_markers).
PRE-REGISTERED READ: paired per-seed on the held-out full row; primary = stable% and err°; steady°/alt always quoted; winner = paired majority across 5 seeds, NEVER best-of-N. Caveat: fitness changes the search trajectory, so this measures "does zscore FIND better genomes", not a bit-level A/B.
BANKED (do NOT re-derive):
  pair 1 (seed 31337002): harmonic NEURONS#0 85.2%/3.26°/2.87°/alt 0.770m · zscore MEMORY#0 94.4%/2.47°/2.19°/alt 0.648m → zscore takes both primaries + steady + jerk + alt; loses mono only.
  run 3 (harmonic, 31337003): MEMORY#1 94.8%/2.39°/2.11°/alt 0.559m.
OPEN FINDING (measured twice, both combines): stage-select passes over a BETTER NEURONS stage because .20 jerk + .10 mono outvote .40 err + .30 stable once attitude saturates. Suspect is C10's WEIGHT VECTOR, not the aggregation. Do NOT "fix" it by selecting on held-out — that leaks. Task #8 covers re-running every banked sweep under the winning aggregation.

IDS: worker UP (13-core budget, ABI 7). The live IDS work is the FITNESS-WEIGHT SWEEP **IDSZ-unswt-quad-16b-*** (flows 5380-5425): 23 arms — the 21 historical WSWEEP tags plus Wb-CTRL (0.1/0.2/0.35/0.35, current production) and C35-CTRL (0.35/0.3/0.3/0.05) — × seeds 20301/20302, UNSW-NB15 **temporal_3way** quad 16b (measured 4.9 min/flow; random_3way is 24.8 and plain random 86.8), **fitness_aggregation=zscore**, interleaved by seed. Verified live in the worker log: `[ArchitectureGA] Fitness calculator: ZRank(...)`. ~10 min/run, ~8 h total.
Two earlier attempts are PAUSED and kept for comparison, do NOT resume without asking: IDSW-unswt-* (5288-5333, harmonic, 5 completed) and IDSWR-unswr-* (5334-5379, random_3way 64b, none run). The 660 prior SP100 cohort flows are also paused and genuinely gated (pause now flips queued→paused, commit 84a85359).
ESCALATE if the worker process count is 0 while flows are queued, or if IDSZ flows move to `failed`.

ONLY add lines beyond the six if:
(a) a NEW controller marker landed — quote every stage's held-out block (stable%/err°/steady°, plus alt where the run prints it), name the arm's alt RANK weight + λ_alt + seed, mark the headline stage, and when both arms of a seed exist print the PAIR table.
(b) an escalation — chain dead before its markers complete, rc!=0, >1 controller running, avail below 4 GiB, a run past 5 h, any "weight_alt > 0 but ... is None", IDS worker down while flows are queued, or any IDSZ flow failed.
(c) the box went IDLE — say so and name the unstarted pending items.
Otherwise stop after the six lines.
