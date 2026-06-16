# Session handoff — controller multi-seed + XDS n=30 + checkpoint/fitness work (16/06/2026)

Read this to resume after a context clear. **Detached runs survive a clear; the
30-min monitor cron does NOT** (session-only) — re-arm it (or check manually) after clearing.

---

## ⚠️ FIRST: what dies vs survives when you clear context
- **SURVIVES** (detached, PPID=1): `seed1@100n/200m` controller run (pid 36185), the IDS
  worker (pid 2262) chewing the XDS n=30 queue. These keep going.
- **DIES**: the controller monitor **cron `57fd3936`** (session-only) and any chat-driven
  watching. After clearing, **re-arm the monitor** (prompt at the bottom) or just check manually.
- The seed1 watcher / rerun orchestrator already did their jobs and exited — nothing to revive.

---

## 1. Controller multi-seed (C10, drone attitude) — where it stands

**★ MILESTONE A DONE — real clean 3-seed mean (50/100 config), best-stage held-out (fresh seed 99990101):**
```
seed1@50/100   81% / 3.84°   (MEMORY won)
seed0-rerun    81% / 3.78°   (NEURONS won; MEMORY overfit 94gen→65held)
seed2-rerun    61% / 4.58°   (NEURONS won; MEMORY overfit 86gen→55held)
─────────────────────────────
MEAN  74.3 ± 9.4%  /  4.07 ± 0.36°
```
vs **sweep C10 3-seed 49.3%/6.36°** (+25pp/−2.3°, big win), vs **c10 single 68%/4.63°** (+6pp),
vs **PID 100%/1.3°** (PID still ahead). 2/3 seeds hit 81%; seed2 the weak draw → the ±9.4% spread.
All clean (no 180°) ⇒ the original seed0/seed2 180° duds were **purely the SIGTERM corruption**.

**🔄 MILESTONE B IN FLIGHT — seed1@100n/200m A/B** (pid 36185, base 31337002):
- Config: NEURONS **100ep, patience 3, check-interval 3**; MEMORY **200ep, patience 6**.
  (Tuned-down patience from the analysis — NEURONS 9-gen tail, MEMORY 18-gen. ~14–17h total.)
- Dir: `logs/controller/c10_seed1_n100m200_20260614/seed1_base31337002/run.out`
- As of last check: STAGE 1 NEURONS ~gen 8, gen-line **22% / 10.38°** (already healthier than
  seed1@50/100's 6%/12.72° at the same point — more eps = better during-search signal).
- **The verdict to report at completion:** seed1@50/100 (81%/3.84°) **vs** seed1@100n/200m
  best-stage held-out. ⚠️ It moves BOTH knobs (NEURONS 50→100 AND MEMORY 100→200) vs the
  baseline → measures the COMBINED effect, NOT MEMORY alone. (To isolate MEMORY later: a 50/200 run.)
- **Check it:** `grep -A2 "HELD-OUT REPORT \[NEURONS\]\|\[MEMORY\]" <that run.out> | grep RESULT`;
  done when `grep -c "PHASED-GA RESULT"` ≥ 1. best-stage = higher stable, then lower err.

**Scripts (all in /tmp, may be wiped on reboot — re-create from the launcher if needed):**
- `scripts/controller_multiseed.py` (committed) — the reusable parallel launcher.
- `/tmp/ctl_rerun_seq.py` — sequential rerun orchestrator (DONE its job).
- `/tmp/ctl_seed1_ep100_watcher.py` — launched seed1@100n/200m (DONE).

## 2. NEXT controller cohort — bake these in (decided this session)
- **`--magnitude-aware-patience` (BAKED INTO `scripts/controller_multiseed.py`, default ON — 16/06).**
  Patience tracks err°/stable% MAGNITUDE, not the rank-WHM. Fixes the seed1@100n/200m MEMORY
  premature stop (gen 27/120 while stable was 26.5→42% climbing). Opt out: `--no-magnitude-aware-patience`.
- **⚠️ RE-EVALUATE the patience-tail cut now that magnitude-aware is on.** The `--neurons-patience 3
  --check-interval 3` / `--memory-patience 6` cut below was derived UNDER the blind tracker — its
  "ALL improvement in first ≤9 gens, tail never improves" premise is partly an ARTIFACT of the blind
  tracker not seeing late gains (seed1@100n/200m proved MEMORY improved to gen 27+). With recovery-
  ∝-magnitude, you WANT runway for genuine late jumps to recover patience. Consider KEEPING the
  larger patience (5/5, mp8) for the first magnitude-aware cohort, then trim once you see where it
  actually plateaus. The launcher still defaults p5/check5/mp8 — decide before launch.
- ~~`--neurons-patience 3 --check-interval 3` (was 5/5); `--memory-patience 6`~~ — superseded above.
- Run controllers **ONE AT A TIME, never parallel** ([[feedback_controller_one_at_a_time_xds_priority]]).
- Lamarckian, skip bits+connections, grid sn{8,12,16}×b{24,30}, pop50, folds5, steps1000, tilt5.0,
  C10 weights err.40/stb.30/jrk.20/mno.10, report-seed 99990101, report-episodes 100.
- **FIRST cohort to run = the clean seed1@100n/200m re-run** (now with magnitude-aware patience) to
  resolve the confounded MILESTONE B A/B (50/100 81%/3.84° vs 100n/200m, which was cut short).

## 3. Fitness / patience redesign (designed, NOT implemented) — `docs/controller_fitness_patience_redesign.md`
Root issue: harmonic fitness is **rank-based → magnitude-blind** (seed2's 20%→70% jump moved the
objective −0.0004). Patience watches it → mis-early-stops. (jerk/mono ARE plumbed+active in clean
runs — the "RESERVED/None" docstring was stale, fixed `981e93e2`.) Plan:
- **(a)** magnitude-scaled patience (recover ∝ err/stable improvement ratio; keep rank selection →
  comparable). Cheap, low-risk, do first.
- **(b)** value-based fitness (exp/convex goodness transforms, weights need not sum to 1). Deep fix,
  re-baselines. Do as a deliberate experiment after.

---

## 4. XDS-cicids (IDS paper) — 96b-Wa to n=30 IN FLIGHT
- **Top-3 decided** (n=5 re-rank): 96b-Wa #1 (99.58/0.09), 64b-Wa #2 (99.56/0.12), 16b-Wa #3
  (99.55/0.12); **16b-Wc dropped** (was #4). All within ~0.03pp F1 (separable-CICIDS). 16b-Wa is
  the efficiency pick (cheapest train+inference at ~tied F1); 96b-Wa the nominal top.
- **96b-Wa extended to n=30**: flows **4151–4175** (25 fresh seeds). As of last check: 7 done + 1
  running + 22 queued. Worker pid 2262 chewing them, ~5 days to drain. GPU-bound (won't fight the
  CPU controller).
- **Re-rank command** (Pareto-scan, best-F1 across genome-types × 7 modes): the inline python in
  `/tmp/xds_rerank_n5.py` (extend the 96b-Wa flow-id list as new ones complete). User watches XDS
  on the dashboard UI (https://macstudio.local:5173; backend API https://localhost:3000, verify=False).
- Flow creation = `scripts/queue_cross_dataset.build_flow("cicids-random", W, weight, seed,
  arch_override=(500,34))` → POST /api/flows (lands `pending`) → POST /{id}/restart (→queued).

## 5. Checkpoint unification — DONE + DEFERRED-DEPLOY
- `phased.PhasedCheckpointManager` (codec-based yaml.gz + SaveCadence) now used by BOTH controller
  and IDS GA (commits 5c4bb756 + 0b203baa). Old `CheckpointManager` deleted. Packed cells/conns
  (D), async between-stage save (C), holdout-sample (A) all committed (93cb4a87, 429ae58f).
- **⚠️ DEFERRED:** the IDS-side migration is Python-only and takes effect on the **next worker
  restart** — which must wait until the XDS cohort is idle (restart cancels the running flow,
  [[feedback_restart_cancels_running_flow]]). Backward-compat loads legacy `.json` checkpoints.

---

## 6. Re-arm the controller monitor after clearing (cron, every 30 min)
Use ScheduleWakeup/CronCreate `8,38 * * * *` with this prompt (observe-only, never touch a process):
> Controller monitor (every 30 min). Relay in chat (user can't see tool output). NEVER
> stop/kill/SIGTERM/touch any process. seed1@100n/200m (base 31337002) in
> logs/controller/c10_seed1_n100m200_20260614/seed1_base31337002/run.out is the only active run.
> Report: pgrep -f "base-seed 31337002" liveness; latest gen line (stage, Gen x/y, err°, stable%,
> patience x/y [counts DOWN: 3/3=full], elapsed); NEURONS+MEMORY held-out when done. When it has
> FINAL_REPORT: report the A/B — seed1@50/100 (81%/3.84°) vs seed1@100n/200m best-stage held-out
> (changes BOTH knobs → combined effect). PushNotification + update memory
> project_wsweep_winner_c10.md. Then stop the cron. Crash (Traceback) → PushNotification + tail.

---

## 7. Key rules / lessons (this session)
- **Never SIGTERM a running controller** — the cancel flag corrupts the population (→180°/CE=inf)
  even if the process keeps running. Caused the seed0/seed2 re-runs. [[feedback_controller_one_at_a_time_xds_priority]]
- **One controller at a time; XDS/IDS (deadline) > controller (pet project)** for resources.
- **Never act on processes without explicit OK** (kill/restart/launch).
- **Don't poll os.kill(pid,0) for liveness on a child you spawned** — a zombie reads as alive;
  use p.poll() / launch detached (init-reaped). (Caused the orchestrator deadlock, fixed.)
- Memory index: `project_wsweep_winner_c10.md` (the controller story + 3-seed mean + tuning),
  `feedback_controller_one_at_a_time_xds_priority.md`, `project_xds_cicids_thermo_round_map.md`.
