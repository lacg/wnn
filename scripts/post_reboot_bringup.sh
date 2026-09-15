#!/usr/bin/env bash
# POST-REBOOT BRING-UP (written 14/09/2026 for the macOS 27 update reboot).
#
# A reboot kills EVERY detached (PPID=1) process: dashboard, IDS worker, memory
# sampler + watchdog, vite, the controller chains and the run they were flying,
# and the marker-repair jobs. This script relaunches all of them, in dependency
# order, each idempotent (skips what is already up), each detached so it survives
# this shell. Run it ONCE after login, from a terminal:
#
#     bash scripts/post_reboot_bringup.sh
#
# What it deliberately does NOT do:
#   - re-arm the Claude tick cron (session-only — CronCreate from
#     docs/controller_status_tick_prompt.md in the CLI session)
#   - lift a HOLD sentinel (experiments/HOLD_CONTROLLER): if present, the
#     controller chain is NOT launched and the script says so — rm it yourself.
#   - touch the IDS DB: the worker's own find_stale_running_flows requeues the
#     flow that was running at the reboot (stale flows REQUEUE, never fail).
set -u
PROJ="/Users/lacg/wnn"
VOL="/Volumes/20260401-WDBlack-SN850X-2TB"
PY="$VOL/wnn/venv/bin/python"
DASH="$VOL/cargo-target/release/wnn-dashboard"     # NEVER dashboard/target (stale 03/07 copy)
cd "$PROJ" || exit 1
unset CONDA_PREFIX || true
export PYTHONPATH="$PROJ/src/wnn:"
log() { echo "[bringup $(date -u +%FT%TZ)] $*"; }

# ---- preconditions ---------------------------------------------------------
[ -f "$VOL/wnn/db/wnn.db" ] || { log "ABORT: $VOL not mounted (DB missing)"; exit 1; }
[ -x "$DASH" ] || { log "ABORT: dashboard binary missing at $DASH"; exit 1; }
"$PY" -c "import ram_controller as c, ram_accelerator as a; print('wheels ok controller ABI', c.ABI_VERSION, 'worker ABI', a.ABI_VERSION)" || { log "ABORT: wheels do not import"; exit 1; }

up() { pgrep -f "$1" >/dev/null 2>&1; }   # caller brackets the pattern's last char

# ---- 1. dashboard (cwd dashboard/, CARGO_TARGET_DIR, log dashboard.out) -----
if up "release/wnn-dashboar[d]"; then log "dashboard already up"; else
	( cd "$PROJ/dashboard" && CARGO_TARGET_DIR="$VOL/cargo-target" nohup "$DASH" >> "$PROJ/dashboard.out" 2>&1 < /dev/null & ) ; disown 2>/dev/null || true
	for i in $(seq 1 30); do curl -sk -o /dev/null https://localhost:3000/api/flows && break; sleep 1; done
	log "dashboard launched (pid $(pgrep -f 'release/wnn-dashboar[d]' | head -1))"
fi

# ---- 2. IDS worker (rayon 13, log /private/tmp/wnn_worker.log) ---------------
if up "wnn.ram.experiments.worke[r]"; then log "worker already up"; else
	RAYON_NUM_THREADS=13 nohup "$PY" -u -B -m wnn.ram.experiments.worker --url https://localhost:3000 --no-ssl-verify \
		>> /private/tmp/wnn_worker.log 2>&1 < /dev/null & disown
	log "worker launched (pid $!) — it requeues the flow that was running at the reboot"
fi

# ---- 3. memory sampler + watchdog -------------------------------------------
if up "controller_mem_sample[r].sh"; then log "mem sampler already up"; else
	nohup bash scripts/controller_mem_sampler.sh > /dev/null 2>&1 < /dev/null & disown; log "mem sampler launched (pid $!)"
fi
if up "controller_mem_watchdo[g].sh"; then log "mem watchdog already up"; else
	nohup bash scripts/controller_mem_watchdog.sh >> logs/controller/mem_watchdog.log 2>&1 < /dev/null & disown; log "mem watchdog launched (pid $!)"
fi

# ---- 4. vite dev server (:5173) ---------------------------------------------
if up "node_modules/.bin/vit[e] dev"; then log "vite already up"; else
	( cd "$PROJ/dashboard/frontend" && nohup npm run dev >> "$PROJ/logs/vite.log" 2>&1 < /dev/null & ) ; disown 2>/dev/null || true
	log "vite launched"
fi

# ---- 5. controller: arm B re-fly, remaining seeds (idempotent chain) --------
bd=$(ls experiments/sweepladder_markers/SL_C_b24n256_cf21_brushless_L4C_g10_s3133700[2-5]_bd.json 2>/dev/null | wc -l | tr -d ' ')
if [ "$bd" -ge 4 ]; then log "arm B re-fly complete ($bd/4) — nothing to fly; box idle BY DESIGN"
elif [ -f experiments/HOLD_CONTROLLER ]; then log "HOLD_CONTROLLER present — NOT launching the arm B chain ($bd/4 banked); rm it to resume"
elif pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null; then log "a controller is already flying — not launching"
else
	nohup bash scripts/arm_b_delta_label_chain.sh >> /private/tmp/arm_b_delta_label.log 2>&1 < /dev/null & disown
	log "arm B chain launched (pid $!) — skips the $bd banked seeds, flies the rest (~5 h each); log /private/tmp/arm_b_delta_label.log"
fi

# ---- 6. marker repair, sequential (all idempotent: done rows are skipped) ---
# second re-score pass (retries the rows the first pass refused) -> stage-select
# recalc for arch-only #0 headlines -> MEMORY-headline runs -> CONNECTIONS-headline runs.
if [ "${SKIP_REPAIR:-0}" = "1" ]; then log "SKIP_REPAIR=1 — repair sequence NOT launched (recalc gate-3 fault open, 15/09)"
elif up "recalc_headline[s].py" || up "rescore_first_report_see[d].py"; then log "a repair job is already running — not relaunching the sequence"; else
	nohup bash -c "
		cd '$PROJ'; export PYTHONPATH='$PROJ/src/wnn:'
		nice -n 10 '$PY' -u scripts/rescore_first_report_seed.py >> logs/controller/rescore_first_seed.log 2>&1
		nice -n 10 '$PY' -u scripts/recalc_headlines.py >> logs/controller/recalc_headlines.log 2>&1
		nice -n 10 '$PY' -u scripts/recalc_headlines.py --headline-stages MEMORY --any-genome >> logs/controller/recalc_headlines_memory.log 2>&1
		nice -n 10 '$PY' -u scripts/recalc_headlines.py --headline-stages CONNECTIONS --any-genome >> logs/controller/recalc_headlines_connections.log 2>&1
	" > /dev/null 2>&1 < /dev/null & disown
	log "repair sequence launched (pid $!): re-score 2nd pass -> recalc arch-only -> MEMORY -> CONNECTIONS"
fi

log "DONE. Remaining by hand: (1) re-arm the tick cron in the Claude session from docs/controller_status_tick_prompt.md;"
log "      (2) verify PPID=1 on everything: ps -o pid,ppid,command -ax | grep -E 'wnn-dashboard|experiments.worker|mem_(sampler|watchdog)|vite dev|delta_label_chain|recalc_headlines' | grep -v grep"
log "      (3) IDS: sqlite3 'file:$VOL/wnn/db/wnn.db?mode=ro' \"select id,name,status from flows where status in ('running','queued') order by id limit 3;\" — the reboot-interrupted flow should be back to queued/running, never failed."
