#!/usr/bin/env bash
# RACING PREDICTIVITY PROBE — chain wrapper (06/09/2026, Luiz: "run the predictivity
# probe before any code"; racing at FOLD boundaries, keep one third).
#
# Runs scripts/racing_fold_probe.py ONLY on an idle box (it is a controller process:
# GPU scoring + Rust DAgger training), in this order, each marker-gated and idempotent:
#   0. MICRO SMOKE — 3 candidates, 1 round x 2 episodes per fold, 2 eval episodes.
#      Exercises every code path the dry run could not (train / export cells /
#      score / fitness / analysis) in under a minute. FAILS CLOSED: no smoke marker,
#      nothing else runs — a broken probe must not burn an hour of the box.
#   A. stage3_connections population of TAB_on b32n256 s31337002, 60 offspring,
#      WITH the exactness reference (one K-fold call vs K single-fold calls). ~60 min.
#      This is the case 4 of 5 CONNECTIONS generations are in (mature population).
#   B. stage0_grid population of the same run, 60 offspring, no reference. ~35 min.
#      The gen-1 case: offspring of the grid pool.
# The recipe argv is the translation A/B's ON arm at b32 n256 gamma=1 (the banked A
# side of the future racing A/B), minus --save-winner (the arm library adds that).
set -u
ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1

LOG="/private/tmp/racing_probe.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
MARK="experiments/racing_markers"
CKPT="logs/controller/translation_ab/ckpt/TAB_on_b32n256_cf21_brushless_L4C_s31337002"
SCRATCH="/private/tmp/racing_probe_ckpt"
mkdir -p "$MARK" "$SCRATCH" logs/controller/racing_probe

# Mirror of scripts/translation_ab_chain.sh run_point (arm=on) at the round-2 winner
# shape, with --save-stage-checkpoints pointed at scratch (the probe never saves).
RECIPE="--levels 16 --lamarckian --skip-stages neurons,bits --max-cells 180000 --max-cells-strict \
--save-stage-checkpoints ${SCRATCH} \
--neurons-gens 5 --neurons-patience 3 --conns-gens 5 --conns-patience 3 --memory-gens 120 --memory-patience 2 \
--pop 50 --num-eval-folds 5 --check-interval 2 --magnitude-aware-patience \
--eval-episodes 100 --memory-eval-episodes 200 --steps 2000 --tilt 5.0 \
--fit-weight-err-sq 0.3125 --fit-weight-stable 0.25 --fit-weight-steady 0.4375 \
--fit-aggregation zscore --zrank-clamp 3.0 --gate-stable 0.70 --gate-err 8.0 \
--delta-gamma 1.0 --grid-bits 32 --grid-output-neurons 256 --max-output-neurons 256 \
--report-episodes 100 --holdout-pop-sample 8 --runs 1 --memory-mode BINARY \
--airframe cf21_brushless --disturbance L4C --teacher mpcof \
--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i \
--obs-collective-cmd --obs-alt-err --obs-vz --translation --reward-lambda-alt 0 \
--grid-state-neurons 0 --max-state-neurons 0 \
--report-seeds 99990101 99990102 99990103 99990104 99990105 --base-seed 31337002"

log() { echo "[racing-probe-chain] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controller_pids() { pgrep -f -- "-m wnn.control.phased_ga" 2>/dev/null || true; }
other_chains() { pgrep -f "scripts/(crn_bits_curve_chain|crn_refly_chain|sweep_ladder_gamma|leak_revisit_chain|translation_ab_chain)\.sh" 2>/dev/null || true; }
wait_idle() {
	local beat=0
	while [ -n "$(controller_pids)" ] || [ -n "$(other_chains)" ]; do
		[ $((beat % 30)) = 0 ] && log "waiting — box busy"
		beat=$((beat + 1)); sleep 60
	done
}

# run_probe <name> <ckpt-file> <extra probe args...>
run_probe() {
	local name="$1" ck="$2"; shift 2
	local out="${MARK}/PROBE_${name}.json" log_out="logs/controller/racing_probe/${name}.out"
	[ -f "$out" ] && { log "SKIP ${name} (marker exists)"; return 0; }
	[ -f "$ck" ] || { log "ABORT — checkpoint missing: $ck"; return 1; }
	wait_idle
	log "===== START ${name} ====="
	/usr/bin/time -l "$VP" -u scripts/racing_fold_probe.py --ckpt "$ck" --recipe-args "$RECIPE" \
		--out "$out" "$@" > "$log_out" 2>&1
	local rc=$?
	log "${name} rc=${rc}"
	[ "$rc" = "0" ] && [ -f "$out" ] || { log "ABORT — ${name} failed (rc=${rc}, marker $( [ -f "$out" ] && echo present || echo MISSING )). Last lines:"; tail -5 "$log_out" | while read -r l; do log "    $l"; done; return 1; }
	return 0
}

log "########## ARMED — smoke → A (stage3 + reference) → B (stage0) ##########"
# 0. Micro smoke: shrink the recipe's training budget through the same flags the
#    smoke test uses (--rg-rounds / --rg-episodes-per-round / --rg-eval-episodes) and
#    the CRN pool size (--eval-episodes). 3 candidates, no reference.
SMOKE_RECIPE="${RECIPE} --rg-rounds 1 --rg-episodes-per-round 2 --rg-eval-episodes 2"
SMOKE_RECIPE="${SMOKE_RECIPE/--eval-episodes 100/--eval-episodes 2}"
run_smoke() {
	local out="${MARK}/PROBE_smoke.json"
	[ -f "$out" ] && { log "SKIP smoke (marker exists)"; return 0; }
	wait_idle
	log "===== START smoke (3 candidates, 1 round x 2 eps, eval 2) ====="
	"$VP" -u scripts/racing_fold_probe.py --ckpt "${CKPT}/stage3_connections.yaml.gz" \
		--recipe-args "$SMOKE_RECIPE" --out "$out" --candidates 3 --reference \
		> logs/controller/racing_probe/smoke.out 2>&1
	local rc=$?
	log "smoke rc=${rc}"
	[ "$rc" = "0" ] && [ -f "$out" ] || { log "ABORT — smoke FAILED; nothing else runs. Last lines:"; tail -8 logs/controller/racing_probe/smoke.out | while read -r l; do log "    $l"; done; exit 1; }
}
run_smoke
run_probe "stage3_s31337002" "${CKPT}/stage3_connections.yaml.gz" --candidates 60 --reference || exit 1
run_probe "stage0_s31337002" "${CKPT}/stage0_grid.yaml.gz" --candidates 60 || exit 1
log "---------- SUMMARY ----------"
for n in stage3_s31337002 stage0_s31337002; do
	"$VP" - "${MARK}/PROBE_${n}.json" >> "$LOG" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print("  %s  (%d candidates, %d inherited cells, folds %d)" % (d["stage"], d["candidates"], d["inherited_cells"], d["folds"]))
for r in d["cuts"]:
	print("    cut after fold %d: spearman %+.3f  top-third kept %d/%d  regret %.4f  true-best survives %s  train-units %.2f"
	      % (r["cut_after_fold"], r["spearman"], r["top_kept"], r["top_size"], r["regret"], r["true_best_survives"], r["train_units"]))
if "exactness" in d:
	e = d["exactness"]; print("    EXACTNESS: identical=%s  max|dreward|=%.6f" % (e["identical"], e["max_abs_reward_delta"]))
print("    timing:", ", ".join("f%d train %ss score %ss" % (t["fold"], t["train_s"], t["score_s"]) for t in d["timing"]))
PY
done
log "########## RACING PROBE COMPLETE ##########"
