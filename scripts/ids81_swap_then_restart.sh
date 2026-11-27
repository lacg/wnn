#!/bin/bash
# IDS-81 swap-then-restart waiter (Luiz 27/11/2026: no ABI bump; swap at worker-idle AFTER 7641 finishes; the
# four restarted flows 6433 6438 7535 7549 must start on the NEW worker). Replaces scripts/ids81_restart_after_7641.sh.
#
#   0. arm: fetch origin/ids81-swap-batch, build its worker wheel (maturin build, NO install) from a git-archive
#      copy, stage the patch `git diff HEAD...origin/ids81-swap-batch -- src/wnn tests dashboard`
#      (three-dot = only the branch's own changes; HEAD has moved on since the branch point). scripts/ is left
#      out so this file can be armed from a copy without colliding with the patch;
#   1. wait until 7641 is completed (failed/cancelled/paused → stand down, '!!');
#   2. require the worker idle: 0 running AND 0 queued (anything queued → stand down: never reorder IDS);
#   3. worker_swap.py --watch-flow 7641 --install-wheel --apply-patch (patch first, wheel second, reverted on a
#      failed install — fails closed); rc != 0 or a marker without the wheel → '!!', the four are NOT restarted;
#   4. verify: new worker PID, installed .so sha256 == the wheel's .so, ABI 30, IDS-81 Python present;
#      then commit the applied patch on the main checkout (history; push best-effort);
#   5. archive the four stale checkpoint folders, restart the four from zero (POST restart from_beginning,
#      PATCH queued), and confirm the first one to start is a child of the NEW worker.
# Every step → logs/ids/ids81_restart.log. '!!' marks a stand-down that needs a human.
#
# Dry run (no build, no swap, no API, fake DB): scripts/ids81_swap_then_restart.sh --dry <fake.db>
set -u
PROJECT=/Users/lacg/wnn
DRY=0; DBFILE=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db
[ "${1:-}" = "--dry" ] && { DRY=1; DBFILE=${2:?--dry needs a fake db path}; }
DB="file:$DBFILE?mode=ro"
API=https://localhost:3000/api/flows
FOUR="6433 6438 7535 7549"
STALE="ids56c-cicids-ce20-zs-binary-2m-max250-xo_blend_neuron-rate0.25-lattice-encon-r20421
ids56uc-cicids-ce20-zs-binary-2m-max250-xo_blend_neuron-rate0.25-nb02-encon-r20419
ids6-ciciot46m-b15ac-desir-r20419 ids6-ciciot46m-b15ac-desir-r20420"
WORK=/Volumes/20260401-WDBlack-SN850X-2TB/ids81-swap
LOG=$PROJECT/logs/ids/ids81_restart.log
[ $DRY = 1 ] && { WORK=$(mktemp -d); LOG=$WORK/ids81_restart.log; }
POLL=${IDS81_POLL:-300}

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
die() { log "!! $* — STANDING DOWN (nothing restarted)"; exit 1; }
q() { sqlite3 "$DB" "$1"; }
st() { q "select status from flows where id=$1;"; }
run() { if [ $DRY = 1 ]; then log "[dry] would run: $*"; else "$@"; fi; }
worker_pid() { pgrep -f "wnn.ram.experiments.worke[r]" | head -1; }

stage_patch_and_wheel() {
	cd "$PROJECT" || die "no $PROJECT"
	run git fetch -q origin ids81-swap-batch || die "fetch failed"
	mkdir -p "$WORK"
	PATCH=$WORK/ids81_swap_batch.patch
	run sh -c "git diff HEAD...origin/ids81-swap-batch -- src/wnn tests dashboard > '$PATCH'"
	run git apply --check "$PATCH" || die "patch does not apply to $(git rev-parse --short HEAD)"
	SRC=$WORK/src_$(date -u +%Y%m%d%H%M); WHEELS=$WORK/wheels
	run mkdir -p "$SRC" "$WHEELS"
	run sh -c "git archive origin/ids81-swap-batch src/wnn/ram/strategies/accelerator | tar -x -C '$SRC'"
	run sh -c "cd '$SRC/src/wnn/ram/strategies/accelerator' && unset CONDA_PREFIX && \
		CARGO_TARGET_DIR=/Volumes/20260401-WDBlack-SN850X-2TB/cargo-target-ids81-wheel \
		nice -n 15 $PROJECT/wnn/bin/maturin build --release -i $PROJECT/wnn/bin/python -o '$WHEELS'" \
		>> "$LOG" 2>&1 || die "wheel build failed"
	WHEEL=$(ls -t "$WHEELS"/ram_accelerator-*.whl 2>/dev/null | head -1)
	[ $DRY = 1 ] && WHEEL=$WORK/ram_accelerator-dry.whl
	[ -n "$WHEEL" ] || die "no wheel in $WHEELS"
	SO_SHA=$( [ $DRY = 1 ] && echo dry || unzip -p "$WHEEL" 'ram_accelerator/*.so' | shasum -a 256 | cut -c1-64)
	log "staged: patch $PATCH ($(wc -l < "$PATCH" 2>/dev/null || echo 0) lines), wheel $WHEEL (.so sha256 $SO_SHA)"
}

wait_for_7641() {
	log "waiting for flow 7641 to finish (poll ${POLL}s)"
	while :; do
		s=$(st 7641)
		case "$s" in
			completed) break ;;
			running|queued) sleep "$POLL" ;;
			*) die "7641 is '$s', not completed" ;;
		esac
	done
	log "7641 completed"
}

require_idle() {
	busy=$(q "select group_concat(id||':'||status) from flows where status in ('running','queued');")
	[ -z "$busy" ] || die "worker not idle after 7641 — running/queued: $busy (never reorder IDS)"
	for f in $FOUR; do [ "$(st $f)" = paused ] || die "flow $f is '$(st $f)', not paused"; done
	log "worker idle (0 running, 0 queued); $FOUR paused"
}

swap() {
	OLD_PID=$(worker_pid); MARK=$WORK/worker_swap_ids81.done; rm -f "$MARK"
	log "swap: old worker pid ${OLD_PID:-none}"
	run sh -c "cd $PROJECT && $PROJECT/wnn/bin/python scripts/worker_swap.py --watch-flow 7641 \
		--install-wheel '$WHEEL' --apply-patch '$PATCH' --db $DBFILE --marker '$MARK' >> '$LOG' 2>&1"
	rc=$?
	[ $DRY = 1 ] && return 0
	[ $rc = 0 ] || die "worker_swap rc=$rc (3 = install failed + patch reverted; see marker $MARK)"
	python3 -c "import json,sys; m=json.load(open('$MARK')); sys.exit(0 if m['installed_wheel'] and m['applied_patch'] and not m['interrupted'] else 1)" \
		|| die "marker says the swap is incomplete: $(cat "$MARK")"
}

verify_new_worker() {
	[ $DRY = 1 ] && { log "[dry] would verify new pid, .so sha, ABI 30, guard modules"; return 0; }
	NEW_PID=""
	for _ in $(seq 1 30); do NEW_PID=$(worker_pid); [ -n "$NEW_PID" ] && [ "$NEW_PID" != "$OLD_PID" ] && break; sleep 10; done
	[ -n "$NEW_PID" ] && [ "$NEW_PID" != "$OLD_PID" ] || die "no NEW worker after the swap (old $OLD_PID)"
	cd "$PROJECT" && PYTHONPATH=$PROJECT/src wnn/bin/python - "$SO_SHA" >> "$LOG" 2>&1 <<'EOF' || die "post-swap verification failed (worker $NEW_PID is up — inspect before any flow runs)"
import glob, hashlib, os, sys
import ram_accelerator as ra
so = glob.glob(os.path.join(os.path.dirname(ra.__file__), "*.so"))[0]
sha = hashlib.sha256(open(so, "rb").read()).hexdigest()
assert ra.ABI_VERSION == 30, ra.ABI_VERSION
assert sha == sys.argv[1], f"installed .so {sha} != wheel .so {sys.argv[1]}"
from wnn.ram.strategies.phased.worker_abi_guard import completed_phase_refusal, stamp_metadata  # IDS-81 (b)
from wnn.ram.experiments.flow_dir_name import flow_dir_name  # IDS-81 (a)
print(f"[verify] ABI {ra.ABI_VERSION}, .so sha256 {sha[:16]}… matches the wheel, IDS-81 Python present")
EOF
	log "verified: new worker pid $NEW_PID, new .so, ABI 30, IDS-81 guard modules present"
	git add -A -- $(git apply --numstat "$PATCH" | cut -f3) && git commit -q -m "IDS-81 swap: apply origin/ids81-swap-batch at worker-idle (wheel + Python)" \
		&& log "committed the applied patch: $(git log --oneline -1)" || log "!! commit of the applied patch failed — tree is patched but uncommitted"
	git push -q origin HEAD >> "$LOG" 2>&1 || log "push failed (non-fatal)"
}

restart_four() {
	A=$PROJECT/checkpoints/_archive_ids81_$(date -u +%Y%m%d)
	run mkdir -p "$A"
	for d in $STALE; do [ -d "$PROJECT/checkpoints/$d" ] && run mv "$PROJECT/checkpoints/$d" "$A/" && log "archived checkpoints/$d"; done
	for f in $FOUR; do
		if [ $DRY = 1 ]; then log "[dry] would POST $API/$f/restart {from_beginning:true}; PATCH status=queued"; continue; fi
		r1=$(curl -sk -X POST $API/$f/restart -H 'Content-Type: application/json' -d '{"from_beginning":true}' -o /dev/null -w '%{http_code}')
		r2=$(curl -sk -X PATCH $API/$f -H 'Content-Type: application/json' -d '{"status":"queued"}' -o /dev/null -w '%{http_code}')
		log "flow $f restart http=$r1 | queued http=$r2 | status now $(st $f)"
	done
}

confirm_on_new_worker() {
	[ $DRY = 1 ] && { log "[dry] would wait for the first of $FOUR to run and check its PPID == new worker"; return 0; }
	for _ in $(seq 1 90); do
		row=$(q "select id||' '||ifnull(pid,'') from flows where id in (${FOUR// /,}) and status='running' limit 1;")
		[ -n "$row" ] && break; sleep 20
	done
	[ -n "$row" ] || { log "!! none of $FOUR started within 30 min — check the worker"; exit 1; }
	set -- $row
	ppid=$(ps -o ppid= -p "${2:-0}" 2>/dev/null | tr -d ' ')
	if [ "$ppid" = "$NEW_PID" ]; then log "flow $1 running (runner pid $2) under the NEW worker $NEW_PID"
	else log "!! flow $1 runner pid ${2:-?} has parent '${ppid:-?}', not the new worker $NEW_PID"; exit 1; fi
}

log "armed (pid $$, dry=$DRY)"
stage_patch_and_wheel
wait_for_7641
require_idle
swap
verify_new_worker
restart_four
confirm_on_new_worker
log "done: swap at idle after 7641, $FOUR restarted from zero on the new worker"
