#!/usr/bin/env bash
# Proof harness for the two dfa1l sweep-resilience scripts:
#   A. scripts/dfa1l_restart_at_cell_boundary.sh  (edge-triggered, one-shot)
#   B. scripts/dfa1l_sweep_supervisor.sh          (level-triggered, self-healing)
#
# Everything runs against FAKE processes in a sandbox; the real sweep is never
# touched (ROOT/SCRIPT/PHASED_PAT/IDS_WORKER are all overridden, and the fake
# patterns FAKECELLPAT / fake_driver.sh match nothing real).
#
# NOTE: every fake redirects stdout to /dev/null before backgrounding. Without
# that, a fake launched inside $(...) holds the command-substitution pipe open
# and the substitution blocks for the fake's whole lifetime.
set -u
W="/Users/lacg/wnn/scripts/dfa1l_restart_at_cell_boundary.sh"
S="/Users/lacg/wnn/scripts/dfa1l_sweep_supervisor.sh"
SB="/private/tmp/claude-501/-Users-lacg-wnn/e3da2191-6720-48f2-af03-8ca966858350/scratchpad/sbx"
pass=0; fail=0
ok()  { echo "  PASS: $1"; pass=$((pass+1)); }
bad() { echo "  FAIL: $1"; fail=$((fail+1)); }
chk() { if [ "$2" = "$3" ]; then ok "$1 ($2)"; else bad "$1 (got '$2' want '$3')"; fi; }
has() { if grep -q "$2" "$1"; then ok "$3"; else bad "$3"; fi; }
ndrv(){ grep -c "FAKE DRIVER STARTED" "$SB/driver.log" 2>/dev/null || echo 0; }

# SAFETY: pkill -f / pgrep -f are GLOBAL. Never match a production script name by
# pattern — a sandbox is only isolated for the names it remembered to fake, and
# an earlier version of this harness killed the live boundary watcher that way.
# Only fake-only patterns (FAKE*) may be pkill'd; anything running a real script
# is tracked by the PID we launched and killed by that PID alone.
KIDS=""
track() { KIDS="$KIDS $1"; }
kill_tracked() { for p in $KIDS; do kill -9 "$p" 2>/dev/null; done; KIDS=""; }

cleanup() {
	kill_tracked
	pkill -9 -f FAKECELLPAT    >/dev/null 2>&1
	pkill -9 -f fake_driver.sh >/dev/null 2>&1
	pkill -9 -f FAKEIDSWORKER  >/dev/null 2>&1
	sleep 1
}

# Guard rail: snapshot production PIDs up front and verify at the end that this
# harness left every one of them alone.
PROD_PIDS="$(pgrep -f 'run_dfa_1layer_study.sh|controller_mem_watchdog.sh' | tr '\n' ' ')"
prod_check() {
	local gone=""
	for p in $PROD_PIDS; do ps -p "$p" >/dev/null 2>&1 || gone="$gone $p"; done
	if [ -z "$gone" ]; then ok "harness left all production PIDs alive ($PROD_PIDS)"
	else bad "harness KILLED production PIDs:$gone"; fi
}

setup() {
	cleanup
	rm -rf "$SB"; mkdir -p "$SB/marks" "$SB/out"
	cat > "$SB/fake_driver.sh" <<'EOF'
#!/usr/bin/env bash
echo "FAKE DRIVER STARTED pid=$$"
sleep 600
EOF
	cat > "$SB/dying_driver.sh" <<'EOF'
#!/usr/bin/env bash
echo "FAKE DRIVER STARTED pid=$$"
exit 1
EOF
	cat > "$SB/python" <<'EOF'
#!/usr/bin/env bash
sleep "${2:-600}"
EOF
	cat > "$SB/idsworker.sh" <<'EOF'
#!/usr/bin/env bash
sleep "${1:-600}"
EOF
	chmod +x "$SB"/*.sh "$SB/python"
	bash "$SB/idsworker.sh" 900 FAKEIDSWORKER >/dev/null 2>&1 &
	IDSPID=$!
	touch "$SB/out/cell.out"
}

start_cell()   { "$SB/python" FAKECELLPAT "$1" >/dev/null 2>&1 & echo $!; }
start_driver() { bash "$SB/fake_driver.sh" >> "$SB/driver.log" 2>&1 & echo $!; }

run_watcher() {  # driver_pid cell_pid marker grace [script]
	WATCHER_ROOT="$SB" WATCHER_SCRIPT="${5:-fake_driver.sh}" \
	WATCHER_DRIVER_LOG="$SB/driver.log" WATCHER_LOG="$SB/watcher.log" \
	WATCHER_IDS_WORKER="$IDSPID" WATCHER_PHASED_PAT="FAKECELLPAT" \
	WATCHER_LOCK="$SB/restart.lock" \
	bash "$W" "$1" "$2" "$3" "$4" >/dev/null 2>&1
	echo $?
}

run_supervisor() {  # poll total_cells max_relaunch min_avail [script] -> exit code
	SUPERVISOR_ROOT="$SB" SUPERVISOR_SCRIPT="${5:-fake_driver.sh}" \
	SUPERVISOR_MARKDIR="$SB/marks" SUPERVISOR_OUTDIR="$SB/out" \
	SUPERVISOR_DRIVER_LOG="$SB/driver.log" SUPERVISOR_LOG="$SB/sup.log" \
	SUPERVISOR_STATE="$SB/relaunches" SUPERVISOR_TRIP="$SB/TRIPPED" \
	SUPERVISOR_IDS_WORKER="$IDSPID" SUPERVISOR_PHASED_PAT="FAKECELLPAT" \
	SUPERVISOR_TOTAL_CELLS="$2" SUPERVISOR_MAX_RELAUNCH_PER_HOUR="$3" \
	SUPERVISOR_MIN_AVAIL_GB="$4" \
	SUPERVISOR_RESTART_LOCK="${FAKE_LOCK:-$SB/restart.lock}" \
	SUPERVISOR_LOCK_STALE_S="${FAKE_LOCK_STALE_S:-1800}" \
	bash "$S" "$1" >/dev/null 2>&1
	echo $?
}

# Backgrounded supervisor, PID-tracked. Launching `bash "$S"` DIRECTLY (not via a
# backgrounded function) matters: backgrounding the function would give us the
# subshell's PID while the real supervisor ran as its child, so the kill would
# miss and we would be back to pattern-matching a production name.
start_supervisor() {  # poll total max_relaunch min_avail [script] -> pid
	SUPERVISOR_ROOT="$SB" SUPERVISOR_SCRIPT="${5:-fake_driver.sh}" \
	SUPERVISOR_MARKDIR="$SB/marks" SUPERVISOR_OUTDIR="$SB/out" \
	SUPERVISOR_DRIVER_LOG="$SB/driver.log" SUPERVISOR_LOG="$SB/sup.log" \
	SUPERVISOR_STATE="$SB/relaunches" SUPERVISOR_TRIP="$SB/TRIPPED" \
	SUPERVISOR_IDS_WORKER="$IDSPID" SUPERVISOR_PHASED_PAT="FAKECELLPAT" \
	SUPERVISOR_TOTAL_CELLS="$2" SUPERVISOR_MAX_RELAUNCH_PER_HOUR="$3" \
	SUPERVISOR_MIN_AVAIL_GB="$4" \
	SUPERVISOR_RESTART_LOCK="${FAKE_LOCK:-$SB/restart.lock}" \
	SUPERVISOR_LOCK_STALE_S="${FAKE_LOCK_STALE_S:-1800}" \
	bash "$S" "$1" >/dev/null 2>&1 &
	echo $!
}

echo "===================== A. BOUNDARY WATCHER ====================="
echo "== syntax =="
if bash -n "$W"; then ok "watcher bash -n clean"; else bad "watcher bash -n"; fi
if bash -n "$S"; then ok "supervisor bash -n clean"; else bad "supervisor bash -n"; fi

echo; echo "== A1: clean completion (marker appears) → restart =="
setup; D=$(start_driver); C=$(start_cell 3)
( sleep 6; echo '{}' > "$SB/marker.json" ) >/dev/null 2>&1 &
chk "exit code" "$(run_watcher "$D" "$C" "$SB/marker.json" 60)" "0"
if ps -p "$D" >/dev/null 2>&1; then bad "old driver still alive"; else ok "old driver killed"; fi
if pgrep -f FAKECELLPAT >/dev/null; then bad "cell survived"; else ok "cell tree killed"; fi
has "$SB/watcher.log" "clean completion" "took clean-completion path"
chk "driver launches" "$(ndrv)" "2"
chk "new driver PPID=1" "$(ps -o ppid= -p "$(pgrep -f fake_driver.sh | head -1)" 2>/dev/null | tr -d ' ')" "1"

echo; echo "== A2: R4 crash path (NO marker, next cell spawns) → restart =="
setup; D=$(start_driver); C=$(start_cell 3)
( sleep 7; "$SB/python" FAKECELLPAT 600 >/dev/null 2>&1 & ) >/dev/null 2>&1 &
chk "exit code" "$(run_watcher "$D" "$C" "$SB/nomarker.json" 60)" "0"
has "$SB/watcher.log" "R4 crash path" "detected no-marker crash path"
chk "driver launches" "$(ndrv)" "2"

echo; echo "== A3: grace expiry (no marker, no next cell) → restart =="
setup; D=$(start_driver); C=$(start_cell 2)
chk "exit code" "$(run_watcher "$D" "$C" "$SB/nomarker.json" 10)" "0"
has "$SB/watcher.log" "grace 10s expired" "logged grace expiry"
chk "driver launches" "$(ndrv)" "2"

echo; echo "== A4 (CHANGED): driver dies mid-cell → wait for orphan, THEN heal =="
setup; D=$(start_driver); C=$(start_cell 25)
( sleep 5; kill -9 "$D" ) >/dev/null 2>&1 &
rc=$(run_watcher "$D" "$C" "$SB/nomarker.json" 30)
chk "exit code (heals, no longer aborts)" "$rc" "0"
has "$SB/watcher.log" "letting the orphaned cell finish" "logged the orphan-wait decision"
has "$SB/watcher.log" "orphaned cell .* finished" "waited for the orphan to finish"
chk "driver launches" "$(ndrv)" "2"

echo; echo "== A5: IDS worker dead → ABORT, no relaunch (exit 3) =="
setup; kill -9 "$IDSPID" 2>/dev/null; sleep 1
D=$(start_driver); C=$(start_cell 2)
chk "exit code" "$(run_watcher "$D" "$C" "$SB/nomarker.json" 10)" "3"
chk "driver launches (no relaunch)" "$(ndrv)" "1"
has "$SB/watcher.log" "IDS worker .* not alive" "logged the IDS gate"

echo; echo "== A6: cell survives 3 kill attempts → ABORT (exit 4) =="
setup; D=$(start_driver); C=$(start_cell 2)
( end=$((SECONDS+60)); while [ "$SECONDS" -lt "$end" ]; do
    pgrep -f FAKECELLPAT >/dev/null || "$SB/python" FAKECELLPAT 30 >/dev/null 2>&1 &
    sleep 1; done ) >/dev/null 2>&1 &
chk "exit code" "$(run_watcher "$D" "$C" "$SB/nomarker.json" 10)" "4"
has "$SB/watcher.log" "kill attempt 1/3" "retried the kill (heals transient survivors)"
has "$SB/watcher.log" "survived 3 kill attempts" "aborted only after retries"
chk "driver launches (no double-run)" "$(ndrv)" "1"
sleep 60

echo; echo "== A7: relaunch keeps failing → 3 attempts then ABORT (exit 6) =="
setup; D=$(start_driver); C=$(start_cell 2)
chk "exit code" "$(run_watcher "$D" "$C" "$SB/nomarker.json" 10 dying_driver.sh)" "6"
has "$SB/watcher.log" "relaunch attempt 3/3" "retried the relaunch 3x"
has "$SB/watcher.log" "failed to start 3 times" "aborted after the cap (no infinite loop)"

echo; echo "===================== B. SUPERVISOR ====================="
echo; echo "== B1: sweep already complete → exit 10, launches nothing =="
setup; for i in 1 2 3; do echo '{}' > "$SB/marks/c$i.json"; done
chk "exit code" "$(run_supervisor 2 3 3 0)" "10"
has "$SB/sup.log" "SWEEP COMPLETE" "logged completion"
chk "driver launches" "$(ndrv)" "0"

echo; echo "== B2: stalled sweep (0 drivers, 0 cells) → self-heals =="
setup
( sleep 12; pkill -9 -f fake_driver.sh ) >/dev/null 2>&1 &
SUP=$(start_supervisor 5 40 3 0); track $SUP
sleep 10
chk "relaunched a driver" "$(ndrv)" "1"
has "$SB/sup.log" "STALLED" "detected the stall"
has "$SB/sup.log" "RELAUNCHED driver" "healed it"
kill_tracked; sleep 1

echo; echo "== B3: healthy (1 driver) → does NOT add a second =="
setup; D=$(start_driver); sleep 1
SUP=$(start_supervisor 3 40 3 0); track $SUP
sleep 8
chk "driver launches (unchanged)" "$(ndrv)" "1"
if grep -q "RELAUNCHED" "$SB/sup.log"; then bad "relaunched despite a healthy driver"; else ok "left the healthy driver alone"; fi
kill_tracked; sleep 1

echo; echo "== B4: orphan cell running, driver gone → waits, does NOT double-run =="
setup; C=$(start_cell 600); sleep 1
SUP=$(start_supervisor 3 40 3 0); track $SUP
sleep 8
chk "driver launches (waited)" "$(ndrv)" "0"
has "$SB/sup.log" "waiting for the orphan" "logged the orphan wait"
kill_tracked; pkill -9 -f FAKECELLPAT; sleep 1

echo; echo "== B5: IDS worker down → gate blocks the relaunch =="
setup; kill -9 "$IDSPID" 2>/dev/null; sleep 1
SUP=$(start_supervisor 3 40 3 0); track $SUP
sleep 8
chk "driver launches (blocked)" "$(ndrv)" "0"
has "$SB/sup.log" "IDS worker .* not alive" "logged the IDS gate"
kill_tracked; sleep 1

echo; echo "== B6: low memory → gate blocks, retries next tick =="
setup
SUP=$(start_supervisor 3 40 3 99999); track $SUP
sleep 8
chk "driver launches (blocked)" "$(ndrv)" "0"
has "$SB/sup.log" "squeezed box" "logged the memory gate"
kill_tracked; sleep 1

echo; echo "== B7: crash-loop brake trips after N relaunches =="
setup
rc=$(run_supervisor 2 40 2 0 dying_driver.sh)
chk "exit code" "$rc" "11"
has "$SB/sup.log" "BRAKE TRIPPED" "brake tripped instead of looping all night"
if [ -f "$SB/TRIPPED" ]; then ok "wrote the trip file"; else bad "no trip file"; fi
n=$(ndrv); if [ "$n" -le 2 ]; then ok "capped relaunches at the brake ($n)"; else bad "exceeded brake ($n)"; fi

echo; echo "== B8: refuses to start while the trip file exists =="
chk "exit code" "$(run_supervisor 2 40 3 0)" "11"
has "$SB/sup.log" "refusing to start" "logged the refusal"

echo; echo "== B9 (RACE): restart lock held → supervisor stands down =="
# Reproduces the collision: 0 drivers + 0 cells (looks stalled) during a PLANNED
# restart. Without the lock the supervisor would add a SECOND driver.
setup; date -u +%FT%TZ > "$SB/restart.lock"
SUP=$(start_supervisor 3 40 3 0); track $SUP
sleep 8
chk "driver launches (stood down)" "$(ndrv)" "0"
has "$SB/sup.log" "standing by" "logged the stand-down"
if grep -q "STALLED" "$SB/sup.log"; then bad "treated the restart window as a stall"; else ok "did not misread the window as a stall"; fi
kill_tracked; sleep 1

echo; echo "== B10: STALE lock (watcher died mid-restart) → heals anyway =="
# The lock must not disable healing forever if the watcher that took it crashed.
setup; date -u +%FT%TZ > "$SB/restart.lock"
SUP=$(FAKE_LOCK_STALE_S=1 start_supervisor 3 40 3 0); track $SUP
sleep 10
chk "relaunched despite the lock" "$(ndrv)" "1"
has "$SB/sup.log" "lock is stale" "logged the stale-lock override"
kill_tracked; pkill -9 -f fake_driver.sh; sleep 1

echo; echo "== B11: watcher holds the lock ONLY while restarting, not while waiting =="
# The whole point of a lock over a pgrep: a watcher waiting on a long cell must
# NOT suppress the supervisor. Assert no lock exists during the wait phase.
setup; D=$(start_driver); C=$(start_cell 30)
WATCHER_ROOT="$SB" WATCHER_SCRIPT="fake_driver.sh" WATCHER_DRIVER_LOG="$SB/driver.log" \
WATCHER_LOG="$SB/watcher.log" WATCHER_IDS_WORKER="$IDSPID" WATCHER_PHASED_PAT="FAKECELLPAT" \
WATCHER_LOCK="$SB/restart.lock" bash "$W" "$D" "$C" "$SB/nomarker.json" 10 >/dev/null 2>&1 &
WPID=$!; track "$WPID"
sleep 6
# Phase 1 — waiting on a live cell. The lock MUST be absent here; this is the
# assertion that would have caught keying stand-down on the process instead.
if [ -f "$SB/restart.lock" ]; then bad "took the lock while merely WAITING (would disable the supervisor for hours)"; else ok "no lock during the wait phase"; fi
if grep -q "took restart lock" "$SB/watcher.log" 2>/dev/null; then bad "logged taking the lock during the wait phase"; else ok "no lock log during the wait phase"; fi

# Phase 2 — end the cell, then WAIT FOR THE WATCHER TO ACTUALLY EXIT before
# asserting. A fixed sleep made the previous version pass for the wrong reason:
# "no lock" and "lock released" were both trivially true because the watcher had
# not reached the lock yet. Bounded, and scoped to OUR pid — never pkill by the
# production script name.
pkill -9 -f FAKECELLPAT
for _ in $(seq 1 40); do ps -p "$WPID" >/dev/null 2>&1 || break; sleep 2; done
if ps -p "$WPID" >/dev/null 2>&1; then bad "watcher never finished within 80s"; else ok "watcher completed its restart phase"; fi
has "$SB/watcher.log" "took restart lock" "did take the lock for the restart phase"
if [ -f "$SB/restart.lock" ]; then bad "lock leaked after the watcher exited"; else ok "lock released on exit (trap fired)"; fi
kill_tracked; sleep 1

echo; echo "== SAFETY: production processes untouched =="
prod_check

cleanup
echo; echo "=========================================="
echo "  PASS=$pass  FAIL=$fail"
[ "$fail" -eq 0 ] && echo "  ALL GREEN" || echo "  *** FAILURES ***"
echo "=========================================="
exit "$fail"
