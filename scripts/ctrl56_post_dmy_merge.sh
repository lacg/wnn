#!/usr/bin/env bash
# CTRL-56 post-block merge (17/11/2026). Waits for "DMY DONE" in ctrl48/runs.log, then:
#   1. refuses if any phased_ga is running (a run loads Python from the main tree);
#   2. merges ctrl56-ckpt-cells into the main tree's checked-out branch (aborts on conflict);
#   3. runs the determinism gate: OLD = the pre-merge commit, NEW1/NEW2 = the merged tree,
#      compared with GATE_EXPECT_CELLS_ONLY=GRID (the one expected difference: the GRID
#      checkpoint now carries cells for its arch-only entries);
#   4. pushes the merged branch only on gate PASS. On FAIL the merge stays LOCAL, unpushed.
# Log: $LOG. Every step is greppable as "[ctrl56-merge]".
set -u
REPO=/Users/lacg/wnn
BR=ctrl56-ckpt-cells
RUNS=$REPO/logs/controller/ctrl48/runs.log
G=$REPO/logs/controller/ctrl56_gate
LOG=${LOG:-$G/merge.log}
mkdir -p "$G"
say() { echo "[ctrl56-merge] $(TZ=America/New_York date '+%d/%m %H:%M %Z') $*" | tee -a "$LOG"; }

say "waiting for 'DMY DONE' in $RUNS"
until grep -q "DMY DONE" "$RUNS"; do sleep 120; done
say "DMY DONE seen"
sleep 60
if pgrep -f "wnn.control.phased_g[a]" >/dev/null; then say "ABORT: a phased_ga is running — not merging under a live run"; exit 2; fi

PRE=$(git -C "$REPO" rev-parse HEAD)
CUR=$(git -C "$REPO" rev-parse --abbrev-ref HEAD)
say "merging $BR into $CUR (pre-merge $PRE)"
if ! git -C "$REPO" merge --no-ff --no-edit "$BR" >>"$LOG" 2>&1; then
	git -C "$REPO" merge --abort >>"$LOG" 2>&1
	say "ABORT: merge conflict — aborted, nothing changed"; exit 3
fi
say "merged: $(git -C "$REPO" log --oneline -1)"

OLD=$G/old_tree
git -C "$REPO" worktree add --detach "$OLD" "$PRE" >>"$LOG" 2>&1 || { say "ABORT: cannot create the old-leg worktree"; exit 4; }
GATE="$REPO/scripts/ctrl_determinism_gate.sh"
for leg in old:$OLD/src new1:$REPO/src new2:$REPO/src; do
	name=${leg%%:*}; src=${leg#*:}
	rm -rf "$G/$name"
	say "gate fly $name ($src)"
	bash "$GATE" fly "$G/$name" "$src" >>"$LOG" 2>&1 || { say "GATE LEG FAILED: $name — merge left local, NOT pushed"; exit 5; }
done
if GATE_EXPECT_CELLS_ONLY=GRID bash "$GATE" compare "$G/new1" "$G/new2" "$G/old" >>"$LOG" 2>&1; then
	say "gate PASS — pushing $CUR"
	git -C "$REPO" push >>"$LOG" 2>&1 && say "pushed $CUR" || say "PUSH FAILED (merge is local)"
else
	say "gate FAIL — merge left LOCAL, NOT pushed (see $LOG)"; exit 6
fi
git -C "$REPO" worktree remove --force "$OLD" >>"$LOG" 2>&1
say "done"
