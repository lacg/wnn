#!/usr/bin/env bash
# SessionStart hook: auto-reload a checkpoint after /clear (or a fresh start/resume).
#
# The /checkpoint save skill drops a one-shot flag file. When the next session
# begins (SessionStart fires AFTER /clear with source="clear"), this hook reads
# the flag, injects an instruction to run `/checkpoint reload`, and deletes the
# flag so it fires exactly once. No flag → no output → normal session.
#
# SessionStart stdout is added to the session context; we use the structured
# JSON form so the instruction is unambiguous.
set -euo pipefail

# Resolve project dir (CLAUDE_PROJECT_DIR is set by the harness; fall back to CWD).
proj="${CLAUDE_PROJECT_DIR:-$PWD}"
flag="${proj}/.claude/.pending-reload"

# No pending reload → silent no-op (exit 0, no stdout).
[ -f "$flag" ] || exit 0

# One-shot: consume the flag before doing anything else so a hook error can't
# wedge the session into a reload loop.
rm -f "$flag"

# Inject the reload instruction as additional context.
cat <<'JSON'
{
  "hookSpecificOutput": {
    "hookEventName": "SessionStart",
    "additionalContext": "A checkpoint auto-reload flag was set by a prior `/checkpoint save`, and this session has just started fresh (post-/clear or restart). Before doing anything else, invoke the `checkpoint` skill in RELOAD mode (equivalent to the user running `/checkpoint reload`): read `.claude/context-snapshot.md`, verify the live PPID=1 processes and markers it lists are still valid, re-create the task list, and give the user the short status + NEXT STEPS. Do this now, automatically, without waiting for the user to ask."
  }
}
JSON
