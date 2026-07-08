---
name: checkpoint
description: Save the full working context to a snapshot file so the conversation can be cleared and resumed without losing state, then reload it in a fresh session. Use when the user says "save the context", "checkpoint", "clear and reload context", "handoff", "context is getting long", or invokes `/checkpoint`, `/checkpoint save`, or `/checkpoint reload`. Mode is chosen by the argument: no arg or `save` → write the snapshot + tell the user to /clear; `reload` → read the snapshot back and resume.
---

# Context checkpoint (save → /clear → reload)

`/clear` wipes the conversation but KEEPS session crons + detached (PPID=1) processes
alive. This skill persists the working state to a file across that boundary so a fresh
session picks up exactly where this one left off. It has two modes, selected by `$1`.

The snapshot file is **`.claude/context-snapshot.md`** (stable path; survives /clear).

---

## Mode SAVE  —  `/checkpoint`  or  `/checkpoint save`

Write a COMPLETE, self-contained handoff to `.claude/context-snapshot.md`. Gather the
live state first (don't write from memory — verify), then write the file, then tell the
user the two-step resume.

### 1. Gather live state (run these, fold results into the snapshot)
- **Tasks:** call `TaskList` — capture every non-deleted task with its status.
- **Detached procs (PPID=1):** `ps -o pid,ppid,etime,command -ax | grep -E "wnn|phased_ga|worker|driver" | grep -v grep` — record what's running, PID, and what each is.
- **Background waiters / markers:** list any `/tmp/wnn_*_done.json` markers and note armed background shell waiters (they DIE on CLI exit — say so).
- **Session crons:** if any were armed this session, list their IDs + schedules (they DIE on CLI exit → must be re-armed after a full restart, survive /clear).
- **Git:** `git -C /Users/lacg/wnn log --oneline -8` and `git -C /Users/lacg/wnn status --short` — recent commits + uncommitted files.
- **In-flight experiments:** any running flows / proof runs / logs being watched (paths + how to read progress).

### 2. Write `.claude/context-snapshot.md` with these sections
```
# Context snapshot — <UTC timestamp>

## RESUME CUE
After /clear, run: `/checkpoint reload`

## What we're doing (1-3 sentences)
<the current objective + where we are in it>

## Tasks
<TaskList output, verbatim, with statuses>

## Live processes (PPID=1 — survive /clear AND CLI exit)
<PID | what it is | how to check its progress (log path / marker)>

## Session-only (DIE on CLI exit — re-arm if the CLI was restarted)
<crons: id + schedule + purpose ; background waiters + what they wait for>

## Recent commits
<git log --oneline -8>

## Uncommitted / in-progress edits
<git status --short + a note on what's half-done>

## NEXT STEPS (do these on reload, in order)
1. <first concrete action>
2. ...
```
Be specific and concrete — the future session has ONLY this file, not the conversation.
Prefer file paths, PIDs, marker files, and exact commands over prose.

### 3. Arm the auto-reload flag
Write a one-shot flag so the next session auto-reloads without a third command:
```bash
touch "$CLAUDE_PROJECT_DIR/.claude/.pending-reload" 2>/dev/null || touch .claude/.pending-reload
```
The `SessionStart` hook (`.claude/hooks/checkpoint-autoreload.sh`, registered in
`.claude/settings.json`) fires AFTER `/clear`, sees this flag, injects "run `/checkpoint
reload`", and deletes the flag (fires exactly once). So the user only has to `/clear`.

### 4. Tell the user (exactly)
> Snapshot saved to `.claude/context-snapshot.md` and auto-reload armed. Just run:
> 1. `/clear`
>
> Reload happens automatically on the fresh session (SessionStart hook) — no third command.
> (Detached procs + session crons keep running across /clear; a full CLI restart also
> kills the crons/waiters — the snapshot lists which to re-arm.)

Do NOT run /clear yourself — it is user-initiated.

---

## Mode RELOAD  —  `/checkpoint reload`

0. Clear any leftover one-shot flag (the hook normally does this, but a manual reload
   should too, so a stale flag can't trigger a second auto-reload):
   `rm -f "$CLAUDE_PROJECT_DIR/.claude/.pending-reload" 2>/dev/null || rm -f .claude/.pending-reload`
1. Read `.claude/context-snapshot.md`. If it is missing, say so and stop.
2. **Verify the live state still matches** (don't trust the file blindly):
   - Re-check the listed PPID=1 procs are still alive (`ps -p <pid>` / `pgrep`).
   - Check any markers named in the snapshot (`ls /tmp/wnn_*_done.json`).
   - Note anything that changed since the snapshot (a proc that finished, a marker that fired).
3. **Re-create the task list**: for each task in the snapshot, call `TaskCreate` to restore it
   (with the right status via `TaskUpdate`), so the future work is tracked again.
4. If the snapshot lists session crons/waiters AND the CLI was restarted (not just /clear),
   re-arm them per the snapshot.
5. Give the user a 3-5 line status: what's running, what changed, and the NEXT STEPS — then
   continue from step 1 of NEXT STEPS.

## Notes
- This is complementary to the durable memory system (`/Users/lacg/.claude/projects/-Users-lacg-wnn/memory/`):
  memories hold cross-session FACTS; this snapshot holds the volatile WORKING state (tasks,
  PIDs, half-done edits, next actions) that would otherwise be lost on /clear.
- The snapshot is overwritten each SAVE — it always reflects the latest checkpoint.
