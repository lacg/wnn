"""Smoke test plan for the flow pause/resume MVP.

This is documentation, NOT an automated test. Running pause/resume against
the live production worker would conflict with whatever the worker is
currently doing — only run this against a fresh test worker.

Locked design (see commit message):
  - Pause granularity = flow level (pauses current GA experiment)
  - Pause check point = end of each GA generation (via existing shutdown_check)
  - Checkpoint = per-gen overwrite of <prefix>_ga.json with current_gen + complete:false
  - Trigger = flows.pause_requested INTEGER column (polled by worker between gens)
  - API = POST /api/flows/<id>/pause + POST /api/flows/<id>/resume
  - After pause = worker proceeds to next queued flow (does NOT park)
  - Resume order = id-DESC pickup (no front-jump)

Manual test:

1) Deploy the new code (user does this when worker is idle):
   - cd /Users/lacg/wnn/dashboard && cargo build --release
   - Stop the running :3000 dashboard, replace binary, restart
   - Stop the running worker, restart it
   - The migration `ALTER TABLE flows ADD COLUMN pause_requested INTEGER NOT NULL DEFAULT 0`
     runs automatically on dashboard startup

2) Queue a small test flow (UI: New Flow → tiny config like 5n×4b on neto_subsample,
   max_iterations=20 so the GA completes in ~1 min and you can pause mid-stream).
   Note the flow id (e.g. 9999).

3) Wait for the worker to pick it up and reach gen ~3 of the GA. You should see
   in the worker log:
     "[ArchitectureGA] Gen 003/020: best=..."

4) Pause from the UI (click the ⏸ Pause button) OR via curl:
   curl -k -X POST -H 'Content-Type: application/json' -d '{}' \
       https://localhost:3000/api/flows/9999/pause

5) Watch the worker log. Within ~10s (the dashboard poll interval used by
   should_stop) it should print:
     "Flow 9999 pause requested via dashboard, pausing at end of generation..."
     "[ArchitectureGA] Shutdown requested at generation N, stopping..."
     "Flow paused at end of generation (per-gen checkpoint already on disk)"
   And the worker should pick up the next queued flow (if any).

6) Verify the flow status is 'paused':
   curl -k https://localhost:3000/api/flows/9999 | jq .status
     → "paused"

7) Verify the per-gen checkpoint exists:
   ls checkpoints/<safe_flow_name>/exp_NN/
     → should contain *_ga.json with current_iteration=<paused gen>, complete=false

8) Resume from the UI (click the ▶ Resume button) OR via curl:
   curl -k -X POST -H 'Content-Type: application/json' -d '{}' \
       https://localhost:3000/api/flows/9999/resume

9) Worker picks the flow back up (id-DESC ordering — no front-jump). In the
   worker log:
     "Resuming from checkpoint at generation N+1"
     "[ArchitectureGA] Gen N+1/020: best=..."

10) Let it finish. Verify the final report looks sane and iterations 0..final
    are all recorded in the DB (some rows from the pre-pause run will have
    been deleted on the resume's set_status(RUNNING) — that's by design, since
    the GA's per-gen checkpoint is the source of truth, not the DB iteration log).

API summary
-----------
POST /api/flows/<id>/pause     → 200 {id, pause_requested: true}
POST /api/flows/<id>/resume    → 200 {flow with status=queued, pause_requested=false}

Bad state returns:
  400  if /pause is called on a flow that's not running/queued/paused
  400  if /resume is called on a flow that's not paused
  404  if flow_id not found
"""
