# Dashboard Stack Review — 12/06/2026

**Scope:** everything the 10/06 architecture review did NOT cover — the Svelte/Vite
frontend (~13.4K lines, 19 files), the `wnn-dashboard` Rust backend (~6.8K lines:
db/api/models/watcher/parser), and the worker↔dashboard boundary
(`dashboard_client.py`, ~1.1K lines). Four parallel review passes (god-pages,
rest-of-frontend, Rust db+api, boundary layer), findings verified with file:line;
the two headline contract drifts were independently re-verified by hand.

**Verdict in one paragraph.** The stack works because the happy path is narrow,
not because the code is defensive. Two worker-client methods have silently
drifted off the API contract (one of them produces the documented Rule-2
"flow with 0 experiments" trap); the backend has three surviving relatives of
the 01/06 `->running` wipe plus zero transactions and zero status-transition
validation; the dashboard's *safe* stale-flow requeue is dead code while the
worker's stale→**failed** logic is what actually causes the "restart kills
flows" pain; and two filesystem holes (path traversal via flow name; arbitrary
file read/delete via checkpoint `file_path`) are one odd POST body away from
disaster. The frontend is stylistically clean (exactly ONE sub-1rem font
violation repo-wide) but has systemic error-state and fetch-race patterns that
brick or mislead pages. `watcher/` and `parser/` are dead code — not even in
the module tree.

---

## Tier 1 — Active bugs + catastrophic-impact holes (fix before anything else)

| # | Finding | Where | Impact |
|---|---------|-------|--------|
| 1.1 | **`create_flow` client drift**: `experiments` nested inside `config`; API wants top-level `experiments` (`#[serde(default)]` → `[]`). Every flow created via `DashboardClient.create_flow` has **0 experiments** = the Rule-2 "completes instantly, does nothing" trap. Live callers: cli.py:250,763, phased_search.py (dashboard registration), flow.py `_register_flow` | dashboard_client.py:261-272 vs api/mod.rs:815-824 | HIGH — silent no-op flows |
| 1.2 | **`link_experiment_to_flow` hits the wrong route**: POSTs `{experiment_id,…}` to `/api/flows/:id/experiments` (= `add_experiment_to_flow`, wants `{experiment: spec}`) → 422 → ConnectionError after retries. Correct route `/experiments/link` exists. This call has been **100% failing**; fallback paths leave orphaned experiments / flows stuck `running` | dashboard_client.py:529-548 vs api/mod.rs:53-54,974-1010 | HIGH |
| 1.3 | **Path traversal in restart**: `safe_name` neutralizes `" "` and `/` but not `..` — a flow named `..` makes restart-from-beginning `remove_dir_all` resolve to the **project root** | api/mod.rs:1268-1285 | CATASTROPHIC, low likelihood |
| 1.4 | **Arbitrary file read/delete via checkpoints**: `file_path` unvalidated at create; `/download` streams any path, DELETE removes any file | api/mod.rs:1507-1552,1571-1599,1602-1671 | HIGH |
| 1.5 | **Fresh-DB bootstrap broken**: `flows.last_heartbeat` exists in no migration/SCHEMA but every flow SELECT + heartbeat UPDATE uses it → new DB file = every flows endpoint 500s. Live DB has it out-of-band only | db/mod.rs:213-555 vs 572,1261 | HIGH for any rebuild/migration |
| 1.6 | **Zero-experiment flows submittable from BOTH ends**: API accepts `experiments: []` silently (api/mod.rs:822-851); flows/new UI lets you submit with no phases (+page.svelte:684-922) | both | the Rule-2 trap, again |
| 1.7 | **LM Validation Progression table reads wrong genome types**: non-IDS branch reads `bestF1Summary/bestFprSummary` under "Best CE / Best Acc" headers → dashes for every LM phase | experiments/[id]/+page.svelte:1062-1067 | visible data bug |
| 1.8 | **One failed request bricks whole pages**: `error` never reset in `loadFlow`/`loadFlows`/`loadCheckpoints`; the `{:else if error}` gate then permanently replaces the page (flow detail: 16 action handlers all feed it; plus `saveExperiment` is a dead feature that *unconditionally* bricks) | flows/[id]:996-1000,327-841,378-384; flows/+page:41-70; checkpoints/+page:15-50 | recurring UX failure |
| 1.9 | **Flow duplication corrupts experiment types**: non-GA/TS phase_types (neurogenesis/lamarckian/lambda_sweep…) duplicated as `'ts'`; per-experiment params dropped | flows/[id]/+page.svelte:585-603 | silent wrong-strategy flows |
| 1.10 | **`--text-tertiary` undefined**: consumed in 6 files / 15 usages; cancelled-status badges render with NO background | app.css (missing) + flows:77, experiments:27, … | visible everywhere |

## Tier 2 — Wipe-family / state-machine robustness (the 01/06 lesson, finished)

| # | Finding | Where |
|---|---------|-------|
| 2.1 | **Gen-0 wipe window survives**: →running wipe guard tests `current_iteration<=0`, not "do iteration rows exist"; crash during gen 0 → re-pickup deletes real rows. No transaction around SELECT+5×DELETE+UPDATE | db/mod.rs:2110-2135 |
| 2.2 | **Unconditional cancel overwrite**: `stop_flow_process` UPDATE has no `WHERE status IN (...)` → TOCTOU flips completed→cancelled; skips completed_at | db/mod.rs:899-944 |
| 2.3 | **ANY→ANY transitions accepted** on flows (PATCH overwrites started_at/NULLs completed_at on →running) and experiments (status is free-text String; typos round-trip as `Pending`) | db/mod.rs:759-876; api/mod.rs:166-173 |
| 2.4 | **Zero transactions in db/mod.rs** (verified): create_flow (flow+seed+N experiments — crash = partial-experiment flow), deletes, restart, the wipe — all multi-statement, all unprotected | db/mod.rs throughout |
| 2.5 | **Stale-flow handling inverted**: dashboard's guarded `requeue_stale_flow` (status→queued, data preserved) is dead code; live behavior is worker-side 90s stale→**failed**. This IS the "restart cancels running flows" pain. A restarted worker (empty `self._running`) can fail flows another process owns | db/mod.rs:1269-1302 (dead) vs worker.py:271-317 |
| 2.6 | **Restart-resume deletes FK-referenced genomes** (`best_genomes.genome_id` → FK violation → 500 mid-mutation, flow stuck) AND throws away the cross-flow `validation_summaries` cache (hours of 46M re-validation) | db/mod.rs:979-1025,1364-1375 |
| 2.7 | **Worker exception handler can wedge flows**: unguarded `get_flow` inside `_handle_flow_exception` — dashboard down (the usual cause) → raises → no requeue/fail marking → flow stuck `running` until stale→failed (loses auto-resume) | worker.py:933-942 |
| 2.8 | **Retry stack hazard**: outer 3×30s × transport `Retry(total=5, backoff)` incl. READ timeouts and POST → hung dashboard blocks synchronous training-loop calls ~10 min; non-idempotent POSTs (checkpoints, flows) can duplicate rows on connection-reset-after-commit | dashboard_client.py:175-240 |

## Tier 3 — Correctness risks

- **No fetch-race protection anywhere in the frontend** (no AbortController/request
  tokens): poll responses can roll status backwards for a tick (re-arming
  Pause/Stop buttons), stale navigations overwrite newer state, leaderboard
  filter changes can render old-filter data. flows/[id] also misses reactive
  reload on id change (stale flow under new URL + poll splices new data into old
  UI); experiments/[id] leaks stale flow context and grid-search results across
  navigations. (god-pages E2-E4, F4-F6; leaderboard) — fix site: one shared
  request-token fetch helper.
- **Flow-control buttons have no double-submit guard** (Stop/Pause/Resume/
  Restart ×2) — duplicate POSTs into the Tier-2 state machine. flows/[id]:712-841.
- **`architecture_type` has three optional, disagreeing sources** (experiment
  column, flow config param, legacy cluster_type) with silent LM fallback —
  the wrong-columns bug family remains structurally possible; `update_experiment
  (cluster_type=…)` is silently dropped by the API (field absent from
  UpdateExperimentRequest). One derivation helper + add field server-side.
- **SQLite posture**: WAL never enabled (sqlx 0.8 doesn't set it) → writer
  blocks all readers; per-WS-client 500ms polling (2 queries/client/500ms, up
  to 500 rows, no `(experiment_id, created_at)` index) is the main reader
  pressure. db/mod.rs:13-26; api/mod.rs:1959-2098.
- **Unclamped `limit`/`offset`**: `?limit=-1` returns entire tables (iterations
  = millions of rows). api/mod.rs:244,800,1489,1792.
- **Leaderboard**: expansion keyed by hash only (expands up to 7 rows, breaks
  virtual-scroll math); Rank header sorts worst-first on first click.
- **Dates**: checkpoints page bypasses `$lib/dateFormat` (US format leaks);
  locale auto-detect defaults to browser (MM/DD on US locale) instead of the
  project DD/MM/24h convention; `new Date("YYYY-MM-DD HH:MM:SS")` parses
  SQLite-naive timestamps as LOCAL time — verify API serialization, likely a
  silent timezone shift.
- **best_genomes schema drift**: migration vs SCHEMA define different shapes
  (threshold_mode missing in migration) → legacy DBs fail the INSERT; conflict
  fallback id-lookup omits threshold_mode (wrong row id); `BestGenome.
  threshold_mode` serde-defaults to `train_cal` (mislabels Pareto provenance).
  db/mod.rs:179-208 vs 507-530, 3035-3059; models/mod.rs:479-504.
- **404→None conflation** in `_request` → `result["id"]` TypeErrors masquerade
  as code bugs; state errors (400) surface as ConnectionError after retries.
- **`-Infinity%`** possible in experiments info card when all best_accuracy
  null (experiments/[id]:425); flows/new memory hint uses 32-bit `1<<bits`
  (negative bytes at ≥31 bits, exactly the 100b regime) :1464.
- **Silent dropdown failures** in SeedCheckpointSelector (3 paths set no error).
- Smaller: `checkpoint_type='phase_end'` not in enum (stored raw, mislabeled on
  read); client can't clear seed_checkpoint_id; `find_checkpoint_by_path` scans
  only latest 100; `check_cached_validation` outage → silent cache-miss →
  hours of re-validation on 46M.

## Tier 4 — Smells / cleanup / structure

- **DELETE dead code**: `dashboard/src/watcher/` + `dashboard/src/parser/`
  (not in module tree; reference types that don't exist; `/api/watch` is
  display-only — document it). Also dead frontend experiment-edit feature
  (flows/[id]:357-384 + unused state), unused `goto` import, stale 17-line
  TODO in stores.ts documenting already-implemented backoff.
- **Frontend dedup** (single best follow-up per the rest-of-frontend pass):
  `lib/api.ts` (fetch wrapper with error handling + stale-response guard — 8+
  copy-pastes with 3 different error behaviors), `lib/format.ts` (percent/CE/
  duration formatters ×5 sites), `lib/statusColors.ts` (duplicated switch ×2+,
  fixes 1.10 once). Consolidate `.status-badge`/btn/filter CSS into app.css.
- **God-file split maps produced** (line-ranged seams, same discipline as D3):
  - `db/mod.rs` (3177) → db/{migrations, schema, flows, flow_lifecycle,
    experiments, iterations, checkpoints, validations, combined_validations,
    gating, best_genomes, parse}
  - `api/mod.rs` (2098) → api/{experiments, live_progress, combined_validations,
    gating, snapshot, flows, flow_lifecycle, checkpoints, watch, best_genomes, ws}
  - `experiments/[id]/+page.svelte` (3387) → 11 components;
    `flows/[id]/+page.svelte` (2994) → 9 components + `useFlowPolling` module;
    `flows/new/+page.svelte` (2153) → 8 components + shared `flowTemplates.ts`
    (also the fix vehicle for 1.9). Full tables in the agent seam maps —
    preserved in section 6 below… (see git history of this commit for the
    raw agent reports if more line detail is needed).
- **types.ts gaps**: `GenomeValidationType` missing `best_f1|best_fpr` (IDS
  reality), `ThresholdResult` missing `acc` — both force `any` erasure in the
  pages; three architecture-type fields (see Tier 3).
- **A11Y**: exactly ONE violation repo-wide — `experiments/[id]/+page.svelte:3265`
  `font-size: 0.8rem`. Everything else ≥1rem. 
- Misc: `currentFlow` store hijacked by any FlowStarted broadcast; hardcoded LM
  baseline 10.5801 in shared stores; `alert()` for gating errors; Svelte 4 +
  Vite 5 on maintenance (deliberate debt; keep Vite patched — `host: 0.0.0.0`
  dev exposure is LAN-deliberate, do not widen).

---

## 5. Proposed execution plan (pending approval — no fixes applied)

| Phase | Contents | Est. |
|-------|----------|------|
| P1 — contract + holes | 1.1, 1.2 (client fixes + a contract test hitting a live dashboard), 1.3, 1.4 (path validation), 1.5 (migration), 1.6 (validate both ends) | 3-4h, high confidence |
| P2 — frontend actives | 1.7, 1.8 (error-banner pattern via new lib/api.ts), 1.9 (shared flowTemplates), 1.10 (+statusColors dedup) | 3-4h |
| P3 — state machine | 2.1-2.7: transition validator shared by flows+experiments, transactions on all multi-statement helpers, conditional cancel, FK-safe + cache-preserving resume, wire graceful requeue (dashboard-side task; worker requeues instead of fails), guard worker exception path | 6-8h, MEDIUM risk — needs the regression flow from the deploy plan |
| P4 — boundary hardening | 2.8 retry stack, 404 handling, WAL+busy_timeout pool options, limit clamps, WS shared-poll + index | 3-4h |
| P5 — races + UX correctness | Tier 3 frontend items (request-token helper, double-submit guards, architecture_type derivation, leaderboard keys, date fixes) | 4-5h |
| P6 — structure | Tier 4: delete dead code, dedup libs, god-file splits per seam maps (mechanical, D3 discipline) | 8-10h |

Sequencing note: P1/P2 are safe pre-deploy (pure additions/client fixes);
P3 should land AFTER the accelerator deploy + regression flow so state-machine
changes are validated against live traffic, not alongside the ABI swap.

## 6. Execution log

- [x] P1 contract + holes — branch `dashboard-p1p2` (d9a9b7fb + a24ca5b6): client
  create_flow top-level experiments + /experiments/link route; restart safe_name
  sanitized; checkpoint file_path validated create/download/delete; create_flow
  400s on empty name/experiments (allow_empty_experiments escape hatch);
  last_heartbeat migration. VERIFIED on a fresh-DB test instance end-to-end
  (incl. '..' canary + real DashboardClient round-trip).
- [x] P2 frontend actives — `dashboard-p1p2` (75d7be79): E1 genome-type swap,
  error-reset + dismissible actionError banners (22 sites), dead edit feature
  removed, duplicateFlow type/phase forwarding, --text-tertiary + statusColors
  dedup, 0.8rem→1rem, zero-phase submit guard. svelte-check 0/0, build OK.
- [x] P3 state machine — branch `dashboard-p3-p6` (ff1f9c60): transition
  validator (ANY→ANY → 409, TOCTOU-safe), conditional cancel + completed_at,
  gen-0 wipe window closed (iteration-rows guard + tx), FK-safe resume that
  PRESERVES the validation cache, create_flow fully transactional, typed
  experiment status (422 on typos), stale flows RE-QUEUED not failed
  (dashboard 180s task + worker fallback). Verified live on a fresh instance.
  DEFERRED: worker _handle_flow_exception guard → post-arch-review-merge
  (D3.2 conflict zone).
- [x] P4 boundary hardening — (6e11e68b): client retries idempotent-only +
  split timeouts + 404 raises on non-GET; WAL/busy_timeout/FK pool; limit
  clamps; ONE shared WS snapshot poller (+ iterations(exp,created_at) index).
- [x] P5 races + UX — (c56776cf): request-token guards everywhere, flow-id
  change reloads, 8-button double-submit guard, hash+threshold_mode
  leaderboard keying, dmy/24h default + UTC-naive timestamp parse, types.ts
  best_f1/best_fpr + acc, lib/{api,format,ids}.ts. svelte-check 0/0 + build.
- [x] P6 structure — (8c03afc6 + 591eb824 + ad321c3a + c56776cf): dead
  watcher/parser deleted; architecture_type PATCH wired end-to-end; db/mod.rs
  → 12 files + api/mod.rs → 11 files (queries shim, zero path changes;
  post-split functional battery re-run green); .gitignore bare-'db' trap
  fixed (was silently untracking the split). (was: REMAINING — now DONE)
- [x] P6c god-pages (d52c788a): experiments/[id] 3426→873 (12 components),
  flows/[id] 3034→1021 (10), flows/new 2165→980 (8 + flowTemplates.ts).
  svelte-check 0/0 + build per page. ALSO DONE: worker
  _handle_flow_exception guard (afbcb9de, on worktree-arch-review-tier1 —
  the D3.2-refactored home of that method). **REVIEW FULLY EXECUTED — zero
  open items.**
