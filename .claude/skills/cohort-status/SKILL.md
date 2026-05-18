---
name: cohort-status
description: Quick one-screen status check for the active OI cohort — completed/queued/running counts, ETA, HSR breakdown. Use when the user asks "how's the cohort going", "what's running", "status check", or wants a fast progress snapshot without regenerating the full report.
---

# Cohort status

One-screen quick check on the active OI-v2 cohort. Fast (~1 second), no file writes.

## Steps

1. Run the status script. By default it auto-detects the active cohort:
   ```bash
   python3 scripts/cohort_status.py
   ```
2. To target a specific cohort:
   ```bash
   python3 scripts/cohort_status.py --cohort PREFIX
   ```
3. Adjust target count if needed (default 112):
   ```bash
   python3 scripts/cohort_status.py --target 30
   ```
4. Show the user the full output verbatim — it's already concise.

## Brief insights to add after the table

- If `running` is 0 and `queued` > 0, the worker may be down — suggest checking
  `pgrep -af wnn.ram.experiments.worker`.
- If ETA shifts significantly between checks (e.g., >2h), suggest checking
  individual flow durations — could be a single slow flow distorting the average.
- HSR breakdown: highlight which HSR values still have 0 completed (the
  experimental coverage gap).

## When to use this vs `/cohort-report`

- `/cohort-status`: a quick "is everything running, what's the ETA, which HSR is
  hot right now" check.
- `/cohort-report`: full 5-table regen, delta tables, best-genome mining,
  paper-prep numbers. Run after `cohort-status` if the numbers look interesting.
