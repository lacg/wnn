"""Budget-aware, type-balanced flow admission policy for the worker scheduler.

Pure logic (no I/O, no DB, no subprocess) so it is fully unit-testable. The worker
imports `admit()` to decide which queued flows to launch each poll.

Policy (locked 06/06/2026, see .claude/plans/controller_dashboard_integration.md):
  * BUDGET = detected CPU cores − reserve (dynamic; ~13 on a 16-core M4 Max).
  * Each flow declares a core need via the `wnn_num_threads` param (default by
    architecture_type: ids=10, controller=3).
  * Admit concurrently while sum(running cores) + need ≤ budget.
  * TYPE-BALANCE: admit the UNDER-represented type first (argmin running count) so
    the running mix stays balanced ±1 by type; when one type is absent the other
    fills the remaining budget (e.g. IDS gone → 4×controller@3).
  * Within a type: OLDEST id first (FIFO). Ties across types break on the oldest
    queued id (global FIFO on ties).
"""
from __future__ import annotations

from typing import Any

# Per-architecture default core need when `wnn_num_threads` is unset on the flow.
DEFAULT_CORES_BY_TYPE: dict[str, int] = {"controller": 3, "ids": 10}
_FALLBACK_CORES = 10
_DEFAULT_TOTAL_CORES = 16
_DEFAULT_RESERVE = 3


def detect_budget(reserve: int = _DEFAULT_RESERVE, cpu_cores: int | None = None) -> int:
    """Dynamic budget = cpu_cores − reserve (min 1). Reads the real core count from
    the Rust accelerator at runtime unless `cpu_cores` is passed (for tests)."""
    if cpu_cores is None:
        try:
            import ram_accelerator
            cpu_cores = int(ram_accelerator.cpu_cores())
        except Exception:
            cpu_cores = _DEFAULT_TOTAL_CORES
    return max(1, cpu_cores - max(0, reserve))


def _params(flow: dict[str, Any]) -> dict[str, Any]:
    return (flow.get("config") or {}).get("params") or {}


def flow_type(flow: dict[str, Any]) -> str:
    """Architecture type of a flow ('ids', 'controller', ...)."""
    return _params(flow).get("architecture_type", "ids")


def flow_cores(flow: dict[str, Any]) -> int:
    """Core need: `wnn_num_threads` param, else the per-type default."""
    p = _params(flow)
    default = DEFAULT_CORES_BY_TYPE.get(p.get("architecture_type", "ids"), _FALLBACK_CORES)
    raw = p.get("wnn_num_threads", default)
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return default


def admit(queued: list[dict[str, Any]],
          running: list[dict[str, Any]],
          budget: int) -> list[dict[str, Any]]:
    """Select flows to launch this poll, honoring the budget + type-balance policy.

    queued:  list of flow dicts (as returned by the dashboard API).
    running: list of {"id", "type", "cores"} for currently-running flows.
    budget:  total core budget.
    Returns the flows to admit (subset of `queued`), in admission order.
    """
    remaining = budget - sum(int(r["cores"]) for r in running)
    type_counts: dict[str, int] = {}
    for r in running:
        type_counts[r["type"]] = type_counts.get(r["type"], 0) + 1
    blocked_ids = {r["id"] for r in running}

    admitted: list[dict[str, Any]] = []
    while True:
        # Group still-admissible queued flows (fit the remaining budget) by type.
        candidates: dict[str, list[dict[str, Any]]] = {}
        for f in queued:
            if f["id"] in blocked_ids:
                continue
            if flow_cores(f) <= remaining:
                candidates.setdefault(flow_type(f), []).append(f)
        if not candidates:
            break

        # Under-represented type first; tie-break on the oldest queued id in that type.
        best_type = min(
            candidates,
            key=lambda t: (type_counts.get(t, 0), min(f["id"] for f in candidates[t])),
        )
        chosen = min(candidates[best_type], key=lambda f: f["id"])  # oldest id (FIFO)

        admitted.append(chosen)
        blocked_ids.add(chosen["id"])
        remaining -= flow_cores(chosen)
        type_counts[best_type] = type_counts.get(best_type, 0) + 1

    return admitted
