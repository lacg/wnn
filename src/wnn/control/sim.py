"""Attitude simulator — re-export from the Rust accelerator.

The implementation lives in
`src/wnn/ram/strategies/accelerator/controller.rs`. This module exists
so Python callers can write `from wnn.control.sim import AttitudeSim`
without reaching into the accelerator package directly.

Per the architecture decision in `project_drone_controller_paper1.md`:
the hot path (per-step physics) is in Rust; Python orchestrates the
outer GA + episode loop.
"""

from __future__ import annotations

from ram_accelerator import AttitudeSim

__all__ = ["AttitudeSim"]
