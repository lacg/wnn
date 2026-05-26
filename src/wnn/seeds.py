"""Reproducible 3-way (train/test/val) + multi-seed management.

A seed must be (a) arbitrary/unbiased — not cherry-picked to flatter a result — and
(b) recorded, so a run reproduces and the values can go straight into a paper. This
module makes that the *default*: you don't pass seeds, you get a UTC-date base
(YYYYMMDD) and three independent partition seeds derived from it; everything is
logged and persisted to the `seed_runs` table. Passing a base (or explicit
train/test/val) overrides, for exact replication.

Partitions (same role as IDS 80/10/10, but the controller generates ICs on the fly,
so each partition is a disjoint *seed* rather than a disjoint row-set):
  - train: fit / evolve against these initial conditions
  - test : model selection / early-stopping (unseen during fit)
  - val  : held-out final report (touched once)

Multi-seed: `run_index` spawns an independent seed set from the same base; run the
whole protocol for run_index 0..N-1 and report mean±std over the val results. Share
the SAME base across substrates (MLP / WNN / PID) so the comparison is controlled.

Usage:
    from wnn.seeds import resolve_seed_set, log_seed_set, record_seed_set
    s = resolve_seed_set(base=args.base_seed, run_index=i)   # base=None → today's date
    log_seed_set(s)
    record_seed_set(s, script="run_mlp_ga", extra={"gens": gens})
    # ... use s.train / s.test / s.val ...
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SeedSet:
    base: int           # master seed (date-derived YYYYMMDD, or explicit)
    run_index: int      # which multi-seed run (0-based)
    train: int          # fit / evolve
    test: int           # model selection / early-stop
    val: int            # held-out final report
    source: str         # "date" (auto) | "explicit"

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (f"[{self.source}] base={self.base} run={self.run_index} | "
                f"train={self.train} test={self.test} val={self.val}")


def default_base_seed() -> int:
    """UTC date as YYYYMMDD — arbitrary, unbiased, self-documenting, monotonic.

    Commit to it before seeing results (do NOT re-roll the date to chase a number —
    that is cherry-picking in disguise)."""
    return int(datetime.now(timezone.utc).strftime("%Y%m%d"))


def derive_seeds(base: int, run_index: int = 0) -> tuple[int, int, int]:
    """Three statistically-independent uint32 seeds (train, test, val), reproducible
    from (base, run_index) via numpy SeedSequence (the recommended way to spawn
    uncorrelated streams — safer than base+1/base+2)."""
    ss = np.random.SeedSequence([int(base), int(run_index)])
    train, test, val = (int(x) for x in ss.generate_state(3, dtype=np.uint32))
    return train, test, val


def resolve_seed_set(base: Optional[int] = None, run_index: int = 0, *,
                     train: Optional[int] = None, test: Optional[int] = None,
                     val: Optional[int] = None) -> SeedSet:
    """Resolve a 3-way seed set. base=None → today's UTC date (built-in; no flag needed).
    Explicit train/test/val override the derived values for exact replication."""
    source = "explicit" if base is not None else "date"
    if base is None:
        base = default_base_seed()
    d_train, d_test, d_val = derive_seeds(base, run_index)
    return SeedSet(
        base=int(base), run_index=int(run_index),
        train=int(train) if train is not None else d_train,
        test=int(test) if test is not None else d_test,
        val=int(val) if val is not None else d_val,
        source=source,
    )


def log_seed_set(s: SeedSet) -> None:
    """Surface the seed set (logger + stdout) so it lands in run logs and can be
    copied into a paper."""
    logger.info("seeds %s", s.summary())
    print(f"[seeds] {s.summary()}")


def _default_db_path() -> str:
    env = os.environ.get("WNN_DB_PATH")
    if env:
        return env
    # src/wnn/seeds.py → project root is three dirnames up, then db/wnn.db
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(root, "db", "wnn.db")


def record_seed_set(s: SeedSet, *, script: str, db_path: Optional[str] = None,
                    extra: Optional[dict] = None) -> Optional[int]:
    """Persist the seed set to the `seed_runs` table (created if missing). Returns the
    row id, or None on failure — recording is best-effort and never raises (the DB may
    be busy with a live worker; a failed record must not break a comparison run)."""
    path = db_path or _default_db_path()
    try:
        conn = sqlite3.connect(path, timeout=30.0)
        try:
            conn.execute("""CREATE TABLE IF NOT EXISTS seed_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                script TEXT NOT NULL,
                source TEXT NOT NULL,
                base INTEGER NOT NULL,
                run_index INTEGER NOT NULL,
                train_seed INTEGER NOT NULL,
                test_seed INTEGER NOT NULL,
                val_seed INTEGER NOT NULL,
                extra_json TEXT
            )""")
            cur = conn.execute(
                "INSERT INTO seed_runs (created_at, script, source, base, run_index, "
                "train_seed, test_seed, val_seed, extra_json) VALUES (?,?,?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), script, s.source, s.base,
                 s.run_index, s.train, s.test, s.val,
                 json.dumps(extra) if extra is not None else None),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()
    except Exception as e:  # best-effort: never break the run over a DB hiccup
        logger.warning("record_seed_set failed (non-fatal): %s", e)
        print(f"[seeds] DB record skipped (non-fatal): {e}")
        return None


__all__ = [
    "SeedSet", "default_base_seed", "derive_seeds", "resolve_seed_set",
    "log_seed_set", "record_seed_set",
]
