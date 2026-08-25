"""Hand-written blocks for docs/ids_results.md, and the guard that protects it.

NO SCRIPT PRODUCES docs/ids_results.md END-TO-END (verified 24/08/2026 — nothing in
scripts/ emits its "canonical source of truth" header). It is hand-assembled from
several generators' stdout. The prose sections that no generator can produce live in
docs/ids_results_preserved/ so a regeneration cannot silently drop them.

POSITION IS THE FILENAME PREFIX = section number x 10:
    00-09   HEAD  — emitted BEFORE the generated sections   (e.g. 00 = section 0)
    10-59         — reserved for GENERATED sections 1-5; no files live here
    60-99   TAIL  — emitted AFTER the generated sections    (60/70/80 = sections 6/7/8)

Only the (future) end-to-end assembler should compose these. A generator that emits a
DIFFERENT report must not append them — doing so injects one document's sections into
another. That was a real bug here on 24/08/2026, caught the same day.
"""

import sys
from pathlib import Path

PRESERVED_DIR = Path(__file__).resolve().parent.parent / "docs" / "ids_results_preserved"
CANONICAL_DOC = "ids_results.md"
HEAD_MAX = 10   # prefixes below this render before the generated sections
TAIL_MIN = 60   # prefixes at or above this render after them


def _blocks():
	if not PRESERVED_DIR.is_dir():
		return []
	out = []
	for q in sorted(PRESERVED_DIR.glob("[0-9]*.md")):
		try:
			n = int(q.name[:2])
		except ValueError:
			print(f"WARNING: {q.name} has no 2-digit position prefix — SKIPPED", file=sys.stderr)
			continue
		out.append((n, q))
	return out


def head_blocks() -> list[str]:
	"""Preserved text that precedes the generated sections."""
	return [q.read_text().rstrip("\n") for n, q in _blocks() if n < HEAD_MAX]


def tail_blocks() -> list[str]:
	"""Preserved text that follows the generated sections."""
	return [q.read_text().rstrip("\n") for n, q in _blocks() if n >= TAIL_MIN]


def orphaned_blocks() -> list[str]:
	"""Files sitting in the GENERATED range — a misfiled block, loud on purpose."""
	return [q.name for n, q in _blocks() if HEAD_MAX <= n < TAIL_MIN]


def guard_canonical_target(out_path: str, force: bool, produces_canonical: bool = False) -> None:
	"""Refuse to replace the canonical paper doc with a report that is not it.

	docs/ids_results.md is a hand-assembled ~11,400-line document. A generator that
	emits something else would destroy all of it, not merely the tail, because these
	tools write with Path.write_text(). Only the end-to-end assembler may pass
	produces_canonical=True.
	"""
	if produces_canonical or force or not out_path:
		return
	if Path(out_path).name == CANONICAL_DOC:
		sys.exit(
			f"REFUSING to write {out_path}: this script does not produce the canonical\n"
			f"IDS results doc — it emits a different report and would replace all of it,\n"
			f"including the generated sections. Write elsewhere, or pass\n"
			f"--force-overwrite-canonical if you genuinely mean to replace the paper's\n"
			f"source of truth."
		)
