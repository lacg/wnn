#!/usr/bin/env python
"""Pick where a watchdog-killed phased_ga run resumes, from its --save-stage-checkpoints dir.

Prints `<path>\\t<mode>` (mode = same | next), or nothing when there is no resume
point (killed during the grid, before any file) — the caller then restarts.

WHY (13/10/2026). controller_arm_lib.sh used `ls -t emergency_stage*.yaml.gz | head -1`.
Emergency dumps are NEVER deleted when their stage finishes, and a stage writes its
first dump only after SaveCadence's first interval (up to --checkpoint-max-interval
gens). So a kill early in stage k — or while stage k-1 was writing its final
checkpoint / scoring its held-out — left stage k-1's STALE mid-stage dump as the
newest emergency file, and the retry re-ran that FINISHED stage from an old
generation, overwriting its final checkpoint.

RULE
  * A final `stage<n>_<name>.yaml.gz` counts only when it is a FINISHED stage
    (wnn.control.stage_completion: `stage_complete: true`, or a legacy final whose
    metrics are not the cancelled-evaluation signature). An interrupted "final" is
    ignored entirely (15/10/2026, _tw515 s3).
  * An emergency dump of stage n is STALE when a FINISHED final of stage n
    exists and is at least as new (that stage finished after the dump; the final
    save always follows the last in-stage dump). Stale dumps are IGNORED, not deleted
    (see below).
  * Of the valid dumps, the highest stage wins -> resume THAT stage (`same`).
  * If the highest FINAL checkpoint is a later stage than every valid dump (or there
    is no valid dump), stage k-1 finished and stage k never dumped -> resume from
    that final checkpoint with `next` (phased_ga starts at k-1+1; MEMORY's final ->
    publish-only: held-out + stage-select + winner save, no GA).
  * Stage number decides, never mtime ordering across stages.

WHY IGNORE, NOT DELETE. The rule is stateless, so it is also correct for every
checkpoint dir already on disk (the _tw20 dir still holds emergency_stage1_neurons,
older than its stage1_neurons final). Deleting would put a destructive write in
phased_ga at the moment the box is fullest, and would throw away the only
mid-stage snapshot if the final checkpoint later proved unreadable. Disk cost of
keeping them: one ~0.4 GB file per finished stage per run.

Usage: python scripts/pick_resume_point.py <stage-checkpoint-dir>
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# The checkpoint-completion rule lives ONCE, in the package (phased_ga's resume
# path applies the same one). This script runs from the shell lib without
# PYTHONPATH, so put this checkout's src/ first.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from wnn.control.stage_completion import checkpoint_is_complete_final  # noqa: E402

_FINAL = re.compile(r"^stage(\d+)_([a-z]+)\.yaml\.gz$")
_DUMP = re.compile(r"^emergency_stage(\d+)_([a-z]+)\.yaml\.gz$")


def pick(stagedir: "str | Path", is_complete=checkpoint_is_complete_final,
         log=None) -> "tuple[str, str] | None":
	"""See the module docstring. `is_complete(path) -> (bool, reason)` decides whether
	a stage-final checkpoint is a FINISHED stage (injectable for tests)."""
	d = Path(stagedir)
	if not d.is_dir():
		return None
	finals: dict = {}
	dumps: dict = {}
	for p in d.iterdir():
		m = _FINAL.match(p.name)
		if m:
			finals[int(m.group(1))] = p
			continue
		m = _DUMP.match(p.name)
		if m:
			dumps[int(m.group(1))] = (m.group(2), p)
	# 15/10/2026: a final only counts if it IS a finished stage — a SIGTERM used to
	# be written up as one (stage4_memory 31 s after its emergency dump, metrics
	# all zero). An unfinished "final" neither resumes (`next`) nor makes its
	# stage's dump stale.
	for n in sorted(finals):
		ok, why = is_complete(finals[n])
		if not ok:
			if log is not None:
				log(f"ignoring {finals[n].name}: {why}")
			del finals[n]
	valid = {}
	for n, (name, p) in dumps.items():
		final = finals.get(n)
		if final is not None and final.name == f"stage{n}_{name}.yaml.gz" \
				and final.stat().st_mtime_ns >= p.stat().st_mtime_ns:
			continue                     # stage n finished after this dump — stale
		valid[n] = p
	best_dump = max(valid) if valid else None
	best_final = max(finals) if finals else None
	if best_dump is not None and (best_final is None or best_dump >= best_final):
		return str(valid[best_dump]), "same"
	if best_final is not None:
		return str(finals[best_final]), "next"
	return None


def main() -> int:
	if len(sys.argv) != 2:
		print(__doc__, file=sys.stderr)
		return 2
	r = pick(sys.argv[1], log=lambda m: print(f"[pick_resume_point] {m}", file=sys.stderr))
	if r is not None:
		print(f"{r[0]}\t{r[1]}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
