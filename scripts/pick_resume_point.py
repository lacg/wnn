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
  * An emergency dump of stage n is STALE when the final `stage<n>_<name>.yaml.gz`
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

import re
import sys
from pathlib import Path

_FINAL = re.compile(r"^stage(\d+)_([a-z]+)\.yaml\.gz$")
_DUMP = re.compile(r"^emergency_stage(\d+)_([a-z]+)\.yaml\.gz$")


def pick(stagedir: "str | Path") -> "tuple[str, str] | None":
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
	valid = {}
	for n, (name, p) in dumps.items():
		final = d / f"stage{n}_{name}.yaml.gz"
		if final.exists() and final.stat().st_mtime_ns >= p.stat().st_mtime_ns:
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
	r = pick(sys.argv[1])
	if r is not None:
		print(f"{r[0]}\t{r[1]}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
