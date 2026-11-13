#!/usr/bin/env python
"""Rebuild a flown run's EXACT phased_ga argv from its .out `[argv]` line (CTRL-48, 12/11/2026).

  rebuild_argv.py <run.out> recalc <ckpt_dir> [extra flags...]
      --recalc-headline <ckpt_dir> + extra flags; the run's own --save-* flags are dropped
      (a recalc trains and writes nothing; the save paths never affect scoring).
  rebuild_argv.py <run.out> fly <ckpt_dir> <winner_path> [extra flags...]
      a NEW run on the same recipe (block 7: the block-6 dc control + one stack): the save
      paths are replaced by the new tag's, extra flags appended.

The only other transformation is the 11/11 terminology cut-over rename
--report-seeds/--report-seed/--report-episodes -> --test-* (same values, same meaning).
Prints the argv as one shell-quoted line. Refuses an .out without exactly one [argv] line,
and an extra flag the original argv already carries (no silent last-wins override)."""
import shlex
import sys

RENAME = {"--report-seeds": "--test-seeds", "--report-seed": "--test-seed",
          "--report-episodes": "--test-episodes"}
SAVE_FLAGS = ("--save-stage-checkpoints", "--save-winner")
# recalc only (13/11/2026, S5p substitute test): the run's OWN stage-select flags are dropped so the re-pick rule's
# flags are the only selection setting. Runs flown with rule P (block 8) carry them; the 30 block-4/5 runs carry none,
# so for those this is a no-op (verified 13/11: 30/30 argv lines have neither flag).
SELECT_FLAGS = ("--val-units", "--select-tail")


def argv_tokens(path):
	with open(path, "rb") as f:
		lines = [l.decode("utf-8", "replace").rstrip("\n") for l in f if l.startswith(b"[argv] ")]
	if len(lines) != 1:
		raise SystemExit(f"{path}: expected exactly one [argv] line, found {len(lines)}")
	return lines[0][len("[argv] "):].split()


def strip_saves(toks, drop=SAVE_FLAGS):
	out, skip = [], False
	for t in toks:
		if skip:
			skip = False
		elif t in drop:
			skip = True
		else:
			out.append(RENAME.get(t, t))
	return out


def refuse_overlap(base, extra):
	clash = sorted({t for t in extra if t.startswith("--")} & set(base))
	if clash:
		raise SystemExit(f"original argv already carries {clash} — refusing a silent override")


def main(a):
	recalc = a[1] == "recalc"
	base = strip_saves(argv_tokens(a[0]), SAVE_FLAGS + SELECT_FLAGS if recalc else SAVE_FLAGS)
	if recalc:
		head, extra = ["--recalc-headline", a[2]], a[3:]
	elif a[1] == "fly":
		head, extra = ["--save-stage-checkpoints", a[2], "--save-winner", a[3]], a[4:]
	else:
		raise SystemExit("mode must be recalc or fly")
	refuse_overlap(base, extra + head)
	print(" ".join(shlex.quote(t) for t in base + extra + head))


if __name__ == "__main__":
	main(sys.argv[1:])
