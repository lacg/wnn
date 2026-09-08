#!/usr/bin/env python3
"""Apply the tie-label fix to scripts/queue_after_ab_chain.sh — SAFELY.

The chain's verdict block labels a per-seed comparison with

    w = "arm" if a[4] < c[4] else "control"

so an EXACT TIE (a[4] == c[4]) is silently credited to the control. The TALLY
below it is already correct — `wa += a<c; wc += c<a` scores a tie for neither —
so only the per-row label lies. The same defect was fixed in
scripts/mutstep_ab_chain.sh, where it had credited a genuine seed-31337003 tie
to arm B.

This is a separate applier because bash resumes a running script at a BYTE
OFFSET: editing scripts/queue_after_ab_chain.sh while pid N is executing it
shifts every byte after the edit and corrupts the rest of the run. So this
script REFUSES while the chain is live. Run it once the chain has exited, then
delete this file — it is a one-shot.
"""
import re
import subprocess
import sys

TARGET = "scripts/queue_after_ab_chain.sh"
OLD = '\t\tw = "arm" if a[4] < c[4] else "control"; wa += a[4] < c[4]; wc += c[4] < a[4]\n'
NEW = (
	'\t\t# A TIE is neither side\'s win — the tally already refuses to score it, so\n'
	'\t\t# the label must refuse too. An `else` here credited every tie to control.\n'
	'\t\tw = "arm" if a[4] < c[4] else ("control" if c[4] < a[4] else "TIE")\n'
	'\t\twa += a[4] < c[4]; wc += c[4] < a[4]\n'
)


def chain_is_running() -> bool:
	r = subprocess.run(["pgrep", "-f", TARGET], capture_output=True, text=True)
	return bool(r.stdout.strip())


def main() -> int:
	src = open(TARGET).read()
	if "else \"TIE\")" in src:
		print(f"already patched — {TARGET} labels ties correctly; delete this applier")
		return 0
	if chain_is_running():
		print(f"REFUSING: bash is still executing {TARGET}.")
		print("Editing it now would shift its byte offsets and corrupt the rest of the run.")
		print("Wait for the chain to exit, then re-run this script.")
		return 1
	if src.count(OLD) != 1:
		print(f"REFUSING: expected exactly 1 occurrence of the defect, found {src.count(OLD)}")
		return 1
	open(TARGET, "w").write(src.replace(OLD, NEW))
	syn = subprocess.run(["bash", "-n", TARGET], capture_output=True, text=True)
	if syn.returncode != 0:
		print("SYNTAX CHECK FAILED — restore from git and investigate:\n" + syn.stderr)
		return 1
	print(f"patched {TARGET}; bash -n clean. Delete this applier.")
	return 0


if __name__ == "__main__":
	sys.exit(main())
