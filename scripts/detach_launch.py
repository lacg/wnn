#!/usr/bin/env python3
"""Launch a command fully detached (own session, PPID=1) so it survives the
parent shell / Claude Code /exit. macOS has no `setsid`, so we use Python's
start_new_session=True (which calls setsid(2)) + redirect IO to a log file.

Usage:
  detach_launch.py <logfile> <cwd> -- <command> [args...]
"""
import os
import sys
import subprocess


def main():
	argv = sys.argv[1:]
	if "--" not in argv or len(argv) < 4:
		print("usage: detach_launch.py <logfile> <cwd> -- <command> [args...]", file=sys.stderr)
		sys.exit(2)
	sep = argv.index("--")
	logfile, cwd = argv[0], argv[1]
	cmd = argv[sep + 1:]
	if not cmd:
		print("no command given after --", file=sys.stderr)
		sys.exit(2)

	logf = open(logfile, "ab", buffering=0)
	p = subprocess.Popen(
		cmd,
		cwd=cwd,
		stdout=logf,
		stderr=subprocess.STDOUT,
		stdin=subprocess.DEVNULL,
		start_new_session=True,   # setsid: detach from our process group/session
		close_fds=True,
		env=os.environ.copy(),
	)
	print(p.pid)


if __name__ == "__main__":
	main()
