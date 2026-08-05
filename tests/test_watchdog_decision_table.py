"""R14c — decision-table tests for scripts/controller_mem_watchdog.sh (v5).

Runs the REAL script unmodified against a mocked environment. Every input the
watchdog reads arrives through an external command (`vm_stat`, `sysctl`, `ps`),
and every action it takes goes through `kill`/`pkill`, so shadowing those on
PATH makes the whole `while true` loop a deterministic function of a scripted
tick sequence. `sleep 15` is called exactly once per iteration, so the mock uses
it as the tick clock and terminates the run when the script runs out.

Why no source edits: the watchdog is a long-running detached process and bash
reads scripts lazily by byte offset — editing the file under a live run is
undefined behavior. Testing it black-box keeps the campaign safe.

Decisions covered (v5 cascade, top to bottom):
  HARD floor      avail < HARD                       -> KILL, ignores CTRL_MIN_RSS
  pressure+SOFT   avail < SOFT & pressure & RSS>MIN  -> KILL      (R5 gate)
                  ... same but RSS<=MIN              -> ride out  (R5)
  thrash>=2       RSS > MIN                          -> PAUSE     (R5 gate)
                  ... RSS <= MIN                     -> ride out  (R5)
  HOG             avail < SOFT & RSS > HOG           -> KILL
                  healthy avail & RSS > HOG          -> nothing (avail-gated)
  CLIMB           ctrl_ticks >= 3 & avail<SOFT       -> KILL
                  first tick (ctrl_ticks < 3)        -> nothing   (R6 warmup)
  selector                                            -> python, never a grep (R7)
  kill_ctrl                                           -> signals wrapper too (R1)

Run:  python3 tests/test_watchdog_decision_table.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WATCHDOG = REPO / "scripts" / "controller_mem_watchdog.sh"

PAGE = 16384
GB = 1073741824

# Defaults the tests drive against (same order as the script's positional args).
HARD, SOFT, HOG, CLIMB, PAUSE_DEEP, PAUSE_TICKS, SWAP_MB, COMP_GB, CTRL_MIN = \
	6, 10, 28, 15, 7.5, 3, 200, 0.8, 4


# ---------------------------------------------------------------------------
# Mock environment
# ---------------------------------------------------------------------------

_VM_STAT = r'''#!/bin/bash
t=$(cat "$WD_TICK")
line=$(awk -v n="$t" 'NR==n+1{print; found=1} END{if(!found) print prev}' "$WD_TICKS" 2>/dev/null)
[ -z "$line" ] && line=$(tail -1 "$WD_TICKS")
avail=$(echo "$line" | awk '{print $1}')
comp=$(echo "$line" | awk '{print $2}')
# Split AVAIL across the four buckets the script sums; compressor separate.
pages=$(echo "$avail" | awk -v g=%(GB)d -v p=%(PAGE)d '{printf "%%d",$1*g/p}')
cpages=$(echo "$comp" | awk -v g=%(GB)d -v p=%(PAGE)d '{printf "%%d",$1*g/p}')
echo "Mach Virtual Memory Statistics: (page size of 16384 bytes)"
echo "Pages free:                                   $pages."
echo "Pages inactive:                               0."
echo "Pages speculative:                            0."
echo "Pages purgeable:                              0."
# The script reads $5 of this line -- keep the real vm_stat field layout
# (Pages=1 occupied=2 by=3 compressor:=4 <count>=5) or comp_gb silently reads 0
# and the pressure signal never fires.
echo "Pages occupied by compressor:            $cpages."
'''

_SYSCTL = r'''#!/bin/bash
t=$(cat "$WD_TICK")
line=$(awk -v n="$t" 'NR==n+1{print; found=1}' "$WD_TICKS" 2>/dev/null)
[ -z "$line" ] && line=$(tail -1 "$WD_TICKS")
sw=$(echo "$line" | awk '{print $3}')
echo "vm.swapusage: total = 3072.00M  used = ${sw}.00M  free = 100.00M  (encrypted)"
'''

# ps mock: serves the four query shapes the watchdog uses.
_PS = r'''#!/bin/bash
case "$*" in
	*"-axo pid=,command="*)
		cat "$WD_PROCS"
		;;
	*"-o rss="*)
		pid=$(echo "$*" | awk '{print $NF}')
		# A per-pid RSS SERIES (one GB value per tick) takes precedence, so a
		# scenario can make a controller grow over time; otherwise a flat value.
		if [ -f "$WD_SERIES/$pid" ]; then
			t=$(cat "$WD_TICK")
			awk -v n="$t" '{v=$1} NR==n+1{printf "%d",$1*1073741824/1024; f=1; exit}
			               END{if(!f) printf "%d",v*1073741824/1024}' "$WD_SERIES/$pid"
		else
			grep "^${pid} " "$WD_RSS" 2>/dev/null | awk '{print $2}'
		fi
		;;
	*"-o ppid="*)
		pid=$(echo "$*" | awk '{print $NF}')
		# reaped -> no such process -> empty, exactly as on a real box
		if [ -f "$WD_REAPED" ] && grep -qx "$pid" "$WD_REAPED"; then exit 0; fi
		grep "^${pid} " "$WD_PPID" 2>/dev/null | awk '{print $2}'
		;;
	*"-o command="*)
		pid=$(echo "$*" | awk '{print $NF}')
		grep -E "^${pid}[[:space:]]" "$WD_PROCS" 2>/dev/null | cut -d' ' -f2-
		;;
esac
exit 0
'''

# kill mock: records signals. `kill -0` reports the target alive unless the
# scenario declared it dies during a graceful pause.
_KILL = r'''#!/bin/bash
sig="$1"; pid="$2"
if [ "$sig" = "-0" ]; then
	[ -f "$WD_DEAD" ] && grep -qx "$pid" "$WD_DEAD" && exit 1
	[ -f "$WD_REAPED" ] && grep -qx "$pid" "$WD_REAPED" && exit 1
	exit 0
fi
echo "kill ${sig} ${pid}" >> "$WD_ACTIONS"
# A SIGKILLed child is REAPED by its /usr/bin/time parent, after which its ppid
# is unreachable. Modelling that is what makes the R1 test discriminate: a
# post-kill `ps -o ppid=` must come back EMPTY, so a watchdog that looks the
# wrapper up after killing leaves it unsignalled (-> rc=1 -> cell abandoned).
[ "$sig" = "-9" ] && echo "$pid" >> "$WD_REAPED"
exit 0
'''

_PKILL = r'''#!/bin/bash
echo "pkill $*" >> "$WD_ACTIONS"
exit 0
'''

# sleep mock: `sleep 15` is the tick clock (one per loop iteration). Everything
# else (90s settle, 1s pause poll) returns instantly.
_SLEEP = r'''#!/bin/bash
if [ "$1" = "15" ]; then
	t=$(cat "$WD_TICK"); t=$((t + 1)); echo "$t" > "$WD_TICK"
	if [ "$t" -ge "$WD_MAX_TICKS" ]; then
		/bin/kill -TERM $PPID 2>/dev/null
		exit 0
	fi
fi
exit 0
'''


class Sandbox:
	"""A mocked /usr/bin for one watchdog scenario."""

	def __init__(self, tmp: Path):
		self.tmp = tmp
		self.bin = tmp / "bin"
		self.bin.mkdir(parents=True, exist_ok=True)
		self.ticks = tmp / "ticks"
		self.tick = tmp / "tick"
		self.procs = tmp / "procs"
		self.rss = tmp / "rss"
		self.ppid = tmp / "ppid"
		self.actions = tmp / "actions"
		self.dead = tmp / "dead"
		self.series = tmp / "series"
		self.series.mkdir(exist_ok=True)
		self.reaped = tmp / "reaped"
		self.reaped.write_text("")
		self.tick.write_text("0\n")
		self.actions.write_text("")
		self.dead.write_text("")
		# `kill` is a bash BUILTIN, so PATH shadowing alone cannot intercept it.
		# bash reads $BASH_ENV before running a non-interactive script, which lets
		# us disable the builtin and fall through to the mock — still no edit to
		# the watchdog source.
		self.bashenv = tmp / "bashenv.sh"
		self.bashenv.write_text("enable -n kill 2>/dev/null || true\n")
		self._write(_VM_STAT % {"GB": GB, "PAGE": PAGE}, "vm_stat")
		self._write(_SYSCTL, "sysctl")
		self._write(_PS, "ps")
		self._write(_KILL, "kill")
		self._write(_PKILL, "pkill")
		self._write(_SLEEP, "sleep")

	def _write(self, body: str, name: str) -> None:
		p = self.bin / name
		p.write_text(body)
		p.chmod(0o755)

	def set_ticks(self, rows: list[tuple[float, float, int]]) -> None:
		"""Each row is (avail_gb, compressor_gb, swap_used_mb) for one tick."""
		self.ticks.write_text("\n".join(f"{a} {c} {s}" for a, c, s in rows) + "\n")

	def set_procs(self, procs: list[tuple[int, str]],
	              rss_gb: dict[int, float] | None = None,
	              ppids: dict[int, int] | None = None) -> None:
		self.procs.write_text("".join(f"{pid} {cmd}\n" for pid, cmd in procs))
		self.rss.write_text("".join(
			f"{pid} {int(g * GB / 1024)}\n" for pid, g in (rss_gb or {}).items()))
		self.ppid.write_text("".join(f"{p} {q}\n" for p, q in (ppids or {}).items()))

	def set_rss_series(self, pid: int, values_gb: list[float]) -> None:
		"""Make one pid's RSS vary per tick — needed to exercise CLIMB, which is
		a 2-tick delta and cannot be provoked with a flat value."""
		(self.series / str(pid)).write_text("\n".join(str(v) for v in values_gb) + "\n")

	def run(self, max_ticks: int = 6, timeout: float = 25.0) -> str:
		env = dict(os.environ)
		env["PATH"] = f"{self.bin}:{env['PATH']}"
		env.update(WD_TICKS=str(self.ticks), WD_TICK=str(self.tick),
		           WD_PROCS=str(self.procs), WD_RSS=str(self.rss),
		           WD_PPID=str(self.ppid), WD_ACTIONS=str(self.actions),
		           WD_DEAD=str(self.dead), WD_MAX_TICKS=str(max_ticks),
		           WD_SERIES=str(self.series), WD_REAPED=str(self.reaped),
		           BASH_ENV=str(self.bashenv))
		p = subprocess.run(
			["bash", str(WATCHDOG), str(HARD), str(SOFT), str(HOG), str(CLIMB),
			 str(PAUSE_DEEP), str(PAUSE_TICKS), str(SWAP_MB), str(COMP_GB), str(CTRL_MIN)],
			cwd=str(REPO), env=env, capture_output=True, text=True, timeout=timeout)
		return p.stdout

	def signals(self) -> str:
		return self.actions.read_text()


def _sandbox(tmp: Path, ctrl_rss_gb: float, extra_procs=(), ctrl_pid=5000,
             wrapper_pid=4999) -> Sandbox:
	"""Standard layout: driver -> /usr/bin/time wrapper -> python controller."""
	sb = Sandbox(tmp)
	procs = list(extra_procs) + [
		(wrapper_pid, "/usr/bin/time -l /venv/bin/python -u -m wnn.control.phased_ga --levels 16"),
		(ctrl_pid, "/opt/homebrew/bin/python3.13 -u -m wnn.control.phased_ga --levels 16"),
	]
	procs.sort(key=lambda t: t[0])
	sb.set_procs(procs, rss_gb={ctrl_pid: ctrl_rss_gb, wrapper_pid: 0.001},
	             ppids={ctrl_pid: wrapper_pid})
	return sb


# ---------------------------------------------------------------------------
# HARD survival floor
# ---------------------------------------------------------------------------

def test_hard_floor_spares_a_tiny_controller():
	"""v6 POLICY (user decision, 31/07/2026) — the INVERSE of the v5 expectation this
	test used to encode. Measured basis: 8/8 v5 HARD-floor kills freed nothing (ctrl
	RSS 0.0-0.3 GB; the IDS worker is the hog), so a sub-threshold controller is NEVER
	SIGKILLed for external pressure at ANY avail level. It alarms and rides out;
	killing it would abandon the run without restoring the floor."""
	with tempfile.TemporaryDirectory() as d:
		sb = _sandbox(Path(d), ctrl_rss_gb=0.2)
		sb.set_ticks([(5.0, 1.0, 100)] * 6)
		out = sb.run()
		assert "CRITICAL HARD floor" in out, f"expected the CRITICAL alarm, got:\n{out}"
		assert "NOT killing (futile)" in out, f"expected the futility ride-out, got:\n{out}"
		assert "kill -9 5000" not in sb.signals(), (
			"v6 must NOT kill a sub-threshold controller for external pressure")
	print("✓ hard_floor_spares_a_tiny_controller")


def test_hard_floor_kill_also_signals_the_wrapper():
	"""R1. The /usr/bin/time wrapper must be captured BEFORE the kill and
	signalled too, so the driver sees rc=137 and its retry fires. A post-kill
	lookup returns empty -> wrapper unsignalled -> rc=1 -> cell ABANDONED."""
	with tempfile.TemporaryDirectory() as d:
		# v6: the HARD kill fires only when the controller is ABOVE CTRL_MIN (it can
		# actually restore the floor). 8 GB > 4 GB keeps this on the kill path.
		sb = _sandbox(Path(d), ctrl_rss_gb=8.0)
		sb.set_ticks([(5.0, 1.0, 100)] * 6)
		sb.run()
		sig = sb.signals()
		assert "kill -9 5000" in sig, "python must be killed"
		assert "kill -9 4999" in sig, (
			f"/usr/bin/time wrapper must ALSO be SIGKILLed (R1) — signals were:\n{sig}")
	print("✓ hard_floor_kill_also_signals_the_wrapper")


# ---------------------------------------------------------------------------
# R5 — CTRL_MIN_RSS gating on EXTERNAL pressure
# ---------------------------------------------------------------------------

def test_soft_pressure_rides_out_a_tiny_controller():
	"""R5. A 0.2GB controller cannot be the cause of a multi-GB IDS spike;
	killing it frees nothing and abandons the cell. Ride out instead."""
	with tempfile.TemporaryDirectory() as d:
		sb = _sandbox(Path(d), ctrl_rss_gb=0.2)
		# avail 8 (< SOFT 10, > HARD 6) with a rising compressor => pressure=1
		sb.set_ticks([(8.0, 1.0, 100), (8.0, 3.0, 100), (8.0, 5.0, 100),
		              (8.0, 7.0, 100), (8.0, 9.0, 100), (8.0, 11.0, 100)])
		out = sb.run()
		assert "riding out" in out, f"expected ride-out for a tiny controller, got:\n{out}"
		assert "REAL exhaustion" not in out, f"must NOT kill a tiny controller:\n{out}"
		assert "kill -9" not in sb.signals(), "no SIGKILL may be issued"
	print("✓ soft_pressure_rides_out_a_tiny_controller")


def test_soft_pressure_kills_a_big_controller():
	"""Same pressure, but a 10GB controller IS big enough to be the cause."""
	with tempfile.TemporaryDirectory() as d:
		sb = _sandbox(Path(d), ctrl_rss_gb=10.0)
		sb.set_ticks([(8.0, 1.0, 100), (8.0, 3.0, 100), (8.0, 5.0, 100),
		              (8.0, 7.0, 100), (8.0, 9.0, 100), (8.0, 11.0, 100)])
		out = sb.run()
		assert "REAL exhaustion" in out, f"expected a kill for a big controller, got:\n{out}"
		assert "kill -9 5000" in sb.signals()
	print("✓ soft_pressure_kills_a_big_controller")


def test_swap_thrash_rides_out_a_tiny_controller():
	"""R5, thrash branch. Sustained swap growth with a tiny controller is the
	IDS worker's doing — pausing the controller frees nothing."""
	with tempfile.TemporaryDirectory() as d:
		sb = _sandbox(Path(d), ctrl_rss_gb=0.2)
		# healthy avail so only the thrash branch can fire; swap +500MB/tick
		sb.set_ticks([(30.0, 1.0, 100), (30.0, 1.0, 600), (30.0, 1.0, 1100),
		              (30.0, 1.0, 1600), (30.0, 1.0, 2100), (30.0, 1.0, 2600)])
		out = sb.run()
		assert "SWAP THRASH" in out, f"thrash branch must be reached, got:\n{out}"
		assert "riding out" in out, f"tiny controller must ride out the thrash:\n{out}"
		# "PAUSE" alone would match the armed banner ("external PAUSE/KILL only
		# when ctrl RSS>...") — match the action line instead.
		assert "graceful PAUSE" not in out, f"must NOT pause a tiny controller:\n{out}"
		assert sb.signals() == "", f"no signal may be sent, got:\n{sb.signals()}"
	print("✓ swap_thrash_rides_out_a_tiny_controller")


def test_swap_thrash_pauses_a_big_controller():
	with tempfile.TemporaryDirectory() as d:
		sb = _sandbox(Path(d), ctrl_rss_gb=10.0)
		sb.dead.write_text("5000\n")     # the controller dumps and exits on SIGTERM
		sb.set_ticks([(30.0, 1.0, 100), (30.0, 1.0, 600), (30.0, 1.0, 1100),
		              (30.0, 1.0, 1600), (30.0, 1.0, 2100), (30.0, 1.0, 2600)])
		out = sb.run()
		assert "graceful PAUSE" in out, f"big controller must be paused, got:\n{out}"
		assert "kill -TERM 5000" in sb.signals(), "PAUSE must SIGTERM the python"
	print("✓ swap_thrash_pauses_a_big_controller")


# ---------------------------------------------------------------------------
# HOG / CLIMB runaway backstops
# ---------------------------------------------------------------------------

def test_hog_kills_only_when_available_is_also_low():
	with tempfile.TemporaryDirectory() as d:
		sb = _sandbox(Path(d), ctrl_rss_gb=30.0)     # > HOG 28
		sb.set_ticks([(8.0, 1.0, 100)] * 6)          # avail < SOFT, no pressure delta
		out = sb.run()
		assert "RUNAWAY" in out, f"expected a HOG kill, got:\n{out}"
	print("✓ hog_kills_only_when_available_is_also_low")


def test_hog_is_ignored_while_available_is_healthy():
	"""A controller may legitimately allocate large into room the box has —
	TERNARY's re-eval held RSS 31GB with 26GB available and was false-killed by
	the ungated guard (15/07)."""
	with tempfile.TemporaryDirectory() as d:
		sb = _sandbox(Path(d), ctrl_rss_gb=30.0)     # > HOG, but plenty of room
		sb.set_ticks([(35.0, 1.0, 100)] * 6)
		out = sb.run()
		assert "RUNAWAY" not in out, f"must not kill a large controller on a healthy box:\n{out}"
		assert "kill -9" not in sb.signals(), "no kill may be issued"
	print("✓ hog_is_ignored_while_available_is_healthy")


def test_first_tick_climb_does_not_false_kill():
	"""R6. climb = rss - prev2, and prev2 is the 0 sentinel until the controller
	has been observed for >=3 ticks — so on tick 1 climb == full RSS and a legit
	large controller was killed as 'CLIMBING' (QA#6). The warmup blocks that."""
	with tempfile.TemporaryDirectory() as d:
		sb = _sandbox(Path(d), ctrl_rss_gb=20.0)     # < HOG 28, but climb would read 20 > 15
		sb.set_ticks([(8.0, 1.0, 100)] * 2)          # avail < SOFT; only 2 ticks
		out = sb.run(max_ticks=2)
		assert "CLIMBING" not in out, (
			f"first-tick climb must not fire before the 3-tick warmup (R6):\n{out}")
		assert "kill -9" not in sb.signals(), "no kill may be issued during warmup"
	print("✓ first_tick_climb_does_not_false_kill")


def test_sustained_climb_kills_after_warmup():
	"""The warmup must not disable the guard — a real climb still gets caught
	once prev2 holds a genuine reading."""
	with tempfile.TemporaryDirectory() as d:
		sb = _sandbox(Path(d), ctrl_rss_gb=1.0)
		sb.set_ticks([(8.0, 1.0, 100)] * 8)          # avail < SOFT, no pressure/thrash
		# Stay under HOG (28) throughout so ONLY the climb branch can fire.
		# climb = rss - prev2; by tick 3 that is 20 - 1 = 19 > CLIMB 15.
		sb.set_rss_series(5000, [1.0, 1.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0])
		out = sb.run(max_ticks=6)
		assert "CLIMBING" in out, f"a real post-warmup climb must be caught:\n{out}"
		assert "kill -9 5000" in sb.signals(), "climb kill must target the python"
	print("✓ sustained_climb_kills_after_warmup")


# ---------------------------------------------------------------------------
# R7 — selector must find the python, never a decoy
# ---------------------------------------------------------------------------

def test_selector_ignores_a_lower_pid_grep_decoy():
	"""R7. A user/agent `grep wnn.control.phased_ga` with a LOWER pid used to be
	selected over the real controller (bare pgrep -f + head -1). Every RSS read
	then measured the decoy, so runaway detection went blind."""
	with tempfile.TemporaryDirectory() as d:
		decoys = [(100, "grep wnn.control.phased_ga"),
		          (200, "tail -f /logs/controller/dfa1l/dfa_9feat_QUAD.out wnn.control.phased_ga")]
		# v6: 8 GB > CTRL_MIN so the HARD kill actually fires and names the pid.
		sb = _sandbox(Path(d), ctrl_rss_gb=8.0, extra_procs=decoys)
		sb.set_ticks([(5.0, 1.0, 100)] * 4)          # HARD floor -> forces a kill, naming the pid
		sb.run(max_ticks=4)
		sig = sb.signals()
		assert "kill -9 5000" in sig, f"the real python must be selected, signals:\n{sig}"
		assert "kill -9 100" not in sig, f"the grep decoy must never be selected:\n{sig}"
		assert "kill -9 200" not in sig, f"the tail decoy must never be selected:\n{sig}"
	print("✓ selector_ignores_a_lower_pid_grep_decoy")


def test_selector_ignores_the_time_wrapper():
	"""The wrapper carries the same argv; selecting it made every kill orphan the
	python at PPID=1 (the 23/07 double-run incidents)."""
	with tempfile.TemporaryDirectory() as d:
		sb = _sandbox(Path(d), ctrl_rss_gb=8.0)      # wrapper pid 4999 < python 5000; >CTRL_MIN so v6 kills
		sb.set_ticks([(5.0, 1.0, 100)] * 4)
		sb.run(max_ticks=4)
		sig = sb.signals()
		# the wrapper IS signalled, but as the captured parent — never as the target
		assert "kill -9 5000" in sig, f"python must be the kill target:\n{sig}"
		first_kill = [l for l in sig.splitlines() if l.startswith("kill -9")][0]
		assert first_kill == "kill -9 5000", (
			f"the python must be killed FIRST as the selected target, got {first_kill}")
	print("✓ selector_ignores_the_time_wrapper")


# ---------------------------------------------------------------------------
# No controller present
# ---------------------------------------------------------------------------

def test_no_controller_is_a_no_op():
	"""Between cells there is no phased_ga at all — the watchdog must idle
	quietly and never touch the IDS worker."""
	with tempfile.TemporaryDirectory() as d:
		sb = Sandbox(Path(d))
		sb.set_procs([(300, "/opt/homebrew/bin/python3.13 -m wnn.ram.experiments.worker")],
		             rss_gb={300: 20.0})
		sb.set_ticks([(4.0, 9.0, 5000)] * 5)         # brutal: below HARD, thrashing
		out = sb.run(max_ticks=5)
		assert sb.signals() == "", (
			f"no controller -> no signals whatsoever, got:\n{sb.signals()}")
		assert "worker" not in out.lower() or "kill" not in out.lower(), \
			"the IDS worker must never be a target"
	print("✓ no_controller_is_a_no_op")


if __name__ == "__main__":
	if not WATCHDOG.exists():
		sys.exit(f"watchdog script not found at {WATCHDOG}")
	if shutil.which("bc") is None:
		sys.exit("bc is required by the watchdog script")
	test_hard_floor_kills_even_a_tiny_controller()
	test_hard_floor_kill_also_signals_the_wrapper()
	test_soft_pressure_rides_out_a_tiny_controller()
	test_soft_pressure_kills_a_big_controller()
	test_swap_thrash_rides_out_a_tiny_controller()
	test_swap_thrash_pauses_a_big_controller()
	test_hog_kills_only_when_available_is_also_low()
	test_hog_is_ignored_while_available_is_healthy()
	test_first_tick_climb_does_not_false_kill()
	test_sustained_climb_kills_after_warmup()
	test_selector_ignores_a_lower_pid_grep_decoy()
	test_selector_ignores_the_time_wrapper()
	test_no_controller_is_a_no_op()
	print("\nAll watchdog decision-table tests passed.")
