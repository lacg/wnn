"""Run provenance for controller markers (multi-axis programme spec, rule R9).

Every banked marker has to say WHICH CODE produced it: the controller wheel
(name-version), its ABI, a hash of the compiled extension actually loaded, the
git HEAD of the Python tree, and the fitness-pool scheme the search ran under.
Without these, "the four anchor markers predate the 08/09 label-fix wheel" is a
sentence in a memory note instead of a field a script can check.

The chain of custody is .out -> grep -> marker JSON (scripts/controller_arm_lib.sh),
so this module prints ONE greppable line, `[provenance] key=value ...`, next to the
existing `fitness_pools=` header. Every accessor is fail-safe: a run must never
die at startup because git is absent or the wheel lacks a metadata record — a
missing value prints as `unknown`, which is itself a finding, not a crash.
"""

import hashlib
import importlib.metadata
import os
import subprocess
from dataclasses import dataclass


PROVENANCE_PREFIX = "[provenance]"
UNKNOWN = "unknown"
_SHA_PREFIX_HEX = 16


@dataclass(frozen=True)
class ControllerProvenance:
	"""What produced a run. Frozen so a marker cannot be re-attributed after the fact."""

	wheel: str          # "ram_controller-2026.212.37" (dist name-version) or "unknown"
	abi: int            # ABI_VERSION of the loaded extension; 0 when unavailable
	wheel_sha256: str   # first 16 hex of sha256 over the compiled .so actually imported
	git: str            # short HEAD of the Python tree, "+dirty" when the tree has edits
	fitness_pools: str  # the same label the "Pop=..." header prints
	# ABI 35 item I: the RESOLVED plant motor lag "T s:source" (e.g. "0.2s:airframe:grafe2026").
	# Printed BEFORE fitness_pools, which must stay last (it may contain spaces).
	motor_lag: str = UNKNOWN
	# ABI 36: what the --obs-pwm feature reported — "delivered" (motor lag on: the
	# student's u_a estimate − anchor), "command" (lag off: the accumulator −
	# anchor), "off". Printed BEFORE fitness_pools.
	obs_pwm: str = UNKNOWN
	# ABI 38 (Luiz 06/11/2026): the TRAINING plant's motor asymmetry —
	# "per_episode:<mag>" (±mag drawn every training episode: a lineage break for
	# every run with motor_asym_mag > 0, e.g. L4C), "fixed" or "off".
	# Printed BEFORE fitness_pools.
	train_asym: str = UNKNOWN
	# ABI 38 (CTRL-27): the controller ARCHITECTURE — "cascade:teacher=<t>:inner=
	# pidfw_rate@<Hz>:outer_hz=<Hz>:range=<T,p,q,r>:gamma=<T,p,q,r>" (the WNN outer
	# loop over the firmware rate PID, lineage _ct38) or "motor:teacher=<t>" (the
	# WNN emits motor pwm). Printed BEFORE fitness_pools.
	arch: str = UNKNOWN
	# ABI 42 (CTRL-39): what the reference recorders fed the student's applied-actuator
	# inputs — "flown" (the command the plant received; wheel RECORDER_INPUTS) or
	# "student" (a pre-fix wheel: motor-level obs_dhat / obs_pwm MEMORY universes were
	# recorded on the student's own command). Printed BEFORE fitness_pools.
	recorder: str = UNKNOWN
	# ABI 44 (08/11/2026): how output neurogenesis lays out the output layer —
	# "segment" (every bank / E-I half resizes at its own tail: the bank-growth fix)
	# or "flat" (a pre-44 wheel: the flat-tail resize that remapped banks). Printed
	# BEFORE fitness_pools.
	neuro: str = UNKNOWN
	# CTRL-45: the axis receptive-field setting — "off" or "axis:cross=<c>".
	rf: str = UNKNOWN
	# CTRL-48: the thermometer calibration "eps:N,regime:R,ladder:L" and the excluded
	# features "off" | "constant" | "names:a+b" | "constant+names:a+b".
	calib: str = UNKNOWN
	drop: str = UNKNOWN
	# CTRL-46 (ABI 46): the DAgger write rule ("last" | "vote:<θ>"), the label attitude
	# ("true" | "estimate") and the Mahony-bias predictor block ("off" | "on").
	write: str = UNKNOWN
	label: str = UNKNOWN
	abp: str = UNKNOWN

	def line(self) -> str:
		"""The single greppable line the ladder copies into the marker."""
		return (f"{PROVENANCE_PREFIX} wheel={self.wheel} abi={self.abi} "
		        f"wheel_sha256={self.wheel_sha256} git={self.git} "
		        f"motor_lag={self.motor_lag} "
		        f"obs_pwm={self.obs_pwm} "
		        f"train_asym={self.train_asym} "
		        f"arch={self.arch} "
		        f"recorder={self.recorder} "
		        f"neuro={self.neuro} "
		        f"rf={self.rf} "
		        f"calib={self.calib} "
		        f"drop={self.drop} "
		        f"write={self.write} "
		        f"label={self.label} "
		        f"abp={self.abp} "
		        f"fitness_pools={self.fitness_pools}")


def obs_pwm_convention(obs_pwm: bool, motor_lag_s: float) -> str:
	"""ABI 36 rule, mirrored from WnnController::obs_pwm_delivered: under motor lag
	the obs_pwm feature IS the delivered-actuator estimate; lag off it is the command."""
	if not obs_pwm:
		return "off"
	return "delivered" if float(motor_lag_s) > 0.0 else "command"


def arch_token(ctbr: "dict | None", teacher: str, action_repeat: int, dt: float) -> str:
	"""ABI 38: the provenance arch= value (see ControllerProvenance.arch)."""
	if ctbr is None:
		return f"motor:teacher={teacher}"
	fmt = lambda xs: ",".join(f"{float(x):g}" for x in xs)
	outer_hz = 1.0 / (max(int(action_repeat), 1) * float(dt))
	tok = (f"cascade:teacher={teacher}:inner=pidfw_rate@{float(ctbr['attitude_hz']):g}Hz:"
	       f"outer_hz={outer_hz:g}:range={fmt(ctbr['range'])}:gamma={fmt(ctbr['gamma'])}")
	# CTRL-38: the cascade input blocks, named only when on (ABI-40 tokens unchanged).
	if ctbr.get("obs_rate_i"):
		tok += ":obs=rate_i"
	if ctbr.get("obs_dhat_plant") is not None:
		tok += ":obs=dhat"
	if ctbr.get("obs_dhat_residual_plant") is not None:   # CTRL-48
		tok += ":obs=dhat_resid"
	return tok


def collect_provenance(fitness_pools: str, motor_lag: str = UNKNOWN, obs_pwm: str = UNKNOWN,
                       train_asym: str = UNKNOWN, arch: str = UNKNOWN,
                       rf: str = UNKNOWN, calib: str = UNKNOWN,
                       drop: str = UNKNOWN, write: str = UNKNOWN, label: str = UNKNOWN,
                       abp: str = UNKNOWN) -> ControllerProvenance:
	"""Gather every field, never raising: each accessor degrades to UNKNOWN / 0."""
	wheel, abi, sha = _wheel_identity()
	return ControllerProvenance(
		wheel=wheel,
		abi=abi,
		wheel_sha256=sha,
		git=_git_head(os.path.dirname(os.path.abspath(__file__))),
		fitness_pools=fitness_pools,
		motor_lag=motor_lag.replace(" ", "_").replace('"', ""),
		obs_pwm=obs_pwm.replace(" ", "_").replace('"', ""),
		train_asym=train_asym.replace(" ", "_").replace('"', ""),
		arch=arch.replace(" ", "_").replace('"', ""),
		recorder=_recorder_inputs(),
		neuro=_neuro_layout(),
		rf=rf.replace(" ", "_").replace('"', ""),
		calib=calib.replace(" ", "_").replace('"', ""),
		drop=drop.replace(" ", "_").replace('"', ""),
		write=write.replace(" ", "_").replace('"', ""),
		label=label.replace(" ", "_").replace('"', ""),
		abp=abp.replace(" ", "_").replace('"', ""),
	)


def _recorder_inputs() -> str:
	"""CTRL-39: "flown" on a fixed wheel, "student" on a pre-ABI-42 one, never raises."""
	try:
		from wnn.control import _accel
		return str(getattr(_accel.require_accel(), "RECORDER_INPUTS", "student"))
	except Exception:
		return UNKNOWN


def _neuro_layout() -> str:
	"""ABI 44: "segment" on a bank-aware wheel, "flat" on a pre-44 one, never raises."""
	try:
		from wnn.control import _accel
		return str(getattr(_accel.require_accel(), "NEURO_LAYOUT", "flat"))
	except Exception:
		return UNKNOWN


def _wheel_identity() -> tuple[str, int, str]:
	"""(dist name-version, ABI, .so hash) of the controller extension in use.

	Goes through the facade so the transition fallback (combined ram_accelerator
	wheel) is reported as what it is, not as a ram_controller build."""
	try:
		from wnn.control import _accel
		mod = _accel.require_accel()
	except Exception:
		return UNKNOWN, 0, UNKNOWN
	name = getattr(mod, "__name__", UNKNOWN).split(".")[0]
	abi = int(getattr(mod, "ABI_VERSION", 0) or 0)
	return _dist_label(name), abi, _module_sha256(mod)


def _dist_label(dist_name: str) -> str:
	try:
		return f"{dist_name}-{importlib.metadata.version(dist_name)}"
	except Exception:
		return f"{dist_name}-{UNKNOWN}"


def _module_sha256(mod) -> str:
	"""Hash of the compiled extension file. `mod.__file__` may be the package
	__init__.py (maturin layout) — hash the .so beside it in that case."""
	try:
		path = getattr(mod, "__file__", None)
		if path is None:
			return UNKNOWN
		if path.endswith("__init__.py"):
			path = _first_extension_in(os.path.dirname(path))
		if path is None:
			return UNKNOWN
		with open(path, "rb") as fh:
			return hashlib.sha256(fh.read()).hexdigest()[:_SHA_PREFIX_HEX]
	except Exception:
		return UNKNOWN


def _first_extension_in(directory: str) -> str | None:
	for entry in sorted(os.listdir(directory)):
		if entry.endswith(".so") or entry.endswith(".pyd"):
			return os.path.join(directory, entry)
	return None


def _git_head(source_dir: str) -> str:
	"""Short HEAD of the tree the Python was imported from, '+dirty' if edited.
	A 2 s timeout so a hung git (network FS, lock) cannot stall a run."""
	try:
		head = _git(source_dir, "rev-parse", "--short", "HEAD")
		if not head:
			return UNKNOWN
		dirty = _git(source_dir, "status", "--porcelain", "--untracked-files=no")
		return head + ("+dirty" if dirty else "")
	except Exception:
		return UNKNOWN


def _git(cwd: str, *args: str) -> str:
	result = subprocess.run(
		["git", *args], cwd=cwd, capture_output=True, text=True, timeout=2.0, check=False,
	)
	return result.stdout.strip() if result.returncode == 0 else ""
