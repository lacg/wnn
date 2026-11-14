"""docs/ctrl_gap_after_block9.md §B: hold-window decomposition of ctrl_gap_student_trace.py
outputs (VAL units). Pure arithmetic on recorded samples; block-6 conventions
(/tmp/steady_block6/ana.py): e_true / e_est are roll, pitch Euler errors against the tilt
reference, b = e_est − e_true is the Mahony bias, abp the student's predicted bias.
Usage: python scripts/ctrl_gap_student_ana.py /tmp/ctrl_gap/both_s1.npz ..."""
import sys
import numpy as np

DEG = np.degrees


def episode_split(v, key):
	"""Share of mean-square carried by the per-episode mean (the frozen offset)."""
	_, inv = np.unique(key, return_inverse=True)
	m = np.bincount(inv, v) / np.bincount(inv)
	return float((m[inv] ** 2).mean() / (v ** 2).mean()), float(DEG(np.abs(m).mean()))


def one(path):
	z = np.load(path)
	R, E, S = z["rows"].astype(np.float64), z["eps"], z["scorer"]
	key = R[:, 0] * 1000 + R[:, 1]
	err, et, yaw, ee, abp, dz = R[:, 3], R[:, 4:6], R[:, 6], R[:, 7:9], R[:, 17:19], R[:, 16]
	b = ee - et
	seen = ee - abp                       # what the student can know of its true error
	mag = lambda v, y: DEG(np.sqrt((v ** 2).sum(1) + y ** 2).mean())
	out = {
		"val scorer stable% / err / steady": (S[:, 1].mean() * 100, S[:, 2].mean(), S[:, 3].mean()),
		"pos3d mean / fifths": (E[:, 5].mean(), *np.nanmean(E[:, 6:11], 0)),
		"mean |dx| |dy| |dz|": tuple(np.nanmean(E[:, 11:14], 0)),
		"hold |dx| |dy| |dz|": tuple(np.nanmean(E[:, 14:17], 0)),
		"steady (replay)": DEG(err.mean()),
		"small-angle steady": mag(et, yaw),
		"roll/pitch only (yaw=0)": mag(et, 0 * yaw),
		"yaw only": DEG(np.abs(yaw).mean()),
		"|e_true| r/p": tuple(DEG(np.abs(et).mean(0))),
		"|e_est| r/p": tuple(DEG(np.abs(ee).mean(0))),
		"|bias b| r/p": tuple(DEG(np.abs(b).mean(0))),
		"|b - abp| r/p (bias the student cannot see)": tuple(DEG(np.abs(b - abp).mean(0))),
		"abp R2 r/p": tuple(np.corrcoef(abp[:, a], b[:, a])[0, 1] ** 2 for a in range(2)),
		"|e_est - abp| r/p (student-visible error)": tuple(DEG(np.abs(seen).mean(0))),
		"corr(visible error, true error) r/p": tuple(np.corrcoef(seen[:, a], et[:, a])[0, 1] for a in range(2)),
		"if visible error were zeroed: steady": mag(et - seen, yaw),
		"if yaw were zeroed too": mag(et - seen, 0 * yaw),
	}
	_, inv = np.unique(key, return_inverse=True)
	off = lambda v: (np.bincount(inv, v) / np.bincount(inv))[inv]
	dm = np.c_[et[:, 0] - off(et[:, 0]), et[:, 1] - off(et[:, 1])]
	out["if every per-episode offset were removed (integral action): steady"] = mag(dm, yaw - off(yaw))
	out["  ... roll/pitch offsets only"] = mag(dm, yaw)
	out["  ... yaw offset only"] = mag(et, yaw - off(yaw))
	unit_of = R[:, 0].astype(int)
	for name, v in (("roll", et[:, 0]), ("pitch", et[:, 1]), ("yaw", yaw)):
		o = off(v)
		um = (np.bincount(unit_of, o) / np.bincount(unit_of))[unit_of]
		out[f"{name}: share of the offset's MS that is per-UNIT (trim)"] = float((um ** 2).mean() / (o ** 2).mean())
	for name, v in (("roll", et[:, 0]), ("pitch", et[:, 1]), ("yaw", yaw)):
		share, off = episode_split(v, key)
		out[f"{name}: share of MS in the episode-mean offset / mean |offset| deg"] = (share, off)
		_, inv = np.unique(key, return_inverse=True)
		m = np.bincount(inv, v) / np.bincount(inv)
		out[f"{name}: within-episode sd deg"] = float(DEG(np.sqrt(((v - m[inv]) ** 2).mean())))
	return out


for p in sys.argv[1:]:
	print(p)
	for k, v in one(p).items():
		print("   ", k, "=", np.round(v, 3).tolist() if hasattr(v, "__len__") else round(float(v), 3))
