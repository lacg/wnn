"""CTRL-83 row B (eq-info MPCOF = THE comparator, docs/ctrl_rival_row_ruling.md §1) on one cell.

  rowb.py <cell> <out.json>

Row B (MPCOF>rate est-fed + attitude-bias compensation) exists only in the cargo probe
(rival_probe_tests::test_units_rival_rows_rescore_f32_lineage). This writes that test's
fixture for the STRESSED config (same builder as scripts/ctrl_indi_val_fixture.py --units
test: classical_baseline's own samplers, nothing re-derived), runs the test, and parses its
[rescore] rows A / B / C. Gates: the test's self_check (this build == the installed wheel
on unit 1 of THIS fixture), and row A here == score_cell.py's mpcofA row of the same cell
when that cache exists (exit 2 on mismatch). Read-only on src/: cargo builds the test binary
into the shared target dir, nothing is edited.
"""
import json, os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/Users/lacg/wnn/scripts")
import numpy as np
import cells as C

# The probe is built from THIS checkout's sources (repo-relative), so a branch's harness never scores row B with
# another checkout's Rust (cell `all`: an ABI-68 probe would ignore the fixture's train_motor_lag_range and fly
# the fixed lag). rival_probe_tests.self_check then ties this build to the installed wheel.
ACC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "src", "wnn", "ram", "strategies", "accelerator"))
PY = "/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
TEST = "rival_probe_tests::test_units_rival_rows_rescore_f32_lineage"
RX = re.compile(r"\[rescore\] row (\w) NEW\(f32\): stable ([\d.]+) err ([\d.]+) steady ([\d.]+) pos ([\d.]+) fails (\d+)/(\d+)")


def write_fixture(cell, out_dir):
	import ctrl_indi_val_fixture as F
	from score_cell import load_run
	from wnn.control._accel import score_classical_baseline
	_, _, a, ec0 = load_run("wnn_s1")
	ec = C.apply(cell, ec0)
	os.makedirs(out_dir, exist_ok=True)
	for i, unit in enumerate(int(u) for u in a.test_seeds):
		f = F._with_defaults(F._fields(a, ec, unit), score_classical_baseline)
		lines = [F._line("unit_seed", unit)] + [F._line(k, v) for k, v in f.items()]
		if i == 0:
			call = {k: v for k, v in f.items() if k not in ("init_qs", "init_omegas", "steps", "stable_deg")}
			r = score_classical_baseline(F.MPCOF, f["init_qs"], f["init_omegas"], f["steps"], f["stable_deg"], **call)
			lines.append(F._line("expect_est_row", [r[0], r[1], r[2], r[4]]))
		open(os.path.join(out_dir, f"unit_{unit}.txt"), "w").write("\n".join(lines) + "\n")


def main():
	cell, out = sys.argv[1], sys.argv[2]
	fix = os.path.join(HERE, "data", "fixtures", cell)
	write_fixture(cell, fix)
	env = {**os.environ, "PYO3_PYTHON": PY, "WNN_RIVAL_RESCORE_FIXTURE": fix,
	       "RAYON_NUM_THREADS": os.environ.get("RAYON_NUM_THREADS", "6")}
	cmd = ["nice", "-n", "15", "cargo", "test", "--release", "-p", "ram_controller", "--lib", "--no-default-features",
	       TEST, "--", "--ignored", "--nocapture", "--test-threads", "1"]
	p = subprocess.run(cmd, cwd=ACC, env=env, capture_output=True, text=True)
	log = p.stdout + p.stderr
	open(out + ".log", "w").write(log)
	if p.returncode != 0 or "[self-check]" not in log:
		print(f"CTRL83 GATE FAIL: row B cargo probe failed (rc {p.returncode}); see {out}.log", file=sys.stderr)
		sys.exit(2)
	rows = {m.group(1): dict(stable=float(m.group(2)), err=float(m.group(3)), steady=float(m.group(4)),
	                         pos=float(m.group(5)), fails=int(m.group(6)), episodes=int(m.group(7))) for m in RX.finditer(log)}
	if sorted(rows) != ["A", "B", "C"]:
		print(f"CTRL83 GATE FAIL: expected rows A/B/C, got {sorted(rows)}", file=sys.stderr)
		sys.exit(2)
	a_npz = os.path.join(HERE, "data", f"mpcofA__{cell}.npz")
	if os.path.exists(a_npz):
		U = np.load(a_npz)["units"]
		if abs(U[:, 2].mean() - rows["A"]["err"]) > 0.0006 or int(U[:, 7].sum()) != rows["A"]["fails"]:
			print(f"CTRL83 GATE FAIL: probe row A {rows['A']} != score_cell mpcofA err {U[:, 2].mean():.4f} fails {int(U[:, 7].sum())}",
			      file=sys.stderr)
			sys.exit(2)
	json.dump(dict(cell=cell, rows=rows, abi_wheel=int(__import__("ram_controller").ABI_VERSION)), open(out, "w"), indent=1)
	b = rows["B"]
	print(f"[ctrl83] mpcofB   {cell:7s} stable {b['stable']:.2f} err {b['err']:.3f} steady {b['steady']:.3f} pos {b['pos']:.4f} "
	      f"fails {b['fails']}/{b['episodes']} (cargo probe; self-check ok)", flush=True)


if __name__ == "__main__":
	main()
