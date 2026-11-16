"""SMOKE (not a result): arm C 10b s1 recipe + --obs-yaw-from-att-est at a minimal budget. Budget
flags are REPLACED in place (every occurrence); test seeds are non-report (7777xxxx)."""
import shlex, sys
src, tag, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
EXTRA = dict(kv.split("=", 1) for kv in sys.argv[4:])  # flag=value overrides (replaced in place, else appended)
toks = [l for l in open(src, errors="replace").read().splitlines() if l.startswith("[argv] ")][0][7:].split()
SET = {"--pop": "4", "--memory-gens": "1", "--neurons-gens": "1", "--conns-gens": "1", "--eval-episodes": "12",
       "--memory-eval-episodes": "12", "--test-episodes": "8", "--calib-episodes": "20", "--val-units": "2",
       "--holdout-pop-sample": "2", "--steps": "1000", "--check-interval": "1"}
SET.update({k: v for k, v in EXTRA.items() if k in toks})
APPEND = [x for k, v in EXTRA.items() if k not in toks for x in (k, v)]
out, i = [], 0
while i < len(toks):
	t = toks[i]
	if t in SET:
		out += [t, SET[t]]; i += 2; continue
	if t == "--test-seeds":
		i += 1
		while i < len(toks) and not toks[i].startswith("--"): i += 1
		out += ["--test-seeds", "77770001", "77770002"]; continue
	if t in ("--save-stage-checkpoints", "--save-winner"):
		i += 2; continue
	out.append(t); i += 1
out += APPEND + ["--obs-yaw-from-att-est", "--save-stage-checkpoints", f"{outdir}/ckpt_{tag}", "--save-winner", f"{outdir}/{tag}_winner.yaml.gz"]
print(shlex.join(out))
