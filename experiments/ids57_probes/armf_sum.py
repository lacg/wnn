"""Summarise armf_gate.py output: VAL differences vs the 2-cluster 34-bit baseline, paired by draw."""
import json, sys
import numpy as np
rows = [json.loads(l) for l in open(sys.argv[1]) if l.startswith("{")]
var = {r["name"]: r for r in rows if r["part"] == "variant"}
val = {}
for r in rows:
	if r["part"] == "val": val.setdefault(r["name"], {})[r["seed"]] = r
base = val["a_b34"]; a1 = val.get("b_A1_b64")
def col(name, f): return np.array([f(val[name][s]) for s in sorted(val[name])])
def d(name, f, ref=None): return np.array([f(val[name][s]) - (ref or f)(base[s]) for s in sorted(val[name])])
F1 = lambda rule: (lambda r: 100 * r["rules"][rule]["f1"]); FPR = lambda rule: (lambda r: 100 * r["rules"][rule]["fpr"])
FZ = lambda r: 100 * r["recall"]["Fuzzers"]; CELLS = lambda r: r["cells_logical"] / 1e6
bc = col("a_b34", CELLS).mean()
print(f"baseline a_b34 (VAL, tuning read): F1 {col('a_b34', F1('max')).mean():.2f}  FPR {col('a_b34', FPR('max')).mean():.2f}  Fuzzers recall {col('a_b34', FZ).mean():.1f}  cells {bc:.2f}M")
if a1:
	a1gain = d("b_A1_b64", F1("max")).mean(); a1cells = col("b_A1_b64", CELLS).mean() - bc
print("\nvariant | n | dF1 max [min..max] | wins | dFPR | dFuzzers recall | cells M | added M | dF1 per added M | vs A1 per M | low-FPR draws")
for name in val:
	n = len(val[name]); df = d(name, F1("max")); dp = d(name, FPR("max")); dz = d(name, FZ); c = col(name, CELLS).mean(); add = c - bc
	per = df.mean() / add if add > 0.01 else float("nan")
	low = int((col(name, FPR("max")) < 0.5).sum())
	print(f"{name} | {n} | {df.mean():+.2f} [{df.min():+.2f}..{df.max():+.2f}] | {int((df > 0).sum())}/{n} | {dp.mean():+.2f} | {dz.mean():+.1f} | {c:.2f} | {add:+.2f} | {per:+.2f} | {per / (a1gain / a1cells):.2f}x | {low}/{n}")
print("\ndecision rules, dF1 vs baseline (mean over draws) / FPR mean")
for name in val:
	# every rule is read against the baseline under ITS production rule (max == the 2-cluster margin)
	print(name, " | ".join(f"{r} {d(name, F1(r), F1('max')).mean():+.2f} ({int((d(name, F1(r), F1('max')) > 0).sum())}/5) / {col(name, FPR(r)).mean():.2f}" for r in ("max", "sum", "pooled", "zmax")), f"| min sigma {col(name, lambda r: r['min_sigma']).min():.4f}")
print("\nat exact false-alarm budgets (max rule, interpolated ROC): attack TPR / Fuzzers recall %, mean over draws (difference vs baseline)")
for name in val:
	print(name, " | ".join(f"FPR {100 * float(f):.1f}: TPR {100 * col(name, lambda r: r['at_fpr'][f]['tpr']).mean():.1f} ({100 * d(name, lambda r: r['at_fpr'][f]['tpr']).mean():+.1f}) Fz {100 * col(name, lambda r: r['at_fpr'][f]['fuzzers']).mean():.1f} ({100 * d(name, lambda r: r['at_fpr'][f]['fuzzers']).mean():+.1f}, {int((d(name, lambda r: r['at_fpr'][f]['fuzzers']) > 0).sum())}/5)" for f in ("0.002", "0.003", "0.005", "0.009")))
print("\nbenign rows won per attack cluster at the max-rule tau (mean over draws) [neurons, bits] ; attack rows won")
for name in val:
	names = var[name]["clusters"][1:]; spec = var[name]["spec"][1:]
	if len(names) < 2: continue
	fp = {c: np.mean([val[name][s]["benign_rows_won"][c] for s in val[name]]) for c in names}
	tp = {c: np.mean([val[name][s]["attack_rows_won"][c] for s in val[name]]) for c in names}
	tot = sum(fp.values())
	print(name, f"FP total {tot:.0f}:", "; ".join(f"{c}[{sp[0]}n,{sp[1]}b] FP {fp[c]:.0f} ({100 * fp[c] / max(tot, 1):.0f}%) TP {tp[c]:.0f}" for c, sp in zip(names, spec)))
print("\nper-category recall at the max-rule tau (mean over draws, %)")
cats = list(base[0]["recall"])
print("variant | " + " | ".join(cats))
for name in val:
	print(name, "| " + " | ".join(f"{100 * np.mean([val[name][s]['recall'][c] for s in val[name]]):.1f}" for c in cats))
