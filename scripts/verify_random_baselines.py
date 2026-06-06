"""Measured RF+XGB baselines for unsw-random + cicids-random, ALIGNED to the WNN's
`random` 80/20 split (same partition + top-20 features the WNN cohort uses), raw
numeric features, n_jobs capped (IDS owns the cores)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_all_baselines import load_raw_dataset, compute_metrics
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

NJOBS = 4
for ds in ["unsw", "cicids"]:
    print(f"\n{'='*60}\n  {ds} — random 80/20, raw top-20 (n_jobs={NJOBS})\n{'='*60}")
    try:
        Xtr, ytr, Xev, yev = load_raw_dataset(ds, split="random")
    except Exception as e:
        print(f"  LOAD FAILED ({ds}, random): {e}"); continue
    for name, clf in [
        ("RF", RandomForestClassifier(n_estimators=100, n_jobs=NJOBS, random_state=42)),
        ("XGB", XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                              n_jobs=NJOBS, random_state=42, eval_metric="logloss", verbosity=0)),
    ]:
        t0 = time.time(); clf.fit(Xtr, ytr); pred = clf.predict(Xev)
        m = compute_metrics(yev, pred)
        print(f"  RESULT {ds:6s} {name:4s}: F1={m['f1']*100:.2f}%  FPR={m['fpr']*100:.2f}%  Acc={m['acc']*100:.2f}%  ({time.time()-t0:.0f}s)")
print("\nDONE")
