"""Thermometer-encoded RF+XGB baselines (verify the paper's thermo numbers + isolate
the raw-vs-thermo delta). Args: dataset split n_bits. Mirrors the WNN's own loader."""
import sys, time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from xgboost import XGBClassifier

def load_thermo(ds_name, split, n_bits):
    if ds_name == "unsw":
        from wnn.ids.dataset import load_unsw_nb15 as L
    elif ds_name == "cicids":
        from wnn.ids.cicids2017 import load_cicids2017 as L
    elif ds_name == "ciciot":
        from wnn.ids.ciciot2023 import load_ciciot2023 as L
    ds = L(n_bits=n_bits, split=split, feature_selection="top20")
    def arr(x): return x.to_numpy_bool() if hasattr(x, "to_numpy_bool") else np.asarray(x)
    Xtr, ytr = arr(ds.X_train), ds.y_train_binary
    if getattr(ds, "X_val", None) is not None:
        Xev = np.concatenate([arr(ds.X_test), arr(ds.X_val)]); yev = np.concatenate([ds.y_test_binary, ds.y_val_binary])
    else:
        Xev, yev = arr(ds.X_test), ds.y_test_binary
    return Xtr, ytr, Xev, yev

def metrics(yt, yp):
    tn, fp, fn, tp = confusion_matrix(yt, yp).ravel()
    return f1_score(yt, yp, average="macro")*100, (fp/(fp+tn))*100, accuracy_score(yt, yp)*100

ds_name, split, n_bits = sys.argv[1], sys.argv[2], int(sys.argv[3])
print(f"### {ds_name} split={split} thermo={n_bits}b", flush=True)
Xtr, ytr, Xev, yev = load_thermo(ds_name, split, n_bits)
print(f"  shapes: train {Xtr.shape}, eval {Xev.shape}", flush=True)
for name, clf in [("RF", RandomForestClassifier(n_estimators=100, n_jobs=4, random_state=42)),
                  ("XGB", XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=4, random_state=42, eval_metric="logloss", verbosity=0))]:
    t0=time.time(); clf.fit(Xtr, ytr); p=clf.predict(Xev); f1, fpr, acc = metrics(yev, p)
    print(f"  RESULT {ds_name}[{split},{n_bits}b] {name:4s}: F1={f1:.2f}%  FPR={fpr:.2f}%  Acc={acc:.2f}%  ({time.time()-t0:.0f}s)", flush=True)
print("DONE", flush=True)
