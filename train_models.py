"""Train both models with PATIENT-HELD-OUT evaluation and write ./artifacts.

    python train_models.py [path/to/MAIN_PROJ_SAMPLE_DATA.xlsx]

* 20% of patients are held out completely. They are NEVER used for fitting or
  threshold calibration, and they are the patients replayed in the dashboard,
  so every number you see live is an honest out-of-sample number.
* Thresholds / risk tiers are calibrated on out-of-fold predictions of the
  training patients only.
"""
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve, accuracy_score)
from sklearn.model_selection import GroupKFold

import idh_core as C

OUT = "artifacts"
DATA = sys.argv[1] if len(sys.argv) > 1 else "MAIN_PROJ_SAMPLE_DATA.xlsx"
t0 = time.time()
os.makedirs(OUT, exist_ok=True)

raw = C.load_raw(DATA)
df = C.build_features(raw)
y = C.target_of(df).values
pat = df["patient"].values
print(f"sessions={len(df)}  IDH rate={y.mean():.1%}  patients(proxy)={len(np.unique(pat))}", flush=True)

# ---- hold out 20% of patients -------------------------------------------------
rng = np.random.RandomState(7)
ids = np.unique(pat)
test_ids = set(rng.choice(ids, size=int(0.2 * len(ids)), replace=False))
is_test = np.array([p in test_ids for p in pat])
tr, te = np.where(~is_test)[0], np.where(is_test)[0]
print(f"train sessions={len(tr)}  held-out sessions={len(te)}  held-out patients={len(test_ids)}", flush=True)

Xs = df[C.FEATURES].values
Xa = df[C.IF_FEATURES].values

# ---- out-of-fold predictions on TRAIN patients (for calibration) -------------
oof_s = np.zeros(len(tr))
oof_a = np.zeros(len(tr))
for k, (a, b) in enumerate(GroupKFold(n_splits=4).split(tr, y[tr], pat[tr]), 1):
    ia, ib = tr[a], tr[b]
    oof_s[b] = C.sup_proba(C.fit_supervised(Xs[ia], y[ia]), Xs[ib])
    am = C.fit_anomaly(Xa[ia], y[ia])
    oof_a[b] = C.anomaly_score(am, Xa[ib])
    print(f"  fold {k} done  ({time.time()-t0:.0f}s)", flush=True)

thr_s = C.best_f1_threshold(y[tr], oof_s)
cut = C.tier_cutoffs(y[tr], oof_s, thr_s)
# anomaly threshold: best F1 on OOF anomaly scores (train patients only)
qs = np.quantile(oof_a, np.arange(0.50, 0.98, 0.01))
f1s = [f1_score(y[tr], (oof_a >= q).astype(int)) for q in qs]
thr_a = float(qs[int(np.argmax(f1s))])
print(f"supervised thr={thr_s:.3f} tiers={cut}  anomaly thr={thr_a:.4f}", flush=True)

# ---- final models on all TRAIN patients ---------------------------------------
sup = C.fit_supervised(Xs[tr], y[tr])
am = C.fit_anomaly(Xa[tr], y[tr])
Zn = am["pre"].transform(Xa[tr][y[tr] == 0])
ref = (Zn.mean(0), Zn.std(0))
normal_scores = np.sort(C.anomaly_score(am, Xa[tr][y[tr] == 0]))

# ---- score everything (held-out rows are the honest ones) ---------------------
p_all = C.sup_proba(sup, Xs)
a_all = C.anomaly_score(am, Xa)


def metrics(yt, score, thr):
    pred = (score >= thr).astype(int)
    fpr, tpr, _ = roc_curve(yt, score)
    step = max(1, len(fpr) // 120)
    return dict(auroc=roc_auc_score(yt, score), auprc=average_precision_score(yt, score),
                f1=f1_score(yt, pred), precision=precision_score(yt, pred, zero_division=0),
                recall=recall_score(yt, pred), accuracy=accuracy_score(yt, pred),
                cm=confusion_matrix(yt, pred).tolist(),
                roc=[fpr[::step].tolist(), tpr[::step].tolist()], thr=float(thr))


yt = y[te]
M = {"supervised": metrics(yt, p_all[te], thr_s),
     "anomaly": metrics(yt, a_all[te], thr_a)}
both = ((p_all[te] >= thr_s) & (a_all[te] >= thr_a)).astype(int)
either = ((p_all[te] >= thr_s) | (a_all[te] >= thr_a)).astype(int)
M["consensus_both"] = dict(precision=precision_score(yt, both, zero_division=0),
                           recall=recall_score(yt, both), f1=f1_score(yt, both), n=int(both.sum()))
M["consensus_either"] = dict(precision=precision_score(yt, either, zero_division=0),
                             recall=recall_score(yt, either), f1=f1_score(yt, either), n=int(either.sum()))
tiers = np.array([C.tier_of(p, cut) for p in p_all[te]])
M["tier_rates"] = {t: dict(n=int((tiers == t).sum()),
                           idh_rate=float(yt[tiers == t].mean()) if (tiers == t).any() else None)
                   for t in ["Normal", "Moderate", "High Risk", "Critical"]}
M["split"] = dict(n_train=int(len(tr)), n_test=int(len(te)), n_test_patients=len(test_ids),
                  idh_rate=float(y.mean()), test_idh_rate=float(yt.mean()))
M["tiers"] = cut
M["baseline_accuracy"] = float(1 - yt.mean())
for k in ("supervised", "anomaly"):
    print(k, {a: round(b, 3) for a, b in M[k].items() if isinstance(b, float)}, flush=True)
print("tier rates", M["tier_rates"])

# ---- replay set: held-out patients with a reasonable number of sessions -------
cnt = pd.Series(pat[te]).value_counts()
rep_ids = [p for p in cnt.index if cnt[p] >= 20]
rep = df[df["patient"].isin(rep_ids)].copy()
rep["p_sup"] = p_all[rep.index]
rep["a_score"] = a_all[rep.index]
rep["tier"] = [C.tier_of(p, cut) for p in rep["p_sup"]]
rep["y"] = y[rep.index]
print("replay patients:", len(rep_ids), "sessions:", len(rep), flush=True)

joblib.dump({"models": sup, "features": C.FEATURES, "thr": thr_s, "cut": cut}, f"{OUT}/supervised.joblib")
joblib.dump({"am": am, "thr": thr_a, "ref": ref, "normal_scores": normal_scores}, f"{OUT}/anomaly.joblib")
joblib.dump(rep, f"{OUT}/replay.joblib")
with open(f"{OUT}/metrics.json", "w") as f:
    json.dump(M, f, indent=1)
print(f"done in {time.time()-t0:.0f}s", flush=True)
