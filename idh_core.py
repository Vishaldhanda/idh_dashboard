"""Shared logic for the IDH early-warning dashboard.

Two models:
  1. SUPERVISED  - XGBoost + CatBoost + HistGradientBoosting ensemble that uses
                   the current session plus the patient's own history.
  2. UNSUPERVISED - Isolation Forest trained on NORMAL sessions only
                   (the 13 features of running_model.py). Anything that looks
                   unlike a normal session gets a high anomaly score.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import DMatrix, XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
SEED = 42
SHEET = "dataset_11"

BASE = ["SBP", "DBP", "pulse_pressure", "WDR", "HRA", "TMP", "IWG", "APR",
        "DIA", "TEC", "SEX_grp", "AGE_val", "HBT"]
ENG = BASE + ["MAP", "fluid_pct", "shock_index", "tmp_per_kg"]
HIST = (["y_prev", "y_prev_rate", "y_last3", "y_last5", "y_count", "since_idh", "session_idx",
         "SBP_last3", "DBP_last3", "SBP_min_prev", "DBP_min_prev", "SBP_std_prev", "DBP_std_prev",
         "WDR_delta"] +
        [f"{c}_{s}" for c in ["SBP", "DBP", "HRA", "IWG"] for s in ["prev", "delta", "pmean", "dev"]])
FEATURES = ENG + HIST

# Unsupervised model uses exactly the 13 features of running_model.py
IF_FEATURES = ["AGE_val", "SEX_grp", "DIA", "TEC", "WDR", "IWG", "HBT", "APR",
               "TMP", "SBP", "DBP", "HRA", "pulse_pressure"]

NICE = {
    "SBP": "Systolic BP", "DBP": "Diastolic BP", "pulse_pressure": "Pulse pressure",
    "WDR": "Dry weight", "HRA": "Heart rate", "TMP": "Transmembrane pressure",
    "IWG": "Inter-dialytic weight gain", "APR": "Arterial pressure", "DIA": "Diabetes",
    "TEC": "Treatment/Access code", "SEX_grp": "Sex", "AGE_val": "Age", "HBT": "Hemoglobin",
    "MAP": "Mean arterial pressure", "fluid_pct": "Fluid gain (% dry wt)",
    "shock_index": "Shock index (HR/SBP)", "tmp_per_kg": "TMP per kg",
    "y_prev": "IDH in last session", "y_prev_rate": "Past IDH rate", "y_last3": "IDH in last 3",
    "y_last5": "IDH in last 5", "y_count": "Past IDH count", "since_idh": "Sessions since IDH",
    "session_idx": "Session number", "WDR_delta": "Dry-weight change",
}


def nice(f):
    if f in NICE:
        return NICE[f]
    for suf, lab in [("_prev", "previous"), ("_delta", "change vs previous"),
                     ("_pmean", "patient average"), ("_dev", "deviation from patient avg"),
                     ("_last3", "avg of last 3"), ("_min_prev", "lowest so far"),
                     ("_std_prev", "variability so far")]:
        if f.endswith(suf):
            return f"{NICE.get(f[:-len(suf)], f[:-len(suf)])} - {lab}"
    return f


# ------------------------------------------------------------------ data
def load_raw(path):
    return pd.read_excel(path, sheet_name=SHEET)


def target_of(df):
    """TARGET may be 'SI'/'NO' text (Mendeley original) or already 0/1."""
    t = df["TARGET"]
    if t.dtype == object or str(t.dtype).startswith("str"):
        return (t.astype(str).str.upper() == "SI").astype(int)
    return t.fillna(0).astype(int)


def assign_patient(df):
    """No patient ID in the file: consecutive rows with the same (age, sex) = one patient.
    REPLACE with a real ID when real hospital data arrives."""
    key = df["AGE_val"].astype(str) + "_" + df["SEX_grp"].astype(str)
    return (key != key.shift()).cumsum().values


def build_features(df, patient=None):
    """Chronological per patient. History features use ONLY earlier sessions (shift(1))."""
    df = df.copy().reset_index(drop=True)
    if patient is None:
        patient = assign_patient(df)
    df["patient"] = patient
    df["pulse_pressure"] = df["SBP"] - df["DBP"]
    df["MAP"] = df["DBP"] + df["pulse_pressure"] / 3
    df["fluid_pct"] = df["IWG"] / df["WDR"] * 100
    df["shock_index"] = df["HRA"] / df["SBP"]
    df["tmp_per_kg"] = df["TMP"] / df["WDR"]
    g = df.groupby("patient")
    y = target_of(df) if "TARGET" in df.columns else pd.Series(np.nan, index=df.index)
    ys = y.groupby(df["patient"])
    df["y_prev"] = ys.shift(1)
    df["y_prev_rate"] = ys.transform(lambda s: s.shift(1).expanding().mean())
    df["y_last3"] = ys.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["y_last5"] = ys.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    df["y_count"] = ys.transform(lambda s: s.shift(1).expanding().sum())

    def since_last(s):
        out, last = [], None
        for i, v in enumerate(s.shift(1).fillna(0).values):
            out.append(np.nan if last is None else i - last)
            if v == 1:
                last = i
        return pd.Series(out, index=s.index)
    df["since_idh"] = ys.transform(since_last)
    df["session_idx"] = g.cumcount()
    for c in ["SBP", "DBP", "HRA", "IWG"]:
        df[f"{c}_prev"] = g[c].shift(1)
        df[f"{c}_delta"] = df[c] - df[f"{c}_prev"]
        df[f"{c}_pmean"] = g[c].transform(lambda s: s.shift(1).expanding().mean())
        df[f"{c}_dev"] = df[c] - df[f"{c}_pmean"]
    for c in ["SBP", "DBP"]:
        df[f"{c}_last3"] = g[c].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        df[f"{c}_min_prev"] = g[c].transform(lambda s: s.shift(1).expanding().min())
        df[f"{c}_std_prev"] = g[c].transform(lambda s: s.shift(1).expanding().std())
    df["WDR_delta"] = df["WDR"] - g["WDR"].shift(1)
    return df


# ------------------------------------------------------------- supervised
def make_models():
    return [
        XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.04, subsample=0.8,
                      colsample_bytree=0.8, scale_pos_weight=4, eval_metric="auc",
                      random_state=SEED, n_jobs=-1),
        CatBoostClassifier(iterations=250, depth=5, learning_rate=0.05,
                           auto_class_weights="Balanced", random_seed=SEED,
                           verbose=0, thread_count=-1),
        HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300,
                                       class_weight="balanced", random_state=SEED),
    ]


def fit_supervised(X, y):
    ms = make_models()
    for m in ms:
        m.fit(X, y)
    return ms


def sup_proba(models, X):
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def best_f1_threshold(y, p):
    ts = np.arange(0.05, 0.95, 0.005)
    f = [f1_score(y, (p >= t).astype(int)) for t in ts]
    return float(ts[int(np.argmax(f))])


def tier_cutoffs(y, oof, thr):
    prec, rec, th = precision_recall_curve(y, oof)
    ok = np.where(rec[:-1] >= 0.80)[0]
    mod = float(th[ok.max()]) if len(ok) else thr * 0.7
    mod = min(mod, thr)
    crit = float(np.quantile(oof, 0.95))
    idx = np.where(prec[:-1] >= 0.60)[0]
    if len(idx):
        crit = min(crit, float(th[idx.min()]))
    crit = max(crit, thr)
    return {"Moderate": mod, "High Risk": float(thr), "Critical": crit}


def tier_of(p, cut):
    if p >= cut["Critical"]:
        return "Critical"
    if p >= cut["High Risk"]:
        return "High Risk"
    if p >= cut["Moderate"]:
        return "Moderate"
    return "Normal"


def explain(models, X_row, features, top=6):
    """Per-feature contribution (log-odds) from the XGBoost member via pred_contribs."""
    xgb = models[0]
    contrib = xgb.get_booster().predict(DMatrix(X_row, feature_names=list(features)),
                                        pred_contribs=True)[0][:-1]
    order = np.argsort(-np.abs(contrib))[:top]
    return [(features[i], float(contrib[i]), float(X_row[0][i])) for i in order]


# ----------------------------------------------------------- unsupervised
def make_iforest():
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    return pipe, IsolationForest(n_estimators=500, contamination="auto", max_features=1.0,
                                 random_state=SEED, n_jobs=-1)


def fit_anomaly(X_all, y_all):
    """Fit scaler on training rows; Isolation Forest on NORMAL rows only."""
    pre, iso = make_iforest()
    Z = pre.fit_transform(X_all)
    iso.fit(Z[y_all == 0])
    return {"pre": pre, "iso": iso}


def anomaly_score(am, X):
    """Higher = more anomalous."""
    return -am["iso"].score_samples(am["pre"].transform(X))


def anomaly_contrib(am, X_row, normal_ref_z, top=3):
    """Which features look most unusual: z-score vs the normal-session population."""
    z = am["pre"].transform(X_row)[0]
    mu, sd = normal_ref_z
    zz = (z - mu) / np.where(sd == 0, 1, sd)
    order = np.argsort(-np.abs(zz))[:top]
    return [(IF_FEATURES[i], float(zz[i])) for i in order]
