"""Simulated dialysis-ward data feed (SYNTHETIC - for demonstrating the data flow only).

Each simulated patient is built from the statistical profile of a real patient group in the
dataset (their mean / spread of SBP, DBP, HRA, IWG, TMP, APR, plus fixed age, sex, dry weight...).
New sessions are drawn with session-to-session correlation around that profile.

The 'outcome' (IDH or not) is drawn from a simple, transparent risk rule - NOT from the trained
model and NOT from real clinical data - so accuracy measured on this feed is NOT validation.
Rule: higher fluid gain %, lower SBP than the patient's usual, and IDH last session raise the odds.
"""
import numpy as np
import pandas as pd

import idh_core as C

RAW_COLS = ["TARGET", "SEX_grp", "AGE_val", "DIA", "TEC", "WDR", "IWG", "HBT", "APR", "TMP", "SBP", "DBP", "HRA"]
DYN = {  # variable: (session-to-session correlation, minimum spread)
    "SBP": (0.4, 8.0), "DBP": (0.4, 5.0), "HRA": (0.4, 4.0),
    "IWG": (0.3, 0.4), "TMP": (0.4, 20.0), "APR": (0.4, 15.0),
}
CLIP = {"SBP": (60, 230), "DBP": (25, 130), "HRA": (40, 150),
        "IWG": (0.2, 6.5), "TMP": (20, 270), "APR": (-310, -40)}
FLUID_MU, FLUID_SD = 2.41, 1.76     # population fluid-gain % (from the dataset)
B0 = -1.85                          # intercept, calibrated so ~20% of sessions are IDH
PATIENT_SESSIONS = 40               # a bed's patient is "discharged" after this many sessions
WARM = 5                            # sessions of history loaded when a patient arrives


def build_profiles(raw):
    """Profiles of every patient group (>= 8 sessions) in the dataset."""
    df = raw.copy()
    df["patient"] = C.assign_patient(df)
    out = []
    for pid, g in df.groupby("patient"):
        if len(g) < 8:
            continue
        prof = {"id": int(pid)}
        for c in DYN:
            prof[c + "_mu"] = float(g[c].mean())
            prof[c + "_sd"] = float(max(g[c].std(), DYN[c][1]))
        for c in ["AGE_val", "SEX_grp", "DIA", "TEC"]:
            prof[c] = float(g[c].mode().iloc[0])
        prof["WDR"] = float(g["WDR"].mean())
        prof["HBT"] = float(g["HBT"].mean())
        out.append(prof)
    return out


class SimPatient:
    def __init__(self, prof, rng, label):
        self.p, self.rng, self.label = prof, rng, label
        self.x = {c: prof[c + "_mu"] for c in DYN}
        self.y_prev = 0
        self.n = 0

    def next_session(self, stress=False):
        p, rng = self.p, self.rng
        row = {}
        for c, (phi, _) in DYN.items():
            mu, sd = p[c + "_mu"], p[c + "_sd"]
            self.x[c] = mu + phi * (self.x[c] - mu) + np.sqrt(1 - phi ** 2) * sd * rng.randn()
            row[c] = self.x[c]
        if stress:   # a clearly risky session: big fluid gain, falling BP, faster heart rate
            row["IWG"] = row["IWG"] * 1.6 + 1.2
            row["SBP"] -= 28
            row["DBP"] -= 10
            row["HRA"] += 12
        for c, (lo, hi) in CLIP.items():
            row[c] = float(np.clip(row[c], lo, hi))
        for c in ["AGE_val", "SEX_grp", "DIA", "TEC", "WDR", "HBT"]:
            row[c] = p[c]
        row["SBP"], row["DBP"], row["HRA"] = round(row["SBP"]), round(row["DBP"]), round(row["HRA"])
        row["IWG"], row["TMP"], row["APR"] = round(row["IWG"], 1), round(row["TMP"] / 5) * 5, round(row["APR"] / 5) * 5
        zf = (row["IWG"] / row["WDR"] * 100 - FLUID_MU) / FLUID_SD
        zs = (row["SBP"] - p["SBP_mu"]) / p["SBP_sd"]
        logit = B0 + 0.8 * zf - 0.7 * zs + 0.7 * self.y_prev
        row["TARGET"] = int(rng.rand() < 1 / (1 + np.exp(-logit)))
        self.y_prev = row["TARGET"]
        self.n += 1
        return row


class Bed:
    def __init__(self, name, profiles, rng):
        self.name, self.profiles, self.rng = name, profiles, rng
        self.seq = 0
        self.log = []          # list of raw-row dicts for the current patient
        self.risk, self.anom = [], []    # per-session history for the chart
        self.new_patient()

    def new_patient(self):
        self.seq += 1
        prof = self.profiles[self.rng.randint(len(self.profiles))]
        self.patient = SimPatient(prof, self.rng, f"{self.name}-P{self.seq}")
        self.log = [self.patient.next_session() for _ in range(WARM)]   # earlier sessions, already recorded
        self.risk, self.anom = [], []

    def advance(self, stress=False):
        """New session arrives. Returns True if a new patient was admitted."""
        admitted = False
        if self.patient.n >= PATIENT_SESSIONS:
            self.new_patient()
            admitted = True
        self.log.append(self.patient.next_session(stress))
        return admitted

    def frame(self):
        return pd.DataFrame(self.log)[RAW_COLS]


def score(bed, sup, ano):
    """Score the newest session of a bed with both models (history excludes its own outcome)."""
    F = C.build_features(bed.frame(), patient=np.zeros(len(bed.log), dtype=int))
    row = F.iloc[[-1]]
    p = float(C.sup_proba(sup["models"], row[C.FEATURES].values)[0])
    a = float(C.anomaly_score(ano["am"], row[C.IF_FEATURES].values)[0])
    return p, a, row


def calibrate(profiles, n=4000, seed=1):
    rng = np.random.RandomState(seed)
    ys = []
    for _ in range(n // 40):
        sp = SimPatient(profiles[rng.randint(len(profiles))], rng, "x")
        ys += [sp.next_session()["TARGET"] for _ in range(40)]
    return float(np.mean(ys))
