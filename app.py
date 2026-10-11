"""IDH Early-Warning - live replay dashboard (supervised + unsupervised).

    python train_models.py      # once
    streamlit run app.py
"""
import json
import os
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import idh_core as C

st.set_page_config(page_title="IDH Early-Warning", page_icon="🩺", layout="wide")

ART = "artifacts"
TIER_COL = {"Normal": "#2E7D32", "Moderate": "#B7791F", "High Risk": "#D9480F", "Critical": "#B42318"}
TIER_ICON = {"Normal": "●", "Moderate": "▲", "High Risk": "◆", "Critical": "■"}
INK, MUTED, GRID = "#1F2937", "#6B7280", "#E5E7EB"
SUP_C, ANO_C = "#1F5FBF", "#7A4EAB"
CHECKLIST = {
    "Normal": ["Routine monitoring - BP every 30 min."],
    "Moderate": ["Check BP every 15 min.", "Review ultrafiltration goal for this session."],
    "High Risk": ["Check BP every 10 min.", "Consider lowering the UF rate / cooling dialysate.",
                  "Have saline bolus ready; alert the dialysis physician."],
    "Critical": ["Continuous BP watch - patient at bedside.", "Reduce UF rate now; consider Trendelenburg / saline bolus.",
                 "Call the physician immediately."],
}

os.chdir(os.path.dirname(os.path.abspath(__file__)))


@st.cache_resource(show_spinner=False)
def ensure_trained():
    """On a fresh server (e.g. Streamlit Cloud) artifacts/ does not exist: train once, ~1-2 min."""
    if os.path.exists(f"{ART}/supervised.joblib"):
        return None
    r = subprocess.run([sys.executable, "train_models.py"], capture_output=True, text=True)
    return None if r.returncode == 0 else (r.stderr or r.stdout)[-2000:]


if not os.path.exists(f"{ART}/supervised.joblib"):
    with st.spinner("First launch: training both models on the dataset (about 1-2 minutes)..."):
        err = ensure_trained()
    if err or not os.path.exists(f"{ART}/supervised.joblib"):
        st.error("Training failed. Details below.")
        st.code(err or "artifacts missing")
        st.stop()


@st.cache_resource
def load():
    sup = joblib.load(f"{ART}/supervised.joblib")
    ano = joblib.load(f"{ART}/anomaly.joblib")
    rep = joblib.load(f"{ART}/replay.joblib")
    with open(f"{ART}/metrics.json") as f:
        met = json.load(f)
    return sup, ano, rep, met


SUP, ANO, REP, MET = load()
RAW_COLS = ["TARGET", "SEX_grp", "AGE_val", "DIA", "TEC", "WDR", "IWG", "HBT", "APR", "TMP", "SBP", "DBP", "HRA"]


def pct_anom(score):
    return float(np.searchsorted(ANO["normal_scores"], score) / len(ANO["normal_scores"]))


def ano_flag(score):
    return score >= ANO["thr"]


def layout(fig, h=320, yfmt=None):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(color=INK, size=13), legend=dict(orientation="h", y=1.14, x=0),
                      hoverlabel=dict(bgcolor="white", font_color=INK))
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID, zeroline=False, tickformat=yfmt)
    return fig


def card(title, big, sub, color, icon=""):
    st.markdown(
        f"""<div style="border:1px solid {GRID};border-left:6px solid {color};border-radius:8px;padding:12px 16px;background:white">
        <div style="color:{MUTED};font-size:13px">{title}</div>
        <div style="font-size:30px;font-weight:700;color:{INK}"><span style="color:{color}">{icon}</span> {big}</div>
        <div style="color:{MUTED};font-size:13px">{sub}</div></div>""", unsafe_allow_html=True)


# --------------------------------------------------------------- sidebar
patients = sorted(REP["patient"].unique())
info = {p: (int((REP.patient == p).sum()), float(REP[REP.patient == p].y.mean())) for p in patients}
st.sidebar.title("🩺 IDH Early-Warning")
pid = st.sidebar.selectbox("Patient (held-out, never trained on)", patients,
                           format_func=lambda p: f"Patient {p} - {info[p][0]} sessions, IDH {info[p][1]:.0%}")
speed = st.sidebar.select_slider("Replay speed", options=[0.5, 1, 2, 4], value=1,
                                 format_func=lambda s: f"{s} session/s")
reveal = st.sidebar.checkbox("Show actual outcome of current session", value=True,
                             help="In real use the outcome is unknown until the session ends.")
c1, c2, c3 = st.sidebar.columns(3)
if "pid" not in st.session_state or st.session_state.pid != pid:
    st.session_state.pid, st.session_state.i, st.session_state.play = pid, 4, False
if c1.button("▶ Play" if not st.session_state.play else "⏸ Pause", width="stretch"):
    st.session_state.play = not st.session_state.play
    st.rerun()
if c2.button("⏭ Step", width="stretch"):
    st.session_state.i = min(st.session_state.i + 1, info[pid][0] - 1)
if c3.button("↺", width="stretch"):
    st.session_state.i, st.session_state.play = 4, False
    st.rerun()
st.sidebar.caption("Replay of de-identified historical sessions. Research prototype - not a medical device.")

P = REP[REP.patient == pid].reset_index(drop=True)
N = len(P)

tab_live, tab_perf, tab_whatif, tab_about = st.tabs(["Live monitor", "Model performance", "What-if", "About & limits"])


# ------------------------------------------------------------------ live
with tab_live:
    @st.fragment(run_every=(1.0 / speed) if st.session_state.play else None)
    def live():
        if st.session_state.play:
            if st.session_state.i < N - 1:
                st.session_state.i += 1
            else:
                st.session_state.play = False
        i = st.session_state.i
        r = P.iloc[i]
        tier = r.tier
        flag = ano_flag(r.a_score)
        sup_flag = r.p_sup >= SUP["thr"]
        ap = pct_anom(r.a_score)

        st.progress((i + 1) / N, text=f"Patient {pid} - session {i + 1} of {N}")
        a, b, c = st.columns(3)
        with a:
            card("Model 1 - Supervised ensemble (XGBoost + CatBoost + HGB)", f"{r.p_sup:.0%}",
                 f"IDH risk - tier: {tier}", TIER_COL[tier], TIER_ICON[tier])
        with b:
            col = ANO_C if flag else "#6B7280"
            card("Model 2 - Unsupervised Isolation Forest", f"{ap:.0%}",
                 "anomaly percentile vs normal sessions - " + ("UNUSUAL" if flag else "typical"), col,
                 "◆" if flag else "●")
        with c:
            if sup_flag and flag:
                lab, col, sub = "Both models alert", TIER_COL["Critical"], "Strongest evidence - act on checklist"
            elif sup_flag:
                lab, col, sub = "Supervised alert", TIER_COL["High Risk"], "Risk pattern learned from labelled IDH"
            elif flag:
                lab, col, sub = "Anomaly only", TIER_COL["Moderate"], "Unusual session, risk model not alarmed"
            else:
                lab, col, sub = "No alert", TIER_COL["Normal"], "Both models quiet"
            card("Consensus", lab, sub, col, "⚑" if (sup_flag or flag) else "●")

        st.write("")
        v = st.columns(5)
        prev = P.iloc[i - 1] if i > 0 else r
        for col, (k, lab, unit) in zip(v, [("SBP", "Systolic BP", "mmHg"), ("DBP", "Diastolic BP", "mmHg"),
                                           ("HRA", "Heart rate", "bpm"), ("IWG", "Weight gain", "kg"),
                                           ("TMP", "TMP", "mmHg")]):
            col.metric(f"{lab} ({unit})", f"{r[k]:.1f}".rstrip("0").rstrip("."), f"{r[k] - prev[k]:+.1f} vs previous",
                       delta_color="off")
        if reveal:
            st.caption(f"Actual outcome of this session: **{'IDH occurred' if r.y == 1 else 'no IDH'}**")

        left, right = st.columns([3, 2])
        with left:
            lo = max(0, i + 1 - 40)          # rolling window: last 40 sessions
            H = P.iloc[lo: i + 1]
            x = np.arange(lo + 1, i + 2)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=H.p_sup, name="Supervised IDH risk", mode="lines",
                                     line=dict(color=SUP_C, width=2)))
            fig.add_trace(go.Scatter(x=x, y=[pct_anom(s) for s in H.a_score], name="Anomaly percentile",
                                     mode="lines", line=dict(color=ANO_C, width=2, dash="dot")))
            seen = H if reveal else H.iloc[:-1]
            hit = seen[seen.y == 1]
            hit = hit.loc[[ix for ix in hit.index if ix >= lo]]
            fig.add_trace(go.Scatter(x=hit.index + 1, y=[1.04] * len(hit), mode="markers", name="Actual IDH",
                                     marker=dict(symbol="triangle-down", size=10, color=TIER_COL["Critical"],
                                                 line=dict(color="white", width=1.5))))
            for t in ["High Risk", "Critical"]:
                fig.add_hline(y=SUP["cut"][t], line=dict(color=TIER_COL[t], width=1, dash="dash"),
                              annotation_text=t, annotation_position="top left",
                              annotation_font=dict(color=MUTED, size=11))
            fig.update_xaxes(range=[lo + 0.5, max(lo + 40, i + 1) + 0.5], title="Session (last 40 shown)")
            fig.update_yaxes(range=[0, 1.1], title="Probability / percentile")
            fig = layout(fig, 340, ".0%")
            st.plotly_chart(fig, width="stretch", key="tl")
        with right:
            feats = C.FEATURES
            top = C.explain(SUP["models"], P[feats].iloc[[i]].values, feats, top=6)
            top = top[::-1]
            fig = go.Figure(go.Bar(x=[t[1] for t in top], y=[C.nice(t[0]) for t in top], orientation="h",
                                   marker_color=[TIER_COL["High Risk"] if t[1] > 0 else SUP_C for t in top],
                                   customdata=[t[2] for t in top],
                                   hovertemplate="%{y}<br>value %{customdata:.1f}<br>effect %{x:+.2f}<extra></extra>"))
            fig.update_layout(title=dict(text="Why this risk? (orange raises, blue lowers)", font=dict(size=14)))
            fig = layout(fig, 340)
            fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, width="stretch", key="why")
            if flag:
                an = C.anomaly_contrib(ANO["am"], P[C.IF_FEATURES].iloc[[i]].values, ANO["ref"])
                st.caption("Most unusual vs normal sessions: " + ", ".join(
                    f"{C.nice(f)} ({z:+.1f}σ)" for f, z in an))

        l2, r2 = st.columns([2, 3])
        with l2:
            st.subheader(f"{TIER_ICON[tier]} Nurse checklist - {tier}")
            for item in CHECKLIST[tier]:
                st.checkbox(item, key=f"ck{i}{item}")
        with r2:
            st.subheader("Alert log")
            log = P.iloc[: i + 1].copy()
            log["Session"] = log.index + 1
            log["Risk"] = log.p_sup.map("{:.0%}".format)
            log["Anomaly pct"] = log.a_score.map(lambda s: f"{pct_anom(s):.0%}")
            log["Both?"] = np.where((log.p_sup >= SUP["thr"]) & (log.a_score >= ANO["thr"]), "yes", "")
            log = log[(log.tier.isin(["High Risk", "Critical"])) | (log["Both?"] == "yes")]
            log = log.rename(columns={"tier": "Tier"})[["Session", "Tier", "Risk", "Anomaly pct", "Both?"]]
            st.dataframe(log.iloc[::-1].head(8), hide_index=True, width="stretch", height=250)
            st.download_button("⬇ Download full session log (CSV)",
                               P.iloc[: i + 1][["p_sup", "tier", "a_score", "y"]].to_csv(index_label="session"),
                               "idh_alert_log.csv", key=f"dl{i}")
    live()


# ------------------------------------------------------------------ performance
with tab_perf:
    m = MET
    s = m["split"]
    st.markdown(f"**Honest evaluation:** {s['n_test_patients']} patients ({s['n_test']:,} sessions) were held out "
                f"entirely - not used to train or to set thresholds. IDH rate in this set: {s['test_idh_rate']:.1%}. "
                f"A model that always says “no IDH” scores **{m['baseline_accuracy']:.0%} accuracy**, so accuracy alone is misleading; "
                "look at AUROC, precision and recall.")
    rows = []
    for k, lab in [("supervised", "Supervised ensemble"), ("anomaly", "Isolation Forest (unsupervised)")]:
        d = m[k]
        rows.append({"Model": lab, "AUROC": d["auroc"], "AUPRC": d["auprc"], "F1": d["f1"],
                     "Precision": d["precision"], "Recall": d["recall"], "Accuracy": d["accuracy"]})
    st.dataframe(pd.DataFrame(rows).set_index("Model").style.format("{:.3f}"), width="stretch")
    st.caption(f"Random-guess AUROC = 0.50 and AUPRC = {s['test_idh_rate']:.2f}.")

    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Chance", line=dict(color=MUTED, dash="dash", width=1)))
        for k, lab, col in [("supervised", "Supervised", SUP_C), ("anomaly", "Isolation Forest", ANO_C)]:
            f, t = m[k]["roc"]
            fig.add_trace(go.Scatter(x=f, y=t, name=f"{lab} (AUROC {m[k]['auroc']:.2f})", line=dict(color=col, width=2)))
        fig.update_xaxes(title="False-positive rate")
        fig.update_yaxes(title="True-positive rate")
        st.plotly_chart(layout(fig, 340), width="stretch")
    with b:
        order = ["Normal", "Moderate", "High Risk", "Critical"]
        rates = [m["tier_rates"][t]["idh_rate"] or 0 for t in order]
        fig = go.Figure(go.Bar(x=order, y=rates, marker_color=[TIER_COL[t] for t in order],
                               text=[f"{r:.0%}<br>n={m['tier_rates'][t]['n']}" for t, r in zip(order, rates)],
                               textposition="outside"))
        fig.update_layout(title=dict(text="Observed IDH rate per risk tier (held-out patients)", font=dict(size=14)))
        fig.update_yaxes(range=[0, 1], title="Observed IDH rate")
        st.plotly_chart(layout(fig, 340, ".0%"), width="stretch")

    st.subheader("Do the two models add up?")
    both, either = m["consensus_both"], m["consensus_either"]
    cons = pd.DataFrame([
        {"Rule": "Supervised alone", "Alerts": int(m["supervised"]["cm"][0][1] + m["supervised"]["cm"][1][1]),
         "Precision": m["supervised"]["precision"], "Recall": m["supervised"]["recall"], "F1": m["supervised"]["f1"]},
        {"Rule": "Both models alert", "Alerts": both["n"], "Precision": both["precision"], "Recall": both["recall"], "F1": both["f1"]},
        {"Rule": "Either model alerts", "Alerts": either["n"], "Precision": either["precision"], "Recall": either["recall"], "F1": either["f1"]},
    ]).set_index("Rule")
    st.dataframe(cons.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}), width="stretch")
    st.info("Honest reading: the Isolation Forest is a weak IDH predictor on its own (AUROC close to 0.5). "
            "Its role is a second, label-free viewpoint that flags sessions unlike normal ones - useful for review and for "
            "new hospitals without labels - while the supervised model carries the prediction.")


# ------------------------------------------------------------------ what-if
with tab_whatif:
    st.markdown("Change the **current session's** readings and watch both models respond. History comes from the selected patient.")
    k = st.slider("Session to edit", 2, N, min(N, st.session_state.i + 1), key="wi_s") - 1
    base = P.iloc[k]
    cc = st.columns(5)
    new = {}
    for col, (f, lo, hi, step) in zip(cc, [("SBP", 50, 200, 1), ("DBP", 20, 120, 1), ("HRA", 40, 160, 1),
                                          ("IWG", 0.0, 6.0, 0.1), ("TMP", 0, 400, 5)]):
        new[f] = col.slider(f"{C.NICE.get(f, f)}", float(lo), float(hi), float(np.clip(base[f], lo, hi)), float(step), key=f"wi_{f}")
    log = P.iloc[: k + 1][RAW_COLS].copy()
    for f, vv in new.items():
        log.loc[log.index[-1], f] = vv
    F = C.build_features(log, patient=np.zeros(len(log), dtype=int))
    p = float(C.sup_proba(SUP["models"], F[C.FEATURES].iloc[[-1]].values)[0])
    sc = float(C.anomaly_score(ANO["am"], F[C.IF_FEATURES].iloc[[-1]].values)[0])
    t0, t1 = base.tier, C.tier_of(p, SUP["cut"])
    a, b, c = st.columns(3)
    with a:
        card("Supervised risk", f"{p:.0%}", f"{t1}  (original {base.p_sup:.0%}, {t0})", TIER_COL[t1], TIER_ICON[t1])
    with b:
        card("Anomaly percentile", f"{pct_anom(sc):.0%}", "unusual" if ano_flag(sc) else "typical",
             ANO_C if ano_flag(sc) else MUTED)
    with c:
        card("Sessions of history used", f"{k}", "previous sessions of this patient", SUP_C)


# ------------------------------------------------------------------ about
with tab_about:
    st.markdown(f"""
### How it works
- **Model 1 - supervised:** XGBoost + CatBoost + HistGradientBoosting, probability-averaged, on {len(C.FEATURES)} features:
  the current session plus the patient's own previous sessions (never the current outcome).
  Four risk tiers are calibrated on out-of-fold predictions of the training patients.
- **Model 2 - unsupervised:** Isolation Forest on 13 session features, trained **only on normal sessions**.
  Anything unlike a normal session gets a high anomaly score (shown as a percentile of normal training sessions).
- **Consensus:** both alert = strongest evidence; one alert = review.

### Limits - please say these out loud when presenting
- Replay of historical data from a public dataset, not live patient monitoring. No hardware is connected.
- The dataset has no patient ID; patients are inferred from consecutive rows with the same age and sex (~200 proxies). Real data will fix this.
- History features need the recorded outcome of previous sessions; the first sessions of a patient have less information.
- Held-out AUROC ≈ {MET['supervised']['auroc']:.2f}: useful, not perfect - roughly {MET['tier_rates']['Critical']['idh_rate']:.0%} of Critical-tier sessions had IDH versus {MET['tier_rates']['Normal']['idh_rate']:.0%} in Normal.
- Research prototype for the capstone; not a medical device and not for clinical decisions.
""")
