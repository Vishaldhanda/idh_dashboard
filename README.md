# IDH Early-Warning - Live Dashboard (Supervised + Unsupervised)

    pip install -r requirements.txt
    python train_models.py        # ~1 min; trains both models, writes ./artifacts
    streamlit run app.py          # opens http://localhost:8501

Model 1: XGBoost + CatBoost + HistGradientBoosting ensemble (47 features incl. patient history).
Model 2: Isolation Forest trained on normal sessions only (13 features from running_model.py).
20% of patients are held out; the dashboard replays only those patients, so all numbers are out-of-sample.
Held-out results: supervised AUROC 0.79; Isolation Forest AUROC 0.54 (weak - see "About & limits" in the app).

Presenting tips: pick a patient, press Play (or Step), then open "Model performance" and "What-if".
Research prototype - not a medical device. Replays historical data; no hardware is connected.

## Deploy on Streamlit Community Cloud
1. Push this folder's contents to a GitHub repo (repo root = this folder).
2. share.streamlit.io -> New app -> pick the repo, main file `app.py` -> Deploy.
3. First launch trains both models automatically (about 1-2 minutes); later launches reuse them.
Do not put real patient data in a public repo.
