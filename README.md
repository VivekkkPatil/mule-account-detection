# 🔍 Mule Account Detection System
> AI/ML-powered financial fraud detection — identifying mule accounts through behavioral analysis, explainability, and intelligent alert generation.

---

## 📊 Results at a Glance

| Metric | Value |
|--------|-------|
| Total Accounts Analyzed | 9,082 |
| Mules Detected | 73 / 81 (90.1%) |
| False Alarms | 25 |
| Accounts Flagged for Review | 98 |
| Investigator Workload Reduction | 98.9% |
| PR-AUC | 0.8932 |
| ROC-AUC | 0.9819 |

---

## 🧠 Problem Statement

Banks face a growing threat from mule accounts — accounts opened with stolen or fake identities to receive, transfer, and conceal fraudulent funds. Traditional rule-based systems fail to detect evolving fraud patterns.

This system builds an AI/ML pipeline that:
- Learns behavioral patterns from 3,922 account-level features
- Identifies suspicious and mule accounts with high recall
- Explains every decision using SHAP values
- Auto-generates investigation reports for compliance officers
- Logs all investigator actions for regulatory audit trails

---

## 🏗️ Architecture

Raw Data (9,082 accounts, 3,922 features)

↓

Leakage Detection & Removal

(Found 2 leaks: post-outcome flag + dataset batch artifact)

↓

Feature Triage (3,922 → 80 features)

Mutual Information + XGBoost dual-ranking

↓

Preprocessing Pipeline

(Median imputation, one-hot encoding, standard scaling)

↓

┌─────────────────────────────────────┐

│  Model A: Statistical (80 features) │

│  Model B: Domain-Informed (98 feats)│

│  Isolation Forest (anomaly scores)  │

└─────────────────────────────────────┘

↓

Ensemble (60% Model A + 40% Model B)

Blended Risk Score → Red / Amber / Green

↓

SHAP Explainability

(Per-account top risk drivers)

↓

LLM Investigation Reports

(Groq / LLaMA-3.3-70b)

↓

Streamlit Dashboard

(Lookup · Graph · Actions · Audit Trail)

---

## 🔑 Key Technical Decisions

### 1. Data Leakage Detection
Discovered and removed two distinct leakage sources before modeling:
- **F3912**: A post-investigation outcome flag (0.97 correlation with target) — would not exist for a new unknown account
- **F2230**: A dataset batch artifact — every mule was labeled "Sep/Nov/Dec25" while every normal account was "Oct25", a data collection artifact with perfect separation

### 2. Dual-Ranking Feature Selection
Used two independent feature ranking methods (Mutual Information + XGBoost importance) and combined their average ranks. A feature had to score well on both methods to make the final set — reducing bias from any single selection method.

### 3. Two Complementary Models + Ensemble
- **Model A** (statistical): 90.1% recall, 40 false alarms
- **Model B** (domain-informed): 85.2% recall, 16 false alarms
- **Ensemble**: 90.1% recall, 25 false alarms — best of both

### 4. Imbalance Handling
- 0.89% positive rate (81 mules / 9,082 accounts)
- Used `scale_pos_weight` (~111x) instead of naive SMOTE in high dimensions
- Stratified 5-fold cross-validation to ensure reliable evaluation
- PR-AUC as primary metric (not ROC-AUC, which is misleading under extreme imbalance)

### 5. Bayesian Hyperparameter Tuning
Used Optuna (30 trials, TPE sampler) to maximize recall at threshold 0.35 — improved from 88.9% to 90.1% without threshold manipulation.

---

## 📁 Project Structure

mule-account-detection/

│

├── data/

│   └── DataSet.csv

│

├── notebooks/

│   └── 01_eda.ipynb

│

├── src/

│   ├── config.py

│   ├── data_loader.py

│   ├── feature_triage.py

│   ├── preprocessing.py

│   ├── evaluate.py

│   ├── explainability.py

│   ├── linkage_graph.py

│   ├── report_generator.py

│   ├── action_logger.py

│   └── modeling/

│       ├── supervised.py

│       ├── unsupervised.py

│       └── ensemble.py

│

├── models/

│   ├── preprocessor.pkl

│   ├── xgb_model_tuned.pkl

│   ├── xgb_model_B.pkl

│   └── isolation_forest.pkl

│

├── outputs/

│   ├── selected_features.json

│   ├── selected_features_modelB.json

│   ├── risk_scores.csv

│   ├── shap_values.csv

│   ├── linkage_graph.html

│   ├── audit_trail.csv

│   ├── reports/

│   └── SAR_reports/

│

├── app/

│   └── streamlit_app.py

│

├── .env

├── .gitignore

├── requirements.txt

└── README.md

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/mule-account-detection.git
cd mule-account-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
Create a `.env` file in the project root:

Get a free API key at [console.groq.com](https://console.groq.com)

### 4. Add the dataset
Place `DataSet.csv` in the `data/` folder.

### 5. Run the notebook
Open `notebooks/01_eda.ipynb` and run all cells in order to:
- Load and explore data
- Run feature triage (takes ~4 min for mutual information step)
- Train all models
- Generate SHAP values and reports

### 6. Launch the dashboard
```bash
streamlit run app/streamlit_app.py
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| ML Models | XGBoost, Isolation Forest |
| Hyperparameter Tuning | Optuna (Bayesian, TPE) |
| Feature Selection | scikit-learn (MI), XGBoost importance |
| Explainability | SHAP |
| LLM Reports | Groq API / LLaMA-3.3-70b |
| Graph Analysis | NetworkX, Pyvis |
| Dashboard | Streamlit, Plotly |
| Data Processing | pandas, NumPy, scikit-learn |

---

## 📈 Dashboard Features

| Tab | Description |
|-----|-------------|
| 📋 Flagged Accounts | All Red/Amber accounts sorted by risk score |
| 🔎 Account Lookup | Per-account risk score, SHAP chart, LLM report |
| 🕸️ Linkage Graph | Interactive behavioral similarity network |
| 📊 Model Comparison | Model A vs B vs Ensemble side-by-side |
| 📈 Model Insights | PR curve, ROC curve, confusion matrix, SHAP summary |
| 🔒 Audit Trail | All investigator actions + SAR reports |

---

## ⚠️ Limitations & Next Steps

- **Dataset scope**: Account-level behavioral data only. Transaction-level analysis would extend detection to individual suspicious transactions.
- **Static model**: Designed for periodic retraining as fraud patterns evolve. Production deployment would include model drift monitoring.
- **Regulatory feeds**: Architecture supports I4C/RBI feed integration — demonstrated on the provided dataset pending real feed access.
- **Graph layer**: Built on attribute similarity (no sender-receiver transaction data available). With real transaction logs, this would become a true money-flow network graph.

---

## 👤 Author

**Vivek Patil**
B.E. Computer Engineering — SIES Graduate School of Technology, Mumbai University
[GitHub](https://github.com/VivekkkPatil) · [LinkedIn](https://www.linkedin.com/in/vivek-patil-8ba9332a7)