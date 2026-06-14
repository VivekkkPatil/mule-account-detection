import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config
from src.data_loader import load_dataset
from src.preprocessing import fit_transform_features
from src.explainability import get_top_drivers
from src.report_generator import generate_report
from src.action_logger import log_action, generate_sar, get_audit_trail, get_account_status
from src.linkage_graph import build_linkage_graph, render_pyvis_graph, get_connected_components

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Mule Account Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# LOAD DATA (cached)
# ─────────────────────────────────────────────
@st.cache_data
def load_all():
    X, y = load_dataset()

    with open(config.SELECTED_FEATURES_PATH) as f:
        selected_features = json.load(f)

    with open(config.OUTPUTS_DIR / "selected_features_modelB.json") as f:
        combined_features = json.load(f)

    X_processed, pipeline, feature_names = fit_transform_features(X, selected_features)
    X_processed_B, pipeline_B, feature_names_B = fit_transform_features(X, combined_features)

    risk_df = pd.read_csv(config.RISK_SCORES_PATH)
    shap_df = pd.read_csv(config.OUTPUTS_DIR / "shap_values.csv")
    oof_preds = np.load(config.OUTPUTS_DIR / "oof_preds_ensemble.npy")

    return (X, y, X_processed, X_processed_B,
            feature_names, feature_names_B,
            selected_features, combined_features,
            risk_df, shap_df, oof_preds)

@st.cache_resource
def load_models():
    model_A = joblib.load(config.MODELS_DIR / "xgb_model_tuned.pkl")
    model_B = joblib.load(config.MODELS_DIR / "xgb_model_B.pkl")
    iso = joblib.load(config.MODELS_DIR / "isolation_forest.pkl")
    return model_A, model_B, iso

(X, y, X_processed, X_processed_B,
 feature_names, feature_names_B,
 selected_features, combined_features,
 risk_df, shap_df, oof_preds) = load_all()

model_A, model_B, iso_model = load_models()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/bank.png", width=60)
st.sidebar.title("Mule Account Detection")
st.sidebar.markdown("**AI-powered Financial Fraud Detection**")
st.sidebar.divider()

st.sidebar.markdown("### 📊 Final Model Performance")
st.sidebar.metric("Mules Caught", "73 / 81", "90.1% recall")
st.sidebar.metric("Workload Reduction", "98.9%", "98 of 9082 flagged")
st.sidebar.metric("False Alarms", "25", "0.28% of normal accounts")
st.sidebar.metric("PR-AUC", "0.8932")
st.sidebar.divider()

st.sidebar.markdown("### 🧠 Model Architecture")
st.sidebar.markdown("""
- **Model A**: Statistical (80 features)
- **Model B**: Domain-Informed (98 features)
- **Ensemble**: 60/40 blend (final)
- Isolation Forest anomaly detection
- SHAP explainability
- Groq/LLaMA-3.3 investigation reports
""")
st.sidebar.divider()

st.sidebar.markdown("### 🔧 Tech Stack")
st.sidebar.markdown("""
XGBoost · Optuna · Isolation Forest
SHAP · NetworkX · Pyvis
Groq/LLaMA-3.3-70b · Streamlit
""")

# ─────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────
st.title("🔍 Mule Account Detection System")
st.markdown("*Detecting financial crime through behavioral AI — not just rules*")

# Top metrics row
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Accounts", "9,082")
m2.metric("🔴 Red", str(risk_df[risk_df["risk_tier"]=="Red"].shape[0]))
m3.metric("🟡 Amber", str(risk_df[risk_df["risk_tier"]=="Amber"].shape[0]))
m4.metric("✅ Mules Caught", "73 / 81")
m5.metric("📉 Workload Cut", "98.9%")
st.divider()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Flagged Accounts",
    "🔎 Account Lookup",
    "🕸️ Linkage Graph",
    "📊 Model Comparison",
    "📈 Model Insights",
    "🔒 Audit Trail"
])

# ══════════════════════════════════════════════
# TAB 1 — FLAGGED ACCOUNTS
# ══════════════════════════════════════════════
with tab1:
    st.subheader("Flagged Accounts Overview")

    tier_filter = st.multiselect(
        "Filter by Risk Tier",
        ["Red", "Amber", "Green"],
        default=["Red", "Amber"]
    )

    filtered = risk_df[risk_df["risk_tier"].isin(tier_filter)].copy()
    filtered = filtered.sort_values("risk_score", ascending=False).reset_index(drop=True)

    filtered["true_label"] = filtered["true_label"].map({1: "✅ MULE", 0: "❌ Normal"})
    filtered["risk_tier"] = filtered["risk_tier"].map({
        "Red": "🔴 Red", "Amber": "🟡 Amber", "Green": "🟢 Green"
    })
    filtered["risk_score"] = filtered["risk_score"].round(4)
    filtered["xgb_proba"] = filtered["xgb_proba"].round(4)

    st.dataframe(
        filtered[["account_idx", "risk_tier", "risk_score",
                  "xgb_proba", "anomaly_score", "true_label"]],
        use_container_width=True,
        height=450
    )

    st.download_button(
        "⬇️ Download Flagged Accounts CSV",
        filtered.to_csv(index=False),
        file_name="flagged_accounts.csv",
        mime="text/csv"
    )

# ══════════════════════════════════════════════
# TAB 2 — ACCOUNT LOOKUP
# ══════════════════════════════════════════════
with tab2:
    st.subheader("Account Risk Lookup")

    account_input = st.number_input(
        "Account Index (0 to 9081)",
        min_value=0, max_value=9081, value=285, step=1
    )

    if st.button("🔍 Analyse Account", type="primary"):
        row = risk_df[risk_df["account_idx"] == account_input]

        if row.empty:
            st.error("Account not found.")
        else:
            row = row.iloc[0]
            tier = row["risk_tier"]
            tier_color = {"Red": "🔴", "Amber": "🟡", "Green": "🟢"}.get(tier, "")
            current_status = get_account_status(int(account_input))

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Risk Score", f"{row['risk_score']:.4f}")
            c2.metric("Risk Tier", f"{tier_color} {tier}")
            c3.metric("XGB Probability", f"{row['xgb_proba']:.4f}")
            c4.metric("True Label", "✅ MULE" if row["true_label"] == 1 else "❌ Normal")
            c5.metric("Current Status", current_status)

            st.divider()

            col_shap, col_report = st.columns([1, 1])

            with col_shap:
                st.markdown("#### 🧠 Top Risk Drivers (SHAP)")
                drivers = get_top_drivers(shap_df, int(account_input), top_n=8)
                driver_df = pd.DataFrame(drivers)
                colors = ["crimson" if d["shap_value"] > 0 else "steelblue" for d in drivers]

                fig = go.Figure(go.Bar(
                    x=driver_df["shap_value"],
                    y=driver_df["feature"],
                    orientation="h",
                    marker_color=colors,
                ))
                fig.update_layout(
                    title="SHAP Values (red=↑risk, blue=↓risk)",
                    xaxis_title="SHAP Value",
                    height=400,
                    yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_report:
                st.markdown("#### 📝 AI Investigation Report")
                if tier in ["Red", "Amber"]:
                    report_path = config.REPORTS_DIR / f"account_{int(account_input)}.txt"
                    if report_path.exists():
                        content = open(report_path, encoding="utf-8").read()
                        report_text = content.split("-"*40)[-1].strip()
                        st.info(report_text)
                    else:
                        with st.spinner("Generating report..."):
                            report_text, _ = generate_report(
                                account_idx=int(account_input),
                                risk_score=row["risk_score"],
                                risk_tier=tier,
                                xgb_proba=row["xgb_proba"],
                                anomaly_score=row["anomaly_score"],
                                shap_df=shap_df,
                            )
                        st.info(report_text)
                else:
                    st.success("✅ Low risk account. No investigation required.")
                    report_text = "Low risk — no investigation required."

            st.divider()
            st.markdown("#### ⚡ Take Action")

            col_act1, col_act2, col_act3, col_act4 = st.columns(4)
            investigator = st.text_input("Investigator Name", value="Compliance Officer")
            reason = st.text_area("Reason", value="AI system flagged high-risk behavioral pattern.")

            with col_act1:
                if st.button("🔒 Freeze Account", type="primary"):
                    log_action(int(account_input), "FREEZE", tier,
                               row["risk_score"], investigator, reason)
                    generate_sar(
                        account_idx=int(account_input),
                        risk_score=row["risk_score"],
                        risk_tier=tier,
                        xgb_proba=row["xgb_proba"],
                        drivers=get_top_drivers(shap_df, int(account_input)),
                        report_text=report_text
                    )
                    st.success(f"Account #{int(account_input)} frozen. SAR generated.")

            with col_act2:
                if st.button("🔎 Flag for Review"):
                    log_action(int(account_input), "FLAG_FOR_REVIEW", tier,
                               row["risk_score"], investigator, reason)
                    st.warning(f"Account #{int(account_input)} flagged for review.")

            with col_act3:
                if st.button("👁️ Monitor"):
                    log_action(int(account_input), "MONITOR", tier,
                               row["risk_score"], investigator, reason)
                    st.info(f"Account #{int(account_input)} set to monitor.")

            with col_act4:
                if st.button("✅ Clear Account"):
                    log_action(int(account_input), "CLEAR", tier,
                               row["risk_score"], investigator, reason)
                    st.success(f"Account #{int(account_input)} cleared.")

# ══════════════════════════════════════════════
# TAB 3 — LINKAGE GRAPH
# ══════════════════════════════════════════════
with tab3:
    st.subheader("🕸️ Account Linkage Graph")
    st.markdown("""
    Accounts are linked based on **behavioral similarity** across 80 features.
    Strongly similar accounts likely share a common fraud network origin.
    Node size = risk score. Colors: 🔴 Confirmed Mule · 🟠 Red Risk · 🟡 Amber · 🟢 Green
    """)

    graph_path = config.OUTPUTS_DIR / "linkage_graph.html"

    if graph_path.exists():
        with open(graph_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=650, scrolling=False)
    else:
        with st.spinner("Building linkage graph..."):
            G, display_idx = build_linkage_graph(X, risk_df, selected_features)
            render_pyvis_graph(G)
            with open(graph_path, "r", encoding="utf-8") as f:
                html_content = f.read()
        st.components.v1.html(html_content, height=650, scrolling=False)

    st.divider()
    st.markdown("#### 🔗 Detected Account Clusters")

    if 'G' not in dir():
        G, display_idx = build_linkage_graph(X, risk_df, selected_features)

    clusters = get_connected_components(G, risk_df)
    if clusters:
        cluster_df = pd.DataFrame(clusters[:10])[
            ["size", "mule_count", "red_count", "avg_risk_score"]
        ]
        cluster_df.index += 1
        cluster_df.columns = ["Cluster Size", "Confirmed Mules",
                               "Red Accounts", "Avg Risk Score"]
        st.dataframe(cluster_df, use_container_width=True)
    else:
        st.info("No clusters found.")

# ══════════════════════════════════════════════
# TAB 4 — MODEL COMPARISON
# ══════════════════════════════════════════════
with tab4:
    st.subheader("📊 Model Comparison")
    st.markdown("Three models trained and evaluated at threshold = 0.35")

    comparison_data = {
        "Model": [
            "Model A — Statistical (80 features)",
            "Model B — Domain-Informed (98 features)",
            "🏆 Ensemble A+B — Final Model"
        ],
        "Features Used": [80, 98, "80 + 98 (60/40 blend)"],
        "Mules Caught": ["73 / 81", "69 / 81", "73 / 81"],
        "Recall": ["90.1%", "85.2%", "90.1%"],
        "False Alarms": [40, 16, 25],
        "Total Flagged": [113, 85, 98],
        "Workload Reduction": ["98.8%", "99.1%", "98.9%"],
    }

    comp_df = pd.DataFrame(comparison_data)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### Key Insight")
    st.info("""
    **Model A** (statistical) catches the most mules (90.1%) using purely data-driven feature selection.

    **Model B** (domain-informed) uses the bank's expert feature list — lower recall but 60% fewer false alarms.

    **Ensemble** combines both: preserves Model A's 90.1% recall while reducing false alarms by 37.5% vs Model A alone.

    This demonstrates that **statistical signal and domain knowledge are complementary** —
    neither alone is optimal; the ensemble captures the best of both.
    """)

    st.divider()
    st.markdown("#### Feature Selection Story")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Original Features", "3,922")
        st.metric("After Missingness Filter", "3,406")
        st.metric("After Variance Filter", "3,125")
    with col2:
        st.metric("Statistical Top-80", "80")
        st.metric("+ Bank Domain Features", "18")
        st.metric("Final Combined (Model B)", "98")

# ══════════════════════════════════════════════
# TAB 5 — MODEL INSIGHTS
# ══════════════════════════════════════════════
with tab5:
    st.subheader("Model Performance Insights")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Risk Score Distribution")
        fig_dist = px.histogram(
            risk_df,
            x="risk_score",
            color=risk_df["true_label"].map({1: "Mule", 0: "Normal"}),
            nbins=50,
            barmode="overlay",
            opacity=0.7,
            color_discrete_map={"Mule": "crimson", "Normal": "steelblue"},
            labels={"color": "Account Type", "risk_score": "Risk Score"}
        )
        fig_dist.add_vline(x=0.3, line_dash="dash",
                           line_color="orange", annotation_text="Amber (0.3)")
        fig_dist.add_vline(x=0.6, line_dash="dash",
                           line_color="red", annotation_text="Red (0.6)")
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_b:
        st.markdown("#### Confusion Matrix")
        cm_img = config.OUTPUTS_DIR / "confusion_matrix.png"
        if cm_img.exists():
            st.image(str(cm_img), use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("#### Precision-Recall Curve")
        pr_img = config.OUTPUTS_DIR / "precision_recall_curve.png"
        if pr_img.exists():
            st.image(str(pr_img), use_container_width=True)

    with col_d:
        st.markdown("#### SHAP Feature Importance")
        shap_img = config.OUTPUTS_DIR / "shap_summary.png"
        if shap_img.exists():
            st.image(str(shap_img), use_container_width=True)

    st.divider()
    st.markdown("#### 💼 Business Impact Summary")
    b1, b2, b3, b4, b5 = st.columns(5)
    b1.metric("Total Accounts", "9,082")
    b2.metric("Mules Detected", "73 / 81")
    b3.metric("Recall", "90.1%")
    b4.metric("Accounts Flagged", "98")
    b5.metric("Workload Reduction", "98.9%")

# ══════════════════════════════════════════════
# TAB 6 — AUDIT TRAIL
# ══════════════════════════════════════════════
with tab6:
    st.subheader("🔒 Investigator Audit Trail")
    st.markdown("All investigator actions are logged here for compliance and traceability.")

    audit_df = get_audit_trail()

    if audit_df.empty:
        st.info("No actions logged yet. Use Account Lookup to take action on flagged accounts.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Actions", len(audit_df))
        col2.metric("Frozen Accounts", len(audit_df[audit_df["action"] == "FREEZE"]))
        col3.metric("Pending Reviews", len(audit_df[audit_df["status"] == "PENDING"]))

        st.divider()

        action_filter = st.multiselect(
            "Filter by Action",
            ["FREEZE", "FLAG_FOR_REVIEW", "MONITOR", "CLEAR", "ESCALATE"],
            default=["FREEZE", "FLAG_FOR_REVIEW", "ESCALATE"]
        )

        filtered_audit = audit_df[audit_df["action"].isin(action_filter)]
        st.dataframe(filtered_audit, use_container_width=True, height=400)

        st.download_button(
            "⬇️ Download Audit Trail CSV",
            audit_df.to_csv(index=False),
            file_name="audit_trail.csv",
            mime="text/csv"
        )

        sar_dir = config.OUTPUTS_DIR / "SAR_reports"
        if sar_dir.exists():
            sar_files = list(sar_dir.glob("*.txt"))
            if sar_files:
                st.divider()
                st.markdown("#### 📄 Generated SAR Reports")
                selected_sar = st.selectbox(
                    "Select SAR to view",
                    [f.name for f in sar_files]
                )
                if selected_sar:
                    sar_content = open(sar_dir / selected_sar, encoding="utf-8").read()
                    st.code(sar_content, language=None)