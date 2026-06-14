import os
import json
from groq import Groq
from src import config
from src.explainability import get_top_drivers


def build_prompt(account_idx, risk_score, risk_tier, drivers, xgb_proba, anomaly_score):
    """
    Build the LLM prompt for a single flagged account.
    """
    drivers_text = "\n".join([
        f"  - {d['feature']}: SHAP value = {d['shap_value']} ({d['direction']})"
        for d in drivers
    ])

    prompt = f"""
You are a senior fraud analyst at a bank investigating potential mule accounts.

You have been given the following risk assessment for Account #{account_idx}:

- Overall Risk Score: {risk_score:.4f} (scale 0-1, higher = more suspicious)
- Risk Tier: {risk_tier}
- XGBoost Mule Probability: {xgb_proba:.4f}
- Anomaly Score: {anomaly_score:.4f}

Top factors driving this account's risk score (SHAP analysis):
{drivers_text}

Write a concise investigation note (3-5 sentences) for this account that:
1. States the risk level and overall assessment
2. Highlights the top 2-3 suspicious factors in plain English
3. Recommends a specific next action (e.g. freeze account, request KYC documents, monitor transactions)

Write professionally, as if this note will be read by a bank compliance officer.
"""
    return prompt.strip()


def generate_report(account_idx, risk_score, risk_tier,
                    xgb_proba, anomaly_score, shap_df, top_n=5):
    """
    Generate an LLM investigation report for a single flagged account.
    """
    client = Groq(api_key=config.GROQ_API_KEY)

    drivers = get_top_drivers(shap_df, account_idx, top_n=top_n)
    prompt = build_prompt(account_idx, risk_score, risk_tier,
                          drivers, xgb_proba, anomaly_score)

    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=300,
    )

    report = response.choices[0].message.content.strip()
    return report, drivers


def generate_all_reports(risk_df, shap_df, tier_filter=["Red", "Amber"]):
    """
    Generate investigation reports for all flagged accounts (Red + Amber).
    Saves individual .txt reports to outputs/reports/.
    Returns a dict of {account_idx: report_text}.
    """
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    flagged = risk_df[risk_df["risk_tier"].isin(tier_filter)]
    print(f"Generating reports for {len(flagged)} flagged accounts...")

    all_reports = {}

    for i, (_, row) in enumerate(flagged.iterrows()):
        account_idx = int(row["account_idx"])
        try:
            report, drivers = generate_report(
                account_idx=account_idx,
                risk_score=row["risk_score"],
                risk_tier=row["risk_tier"],
                xgb_proba=row["xgb_proba"],
                anomaly_score=row["anomaly_score"],
                shap_df=shap_df,
            )

            # Save individual report
            report_path = config.REPORTS_DIR / f"account_{account_idx}.txt"
            with open(report_path, "w") as f:
                f.write(f"ACCOUNT: {account_idx}\n")
                f.write(f"RISK TIER: {row['risk_tier']}\n")
                f.write(f"RISK SCORE: {row['risk_score']:.4f}\n")
                f.write(f"TRUE LABEL: {'MULE' if row['true_label'] == 1 else 'NORMAL'}\n")
                f.write("-" * 40 + "\n")
                f.write(report + "\n")

            all_reports[account_idx] = report

            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(flagged)} reports done...")

        except Exception as e:
            print(f"  Failed for account {account_idx}: {e}")
            all_reports[account_idx] = "Report generation failed."

    print(f"All reports saved to {config.REPORTS_DIR}")
    return all_reports