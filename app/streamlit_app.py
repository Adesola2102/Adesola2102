"""
User Interface (Chapter Three, Component 5).

A Streamlit application that lets a user submit either an email or a URL and
see, side by side (FR6):

* the classifier's verdict and confidence score,
* the raw SHAP feature-importance values (bar chart + table),
* the raw LIME feature-importance values (bar chart + table),
* the single plain-language explanation produced by the LLM Translation
  Layer.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from phishguard.data.email_features import EmailInput  # noqa: E402
from phishguard.pipeline import PhishingExplanationPipeline  # noqa: E402

st.set_page_config(
    page_title="Explainable Phishing Translation Layer",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading classifiers, SHAP/LIME explainers, and translation layer...")
def load_pipeline() -> PhishingExplanationPipeline:
    return PhishingExplanationPipeline()


def contributions_frame(result) -> pd.DataFrame:
    rows = [
        {"feature": c.feature, "value": round(c.raw_value, 3), "contribution": round(c.contribution, 4)}
        for c in sorted(result.contributions, key=lambda c: abs(c.contribution), reverse=True)[:10]
    ]
    return pd.DataFrame(rows)


def render_result(analysis) -> None:
    verdict = analysis.verdict
    confidence = analysis.confidence
    colour = "🔴" if verdict == "phishing" else "🟢"

    st.subheader(f"{colour} Verdict: {verdict.upper()}  ({confidence:.1%} confidence)")

    st.markdown("### 🗣️ Plain-Language Explanation (LLM Translation Layer)")
    st.info(analysis.translation.sentence)
    st.caption(f"Translation backend: `{analysis.translation.backend_name}`")

    st.markdown("### 🔬 Raw XAI Output (for comparison)")
    shap_col, lime_col = st.columns(2)

    with shap_col:
        st.markdown("**SHAP (TreeExplainer) — top contributing features**")
        shap_df = contributions_frame(analysis.shap)
        st.bar_chart(shap_df.set_index("feature")["contribution"])
        st.dataframe(shap_df, hide_index=True, use_container_width=True)

    with lime_col:
        st.markdown("**LIME (Tabular Explainer) — top contributing features**")
        lime_df = contributions_frame(analysis.lime)
        st.bar_chart(lime_df.set_index("feature")["contribution"])
        st.dataframe(lime_df, hide_index=True, use_container_width=True)

    with st.expander("Show the exact prompt sent to the translation backend"):
        st.code(analysis.translation.prompt, language="text")


def main() -> None:
    st.title("🛡️ Explainable Translation Layer for Phishing Alerts")
    st.caption(
        "Converts SHAP/LIME phishing-detection outputs into a human-readable "
        "explanation using a lightweight, locally-run translation layer."
    )

    pipeline = load_pipeline()

    tab_email, tab_url = st.tabs(["📧 Analyse an Email", "🔗 Analyse a URL"])

    with tab_email:
        st.markdown("Paste an email's subject and body below.")
        example = st.selectbox(
            "Or load an example",
            [
                "(none)",
                "Phishing example - urgent account suspension",
                "Legitimate example - internal meeting notes",
            ],
            key="email_example",
        )
        subject_default, body_default = "", ""
        if example == "Phishing example - urgent account suspension":
            subject_default = "URGENT: Your account will be suspended"
            body_default = (
                "Dear customer, we detected unusual activity on your account. "
                "Click here immediately to verify your password: "
                "http://secure-update.account-verify.xyz/login or your account "
                "will be suspended within 24 hours!!!"
            )
        elif example == "Legitimate example - internal meeting notes":
            subject_default = "Notes from yesterday's planning meeting"
            body_default = (
                "Hi team, please find attached the notes from yesterday's "
                "planning meeting. Let me know if you have any questions. "
                "Thanks, Sarah"
            )

        subject = st.text_input("Subject", value=subject_default, key="email_subject")
        body = st.text_area("Body", value=body_default, height=180, key="email_body")

        if st.button("Analyse Email", type="primary", key="analyse_email"):
            if not subject.strip() and not body.strip():
                st.warning("Please enter an email subject or body.")
            else:
                with st.spinner("Classifying, computing SHAP/LIME, translating..."):
                    result = pipeline.analyze_email(EmailInput(subject=subject, body=body))
                render_result(result)

    with tab_url:
        st.markdown("Paste a URL below.")
        example_url = st.selectbox(
            "Or load an example",
            [
                "(none)",
                "Phishing example - lookalike domain",
                "Legitimate example - well-known site",
            ],
            key="url_example",
        )
        url_default = ""
        if example_url == "Phishing example - lookalike domain":
            url_default = "http://secure-paypal-verification.account-update.xyz/login.php?id=93820"
        elif example_url == "Legitimate example - well-known site":
            url_default = "https://www.wikipedia.org/"

        url = st.text_input("URL", value=url_default, key="url_input")

        if st.button("Analyse URL", type="primary", key="analyse_url"):
            if not url.strip():
                st.warning("Please enter a URL.")
            else:
                with st.spinner("Classifying, computing SHAP/LIME, translating..."):
                    result = pipeline.analyze_url(url.strip())
                render_result(result)

    st.divider()
    st.caption(
        "Prototype system built for the MIT Professional Master's Project: "
        "\"An Explainable Translation Layer for Converting SHAP/LIME Outputs "
        "into Human-Readable Phishing Alerts Using a Lightweight Open-Source "
        "LLM\" — runs fully offline/locally (no external API calls)."
    )


if __name__ == "__main__":
    main()
