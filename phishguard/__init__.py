"""
phishguard
==========

An Explainable Translation Layer for Converting SHAP/LIME Outputs into
Human-Readable Phishing Alerts Using a Lightweight Open-Source LLM.

This package implements the five-component pipeline described in Chapter
Three of the project report:

    1. Data Ingestion & Feature Extraction   -> phishguard.data
    2. Phishing Classification (Random Forest) -> phishguard.models
    3. XAI Module (SHAP + LIME)                -> phishguard.xai
    4. LLM Translation Layer                   -> phishguard.translation
    5. User Interface                          -> app/streamlit_app.py

``phishguard.pipeline`` wires all four backend components together into a
single ``PhishingExplanationPipeline`` used by both the Streamlit UI and the
evaluation scripts.
"""

__version__ = "0.1.0"
