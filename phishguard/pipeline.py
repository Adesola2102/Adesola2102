"""
End-to-end pipeline tying together all five components described in
Chapter Three: feature extraction -> classification -> XAI -> translation
-> (consumed by the) user interface.

This is the single object both ``app/streamlit_app.py`` and the evaluation
scripts (``scripts/run_evaluation.py``) use, so the UI and the offline
evaluation always exercise identically-configured components.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from phishguard.data.email_features import (
    FEATURE_NAMES as EMAIL_FEATURE_NAMES,
)
from phishguard.data.email_features import EmailInput, extract_email_features
from phishguard.data.url_features import FEATURE_NAMES as URL_FEATURE_NAMES
from phishguard.data.url_features import extract_url_features
from phishguard.models.classifier import PhishingClassifier
from phishguard.translation.backend import LLMBackend
from phishguard.translation.translator import PlainLanguageTranslator, TranslationResult
from phishguard.xai.explainer import ExplanationResult, PhishingExplainer

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "raw"


@dataclass
class AnalysisResult:
    input_type: str  # "email" | "url"
    raw_input: str
    verdict: str
    confidence: float
    shap: ExplanationResult
    lime: ExplanationResult
    translation: TranslationResult


class PhishingExplanationPipeline:
    """Loads both classifiers + explainers once and serves both input types."""

    def __init__(self, backend: LLMBackend | None = None, background_size: int = 200,
                 lime_samples: int = 400):
        self.translator = PlainLanguageTranslator(backend=backend)

        self.email_classifier = PhishingClassifier.load(MODELS_DIR / "email_classifier.joblib")
        self.url_classifier = PhishingClassifier.load(MODELS_DIR / "url_classifier.joblib")

        email_bg_df = pd.read_csv(DATA_DIR / "email_dataset.csv").fillna("").sample(
            background_size, random_state=1
        )
        email_bg = pd.DataFrame(
            [
                extract_email_features(EmailInput(subject=r.subject, body=r.body))
                for r in email_bg_df.itertuples()
            ],
            columns=EMAIL_FEATURE_NAMES,
        )
        self.email_explainer = PhishingExplainer(
            self.email_classifier, email_bg, lime_samples=lime_samples
        )

        url_bg_df = pd.read_csv(DATA_DIR / "url_dataset.csv").sample(
            background_size, random_state=1
        )
        url_bg = pd.DataFrame(
            [extract_url_features(u) for u in url_bg_df["url"]], columns=URL_FEATURE_NAMES
        )
        self.url_explainer = PhishingExplainer(
            self.url_classifier, url_bg, lime_samples=lime_samples
        )

    # ------------------------------------------------------------------ #
    def analyze_email(self, email: EmailInput) -> AnalysisResult:
        features = extract_email_features(email)
        shap_result = self.email_explainer.explain_shap(features)
        lime_result = self.email_explainer.explain_lime(features)
        translation = self.translator.translate(shap_result, lime_result, input_type="email")
        return AnalysisResult(
            input_type="email",
            raw_input=email.full_text,
            verdict=shap_result.predicted_class,
            confidence=shap_result.confidence,
            shap=shap_result,
            lime=lime_result,
            translation=translation,
        )

    def analyze_url(self, url: str) -> AnalysisResult:
        features = extract_url_features(url)
        shap_result = self.url_explainer.explain_shap(features)
        lime_result = self.url_explainer.explain_lime(features)
        translation = self.translator.translate(shap_result, lime_result, input_type="url")
        return AnalysisResult(
            input_type="url",
            raw_input=url,
            verdict=shap_result.predicted_class,
            confidence=shap_result.confidence,
            shap=shap_result,
            lime=lime_result,
            translation=translation,
        )
