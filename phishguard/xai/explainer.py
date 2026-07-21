"""
XAI Module (Chapter Three, Component 3).

Wraps SHAP's ``TreeExplainer`` (exact Shapley values for the Random Forest)
and LIME's ``LimeTabularExplainer`` (locally-faithful linear approximation)
around a fitted ``PhishingClassifier``, producing a common
``ExplanationResult`` structure that both the Streamlit UI and the LLM
Translation Layer consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from phishguard.models.classifier import PhishingClassifier

POSITIVE_LABEL = "phishing"


@dataclass
class FeatureContribution:
    feature: str
    raw_value: float
    contribution: float  # signed: positive pushes toward "phishing"

    @property
    def direction(self) -> str:
        return "phishing" if self.contribution > 0 else "legitimate"


@dataclass
class ExplanationResult:
    method: str  # "shap" | "lime"
    predicted_class: str
    confidence: float
    base_value: float
    contributions: List[FeatureContribution]

    def top(self, k: int = 3) -> List[FeatureContribution]:
        return sorted(self.contributions, key=lambda c: abs(c.contribution), reverse=True)[:k]


class PhishingExplainer:
    """Produces SHAP and LIME local explanations for a single prediction."""

    def __init__(self, classifier: PhishingClassifier, background: pd.DataFrame,
                 lime_samples: int = 500):
        self.classifier = classifier
        self.feature_names = classifier.feature_names

        self._shap_explainer = shap.TreeExplainer(classifier.rf)

        bg = classifier._to_frame(background)
        bg_transformed = classifier.transformed_input(bg)
        classes = list(classifier.classes_())
        self._positive_idx = classes.index(POSITIVE_LABEL)

        self._lime_explainer = LimeTabularExplainer(
            training_data=bg_transformed,
            feature_names=self.feature_names,
            class_names=classes,
            discretize_continuous=True,
            mode="classification",
            random_state=42,
        )
        self._lime_samples = lime_samples

    # ------------------------------------------------------------------ #
    def explain_shap(self, features: dict) -> ExplanationResult:
        frame = self.classifier._to_frame(features)
        x_transformed = self.classifier.transformed_input(frame)

        explanation = self._shap_explainer(x_transformed)
        # explanation.values shape: (n_samples, n_features, n_classes)
        values = np.asarray(explanation.values)
        if values.ndim == 3:
            per_feature = values[0, :, self._positive_idx]
            base_value = float(np.asarray(explanation.base_values)[0, self._positive_idx])
        else:  # some shap versions return a plain 2D array for binary tasks
            per_feature = values[0, :]
            base_value = float(np.asarray(explanation.base_values).ravel()[0])

        label, confidence = self.classifier.verdict(features)
        contributions = [
            FeatureContribution(
                feature=name,
                raw_value=float(x_transformed[0, i]),
                contribution=float(per_feature[i]),
            )
            for i, name in enumerate(self.feature_names)
        ]
        return ExplanationResult(
            method="shap",
            predicted_class=label,
            confidence=confidence,
            base_value=base_value,
            contributions=contributions,
        )

    # ------------------------------------------------------------------ #
    def explain_lime(self, features: dict) -> ExplanationResult:
        frame = self.classifier._to_frame(features)
        x_transformed = self.classifier.transformed_input(frame)[0]

        def predict_fn(data: np.ndarray) -> np.ndarray:
            return self.classifier.pipeline.named_steps["rf"].predict_proba(
                np.asarray(data)
            )

        lime_exp = self._lime_explainer.explain_instance(
            x_transformed,
            predict_fn,
            labels=(self._positive_idx,),
            num_features=len(self.feature_names),
            num_samples=self._lime_samples,
        )
        weight_map = dict(lime_exp.as_map()[self._positive_idx])

        label, confidence = self.classifier.verdict(features)
        contributions = [
            FeatureContribution(
                feature=name,
                raw_value=float(x_transformed[i]),
                contribution=float(weight_map.get(i, 0.0)),
            )
            for i, name in enumerate(self.feature_names)
        ]
        return ExplanationResult(
            method="lime",
            predicted_class=label,
            confidence=confidence,
            base_value=float(lime_exp.intercept[self._positive_idx]),
            contributions=contributions,
        )

    # ------------------------------------------------------------------ #
    def explain(self, features: dict) -> dict:
        return {
            "shap": self.explain_shap(features),
            "lime": self.explain_lime(features),
        }
