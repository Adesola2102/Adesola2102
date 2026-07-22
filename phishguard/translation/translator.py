"""
LLM Translation Layer orchestration (Chapter Three, Component 4).

``PlainLanguageTranslator`` reconciles the SHAP and LIME explanations for a
single prediction into one ranked list of contributing factors (Key Feature
#1 in the proposal: "dual XAI usage ... reconciled into a single translated
explanation"), builds the engineered prompt described in Chapter Three, and
delegates text generation to whichever ``LLMBackend`` it was configured
with.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from phishguard.translation.backend import LLMBackend, get_default_backend
from phishguard.translation.phrases import is_coherent
from phishguard.xai.explainer import ExplanationResult

PROMPT_TEMPLATE = (
    "You are a cybersecurity assistant. Based on {input_desc}, a phishing "
    "detection system found the result {verdict_upper} with {confidence_pct}% "
    "confidence. Top contributing factors (from SHAP and LIME feature-importance "
    "analysis) include: {factor_list}. Write one simple sentence explaining why "
    "this {input_noun} was flagged, for an IT helpdesk person who is not "
    "technically knowledgeable."
)


@dataclass
class TranslationResult:
    sentence: str
    verdict: str
    confidence: float
    factors_used: List[Tuple[str, float, bool]]
    prompt: str
    backend_name: str


def _combine_contributions(
    shap_result: ExplanationResult, lime_result: ExplanationResult, top_k: int = 3
) -> List[Tuple[str, float, bool]]:
    """Reconcile SHAP + LIME into one ranked list of (feature, value, is_risk).

    Both explainers' per-feature contributions are min-max normalised to
    [-1, 1] independently (SHAP and LIME weights live on different numeric
    scales) and then averaged, so a feature that both methods agree is
    important ranks above one only one method flags strongly.
    """

    def _normalise(result: ExplanationResult) -> dict:
        max_abs = max((abs(c.contribution) for c in result.contributions), default=1.0)
        max_abs = max_abs or 1.0
        return {c.feature: c.contribution / max_abs for c in result.contributions}

    shap_norm = _normalise(shap_result)
    lime_norm = _normalise(lime_result)

    combined = {}
    values = {c.feature: c.raw_value for c in shap_result.contributions}
    for feature in shap_norm:
        s = shap_norm.get(feature, 0.0)
        l = lime_norm.get(feature, 0.0)
        combined[feature] = (s + l) / 2.0

    predicted_positive = shap_result.predicted_class == "phishing"
    predicted_verdict = shap_result.predicted_class
    # Only surface factors that (a) point the *same* direction as the
    # verdict, and (b) are coherent - i.e. the feature's actual value
    # plausibly explains that direction, rather than being a spurious
    # small-sample correlation SHAP/LIME happened to rank highly (see
    # phrases.is_coherent / Chapter Two's discussion of explanation
    # faithfulness, Cambria et al., 2023).
    relevant = [
        (feature, values[feature], score)
        for feature, score in combined.items()
        if (score > 0) == predicted_positive
        and abs(score) > 1e-6
        and is_coherent(feature, values[feature], predicted_verdict)
    ]
    relevant.sort(key=lambda item: abs(item[2]), reverse=True)
    return [(feature, value, predicted_positive) for feature, value, _ in relevant[:top_k]]


class PlainLanguageTranslator:
    """Turns (SHAP result, LIME result) into a single plain-language sentence."""

    def __init__(self, backend: LLMBackend | None = None, top_k: int = 3):
        self.backend = backend or get_default_backend()
        self.top_k = top_k

    def translate(
        self,
        shap_result: ExplanationResult,
        lime_result: ExplanationResult,
        input_type: str = "email",
    ) -> TranslationResult:
        factors = _combine_contributions(shap_result, lime_result, self.top_k)
        input_noun = "email" if input_type == "email" else "web link"
        input_desc = (
            f"an analysis of a submitted {input_noun}"
        )

        factor_descriptions = ", ".join(
            f"{feature} ({value:.2f})" for feature, value, _ in factors
        ) or "overall statistical pattern of the input"

        prompt = PROMPT_TEMPLATE.format(
            input_desc=input_desc,
            verdict_upper=shap_result.predicted_class.upper(),
            confidence_pct=round(shap_result.confidence * 100),
            factor_list=factor_descriptions,
            input_noun=input_noun,
        )

        context = {
            "verdict": shap_result.predicted_class,
            "confidence": shap_result.confidence,
            "input_noun": input_noun,
            "factors": factors,
        }

        sentence = self.backend.generate(prompt, context)

        return TranslationResult(
            sentence=sentence,
            verdict=shap_result.predicted_class,
            confidence=shap_result.confidence,
            factors_used=factors,
            prompt=prompt,
            backend_name=self.backend.name,
        )
