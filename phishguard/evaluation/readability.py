"""
Evaluation utilities for Objective V: "Evaluate the readability, perceived
credibility, and impact on decision-making speed of the generated plain
language explanations by comparing them with raw SHAP/LIME explanations
among non-expert users."

Two complementary evaluation instruments are provided:

1. Automated readability metrics (this module) - an objective, fully
   reproducible proxy for "how hard is this explanation to read", computed
   with the standard Flesch Reading Ease / Flesch-Kincaid Grade Level
   formulas over (a) the raw SHAP/LIME feature-importance output rendered
   as text, exactly as a data-scientist would read it, versus (b) the
   plain-language sentence produced by the translation layer.

2. A structured non-expert questionnaire (see
   docs/user_evaluation_questionnaire.md) covering perceived credibility and
   decision-making speed/confidence - dimensions that genuinely require
   human participants and could not be collected inside this automated
   build environment. That instrument is provided ready for the student to
   administer with real IT-helpdesk/SOC-analyst participants, exactly as
   described in Chapter Three's Evaluation phase.

Reporting both together, rather than only the automated proxy, is a
deliberate methodological choice: readability formulas measure surface-level
linguistic complexity, not comprehension or trust, so the automated numbers
below should be read as necessary-but-not-sufficient evidence for Objective V.
"""

from __future__ import annotations

import textstat

from phishguard.xai.explainer import ExplanationResult


def raw_shap_text(result: ExplanationResult, top_k: int = 5) -> str:
    """Render a SHAP/LIME explanation exactly as a data scientist would read
    it in a notebook: signed numeric feature-importance values, no natural
    language. This is the "raw XAI output" baseline that the plain-language
    sentence is compared against."""

    top = result.top(top_k)
    parts = [f"{c.feature}: {c.contribution:+.3f}" for c in top]
    return (
        f"[{result.method.upper()}] predicted_class={result.predicted_class}, "
        f"confidence={result.confidence:.3f}, base_value={result.base_value:.3f}, "
        f"top_features=({', '.join(parts)})"
    )


def readability_metrics(text: str) -> dict:
    """Standard readability formulas (higher Flesch Reading Ease = easier;
    lower Flesch-Kincaid Grade = easier)."""

    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "word_count": textstat.lexicon_count(text, removepunct=True),
        "sentence_count": max(textstat.sentence_count(text), 1),
    }
