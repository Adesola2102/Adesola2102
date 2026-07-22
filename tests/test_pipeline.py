"""
Integration tests for the full five-component pipeline (Chapter Five,
"Testing Strategy" -> integration testing / user-acceptance-style golden
path checks). These require the trained classifiers to exist under
models/ (produced by scripts/train_email_classifier.py and
scripts/train_url_classifier.py) and are skipped otherwise.
"""

from __future__ import annotations

import pytest

from phishguard.data.email_features import EmailInput
from phishguard.pipeline import PhishingExplanationPipeline
from tests.conftest import requires_trained_models


@pytest.fixture(scope="module")
def pipeline() -> PhishingExplanationPipeline:
    return PhishingExplanationPipeline(background_size=100, lime_samples=200)


@requires_trained_models
def test_obvious_phishing_url_is_flagged(pipeline):
    result = pipeline.analyze_url(
        "http://secure-paypal-verification.account-update.xyz/login.php?id=93820"
    )
    assert result.verdict == "phishing"
    assert result.confidence > 0.5
    assert "PHISHING" in result.translation.sentence
    assert result.translation.sentence.strip().endswith(".")


@requires_trained_models
def test_well_known_domain_is_legitimate(pipeline):
    result = pipeline.analyze_url("https://www.wikipedia.org/")
    assert result.verdict == "legitimate"
    assert "LEGITIMATE" in result.translation.sentence


@requires_trained_models
def test_obvious_phishing_email_is_flagged(pipeline):
    email = EmailInput(
        subject="URGENT: Your account will be suspended",
        body=(
            "Dear customer, we detected unusual activity on your account. "
            "Click here immediately to verify your password: "
            "http://secure-update.account-verify.xyz/login or your account "
            "will be suspended within 24 hours!!!"
        ),
    )
    result = pipeline.analyze_email(email)
    assert result.verdict == "phishing"
    assert "PHISHING" in result.translation.sentence


@requires_trained_models
def test_calm_internal_email_is_legitimate(pipeline):
    email = EmailInput(
        subject="Notes from yesterday's planning meeting",
        body=(
            "Hi team, please find attached the notes from yesterday's "
            "planning meeting. Let me know if you have any questions. "
            "Thanks, Sarah"
        ),
    )
    result = pipeline.analyze_email(email)
    assert result.verdict == "legitimate"


@requires_trained_models
def test_translation_sentence_is_a_single_concise_sentence(pipeline):
    result = pipeline.analyze_url("http://bit.ly/free-prize-claim-now")
    sentence = result.translation.sentence
    assert sentence.count(".") <= 2  # allows e.g. "24 hours" style numbers, but stays terse
    assert len(sentence.split()) < 60


@requires_trained_models
def test_shap_and_lime_both_available_for_ui_side_by_side_view(pipeline):
    result = pipeline.analyze_url("http://192.168.1.10/wp-login.php")
    assert len(result.shap.contributions) > 0
    assert len(result.lime.contributions) > 0
    assert result.shap.predicted_class == result.lime.predicted_class == result.verdict
