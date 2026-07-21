from phishguard.translation.backend import TemplateBackend
from phishguard.translation.phrases import describe, is_coherent
from phishguard.translation.translator import PlainLanguageTranslator
from phishguard.xai.explainer import ExplanationResult, FeatureContribution


def _explanation(method: str, verdict: str, confidence: float, contributions) -> ExplanationResult:
    return ExplanationResult(
        method=method,
        predicted_class=verdict,
        confidence=confidence,
        base_value=0.5,
        contributions=[
            FeatureContribution(feature=f, raw_value=v, contribution=c)
            for f, v, c in contributions
        ],
    )


def test_is_coherent_flags_present_risk_feature_as_phishing():
    # urgent_keyword_count present (>0) is normally a phishing signal
    assert is_coherent("urgent_keyword_count", 3.0, "phishing") is True
    assert is_coherent("urgent_keyword_count", 3.0, "legitimate") is False


def test_is_coherent_flags_absent_feature_as_legitimate():
    assert is_coherent("urgent_keyword_count", 0.0, "legitimate") is True
    assert is_coherent("urgent_keyword_count", 0.0, "phishing") is False


def test_is_coherent_handles_inverse_polarity_feature_has_https():
    # HTTPS present is normally the *safer* signal
    assert is_coherent("has_https", 1.0, "legitimate") is True
    assert is_coherent("has_https", 1.0, "phishing") is False
    assert is_coherent("has_https", 0.0, "phishing") is True


def test_describe_never_crashes_on_unknown_feature():
    text = describe("some_future_feature_xyz", 1.0)
    assert "some future feature xyz" in text


def test_template_backend_produces_single_sentence_mentioning_verdict():
    backend = TemplateBackend()
    context = {
        "verdict": "phishing",
        "confidence": 0.87,
        "input_noun": "email",
        "factors": [("urgent_keyword_count", 3.0, True), ("credential_keyword_count", 2.0, True)],
    }
    sentence = backend.generate("unused prompt", context)
    assert "PHISHING" in sentence
    assert "87%" in sentence
    assert sentence.count(".") == 1  # single concise sentence (FR5)


def test_translator_filters_incoherent_factor_from_final_sentence():
    """Regression test for the bug where an absent feature (e.g. no
    attachment mentioned) was described as present just because SHAP/LIME
    ranked it highly for a noisy/spurious reason."""

    shap_result = _explanation(
        "shap",
        "phishing",
        0.9,
        [
            ("urgent_keyword_count", 5.0, 0.3),
            # attachment_keyword is 0 (absent) yet has a *positive* SHAP
            # contribution toward phishing here - an incoherent/spurious
            # signal that must not surface as "references an attachment".
            ("attachment_keyword", 0.0, 0.25),
            ("exclamation_count", 4.0, 0.2),
        ],
    )
    lime_result = _explanation(
        "lime",
        "phishing",
        0.9,
        [
            ("urgent_keyword_count", 5.0, 0.4),
            ("attachment_keyword", 0.0, 0.3),
            ("exclamation_count", 4.0, 0.25),
        ],
    )

    translator = PlainLanguageTranslator(backend=TemplateBackend())
    result = translator.translate(shap_result, lime_result, input_type="email")

    used_features = [f for f, _, _ in result.factors_used]
    assert "attachment_keyword" not in used_features
    assert "attachment" not in result.sentence.lower()
    assert "urgent_keyword_count" in used_features


def test_translator_surfaces_coherent_factors_for_legitimate_verdict():
    shap_result = _explanation(
        "shap",
        "legitimate",
        0.95,
        [("urgent_keyword_count", 0.0, -0.3), ("has_https", 1.0, -0.2)],
    )
    lime_result = _explanation(
        "lime",
        "legitimate",
        0.95,
        [("urgent_keyword_count", 0.0, -0.35), ("has_https", 1.0, -0.25)],
    )
    translator = PlainLanguageTranslator(backend=TemplateBackend())
    result = translator.translate(shap_result, lime_result, input_type="url")

    assert result.verdict == "legitimate"
    assert "LEGITIMATE" in result.sentence
    assert "does not use urgent" in result.sentence or "HTTPS" in result.sentence
