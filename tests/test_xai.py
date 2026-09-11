from phishguard.xai.explainer import PhishingExplainer


def test_shap_explanation_has_all_features(tiny_classifier, tiny_background):
    explainer = PhishingExplainer(tiny_classifier, tiny_background, lime_samples=200)
    result = explainer.explain_shap({"signal_a": 0.95, "signal_b": 0.1, "noise": 0.5})
    assert {c.feature for c in result.contributions} == {"signal_a", "signal_b", "noise"}
    assert result.predicted_class == "phishing"


def test_shap_top_feature_is_the_true_signal(tiny_classifier, tiny_background):
    explainer = PhishingExplainer(tiny_classifier, tiny_background, lime_samples=200)
    result = explainer.explain_shap({"signal_a": 0.95, "signal_b": 0.9, "noise": 0.9})
    top = result.top(1)[0]
    assert top.feature == "signal_a"
    assert top.direction == "phishing"


def test_lime_explanation_has_all_features(tiny_classifier, tiny_background):
    explainer = PhishingExplainer(tiny_classifier, tiny_background, lime_samples=200)
    result = explainer.explain_lime({"signal_a": 0.95, "signal_b": 0.1, "noise": 0.5})
    assert {c.feature for c in result.contributions} == {"signal_a", "signal_b", "noise"}


def test_legitimate_prediction_has_negative_top_contribution(tiny_classifier, tiny_background):
    explainer = PhishingExplainer(tiny_classifier, tiny_background, lime_samples=200)
    result = explainer.explain_shap({"signal_a": 0.05, "signal_b": 0.5, "noise": 0.5})
    assert result.predicted_class == "legitimate"
    top = result.top(1)[0]
    assert top.feature == "signal_a"
    assert top.direction == "legitimate"


def test_explain_returns_both_methods(tiny_classifier, tiny_background):
    explainer = PhishingExplainer(tiny_classifier, tiny_background, lime_samples=200)
    both = explainer.explain({"signal_a": 0.8, "signal_b": 0.2, "noise": 0.5})
    assert set(both.keys()) == {"shap", "lime"}
    assert both["shap"].method == "shap"
    assert both["lime"].method == "lime"


def test_lime_explanation_is_reproducible_across_repeated_calls(
    tiny_classifier, tiny_background
):
    """Regression test for Section 4.4.9.

    LimeTabularExplainer holds its random_state as a stateful RandomState that
    advances on each explain_instance call, so seeding it only at construction
    left repeated explanations of the same input non-reproducible. The explainer
    re-seeds per call; this test fails if that is removed.
    """
    explainer = PhishingExplainer(tiny_classifier, tiny_background, lime_samples=200)
    features = {"signal_a": 0.8, "signal_b": 0.2, "noise": 0.5}

    runs = [
        tuple(
            (c.feature, round(c.contribution, 10))
            for c in explainer.explain_lime(features).contributions
        )
        for _ in range(4)
    ]

    assert len(set(runs)) == 1, "LIME attributions drifted across repeated calls"
