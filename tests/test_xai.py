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
