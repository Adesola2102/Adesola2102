import numpy as np
import pandas as pd

from phishguard.models.classifier import PhishingClassifier


def test_verdict_matches_learned_separable_pattern(tiny_classifier):
    label, confidence = tiny_classifier.verdict({"signal_a": 0.95, "signal_b": 0.1, "noise": 0.5})
    assert label == "phishing"
    assert confidence > 0.5

    label, confidence = tiny_classifier.verdict({"signal_a": 0.05, "signal_b": 0.9, "noise": 0.5})
    assert label == "legitimate"
    assert confidence > 0.5


def test_predict_proba_sums_to_one(tiny_classifier):
    proba = tiny_classifier.predict_proba({"signal_a": 0.5, "signal_b": 0.5, "noise": 0.5})
    assert proba.shape == (1, 2)
    assert np.isclose(proba.sum(), 1.0)


def test_feature_importance_favours_true_signal(tiny_classifier):
    importances = dict(zip(tiny_classifier.feature_names, tiny_classifier.rf.feature_importances_))
    assert importances["signal_a"] > importances["noise"]
    assert importances["signal_a"] > importances["signal_b"]


def test_save_and_load_round_trip(tmp_path, tiny_classifier):
    path = tmp_path / "clf.joblib"
    tiny_classifier.save(path)
    loaded = PhishingClassifier.load(path)

    sample = {"signal_a": 0.9, "signal_b": 0.2, "noise": 0.4}
    assert loaded.verdict(sample) == tiny_classifier.verdict(sample)


def test_accepts_dataframe_dict_and_ndarray_inputs(tiny_classifier):
    row = {"signal_a": 0.8, "signal_b": 0.3, "noise": 0.5}
    from_dict = tiny_classifier.predict(row)
    from_frame = tiny_classifier.predict(pd.DataFrame([row]))
    from_array = tiny_classifier.predict(np.array([[0.8, 0.3, 0.5]]))
    assert from_dict[0] == from_frame[0] == from_array[0]
