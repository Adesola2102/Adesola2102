"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from phishguard.models.classifier import PhishingClassifier  # noqa: E402

TINY_FEATURES = ["signal_a", "signal_b", "noise"]


@pytest.fixture()
def tiny_classifier() -> PhishingClassifier:
    """A small, perfectly-separable synthetic classifier for fast unit tests
    of the classifier/XAI/translation layers, independent of the real
    trained models."""

    rng = np.random.RandomState(0)
    n = 200
    signal_a = rng.uniform(0, 1, n)
    signal_b = rng.uniform(0, 1, n)
    noise = rng.uniform(0, 1, n)
    # phishing whenever signal_a is high; signal_b and noise are irrelevant
    label = np.where(signal_a > 0.6, "phishing", "legitimate")

    X = pd.DataFrame({"signal_a": signal_a, "signal_b": signal_b, "noise": noise})
    clf = PhishingClassifier(feature_names=TINY_FEATURES, name="tiny", n_estimators=50)
    clf.fit(X, label)
    return clf


@pytest.fixture()
def tiny_background() -> pd.DataFrame:
    rng = np.random.RandomState(1)
    n = 60
    return pd.DataFrame(
        {
            "signal_a": rng.uniform(0, 1, n),
            "signal_b": rng.uniform(0, 1, n),
            "noise": rng.uniform(0, 1, n),
        }
    )


MODELS_DIR = ROOT / "models"


def _models_present() -> bool:
    return (MODELS_DIR / "email_classifier.joblib").exists() and (
        MODELS_DIR / "url_classifier.joblib"
    ).exists()


requires_trained_models = pytest.mark.skipif(
    not _models_present(),
    reason="Trained models not found - run scripts/train_email_classifier.py "
    "and scripts/train_url_classifier.py first.",
)
