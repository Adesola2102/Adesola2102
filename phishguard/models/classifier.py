"""
Phishing Classification Module (Chapter Three, Component 2).

Thin, explicit wrapper around a scikit-learn ``RandomForestClassifier`` that:

* keeps track of the feature-name order (required so SHAP/LIME explanations
  can be mapped back to human-readable feature names),
* exposes a ``verdict`` convenience method returning (label, confidence)
  exactly as described in FR1/FR2 (binary output + confidence score, where
  confidence is the proportion of trees voting for the predicted class),
* is trivially picklable via joblib for reuse by the Streamlit UI and the
  evaluation scripts without retraining.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

POSITIVE_LABEL = "phishing"
NEGATIVE_LABEL = "legitimate"


@dataclass
class TrainingMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list
    n_train: int
    n_test: int

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "confusion_matrix": self.confusion_matrix,
            "n_train": self.n_train,
            "n_test": self.n_test,
        }


class PhishingClassifier:
    """Random Forest phishing/legitimate classifier over a fixed feature set."""

    def __init__(self, feature_names: Sequence[str], name: str = "classifier",
                 n_estimators: int = 300, random_state: int = 42):
        self.feature_names = list(feature_names)
        self.name = name
        self.random_state = random_state
        self.pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        self._fitted = False

    @property
    def rf(self) -> RandomForestClassifier:
        return self.pipeline.named_steps["rf"]

    def _to_frame(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X[self.feature_names]
        if isinstance(X, dict):
            return pd.DataFrame([X])[self.feature_names]
        return pd.DataFrame(X, columns=self.feature_names)

    def fit(self, X, y) -> "PhishingClassifier":
        frame = self._to_frame(X)
        self.pipeline.fit(frame, y)
        self._fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        return self.pipeline.predict(self._to_frame(X))

    def predict_proba(self, X) -> np.ndarray:
        return self.pipeline.predict_proba(self._to_frame(X))

    def classes_(self) -> np.ndarray:
        return self.rf.classes_

    def verdict(self, features: dict) -> tuple:
        """Return (label, confidence) for a single feature dict."""

        proba = self.predict_proba(features)[0]
        classes = self.classes_()
        best_idx = int(np.argmax(proba))
        return str(classes[best_idx]), float(proba[best_idx])

    def transformed_input(self, X) -> np.ndarray:
        """Apply only the imputer step - used to feed SHAP/LIME the exact
        numeric matrix the Random Forest itself consumes."""

        frame = self._to_frame(X)
        return self.pipeline.named_steps["imputer"].transform(frame)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "PhishingClassifier":
        return joblib.load(path)
