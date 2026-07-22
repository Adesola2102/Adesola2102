"""
Trains the Random Forest email phishing classifier on data/raw/email_dataset.csv
and saves the fitted model + evaluation metrics.

Usage:
    python scripts/train_email_classifier.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from phishguard.data.email_features import (  # noqa: E402
    FEATURE_NAMES,
    EmailInput,
    extract_email_features,
)
from phishguard.models.classifier import PhishingClassifier  # noqa: E402

DATA_PATH = ROOT / "data" / "raw" / "email_dataset.csv"
MODEL_PATH = ROOT / "models" / "email_classifier.joblib"
METRICS_PATH = ROOT / "reports" / "email_classifier_metrics.json"


def main() -> None:
    df = pd.read_csv(DATA_PATH).fillna("")
    print(f"Loaded {len(df)} labelled emails from {DATA_PATH}")

    emails = [EmailInput(subject=r.subject, body=r.body) for r in df.itertuples()]
    feature_rows = [extract_email_features(e) for e in emails]
    X = pd.DataFrame(feature_rows, columns=FEATURE_NAMES).to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train = pd.DataFrame(X_train, columns=FEATURE_NAMES)
    X_test = pd.DataFrame(X_test, columns=FEATURE_NAMES)

    clf = PhishingClassifier(feature_names=FEATURE_NAMES, name="email_classifier")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label="phishing"),
        "recall": recall_score(y_test, y_pred, pos_label="phishing"),
        "f1": f1_score(y_test, y_pred, pos_label="phishing"),
        "confusion_matrix": confusion_matrix(
            y_test, y_pred, labels=["legitimate", "phishing"]
        ).tolist(),
        "confusion_matrix_labels": ["legitimate", "phishing"],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_importances": dict(
            sorted(
                zip(FEATURE_NAMES, clf.rf.feature_importances_.tolist()),
                key=lambda kv: kv[1],
                reverse=True,
            )
        ),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    clf.save(MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")
    print(json.dumps({k: v for k, v in metrics.items() if k != "feature_importances"}, indent=2))
    print("\nTop 5 features:")
    for name, imp in list(metrics["feature_importances"].items())[:5]:
        print(f"  {name}: {imp:.4f}")


if __name__ == "__main__":
    main()
