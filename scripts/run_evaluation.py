"""
Objective V evaluation: compares the plain-language translation layer
output against the raw SHAP/LIME output it was derived from, using
automated readability metrics, and writes a Markdown + JSON evaluation
report consumed directly by Chapter Five ("Testing, Results & Evaluation").

Usage:
    python scripts/run_evaluation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from phishguard.data.email_features import EmailInput  # noqa: E402
from phishguard.evaluation.readability import raw_shap_text, readability_metrics  # noqa: E402
from phishguard.pipeline import PhishingExplanationPipeline  # noqa: E402

N_SAMPLES_PER_CLASS = 15
REPORTS_DIR = ROOT / "reports"


def _avg(rows: list, key: str) -> float:
    return sum(r[key] for r in rows) / len(rows)


def _stratified_sample(df: pd.DataFrame, n_per_class: int) -> pd.DataFrame:
    parts = [
        group.sample(min(n_per_class, len(group)), random_state=7)
        for _, group in df.groupby("label")
    ]
    return pd.concat(parts).reset_index(drop=True)


def evaluate_urls(pipeline: PhishingExplanationPipeline) -> list:
    df = pd.read_csv(ROOT / "data" / "raw" / "url_dataset.csv")
    sample = _stratified_sample(df, N_SAMPLES_PER_CLASS)

    rows = []
    for _, row in sample.iterrows():
        result = pipeline.analyze_url(row["url"])
        raw_text = raw_shap_text(result.shap)
        translated_text = result.translation.sentence
        rows.append(
            {
                "input_type": "url",
                "input": row["url"],
                "true_label": row["label"],
                "predicted_label": result.verdict,
                "confidence": result.confidence,
                "raw_shap_text": raw_text,
                "translated_sentence": translated_text,
                "raw": readability_metrics(raw_text),
                "translated": readability_metrics(translated_text),
            }
        )
    return rows


def evaluate_emails(pipeline: PhishingExplanationPipeline) -> list:
    df = pd.read_csv(ROOT / "data" / "raw" / "email_dataset.csv").fillna("")
    sample = _stratified_sample(df, N_SAMPLES_PER_CLASS)

    rows = []
    for _, row in sample.iterrows():
        email = EmailInput(subject=row["subject"], body=row["body"])
        result = pipeline.analyze_email(email)
        raw_text = raw_shap_text(result.shap)
        translated_text = result.translation.sentence
        rows.append(
            {
                "input_type": "email",
                "input": (row["subject"] or "")[:80],
                "true_label": row["label"],
                "predicted_label": result.verdict,
                "confidence": result.confidence,
                "raw_shap_text": raw_text,
                "translated_sentence": translated_text,
                "raw": readability_metrics(raw_text),
                "translated": readability_metrics(translated_text),
            }
        )
    return rows


def summarise(rows: list) -> dict:
    accuracy = sum(1 for r in rows if r["predicted_label"] == r["true_label"]) / len(rows)
    return {
        "n": len(rows),
        "accuracy_on_sample": accuracy,
        "raw_flesch_reading_ease_avg": _avg([r["raw"] for r in rows], "flesch_reading_ease"),
        "translated_flesch_reading_ease_avg": _avg(
            [r["translated"] for r in rows], "flesch_reading_ease"
        ),
        "raw_flesch_kincaid_grade_avg": _avg([r["raw"] for r in rows], "flesch_kincaid_grade"),
        "translated_flesch_kincaid_grade_avg": _avg(
            [r["translated"] for r in rows], "flesch_kincaid_grade"
        ),
        "raw_word_count_avg": _avg([r["raw"] for r in rows], "word_count"),
        "translated_word_count_avg": _avg([r["translated"] for r in rows], "word_count"),
    }


def render_markdown_report(url_rows: list, email_rows: list, url_summary: dict,
                            email_summary: dict) -> str:
    lines = []
    lines.append("# Objective V Evaluation Report")
    lines.append("")
    lines.append(
        "Automated comparison of the LLM Translation Layer's plain-language "
        "sentences against the raw SHAP feature-importance output they were "
        "derived from, using standard readability formulas (Flesch Reading "
        "Ease: higher = easier; Flesch-Kincaid Grade: lower = easier)."
    )
    lines.append("")

    for name, summary in [("URL classifier", url_summary), ("Email classifier", email_summary)]:
        lines.append(f"## {name} sample (n={summary['n']})")
        lines.append("")
        lines.append(f"- Sample accuracy on this evaluation batch: **{summary['accuracy_on_sample']:.1%}**")
        lines.append(
            f"- Flesch Reading Ease — raw SHAP text: **{summary['raw_flesch_reading_ease_avg']:.1f}** "
            f"vs translated sentence: **{summary['translated_flesch_reading_ease_avg']:.1f}** "
            f"(higher is easier to read)"
        )
        lines.append(
            f"- Flesch-Kincaid Grade Level — raw SHAP text: **{summary['raw_flesch_kincaid_grade_avg']:.1f}** "
            f"vs translated sentence: **{summary['translated_flesch_kincaid_grade_avg']:.1f}** "
            f"(lower is easier to read)"
        )
        lines.append(
            f"- Average length — raw SHAP text: **{summary['raw_word_count_avg']:.1f} words** "
            f"vs translated sentence: **{summary['translated_word_count_avg']:.1f} words**"
        )
        lines.append("")

    lines.append("## Example side-by-side comparisons")
    lines.append("")
    for label, rows in [("URL", url_rows[:5]), ("Email", email_rows[:5])]:
        lines.append(f"### {label} examples")
        lines.append("")
        for r in rows:
            match = "correct" if r["predicted_label"] == r["true_label"] else "MISCLASSIFIED"
            lines.append(f"**Input:** `{r['input']}`  ")
            lines.append(
                f"**True label:** {r['true_label']} | **Predicted:** {r['predicted_label']} "
                f"({r['confidence']:.1%}) — *{match}*  "
            )
            lines.append(f"**Raw SHAP output:** {r['raw_shap_text']}  ")
            lines.append(f"**Plain-language translation:** {r['translated_sentence']}")
            lines.append("")

    lines.append(
        "\n> **Note on scope:** these readability metrics are an objective, "
        "reproducible proxy for surface-level linguistic complexity. They do "
        "not, by themselves, measure perceived credibility or "
        "decision-making speed - those dimensions of Objective V require "
        "human participants and are covered by the structured questionnaire "
        "in `docs/user_evaluation_questionnaire.md`, intended for "
        "administration with real non-expert participants (IT helpdesk "
        "staff / general users) as described in Chapter Three."
    )
    return "\n".join(lines)


def main() -> None:
    pipeline = PhishingExplanationPipeline(background_size=150, lime_samples=300)

    print(f"Evaluating {N_SAMPLES_PER_CLASS * 2} URLs and {N_SAMPLES_PER_CLASS * 2} emails...")
    url_rows = evaluate_urls(pipeline)
    email_rows = evaluate_emails(pipeline)

    url_summary = summarise(url_rows)
    email_summary = summarise(email_rows)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "evaluation_metrics.json").write_text(
        json.dumps(
            {"url": url_summary, "email": email_summary},
            indent=2,
        )
    )
    (REPORTS_DIR / "evaluation_examples.json").write_text(
        json.dumps({"url": url_rows, "email": email_rows}, indent=2)
    )

    report_md = render_markdown_report(url_rows, email_rows, url_summary, email_summary)
    (REPORTS_DIR / "evaluation_report.md").write_text(report_md)

    print(json.dumps({"url": url_summary, "email": email_summary}, indent=2))
    print(f"\nWrote reports/evaluation_report.md, evaluation_metrics.json, evaluation_examples.json")


if __name__ == "__main__":
    main()
