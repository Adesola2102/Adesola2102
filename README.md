# Explainable Translation Layer for Phishing Alerts

A working prototype for the MIT Professional Master's Project:

> **An Explainable Translation Layer for Converting SHAP/LIME Outputs into
> Human-Readable Phishing Alerts Using a Lightweight Open-Source LLM**
> Zulikhat Adesola Adeyanju (2025/A/MIT/0231), Department of Information
> Technology, School of Computing, Miva Open University, Abuja.

This repository implements the five-component pipeline described in
Chapter Three of the project report — Data Ingestion & Feature Extraction →
Phishing Classification (Random Forest) → XAI (SHAP + LIME) → LLM
Translation Layer → User Interface (Streamlit) — and is the basis for
Chapter Four (System Implementation) and Chapter Five (Testing, Results &
Evaluation).

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. (Optional - data/raw/*.csv is already committed) re-fetch/resample data
python scripts/download_data.py

# 2. Train both classifiers (also already committed under models/)
python scripts/train_email_classifier.py
python scripts/train_url_classifier.py

# 3. Run the tests
pytest -q

# 4. Launch the UI
streamlit run app/streamlit_app.py

# 5. Regenerate the Objective-V evaluation report
python scripts/run_evaluation.py
```

## Repository layout

```
phishguard/
  data/            Feature extraction (email_features.py, url_features.py)
  models/          PhishingClassifier (Random Forest wrapper)
  xai/             SHAP TreeExplainer + LIME TabularExplainer wrapper
  translation/      Prompt template, phrase dictionary, pluggable LLM backends
  evaluation/       Readability-based Objective V evaluation utilities
  pipeline.py       End-to-end orchestration used by both the UI and evaluation
app/
  streamlit_app.py  User Interface (Component 5)
scripts/
  download_data.py            Reproducible dataset acquisition
  train_email_classifier.py   Trains + saves models/email_classifier.joblib
  train_url_classifier.py     Trains + saves models/url_classifier.joblib
  run_evaluation.py           Produces reports/evaluation_report.md
tests/              pytest unit + integration tests (50 tests)
data/raw/           Committed, reproducible data samples (see Data sources)
models/             Committed trained classifiers (.joblib)
reports/            Generated metrics/evaluation artifacts (regenerable)
docs/               user_evaluation_questionnaire.md (human-subject instrument)
```

## Data sources

The classifiers are trained on real, publicly available data fetched by
`scripts/download_data.py` (see that file's docstring for full detail and
the substitutions made because Nazario/CSDMC2010/PhishTank/ISCX-URL-2016
were not directly fetchable from this build environment):

- **Email:** Enron-Spam corpus (Metsis, Androutsopoulos & Paliouras, 2006),
  6,000-email balanced sample.
- **URL:** Phishing.Database active-phishing-domain blocklist + OpenDNS
  top-domains list, 12,000-URL balanced sample.

## Results snapshot

See `reports/*_classifier_metrics.json` and `reports/evaluation_report.md`
for full numbers (regenerate with the commands above). At last run:

| Classifier | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| URL (Random Forest) | 88.1% | 92.6% | 82.9% | 87.5% |
| Email (Random Forest) | 78.8% | 78.0% | 80.0% | 79.0% |

## Design notes / known limitations

These are discussed in full in Chapter Four ("Challenges Encountered and
Mitigation Strategies") and Chapter Five ("Discussion of Results"):

1. **Translation backend:** ships with a deterministic, template-based
   plain-language generator (`TemplateBackend`) rather than a downloaded
   LLM, because this build environment blocks outbound access to Hugging
   Face. The backend interface (`phishguard/translation/backend.py`)
   is pluggable — an `LlamaCppBackend` for a local quantized GGUF model
   (Phi-3-mini / Mistral-7B / Llama-3.2-3B) is implemented and unit-tested,
   and activates automatically if `PHISHGUARD_LLM_MODEL_PATH` points to a
   real model file.
2. **Faithfulness filter:** the translation layer only surfaces SHAP/LIME
   factors whose direction is semantically coherent with their actual
   value (`phishguard/translation/phrases.py::is_coherent`), to avoid
   generating misleading sentences from small-sample spurious
   correlations.
3. **Email corpus substitution:** Enron-Spam (spam/ham) is used in place of
   the Nazario/CSDMC2010 phishing corpora named in the proposal, for the
   reasons documented in `scripts/download_data.py`.
