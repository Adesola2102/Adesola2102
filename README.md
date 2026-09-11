# Explainable Translation Layer for Phishing Alerts

An explainable phishing detection system built for the MIT Professional
Master's Project:

> **An Explainable Translation Layer for Converting SHAP/LIME Outputs into
> Human-Readable Phishing Alerts Using a Lightweight Open-Source LLM**
> Zulikhat Adesola Adeyanju (2025/A/MIT/0231), Department of Information
> Technology, School of Computing, Miva Open University, Abuja.

The system is a layered, service-oriented architecture designed for
local-first deployment. A seven-layer pipeline — Input → Feature Extraction →
Prediction → XAI Attribution → Translation → Generation → Presentation — takes
an email or URL through to a plain-language explanation, reconciling SHAP and
LIME attributions and enforcing semantic coherence before any sentence is
generated. Each layer consumes only the output of the layer before it, so the
processing engine is decoupled from the presentation layer and could sit behind
an email gateway or a SOC dashboard instead of the Streamlit interface without
change to the layers beneath it.

Everything runs offline on ordinary CPU hardware: no GPU, and no network call
at inference time. This repository is the basis for Chapter Four (System
Implementation) and Chapter Five (Testing, Results & Evaluation) of the report.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Train both classifiers
#    -> models/*.joblib, reports/*_classifier_metrics.json
python scripts/train_url_classifier.py
python scripts/train_email_classifier.py

# 2. Regenerate the Objective-V evaluation
#    -> reports/evaluation_report.md, evaluation_metrics.json, evaluation_examples.json
python scripts/run_evaluation.py

# 3. Measure runtime performance
#    -> reports/performance_metrics.json
python scripts/measure_performance.py

# 4. Run the tests (66 tests)
pytest -q

# 5. Launch the UI
streamlit run app/streamlit_app.py
```

Models and reports are committed, so steps 1–3 are only needed to reproduce
them. They do reproduce: re-running the training scripts on the committed data
regenerates both `.joblib` models byte-for-byte identically, along with
`url_classifier_metrics.json`, `email_classifier_metrics.json`,
`evaluation_metrics.json` and `evaluation_report.md`. Only
`evaluation_examples.json` varies, in a handful of confidence values at the
sixteenth decimal place (floating-point summation order in the forest's vote
averaging); predicted labels, attributions and generated sentences are
unaffected.

### A note on `scripts/download_data.py`

This script is **not** part of the sequence above, and running it will change
your results. It re-fetches from live phishing feeds that refresh every 24
hours: the sampling is seeded and reproducible, but the pool being sampled is
not fixed, so it overwrites `data/raw/*.csv` with whatever the feeds carry
today. The committed CSVs are the ones that produced every number reported
below and in the report. Run it only when a deliberately fresh dataset is
wanted, and expect the metrics to move.

## Repository layout

```
phishguard/
  data/             Feature extraction (email_features.py, url_features.py)
  models/           PhishingClassifier (Random Forest wrapper)
  xai/              SHAP TreeExplainer + LIME TabularExplainer wrapper
  translation/      Prompt template, phrase dictionary, pluggable LLM backends
  evaluation/       Readability-based Objective V evaluation utilities
  pipeline.py       End-to-end orchestration used by both the UI and evaluation
app/
  streamlit_app.py  Presentation layer (Streamlit)
scripts/
  download_data.py            Dataset acquisition (see note above)
  train_url_classifier.py     Trains + saves models/url_classifier.joblib
  train_email_classifier.py   Trains + saves models/email_classifier.joblib
  run_evaluation.py           Produces reports/evaluation_report.md
  measure_performance.py      Produces reports/performance_metrics.json
tests/              pytest unit + integration tests (66 tests)
data/raw/           Committed datasets (see Data sources)
models/             Committed trained classifiers (.joblib)
reports/            Generated metrics/evaluation artifacts (regenerable)
docs/               user_evaluation_questionnaire.md (human-subject instrument)
```

## Data sources

Both classifiers are trained on genuine, directly-labelled phishing data
fetched by `scripts/download_data.py`, whose docstring records every source URL
and the reasoning behind two deliberate construction choices (why the
legitimate URL class stops at roughly 1,000 rows, and why the http/https scheme
is re-randomised):

- **Email — 6,000 messages, 3,000 phishing / 3,000 legitimate.** Phishing from
  the Nazario Phishing Corpus and a Nigerian/419 advance-fee fraud set;
  legitimate from the ham portion of the Enron corpus (Metsis, Androutsopoulos
  & Paliouras, 2006).
- **URL — 1,995 URLs, 1,000 phishing / 995 legitimate.** Phishing from
  PhishTank's verified feed and Phishing.Database's active-links list;
  legitimate from a labelled set that preserves real paths and query strings
  rather than bare domains.

The corpora carry subject and body only, with no header block. Four
header-authenticity features (`sender_domain_mismatch`, `reply_to_mismatch`,
`spf_fail`, `dkim_fail`) therefore sit at a neutral default during training and
carry zero importance in the trained model. This is documented as a data
constraint in Chapter Four rather than worked around.

## Results

Full numbers in `reports/*_classifier_metrics.json`,
`reports/evaluation_report.md` and `reports/performance_metrics.json`.

**Detection**

| Classifier | Accuracy | Precision | Recall | F1 | Test set |
|---|---|---|---|---|---|
| URL (Random Forest) | 92.23% | 91.22% | 93.50% | 92.35% | 399 |
| Email (Random Forest) | 96.75% | 98.95% | 94.50% | 96.68% | 1,200 |

**Explanation readability** — the project's central result, measured over 60
predictions by comparing each translated sentence against the raw SHAP/LIME
output it replaces:

| | URL | Email |
|---|---|---|
| Flesch Reading Ease, raw | 8.7 | −51.3 |
| Flesch Reading Ease, translated | 36.0 | 24.0 |
| **Improvement** | **+27.3 points** | **+75.4 points** |

Flesch-Kincaid Grade Level rises rather than falls (12.6 → 16.2 for URLs).
This is a property of applying a syllable-counting prose formula to non-prose
attribution output, and is reported as a methodological finding in Chapter Five
rather than suppressed.

**Performance** — 0.11 s mean per analysis on a single CPU core with no GPU,
measured over 15 runs; 12.2 MB of models on disk.

## Design notes / known limitations

Discussed in full in Chapter Four ("Challenges Encountered and Mitigation
Strategies") and Chapter Six ("Conclusion"):

1. **Translation backend.** Ships with a deterministic, template-based
   plain-language generator (`TemplateBackend`) rather than a downloaded LLM,
   because this build environment blocks outbound access to Hugging Face. The
   backend interface (`phishguard/translation/backend.py`) is pluggable — an
   `LlamaCppBackend` for a local quantized GGUF model is implemented, and
   activates automatically if `PHISHGUARD_LLM_MODEL_PATH` points to a real
   model file. **It has never been run against real weights.** Its
   availability gating, offline fallback and output post-processing are
   covered by tests, and a separate integration test drives the real pipeline
   through a non-template backend to evidence that the generator is genuinely
   replaceable — but no result anywhere in this project describes the output
   of a language model.
2. **Faithfulness filter.** The translation layer only surfaces SHAP/LIME
   factors whose direction is semantically coherent with their actual value
   (`phishguard/translation/phrases.py::is_coherent`), so that a statistically
   prominent factor is withheld rather than asserted misleadingly. This was
   added in response to an observed failure and is guarded by a regression
   test.
3. **Readability measured, comprehension not.** Translation is shown to produce
   measurably more readable output. Whether users understand it better, trust
   it more or act on it faster was not measured; the instrument for that study
   is in `docs/`.

Three of the tests are regression tests written after specific defects found
during development — the faithfulness failure, a malformed-URL crash, and an
explanation-reproducibility defect in which LIME's stateful random generator
advanced between calls. Each reproduces its bug and fails without the
corresponding fix.
