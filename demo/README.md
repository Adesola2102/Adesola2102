# PhishLens — standalone browser demo

`phishlens.html` is a single, self-contained HTML file (no build step, no
server, no dependencies) that lets anyone try the system's core idea —
verdict + confidence + a plain-language explanation, next to the raw
model output it was derived from — by opening the file directly in a
browser.

It embeds a lighter, browser-sized version of the real classifiers (50
trees, depth-limited, exported from the same training data and pipeline
described in `scripts/train_email_classifier.py` /
`scripts/train_url_classifier.py` and Chapter Four of the project report)
and reimplements the feature-extraction and phrase/translation logic from
`phishguard/data/*_features.py` and `phishguard/translation/*` in
JavaScript, so it can run entirely client-side with no installation.

Because it has no SHAP/LIME dependency available in-browser, per-feature
"raw model output" contributions are computed with a path-based attribution
over the actual decision trees (the Saabas method — a fast, well-known
relative of SHAP), rather than exact Shapley values. This is disclosed on
the page itself. The authoritative implementation — full-size classifiers,
exact SHAP TreeExplainer + LIME, 50 automated tests — is the Python
package in `phishguard/`, documented in `docs/Chapter_Four_and_Five.docx`.

## Usage

Just open `phishlens.html` in any modern browser (double-click it, or
`open demo/phishlens.html` / drag it into a browser tab). No server, no
`pip install`, no internet connection required — everything, including the
trained model weights, is embedded in the file.
