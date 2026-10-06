#!/usr/bin/env python3
"""
Stage-by-stage walkthrough of the PhishGuard pipeline, for demonstration.

Shows what happens to a single input at every layer boundary described in
Chapter Three: Input -> Feature Extraction -> Prediction -> XAI Attribution
-> Reconciliation -> Coherence -> Generation. Each stage pauses so it can be
narrated.

Usage:
    python scripts/demo_walkthrough.py                       # default URL
    python scripts/demo_walkthrough.py --url "http://..."
    python scripts/demo_walkthrough.py --email "Subject" "Body text"
    python scripts/demo_walkthrough.py --auto 6              # 6s per stage
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from phishguard.data.email_features import (  # noqa: E402
    FEATURE_NAMES as EMAIL_FEATURES,
)
from phishguard.data.email_features import EmailInput, extract_email_features  # noqa: E402
from phishguard.data.url_features import FEATURE_NAMES as URL_FEATURES  # noqa: E402
from phishguard.data.url_features import extract_url_features, is_analysable_url  # noqa: E402
from phishguard.pipeline import PhishingExplanationPipeline  # noqa: E402
from phishguard.translation.phrases import describe, is_coherent  # noqa: E402

W = 74
PAUSE = None  # seconds, or None to wait for Enter


def stage(n, title, subtitle=""):
    print("\n" + "─" * W)
    print(f"  STAGE {n}   {title}")
    if subtitle:
        print(f"            {subtitle}")
    print("─" * W)


def hold(note=""):
    if note:
        print(f"\n   ▸ {note}")
    if PAUSE is None:
        try:
            input("\n   [Enter to continue] ")
        except EOFError:
            print()
    else:
        time.sleep(PAUSE)


def run(pipeline, kind, raw, email=None):
    is_url = kind == "url"
    names = URL_FEATURES if is_url else EMAIL_FEATURES
    clf = pipeline.url_classifier if is_url else pipeline.email_classifier
    explainer = pipeline.url_explainer if is_url else pipeline.email_explainer

    # ---------------------------------------------------------------- 1
    stage(1, "INPUT", "what the user types into the interface")
    if is_url:
        print(f"\n   {raw}")
        print(f"\n   characters          {len(raw)}")
        print(f"   passes input gate   {is_analysable_url(raw)}")
        print(f"   ('banana' would be  {is_analysable_url('banana')} - rejected, no verdict given)")
    else:
        print(f"\n   Subject:  {email.subject}")
        body = email.body if len(email.body) < 240 else email.body[:237] + "..."
        print(f"   Body:     {body}")
    hold("The system has a string. It cannot do arithmetic on a string.")

    # ---------------------------------------------------------------- 2
    stage(2, "FEATURE EXTRACTION",
          f"the string becomes {len(names)} measurable numbers")
    feats = extract_url_features(raw) if is_url else extract_email_features(email)
    print()
    for i, n in enumerate(names):
        marker = ""
        if feats[n] and n in ("suspicious_tld", "has_ip_address", "is_shortener",
                              "brand_keyword_in_subdomain", "generic_greeting",
                              "any_embedded_ip_url"):
            marker = "   <= risk indicator present"
        print(f"   {i+1:>2}. {n:<30} {feats[n]:>10.4f}{marker}")
    hold("This is the only step that loses information. "
         "Nothing after this sees the original text.")

    # ---------------------------------------------------------------- 3
    stage(3, "CLASSIFICATION", "300 decision trees vote")
    proba = clf.predict_proba(feats)[0]
    classes = list(clf.classes_())
    print()
    for c, pr in zip(classes, proba):
        bar = "█" * int(pr * 40)
        print(f"   {c:<12} {pr*300:6.1f} / 300 trees   {bar} {pr:.1%}")
    label, conf = clf.verdict(feats)
    print(f"\n   VERDICT   {label.upper()}   confidence {conf:.1%}")
    hold("Confidence is literally the share of trees that voted this way.")

    # ---------------------------------------------------------------- 4
    stage(4, "XAI ATTRIBUTION", "why did the forest decide that?")
    s = explainer.explain_shap(feats)
    l = explainer.explain_lime(feats)
    pos = classes.index("phishing")
    total = sum(c.contribution for c in s.contributions)
    print(f"\n   SHAP  (exact Shapley values, TreeExplainer)")
    for c in s.top(5):
        print(f"      {c.feature:<30} {c.contribution:+.4f}  -> {c.direction}")
    print(f"\n      base value {s.base_value:.4f} + contributions {total:+.4f}"
          f" = {s.base_value + total:.4f}")
    print(f"      the model predicted               = {proba[pos]:.4f}")
    print(f"      they match to {abs(s.base_value + total - proba[pos]):.1e}"
          "  <= the attribution accounts for the prediction exactly")
    print(f"\n   LIME  (local linear approximation, 400 perturbed samples)")
    for c in l.top(5):
        print(f"      {c.feature:<30} {c.contribution:+.4f}  -> {c.direction}")
    hold("Two independent methods. Note they do not agree on the ranking.")

    # ---------------------------------------------------------------- 5
    stage(5, "RECONCILIATION", "two different scales become one ranking")

    def norm(r):
        m = max((abs(c.contribution) for c in r.contributions), default=1.0) or 1.0
        return m, {c.feature: c.contribution / m for c in r.contributions}

    sm, sn = norm(s)
    lm, ln = norm(l)
    print(f"\n   largest SHAP weight {sm:.4f}   largest LIME weight {lm:.4f}"
          f"   ({max(sm,lm)/min(sm,lm):.1f}x apart)")
    print("   -> each is divided by its own maximum, then the two are averaged\n")
    vals = {c.feature: c.raw_value for c in s.contributions}
    comb = {f: (sn[f] + ln.get(f, 0.0)) / 2 for f in sn}
    order = sorted(comb.items(), key=lambda kv: abs(kv[1]), reverse=True)
    print(f"   {'feature':<30}{'SHAP':>8}{'LIME':>8}{'mean':>8}")
    for f, sc in order[:6]:
        print(f"   {f:<30}{sn[f]:+8.2f}{ln.get(f,0):+8.2f}{sc:+8.2f}")
    hold("A factor both methods rank highly outranks one only a single method likes.")

    # ---------------------------------------------------------------- 6
    stage(6, "COHERENCE CHECK", "would saying this out loud make sense?")
    positive = label == "phishing"
    aligned = [(f, sc) for f, sc in order if (sc > 0) == positive and abs(sc) > 1e-6]
    kept = [(f, sc) for f, sc in aligned if is_coherent(f, vals[f], label)]
    dropped = [(f, sc) for f, sc in aligned if not is_coherent(f, vals[f], label)]
    print(f"\n   {len(order)} features"
          f"  ->  {len(aligned)} point the same way as the verdict"
          f"  ->  {len(kept)} coherent")
    if dropped:
        print(f"\n   REJECTED by the coherence check ({len(dropped)}):")
        for f, sc in dropped[:4]:
            rank = [x[0] for x in order].index(f) + 1
            print(f"      {f} (rank #{rank}, value {vals[f]:.2f})")
            print(f"         would have claimed: \"it {describe(f, vals[f])}\"")
            print(f"         ...as a reason this is {label.upper()}. It is not.")
    else:
        print("\n   nothing rejected on this input - every aligned factor is coherent")
    hold("This is the step that stops the system producing a fluent falsehood.")

    # ---------------------------------------------------------------- 7
    stage(7, "GENERATION", "the three surviving factors become clauses")
    print()
    for i, (f, sc) in enumerate(kept[:3], 1):
        print(f"   {i}. {f}  = {vals[f]:.2f}")
        print(f"      -> \"{describe(f, vals[f])}\"")
    result = (pipeline.analyze_url(raw) if is_url else pipeline.analyze_email(email))
    hold("Each factor maps to one pre-written, grammatical clause.")

    # ---------------------------------------------------------------- 8
    stage(8, "OUTPUT", "what the user actually reads")
    import textwrap
    print()
    print(f"   VERDICT: {result.verdict.upper()}  ({result.confidence:.1%} confidence)")
    print()
    for line in textwrap.wrap(result.translation.sentence, W - 6):
        print(f"   {line}")
    print(f"\n   (the interface shows all {len(names)} raw attributions beside this)")
    print("\n" + "─" * W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--email", nargs=2, metavar=("SUBJECT", "BODY"))
    ap.add_argument("--auto", type=float, help="seconds per stage instead of Enter")
    a = ap.parse_args()

    global PAUSE
    PAUSE = a.auto

    print("\n  Loading classifiers, SHAP/LIME explainers and the translation layer...")
    pipeline = PhishingExplanationPipeline()
    print("  Ready.")

    if a.email:
        run(pipeline, "email", a.email[0], EmailInput(subject=a.email[0], body=a.email[1]))
    else:
        url = a.url or "http://secure-paypal-verification.account-update.xyz/login.php?id=93820"
        run(pipeline, "url", url)


if __name__ == "__main__":
    main()
