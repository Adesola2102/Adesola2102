"""
Data Ingestion & Feature Extraction Module - email side.

Converts a raw email (subject + body, with optional header metadata) into
the numerical feature vector consumed by the Random Forest phishing
classifier (Chapter Three, Component 1 / Component 2).

Feature families implemented, matching the methodology in Chapter Three:

* Header features   - sender/reply-to domain mismatch, SPF/DKIM pass flags.
  These are populated from the caller-supplied ``EmailInput`` when header
  metadata is available (e.g. a live .eml upload in the UI). The benchmark
  training corpus (Enron-Spam, see scripts/download_data.py) only contains
  subject/body text with no header block, so during training these features
  are supplied with a neutral default and the classifier instead leans on
  the richer body/URL-derived signals below. This is disclosed as a
  build-environment data constraint in Chapter Four.
* Body features     - urgency/pressure language, financial/credential
  keywords, generic-greeting detection, punctuation/formatting anomalies.
* Embedded-URL features - counts and lexical properties of any links found
  in the email body, reusing ``phishguard.data.url_features``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from phishguard.data.url_features import URL_REGEX, extract_url_features

FEATURE_NAMES = [
    # header-derived (neutral 0.5 default when unknown)
    "sender_domain_mismatch",
    "reply_to_mismatch",
    "spf_fail",
    "dkim_fail",
    # body / language features
    "word_count",
    "exclamation_count",
    "all_caps_word_ratio",
    "urgent_keyword_count",
    "financial_keyword_count",
    "credential_keyword_count",
    "generic_greeting",
    "reward_keyword_count",
    "html_tag_ratio",
    "non_alpha_ratio",
    "attachment_keyword",
    # embedded URL features (aggregated across links found in the body)
    "num_urls_in_body",
    "link_to_word_ratio",
    "max_url_suspicion_score",
    "any_embedded_ip_url",
    "any_embedded_shortener",
    "avg_embedded_url_length",
]

_URGENT_KEYWORDS = [
    "urgent", "immediately", "verify your account", "verify now", "act now",
    "action required", "suspend", "suspended", "expire", "expires",
    "expiring", "limited time", "within 24 hours", "final notice",
    "unusual activity", "unauthorized", "security alert", "restricted",
    "your account will be", "confirm your", "click here", "log in now",
]
_FINANCIAL_KEYWORDS = [
    "bank", "wire transfer", "invoice", "payment", "credit card", "billing",
    "refund", "tax", "irs", "paypal", "account number", "routing number",
    "wallet", "bitcoin", "crypto",
]
_CREDENTIAL_KEYWORDS = [
    "password", "username", "login", "log in", "ssn", "social security",
    "pin number", "security question", "one-time code", "otp", "2fa",
    "reset your password", "update your information",
]
_REWARD_KEYWORDS = [
    "congratulations", "winner", "you have won", "claim your prize",
    "free gift", "lottery", "selected to receive", "cashback",
]
_GENERIC_GREETINGS = [
    "dear customer", "dear user", "dear valued customer", "dear account holder",
    "dear sir/madam", "dear member", "hello dear",
]
_HTML_TAG_RE = re.compile(r"<\s*(a|img|table|div|span|font)\b", re.IGNORECASE)


@dataclass
class EmailInput:
    """Raw email supplied to the pipeline. Header fields are optional."""

    subject: str = ""
    body: str = ""
    sender_display_domain: Optional[str] = None
    sender_actual_domain: Optional[str] = None
    reply_to_domain: Optional[str] = None
    spf_pass: Optional[bool] = None
    dkim_pass: Optional[bool] = None
    extra_urls: list = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return f"{self.subject}\n{self.body}"


def _count_keywords(text: str, keywords: list) -> int:
    text_l = text.lower()
    return sum(text_l.count(kw) for kw in keywords)


def _has_keyword(text: str, keywords: list) -> bool:
    text_l = text.lower()
    return any(kw in text_l for kw in keywords)


def _all_caps_ratio(words: list) -> float:
    alpha_words = [w for w in words if w.isalpha() and len(w) > 1]
    if not alpha_words:
        return 0.0
    caps = [w for w in alpha_words if w.isupper()]
    return len(caps) / len(alpha_words)


def extract_email_features(email: EmailInput) -> dict:
    """Return a flat dict of numeric features for ``email``."""

    text = email.full_text
    words = re.findall(r"[A-Za-z']+", text)
    word_count = max(len(words), 1)

    urls = URL_REGEX.findall(text) + list(email.extra_urls)
    url_feature_rows = [extract_url_features(u) for u in urls]

    if url_feature_rows:
        suspicion_scores = [
            row["digit_ratio"] + row["has_ip_address"] + row["suspicious_tld"]
            + row["has_at_symbol"] + row["brand_keyword_in_subdomain"]
            for row in url_feature_rows
        ]
        max_suspicion = max(suspicion_scores)
        any_ip = float(any(row["has_ip_address"] for row in url_feature_rows))
        any_shortener = float(any(row["is_shortener"] for row in url_feature_rows))
        avg_len = sum(row["url_length"] for row in url_feature_rows) / len(url_feature_rows)
    else:
        max_suspicion = 0.0
        any_ip = 0.0
        any_shortener = 0.0
        avg_len = 0.0

    def _domain_mismatch(a: Optional[str], b: Optional[str]) -> float:
        if not a or not b:
            return 0.5  # unknown / neutral
        return float(a.strip().lower() != b.strip().lower())

    non_alpha_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())

    features = {
        "sender_domain_mismatch": _domain_mismatch(
            email.sender_display_domain, email.sender_actual_domain
        ),
        "reply_to_mismatch": _domain_mismatch(
            email.reply_to_domain, email.sender_actual_domain
        ),
        "spf_fail": 0.5 if email.spf_pass is None else float(not email.spf_pass),
        "dkim_fail": 0.5 if email.dkim_pass is None else float(not email.dkim_pass),
        "word_count": float(word_count),
        "exclamation_count": float(text.count("!")),
        "all_caps_word_ratio": _all_caps_ratio(words),
        "urgent_keyword_count": float(_count_keywords(text, _URGENT_KEYWORDS)),
        "financial_keyword_count": float(_count_keywords(text, _FINANCIAL_KEYWORDS)),
        "credential_keyword_count": float(_count_keywords(text, _CREDENTIAL_KEYWORDS)),
        "generic_greeting": float(_has_keyword(text, _GENERIC_GREETINGS)),
        "reward_keyword_count": float(_count_keywords(text, _REWARD_KEYWORDS)),
        "html_tag_ratio": len(_HTML_TAG_RE.findall(text)) / word_count,
        "non_alpha_ratio": non_alpha_chars / max(len(text), 1),
        "attachment_keyword": float(_has_keyword(text, ["attached", "attachment"])),
        "num_urls_in_body": float(len(urls)),
        "link_to_word_ratio": len(urls) / word_count,
        "max_url_suspicion_score": max_suspicion,
        "any_embedded_ip_url": any_ip,
        "any_embedded_shortener": any_shortener,
        "avg_embedded_url_length": avg_len,
    }
    return features


def build_email_feature_frame(emails: list) -> pd.DataFrame:
    """Vectorised helper: list[EmailInput] -> feature DataFrame."""

    rows = [extract_email_features(e) for e in emails]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)
