"""
Data Ingestion & Feature Extraction Module - URL side.

Extracts lexical and host-based features directly from a URL string,
mirroring the feature families named in Chapter Three (URL length, presence
of IP addresses, number of subdomains, entropy, URL-shortener use, etc.) and
the feature set used by the PhishTank / ISCX-URL-2016-style literature cited
in Chapter Two.

No live network calls (WHOIS, DNS, content fetch) are made — the module is
purely lexical/string-based so that the classifier can run fully offline
(NFR5 / FR7 in Chapter Three).
"""

from __future__ import annotations

import math
import re
from types import SimpleNamespace
from urllib.parse import urlsplit

import pandas as pd

URL_REGEX = re.compile(r"(?:https?://|www\.)[^\s\"'<>]+", re.IGNORECASE)

FEATURE_NAMES = [
    "url_length",
    "domain_length",
    "num_dots",
    "num_hyphens",
    "num_digits",
    "digit_ratio",
    "num_subdomains",
    "has_ip_address",
    "has_at_symbol",
    "has_https",
    "num_special_chars",
    "suspicious_tld",
    "is_shortener",
    "domain_entropy",
    "brand_keyword_in_subdomain",
    "num_path_segments",
    "path_length",
    "query_length",
]

_IPV4_RE = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")

_SUSPICIOUS_TLDS = {
    "xyz", "top", "club", "work", "info", "online", "site", "icu", "buzz",
    "rest", "click", "link", "gq", "tk", "ml", "ga", "cf", "live", "fit",
    "kim", "loan", "men", "review", "download", "stream", "party", "trade",
    "date", "win", "bid", "accountant", "science", "faith", "cam", "cyou",
}

_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "shorte.st", "adf.ly", "rb.gy",
}

_MULTI_LABEL_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk", "gov.uk", "com.ng", "org.ng", "gov.ng",
    "edu.ng", "co.za", "com.au", "co.jp", "com.br", "co.in", "com.ng",
}

_BRAND_KEYWORDS = [
    "paypal", "apple", "microsoft", "google", "amazon", "facebook",
    "instagram", "netflix", "bank", "chase", "wellsfargo", "irs", "dhl",
    "fedex", "ups", "whatsapp", "coinbase", "binance", "outlook", "office365",
    "icloud", "verizon", "att", "hsbc", "barclays", "gtbank", "zenithbank",
    "firstbank", "accessbank",
]


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def _split_registrable_domain(hostname: str) -> tuple[str, list]:
    """Return (registrable_domain, subdomain_labels) using a small suffix list."""

    labels = hostname.split(".")
    if len(labels) <= 2:
        return hostname, []
    last_two = ".".join(labels[-2:])
    if last_two in _MULTI_LABEL_SUFFIXES and len(labels) > 2:
        registrable = ".".join(labels[-3:])
        subdomains = labels[:-3]
    else:
        registrable = last_two
        subdomains = labels[:-2]
    return registrable, subdomains


def extract_url_features(url: str) -> dict:
    """Return a flat dict of numeric features describing ``url``."""

    raw = url.strip()
    if "://" not in raw:
        raw = "http://" + raw

    try:
        parts = urlsplit(raw)
        hostname = (parts.hostname or "").lower()
        netloc = parts.netloc.lower()
    except ValueError:
        # Real-world phishing text sometimes embeds malformed/obfuscated
        # "URLs" (stray brackets, broken bracketed-IPv6 syntax, etc.) that
        # urlsplit rejects outright. Treat these as unparseable rather than
        # crashing the pipeline: string-level features (length, digit ratio,
        # special-char count below) still apply, host-structure features
        # just fall back to their empty/absent defaults - a malformed URL is
        # itself a signal a well-formed one would never produce.
        parts = SimpleNamespace(scheme="", netloc="", path="", query="")
        hostname = ""
        netloc = ""

    registrable_domain, subdomains = _split_registrable_domain(hostname)
    # "www" is by far the most common legitimate subdomain label; counting it
    # as a nesting/cloaking signal would penalise a large fraction of
    # ordinary legitimate sites, so it is excluded from the subdomain count.
    subdomains = [s for s in subdomains if s != "www"]
    tld = registrable_domain.split(".")[-1] if "." in registrable_domain else ""
    subdomain_str = ".".join(subdomains)

    digits = sum(c.isdigit() for c in raw)
    special_chars = sum(
        1 for c in raw if not c.isalnum() and c not in "://.-_/?=&%"
    )

    features = {
        "url_length": float(len(raw)),
        "domain_length": float(len(hostname)),
        "num_dots": float(hostname.count(".")),
        "num_hyphens": float(hostname.count("-")),
        "num_digits": float(digits),
        "digit_ratio": digits / max(len(raw), 1),
        "num_subdomains": float(len(subdomains)),
        "has_ip_address": float(bool(_IPV4_RE.match(hostname))),
        "has_at_symbol": float("@" in raw),
        "has_https": float(parts.scheme == "https"),
        "num_special_chars": float(special_chars),
        "suspicious_tld": float(tld in _SUSPICIOUS_TLDS),
        "is_shortener": float(registrable_domain in _SHORTENERS),
        "domain_entropy": _shannon_entropy(hostname),
        "brand_keyword_in_subdomain": float(
            any(b in subdomain_str for b in _BRAND_KEYWORDS)
            and not any(b in registrable_domain for b in _BRAND_KEYWORDS)
        ),
        "num_path_segments": float(len([p for p in parts.path.split("/") if p])),
        "path_length": float(len(parts.path)),
        "query_length": float(len(parts.query)),
    }
    return features


def build_url_feature_frame(urls: list) -> pd.DataFrame:
    rows = [extract_url_features(u) for u in urls]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)
