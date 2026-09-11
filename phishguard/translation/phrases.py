"""
Feature -> plain-English phrase dictionary used by the LLM Translation Layer.

Each feature has two small template functions:

* ``present`` - clause used when the feature's value is "elevated" (a flag
  is true, a keyword count is > 0, or - for continuous features with no
  natural absent state, such as URL length - simply "this is the value").
* ``absent``  - clause used when the feature's value is at/near its
  baseline (a flag is false, a keyword count is 0).

Separately, ``HIGH_RISK_WHEN_PRESENT`` records, for each feature, whether an
*elevated* value is normally the phishing-risk direction (true for the large
majority - e.g. more urgent-language keywords is riskier) or the opposite
(false for the two exceptions: ``has_https`` and ``word_count``, where an
elevated value is normally the *safer* signal).

``translator.py`` uses this polarity table as a faithfulness/coherence
check: SHAP/LIME occasionally rank a feature whose value sits at its "safe"
setting as the top contributor toward a PHISHING verdict (or vice-versa) -
a small-sample spurious correlation rather than a real signal. Rather than
generate a sentence that contradicts the feature's own plain-English
meaning (e.g. claiming an email "references an attachment" when it plainly
does not), such incoherent factors are filtered out before the sentence is
built. This directly addresses the explanation-faithfulness concern raised
in Cambria et al. (2023) and discussed in Chapter Two.
"""

from __future__ import annotations

from typing import Callable, Dict, Set

PhraseFn = Callable[[float], str]


def _n(value: float) -> int:
    return int(round(value))


def _count(value: float, singular: str, plural: str | None = None) -> str:
    """Render a count with a correctly agreeing noun.

    The sentences these phrases build are the project's output, and their
    readability is what Chapter Five measures, so "1 dots" and the evasive
    "1 dot(s)" are both defects rather than cosmetic details.
    """

    n = _n(value)
    return f"{n} {singular if n == 1 else (plural or singular + 's')}"


def _pair(present: PhraseFn, absent: PhraseFn) -> Dict[str, PhraseFn]:
    return {"present": present, "absent": absent}


FEATURE_PHRASES: Dict[str, Dict[str, PhraseFn]] = {
    # ---------------------------------------------------------------- URL
    "suspicious_tld": _pair(
        lambda v: "uses an unusual, high-risk domain ending",
        lambda v: "uses a common, mainstream domain ending",
    ),
    "num_subdomains": _pair(
        lambda v: f"contains {_count(v, 'nested sub-domain')}, a common cloaking trick",
        lambda v: "has a simple, normal domain structure",
    ),
    "url_length": _pair(
        lambda v: f"has an unusually long web address ({_count(v, 'character')})",
        lambda v: "has a normal-length web address",
    ),
    "domain_length": _pair(
        lambda v: f"has an unusually long domain name ({_count(v, 'character')})",
        lambda v: "has a normal-length domain name",
    ),
    "num_dots": _pair(
        lambda v: f"has an unusually complex domain name with {_count(v, 'dot')}",
        lambda v: "has a simple domain name",
    ),
    "num_hyphens": _pair(
        lambda v: f"contains {_count(v, 'hyphen')} in the domain, often used to imitate a real brand",
        lambda v: "has no unusual hyphens in the domain",
    ),
    "num_digits": _pair(
        lambda v: "mixes unusual digits into the web address",
        lambda v: "does not mix unusual digits into the web address",
    ),
    "digit_ratio": _pair(
        lambda v: "is made up largely of digits",
        lambda v: "is not dominated by digits",
    ),
    "has_ip_address": _pair(
        lambda v: "points to a raw numeric IP address instead of a normal domain name",
        lambda v: "points to a properly registered domain name rather than a raw IP address",
    ),
    "has_at_symbol": _pair(
        lambda v: "contains an '@' symbol that can hide its real destination",
        lambda v: "does not contain any hidden-destination tricks such as an '@' symbol",
    ),
    "has_https": _pair(
        lambda v: "uses a standard secure HTTPS connection",
        lambda v: "does not use a secure HTTPS connection",
    ),
    "num_special_chars": _pair(
        lambda v: "contains unusual special characters",
        lambda v: "does not contain unusual special characters",
    ),
    "is_shortener": _pair(
        lambda v: "uses a link-shortening service that hides its true destination",
        lambda v: "does not rely on a link-shortening service",
    ),
    "domain_entropy": _pair(
        lambda v: "has a domain name that looks randomly generated",
        lambda v: "has a domain name that reads like a normal word/brand",
    ),
    "brand_keyword_in_subdomain": _pair(
        lambda v: "places a trusted brand name in the web address in a misleading way",
        lambda v: "does not misuse a trusted brand name in its web address",
    ),
    "num_path_segments": _pair(
        lambda v: "has an unusually complex folder structure in its link",
        lambda v: "has a simple, shallow link structure",
    ),
    "path_length": _pair(
        lambda v: "has an unusually long web-address path",
        lambda v: "has a short, normal web-address path",
    ),
    "query_length": _pair(
        lambda v: "carries unusually long tracking parameters",
        lambda v: "does not carry unusual tracking parameters",
    ),
    # -------------------------------------------------------------- EMAIL
    "sender_domain_mismatch": _pair(
        lambda v: "was sent from an address whose display name does not match its actual domain",
        lambda v: "was sent from an address whose display name matches its actual domain",
    ),
    "reply_to_mismatch": _pair(
        lambda v: "would redirect any reply to a different address than the visible sender",
        lambda v: "keeps replies going to the same address as the visible sender",
    ),
    "spf_fail": _pair(
        lambda v: "fails the sender's SPF authentication check",
        lambda v: "passes the sender's SPF authentication check",
    ),
    "dkim_fail": _pair(
        lambda v: "fails the sender's DKIM authentication check",
        lambda v: "passes the sender's DKIM authentication check",
    ),
    "word_count": _pair(
        lambda v: "has a normal amount of message content",
        lambda v: "is unusually short, giving little legitimate context",
    ),
    "exclamation_count": _pair(
        lambda v: "uses excessive exclamation marks to create urgency",
        lambda v: "is written in a calm, measured tone",
    ),
    "all_caps_word_ratio": _pair(
        lambda v: "uses excessive capital letters to create alarm",
        lambda v: "does not rely on capital letters for emphasis",
    ),
    "urgent_keyword_count": _pair(
        lambda v: "uses urgent, pressuring language",
        lambda v: "does not use urgent or pressuring language",
    ),
    "financial_keyword_count": _pair(
        lambda v: "references bank, payment or other financial details",
        lambda v: "does not reference financial or payment details",
    ),
    "credential_keyword_count": _pair(
        lambda v: "asks for a password, login, or other sensitive credentials",
        lambda v: "does not ask for a password or other credentials",
    ),
    "generic_greeting": _pair(
        lambda v: "uses a generic greeting instead of your real name",
        lambda v: "addresses the recipient normally rather than with a generic greeting",
    ),
    "reward_keyword_count": _pair(
        lambda v: "promises a prize, refund, or reward",
        lambda v: "does not promise a prize, refund, or reward",
    ),
    "html_tag_ratio": _pair(
        lambda v: "contains embedded HTML/web formatting rather than plain text",
        lambda v: "is written mostly as plain text",
    ),
    "non_alpha_ratio": _pair(
        lambda v: "contains an unusual number of symbols or special characters",
        lambda v: "reads as normal, clean text",
    ),
    "attachment_keyword": _pair(
        lambda v: "references an attached file",
        lambda v: "does not reference any attachment",
    ),
    "num_urls_in_body": _pair(
        lambda v: f"contains {_count(v, 'embedded link')}",
        lambda v: "contains few or no embedded links",
    ),
    "link_to_word_ratio": _pair(
        lambda v: "is unusually link-heavy relative to its length",
        lambda v: "is not unusually link-heavy",
    ),
    "max_url_suspicion_score": _pair(
        lambda v: "contains links with multiple classic phishing warning signs",
        lambda v: "contains links that look ordinary",
    ),
    "any_embedded_ip_url": _pair(
        lambda v: "contains a link pointing to a raw IP address",
        lambda v: "does not contain any links pointing to a raw IP address",
    ),
    "any_embedded_shortener": _pair(
        lambda v: "contains a link that uses a link-shortening service",
        lambda v: "does not contain any shortened links",
    ),
    "avg_embedded_url_length": _pair(
        lambda v: "contains unusually long links",
        lambda v: "contains normal-length links",
    ),
}

# Features where an ELEVATED value is normally the *safer* direction rather
# than the risk direction. Every feature not listed here defaults to True
# (elevated = riskier), which matches the large majority of the phrase set
# above.
HIGH_RISK_WHEN_PRESENT: Dict[str, bool] = {
    "has_https": False,
    "word_count": False,
}

# Features that behave like flags/counts, i.e. "present" means value >= 0.5
# (flags) or value > 0 (counts). Everything else is treated as an
# always-descriptive magnitude (e.g. url_length, domain_entropy) where the
# "present" phrase is always used and simply narrates the numeric value.
_BINARY_FEATURES: Set[str] = {
    "suspicious_tld", "has_ip_address", "has_at_symbol", "has_https",
    "is_shortener", "brand_keyword_in_subdomain", "sender_domain_mismatch",
    "reply_to_mismatch", "spf_fail", "dkim_fail", "generic_greeting",
    "attachment_keyword", "any_embedded_ip_url", "any_embedded_shortener",
}
_COUNT_FEATURES: Set[str] = {
    "num_subdomains", "num_hyphens", "num_digits", "num_special_chars",
    "num_dots", "num_path_segments", "urgent_keyword_count",
    "financial_keyword_count", "credential_keyword_count",
    "reward_keyword_count", "exclamation_count", "num_urls_in_body",
    "all_caps_word_ratio", "html_tag_ratio", "non_alpha_ratio",
    "link_to_word_ratio", "digit_ratio", "max_url_suspicion_score",
}
_WORD_COUNT_THRESHOLD = 15.0  # below this, an email is treated as "short"


def is_present(feature: str, value: float) -> bool:
    """Whether ``value`` counts as the "elevated" state for ``feature``."""

    if feature == "word_count":
        return value >= _WORD_COUNT_THRESHOLD
    if feature in _BINARY_FEATURES:
        return value >= 0.5
    if feature in _COUNT_FEATURES:
        return value > 0
    return True  # continuous magnitude features are always "descriptive"


def expected_verdict_when_present(feature: str) -> str:
    high_risk = HIGH_RISK_WHEN_PRESENT.get(feature, True)
    return "phishing" if high_risk else "legitimate"


def is_coherent(feature: str, value: float, predicted_verdict: str) -> bool:
    """True if ``feature``'s actual value plausibly explains why the model
    predicted ``predicted_verdict`` (used to filter out spurious/noisy
    SHAP-LIME factors before they are turned into a sentence)."""

    present = is_present(feature, value)
    expected_if_present = expected_verdict_when_present(feature)
    expected_if_absent = "legitimate" if expected_if_present == "phishing" else "phishing"
    expected = expected_if_present if present else expected_if_absent
    return expected == predicted_verdict


def describe(feature: str, value: float) -> str:
    """Return the natural-language clause describing ``feature`` at ``value``."""

    meta = FEATURE_PHRASES.get(feature)
    if meta is None:
        clean = feature.replace("_", " ")
        return f"shows an unusual value for '{clean}'"
    present = is_present(feature, value)
    return meta["present" if present else "absent"](value)
