"""
Tests for the Input Layer gate (Chapter Three, Section 3.3.1).

The feature extractors are total by design: any string yields a complete
feature vector, because extraction runs over tens of thousands of rows during
dataset construction and one malformed row must not halt it. At the interface
that same property is a liability, since a bare word is silently rewritten to
"http://<word>" and comes back with a confident verdict and an explanation
describing properties the input does not have. is_analysable_url() is the gate
that refuses such input instead of explaining it.
"""

from __future__ import annotations

import pytest

from phishguard.data.url_features import extract_url_features, is_analysable_url
from phishguard.translation.phrases import describe


# ------------------------------------------------------------------ rejected

@pytest.mark.parametrize(
    "value",
    [
        "banana",           # the case named in Chapter Three
        "hello",
        "x",
        "",
        "   ",
        "not a url at all",   # no dotted host survives the spaces
        "exa mple.com",       # whitespace inside the host itself
        "https://exa mple.com/path",
        "http://",           # scheme with no host
        "://",
        "localhost.",        # trailing dot leaves an empty final label
        "-leading-hyphen.com",
        "trailing-hyphen-.com",
    ],
)
def test_structurally_implausible_input_is_refused(value):
    assert is_analysable_url(value) is False


# ------------------------------------------------------------------ accepted

@pytest.mark.parametrize(
    "value",
    [
        "https://example.com",
        "example.com",                                    # scheme is optional
        "http://secure-paypal-verification.account-update.xyz/login.php?id=93820",
        "https://en.wikipedia.org/wiki/Estrildid_finch",
        "http://192.168.1.1/admin",                       # raw IP is analysable
        "https://sub.domain.co.uk/path?query=1",
        "localhost:8501",
        "https://q-r.to/bfMBCf",                          # shortener
        # The gate asks only whether the HOST is usable, so a stray space in
        # the path does not make the input unanalysable - every host-derived
        # feature is still meaningful.
        "https://example.com/pa th",
    ],
)
def test_genuine_web_addresses_are_accepted(value):
    assert is_analysable_url(value) is True


# ------------------------------------------------- why the gate has to exist

def test_a_refused_input_would_otherwise_get_a_confident_false_explanation():
    """The regression this gate prevents.

    Without it, "banana" becomes "http://banana", receives a full feature
    vector, and is described with clauses about a web address it does not have.
    This asserts the underlying behaviour still exists (the extractors remain
    total, which Section 4.2.2 requires) so that the gate is demonstrably the
    thing standing between that behaviour and the user.
    """
    assert is_analysable_url("banana") is False

    # The extractor still happily describes it, which is exactly the problem.
    features = extract_url_features("banana")
    assert len(features) == 18
    assert describe("url_length", features["url_length"])


def test_the_evaluation_corpus_is_unaffected_by_the_gate():
    """Every input used in Chapter Five's evaluation must still be analysable,
    so adding the gate cannot have changed any reported result."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "reports" / "evaluation_examples.json"
    if not path.exists():
        pytest.skip("evaluation_examples.json not present")

    urls = [row["input"] for row in json.loads(path.read_text())["url"]]
    refused = [u for u in urls if not is_analysable_url(u)]
    assert refused == [], f"gate would have refused evaluated input: {refused}"


# ------------------------------------------------------------- pluralisation

@pytest.mark.parametrize(
    "feature,value,expected",
    [
        ("num_dots", 1, "has an unusually complex domain name with 1 dot"),
        ("num_dots", 3, "has an unusually complex domain name with 3 dots"),
        ("num_hyphens", 1, "contains 1 hyphen in the domain, often used to imitate a real brand"),
        ("num_subdomains", 1, "contains 1 nested sub-domain, a common cloaking trick"),
        ("num_subdomains", 2, "contains 2 nested sub-domains, a common cloaking trick"),
        ("url_length", 1, "has an unusually long web address (1 character)"),
        ("url_length", 296, "has an unusually long web address (296 characters)"),
    ],
)
def test_counts_agree_with_their_nouns(feature, value, expected):
    """The generated sentence is the project's deliverable and its readability
    is what Chapter Five measures, so "1 dots" is a defect, not a detail."""
    assert describe(feature, value) == expected
