import pytest

from phishguard.data.url_features import FEATURE_NAMES, extract_url_features


def test_returns_all_expected_features():
    feats = extract_url_features("https://www.example.com/")
    assert set(feats.keys()) == set(FEATURE_NAMES)


def test_detects_ip_address_host():
    feats = extract_url_features("http://192.168.1.5/login")
    assert feats["has_ip_address"] == 1.0


def test_normal_domain_has_no_ip_flag():
    feats = extract_url_features("https://www.google.com/")
    assert feats["has_ip_address"] == 0.0


def test_detects_suspicious_tld():
    feats = extract_url_features("http://free-gift-cards.xyz/")
    assert feats["suspicious_tld"] == 1.0


def test_common_tld_not_flagged_suspicious():
    feats = extract_url_features("https://www.wikipedia.org/")
    assert feats["suspicious_tld"] == 0.0


def test_detects_url_shortener():
    feats = extract_url_features("http://bit.ly/abc123")
    assert feats["is_shortener"] == 1.0


def test_detects_at_symbol_obfuscation():
    feats = extract_url_features("http://real-bank.com@malicious.tk/login")
    assert feats["has_at_symbol"] == 1.0


def test_www_subdomain_excluded_from_subdomain_count():
    with_www = extract_url_features("https://www.example.com/")
    without_www = extract_url_features("https://example.com/")
    assert with_www["num_subdomains"] == without_www["num_subdomains"] == 0.0


def test_nested_subdomains_counted():
    feats = extract_url_features("http://login.secure.account-verify.example.com/")
    assert feats["num_subdomains"] >= 2.0


def test_multi_label_suffix_handled():
    # "co.uk" should be treated as a single suffix, not two separate labels
    feats = extract_url_features("https://shop.example.co.uk/")
    assert feats["num_subdomains"] == 1.0  # only "shop" is a real subdomain


def test_https_flag():
    assert extract_url_features("https://example.com/")["has_https"] == 1.0
    assert extract_url_features("http://example.com/")["has_https"] == 0.0


def test_brand_keyword_in_subdomain_detected_for_lookalike():
    feats = extract_url_features("http://paypal.security-check.xyz/")
    assert feats["brand_keyword_in_subdomain"] == 1.0


def test_brand_keyword_in_registrable_domain_not_flagged():
    # paypal.com itself should NOT be flagged as a misleading lookalike
    feats = extract_url_features("https://www.paypal.com/")
    assert feats["brand_keyword_in_subdomain"] == 0.0


def test_url_without_scheme_is_handled():
    feats = extract_url_features("www.example.com")
    assert feats["domain_length"] > 0


def test_malformed_bracketed_url_does_not_crash_the_extractor():
    """Regression test for the defect in Section 4.4.5.

    urlsplit() rejects an unbalanced square bracket as a malformed IPv6
    literal. The extractor is a total function by design (Section 4.2.2), so
    it must absorb that rather than propagate the exception: string-level
    features still describe the input, and host-derived features fall back to
    their empty defaults.
    """
    from urllib.parse import urlsplit

    malformed = "http://[unbalanced.example.com/login"
    with pytest.raises(ValueError):
        urlsplit(malformed)  # the condition this test exists for

    feats = extract_url_features(malformed)
    assert set(feats.keys()) == set(FEATURE_NAMES)
    assert all(isinstance(v, (int, float)) for v in feats.values())
    # String-level features still apply...
    assert feats["url_length"] == float(len(malformed))
    # ...while host-structure features take their absent defaults.
    assert feats["domain_length"] == 0.0
    assert feats["has_ip_address"] == 0.0
