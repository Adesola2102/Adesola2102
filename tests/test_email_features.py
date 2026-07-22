from phishguard.data.email_features import EmailInput, FEATURE_NAMES, extract_email_features


def test_returns_all_expected_features():
    feats = extract_email_features(EmailInput(subject="Hi", body="Just checking in."))
    assert set(feats.keys()) == set(FEATURE_NAMES)


def test_detects_urgent_language():
    feats = extract_email_features(
        EmailInput(subject="Act now", body="Your account will be suspended, act now!")
    )
    assert feats["urgent_keyword_count"] >= 2


def test_calm_email_has_no_urgent_language():
    feats = extract_email_features(
        EmailInput(subject="Lunch tomorrow?", body="Are you free for lunch tomorrow at noon?")
    )
    assert feats["urgent_keyword_count"] == 0


def test_detects_credential_request():
    feats = extract_email_features(
        EmailInput(subject="Reset", body="Please reset your password and confirm your username.")
    )
    assert feats["credential_keyword_count"] >= 2


def test_detects_generic_greeting():
    feats = extract_email_features(EmailInput(subject="", body="Dear customer, please respond."))
    assert feats["generic_greeting"] == 1.0


def test_personal_greeting_not_flagged_generic():
    feats = extract_email_features(EmailInput(subject="", body="Hi Sarah, please respond."))
    assert feats["generic_greeting"] == 0.0


def test_no_attachment_keyword_when_absent():
    feats = extract_email_features(
        EmailInput(subject="Quick question", body="Do you have five minutes to chat today?")
    )
    assert feats["attachment_keyword"] == 0.0


def test_attachment_keyword_detected():
    feats = extract_email_features(
        EmailInput(subject="Report", body="Please find the report attached.")
    )
    assert feats["attachment_keyword"] == 1.0


def test_embedded_ip_url_detected_in_body():
    feats = extract_email_features(
        EmailInput(subject="Verify", body="Click http://192.168.0.1/verify now.")
    )
    assert feats["any_embedded_ip_url"] == 1.0
    assert feats["num_urls_in_body"] == 1.0


def test_no_urls_in_plain_message():
    feats = extract_email_features(EmailInput(subject="Hi", body="No links here at all."))
    assert feats["num_urls_in_body"] == 0.0
    assert feats["any_embedded_ip_url"] == 0.0


def test_exclamation_and_caps_counted():
    feats = extract_email_features(
        EmailInput(subject="WARNING", body="ACT NOW!!! YOUR ACCOUNT IS AT RISK!!!")
    )
    assert feats["exclamation_count"] == 6
    assert feats["all_caps_word_ratio"] > 0.5


def test_header_fields_default_to_neutral_when_unknown():
    feats = extract_email_features(EmailInput(subject="Hi", body="Hello there."))
    assert feats["sender_domain_mismatch"] == 0.5
    assert feats["spf_fail"] == 0.5


def test_header_mismatch_detected_when_supplied():
    feats = extract_email_features(
        EmailInput(
            subject="Hi",
            body="Hello",
            sender_display_domain="paypal.com",
            sender_actual_domain="totally-not-paypal.tk",
        )
    )
    assert feats["sender_domain_mismatch"] == 1.0
