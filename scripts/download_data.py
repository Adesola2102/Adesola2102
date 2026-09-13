"""
Dataset acquisition script for the Explainable Phishing Translation Layer project.

Downloads and samples the raw benchmark data used to train the email and URL
phishing classifiers, then writes fixed-size, reproducible CSV samples into
``data/raw/`` so the rest of the pipeline never needs network access again
(see NFR5/FR7 in Chapter Three - local, offline execution).

Data sources
------------
Email (genuine phishing/fraud vs. legitimate):
    - Phishing: the Nazario Phishing Corpus (Nazario, 2004-2007) plus a
      Nigerian/"419" advance-fee fraud email set, both genuine real-world
      malicious/social-engineering email (not spam-relabeled-as-phishing),
      parsed into CSV and republished by rokibulroni:
      https://github.com/rokibulroni/Phishing-Email-Dataset
      (files: Nazario.csv, Nigerian_Fraud.csv)
    - Legitimate: the ham (non-spam) portion of the Enron corpus (Metsis,
      Androutsopoulos & Paliouras, 2006), republished as CSV in the same
      collection (file: Enron.csv, label == 0).

    NOTE ON AN EARLIER SUBSTITUTION (since corrected): the project proposal
    names the Nazario Phishing Corpus as the intended email phishing source.
    An earlier build of this script used the Enron corpus's *spam* class
    (relabeled "phishing") instead, because Nazario's original distribution
    point (an ad-hoc mailing-list archive at monkey.org, with no stable
    direct-download URL) was not reliably fetchable from an unattended,
    non-interactive build environment. Spam and phishing overlap in surface
    features but are not the same category, which measurably affected
    classifier accuracy. The rokibulroni mirror above republishes the actual
    Nazario corpus (plus Nigerian Fraud emails) as directly-fetchable CSVs,
    so this build now uses the originally-proposed source. A few
    mbox-parsing artifacts (e.g. "DON'T DELETE THIS MESSAGE -- FOLDER
    INTERNAL DATA" placeholder rows, ~0.5% of rows) are filtered out below.

URL (phishing vs. legitimate, full URLs with real paths/query strings):
    - Phishing: PhishTank's verified-phish feed, plus Phishing.Database's
      full-link list (not just its bare-domain list), combined and
      deduplicated:
      https://github.com/ProKn1fe/phishtank-database (PhishTank mirror,
        refreshed every 24h - phishtank.com's own data.phishtank.com feed
        is not directly reachable from this build environment's network
        policy, so this GitHub mirror is used instead)
      https://github.com/mitchellkrogza/Phishing.Database
        (file: phishing-links-ACTIVE.txt)
    - Legitimate: a labelled URL set giving many distinct real domains
      *with* their real paths (not just bare domains), republished on
      GitHub:
      https://github.com/jishnusaurav/Phishing-attack-PCAP-analysis-using-scapy
      (file: Phishing-Website-Detection/datasets/legitimate-urls.csv,
      reconstructed from its Protocol/Domain/Path columns)

    NOTE ON SAMPLE SIZE (~1,000/class instead of the email side's 3,000):
    hundreds of thousands of phishing URLs are readily available, but a
    directly-fetchable *legitimate* URL corpus that preserves genuine paths
    and query strings (rather than bare domains) is comparatively scarce -
    the source above tops out around 1,000 usable rows across ~670 distinct
    domains. The alternative (pad the legitimate class with bare domains
    from a larger list, e.g. OpenDNS's top-domains list, to hit a bigger
    sample size) was rejected: it would let the classifier partly "cheat"
    by learning "has a path/query -> more likely phishing" - an artefact of
    how the data was built, not a genuine phishing signal. A smaller but
    structurally matched dataset, where both classes have genuine path/
    query diversity, is preferred over a larger but structurally lopsided
    one.

    NOTE ON SCHEME (http/https): the legitimate-URL source above predates
    HTTPS's near-universal adoption, so *as collected*, 100% of its rows
    are http:// while the (currently-live) phishing feeds are ~79% https -
    a dataset-vintage confound with nothing to do with phishing (verified:
    with scheme left as-collected, has_https became the single dominant
    trained feature at 41% importance, for entirely the wrong reason).
    build_url_dataset() re-randomizes scheme independent of label (70%
    https, matching modern web-wide adoption) rather than letting the
    classifier learn "which decade is this URL from" as a phishing proxy -
    the same class of shortcut the URL scheme was already kept independent
    of label to avoid before real full URLs replaced bare domains here.

Usage
-----
    python scripts/download_data.py

Re-running this script overwrites data/raw/email_dataset.csv and
data/raw/url_dataset.csv with a freshly-sampled (but seeded/reproducible) cut
of the latest upstream lists.
"""

from __future__ import annotations

import io
import random
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
EMAIL_SAMPLE_PER_CLASS = 3000
URL_SAMPLE_PER_CLASS = 1000

PHISHING_EMAIL_REPO = (
    "https://raw.githubusercontent.com/rokibulroni/Phishing-Email-Dataset/main"
)
NAZARIO_URL = f"{PHISHING_EMAIL_REPO}/Nazario.csv"
NIGERIAN_FRAUD_URL = f"{PHISHING_EMAIL_REPO}/Nigerian_Fraud.csv"
ENRON_HAM_URL = f"{PHISHING_EMAIL_REPO}/Enron.csv"
PHISHTANK_MIRROR_URL = (
    "https://raw.githubusercontent.com/ProKn1fe/phishtank-database/"
    "master/online-valid.json"
)
PHISHING_DATABASE_LINKS_URL = (
    "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/"
    "master/phishing-links-ACTIVE.txt"
)
LEGIT_URLS_WITH_PATHS_URL = (
    "https://raw.githubusercontent.com/jishnusaurav/"
    "Phishing-attack-PCAP-analysis-using-scapy/master/"
    "Phishing-Website-Detection/datasets/legitimate-urls.csv"
)


def fetch(url: str, timeout: int = 60) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def _drop_mbox_artifacts(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out mbox-folder-internal placeholder rows left over from
    parsing the raw Nazario/Nigerian-Fraud mailbox archives into CSV."""
    sender = df.get("sender", pd.Series("", index=df.index)).fillna("")
    subject = df["subject"].fillna("")
    body = df["body"].fillna("")
    junk = (
        sender.str.contains("MAILER-DAEMON", na=False)
        | subject.str.contains("DON'T DELETE THIS MESSAGE", na=False)
        | (body.str.strip().str.len() < 5)
    )
    return df[~junk]


def build_email_dataset() -> None:
    print(f"Downloading genuine phishing corpus from {NAZARIO_URL} ...")
    nazario = pd.read_csv(io.BytesIO(fetch(NAZARIO_URL)))
    print(f"Downloading Nigerian Fraud corpus from {NIGERIAN_FRAUD_URL} ...")
    nigerian = pd.read_csv(io.BytesIO(fetch(NIGERIAN_FRAUD_URL)))
    print(f"Downloading Enron corpus (for the ham/legitimate class) from {ENRON_HAM_URL} ...")
    enron = pd.read_csv(io.BytesIO(fetch(ENRON_HAM_URL)))

    phishing = pd.concat([nazario, nigerian], ignore_index=True)
    phishing = _drop_mbox_artifacts(phishing)
    phishing = phishing[["subject", "body"]].copy()
    phishing["subject"] = phishing["subject"].fillna("")
    phishing["body"] = phishing["body"].fillna("")
    phishing["label"] = "phishing"

    legitimate = enron[enron["label"].astype(str) == "0"][["subject", "body"]].copy()
    legitimate["subject"] = legitimate["subject"].fillna("")
    legitimate["body"] = legitimate["body"].fillna("")
    legitimate["label"] = "legitimate"

    df = pd.concat([phishing, legitimate], ignore_index=True)

    rng = random.Random(SEED)
    parts = []
    for label, group in df.groupby("label"):
        n = min(EMAIL_SAMPLE_PER_CLASS, len(group))
        idx = rng.sample(list(group.index), n)
        parts.append(group.loc[idx])
    sample = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)

    out_path = RAW_DIR / "email_dataset.csv"
    sample.to_csv(out_path, index=False)
    print(f"Wrote {len(sample)} labelled emails -> {out_path}")
    print(sample["label"].value_counts())


def build_url_dataset() -> None:
    print(f"Downloading PhishTank feed (GitHub mirror) from {PHISHTANK_MIRROR_URL} ...")
    phishtank = pd.read_json(io.BytesIO(fetch(PHISHTANK_MIRROR_URL)))
    phishtank_urls = phishtank["url"].dropna().astype(str).tolist()

    print(f"Downloading Phishing.Database full links from {PHISHING_DATABASE_LINKS_URL} ...")
    links_raw = fetch(PHISHING_DATABASE_LINKS_URL).decode("utf-8", errors="ignore")
    database_urls = [
        line.strip()
        for line in links_raw.splitlines()
        if line.strip().startswith(("http://", "https://"))
    ]

    print(f"Downloading legitimate URLs (with real paths) from {LEGIT_URLS_WITH_PATHS_URL} ...")
    legit_df = pd.read_csv(io.BytesIO(fetch(LEGIT_URLS_WITH_PATHS_URL)))
    legit_urls = (
        legit_df["Protocol"].astype(str)
        + "://"
        + legit_df["Domain"].astype(str)
        + legit_df["Path"].fillna("").astype(str)
    ).tolist()

    rng = random.Random(SEED)

    # dict.fromkeys dedupes exact-string overlap (Phishing.Database partly
    # aggregates from other feeds, PhishTank included) while preserving order.
    phishing_pool = list(dict.fromkeys(phishtank_urls + database_urls))
    phishing_sample = rng.sample(
        phishing_pool, min(URL_SAMPLE_PER_CLASS, len(phishing_pool))
    )

    legit_pool = list(dict.fromkeys(legit_urls))
    legit_sample = rng.sample(legit_pool, min(URL_SAMPLE_PER_CLASS, len(legit_pool)))

    # NOTE ON SCHEME (http/https): the legitimate-URL source is an older
    # (pre-HTTPS-ubiquity) crawl, while PhishTank/Phishing.Database reflect
    # today's web - so *as observed*, scheme is almost perfectly correlated
    # with label (100% of legitimate rows are http, ~79% of phishing rows
    # are https) purely because of when each source was collected, not
    # because that reflects reality. Left alone, has_https became the
    # single dominant feature (41% of importance) for entirely the wrong
    # reason. Re-randomizing scheme independent of label - as this script
    # already did before real full URLs replaced bare domains - removes
    # that dataset-vintage confound instead of letting the classifier learn
    # "which decade is this URL from" as a proxy for phishing.
    def _rerandomize_scheme(url: str) -> str:
        scheme = "https" if rng.random() < 0.7 else "http"
        return re.sub(r"^https?://", f"{scheme}://", url, count=1)

    rows = [{"url": _rerandomize_scheme(u), "label": "phishing"} for u in phishing_sample]
    rows += [{"url": _rerandomize_scheme(u), "label": "legitimate"} for u in legit_sample]

    df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    out_path = RAW_DIR / "url_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} labelled URLs -> {out_path}")
    print(df["label"].value_counts())


def main() -> None:
    build_email_dataset()
    build_url_dataset()


if __name__ == "__main__":
    sys.exit(main())
