"""
Dataset acquisition script for the Explainable Phishing Translation Layer project.

Downloads and samples the raw benchmark data used to train the email and URL
phishing classifiers, then writes fixed-size, reproducible CSV samples into
``data/raw/`` so the rest of the pipeline never needs network access again
(see NFR5/FR7 in Chapter Three - local, offline execution).

Data sources
------------
Email (legitimate vs. suspicious/social-engineering-style):
    Enron-Spam dataset (Metsis, Androutsopoulos & Paliouras, 2006), republished
    as a single CSV by MWiechmann:
    https://github.com/MWiechmann/enron_spam_data

    NOTE ON SUBSTITUTION: The project proposal names the Nazario Phishing
    Corpus and the CSDMC2010 Spam Corpus as the intended email sources. Both
    are distributed as ad-hoc mailing-list archives / competition mirrors that
    are not reliably fetchable from an unattended, non-interactive build
    environment (no stable direct-download URL, and several mirrors require
    manual sign-up). The Enron-Spam corpus is the closest widely-cited,
    directly-fetchable academic substitute with the same binary
    legitimate-vs-illegitimate email structure, and is used here instead. This
    substitution is documented as a build-environment constraint in Chapter
    Four (Challenges Encountered and Mitigation Strategies).

URL / domain (phishing vs. legitimate):
    - Phishing domains: Phishing.Database project (continuously updated,
      community-maintained active-phishing-domain blocklist), used here as a
      direct substitute for the PhishTank feed (which now requires a
      registered API key to query programmatically):
      https://github.com/mitchellkrogza/Phishing.Database
    - Legitimate domains: OpenDNS public top-domains list:
      https://github.com/opendns/public-domain-lists

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
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
EMAIL_SAMPLE_PER_CLASS = 3000
URL_SAMPLE_PER_CLASS = 6000

ENRON_ZIP_URL = (
    "https://raw.githubusercontent.com/MWiechmann/enron_spam_data/"
    "master/enron_spam_data.zip"
)
PHISHING_DOMAINS_URL = (
    "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/"
    "master/phishing-domains-ACTIVE.txt"
)
LEGIT_DOMAINS_URL = (
    "https://raw.githubusercontent.com/opendns/public-domain-lists/"
    "master/opendns-top-domains.txt"
)


def fetch(url: str, timeout: int = 60) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def build_email_dataset() -> None:
    print(f"Downloading Enron-Spam corpus from {ENRON_ZIP_URL} ...")
    raw = fetch(ENRON_ZIP_URL)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
        with zf.open(csv_name) as f:
            df = pd.read_csv(f)

    df = df.rename(columns={"Spam/Ham": "label_raw"})
    df["label"] = df["label_raw"].map({"spam": "phishing", "ham": "legitimate"})
    df["subject"] = df["Subject"].fillna("")
    df["body"] = df["Message"].fillna("")
    df = df[["subject", "body", "label"]].dropna(subset=["label"])

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
    print(f"Downloading phishing domains from {PHISHING_DOMAINS_URL} ...")
    phishing_raw = fetch(PHISHING_DOMAINS_URL).decode("utf-8", errors="ignore")
    phishing_domains = [
        line.strip() for line in phishing_raw.splitlines() if line.strip()
    ]

    print(f"Downloading legitimate domains from {LEGIT_DOMAINS_URL} ...")
    legit_raw = fetch(LEGIT_DOMAINS_URL).decode("utf-8", errors="ignore")
    legit_domains = [line.strip() for line in legit_raw.splitlines() if line.strip()]

    rng = random.Random(SEED)
    phishing_sample = rng.sample(
        phishing_domains, min(URL_SAMPLE_PER_CLASS, len(phishing_domains))
    )
    legit_sample = rng.sample(
        legit_domains, min(URL_SAMPLE_PER_CLASS, len(legit_domains))
    )

    # NOTE: scheme (http/https) is assigned independently of the label (70%
    # https, matching the overall modern-web HTTPS adoption rate) rather than
    # being deterministically tied to phishing/legitimate. The raw domain
    # lists do not record which scheme was actually observed, and encoding
    # scheme-by-label would let the classifier "cheat" by keying on an
    # artefact of this data-construction step instead of a genuine lexical
    # signal (HTTPS adoption among phishing sites is now high in practice).
    def _rows_for(domains: list, label: str) -> list:
        out = []
        for d in domains:
            scheme = "https" if rng.random() < 0.7 else "http"
            out.append({"url": f"{scheme}://{d}/", "label": label})
        return out

    rows = _rows_for(phishing_sample, "phishing")
    rows += _rows_for(legit_sample, "legitimate")

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
