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
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
EMAIL_SAMPLE_PER_CLASS = 3000
URL_SAMPLE_PER_CLASS = 6000

PHISHING_EMAIL_REPO = (
    "https://raw.githubusercontent.com/rokibulroni/Phishing-Email-Dataset/main"
)
NAZARIO_URL = f"{PHISHING_EMAIL_REPO}/Nazario.csv"
NIGERIAN_FRAUD_URL = f"{PHISHING_EMAIL_REPO}/Nigerian_Fraud.csv"
ENRON_HAM_URL = f"{PHISHING_EMAIL_REPO}/Enron.csv"
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
