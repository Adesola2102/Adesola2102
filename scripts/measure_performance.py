"""Measure start-up and per-analysis latency for the PhishGuard pipeline.

Writes ``reports/performance_metrics.json`` with the measured timings
alongside the hardware and library versions in force, so that the figures
reported in Chapter Four can be reproduced rather than assumed.

Two start-up figures are reported because they differ by a large factor:

* *cold*  - the model files are not held in the operating system file cache.
  The cache is dropped via ``/proc/sys/vm/drop_caches`` where the running
  user is permitted to do so; when it is not, the run is recorded as
  ``cache_dropped: false`` and the cold figure should be read as an
  ordinary fresh-process start rather than a true cold one.
* *warm*  - the same load repeated with the files already cached.

Each start-up measurement runs in a fresh subprocess, so it includes
interpreter start, imports and unpickling, which is what a user waits for.

Run from the project root::

    python scripts/measure_performance.py

To approximate a single-core machine, pin the process and disable the
maths-library thread pools::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        taskset -c 0 python scripts/measure_performance.py
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_DIR = ROOT / "models"
OUT_PATH = ROOT / "reports" / "performance_metrics.json"

ANALYSIS_RUNS = 15

SAMPLE_URL = "http://maceoandstacey.com/updates/webmail-portal-rd337"
SAMPLE_EMAIL_SUBJECT = "Urgent: verify your account"
SAMPLE_EMAIL_BODY = (
    "Dear customer, our records show your account will be suspended. "
    "Please confirm your details immediately to avoid interruption."
)

# Loads both classifiers in a fresh interpreter and prints the elapsed seconds.
LOAD_SNIPPET = """
import time, warnings
warnings.filterwarnings("ignore")
start = time.perf_counter()
import joblib
joblib.load({email!r})
joblib.load({url!r})
print(time.perf_counter() - start)
"""


def drop_file_cache() -> bool:
    """Best-effort drop of the OS page cache. True if it actually happened."""
    try:
        subprocess.run(["sync"], check=True)
        with open("/proc/sys/vm/drop_caches", "w") as handle:
            handle.write("3")
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def time_model_load() -> float:
    """Seconds to load both classifiers in a fresh interpreter."""
    snippet = LOAD_SNIPPET.format(
        email=str(MODEL_DIR / "email_classifier.joblib"),
        url=str(MODEL_DIR / "url_classifier.joblib"),
    )
    proc = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    return float(proc.stdout.strip().splitlines()[-1])


def summarise(samples: list[float]) -> dict:
    return {
        "runs": len(samples),
        "mean_s": round(statistics.mean(samples), 4),
        "median_s": round(statistics.median(samples), 4),
        "min_s": round(min(samples), 4),
        "max_s": round(max(samples), 4),
    }


def measure_analysis() -> dict:
    import warnings

    warnings.filterwarnings("ignore")
    from phishguard.data.email_features import EmailInput
    from phishguard.pipeline import PhishingExplanationPipeline

    pipeline = PhishingExplanationPipeline()
    email = EmailInput(subject=SAMPLE_EMAIL_SUBJECT, body=SAMPLE_EMAIL_BODY)

    # One untimed pass each, so lazy initialisation is not charged to run one.
    pipeline.analyze_url(SAMPLE_URL)
    pipeline.analyze_email(email)

    url_times, email_times = [], []
    for _ in range(ANALYSIS_RUNS):
        start = time.perf_counter()
        pipeline.analyze_url(SAMPLE_URL)
        url_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        pipeline.analyze_email(email)
        email_times.append(time.perf_counter() - start)

    return {"url_analysis": summarise(url_times), "email_analysis": summarise(email_times)}


def library_versions() -> dict:
    from importlib import metadata

    # Distribution name differs from the import name for scikit-learn, and
    # lime exposes no __version__ attribute, so fall back to package metadata.
    dist_names = {"sklearn": "scikit-learn"}
    versions = {}
    for name in ("sklearn", "shap", "lime", "pandas", "numpy", "joblib"):
        version = getattr(__import__(name), "__version__", None)
        if version is None:
            try:
                version = metadata.version(dist_names.get(name, name))
            except metadata.PackageNotFoundError:
                version = None
        versions[name] = version
    return versions


def cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def total_memory_gb() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                return round(int(line.split()[1]) / 1024 / 1024, 1)
    except (OSError, ValueError):
        pass
    return None


def main() -> None:
    model_sizes = {
        path.name: path.stat().st_size
        for path in sorted(MODEL_DIR.glob("*.joblib"))
    }
    total_bytes = sum(model_sizes.values())

    cache_dropped = drop_file_cache()
    cold_s = time_model_load()
    warm_s = time_model_load()

    results = {
        "environment": {
            "cpu_model": cpu_model(),
            "cpu_count_total": os.cpu_count(),
            "cpus_available_to_process": len(os.sched_getaffinity(0)),
            "total_memory_gb": total_memory_gb(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "thread_env": {
                var: os.environ.get(var)
                for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            },
            "libraries": library_versions(),
        },
        "models": {
            "files": model_sizes,
            "total_bytes": total_bytes,
            "total_mb": round(total_bytes / 1024 / 1024, 1),
        },
        "startup": {
            "cache_dropped": cache_dropped,
            "cold_load_s": round(cold_s, 4),
            "warm_load_s": round(warm_s, 4),
        },
        "analysis": measure_analysis(),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"\nWrote {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
