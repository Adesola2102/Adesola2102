"""
Live pipeline view - the same analysis the main interface performs, but with
every layer boundary surfaced as it happens.

``app/streamlit_app.py`` shows an input and a sentence; everything between
them is invisible. This runs the identical ``PhishingExplanationPipeline`` on
whatever the user submits and renders each stage as it completes, so the path
from a raw string through feature extraction, the Random Forest vote, the two
XAI attributions, reconciliation and the coherence check can be followed on
any input, not a prepared one.

The real pipeline completes in roughly two tenths of a second, which is too
fast to watch. A pacing delay is therefore inserted *between* stages for
demonstration. It is presentation only: each stage reports the wall-clock
time its own computation actually took, measured around the call itself and
excluding the delay, so what is displayed stays honest about how fast the
system is.

Run with:
    streamlit run app/pipeline_view.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from phishguard.data.email_features import (  # noqa: E402
    FEATURE_NAMES as EMAIL_FEATURES,
)
from phishguard.data.email_features import EmailInput, extract_email_features  # noqa: E402
from phishguard.data.url_features import FEATURE_NAMES as URL_FEATURES  # noqa: E402
from phishguard.data.url_features import extract_url_features, is_analysable_url  # noqa: E402
from phishguard.pipeline import PhishingExplanationPipeline  # noqa: E402
from phishguard.translation.phrases import describe, is_coherent  # noqa: E402

st.set_page_config(page_title="PhishGuard - Live Pipeline View", page_icon="🛡️",
                   layout="wide")


@st.cache_resource(show_spinner="Loading classifiers, SHAP/LIME explainers and translation layer...")
def get_pipeline() -> PhishingExplanationPipeline:
    return PhishingExplanationPipeline()


def ms(seconds: float) -> str:
    return f"{seconds * 1000:.1f} ms"


def run_pipeline(kind: str, raw: str, email: EmailInput | None, pace: float,
                 keep_open: bool) -> None:
    pipeline = get_pipeline()
    is_url = kind == "url"
    names = URL_FEATURES if is_url else EMAIL_FEATURES
    clf = pipeline.url_classifier if is_url else pipeline.email_classifier
    explainer = pipeline.url_explainer if is_url else pipeline.email_explainer
    total = 0.0

    # ------------------------------------------------------- 1. input
    with st.status("**Layer 1 - Input**", expanded=True) as s:
        t = time.perf_counter()
        ok = is_analysable_url(raw) if is_url else bool((email.subject + email.body).strip())
        dt = time.perf_counter() - t
        total += dt
        st.code(raw, language=None)
        if not ok:
            s.update(label=f"**Layer 1 - Input** · rejected · {ms(dt)}",
                     state="error", expanded=True)
            st.error(
                "This input is not structurally analysable, so the pipeline stops here. "
                "No verdict and no explanation are produced - the system declines rather "
                "than guessing."
            )
            return
        st.success(f"Accepted - structurally analysable. Validation took {ms(dt)}.")
        s.update(label=f"**Layer 1 - Input** · accepted · {ms(dt)}", state="complete", expanded=keep_open)
    time.sleep(pace)

    # ------------------------------------- 2. feature extraction
    with st.status("**Layer 2 - Feature extraction**", expanded=True) as s:
        t = time.perf_counter()
        feats = extract_url_features(raw) if is_url else extract_email_features(email)
        dt = time.perf_counter() - t
        total += dt
        st.caption(
            f"The string becomes {len(names)} numbers. This is the only step that "
            "loses information - nothing downstream sees the original text."
        )
        frame = pd.DataFrame(
            [{"feature": n, "value": round(float(feats[n]), 4)} for n in names]
        )
        c1, c2, c3 = st.columns(3)
        third = (len(frame) + 2) // 3
        for col, chunk in zip((c1, c2, c3),
                              (frame[:third], frame[third:2 * third], frame[2 * third:])):
            col.dataframe(chunk, hide_index=True, use_container_width=True)
        s.update(label=f"**Layer 2 - Feature extraction** · {len(names)} features · {ms(dt)}",
                 state="complete", expanded=keep_open)
    time.sleep(pace)

    # ------------------------------------------------ 3. classification
    with st.status("**Layer 3 - Classification (Random Forest, 300 trees)**",
                   expanded=True) as s:
        t = time.perf_counter()
        proba = clf.predict_proba(feats)[0]
        label, confidence = clf.verdict(feats)
        dt = time.perf_counter() - t
        total += dt
        classes = [str(c) for c in clf.classes_()]
        cols = st.columns(len(classes) + 1)
        for col, cls, pr in zip(cols, classes, proba):
            col.metric(f"{cls}", f"{pr * 300:.1f} / 300", f"{pr:.1%} of trees")
        cols[-1].metric("Verdict", label.upper(), f"{confidence:.1%} confidence")
        st.progress(float(confidence))
        st.caption(
            "Confidence is the proportion of the 300 trees that voted for the "
            "predicted class - not a calibrated probability."
        )
        s.update(label=f"**Layer 3 - Classification** · {label.upper()} · {ms(dt)}",
                 state="complete", expanded=keep_open)
    time.sleep(pace)

    # ----------------------------------------------- 4. XAI attribution
    with st.status("**Layer 4 - XAI attribution (SHAP + LIME)**", expanded=True) as s:
        t = time.perf_counter()
        shap_result = explainer.explain_shap(feats)
        t_shap = time.perf_counter() - t
        t2 = time.perf_counter()
        lime_result = explainer.explain_lime(feats)
        t_lime = time.perf_counter() - t2
        dt = t_shap + t_lime
        total += dt

        pos = classes.index("phishing")
        contrib_sum = sum(c.contribution for c in shap_result.contributions)
        reconstructed = shap_result.base_value + contrib_sum

        left, right = st.columns(2)
        with left:
            st.markdown(f"**SHAP** - exact Shapley values · {ms(t_shap)}")
            sdf = pd.DataFrame(
                [{"feature": c.feature, "contribution": round(c.contribution, 4)}
                 for c in shap_result.top(8)]
            )
            st.bar_chart(sdf.set_index("feature")["contribution"])
            st.caption(
                f"base {shap_result.base_value:.4f} + contributions {contrib_sum:+.4f} "
                f"= **{reconstructed:.4f}**, and the model predicted "
                f"**{proba[pos]:.4f}** - the attribution reconstructs the prediction "
                f"to {abs(reconstructed - proba[pos]):.1e}."
            )
        with right:
            st.markdown(f"**LIME** - local linear approximation · {ms(t_lime)}")
            ldf = pd.DataFrame(
                [{"feature": c.feature, "contribution": round(c.contribution, 4)}
                 for c in lime_result.top(8)]
            )
            st.bar_chart(ldf.set_index("feature")["contribution"])
            st.caption("Fitted on 400 perturbed samples drawn around this one input.")
        s.update(label=f"**Layer 4 - XAI attribution** · SHAP {ms(t_shap)}, "
                       f"LIME {ms(t_lime)}", state="complete", expanded=keep_open)
    time.sleep(pace)

    # ------------------------------------------------ 5. reconciliation
    with st.status("**Layer 5 - Reconciliation**", expanded=True) as s:
        t = time.perf_counter()

        def normalise(result):
            m = max((abs(c.contribution) for c in result.contributions), default=1.0) or 1.0
            return m, {c.feature: c.contribution / m for c in result.contributions}

        s_max, s_norm = normalise(shap_result)
        l_max, l_norm = normalise(lime_result)
        values = {c.feature: c.raw_value for c in shap_result.contributions}
        combined = {f: (s_norm[f] + l_norm.get(f, 0.0)) / 2 for f in s_norm}
        ranking = sorted(combined.items(), key=lambda kv: abs(kv[1]), reverse=True)
        dt = time.perf_counter() - t
        total += dt

        st.caption(
            f"Largest SHAP weight {s_max:.4f}; largest LIME weight {l_max:.4f} - "
            f"{max(s_max, l_max) / min(s_max, l_max):.1f}× apart, so each is divided "
            "by its own maximum before the two are averaged."
        )
        st.dataframe(
            pd.DataFrame([
                {"feature": f, "SHAP (norm)": round(s_norm[f], 3),
                 "LIME (norm)": round(l_norm.get(f, 0.0), 3), "mean": round(sc, 3)}
                for f, sc in ranking[:8]
            ]), hide_index=True, use_container_width=True)
        s.update(label=f"**Layer 5 - Reconciliation** · {ms(dt)}", state="complete", expanded=keep_open)
    time.sleep(pace)

    # --------------------------------------------- 6. coherence check
    with st.status("**Layer 6 - Coherence check**", expanded=True) as s:
        t = time.perf_counter()
        positive = label == "phishing"
        aligned = [(f, sc) for f, sc in ranking
                   if (sc > 0) == positive and abs(sc) > 1e-6]
        kept = [(f, sc) for f, sc in aligned if is_coherent(f, values[f], label)]
        dropped = [(f, sc) for f, sc in aligned if not is_coherent(f, values[f], label)]
        dt = time.perf_counter() - t
        total += dt

        st.markdown(
            f"{len(ranking)} features → **{len(aligned)}** point the same way as the "
            f"verdict → **{len(kept)}** survive the coherence check"
            + (f" → **{len(dropped)} rejected**" if dropped else "")
        )
        if dropped:
            st.error("**Rejected - these would have been false if spoken aloud:**")
            for f, sc in dropped:
                rank = [x[0] for x in ranking].index(f) + 1
                st.markdown(
                    f"- `{f}` = {values[f]:.2f} (ranked #{rank} by weight) - would have "
                    f"claimed *“it {describe(f, values[f])}”* as a reason this is "
                    f"**{label.upper()}**."
                )
            st.caption(
                "These stay visible in the raw attribution panel. They are excluded "
                "from the sentence, not hidden from the user."
            )
        else:
            st.success("Every aligned factor is coherent on this input - nothing rejected.")
        s.update(label=f"**Layer 6 - Coherence check** · {len(dropped)} rejected · {ms(dt)}",
                 state="complete", expanded=keep_open)
    time.sleep(pace)

    # ------------------------------------------------- 7. generation
    with st.status("**Layer 7 - Translation and generation**", expanded=True) as s:
        # Only the translation step is timed here. Calling analyze_url/analyze_email
        # would re-run feature extraction, the forest and both explainers, and report
        # all of that as the generation cost.
        t = time.perf_counter()
        translation = pipeline.translator.translate(
            shap_result, lime_result, input_type=kind
        )
        dt = time.perf_counter() - t
        total += dt
        for feature, value, _ in translation.factors_used:
            st.markdown(f"`{feature}` = {value:.2f} → *“{describe(feature, value)}”*")
        with st.expander("The exact prompt handed to the generation backend"):
            st.code(translation.prompt, language="text")
        st.caption(f"Backend: `{translation.backend_name}` - deterministic, "
                   "offline, no network call.")
        s.update(label=f"**Layer 7 - Translation and generation** · {ms(dt)}",
                 state="complete", expanded=keep_open)
    time.sleep(pace)

    # ----------------------------------------------------- 8. output
    st.markdown("### Final output")
    colour = "🔴" if label == "phishing" else "🟢"
    st.subheader(f"{colour} {label.upper()} - {confidence:.1%} confidence")
    st.info(translation.sentence)
    st.caption(
        f"Total computation across all layers: **{ms(total)}**. The pauses between "
        "layers above are a presentation delay so each stage can be read; they are "
        "not part of the system's running time."
    )


def main() -> None:
    st.title("🛡️ PhishGuard - Live Pipeline View")
    st.caption(
        "The same analysis the main interface performs, with every layer surfaced "
        "as it runs. Works on any input."
    )

    pace = st.sidebar.slider(
        "Pause between layers (seconds)", 0.0, 3.0, 1.2, 0.1,
        help="Presentation delay only. Each layer reports the time its own "
             "computation actually took.",
    )
    st.sidebar.caption(
        "The pipeline itself completes in roughly 200 ms, which is too fast to "
        "watch. This delay paces the display; it does not slow the system."
    )

    keep_open = st.sidebar.checkbox(
        "Keep each layer expanded", value=True,
        help="Leave on for demonstrating; turn off for a compact summary.",
    )

    url_tab, email_tab = st.tabs(["🔗 Analyse a URL", "📧 Analyse an email"])

    with url_tab:
        url = st.text_input("Web address", placeholder="http://example.com/login",
                            key="url_in")
        st.caption("Try anything - including something that is not a link at all.")
        if st.button("Run the pipeline", key="run_url", type="primary"):
            if not url.strip():
                st.warning("Enter a web address first.")
            else:
                run_pipeline("url", url.strip(), None, pace, keep_open)

    with email_tab:
        subject = st.text_input("Subject", key="subj_in")
        body = st.text_area("Body", height=160, key="body_in")
        if st.button("Run the pipeline", key="run_email", type="primary"):
            if not (subject.strip() or body.strip()):
                st.warning("Enter a subject or a body first.")
            else:
                run_pipeline("email", f"{subject}\n\n{body}",
                             EmailInput(subject=subject, body=body), pace, keep_open)


if __name__ == "__main__":
    main()
