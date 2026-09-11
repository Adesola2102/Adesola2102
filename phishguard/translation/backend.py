"""
LLM Translation Layer - pluggable inference backends (Chapter Three,
Component 4).

The methodology calls for "a lightweight, open-source large language model"
(Phi-3-mini / Mistral-7B / Llama-3.2-3B, run locally via llama.cpp / Hugging
Face Transformers). To keep the translation layer genuinely swappable, every
backend implements the same tiny ``LLMBackend`` interface:

    generate(prompt: str, context: dict) -> str

``TemplateBackend`` is the default backend shipped with this build. It is a
deterministic, controlled natural-language generator: given the same
structured ``context`` (verdict, confidence, ranked SHAP/LIME factors) it
always produces the same grammatically well-formed sentence, using the
phrase library in ``phishguard.translation.phrases``. It requires no model
download, no GPU, and no network access at all - which trivially satisfies
FR7 ("execute locally without internet access or API requests") and NFR5
(privacy) rather than merely approximating them.

``LlamaCppBackend`` is provided for the case where a quantized GGUF build of
one of the three candidate models (Phi-3-mini, Mistral-7B, Llama-3.2-3B) is
available on disk - it shells out to ``llama-cpp-python`` if installed and a
model path is configured. In the sandboxed environment this project was
built in, outbound access to Hugging Face (the natural source of GGUF
weights) is blocked by the environment's network policy, so this backend has
never been run against real weights; see Chapter Four, "Challenges
Encountered and Mitigation Strategies" for the full discussion.

What *is* covered by tests (``tests/test_llm_backend.py``) is everything the
class does around the model: the availability gate, the fallback to
``TemplateBackend`` when no local model is configured, and the
post-processing that enforces the single-sentence requirement on generated
output. The replaceability of the Generation Layer is covered separately by
an integration test that drives the real pipeline through a non-template
backend. None of that evidences the fluency of a real model's output - only
that swapping one in requires no code change, just
``PHISHGUARD_LLM_MODEL_PATH``.
"""

from __future__ import annotations

import os
import re
import textwrap
from abc import ABC, abstractmethod
from typing import Optional

from phishguard.translation.phrases import describe


class LLMBackend(ABC):
    """Common interface every translation backend must implement."""

    name = "base"

    @abstractmethod
    def generate(self, prompt: str, context: dict) -> str:
        """Return a single plain-language sentence explaining the verdict."""
        raise NotImplementedError

    def is_available(self) -> bool:
        return True


class TemplateBackend(LLMBackend):
    """Deterministic, offline, zero-dependency slot-filling generator.

    This is the default backend used by the shipped prototype. It builds the
    explanation directly from the structured ``context`` produced by
    ``PlainLanguageTranslator`` (verdict, confidence, ranked contributing
    factors from SHAP/LIME) rather than by parsing the free-text ``prompt``,
    since a template engine does not need natural-language understanding of
    its own prompt - it already has the structured data the prompt was built
    from. The ``prompt`` argument is still accepted (and returned as
    ``debug_prompt`` by the translator) purely so this backend is a drop-in
    replacement for a real LLM backend that *would* read the prompt text.
    """

    name = "template"

    def generate(self, prompt: str, context: dict) -> str:
        verdict = context["verdict"]
        confidence_pct = round(context["confidence"] * 100)
        input_noun = context.get("input_noun", "message")
        factors = context["factors"]  # list of (feature, value, is_risk)

        clauses = [describe(f, v) for f, v, _is_risk in factors]
        clauses = clauses[:3] if clauses else []

        if verdict == "phishing":
            if not clauses:
                reason = "it shows several combined statistical warning signs"
            elif len(clauses) == 1:
                reason = clauses[0]
            elif len(clauses) == 2:
                reason = f"{clauses[0]} and {clauses[1]}"
            else:
                reason = f"{clauses[0]}, {clauses[1]}, and {clauses[2]}"
            sentence = (
                f"This {input_noun} was flagged as PHISHING with {confidence_pct}% "
                f"confidence because it {reason}."
            )
        else:
            if not clauses:
                reason = "no significant phishing warning signs were detected"
            elif len(clauses) == 1:
                reason = clauses[0]
            elif len(clauses) == 2:
                reason = f"{clauses[0]} and {clauses[1]}"
            else:
                reason = f"{clauses[0]}, {clauses[1]}, and {clauses[2]}"
            sentence = (
                f"This {input_noun} looks LEGITIMATE with {confidence_pct}% confidence: "
                f"it {reason}."
            )
        return sentence


class LlamaCppBackend(LLMBackend):
    """Optional backend for a locally-hosted quantized GGUF model.

    Requires ``llama-cpp-python`` (pip-installable, CPU-only build works) and
    a GGUF model file (e.g. a 4-bit quantized Phi-3-mini) available on disk.
    Never downloads anything itself - the model file must already exist
    locally, which keeps this backend consistent with FR7/NFR5.
    """

    name = "llama-cpp"

    def __init__(self, model_path: Optional[str] = None, n_ctx: int = 2048,
                 max_tokens: int = 80, temperature: float = 0.2):
        self.model_path = model_path or os.environ.get("PHISHGUARD_LLM_MODEL_PATH")
        self.n_ctx = n_ctx
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._llm = None

    def is_available(self) -> bool:
        if not self.model_path or not os.path.exists(self.model_path):
            return False
        try:
            import llama_cpp  # noqa: F401
        except ImportError:
            return False
        return True

    def _load(self):
        if self._llm is None:
            from llama_cpp import Llama

            self._llm = Llama(model_path=self.model_path, n_ctx=self.n_ctx, verbose=False)
        return self._llm

    def generate(self, prompt: str, context: dict) -> str:
        llm = self._load()
        output = llm(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=["\n\n"],
        )
        text = output["choices"][0]["text"].strip()
        # Enforce the "single concise sentence" requirement (FR5) even if the
        # model rambles across several sentences.
        first_sentence = re.split(r"(?<=[.!?])\s", text.strip())[0]
        return textwrap.shorten(first_sentence, width=400, placeholder="...")


def get_default_backend() -> LLMBackend:
    """Selects the best available backend: a configured local GGUF model if
    present, otherwise the always-available template backend."""

    llama_backend = LlamaCppBackend()
    if llama_backend.is_available():
        return llama_backend
    return TemplateBackend()
