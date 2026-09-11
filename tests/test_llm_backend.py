"""
Tests for the pluggable generation backend (Chapter Three, Component 4).

The project's design claim is that the Generation Layer is the replaceable
component: the same pipeline should drive a deterministic template generator
or a locally hosted quantized language model without change to any other
layer. No GGUF weights could be obtained in the build environment (Hugging
Face is blocked by network policy, see Chapter Four), so LlamaCppBackend is
never exercised against real weights here.

What can be tested without weights is everything the class does *around* the
model: the availability gate that decides whether a local model is usable at
all, the selection logic that falls back to the template generator when it is
not, and the post-processing that enforces the single-sentence requirement
(FR5) on output a language model has produced. Those are covered below, along
with a substitution test that runs the real pipeline end to end on a
non-template backend, which is the part that actually evidences the
replaceability claim.
"""

from __future__ import annotations

import sys
import types

import pytest

from phishguard.data.email_features import EmailInput
from phishguard.pipeline import PhishingExplanationPipeline
from phishguard.translation.backend import (
    LlamaCppBackend,
    LLMBackend,
    TemplateBackend,
    get_default_backend,
)
from tests.conftest import requires_trained_models


class _StubLlama:
    """Stands in for a loaded llama_cpp.Llama object.

    Returns the response shape llama-cpp-python actually produces,
    ``{"choices": [{"text": ...}]}``, so the post-processing under test is
    driven by the same structure a real model would hand it.
    """

    def __init__(self, text: str):
        self.text = text
        self.calls: list[dict] = []

    def __call__(self, prompt: str, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        return {"choices": [{"text": self.text}]}


def _backend_with(text: str) -> LlamaCppBackend:
    """A LlamaCppBackend whose model is already 'loaded' as a stub, bypassing
    _load() so no weights are needed."""
    backend = LlamaCppBackend(model_path="/nonexistent/model.gguf")
    backend._llm = _StubLlama(text)
    return backend


# --------------------------------------------------------------- availability


def test_is_unavailable_when_no_model_path_is_configured(monkeypatch):
    monkeypatch.delenv("PHISHGUARD_LLM_MODEL_PATH", raising=False)
    assert LlamaCppBackend().is_available() is False


def test_is_unavailable_when_configured_path_does_not_exist(monkeypatch):
    """The runtime is made to look installed so that a missing file is the only
    thing that can fail the check - otherwise this passes on a machine without
    llama-cpp-python regardless of whether the path is ever inspected."""
    monkeypatch.delenv("PHISHGUARD_LLM_MODEL_PATH", raising=False)
    monkeypatch.setitem(sys.modules, "llama_cpp", types.ModuleType("llama_cpp"))
    assert LlamaCppBackend(model_path="/no/such/model.gguf").is_available() is False


def test_is_unavailable_when_no_path_is_set_even_if_runtime_is_installed(monkeypatch):
    monkeypatch.delenv("PHISHGUARD_LLM_MODEL_PATH", raising=False)
    monkeypatch.setitem(sys.modules, "llama_cpp", types.ModuleType("llama_cpp"))
    assert LlamaCppBackend().is_available() is False


def test_reads_model_path_from_the_documented_environment_variable(monkeypatch, tmp_path):
    model = tmp_path / "phi3-mini-q4.gguf"
    model.write_bytes(b"not real weights")
    monkeypatch.setenv("PHISHGUARD_LLM_MODEL_PATH", str(model))
    assert LlamaCppBackend().model_path == str(model)


def test_is_unavailable_when_weights_exist_but_llama_cpp_is_not_installed(
    monkeypatch, tmp_path
):
    """A model file alone is not enough - the runtime has to be installed too."""
    model = tmp_path / "model.gguf"
    model.write_bytes(b"not real weights")
    monkeypatch.setitem(sys.modules, "llama_cpp", None)  # forces ImportError
    assert LlamaCppBackend(model_path=str(model)).is_available() is False


def test_is_available_when_both_weights_and_runtime_are_present(monkeypatch, tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"not real weights")
    monkeypatch.setitem(sys.modules, "llama_cpp", types.ModuleType("llama_cpp"))
    assert LlamaCppBackend(model_path=str(model)).is_available() is True


# ----------------------------------------------------------------- selection


def test_default_backend_falls_back_to_the_template_generator(monkeypatch):
    """With no local model configured the system must still work offline."""
    monkeypatch.delenv("PHISHGUARD_LLM_MODEL_PATH", raising=False)
    assert isinstance(get_default_backend(), TemplateBackend)


def test_default_backend_prefers_a_local_model_when_one_is_available(monkeypatch):
    monkeypatch.setattr(LlamaCppBackend, "is_available", lambda self: True)
    assert isinstance(get_default_backend(), LlamaCppBackend)


def test_llama_backend_satisfies_the_backend_interface():
    """Swapping generators must not require changes elsewhere, which holds
    only if the class really implements the shared interface."""
    backend = LlamaCppBackend()
    assert isinstance(backend, LLMBackend)
    assert backend.name == "llama-cpp"
    assert callable(backend.generate)


# ----------------------------------------------------- output post-processing


def test_rambling_model_output_is_reduced_to_one_sentence():
    """FR5 requires a single sentence. A language model asked for one will
    often produce several, so the backend has to enforce it rather than trust
    the model."""
    backend = _backend_with(
        "This link is dangerous. It uses an IP address instead of a domain. "
        "You should not click it. Contact your IT department."
    )
    sentence = backend.generate("prompt", {})
    assert sentence == "This link is dangerous."
    assert sentence.count(".") == 1


def test_leading_and_trailing_whitespace_is_stripped():
    backend = _backend_with("\n\n  The message is suspicious.  \n")
    assert backend.generate("prompt", {}) == "The message is suspicious."


def test_a_single_overlong_sentence_is_truncated():
    """Guards the UI against a model that never emits a full stop."""
    backend = _backend_with("word " * 400)
    sentence = backend.generate("prompt", {})
    assert len(sentence) <= 400
    assert sentence.endswith("...")


def test_generation_parameters_are_passed_through_to_the_model():
    backend = _backend_with("A verdict sentence.")
    backend.max_tokens = 64
    backend.temperature = 0.1
    backend.generate("the constructed prompt", {})

    call = backend._llm.calls[0]
    assert call["prompt"] == "the constructed prompt"
    assert call["max_tokens"] == 64
    assert call["temperature"] == 0.1


# ------------------------------------------------------- substitution proof


class _ScriptedLlmBackend(LLMBackend):
    """A non-template backend standing in for a local language model.

    Deliberately not a TemplateBackend subclass: if the pipeline can drive
    this, the Generation Layer is genuinely replaceable rather than replaceable
    in principle.
    """

    name = "scripted-llm"

    def __init__(self):
        self.contexts: list[dict] = []
        self.prompts: list[str] = []

    def generate(self, prompt: str, context: dict) -> str:
        self.prompts.append(prompt)
        self.contexts.append(context)
        return "Generated by a substituted backend."


@requires_trained_models
def test_pipeline_runs_end_to_end_on_a_substituted_backend():
    """The replaceability claim in Chapter Six, exercised rather than asserted:
    the real pipeline, real classifiers and real SHAP/LIME, with only the
    generator swapped out."""
    backend = _ScriptedLlmBackend()
    pipeline = PhishingExplanationPipeline(
        backend=backend, background_size=100, lime_samples=200
    )

    result = pipeline.analyze_url(
        "http://secure-paypal-verification.account-update.xyz/login.php?id=93820"
    )

    # The substituted generator produced the sentence, not the template one.
    assert result.translation.sentence == "Generated by a substituted backend."
    # Every layer beneath the generator still did its work.
    assert result.verdict == "phishing"
    assert result.confidence > 0.5
    assert result.shap.contributions
    assert result.lime.contributions
    # And the generator was handed the reconciled factors to work from.
    assert backend.prompts and backend.contexts


@requires_trained_models
def test_substituted_backend_receives_the_reconciled_factors():
    """What a language model would be given is the reconciled, coherence-
    filtered factor set - not the raw attribution dump."""
    backend = _ScriptedLlmBackend()
    pipeline = PhishingExplanationPipeline(
        backend=backend, background_size=100, lime_samples=200
    )
    pipeline.analyze_email(
        EmailInput(
            subject="URGENT: Verify your account NOW",
            body="Your account will be SUSPENDED. Click http://1.2.3.4/verify to confirm.",
        )
    )

    context = backend.contexts[0]
    assert context.get("verdict")
    assert context.get("factors"), "generator must receive ranked factors"
