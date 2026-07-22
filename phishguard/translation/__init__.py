from phishguard.translation.backend import LLMBackend, TemplateBackend, get_default_backend
from phishguard.translation.translator import PlainLanguageTranslator, TranslationResult

__all__ = [
    "LLMBackend",
    "TemplateBackend",
    "get_default_backend",
    "PlainLanguageTranslator",
    "TranslationResult",
]
