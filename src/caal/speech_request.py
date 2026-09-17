"""Immutable speech metadata carried with text across the SDK task boundary."""

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class SpeechProfile:
    language: str | None
    voice: str = "jarvis-designed"
    model: str = "qwen-trial"


speech_profile: ContextVar[SpeechProfile | None] = ContextVar("speech_profile", default=None)


class SpeechText(str):
    """A normal SDK text chunk with the producing turn's immutable profile."""

    __slots__ = ("_profile",)

    @property
    def profile(self):
        return self._profile

    def __setattr__(self, name, value):
        raise AttributeError("Speech metadata is immutable")

    def __new__(cls, text: str, language: str):
        value = super().__new__(cls, text)
        object.__setattr__(value, "_profile", SpeechProfile(language))
        return value


def turn_language(agent, chat_ctx) -> str:
    """Prefer the completed user turn's metadata over later session changes."""
    for item in reversed(chat_ctx.items):
        if getattr(item, "role", None) == "user":
            value = getattr(item, "extra", {}).get("caal_reply_language")
            if value in ("en", "es"):
                return value
            break
    return getattr(getattr(agent, "_language_session", None), "current", "en")
