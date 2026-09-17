"""Which language JARVIS answers a turn in. Pure: no model call, no network.

The only new signal this needs is the one the speech server already returns on
every transcription (see reports/bilingual/DESIGN.md), so a turn costs nothing
extra to classify.

Two measured facts shape the rules:

* A short utterance is not evidence. The live Whisper server transcribes the
  Spanish ``Sí.`` as ``" See"`` and labels it ``en``. A session must therefore
  never change language on a brief acknowledgement, which is also what makes
  ``sí`` / ``okay`` feel sticky rather than jarring.
* Detection is the primary signal, but Spanish orthography (``¿``, ``¡``, ``ñ``,
  accented vowels, or several Spanish function words) is an unambiguous positive
  override: English has no comparable marker, and Whisper's bias runs toward
  English, so only the Spanish direction gets an orthographic escape hatch.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

AUTO = "auto"
EN = "en"
ES = "es"

SUPPORTED = (AUTO, EN, ES)
#: Languages a session can actually speak, in contrast to the ``auto`` setting.
SPOKEN = (EN, ES)

#: A turn shorter than this cannot be evidence on its own.
MIN_EVIDENTIAL_WORDS = 2

_SPANISH_LETTERS = set("ñáéíóúü")
#: Only characters English never uses. ``¿``/``¡`` are handled separately as a
#: stronger signal because no English text contains them at all.
_INVERTED = set("¿¡")

#: Spanish function words that are *not* also ordinary English words. Words like
#: ``no``, ``me``, ``son``, ``a`` and ``si`` were deliberately removed: they made
#: English sentences score as Spanish, which is the defect this replaces.
_SPANISH_FUNCTION_WORDS = frozenset(
    """
    el la los las un una unos unas del al pero porque que cual quien cuando donde
    para por con sin sobre entre hasta desde muy más menos también tambien ya sí
    está están estoy estamos estás eres somos ser estar hay tengo tienes tiene tenemos
    puedes puedo quiero quieres necesito necesitas mis tus sus nuestro nuestra
    te nos les esto esta eso esos esas aquí ahí allí ahora mañana hoy ayer
    correo correos cita citas recordatorio recordatorios lámpara luz oficina
    """.split()
)

#: English function words that are not Spanish words, used only to refuse a
#: Spanish detection on plainly English text. Whisper mislabels in both
#: directions; English is the incumbent behaviour and must not drift.
_ENGLISH_FUNCTION_WORDS = frozenset(
    """
    the and is are was were to of in on at for with about from that this these those
    it its my your our their he she we they you i but or if when what how why
    can will would should do does did have has had be been am there here
    please my me tomorrow today yesterday email emails calendar meeting reminder
    """.split()
)

#: Short acknowledgements and fillers. A turn made only of these is not evidence
#: in either direction: the live Whisper server transcribes Spanish ``Sí.`` as
#: ``" See"`` tagged ``en``. Nothing here authorizes anything — see
#: ``caal.handoff_intent`` for the confirmation vocabulary, which is separate and
#: deliberately does **not** contain ``see``.
_ACKNOWLEDGEMENTS = frozenset(
    """
    yes yeah yep yup ok okay okey alright right sure fine good great no nope nah
    see sea si sí claro vale bueno dale listo exacto correcto adelante
    thanks thank you gracias please por favor hmm uhm um uh mm hm
    """.split()
)

_LANGUAGE_WORDS = {
    "english": EN,
    "inglés": EN,
    "ingles": EN,
    "spanish": ES,
    "español": ES,
    "espanol": ES,
    "castellano": ES,
}

_INSTRUCTION_VERBS = (
    "habla|hablar|hablame|háblame|responde|responder|respóndeme|respondeme|contesta|contestar"
    "|cambia|cambiar|sigue|continúa|continua|dime|explica|escribe"
    "|speak|say|reply|respond|answer|switch|talk|change|use|continue|keep"
)

_INSTRUCTION = re.compile(
    rf"\b(?:{_INSTRUCTION_VERBS})\b[^.?!]{{0,40}}?\b({'|'.join(_LANGUAGE_WORDS)})\b",
    re.IGNORECASE,
)
#: The bare prepositional form: "en español", "in English".
_BARE_INSTRUCTION = re.compile(
    rf"\b(?:en|in)\s+(?:el\s+|the\s+)?({'|'.join(_LANGUAGE_WORDS)})\b", re.IGNORECASE
)
#: The natural typed/spoken switch: "Spanish please", "español por favor".
_POLITE_INSTRUCTION = re.compile(
    rf"\b({'|'.join(_LANGUAGE_WORDS)})\b\s*,?\s*(?:please|porfavor|por\s+favor)\b",
    re.IGNORECASE,
)
#: The whole turn is nothing but the language name: "castellano", "en español".
_NAMED_ONLY = re.compile(
    rf"^\W*(?:en\s+|in\s+)?({'|'.join(_LANGUAGE_WORDS)})\W*$", re.IGNORECASE
)
#: "I don't speak Spanish" / "no hablo español" is a statement, not a request.
_NEGATED = re.compile(
    r"\b(?:no|not|don'?t|doesn'?t|can'?t|won'?t|never|nunca|jamás|jamas)\b[^.?!]{0,30}?"
    r"\b(?:speak|speaks|speaking|understand|understands|read|reads|write|writes"
    r"|hablo|hablas|habla|hablar|hablamos|entiendo|entiendes|entiende|leo|lees)\b",
    re.IGNORECASE,
)

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


@dataclass(frozen=True)
class LanguageReading:
    """What the local model read one turn as. Holds no words of the turn.

    ``language`` is always exactly ``en`` or ``es``: it is produced only by
    :func:`read_semantic`, which is the one place a model-supplied value is
    turned into something this module will act on.

    ``switch_requested`` is the separate question of whether the speaker asked
    to *change* language rather than merely spoke one. It is the only reading
    allowed past a pinned account preference, which is why it is a field of its
    own rather than an inference from ``language``.
    """

    language: str
    switch_requested: bool = False


def read_semantic(language: object, *, switch: object = False) -> LanguageReading | None:
    """Validate a model-supplied language reading, or refuse it.

    Deterministic and closed: two strings are readings and everything else --
    another language, a language *name*, a different case, an empty string, a
    non-string -- is no reading at all. A refusal is not an error and not a
    guess; the caller falls back to the speech server evidence, so the worst a
    confused model can do here is cost the semantic reading.
    """
    if type(language) is not str or language not in SPOKEN:
        return None
    return LanguageReading(language, switch is True)


def normalize(preference: str | None) -> str:
    """Coerce a stored or supplied preference to a supported value."""
    if isinstance(preference, str) and preference in SUPPORTED:
        return preference
    return AUTO


def explicit_request(transcript: str | None) -> str | None:
    """The language the speaker asked for in so many words, or ``None``.

    This outranks detection: a direct instruction is the clearest signal there
    is, and it is the only thing allowed past the short-utterance guard.
    """
    if not isinstance(transcript, str) or not transcript.strip():
        return None
    if _NEGATED.search(transcript):
        return None
    for pattern in (_POLITE_INSTRUCTION, _NAMED_ONLY, _BARE_INSTRUCTION, _INSTRUCTION):
        match = pattern.search(transcript)
        if match:
            return _LANGUAGE_WORDS[match.group(1).lower()]
    return None


def _words(transcript: str) -> list[str]:
    return _WORD.findall(transcript.lower())


def _looks_spanish(transcript: str) -> bool:
    """Structural evidence strong enough to override an ``en`` detection.

    A diacritic alone is **not** enough. ``call José about the piñata order`` is
    an English sentence, and stage 1 flipped the whole session to Spanish on it.
    Spanish now has to show grammar, not just a character.
    """
    lowered = transcript.lower()
    if _INVERTED & set(lowered):
        return True
    words = _words(transcript)
    function_words = sum(1 for word in words if word in _SPANISH_FUNCTION_WORDS)
    letters = bool(_SPANISH_LETTERS & set(lowered))
    return function_words >= 2 or (function_words >= 1 and letters)


def _looks_english(transcript: str) -> bool:
    """Enough plain English function words to refuse a Spanish detection."""
    words = _words(transcript)
    return sum(1 for word in words if word in _ENGLISH_FUNCTION_WORDS) >= 2


def is_evidential(transcript: str | None) -> bool:
    """Whether a turn can be used as language evidence at all.

    Stage 1 used a four-word floor, which silently discarded real commands like
    ``apaga la luz``. The thing that actually makes a turn unusable is not its
    length but that it is a bare acknowledgement, so that is what is tested.
    """
    if not isinstance(transcript, str):
        return False
    words = _words(transcript)
    if len(words) < MIN_EVIDENTIAL_WORDS:
        return False
    return any(word not in _ACKNOWLEDGEMENTS for word in words)


def resolve(
    preference: str | None,
    *,
    detected: str | None = None,
    transcript: str | None = None,
    current: str | None = None,
    semantic: LanguageReading | None = None,
) -> str:
    """The language to answer this turn in.

    ``preference`` is the account setting (``auto``/``en``/``es``), ``detected``
    the code the speech server returned, ``transcript`` what it heard,
    ``current`` the language the session is already speaking, and ``semantic``
    what the local model read the turn as, if it read it at all.

    The order is fixed and is the whole policy:

    1. An explicit ask to switch -- read semantically, or matched in so many
       words -- outranks everything, including a pinned account preference.
       That is what makes "answer me in Spanish from now on" work on an
       account that is pinned to English.
    2. A pinned ``en``/``es`` preference. The account chose; an ordinary turn
       spoken in the other language does not un-choose it.
    3. The semantic reading, for a turn that is evidence at all. A bare
       ``sí``/``okay`` is not: it is answered in whatever is already being
       spoken, however the model read it.
    4. The speech-server evidence path, unchanged from stage 2. This is where
       a turn lands when the local model was unreachable, timed out, or said
       something unusable -- a failed reading costs the reading and nothing
       else.
    5. The language already being spoken.
    """
    settled = current if current in SPOKEN else EN
    if semantic is not None and semantic.switch_requested:
        return semantic.language
    instruction = explicit_request(transcript)
    if instruction:
        return instruction
    preference = normalize(preference)
    if preference in SPOKEN:
        return preference
    if not is_evidential(transcript):
        return settled
    if semantic is not None:
        return semantic.language
    text = transcript or ""
    if _looks_spanish(text):
        return ES
    if detected == ES and _looks_english(text):
        return EN
    if detected in SPOKEN:
        return detected
    if _looks_english(text):
        return EN
    return settled


#: Appended to the system instructions for the turn. English resolves to the
#: empty string, so an English session's prompt is byte-for-byte what it is
#: today and existing English behaviour cannot drift.
_DIRECTIVE_ES = (
    "Responde en español latinoamericano neutro. Mantén exactamente el mismo comportamiento, "
    "las mismas herramientas y las mismas confirmaciones que en inglés; solo cambia el idioma "
    "de tu respuesta hablada. No traduzcas ni alteres nombres propios, direcciones de correo "
    "(email), URLs, identificadores de cuenta, nombres de archivos ni horas y fechas: "
    "reprodúcelos exactamente como aparecen. No inventes información."
)

_DIRECTIVES = {ES: _DIRECTIVE_ES, EN: ""}


def reply_directive(language: str | None) -> str:
    """The instruction appended for a turn answered in ``language``."""
    return _DIRECTIVES.get(language or EN, "")


class LanguageSession:
    """The language one voice session is speaking, and how it may change.

    Semantics, deliberately narrow:

    * The stored account preference (``auto``/``en``/``es``) is read once, from
      the verified principal, when the session starts. ``en``/``es`` pin it.
    * A spoken instruction ("habla en español") is a conversational override
      that outranks the account preference **for this session only**; the stored
      preference is never written from here.
    * Otherwise ``auto`` follows the speaker, subject to the short-utterance
      guard in :func:`resolve`.
    """

    def __init__(self, preference: str | None = AUTO, *, default: str = EN) -> None:
        self.preference = normalize(preference)
        self.override: str | None = None
        self.current = self.preference if self.preference in SPOKEN else default

    @property
    def effective_preference(self) -> str:
        return self.override or self.preference

    def observe(
        self,
        transcript: str | None,
        detected: str | None = None,
        *,
        semantic: LanguageReading | None = None,
    ) -> str:
        """Fold one user turn in and return the language to answer it in.

        ``semantic`` is the local model's reading of this turn, when there is
        one. A reading of a *switch* becomes the session override, exactly as a
        matched instruction does; an ordinary reading decides this turn without
        overriding the account preference. The reading is session state and
        nothing else: it is never written to the stored preference, and one
        session's reading cannot reach another's.
        """
        if semantic is not None and semantic.switch_requested:
            self.override = semantic.language
            self.current = semantic.language
            return self.current
        instruction = explicit_request(transcript)
        if instruction:
            self.override = instruction
            self.current = instruction
            return self.current
        self.current = resolve(
            self.effective_preference,
            detected=detected,
            transcript=transcript,
            current=self.current,
            semantic=semantic,
        )
        return self.current

    def directive(self) -> str:
        return reply_directive(self.current)


def strip_accents(text: str) -> str:
    """Accent-insensitive form, for matching spoken words against fixed tokens."""
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))
