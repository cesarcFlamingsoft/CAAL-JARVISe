"""The language of the replies CAAL says itself, as opposed to the model's.

A voice turn has two kinds of spoken text. Most of it is the model's answer, and
the reply-language directive (see :mod:`caal.language_policy`) already decides
that. The rest is **fixed outcome strings**: the constants handler code says when
it does something itself -- the phone-handoff confirmation question, the refusal
when there is no approved number, the acknowledgement that work was queued, the
end-of-call line. No prompt can reach those, so before this module a Spanish
session heard Spanish answers and English outcomes.

What this is, deliberately:

* **A table, not a translator.** Only an exact, whole, known English constant is
  localized, keyed by the English string itself. A tool result, a model answer,
  a composed sentence, or a near-miss passes through untouched. Nothing is sent
  anywhere to be translated, and no model is consulted.
* **Same risk, same policy.** Each pair was written to say the same thing: a
  question stays a question, a refusal stays a refusal, and the arguments and
  identifiers in an outcome are not the kind of thing the table contains.
  Localizing happens where a reply is *spoken*, downstream of every decision, so
  it cannot widen what authorizes an action: a Spanish session needs exactly the
  same pending confirmation an English one does.
* **English is untouched.** The English constants keep their bytes and are still
  what is spoken when the turn is English, which is the default for everything
  that does not say otherwise.

The turn's language is published in a context variable set at the top of the
local-command path, the same way :mod:`caal.company_privacy` re-decides session
privacy there: the handlers that speak these replies run in that task, so they
see it. It is a *read* of session state, never a value taken from a message.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar

from .language_policy import EN, ES, SPOKEN

logger = logging.getLogger(__name__)

__all__ = [
    "END_CALL_REPLY",
    "begin_turn",
    "current_language",
    "localize",
    "reply_language",
    "spanish_pairs",
]

#: The language the replies of the current turn are said in. English unless a
#: turn says otherwise, so anything that never calls :func:`begin_turn` -- every
#: existing caller -- behaves exactly as it did.
reply_language: ContextVar[str] = ContextVar("reply_language", default=EN)

#: Said by ``caal.call_termination`` when a call is ended. Defined here as the
#: constant it already was, so the table can be keyed on it.
END_CALL_REPLY = "Ending the call. Goodbye."


def begin_turn(agent: object, *, language: str | None = None) -> str:
    """Publish the language this turn's own replies are said in.

    Read from the session's :class:`caal.language_policy.LanguageSession` and
    nothing else. Never raises: a broken or absent session means English, which
    is what these replies were before.

    ``language`` is an explicit value for *this* turn, passed by the caller that
    just decided it -- the local model's reading of the turn now being answered.
    It is used because a reading made inside another task cannot be published
    through this context variable: the value has to travel. An unusable value is
    ignored and the session is read as before, so the explicit form can only be
    as current as the session, never staler.
    """
    if language in SPOKEN:
        reply_language.set(language)
        return language
    language = EN
    try:
        session = getattr(agent, "_language_session", None)
        current = getattr(session, "current", None)
        if current in SPOKEN:
            language = current
    except Exception:  # noqa: BLE001 - a reply language must never break a turn
        logger.debug("Could not read the session reply language", exc_info=True)
    reply_language.set(language)
    return language


def current_language() -> str:
    """The language this turn's fixed replies are said in."""
    language = reply_language.get()
    return language if language in SPOKEN else EN


def localize(text: str) -> str:
    """The fixed reply to say, in this turn's language.

    Returns ``text`` itself -- the same object -- for an English turn, for an
    unknown string, and for anything that is not a string. That is the whole
    safety argument: a reply this module does not recognise is spoken exactly as
    the handler wrote it.
    """
    if not isinstance(text, str) or current_language() != ES:
        return text
    return spanish_pairs().get(text, text)


def speech_reply(text: str) -> str:
    """Localize at the speech boundary and carry that turn's language with it."""
    from .speech_request import SpeechText

    return SpeechText(localize(text), current_language())


_PAIRS: dict[str, str] | None = None


def spanish_pairs() -> dict[str, str]:
    """The vetted English -> neutral-Spanish table, built once.

    Built lazily and keyed on the constants *imported from the modules that own
    them*, so a key cannot drift from the string actually spoken: change the
    English constant and the pair stops matching, which the coverage test in
    ``tests/test_bilingual_outcome_strings.py`` fails on.
    """
    global _PAIRS
    if _PAIRS is None:
        _PAIRS = _build()
    return _PAIRS


def _build() -> dict[str, str]:
    from . import background_task_session as work
    from . import end_call_intent as end
    from . import handoff_intent as handoff

    return {
        # --- phone handoff: a question, a refusal, an outcome ----------------
        handoff.ASK_CLARIFICATION_REPLY: (
            "Para confirmar: ¿quiere seguir hablando por teléfono en lugar de aquí?"
        ),
        handoff.ASK_CONFIRMATION_REPLY: (
            "Puedo continuar esta conversación llamando a su teléfono aprobado. "
            "¿Quiere que le llame ahora?"
        ),
        handoff.CANCELLED_REPLY: (
            "Entendido. No llamaré a su teléfono. Podemos seguir aquí."
        ),
        handoff.STARTING_REPLY: (
            "Llamando a su teléfono ahora. Retomo esta conversación allí."
        ),
        handoff.NO_DESTINATION_REPLY: (
            "No puedo hacer eso. No tengo exactamente un número de teléfono aprobado "
            "registrado, así que no haré la llamada. Podemos seguir aquí."
        ),
        handoff.NO_CALLBACK_NUMBER_REPLY: (
            "No puedo llamarle porque no hay un número de devolución de llamada aprobado "
            "en su perfil. Un administrador debe configurarlo primero en su perfil, "
            "así que seguiremos aquí."
        ),
        handoff.FAILED_REPLY: (
            "No pude iniciar esa llamada, así que seguiremos aquí."
        ),
        # --- end of call ----------------------------------------------------
        end.ASK_END_CALL_REPLY: (
            "Parece que está terminando. ¿Quiere que finalice la llamada?"
        ),
        end.ASK_CALLBACK_REPLY: (
            "Parece que quiere que cuelgue y le llame de vuelta cuando termine esa tarea. "
            "¿Lo hago?"
        ),
        end.NO_TASK_ASK_END_CALL_REPLY: (
            "No hay ninguna tarea en segundo plano en este momento, así que no puedo "
            "llamarle de vuelta. ¿Quiere que finalice la llamada de todos modos?"
        ),
        end.NO_TASK_REPLY: (
            "No hay ninguna tarea en segundo plano en este momento, así que no hay nada "
            "sobre lo cual llamarle de vuelta. Me quedo en la línea."
        ),
        end.STAY_ON_LINE_REPLY: "Entendido. Me quedo en la línea.",
        end.CALLBACK_OFFER_DECLINED_REPLY: (
            "De acuerdo. Seguiré trabajando en eso en segundo plano y le aviso cuando "
            "esté listo."
        ),
        END_CALL_REPLY: "Finalizando la llamada. Hasta luego.",
        # --- durable work: acknowledgements, status, refusals ----------------
        work.BACKGROUND_ACK_REPLY: (
            "Entendido. Trabajaré en eso en segundo plano y le aviso cuando esté listo."
        ),
        work.BACKGROUND_BUSY_REPLY: (
            "No puedo aceptar más trabajo en segundo plano ahora mismo. "
            "Terminemos primero lo que ya está en curso."
        ),
        work.BACKGROUND_STATUS_IDLE_REPLY: (
            "No estoy trabajando en nada en segundo plano en este momento."
        ),
        work.BACKGROUND_STATUS_WORKING_REPLY: (
            "Sigo trabajando en eso en segundo plano. Le aviso en cuanto esté listo."
        ),
        work.BACKGROUND_CANCELLED_REPLY: (
            "Entendido. He detenido esa tarea en segundo plano."
        ),
        work.BACKGROUND_NOTHING_TO_CANCEL_REPLY: (
            "No hay ninguna tarea en segundo plano en este momento."
        ),
        work.CALLBACK_ARMED_REPLY: (
            "Entendido. Cuelgo ahora y le llamo de vuelta en cuanto esté listo. Hasta luego."
        ),
        work.CALLBACK_NOTHING_RUNNING_REPLY: (
            "No hay ninguna tarea en segundo plano en este momento, así que me quedo "
            "en la línea."
        ),
        work.LONG_WORK_ACK_REPLY: (
            "Entendido. Empiezo con eso ahora y le aviso en cuanto esté listo."
        ),
        work.LONG_WORK_OFFER_CALLBACK_REPLY: (
            "Entendido. Empiezo con eso ahora. Esto va a tardar un rato. "
            "¿Quiere que le llame cuando esté listo?"
        ),
        work.CODING_ACK_REPLY: (
            "Entendido. Eso es un trabajo de programación, así que usaré la mejor ruta "
            "de programación disponible y le diré lo que encuentre."
        ),
        work.CODING_OFFER_CALLBACK_REPLY: (
            "Entendido. Eso es un trabajo de programación, así que usaré la mejor ruta "
            "de programación disponible. Esto va a tardar un rato. "
            "¿Quiere que le llame cuando esté listo?"
        ),
    }
