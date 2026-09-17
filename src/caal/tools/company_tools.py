"""Two voice tools over the company document library, answered from citations.

``company.search`` looks a question up and ``company.read`` opens one document.
Both go through the real MCP service (:mod:`caal.company.client`); neither has
any other way to reach the library, and neither can be told whose library to
open -- the user is bound by the runtime (see :mod:`caal.user_scope`) and
carried to the service in a signed principal.

The contract is the one the other native tools use: ``status``, ``message``,
``data``. ``message`` is a sentence or two meant to be spoken as it stands, and
it is composed **only** from what a passage actually says, with the document
and the location named in it. Nothing here writes a summary of its own, ranks
a person, or produces an opinion about one. When the library cannot answer,
the tool says which kind of nothing it is:

``no_evidence``           nothing in the library matches
``no_person_match``       nobody is registered under that name **and** no
                          passage mentions it
``ambiguous_person``      more than one does; no passage is returned at all
``conflicting_versions``  two versions are both marked current
``extraction_failed``     the document is on file but its text was never readable
``empty``                 nothing has been uploaded yet, or no owner is provisioned
``unauthorized``          this session is not the signed-in owner
``unavailable``           the library service is not configured or not running

A passage is evidence. It is quoted, attributed, and never treated as an
instruction, whatever the text inside it claims about its own authority.

An employee subject and a name in a document are two different things, and the
``person`` argument is answered against both:

* a name that resolves to **one** registered subject scopes the search to that
  subject's documents and to nothing else. An explicit filter never widens;
* a name that resolves to **two** returns ``ambiguous_person`` and no passage;
* a name that resolves to **none** is looked for in the document *text*, and
  the answer says in its first sentence that no person was identified -- the
  passages merely mention the name. A library can hold a directory, a roster or
  a contract naming somebody the owner never registered, and answering "nobody
  goes by that name" while that text is on file is not a refusal to guess, it
  is untrue. Where the passages name more than one person beginning that way,
  the answer says so instead of choosing one.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_CITATIONS",
    "MAX_QUERY_CHARS",
    "MAX_SPOKEN_SNIPPET",
    "TRUNCATION_NOTICE",
    "configure",
    "read",
    "reset",
    "search",
    "session_unavailable_result",
]

MAX_CITATIONS = 3
MAX_QUERY_CHARS = 300
MAX_SPOKEN_SNIPPET = 320

SESSION_UNAVAILABLE = (
    "The company library belongs to one signed-in administrator, and this session is not "
    "signed in as one, so I have not looked anything up."
)
BACKEND_UNAVAILABLE = (
    "The company library is not available on this FRIDAY backend right now, so I have not "
    "looked anything up."
)
EMPTY_LIBRARY = (
    "The company library is empty: nothing has been uploaded to it yet. You can add "
    "documents under Admin, Company Library."
)
#: Said whenever a passage was longer than what is read out, so a clause that
#: was cut is never mistaken for a passage that ended there.
TRUNCATION_NOTICE = (
    "That passage is longer than I read out; the rest of it is in the citation."
)
UNREGISTERED_PERSON = (
    "Nobody is registered in the company library under that name, so I have not "
    "identified a person. These passages are documents whose text mentions the name."
)
UNREGISTERED_AND_UNMENTIONED = (
    "Nobody is registered in the company library under that name, and no document text "
    "in it mentions the name either, so there is nothing for me to read."
)
#: How many distinct full names a passage set may name before the answer stops
#: treating the question as being about one person.
MAX_NAME_VARIANTS = 4
_EVIDENCE = "quoted_document_text"
_CLASSIFICATION_WORDS = dict(
    policy="policy", hr="HR document", contract="contract", general="document"
)


# --- binding to the runtime ---------------------------------------------------------------


_lock = threading.Lock()
_provider: Callable[[], Any] | None = None


def configure(provider: Callable[[], Any]) -> None:
    """Bind the tools to a company MCP client factory (tests; embedding runtimes)."""
    global _provider
    with _lock:
        _provider = provider


def reset() -> None:
    global _provider
    with _lock:
        _provider = None


def _default_client() -> Any | None:
    """The process-wide company client, when the library is configured."""
    from caal.company import runtime  # lazy: keep the registry import light

    return runtime.get_client()


def _client() -> Any | None:
    with _lock:
        provider = _provider
    try:
        return (provider or _default_client)()
    except Exception as exc:  # noqa: BLE001 - a tool answers, it never raises into the LLM
        logger.error("The company library client could not be built: %s", type(exc).__name__)
        return None


# --- the contract --------------------------------------------------------------------------


def session_unavailable_result() -> dict[str, Any]:
    """The refusal the runtime gives a session that is not the signed-in owner."""
    return _result("unauthorized", SESSION_UNAVAILABLE)


def _result(status: str, message: str, **data: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"citations": [], "treat_as": _EVIDENCE}
    payload.update(data)
    return {"status": status, "message": message, "data": payload}


def _version_phrase(citation: dict[str, Any]) -> str:
    """How a citation is said out loud, including when it is not in force.

    A draft and a superseded version are answers to an explicit request for
    one, so the status is spoken rather than left for the listener to assume
    they were told the current rule.
    """
    parts = [_CLASSIFICATION_WORDS.get(citation.get("classification") or "", "document")]
    status = citation.get("version_status")
    if isinstance(status, str) and status in ("draft", "superseded"):
        parts.append("draft, not in force" if status == "draft" else "superseded, not current")
    label = citation.get("version_label")
    if isinstance(label, str) and label.strip():
        parts.append(f"version {label.strip()}")
    effective = citation.get("effective_date")
    if isinstance(effective, str) and effective.strip():
        parts.append(f"effective {effective.strip()}")
    return ", ".join(parts)


#: Punctuation a title line may carry that the title field does not. A heading
#: typed as "Acme agreement." is the same heading as "Acme agreement".
_TITLE_TRIM = " \t.,:;!?-–—"


def _echoes_title(snippet: str, title: object) -> bool:
    """Whether a passage says nothing the citation does not already say.

    Short documents put their own title in the index, and BM25 favours short
    passages, so the title line can outrank the clause that answers the
    question. Quoting it back would be a citation with no evidence in it.
    """
    heading = " ".join(str(title or "").split()).casefold().strip(_TITLE_TRIM)
    body = snippet.casefold().strip(_TITLE_TRIM)
    return bool(heading) and bool(body) and body in heading


def _spoken(citations: list[dict[str, Any]]) -> str:
    """One sentence per passage: where it came from, then what it says. Nothing added.

    A passage longer than :data:`MAX_SPOKEN_SNIPPET` is cut at a word, and the
    cut is **said**. An address or a clause sitting past the limit would
    otherwise disappear out of a sentence that sounded complete; the full
    passage stays in the citation either way.
    """
    lines = []
    truncated = False
    informative = [
        citation
        for citation in citations
        if not _echoes_title(
            " ".join(str(citation.get("snippet") or "").split()), citation.get("title")
        )
    ]
    for citation in informative or citations:
        passage = " ".join(str(citation.get("snippet") or "").split())
        if not passage:
            continue
        snippet = passage[:MAX_SPOKEN_SNIPPET]
        if len(passage) > len(snippet):
            truncated = True
            spaced = snippet.rsplit(" ", 1)[0]
            snippet = (spaced or snippet) + " …"
        lines.append(
            f"{citation.get('title')} ({_version_phrase(citation)}), "
            f"{citation.get('location')}: “{snippet}”"
        )
    if truncated:
        lines.append(TRUNCATION_NOTICE)
    return " ".join(lines)


# A spoken name, and up to two capitalised words after it: enough to tell
# "FIXTURE Person A" from "FIXTURE Person B" in a passage, and deliberately not
# a name parser. It is only ever used to decide whether the passages are about
# more than one person, never to assert who anybody is.
def _name_pattern(name: str) -> re.Pattern[str] | None:
    words = re.findall(r"[^\W\d_]+", name, re.UNICODE)[:3]
    if not words:
        return None
    return re.compile(
        r"\b" + r"\s+".join(re.escape(word) for word in words) + r"(?:\s+[A-Z][^\W\d_]+){0,2}",
        re.IGNORECASE,
    )


def _mentions_name(name: str, snippet: object) -> bool:
    """Whether a passage really says the name, rather than merely ranking for it.

    The index matches *any* word of a query, so a search that carries a name
    into it comes back with passages that share only the common words. Saying
    "these mention the name" about those would be the same untruth in the other
    direction, so the last word of the name -- the surname, or the whole of a
    one-word name -- has to actually be in the passage.
    """
    words = re.findall(r"[^\W\d_]+", name, re.UNICODE)
    if not words:
        return False
    text = " ".join(str(snippet or "").split())
    return re.search(rf"\b{re.escape(words[-1])}\b", text, re.IGNORECASE) is not None


def _mentioning(name: str, results: list[Any]) -> list[Any]:
    """The ranked results that name the person, in the order the index gave them."""
    return [
        item
        for item in results
        if isinstance(item, dict) and _mentions_name(name, item.get("snippet"))
    ]


def _name_variants(name: str, citations: list[dict[str, Any]]) -> list[str]:
    """The distinct people the passages could be naming, longest form kept.

    "Quill" in one passage and "Quill Marlow" in another is one person as far
    as this can tell, so a bare mention that another mention extends is folded
    into it. Two different surnames are two candidates and stay two.
    """
    pattern = _name_pattern(name)
    if pattern is None:
        return []
    found: dict[str, str] = {}
    for citation in citations:
        passage = " ".join(str(citation.get("snippet") or "").split())
        for match in pattern.finditer(passage):
            cleaned = " ".join(match.group(0).split())
            found.setdefault(cleaned.casefold(), cleaned)
    variants = sorted(found.values(), key=len, reverse=True)
    kept: list[str] = []
    for variant in variants:
        if not any(longer.casefold().startswith(variant.casefold()) for longer in kept):
            kept.append(variant)
    return kept[:MAX_NAME_VARIANTS]


def _citations(results: list[Any]) -> list[dict[str, Any]]:
    kept = []
    for item in results[:MAX_CITATIONS]:
        if not isinstance(item, dict):
            continue
        kept.append(
            {
                "document_id": item.get("document_id"),
                "version_id": item.get("version_id"),
                "chunk_id": item.get("chunk_id"),
                "title": item.get("title"),
                "location": item.get("location"),
                "classification": item.get("classification"),
                "version_status": item.get("version_status"),
                "version_label": item.get("version_label"),
                "effective_date": item.get("effective_date"),
                "snippet": item.get("snippet"),
            }
        )
    return kept


def _person_query(text: str, unregistered: str | None) -> str:
    """The words sent to the index when the registry knew nothing about the name.

    The name is the only thing narrowing this search -- there is no subject to
    scope it with -- so it has to be in the query. It is added once, and only
    when the user's own words did not already carry it.
    """
    if not unregistered:
        return text
    words = [word for word in re.findall(r"[^\W\d_]+", unregistered, re.UNICODE) if word]
    lowered = text.casefold()
    missing = [word for word in words if word.casefold() not in lowered]
    if not missing:
        return text
    return " ".join([text, *missing]).strip()[:MAX_QUERY_CHARS]


def _person_identity(
    person: str | None, unregistered: str | None, citations: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """What this answer is entitled to claim about whose documents these are.

    ``None`` when no person was asked for at all. Otherwise a caveat to speak
    first -- empty for a registered subject, which needs none -- and a record
    of how the name was resolved, so a caller never has to infer it from the
    wording of a sentence.
    """
    if not person:
        return None
    if not unregistered:
        return {
            "caveat": "",
            "data": {
                "name_asked": str(person),
                "resolution": "registered_subject",
                "verified": True,
                "variants": [],
            },
        }
    variants = _name_variants(unregistered, citations)
    caveat = UNREGISTERED_PERSON
    if len(variants) > 1:
        caveat = (
            f"{UNREGISTERED_PERSON} They name more than one person whose name starts that "
            f"way ({len(variants)} of them), so which one is meant is not settled."
        )
    return {
        "caveat": caveat,
        "data": {
            "name_asked": str(person),
            "resolution": "text_mention",
            "verified": False,
            "variants": variants,
        },
    }


# --- the tools -----------------------------------------------------------------------------


async def _cross_language(
    client: Any,
    user_id: str,
    arguments: dict[str, Any],
    text: str,
    unregistered: str | None,
    results: list[Any],
    status: object,
) -> tuple[list[Any], object]:
    """Search the same question again in the other language, and merge the rankings.

    The index matches words, so a Spanish question cannot reach an English
    contract on its own. :mod:`caal.company.query_expansion` renders the
    question -- and only the question -- into ``en``/``es`` with the configured
    **local** model; each rendering is searched with **this same arguments
    dict**, every declared and undeclared filter copied verbatim and only
    ``query`` replaced, so an expansion can never widen a scope the caller
    narrowed. With no translator configured, or a failing one, the expansion is
    empty and this is the original search unchanged.

    A conflict is not something an extra search may hide: if any of them says
    two versions are both current, that is the status of the answer.
    """
    from caal.company import query_expansion
    from caal.company.client import CompanyClientError

    if not query_expansion.enabled():
        return results, status
    # The subject of the tool call is carried into the expansion, which checks
    # it against the session's authorization *before* the question is composed
    # or sent. A session that is not the library owner searches its own words
    # and nothing leaves for the translator.
    variants = (await query_expansion.expand(text, user_id=user_id))[1:]
    if not variants:
        return results, status

    def _kept(items: list[Any]) -> list[Any]:
        # An unregistered name decides identity before the merge as well as
        # after it, so an expansion cannot spend the three citations on
        # passages about somebody else who ranks for the same words.
        kept = _mentioning(unregistered, items) if unregistered else list(items)
        # A short document indexes its own title as a chunk, and BM25 favours
        # short chunks, so the title line can outrank the clause underneath it.
        # `_spoken` already declines to quote such a passage; with several
        # searches competing for three citations it must not also take the
        # place of the clause that answers the question. Demoted, not dropped:
        # a document whose only hit is its title still has a hit.
        echoes = [
            item
            for item in kept
            if _echoes_title(
                " ".join(str(item.get("snippet") or "").split()), item.get("title")
            )
        ]
        if not echoes:
            return kept
        return [item for item in kept if item not in echoes] + echoes

    lists = [_kept(results)]
    for variant in variants:
        attempt = dict(arguments)
        attempt["query"] = _person_query(variant, unregistered) or variant
        try:
            payload = await client.call("company_search", attempt, subject=user_id)
        except CompanyClientError:
            # The original search already succeeded. A failed extra one costs
            # the cross-language hit, never the answer.
            logger.warning("a cross-language company search failed; keeping the original hits")
            continue
        outcome = payload.get("status")
        if outcome == "conflicting_versions":
            status = outcome
        elif outcome != "ok":
            continue
        lists.append(_kept(payload.get("results") or []))
    return query_expansion.aggregate(lists, MAX_CITATIONS), status


async def search(
    *,
    user_id: str | None,
    query: str,
    classification: str | None = None,
    person: str | None = None,
) -> dict[str, Any]:
    """Look a question up in the company library and answer from cited passages."""
    if not user_id:
        return session_unavailable_result()
    client = _client()
    if client is None:
        return _result("unavailable", BACKEND_UNAVAILABLE)
    from caal.company.client import CompanyClientError

    text = str(query or "").strip()[:MAX_QUERY_CHARS]
    if not text and not person:
        return _result("invalid_request", "Tell me what to look up in the company library.")

    subject_id: str | None = None
    # A name the registry does not know. The lookup carries on into the
    # document text with it, and every answer below says it was never resolved
    # to a person.
    unregistered: str | None = None
    if person:
        try:
            resolved = await client.call("company_people", {"name": str(person)}, subject=user_id)
        except CompanyClientError:
            return _result("unavailable", BACKEND_UNAVAILABLE)
        outcome = resolved.get("status")
        if outcome == "ambiguous_subject":
            # Two people, one name. Naming either of them would be a guess
            # about a person, which this library never makes.
            candidates = resolved.get("candidates") or []
            return _result(
                "ambiguous_person",
                (
                    f"{len(candidates)} people in the company library go by that name, so I "
                    "have not opened either record. Tell me which one you mean."
                ),
                candidates=[
                    {"subject_id": c.get("subject_id"), "display_name": c.get("display_name")}
                    for c in candidates
                    if isinstance(c, dict)
                ],
            )
        if outcome != "ok":
            unregistered = str(person)
        else:
            subject_id = resolved.get("subject_id")

    arguments: dict[str, Any] = {"query": _person_query(text, unregistered) or str(person)}
    if classification:
        arguments["classification"] = str(classification)
    if subject_id:
        arguments["subject"] = subject_id
    try:
        payload = await client.call("company_search", arguments, subject=user_id)
    except CompanyClientError:
        return _result("unavailable", BACKEND_UNAVAILABLE)

    status = payload.get("status")
    if status == "forbidden":
        return session_unavailable_result()
    if status == "unconfigured":
        return _result("empty", EMPTY_LIBRARY)
    results = payload.get("results") or []
    if text:
        results, status = await _cross_language(
            client, user_id, arguments, text, unregistered, results, status
        )
    if unregistered:
        results = _mentioning(unregistered, results)
    citations = _citations(results)
    if not citations:
        try:
            report = await client.call("company_status", {}, subject=user_id)
        except CompanyClientError:
            report = {}
        if not report.get("document_count"):
            return _result("empty", EMPTY_LIBRARY)
        if unregistered:
            # Both halves of the truth: not a registered person, and not a name
            # the documents say either. Only the first half was said before.
            return _result("no_person_match", UNREGISTERED_AND_UNMENTIONED)
        return _result(
            "no_evidence",
            (
                "Nothing in the company library mentions that. I only answer from the "
                "documents uploaded to it, so I have not guessed."
            ),
        )

    identity = _person_identity(person, unregistered, citations)
    spoken = _spoken(citations)
    if identity is not None and identity["caveat"]:
        spoken = f"{identity['caveat']} {spoken}"
    extra: dict[str, Any] = {} if identity is None else {"person_identity": identity["data"]}
    if status == "conflicting_versions":
        return _result(
            "conflicting_versions",
            (
                "Two versions of that document are both marked current, so I cannot say "
                f"which one applies. Here is what each says. {spoken}"
            ),
            citations=citations,
            conflicts=payload.get("conflicts") or [],
            **extra,
        )
    return _result("ok", spoken, citations=citations, **extra)


async def read(
    *, user_id: str | None, document_id: str, version_id: str | None = None
) -> dict[str, Any]:
    """Open one company document and return a bounded excerpt with its metadata."""
    if not user_id:
        return session_unavailable_result()
    client = _client()
    if client is None:
        return _result("unavailable", BACKEND_UNAVAILABLE)
    from caal.company.client import CompanyClientError

    arguments: dict[str, Any] = {"document_id": str(document_id or "")}
    if version_id:
        arguments["version_id"] = str(version_id)
    try:
        payload = await client.call("company_fetch", arguments, subject=user_id)
    except CompanyClientError:
        return _result("unavailable", BACKEND_UNAVAILABLE)

    status = payload.get("status")
    if status == "forbidden":
        return session_unavailable_result()
    if status == "unconfigured":
        return _result("empty", EMPTY_LIBRARY)
    if status == "conflicting_versions":
        # The same refusal a search gives. A read must not be the quiet way
        # past a conflict the owner has not resolved.
        return _result(
            "conflicting_versions",
            (
                "Two versions of that document are both marked current, so I have not "
                "picked one. Say which version you mean, or set one of them superseded."
            ),
            conflicts=payload.get("conflicts") or [],
        )
    if status == "no_current_version":
        return _result(
            "no_evidence",
            (
                "That document has no current version, only drafts or superseded history. "
                "Name the version if you want me to read one of those."
            ),
        )
    if status == "extraction_failed":
        return _result(
            "extraction_failed",
            (
                "That document is on file, but its text could not be read when it was "
                "uploaded, so there is nothing for me to quote from it."
            ),
            reason=payload.get("reason"),
        )
    if status != "ok":
        return _result(
            "no_evidence", "No document in the company library has that identifier."
        )
    excerpt = " ".join(str(payload.get("excerpt") or "").split())
    detail = {
        "document_id": payload.get("document_id"),
        "version_id": payload.get("version_id"),
        "title": payload.get("title"),
        "classification": payload.get("classification"),
        "version_status": payload.get("version_status"),
        "effective_date": payload.get("effective_date"),
        "version_label": payload.get("version_label"),
        "locations": payload.get("locations") or [],
        "excerpt": payload.get("excerpt"),
        "truncated": bool(payload.get("truncated")),
    }
    spoken = (
        f"{payload.get('title')} ({_version_phrase(payload)}): "
        f"“{excerpt[:MAX_SPOKEN_SNIPPET]}”"
    )
    return _result("ok", spoken, citations=[detail], **detail)
