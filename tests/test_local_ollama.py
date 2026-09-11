"""What an operator is allowed to point JARVIS local model at, and what comes back.

The endpoint is typed by an authenticated operator, so it is a request the
backend makes on someone elses say-so: exactly the shape of a server-side
request forgery. The rule pinned here is narrow on purpose -- plain ``http``,
an explicit port, and a host that is either one of two named local aliases or
an IP literal in a loopback / RFC1918 / IPv6 ULA-link-local range. Everything
else is refused offline, before a socket is opened.

Discovery is bounded and quiet: short timeouts, a capped list of names that
look like model names, and failures reduced to a short code. No test here
allows the endpoint, the payload, or an upstream error string into a log.
"""

from __future__ import annotations

import logging

import httpx
import pytest

from caal.local_ollama import (
    DEFAULT_ENDPOINT,
    MAX_MODELS,
    DiscoveryError,
    EndpointError,
    configured_endpoint,
    discover_models,
    normalize_endpoint,
    resolve_local_alias,
)

# --- what is accepted ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("http://localhost:11434", "http://localhost:11434"),
        ("http://host.docker.internal:11434", "http://host.docker.internal:11434"),
        ("http://127.0.0.1:11434", "http://127.0.0.1:11434"),
        ("http://192.168.1.50:11434", "http://192.168.1.50:11434"),
        ("http://10.0.0.12:11434", "http://10.0.0.12:11434"),
        ("http://172.16.5.4:8080", "http://172.16.5.4:8080"),
        ("http://172.31.255.254:1", "http://172.31.255.254:1"),
        ("http://[::1]:11434", "http://[::1]:11434"),
        ("http://[fd00::1]:11434", "http://[fd00::1]:11434"),
        ("http://[fe80::1]:11434", "http://[fe80::1]:11434"),
        # Cosmetic differences an operator will type; the stored form is one form.
        ("  http://localhost:11434/  ", "http://localhost:11434"),
        ("HTTP://LocalHost:11434", "http://localhost:11434"),
        ("http://[FD00::0:1]:11434", "http://[fd00::1]:11434"),
    ],
)
def test_a_local_endpoint_is_accepted_and_normalized(raw, expected):
    assert normalize_endpoint(raw) == expected


# --- what is refused ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        ("https://localhost:11434", "scheme_not_http"),
        ("HTTPS://192.168.1.50:11434", "scheme_not_http"),
        ("file:///etc/passwd", "scheme_not_http"),
        ("ftp://localhost:11434", "scheme_not_http"),
        ("localhost:11434", "scheme_not_http"),
        ("http://user:secret@localhost:11434", "credentials_not_allowed"),
        ("http://token@192.168.1.50:11434", "credentials_not_allowed"),
        ("http://localhost:11434/api/tags", "path_not_allowed"),
        ("http://localhost:11434/?x=1", "path_not_allowed"),
        ("http://localhost:11434#frag", "path_not_allowed"),
        ("http://localhost", "port_required"),
        ("http://192.168.1.50", "port_required"),
        ("http://localhost:0", "invalid_port"),
        ("http://localhost:65536", "invalid_port"),
        ("http://localhost:abc", "invalid_port"),
        ("http://localhost:", "port_required"),
        # Public names and addresses, however they are dressed up.
        ("http://ollama.example.com:11434", "host_not_local"),
        ("http://8.8.8.8:11434", "host_not_local"),
        ("http://169.254.169.254:80", "host_not_local"),
        ("http://172.32.0.1:11434", "host_not_local"),
        ("http://100.64.0.1:11434", "host_not_local"),
        ("http://[2606:4700::1111]:11434", "host_not_local"),
        ("http://[::ffff:8.8.8.8]:11434", "host_not_local"),
        ("http://[::ffff:192.168.1.50]:11434", "host_not_local"),
        ("http://localhost.attacker.example:11434", "host_not_local"),
        ("http://127.0.0.1.attacker.example:11434", "host_not_local"),
        # Nothing usable at all.
        ("", "invalid_endpoint"),
        ("   ", "invalid_endpoint"),
        ("http://", "invalid_endpoint"),
        ("http:// localhost:11434", "invalid_endpoint"),
        ("http://localhost:11434\nX: 1", "invalid_endpoint"),
        ("http://l\u03bfcalhost:11434", "invalid_endpoint"),
        ("http://" + "a" * 400 + ":11434", "invalid_endpoint"),
        (None, "invalid_endpoint"),
        (11434, "invalid_endpoint"),
        (True, "invalid_endpoint"),
    ],
)
def test_anything_that_is_not_a_local_http_endpoint_is_refused(raw, code):
    with pytest.raises(EndpointError) as caught:
        normalize_endpoint(raw)
    assert caught.value.code == code


def test_a_refusal_carries_a_short_safe_code_and_a_sentence():
    with pytest.raises(EndpointError) as caught:
        normalize_endpoint("http://user:hunter2@evil.example.com:80")
    error = caught.value
    assert error.code.replace("_", "").isalpha()
    assert "hunter2" not in str(error)
    assert "evil.example.com" not in str(error)
    assert len(error.message) < 200


# --- named aliases ------------------------------------------------------------------------


def test_localhost_resolves_without_asking_a_resolver():
    def explode(*_args, **_kwargs):  # pragma: no cover - must never run
        raise AssertionError("localhost must not be resolved through DNS")

    assert resolve_local_alias("localhost", getaddrinfo=explode) == ["127.0.0.1"]


def test_an_alias_that_resolves_to_a_local_address_is_allowed():
    def resolver(*_args, **_kwargs):
        return [(2, 1, 6, "", ("192.168.65.2", 11434))]

    assert resolve_local_alias("host.docker.internal", getaddrinfo=resolver) == ["192.168.65.2"]


def test_an_alias_that_resolves_off_the_local_network_is_refused():
    def resolver(*_args, **_kwargs):
        return [(2, 1, 6, "", ("192.168.65.2", 11434)), (2, 1, 6, "", ("8.8.8.8", 11434))]

    with pytest.raises(EndpointError) as caught:
        resolve_local_alias("host.docker.internal", getaddrinfo=resolver)
    assert caught.value.code == "host_not_local"


def test_an_alias_that_does_not_resolve_is_refused_as_unresolvable():
    def resolver(*_args, **_kwargs):
        raise OSError("no such host")

    with pytest.raises(EndpointError) as caught:
        resolve_local_alias("host.docker.internal", getaddrinfo=resolver)
    assert caught.value.code == "unresolvable"


def test_an_ip_literal_needs_no_resolution():
    def explode(*_args, **_kwargs):  # pragma: no cover - must never run
        raise AssertionError("an IP literal must not be resolved")

    assert resolve_local_alias("10.0.0.12", getaddrinfo=explode) == ["10.0.0.12"]


# --- discovery ----------------------------------------------------------------------------


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


TAGS = {
    "models": [
        {"name": "qwen3:8b", "digest": "sha256:abc"},
        {"name": "llama3.2:3b"},
        {"name": "mistral-small3.2:latest"},
    ]
}


@pytest.mark.asyncio
async def test_discovery_reads_api_tags_from_the_given_endpoint():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=TAGS)

    async with _client(handler) as client:
        models = await discover_models("http://192.168.1.50:11434", client=client)

    assert models == ["llama3.2:3b", "mistral-small3.2:latest", "qwen3:8b"]
    assert str(seen[0].url) == "http://192.168.1.50:11434/api/tags"
    assert seen[0].method == "GET"


@pytest.mark.asyncio
async def test_discovery_refuses_an_endpoint_that_is_not_local_before_connecting():
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never runs
        raise AssertionError("no request may be made to a non-local endpoint")

    async with _client(handler) as client:
        with pytest.raises(EndpointError) as caught:
            await discover_models("http://169.254.169.254:80", client=client)
    assert caught.value.code == "host_not_local"


@pytest.mark.asyncio
async def test_discovery_keeps_only_plausible_model_names_and_caps_the_list():
    payload = {
        "models": [
            {"name": "qwen3:8b"},
            {"name": "qwen3:8b"},
            {"name": ""},
            {"name": None},
            {"name": "bad name with spaces"},
            {"name": "x" * 500},
            {"nope": "no name at all"},
            "not-an-object",
            *[{"name": f"model{index:04d}:v1"} for index in range(MAX_MODELS + 50)],
        ]
    }

    async with _client(lambda request: httpx.Response(200, json=payload)) as client:
        models = await discover_models("http://localhost:11434", client=client)

    assert len(models) == MAX_MODELS
    assert models == sorted(models)
    assert all(" " not in name and len(name) <= 120 for name in models)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("answer", "code"),
    [
        (httpx.Response(500, text="boom"), "upstream_error"),
        (httpx.Response(404, text="nope"), "upstream_error"),
        (httpx.Response(200, text="<html>not json</html>"), "unexpected_response"),
        (httpx.Response(200, json={"models": "nope"}), "unexpected_response"),
        (httpx.Response(200, json=["a"]), "unexpected_response"),
        (httpx.ConnectError("refused"), "unreachable"),
        (httpx.ConnectTimeout("slow"), "timeout"),
        (httpx.ReadTimeout("slow"), "timeout"),
    ],
)
async def test_a_failed_discovery_is_reduced_to_a_short_code(answer, code):
    def handler(request: httpx.Request) -> httpx.Response:
        if isinstance(answer, Exception):
            raise answer
        return answer

    async with _client(handler) as client:
        with pytest.raises(DiscoveryError) as caught:
            await discover_models("http://localhost:11434", client=client)
    assert caught.value.code == code
    assert caught.value.message


@pytest.mark.asyncio
async def test_discovery_answers_an_empty_ollama_with_an_empty_list():
    async with _client(lambda request: httpx.Response(200, json={"models": []})) as client:
        assert await discover_models("http://localhost:11434", client=client) == []


class _Recorder(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.mark.asyncio
async def test_discovery_never_logs_the_endpoint_or_the_payload():
    """Whatever this module says about a discovery, it says without the details.

    The handler is attached to this module logger directly rather than through
    caplog: the address and the payload must be absent from what *this* module
    writes, whatever the rest of the test run does to the root logger. httpx
    logs its own request line, which is a library-wide behaviour.
    """
    secret_model = "internal-project-codename:latest"
    recorder = _Recorder()
    module_logger = logging.getLogger("caal.local_ollama")
    module_logger.addHandler(recorder)
    previous = module_logger.level
    module_logger.setLevel(logging.DEBUG)
    try:

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"models": [{"name": secret_model}]})

        async with _client(handler) as client:
            await discover_models("http://192.168.1.50:11434", client=client)
        async with _client(lambda r: httpx.Response(500, text="stack trace here")) as client:
            with pytest.raises(DiscoveryError):
                await discover_models("http://192.168.1.50:11434", client=client)
    finally:
        module_logger.removeHandler(recorder)
        module_logger.setLevel(previous)

    assert recorder.messages
    written = " ".join(recorder.messages)
    assert secret_model not in written
    assert "stack trace here" not in written
    assert "192.168.1.50" not in written


@pytest.mark.asyncio
async def test_discovery_does_not_follow_a_redirect_off_the_local_network():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host != "localhost":  # pragma: no cover - must never run
            raise AssertionError("a redirect must not be followed")
        return httpx.Response(302, headers={"Location": "http://169.254.169.254/latest/meta"})

    async with _client(handler) as client:
        with pytest.raises(DiscoveryError) as caught:
            await discover_models("http://localhost:11434", client=client)
    assert caught.value.code == "upstream_error"


# --- what the runtime uses ----------------------------------------------------------------


def test_the_saved_host_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://host.docker.internal:11434")
    saved = {"ollama_host": "http://10.0.0.12:11434"}
    assert configured_endpoint(saved) == "http://10.0.0.12:11434"


def test_the_environment_is_used_when_nothing_is_saved(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://host.docker.internal:11434")
    assert configured_endpoint({}) == "http://host.docker.internal:11434"
    assert configured_endpoint({"ollama_host": ""}) == "http://host.docker.internal:11434"


def test_an_unusable_saved_host_falls_back_rather_than_failing(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert configured_endpoint({"ollama_host": "https://ollama.example.com"}) == DEFAULT_ENDPOINT
    assert configured_endpoint({"ollama_host": None}) == DEFAULT_ENDPOINT


def test_an_unusable_environment_host_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://8.8.8.8:11434")
    assert configured_endpoint({}) == DEFAULT_ENDPOINT
