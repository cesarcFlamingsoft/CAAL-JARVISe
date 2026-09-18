from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from caal import profile_crypto, user_api, visual_api, webhooks
from caal.internal_auth import AUDIENCE_BACKEND, RateLimiter, mint_principal
from caal.profile_crypto import KeyRing
from caal.security_config import MultiUserConfig
from caal.user_api import IdentityRuntime
from caal.user_store import MEMBER, Actor, UserStore
from caal.visual_api import MAX_IMAGE_BYTES, VisualRuntime

SECRET = "v" * 48
NOW = 1_700_000_000
PROMPT = "Briefly describe what is visible in this camera view."


def jpeg(width: int = 32, height: int = 24) -> str:
    output = io.BytesIO()
    Image.new("RGB", (width, height), "navy").save(output, "JPEG")
    return base64.b64encode(output.getvalue()).decode("ascii")


class Harness:
    def __init__(self, tmp_path) -> None:
        keyring = KeyRing.from_env(profile_crypto.generate_key_material(version=1))
        self.store = UserStore(tmp_path / "users.sqlite3", keyring=keyring)
        config = MultiUserConfig(
            internal_auth_secret=SECRET,
            keyring=keyring,
            bootstrap_admin_email="owner@example.com",
            store_path=tmp_path / "users.sqlite3",
        )
        self.identity = IdentityRuntime(
            config,
            store=self.store,
            mutation_limiter=RateLimiter(limit=100, window_seconds=60),
            clock=lambda: NOW,
        )
        self.user = self.store.create_user(
            email="member@example.com",
            display_name="Member",
            role=MEMBER,
            actor=Actor.system(),
            now=NOW,
        ).user_id
        self.requests: list[httpx.Request] = []
        self.answers: list[httpx.Response | Exception] = []
        self.runtime = VisualRuntime(
            self.identity,
            load=lambda: {
                "ollama_host": "http://localhost:11434",
                "ollama_model": "gemma4:e4b",
                "hermes_api_key": "must-never-be-used",
            },
            transport=httpx.MockTransport(self.handle),
            limiter=RateLimiter(limit=2, window_seconds=60),
        )

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    def bearer(self, *, binding: bool = True) -> dict[str, str]:
        claims = {"session_binding": "a" * 64} if binding else None
        token = mint_principal(
            secret=SECRET,
            subject=self.user,
            audience=AUDIENCE_BACKEND,
            claims=claims,
            now=NOW,
        )
        return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def harness(tmp_path):
    return Harness(tmp_path)


@pytest.fixture
def client(harness):
    overrides = webhooks.app.dependency_overrides
    overrides[user_api.get_runtime] = lambda: harness.identity
    overrides[visual_api.get_visual_runtime] = lambda: harness.runtime
    try:
        with TestClient(webhooks.app) as value:
            yield value
    finally:
        overrides.pop(user_api.get_runtime, None)
        overrides.pop(visual_api.get_visual_runtime, None)


def body(**changes):
    value = {
        "image": jpeg(),
        "prompt": PROMPT,
        "company_private": False,
    }
    value.update(changes)
    return value


def test_requires_user_and_bff_session_owner_binding(client, harness):
    assert client.post("/users/me/visual/analyze", json=body()).status_code == 401
    response = client.post(
        "/users/me/visual/analyze", headers=harness.bearer(binding=False), json=body()
    )
    assert response.status_code == 403
    assert response.json() == {"detail": "owner_binding_required"}
    assert harness.requests == []


@pytest.mark.parametrize(
    "changes",
    [
        {"image": "not base64!!"},
        {"image": base64.b64encode(b"not a jpeg").decode()},
        {"image": base64.b64encode(b"x" * (MAX_IMAGE_BYTES + 1)).decode()},
        {"image": jpeg(641, 1)},
        {"image": jpeg(640, 640), "extra": True},
        {"prompt": "x" * 241},
        {"prompt": "hidden\ntext"},
        {"prompt": "What does our private employment contract say?"},
    ],
)
def test_rejects_malformed_or_out_of_bounds_input_before_ollama(client, harness, changes):
    response = client.post(
        "/users/me/visual/analyze", headers=harness.bearer(), json=body(**changes)
    )
    assert response.status_code == 422
    assert harness.requests == []


def test_company_mode_is_blocked_before_decode_or_model(client, harness, monkeypatch):
    decoded = False

    def should_not_decode(_value):
        nonlocal decoded
        decoded = True
        raise AssertionError

    monkeypatch.setattr(visual_api, "decode_jpeg", should_not_decode)
    response = client.post(
        "/users/me/visual/analyze",
        headers=harness.bearer(),
        json=body(image="not-even-base64", company_private=True),
    )
    assert response.status_code == 403
    assert response.json() == {"detail": "company_mode_blocked"}
    assert decoded is False
    assert harness.requests == []


def test_active_runtime_model_overrides_stale_saved_model(client, harness, monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "gemma4:e4b")
    harness.runtime._load = lambda: {
        "ollama_host": "http://localhost:11434",
        "ollama_model": "qwen3:8b",
    }
    harness.answers = [
        httpx.Response(200, json={"capabilities": ["vision"]}),
        httpx.Response(200, json={"message": {"content": "A local view."}}),
    ]
    response = client.post("/users/me/visual/analyze", headers=harness.bearer(), json=body())
    assert response.status_code == 200
    assert json.loads(harness.requests[0].content) == {"model": "gemma4:e4b"}
    assert json.loads(harness.requests[1].content)["model"] == "gemma4:e4b"


def test_checks_vision_then_sends_a_tool_free_local_chat_payload(client, harness):
    harness.answers = [
        httpx.Response(200, json={"capabilities": ["completion", "vision"]}),
        httpx.Response(200, json={"message": {"role": "assistant", "content": "  A blue view.  "}}),
    ]
    response = client.post("/users/me/visual/analyze", headers=harness.bearer(), json=body())
    assert response.status_code == 200
    assert response.json() == {"description": "A blue view."}
    assert [request.url.path for request in harness.requests] == ["/api/show", "/api/chat"]
    assert json.loads(harness.requests[0].content) == {"model": "gemma4:e4b"}
    payload = json.loads(harness.requests[1].content)
    assert payload["model"] == "gemma4:e4b"
    assert payload["stream"] is False
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][0]["content"] == PROMPT
    assert payload["messages"][0]["images"] == [body()["image"]]
    assert "tools" not in payload
    assert payload["options"]["num_predict"] <= 300
    assert all(request.url.host == "localhost" for request in harness.requests)
    assert "hermes" not in repr(payload).lower()


def test_retries_one_empty_local_vision_response(client, harness):
    harness.answers = [
        httpx.Response(200, json={"capabilities": ["vision"]}),
        httpx.Response(200, json={"message": {"content": ""}}),
        httpx.Response(200, json={"message": {"content": "A blue view."}}),
    ]
    response = client.post("/users/me/visual/analyze", headers=harness.bearer(), json=body())
    assert response.status_code == 200
    assert response.json() == {"description": "A blue view."}
    assert [request.url.path for request in harness.requests] == [
        "/api/show",
        "/api/chat",
        "/api/chat",
    ]


def test_retries_one_transient_local_vision_failure(client, harness):
    harness.answers = [
        httpx.Response(200, json={"capabilities": ["vision"]}),
        httpx.Response(503, text="temporary"),
        httpx.Response(200, json={"message": {"content": "A blue view."}}),
    ]
    response = client.post("/users/me/visual/analyze", headers=harness.bearer(), json=body())
    assert response.status_code == 200
    assert response.json() == {"description": "A blue view."}


def test_accepts_a_bounded_large_show_response_but_keeps_chat_response_strict(client, harness):
    harness.answers = [
        httpx.Response(200, json={"capabilities": ["vision"], "model_info": {"x": "z" * 160_000}}),
        httpx.Response(200, json={"message": {"content": "A blue view."}}),
    ]
    response = client.post("/users/me/visual/analyze", headers=harness.bearer(), json=body())
    assert response.status_code == 200
    assert response.json() == {"description": "A blue view."}


@pytest.mark.parametrize(
    "answer",
    [
        httpx.Response(200, json={"capabilities": ["completion"]}),
        httpx.Response(503, text="private upstream detail"),
        httpx.ConnectError("do not expose", request=httpx.Request("POST", "http://local")),
    ],
)
def test_missing_or_unavailable_vision_has_one_bounded_failure(client, harness, answer):
    harness.answers = [answer]
    response = client.post("/users/me/visual/analyze", headers=harness.bearer(), json=body())
    assert response.status_code == 503
    assert response.json() == {"detail": "vision_unavailable"}
    assert len(harness.requests) == 1


def test_rate_limit_is_per_user_and_prevents_model_access(client, harness):
    harness.answers = [
        httpx.Response(200, json={"capabilities": ["vision"]}),
        httpx.Response(200, json={"message": {"content": "one"}}),
        httpx.Response(200, json={"capabilities": ["vision"]}),
        httpx.Response(200, json={"message": {"content": "two"}}),
    ]
    assert (
        client.post("/users/me/visual/analyze", headers=harness.bearer(), json=body()).status_code
        == 200
    )
    assert (
        client.post("/users/me/visual/analyze", headers=harness.bearer(), json=body()).status_code
        == 200
    )
    limited = client.post("/users/me/visual/analyze", headers=harness.bearer(), json=body())
    assert limited.status_code == 429
    assert len(harness.requests) == 4


def test_response_is_bounded_and_no_image_or_prompt_is_logged(client, harness, caplog):
    harness.answers = [
        httpx.Response(200, json={"capabilities": ["vision"]}),
        httpx.Response(200, json={"message": {"content": "z" * 5000}, "secret": "raw"}),
    ]
    request_body = body()
    response = client.post("/users/me/visual/analyze", headers=harness.bearer(), json=request_body)
    assert len(response.json()["description"]) <= 1200
    logs = caplog.text
    assert PROMPT not in logs
    assert request_body["image"] not in logs
    assert "raw" not in logs


def test_module_has_no_conversation_persistence_or_external_routing_surface():
    source = Path(visual_api.__file__).read_text(encoding="utf-8") if visual_api.__file__ else ""
    for forbidden in (
        "conversation_ledger",
        "memory_tools",
        "model_routing",
        "hermes",
        "groq",
        "cloudflare",
    ):
        assert forbidden not in source.lower()


def test_live_endpoint_overrides_stale_saved_endpoint(client, harness, monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://192.168.1.20:11434")
    harness.answers = [
        httpx.Response(200, json={"capabilities": ["vision"]}),
        httpx.Response(200, json={"message": {"content": "A mug."}}),
    ]
    response = client.post("/users/me/visual/analyze", headers=harness.bearer(), json=body())
    assert response.status_code == 200
    assert {r.url.host for r in harness.requests} == {"192.168.1.20"}


@pytest.mark.parametrize(
    "endpoint", ["https://example.com:443", "http://169.254.169.254:80", "bad"]
)
def test_invalid_live_endpoint_fails_closed_without_fallback_or_logs(
    client, harness, monkeypatch, caplog, endpoint
):
    monkeypatch.setenv("OLLAMA_HOST", endpoint)
    response = client.post("/users/me/visual/analyze", headers=harness.bearer(), json=body())
    assert response.status_code == 503
    assert harness.requests == []
    assert endpoint not in caplog.text
