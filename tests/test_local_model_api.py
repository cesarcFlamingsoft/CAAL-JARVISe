"""HTTP contract for choosing the local model JARVIS runs on.

The endpoint and the model are deployment-wide settings, so the routes sit
behind the same internal trust boundary as the rest of the identity API and
writes are for administrators only: the BFF proves itself with a single-use
signed principal, the backend loads the user from its own database, and an
ordinary member may read the current choice but not change it.

Pinned properties: an unauthenticated caller gets nothing; a member cannot
write; discovery reads the endpoint given (or the saved one) and nothing
else; an endpoint outside the local network is refused before a socket is
opened; upstream trouble comes back as a short code rather than an upstream
string; a save writes the two settings keys and leaves routing alone; and no
response carries a secret from the settings file.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from caal import local_model_api, profile_crypto, user_api, webhooks
from caal.internal_auth import AUDIENCE_BACKEND, RateLimiter, mint_principal
from caal.local_model_api import LocalModelRuntime
from caal.profile_crypto import KeyRing
from caal.security_config import MultiUserConfig
from caal.user_api import IdentityRuntime
from caal.user_store import ADMIN, MEMBER, Actor, UserStore

SECRET = "s" * 48
BOOTSTRAP = "cesarc@mexcantech.com"
NOW = 1_700_000_000
HERMES_KEY = "sk-hermes-should-never-be-here"

TAGS = {"models": [{"name": "qwen3:8b"}, {"name": "llama3.2:3b"}]}


class Harness:
    def __init__(self, tmp_path) -> None:
        self.now = NOW
        self.keyring = KeyRing.from_env(profile_crypto.generate_key_material(version=1))
        self.store = UserStore(tmp_path / "assistant.sqlite3", keyring=self.keyring)
        self.config = MultiUserConfig(
            internal_auth_secret=SECRET,
            keyring=self.keyring,
            bootstrap_admin_email=BOOTSTRAP,
            store_path=tmp_path / "assistant.sqlite3",
        )
        self.identity = IdentityRuntime(
            self.config,
            store=self.store,
            mutation_limiter=RateLimiter(limit=100, window_seconds=60),
            clock=lambda: self.now,
        )
        self.settings = dict(
            llm_provider="routed",
            ollama_host="http://localhost:11434",
            ollama_model="qwen3:8b",
            hermes_api_key=HERMES_KEY,
        )
        self.saved: list[dict] = []
        self.requests: list[httpx.Request] = []
        self.answers: list[object] = []
        self.runtime = LocalModelRuntime(
            self.identity,
            load=lambda: dict(self.settings),
            save=self._save,
            transport=httpx.MockTransport(self._handle),
        )
        self.admin = self.user("ada@example.com", ADMIN)
        self.member = self.user("mo@example.com", MEMBER)

    def _save(self, values: dict) -> None:
        self.saved.append(dict(values))
        self.settings.update(values)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answer = self.answers.pop(0) if self.answers else httpx.Response(200, json=TAGS)
        if isinstance(answer, Exception):
            raise answer
        return answer

    def user(self, email: str, role: str) -> str:
        return self.store.create_user(
            email=email,
            display_name=email.split("@")[0],
            role=role,
            actor=Actor.system(),
            now=self.now,
        ).user_id

    def bearer(self, user_id: str) -> dict[str, str]:
        token = mint_principal(
            secret=SECRET, subject=user_id, audience=AUDIENCE_BACKEND, now=self.now
        )
        return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def harness(tmp_path):
    return Harness(tmp_path)


@pytest.fixture
def client(harness):
    overrides = webhooks.app.dependency_overrides
    overrides[user_api.get_runtime] = lambda: harness.identity
    overrides[local_model_api.get_local_model_runtime] = lambda: harness.runtime
    try:
        with TestClient(webhooks.app) as test_client:
            yield test_client
    finally:
        overrides.pop(user_api.get_runtime, None)
        overrides.pop(local_model_api.get_local_model_runtime, None)


# --- the identity boundary ----------------------------------------------------------------


def test_every_route_refuses_an_unauthenticated_caller(client):
    calls = [
        ("get", "/users/me/local-model", None),
        ("post", "/users/me/local-model/models", {"endpoint": "http://localhost:11434"}),
        (
            "put",
            "/users/me/local-model",
            {"endpoint": "http://localhost:11434", "model": "qwen3:8b"},
        ),
    ]
    for method, path, body in calls:
        response = getattr(client, method)(path, **(dict(json=body) if body else {}))
        assert response.status_code == 401, path


def test_a_member_may_read_the_choice_but_not_change_it(client, harness):
    read = client.get("/users/me/local-model", headers=harness.bearer(harness.member))
    assert read.status_code == 200

    write = client.put(
        "/users/me/local-model",
        headers=harness.bearer(harness.member),
        json={"endpoint": "http://10.0.0.12:11434", "model": "qwen3:8b"},
    )
    assert write.status_code == 403
    discover = client.post(
        "/users/me/local-model/models",
        headers=harness.bearer(harness.member),
        json={"endpoint": "http://10.0.0.12:11434"},
    )
    assert discover.status_code == 403
    assert harness.saved == []
    assert harness.requests == []


# --- reading the current choice -----------------------------------------------------------


def test_the_current_choice_describes_the_routed_behaviour(client, harness):
    body = client.get("/users/me/local-model", headers=harness.bearer(harness.admin)).json()
    assert body["endpoint"] == "http://localhost:11434"
    assert body["model"] == "qwen3:8b"
    assert body["local_only"] is True
    assert body["applies_to"] == "new_sessions"
    routing = body["routing"]
    assert routing["primary"] == "ollama"
    assert routing["escalation"] == "hermes"
    assert routing["coding"] == "hermes_delegation"
    assert HERMES_KEY not in repr(body)


def test_an_unusable_saved_endpoint_reads_back_as_the_default(client, harness, monkeypatch):
    # With no usable saved value and nothing in the environment either, the
    # answer is the default rather than a refusal.
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    harness.settings["ollama_host"] = "https://ollama.example.com"
    body = client.get("/users/me/local-model", headers=harness.bearer(harness.admin)).json()
    assert body["endpoint"] == "http://localhost:11434"


# --- discovery ----------------------------------------------------------------------------


def test_discovery_reads_the_endpoint_the_operator_typed(client, harness):
    response = client.post(
        "/users/me/local-model/models",
        headers=harness.bearer(harness.admin),
        json={"endpoint": "http://192.168.1.50:11434"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["endpoint"] == "http://192.168.1.50:11434"
    assert body["models"] == ["llama3.2:3b", "qwen3:8b"]
    assert str(harness.requests[0].url) == "http://192.168.1.50:11434/api/tags"


def test_discovery_falls_back_to_the_saved_endpoint(client, harness):
    harness.settings["ollama_host"] = "http://10.0.0.12:11434"
    response = client.post(
        "/users/me/local-model/models", headers=harness.bearer(harness.admin), json={}
    )
    assert response.status_code == 200
    assert response.json()["endpoint"] == "http://10.0.0.12:11434"
    assert str(harness.requests[0].url) == "http://10.0.0.12:11434/api/tags"


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://localhost:11434",
        "http://ollama.example.com:11434",
        "http://8.8.8.8:11434",
        "http://user:secret@localhost:11434",
        "http://localhost:11434/api/tags",
        "http://localhost",
        "http://169.254.169.254:80",
    ],
)
def test_discovery_refuses_an_endpoint_off_the_local_network(client, harness, endpoint):
    response = client.post(
        "/users/me/local-model/models",
        headers=harness.bearer(harness.admin),
        json={"endpoint": endpoint},
    )
    assert response.status_code == 422
    assert harness.requests == []
    detail = response.json()["detail"]
    assert isinstance(detail, str) and detail.replace("_", "").isalpha()
    assert "secret" not in response.text


def test_upstream_trouble_comes_back_as_a_short_code(client, harness):
    harness.answers.append(httpx.Response(500, text="ollama stack trace"))
    response = client.post(
        "/users/me/local-model/models",
        headers=harness.bearer(harness.admin),
        json={"endpoint": "http://localhost:11434"},
    )
    assert response.status_code == 502
    assert response.json()["detail"] == "upstream_error"
    assert "stack trace" not in response.text


def test_an_unreachable_ollama_comes_back_as_unreachable(client, harness):
    harness.answers.append(httpx.ConnectError("connection refused"))
    response = client.post(
        "/users/me/local-model/models",
        headers=harness.bearer(harness.admin),
        json={"endpoint": "http://localhost:11434"},
    )
    assert response.status_code == 502
    assert response.json()["detail"] == "unreachable"


# --- saving -------------------------------------------------------------------------------


def test_saving_updates_only_the_endpoint_and_model_while_preserving_other_settings(client, harness):
    response = client.put(
        "/users/me/local-model",
        headers=harness.bearer(harness.admin),
        json={"endpoint": "  http://192.168.1.50:11434/  ", "model": "llama3.2:3b"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["endpoint"] == "http://192.168.1.50:11434"
    assert body["model"] == "llama3.2:3b"
    assert body["applies_to"] == "new_sessions"
    assert len(harness.saved) == 1
    saved = harness.saved[0]
    assert saved["ollama_host"] == "http://192.168.1.50:11434"
    assert saved["ollama_model"] == "llama3.2:3b"
    assert saved["llm_provider"] == "routed"
    assert saved["hermes_api_key"] == HERMES_KEY


def test_saving_leaves_the_routed_provider_alone(client, harness):
    client.put(
        "/users/me/local-model",
        headers=harness.bearer(harness.admin),
        json={"endpoint": "http://192.168.1.50:11434", "model": "llama3.2:3b"},
    )
    assert harness.settings["llm_provider"] == "routed"
    assert harness.saved[0]["llm_provider"] == "routed"


def test_saving_does_not_ask_ollama_anything(client, harness):
    client.put(
        "/users/me/local-model",
        headers=harness.bearer(harness.admin),
        json={"endpoint": "http://192.168.1.50:11434", "model": "llama3.2:3b"},
    )
    assert harness.requests == []


@pytest.mark.parametrize(
    ("body", "detail"),
    [
        ({"endpoint": "https://192.168.1.50:11434", "model": "qwen3:8b"}, "scheme_not_http"),
        ({"endpoint": "http://ollama.example.com:11434", "model": "qwen3:8b"}, "host_not_local"),
        ({"endpoint": "http://localhost:11434", "model": "bad model name"}, "invalid_model"),
        ({"endpoint": "http://localhost:11434", "model": ""}, "invalid_model"),
        ({"endpoint": "http://localhost:11434", "model": "x" * 200}, "invalid_model"),
    ],
)
def test_a_refused_save_changes_nothing(client, harness, body, detail):
    response = client.put("/users/me/local-model", headers=harness.bearer(harness.admin), json=body)
    assert response.status_code == 422
    assert response.json()["detail"] == detail
    assert harness.saved == []


def test_a_save_with_unknown_fields_is_refused(client, harness):
    response = client.put(
        "/users/me/local-model",
        headers=harness.bearer(harness.admin),
        json={"endpoint": "http://localhost:11434", "model": "qwen3:8b", "llm_provider": "groq"},
    )
    assert response.status_code == 422
    assert harness.saved == []


def test_a_saved_choice_is_what_the_next_read_and_the_next_session_see(client, harness):
    client.put(
        "/users/me/local-model",
        headers=harness.bearer(harness.admin),
        json={"endpoint": "http://10.0.0.12:11434", "model": "llama3.2:3b"},
    )
    body = client.get("/users/me/local-model", headers=harness.bearer(harness.admin)).json()
    assert body["endpoint"] == "http://10.0.0.12:11434"
    assert body["model"] == "llama3.2:3b"
