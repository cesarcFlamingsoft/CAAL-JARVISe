"""HTTP contract for the signed-in user own local reminders on the dashboard.

``/users/me/dashboard/reminders`` sits behind the same internal trust boundary
as the rest of the identity API: the BFF proves itself with a single-use signed
principal, the backend loads the user from its own database, and only that
user own reminders are read. Reminders are private, so the boundary is the
whole feature: there is no unauthenticated path to one, no way to name another
owner, and nothing in a response or a log line that is not this caller own.

``/users/me/dashboard/reminders/delivery`` is the minimal edit surface: the
owner reads and sets the channels *future* reminders of theirs will use. It
never touches a reminder that already exists and it can never name a phone
number or a chat.
"""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from caal import profile_crypto, user_api, webhooks
from caal.internal_auth import AUDIENCE_BACKEND, RateLimiter, mint_principal
from caal.profile_crypto import KeyRing
from caal.security_config import MultiUserConfig
from caal.tools import alarms_tools, reminder_delivery, reminders_tools
from caal.user_api import IdentityRuntime
from caal.user_store import MEMBER, Actor, UserStore

SECRET = "s" * 48
BOOTSTRAP = "cesarc@mexcantech.com"
NOW = 1_700_000_000  # 2023-11-14T22:13:20Z


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
        self.runtime = IdentityRuntime(
            self.config,
            store=self.store,
            mutation_limiter=RateLimiter(limit=100, window_seconds=60),
            clock=lambda: self.now,
        )
        self.ana = self.user("ana@example.com")
        self.bo = self.user("bo@example.com")

    def user(self, email: str) -> str:
        return self.store.create_user(
            email=email,
            display_name=email.split("@")[0],
            role=MEMBER,
            actor=Actor.system(),
            now=self.now,
        ).user_id

    def bearer(self, user_id: str) -> dict:
        token = mint_principal(
            secret=SECRET, subject=user_id, audience=AUDIENCE_BACKEND, now=self.now
        )
        return dict(Authorization=f"Bearer {token}")


@pytest.fixture
def harness(tmp_path, monkeypatch):
    path = tmp_path / "assistant.sqlite3"
    monkeypatch.setattr(alarms_tools, "STORE_PATH", path)
    monkeypatch.setattr(reminders_tools, "STORE_PATH", path)
    monkeypatch.setattr(reminder_delivery, "STORE_PATH", path)
    built = Harness(tmp_path)
    monkeypatch.setattr(reminder_delivery, "telegram_owner", lambda: built.ana)
    monkeypatch.setattr(reminder_delivery, "telegram_configured", lambda: True)
    monkeypatch.setattr(reminder_delivery, "resolve_callback_number", lambda user_id: None)
    return built


@pytest.fixture
def client(harness):
    webhooks.app.dependency_overrides[user_api.get_runtime] = lambda: harness.runtime
    try:
        with TestClient(webhooks.app) as test_client:
            yield test_client
    finally:
        webhooks.app.dependency_overrides.pop(user_api.get_runtime, None)


def _create(user_id: str, title: str, due: str | None = None, delivery=None) -> None:
    reminders_tools.create_reminder(
        title=title, due=due, delivery=delivery, user_id=user_id, now=NOW
    )


# --- the feed ---------------------------------------------------------------------------


def test_the_feed_shows_only_the_callers_own_reminders(client, harness):
    _create(harness.ana, "Call the clinic", "PT30M", ["speak", "telegram"])
    _create(harness.bo, "Bo private thing", "PT30M")

    body = client.get("/users/me/dashboard/reminders", headers=harness.bearer(harness.ana)).json()

    assert [item["title"] for item in body["reminders"]] == ["Call the clinic"]
    assert [channel["channel"] for channel in body["reminders"][0]["delivery"]] == [
        "speak",
        "telegram",
    ]
    assert all(channel["state"] == "pending" for channel in body["reminders"][0]["delivery"])
    assert body["reminders"][0]["timed"] is True
    assert body["reminders"][0]["due"].endswith("Z")


def test_an_undated_reminder_is_listed_and_offers_no_delivery(client, harness):
    _create(harness.ana, "Buy stamps")

    body = client.get("/users/me/dashboard/reminders", headers=harness.bearer(harness.ana)).json()

    assert body["reminders"][0]["timed"] is False
    assert body["reminders"][0]["due"] is None
    assert body["reminders"][0]["delivery"] == []


def test_an_empty_list_is_an_answer_not_an_error(client, harness):
    response = client.get("/users/me/dashboard/reminders", headers=harness.bearer(harness.bo))

    assert response.status_code == 200
    assert response.json()["reminders"] == []


def test_an_unknown_caller_gets_nothing_at_all(client, harness):
    _create(harness.ana, "Call the clinic", "PT30M")

    assert client.get("/users/me/dashboard/reminders").status_code == 401
    assert (
        client.get(
            "/users/me/dashboard/reminders", headers=dict(Authorization="Bearer nonsense")
        ).status_code
        == 401
    )


def test_the_feed_reports_which_channels_this_owner_may_use(client, harness):
    body = client.get("/users/me/dashboard/reminders", headers=harness.bearer(harness.ana)).json()
    other = client.get("/users/me/dashboard/reminders", headers=harness.bearer(harness.bo)).json()

    assert body["available"] == ["speak", "telegram"]
    assert other["available"] == ["speak"]


def test_the_feed_is_read_only(client, harness):
    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)(
            "/users/me/dashboard/reminders", headers=harness.bearer(harness.ana)
        )
        assert response.status_code == 405


# --- the owner default for future reminders ---------------------------------------------


def test_the_owner_reads_and_sets_the_default_for_future_reminders(client, harness):
    read = client.get(
        "/users/me/dashboard/reminders/delivery", headers=harness.bearer(harness.ana)
    )
    assert read.status_code == 200
    assert read.json() == dict(delivery=["speak"], available=["speak", "telegram"], saved=False)

    saved = client.put(
        "/users/me/dashboard/reminders/delivery",
        headers=harness.bearer(harness.ana),
        json=dict(delivery=["telegram", "speak"]),
    )

    assert saved.status_code == 200
    assert saved.json()["delivery"] == ["speak", "telegram"]
    assert saved.json()["saved"] is True
    assert reminder_delivery.default_channels(harness.ana) == ("speak", "telegram")


def test_a_saved_default_is_what_a_new_reminder_uses(client, harness):
    client.put(
        "/users/me/dashboard/reminders/delivery",
        headers=harness.bearer(harness.ana),
        json=dict(delivery=["telegram"]),
    )

    _create(harness.ana, "Call the clinic", "PT30M")
    body = client.get("/users/me/dashboard/reminders", headers=harness.bearer(harness.ana)).json()

    assert [c["channel"] for c in body["reminders"][0]["delivery"]] == ["telegram"]


def test_a_channel_this_owner_may_not_use_is_refused_not_saved(client, harness):
    response = client.put(
        "/users/me/dashboard/reminders/delivery",
        headers=harness.bearer(harness.bo),
        json=dict(delivery=["telegram"]),
    )

    assert response.status_code == 422
    assert reminder_delivery.default_channels(harness.bo) == ("speak",)


def test_a_default_cannot_be_set_for_anybody_else(client, harness):
    """There is no owner in the payload, and an extra field is refused outright."""
    response = client.put(
        "/users/me/dashboard/reminders/delivery",
        headers=harness.bearer(harness.bo),
        json=dict(delivery=["speak"], user_id=harness.ana),
    )

    assert response.status_code == 422
    assert reminder_delivery.default_channels(harness.ana) == ("speak",)


@pytest.mark.parametrize(
    "payload",
    [
        dict(delivery=["sms"]),
        dict(delivery=[]),
        dict(delivery="speak"),
        dict(delivery=["+15551230000"]),
        dict(),
    ],
)
def test_a_malformed_choice_is_refused(client, harness, payload):
    response = client.put(
        "/users/me/dashboard/reminders/delivery",
        headers=harness.bearer(harness.ana),
        json=payload,
    )

    assert response.status_code == 422


def test_an_unknown_caller_cannot_read_or_change_a_default(client, harness):
    assert client.get("/users/me/dashboard/reminders/delivery").status_code == 401
    assert (
        client.put(
            "/users/me/dashboard/reminders/delivery", json=dict(delivery=["speak"])
        ).status_code
        == 401
    )


# --- privacy ---------------------------------------------------------------------------------


def test_nothing_private_reaches_the_log(client, harness, caplog):
    target = logging.getLogger("caal.dashboard_api")
    target.setLevel(logging.DEBUG)
    target.addHandler(caplog.handler)
    try:
        _create(harness.ana, "Biopsy results", "PT30M", ["speak"])
        client.get("/users/me/dashboard/reminders", headers=harness.bearer(harness.ana))
        client.put(
            "/users/me/dashboard/reminders/delivery",
            headers=harness.bearer(harness.ana),
            json=dict(delivery=["speak"]),
        )
    finally:
        target.removeHandler(caplog.handler)

    text = "\n".join(record.getMessage() for record in caplog.records)
    for secret in ("Biopsy", "results", harness.ana, harness.bo):
        assert secret not in text
