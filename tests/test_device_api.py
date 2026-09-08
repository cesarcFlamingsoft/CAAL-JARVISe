"""API coverage for the authenticated device session and handoff endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from caal import device_registry, webhooks

ENROLLMENT_TOKEN = "enrollment-secret"
ROOM_NAME = "jarvis-private-room"


@pytest.fixture(autouse=True)
def isolated_store(monkeypatch, tmp_path):
    """Keep every test on its own database and enrollment secret."""
    monkeypatch.setattr(device_registry, "STORE_PATH", tmp_path / "assistant.sqlite3")
    monkeypatch.setenv("CAAL_DEVICE_ENROLLMENT_TOKEN", ENROLLMENT_TOKEN)


@pytest.fixture()
def client():
    with TestClient(webhooks.app) as test_client:
        yield test_client


def _register(
    client, device_id="phone-1", label="Cesar's iPhone", transport="mobile", token=ENROLLMENT_TOKEN
):
    return client.post(
        "/devices/register",
        headers={"X-Caal-Device-Token": token},
        json={
            "device_id": device_id,
            "room_name": ROOM_NAME,
            "label": label,
            "transport": transport,
        },
    )


def _session_token(client, **kwargs) -> str:
    response = _register(client, **kwargs)
    assert response.status_code == 200
    return response.json()["session_token"]


def test_register_issues_a_session_token_without_echoing_the_room(client) -> None:
    response = _register(client)

    assert response.status_code == 200
    body = response.json()
    assert body["device_id"] == "phone-1"
    assert body["label"] == "Cesar's iPhone"
    assert body["transport"] == "mobile"
    assert body["expires_in"] == device_registry.SESSION_TTL_SECONDS
    assert len(body["session_token"]) >= 32
    assert ROOM_NAME not in response.text


def test_register_requires_the_enrollment_token(client, monkeypatch) -> None:
    assert _register(client, token="wrong-secret").status_code == 401

    unauthenticated = client.post(
        "/devices/register",
        json={
            "device_id": "phone-1",
            "room_name": ROOM_NAME,
            "label": "Phone",
            "transport": "mobile",
        },
    )
    assert unauthenticated.status_code == 401

    monkeypatch.delenv("CAAL_DEVICE_ENROLLMENT_TOKEN", raising=False)
    assert _register(client).status_code == 401


def test_register_rejects_invalid_labels_and_transports(client) -> None:
    assert _register(client, label="   ").status_code == 422
    assert _register(client, label="x" * 200).status_code == 422
    assert _register(client, transport="carrier-pigeon").status_code == 422


def test_heartbeat_keeps_a_session_alive_and_rejects_bad_tokens(client) -> None:
    token = _session_token(client)

    beat = client.post("/devices/heartbeat", headers={"Authorization": f"Bearer {token}"})

    assert beat.status_code == 200
    assert beat.json()["status"] == "ok"
    assert beat.json()["expires_in"] == device_registry.SESSION_TTL_SECONDS
    assert ROOM_NAME not in beat.text

    assert client.post("/devices/heartbeat").status_code == 401
    assert (
        client.post("/devices/heartbeat", headers={"Authorization": "Bearer nope"}).status_code
        == 401
    )
    assert client.post("/devices/heartbeat", headers={"Authorization": token}).status_code == 401


def test_expired_sessions_are_rejected_everywhere(client) -> None:
    stale = device_registry.register_device_session(
        device_id="phone-1",
        room_name=ROOM_NAME,
        label="Stale Phone",
        transport="mobile",
        now=1000,
    )
    headers = {"Authorization": f"Bearer {stale.session_id}"}

    assert client.post("/devices/heartbeat", headers=headers).status_code == 401
    assert client.get("/devices/active", headers=headers).status_code == 401
    assert client.post("/devices/handoff", headers=headers, json={"context": {}}).status_code == 401


def test_active_listing_is_friendly_and_leaks_nothing(client) -> None:
    phone_token = _session_token(client, device_id="phone-1", label="Cesar's iPhone")
    laptop_token = _session_token(
        client, device_id="laptop-1", label="Studio Laptop", transport="web"
    )

    response = client.get("/devices/active", headers={"Authorization": f"Bearer {laptop_token}"})

    assert response.status_code == 200
    devices = response.json()["devices"]
    assert {device["label"] for device in devices} == {"Cesar's iPhone", "Studio Laptop"}
    assert [device["is_self"] for device in devices if device["label"] == "Studio Laptop"] == [True]
    assert all(set(device) == {"label", "transport", "last_seen", "is_self"} for device in devices)
    assert ROOM_NAME not in response.text
    assert phone_token not in response.text
    assert laptop_token not in response.text
    assert "device_id" not in response.text


def test_active_listing_requires_a_session(client) -> None:
    _register(client)

    assert client.get("/devices/active").status_code == 401
    assert (
        client.get("/devices/active", headers={"Authorization": "Bearer nope"}).status_code == 401
    )


def test_handoff_create_returns_an_id_only_to_the_owning_client(client) -> None:
    token = _session_token(client)

    response = client.post(
        "/devices/handoff",
        headers={"Authorization": f"Bearer {token}"},
        json={"context": {"summary": "Reviewing the deploy plan"}},
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["handoff_id"]) >= 32
    assert body["expires_in"] == device_registry.HANDOFF_TTL_SECONDS
    assert ROOM_NAME not in response.text
    assert token not in response.text
    assert "Reviewing the deploy plan" not in response.text


def test_handoff_create_requires_a_session_and_bounded_context(client) -> None:
    token = _session_token(client)
    headers = {"Authorization": f"Bearer {token}"}

    assert client.post("/devices/handoff", json={"context": {}}).status_code == 401

    oversized = client.post(
        "/devices/handoff",
        headers=headers,
        json={"context": {f"key-{index}": index for index in range(40)}},
    )
    nested = client.post(
        "/devices/handoff", headers=headers, json={"context": {"nested": {"no": "deep"}}}
    )

    assert oversized.status_code == 422
    assert nested.status_code == 422


def test_handoff_is_claimable_once_by_another_authenticated_device(client) -> None:
    phone_token = _session_token(client, device_id="phone-1")
    laptop_token = _session_token(
        client, device_id="laptop-1", label="Studio Laptop", transport="web"
    )
    handoff_id = client.post(
        "/devices/handoff",
        headers={"Authorization": f"Bearer {phone_token}"},
        json={"context": {"summary": "Reviewing the deploy plan"}},
    ).json()["handoff_id"]
    headers = {"Authorization": f"Bearer {laptop_token}"}

    claimed = client.post(
        "/devices/handoff/claim", headers=headers, json={"handoff_id": handoff_id}
    )
    replayed = client.post(
        "/devices/handoff/claim", headers=headers, json={"handoff_id": handoff_id}
    )

    assert claimed.status_code == 200
    assert claimed.json()["status"] == "claimed"
    assert claimed.json()["context"] == {"summary": "Reviewing the deploy plan"}
    assert ROOM_NAME not in claimed.text
    assert replayed.status_code == 404


def test_claim_requires_a_session_and_hides_unknown_handoffs(client) -> None:
    token = _session_token(client)
    headers = {"Authorization": f"Bearer {token}"}

    assert client.post("/devices/handoff/claim", json={"handoff_id": "abc"}).status_code == 401

    unknown = client.post("/devices/handoff/claim", headers=headers, json={"handoff_id": "abc"})
    assert unknown.status_code == 404
    assert unknown.json()["detail"] == "Handoff is not available."
