"""Owner-only, browser-safe durable work feed contract."""
import logging

import pytest
from fastapi.testclient import TestClient
from test_dashboard_reminders_api import Harness

from caal import background_tasks as tasks
from caal import ha_policy, user_api, webhooks


@pytest.fixture
def harness(tmp_path, monkeypatch):
    monkeypatch.setattr(tasks, "STORE_PATH", tmp_path / "assistant.sqlite3")
    # Admission policy is tested elsewhere; seed durable work for both owners.
    monkeypatch.setattr(ha_policy, "require_delegated_scope", lambda scope: None)
    return Harness(tmp_path)


@pytest.fixture
def client(harness):
    webhooks.app.dependency_overrides[user_api.get_runtime] = lambda: harness.runtime
    try:
        with TestClient(webhooks.app) as test_client:
            yield test_client
    finally:
        webhooks.app.dependency_overrides.pop(user_api.get_runtime, None)


def test_work_is_owned_bounded_redacted_and_newest_first(client, harness, caplog):
    for index in range(15):
        tasks.enqueue(f"Research item {index} password=private-credential " + "x" * 160,
                      user_id=harness.ana, session_key="private-session")
    tasks.enqueue("Other owner private work", user_id=harness.bo)
    tasks.enqueue("Legacy private work")
    with caplog.at_level(logging.INFO):
        response = client.get("/users/me/dashboard/work", headers=harness.bearer(harness.ana))
    assert response.status_code == 200
    assert "no-store" in response.headers["cache-control"]
    body = response.json()
    assert set(body) == {"generated_at", "items"}
    assert type(body["generated_at"]) is int
    assert len(body["items"]) == 12
    assert [item["title"].split()[2] for item in body["items"]] == [
        str(i) for i in range(14, 2, -1)
    ]
    for item in body["items"]:
        assert set(item) == {
            "title",
            "status",
            "created_at",
            "updated_at",
            "started_at",
            "finished_at",
        }
        assert len(item["title"]) <= 120
        assert "[REDACTED]" in item["title"]
        assert item["status"] == "queued"
        assert type(item["created_at"]) is int
        assert type(item["updated_at"]) is int
        assert item["started_at"] is None and item["finished_at"] is None
    for forbidden in (
        "private-credential",
        "private-session",
        "Other owner",
        "Legacy private",
        harness.ana,
    ):
        assert forbidden not in response.text
        assert forbidden not in caplog.text
    assert "Research item" not in caplog.text
    other = client.get("/users/me/dashboard/work", headers=harness.bearer(harness.bo)).json()
    assert [item["title"] for item in other["items"]] == ["Other owner private work"]


def test_work_requires_auth_and_rejects_all_parameters(client, harness):
    assert client.get("/users/me/dashboard/work").status_code == 401
    for key in ("user_id", "task_id", "session_key", "owner", "limit"):
        assert client.get(f"/users/me/dashboard/work?{key}=anything",
                          headers=harness.bearer(harness.ana)).status_code == 422


def test_work_all_lifecycle_states_and_no_mutation(client, harness):
    for index, status in enumerate(sorted(tasks.ALL_STATUSES)):
        task = tasks.enqueue("Prepare report", user_id=harness.ana)
        with tasks._connect() as db:
            db.execute("UPDATE background_tasks SET status=?, created_at=?, updated_at=?, "
                       "started_at=?, finished_at=?, result=?, error=? WHERE task_id=?",
                       (status, 100 + index, 200 + index, 150, 200, "private-result",
                        "private-exception", task.task_id))
    before = tasks.list_tasks(user_id=harness.ana)
    response = client.get("/users/me/dashboard/work", headers=harness.bearer(harness.ana))
    assert response.status_code == 200
    items = response.json()["items"]
    assert {item["status"] for item in items} == tasks.ALL_STATUSES
    assert [item["created_at"] for item in items] == list(range(105, 99, -1))
    assert all(
        type(item[key]) is int
        for item in items
        for key in ("updated_at", "started_at", "finished_at")
    )
    assert "private-result" not in response.text and "private-exception" not in response.text
    assert tasks.list_tasks(user_id=harness.ana) == before


def test_work_empty_feed_and_single_use_principal(client, harness):
    headers = harness.bearer(harness.ana)
    response = client.get("/users/me/dashboard/work", headers=headers)
    assert response.status_code == 200
    assert response.json()["items"] == []
    assert client.get("/users/me/dashboard/work", headers=headers).status_code == 401
