from __future__ import annotations


def test_local_reminder_is_persisted_and_returns_speakable_result(monkeypatch, tmp_path):
    from caal.tools import reminders_tools

    monkeypatch.setattr(reminders_tools, "STORE_PATH", tmp_path / "assistant.sqlite3")

    created = reminders_tools.create_reminder(
        title="Call the dentist",
        due="2026-07-29T09:00:00-06:00",
        list_name="Personal",
        notes="Ask about the cleaning.",
    )

    assert created["status"] == "ok"
    assert created["message"] == "Created reminder: Call the dentist."
    assert created["data"]["title"] == "Call the dentist"
    assert created["data"]["list"] == "Personal"

    reminders = reminders_tools.list_reminders()

    assert len(reminders["data"]["reminders"]) == 1
    assert reminders["data"]["reminders"][0]["notes"] == "Ask about the cleaning."


def test_timer_is_persisted_and_claimed_exactly_once(monkeypatch, tmp_path):
    from caal.tools import alarms_tools

    monkeypatch.setattr(alarms_tools, "STORE_PATH", tmp_path / "assistant.sqlite3")
    created = alarms_tools.set_alarm(label="Tea timer", when="10m", kind="timer", now=1_000)

    assert created["status"] == "ok"
    assert created["data"]["due_at"] == 1_600
    assert alarms_tools.claim_due_alarms(now=1_599) == []

    due = alarms_tools.claim_due_alarms(now=1_600)
    assert [alarm["label"] for alarm in due] == ["Tea timer"]
    assert alarms_tools.claim_due_alarms(now=1_600) == []


def test_due_alarms_are_spoken_once(monkeypatch):
    import asyncio

    from caal import alarm_delivery

    class Session:
        def __init__(self):
            self.messages = []

        async def say(self, message):
            self.messages.append(message)

    monkeypatch.setattr(
        alarm_delivery,
        "claim_due_alarms",
        lambda: [{"label": "Tea timer", "kind": "timer"}],
    )
    session = Session()

    assert asyncio.run(alarm_delivery.announce_due_alarms(session)) == 1
    assert session.messages == ["Your timer 'Tea timer' is finished."]
