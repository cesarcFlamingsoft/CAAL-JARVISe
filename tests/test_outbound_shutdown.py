"""Ending an outbound job cleanly when the dial never happens.

A failed dial (LiveKit answering ``sip not connected (redis required)``) ended
the job with ``TypeError: object NoneType can not be used in await expression``:
``JobContext.shutdown`` returns ``None`` in the runtime LiveKit version, and the
call site awaited it. The unhandled error replaced the clean shutdown, and the
worker logged a job failure instead of a quiet end.

Pinned properties:

* the shutdown helper accepts both shapes of ``ctx.shutdown`` -- the sync one
  the runtime has and the coroutine one older versions and mocks have;
* a dial that fails still releases the job exactly once, says nothing to the
  callee, and leaves no voicemail;
* the outbound request is constructed with the fields the call needs. Nothing
  here dials: the SIP client is a recorder.
"""

from __future__ import annotations

import asyncio
import importlib.util
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from caal.outbound_runtime import OutboundRoomConfig, call_timeouts, shutdown_job

ATTEMPT_ID = "attempt-1"
DESTINATION = "+17805558345"
TRUNK = "ST_trunk"

_voice_agent = None


def _load_voice_agent():
    """Load voice_agent.py once; it is a script, not an importable package."""
    global _voice_agent
    if _voice_agent is None:
        module_path = Path(__file__).parents[1] / "voice_agent.py"
        spec = importlib.util.spec_from_file_location("voice_agent_shutdown_test", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _voice_agent = module
    return _voice_agent


def run(coro):
    return asyncio.run(coro)


class SyncShutdownContext:
    """``ctx.shutdown`` as the runtime LiveKit version defines it: it returns None."""

    def __init__(self) -> None:
        self.reasons: list[str] = []

    def shutdown(self, reason: str = "") -> None:
        self.reasons.append(reason)


class AsyncShutdownContext:
    """``ctx.shutdown`` as a coroutine, which other versions and test doubles use."""

    def __init__(self) -> None:
        self.reasons: list[str] = []

    async def shutdown(self, reason: str = "") -> None:
        self.reasons.append(reason)


class RecordingSIP:
    """A SIP client that records the request and fails the way LiveKit did."""

    def __init__(self, error: Exception | None = None) -> None:
        self.requests: list[object] = []
        self._error = error

    async def create_sip_participant(self, request):
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        return SimpleNamespace(participant_identity="caal-outbound-" + ATTEMPT_ID)


class FakeSession:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def say(self, text: str, **_) -> None:
        self.spoken.append(text)


# --- the compatibility helper -----------------------------------------------------------


def test_a_sync_shutdown_is_awaited_without_a_type_error() -> None:
    ctx = SyncShutdownContext()

    run(shutdown_job(ctx, "outbound call not answered by a human"))

    assert ctx.reasons == ["outbound call not answered by a human"]


def test_an_async_shutdown_is_awaited_the_same_way() -> None:
    ctx = AsyncShutdownContext()

    run(shutdown_job(ctx, "outbound trunk unavailable"))

    assert ctx.reasons == ["outbound trunk unavailable"]


def test_a_context_that_cannot_shut_down_does_not_raise_into_the_job() -> None:
    class Broken:
        def shutdown(self, reason: str = "") -> None:
            raise RuntimeError("already closed")

    run(shutdown_job(Broken(), "invalid outbound room configuration"))
    run(shutdown_job(SimpleNamespace(), "no shutdown at all"))


# --- the failed dial ---------------------------------------------------------------------


def test_a_dial_that_fails_before_ringing_ends_the_job_quietly(monkeypatch) -> None:
    """What production hit: LiveKit refuses the dial because SIP has no Redis."""
    voice_agent = _load_voice_agent()
    monkeypatch.setenv("LIVEKIT_OUTBOUND_TRUNK_ID", TRUNK)
    sip = RecordingSIP(error=RuntimeError("sip not connected (redis required)"))
    reasons: list[str] = []
    ctx = SimpleNamespace(
        api=SimpleNamespace(sip=sip),
        room=SimpleNamespace(name="caal-outbound-" + ATTEMPT_ID),
        shutdown=lambda reason="": reasons.append(reason),
    )
    session = FakeSession()
    config = OutboundRoomConfig(attempt_id=ATTEMPT_ID, destination=DESTINATION)

    answered = run(voice_agent.run_outbound_call(ctx, session, config))

    assert answered is False
    assert reasons == ["outbound call not answered by a human"]
    assert session.spoken == [], "nothing is said into a call that never connected"


def test_a_missing_trunk_ends_the_job_before_any_request_is_built(monkeypatch) -> None:
    voice_agent = _load_voice_agent()
    monkeypatch.delenv("LIVEKIT_OUTBOUND_TRUNK_ID", raising=False)
    sip = RecordingSIP()
    reasons: list[str] = []
    ctx = SimpleNamespace(
        api=SimpleNamespace(sip=sip),
        room=SimpleNamespace(name="caal-outbound-" + ATTEMPT_ID),
        shutdown=lambda reason="": reasons.append(reason),
    )
    config = OutboundRoomConfig(attempt_id=ATTEMPT_ID, destination=DESTINATION)

    answered = run(voice_agent.run_outbound_call(ctx, FakeSession(), config))

    assert answered is False
    assert sip.requests == []
    assert reasons == ["outbound trunk unavailable"]


# --- constructing the request (no dial) --------------------------------------------------


def test_the_outbound_request_carries_the_trunk_destination_room_and_timeouts() -> None:
    """A construction smoke test: no LiveKit connection, no PSTN call."""
    api = pytest.importorskip("livekit.api")
    ringing, duration = call_timeouts("30", "900")

    request = api.CreateSIPParticipantRequest(
        sip_trunk_id=TRUNK,
        sip_call_to=DESTINATION,
        room_name="caal-outbound-" + ATTEMPT_ID,
        participant_identity="caal-outbound-" + ATTEMPT_ID,
        participant_name="JARVIS outbound call",
        wait_until_answered=True,
        ringing_timeout=ringing,
        max_call_duration=duration,
    )

    assert request.sip_trunk_id == TRUNK
    assert request.sip_call_to == DESTINATION
    assert request.room_name == "caal-outbound-" + ATTEMPT_ID
    assert request.wait_until_answered is True
    assert ringing == timedelta(seconds=30) and duration == timedelta(seconds=900)
