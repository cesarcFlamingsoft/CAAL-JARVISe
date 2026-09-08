from __future__ import annotations

import pytest

from caal.outbound_calls import (
    OutboundCallPolicy,
    OutboundCallRequest,
    OutboundCallStatus,
    verify_control_token,
)


def test_outbound_policy_accepts_only_allowlisted_e164_destination() -> None:
    policy = OutboundCallPolicy.from_csv("+17805558345")

    request = policy.authorize("+17805558345")

    assert request.destination == "+17805558345"
    assert request.status is OutboundCallStatus.AUTHORIZED


def test_outbound_policy_rejects_unapproved_destination() -> None:
    policy = OutboundCallPolicy.from_csv("+17805558345")

    with pytest.raises(PermissionError, match="not approved"):
        policy.authorize("+17805550000")


def test_outbound_policy_rejects_non_e164_destination() -> None:
    policy = OutboundCallPolicy.from_csv("+17805558345")

    with pytest.raises(ValueError, match="E.164"):
        policy.authorize("780-555-8345")


def test_outbound_request_keeps_destination_out_of_room_metadata() -> None:
    request = OutboundCallRequest(destination="+17805558345", attempt_id="attempt-123")

    metadata = request.dispatch_metadata()

    assert metadata == {
        "caal_outbound": True,
        "attempt_id": "attempt-123",
        "destination": "+17805558345",
    }


def test_control_token_must_match_exactly() -> None:
    assert verify_control_token("expected", "expected") is True
    assert verify_control_token("expected", "wrong") is False
    assert verify_control_token("expected", None) is False
