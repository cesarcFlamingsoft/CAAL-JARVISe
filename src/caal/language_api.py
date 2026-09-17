"""Authenticated, narrow reply-language selection on the existing user boundary.

Deliberately has no admin or deployment-wide write path: the language JARVIS
answers in is a personal choice, so one user changing it can never move another
user's session.
"""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict

from .language_policy import AUTO
from .language_store import LanguageStore
from .user_api import CurrentUser, IdentityRuntime, get_runtime, throttle_mutation

router = APIRouter(tags=["language"])


class LanguageSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    language: Literal["auto", "en", "es"]


def view(user, identity):
    preference = LanguageStore(identity).preference(user.profile.user_id) if identity else AUTO
    return {
        "language": preference,
        "source": "personal" if preference != AUTO else "default",
        # The choice is read once, against the signed principal, when a voice
        # session starts; an in-flight session keeps the language it began with.
        "applies_to": "new_sessions",
    }


@router.get("/users/me/language")
def read_language(
    user: CurrentUser = Depends(throttle_mutation),
    identity: IdentityRuntime = Depends(get_runtime),
):
    return view(user, identity)


@router.put("/users/me/language")
def write_language(
    body: LanguageSelection,
    user: CurrentUser = Depends(throttle_mutation),
    identity: IdentityRuntime = Depends(get_runtime),
):
    if identity is None:
        raise HTTPException(503, "not_configured")
    LanguageStore(identity).save_preference(user.profile.user_id, body.language)
    return view(user, identity)
