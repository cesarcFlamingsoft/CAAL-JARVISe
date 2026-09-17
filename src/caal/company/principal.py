"""Who may ask the company library a question.

The rule this module exists to enforce:

    A service bearer alone MUST NOT authorize a model-supplied owner id.

So authorization needs two independent things, and then a third check that
trusts neither of them:

1. the **service bearer** proves the caller is the JARVIS runtime rather than
   any other process that can reach loopback. Compared in constant time;
2. a **signed principal** (:mod:`caal.internal_auth`) proves *which verified
   session* is asking. It is pinned to the ``caal-company-mcp`` audience and
   the ``caal-company-client`` issuer, carries an action scope, lives about a
   minute, and is single use, so a captured header cannot be replayed;
3. the subject is then **re-read from the identity store**: it must be an
   active administrator, and it must be the owner of this library. The role
   is never taken from a claim, a header, or a tool argument.

Failures are deliberately uninformative to the caller (``unauthorized`` /
``forbidden``) and never echo a token. Nothing here logs a subject.
"""

from __future__ import annotations

import hmac
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from caal.internal_auth import NonceStore, Principal, PrincipalError, verify_principal

from .config import CompanyConfig

logger = logging.getLogger(__name__)

__all__ = [
    "ACTION_CLAIM",
    "ACTION_READ",
    "AUDIENCE_COMPANY_MCP",
    "ISSUER_COMPANY_CLIENT",
    "PRINCIPAL_TTL_SECONDS",
    "CompanyAuthError",
    "CompanyAuthorizer",
    "UserFacts",
]

AUDIENCE_COMPANY_MCP = "caal-company-mcp"
ISSUER_COMPANY_CLIENT = "caal-company-client"
ACTION_CLAIM = "act"
ACTION_READ = "company.read"
PRINCIPAL_TTL_SECONDS = 60


@dataclass(frozen=True)
class UserFacts:
    """What the identity store says about a subject, and nothing else."""

    user_id: str
    role: str
    status: str


class CompanyAuthError(Exception):
    """Refusal. ``status`` is the HTTP status the transport should answer with."""

    def __init__(self, status: int, code: str) -> None:
        super().__init__(code)
        self.status = int(status)
        self.code = code


class OwnerSource(Protocol):
    def __call__(self) -> str | None: ...


@dataclass
class CompanyAuthorizer:
    """Turns two headers into the one opaque owner id the library will act for."""

    config: CompanyConfig
    resolve_user: Callable[[str], UserFacts | None]
    nonce_store: NonceStore
    owner_id: OwnerSource | None = None
    now: Callable[[], float] = field(default=time.time)

    def authorize(
        self, *, bearer: object, principal_token: object, action: str = ACTION_READ
    ) -> str:
        """Return the owner id this request may act for, or raise :class:`CompanyAuthError`."""
        self._check_bearer(bearer)
        principal = self._check_principal(principal_token, action)
        facts = self._facts(principal)
        owner = self.owner_id() if self.owner_id is not None else None
        if owner is not None and not hmac.compare_digest(owner, facts.user_id):
            # An administrator of this deployment who is not the owner of this
            # library. Refused: there is no shared company library.
            raise CompanyAuthError(403, "forbidden")
        return facts.user_id

    # --- the three checks ------------------------------------------------------------

    def _check_bearer(self, bearer: object) -> None:
        expected = self.config.mcp_token
        if not expected:
            raise CompanyAuthError(503, "not_configured")
        if not isinstance(bearer, str) or not bearer:
            raise CompanyAuthError(401, "unauthorized")
        if not hmac.compare_digest(bearer.encode("utf-8"), expected.encode("utf-8")):
            raise CompanyAuthError(401, "unauthorized")

    def _check_principal(self, token: object, action: str) -> Principal:
        secret = self.config.internal_auth_secret
        if not secret:
            raise CompanyAuthError(503, "not_configured")
        if not isinstance(token, str) or not token:
            # The bearer got this far on its own. It goes no further.
            raise CompanyAuthError(401, "unauthorized")
        try:
            principal = verify_principal(
                token,
                secret=secret,
                audience=AUDIENCE_COMPANY_MCP,
                issuer=ISSUER_COMPANY_CLIENT,
                now=self.now(),
                nonce_store=self.nonce_store,
                max_ttl_seconds=PRINCIPAL_TTL_SECONDS,
            )
        except PrincipalError as exc:
            raise CompanyAuthError(401, "unauthorized") from exc
        if principal.claims.get(ACTION_CLAIM) != action:
            raise CompanyAuthError(401, "unauthorized")
        return principal

    def _facts(self, principal: Principal) -> UserFacts:
        from caal.user_scope import is_valid_user_id

        subject = principal.subject
        if not is_valid_user_id(subject):
            raise CompanyAuthError(401, "unauthorized")
        try:
            facts = self.resolve_user(subject)
        except Exception as exc:  # noqa: BLE001 - an identity outage is a refusal
            logger.error("The company authorizer could not read the identity store")
            raise CompanyAuthError(503, "unavailable") from exc
        if facts is None:
            raise CompanyAuthError(401, "unauthorized")
        if facts.status != "active":
            raise CompanyAuthError(403, "forbidden")
        if facts.role != "admin":
            # No member, satellite or device reads the company library in this
            # release. There is no tier below administrator yet.
            raise CompanyAuthError(403, "forbidden")
        return facts
