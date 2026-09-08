"""The production :class:`TokenExchanger`: a bounded HTTPS transport plus identity.

Redeeming an authorization code is two things, and this module does both
within fixed limits:

1. **The exchange.** One ``POST`` of form fields to the provider's token
   endpoint, authenticated with the application's client id and secret in
   the body (what Google, Microsoft and Zoho all accept), carrying the PKCE
   verifier where the provider uses one. Only ``https`` endpoints are ever
   contacted, redirects are never followed, the response is read within a
   byte budget, every socket operation has a timeout, and the whole exchange
   has a wall-clock budget on top. Zoho reports failures with a ``200`` and
   an ``error`` field; that is a refusal like any other.

2. **The identity.** Which account was linked is decided by the provider,
   never by what a browser sent and never by a claim this module cannot
   verify. After the exchange, one ``GET`` of the provider's own identity
   endpoint over https (Google userinfo, Microsoft Graph ``/me``, Zoho's
   accounts server), authenticated with the access token just issued,
   supplies the account id and a label. That call is subject to the same
   timeout, no-redirect and byte bounds as the exchange. A grant whose
   account the provider will not name is refused: two accounts from one
   provider could not be kept apart.

   An ``id_token`` in the token response is deliberately ignored. Google
   and Microsoft return one, but this module holds no provider signing key
   and performs no JWS verification, so its claims are unverified bytes;
   parsing them would let whoever can shape that response choose which
   account a grant is filed under. Nothing here decodes, checks or logs it.

Every failure becomes one :class:`TokenExchangeError` with a bounded
``reason`` and the names of the settings an operator would look at. The
provider's wording is never quoted, and nothing here logs a code, a token,
a secret, an account id or an email.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from urllib.parse import urlsplit

import httpx

from .oauth_providers import ProviderConfig, TokenExchangeError, TokenGrant

logger = logging.getLogger(__name__)

__all__ = [
    "CONNECT_TIMEOUT_SECONDS",
    "MAX_REQUESTS_PER_EXCHANGE",
    "MAX_RESPONSE_BYTES",
    "TIMEOUT_SECONDS",
    "HttpTokenExchanger",
]

# Per-operation socket budget and the connect budget within it.
TIMEOUT_SECONDS = 10.0
CONNECT_TIMEOUT_SECONDS = 5.0
_MAX_TIMEOUT_SECONDS = 30.0
# Token responses are a few hundred bytes; profile answers a few kilobytes.
MAX_RESPONSE_BYTES = 64 * 1024
_MIN_RESPONSE_BYTES = 1024
_MAX_RESPONSE_BYTES_LIMIT = 256 * 1024
# The token POST and at most one identity GET.
MAX_REQUESTS_PER_EXCHANGE = 2

_MAX_CODE_LENGTH = 4096
_MAX_TOKEN_LENGTH = 8192
_MAX_LABEL_LENGTH = 254
_MAX_ACCOUNT_ID_LENGTH = 256
_MAX_TOKEN_LIFETIME_SECONDS = 10 * 365 * 86400
_USER_AGENT = "CAAL-JARVIS provider-connections"

# Google expands the OpenID shorthand scopes in the ``scope`` it echoes back.
_GOOGLE_SCOPE_ALIASES = {
    "https://www.googleapis.com/auth/userinfo.email": "email",
    "https://www.googleapis.com/auth/userinfo.profile": "profile",
}


def _is_https(url: str) -> bool:
    parts = urlsplit(url)
    return parts.scheme == "https" and bool(parts.netloc)


def _token_like(value: object, *, limit: int = _MAX_TOKEN_LENGTH) -> str | None:
    """A non-empty, printable, whitespace-free string within ``limit``, else None."""
    if not isinstance(value, str) or not value or len(value) > limit:
        return None
    if not value.isprintable() or any(ch.isspace() for ch in value):
        return None
    return value


def _label_like(value: object) -> str | None:
    if not isinstance(value, str) or not value.isprintable():
        return None
    label = " ".join(value.split())
    if not label or len(label) > _MAX_LABEL_LENGTH:
        return None
    return label


def _account_id_like(value: object) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        value = str(value)
    return _token_like(value, limit=_MAX_ACCOUNT_ID_LENGTH)


def _expires_in(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, str) and value.isdigit():
        value = int(value)
    if not isinstance(value, int) or not 0 <= value <= _MAX_TOKEN_LIFETIME_SECONDS:
        return None
    return value


def _json_object(body: bytes) -> dict[str, Any] | None:
    try:
        loaded = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _split_scopes(value: str) -> list[str]:
    # Space-separated per RFC 6749; Zoho is known to use commas in places.
    return [part for part in value.replace(",", " ").split() if part]


def _error_code(data: dict[str, Any]) -> str | None:
    """The provider's short error code, if it is one, never its description."""
    code = data.get("error")
    return code if isinstance(code, str) and 0 < len(code) <= 64 and code.isprintable() else None


class HttpTokenExchanger:
    """Redeems authorization codes over ``httpx`` within fixed bounds.

    ``transport`` is for tests (``httpx.MockTransport``); production uses the
    default.
    """

    def __init__(
        self,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_seconds: float = TIMEOUT_SECONDS,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
    ) -> None:
        if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool):
            raise ValueError("timeout_seconds must be a number of seconds")
        if not 0 < float(timeout_seconds) <= _MAX_TIMEOUT_SECONDS:
            raise ValueError(f"timeout_seconds must be within (0, {_MAX_TIMEOUT_SECONDS}]")
        if isinstance(max_response_bytes, bool) or not isinstance(max_response_bytes, int):
            raise ValueError("max_response_bytes must be a whole number of bytes")
        if not _MIN_RESPONSE_BYTES <= max_response_bytes <= _MAX_RESPONSE_BYTES_LIMIT:
            raise ValueError(
                f"max_response_bytes must be within [{_MIN_RESPONSE_BYTES}, "
                f"{_MAX_RESPONSE_BYTES_LIMIT}]"
            )
        self._transport = transport
        self._timeout = float(timeout_seconds)
        self._max_bytes = int(max_response_bytes)

    def __repr__(self) -> str:
        return (
            f"HttpTokenExchanger(timeout_seconds={self._timeout}, "
            f"max_response_bytes={self._max_bytes})"
        )

    @property
    def timeout_seconds(self) -> float:
        return self._timeout

    @property
    def max_response_bytes(self) -> int:
        return self._max_bytes

    # --- entry point ---------------------------------------------------------------

    async def exchange(
        self,
        config: ProviderConfig,
        *,
        code: str,
        redirect_uri: str,
        code_verifier: str | None,
    ) -> TokenGrant:
        if _token_like(code, limit=_MAX_CODE_LENGTH) is None:
            raise ValueError("The authorization code has an invalid shape")
        if redirect_uri != config.redirect_uri:
            raise ValueError("The redirect URI must be the configured one")
        if config.pkce and _token_like(code_verifier) is None:
            raise ValueError("A PKCE verifier is required for this provider")
        if not _is_https(config.token_endpoint):
            raise TokenExchangeError("Token endpoint is not https", reason="transport")

        budget = self._timeout * MAX_REQUESTS_PER_EXCHANGE
        try:
            return await asyncio.wait_for(
                self._exchange(config, code=code, code_verifier=code_verifier), timeout=budget
            )
        except asyncio.TimeoutError as exc:
            raise TokenExchangeError(
                "The exchange exceeded its time budget", reason="transport"
            ) from exc

    async def _exchange(
        self, config: ProviderConfig, *, code: str, code_verifier: str | None
    ) -> TokenGrant:
        timeout = httpx.Timeout(self._timeout, connect=min(CONNECT_TIMEOUT_SECONDS, self._timeout))
        async with httpx.AsyncClient(
            transport=self._transport,
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
            headers={"Accept": "application/json", "User-Agent": _USER_AGENT},
        ) as client:
            data = await self._redeem(client, config, code=code, code_verifier=code_verifier)
            access_token = _token_like(data.get("access_token"))
            if access_token is None:
                raise TokenExchangeError("No usable access token", reason="malformed_response")
            token_type = data.get("token_type")
            if token_type is not None and (
                not isinstance(token_type, str) or token_type.lower() != "bearer"
            ):
                raise TokenExchangeError("Unsupported token type", reason="malformed_response")
            refresh_token: str | None = None
            if data.get("refresh_token") is not None:
                refresh_token = _token_like(data.get("refresh_token"))
                if refresh_token is None:
                    raise TokenExchangeError("Unusable refresh token", reason="malformed_response")
            echoed = self._echoed_scopes(data)
            granted = echoed if echoed is not None else tuple(config.scopes)

            # Whatever else the token response carried (an ``id_token`` included)
            # is not consulted: only the provider, asked directly, names the account.
            try:
                account_id, label = await self._resolve_identity(client, config, access_token)
            except TokenExchangeError as exc:
                if exc.reason == "identity_unavailable" and not self._identity_scopes_granted(
                    config, echoed
                ):
                    raise TokenExchangeError(
                        "The identity scope was not granted",
                        reason="insufficient_scope",
                        settings=(config.env_scopes,),
                    ) from exc
                raise

        return TokenGrant(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=_expires_in(data.get("expires_in")),
            scopes=granted,
            provider_account_id=account_id,
            account_label=label,
        )

    # --- the token request -----------------------------------------------------------

    async def _redeem(
        self,
        client: httpx.AsyncClient,
        config: ProviderConfig,
        *,
        code: str,
        code_verifier: str | None,
    ) -> dict[str, Any]:
        form: dict[str, str] = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": config.redirect_uri,
            "client_id": config.client_id,
            "client_secret": config.client_secret,
        }
        if config.pkce and code_verifier is not None:
            form["code_verifier"] = code_verifier
        request = client.build_request("POST", config.token_endpoint, data=form)
        status, body = await self._send(client, request)
        if 300 <= status < 400:
            raise TokenExchangeError("The token endpoint redirected", reason="malformed_response")
        data = _json_object(body)
        if data is None:
            if 200 <= status < 300:
                raise TokenExchangeError("Token response is not JSON", reason="malformed_response")
            raise TokenExchangeError(f"Token endpoint answered {status}", reason="provider_refused")
        error = _error_code(data)
        if error is not None or not 200 <= status < 300:
            if error == "invalid_scope":
                raise TokenExchangeError(
                    "The provider rejected the requested scopes",
                    reason="insufficient_scope",
                    settings=(config.env_scopes,),
                )
            raise TokenExchangeError(
                f"The provider refused the exchange ({error or status})",
                reason="provider_refused",
                provider_code=(
                    error if error is not None and error.isascii() and error.islower() else None
                ),
            )
        return data

    async def _send(self, client: httpx.AsyncClient, request: httpx.Request) -> tuple[int, bytes]:
        """Send one request; return status and a body read within the byte budget."""
        if not _is_https(str(request.url)):
            raise TokenExchangeError("Refusing a non-https request", reason="transport")
        try:
            response = await client.send(request, stream=True)
        except (httpx.HTTPError, OSError) as exc:
            raise TokenExchangeError(
                f"Transport failure: {type(exc).__name__}", reason="transport"
            ) from exc
        try:
            declared = response.headers.get("content-length")
            if declared is not None and (not declared.isdigit() or int(declared) > self._max_bytes):
                raise TokenExchangeError("Response too large", reason="malformed_response")
            chunks: list[bytes] = []
            total = 0
            try:
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > self._max_bytes:
                        raise TokenExchangeError("Response too large", reason="malformed_response")
                    chunks.append(chunk)
            except (httpx.HTTPError, OSError) as exc:
                raise TokenExchangeError(
                    f"Transport failure: {type(exc).__name__}", reason="transport"
                ) from exc
        finally:
            await response.aclose()
        return response.status_code, b"".join(chunks)

    # --- scopes --------------------------------------------------------------------------

    @staticmethod
    def _echoed_scopes(data: dict[str, Any]) -> tuple[str, ...] | None:
        raw = data.get("scope")
        if raw is None:
            return None
        if not isinstance(raw, str):
            raise TokenExchangeError("Unusable scope field", reason="malformed_response")
        scopes = tuple(dict.fromkeys(_split_scopes(raw)))
        for scope in scopes:
            if _token_like(scope, limit=256) is None:
                raise TokenExchangeError("Unusable scope field", reason="malformed_response")
        return scopes or None

    @staticmethod
    def _identity_scopes_granted(config: ProviderConfig, echoed: tuple[str, ...] | None) -> bool:
        """Whether an echoed scope list covers the identity scopes; unknown if none echoed."""
        if echoed is None:
            return True
        normalized = {_GOOGLE_SCOPE_ALIASES.get(scope, scope) for scope in echoed}
        return set(config.identity_scopes) <= normalized

    # --- identity --------------------------------------------------------------------------

    async def _resolve_identity(
        self, client: httpx.AsyncClient, config: ProviderConfig, access_token: str
    ) -> tuple[str, str | None]:
        """Ask the provider's identity endpoint, with the new access token, who this is.

        This is the only source of the account id. It goes through :meth:`_send`,
        so it is https-only, never follows a redirect, and is read within the
        same timeout and byte budget as the token request.
        """
        unavailable = TokenExchangeError(
            "The provider did not identify the account",
            reason="identity_unavailable",
            settings=(config.env_scopes,),
        )
        if not _is_https(config.identity_endpoint):
            raise unavailable
        scheme = "Zoho-oauthtoken" if config.provider == "zoho" else "Bearer"
        request = client.build_request(
            "GET", config.identity_endpoint, headers={"Authorization": f"{scheme} {access_token}"}
        )
        try:
            status, body = await self._send(client, request)
        except TokenExchangeError as exc:
            raise unavailable from exc
        data = _json_object(body) if 200 <= status < 300 else None
        if data is None or _error_code(data) is not None:
            raise unavailable
        if config.provider == "google":
            account_id = _account_id_like(data.get("sub"))
            label = _label_like(data.get("email"))
        elif config.provider == "microsoft":
            account_id = _account_id_like(data.get("id"))
            label = _label_like(data.get("mail")) or _label_like(data.get("userPrincipalName"))
        else:
            account_id = _account_id_like(data.get("ZUID"))
            label = _label_like(data.get("Email"))
        if account_id is None:
            raise unavailable
        return account_id, label
