"""The MCP client JARVIS uses to reach its own company library.

Thin on purpose: it mints a principal, speaks the protocol, and hands the
payload back. It holds no results, caches nothing between calls and knows no
owner of its own -- the owner of a call is the ``subject`` the trusted
dispatcher passes in, which is the verified user of the session and never
anything the model wrote.

One MCP session per call, and therefore one freshly minted principal per HTTP
request inside it. That is what makes the service's single-use replay control
meaningful: a token reused across the requests of one session would be
indistinguishable from a captured token replayed by somebody else. The
service runs stateless, so a session costs an ``initialize`` on a loopback
socket and nothing is left behind on the server between calls -- which is
also why no cache here can ever be read by the wrong owner: there isn't one.

Every failure is an exception with a short, non-echoing message. Nothing here
logs a query, a passage, a title or a subject.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

from caal.internal_auth import PrincipalError, mint_principal

from .config import CompanyConfig
from .mcp_server import PrincipalAuth, tool_payload
from .principal import (
    ACTION_CLAIM,
    ACTION_READ,
    AUDIENCE_COMPANY_MCP,
    ISSUER_COMPANY_CLIENT,
    PRINCIPAL_TTL_SECONDS,
)

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_TIMEOUT_SECONDS", "CompanyClientError", "CompanyMcpClient"]

DEFAULT_TIMEOUT_SECONDS = 15.0


class CompanyClientError(RuntimeError):
    """The company library could not be reached or refused the call."""


class CompanyMcpClient:
    """A real MCP client for the loopback company library service."""

    def __init__(self, *, config: CompanyConfig, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        self._config = config
        self._timeout = float(timeout)

    @property
    def url(self) -> str:
        return self._config.mcp_url

    def _auth(self, subject: str) -> PrincipalAuth:
        from caal.user_scope import is_valid_user_id

        if not is_valid_user_id(subject):
            raise CompanyClientError("This call is not bound to a verified user")
        secret = self._config.internal_auth_secret
        bearer = self._config.mcp_token
        if not secret or not bearer:
            raise CompanyClientError("The company library service is not configured")

        def mint() -> str:
            try:
                return mint_principal(
                    secret=secret,
                    subject=subject,
                    audience=AUDIENCE_COMPANY_MCP,
                    issuer=ISSUER_COMPANY_CLIENT,
                    ttl_seconds=PRINCIPAL_TTL_SECONDS,
                    claims={ACTION_CLAIM: ACTION_READ},
                )
            except PrincipalError as exc:
                raise CompanyClientError("The company principal could not be signed") from exc

        return PrincipalAuth(mint, bearer=bearer)

    async def call(self, name: str, arguments: dict[str, Any], *, subject: str) -> dict[str, Any]:
        """Run one MCP tool as ``subject`` and return its payload."""
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        auth = self._auth(subject)
        timeout = timedelta(seconds=self._timeout)
        try:
            async with streamablehttp_client(
                self.url, auth=auth, timeout=timeout, sse_read_timeout=timeout
            ) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(name, dict(arguments or {}))
        except CompanyClientError:
            raise
        except Exception as exc:  # noqa: BLE001 - the transport has many failure shapes
            logger.error("The company library service call failed (%s)", type(exc).__name__)
            raise CompanyClientError("The company library service did not answer") from exc
        if result.isError:
            raise CompanyClientError("The company library refused that request")
        payload = tool_payload(result)
        if not payload:
            raise CompanyClientError("The company library returned nothing readable")
        return payload

    async def list_tools(self, *, subject: str) -> tuple[str, ...]:
        """The tool names the service publishes. Used by the health check."""
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        auth = self._auth(subject)
        timeout = timedelta(seconds=self._timeout)
        try:
            async with streamablehttp_client(
                self.url, auth=auth, timeout=timeout, sse_read_timeout=timeout
            ) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    listed = await session.list_tools()
        except Exception as exc:  # noqa: BLE001
            raise CompanyClientError("The company library service did not answer") from exc
        return tuple(sorted(tool.name for tool in listed.tools))
