"""Where the company library lives, and what opens it.

Three things come from the operator environment and nowhere else:

``CAAL_COMPANY_LIBRARY_DIR``
    the directory holding the encrypted originals and the encrypted index.
    It defaults to ``~/.caal/company-library`` and is deliberately *outside*
    the repository: company documents are not source, and a bind-mounted
    checkout must not be able to carry them anywhere. Created ``0700``.

``CAAL_COMPANY_LIBRARY_KEYS``
    a versioned AES-256-GCM ring in the same format the profile ring uses
    (``v1:<base64url 32 bytes>``), parsed by :class:`caal.profile_crypto.KeyRing`.
    It is a *separate* ring from the profile keys on purpose: one being
    compromised does not open the other.

``CAAL_COMPANY_OWNER_USER_ID``
    the opaque ``usr_...`` id of the one administrator this library belongs
    to. It is **provisioned by the operator before the first upload** and is
    then immutable: nobody claims the library by being first to write to it,
    and no user is ever chosen by name. Unset means the library is
    ``unconfigured`` -- it answers nothing, stores nothing, and says so.

``CAAL_COMPANY_NAME``
    optional. Unset means the organisation is unnamed, and no name is
    invented anywhere -- not in the UI, not in a prompt, not in an answer.

``CAAL_COMPANY_ROLE``
    ``owner`` (the single writer -- the CAAL backend) or ``reader`` (every
    other process, including the MCP service). Defaults to ``reader``, because
    the safe mistake is a process that cannot write.

``CAAL_COMPANY_READER_STATE_DIR``
    optional, and only meaningful for a reader. A reader's library mount is
    read only -- that is how "exactly one writer" is enforced by the platform
    and not only by a lock -- but a reader still has one thing it must write:
    the single-use principal nonces that stop a spent principal being replayed
    against another worker. That state does not belong in the library, so it
    gets its own writable directory. Unset means the library directory, which
    is correct for the writer and was the only case before the MCP service ran
    with ``:ro``.

The MCP settings (``CAAL_COMPANY_MCP_HOST``/``_PORT``/``_TOKEN``) describe a
loopback service. The host is validated to be a loopback address: this
service is never published.

Nothing here logs, and the config never renders its key material or its
service token.
"""

from __future__ import annotations

import ipaddress
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from caal.profile_crypto import KeyRing, KeyRingError

__all__ = [
    "DEFAULT_MCP_HOST",
    "DEFAULT_MCP_PORT",
    "MIN_TOKEN_LENGTH",
    "ROLE_OWNER",
    "ROLE_READER",
    "CompanyConfig",
    "CompanyConfigError",
]

DEFAULT_MCP_HOST = "127.0.0.1"
DEFAULT_MCP_PORT = 8791
MIN_TOKEN_LENGTH = 32
MAX_COMPANY_NAME = 80
ROLE_OWNER = "owner"
ROLE_READER = "reader"


class CompanyConfigError(ValueError):
    """The company library is not configured, or is configured wrongly."""


def _loopback(host: str) -> str:
    """Accept only an address that cannot be reached from off the machine."""
    if host == "localhost":
        return host
    try:
        if not ipaddress.ip_address(host).is_loopback:
            raise CompanyConfigError("The company MCP service may only bind a loopback address")
    except ValueError as exc:
        raise CompanyConfigError("The company MCP host is not an IP address") from exc
    return host


@dataclass(frozen=True)
class CompanyConfig:
    """Everything the library needs. Never prints its secrets."""

    data_dir: Path
    keys: KeyRing = field(repr=False)
    owner_user_id: str | None = None
    role: str = ROLE_READER
    company_name: str | None = None
    mcp_host: str = DEFAULT_MCP_HOST
    mcp_port: int = DEFAULT_MCP_PORT
    mcp_token: str | None = field(default=None, repr=False)
    internal_auth_secret: str | None = field(default=None, repr=False)
    #: Where a reader keeps the little state it must write. ``None`` means the
    #: library directory, which is right for the single writer.
    reader_state_dir: Path | None = None

    @property
    def mcp_url(self) -> str:
        return f"http://{self.mcp_host}:{self.mcp_port}/mcp"

    @property
    def state_dir(self) -> Path:
        """Where this process may write its own bookkeeping.

        The library directory for the writer; a dedicated directory for a
        reader, whose library mount is read only.
        """
        return self.reader_state_dir or self.data_dir

    @property
    def sources_dir(self) -> Path:
        return self.data_dir / "sources"

    @property
    def index_path(self) -> Path:
        return self.data_dir / "index.enc"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> CompanyConfig:
        """Build the config or raise :class:`CompanyConfigError`. Fails closed."""
        source = os.environ if env is None else env
        raw_keys = (source.get("CAAL_COMPANY_LIBRARY_KEYS") or "").strip()
        if not raw_keys:
            raise CompanyConfigError("CAAL_COMPANY_LIBRARY_KEYS is not configured")
        try:
            keys = KeyRing.from_env(raw_keys)
        except KeyRingError as exc:
            raise CompanyConfigError("CAAL_COMPANY_LIBRARY_KEYS is malformed") from exc

        role = (source.get("CAAL_COMPANY_ROLE") or ROLE_READER).strip().lower()
        if role not in (ROLE_OWNER, ROLE_READER):
            raise CompanyConfigError("CAAL_COMPANY_ROLE must be owner or reader")

        directory = (source.get("CAAL_COMPANY_LIBRARY_DIR") or "").strip()
        data_dir = Path(directory) if directory else Path.home() / ".caal" / "company-library"
        data_dir = data_dir.expanduser()
        # Only the writer prepares the library. A reader's mount is read only
        # on purpose, so creating or chmod-ing it would fail -- and succeeding
        # would mean the reader could write, which is the thing being
        # prevented. A reader needs the directory to exist and nothing else.
        if role == ROLE_OWNER:
            try:
                data_dir.mkdir(parents=True, exist_ok=True)
                data_dir.chmod(0o700)
                (data_dir / "sources").mkdir(exist_ok=True)
                (data_dir / "sources").chmod(0o700)
            except OSError as exc:
                raise CompanyConfigError("The company library directory is not writable") from exc
        # A reader whose directory does not exist yet simply has nothing to
        # serve, and says so. It is not an error: the MCP service is allowed to
        # start before the writer has opened the library for the first time.

        state_text = (source.get("CAAL_COMPANY_READER_STATE_DIR") or "").strip()
        reader_state_dir: Path | None = None
        if state_text:
            reader_state_dir = Path(state_text).expanduser()
            try:
                reader_state_dir.mkdir(parents=True, exist_ok=True)
                reader_state_dir.chmod(0o700)
            except OSError as exc:
                raise CompanyConfigError(
                    "CAAL_COMPANY_READER_STATE_DIR is not writable"
                ) from exc

        name = (source.get("CAAL_COMPANY_NAME") or "").strip() or None
        if name is not None and (len(name) > MAX_COMPANY_NAME or not name.isprintable()):
            raise CompanyConfigError("CAAL_COMPANY_NAME is malformed")

        port_text = (source.get("CAAL_COMPANY_MCP_PORT") or "").strip()
        try:
            port = int(port_text) if port_text else DEFAULT_MCP_PORT
        except ValueError as exc:
            raise CompanyConfigError("CAAL_COMPANY_MCP_PORT is not a number") from exc
        if not 1024 <= port <= 65535:
            raise CompanyConfigError("CAAL_COMPANY_MCP_PORT is out of range")

        token = (source.get("CAAL_COMPANY_MCP_TOKEN") or "").strip() or None
        if token is not None and len(token) < MIN_TOKEN_LENGTH:
            raise CompanyConfigError("CAAL_COMPANY_MCP_TOKEN is too short")

        secret = (source.get("CAAL_INTERNAL_AUTH_SECRET") or "").strip() or None

        # The owner is provisioned, never claimed. An unset value is a library
        # that is not yet open for business; a malformed one is a refusal
        # rather than a library bound to something that is not a user id.
        from caal.user_scope import is_valid_user_id

        owner = (source.get("CAAL_COMPANY_OWNER_USER_ID") or "").strip() or None
        if owner is not None and not is_valid_user_id(owner):
            raise CompanyConfigError("CAAL_COMPANY_OWNER_USER_ID is not a user identifier")

        return cls(
            data_dir=data_dir,
            keys=keys,
            owner_user_id=owner,
            role=role,
            company_name=name,
            mcp_host=_loopback((source.get("CAAL_COMPANY_MCP_HOST") or DEFAULT_MCP_HOST).strip()),
            mcp_port=port,
            mcp_token=token,
            internal_auth_secret=secret,
            reader_state_dir=reader_state_dir,
        )
