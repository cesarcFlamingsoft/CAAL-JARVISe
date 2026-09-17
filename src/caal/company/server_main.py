"""Run the company library MCP service. Loopback only, supervised separately.

``python -m caal.company.server_main``

It builds three things and then serves:

* the **library**, from ``CAAL_COMPANY_LIBRARY_DIR`` and the company key ring,
  opened as a :data:`~caal.company.store.ROLE_READER`. This process never
  writes: not on a call, not on shutdown, not ever. It refreshes from the
  writer's published index before every authorization and every read, so a
  document uploaded a second ago is visible and a document deleted a second
  ago is not;
* the **authorizer**, whose ``resolve_user`` reads the same
  ``assistant.sqlite3`` the backend uses, so an administrator who is suspended
  or demoted stops being able to read the library on their very next request;
* the **ASGI app**, which refuses every request that does not carry both the
  service bearer and a fresh signed principal.

The owner is **provisioned, not claimed**: ``CAAL_COMPANY_OWNER_USER_ID`` must
name a user who exists in the identity database and is an active
administrator, and the service refuses to start otherwise. Nothing about being
first to upload makes anyone the owner.

It binds ``127.0.0.1`` (the config refuses anything else), publishes nothing,
and makes no outbound connection. Exits non-zero with a one-line reason when
it is not configured, so a supervisor's log says what is missing without ever
printing a value.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import replace

logger = logging.getLogger(__name__)

__all__ = ["main"]


def _user_resolver(config):
    """Read role and status from the identity database, on every request."""
    from caal.company.principal import UserFacts
    from caal.security_config import load_multi_user_config
    from caal.user_store import UserStore

    loaded = load_multi_user_config()
    if loaded.config is None:
        raise SystemExit("company-mcp: multi-user identity is not configured")
    store = UserStore(loaded.config.store_path, keyring=loaded.config.keyring)

    def resolve(user_id: str) -> UserFacts | None:
        profile = store.get_user(user_id)
        if profile is None:
            return None
        return UserFacts(user_id=profile.user_id, role=profile.role, status=profile.status)

    return resolve, loaded.config.internal_auth_secret


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    from caal.company.config import ROLE_READER, CompanyConfig, CompanyConfigError
    from caal.company.mcp_server import build_app
    from caal.company.principal import CompanyAuthorizer
    from caal.company.service import CompanyLibrary
    from caal.internal_auth import SqliteNonceStore

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    try:
        config = CompanyConfig.from_env()
    except CompanyConfigError as exc:
        print(f"company-mcp: {exc}", file=sys.stderr)
        return 2
    if not config.mcp_token:
        print("company-mcp: CAAL_COMPANY_MCP_TOKEN is not configured", file=sys.stderr)
        return 2
    if not config.owner_user_id:
        print(
            "company-mcp: CAAL_COMPANY_OWNER_USER_ID is not provisioned; the library has "
            "no owner and will not be served",
            file=sys.stderr,
        )
        return 2

    resolve, secret = _user_resolver(config)
    if not config.internal_auth_secret:
        # The identity secret is the same one the backend validated; taking it
        # from there keeps one source of truth instead of a second env var.
        config = replace(config, internal_auth_secret=secret)

    # The provisioned owner has to be a real, active administrator of this
    # deployment before a single byte is served. A configured id that is not
    # one is an operator mistake, and it fails here rather than at the first
    # question somebody asks.
    facts = resolve(config.owner_user_id)
    if facts is None:
        print(
            "company-mcp: the provisioned owner is not a user of this deployment",
            file=sys.stderr,
        )
        return 2
    if facts.status != "active" or facts.role != "admin":
        print(
            "company-mcp: the provisioned owner is not an active administrator",
            file=sys.stderr,
        )
        return 2

    # This process is a reader whatever the environment says. The single
    # writer is the CAAL backend; two writers is the failure this release
    # exists to make impossible.
    library = CompanyLibrary(replace(config, role=ROLE_READER))
    authorizer = CompanyAuthorizer(
        config=config,
        resolve_user=resolve,
        # Shared across processes: a principal spent by one worker cannot be
        # spent again by another.
        # Not the library directory: a reader's library mount is read only, and
        # this is the one thing a reader has to write. See
        # CompanyConfig.state_dir and CAAL_COMPANY_READER_STATE_DIR.
        nonce_store=SqliteNonceStore(config.state_dir / "nonces.sqlite3"),
        owner_id=library.owner_id,
    )
    app = build_app(library=library, authorizer=authorizer, config=config)
    logger.info("The company library MCP service is starting on loopback, read only")
    try:
        uvicorn.run(
            app,
            host=config.mcp_host,
            port=config.mcp_port,
            log_level="info",
            access_log=False,
            # Nothing but the local runtime speaks to this service.
            forwarded_allow_ips=[],
        )
    finally:
        # A reader's close writes nothing: see CompanyStore.close.
        library.close()
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main(sys.argv[1:]))
