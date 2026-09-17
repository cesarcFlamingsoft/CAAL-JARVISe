"""The process-wide company library objects, built once and only if configured.

Two consumers, two objects:

* the **management plane** (the authenticated admin API) wants the
  :class:`~caal.company.service.CompanyLibrary` itself, in process, because it
  writes. There is exactly one process in the deployment that may do this --
  the one configured ``CAAL_COMPANY_ROLE=owner`` -- and the store takes an
  exclusive lock to make that true rather than merely intended;
* the **query plane** (the model's native tools) wants a
  :class:`~caal.company.client.CompanyMcpClient`, because it must go through
  the real MCP service like any other client would.

Both are lazy and both fail closed: a deployment without
``CAAL_COMPANY_LIBRARY_KEYS`` or without ``CAAL_COMPANY_OWNER_USER_ID`` has no
usable company library, the tools say so honestly, and nothing else changes. A
configuration error is logged once, by shape, and never with its values.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["get_client", "get_config", "get_library", "is_configured", "reset"]

_lock = threading.Lock()
_config: Any | None = None
_library: Any | None = None
_client: Any | None = None
_attempted = False


def reset() -> None:
    """Forget the cached objects (tests, and a reload of the configuration)."""
    global _config, _library, _client, _attempted
    with _lock:
        if _library is not None:
            try:
                _library.close()
            except Exception:  # noqa: BLE001 - a close failure must not wedge a reset
                logger.error("The company library could not be closed cleanly")
        _config = _library = _client = None
        _attempted = False


def get_config() -> Any | None:
    """The company configuration, or ``None`` when this deployment has none."""
    global _config, _attempted
    with _lock:
        if _config is not None or _attempted:
            return _config
        _attempted = True
        from .config import CompanyConfig, CompanyConfigError

        try:
            _config = CompanyConfig.from_env()
        except CompanyConfigError as exc:
            logger.info("The company library is not configured (%s)", type(exc).__name__)
            _config = None
        return _config


def is_configured() -> bool:
    return get_config() is not None


def get_library() -> Any | None:
    """The in-process library. Only the authenticated management plane uses this.

    ``None`` when the deployment has no company configuration, when another
    process already holds the writer lock, or when the library on disk is
    bound to a different provisioned owner than this configuration names.
    Every one of those is a refusal, never a second mutable copy.
    """
    global _library
    config = get_config()
    if config is None:
        return None
    with _lock:
        if _library is None:
            from .service import CompanyLibrary

            try:
                _library = CompanyLibrary(config)
            except Exception as exc:  # noqa: BLE001 - a bad snapshot must not crash the backend
                logger.error("The company library could not be opened: %s", type(exc).__name__)
                return None
        return _library


def get_client() -> Any | None:
    """The MCP client the model's tools use. ``None`` when nothing is configured."""
    global _client
    config = get_config()
    if config is None or not config.mcp_token or not config.internal_auth_secret:
        return None
    with _lock:
        if _client is None:
            from .client import CompanyMcpClient

            _client = CompanyMcpClient(config=config)
        return _client
