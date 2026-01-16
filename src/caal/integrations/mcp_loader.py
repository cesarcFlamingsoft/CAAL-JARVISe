"""MCP Server Configuration Loader

Loads MCP server definitions from settings, environment variables, and optional JSON config file.
Settings take priority, then env vars, then JSON file.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx
from livekit.agents import mcp

# Timeout for pre-flight connection test (seconds)
# This validates connectivity before calling MCP initialize() which can hang
CONNECTION_TEST_TIMEOUT = 3.0

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server."""
    name: str
    url: str
    auth_token: str | None = None
    transport: Literal["sse", "streamable_http"] | None = None
    timeout: float = 10.0


def load_mcp_config(settings: dict[str, Any] | None = None) -> list[MCPServerConfig]:
    """Load MCP server configurations from settings, env vars, and optional JSON file.

    Priority order:
    1. Settings (from frontend/settings.json)
    2. Environment variables
    3. JSON config file (mcp_servers.json)

    Args:
        settings: Optional settings dict. If not provided, attempts to load from settings module.

    Environment variables (fallback):
        N8N_MCP_URL: n8n MCP server URL
        N8N_MCP_TOKEN: Bearer token for n8n (optional)
        N8N_MCP_TIMEOUT: Request timeout in seconds (optional, default 10.0)

    JSON file (mcp_servers.json):
        {
            "servers": [
                {
                    "name": "server_name",
                    "url": "http://...",
                    "token": "optional_token",
                    "transport": "sse" | "streamable_http",
                    "timeout": 10.0
                }
            ]
        }

    Returns:
        List of MCPServerConfig objects
    """
    servers = []

    # Try to load settings if not provided
    if settings is None:
        try:
            from .. import settings as settings_module
            settings = settings_module.load_settings()
        except Exception:
            settings = {}

    # 1. Home Assistant MCP Server - settings only
    hass_enabled = settings.get("hass_enabled", False)
    if hass_enabled:
        hass_host = settings.get("hass_host")
        hass_token = settings.get("hass_token")
        if hass_host:
            # Build MCP URL from host
            hass_mcp_url = f"{hass_host.rstrip('/')}/api/mcp"
            servers.append(MCPServerConfig(
                name="home_assistant",
                url=hass_mcp_url,
                auth_token=hass_token,
                transport="streamable_http",  # HASS MCP uses Streamable HTTP
                timeout=10.0,
            ))
            logger.debug(f"Loaded MCP server config: home_assistant ({hass_mcp_url})")
        else:
            logger.warning("Home Assistant enabled but no host configured")
    else:
        logger.info("Home Assistant not configured - HASS MCP tools will not be available")

    # 2. n8n MCP Server - settings first, then env vars
    # Check if explicitly disabled in settings
    n8n_enabled = settings.get("n8n_enabled")

    # If n8n_enabled is explicitly False, don't load n8n
    if n8n_enabled is False:
        logger.info("n8n disabled in settings - n8n MCP tools will not be available")
        n8n_url = None
        n8n_token = None
    else:
        # Try settings first
        n8n_url = settings.get("n8n_url") if n8n_enabled else None
        n8n_token = settings.get("n8n_token") if n8n_enabled else None

        # Fall back to env vars if settings not configured (legacy support)
        if not n8n_url:
            n8n_url = os.environ.get("N8N_MCP_URL")
            n8n_token = os.environ.get("N8N_MCP_TOKEN")

    if n8n_url:
        servers.append(MCPServerConfig(
            name="n8n",
            url=n8n_url,
            auth_token=n8n_token,
            transport="streamable_http",  # n8n uses /http suffix which needs explicit transport
            timeout=float(os.environ.get("N8N_MCP_TIMEOUT", "10.0")),
        ))
        logger.debug(f"Loaded MCP server config: n8n ({n8n_url})")
    else:
        logger.info("n8n not configured - n8n MCP tools will not be available")

    # 3. Additional MCP servers from JSON config (optional)
    config_path = Path("mcp_servers.json")
    if config_path.exists():
        try:
            with open(config_path) as f:
                data = json.load(f)
                for server in data.get("servers", []):
                    name = server.get("name")
                    url = server.get("url")
                    if not name or not url:
                        logger.warning(f"Skipping MCP server with missing name or url: {server}")
                        continue

                    servers.append(MCPServerConfig(
                        name=name,
                        url=url,
                        auth_token=server.get("token"),
                        transport=server.get("transport"),
                        timeout=server.get("timeout", 10.0),
                    ))
                    logger.debug(f"Loaded MCP server config from JSON: {name} ({url})")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse mcp_servers.json: {e}")
        except Exception as e:
            logger.error(f"Failed to load mcp_servers.json: {e}")

    if not servers:
        logger.warning("No MCP servers configured - no MCP tools will be available")

    return servers


@dataclass
class MCPInitError:
    """Error from MCP server initialization."""

    name: str
    error: str


async def initialize_mcp_servers(
    configs: list[MCPServerConfig],
) -> tuple[dict[str, mcp.MCPServerHTTP], list[MCPInitError]]:
    """Initialize MCP servers from config list.

    Args:
        configs: List of MCPServerConfig objects

    Returns:
        Tuple of (servers_dict, errors_list) where:
        - servers_dict maps server name to initialized MCPServerHTTP instance
        - errors_list contains MCPInitError for any servers that failed
    """
    servers = {}
    errors = []

    for config in configs:
        try:
            headers = {}
            if config.auth_token:
                headers["Authorization"] = f"Bearer {config.auth_token}"

            server = mcp.MCPServerHTTP(
                url=config.url,
                headers=headers if headers else None,
                timeout=config.timeout,
            )

            # Set transport type if specified
            # Newer LiveKit versions use transport_type param, older use private attr
            if config.transport == "streamable_http":
                server._use_streamable_http = True
            elif config.transport == "sse":
                server._use_streamable_http = False
            # If transport not specified, let LiveKit auto-detect from URL

            # Pre-flight connection test with httpx (reliable timeout)
            # This prevents the MCP library from hanging on bad connections
            async with httpx.AsyncClient(timeout=CONNECTION_TEST_TIMEOUT) as client:
                try:
                    resp = await client.post(
                        config.url,
                        headers=headers if headers else None,
                        json={"jsonrpc": "2.0", "method": "ping", "id": 1},
                    )
                    # 401/403 = auth issue, other 4xx/5xx = server issue
                    if resp.status_code == 401:
                        raise httpx.HTTPStatusError(
                            "Unauthorized - check your token",
                            request=resp.request,
                            response=resp,
                        )
                    elif resp.status_code == 403:
                        raise httpx.HTTPStatusError(
                            "Forbidden - check your token permissions",
                            request=resp.request,
                            response=resp,
                        )
                except httpx.TimeoutException:
                    raise TimeoutError(f"Connection timed out after {CONNECTION_TEST_TIMEOUT}s")

            # Initialize MCP server (connection already validated)
            await server.initialize()
            servers[config.name] = server
            logger.info(f"Initialized MCP server: {config.name}")

        except TimeoutError:
            error_msg = f"Connection timed out after {CONNECTION_TEST_TIMEOUT}s"
            logger.error(f"Failed to initialize MCP server {config.name}: {error_msg}")
            errors.append(MCPInitError(name=config.name, error=error_msg))

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to initialize MCP server {config.name}: {error_msg}")
            errors.append(MCPInitError(name=config.name, error=error_msg))

    return servers, errors
