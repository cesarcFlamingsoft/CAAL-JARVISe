"""Restricted, independently enrolled Jarvis voice satellites."""

from homeassistant.const import Platform
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers import entity_registry as er

from .client import BridgeClient

DOMAIN = "jarvis_satellite"
PLATFORMS = [Platform.CONVERSATION, Platform.TTS]


async def async_setup_entry(hass, entry):
    client = BridgeClient(entry.data["backend"], entry.data["credential"])
    try:
        identity = await client.identity()
        verify_registry(hass, identity)
        if entry.unique_id is not None and entry.unique_id != identity["satellite_id"]:
            raise ValueError("Satellite identity changed")
    except Exception:
        await client.close()
        raise ConfigEntryNotReady("Enrolled satellite binding or backend unavailable") from None
    entry.runtime_data = client
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass, entry):
    if await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        await entry.runtime_data.close()
        return True
    return False


def verify_registry(hass, identity):
    from homeassistant.helpers import device_registry as dr

    satellite = er.async_get(hass).async_get(identity["satellite_id"])
    device = dr.async_get(hass).async_get(identity["device_id"])
    if (
        satellite is None
        or satellite.device_id != identity["device_id"]
        or satellite.disabled_by
        or not satellite.config_entry_id
        or device is None
        or device.disabled_by
    ):
        raise ValueError("Enrolled satellite registry binding unavailable")
    return satellite


async def async_migrate_entry(hass, entry):
    """Preserve the working entry and credential; bind identity from backend at setup."""
    if entry.version > 2:
        return False
    if entry.version == 1:
        hass.config_entries.async_update_entry(entry, version=2)
    return True
