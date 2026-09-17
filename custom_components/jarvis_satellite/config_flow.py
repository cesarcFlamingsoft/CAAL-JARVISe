"""HA administrator setup using a separately issued FRIDAY service credential."""

import aiohttp
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.helpers import selector

from . import DOMAIN, verify_registry
from .client import BridgeClient


class JarvisSatelliteConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    VERSION = 2

    async def async_step_user(self, user_input=None):
        errors = {}
        if user_input is not None:
            client = None
            try:
                client = BridgeClient(user_input["backend"], user_input["credential"])
                identity = await client.identity()
                satellite = verify_registry(self.hass, identity)
                await self.async_set_unique_id(identity["satellite_id"])
                self._abort_if_unique_id_configured()
                return self.async_create_entry(
                    title="FRIDAY "
                    + (satellite.name or satellite.original_name or identity["satellite_id"]),
                    data=user_input,
                )
            except (ValueError, OSError, TimeoutError, aiohttp.ClientError):
                errors["base"] = "cannot_connect"
            finally:
                if client:
                    await client.close()
        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required("backend"): str,
                    vol.Required("credential"): selector.TextSelector(
                        selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)
                    ),
                }
            ),
            errors=errors,
        )

    async def async_step_reconfigure(self, user_input=None):
        """Validate a rotated credential and reload only this integration entry."""
        errors = {}
        if user_input is not None:
            client = None
            try:
                client = BridgeClient(user_input["backend"], user_input["credential"])
                identity = await client.identity()
                verify_registry(self.hass, identity)
                if identity["satellite_id"] != self._get_reconfigure_entry().unique_id:
                    raise ValueError("Cannot move an enrollment to another device")
                return self.async_update_reload_and_abort(
                    self._get_reconfigure_entry(), data=user_input
                )
            except (ValueError, OSError, TimeoutError, aiohttp.ClientError):
                errors["base"] = "cannot_connect"
            finally:
                if client:
                    await client.close()
        return self.async_show_form(
            step_id="reconfigure",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        "backend", default=self._get_reconfigure_entry().data["backend"]
                    ): str,
                    vol.Required("credential"): selector.TextSelector(
                        selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)
                    ),
                }
            ),
            errors=errors,
        )
