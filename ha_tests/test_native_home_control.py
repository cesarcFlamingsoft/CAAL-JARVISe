"""Real HA REST/context and virtual lights only; no network to production HA."""

import pytest
from homeassistant.const import EVENT_CALL_SERVICE
from homeassistant.setup import async_setup_component


@pytest.mark.asyncio
async def test_native_light_rest_records_authenticated_user(hass, hass_client, hass_admin_user):
    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(hass, "api", {})
    from homeassistant.components.light import DATA_COMPONENT, ColorMode, LightEntity

    assert await async_setup_component(hass, "light", {})

    class VirtualLight(LightEntity):
        _attr_name = "Isolated test light"
        _attr_unique_id = "isolated-test-light"
        _attr_is_on = False
        _attr_supported_color_modes = {ColorMode.ONOFF}
        _attr_color_mode = ColorMode.ONOFF

        async def async_turn_on(self, **kwargs):
            self._attr_is_on = True
            self.async_write_ha_state()

        async def async_turn_off(self, **kwargs):
            self._attr_is_on = False
            self.async_write_ha_state()

    await hass.data[DATA_COMPONENT].async_add_entities([VirtualLight()])
    await hass.async_block_till_done()
    lights = hass.states.async_all("light")
    assert lights
    target = lights[0].entity_id
    calls = []
    hass.bus.async_listen(EVENT_CALL_SERVICE, lambda event: calls.append(event))
    client = await hass_client()
    response = await client.get("/api/states")
    assert response.status == 200
    assert any(row["entity_id"] == target for row in await response.json())
    response = await client.post("/api/services/light/turn_on", json={"entity_id": [target]})
    assert response.status == 200
    await hass.async_block_till_done()
    assert hass.states.get(target).state == "on"
    request = next(
        e for e in calls if e.data.get("domain") == "light" and e.data.get("service") == "turn_on"
    )
    assert request.context.user_id == hass_admin_user.id
    response = await client.post("/api/services/light/turn_off", json={"entity_id": [target]})
    assert response.status == 200
    await hass.async_block_till_done()
    assert hass.states.get(target).state == "off"
