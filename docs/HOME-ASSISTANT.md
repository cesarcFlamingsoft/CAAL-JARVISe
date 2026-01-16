# Home Assistant Integration

CAAL integrates with Home Assistant via MCP (Model Context Protocol) using simplified wrapper tools that provide a consistent interface for voice control.

## Quick Start

1. Enable Home Assistant in the setup wizard or settings
2. Enter your Home Assistant URL (e.g., `http://homeassistant.local:8123`)
3. Add a Long-Lived Access Token from HA (Settings → Security → Long-lived access tokens)

## How It Works

CAAL connects to Home Assistant's MCP server but exposes only two simplified tools to the LLM:

| Wrapper Tool | Purpose |
|--------------|---------|
| `hass_control(action, target, value)` | Control devices |
| `hass_get_state(target)` | Get device status |

This simplification (from 15 raw MCP tools to 2 wrapper tools) dramatically improves LLM tool-calling reliability.

## hass_control

Control Home Assistant devices with a simple action/target interface.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `action` | string | Yes | The action to perform (see table below) |
| `target` | string | Yes | Device name (e.g., "office lamp", "apple tv") |
| `value` | integer | No | Value for `set_volume` (0-100) |

### Supported Actions

| Action | HASS MCP Tool | Description |
|--------|---------------|-------------|
| `turn_on` | HassTurnOn | Turn on a device/switch |
| `turn_off` | HassTurnOff | Turn off a device/switch |
| `pause` | HassMediaPause | Pause media playback |
| `play` | HassMediaUnpause | Resume media playback |
| `next` | HassMediaNext | Skip to next track |
| `previous` | HassMediaPrevious | Go to previous track |
| `volume_up` | HassSetVolumeRelative | Increase volume |
| `volume_down` | HassSetVolumeRelative | Decrease volume |
| `set_volume` | HassSetVolume | Set volume to specific level (requires `value`) |
| `mute` | HassMediaPlayerMute | Mute audio |
| `unmute` | HassMediaPlayerUnmute | Unmute audio |

### Examples

```
"Turn on the office lamp"
→ hass_control(action="turn_on", target="office lamp")

"Pause the Apple TV"
→ hass_control(action="pause", target="apple tv")

"Set the soundbar volume to 30"
→ hass_control(action="set_volume", target="soundbar", value=30)

"Mute the living room speaker"
→ hass_control(action="mute", target="living room speaker")
```

## hass_get_state

Get the current state of Home Assistant devices.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `target` | string | No | Device name to filter (omit for all devices) |

### Examples

```
"What's the status of the garage door?"
→ hass_get_state(target="garage door")

"What devices are on?"
→ hass_get_state()
```

## Prompt Configuration

The default prompt (`prompt/default.md`) includes instructions for using these tools:

```markdown
# Home Control (hass_control)

Control devices with: `hass_control(action, target, value)`
- **action**: turn_on, turn_off, volume_up, volume_down, set_volume, mute, unmute, pause, play, next, previous
- **target**: Device name like "office lamp" or "apple tv"
- **value**: Only for set_volume (0-100)

Examples:
- "turn on the office lamp" → `hass_control(action="turn_on", target="office lamp")`
- "set apple tv volume to 50" → `hass_control(action="set_volume", target="apple tv", value=50)`

Act immediately - don't ask for confirmation. Confirm AFTER the action completes.
```

## Advanced: Raw MCP Tools

For power users who need full access to all 15 HASS MCP tools:

1. Add Home Assistant manually via `mcp_servers.json`:

```json
{
  "servers": [
    {
      "name": "hass_raw",
      "url": "http://homeassistant.local:8123/api/mcp",
      "token": "your-long-lived-token",
      "transport": "streamable_http"
    }
  ]
}
```

2. Create a custom prompt (`prompt/custom.md`) with instructions for the full tool set

Note: The wrapper tools will still be available alongside raw tools when using wizard-configured HASS.

## Troubleshooting

### "Home Assistant is not connected"

- Check that HASS URL is reachable from the CAAL container
- Verify the Long-Lived Access Token is valid
- Check HASS logs for MCP connection errors

### Device not found

- Device names must match exactly as shown in Home Assistant
- Use `hass_get_state()` to see available devices and their names
- Names are case-insensitive

### Action not working

- Ensure the device supports the action (e.g., lights don't support `pause`)
- Check Home Assistant for device-specific requirements
