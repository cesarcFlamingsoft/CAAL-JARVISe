"""The LiveKit runtime configuration, and why SIP silently stopped dialing.

A LiveKit-only ``docker compose up -d livekit`` run from the base compose file
recreated ``caal-livekit`` without the telephony overlay. The overlay was the
only thing that mounted a Redis-backed LiveKit configuration, so the container
came back with keys and rtc but no ``redis`` section, and every outbound call
died before the dial with ``sip not connected (redis required)`` -- while
``caal-sip`` and ``caal-redis`` stayed up and healthy, so nothing looked broken.

The configuration is now rendered by one script, from the same templates in
every composition, and the Redis section is driven by ``LIVEKIT_REDIS_ADDRESS``
rather than by which file happened to be mounted. Pinned properties:

* a telephony composition renders a config with Redis in it, in HTTPS mode as
  well as LAN mode -- SIP works and the public TURN/TLS deployment is intact;
* a base, non-telephony composition renders no Redis at all: SIP is never
  enabled by accident, and a single-node deployment keeps its behaviour;
* the browser media port range the container publishes and the range LiveKit
  advertises in its ICE candidates stay identical (50100-50200);
* the telephony overlay owns the Redis address and the SIP service, and the
  base compose only passes through whatever the operator set.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]
RENDER_SCRIPT = ROOT / "livekit-render-config.sh"
LAN_TEMPLATE = ROOT / "livekit.yaml"
TLS_TEMPLATE = ROOT / "livekit-tailscale.yaml.template"
BASE_COMPOSE = ROOT / "docker-compose.yaml"
APPLE_COMPOSE = ROOT / "docker-compose.apple.yaml"
TELEPHONY_COMPOSE = ROOT / "docker-compose.telephony.yaml"

MEDIA_PORTS = "50100-50200:50100-50200/udp"
KEY = "test-api-key"
SECRET = "test-api-secret"

pytestmark = pytest.mark.skipif(
    shutil.which("envsubst") is None, reason="envsubst renders the LiveKit templates"
)


def _render(tmp_path: Path, **environment: str) -> dict:
    """Run the container own render script the way the container runs it."""
    out = tmp_path / "livekit.yaml"
    env = dict(os.environ)
    env.pop("HTTPS_DOMAIN", None)
    env.pop("LIVEKIT_REDIS_ADDRESS", None)
    env.update(
        LIVEKIT_API_KEY=KEY,
        LIVEKIT_API_SECRET=SECRET,
        CAAL_HOST_IP="10.0.0.64",
        LIVEKIT_LAN_TEMPLATE=str(LAN_TEMPLATE),
        LIVEKIT_TLS_TEMPLATE=str(TLS_TEMPLATE),
    )
    env.update(environment)
    done = subprocess.run(
        ["/bin/sh", str(RENDER_SCRIPT), str(out)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return yaml.safe_load(out.read_text())


def _as_map(values: object) -> dict[str, str]:
    """An environment block as a map, whether it was written as a list or a map."""
    if isinstance(values, dict):
        return {str(name): str(value) for name, value in values.items()}
    if isinstance(values, list):
        pairs = [str(item).split("=", 1) for item in values]
        return {pair[0]: (pair[1] if len(pair) > 1 else "") for pair in pairs}
    return {}


def _merge_service(into: dict, override: dict) -> dict:
    """One service merged the way ``docker compose -f a -f b`` merges it."""
    for key, value in override.items():
        if key == "environment":
            into[key] = {**_as_map(into.get(key)), **_as_map(value)}
        elif key in ("ports", "volumes") and isinstance(value, list):
            kept = list(into.get(key) or [])
            into[key] = kept + [item for item in value if item not in kept]
        else:
            into[key] = value
    return into


def _compose(*paths: Path) -> dict:
    merged: dict = {}
    for path in paths:
        document = yaml.safe_load(path.read_text())
        for section, value in document.items():
            if section == "services":
                for name, service in value.items():
                    target = merged.setdefault("services", {}).setdefault(name, {})
                    _merge_service(target, service)
            else:
                merged[section] = value
    return merged


def _environment(service: dict) -> dict[str, str]:
    return _as_map(service.get("environment"))


# --- what the container renders ---------------------------------------------------------


def test_a_telephony_composition_renders_a_redis_backed_livekit_config(tmp_path) -> None:
    config = _render(tmp_path, LIVEKIT_REDIS_ADDRESS="redis:6379")

    assert config["redis"]["address"] == "redis:6379", "SIP requires LiveKit Redis deployment"
    assert config["keys"] == {KEY: SECRET}


def test_a_telephony_composition_keeps_the_public_https_deployment(tmp_path) -> None:
    """Redis is added to the HTTPS configuration, not swapped in place of it."""
    config = _render(
        tmp_path, LIVEKIT_REDIS_ADDRESS="redis:6379", HTTPS_DOMAIN="jarvis.example.com"
    )

    assert config["redis"]["address"] == "redis:6379"
    assert config["turn"]["enabled"] is True
    assert config["turn"]["domain"] == "jarvis.example.com"
    assert config["keys"] == {KEY: SECRET}
    assert (config["rtc"]["port_range_start"], config["rtc"]["port_range_end"]) == (50100, 50200)


def test_a_base_composition_never_forces_redis(tmp_path) -> None:
    lan = _render(tmp_path)
    https = _render(tmp_path, HTTPS_DOMAIN="jarvis.example.com")

    assert "redis" not in lan, "a base deployment must not be given a Redis dependency"
    assert "redis" not in https
    assert lan["keys"] == {KEY: SECRET}


def test_the_rendered_media_ports_match_the_published_ones(tmp_path) -> None:
    """LiveKit advertises these ports itself; Docker does not translate them."""
    for environment in ({}, dict(HTTPS_DOMAIN="jarvis.example.com")):
        config = _render(tmp_path, **environment)
        rtc = config["rtc"]
        assert (rtc["port_range_start"], rtc["port_range_end"]) == (50100, 50200), environment


# --- what each composition declares -----------------------------------------------------


@pytest.mark.parametrize("path", [BASE_COMPOSE, APPLE_COMPOSE])
def test_every_livekit_service_renders_through_the_one_script(path: Path) -> None:
    livekit = _compose(path)["services"]["livekit"]
    command = " ".join(livekit["command"])
    mounts = " ".join(livekit["volumes"])

    assert "livekit-render-config.sh" in mounts, "the render script is mounted, not inlined"
    assert "livekit-render-config.sh" in command
    assert MEDIA_PORTS in livekit["ports"]
    assert "LIVEKIT_REDIS_ADDRESS" in _environment(livekit), (
        "the operator Redis address must survive a livekit-only recreate"
    )


@pytest.mark.parametrize("path", [BASE_COMPOSE, APPLE_COMPOSE])
def test_a_base_composition_defines_neither_sip_nor_redis(path: Path) -> None:
    services = _compose(path)["services"]

    assert "sip" not in services and "redis" not in services


def test_the_telephony_overlay_owns_redis_and_sip() -> None:
    merged = _compose(APPLE_COMPOSE, TELEPHONY_COMPOSE)
    services = merged["services"]
    livekit = services["livekit"]
    values = _environment(livekit)

    assert "redis" in services and "sip" in services
    assert "redis" in values["LIVEKIT_REDIS_ADDRESS"]
    assert "redis" in livekit["depends_on"]
    assert values.get("HTTPS_DOMAIN", "") != "", (
        "the overlay must not disable the public HTTPS deployment to get Redis"
    )
    assert MEDIA_PORTS in livekit["ports"], "browser media ports stay aligned under telephony"


def test_the_telephony_overlay_does_not_mount_a_second_livekit_config() -> None:
    """One template per mode, in every composition: no per-overlay config file."""
    livekit = yaml.safe_load(TELEPHONY_COMPOSE.read_text())["services"]["livekit"]

    for mount in livekit.get("volumes", []):
        assert "livekit-lan.yaml" not in mount, "the overlay must not swap the base template"
