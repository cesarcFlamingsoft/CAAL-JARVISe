from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_livekit_templates_advertise_external_single_port_ice():
    for name in ("livekit.yaml", "livekit-tailscale.yaml.template"):
        template = (ROOT / name).read_text(encoding="utf-8")
        assert "use_external_ip: true" in template
        assert "udp_port: 7881" in template
