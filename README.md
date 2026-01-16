# CAAL

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LiveKit](https://img.shields.io/badge/LiveKit-Agents-purple.svg)](https://docs.livekit.io/agents/)

> **Local voice assistant with n8n workflow integrations and Home Assistant control**

Built on [LiveKit Agents](https://docs.livekit.io/agents/). Runs fully local with [Ollama](https://ollama.ai/) + [Speaches](https://github.com/speaches-ai/speaches) + [Kokoro](https://github.com/remsky/Kokoro-FastAPI), or GPU-free with [Groq](https://groq.com/) + [Piper](https://github.com/rhasspy/piper).

![CAAL Voice Assistant](frontend/.github/assets/readme-hero.webp)

## Features

- **First-Start Wizard** - Configure everything from the browser, only one edit in `.env` required
- **Flexible Providers** - Ollama (local) or Groq (cloud) for LLM/STT, Kokoro or Piper for TTS
- **Home Assistant** - Native MCP integration with simplified `hass_control` and `hass_get_state` tools
- **n8n Workflows** - Expandable LLM tool capability - any n8n workflow can become a tool for CAAL
- **Wake Word Detection** - "Hey Cal" activation via OpenWakeWord (server-side)
- **Web Search** - DuckDuckGo integration for real-time information
- **Webhook API** - External triggers for announcements and tool reload
- **Mobile App** - Flutter client for Android and iOS

## Quick Start

```bash
git clone https://github.com/CoreWorxLab/caal.git
cd caal
cp .env.example .env
nano .env  # Set CAAL_HOST_IP to your server's LAN IP

# GPU mode (Ollama + Kokoro)
docker compose up -d

# CPU-only mode (Groq + Piper) - no GPU required
docker compose -f docker-compose.cpu.yaml up -d
```

Open `http://YOUR_SERVER_IP:3000` and complete the setup wizard.

| Mode | Hardware | Command |
|------|----------|---------|
| **GPU** | Linux + NVIDIA GPU | `docker compose up -d` |
| **CPU-only** | Any Docker host | `docker compose -f docker-compose.cpu.yaml up -d` |
| **Apple Silicon** | M1/M2/M3/M4 Mac | [docs/APPLE-SILICON.md](docs/APPLE-SILICON.md) |
| **Distributed** | GPU Server + Mac | [docs/DISTRIBUTED-DEPLOYMENT.md](docs/DISTRIBUTED-DEPLOYMENT.md) |

---

## GPU Mode (NVIDIA Linux)

Full local stack with GPU-accelerated STT (Speaches), LLM (Ollama) and TTS (Kokoro).

### Requirements

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- 12GB+ VRAM recommended

### Installation

```bash
git clone https://github.com/CoreWorxLab/caal.git
cd caal
cp .env.example .env
nano .env  # Set CAAL_HOST_IP

docker compose up -d
```

The setup wizard will guide you through LLM (Ollama), TTS, and integration configuration.

---

## CPU-Only Mode (No GPU)

Run CAAL without a GPU using Groq for LLM/STT and Piper (CPU) for TTS.

```bash
docker compose -f docker-compose.cpu.yaml up -d
```

For HTTPS:
```bash
docker compose -f docker-compose.cpu.yaml --profile https up -d
```

In the setup wizard:
1. Select **Groq** as LLM provider and enter your [free API key](https://console.groq.com/)
2. Select **Piper** as TTS provider (models download automatically)

> **Note:** Voice data is sent to Groq's API. For fully local operation, use GPU mode with Ollama.

---

## Apple Silicon (macOS)

CAAL runs on Apple Silicon Macs using [mlx-audio](https://github.com/Blaizzy/mlx-audio) for Metal-accelerated STT/TTS.

```bash
./start-apple.sh
```

**See [docs/APPLE-SILICON.md](docs/APPLE-SILICON.md) for full setup instructions.**

---

## Distributed Deployment

Run the GPU-intensive backend on a Linux server while using the frontend on a Mac or another device.

**See [docs/DISTRIBUTED-DEPLOYMENT.md](docs/DISTRIBUTED-DEPLOYMENT.md) for full setup instructions.**

---

## Network Modes

CAAL supports three network configurations:

| Mode          | Voice From        | Access URL                                 | Command                                |
| ------------- | ----------------- | ------------------------------------------ | -------------------------------------- |
| **LAN HTTP**  | Host machine only | `http://localhost:3000`                    | `docker compose up -d`                 |
| **LAN HTTPS** | Any LAN device    | `https://192.168.1.100:3443`               | `docker compose --profile https up -d` |
| **Tailscale** | Anywhere          | `https://your-machine.tailnet.ts.net:3443` | `docker compose --profile https up -d` |

> **Why?** Browsers block microphone access on HTTP except from localhost. HTTPS is required for voice from other devices.
>
> **Note:** For utilization with mobile app as the client, only LAN HTTP is required, not HTTPS

### LAN HTTP (Default)

```bash
CAAL_HOST_IP=192.168.1.100  # Set in .env
docker compose up -d
```

### LAN HTTPS

Self-signed certificates are auto-generated if none exist in `./certs/`.

```bash
# Configure .env
CAAL_HOST_IP=192.168.1.100
HTTPS_DOMAIN=192.168.1.100

# Start with HTTPS profile (certs auto-generated)
docker compose --profile https up -d
```

Access: `https://192.168.1.100:3443`

> **Trusted certs:** For browser-trusted certs without warnings, use [mkcert](https://github.com/FiloSottile/mkcert):
> ```bash
> mkcert -install && mkcert 192.168.1.100
> mkdir -p certs && mv 192.168.1.100.pem certs/server.crt && mv 192.168.1.100-key.pem certs/server.key
> ```

### Tailscale (Remote Access)

```bash
# Generate Tailscale certs
tailscale cert your-machine.tailnet.ts.net
mkdir -p certs && mv your-machine.tailnet.ts.net.crt certs/server.crt && mv your-machine.tailnet.ts.net.key certs/server.key

# Configure .env
CAAL_HOST_IP=100.x.x.x                         # tailscale ip -4
HTTPS_DOMAIN=your-machine.tailnet.ts.net

# Start
docker compose --profile https up -d
```

Access: `https://your-machine.tailnet.ts.net:3443`

---

## Configuration

### Environment Variables

Only `CAAL_HOST_IP` is required. Everything else is configured via the web UI.

| Variable | Description | Required |
|----------|-------------|----------|
| `CAAL_HOST_IP` | Your server's LAN/Tailscale IP | Yes |
| `HTTPS_DOMAIN` | Domain for HTTPS mode | No |

See `.env.example` for additional options (ports, default models).

### Settings Panel

After setup, click the gear icon to access the settings panel:

- **Agent** - Agent name, voice selection, wake greetings
- **Prompt** - Default or custom system prompt
- **Providers** - LLM provider (Ollama/Groq), TTS provider (Kokoro/Piper)
- **LLM Settings** - Temperature, context size, max turns, turn detection settings
- **Integrations** - Home Assistant and n8n connection configuration
- **Wake Word** - Enable/disable, model selection, threshold, timeout

---

## Integrations

### Home Assistant

Control your smart home with voice commands. CAAL exposes two simplified tools:

- `hass_control(action, target, value)` - Control devices
  - Actions: `turn_on`, `turn_off`, `set_volume`, `volume_up`, `volume_down`, `mute`, `unmute`, `pause`, `play`, `next`, `previous`
  - Value: 0-100 for `set_volume`
- `hass_get_state(target)` - Query device states

**Setup:**
1. Create a [Long-Lived Access Token](https://www.home-assistant.io/docs/authentication/#your-account-profile) in Home Assistant
2. In CAAL settings, enable Home Assistant and enter your host URL and token
3. Restart the agent - CAAL auto-discovers your devices

See [docs/HOME-ASSISTANT.md](docs/HOME-ASSISTANT.md) for action mappings and examples.

### n8n Workflows

Extend CAAL with any API, database, or service via n8n workflows exposed through MCP.

**Setup n8n:**
1. Enable MCP: **Settings > MCP Access > Enable MCP**
2. Set connection method to **Access Token** and copy the token
3. In CAAL settings, enable n8n and enter your MCP URL and token

**Import example workflows:**
```bash
cd n8n-workflows
cp config.env.example config.env
nano config.env  # Set your n8n IP and API key
python setup.py  # Creates all workflows
```

See [docs/N8N-WORKFLOWS.md](docs/N8N-WORKFLOWS.md) for how to create your own workflows.

### Wake Word Detection

Enable "Hey Cal" wake word in the settings panel. Two options:

- **OpenWakeWord (Server-side)** - Runs on the server, works with any client
- **Picovoice (Client-side)** - Requires access key and trained model per device

---

## Webhook API

The agent exposes a REST API on port 8889 for external integrations.

**Core Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/announce` | POST | Make CAAL speak a message |
| `/wake` | POST | Trigger wake word greeting |
| `/reload-tools` | POST | Refresh MCP tool cache |
| `/health` | GET | Health check |

**Settings & Configuration:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/settings` | GET/POST | Read/update settings |
| `/prompt` | GET/POST | Read/update system prompt |
| `/voices` | GET | List available TTS voices |
| `/models` | GET | List available Ollama models |

**Wake Word:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/wake-word/status` | GET | Get wake word status |
| `/wake-word/enable` | POST | Enable wake word detection |
| `/wake-word/disable` | POST | Disable wake word detection |
| `/wake-word/models` | GET | List available wake word models |

```bash
curl -X POST http://localhost:8889/announce \
  -H "Content-Type: application/json" \
  -d '{"message": "Package delivered"}'
```

---

## Mobile App

Android app available from [GitHub Releases](https://github.com/CoreWorxLab/caal/releases). Download the APK and install on your device.

**Building from source:**
```bash
cd mobile
flutter pub get
flutter build apk
```

See [mobile/README.md](mobile/README.md) for full documentation.

---

## Development

```bash
# Install dependencies
uv sync

# Start infrastructure
docker compose up -d livekit speaches kokoro

# Run agent locally
uv run voice_agent.py dev

# Run frontend locally
cd frontend && pnpm install && pnpm dev
```

**Commands:**
```bash
uv run ruff check src/   # Lint
uv run mypy src/         # Type check
uv run pytest            # Test
```

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│  Docker Compose Stack                                                 │
│                                                                       │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐       │
│  │  Frontend  │  │  LiveKit   │  │  Speaches  │  │Kokoro/Piper│       │
│  │  (Next.js) │  │   Server   │  │(STT, GPU)  │  │  (TTS)     │       │
│  │   :3000    │  │   :7880    │  │   :8000    │  │   :8880    │       │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘       │
│        │               │               │               │              │
│        │               └───────────────┼───────────────┘              │
│        └───────────────────────┐       │                              │
│                                │       │                              │
│                          ┌─────┴───────┴─────┐                        │
│                          │       Agent       │                        │
│                          │  (Voice Pipeline) │                        │
│                          │  :8889 (webhooks) │                        │
│                          └─────────┬─────────┘                        │
│                                    │                                  │
└────────────────────────────────────┼──────────────────────────────────┘
                                     │
            ┌────────────────────────┼────────────────────────┐
            │                        │                        │
      ┌─────┴─────┐           ┌──────┴──────┐          ┌──────┴──────┐
      │Ollama/Groq│           │     n8n     │          │    Home     │
      │   (LLM)   │           │  Workflows  │          │  Assistant  │
      └───────────┘           └─────────────┘          └─────────────┘
                       External Services (via MCP)
```

---

## Troubleshooting

### WebRTC Not Connecting

1. Check `CAAL_HOST_IP` matches your network mode
2. Verify firewall ports: 3000, 7880, 7881, 50000-50100 (UDP)
3. Check logs: `docker compose logs livekit | grep -i "ice\|error"`

### Ollama Connection Failed

```bash
# Ensure Ollama binds to network
OLLAMA_HOST=0.0.0.0 ollama serve

# From Docker, use host.docker.internal
OLLAMA_HOST=http://host.docker.internal:11434
```

### First Start Is Slow

Normal - models download on first run (~2-5 minutes):
```bash
docker compose logs -f speaches kokoro
```

### Integration Connection Errors

If Home Assistant or n8n fail to connect, you'll see a toast notification with the error. Check:
- Host URL is reachable from the Docker container
- Access token is valid and has correct permissions
- For n8n: MCP Access is enabled in Settings

---

## Related Projects

- [LiveKit Agents](https://github.com/livekit/agents) - Voice agent framework
- [Speaches](https://github.com/speaches-ai/speaches) - Faster-Whisper STT server (also includes Piper TTS)
- [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI) - Kokoro TTS server
- [Piper](https://github.com/rhasspy/piper) - Fast local TTS (CPU-friendly)
- [mlx-audio](https://github.com/Blaizzy/mlx-audio) - STT/TTS for Apple Silicon
- [Ollama](https://ollama.ai/) - Local LLM server
- [Groq](https://groq.com/) - Fast cloud LLM inference (free tier available)
- [n8n](https://n8n.io/) - Workflow automation
- [Home Assistant](https://www.home-assistant.io/) - Smart home platform

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.
