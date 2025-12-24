# CAAL

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LiveKit](https://img.shields.io/badge/LiveKit-Agents-purple.svg)](https://docs.livekit.io/agents/)

**Voice assistant with n8n workflow integrations.**

Built on [LiveKit Agents](https://docs.livekit.io/agents/) with fully local STT/TTS/LLM using [Speaches](https://github.com/speaches-ai/speaches) (Faster-Whisper STT), [Kokoro](https://github.com/remsky/Kokoro-FastAPI) (TTS), and [Ollama](https://ollama.ai/).

## Features

- **Local Voice Pipeline**: Speaches (Faster-Whisper STT) + Kokoro (TTS) + Ollama LLM
- **Wake Word Detection**: "Hey Cal" activation via Picovoice Porcupine
- **n8n Integrations**: Home Assistant, APIs, databases - anything n8n can connect to
- **Web Search**: DuckDuckGo integration for real-time information
- **Webhook API**: External triggers for announcements and tool reload

## Quick Start (Docker)

```bash
# Clone and configure
git clone https://github.com/CoreWorxLab/caal.git
cd caal
cp .env.example .env
nano .env  # Set CAAL_HOST_IP, OLLAMA_HOST, N8N_MCP_URL, N8N_MCP_TOKEN

# Deploy
docker compose up -d
```

Open `http://YOUR_SERVER_IP:3000` from any device on your network.

**Requirements:**
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (for GPU acceleration)
- [Ollama](https://ollama.ai/) running on your network
- [n8n](https://n8n.io/) with MCP enabled (Settings > MCP Access)
- 12GB+ VRAM recommended

## Network Modes

CAAL supports three network configurations:

| Mode | Voice From | Access URL | Command |
|------|------------|------------|---------|
| **LAN HTTP** | Host machine only | `http://localhost:3000` | `docker compose up -d` |
| **LAN HTTPS** | Any LAN device | `https://192.168.1.100` | `docker compose --profile https up -d` |
| **Tailscale** | Anywhere | `https://your-machine.tailnet.ts.net` | `docker compose --profile https up -d` |

> **Why the difference?** Browsers block microphone access on HTTP except from localhost. HTTPS is required for voice from other devices.

### LAN HTTP Mode (Default)

Simplest setup. Voice works from the host machine; other devices can use text chat:

```bash
# Set your LAN IP in .env
CAAL_HOST_IP=192.168.1.100

# Start
docker compose up -d
```

### LAN HTTPS Mode (mkcert)

Full voice from any device on your LAN using locally-trusted certificates:

**1. Install mkcert and generate certs:**
```bash
# Install mkcert (Arch/Manjaro)
sudo pacman -S mkcert

# Install mkcert (Ubuntu/Debian)
sudo apt install mkcert

# Install mkcert (macOS)
brew install mkcert

# Install local CA (one-time, may need browser restart)
mkcert -install

# Generate cert for your LAN IP
mkcert 192.168.1.100

# Move to certs folder with standard names
mkdir -p certs
mv 192.168.1.100.pem certs/server.crt
mv 192.168.1.100-key.pem certs/server.key
```

**2. Configure `.env`:**
```bash
CAAL_HOST_IP=192.168.1.100
HTTPS_DOMAIN=192.168.1.100
```

**3. Set key permissions and rebuild frontend:**
```bash
chmod 644 certs/server.key  # nginx needs read access

# Frontend bakes in wss:// URL at build time - must rebuild
docker compose --profile https build frontend
```

**4. Start with HTTPS profile:**
```bash
docker compose --profile https up -d
```

**5. Access from any LAN device:**
```
https://192.168.1.100
```

> **Note:** Other devices on your LAN need the mkcert CA installed to avoid certificate warnings. Run `mkcert -CAROOT` to find the CA cert, then install it on other devices.

### Tailscale Mode (Remote Access)

Access CAAL from anywhere with HTTPS via [Tailscale](https://tailscale.com/):

**1. Generate Tailscale certificates:**
```bash
# Get your Tailscale hostname
tailscale status | head -1

# Generate certs (replace with your hostname)
tailscale cert your-machine.tailnet.ts.net

# Move certs to project with standard names
mkdir -p certs
mv your-machine.tailnet.ts.net.crt certs/server.crt
mv your-machine.tailnet.ts.net.key certs/server.key
```

**2. Configure `.env`:**
```bash
CAAL_HOST_IP=100.x.x.x                         # Your Tailscale IP (tailscale ip -4)
HTTPS_DOMAIN=your-machine.tailnet.ts.net       # Your Tailscale hostname
```

**3. Rebuild frontend and start:**
```bash
# Frontend bakes in wss:// URL at build time - must rebuild
docker compose --profile https build frontend

# Start all services
docker compose --profile https up -d
```

**4. Access from any Tailscale device:**
```
https://your-machine.tailnet.ts.net
```

## Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│  Docker Compose Stack                                                 │
│                                                                       │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐       │
│  │  Frontend  │  │  LiveKit   │  │  Speaches  │  │   Kokoro   │       │
│  │  (Next.js) │  │   Server   │  │ (STT, GPU) │  │ (TTS, GPU) │       │
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
                   ┌─────────────────┼─────────────────┐
                   │                 │                 │
             ┌─────┴─────┐     ┌─────┴─────┐     ┌─────┴─────┐
             │  Ollama   │     │    n8n    │     │   Your    │
             │   (LLM)   │     │ Workflows │     │   APIs    │
             └───────────┘     └───────────┘     └───────────┘
                    External Services (on your network)
```

## Configuration

### Environment Variables (`.env`)

| Variable | Description | Default |
|----------|-------------|---------|
| `CAAL_HOST_IP` | Your server's LAN IP (required for WebRTC) | - |
| `N8N_MCP_URL` | n8n MCP server URL (required) | - |
| `LIVEKIT_URL` | LiveKit server URL | `ws://localhost:7880` |
| `SPEACHES_URL` | Speaches STT server URL | `http://localhost:8000` |
| `KOKORO_URL` | Kokoro TTS server URL | `http://localhost:8880` |
| `WHISPER_MODEL` | Faster-Whisper model | `Systran/faster-whisper-medium` |
| `TTS_VOICE` | Kokoro voice name | `am_puck` |
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | LLM model name | `qwen3:8b` |
| `OLLAMA_THINK` | Enable thinking mode (slower) | `false` |
| `PORCUPINE_ACCESS_KEY` | Picovoice key for wake word | - |

## n8n Workflow Integration

CAAL discovers tools from n8n workflows via MCP. Each workflow with a webhook trigger becomes a voice command.

### Quick Start

Example workflows are included in the `n8n-workflows/` folder:

```bash
cd n8n-workflows
cp config.env.example config.env
nano config.env  # Set your n8n IP and API key
python setup.py  # Creates all workflows in n8n
```

### Setup n8n

1. Enable MCP in n8n: **Settings > MCP Access > Enable MCP**
2. Set connection method to **Access Token** and copy the token
3. Enable workflow access in each workflow's settings
4. Set `N8N_MCP_URL` in `.env` to your n8n MCP endpoint (e.g., `http://192.168.1.100:5678/mcp-server/http`)

### Included Workflows

| Workflow | Voice Command |
|----------|---------------|
| `espn_get_nfl_scores` | "What are the NFL scores?" |
| `calendar_get_events` | "What's on my calendar today?" |
| `hass_control` | "Turn on the office lamp" |
| `radarr_search_movies` | "Do I have any Batman movies?" |

See `n8n-workflows/README.md` for full documentation.

## Wake Word Detection

CAAL supports "Hey Cal" wake word detection using Picovoice Porcupine.

**Setup:**
1. Get a free access key from [Picovoice Console](https://console.picovoice.ai/)
2. Train a custom "Hey Cal" wake word and download the **Web (WASM)** model
3. Place file in `frontend/public/`:
   - `hey_cal.ppn` - Custom wake word model (must replace with your own)
4. Add to `.env`: `PORCUPINE_ACCESS_KEY=your_key_here`
5. Rebuild frontend: `docker compose build frontend && docker compose up -d`

**Usage:**
- Toggle wake word on/off with the ear icon in the control bar
- Say "Hey Cal" to activate - CAAL responds with a greeting
- Conversation continues until agent finishes speaking

## Webhook API

External systems can trigger CAAL actions via HTTP:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/announce` | POST | Make CAAL speak a message |
| `/wake` | POST | Trigger wake word greeting |
| `/reload-tools` | POST | Refresh MCP tool cache |
| `/health` | GET | Health check |

**Example - Announce:**
```bash
curl -X POST http://localhost:8889/announce \
  -H "Content-Type: application/json" \
  -d '{"message": "Package delivered at front door"}'
```

**Example - Reload Tools:**
```bash
curl -X POST http://localhost:8889/reload-tools \
  -H "Content-Type: application/json" \
  -d '{"tool_name": "calendar_create_event"}'
```

## Local Development

```bash
# Install dependencies
uv sync

# Start infrastructure (LiveKit + Speaches + Kokoro)
docker compose up -d livekit speaches kokoro

# Run agent locally
uv run voice_agent.py dev

# Run frontend locally
cd frontend && pnpm install && pnpm dev
```

**Development commands:**
```bash
uv run ruff check src/        # Lint
uv run mypy src/              # Type check
uv run pytest                 # Test
```

## Project Structure

```
caal/
├── voice_agent.py              # Main entry point
├── .env                        # Environment variables
├── docker-compose.yaml         # Docker deployment
├── prompt/
│   └── default.md              # System prompt template
├── frontend/                   # Next.js web interface
│   ├── public/                 # Wake word models go here
│   └── components/             # UI components
├── n8n-workflows/              # Example n8n workflows
│   ├── setup.py                # One-command deployment
│   ├── config.env.example      # Configuration template
│   └── *.json                  # Workflow definitions
└── src/caal/
    ├── integrations/           # n8n MCP, web search
    ├── llm/                    # Ollama with think parameter
    ├── webhooks.py             # HTTP API endpoints
    └── utils/                  # Formatting helpers
```

## Troubleshooting

### WebRTC Not Connecting

**Symptom**: Frontend loads but voice doesn't work

1. **Check CAAL_HOST_IP** in `.env` - must match your network mode:
   - LAN HTTP/HTTPS: your LAN IP (e.g., `192.168.1.100`)
   - Tailscale: your Tailscale IP (`tailscale ip -4`)

2. **Check firewall** - these ports must be open:
   | Port | Protocol | Purpose |
   |------|----------|---------|
   | 3000 | TCP | Web UI |
   | 7880 | TCP | WebSocket signaling |
   | 7881 | TCP/UDP | WebRTC fallback |
   | 50000-50100 | UDP | WebRTC media |

3. **Check LiveKit logs**:
   ```bash
   docker compose logs livekit | grep -i "ice\|error"
   ```

### Agent Not Processing Voice

```bash
# Check agent logs
docker compose logs -f agent

# Verify Speaches (STT) is healthy
curl http://localhost:8000/health

# Verify Kokoro (TTS) is healthy
curl http://localhost:8880/health

# Verify Ollama is reachable
curl http://YOUR_OLLAMA_IP:11434/api/tags
```

### Ollama Connection Failed

**Symptom**: Agent logs show "error connecting to Ollama"

Ollama defaults to localhost only. Start it with network binding:

```bash
OLLAMA_HOST=0.0.0.0 ollama serve
```

Or set in your shell profile:
```bash
export OLLAMA_HOST=0.0.0.0
```

### Frontend Connection Timeout

**Symptom**: Frontend times out waiting for agent, especially on first connection

Ollama unloads models after 5 minutes by default. On slower drives (HDD), reloading takes too long.

**Option 1** - Keep model loaded:
```bash
OLLAMA_HOST=0.0.0.0 OLLAMA_KEEP_ALIVE=24h ollama serve
```

**Option 2** - Pre-load model before connecting:
```bash
ollama run qwen3:8b  # or your configured model
```

### n8n Tools Not Loading

1. Verify `N8N_MCP_URL` and `N8N_MCP_TOKEN` in `.env`
2. Check n8n has MCP enabled (Settings > MCP Access)
3. Ensure workflows have webhook triggers and are active

### First Start Is Slow

Normal - models download on first run (~2-5 minutes). Watch with:
```bash
docker compose logs -f speaches kokoro
```

## Production Hardening

### Generate Secure LiveKit Keys

```bash
# Generate new API keys
docker run --rm livekit/livekit-server generate-keys

# Update .env and livekit.yaml with generated values
```

### HTTPS

For HTTPS, see [Network Modes](#network-modes). Options:
- **LAN HTTPS (mkcert)**: Full voice from any device on your local network
- **Tailscale**: Full voice from anywhere via Tailscale network

Both use the same `--profile https` and nginx for TLS termination.

## Known Issues

1. **No Streaming STT**: Faster-Whisper uses batch processing (waits for speech to end). This is a fundamental limitation of Whisper-based solutions.

2. **Wake Word Models**: The Python `.ppn` models don't work in browser - you need the Web (WASM) version from Picovoice.

## Related Projects

- [LiveKit Agents](https://github.com/livekit/agents) - Voice agent framework
- [Speaches](https://github.com/speaches-ai/speaches) - Faster-Whisper STT server
- [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI) - Kokoro TTS server
- [Ollama](https://ollama.ai/) - Local LLM server
- [n8n](https://n8n.io/) - Workflow automation
- [Picovoice Porcupine](https://picovoice.ai/platform/porcupine/) - Wake word engine

## License

MIT License - see [LICENSE](LICENSE) for details.
