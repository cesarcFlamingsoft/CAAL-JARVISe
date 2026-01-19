#!/bin/bash
# CAAL Startup Script for Apple Silicon
# Usage: ./start-apple.sh [--stop] [--build]
#
# Runs mlx-audio and agent natively on macOS for GPU acceleration (MPS)
# Only livekit and frontend run in Docker

set -e
cd "$(dirname "$0")"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log() { echo -e "${GREEN}[CAAL]${NC} $1"; }
warn() { echo -e "${YELLOW}[CAAL]${NC} $1"; }
error() { echo -e "${RED}[CAAL]${NC} $1"; }

# PID and log files
MLX_PID_FILE="/tmp/caal-mlx-audio.pid"
MLX_LOG_FILE="/tmp/caal-mlx-audio.log"
AGENT_PID_FILE="/tmp/caal-agent.pid"
AGENT_LOG_FILE="/tmp/caal-agent.log"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MLX_VENV="$SCRIPT_DIR/.mlx-audio-venv"
MLX_PYTHON="$MLX_VENV/bin/python"
AGENT_VENV="$SCRIPT_DIR/.agent-venv"
AGENT_PYTHON="$AGENT_VENV/bin/python"

# Load .env file if it exists
if [ -f "$SCRIPT_DIR/.env" ]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/.env"
fi

# Set Docker profile based on HTTPS_DOMAIN
if [ -n "${HTTPS_DOMAIN}" ]; then
    DOCKER_PROFILE="--profile https"
else
    DOCKER_PROFILE=""
fi

banner() {
    echo -e "${CYAN}${BOLD}"
    cat << 'EOF'
    ███╗   ███╗██╗     ██╗  ██╗       ██████╗ █████╗  █████╗ ██╗
    ████╗ ████║██║     ╚██╗██╔╝      ██╔════╝██╔══██╗██╔══██╗██║
    ██╔████╔██║██║      ╚███╔╝ █████╗██║     ███████║███████║██║
    ██║╚██╔╝██║██║      ██╔██╗ ╚════╝██║     ██╔══██║██╔══██║██║
    ██║ ╚═╝ ██║███████╗██╔╝ ██╗      ╚██████╗██║  ██║██║  ██║███████╗
    ╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝       ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝
EOF
    echo -e "${NC}"
    echo -e "  ${BOLD}Voice Assistant for Apple Silicon (Native GPU)${NC}"
    echo ""
}

# Load a model with progress feedback
load_model() {
    local model="$1"
    local name="$2"

    curl -s -X POST "http://localhost:8001/v1/models?model_name=$model" > /dev/null &
    local pid=$!

    local spin=('⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏')
    local i=0

    while kill -0 $pid 2>/dev/null; do
        printf "\r${GREEN}[CAAL]${NC} Loading %s ${CYAN}%s${NC} " "$name" "${spin[$i]}"
        i=$(( (i + 1) % 10 ))
        sleep 0.1
    done

    wait $pid
    printf "\r${GREEN}[CAAL]${NC} ✓ %s loaded                    \n" "$name"
}

stop_all() {
    echo ""
    log "Stopping CAAL..."

    # Stop Docker (livekit, frontend)
    log "Stopping Docker containers..."
    docker compose -f docker-compose.apple.yaml $DOCKER_PROFILE down || true

    # Stop agent
    if [ -f "$AGENT_PID_FILE" ]; then
        PID=$(cat "$AGENT_PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID" 2>/dev/null || true
            log "Agent stopped (PID $PID)"
        fi
        rm -f "$AGENT_PID_FILE"
    fi

    # Stop mlx-audio
    if [ -f "$MLX_PID_FILE" ]; then
        PID=$(cat "$MLX_PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID" 2>/dev/null || true
            log "mlx-audio stopped (PID $PID)"
        fi
        rm -f "$MLX_PID_FILE"
    fi

    echo ""
    log "CAAL stopped."
    exit 0
}

# Handle flags
BUILD_FLAG=""
for arg in "$@"; do
    case $arg in
        --stop)
            stop_all
            ;;
        --build)
            BUILD_FLAG="--build"
            log "Will rebuild Docker images and agent venv..."
            # Force rebuild of agent venv
            rm -rf "$AGENT_VENV"
            ;;
    esac
done

setup_mlx_audio() {
    log "Setting up mlx-audio environment..."

    # Check if venv exists but is corrupted (missing pip - can happen with uv-created venvs)
    if [ -d "$MLX_VENV" ] && ! "$MLX_VENV/bin/python" -c "import pip" 2>/dev/null; then
        warn "Virtual environment is corrupted (missing pip). Recreating..."
        rm -rf "$MLX_VENV"
    fi

    # Create virtual environment if it doesn't exist
    # Use Python 3.11 explicitly for compatibility with mlx-audio dependencies
    if [ ! -d "$MLX_VENV" ]; then
        log "Creating virtual environment at $MLX_VENV..."
        if command -v python3.11 &> /dev/null; then
            python3.11 -m venv "$MLX_VENV"
        else
            warn "Python 3.11 not found, using default python3 (may have compatibility issues)"
            python3 -m venv "$MLX_VENV"
        fi
    fi

    # Verify pip is available
    if ! "$MLX_VENV/bin/python" -c "import pip" 2>/dev/null; then
        error "Failed to create virtual environment with pip. Please check your Python installation."
        exit 1
    fi

    # Upgrade pip
    "$MLX_VENV/bin/pip" install --upgrade pip -q

    # Install mlx-audio and all dependencies
    log "Installing mlx-audio and dependencies (this may take a few minutes)..."

    # Install all mlx-audio dependencies in one command
    "$MLX_VENV/bin/pip" install -q \
        mlx-audio \
        soundfile fastapi uvicorn webrtcvad python-multipart \
        numba tiktoken scipy tqdm \
        loguru misaki num2words spacy phonemizer-fork espeakng-loader torch

    log "✓ mlx-audio environment ready"
    echo ""

    # Pre-download models
    log "Pre-downloading models (first time may take a few minutes)..."
    echo ""

    log "Downloading Whisper STT model..."
    "$MLX_PYTHON" -c "
from huggingface_hub import snapshot_download
snapshot_download('mlx-community/whisper-medium-mlx', local_files_only=False)
print('Done')
" 2>/dev/null || warn "Whisper model will be downloaded on first use"

    log "Downloading Kokoro TTS model..."
    "$MLX_PYTHON" -c "
from huggingface_hub import snapshot_download
snapshot_download('prince-canuma/Kokoro-82M', local_files_only=False)
print('Done')
" 2>/dev/null || warn "Kokoro model will be downloaded on first use"

    echo ""
    log "✓ Models ready"
}

setup_agent() {
    log "Setting up native agent environment (with MPS GPU support)..."

    # Check if venv exists but is corrupted
    if [ -d "$AGENT_VENV" ] && ! "$AGENT_VENV/bin/python" -c "import pip" 2>/dev/null; then
        warn "Agent virtual environment is corrupted. Recreating..."
        rm -rf "$AGENT_VENV"
    fi

    # Create virtual environment if it doesn't exist
    if [ ! -d "$AGENT_VENV" ]; then
        log "Creating virtual environment at $AGENT_VENV..."
        if command -v python3.11 &> /dev/null; then
            python3.11 -m venv "$AGENT_VENV"
        else
            warn "Python 3.11 not found, using default python3"
            python3 -m venv "$AGENT_VENV"
        fi
    fi

    # Verify pip is available
    if ! "$AGENT_VENV/bin/python" -c "import pip" 2>/dev/null; then
        error "Failed to create agent virtual environment with pip."
        exit 1
    fi

    # Upgrade pip
    "$AGENT_VENV/bin/pip" install --upgrade pip -q

    log "Installing agent dependencies (this may take a few minutes)..."

    # Install PyTorch with MPS support (Apple Silicon GPU)
    "$AGENT_VENV/bin/pip" install -q torch torchvision torchaudio

    # Install the CAAL package and dependencies
    "$AGENT_VENV/bin/pip" install -q -e "$SCRIPT_DIR"

    # Install DeepFilterNet for noise suppression (uses MPS!)
    "$AGENT_VENV/bin/pip" install -q deepfilternet

    log "✓ Agent environment ready (PyTorch MPS enabled)"
    echo ""
}

banner
log "Starting CAAL (Native Apple Silicon mode)..."
echo ""

# Check Ollama
printf "${GREEN}[CAAL]${NC} Checking Ollama... "
if ! curl -s http://10.0.0.64:11434/api/tags > /dev/null 2>&1; then
    echo ""
    error "Ollama is not accessible on localhost:11434"
    error "Run: ollama serve"
    exit 1
fi
echo -e "${GREEN}✓${NC}"

# Check/setup mlx-audio environment
if [ ! -f "$MLX_PYTHON" ] || \
   ! "$MLX_PYTHON" -c "import pip" 2>/dev/null || \
   ! "$MLX_PYTHON" -c "import mlx_audio" 2>/dev/null; then
    setup_mlx_audio
else
    printf "${GREEN}[CAAL]${NC} Checking mlx-audio... "
    echo -e "${GREEN}✓${NC}"
fi

# Check/setup agent environment
if [ ! -f "$AGENT_PYTHON" ] || \
   ! "$AGENT_PYTHON" -c "import pip" 2>/dev/null || \
   ! "$AGENT_PYTHON" -c "import livekit.agents" 2>/dev/null; then
    setup_agent
else
    printf "${GREEN}[CAAL]${NC} Checking agent environment... "
    echo -e "${GREEN}✓${NC}"
fi

# Check if mlx-audio is already running
MLX_RUNNING=false
if [ -f "$MLX_PID_FILE" ]; then
    PID=$(cat "$MLX_PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        MLX_RUNNING=true
        log "✓ mlx-audio already running (PID $PID)"
    fi
fi

# Start mlx-audio if not running
if [ "$MLX_RUNNING" = false ]; then
    printf "${GREEN}[CAAL]${NC} Starting mlx-audio server... "

    # Start in background using dedicated venv
    nohup "$MLX_PYTHON" -m mlx_audio.server --host 0.0.0.0 --port 8001 > "$MLX_LOG_FILE" 2>&1 &
    MLX_PID=$!
    echo $MLX_PID > "$MLX_PID_FILE"

    # Wait for server to be ready
    for i in {1..30}; do
        if curl -s http://localhost:8001/docs > /dev/null 2>&1; then
            break
        fi
        sleep 1
    done

    if ! curl -s http://localhost:8001/docs > /dev/null 2>&1; then
        echo ""
        error "mlx-audio failed to start. Logs: $MLX_LOG_FILE"
        exit 1
    fi

    echo -e "${GREEN}✓${NC} (PID $MLX_PID)"
    echo ""

    # Preload models with progress
    log "Preloading models (first time may take a few minutes)..."
    echo ""

    load_model "mlx-community/whisper-medium-mlx" "Whisper STT"
    load_model "prince-canuma/Kokoro-82M" "Kokoro TTS"

    echo ""
fi

# Start Docker services (livekit, frontend only - agent runs natively)
# Add nginx if HTTPS is enabled
DOCKER_SERVICES="livekit frontend"
if [ -n "${HTTPS_DOMAIN}" ]; then
    DOCKER_SERVICES="$DOCKER_SERVICES nginx"
fi
log "Starting Docker services ($DOCKER_SERVICES)..."
docker compose -f docker-compose.apple.yaml up -d $BUILD_FLAG $DOCKER_SERVICES

# Wait for livekit
printf "${GREEN}[CAAL]${NC} Waiting for LiveKit"
for i in {1..10}; do
    if curl -s http://localhost:7880 > /dev/null 2>&1; then
        break
    fi
    printf "."
    sleep 1
done
echo -e " ${GREEN}✓${NC}"

# Check if agent is already running
AGENT_RUNNING=false
if [ -f "$AGENT_PID_FILE" ]; then
    PID=$(cat "$AGENT_PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        AGENT_RUNNING=true
        log "✓ Agent already running (PID $PID)"
    fi
fi

# Start agent natively if not running
if [ "$AGENT_RUNNING" = false ]; then
    printf "${GREEN}[CAAL]${NC} Starting native agent (MPS GPU)... "

    # Set environment variables for the agent
    export LIVEKIT_URL="${LIVEKIT_URL:-ws://localhost:7880}"
    export LIVEKIT_API_KEY="${LIVEKIT_API_KEY:-devkey}"
    export LIVEKIT_API_SECRET="${LIVEKIT_API_SECRET:-secret}"
    export SPEACHES_URL="${MLX_AUDIO_URL:-http://localhost:8001}"
    export KOKORO_URL="${MLX_AUDIO_URL:-http://localhost:8001}"
    export WHISPER_MODEL="${WHISPER_MODEL:-mlx-community/whisper-medium-mlx}"
    export TTS_MODEL="${TTS_MODEL:-prince-canuma/Kokoro-82M}"
    export TTS_VOICE="${TTS_VOICE:-af_heart}"
    export CAAL_SETTINGS_PATH="$SCRIPT_DIR/settings.json"
    export CAAL_PROMPT_DIR="$SCRIPT_DIR/prompt"

    # Start agent in background
    nohup "$AGENT_PYTHON" "$SCRIPT_DIR/voice_agent.py" dev > "$AGENT_LOG_FILE" 2>&1 &
    AGENT_PID=$!
    echo $AGENT_PID > "$AGENT_PID_FILE"

    # Wait for agent to initialize
    sleep 3

    if ! kill -0 "$AGENT_PID" 2>/dev/null; then
        echo ""
        error "Agent failed to start. Logs: $AGENT_LOG_FILE"
        tail -20 "$AGENT_LOG_FILE"
        exit 1
    fi

    echo -e "${GREEN}✓${NC} (PID $AGENT_PID)"
fi

# Wait for frontend
printf "${GREEN}[CAAL]${NC} Waiting for frontend"
for i in {1..15}; do
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        break
    fi
    printf "."
    sleep 1
done
echo -e " ${GREEN}✓${NC}"

echo ""
echo -e "${CYAN}══════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  CAAL is ready! (Native Apple Silicon Mode)${NC}"
echo -e "${CYAN}══════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${BOLD}GPU Acceleration:${NC}  PyTorch MPS (DeepFilterNet)"
echo -e "  ${BOLD}                    ${NC}  MLX (Whisper STT, Kokoro TTS)"
echo ""
if [ -n "${HTTPS_DOMAIN}" ]; then
    echo -e "  ${BOLD}Web interface:${NC}  https://${HTTPS_DOMAIN}:3443"
else
    echo -e "  ${BOLD}Web interface:${NC}  http://localhost:3000"
fi
echo -e "  ${BOLD}Stop command:${NC}   ./start-apple.sh --stop"
echo ""
echo -e "  ${BOLD}Logs:${NC}"
echo -e "    mlx-audio:  tail -f $MLX_LOG_FILE"
echo -e "    agent:      tail -f $AGENT_LOG_FILE"
echo ""
