#!/bin/bash
set -euo pipefail

PATH="/Applications/Docker.app/Contents/Resources/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export PATH

CAAL_DIR="/Users/cesar/caal"
LOG_DIR="$CAAL_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/ensure-services.log"

exec >>"$LOG_FILE" 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ensure-services start"

ok=true

check_url() {
  local url="$1"
  curl -fsS --max-time 5 "$url" >/dev/null 2>&1
}

check_container() {
  local name="$1"
  docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null | grep -qx true
}

# Wait briefly for Docker after login/restart. Try to launch Docker Desktop first;
# Docker's helper is configured as a login item, but after a hard reboot it can
# lag behind user LaunchAgents.
if ! docker ps >/dev/null 2>&1; then
  open -gj -a Docker >/dev/null 2>&1 || true
fi
for i in {1..60}; do
  if docker ps >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

if ! docker ps >/dev/null 2>&1; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Docker not ready"
  exit 1
fi

if ! check_url "http://localhost:8001/docs"; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] mlx-audio check failed"
  ok=false
fi

if ! check_url "http://localhost:8002/health"; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] deepfilter check failed"
  ok=false
fi

for c in caal-nginx caal-agent caal-frontend caal-livekit; do
  if ! check_container "$c"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] container $c not running"
    ok=false
  fi
done

if [ "$ok" = true ]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] all services healthy"
  exit 0
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] unhealthy services detected; restarting CAAL"
cd "$CAAL_DIR"
./start-apple.sh

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ensure-services completed"
