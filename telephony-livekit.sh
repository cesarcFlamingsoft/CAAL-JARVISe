#!/bin/bash
# Recreate and verify LiveKit under the correct compose set.
# =========================================================
#
# Why this exists: "docker compose up -d livekit" from the base compose file
# alone recreated caal-livekit without the telephony overlay. LiveKit came back
# with no Redis section, and every outbound call then failed inside LiveKit with
# "sip not connected (redis required)" -- while caal-sip and caal-redis stayed
# up and healthy, so nothing looked wrong from the outside.
#
# Usage:
#   ./telephony-livekit.sh check       # verify the running stack, change nothing
#   ./telephony-livekit.sh recreate    # recreate livekit under the exact set
#   ./telephony-livekit.sh config      # validate the compose set; never print it
#
# The compose set is explicit, from files and .env -- never from a container
# label:
#   CAAL_BASE_COMPOSE   base file (default: apple on arm64 macOS, else the
#                       generic docker-compose.yaml)
#   CAAL_TELEPHONY=1    in .env: this deployment runs SIP, so the telephony
#                       overlay is part of every compose command. Unset, this
#                       script never enables SIP on its own.
#
# Nothing here prints a credential: the config check reports key presence only.
set -euo pipefail

cd "$(dirname "$0")"

ok()   { echo "[ok]   $1"; }
warn() { echo "[warn] $1"; }
bad()  { echo "[fail] $1"; }

# .env is the deployment own declaration of what it runs.
TELEPHONY_SETTING="${CAAL_TELEPHONY:-}"
if [ -f ./.env ]; then
  FROM_ENV=$(grep -E '^CAAL_TELEPHONY=' ./.env | tail -n 1 | cut -d= -f2 | tr -d '[:space:]')
  if [ -n "$FROM_ENV" ]; then
    TELEPHONY_SETTING="$FROM_ENV"
  fi
fi

if [ -z "${CAAL_BASE_COMPOSE:-}" ]; then
  if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
    CAAL_BASE_COMPOSE="docker-compose.apple.yaml"
  else
    CAAL_BASE_COMPOSE="docker-compose.yaml"
  fi
fi

COMPOSE_ARGS=(-f "$CAAL_BASE_COMPOSE")
TELEPHONY=no
case "$TELEPHONY_SETTING" in
  1|true|yes)
    COMPOSE_ARGS+=(-f docker-compose.telephony.yaml)
    TELEPHONY=yes
    ;;
esac
if [ -n "${HTTPS_DOMAIN:-}" ] || grep -qsE '^HTTPS_DOMAIN=.+' ./.env; then
  COMPOSE_ARGS+=(--profile https)
fi

echo "compose set: docker compose ${COMPOSE_ARGS[*]}"
echo "telephony:   $TELEPHONY"

validate() {
  docker compose "${COMPOSE_ARGS[@]}" config -q
  ok "compose configuration is valid"
}

# Report only whether a key is configured, never its value.
check_rendered_config() {
  local present
  present=$(docker exec caal-livekit sh -c "grep -c '^redis:' /etc/livekit.yaml || true" 2>/dev/null | tr -d '[:space:]')
  if [ "${present:-0}" != "0" ]; then
    ok "livekit: redis configured in /etc/livekit.yaml: yes"
  else
    bad "livekit: redis configured in /etc/livekit.yaml: no (SIP cannot dial)"
    return 1
  fi
}

check_redis() {
  if docker exec caal-redis redis-cli ping >/dev/null 2>&1; then
    ok "redis: responding to PING"
  else
    bad "redis: not responding"
    return 1
  fi
}

check_sip() {
  if ! docker inspect -f '{{.State.Running}}' caal-sip 2>/dev/null | grep -qx true; then
    bad "sip: caal-sip is not running"
    return 1
  fi
  # The SIP service registers with LiveKit through Redis; a connected service
  # stops repeating the redis error in its log.
  local recent
  recent=$(docker logs --since 5m caal-sip 2>&1 | tail -n 50 || true)
  if echo "$recent" | grep -qi 'redis required'; then
    bad "sip: LiveKit is still refusing SIP (redis required)"
    return 1
  fi
  ok "sip: caal-sip is running with no redis errors in the last 5 minutes"
}

check_livekit_health() {
  local code
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://localhost:7880 || true)
  if [ "$code" = "200" ]; then
    ok "livekit: HTTP $code on :7880"
  else
    bad "livekit: HTTP ${code:-none} on :7880"
    return 1
  fi
}

check_all() {
  local failed=0
  check_livekit_health || failed=1
  if [ "$TELEPHONY" = "yes" ]; then
    check_rendered_config || failed=1
    check_redis || failed=1
    check_sip || failed=1
  else
    warn "telephony is not enabled for this deployment; skipping SIP checks"
  fi
  return $failed
}

case "${1:-check}" in
  config)
    validate
    # `docker compose config` expands env_file values and would print every
    # deployment credential. The compose set was already printed above and
    # config -q gives the only useful diagnostic without that disclosure.
    ok "compose set is safe to use (rendered configuration withheld)"
    ;;
  check)
    validate
    check_all
    ;;
  recreate)
    validate
    echo "recreating caal-livekit under the compose set above..."
    docker compose "${COMPOSE_ARGS[@]}" up -d --force-recreate --no-deps livekit
    for _ in $(seq 1 30); do
      if docker inspect -f '{{.State.Health.Status}}' caal-livekit 2>/dev/null | grep -qx healthy; then
        break
      fi
      sleep 2
    done
    # SIP registers with LiveKit through Redis; restart it so it re-registers
    # against the LiveKit process that just came up.
    if [ "$TELEPHONY" = "yes" ]; then
      docker compose "${COMPOSE_ARGS[@]}" restart sip >/dev/null
      sleep 5
    fi
    check_all
    ;;
  *)
    echo "usage: $0 [check|recreate|config]" >&2
    exit 2
    ;;
esac
