#!/bin/sh
# Render the LiveKit server configuration this container runs with.
# =================================================================
#
# One script, used by every CAAL composition, so the configuration a LiveKit
# container comes up with depends only on its environment -- never on which
# compose file happened to be used for the last `docker compose up -d livekit`.
#
# Inputs (environment):
#   HTTPS_DOMAIN            non-empty -> the public TURN/TLS template
#   LIVEKIT_REDIS_ADDRESS   non-empty -> append LiveKit's Redis section, which
#                           LiveKit requires before it will accept SIP
#                           ("sip not connected (redis required)"). Empty, the
#                           default, renders a single-node config with no Redis.
#   LIVEKIT_API_KEY / LIVEKIT_API_SECRET / CAAL_HOST_IP  templated as before.
#
# Usage: livekit-render-config.sh [output-path]
#
# Nothing here prints a credential or the rendered file: only which mode was
# chosen and whether Redis is configured.
set -eu

OUT="${1:-/etc/livekit.yaml}"
LAN_TEMPLATE="${LIVEKIT_LAN_TEMPLATE:-/etc/livekit-lan.yaml}"
TLS_TEMPLATE="${LIVEKIT_TLS_TEMPLATE:-/etc/livekit.yaml.template}"

if [ -n "${HTTPS_DOMAIN:-}" ]; then
  echo "livekit-config: HTTPS mode (TURN/TLS enabled)"
  envsubst < "$TLS_TEMPLATE" > "$OUT"
else
  echo "livekit-config: LAN mode"
  envsubst < "$LAN_TEMPLATE" > "$OUT"
fi

# Redis is additive: a telephony deployment keeps the public HTTPS/TURN
# configuration it had and gains the Redis section SIP needs.
if [ -n "${LIVEKIT_REDIS_ADDRESS:-}" ]; then
  if grep -q '^redis:' "$OUT"; then
    echo "livekit-config: redis configured: yes (already in template)"
  else
    printf '\nredis:\n  address: %s\n' "$LIVEKIT_REDIS_ADDRESS" >> "$OUT"
    echo "livekit-config: redis configured: yes (SIP/telephony enabled)"
  fi
else
  echo "livekit-config: redis configured: no (single node, SIP disabled)"
fi
