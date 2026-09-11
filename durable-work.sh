#!/bin/bash
# CAAL durable background work: operations.
#
#   ./durable-work.sh health      liveness of the worker container
#   ./durable-work.sh status      queue and dispatch counts (authenticated)
#   ./durable-work.sh verify      full pre-flight: compose config, service
#                                 health, worker registration, durable runner
#                                 health, and a safe callback-dispatch
#                                 construction. Places no call.
#   ./durable-work.sh wake        ask the worker to tick now
#   ./durable-work.sh orphans     how many pre-lease tasks await adoption
#   ./durable-work.sh adopt       adopt them (each may end in a real callback)
#   ./durable-work.sh restart     restart only the worker
#   ./durable-work.sh logs        follow the worker log
#
# The worker publishes no host port. Everything here talks to it from inside
# the compose network, authenticated with CAAL_INTERNAL_AUTH_SECRET, which is
# read from .env, passed as an environment variable, and never printed.
# Nothing here prints task text, a task id, a user id, or a phone number: the
# status surface is counts only, by design.

set -euo pipefail
cd "$(dirname "$0")"

log()  ( echo "[durable-work] $1" )
warn() ( echo "[durable-work] WARNING: $1" )
fail() ( echo "[durable-work] ERROR: $1" >&2 )

COMPOSE_FILES="-f docker-compose.apple.yaml"
TELEPHONY="$(grep -E '^CAAL_TELEPHONY=' .env 2>/dev/null | head -n 1 | cut -d= -f2- | tr -d "\"'" || true)"
case "$TELEPHONY" in
    1|true|yes)
        COMPOSE_FILES="$COMPOSE_FILES -f docker-compose.telephony.yaml"
        ;;
esac

compose() ( docker compose $COMPOSE_FILES "$@" )

# The shared internal secret, read straight from .env into the environment of
# the process that needs it, so it never reaches a command line or a log.
worker_secret() (
    grep -E '^CAAL_INTERNAL_AUTH_SECRET=' .env 2>/dev/null \
        | head -n 1 | cut -d= -f2- | tr -d "\"'"
)

worker_call() (
    METHOD="$1"
    CALL_PATH="$2"
    SECRET="$(worker_secret)"
    if [ -z "$SECRET" ]; then
        fail "CAAL_INTERNAL_AUTH_SECRET is not set in .env; the control plane stays closed."
        return 3
    fi
    compose exec -T \
        -e "CAAL_WORKER_SECRET=$SECRET" \
        -e "CAAL_CALL_METHOD=$METHOD" \
        -e "CAAL_CALL_PATH=$CALL_PATH" \
        worker python - <<'PY'
import json
import os
import urllib.error
import urllib.request

port = os.getenv("CAAL_WORKER_PORT", "8890")
url = "http://127.0.0.1:" + port + os.environ["CAAL_CALL_PATH"]
request = urllib.request.Request(url, method=os.environ["CAAL_CALL_METHOD"])
request.add_header("X-CAAL-Worker-Token", os.environ["CAAL_WORKER_SECRET"])
try:
    with urllib.request.urlopen(request, timeout=10) as response:
        body = response.read().decode("utf-8")
except urllib.error.HTTPError as error:
    raise SystemExit("worker refused the call: HTTP " + str(error.code))
print(json.dumps(json.loads(body), indent=2, sort_keys=True))
PY
)

cmd_health() (
    compose exec -T worker python - <<'PY'
import json
import os
import urllib.request

port = os.getenv("CAAL_WORKER_PORT", "8890")
with urllib.request.urlopen("http://127.0.0.1:" + port + "/healthz", timeout=10) as response:
    print(json.dumps(json.loads(response.read().decode("utf-8")), sort_keys=True))
PY
)

cmd_status() ( worker_call GET /status )

cmd_wake() ( worker_call POST /internal/wake )

cmd_orphans() (
    compose exec -T worker python - <<'PY'
from caal import background_tasks

print("orphaned_running=" + str(background_tasks.orphaned_running_count()))
print("counts=" + str(background_tasks.queue_counts()))
PY
)

cmd_adopt() (
    warn "Adoption returns pre-lease running work to the queue."
    warn "Such a task may carry a callback the caller confirmed. When it settles,"
    warn "JARVIS will place a real outbound call to its owner's approved number."
    if [ "${1:-}" != "--yes" ]; then
        printf "Type ADOPT to continue: "
        read -r CONFIRM
        if [ "$CONFIRM" != "ADOPT" ]; then
            fail "Not adopted."
            return 1
        fi
    fi
    compose exec -T worker python - <<'PY'
from caal import background_tasks

print("adopted=" + str(background_tasks.adopt_orphaned_running()))
PY
)

# Construct a callback exactly the way the dispatcher does -- policy, metadata
# and all -- against a synthetic task and a synthetic resolver, and report only
# the shape. No store row is touched, no room is created, nothing is dialed.
cmd_verify_dispatch() (
    compose exec -T worker python - <<'PY'
from caal.durable_work import CallbackRefusedError, build_callback_request

TASK = "bt_" + "0" * 16
USER = "usr_" + "f" * 24
NUMBER = "+15550000000"

request = build_callback_request(
    task_id=TASK,
    user_id=USER,
    destination=None,
    resolve_user_destination=lambda user_id: NUMBER if user_id == USER else None,
)
metadata = request.dispatch_metadata()
print("construction=ok")
print("metadata_keys=" + ",".join(sorted(str(key) for key in metadata)))
print("destination_is_resolved=" + str(metadata["destination"] == NUMBER))
print("carries_task=" + str(metadata["callback_task_id"] == TASK))
print("carries_user=" + str(metadata["user_id"] == USER))
print("carries_no_transcript=" + str("handoff_context" not in metadata))
print("repr_hides_number=" + str(NUMBER not in repr(request)))

for label, user, number in (
    ("unknown_owner", "usr_" + "e" * 24, None),
    ("model_chosen_number", USER, "+19998887777"),
):
    try:
        build_callback_request(
            task_id=TASK,
            user_id=user,
            destination=number,
            resolve_user_destination=lambda user_id: NUMBER if user_id == USER else None,
        )
    except CallbackRefusedError:
        print("refused_" + label + "=True")
    else:
        raise SystemExit("SECURITY: " + label + " was not refused")
PY
)

cmd_verify() (
    log "1. compose configuration"
    compose config --quiet
    log "   ok"

    log "2. worker service state"
    compose ps worker

    log "3. worker liveness"
    cmd_health

    log "4. worker registration and durable runner health (counts only)"
    cmd_status

    log "5. safe callback-dispatch construction (no call is placed)"
    cmd_verify_dispatch

    log "6. pre-lease work awaiting explicit adoption"
    cmd_orphans

    log "verify: complete"
)

cmd_restart() ( compose up -d --no-deps worker && compose ps worker )

cmd_logs() ( compose logs -f --tail 100 worker )

case "${1:-verify}" in
    health)  cmd_health ;;
    status)  cmd_status ;;
    wake)    cmd_wake ;;
    orphans) cmd_orphans ;;
    adopt)   cmd_adopt "${2:-}" ;;
    verify)  cmd_verify ;;
    verify-dispatch) cmd_verify_dispatch ;;
    restart) cmd_restart ;;
    logs)    cmd_logs ;;
    *)
        fail "unknown command: $1"
        echo "commands: health status wake orphans adopt verify verify-dispatch restart logs" >&2
        exit 2
        ;;
esac
