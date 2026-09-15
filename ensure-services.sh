#!/bin/bash
# launchd runs this every five minutes; recovery shares the startup lock.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec /usr/bin/python3 "$ROOT/startup_runtime.py" "$@"
