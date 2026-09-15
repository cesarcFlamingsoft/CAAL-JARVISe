#!/bin/bash
# Recover the verified macOS deployment; --check / --dry-run change no services.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec /usr/bin/python3 "$ROOT/startup_runtime.py" "$@"
