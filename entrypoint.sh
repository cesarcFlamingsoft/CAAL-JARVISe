#!/bin/bash
# CAAL Agent Entrypoint
# Creates config files from defaults if they don't exist, then runs as agent user

set -e

CONFIG_DIR="/app/config"
DATA_DIR="${CAAL_DATA_DIR:-/app/data}"

# Ensure runtime directories are writable by the unprivileged agent process.
# Named volumes are initially root-owned, so without this the SQLite-backed
# device registry (and memory tools) can read but cannot create/update state.
mkdir -p "$DATA_DIR"
# Existing named volumes can contain root-owned SQLite files created by an older
# container. The directory alone is insufficient: SQLite needs to update the
# database file and may create journal/WAL siblings alongside it.
chown -R agent:agent "$DATA_DIR"

# Ensure config directory exists and is writable by agent
mkdir -p "$CONFIG_DIR"
chown agent:agent "$CONFIG_DIR"

# settings.json - copy default if missing
if [ ! -f "$CONFIG_DIR/settings.json" ]; then
    echo "Creating settings.json from defaults..."
    cp /app/settings.default.json "$CONFIG_DIR/settings.json"
    chown agent:agent "$CONFIG_DIR/settings.json"
fi

# mcp_servers.json - copy default if missing
if [ ! -f "$CONFIG_DIR/mcp_servers.json" ]; then
    echo "Creating mcp_servers.json from defaults..."
    cp /app/mcp_servers.default.json "$CONFIG_DIR/mcp_servers.json"
    chown agent:agent "$CONFIG_DIR/mcp_servers.json"
fi

# Create symlinks from /app to config files (for code that expects them in /app)
# Skip if files are already mounted directly (e.g., via docker-compose volumes)
if [ ! -e /app/settings.json ]; then
    ln -sf "$CONFIG_DIR/settings.json" /app/settings.json
fi
if [ ! -e /app/mcp_servers.json ]; then
    ln -sf "$CONFIG_DIR/mcp_servers.json" /app/mcp_servers.json
fi

# Drop privileges and execute the main command as agent user
exec gosu agent "$@"
