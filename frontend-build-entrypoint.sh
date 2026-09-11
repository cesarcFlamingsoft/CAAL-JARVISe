#!/bin/sh
# Install the published frontend build, then start the Next.js server.
#
# This runs as the entrypoint of every caal-frontend container. The image bakes
# a /app/.next from whenever it was last built, and nothing in the stack
# rebuilds it on a normal `docker compose up -d`: a recreate during an unrelated
# deployment silently brought back a months-old build and the dashboard
# vanished. So the image is treated as a Node runtime only, and the application
# comes from the artifact published by ./publish-frontend-build.sh and mounted
# read-only at /app/.next-deploy.
#
# The artifact is mounted at a *side* path on purpose. Docker creates a missing
# bind-mount source as an empty directory, and an empty directory mounted
# straight over /app/.next would leave the container with no application at all
# -- worse than the stale build this replaces.
#
# Environment:
#   CAAL_FRONTEND_ALLOW_BAKED_BUILD=true  serve the image build when nothing is
#                                         published (first run from a fresh
#                                         image); off by default, because
#                                         falling back silently is the bug.
#   CAAL_FRONTEND_EXPECTED_BUILD_ID       refuse to start on any other build id
#   CAAL_FRONTEND_APP_DIR                 default /app
#   CAAL_FRONTEND_BUILD_DIR               default /app/.next-deploy
#   CAAL_FRONTEND_RUN_USER                default nextjs
#
# See docs/FRONTEND-DEPLOYMENT.md.

set -e

APP_DIR=$CAAL_FRONTEND_APP_DIR
if [ -z "$APP_DIR" ]; then
    APP_DIR=/app
fi
BUILD_DIR=$CAAL_FRONTEND_BUILD_DIR
if [ -z "$BUILD_DIR" ]; then
    BUILD_DIR=$APP_DIR/.next-deploy
fi
RUN_USER=$CAAL_FRONTEND_RUN_USER
if [ -z "$RUN_USER" ]; then
    RUN_USER=nextjs
fi

# Records the build id fully installed into APP_DIR. A half-finished install
# leaves no marker, so the next start redoes it instead of trusting the mix.
MARKER=$APP_DIR/.caal-deployed-build-id

if [ "$#" -eq 0 ]; then
    set -- node server.js
fi

log() (
    echo "[frontend-build] $1" >&2
)

read_id() (
    if [ -s "$1" ]; then
        cat "$1"
    fi
)

# --- Is there a complete artifact to install? -------------------------------
# BUILD_ID is written last by the publisher and must agree with the one inside
# .next, so neither an empty mount nor a torn copy can look complete.
ARTIFACT_ID=$(read_id "$BUILD_DIR/BUILD_ID")
INNER_ID=$(read_id "$BUILD_DIR/.next/BUILD_ID")
if [ ! -f "$BUILD_DIR/server.js" ] || [ -z "$ARTIFACT_ID" ] || [ "$ARTIFACT_ID" != "$INNER_ID" ]; then
    ARTIFACT_ID=
fi

BAKED_ID=$(read_id "$APP_DIR/.next/BUILD_ID")

if [ -z "$ARTIFACT_ID" ]; then
    if [ "$CAAL_FRONTEND_ALLOW_BAKED_BUILD" = "true" ] && [ -n "$BAKED_ID" ]; then
        log "WARNING: no published build at $BUILD_DIR."
        log "WARNING: serving the baked image build $BAKED_ID because"
        log "WARNING: CAAL_FRONTEND_ALLOW_BAKED_BUILD=true. This build is as old"
        log "WARNING: as the image and may be missing the dashboard."
        cd "$APP_DIR"
        if [ "$(id -u)" = "0" ] && [ "$RUN_USER" != root ]; then
            exec su -s /bin/sh "$RUN_USER" -c 'exec "$0" "$@"' -- "$@"
        fi
        exec "$@"
    fi
    log "ERROR: no published frontend build at $BUILD_DIR."
    log "ERROR: refusing to serve the build baked into the image -- it drifts"
    log "ERROR: from the deployed dashboard and that failure is invisible."
    log "ERROR: On the Docker host run: ./publish-frontend-build.sh"
    log "ERROR: then: docker compose up -d --force-recreate frontend"
    log "ERROR: To serve the image build anyway, set CAAL_FRONTEND_ALLOW_BAKED_BUILD=true."
    exit 1
fi

if [ -n "$CAAL_FRONTEND_EXPECTED_BUILD_ID" ] && \
   [ "$CAAL_FRONTEND_EXPECTED_BUILD_ID" != "$ARTIFACT_ID" ]; then
    log "ERROR: published build is $ARTIFACT_ID but this deployment is pinned to"
    log "ERROR: $CAAL_FRONTEND_EXPECTED_BUILD_ID. Not starting."
    exit 1
fi

# --- Install ----------------------------------------------------------------
INSTALLED_ID=$(read_id "$MARKER")
if [ "$INSTALLED_ID" != "$ARTIFACT_ID" ] || [ "$BAKED_ID" != "$ARTIFACT_ID" ]; then
    log "installing build $ARTIFACT_ID over $BAKED_ID"
    rm -f "$MARKER"
    # Each component is staged beside its destination and swapped in, so an
    # interrupted install leaves the previous one whole and rolls back by
    # restarting rather than by restoring anything by hand.
    for component in .next public node_modules server.js package.json; do
        SRC=$BUILD_DIR/$component
        DST=$APP_DIR/$component
        if [ ! -e "$SRC" ]; then
            continue
        fi
        rm -rf "$DST.incoming" "$DST.previous"
        cp -a "$SRC" "$DST.incoming"
        if [ "$(id -u)" = "0" ] && [ "$RUN_USER" != root ]; then
            # .next/cache is written at runtime (ISR, image optimisation).
            chown -R "$RUN_USER" "$DST.incoming"
        fi
        if [ -e "$DST" ]; then
            mv "$DST" "$DST.previous"
        fi
        mv "$DST.incoming" "$DST"
    done
    printf '%s\n' "$ARTIFACT_ID" > "$MARKER"
else
    log "build $ARTIFACT_ID already installed"
fi

# --- Verify before serving --------------------------------------------------
SERVED_ID=$(read_id "$APP_DIR/.next/BUILD_ID")
if [ "$SERVED_ID" != "$ARTIFACT_ID" ]; then
    log "ERROR: installed $ARTIFACT_ID but $APP_DIR/.next reports '$SERVED_ID'."
    exit 1
fi
log "serving build $SERVED_ID"

cd "$APP_DIR"
if [ "$(id -u)" = "0" ] && [ "$RUN_USER" != root ]; then
    # Root is only needed to write into /app; the server itself runs as the
    # unprivileged user the image created.
    exec su -s /bin/sh "$RUN_USER" -c 'exec "$0" "$@"' -- "$@"
fi
exec "$@"
