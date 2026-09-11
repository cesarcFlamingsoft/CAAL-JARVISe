#!/bin/sh
# Publish the locally built Next.js frontend as a deployable artifact.
#
#   ./publish-frontend-build.sh              publish frontend/.next
#   ./publish-frontend-build.sh --if-needed  publish if there is a local build,
#                                            otherwise keep the published one
#
# Nothing in the stack rebuilds the frontend image on a normal
# `docker compose up -d`, so the build baked into that image drifts: recreating
# caal-frontend during an unrelated deployment brought back a months-old build
# and the dashboard vanished. The frontend containers therefore serve *this*
# artifact, not the baked one -- see frontend-build-entrypoint.sh and
# docs/FRONTEND-DEPLOYMENT.md.
#
# The artifact is a self-contained Next.js standalone runtime:
#
#   frontend/.next-deploy/
#     BUILD_ID        the build id, written last: its presence means complete
#     server.js       standalone server
#     node_modules/   only what the build traced
#     .next/          server output, manifests and static/
#     public/         fonts, wake word, hand-tracking wasm and models
#
# The previous artifact is kept at frontend/.next-deploy.previous for rollback.

set -e

ROOT=$(cd "$(dirname "$0")" && pwd)
FRONTEND=$CAAL_FRONTEND_DIR
if [ -z "$FRONTEND" ]; then
    FRONTEND=$ROOT/frontend
fi

BUILD=$FRONTEND/.next
ARTIFACT=$FRONTEND/.next-deploy
PREVIOUS=$ARTIFACT.previous
STAGING=$ARTIFACT.staging.$$

IF_NEEDED=false
for arg in "$@"; do
    case "$arg" in
        --if-needed)
            IF_NEEDED=true
            ;;
        *)
            echo "publish-frontend-build.sh: unknown option: $arg" >&2
            exit 2
            ;;
    esac
done

trap 'rm -rf "$STAGING"' EXIT INT TERM

# --- Is there anything to publish? ------------------------------------------
if [ ! -s "$BUILD/BUILD_ID" ]; then
    if [ "$IF_NEEDED" = true ] && [ -s "$ARTIFACT/BUILD_ID" ]; then
        echo "No local build in $BUILD; keeping published build $(cat "$ARTIFACT/BUILD_ID")"
        exit 0
    fi
    echo "publish-frontend-build.sh: no build in $BUILD -- run 'pnpm build' in $FRONTEND first" >&2
    exit 1
fi

BUILD_ID=$(cat "$BUILD/BUILD_ID")

# --- Locate the standalone application root ---------------------------------
# Next traces the application under its inferred workspace root, so the server
# is at .next/standalone/<some>/<path>/server.js rather than at the top. Find it
# rather than hard-code a path that moves with the checkout location. Traced
# dependencies ship their own server.js files; those are not the entrypoint.
SERVER=$(find "$BUILD/standalone" -name node_modules -prune -o \
    -name server.js -type f -print 2>/dev/null | sort | head -1)
if [ -z "$SERVER" ]; then
    echo "publish-frontend-build.sh: no standalone server.js under $BUILD/standalone" >&2
    exit 1
fi
APP_DIR=$(dirname "$SERVER")

# --- Refuse anything that is not one coherent build -------------------------
if [ ! -s "$APP_DIR/.next/BUILD_ID" ]; then
    echo "publish-frontend-build.sh: $APP_DIR/.next/BUILD_ID is missing -- rebuild" >&2
    exit 1
fi
STANDALONE_ID=$(cat "$APP_DIR/.next/BUILD_ID")
if [ "$STANDALONE_ID" != "$BUILD_ID" ]; then
    echo "publish-frontend-build.sh: torn build -- .next is $BUILD_ID but standalone is $STANDALONE_ID" >&2
    exit 1
fi
if [ ! -d "$BUILD/static" ]; then
    echo "publish-frontend-build.sh: $BUILD/static is missing -- rebuild" >&2
    exit 1
fi

# --- Stage ------------------------------------------------------------------
rm -rf "$STAGING"
mkdir -p "$STAGING"
cp -a "$APP_DIR/." "$STAGING/"
rm -rf "$STAGING/.next/static" "$STAGING/public"
cp -a "$BUILD/static" "$STAGING/.next/static"
if [ -d "$FRONTEND/public" ]; then
    cp -a "$FRONTEND/public" "$STAGING/public"
fi
# The local build cache is ~1 GB of regenerable local state and the trace is a
# profiling artefact. Neither belongs in a deployment.
rm -rf "$STAGING/.next/cache" "$STAGING/.next/trace"

if [ ! -s "$STAGING/.next/BUILD_ID" ] || [ ! -f "$STAGING/server.js" ]; then
    echo "publish-frontend-build.sh: staged runtime is incomplete -- not publishing" >&2
    exit 1
fi

# Written last: frontend-build-entrypoint.sh treats this marker as the proof
# that the artifact is complete, so a torn copy is never installed.
printf '%s\n' "$BUILD_ID" > "$STAGING/BUILD_ID"

# --- Swap -------------------------------------------------------------------
rm -rf "$PREVIOUS"
if [ -e "$ARTIFACT" ]; then
    mv "$ARTIFACT" "$PREVIOUS"
fi
mv "$STAGING" "$ARTIFACT"

echo "Published frontend build $BUILD_ID to $ARTIFACT"
echo "Deploy it: docker compose up -d --force-recreate frontend"
