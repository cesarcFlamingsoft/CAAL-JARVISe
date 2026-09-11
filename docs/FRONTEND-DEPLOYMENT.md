# Frontend Build Deployment

How the dashboard gets onto `caal-frontend`, and why it is not the build inside
the image.

## The failure this prevents

Bringing the durable-work services up recreated `caal-frontend` from its
existing image. That image bakes a `/app/.next` from whenever it was last
built -- a 14 MB build from September 5 -- so the container came back healthy,
answering 200, and serving an application with no dashboard in it. Every other
container was current. Nothing in `docker ps`, the healthcheck or the logs said
anything was wrong, because a stale Next.js build is a perfectly healthy
Next.js app.

Nothing in this stack rebuilds the frontend image on a normal `docker compose
up -d`, `--force-recreate`, `start-apple.sh` or `ensure-services.sh` run. So the
baked build cannot be the source of truth, and "remember to rebuild" is not a
fix -- a recreate nobody thought of as a frontend deployment is exactly how this
happened.

## How it works now

1. You build the frontend locally: `cd frontend && pnpm build`.
2. `./publish-frontend-build.sh` copies that build into a self-contained
   deployment artifact at `frontend/.next-deploy/`.
3. Every composition mounts that artifact **read-only** into the frontend
   container at `/app/.next-deploy`.
4. `frontend-build-entrypoint.sh` runs as the container entrypoint. It installs
   the artifact over the baked `/app` and then starts the server -- or refuses
   to start at all.

The image is a Node runtime. The application is the artifact.

### The artifact

`publish-frontend-build.sh` assembles a complete Next.js standalone runtime:

```
frontend/.next-deploy/
  BUILD_ID        written last; its presence is the "this copy is complete" proof
  server.js       standalone server
  node_modules/   only what the build traced
  .next/          server output, manifests and static/
  public/         fonts, wake word, hand-tracking wasm and models
```

`public/` is part of the deployment, not a detail: the hand-tracking wasm and
models live there, and the stale image was missing them entirely.

`.next/cache` (about 1 GB of regenerable local state) and `.next/trace` are
excluded. The build is published as a whole or not at all: a `.next` and a
`standalone/` with different build ids is refused as a torn build.

The previous artifact is kept at `frontend/.next-deploy.previous/` for rollback.

### Why the mount is not on `/app/.next`

Docker creates a missing bind-mount source as an empty directory. Mounting
`./frontend/.next-deploy` straight onto `/app/.next` would mean that a host
without a published build gets an **empty** `.next` mounted over the runtime --
no application at all, which is worse than the stale build this replaces. It
would also make `.next/cache` read-only, which Next writes to at runtime for
ISR and image optimisation.

So the artifact is mounted at a side path and installed by the entrypoint,
component by component, each one staged next to its destination and swapped in.
An interrupted install leaves the previous one whole and rolls back on restart.

### Why the container starts as root

Installing into `/app` needs root; `/app` is root-owned in the image. The
entrypoint chowns what it installs and then drops to the image's unprivileged
`nextjs` user with `su` before it execs the server. The server process runs
exactly as it did before.

## Operations

### Deploy a new dashboard build

```bash
cd frontend && pnpm build && cd ..
./publish-frontend-build.sh
docker compose up -d --force-recreate frontend      # or -f docker-compose.apple.yaml
```

Confirm the container is serving what you published:

```bash
cat frontend/.next-deploy/BUILD_ID
docker exec caal-frontend cat /app/.next/BUILD_ID   # must match
curl -sf -o /dev/null -w '%{http_code}\n' http://localhost:3000
```

`./start-apple.sh` publishes automatically (`--if-needed`) before it brings
Docker up, so the normal restart path needs no extra step.

### Roll back

```bash
rm -rf frontend/.next-deploy
mv frontend/.next-deploy.previous frontend/.next-deploy
docker compose up -d --force-recreate frontend
```

### When the frontend refuses to start

```
[frontend-build] ERROR: no published frontend build at /app/.next-deploy.
```

That is the guard working: there is no artifact to serve, and serving the image
build instead is the regression. The container exits 1, and `restart:
unless-stopped` turns that into a visible restart loop rather than a frontend
that quietly serves the wrong thing. Run `./publish-frontend-build.sh` on the
Docker host and recreate.

If nothing was published, Docker will also have created an empty
`frontend/.next-deploy/` on the host as the bind source. Publishing over it is
safe -- the publisher moves it aside like any previous artifact.

On a genuinely fresh install -- a freshly built image, nothing published yet --
set `CAAL_FRONTEND_ALLOW_BAKED_BUILD=true` in `.env` to serve the image build.
It logs a loud warning every start. Publish a real build and remove it.

### Pinning a build

`CAAL_FRONTEND_EXPECTED_BUILD_ID=<id>` makes the container refuse to start on
any other build id. Useful to freeze a deployment on a known-good build while
someone is building on the same host.

## Tests

`tests/test_frontend_build_deployment.py` pins all of it without Docker: the
compose configuration for base, apple and cpu; that no composition ever mounts
over `/app/.next`; and the publish/install scripts end to end against a
miniature frontend tree and a miniature stale container -- including that an
empty mount, a half-written artifact and a missing artifact each stop the
container instead of quietly serving the baked build.

```bash
uv run pytest tests/test_frontend_build_deployment.py
```
