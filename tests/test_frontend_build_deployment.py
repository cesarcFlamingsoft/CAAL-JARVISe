"""The deployed frontend build, and why the dashboard disappeared.

Bringing the durable-work services up recreated ``caal-frontend`` from its
image. The image carries a ``/app/.next`` baked at build time, so the container
came back serving a months-old build id while every other container was
current: the dashboard was simply gone, and nothing in ``docker ps`` looked
wrong because the stale build is a perfectly healthy Next.js app.

Nothing in the stack rebuilds the frontend image on a normal
``docker compose up -d`` / ``--force-recreate`` / ``start-apple.sh`` run, so the
baked build can never be the source of truth. The build the operator verified
locally is published once, as a versioned artifact, and mounted read-only into
every frontend container. Pinned properties:

* the artifact is mounted at a *side* path, never over ``/app/.next`` -- Docker
  silently creates a missing bind-mount source, and an empty directory mounted
  over the runtime would break the frontend far worse than a stale one does;
* a recreate installs the artifact over whatever the image baked, so the build
  id the container serves is the build id the operator published;
* a missing or half-written artifact fails the container loudly instead of
  falling back to the baked build -- that silent fallback is the regression;
* falling back to the baked image build stays possible, but only when an
  operator asks for it (``CAAL_FRONTEND_ALLOW_BAKED_BUILD``), e.g. on a first
  run from a freshly built image with nothing published yet;
* every composition that runs a frontend -- base, apple, cpu -- does all of the
  above, because the regression came from one composition differing.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]
ENTRYPOINT = ROOT / "frontend-build-entrypoint.sh"
PUBLISH = ROOT / "publish-frontend-build.sh"
START_APPLE = ROOT / "start-apple.sh"

FRONTEND_COMPOSE = {
    "base": ROOT / "docker-compose.yaml",
    "apple": ROOT / "docker-compose.apple.yaml",
    "cpu": ROOT / "docker-compose.cpu.yaml",
}

ARTIFACT_MOUNT = "./frontend/.next-deploy:/app/.next-deploy:ro"
ENTRYPOINT_MOUNT = "./frontend-build-entrypoint.sh:/app/frontend-build-entrypoint.sh:ro"

STALE_ID = "ypwrhVbm-aJlACr-uIvHX"
CURRENT_ID = "FFVn4DEzTllJ9-B740eBx"


# ---------------------------------------------------------------------------
# Compose configuration
# ---------------------------------------------------------------------------


def _frontend(compose: Path) -> dict:
    return yaml.safe_load(compose.read_text())["services"]["frontend"]


@pytest.mark.parametrize("name", sorted(FRONTEND_COMPOSE))
def test_frontend_mounts_the_published_build_read_only(name: str) -> None:
    """Every composition mounts the published artifact, read-only."""
    service = _frontend(FRONTEND_COMPOSE[name])
    assert ARTIFACT_MOUNT in service["volumes"]
    assert ENTRYPOINT_MOUNT in service["volumes"]


@pytest.mark.parametrize("name", sorted(FRONTEND_COMPOSE))
def test_no_composition_mounts_over_the_next_runtime(name: str) -> None:
    """``/app/.next`` is never a bind target: a missing source would empty it."""
    service = _frontend(FRONTEND_COMPOSE[name])
    targets = {str(volume).split(":")[1] for volume in service["volumes"]}
    assert "/app/.next" not in targets


@pytest.mark.parametrize("name", sorted(FRONTEND_COMPOSE))
def test_frontend_installs_the_build_before_serving(name: str) -> None:
    """The entrypoint is the installer, and the command is still the server.

    Compose clears the image ``CMD`` whenever ``entrypoint`` is overridden, so
    the command has to be spelled out or the container starts with no server.
    """
    service = _frontend(FRONTEND_COMPOSE[name])
    assert service["entrypoint"] == ["/bin/sh", "/app/frontend-build-entrypoint.sh"]
    assert service["command"] == ["node", "server.js"]
    # Installing into /app needs root; the entrypoint drops back to the image
    # user before it execs the server.
    assert str(service["user"]) == "0:0"


def test_startup_script_publishes_before_composing() -> None:
    """``start-apple.sh`` publishes the local build before it brings Docker up."""
    text = START_APPLE.read_text()
    publish = text.find("publish-frontend-build.sh")
    compose_up = text.find("docker compose $COMPOSE_FILES $DOCKER_PROFILE up -d")
    assert publish != -1, "start-apple.sh never publishes the frontend build"
    assert compose_up != -1
    assert publish < compose_up


@pytest.mark.parametrize("script", [ENTRYPOINT, PUBLISH])
def test_scripts_are_executable(script: Path) -> None:
    assert script.exists(), f"{script.name} is missing"
    assert script.stat().st_mode & stat.S_IXUSR


# ---------------------------------------------------------------------------
# Fixtures: a frontend tree and a container filesystem, in miniature
# ---------------------------------------------------------------------------


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _fake_frontend(root: Path, build_id: str = CURRENT_ID) -> Path:
    """A ``frontend/`` directory shaped like a real ``next build`` output.

    Including the quirk that breaks a naive copy: ``.next/standalone`` nests the
    application under its own traced root, so the application directory is not
    ``standalone/`` itself.
    """
    frontend = root / "frontend"
    app = frontend / ".next" / "standalone" / "caal" / "frontend"
    _write(frontend / ".next" / "BUILD_ID", build_id)
    _write(frontend / ".next" / "static" / "chunks" / "main.js", "// static")
    _write(frontend / ".next" / "cache" / "huge.bin", "x" * 1024)
    _write(frontend / ".next" / "trace", "trace")
    _write(app / "server.js", "// server")
    _write(app / "package.json", "package json placeholder")
    _write(app / "node_modules" / "next" / "index.js", "// next")
    _write(app / ".next" / "BUILD_ID", build_id)
    _write(app / ".next" / "server" / "app" / "dashboard.js", "// dashboard")
    _write(frontend / "public" / "hands" / "wasm" / "vision.wasm", "wasm")
    return frontend


def _fake_container(root: Path, baked_id: str = STALE_ID) -> Path:
    """An ``/app`` shaped like the image runner stage, with a stale build."""
    app = root / "app"
    _write(app / ".next" / "BUILD_ID", baked_id)
    _write(app / ".next" / "server" / "app" / "old.js", "// old")
    _write(app / "server.js", "// baked server")
    _write(app / "public" / "caal-logo.svg", "<svg/>")
    _write(app / "node_modules" / "next" / "index.js", "// baked next")
    return app


FAKE_NODE = """#!/bin/sh
printf '%s\\n' "$@" > "$FAKE_NODE_ARGS"
pwd > "$FAKE_NODE_CWD"
"""


def _fake_node(root: Path) -> Path:
    """A ``node`` that records how it was invoked instead of serving."""
    bindir = root / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    node = bindir / "node"
    node.write_text(FAKE_NODE)
    node.chmod(0o755)
    return bindir


def _publish(frontend: Path, *args: str):
    env = dict(os.environ, CAAL_FRONTEND_DIR=str(frontend))
    return subprocess.run(
        ["/bin/sh", str(PUBLISH), *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _start(app: Path, artifact: Path, tmp_path: Path, **environment: str):
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(_fake_node(tmp_path)), env["PATH"]])
    env["FAKE_NODE_ARGS"] = str(tmp_path / "node-args")
    env["FAKE_NODE_CWD"] = str(tmp_path / "node-cwd")
    env["CAAL_FRONTEND_APP_DIR"] = str(app)
    env["CAAL_FRONTEND_BUILD_DIR"] = str(artifact)
    env.pop("CAAL_FRONTEND_ALLOW_BAKED_BUILD", None)
    env.pop("CAAL_FRONTEND_EXPECTED_BUILD_ID", None)
    env.update(environment)
    return subprocess.run(
        ["/bin/sh", str(ENTRYPOINT), "node", "server.js"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _served(app: Path) -> str:
    build_id = app / ".next" / "BUILD_ID"
    return build_id.read_text().strip() if build_id.exists() else ""


# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------


def test_publish_collects_the_whole_runtime(tmp_path: Path) -> None:
    """The artifact is a self-contained runtime, not just ``.next``.

    ``public/`` matters as much as ``.next``: the hand-tracking wasm and models
    live there, and the stale image was missing them entirely.
    """
    frontend = _fake_frontend(tmp_path)
    done = _publish(frontend)
    assert done.returncode == 0, done.stderr

    artifact = frontend / ".next-deploy"
    assert (artifact / "BUILD_ID").read_text().strip() == CURRENT_ID
    assert (artifact / ".next" / "BUILD_ID").read_text().strip() == CURRENT_ID
    assert (artifact / ".next" / "static" / "chunks" / "main.js").exists()
    assert (artifact / ".next" / "server" / "app" / "dashboard.js").exists()
    assert (artifact / "public" / "hands" / "wasm" / "vision.wasm").exists()
    assert (artifact / "node_modules" / "next" / "index.js").exists()
    assert (artifact / "server.js").exists()
    # The 1 GB local build cache is a local artefact, not part of a deployment.
    assert not (artifact / ".next" / "cache").exists()


def test_publish_keeps_the_previous_artifact_for_rollback(tmp_path: Path) -> None:
    frontend = _fake_frontend(tmp_path)
    assert _publish(frontend).returncode == 0
    _write(frontend / ".next" / "BUILD_ID", "second-build-id")
    _write(
        frontend / ".next" / "standalone" / "caal" / "frontend" / ".next" / "BUILD_ID",
        "second-build-id",
    )
    assert _publish(frontend).returncode == 0

    current = frontend / ".next-deploy" / "BUILD_ID"
    assert current.read_text().strip() == "second-build-id"
    previous = frontend / ".next-deploy.previous" / "BUILD_ID"
    assert previous.read_text().strip() == CURRENT_ID


def test_publish_refuses_a_torn_build(tmp_path: Path) -> None:
    """A ``.next`` and a ``standalone`` from different builds never ship."""
    frontend = _fake_frontend(tmp_path)
    _write(frontend / ".next" / "BUILD_ID", "outer-id")
    done = _publish(frontend)
    assert done.returncode != 0
    assert not (frontend / ".next-deploy").exists()


def test_publish_if_needed_keeps_a_good_artifact_without_a_local_build(
    tmp_path: Path,
) -> None:
    """``--if-needed`` lets a startup script run on a host that never built."""
    frontend = _fake_frontend(tmp_path)
    assert _publish(frontend).returncode == 0
    subprocess.run(["rm", "-rf", str(frontend / ".next")], check=True)

    done = _publish(frontend, "--if-needed")
    assert done.returncode == 0, done.stderr
    kept = frontend / ".next-deploy" / "BUILD_ID"
    assert kept.read_text().strip() == CURRENT_ID

    # With no artifact to fall back on there is nothing to deploy, and saying so
    # beats starting a container that serves the wrong thing.
    subprocess.run(["rm", "-rf", str(frontend / ".next-deploy")], check=True)
    assert _publish(frontend, "--if-needed").returncode != 0


# ---------------------------------------------------------------------------
# Recreating the container
# ---------------------------------------------------------------------------


def test_recreate_serves_the_published_build(tmp_path: Path) -> None:
    """The regression, end to end: publish, recreate, keep the dashboard."""
    frontend = _fake_frontend(tmp_path)
    assert _publish(frontend).returncode == 0
    app = _fake_container(tmp_path)

    done = _start(app, frontend / ".next-deploy", tmp_path)
    assert done.returncode == 0, done.stderr + done.stdout
    assert _served(app) == CURRENT_ID
    assert (app / ".next" / "server" / "app" / "dashboard.js").exists()
    assert not (app / ".next" / "server" / "app" / "old.js").exists()
    assert (app / "public" / "hands" / "wasm" / "vision.wasm").exists()
    assert (tmp_path / "node-args").read_text().split() == ["server.js"]
    assert (tmp_path / "node-cwd").read_text().strip() == str(app)


def test_recreate_is_idempotent(tmp_path: Path) -> None:
    frontend = _fake_frontend(tmp_path)
    assert _publish(frontend).returncode == 0
    app = _fake_container(tmp_path)

    assert _start(app, frontend / ".next-deploy", tmp_path).returncode == 0
    second = _start(app, frontend / ".next-deploy", tmp_path)
    assert second.returncode == 0, second.stderr
    assert _served(app) == CURRENT_ID


def test_an_empty_mount_never_reaches_the_runtime(tmp_path: Path) -> None:
    """Docker creates a missing bind source. That must not start a server."""
    app = _fake_container(tmp_path)
    empty = tmp_path / "frontend" / ".next-deploy"
    empty.mkdir(parents=True)

    done = _start(app, empty, tmp_path)
    assert done.returncode != 0
    assert not (tmp_path / "node-args").exists(), "served with no published build"
    # The runtime is left exactly as it was, so a rollback is a restart away.
    assert _served(app) == STALE_ID
    assert "publish-frontend-build.sh" in done.stderr


def test_a_half_written_artifact_never_reaches_the_runtime(tmp_path: Path) -> None:
    """``BUILD_ID`` is written last, so a torn copy has no marker to trust."""
    frontend = _fake_frontend(tmp_path)
    assert _publish(frontend).returncode == 0
    artifact = frontend / ".next-deploy"
    (artifact / "BUILD_ID").unlink()
    app = _fake_container(tmp_path)

    done = _start(app, artifact, tmp_path)
    assert done.returncode != 0
    assert _served(app) == STALE_ID


def test_the_baked_build_is_never_a_silent_fallback(tmp_path: Path) -> None:
    """The whole regression, in one test."""
    app = _fake_container(tmp_path)
    missing = tmp_path / "nothing-published"

    refused = _start(app, missing, tmp_path)
    assert refused.returncode != 0
    assert not (tmp_path / "node-args").exists()

    allowed = _start(app, missing, tmp_path, CAAL_FRONTEND_ALLOW_BAKED_BUILD="true")
    assert allowed.returncode == 0, allowed.stderr
    assert (tmp_path / "node-args").exists()
    assert _served(app) == STALE_ID
    assert "baked" in allowed.stderr.lower()


def test_an_unexpected_build_id_stops_the_container(tmp_path: Path) -> None:
    """An operator can pin the build id a deployment is allowed to serve."""
    frontend = _fake_frontend(tmp_path)
    assert _publish(frontend).returncode == 0
    app = _fake_container(tmp_path)

    done = _start(
        app,
        frontend / ".next-deploy",
        tmp_path,
        CAAL_FRONTEND_EXPECTED_BUILD_ID="some-other-build",
    )
    assert done.returncode != 0
    assert _served(app) == STALE_ID

    pinned = _start(
        app,
        frontend / ".next-deploy",
        tmp_path,
        CAAL_FRONTEND_EXPECTED_BUILD_ID=CURRENT_ID,
    )
    assert pinned.returncode == 0, pinned.stderr
    assert _served(app) == CURRENT_ID
