"""Current macOS deployment recovery. Never build, publish, or recreate on startup."""

import fcntl
import hashlib
import json
import os
import subprocess
import time
import urllib.request
from contextlib import contextmanager, nullcontext
from pathlib import Path


class StartupError(RuntimeError):
    pass


def load_manifest(path):
    data = json.loads(Path(path).read_text())
    for name in data["compose_files"]:
        if not Path(name).is_file():
            raise StartupError("Required file missing: " + name)
    for name in [str(path)] + data["private_files"] + data.get("compose_env_files", []):
        p = Path(name)
        if not p.is_file():
            raise StartupError("Required file missing: " + name)
        if p.stat().st_mode & 0o777 != 0o600 or p.stat().st_uid != os.getuid():
            raise StartupError("Private file must be owned by current user and mode 0600: " + name)
    if "artifact" in data:
        artifact = Path(data["artifact"])
        for marker in ["BUILD_ID", ".next/BUILD_ID"]:
            if (
                not (artifact / marker).is_file()
                or (artifact / marker).read_text().strip() != data["build_id"]
            ):
                raise StartupError("Frontend artifact mismatch")
        for name, expected in data["artifact_hashes"].items():
            p = artifact / name
            if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
                raise StartupError("Frontend artifact content mismatch")
    for name, expected in data.get("file_hashes", {}).items():
        p = Path(name)
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            raise StartupError("Pinned runtime file changed: " + name)
    return data


def compose_command(data):
    command = [
        data["docker"],
        "--context",
        data["context"],
        "compose",
        "--project-directory",
        data["root"],
        "--project-name",
        "caal",
    ]
    for name in data["compose_files"]:
        command.extend(["-f", name])
    for profile in data["profiles"]:
        command.extend(["--profile", profile])
    return command


def recover(runtime):
    runtime.wait_docker()
    runtime.validate()
    runtime.native_health()
    if not runtime.containers_ready():
        runtime.start_containers()
    runtime.verify()


class Runtime:
    def __init__(self, data, token=None):
        self.data = data
        self.compose = compose_command(data)
        self.docker = self.compose[:3]
        self.token = token if token is not None else Path(data["token_file"]).read_text().strip()

    def start_containers(self):
        self.run(
            self.compose
            + [
                "up",
                "-d",
                "--no-recreate",
                "--no-build",
                "--pull",
                "never",
                "--wait",
                "--wait-timeout",
                "120",
            ],
            timeout=150,
        )

    def compose_env(self):
        """Interpolation values for Compose that are not in the project .env.

        The company overlays use ``${VAR:?}`` so a half-configured library
        cannot start, and their values are deliberately outside the repository
        in a 0600 file. `load_manifest` has already checked the mode and owner
        of every entry here, exactly as it does for the other private files.
        Returned, never printed: `run` suppresses subprocess output.
        """
        values = {}
        for name in self.data.get("compose_env_files", []):
            for line in Path(name).read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    values[key.strip()] = value.strip()
        return values

    def run(self, args, timeout=30, stream=False):
        env = {
            "HOME": str(Path.home()),
            "PATH": (
                "/Applications/Docker.app/Contents/Resources/bin:/opt/homebrew/bin:"
                "/usr/bin:/bin:/usr/sbin:/sbin"
            ),
            "CAAL_QWEN_TRIAL_TOKEN": self.token,
        }
        env.update(self.compose_env())
        try:
            result = subprocess.run(
                args,
                cwd=self.data["root"],
                env=env,
                capture_output=not stream,
                text=True,
                timeout=None if stream else timeout,
            )
        except subprocess.TimeoutExpired:
            raise StartupError("Command timed out; output suppressed") from None
        if result.returncode:
            raise StartupError("Command failed; output suppressed")
        return result.stdout or ""

    def wait_docker(self, attempts=24):
        for attempt in range(attempts):
            try:
                self.run(self.docker + ["info"], timeout=3)
                return
            except StartupError:
                if attempt == 0:
                    try:
                        self.run(["/usr/bin/open", "-gj", "-a", "Docker"], timeout=5)
                    except StartupError:
                        pass
                if attempt + 1 < attempts:
                    time.sleep(2)
        raise StartupError("Docker readiness deadline exceeded")

    def http(self, url, authenticated=False):
        headers = {"Authorization": "Bearer " + self.token} if authenticated else {}
        request = urllib.request.Request(url, headers=headers)
        # Local checks must never inherit an HTTP proxy from a login shell.
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
            request, timeout=3
        ) as response:
            body = response.read()
            return (
                json.loads(body)
                if response.headers.get_content_type() == "application/json"
                else {}
            )

    def native_health(self, attempts=20):
        for attempt in range(attempts):
            try:
                for label in self.data["native_labels"]:
                    state = self.run(
                        ["/bin/launchctl", "print", f"gui/{os.getuid()}/{label}"], timeout=3
                    )
                    if "state = running" not in state or "pid = " not in state:
                        raise StartupError("Native supervisor not running")
                self.http("http://127.0.0.1:8001/v1/models")
                self.http("http://127.0.0.1:8002/health")
                health = self.http("http://127.0.0.1:18003/health", authenticated=True)
                voice = self.http("http://127.0.0.1:18003/voice", authenticated=True)
                if (
                    health.get("status") != "ok"
                    or voice.get("style_sha256") != self.data["style_hash"]
                ):
                    raise StartupError("Native Qwen verification failed")
                return
            except (StartupError, OSError, ValueError):
                if attempt + 1 < attempts:
                    time.sleep(2)
        raise StartupError("Native readiness deadline exceeded; launchd retains supervision")

    def inspect(self, name):
        return json.loads(self.run(self.docker + ["inspect", name]))[0]

    def validate(self):
        # Never render expanded configuration into stdout, logs, or files.
        self.run(self.compose + ["config", "-q"])
        for ref, expected_id in self.data.get("image_refs", {}).items():
            image = json.loads(self.run(self.docker + ["image", "inspect", ref]))[0]
            if image["Id"] != expected_id:
                raise StartupError("Pinned image tag drift: " + ref)
        for name, expected in self.data["containers"].items():
            container = self.inspect(name)
            env = dict(item.split("=", 1) for item in container["Config"]["Env"])
            digest = hashlib.sha256(json.dumps(env, sort_keys=True).encode()).hexdigest()
            if digest != expected["env_hash"] or container["Image"] != expected["image"]:
                raise StartupError("Container environment or image drift: " + name)

    def containers_ready(self):
        return all(self.inspect(name)["State"]["Running"] for name in self.data["containers"])

    def verify(self, attempts=12):
        for attempt in range(attempts):
            try:
                for name in self.data["containers"]:
                    state = self.inspect(name)["State"]
                    if (
                        not state["Running"]
                        or state.get("Health", {}).get("Status", "healthy") != "healthy"
                    ):
                        raise StartupError("Container not healthy: " + name)
                if self.http("http://127.0.0.1:8889/health").get("status") != "ok":
                    raise StartupError("Agent not healthy")
                self.http("http://127.0.0.1:3000")
                self.http("http://127.0.0.1:7880")
                break
            except (StartupError, OSError, ValueError):
                if attempt + 1 == attempts:
                    raise StartupError("Container health deadline exceeded") from None
                time.sleep(2)
        served = self.run(
            self.docker + ["exec", "caal-frontend", "cat", "/app/.next/BUILD_ID"]
        ).strip()
        if served != self.data["build_id"]:
            raise StartupError("Served frontend build mismatch")
        self.run(
            self.docker
            + [
                "exec",
                "caal-worker",
                "python",
                "-c",
                "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8890/healthz',timeout=5)",
            ]
        )

    def worker_status(self):
        code = """import urllib.request,json
with urllib.request.urlopen('http://127.0.0.1:8081/',timeout=5) as r:
 healthy=r.status==200
with urllib.request.urlopen('http://127.0.0.1:8081/worker',timeout=5) as r:
 w=json.load(r)
print(json.dumps({'healthy':healthy,'named_agent':w.get('agent_name')=='caal','active_jobs':w.get('active_jobs',0)}))
"""
        return json.loads(
            self.run(self.docker + ["exec", "caal-agent", "/app/.venv/bin/python", "-c", code])
        )

    def idle_gate(self):
        if self.worker_status().get("active_jobs") != 0:
            raise StartupError("Active agent jobs or unknown worker state")
        health = self.http("http://127.0.0.1:8889/health")
        if health.get("status") != "ok" or health.get("active_sessions") != []:
            raise StartupError("Active sessions or unknown agent state")
        code = """import asyncio,os,json
from livekit import api
async def main():
 c=api.LiveKitAPI(os.environ['LIVEKIT_URL'],os.environ['LIVEKIT_API_KEY'],os.environ['LIVEKIT_API_SECRET'])
 try:
  r=await c.room.list_rooms(api.ListRoomsRequest())
  print(json.dumps({'rooms':len(r.rooms),'participants':sum(x.num_participants for x in r.rooms)}))
 finally: await c.aclose()
asyncio.run(main())
"""
        rooms = json.loads(
            self.run(self.docker + ["exec", "caal-agent", "/app/.venv/bin/python", "-c", code])
        )
        if rooms != {"rooms": 0, "participants": 0}:
            raise StartupError("Active rooms or unknown LiveKit state")

    def operator_compose(self, args):
        if args in [["config", "-q"], ["config", "--quiet"]]:
            self.validate()
            return ""
        if args in [
            ["up", "-d", "--force-recreate", "--no-deps", "livekit"],
            ["up", "-d", "--no-deps", "worker"],
            ["restart", "sip"],
        ]:
            self.validate()
            self.idle_gate()
            extra = ["--no-build", "--pull", "never"] if args[0] == "up" else []
            return self.run(self.compose + args + extra, timeout=150)
        if not args or args[0] not in ["ps", "exec", "logs"]:
            raise StartupError("Unsafe Compose action refused; use a separately gated deployment")
        self.validate()
        return self.run(self.compose + args, timeout=300, stream=args[0] == "logs")


@contextmanager
def startup_lock(path):
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise StartupError("Startup already in progress") from None
        yield
    finally:
        os.close(fd)


def main(args=None, manifest_path=None):
    import sys

    args = sys.argv[1:] if args is None else args
    path = (
        Path(manifest_path) if manifest_path else Path.home() / ".config/caal/startup/manifest.json"
    )
    data = load_manifest(path)
    lock = nullcontext() if args[:2] == ["compose", "logs"] else startup_lock(data["lock_file"])
    with lock:
        runtime = Runtime(data)
        if args and args[0] == "compose":
            print(runtime.operator_compose(args[1:]), end="")
        elif args in [["--check"], ["--dry-run"]]:
            runtime.validate()
            runtime.native_health(attempts=1)
            runtime.verify(attempts=1)
            print("Startup check passed; no services changed.")
        elif not args:
            recover(runtime)
            print("Startup complete; verified current runtime and frontend artifact.")
        else:
            raise StartupError(
                "Unsupported startup action; builds, publication and recreation "
                "require a separate reviewed deployment"
            )


if __name__ == "__main__":
    try:
        main()
    except (StartupError, OSError, ValueError, KeyError):
        # No tracebacks: subprocess/config exceptions can carry credentials.
        import sys

        print(
            "Startup refused: validation, readiness or command failed; no automatic recreation. "
            "Run --check and inspect the private manifest.",
            file=sys.stderr,
        )
        sys.exit(1)
