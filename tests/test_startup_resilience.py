"""Reboot recovery must fail closed before touching the running stack."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT / "startup_runtime.py").is_file(), "Missing canonical startup helper")
        spec = importlib.util.spec_from_file_location(
            "startup_runtime", ROOT / "startup_runtime.py"
        )
        self.ops = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.ops)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = self.root / "manifest.json"
        overlay = self.root / "overlay.json"
        overlay.write_text("{}")
        overlay.chmod(0o600)
        self.data = {"compose_files": [str(overlay)], "private_files": [str(overlay)]}
        self.save()

    def save(self):
        self.manifest.write_text(json.dumps(self.data))
        self.manifest.chmod(0o600)

    def test_missing_overlay_fails_closed(self):
        Path(self.data["compose_files"][0]).unlink()
        with self.assertRaisesRegex(self.ops.StartupError, "Required file missing"):
            self.ops.load_manifest(self.manifest)

    def test_private_overlay_permissions_fail_closed(self):
        Path(self.data["private_files"][0]).chmod(0o644)
        with self.assertRaisesRegex(self.ops.StartupError, "Private file must be owned.*0600"):
            self.ops.load_manifest(self.manifest)

    def test_artifact_mismatch_fails_closed(self):
        artifact = self.root / "artifact"
        (artifact / ".next").mkdir(parents=True)
        (artifact / "BUILD_ID").write_text("verified")
        (artifact / ".next/BUILD_ID").write_text("old")
        self.data.update(artifact=str(artifact), build_id="verified", artifact_hashes={})
        self.save()
        with self.assertRaisesRegex(self.ops.StartupError, "Frontend artifact mismatch"):
            self.ops.load_manifest(self.manifest)

    def test_artifact_content_tamper_fails_closed(self):
        import hashlib

        artifact = self.root / "artifact"
        (artifact / ".next").mkdir(parents=True)
        for name in ["BUILD_ID", ".next/BUILD_ID"]:
            (artifact / name).write_text("verified")
        (artifact / "server.js").write_text("downgraded")
        self.data.update(
            artifact=str(artifact),
            build_id="verified",
            artifact_hashes={"server.js": hashlib.sha256(b"verified").hexdigest()},
        )
        self.save()
        with self.assertRaisesRegex(self.ops.StartupError, "Frontend artifact content mismatch"):
            self.ops.load_manifest(self.manifest)

    def test_config_drift_fails_closed(self):
        self.data["file_hashes"] = {self.data["compose_files"][0]: "wrong"}
        self.save()
        with self.assertRaisesRegex(self.ops.StartupError, "Pinned runtime file changed"):
            self.ops.load_manifest(self.manifest)

    def test_healthy_start_is_idempotent_and_uses_complete_compose(self):
        from unittest.mock import Mock

        self.assertTrue(hasattr(self.ops, "recover"), "Recovery function missing")
        runtime = Mock()
        runtime.containers_ready.return_value = True
        self.ops.recover(runtime)
        self.ops.recover(runtime)
        runtime.start_containers.assert_not_called()
        runtime.wait_docker.assert_called()
        runtime.validate.assert_called()
        runtime.native_health.assert_called()
        runtime.verify.assert_called()
        self.assertEqual(
            self.ops.compose_command(
                {
                    "root": "/project",
                    "compose_files": ["a", "b", "c"],
                    "profiles": ["https"],
                    "docker": "docker",
                    "context": "desktop-linux",
                }
            ),
            [
                "docker",
                "--context",
                "desktop-linux",
                "compose",
                "--project-directory",
                "/project",
                "--project-name",
                "caal",
                "-f",
                "a",
                "-f",
                "b",
                "-f",
                "c",
                "--profile",
                "https",
            ],
        )

    def test_recovery_uses_no_recreate_no_build_and_bounded_wait(self):
        from unittest.mock import Mock

        self.assertTrue(hasattr(self.ops, "Runtime"), "Runtime missing")
        runtime = self.ops.Runtime(
            {
                "root": "/project",
                "compose_files": ["a", "b"],
                "profiles": ["https"],
                "docker": "docker",
                "context": "desktop-linux",
                "token_file": str(self.root / "token"),
            },
            token="test-token",
        )
        runtime.run = Mock(return_value="")
        runtime.start_containers()
        args = runtime.run.call_args.args[0]
        self.assertEqual(
            args[-9:],
            [
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
        )
        runtime.run.assert_called_once()
        self.assertLessEqual(runtime.run.call_args.kwargs["timeout"], 150)

    def test_runner_hides_secrets_and_ignores_shell_environment_drift(self):
        import subprocess
        from unittest.mock import patch

        runtime = self.ops.Runtime(
            {
                "root": str(self.root),
                "compose_files": ["a"],
                "profiles": [],
                "docker": "docker",
                "context": "desktop-linux",
            },
            token="private-token",
        )
        self.assertTrue(hasattr(runtime, "run"), "Safe runner missing")
        with patch(
            "subprocess.run", return_value=subprocess.CompletedProcess([], 1, "secret", "secret")
        ) as child:
            with self.assertRaisesRegex(
                self.ops.StartupError, "^Command failed; output suppressed$"
            ):
                runtime.run(["docker", "info"])
            env = child.call_args.kwargs["env"]
            self.assertEqual(env["CAAL_QWEN_TRIAL_TOKEN"], "private-token")
            self.assertNotIn("COMPOSE_FILE", env)
            self.assertNotIn("CAAL_BOOTSTRAP_ADMIN_PASSWORD_HASH", env)
            self.assertGreater(child.call_args.kwargs["timeout"], 0)

    def test_docker_readiness_is_bounded(self):
        from unittest.mock import Mock, patch

        runtime = self.ops.Runtime(
            {
                "root": str(self.root),
                "compose_files": ["a"],
                "profiles": [],
                "docker": "docker",
                "context": "desktop-linux",
            },
            token="test",
        )
        self.assertTrue(hasattr(runtime, "wait_docker"), "Docker wait missing")
        runtime.run = Mock(side_effect=self.ops.StartupError("unavailable"))
        with patch("time.sleep"):
            with self.assertRaisesRegex(self.ops.StartupError, "Docker readiness deadline"):
                runtime.wait_docker(attempts=2)
        self.assertEqual(runtime.run.call_count, 3)

    def test_native_health_checks_supervision_without_launch_or_kill(self):
        from unittest.mock import Mock

        runtime = self.ops.Runtime(
            {
                "root": str(self.root),
                "compose_files": ["a"],
                "profiles": [],
                "docker": "docker",
                "context": "desktop-linux",
                "native_labels": [
                    "com.caal.mlx-audio",
                    "com.caal.deepfilter",
                    "com.caal.qwen-trial",
                ],
                "style_hash": "approved",
            },
            token="test",
        )
        self.assertTrue(hasattr(runtime, "native_health"), "Native health missing")
        runtime.run = Mock(return_value="state = running\n pid = 123\n")
        runtime.http = Mock(
            side_effect=[{}, {}, {"status": "ok", "busy": False}, {"style_sha256": "approved"}]
        )
        runtime.native_health(attempts=1)
        for call in runtime.run.call_args_list:
            self.assertEqual(call.args[0][:2], ["/bin/launchctl", "print"])
        self.assertEqual(runtime.http.call_count, 4)
        runtime.http = Mock(side_effect=OSError("unavailable"))
        with self.assertRaisesRegex(self.ops.StartupError, "Native readiness deadline"):
            runtime.native_health(attempts=1)

    def test_live_environment_and_image_drift_refused_before_start(self):
        from unittest.mock import Mock

        runtime = self.ops.Runtime(
            {
                "root": str(self.root),
                "compose_files": ["a"],
                "profiles": [],
                "docker": "docker",
                "context": "desktop-linux",
                "containers": {"caal-agent": {"env_hash": "expected", "image": "sha256:approved"}},
            },
            token="test",
        )
        self.assertTrue(hasattr(runtime, "validate"), "Runtime validation missing")
        runtime.run = Mock(return_value="")
        runtime.inspect = Mock(
            return_value={"Config": {"Env": ["KEY=drifted"]}, "Image": "sha256:approved"}
        )
        with self.assertRaisesRegex(self.ops.StartupError, "Container environment or image drift"):
            runtime.validate()
        self.assertFalse(any("up" in call.args[0] for call in runtime.run.call_args_list))

    def test_running_but_unhealthy_is_not_restarted(self):
        from unittest.mock import Mock

        runtime = self.ops.Runtime(
            {
                "root": str(self.root),
                "compose_files": ["a"],
                "profiles": [],
                "docker": "docker",
                "context": "desktop-linux",
                "containers": {"caal-agent": {}},
            },
            token="test",
        )
        self.assertTrue(hasattr(runtime, "containers_ready"), "Container readiness missing")
        runtime.inspect = Mock(
            return_value={"State": {"Running": True, "Health": {"Status": "unhealthy"}}}
        )
        self.assertTrue(runtime.containers_ready())
        runtime.inspect = Mock(return_value={"State": {"Running": False}})
        self.assertFalse(runtime.containers_ready())

    def test_verification_refuses_served_build_downgrade(self):
        from unittest.mock import Mock

        runtime = self.ops.Runtime(
            {
                "root": str(self.root),
                "compose_files": ["a"],
                "profiles": [],
                "docker": "docker",
                "context": "desktop-linux",
                "containers": {},
                "build_id": "verified",
            },
            token="test",
        )
        self.assertTrue(hasattr(runtime, "verify"), "Final verification missing")
        runtime.http = Mock(return_value={"status": "ok"})
        runtime.run = Mock(return_value="old")
        with self.assertRaisesRegex(self.ops.StartupError, "Served frontend build mismatch"):
            runtime.verify(attempts=1)

    def test_concurrent_start_fails_without_entering_recovery(self):
        self.assertTrue(hasattr(self.ops, "startup_lock"), "Startup lock missing")
        with self.ops.startup_lock(self.root / "startup.lock"):
            with self.assertRaisesRegex(self.ops.StartupError, "Startup already in progress"):
                with self.ops.startup_lock(self.root / "startup.lock"):
                    self.fail("Second startup entered")
        with self.ops.startup_lock(self.root / "startup.lock"):
            pass

    def test_cli_check_does_not_start_services(self):
        from unittest.mock import Mock, patch

        self.assertTrue(hasattr(self.ops, "main"), "Startup CLI missing")
        self.data["lock_file"] = str(self.root / "lock")
        self.save()
        runtime = Mock()
        with patch.object(self.ops, "Runtime", return_value=runtime):
            self.ops.main(["--check"], manifest_path=self.manifest)
        runtime.wait_docker.assert_not_called()
        runtime.start_containers.assert_not_called()
        runtime.validate.assert_called_once()
        runtime.native_health.assert_called_once_with(attempts=1)
        runtime.verify.assert_called_once_with(attempts=1)

    def test_entry_scripts_share_canonical_manifest(self):
        for name in ["start-apple.sh", "ensure-services.sh"]:
            script = (ROOT / name).read_text()
            self.assertIn("startup_runtime.py", script)
            self.assertNotIn("nohup", script)
            self.assertNotIn("publish-frontend-build.sh", script)
        for name in ["durable-work.sh", "telephony-livekit.sh"]:
            script = (ROOT / name).read_text()
            self.assertIn("startup_runtime.py", script)
            self.assertNotIn("docker compose ", script[script.index("set -euo pipefail") :])

    def test_shared_compose_refuses_secret_render_and_unsafe_mutations(self):
        from unittest.mock import Mock

        runtime = self.ops.Runtime(
            {
                "root": str(self.root),
                "compose_files": ["a"],
                "profiles": [],
                "docker": "docker",
                "context": "desktop-linux",
            },
            token="test",
        )
        self.assertTrue(hasattr(runtime, "operator_compose"), "Shared compose entry missing")
        runtime.run = Mock(return_value="")
        runtime.validate = Mock()
        for args in [["config"], ["config", "--format", "json"], ["down"], ["up", "--build"]]:
            with self.assertRaises(self.ops.StartupError):
                runtime.operator_compose(args)
        runtime.run.assert_not_called()
        runtime.operator_compose(["config", "-q"])
        runtime.validate.assert_called_once()

    def test_manual_recreation_requires_independent_idle_gates(self):
        from unittest.mock import Mock

        runtime = self.ops.Runtime(
            {
                "root": str(self.root),
                "compose_files": ["a"],
                "profiles": [],
                "docker": "docker",
                "context": "desktop-linux",
            },
            token="test",
        )
        runtime.validate = Mock()
        runtime.run = Mock(return_value="")
        runtime.idle_gate = Mock(side_effect=self.ops.StartupError("active"))
        args = ["up", "-d", "--force-recreate", "--no-deps", "livekit"]
        with self.assertRaisesRegex(self.ops.StartupError, "active"):
            runtime.operator_compose(args)
        runtime.run.assert_not_called()
        runtime.idle_gate = Mock()
        runtime.operator_compose(args)
        runtime.idle_gate.assert_called_once()
        command = runtime.run.call_args.args[0]
        self.assertIn("--no-build", command)
        self.assertEqual(command[-3:], ["--no-build", "--pull", "never"])

    def test_idle_gate_rejects_sessions_rooms_and_unknown_state(self):
        from unittest.mock import Mock

        runtime = self.ops.Runtime(
            {
                "root": str(self.root),
                "compose_files": ["a"],
                "profiles": [],
                "docker": "docker",
                "context": "desktop-linux",
            },
            token="test",
        )
        self.assertTrue(hasattr(runtime, "idle_gate"), "Independent idle gate missing")
        runtime.worker_status = Mock(return_value={"active_jobs": 0})
        runtime.http = Mock(return_value={"status": "ok", "active_sessions": ["active"]})
        runtime.run = Mock(return_value='{"rooms":0,"participants":0}')
        with self.assertRaises(self.ops.StartupError):
            runtime.idle_gate()
        runtime.run.assert_not_called()
        runtime.http.return_value = {"status": "ok", "active_sessions": []}
        runtime.run.return_value = '{"rooms":1,"participants":0}'
        with self.assertRaises(self.ops.StartupError):
            runtime.idle_gate()
        runtime.run.return_value = "{}"
        with self.assertRaises(self.ops.StartupError):
            runtime.idle_gate()
        runtime.run.return_value = '{"rooms":0,"participants":0}'
        runtime.idle_gate()

    def test_launchagent_configuration_has_single_native_owner(self):
        import plistlib

        agents = Path.home() / "Library/LaunchAgents"
        for name in ["mlx-audio", "deepfilter", "qwen-trial", "cloudflared"]:
            data = plistlib.loads((agents / ("com.caal." + name + ".plist")).read_bytes())
            self.assertTrue(data["RunAtLoad"])
            self.assertTrue(data["KeepAlive"])
        startup = plistlib.loads((agents / "com.caal.startup.plist").read_bytes())
        self.assertEqual(startup["ProgramArguments"], ["/bin/bash", str(ROOT / "start-apple.sh")])
        self.assertFalse(startup["KeepAlive"])
        friday = plistlib.loads((agents / "local.friday.stt.plist").read_bytes())
        self.assertTrue(friday["Disabled"])
        self.assertFalse(friday["RunAtLoad"])
        self.assertFalse(friday["KeepAlive"])

    def test_idle_gate_refuses_actual_active_worker_job(self):
        from unittest.mock import Mock

        runtime = self.ops.Runtime(
            {
                "root": str(self.root),
                "compose_files": ["a"],
                "profiles": [],
                "docker": "docker",
                "context": "desktop-linux",
            },
            token="test",
        )
        runtime.http = Mock(return_value={"status": "ok", "active_sessions": []})
        runtime.run = Mock(return_value='{"rooms":0,"participants":0}')
        runtime.worker_status = Mock(return_value={"active_jobs": 1})
        with self.assertRaisesRegex(self.ops.StartupError, "Active agent jobs"):
            runtime.idle_gate()

    def test_worker_status_uses_live_sdk_http_endpoint(self):
        from unittest.mock import Mock

        runtime = self.ops.Runtime(
            {
                "root": str(self.root),
                "compose_files": ["a"],
                "profiles": [],
                "docker": "docker",
                "context": "desktop-linux",
            },
            token="test",
        )
        self.assertTrue(hasattr(runtime, "worker_status"), "Worker HTTP inspection missing")
        runtime.run = Mock(return_value='{"active_jobs":0,"healthy":true,"named_agent":true}')
        self.assertEqual(runtime.worker_status()["active_jobs"], 0)
        code = runtime.run.call_args.args[0][-1]
        self.assertIn("127.0.0.1:8081/worker", code)
        self.assertIn("127.0.0.1:8081/", code)

    def test_desired_image_tag_drift_fails_before_manual_recreation(self):
        from unittest.mock import Mock

        runtime = self.ops.Runtime(
            {
                "root": str(self.root),
                "compose_files": ["a"],
                "profiles": [],
                "docker": "docker",
                "context": "desktop-linux",
                "containers": {},
                "image_refs": {"caal-agent:latest": "approved"},
            },
            token="test",
        )
        runtime.run = Mock(side_effect=["", '[{"Id":"old"}]'])
        with self.assertRaisesRegex(self.ops.StartupError, "Pinned image tag drift"):
            runtime.validate()

    def test_operator_logs_keep_live_streaming(self):
        from unittest.mock import Mock

        runtime = self.ops.Runtime(
            {
                "root": str(self.root),
                "compose_files": ["a"],
                "profiles": [],
                "docker": "docker",
                "context": "desktop-linux",
            },
            token="test",
        )
        runtime.run = Mock(return_value="")
        runtime.validate = Mock()
        runtime.operator_compose(["logs", "-f", "--tail", "100", "worker"])
        self.assertTrue(runtime.run.call_args.kwargs.get("stream"))

    def test_following_logs_does_not_hold_startup_lock(self):
        from unittest.mock import Mock, patch

        self.data["lock_file"] = str(self.root / "lock")
        self.save()
        runtime = Mock()
        runtime.operator_compose.return_value = ""
        with (
            patch.object(self.ops, "Runtime", return_value=runtime),
            patch.object(self.ops, "startup_lock") as lock,
        ):
            self.ops.main(["compose", "logs", "-f", "worker"], manifest_path=self.manifest)
        lock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
