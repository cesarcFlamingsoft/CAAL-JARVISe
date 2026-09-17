"""The company library's runtime dependency and its deployment topology, declared.

Two gaps this covers, both found by inspecting the real deployment rather than
the design:

**pypdf.** ``caal.company.extraction`` reads PDFs through pypdf's page tree,
and the host ``.venv`` has it because someone installed it by hand. The runtime
image does not: ``reports/company-knowledge/verify-topology.json`` recorded
``"pypdf_installed": false`` inside the running agent container. An ad-hoc
``pip install`` in a container disappears the next time the container is
recreated, so a PDF that indexed on Monday is ``needs_ocr`` on Tuesday. The
dependency is therefore *declared*, with a supported version range, and the
image installs it in a build layer.

**The reader process.** The model's company tools go through an MCP service
that must be a **separate, read-only process**: exactly one writer holds the
``flock``, and a reader that could write is the defect the store rewrite fixed.
That process needs the agent's network namespace (the service is on private
loopback 8791, published nowhere), the company library volume, and the
identity database the owner check resolves against. This test holds the compose
overlay that declares it to those requirements.

Nothing here starts, recreates or deploys anything. It reads files.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
DOCKERFILE = ROOT / "Dockerfile"
OVERLAY = ROOT / "docker-compose.company.yaml"


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text())


@pytest.fixture(scope="module")
def overlay() -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(OVERLAY.read_text())


def _environment(service: dict) -> dict[str, str]:
    """The service's environment as a mapping.

    Every compose file in this repository writes it as a ``- KEY=value`` list,
    which is what an operator reading the file will expect to see; this turns
    that into something a test can assert on.
    """
    raw = service["environment"]
    if isinstance(raw, dict):
        return {str(key): str(value) for key, value in raw.items()}
    pairs = {}
    for item in raw:
        name, _, value = str(item).partition("=")
        pairs[name] = value
    return pairs


# --- pypdf -------------------------------------------------------------------------------------


def test_pypdf_is_declared_with_a_supported_version_range(pyproject):
    extras = pyproject["project"]["optional-dependencies"]
    assert "company" in extras, "the company extra must exist"
    declared = [item for item in extras["company"] if item.lower().startswith("pypdf")]
    assert len(declared) == 1, extras["company"]
    specifier = declared[0]
    # A lower bound, because the page-tree API this uses is not in every
    # version, and an upper bound, because a major bump is a breaking change
    # in a library that parses untrusted files.
    assert re.search(r">=\s*\d+", specifier), specifier
    assert re.search(r"<\s*\d+", specifier), specifier


def test_the_extraction_module_is_the_only_thing_that_needs_it():
    """If this stops being true the extra is the wrong shape."""
    importers = [
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "src").rglob("*.py")
        if re.search(r"^\s*(import|from)\s+pypdf", path.read_text(), re.MULTILINE)
    ]
    assert importers == ["src/caal/company/extraction.py"], importers


def test_the_runtime_image_installs_it_in_a_build_layer():
    """Durable: part of the image, not a pip command run inside a live container."""
    dockerfile = DOCKERFILE.read_text()
    assert re.search(r"uv pip install[^\n]*pypdf", dockerfile), (
        "the Dockerfile must install pypdf"
    )


def test_the_declared_range_and_the_image_agree(pyproject):
    extras = pyproject["project"]["optional-dependencies"]["company"]
    specifier = next(item for item in extras if item.lower().startswith("pypdf"))
    bound = re.search(r">=\s*([\d.]+)", specifier).group(1)
    assert bound in DOCKERFILE.read_text(), (
        "the image must install the version the project declares"
    )


def test_extraction_reports_pdf_support_from_the_import_not_from_a_flag():
    """The status route tells the owner whether PDFs work; it must not guess."""
    from caal.company.extraction import extraction_capabilities

    capabilities = extraction_capabilities()
    assert set(capabilities) >= {"pdf_supported"}
    try:
        import pypdf  # noqa: F401
    except ImportError:
        assert capabilities["pdf_supported"] is False
    else:
        assert capabilities["pdf_supported"] is True


# --- the reader process topology ---------------------------------------------------------------


def test_the_overlay_declares_exactly_one_extra_service(overlay):
    assert sorted(overlay["services"]) == ["company-mcp"]


def test_the_reader_shares_the_agent_network_namespace_and_publishes_nothing(overlay):
    service = overlay["services"]["company-mcp"]
    assert service["network_mode"] == "service:agent"
    assert "ports" not in service, "the MCP service must not be published"
    assert "networks" not in service, "network_mode and networks are mutually exclusive"


def test_the_reader_runs_the_read_only_entrypoint_explicitly(overlay):
    service = overlay["services"]["company-mcp"]
    assert service["command"] == ["python", "-m", "caal.company.server_main"]
    assert _environment(service)["CAAL_COMPANY_ROLE"] == "reader"


def test_the_reader_cannot_be_the_writer(overlay):
    """Belt and braces: the role says reader and the mount is read only."""
    service = overlay["services"]["company-mcp"]
    library = [
        mount
        for mount in service["volumes"]
        if str(mount).startswith("caal-company:")
    ]
    assert library == ["caal-company:/app/company:ro"]


def test_the_reader_reads_the_identity_database_read_only(overlay):
    """The provisioned owner is resolved against the authoritative identity store."""
    service = overlay["services"]["company-mcp"]
    assert "caal-memory:/app/data:ro" in service["volumes"]


def test_the_read_only_mount_still_leaves_somewhere_to_write_a_nonce(overlay):
    """Principal replay protection writes; a read-only library cannot hold it."""
    service = overlay["services"]["company-mcp"]
    dedicated = [
        mount for mount in service["volumes"] if "caal-company-reader" in str(mount)
    ]
    assert dedicated == ["caal-company-reader:/app/reader-state"]
    assert _environment(service)["CAAL_COMPANY_READER_STATE_DIR"] == "/app/reader-state"


def test_the_company_volume_is_named_and_is_not_offered_to_the_frontend(overlay):
    assert "caal-company" in overlay["volumes"]
    assert overlay["volumes"]["caal-company"]["name"] == "caal-company"
    assert "frontend" not in overlay["services"]


def test_the_owner_and_the_keys_must_be_provided_rather_than_defaulted(overlay):
    """No generated key and no guessed owner: an unset variable stays unset."""
    environment = _environment(overlay["services"]["company-mcp"])
    for name in (
        "CAAL_COMPANY_OWNER_USER_ID",
        "CAAL_COMPANY_LIBRARY_KEYS",
        "CAAL_COMPANY_MCP_TOKEN",
    ):
        value = environment[name]
        assert value == "${%s:?%s must be provided}" % (name, name), (name, value)


def test_the_overlay_adds_no_service_and_changes_no_existing_one(overlay):
    """It is an overlay, not a redefinition: nothing else appears in it."""
    assert set(overlay) <= {"services", "volumes"}
    assert set(overlay["services"]) == {"company-mcp"}


def test_the_reader_uses_the_same_image_as_the_agent(overlay):
    """One build, two roles: the reader is the same code with a different role."""
    service = overlay["services"]["company-mcp"]
    assert service["image"] == "caal-agent:latest"
    assert "build" not in service, "the overlay must not trigger a second build"


def test_the_reader_is_supervised_the_way_every_other_service_is(overlay):
    service = overlay["services"]["company-mcp"]
    assert service["restart"] == "unless-stopped"
    assert service["depends_on"] == ["agent"]


def test_the_reader_does_not_inherit_the_shared_env_file(overlay):
    """A read-only query service has no use for the deployment's other secrets.

    ``./.env`` carries every unrelated integration secret this deployment has --
    provider OAuth, telephony, Telegram, the harness -- and Compose can only
    append to an ``env_file`` list, never subtract. So the reader declares no
    shared env file at all and names the narrow set it genuinely needs.
    """
    service = overlay["services"]["company-mcp"]
    assert "env_file" not in service


def test_the_reader_names_every_variable_it_needs_and_no_others(overlay):
    """The whole environment is the narrow surface, written down explicitly."""
    environment = _environment(overlay["services"]["company-mcp"])
    assert set(environment) == {
        "CAAL_COMPANY_ROLE",
        "CAAL_COMPANY_LIBRARY_DIR",
        "CAAL_COMPANY_READER_STATE_DIR",
        "CAAL_COMPANY_MCP_HOST",
        "CAAL_COMPANY_MCP_PORT",
        "CAAL_DATA_DIR",
        "CAAL_COMPANY_OWNER_USER_ID",
        "CAAL_COMPANY_LIBRARY_KEYS",
        "CAAL_COMPANY_MCP_TOKEN",
        "CAAL_INTERNAL_AUTH_SECRET",
        "CAAL_PROFILE_ENCRYPTION_KEYS",
        "CAAL_BOOTSTRAP_ADMIN_EMAIL",
    }
    # It reads role and status for an already-signed principal; it never
    # authenticates anybody, so the bootstrap password hash is not passed.
    assert "CAAL_BOOTSTRAP_ADMIN_PASSWORD_HASH" not in environment


def test_the_reader_is_pinned_to_the_reader_role(overlay):
    """With no env file to override it, the declared role is the only role."""
    assert _environment(overlay["services"]["company-mcp"])["CAAL_COMPANY_ROLE"] == "reader"


def test_the_reader_drops_privilege_rather_than_running_the_image_entrypoint(overlay):
    """`entrypoint.sh` chowns `$CAAL_DATA_DIR`, which this service mounts `:ro`.

    Under ``set -e`` that chown fails and the reader never starts, so the
    reader bypasses it and drops to the unprivileged user directly. It does not
    need the writer's config bootstrap: no /app/config, no settings.json.
    """
    service = overlay["services"]["company-mcp"]
    assert service["entrypoint"] == ["/usr/sbin/gosu", "agent"]
    assert service["command"] == ["python", "-m", "caal.company.server_main"]
    assert "user" not in service, "privilege is dropped by the entrypoint, not re-granted"


# --- what a read-only mount actually requires of the code --------------------------------------


def test_a_reader_opens_a_read_only_library_directory(tmp_path):
    """The `:ro` mount is real: `from_env` must not try to create or chmod it."""
    import os
    import stat

    from caal import profile_crypto
    from caal.company.config import CompanyConfig

    library = tmp_path / "library"
    (library / "sources").mkdir(parents=True)
    os.chmod(library, stat.S_IRUSR | stat.S_IXUSR)
    try:
        config = CompanyConfig.from_env(
            {
                "CAAL_COMPANY_LIBRARY_DIR": str(library),
                "CAAL_COMPANY_LIBRARY_KEYS": profile_crypto.generate_key_material(version=1),
                "CAAL_COMPANY_OWNER_USER_ID": "usr_" + "a1" * 12,
                "CAAL_COMPANY_ROLE": "reader",
                "CAAL_COMPANY_READER_STATE_DIR": str(tmp_path / "reader-state"),
            }
        )
    finally:
        os.chmod(library, stat.S_IRWXU)
    assert config.role == "reader"
    assert config.data_dir == library


def test_the_nonce_store_goes_to_the_dedicated_writable_directory(tmp_path):
    """Replay protection writes; the library mount cannot hold it for a reader."""
    from caal import profile_crypto
    from caal.company.config import CompanyConfig

    state = tmp_path / "reader-state"
    # The library volume always exists in the deployed topology; a reader
    # requires it rather than creating it, which is the point.
    (tmp_path / "library" / "sources").mkdir(parents=True)
    config = CompanyConfig.from_env(
        {
            "CAAL_COMPANY_LIBRARY_DIR": str(tmp_path / "library"),
            "CAAL_COMPANY_LIBRARY_KEYS": profile_crypto.generate_key_material(version=1),
            "CAAL_COMPANY_ROLE": "reader",
            "CAAL_COMPANY_READER_STATE_DIR": str(state),
        }
    )
    assert config.state_dir == state
    assert state.is_dir()


def test_the_writer_keeps_its_state_beside_its_library(tmp_path):
    """Unset means unchanged: the single writer's own directory is writable."""
    from caal import profile_crypto
    from caal.company.config import CompanyConfig

    config = CompanyConfig.from_env(
        {
            "CAAL_COMPANY_LIBRARY_DIR": str(tmp_path / "library"),
            "CAAL_COMPANY_LIBRARY_KEYS": profile_crypto.generate_key_material(version=1),
            "CAAL_COMPANY_ROLE": "owner",
        }
    )
    assert config.state_dir == config.data_dir


def test_the_mcp_service_puts_its_nonces_in_the_state_directory():
    import inspect

    from caal.company import server_main

    source = inspect.getsource(server_main)
    assert "config.state_dir / \"nonces.sqlite3\"" in source
