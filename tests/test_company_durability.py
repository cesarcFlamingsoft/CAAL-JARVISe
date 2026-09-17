"""The company library across process boundaries, crashes and injected faults.

Everything here uses **separately spawned interpreters** driving their own
`CompanyLibrary` objects over a pipe. That is the point: the defects this file
exists to hold down were all invisible to a test that shared one library object
between "the writer" and "the reader", because a shared object cannot have a
stale snapshot, cannot lose a race for the last write, and cannot be killed
half way through a mutation. Nothing here monkeypatches the store under test.

What is pinned:

* a reader in another process sees an upload without being restarted, and
  stops seeing a deleted document without being restarted;
* a reader's shutdown changes nothing on disk -- it cannot overwrite a newer
  index with the one it loaded, and it cannot resurrect a deletion;
* there is exactly one writer, enforced by an exclusive lock, not by intent;
* the owner is provisioned, and two differently configured processes cannot
  both bind the same library;
* an ingest that fails at the source write, at the index encryption, at the
  index fsync or at the index rename leaves nothing searchable, leaves no
  orphaned original, and cannot be persisted afterwards by a close;
* a crash before the index publish loses the whole mutation; a crash after it
  keeps the whole mutation; either way reopening is consistent;
* a deletion whose original cannot be unlinked says so, and reopening cleans
  the orphan up rather than leaving it to be found later.

Every fixture is unmistakably synthetic and lives in a temp directory with its
own freshly generated key ring.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from caal import profile_crypto

OWNER = "usr_" + "a1" * 12
OTHER = "usr_" + "b2" * 12
CLAUSE = "FIXTURE termination clause: thirty days written notice."

REPO_ROOT = Path(__file__).resolve().parents[1]

# The child. A real interpreter, a real CompanyLibrary, one JSON command per
# line. It shares no object with the test process.
_CHILD = r"""
import json, os, sys
from caal.company.config import CompanyConfig
from caal.company.service import CompanyLibrary

config = CompanyConfig.from_env()
library = CompanyLibrary(config)
sys.stdout.write(json.dumps({"opened": True, "role": library.role}) + "\n")
sys.stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    command = json.loads(line)
    operation = command.pop("op")
    try:
        if operation == "arm":
            # Arm the store's fault point *after* the library is open, so the
            # failure lands on the mutation under test rather than on the
            # binding this process did on the way up.
            os.environ["CAAL_COMPANY_FAULT_ENABLE"] = "1"
            os.environ["CAAL_COMPANY_FAULT"] = command["fault"]
            answer = {"armed": command["fault"]}
        elif operation == "status":
            report = library.status(owner=command["owner"])
            answer = {
                "status": report.get("status"),
                "document_count": report.get("document_count"),
                "generation": report.get("generation"),
                "passage_count": report.get("passage_count"),
            }
        elif operation == "owner":
            answer = {"owner_id": library.owner_id()}
        elif operation == "ingest":
            result = library.ingest(
                owner=command["owner"],
                filename=command.get("filename", "FIXTURE.txt"),
                data=command["text"].encode("utf-8"),
                title=command.get("title", "FIXTURE Contract"),
                classification="contract",
                status="current",
            )
            answer = {
                "status": result.status,
                "document_id": result.document_id,
                "version_id": result.version_id,
            }
        elif operation == "search":
            found = library.search(owner=command["owner"], query=command["query"])
            answer = {"status": found["status"], "hits": len(found["results"])}
        elif operation == "delete":
            answer = library.delete_document(
                owner=command["owner"], document_id=command["document_id"]
            )
        elif operation == "sources":
            answer = {
                "files": sorted(p.name for p in config.sources_dir.iterdir())
            }
        elif operation == "close":
            library.close()
            answer = {"closed": True}
        else:
            answer = {"error": "unknown_op"}
    except BaseException as exc:
        answer = {"error": type(exc).__name__, "message": str(exc)[:200]}
    sys.stdout.write(json.dumps(answer) + "\n")
    sys.stdout.flush()
    if operation == "close":
        break
"""


class Child:
    """One separately spawned interpreter holding its own library."""

    def __init__(self, process: subprocess.Popen) -> None:
        self.process = process

    def send(self, **command) -> dict:
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(json.dumps(command) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise AssertionError("the child interpreter died before answering")
        return json.loads(line)

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.send(op="close")
            except Exception:  # noqa: BLE001 - it may already be gone
                pass
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:  # pragma: no cover
            self.process.kill()


@pytest.fixture()
def library_dir(tmp_path) -> Path:
    return tmp_path / "company-library"


@pytest.fixture()
def environment(library_dir) -> dict[str, str]:
    base = dict(os.environ)
    base.update(
        {
            "CAAL_COMPANY_LIBRARY_DIR": str(library_dir),
            "CAAL_COMPANY_LIBRARY_KEYS": profile_crypto.generate_key_material(version=1),
            "CAAL_COMPANY_NAME": "FIXTURE Org",
            "CAAL_COMPANY_OWNER_USER_ID": OWNER,
        }
    )
    for name in ("CAAL_COMPANY_FAULT", "CAAL_COMPANY_FAULT_ENABLE", "CAAL_COMPANY_ROLE"):
        base.pop(name, None)
    return base


@pytest.fixture()
def spawn(environment):
    started: list[Child] = []

    def _spawn(role: str = "owner", *, owner: str | None = None):
        child_env = dict(environment)
        child_env["CAAL_COMPANY_ROLE"] = role
        if owner is not None:
            child_env["CAAL_COMPANY_OWNER_USER_ID"] = owner
        process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-c", _CHILD],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=child_env,
            cwd=str(REPO_ROOT),
        )
        child = Child(process)
        started.append(child)
        return child

    yield _spawn
    for child in reversed(started):
        child.close()


def _opened(child: Child) -> dict:
    assert child.process.stdout is not None
    return json.loads(child.process.stdout.readline())


def _fingerprint(directory: Path) -> dict[str, str]:
    """Every file under the library directory, by content hash."""
    prints = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            prints[str(path.relative_to(directory))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return prints


# --- cross-process visibility ------------------------------------------------------------------


def test_a_reader_has_nothing_to_serve_before_the_writer_has_opened(spawn):
    reader = spawn("reader")
    _opened(reader)
    report = reader.send(op="status", owner=OWNER)
    # Not "forbidden", and certainly not an empty success: the library has not
    # been initialised by its owner process yet, and the reader says so.
    assert report["status"] == "unconfigured"


def test_a_separately_spawned_reader_sees_an_upload_without_restarting(spawn):
    writer = spawn("owner")
    _opened(writer)
    reader = spawn("reader")
    _opened(reader)

    assert reader.send(op="status", owner=OWNER)["document_count"] == 0
    ingested = writer.send(op="ingest", owner=OWNER, text=CLAUSE)
    assert ingested["status"] == "indexed"

    # The same long-lived reader object, no restart.
    report = reader.send(op="status", owner=OWNER)
    assert report["document_count"] == 1
    assert reader.send(op="search", owner=OWNER, query="termination")["hits"] == 1


def test_a_reader_shutdown_changes_nothing_on_disk(spawn, library_dir):
    writer, reader = spawn("owner"), spawn("reader")
    _opened(writer)
    _opened(reader)
    writer.send(op="ingest", owner=OWNER, text=CLAUSE)
    reader.send(op="status", owner=OWNER)

    writer.send(op="ingest", owner=OWNER, text="FIXTURE second clause: notice period.",
                title="FIXTURE Second")
    before = _fingerprint(library_dir)
    reader.close()
    assert _fingerprint(library_dir) == before

    fresh = spawn("reader")
    _opened(fresh)
    # Both uploads survive the stale reader's shutdown.
    assert fresh.send(op="status", owner=OWNER)["document_count"] == 2


def test_a_reader_stops_answering_from_a_deleted_document(spawn):
    writer, reader = spawn("owner"), spawn("reader")
    _opened(writer)
    _opened(reader)
    ingested = writer.send(op="ingest", owner=OWNER, text=CLAUSE)
    assert reader.send(op="search", owner=OWNER, query="termination")["hits"] == 1

    writer.send(op="delete", owner=OWNER, document_id=ingested["document_id"])

    # No restart, no cache invalidation call: the reader refreshes itself.
    assert reader.send(op="search", owner=OWNER, query="termination")["hits"] == 0


def test_a_deleted_document_is_not_resurrected_by_a_reader_restart(spawn):
    writer, reader = spawn("owner"), spawn("reader")
    _opened(writer)
    _opened(reader)
    ingested = writer.send(op="ingest", owner=OWNER, text=CLAUSE)
    reader.send(op="search", owner=OWNER, query="termination")
    writer.send(op="delete", owner=OWNER, document_id=ingested["document_id"])
    reader.close()

    revived = spawn("reader")
    _opened(revived)
    assert revived.send(op="status", owner=OWNER)["document_count"] == 0
    assert writer.send(op="sources", owner=OWNER)["files"] == []


# --- one writer --------------------------------------------------------------------------------


def test_two_writers_cannot_open_the_same_library(spawn):
    first = spawn("owner")
    _opened(first)
    second = spawn("owner")
    assert second.process.stdout is not None
    assert second.process.stdout.readline() == ""
    second.process.wait(timeout=20)
    assert second.process.returncode != 0
    assert "StoreLockedError" in (second.process.stderr.read() if second.process.stderr else "")


def test_a_writer_can_be_replaced_after_the_first_lets_go(spawn):
    first = spawn("owner")
    _opened(first)
    first.send(op="ingest", owner=OWNER, text=CLAUSE)
    first.close()

    second = spawn("owner")
    _opened(second)
    assert second.send(op="status", owner=OWNER)["document_count"] == 1


# --- the provisioned owner ---------------------------------------------------------------------


def test_the_owner_is_provisioned_and_never_claimed_by_writing_first(spawn):
    writer = spawn("owner")
    _opened(writer)
    assert writer.send(op="owner")["owner_id"] == OWNER
    # A different administrator writing first changes nothing.
    refused = writer.send(op="ingest", owner=OTHER, text=CLAUSE)
    assert refused["error"] == "PermissionError"
    assert writer.send(op="owner")["owner_id"] == OWNER


def test_two_independently_configured_owners_cannot_both_bind(spawn):
    first = spawn("owner", owner=OWNER)
    _opened(first)
    assert first.send(op="owner")["owner_id"] == OWNER
    first.close()

    # A second process, configured for somebody else, against the same library.
    other = spawn("owner", owner=OTHER)
    assert other.process.stdout is not None
    assert other.process.stdout.readline() == ""
    other.process.wait(timeout=20)
    assert other.process.returncode != 0
    stderr = other.process.stderr.read() if other.process.stderr else ""
    assert "CompanyOwnerMismatchError" in stderr

    # And the binding on disk is untouched.
    again = spawn("owner", owner=OWNER)
    _opened(again)
    assert again.send(op="owner")["owner_id"] == OWNER


def test_an_unprovisioned_library_answers_nothing(spawn, environment):
    environment.pop("CAAL_COMPANY_OWNER_USER_ID")
    child = spawn("owner")
    _opened(child)
    assert child.send(op="status", owner=OWNER)["status"] == "unconfigured"
    assert child.send(op="ingest", owner=OWNER, text=CLAUSE)["error"] == "PermissionError"


# --- injected faults ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fault", ["source_write", "source_fsync", "source_rename", "index_encrypt",
              "index_fsync", "index_rename"]
)
def test_an_ingest_that_fails_leaves_nothing_searchable(spawn, fault):
    """A real failure, in a real separate process, at a named point."""
    broken = spawn("owner")
    _opened(broken)
    assert broken.send(op="arm", fault=fault)["armed"] == fault
    outcome = broken.send(op="ingest", owner=OWNER, text=CLAUSE)
    assert outcome["error"] in ("OSError", "ValueError"), outcome

    # Still the same process: the failed ingest is not searchable in memory.
    assert broken.send(op="search", owner=OWNER, query="termination")["hits"] == 0
    assert broken.send(op="status", owner=OWNER)["document_count"] == 0
    # And closing it cannot persist what failed.
    broken.close()

    reopened = spawn("owner")
    _opened(reopened)
    assert reopened.send(op="status", owner=OWNER)["document_count"] == 0
    assert reopened.send(op="search", owner=OWNER, query="termination")["hits"] == 0
    # No orphaned encrypted original either: reopening reconciles them.
    assert reopened.send(op="sources", owner=OWNER)["files"] == []


def test_a_crash_before_the_index_publish_loses_the_whole_mutation(spawn):
    crashing = spawn("owner")
    _opened(crashing)
    crashing.send(op="arm", fault="crash_before_publish")
    with pytest.raises(AssertionError):
        crashing.send(op="ingest", owner=OWNER, text=CLAUSE)
    crashing.process.wait(timeout=20)
    assert crashing.process.returncode == 97

    reopened = spawn("owner")
    _opened(reopened)
    assert reopened.send(op="status", owner=OWNER)["document_count"] == 0
    assert reopened.send(op="sources", owner=OWNER)["files"] == []


def test_a_crash_after_the_index_publish_keeps_the_whole_mutation(spawn):
    crashing = spawn("owner")
    _opened(crashing)
    crashing.send(op="arm", fault="crash_after_publish")
    with pytest.raises(AssertionError):
        crashing.send(op="ingest", owner=OWNER, text=CLAUSE)
    crashing.process.wait(timeout=20)
    assert crashing.process.returncode == 97

    reopened = spawn("owner")
    _opened(reopened)
    assert reopened.send(op="status", owner=OWNER)["document_count"] == 1
    assert reopened.send(op="search", owner=OWNER, query="termination")["hits"] == 1
    # The original committed before the catalogue did, so it is there.
    assert len(reopened.send(op="sources", owner=OWNER)["files"]) == 1


def test_a_deletion_that_cannot_remove_the_original_says_so(spawn):
    writer = spawn("owner")
    _opened(writer)
    ingested = writer.send(op="ingest", owner=OWNER, text=CLAUSE)
    writer.close()

    broken = spawn("owner")
    _opened(broken)
    broken.send(op="arm", fault="source_unlink")
    outcome = broken.send(op="delete", owner=OWNER, document_id=ingested["document_id"])
    assert outcome["status"] == "deleted_with_errors"
    assert outcome["sources_removed"] is False
    # The catalogue change committed, so it is gone from the library...
    assert broken.send(op="status", owner=OWNER)["document_count"] == 0
    # ...and the original really did survive, which is what was reported.
    assert len(broken.send(op="sources", owner=OWNER)["files"]) == 1
    broken.close()

    # Reopening reconciles the orphan rather than leaving it to be found later.
    reopened = spawn("owner")
    _opened(reopened)
    assert reopened.send(op="sources", owner=OWNER)["files"] == []
