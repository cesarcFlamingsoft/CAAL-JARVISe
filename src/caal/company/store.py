"""The encrypted store underneath the company library.

Three things live in the library directory, and only one of them is readable:

``sources/<version>.enc``
    the original uploaded bytes, AES-256-GCM under the company key ring and
    bound by associated data to the owner, the document and the version. A
    blob copied between versions or between owners fails authentication
    instead of decrypting into somebody else's answer.

``index.enc``
    the whole catalogue -- documents, versions, subjects, chunk text and the
    FTS5 full-text index -- as one encrypted snapshot.

``writer.lock``
    an empty file. Its only content is the ``flock`` on it.

The index is a **SQLite database held in memory**. It is persisted by
``Connection.serialize()``, encrypted, and written durably; it is restored by
decrypting and ``Connection.deserialize()``. That is the deliberate answer to
the obvious hole in "encrypted document storage": a plaintext FTS file sitting
next to the ciphertext would undercut the whole claim. The cost is stated
rather than hidden -- the working copy is plaintext *in process memory* while
the service runs -- and the capacity is bounded so that statement stays true.

Two roles, and the difference between them is the whole of the cross-process
story:

:data:`ROLE_OWNER`
    the single writer. It takes an **exclusive** ``flock`` on ``writer.lock``
    at open and holds it for its whole life, so a second writer cannot start
    at all rather than racing for the last snapshot. It is the only role that
    may publish an index.

:data:`ROLE_READER`
    every other process. It takes no lock, **never writes anything**, and
    :meth:`refresh` before each read: the identity of ``index.enc`` is checked
    and the in-memory database is rebuilt from it when the writer has moved
    on. A reader that shuts down leaves the directory byte-for-byte as it
    found it -- in particular :meth:`close` does not snapshot, in either role,
    so no shutdown can resurrect a deleted document or discard a new one.

The durability protocol for one mutation, in order:

1. encrypted originals for the mutation are written **before** the catalogue
   commits, each one durably (``fsync`` on the file, atomic rename, ``fsync``
   on the directory). A failure here aborts before anything is searchable;
2. the catalogue changes inside one ``BEGIN IMMEDIATE`` transaction which also
   bumps the generation counter, and commits;
3. ``index.enc`` is replaced durably. **This rename is the commit point**: a
   crash before it leaves the previous library, a crash after it leaves the
   next one. If publishing fails, the in-memory database is rolled back to the
   bytes that were last successfully published, so a failed mutation is not
   searchable and cannot be persisted later by anything;
4. originals belonging to removed versions are unlinked *after* the commit
   point, and a failure is **reported**, never swallowed.

Whatever a crash interrupts, reopening as the owner reconciles the two: an
original whose version is not in the catalogue is deleted, because either its
ingest never committed or its deletion did.

Nothing here logs a title, a query, a passage or an owner id.
"""

from __future__ import annotations

import base64
import contextlib
import errno
import logging
import os
import sqlite3
import sys
from collections.abc import Iterator
from pathlib import Path

from caal.profile_crypto import DecryptionError, KeyRing

from .config import CompanyConfig

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_SNAPSHOT_BYTES",
    "ROLE_OWNER",
    "ROLE_READER",
    "SCHEMA_VERSION",
    "CompanyStore",
    "SourceRemovalError",
    "StoreLockedError",
    "StoreReadOnlyError",
    "StoreUnsupportedError",
    "fault_point",
    "interpreter_supports_store",
]

SCHEMA_VERSION = 2
MAX_SNAPSHOT_BYTES = 40 * 1024 * 1024
ROLE_OWNER = "owner"
ROLE_READER = "reader"
_INDEX_AAD = "company:index:v1"
_LOCK_NAME = "writer.lock"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS library (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS versions (
    version_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(document_id),
    classification TEXT NOT NULL,
    status TEXT NOT NULL,
    effective_date TEXT,
    version_label TEXT,
    filename TEXT NOT NULL,
    content_sha TEXT NOT NULL,
    byte_count INTEGER NOT NULL,
    ingest_status TEXT NOT NULL,
    reason TEXT,
    page_count INTEGER,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    coverage TEXT,
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS versions_by_document ON versions(document_id, status);
CREATE TABLE IF NOT EXISTS subjects (
    subject_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    aliases TEXT NOT NULL DEFAULT '',
    created_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS version_subjects (
    version_id TEXT NOT NULL REFERENCES versions(version_id),
    subject_id TEXT NOT NULL REFERENCES subjects(subject_id),
    PRIMARY KEY (version_id, subject_id)
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    version_id TEXT NOT NULL REFERENCES versions(version_id),
    ordinal INTEGER NOT NULL,
    location TEXT NOT NULL,
    body TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_by_version ON chunks(version_id, ordinal);
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
    body,
    chunk_id UNINDEXED,
    tokenize = 'porter unicode61'
);
"""


class StoreUnsupportedError(RuntimeError):
    """This interpreter, or this snapshot, cannot be used."""


class StoreLockedError(RuntimeError):
    """Another process already owns the single writer lock for this library."""


class StoreReadOnlyError(RuntimeError):
    """A reader was asked to write. Readers never write."""


class SourceRemovalError(RuntimeError):
    """Encrypted originals survived a deletion that otherwise committed."""

    def __init__(self, version_ids: tuple[str, ...]) -> None:
        super().__init__("some encrypted originals could not be removed")
        self.version_ids = tuple(version_ids)


# --- the deliberate test affordance -------------------------------------------------------------
#
# Fault injection has to cross a process boundary: the only honest way to show
# that a crash between the source write and the index publish reopens
# consistently is to make a separately spawned interpreter fail there for real.
# Monkeypatching an object the test also holds would prove nothing about the
# topology. So the store recognises one fault point, and only when *two*
# environment variables agree -- one of which exists solely to say "this
# process is a test".


def fault_point() -> str | None:
    """The injected failure point for this process, or ``None`` in normal operation."""
    if os.environ.get("CAAL_COMPANY_FAULT_ENABLE") != "1":
        return None
    point = (os.environ.get("CAAL_COMPANY_FAULT") or "").strip()
    return point or None


def _fault(point: str) -> None:
    if fault_point() == point:
        if point.startswith("crash_"):
            os._exit(97)
        raise OSError(errno.EIO, f"injected company store fault at {point}")


# --- durable primitives ---------------------------------------------------------------------------


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _durable_write(path: Path, text: str, *, fault: str | None = None) -> None:
    """Write ``text`` to ``path`` so that a crash leaves the old bytes or the new."""
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, text.encode("utf-8"))
            if fault:
                _fault(f"{fault}_fsync")
            os.fsync(fd)
        finally:
            os.close(fd)
        if fault:
            _fault(f"{fault}_rename")
        os.replace(temporary, path)
    except BaseException:
        # A half-written temporary is not evidence of anything and must not be
        # left for a later reconcile to puzzle over.
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise
    _fsync_dir(path.parent)


class CompanyStore:
    """The encrypted catalogue and the encrypted originals, as one object."""

    def __init__(self, config: CompanyConfig, *, role: str = ROLE_OWNER) -> None:
        if not hasattr(sqlite3.Connection, "serialize"):  # pragma: no cover - 3.10 only
            raise StoreUnsupportedError(
                "The company library needs Python 3.11 or newer for an encrypted in-memory index"
            )
        if role not in (ROLE_OWNER, ROLE_READER):
            raise ValueError("role must be owner or reader")
        self._config = config
        self._role = role
        self._keys: KeyRing = config.keys
        self._lock_handle: int | None = None
        self._loaded_identity: tuple[int, int, int] | None = None
        self._published: bytes | None = None
        if role == ROLE_OWNER:
            self._take_writer_lock()
        # Autocommit: this module drives its own BEGIN IMMEDIATE around the
        # multi-statement writes, and nothing else should open one implicitly.
        self._connection = sqlite3.connect(
            ":memory:", check_same_thread=False, isolation_level=None
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        try:
            self._restore()
            if role == ROLE_OWNER:
                self._reconcile_sources()
        except BaseException:
            self._release_writer_lock()
            raise

    # --- lifecycle ----------------------------------------------------------------------

    @property
    def role(self) -> str:
        return self._role

    @property
    def writable(self) -> bool:
        return self._role == ROLE_OWNER

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection

    def close(self) -> None:
        """Let go. **Never** writes, in either role.

        A snapshot on shutdown is how a stale reader overwrites a newer
        library, and how a crashed mutation gets persisted after the fact.
        Every successful mutation has already published; there is nothing left
        here that a close is entitled to keep.
        """
        try:
            self._connection.close()
        finally:
            self._release_writer_lock()

    def _take_writer_lock(self) -> None:
        import fcntl

        path = self._config.data_dir / _LOCK_NAME
        handle = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(handle)
            raise StoreLockedError(
                "Another process is already the writer for this company library"
            ) from exc
        self._lock_handle = handle

    def _release_writer_lock(self) -> None:
        handle, self._lock_handle = self._lock_handle, None
        if handle is None:
            return
        with contextlib.suppress(OSError):
            os.close(handle)

    # --- restoring and refreshing -------------------------------------------------------

    def _identity(self) -> tuple[int, int, int] | None:
        """What the current ``index.enc`` is, cheaply. ``None`` when there is none."""
        try:
            stat = os.stat(self._config.index_path)
        except FileNotFoundError:
            return None
        return (stat.st_ino, stat.st_mtime_ns, stat.st_size)

    def _fresh_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _restore(self) -> None:
        path = self._config.index_path
        identity = self._identity()
        if identity is None:
            self._connection.executescript(_SCHEMA)
            self._connection.execute(
                "INSERT OR REPLACE INTO library (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            self._connection.execute(
                "INSERT OR IGNORE INTO library (key, value) VALUES ('generation', '0')"
            )
            self._connection.commit()
            self._loaded_identity = None
            self._published = self._connection.serialize()
            return
        token = path.read_text("utf-8")
        # A DecryptionError here is the right outcome: a snapshot this ring
        # cannot open must stop the service, never be silently replaced.
        raw = base64.b64decode(self._keys.decrypt(token, aad=_INDEX_AAD).encode("ascii"))
        self._connection.deserialize(raw)
        self._connection.row_factory = sqlite3.Row
        # A snapshot from a future schema is refused rather than half-read.
        stored = self._connection.execute(
            "SELECT value FROM library WHERE key = 'schema_version'"
        ).fetchone()
        if stored is not None and int(stored["value"]) > SCHEMA_VERSION:
            raise StoreUnsupportedError("The company library was written by a newer FRIDAY")
        self._connection.executescript(_SCHEMA)
        self._migrate()
        self._connection.execute(
            "INSERT OR REPLACE INTO library (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self._connection.execute(
            "INSERT OR IGNORE INTO library (key, value) VALUES ('generation', '0')"
        )
        self._connection.commit()
        self._loaded_identity = identity
        # The rollback point is what this process would publish right now, not
        # the bytes it read: a schema migration applied above belongs to it.
        self._published = self._connection.serialize()

    def _migrate(self) -> None:
        """Add columns a newer schema introduced. ``CREATE IF NOT EXISTS`` cannot."""
        columns = {
            row["name"] for row in self._connection.execute("PRAGMA table_info(versions)")
        }
        if "coverage" not in columns:
            self._connection.execute("ALTER TABLE versions ADD COLUMN coverage TEXT")

    def refresh(self) -> bool:
        """Reload when the writer has published since this object last looked.

        Called before every authorization decision and every read, in both
        roles: a reader must not answer from a catalogue the owner has moved
        past, and must never resurrect a version the owner deleted.

        Returns whether anything was reloaded.
        """
        identity = self._identity()
        if identity == self._loaded_identity:
            return False
        if self.writable:
            # The owner holds the exclusive lock, so nothing else can have
            # published. A changed identity here means the file was replaced
            # underneath the single writer, which is not a recoverable state.
            raise StoreUnsupportedError(
                "The company library index changed underneath its writer"
            )
        replacement = self._fresh_connection()
        previous = self._connection
        self._connection = replacement
        try:
            self._restore()
        except BaseException:
            self._connection = previous
            with contextlib.suppress(Exception):
                replacement.close()
            raise
        with contextlib.suppress(Exception):
            previous.close()
        return True

    @property
    def generation(self) -> int:
        row = self._connection.execute(
            "SELECT value FROM library WHERE key = 'generation'"
        ).fetchone()
        return int(row["value"]) if row is not None else 0

    # --- publishing ---------------------------------------------------------------------

    def _require_owner(self) -> None:
        if not self.writable:
            raise StoreReadOnlyError("This company library process is a reader and never writes")

    def publish(self) -> None:
        """Serialize, encrypt and durably replace ``index.enc``. Owner only."""
        self._require_owner()
        raw = self._connection.serialize()
        if len(raw) > MAX_SNAPSHOT_BYTES:
            raise ValueError("The company library index has outgrown its bound")
        _fault("index_encrypt")
        token = self._keys.encrypt(base64.b64encode(raw).decode("ascii"), aad=_INDEX_AAD)
        _durable_write(self._config.index_path, token, fault="index")
        self._loaded_identity = self._identity()
        self._published = raw

    def _rollback_memory(self) -> None:
        """Put the in-memory database back to the last successfully published bytes."""
        replacement = self._fresh_connection()
        previous = self._connection
        self._connection = replacement
        if self._published is None:  # pragma: no cover - set in every constructor path
            self._connection.executescript(_SCHEMA)
        else:
            self._connection.deserialize(self._published)
            self._connection.row_factory = sqlite3.Row
        with contextlib.suppress(Exception):
            previous.close()

    @contextlib.contextmanager
    def mutation(self) -> Iterator["Mutation"]:
        """One durable, all-or-nothing change to the library. Owner only.

        The body adds and removes encrypted originals through the yielded
        :class:`Mutation` and changes the catalogue through
        :attr:`connection`. Nothing it did is visible to any process, and
        nothing it did can be persisted later, unless this block reaches the
        index publish.
        """
        self._require_owner()
        pending = Mutation(self)
        self._connection.execute("BEGIN IMMEDIATE")
        committed = False
        try:
            yield pending
            pending.write_additions()
            self._connection.execute(
                "INSERT OR REPLACE INTO library (key, value) VALUES ('generation', ?)",
                (str(self.generation + 1),),
            )
            self._connection.execute("COMMIT")
            committed = True
            _fault("crash_before_publish")
            self.publish()
            _fault("crash_after_publish")
        except BaseException:
            if not committed:
                with contextlib.suppress(Exception):
                    self._connection.execute("ROLLBACK")
            # Either the transaction never committed, or it committed in memory
            # and the publish failed. Both are undone the same way: go back to
            # the bytes that are actually on disk.
            self._rollback_memory()
            pending.discard_additions()
            raise
        failed = pending.apply_removals()
        if failed:
            # The catalogue no longer lists these versions, which is the truth
            # the reader must see, and it was published before this ran. What
            # is not true is that their originals are gone, so say so.
            raise SourceRemovalError(failed)

    # --- encrypted originals -------------------------------------------------------------

    def source_path(self, version_id: str) -> Path:
        return self._config.sources_dir / f"{version_id}.enc"

    @staticmethod
    def source_aad(*, owner: str, document_id: str, version_id: str) -> str:
        return f"company:src:{owner}:{document_id}:{version_id}"

    def _encrypt_source(
        self, data: bytes, *, owner: str, document_id: str, version_id: str
    ) -> str:
        return self._keys.encrypt(
            base64.b64encode(data).decode("ascii"),
            aad=self.source_aad(owner=owner, document_id=document_id, version_id=version_id),
        )

    def read_source(self, *, owner: str, document_id: str, version_id: str) -> bytes | None:
        path = self.source_path(version_id)
        if not path.exists():
            return None
        try:
            plain = self._keys.decrypt(
                path.read_text("utf-8"),
                aad=self.source_aad(owner=owner, document_id=document_id, version_id=version_id),
            )
        except DecryptionError:
            logger.error("A company source blob failed authentication and was not returned")
            return None
        return base64.b64decode(plain.encode("ascii"))

    def _reconcile_sources(self) -> int:
        """Delete originals the catalogue does not list. Owner only, at open.

        An original with no version row is the residue of exactly one of two
        interrupted mutations: an ingest whose index never published, or a
        deletion whose index did. Neither is something this library holds, so
        both end the same way.
        """
        self._require_owner()
        known = {
            row["version_id"]
            for row in self._connection.execute("SELECT version_id FROM versions")
        }
        removed = 0
        try:
            entries = list(self._config.sources_dir.iterdir())
        except OSError:  # pragma: no cover - the config created this directory
            return 0
        for entry in entries:
            # A ``.tmp`` here is the residue of a write that was killed before
            # its rename -- by a crash, since a failure cleans up after itself.
            temporary = entry.name.endswith(".tmp")
            if not temporary and (entry.suffix != ".enc" or entry.stem in known):
                continue
            try:
                entry.unlink()
                removed += 1
            except FileNotFoundError:
                continue
            except OSError:
                logger.error("An orphaned company source blob could not be removed at open")
        if removed:
            logger.info("Reconciled %d orphaned company source blob(s) at open", removed)
        return removed


class Mutation:
    """The encrypted originals one mutation adds and removes."""

    def __init__(self, store: CompanyStore) -> None:
        self._store = store
        self._additions: list[tuple[str, str]] = []
        self._written: list[Path] = []
        self._removals: list[str] = []

    def add_source(
        self, data: bytes, *, owner: str, document_id: str, version_id: str
    ) -> None:
        """Queue one encrypted original. It is written before the catalogue commits."""
        token = self._store._encrypt_source(
            bytes(data), owner=owner, document_id=document_id, version_id=version_id
        )
        self._additions.append((version_id, token))

    def remove_source(self, version_id: str) -> None:
        """Queue one encrypted original for removal, after the commit point."""
        self._removals.append(version_id)

    def write_additions(self) -> None:
        for version_id, token in self._additions:
            path = self._store.source_path(version_id)
            _fault("source_write")
            _durable_write(path, token, fault="source")
            self._written.append(path)

    def discard_additions(self) -> None:
        for path in self._written:
            with contextlib.suppress(OSError):
                path.unlink()
        self._written.clear()

    def apply_removals(self) -> tuple[str, ...]:
        """Unlink the queued originals. Returns the ones that survived."""
        failed: list[str] = []
        for version_id in self._removals:
            path = self._store.source_path(version_id)
            try:
                _fault("source_unlink")
                path.unlink()
            except FileNotFoundError:
                continue
            except OSError:
                logger.error("A company source blob could not be removed")
                failed.append(version_id)
        return tuple(failed)


def interpreter_supports_store() -> bool:
    """Whether this Python can hold the encrypted in-memory index (3.11+)."""
    return sys.version_info >= (3, 11) and hasattr(sqlite3.Connection, "serialize")
