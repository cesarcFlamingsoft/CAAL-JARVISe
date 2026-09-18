"""Encrypted, process-keyed, short-lived visual-frame cache."""
from __future__ import annotations

import hashlib
import os
import secrets
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

TTL_SECONDS = 30 * 60
MAX_BYTES = 1 << 30


@dataclass
class _Entry:
    created: float
    accessed: float
    size: int


class VisualFrameCache:
    """One encrypted compact JPEG per authenticated browser binding.

    The encryption and filename keys only exist in this process.  Initializing a
    cache removes any prior process's directory contents, so a restart makes old
    frames irrecoverable even if a temporary directory was not cleaned by the OS.
    """

    def __init__(self, root: Path | None = None, *, clock=time.time) -> None:
        self.root = root or Path(tempfile.gettempdir()) / "caal-visual-cache"
        # No cache survives construction by a new process/key.  This is also
        # deliberately best-effort: a stale unreadable file must not prevent a
        # process from serving its own isolated cache.
        if self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)
        self._crypt = Fernet(Fernet.generate_key())
        self._name_key = secrets.token_bytes(32)
        self._clock = clock
        self._entries: dict[Path, _Entry] = {}

    def _path(self, user_id: str, session_binding: str) -> Path:
        digest = hashlib.blake2b(
            f"{user_id}\0{session_binding}".encode(), key=self._name_key, digest_size=32
        ).hexdigest()
        return self.root / digest

    def _remove(self, path: Path) -> None:
        self._entries.pop(path, None)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass

    def purge(self) -> None:
        now = self._clock()
        for path, entry in list(self._entries.items()):
            if now - entry.created > TTL_SECONDS:
                self._remove(path)
        # This process creates all entries under its fresh root. Still count
        # actual file sizes so the one-gigabyte limit is hard at rest.
        entries: list[tuple[float, int, Path]] = []
        for path, entry in list(self._entries.items()):
            try:
                size = path.stat().st_size
            except OSError:
                self._entries.pop(path, None)
                continue
            entry.size = size
            entries.append((entry.accessed, size, path))
        total = sum(size for _, size, _ in entries)
        for _, size, path in sorted(entries):
            if total <= MAX_BYTES:
                break
            self._remove(path)
            total -= size

    def put(self, user_id: str, session_binding: str, jpeg_b64: str) -> None:
        now = self._clock()
        self.purge()
        path = self._path(user_id, session_binding)
        # A random, hidden temporary filename makes the only filesystem names
        # opaque too. replace() gives readers either a complete old token or a
        # complete new token, never a partial frame.
        temporary = self.root / f".{secrets.token_hex(24)}"
        try:
            encoded = self._crypt.encrypt(jpeg_b64.encode("ascii"))
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            os.chmod(path, 0o600)
            self._entries[path] = _Entry(created=now, accessed=now, size=len(encoded))
            self.purge()
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass

    def get(self, user_id: str, session_binding: str) -> str | None:
        self.purge()
        path = self._path(user_id, session_binding)
        entry = self._entries.get(path)
        if entry is None or self._clock() - entry.created > TTL_SECONDS:
            self._remove(path)
            return None
        try:
            value = self._crypt.decrypt(path.read_bytes()).decode("ascii")
        except (OSError, InvalidToken, UnicodeDecodeError):
            self._remove(path)
            return None
        entry.accessed = self._clock()
        return value

    def clear(self, user_id: str, session_binding: str) -> None:
        self._remove(self._path(user_id, session_binding))
