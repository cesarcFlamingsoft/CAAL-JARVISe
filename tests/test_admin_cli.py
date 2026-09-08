"""The operator commands for standalone sign-in.

The property that matters most here: a generated password reaches standard
output and nothing else -- not the database, not the audit trail, not a log.
"""

from __future__ import annotations

import pytest

from caal import admin_cli, profile_crypto
from caal.local_auth import LocalAuth
from caal.password_hash import is_password_hash
from caal.security_config import (
    ENV_BOOTSTRAP_ADMIN_EMAIL,
    ENV_BOOTSTRAP_ADMIN_PASSWORD_HASH,
    ENV_DATA_DIR,
    ENV_INTERNAL_AUTH_SECRET,
    ENV_PROFILE_ENCRYPTION_KEYS,
)
from caal.user_store import ADMIN, UserStore

EMAIL = "cesarc@mexcantech.com"


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_INTERNAL_AUTH_SECRET, "s" * 48)
    monkeypatch.setenv(ENV_PROFILE_ENCRYPTION_KEYS, profile_crypto.generate_key_material(version=1))
    monkeypatch.setenv(ENV_BOOTSTRAP_ADMIN_EMAIL, EMAIL)
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path))
    monkeypatch.delenv(ENV_BOOTSTRAP_ADMIN_PASSWORD_HASH, raising=False)
    return tmp_path


def _password_from(output: str) -> str:
    for line in output.splitlines():
        if line.strip().startswith("Password:"):
            return line.split("Password:", 1)[1].strip()
    raise AssertionError(f"no password in output: {output!r}")


def _hash_from(output: str) -> str:
    for line in output.splitlines():
        if ENV_BOOTSTRAP_ADMIN_PASSWORD_HASH in line:
            return line.split("=", 1)[1].strip()
    raise AssertionError("no hash line in output")


class TestGenerate:
    def test_prints_a_password_and_a_matching_hash(self, env, capsys) -> None:
        assert admin_cli.main(["generate", "--email", EMAIL]) == 0

        output = capsys.readouterr().out
        password = _password_from(output)
        hashed = _hash_from(output)
        assert len(password) == 24
        assert is_password_hash(hashed)
        from caal.password_hash import verify_password

        assert verify_password(password, hashed).ok

    def test_never_writes_the_plaintext_into_the_hash_line(self, env, capsys) -> None:
        admin_cli.main(["generate", "--email", EMAIL])

        output = capsys.readouterr().out
        assert _password_from(output) not in _hash_from(output)

    def test_touches_no_database(self, env, capsys) -> None:
        admin_cli.main(["generate", "--email", EMAIL])
        capsys.readouterr()

        assert not (env / "assistant.sqlite3").exists()

    def test_refuses_a_malformed_email(self, env, capsys) -> None:
        assert admin_cli.main(["generate", "--email", "not-an-email"]) == 2


class TestSeed:
    def _seed_env(self, monkeypatch, capsys) -> str:
        admin_cli.main(["generate", "--email", EMAIL])
        output = capsys.readouterr().out
        monkeypatch.setenv(ENV_BOOTSTRAP_ADMIN_PASSWORD_HASH, _hash_from(output))
        return _password_from(output)

    def test_creates_the_administrator_who_must_then_change_the_password(
        self, env, monkeypatch, capsys
    ) -> None:
        password = self._seed_env(monkeypatch, capsys)

        assert admin_cli.main(["seed"]) == 0
        assert "Created the administrator" in capsys.readouterr().out

        store = UserStore(env / "assistant.sqlite3", keyring=None)
        auth = LocalAuth(store)
        profiles = store.list_users()
        assert len(profiles) == 1
        assert profiles[0].email == EMAIL
        assert profiles[0].role == ADMIN
        result = auth.authenticate(EMAIL, password)
        assert result.ok
        assert result.must_change_password is True

    def test_is_idempotent(self, env, monkeypatch, capsys) -> None:
        self._seed_env(monkeypatch, capsys)
        admin_cli.main(["seed"])
        capsys.readouterr()

        assert admin_cli.main(["seed"]) == 0
        assert "Nothing to do" in capsys.readouterr().out

    def test_refuses_without_a_configured_hash(self, env, capsys) -> None:
        assert admin_cli.main(["seed"]) == 2
        assert ENV_BOOTSTRAP_ADMIN_PASSWORD_HASH in capsys.readouterr().err

    def test_never_stores_the_plaintext_anywhere_in_the_database(
        self, env, monkeypatch, capsys
    ) -> None:
        password = self._seed_env(monkeypatch, capsys)
        admin_cli.main(["seed"])
        capsys.readouterr()

        blob = (env / "assistant.sqlite3").read_bytes()
        assert password.encode() not in blob


class TestReset:
    def test_issues_a_new_one_time_password_for_an_existing_user(
        self, env, monkeypatch, capsys
    ) -> None:
        admin_cli.main(["generate", "--email", EMAIL])
        monkeypatch.setenv(ENV_BOOTSTRAP_ADMIN_PASSWORD_HASH, _hash_from(capsys.readouterr().out))
        admin_cli.main(["seed"])
        capsys.readouterr()

        assert admin_cli.main(["reset", "--email", EMAIL]) == 0
        issued = _password_from(capsys.readouterr().out)

        store = UserStore(env / "assistant.sqlite3", keyring=None)
        result = LocalAuth(store).authenticate(EMAIL, issued)
        assert result.ok
        assert result.must_change_password is True

    def test_refuses_an_unknown_account_without_saying_more(self, env, capsys) -> None:
        assert admin_cli.main(["reset", "--email", "nobody@example.com"]) == 1
        assert "No account" in capsys.readouterr().err


class TestStatus:
    def test_reports_accounts_without_printing_emails(self, env, monkeypatch, capsys) -> None:
        admin_cli.main(["generate", "--email", EMAIL])
        monkeypatch.setenv(ENV_BOOTSTRAP_ADMIN_PASSWORD_HASH, _hash_from(capsys.readouterr().out))
        admin_cli.main(["seed"])
        capsys.readouterr()

        assert admin_cli.main(["status"]) == 0
        output = capsys.readouterr().out
        assert "password=yes" in output
        assert "admin" in output
        assert EMAIL not in output

    def test_reports_an_unconfigured_deployment(self, monkeypatch, capsys) -> None:
        for name in (ENV_INTERNAL_AUTH_SECRET, ENV_PROFILE_ENCRYPTION_KEYS,
                     ENV_BOOTSTRAP_ADMIN_EMAIL):
            monkeypatch.delenv(name, raising=False)

        assert admin_cli.main(["status"]) == 1
