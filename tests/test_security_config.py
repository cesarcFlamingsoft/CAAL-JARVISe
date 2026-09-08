"""Startup validation for the multi-user security configuration.

Multi-user JARVIS is enabled only when every required, non-secret and secret
setting is present and valid. Anything less is reported loudly, by variable
name and never by value, and leaves the user-scoped endpoints failing closed.
"""

from __future__ import annotations

import logging

import pytest

from caal import profile_crypto
from caal.security_config import (
    ENV_ACCESS_AUD,
    ENV_ACCESS_TEAM_DOMAIN,
    ENV_BOOTSTRAP_ADMIN_EMAIL,
    ENV_INTERNAL_AUTH_SECRET,
    ENV_PROFILE_ENCRYPTION_KEYS,
    REQUIRED_ENV,
    load_multi_user_config,
    log_startup_status,
)

SECRET = "k" * 48
KEYS = profile_crypto.generate_key_material(version=1)
TEAM = "https://flamingsoftinc.cloudflareaccess.com"
AUD = "d1d09a2c79e964918d59077b9bb5b3a7b67ff76a04a80822eb8a3ce5f46354ac"


def _env(**overrides) -> dict[str, str]:
    env = {
        ENV_INTERNAL_AUTH_SECRET: SECRET,
        ENV_PROFILE_ENCRYPTION_KEYS: KEYS,
        ENV_BOOTSTRAP_ADMIN_EMAIL: "CesarC@MexcanTech.com",
        ENV_ACCESS_TEAM_DOMAIN: TEAM,
        ENV_ACCESS_AUD: AUD,
        "CAAL_DATA_DIR": "/tmp/caal-test-data",
    }
    env.update(overrides)
    return {key: value for key, value in env.items() if value is not None}


def test_complete_configuration_enables_multi_user() -> None:
    status = load_multi_user_config(_env())

    assert status.enabled is True
    assert status.attempted is True
    assert status.problems == ()
    config = status.config
    assert config is not None
    assert config.bootstrap_admin_email == "cesarc@mexcantech.com"
    assert config.access.team_domain == TEAM
    assert config.access.audience == AUD
    assert config.keyring.active_version == 1
    assert str(config.store_path).endswith("assistant.sqlite3")
    assert "/tmp/caal-test-data" in str(config.store_path)
    for rendered in (repr(config), str(config), status.describe()):
        assert SECRET not in rendered
        assert KEYS.split(":", 1)[1] not in rendered
        assert "mexcantech" not in rendered


def test_no_configuration_at_all_is_reported_as_legacy_single_user() -> None:
    status = load_multi_user_config({"CAAL_DATA_DIR": "/tmp/x"})

    assert status.enabled is False
    assert status.attempted is False
    assert {problem.name for problem in status.problems} == set(REQUIRED_ENV)
    assert all(problem.problem == "missing" for problem in status.problems)
    assert "single-user" in status.describe().lower()


def test_partial_configuration_fails_closed_and_names_only_what_is_missing() -> None:
    status = load_multi_user_config(_env(**{ENV_INTERNAL_AUTH_SECRET: None}))

    assert status.enabled is False
    assert status.attempted is True
    assert [(p.name, p.problem) for p in status.problems] == [(ENV_INTERNAL_AUTH_SECRET, "missing")]
    text = status.describe()
    assert ENV_INTERNAL_AUTH_SECRET in text
    assert "fail closed" in text.lower() or "disabled" in text.lower()


@pytest.mark.parametrize(
    "name, value",
    [
        (ENV_INTERNAL_AUTH_SECRET, "too-short"),
        (ENV_INTERNAL_AUTH_SECRET, "   "),
        (ENV_PROFILE_ENCRYPTION_KEYS, "v1:not-a-key"),
        (ENV_BOOTSTRAP_ADMIN_EMAIL, "cesar"),
        (ENV_ACCESS_TEAM_DOMAIN, "http://flamingsoftinc.cloudflareaccess.com"),
        (ENV_ACCESS_TEAM_DOMAIN, "https://example.com"),
        (ENV_ACCESS_AUD, "not-hex"),
    ],
)
def test_invalid_values_are_reported_by_name_without_echoing_them(name: str, value: str) -> None:
    status = load_multi_user_config(_env(**{name: value}))

    assert status.enabled is False
    assert [p.name for p in status.problems] == [name]
    assert status.problems[0].problem.startswith("invalid")
    if value.strip():  # an all-whitespace value has nothing that could be echoed
        assert value.strip() not in status.describe()


def test_every_problem_is_listed_at_once() -> None:
    status = load_multi_user_config(
        _env(**{ENV_INTERNAL_AUTH_SECRET: "short", ENV_ACCESS_AUD: None})
    )

    assert {p.name for p in status.problems} == {ENV_INTERNAL_AUTH_SECRET, ENV_ACCESS_AUD}


def test_startup_logging_is_loud_for_partial_config_and_quiet_for_legacy(caplog) -> None:
    # Outside the ``caal`` hierarchy: the voice agent turns propagation off for
    # that tree, which would hide records from caplog when the suite runs whole.
    logger = logging.getLogger("security_config_startup_test")
    logger.propagate = True

    with caplog.at_level(logging.INFO, logger=logger.name):
        log_startup_status(load_multi_user_config(_env()), logger=logger)
        log_startup_status(load_multi_user_config({}), logger=logger)
        log_startup_status(
            load_multi_user_config(_env(**{ENV_PROFILE_ENCRYPTION_KEYS: None})), logger=logger
        )

    levels = [record.levelno for record in caplog.records]
    assert levels == [logging.INFO, logging.WARNING, logging.ERROR]
    assert ENV_PROFILE_ENCRYPTION_KEYS in caplog.records[2].getMessage()
    assert SECRET not in caplog.text and "mexcantech" not in caplog.text


# --- standalone (Cloudflare-free) password sign-in ---------------------------------
#
# Cloudflare Access is an *optional* alternate identity provider. A deployment
# that never heard of Cloudflare must still enable multi-user identity, and
# must not lose any local-auth control by leaving those variables unset.

from caal.password_hash import Argon2Params, hash_password  # noqa: E402
from caal.security_config import (  # noqa: E402
    ENV_BOOTSTRAP_ADMIN_PASSWORD_HASH,
    ENV_PASSWORD_LOGIN,
    ENV_SESSION_ABSOLUTE_HOURS,
    ENV_SESSION_IDLE_MINUTES,
)

CHEAP = Argon2Params(memory_kib=64, iterations=1, lanes=1)
BOOTSTRAP_HASH = hash_password("a one time bootstrap password", params=CHEAP)


def _local_env(**overrides) -> dict[str, str]:
    env = _env(**{ENV_ACCESS_TEAM_DOMAIN: None, ENV_ACCESS_AUD: None})
    env.update(overrides)
    return {key: value for key, value in env.items() if value is not None}


def test_cloudflare_is_not_required_for_multi_user_identity() -> None:
    status = load_multi_user_config(_local_env())

    assert status.enabled is True
    assert status.problems == ()
    assert status.config.access is None
    assert status.config.password_login is True
    assert ENV_ACCESS_TEAM_DOMAIN not in REQUIRED_ENV
    assert ENV_ACCESS_AUD not in REQUIRED_ENV


def test_describe_says_which_identity_providers_are_active() -> None:
    local = load_multi_user_config(_local_env()).describe()
    both = load_multi_user_config(_env()).describe()

    assert "password" in local.lower()
    assert "cloudflare" not in local.lower()
    assert "cloudflare" in both.lower()


@pytest.mark.parametrize("missing", [ENV_ACCESS_TEAM_DOMAIN, ENV_ACCESS_AUD])
def test_half_configured_cloudflare_is_an_error_not_a_silent_downgrade(missing: str) -> None:
    status = load_multi_user_config(_env(**{missing: None}))

    assert status.enabled is False
    assert [p.name for p in status.problems] == [missing]


def test_disabling_password_login_requires_an_identity_provider() -> None:
    status = load_multi_user_config(_local_env(**{ENV_PASSWORD_LOGIN: "false"}))

    assert status.enabled is False
    assert [p.name for p in status.problems] == [ENV_PASSWORD_LOGIN]
    assert "no way to sign in" in status.problems[0].problem


def test_password_login_may_be_disabled_when_cloudflare_is_configured() -> None:
    status = load_multi_user_config(_env(**{ENV_PASSWORD_LOGIN: "false"}))

    assert status.enabled is True
    assert status.config.password_login is False


class TestBootstrapPasswordHash:
    def test_is_optional(self) -> None:
        assert load_multi_user_config(_local_env()).config.bootstrap_admin_password_hash is None

    def test_is_accepted_and_never_printed(self) -> None:
        status = load_multi_user_config(
            _local_env(**{ENV_BOOTSTRAP_ADMIN_PASSWORD_HASH: BOOTSTRAP_HASH})
        )

        assert status.config.bootstrap_admin_password_hash == BOOTSTRAP_HASH
        for rendered in (repr(status.config), str(status.config), status.describe()):
            assert BOOTSTRAP_HASH not in rendered
            assert BOOTSTRAP_HASH.split("$")[-1] not in rendered

    @pytest.mark.parametrize(
        "value",
        ["a one time bootstrap password", "hunter2", "$argon2id$", "not-a-hash"],
    )
    def test_a_plaintext_password_is_refused_outright(self, value: str) -> None:
        status = load_multi_user_config(
            _local_env(**{ENV_BOOTSTRAP_ADMIN_PASSWORD_HASH: value})
        )

        assert status.enabled is False
        assert [p.name for p in status.problems] == [ENV_BOOTSTRAP_ADMIN_PASSWORD_HASH]
        assert value not in status.describe()


class TestSessionLifetimes:
    def test_defaults_are_sane(self) -> None:
        config = load_multi_user_config(_local_env()).config

        assert config.session_policy.idle_seconds == 8 * 3600
        assert config.session_policy.absolute_seconds == 7 * 86400

    def test_operator_may_shorten_them(self) -> None:
        config = load_multi_user_config(
            _local_env(**{ENV_SESSION_IDLE_MINUTES: "30", ENV_SESSION_ABSOLUTE_HOURS: "12"})
        ).config

        assert config.session_policy.idle_seconds == 1800
        assert config.session_policy.absolute_seconds == 12 * 3600

    @pytest.mark.parametrize(
        "name, value",
        [
            (ENV_SESSION_IDLE_MINUTES, "0"),
            (ENV_SESSION_IDLE_MINUTES, "-5"),
            (ENV_SESSION_IDLE_MINUTES, "abc"),
            (ENV_SESSION_IDLE_MINUTES, "999999"),
            (ENV_SESSION_ABSOLUTE_HOURS, "0"),
            (ENV_SESSION_ABSOLUTE_HOURS, "not-a-number"),
        ],
    )
    def test_nonsense_lifetimes_are_refused(self, name: str, value: str) -> None:
        status = load_multi_user_config(_local_env(**{name: value}))

        assert status.enabled is False
        assert [p.name for p in status.problems] == [name]

    def test_an_idle_timeout_longer_than_the_absolute_cap_is_refused(self) -> None:
        status = load_multi_user_config(
            _local_env(**{ENV_SESSION_IDLE_MINUTES: "600", ENV_SESSION_ABSOLUTE_HOURS: "1"})
        )

        assert status.enabled is False
        assert [p.name for p in status.problems] == [ENV_SESSION_IDLE_MINUTES]
