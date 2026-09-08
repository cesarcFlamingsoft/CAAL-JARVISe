"""Provider catalogue and configuration for user-linked OAuth connections.

The catalogue is fixed (google, microsoft, zoho); configuration comes from the
operator's environment and is reported by *name* when it is missing or
unusable. Nothing here holds or asks for a person's password: the only
credentials are the application's own client id and secret, and even those
never render.
"""

from __future__ import annotations

import dataclasses
from urllib.parse import parse_qs, urlsplit

import pytest

from caal import oauth_providers
from caal.oauth_providers import (
    ENV_PUBLIC_ORIGIN,
    OAUTH_CALLBACK_PATH,
    PROVIDERS,
    load_provider_registry,
)

ORIGIN = "https://jarvis.example.com"
GOOGLE_SECRET = "google-client-secret-value"
GOOGLE_ENV = {
    "CAAL_OAUTH_GOOGLE_CLIENT_ID": "google-client-id.apps.googleusercontent.com",
    "CAAL_OAUTH_GOOGLE_CLIENT_SECRET": GOOGLE_SECRET,
    ENV_PUBLIC_ORIGIN: ORIGIN,
}
OTHERS_ENV = {
    "CAAL_OAUTH_MICROSOFT_CLIENT_ID": "ms-client-id",
    "CAAL_OAUTH_MICROSOFT_CLIENT_SECRET": "ms-client-secret",
    "CAAL_OAUTH_ZOHO_CLIENT_ID": "zoho-client-id",
    "CAAL_OAUTH_ZOHO_CLIENT_SECRET": "zoho-client-secret",
    ENV_PUBLIC_ORIGIN: ORIGIN,
}


def test_catalogue_is_exactly_the_three_supported_providers() -> None:
    assert PROVIDERS == ("google", "microsoft", "zoho")
    for name in PROVIDERS:
        spec = oauth_providers.spec_for(name)
        assert spec is not None and spec.provider == name
        assert spec.display_name
        assert spec.default_scopes
    assert oauth_providers.spec_for("apple") is None
    assert oauth_providers.spec_for("Google") is None
    assert oauth_providers.spec_for(None) is None


def test_nothing_models_or_asks_for_a_user_password() -> None:
    for cls in (
        oauth_providers.ProviderSpec,
        oauth_providers.ProviderConfig,
        oauth_providers.TokenGrant,
    ):
        names = {f.name.lower() for f in dataclasses.fields(cls)}
        assert not any("password" in n for n in names), cls
    assert not any("PASSWORD" in n for n in oauth_providers.all_env_names())
    assert ENV_PUBLIC_ORIGIN in oauth_providers.all_env_names()


def test_unconfigured_registry_reports_missing_names_only() -> None:
    registry = load_provider_registry({})
    for name in PROVIDERS:
        assert registry.configured(name) is False
        assert registry.get(name) is None
    assert registry.missing("google") == (
        "CAAL_OAUTH_GOOGLE_CLIENT_ID",
        "CAAL_OAUTH_GOOGLE_CLIENT_SECRET",
        ENV_PUBLIC_ORIGIN,
    )
    assert registry.missing("microsoft") == (
        "CAAL_OAUTH_MICROSOFT_CLIENT_ID",
        "CAAL_OAUTH_MICROSOFT_CLIENT_SECRET",
        ENV_PUBLIC_ORIGIN,
    )
    assert registry.missing("zoho") == (
        "CAAL_OAUTH_ZOHO_CLIENT_ID",
        "CAAL_OAUTH_ZOHO_CLIENT_SECRET",
        ENV_PUBLIC_ORIGIN,
    )
    assert registry.problems == ()
    assert [entry["provider"] for entry in registry.availability()] == list(PROVIDERS)
    assert all(entry["configured"] is False for entry in registry.availability())


def test_configured_provider_builds_an_authorization_url_without_its_secret() -> None:
    registry = load_provider_registry(GOOGLE_ENV)
    config = registry.get("google")

    assert config is not None and registry.configured("google")
    assert registry.missing("google") == ()
    assert config.redirect_uri == ORIGIN + OAUTH_CALLBACK_PATH
    assert config.pkce is True

    url = config.authorization_url(state="opaque-state", code_challenge="chal")
    parts = urlsplit(url)
    assert (
        f"{parts.scheme}://{parts.netloc}{parts.path}"
        == "https://accounts.google.com/o/oauth2/v2/auth"
    )
    query = parse_qs(parts.query, strict_parsing=True)
    assert query["client_id"] == [GOOGLE_ENV["CAAL_OAUTH_GOOGLE_CLIENT_ID"]]
    assert query["redirect_uri"] == [config.redirect_uri]
    assert query["response_type"] == ["code"]
    assert query["state"] == ["opaque-state"]
    assert query["code_challenge"] == ["chal"]
    assert query["code_challenge_method"] == ["S256"]
    assert query["access_type"] == ["offline"]
    assert set(query["scope"][0].split()) == set(config.scopes)
    assert "client_secret" not in query
    assert GOOGLE_SECRET not in url

    rendered = repr(registry) + str(registry) + repr(config) + str(config)
    assert GOOGLE_SECRET not in rendered
    assert [entry for entry in registry.availability() if entry["provider"] == "google"] == [
        {"provider": "google", "display_name": "Google", "configured": True}
    ]


def test_microsoft_tenant_and_zoho_accounts_domain_are_optional_overrides() -> None:
    registry = load_provider_registry(OTHERS_ENV)
    microsoft = registry.get("microsoft")
    zoho = registry.get("zoho")
    assert microsoft is not None and zoho is not None
    assert (
        microsoft.authorization_endpoint
        == "https://login.microsoftonline.com/common/oauth2/v2.0/authorize"
    )
    assert microsoft.pkce is True
    assert zoho.authorization_endpoint == "https://accounts.zoho.com/oauth/v2/auth"
    # Zoho's web-server flow is used without PKCE; the URL must still be complete.
    query = parse_qs(urlsplit(zoho.authorization_url(state="s", code_challenge=None)).query)
    assert "code_challenge" not in query and query["access_type"] == ["offline"]

    tuned = load_provider_registry(
        {
            **OTHERS_ENV,
            "CAAL_OAUTH_MICROSOFT_TENANT": "contoso.onmicrosoft.com",
            "CAAL_OAUTH_ZOHO_ACCOUNTS_DOMAIN": "https://accounts.zoho.eu",
            "CAAL_OAUTH_ZOHO_SCOPES": "ZohoMail.messages.READ  ZohoCalendar.event.READ",
        }
    )
    assert tuned.get("microsoft").authorization_endpoint == (
        "https://login.microsoftonline.com/contoso.onmicrosoft.com/oauth2/v2.0/authorize"
    )
    assert tuned.get("zoho").authorization_endpoint == "https://accounts.zoho.eu/oauth/v2/auth"
    assert tuned.get("zoho").token_endpoint == "https://accounts.zoho.eu/oauth/v2/token"
    # An override replaces the product scopes; the identity scope is always kept.
    assert tuned.get("zoho").scopes == (
        "AaaServer.profile.READ",
        "ZohoMail.messages.READ",
        "ZohoCalendar.event.READ",
    )


def test_invalid_values_are_reported_by_name_and_leave_the_provider_unconfigured() -> None:
    bad_origin = load_provider_registry({**GOOGLE_ENV, ENV_PUBLIC_ORIGIN: "jarvis.example.com/app"})
    assert bad_origin.configured("google") is False
    assert bad_origin.missing("google") == (ENV_PUBLIC_ORIGIN,)
    assert [p.name for p in bad_origin.problems] == [ENV_PUBLIC_ORIGIN]
    assert bad_origin.problems[0].problem.startswith("invalid")
    assert "jarvis.example.com/app" not in bad_origin.describe()

    bad_tenant = load_provider_registry({**OTHERS_ENV, "CAAL_OAUTH_MICROSOFT_TENANT": "evil/../x"})
    assert bad_tenant.configured("microsoft") is False
    assert bad_tenant.missing("microsoft") == ("CAAL_OAUTH_MICROSOFT_TENANT",)
    assert bad_tenant.configured("zoho") is True

    bad_zoho = load_provider_registry(
        {**OTHERS_ENV, "CAAL_OAUTH_ZOHO_ACCOUNTS_DOMAIN": "http://accounts.zoho.com"}
    )
    assert bad_zoho.configured("zoho") is False
    assert bad_zoho.missing("zoho") == ("CAAL_OAUTH_ZOHO_ACCOUNTS_DOMAIN",)

    partial = load_provider_registry(
        {"CAAL_OAUTH_GOOGLE_CLIENT_ID": "id-only", ENV_PUBLIC_ORIGIN: ORIGIN}
    )
    assert partial.configured("google") is False
    assert partial.missing("google") == ("CAAL_OAUTH_GOOGLE_CLIENT_SECRET",)

    blank = load_provider_registry({**GOOGLE_ENV, "CAAL_OAUTH_GOOGLE_CLIENT_SECRET": "   "})
    assert blank.missing("google") == ("CAAL_OAUTH_GOOGLE_CLIENT_SECRET",)

    local_dev = load_provider_registry({**GOOGLE_ENV, ENV_PUBLIC_ORIGIN: "http://localhost:3000"})
    assert local_dev.configured("google") is True
    assert local_dev.get("google").redirect_uri == "http://localhost:3000" + OAUTH_CALLBACK_PATH


def test_unknown_provider_names_are_refused_everywhere() -> None:
    registry = load_provider_registry(GOOGLE_ENV)
    for bad in ("apple", "GOOGLE", "", None, "google "):
        assert registry.get(bad) is None
        assert registry.configured(bad) is False
        assert registry.missing(bad) == ()


# --- production defaults: read-only scopes, provider delimiters, identity ---------------


READ_ONLY_DEFAULTS = {
    "google": {
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/calendar.readonly",
    },
    "microsoft": {"Mail.Read", "Calendars.Read", "offline_access"},
    "zoho": {
        "ZohoMail.accounts.READ",
        "ZohoMail.messages.READ",
        "ZohoCalendar.calendar.READ",
        "ZohoCalendar.event.READ",
    },
}
WRITE_MARKERS = (
    "gmail.modify",
    "gmail.send",
    "gmail.compose",
    "/auth/gmail\b",
    "/auth/calendar\b",
    "calendar.events\b",
    "Mail.ReadWrite",
    "Mail.Send",
    "Calendars.ReadWrite",
    ".ALL",
    ".CREATE",
    ".UPDATE",
    ".DELETE",
    ".WRITE",
)


def test_default_scopes_are_read_only_mail_and_calendar_plus_identity() -> None:
    import re

    for name, required in READ_ONLY_DEFAULTS.items():
        spec = oauth_providers.spec_for(name)
        assert required <= set(spec.default_scopes), name
        assert set(spec.identity_scopes) <= set(spec.default_scopes), name
        assert spec.identity_scopes, name
        for scope in spec.default_scopes:
            for marker in WRITE_MARKERS:
                assert re.search(marker, scope) is None, (name, scope, marker)
    assert oauth_providers.spec_for("google").identity_scopes == ("openid", "email")
    assert oauth_providers.spec_for("microsoft").identity_scopes == ("openid", "email")
    assert oauth_providers.spec_for("zoho").identity_scopes == ("AaaServer.profile.READ",)


def test_identity_scopes_survive_an_operator_scope_override() -> None:
    registry = load_provider_registry(
        {
            **GOOGLE_ENV,
            **OTHERS_ENV,
            "CAAL_OAUTH_GOOGLE_SCOPES": "https://www.googleapis.com/auth/gmail.readonly",
            "CAAL_OAUTH_ZOHO_SCOPES": "ZohoMail.messages.READ",
        }
    )
    google = registry.get("google")
    zoho = registry.get("zoho")
    assert google is not None and zoho is not None
    assert set(google.identity_scopes) <= set(google.scopes)
    assert "https://www.googleapis.com/auth/gmail.readonly" in google.scopes
    assert "https://www.googleapis.com/auth/calendar.readonly" not in google.scopes
    assert zoho.scopes == ("AaaServer.profile.READ", "ZohoMail.messages.READ")
    assert registry.problems == ()


def test_zoho_authorization_url_uses_zohos_comma_delimiter_and_others_use_spaces() -> None:
    registry = load_provider_registry({**GOOGLE_ENV, **OTHERS_ENV})
    zoho = registry.get("zoho")
    assert zoho.scope_delimiter == ","
    query = parse_qs(urlsplit(zoho.authorization_url(state="s", code_challenge=None)).query)
    (scope,) = query["scope"]
    assert " " not in scope
    assert set(scope.split(",")) == set(zoho.scopes)
    assert query["access_type"] == ["offline"] and query["prompt"] == ["consent"]

    for name in ("google", "microsoft"):
        config = registry.get(name)
        assert config.scope_delimiter == " "
        (scope,) = parse_qs(
            urlsplit(config.authorization_url(state="s", code_challenge="c")).query
        )["scope"]
        assert set(scope.split(" ")) == set(config.scopes)

    # A second account must be selectable at Google and Microsoft.
    google_prompt = parse_qs(
        urlsplit(registry.get("google").authorization_url(state="s", code_challenge="c")).query
    )["prompt"]
    assert google_prompt == ["select_account consent"]
    microsoft_prompt = parse_qs(
        urlsplit(registry.get("microsoft").authorization_url(state="s", code_challenge="c")).query
    )["prompt"]
    assert microsoft_prompt == ["select_account"]


def test_every_configured_provider_names_an_https_identity_endpoint() -> None:
    registry = load_provider_registry({**GOOGLE_ENV, **OTHERS_ENV})
    assert registry.get("google").identity_endpoint == (
        "https://openidconnect.googleapis.com/v1/userinfo"
    )
    assert registry.get("microsoft").identity_endpoint == "https://graph.microsoft.com/v1.0/me"
    assert registry.get("zoho").identity_endpoint == "https://accounts.zoho.com/oauth/user/info"
    tuned = load_provider_registry(
        {**OTHERS_ENV, "CAAL_OAUTH_ZOHO_ACCOUNTS_DOMAIN": "https://accounts.zoho.in"}
    )
    assert tuned.get("zoho").identity_endpoint == "https://accounts.zoho.in/oauth/user/info"


def test_zoho_data_center_codes_map_to_known_accounts_origins_only() -> None:
    origin = oauth_providers.zoho_accounts_origin
    assert origin("us") == "https://accounts.zoho.com"
    assert origin("eu") == "https://accounts.zoho.eu"
    assert origin("in") == "https://accounts.zoho.in"
    assert origin("au") == "https://accounts.zoho.com.au"
    for bad in ("", "US", "xx", "us ", None, 7, "https://accounts.zoho.com"):
        assert origin(bad) is None, bad
    assert oauth_providers.origin_of("https://accounts.zoho.com/oauth/v2/token") == (
        "https://accounts.zoho.com"
    )


def test_exchange_errors_carry_a_bounded_reason_and_setting_names_only() -> None:
    assert set(oauth_providers.EXCHANGE_REASONS) == {
        "provider_refused",
        "transport",
        "malformed_response",
        "identity_unavailable",
        "insufficient_scope",
        "datacenter_mismatch",
    }
    plain = oauth_providers.TokenExchangeError("refused")
    assert plain.reason == "provider_refused" and plain.settings == ()
    scoped = oauth_providers.TokenExchangeError(
        "scope", reason="insufficient_scope", settings=("CAAL_OAUTH_ZOHO_SCOPES",)
    )
    assert scoped.reason == "insufficient_scope"
    assert scoped.settings == ("CAAL_OAUTH_ZOHO_SCOPES",)
    with pytest.raises(ValueError):
        oauth_providers.TokenExchangeError("x", reason="something_else")

    grant = oauth_providers.TokenGrant(
        access_token="ya29.SECRET",
        refresh_token="1//SECRET",
        provider_account_id="1080000",
        account_label="ana@gmail.com",
    )
    rendered = repr(grant) + str(grant)
    for hidden in ("SECRET", "1080000", "ana@gmail.com"):
        assert hidden not in rendered
