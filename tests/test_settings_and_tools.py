import importlib
import json

REDACTED = "********"


def configure_settings_module(monkeypatch, tmp_path):
    from caal import settings as settings_module

    settings_path = tmp_path / "settings.json"
    prompt_dir = tmp_path / "prompt"
    prompt_dir.mkdir()
    (prompt_dir / "default.md").write_text("default prompt")

    monkeypatch.setattr(settings_module, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(settings_module, "PROMPT_DIR", prompt_dir)
    settings_module._settings_cache = None
    return settings_module, settings_path


def test_safe_settings_redacts_secret_values(monkeypatch, tmp_path):
    settings_module, settings_path = configure_settings_module(monkeypatch, tmp_path)
    settings_path.write_text(
        json.dumps(
            {
                "groq_api_key": "gsk_live_secret",
                "hass_token": "ha_secret",
                "n8n_token": "n8n_secret",
                "friday_token": "friday_secret",
            }
        )
    )

    safe = settings_module.load_settings_safe()

    assert safe["groq_api_key"] == REDACTED
    assert safe["hass_token"] == REDACTED
    assert safe["n8n_token"] == REDACTED
    assert safe["friday_token"] == REDACTED


def test_save_settings_preserves_existing_secret_when_redacted_placeholder_is_submitted(
    monkeypatch, tmp_path
):
    settings_module, settings_path = configure_settings_module(monkeypatch, tmp_path)
    settings_path.write_text(
        json.dumps(
            {
                "groq_api_key": "gsk_live_secret",
                "hass_token": "ha_secret",
                "n8n_token": "n8n_secret",
                "friday_token": "friday_secret",
            }
        )
    )

    settings_module.save_settings(
        {
            "agent_name": "Jarvis",
            "groq_api_key": REDACTED,
            "hass_token": REDACTED,
            "n8n_token": REDACTED,
            "friday_token": REDACTED,
        }
    )

    saved = json.loads(settings_path.read_text())
    assert saved["agent_name"] == "Jarvis"
    assert saved["groq_api_key"] == "gsk_live_secret"
    assert saved["hass_token"] == "ha_secret"
    assert saved["n8n_token"] == "n8n_secret"
    assert saved["friday_token"] == "friday_secret"


def test_native_assistant_defaults_disable_legacy_n8n_and_enable_core_capabilities():
    from caal import settings as settings_module

    defaults = settings_module.DEFAULT_SETTINGS

    assert defaults["n8n_enabled"] is False
    assert defaults["native_tools_enabled"] is True
    assert defaults["email_accounts"] == []
    assert defaults["calendar_sources"] == []
    assert defaults["reminders_provider"] == "local"
    assert defaults["alarms_enabled"] is True


def test_native_tool_registry_contains_core_assistant_capabilities():
    registry_module = importlib.import_module("caal.tools.registry")
    registry = registry_module.create_default_registry()

    tool_names = set(registry.names())

    assert "email.send" in tool_names
    assert "email.search" in tool_names
    assert "calendar.list_events" in tool_names
    assert "calendar.create_event" in tool_names
    assert "reminders.create" in tool_names
    assert "alarms.set" in tool_names
    assert registry.get("email.send").requires_confirmation is True
    assert registry.get("calendar.create_event").requires_confirmation is True


def test_safe_settings_recursively_redacts_email_and_calendar_secrets(monkeypatch, tmp_path):
    settings_module, settings_path = configure_settings_module(monkeypatch, tmp_path)
    settings_path.write_text(
        json.dumps(
            {
                "email_accounts": [
                    {
                        "id": "personal",
                        "provider": "zoho",
                        "smtp_password": "smtp_secret",
                        "imap_password": "imap_secret",
                    }
                ],
                "calendar_sources": [
                    {
                        "id": "work",
                        "provider": "zoho_caldav",
                        "password": "calendar_secret",
                        "access_token": "calendar_token",
                    }
                ],
            }
        )
    )

    safe = settings_module.load_settings_safe()

    assert safe["email_accounts"][0]["smtp_password"] == REDACTED
    assert safe["email_accounts"][0]["imap_password"] == REDACTED
    assert safe["calendar_sources"][0]["password"] == REDACTED
    assert safe["calendar_sources"][0]["access_token"] == REDACTED


def test_save_settings_preserves_nested_secrets_when_placeholders_are_submitted(
    monkeypatch, tmp_path
):
    settings_module, settings_path = configure_settings_module(monkeypatch, tmp_path)
    settings_path.write_text(
        json.dumps(
            {
                "email_accounts": [
                    {
                        "id": "personal",
                        "provider": "zoho",
                        "smtp_password": "smtp_secret",
                        "imap_password": "imap_secret",
                    }
                ],
                "calendar_sources": [
                    {
                        "id": "work",
                        "provider": "zoho_caldav",
                        "password": "calendar_secret",
                    }
                ],
            }
        )
    )

    settings_module.save_settings(
        {
            "email_accounts": [
                {
                    "id": "personal",
                    "provider": "zoho",
                    "smtp_password": REDACTED,
                    "imap_password": REDACTED,
                    "display_name": "Personal Mail",
                }
            ],
            "calendar_sources": [
                {
                    "id": "work",
                    "provider": "zoho_caldav",
                    "password": REDACTED,
                    "display_name": "Work Calendar",
                }
            ],
        }
    )

    saved = json.loads(settings_path.read_text())
    assert saved["email_accounts"][0]["smtp_password"] == "smtp_secret"
    assert saved["email_accounts"][0]["imap_password"] == "imap_secret"
    assert saved["email_accounts"][0]["display_name"] == "Personal Mail"
    assert saved["calendar_sources"][0]["password"] == "calendar_secret"
    assert saved["calendar_sources"][0]["display_name"] == "Work Calendar"


def test_zoho_email_provider_preset_fills_runtime_ui_defaults():
    from caal import settings as settings_module

    account = settings_module.apply_email_provider_preset(
        {
            "id": "personal",
            "provider": "zoho",
            "email": "cesar@example.com",
        }
    )

    assert account["imap_host"] == "imap.zoho.com"
    assert account["imap_port"] == 993
    assert account["imap_ssl"] is True
    assert account["smtp_host"] == "smtp.zoho.com"
    assert account["smtp_port"] == 587
    assert account["smtp_starttls"] is True
    assert account["imap_username"] == "cesar@example.com"
    assert account["smtp_username"] == "cesar@example.com"
