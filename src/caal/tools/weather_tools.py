"""Read the dashboard weather for the runtime-bound user, without changing location."""

from __future__ import annotations

from caal.user_scope import is_valid_user_id


def _runtime():
    from caal import user_api, weather_api

    return weather_api.get_weather_runtime(user_api.get_runtime())


def session_unavailable_result() -> dict:
    return {
        "status": "unauthorized",
        "message": "Weather requires a verified signed-in user. No location was read.",
        "data": {},
    }


async def current_weather(*, user_id: str | None = None) -> dict:
    """Only the execution layer supplies user_id; no model location overrides."""
    if not is_valid_user_id(user_id):
        return session_unavailable_result()
    try:
        runtime = _runtime()
        if runtime is None:
            raise RuntimeError("weather_not_configured")
        snapshot = await runtime.client.current(user_id)
        data = dict(generated_at=runtime.now(), **snapshot.view())
    except Exception:
        # Never expose coordinates, identities, URLs or provider exception text.
        return {
            "status": "unavailable",
            "message": "The dashboard weather backend is unavailable. Do not invent conditions.",
            "data": {},
        }
    instructions = {
        "ok": "Answer from these current conditions in the conversation language.",
        "stale": "These are older cached conditions: the upstream refresh failed. "
        "Explicitly say they are stale; do not present them as current.",
        "no_location": "No weather location is set or the shared browser location expired. "
        "Ask the user to choose a city or share location in the dashboard weather widget. "
        "Do not infer location from timezone.",
        "unavailable": "The weather provider is unavailable and there is no cached reading. "
        "Do not invent conditions.",
    }
    return {
        "status": snapshot.state,
        "message": instructions[snapshot.state] + " This tool provides current observations only, "
        "not future forecasts. Report the conditions label faithfully, translating it if needed. "
        "Zero precipitation means none measured at observation time, never none expected. "
        "Never predict later weather or invent missing measurements.",
        "data": data,
    }


def spoken_failure(result: dict, language: str) -> str | None:
    """Actionable failures only; observations still come from the scoped backend."""
    replies = {
        "no_location": (
            "Choose a city or share your location in the dashboard weather widget, then ask again.",
            "Elige una ciudad o comparte tu ubicación en el widget del tiempo del panel "
            "y vuelve a preguntar.",
        ),
        "unavailable": (
            "I couldn't get the weather right now. Please try again in a moment.",
            "No pude consultar el tiempo ahora. Vuelve a intentarlo en un momento.",
        ),
        "unauthorized": (
            "Please sign in to use the weather location saved in your dashboard.",
            "Inicia sesión para usar la ubicación del tiempo guardada en tu panel.",
        ),
    }
    pair = replies.get(result.get("status"))
    return pair[language == "es"] if pair else None
