"""Keep the suite hermetic with respect to the developer's real environment.

``voice_agent.py`` calls ``load_dotenv()`` at import time, and several tests
import it. Without the fixture below, a machine whose ``.env`` actually
configures multi-user identity -- which is exactly what a working deployment
looks like -- would run the tests against that real configuration: the cached
identity runtime would be built from the operator's settings and would try to
open the container's data directory.

So the tests neither read ``.env`` nor inherit any identity variable from the
process. A test that wants identity configured sets it explicitly, which also
makes each test say what it depends on.
"""

from __future__ import annotations

import dotenv
import pytest

from caal import user_api
from caal.security_config import ENV_DATA_DIR, OPTIONAL_ENV, REQUIRED_ENV


@pytest.fixture(autouse=True)
def hermetic_environment(monkeypatch: pytest.MonkeyPatch):
    """Neutralize ``.env`` loading and clear inherited identity settings."""
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    for name in (*REQUIRED_ENV, *OPTIONAL_ENV, ENV_DATA_DIR):
        monkeypatch.delenv(name, raising=False)
    # The runtime is process-wide and cached; a stale one would outlive the
    # environment it was built from.
    user_api.reset_runtime()
    yield
    user_api.reset_runtime()
