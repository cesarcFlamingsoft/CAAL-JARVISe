"""Reply-language preference in the identity database; never in global settings.

Mirrors ``caal.tts_store`` so a personal choice can never reach another user's
session: there is one row per verified ``user_id`` and no deployment-wide key.
"""

from contextlib import closing

from .language_policy import AUTO, SUPPORTED


class LanguageStore:
    def __init__(self, identity):
        self.identity = identity
        with closing(identity.store.connect()) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS language_preferences (user_id TEXT PRIMARY KEY, "
                "language TEXT NOT NULL)"
            )

    def preference(self, user_id):
        """The user's setting, defaulting to ``auto`` for anyone who never chose."""
        if not user_id:
            return AUTO
        with closing(self.identity.store.connect()) as db:
            row = db.execute(
                "SELECT language FROM language_preferences WHERE user_id = ?", (user_id,)
            ).fetchone()
        return row["language"] if row and row["language"] in SUPPORTED else AUTO

    def save_preference(self, user_id, language):
        if language not in SUPPORTED:
            raise ValueError(f"Unsupported language preference: {language!r}")
        with closing(self.identity.store.connect()) as db:
            db.execute(
                "INSERT INTO language_preferences VALUES (?, ?) ON CONFLICT(user_id) DO UPDATE "
                "SET language=excluded.language",
                (user_id, language),
            )
