"""TTS preferences in the existing identity database; never in global settings."""

import json
from contextlib import closing


class TTSStore:
    def __init__(self, identity):
        self.identity = identity
        with closing(identity.store.connect()) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS tts_preferences (user_id TEXT PRIMARY KEY, "
                "payload TEXT NOT NULL)"
            )

    def preference(self, user_id):
        with closing(self.identity.store.connect()) as db:
            row = db.execute(
                "SELECT payload FROM tts_preferences WHERE user_id = ?", (user_id,)
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    def save_preference(self, user_id, value):
        with closing(self.identity.store.connect()) as db:
            db.execute(
                "INSERT INTO tts_preferences VALUES (?, ?) ON CONFLICT(user_id) DO UPDATE "
                "SET payload=excluded.payload",
                (user_id, json.dumps(value)),
            )

    def config(self):
        with closing(self.identity.store.connect()) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS tts_voicebox_config (id INTEGER PRIMARY KEY "
                "CHECK(id=1), payload TEXT NOT NULL)"
            )
            row = db.execute("SELECT payload FROM tts_voicebox_config WHERE id=1").fetchone()
        if not row:
            return None
        return json.loads(
            self.identity.config.keyring.decrypt(row["payload"], aad="tts:voicebox:config")
        )

    def save_config(self, value):
        self.config()
        encrypted = self.identity.config.keyring.encrypt(
            json.dumps(value), aad="tts:voicebox:config"
        )
        with closing(self.identity.store.connect()) as db:
            db.execute(
                "INSERT INTO tts_voicebox_config VALUES (1, ?) ON CONFLICT(id) DO UPDATE "
                "SET payload=excluded.payload",
                (encrypted,),
            )
