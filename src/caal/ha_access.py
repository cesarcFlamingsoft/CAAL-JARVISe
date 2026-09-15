"""HA capability grants. Provider identities are separate, encrypted connections."""

import json
import secrets
import time
from contextlib import closing

from caal.user_store import Actor


class HAStore:
    def __init__(self, identity):
        self.identity = identity
        with closing(identity.store.connect()) as db:
            db.execute(
                (
                    "CREATE TABLE IF NOT EXISTS ha_connections (id TEXT PRIMARY KEY, owner_"
                    "id TEXT NOT NULL, account_key TEXT NOT NULL, payload TEXT, UNIQUE(owne"
                    "r_id, account_key))"
                )
            )
            db.execute(
                (
                    "CREATE TABLE IF NOT EXISTS ha_grants (user_id TEXT PRIMARY KEY, enable"
                    "d INTEGER NOT NULL, connection_id TEXT, revision INTEGER NOT NULL DEFA"
                    "ULT 1)"
                )
            )

    def active(self, user_id):
        user = self.identity.store.get_user(user_id)
        if user is None or user.status != "active":
            raise PermissionError("ha_access_denied")
        return user

    def grant(self, actor_id, user_id, *, enabled, connection_id):
        actor = self.active(actor_id)
        if actor.role != "admin":
            raise PermissionError("ha_admin_required")
        self.active(user_id)
        if type(enabled) is not bool:
            raise ValueError("invalid_grant")
        if connection_id is not None:
            connection = self.connection(connection_id)
            if not connection or connection["owner_id"] not in (actor_id, user_id):
                raise PermissionError("ha_identity_not_assignable")
        with closing(self.identity.store.connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                db.execute(
                    (
                        "INSERT INTO ha_grants VALUES (?, ?, ?, 1) ON CONFLICT(user_id) DO UPDA"
                        "TE SET enabled=excluded.enabled, connection_id=excluded.connection_id,"
                        " revision=ha_grants.revision+1"
                    ),
                    (user_id, int(enabled), connection_id),
                )
                self.identity.store._write_audit(
                    db,
                    actor=Actor.for_user(actor),
                    action="ha.grant" if enabled else "ha.revoke",
                    target_id=user_id,
                    detail={},
                    now=int(time.time()),
                )
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise

    def access(self, scope):
        denied = {"enabled": False, "status": "denied", "connection_id": None}
        try:
            user = self.active(getattr(scope, "user_id", None))
        except PermissionError:
            return denied
        with closing(self.identity.store.connect()) as db:
            row = db.execute("SELECT * FROM ha_grants WHERE user_id=?", (scope.user_id,)).fetchone()
        if not row and user.role == "admin":
            return {
                "enabled": True,
                "status": "service_account",
                "service_account": True,
                "connection_id": None,
            }
        if not row or not row["enabled"]:
            return denied
        connection = self.connection(row["connection_id"]) if row["connection_id"] else None
        return {
            "enabled": True,
            "status": "connected" if connection else "connection_required",
            "connection_id": row["connection_id"],
            "revision": row["revision"],
        }

    def connection(self, connection_id):
        with closing(self.identity.store.connect()) as db:
            row = db.execute(
                "SELECT * FROM ha_connections WHERE id=? AND payload IS NOT NULL", (connection_id,)
            ).fetchone()
        if not row:
            return None
        try:
            self.active(row["owner_id"])
        except PermissionError:
            return None
        data = json.loads(
            self.identity.config.keyring.decrypt(
                row["payload"], aad=f"ha:{row['owner_id']}:{row['id']}"
            )
        )
        return dict(data, id=row["id"], owner_id=row["owner_id"])

    def save_connection(self, owner_id, provider_user, tokens, *, endpoint, client_id):
        self.active(owner_id)
        account = provider_user.get("id")
        if not isinstance(account, str) or not account or len(account) > 128:
            raise ValueError("invalid_ha_identity")
        if not isinstance(tokens.get("access_token"), str) or not tokens["access_token"]:
            raise ValueError("invalid_ha_credentials")
        # Account key stays encrypted in payload; equality key is keyed to this owner.
        import hashlib

        account_key = hashlib.sha256((owner_id + endpoint + account).encode()).hexdigest()
        with closing(self.identity.store.connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute(
                "SELECT id FROM ha_connections WHERE owner_id=? AND account_key=?",
                (owner_id, account_key),
            ).fetchone()
            cid = old["id"] if old else "ha_" + secrets.token_hex(12)
            data = dict(
                tokens,
                provider_user=provider_user,
                endpoint=endpoint,
                client_id=client_id,
                expires_at=int(time.time()) + int(tokens.get("expires_in", 1800)),
            )
            encrypted = self.identity.config.keyring.encrypt(
                json.dumps(data), aad=f"ha:{owner_id}:{cid}"
            )
            db.execute(
                (
                    "INSERT INTO ha_connections VALUES (?, ?, ?, ?) ON CONFLICT(id) DO UPDA"
                    "TE SET payload=excluded.payload"
                ),
                (cid, owner_id, account_key, encrypted),
            )
            db.execute(
                (
                    "UPDATE ha_grants SET connection_id=?, revision=revision+1 WHERE user_i"
                    "d=? AND enabled=1 AND connection_id IS NULL"
                ),
                (cid, owner_id),
            )
            db.execute("COMMIT")
        return cid

    def credentials(self, scope):
        access = self.access(scope)
        if access["status"] != "connected":
            raise PermissionError("ha_connection_required")
        data = self.connection(access["connection_id"])
        if data is None:
            raise PermissionError("ha_connection_required")
        return data

    def disconnect(self, owner_id, connection_id, *, expected_token=None):
        if expected_token is None:
            self.active(owner_id)
        with closing(self.identity.store.connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            if expected_token is not None:
                row = db.execute(
                    "SELECT payload FROM ha_connections WHERE id=? AND owner_id=?",
                    (connection_id, owner_id),
                ).fetchone()
                if not row or row["payload"] is None:
                    db.execute("ROLLBACK")
                    return
                current = json.loads(
                    self.identity.config.keyring.decrypt(
                        row["payload"], aad=f"ha:{owner_id}:{connection_id}"
                    )
                )
                if current["access_token"] != expected_token:
                    db.execute("ROLLBACK")
                    return
            db.execute(
                "UPDATE ha_connections SET payload=NULL WHERE id=? AND owner_id=?",
                (connection_id, owner_id),
            )
            db.execute("COMMIT")

    def choices(self, actor_id, target_id):
        actor = self.active(actor_id)
        if actor_id != target_id and actor.role != "admin":
            raise PermissionError("ha_admin_required")
        with closing(self.identity.store.connect()) as db:
            rows = db.execute(
                "SELECT id FROM ha_connections WHERE owner_id IN (?, ?) "
                "AND payload IS NOT NULL LIMIT 32",
                (actor_id, target_id),
            ).fetchall()
        result = []
        for row in rows:
            c = self.connection(row["id"])
            if c:
                result.append(
                    {
                        "id": c["id"],
                        "label": str(c["provider_user"].get("name") or "Home Assistant user")[:80],
                        "shared": c["owner_id"] != target_id,
                        "ha_admin": bool(c["provider_user"].get("is_admin")),
                    }
                )
        return result

    def start_state(self, owner_id, endpoint, client_id, *, now=None):
        self.active(owner_id)
        now = int(time.time()) if now is None else now
        state = secrets.token_urlsafe(24)
        with closing(self.identity.store.connect()) as db:
            db.execute(
                (
                    "CREATE TABLE IF NOT EXISTS ha_oauth_states (state TEXT PRIMARY KEY, ow"
                    "ner_id TEXT, endpoint TEXT, client_id TEXT, expires INTEGER)"
                )
            )
            db.execute(
                "DELETE FROM ha_oauth_states WHERE expires<=? OR owner_id=?", (now, owner_id)
            )
            db.execute(
                "INSERT INTO ha_oauth_states VALUES (?, ?, ?, ?, ?)",
                (state, owner_id, endpoint, client_id, now + 600),
            )
        return state

    def consume_state(self, owner_id, state, *, now=None):
        self.active(owner_id)
        now = int(time.time()) if now is None else now
        with closing(self.identity.store.connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM ha_oauth_states WHERE state=? AND owner_id=? AND expires>?",
                (state, owner_id, now),
            ).fetchone()
            if not row:
                db.execute("ROLLBACK")
                raise PermissionError("invalid_ha_state")
            db.execute("DELETE FROM ha_oauth_states WHERE state=?", (state,))
            db.execute("COMMIT")
        return dict(row)

    def refresh_connection(self, previous, fresh):
        """Compare-and-update existing credentials; disconnect always wins over resurrection."""
        self.active(previous["owner_id"])
        with closing(self.identity.store.connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT payload FROM ha_connections WHERE id=? AND owner_id=?",
                (previous["id"], previous["owner_id"]),
            ).fetchone()
            if not row or row["payload"] is None:
                db.execute("ROLLBACK")
                raise PermissionError("ha_connection_required")
            aad = f"ha:{previous['owner_id']}:{previous['id']}"
            data = json.loads(self.identity.config.keyring.decrypt(row["payload"], aad=aad))
            if data["access_token"] != previous["access_token"]:
                db.execute("ROLLBACK")
                raise PermissionError("ha_credentials_changed")
            data["access_token"] = fresh["access_token"]
            data["expires_at"] = int(time.time()) + int(fresh.get("expires_in", 1800))
            db.execute(
                "UPDATE ha_connections SET payload=? WHERE id=?",
                (self.identity.config.keyring.encrypt(json.dumps(data), aad=aad), previous["id"]),
            )
            db.execute("COMMIT")
