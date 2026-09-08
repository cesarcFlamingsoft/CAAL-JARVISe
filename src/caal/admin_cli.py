"""Operator commands for standalone sign-in.

Three things an operator needs and should never have to improvise:

``generate``  mint a random one-time password, print it once, and print the
              ``CAAL_BOOTSTRAP_ADMIN_PASSWORD_HASH=...`` line to paste into
              ``.env``. The plaintext is never written to disk by this command.
``seed``      apply the configured bootstrap hash to the database now, instead
              of waiting for the next backend start. Idempotent.
``reset``     issue a fresh one-time password for an existing user, ending
              their sessions and forcing a change at next sign-in.
``status``    report whether the configuration validates, by variable name.

The generated password is written to standard output and nowhere else: not to
a log, not to the audit trail, not to the database. Only its hash is stored.

    python3 -m caal.admin_cli generate --email you@example.com
    python3 -m caal.admin_cli seed
    python3 -m caal.admin_cli reset --email someone@example.com
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .access_jwt import normalize_email
from .local_auth import LocalAuth
from .password_hash import generate_password, hash_password
from .security_config import (
    ENV_BOOTSTRAP_ADMIN_PASSWORD_HASH,
    MultiUserConfig,
    load_multi_user_config,
    log_startup_status,
)
from .user_store import ADMIN, Actor, UserStore

__all__ = ["main"]


def _config() -> MultiUserConfig:
    status = load_multi_user_config()
    if status.config is None:
        print(status.describe(), file=sys.stderr)
        raise SystemExit(2)
    return status.config


def _auth(config: MultiUserConfig) -> LocalAuth:
    store = UserStore(config.store_path, keyring=config.keyring)
    return LocalAuth(store, policy=config.session_policy)


def _print_credential(email: str, password: str, *, hashed: str | None = None) -> None:
    """The one place a plaintext password is ever rendered."""
    print()
    print("  Account:  " + email)
    print("  Password: " + password)
    print()
    print("  This is shown once. Store it in a password manager now.")
    print("  It must be changed at first sign-in.")
    if hashed is not None:
        print()
        print("  Add this line to your .env (the hash, never the password):")
        print()
        print(f"  {ENV_BOOTSTRAP_ADMIN_PASSWORD_HASH}={hashed}")
    print()


def _generate(args: argparse.Namespace) -> int:
    """Mint a one-time password and its hash without touching the database."""
    email = normalize_email(args.email)
    password = generate_password(length=args.length)
    _print_credential(email, password, hashed=hash_password(password))
    return 0


def _seed(args: argparse.Namespace) -> int:
    config = _config()
    if not config.bootstrap_admin_password_hash:
        print(
            f"{ENV_BOOTSTRAP_ADMIN_PASSWORD_HASH} is not set; nothing to seed. "
            "Run `generate` first and put the hash in your .env.",
            file=sys.stderr,
        )
        return 2
    outcome = _auth(config).ensure_bootstrap_admin(
        config.bootstrap_admin_email, config.bootstrap_admin_password_hash
    )
    if outcome.refused:
        print(
            "Refused: this deployment already has users, so the bootstrap "
            "administrator was not created. Add the account from the admin panel.",
            file=sys.stderr,
        )
        return 1
    if outcome.created:
        print(f"Created the administrator account ({outcome.user_id}) with the bootstrap password.")
    elif outcome.credential_installed:
        print(f"Installed the bootstrap password on the existing account ({outcome.user_id}).")
    else:
        print(
            f"Nothing to do: the account ({outcome.user_id}) already has a password. "
            "The bootstrap hash is never re-applied over one."
        )
    return 0


def _reset(args: argparse.Namespace) -> int:
    config = _config()
    auth = _auth(config)
    email = normalize_email(args.email)
    profile = auth.store.get_user_by_email(email)
    if profile is None:
        print("No account with that email.", file=sys.stderr)
        return 1
    password = auth.admin_reset_password(profile.user_id, actor=Actor.system())
    _print_credential(email, password)
    print("  Their existing sessions have been signed out.")
    print()
    return 0


def _status(args: argparse.Namespace) -> int:
    status = load_multi_user_config()
    log_startup_status(status)
    print(status.describe())
    if status.config is None:
        return 1
    store = UserStore(status.config.store_path, keyring=status.config.keyring)
    users = store.list_users()
    auth = LocalAuth(store, policy=status.config.session_policy)
    print(f"Store: {status.config.store_path} (schema v{store.schema_version()})")
    print(f"Users: {len(users)}")
    for profile in users:
        # Opaque ids and roles only: no emails on a terminal that may be logged.
        print(
            f"  {profile.user_id}  {profile.role:<6} {profile.status:<9} "
            f"password={'yes' if auth.has_password(profile.user_id) else 'no'}"
            + ("  <- bootstrap admin" if profile.role == ADMIN else "")
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m caal.admin_cli", description=__doc__.split("\n\n")[0]
    )
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="mint a one-time password and its hash")
    generate.add_argument("--email", required=True, help="the administrator's email")
    generate.add_argument("--length", type=int, default=24, help="password length (default 24)")
    generate.set_defaults(func=_generate)

    seed = sub.add_parser("seed", help="apply the configured bootstrap hash to the database")
    seed.set_defaults(func=_seed)

    reset = sub.add_parser("reset", help="issue a fresh one-time password for a user")
    reset.add_argument("--email", required=True)
    reset.set_defaults(func=_reset)

    status = sub.add_parser("status", help="report configuration and accounts")
    status.set_defaults(func=_status)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
