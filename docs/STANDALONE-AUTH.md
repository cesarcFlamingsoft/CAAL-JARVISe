# Standalone authentication (no Cloudflare, no identity provider)

JARVIS/CAAL signs people in on its own: an email and a password, checked
against Argon2id hashes in the deployment's own SQLite database, with
server-side sessions. It needs no Cloudflare Access, no OAuth provider, and no
outbound network. Cloudflare Access is still supported as an *optional
additional* provider; see [Optional: Cloudflare Access](#optional-cloudflare-access).

For what users and administrators can do once they are signed in -- roles,
callback numbers, the audit trail, per-user memory isolation -- see
[MULTI-USER-IDENTITY.md](MULTI-USER-IDENTITY.md). This document is only about
how someone proves who they are.

## How it works

```
Browser ──password──▶ Next.js BFF ──signed principal──▶ CAAL backend
        ◀─session cookie──┘                             │  Argon2id verify
                                                        │  session row (SHA-256 of token)
        ──cookie──▶ BFF ──session token──▶ backend ──────┘  re-resolved every request
```

1. The browser posts an email and password to the BFF. The request must carry
   our own `Origin`, a valid double-submit CSRF token, and pass a rate limit.
2. The BFF forwards it once to the CAAL backend over the internal network,
   proving itself with a short-lived, single-use HMAC principal.
3. The backend verifies the password against the account's **Argon2id** hash
   (64 MiB, t=3, p=2), applies per-account lockout, and on success creates a
   session row. It returns an opaque 256-bit token.
4. The BFF puts that token in an **HttpOnly, SameSite=Lax, Secure** cookie. The
   token never appears in a response body, so no script can read it.
5. On every later request the BFF hands the token back to the backend, which
   re-reads the session, the user's role, their status and their
   forced-password-change flag **from the database**. Nothing is trusted from
   inside the cookie, so suspending or resetting someone takes effect on their
   very next request.

The database stores only the **SHA-256 digest** of a session token, so a
database leak hands out no live sessions.

## Configuration

| Variable | Where | Secret | Value |
| --- | --- | --- | --- |
| `CAAL_INTERNAL_AUTH_SECRET` | agent, frontend | **yes** | ≥ 32 random characters: `python3 -c "import secrets; print(secrets.token_urlsafe(48))"` |
| `CAAL_PROFILE_ENCRYPTION_KEYS` | agent | **yes** | `v1:<base64url 32 bytes>`: `python3 -c "from caal.profile_crypto import generate_key_material as g; print(g(version=1))"` |
| `CAAL_BOOTSTRAP_ADMIN_EMAIL` | agent | no | The first administrator's email |
| `CAAL_BOOTSTRAP_ADMIN_PASSWORD_HASH` | agent | **yes** | Argon2id hash of their one-time password — **never the password itself** |
| `CAAL_IDENTITY_API_URL` | frontend | no | Backend base URL, `http://agent:8889` in Docker (compose sets it) |
| `CAAL_PUBLIC_ORIGIN` | frontend | no | Public origin of the portal, e.g. `https://jarvis.example.com` |
| `CAAL_PASSWORD_LOGIN` | agent, frontend | no | `false` disables password sign-in; only allowed when Access is configured |
| `CAAL_SESSION_IDLE_MINUTES` | agent | no | Sliding timeout, default `480` (8 h) |
| `CAAL_SESSION_ABSOLUTE_HOURS` | agent | no | Hard ceiling, default `168` (7 d) |
| `CAAL_ALLOW_INSECURE_COOKIES` | frontend | no | `true` permits a session cookie over plain HTTP (LAN only) |
| `CAAL_REQUIRE_IDENTITY_FOR_SESSIONS` | frontend | no | `true` refuses anonymous voice sessions |

The first three are required. Anything missing or invalid is reported **by
variable name only** at startup, the identity endpoints answer `503`, and voice
sessions carry no user until it is fixed. Never prefix any of these with
`NEXT_PUBLIC_`.

### TLS is the default, and plain HTTP fails closed

A session cookie is a bearer credential. Over HTTPS it is issued `Secure`. Over
plain HTTP the BFF **refuses to sign anyone in** and answers
`400 insecure_transport`, rather than silently handing out a token that travels
in the clear.

A LAN deployment with no TLS can opt in with `CAAL_ALLOW_INSECURE_COOKIES=true`.
That is a real downgrade — anyone on the network segment can read the session
token — so it is off by default and is called out in the startup summary.

## Bootstrapping the first administrator

The plaintext password is never in the source tree, the image, or `.env`. Only
its hash is stored, and the account is forced to change it at first sign-in.

1. **Mint a one-time password and its hash.** This touches no database:

   ```bash
   python3 -m caal.admin_cli generate --email you@example.com
   ```

   It prints the password once, and the `CAAL_BOOTSTRAP_ADMIN_PASSWORD_HASH=...`
   line to paste into `.env`. Put the password straight into a password
   manager; it is not recoverable.

2. **Put the hash in `.env`** (which is gitignored) along with
   `CAAL_BOOTSTRAP_ADMIN_EMAIL`, `CAAL_INTERNAL_AUTH_SECRET` and
   `CAAL_PROFILE_ENCRYPTION_KEYS`.

3. **Seed the account.** The backend does this automatically on startup, or you
   can do it now:

   ```bash
   python3 -m caal.admin_cli seed
   python3 -m caal.admin_cli status
   ```

4. **Sign in** at `/login` with that email and the one-time password. You are
   sent straight to `/change-password` and can reach nothing else until you
   have chosen your own password.

The seed is **idempotent and one-way**: once the account has any password of
its own, the bootstrap hash is never re-applied. Restarting the deployment
therefore cannot reinstate the one-time password after you have changed it, and
leaving the variable in `.env` afterwards is inert.

It is also **deny-by-default**: an account is only *created* when the user table
is empty. In a deployment that already has users, a leaked
`CAAL_BOOTSTRAP_ADMIN_PASSWORD_HASH` cannot mint a new administrator; the
attempt is refused and logged as a `SECURITY:` error.

## Password reset is deliberately disabled

There is **no self-service password reset**. CAAL has no mail transport, and an
emailed reset link would be a recovery channel the deployment cannot secure —
in practice it would become the weakest way into every account.

Instead, an administrator issues a one-time password out of band:

* **In the admin panel:** *Reset password* on the user's row. The generated
  password is displayed exactly once and cannot be retrieved again.
* **On the command line:** `python3 -m caal.admin_cli reset --email them@example.com`

Either way the user's live sessions are revoked immediately and they must
choose a new password at their next sign-in.

If *every* administrator loses their password, recover by generating a fresh
hash and applying it directly:

```bash
python3 -m caal.admin_cli generate --email admin@example.com   # note the password
# put the hash in .env, then, because the account already exists:
python3 -m caal.admin_cli reset --email admin@example.com
```

## Security controls

- **Hashing.** Argon2id via OpenSSL through `cryptography` (m=64 MiB, t=3,
  p=2), per-password random 16-byte salt, self-describing PHC encoding so the
  cost can be raised later without invalidating stored hashes; a verified
  password whose hash is below current policy is transparently upgraded.
  `hashlib.scrypt` is the fallback where the linked OpenSSL predates Argon2.
- **Anti-enumeration.** An unknown email, an account with no password, a
  suspended account and a wrong password all return the same
  `invalid_credentials`. The unknown-account path performs a dummy Argon2
  verification so the four cannot be told apart by timing either.
- **Lockout.** Five consecutive failures lock an account; the window doubles
  with each further failure up to 15 minutes. A lockout is only *disclosed* to
  a caller who already supplied the correct password, so it cannot be used to
  discover which accounts exist.
- **Rate limiting.** Per browser (the BFF signs a hashed client key into its
  assertion, so all traffic arriving from one BFF is not treated as one
  caller), plus a limit at the BFF edge, plus the per-account lockout.
- **Sessions.** 256-bit opaque tokens, stored only as SHA-256 digests, with a
  sliding idle timeout, a hard absolute ceiling, a per-user cap, and
  revocation on password change, on administrative reset, and on sign-out.
- **Cookies.** `HttpOnly`, `SameSite=Lax`, `Secure` (see above), `Path=/`.
- **CSRF.** Double-submit token in an `HttpOnly`, `SameSite=Strict` cookie,
  echoed in `X-CAAL-CSRF`, compared in constant time, on *every* state-changing
  route including sign-in and sign-out. The token is rotated on sign-in and on
  password change, so one planted beforehand cannot be replayed.
- **Origin.** Every mutation must carry our own `Origin` (or `Referer`); a
  request with neither is refused.
- **Open redirects.** `?next=` is reduced to a same-origin path by parsing it
  against a throwaway base and requiring the origin to survive, which rejects
  `//evil`, `/\evil`, absolute URLs and `javascript:` in one rule. `/login`
  itself is never a target.
- **Authorization.** Roles and status are read from the database on every
  request, never from a token claim. A user with a forced-change flag is
  refused every route except changing their password and signing out — enforced
  in the API, not only in the UI.
- **Audit.** `auth.login` (with outcome), `auth.lockout`, `auth.password.change`,
  `auth.password.reset`, `auth.password.set` and `auth.bootstrap` are recorded
  with opaque user ids only. Passwords, tokens, emails and client addresses
  never reach the audit trail or the logs.

## Optional: Cloudflare Access

Setting **both** `CF_ACCESS_TEAM_DOMAIN` and `CF_ACCESS_AUD` additionally
accepts a verified Cloudflare Access assertion as identity, mapped to the same
opaque users. Setting only one of the pair is a configuration error, not a
silent downgrade.

Enabling it weakens nothing above: password sign-in keeps its own hashing,
lockout, session and CSRF controls, and `POST /auth/resolve` — the only route
that consults an Access assertion — does not exist at all when Access is
unconfigured.

The session cookie is tried first; a valid Access assertion is only consulted
if no session resolves.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `400 insecure_transport` on sign-in | Plain HTTP without `CAAL_ALLOW_INSECURE_COOKIES=true`. Use TLS. |
| `403 csrf` | The page was loaded before the CSRF cookie existed. Reload. |
| `403 bad_origin` | A proxy is stripping `Origin`/`Referer`, or `CAAL_PUBLIC_ORIGIN` does not match the real origin. |
| `503 identity_not_configured` | A required variable is missing or invalid; the agent log names it. |
| `429 locked` | Per-account lockout. Wait, or issue a reset. |
| Signed in but everything is `403 password_change_required` | Expected with a one-time password. Go to `/change-password`. |
| Seed says "Refused: this deployment already has users" | Deny-by-default. Create the account from the admin panel instead. |
