# Multi-user identity, authorization and administration

JARVIS/CAAL can run for several people at once. Each person is a **user** with
an opaque id, a verified email, a display name, a role (`admin` or `member`), a
status (`active` or `suspended`) and at most one **approved callback number**.
JARVIS saves and recalls memories per user, dials only that user's approved
number, and never lets one user see or act on another user's conversations,
background tasks, memories, callbacks or devices.

Until every setting below is configured, JARVIS runs exactly as before: one
shared single-user deployment. It never pretends to be multi-user.

> **Sign-in has moved.** Authentication is now standalone -- email and password
> against Argon2id hashes in CAAL's own database, with server-side sessions and
> no identity provider required. See **[STANDALONE-AUTH.md](STANDALONE-AUTH.md)**
> for how people sign in, how the first administrator is bootstrapped, and how
> passwords are reset. Cloudflare Access, described below, is now an *optional
> additional* provider rather than a requirement.

## How identity works

```
Browser ──Cloudflare Access (signed JWT)──▶ Next.js BFF ──signed principal──▶ CAAL backend
                                                │                                  │
                                                └──signed room configuration──▶ LiveKit ──▶ voice agent
```

1. **Cloudflare Access** in front of the public portal authenticates the
   person and attaches a signed JWT (`Cf-Access-Jwt-Assertion`) to every
   request. The plain `Cf-Access-Authenticated-User-Email` header is never
   trusted.
2. The **Next.js BFF** verifies that JWT cryptographically against the team's
   published keys (`https://<team>.cloudflareaccess.com/cdn-cgi/access/certs`,
   cached, rotated on unknown key id, refreshed at most once per cooldown),
   checking algorithm (`RS256` only), issuer (the team domain), the
   application audience tag, lifetime, and that a verified `email` is present
   (service tokens and login "meta" tokens are refused).
3. The BFF then asks the **CAAL backend** to resolve the email to a user with a
   short-lived, single-use, HMAC-signed *identity assertion* **and** the
   original Access JWT. The backend verifies both independently and refuses
   if they disagree. Only an opaque `user_id`, role, status and display name
   come back. Unknown emails are refused (deny by default).
4. Every later BFF → backend call carries a fresh single-use *backend
   principal* naming that opaque id. The backend loads the user from its own
   database and decides authorization there, never from token claims.
5. For a voice session the BFF mints a room-bound *agent principal* and puts it
   in the **LiveKit room configuration it signs**, so the voice agent receives
   it as job metadata that no participant can read or forge. Room names are
   server-generated and unguessable; nothing about the user appears in room
   metadata, participant identities, or browser responses.
6. A telephone caller is identified only after the DTMF PIN gate, by matching
   caller-id against the users' approved numbers. An outbound leg is
   identified by the opaque id the trusted coordinator dispatched, and the
   worker re-reads that user's approved number from the database before the
   SIP call is placed.

Anything that fails to verify leaves a session **anonymous**: it can still
talk to JARVIS, but memory, phone handoff and callbacks are refused with a
spoken explanation.

### Bootstrap

With standalone sign-in the first administrator is seeded from
`CAAL_BOOTSTRAP_ADMIN_EMAIL` plus `CAAL_BOOTSTRAP_ADMIN_PASSWORD_HASH`; see
[STANDALONE-AUTH.md](STANDALONE-AUTH.md#bootstrapping-the-first-administrator).

With Cloudflare Access configured, the alternative still applies: the very
first time the configured `CAAL_BOOTSTRAP_ADMIN_EMAIL` signs in through
Cloudflare Access into an empty user table, that account is created as
the only administrator. The pre-multi-user memory store (the single-user
memories) is adopted by that administrator once, and the adoption is recorded
in the audit trail as a count. No other email can ever bootstrap, and the
bootstrap never happens again once a user exists.

## Configuration

The Cloudflare values below are **optional** -- they enable Access as an
additional identity provider. The required values are
`CAAL_INTERNAL_AUTH_SECRET`, `CAAL_PROFILE_ENCRYPTION_KEYS` and
`CAAL_BOOTSTRAP_ADMIN_EMAIL`, documented in
[STANDALONE-AUTH.md](STANDALONE-AUTH.md#configuration). A partial or invalid configuration is reported **by variable name
only** as an error at startup, the identity endpoints answer `503`, and voice
sessions carry no user until it is fixed.

| Variable | Where | Secret | Value |
| --- | --- | --- | --- |
| `CF_ACCESS_TEAM_DOMAIN` | agent, frontend | no | `https://<team>.cloudflareaccess.com` (the host in the Access login redirect) |
| `CF_ACCESS_AUD` | agent, frontend | no | The Access application's Audience (AUD) tag, 64 hex characters (Zero Trust → Access → Applications → the portal app → Overview) |
| `CAAL_INTERNAL_AUTH_SECRET` | agent, frontend | **yes** | ≥ 32 random characters, e.g. `python3 -c "import secrets; print(secrets.token_urlsafe(48))"` |
| `CAAL_PROFILE_ENCRYPTION_KEYS` | agent | **yes** | `v1:<base64url 32 bytes>[,v2:…]`, e.g. `python3 -c "from caal.profile_crypto import generate_key_material as g; print(g(version=1))"` |
| `CAAL_BOOTSTRAP_ADMIN_EMAIL` | agent | no | The Access login email of the first administrator |
| `CAAL_IDENTITY_API_URL` | frontend | no | Backend base URL, `http://agent:8889` in Docker (compose sets it) |
| `CAAL_PUBLIC_ORIGIN` | frontend | no | Public origin of the portal, e.g. `https://jarvis.example.com`; used for origin checks |
| `CAAL_REQUIRE_IDENTITY_FOR_SESSIONS` | frontend | no | `true` to refuse voice sessions without a verified identity (default `false`) |

The agent reads them from `.env` (compose passes the whole file); the frontend
gets them through `docker-compose*.yaml`. Never prefix any of these with
`NEXT_PUBLIC_`.

For this deployment the two non-secret Cloudflare values were discovered from
the live Access login redirect (`kid=` query parameter and the `aud` claim of
the login meta token, both verified against the team JWKS):

```
CF_ACCESS_TEAM_DOMAIN=https://flamingsoftinc.cloudflareaccess.com
CF_ACCESS_AUD=d1d09a2c79e964918d59077b9bb5b3a7b67ff76a04a80822eb8a3ce5f46354ac
```

Confirm the AUD tag in the Cloudflare Zero Trust dashboard before enabling.

### Enabling, step by step

1. Add the variables to `.env` (agent) and make sure the frontend service in
   your compose file passes them (both `docker-compose.yaml` and
   `docker-compose.apple.yaml` already do).
2. Rebuild and restart only the two affected services:

   ```bash
   docker compose -f docker-compose.apple.yaml -f docker-compose.telephony.yaml up -d --build agent frontend
   ```

3. Check the agent log for `Multi-user identity is configured` (an error line
   starting with `SECURITY CONFIGURATION ERROR` means something is missing or
   invalid, and names it), and check `GET http://localhost:8889/identity/status`
   returns `{"configured": true}`.
4. Sign in to the portal as the bootstrap administrator. The account is created
   on first sign-in. Open **Admin** (top right) to create users, assign roles,
   suspend accounts and approve callback numbers.

### Key rotation

Add a new, higher-numbered key to `CAAL_PROFILE_ENCRYPTION_KEYS` (keep the old
one), restart, then re-encrypt stored numbers:

```bash
docker compose exec agent python -c "from caal.user_api import get_runtime; print(get_runtime().store.rotate_encryption())"
```

Once it prints `0` on a second run, the old key can be removed.

## What users and administrators can do

| | Member | Administrator |
| --- | --- | --- |
| Talk to JARVIS, save and recall **own** memories | yes | yes |
| Phone handoff / callback to **own** approved number | yes (if a number is approved) | yes |
| View own profile, change own display name | yes | yes |
| See or change own role, status, callback number | no | via the admin panel |
| List, create, edit, suspend users; set roles | no | yes |
| Set or clear another user's callback number | no | yes |
| Read the audit trail | no | yes |

The last active administrator cannot be demoted or suspended. The approved
callback number is stored AES-256-GCM encrypted, bound to its row, indexed only
through a keyed blind index, and is never returned to any browser, spoken, or
logged; the UI only ever shows *whether* a number is on file.

## Security controls, by layer

- **Access JWT**: RS256 only, issuer and audience pinned, `exp`/`nbf`/`iat`
  with 30 s tolerance, verified `email` required, JWKS cached with bounded
  refresh on unknown key id, fetch failures never trusted.
- **Internal principals**: HS256 with a ≥ 32-character shared secret,
  fixed issuer, single audience per purpose (`caal-identity`, `caal-backend`,
  `caal-agent`), every claim required, lifetime capped at 300 s regardless of
  issuer, `jti` single-use for HTTP calls, room-bound for the agent.
- **Backend API**: deny by default, roles and status read from the database on
  every call, `Cache-Control: no-store` on every identity response including
  errors, rate limits per client and per actor, strict Pydantic schemas with
  unknown fields rejected, parameterized SQL only, versioned schema migrations.
- **BFF**: `Origin`/`Referer` check plus a double-submit CSRF token (HttpOnly,
  `SameSite=Strict`, `Secure` on TLS) on every mutation, per-client and
  per-actor rate limits, bounded JSON bodies, masked emails and never phone
  numbers in browser responses, `no-store`/`nosniff`/`DENY` headers on
  `/admin`, `/account` and `/api/*`.
- **Isolation**: conversations, background tasks, callbacks and devices carry
  the owner's opaque id and every read and write is scoped to it; the shared
  Telegram fallback channel is never used for a user-scoped outcome.
- **Audit**: administrative mutations and sensitive profile changes are
  recorded with opaque actor and target ids, the action, and field names only.
  The trail is bounded (10 000 rows, 500 per page).
- **Logs** never contain emails, phone numbers, memory contents or tokens.

## Legacy and LAN behaviour

- With identity unconfigured, everything behaves as the single-user JARVIS.
- With identity configured, a request that carries **no** Access assertion at
  all (a LAN client that bypasses Cloudflare) gets an *anonymous* session:
  general assistant features work, but memory, handoff and callbacks are
  refused. Set `CAAL_REQUIRE_IDENTITY_FOR_SESSIONS=true` to refuse such
  sessions entirely.
- A caller whose Access identity verifies but who has no account is refused
  (`no_account`); an administrator must create the account first.
- The mobile app talks to the backend on port 8889 directly and is a legacy
  (anonymous) client until it learns to present an Access identity.

## Limitations

- Hermes runs its own tool loop; when it is the LLM provider, JARVIS's memory
  tools are not the memory path and this per-user scoping does not apply to
  Hermes-side memory.
- Per-user out-of-session notification channels do not exist yet; a user's
  finished background work is announced in their next session or through
  their own phone callback, never through the shared Telegram channel.
