# Telephony (self-hosted LiveKit SIP)

Outbound and inbound PSTN calls run through LiveKit's SIP service, which needs
three containers together: `caal-livekit`, `caal-redis` and `caal-sip`.

## The failure this layout exists to prevent

LiveKit only accepts SIP when it is deployed with Redis. Without it, the SIP
service is running and healthy, Redis is running and healthy, the agent
dispatches an outbound room -- and the dial fails inside LiveKit with:

```
sip not connected (redis required)
```

That is exactly what happened in production: `caal-livekit` was recreated with
`docker compose -f docker-compose.yaml up -d livekit`, from the base file only.
The telephony overlay was the only thing that gave LiveKit a Redis-backed
configuration, so the container came back without one. Nothing else changed,
nothing looked unhealthy, and every call failed before it rang.

## How it is arranged now

* **One renderer.** `livekit-render-config.sh` is mounted into the LiveKit
  container by every compose file and renders `/etc/livekit.yaml` from the same
  two templates (`livekit.yaml` for LAN, `livekit-tailscale.yaml.template` for
  the public HTTPS/TURN deployment).
* **Redis is additive and environment-driven.** The renderer appends LiveKit's
  `redis:` section when `LIVEKIT_REDIS_ADDRESS` is set, and only then. Telephony
  no longer replaces the LiveKit configuration, so SIP and the public HTTPS/TURN
  deployment coexist.
* **The deployment declares itself in `.env`,** not in container state:

  ```
  CAAL_TELEPHONY=1
  LIVEKIT_REDIS_ADDRESS=redis:6379
  ```

  `CAAL_TELEPHONY=1` makes `start-apple.sh` and `telephony-livekit.sh` include
  `docker-compose.telephony.yaml` in *every* compose command.
  `LIVEKIT_REDIS_ADDRESS` in `.env` means that even a LiveKit-only recreate from
  the base compose file still comes up Redis-backed.
* **A base install is unaffected.** With neither variable set, no Redis section
  is rendered, no SIP or Redis service is defined, and nothing enables SIP by
  accident.
* **Media ports stay aligned.** LiveKit advertises its own UDP range in ICE
  candidates and Docker does not translate them, so the published range, the
  container range and the range in both templates are all `50100-50200`.

## Operating it

```bash
./telephony-livekit.sh check       # verify the running stack; changes nothing
./telephony-livekit.sh recreate    # recreate LiveKit under the exact compose set
./telephony-livekit.sh config      # validate the merged compose config (secrets withheld)
```

`recreate` is the only supported way to restart LiveKit alone on a telephony
deployment. It validates the compose set first, recreates `livekit` with
`--no-deps`, restarts `sip` so it re-registers against the new LiveKit process,
and then verifies:

* LiveKit answers HTTP 200 on `:7880`;
* `/etc/livekit.yaml` inside the container has a `redis` section (presence only
  is reported -- never the file, never a credential);
* Redis answers `PING`;
* `caal-sip` is running with no `redis required` errors in its recent log.

Bringing the whole stack up is unchanged: `./start-apple.sh` now includes the
telephony overlay by itself when `CAAL_TELEPHONY=1`.

Never run `docker compose -f docker-compose.yaml up -d livekit` on this
deployment; it is the exact command that caused the outage above.

## Router prerequisites

Public PSTN calling still needs TCP + UDP 5060 and UDP 10000-10100 forwarded to
this host.
