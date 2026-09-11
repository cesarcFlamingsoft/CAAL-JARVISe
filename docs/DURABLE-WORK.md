# Durable background work and callbacks

## The failure this replaces

On 2026-09-11 an outbound JARVIS call was answered by a human, work was queued
during the call, the caller explicitly confirmed a callback, and exactly one
callback was armed. Two seconds later the outbound LiveKit room was deleted and
its job process exited. The work runner lived inside that job. The work was
left mid-flight, a later process shutdown recorded it as interrupted, and
interrupted work is never called back -- so the callback was dropped.

Nothing logged a dispatch failure, because no dispatch was ever attempted. This
was not a SIP fault: the call path was healthy the whole time.

The root cause was ownership. `BackgroundTaskBridge.close()` promised the work
would carry on "in this process", but the process it meant was a LiveKit job
whose entire reason to exist had just ended.

## The shape of the fix

Work and callbacks are owned by a supervised service that has no relationship
to any room, job, or session.

```
   voice session (any transport)          durable work service (caal-worker)
   ------------------------------         -----------------------------------
   schedules work  ------------------->   SQLite queue  <---- leases work
   arms a callback ------------------->   (the only shared state)
   may run work under a lease             resumes expired leases
   hands leases back when it ends         settles work
   speaks outcomes while it is open       dispatches callbacks
                                              |
                                              v
                                  isolated outbound LiveKit job
                                  (normal AMD / no-voicemail path)
```

### Leases: work that outlives its runner

A process that runs a task takes a **lease** on it: `lease_owner` plus
`lease_expires_at` on the task row, renewed on a heartbeat. Consequences:

* a session or job that ends **releases** its leases, so each task returns to
  `queued` with its callback authorization intact. Nothing is recorded as
  interrupted, and nothing is announced, because nothing finished;
* a process that dies without releasing anything simply stops renewing. Any
  supervisor reclaims the lease once it expires and resumes the work;
* two processes cannot run the same task, because a lease is taken with one
  atomic conditional update. The agent and the worker can both be running.

`attempts` bounds resumption. Work that has been resumed
`MAX_TASK_ATTEMPTS` times is failed rather than requeued -- and failure is a
settled outcome, so a caller waiting on a callback still hears from JARVIS
instead of waiting forever.

### Callbacks: one authorization, one call, one channel

An armed callback is an authorization, not a queued phone call. Dispatch is:

1. **Claim atomically.** `claim_callback_dispatch` wins the authorization *and*
   the right to announce the outcome in a single transaction. If the outcome
   was already announced elsewhere, the authorization is consumed as
   `superseded` and nothing is dialed: exactly one owner is told exactly once,
   on exactly one channel.
2. **Resolve the destination server-side.** A user-bound callback stores no
   number at all. The number is read from that user's profile at dispatch time,
   the policy is built from that same read, and a stored destination that
   disagrees refuses the call rather than widening it. No model output, no
   caller utterance, and no stored task text can influence the destination, and
   there is no webhook URL anywhere in this path.
3. **Hand off in two steps.** `reserve` creates the isolated room; `dispatch`
   creates the agent dispatch. Splitting them is what makes failure legible.
4. **Fail honestly.**

| Failure | What is true | What happens |
| --- | --- | --- |
| `reserve` fails | no job exists, nothing can dial | authorization released, announcement un-claimed, retried with exponential backoff up to `MAX_CALLBACK_ATTEMPTS` |
| `dispatch` fails | the request was issued and **may** have been accepted | never retried (a retry could ring the same person twice). The authorization is consumed as `uncertain`, the announcement is un-claimed, and the ordinary channels report the outcome |
| no approved number, or the callback's owner is not the work's owner | the call must not happen | consumed as `abandoned`, announcement un-claimed, fallback reports the outcome |
| attempts exhausted | LiveKit is not reachable | same as above |

In no case is delivery recorded for a call that was not placed. The worst case
is that an owner hears an outcome twice through two channels after an
infrastructure error; the case that is designed out is dialing twice.

The outbound job itself is unchanged: it carries `callback_task_id` and the
verified `user_id` in dispatch metadata, re-resolves the number before creating
a SIP participant, and keeps the existing AMD and no-voicemail behaviour.

### In-session delivery races safely

While a session is open it still speaks outcomes. It now claims notifications
with `exclude_callback_armed=True`, so an outcome covered by an unclaimed
callback is left alone -- the user asked to be *called back* about that one.
The exclusion is part of the same conditional `UPDATE` as the claim, so a
session and the dispatcher racing over one task cannot both win. The Telegram
fallback is unchanged and still refuses to carry a task whose `user_id` is not
the bridge's own, so one user's result can never reach another's channel.

### Notification policy: silence by default, one line at the end

A callback that is still being worked on says **nothing**. No chat message is
sent for an attempt, a backed-off retry, a worker restart, a reclaimed lease,
or a hand-off whose fate is unknown (it may have been received, so a message
could contradict a real call).

Exactly one short line is owed when a callback is *finished* failing:

* its bounded retry budget is spent (`DispatchOutcome.ABANDONED`), or
* it was explicitly cancelled outside the session that armed it
  (`disarm_callback(task_id, notify_owner=True)`).

That queues one row in `background_callback_notices` -- one per task, ever --
which a session in the owner scope claims atomically and sends as the constant
`CALLBACK_ABANDONED_NOTICE`. The text names no task, no id, no number and no
outcome, does not claim a call took place, and never asks for another number:
callback destinations are read from the profile an administrator controls and
cannot be set from chat. Verification (dry-run) mode queues nothing at all.

A callback leg that was placed but reached no human now messages nobody. It
calls `release_callback_announcement(task_id)`, which hands the outcome back to
the ordinary deduplicated channels (the session, the dashboard, or ordinary
out-of-session delivery) instead of sending a chat line per unanswered attempt.

Non-callback outbound hand-offs are silent unless an operator sets
`CAAL_OUTBOUND_NOTIFY_FAILED_HANDOFF`, and even then send one fixed line with
no destination and no prompt. The AMD and no-voicemail behaviour on the call
itself is unchanged either way.

## Why a worker service, not a webhook

The worker is a long-lived process with a restart policy, a health probe, and
its own supervision loop. It is correct with no external input at all: it polls.

`POST /internal/wake` exists as an optimisation -- an authenticated nudge to
tick now rather than at the next poll. It is:

* authenticated with the shared `CAAL_INTERNAL_AUTH_SECRET` (constant-time
  comparison, fails closed with 503 when the secret is missing or too short);
* reachable only on the internal compose network -- the worker publishes **no**
  host port;
* never required for correctness. Dropping every wake-up call costs latency,
  not delivery.

`GET /healthz` is unauthenticated for the container probe and returns nothing
but liveness. `GET /status` is authenticated and returns counts and timings
only -- never a task id, a user id, a number, or any task text.

## Operating it

The service starts and restarts with the stack:

```
./start-apple.sh                 # includes caal-worker
./durable-work.sh verify         # config, service, runner, dispatch construction
./durable-work.sh status         # counts only
./durable-work.sh logs
```

`verify` proves the whole dispatch path -- policy, authorization, metadata
shape, and the refusals -- against a synthetic task and resolver. It touches no
stored row and places no call.

To exercise *real* pending callbacks without dialing, set
`CAAL_CALLBACK_DISPATCH_DRY_RUN=1` in `.env` and restart the worker: each due
callback is claimed, its outbound request is built and authorized, and then the
authorization is put back exactly as it was found.

### Runtime source without rebuilding

`src/caal` is bind-mounted read-only into both the agent and the worker, at
both import paths the image has (`/app/.venv/.../site-packages/caal`, which
wins on `sys.path`, and `/app/src/caal`). Both services run the same pinned
image, `caal-agent:latest`. So a code change reaches both processes with a
restart and no rebuild, and neither depends on a one-off copy into a container
that a restart would discard.

### Work that predates leases

A task left `running` by the old session-bound runner carries no lease. Such a
row is **never** adopted automatically, by the agent or by the worker: it may
carry an armed callback, and resuming it can end in JARVIS phoning its owner.
That is an operator's decision, not a side effect of a restart.

```
./durable-work.sh orphans        # how many, counts only
./durable-work.sh adopt          # explicit, with a typed confirmation
```

Adoption returns those tasks to the queue. From there they are ordinary durable
work: they run, they settle, and only then -- on a legitimate terminal state --
does any callback armed on them get dispatched.
