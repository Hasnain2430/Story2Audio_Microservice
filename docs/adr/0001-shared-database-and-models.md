# ADR-0001 — Services share one database and one set of ORM models

**Status:** Accepted (Phase 1)

## Context

`gateway`, `story-worker` and `tts-worker` all read and write the `jobs` table. The gateway
creates the row; the story worker sets `writing`/`written` and the story text; the TTS
worker sets `synthesizing`/`done` and the audio keys. Three services, one table.

Strict microservice practice says each service owns its own datastore and other services
reach it only through its API.

## Options

1. **Shared database, models defined once in `packages/shared`.**
2. **Shared database, each service defining its own model classes.**
3. **Gateway owns the database and exposes an internal write-API that the workers call to
   report status and results.**

## Decision

Option 1.

Option 2 is the worst of both: the same coupling, plus guaranteed drift. Two definitions of
one table diverge the moment someone adds a column to the writer and forgets the reader,
and the symptom shows up as a worker silently writing a field the gateway never reads.

Option 3 is the orthodox answer, and it is genuinely better when services are owned by
different teams and deploy independently. Neither is true here — these three deploy
together from one repository, on one release. What Option 3 would actually buy is an extra
network hop on every status transition, a new retry and partial-failure surface, and an
internal API to version. That is real cost for independence we do not need.

So: one database, one definition, and say so out loud rather than pretending otherwise.

## Consequences

- Migrations are owned **solely by the gateway**. Workers never issue DDL. A schema change
  ships in the gateway's Alembic history and workers pick up the new model on the same
  release.
- Each service connects with a **database role scoped to the columns it writes**. Shared
  schema is not shared privilege: a bug in the TTS worker must not be able to rewrite a
  prompt or delete a user.
- Every worker status update is a **conditional UPDATE asserting the previous status**
  (`enums.ALLOWED_TRANSITIONS`). Celery is at-least-once, so a duplicate delivery must not
  move a job backwards.
- If the services ever do need independent deployment, the migration path is Option 3 —
  and `packages/shared` is where the write-API client would live, so the change is
  contained.

## Revisit when

The services stop deploying as one unit, or a second team takes ownership of one of them.
