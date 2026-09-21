-- Per-service database roles (ADR-0001).
--
-- The gateway and both workers share one database and one set of ORM models. Shared
-- schema must not mean shared privilege: a bug in the TTS worker should not be able to
-- rewrite a prompt, read another user's email, or drop a table.
--
-- Applied at provisioning time, not from a migration: a migration would have to carry
-- passwords. Run once per environment with the passwords supplied from the secret store:
--
--     psql "$ADMIN_DATABASE_URL" \
--       -v gateway_password="'...'" \
--       -v story_worker_password="'...'" \
--       -v tts_worker_password="'...'" \
--       -f infra/sql/roles.sql
--
-- Idempotent: safe to re-run after a schema change to re-apply grants.

\set ON_ERROR_STOP on

-- --- Roles -----------------------------------------------------------------------------

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'story2audio_gateway') THEN
        CREATE ROLE story2audio_gateway LOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'story2audio_story_worker') THEN
        CREATE ROLE story2audio_story_worker LOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'story2audio_tts_worker') THEN
        CREATE ROLE story2audio_tts_worker LOGIN;
    END IF;
END
$$;

ALTER ROLE story2audio_gateway      WITH PASSWORD :gateway_password;
ALTER ROLE story2audio_story_worker WITH PASSWORD :story_worker_password;
ALTER ROLE story2audio_tts_worker   WITH PASSWORD :tts_worker_password;

GRANT USAGE ON SCHEMA public TO
    story2audio_gateway, story2audio_story_worker, story2audio_tts_worker;

-- --- Gateway ---------------------------------------------------------------------------
-- Owns the public surface: creates users, voices and jobs, reads everything, cancels
-- jobs. It does not run migrations at runtime; DDL is applied by the admin role during
-- deployment.

GRANT SELECT, INSERT, UPDATE, DELETE ON users, voices, jobs TO story2audio_gateway;

-- --- story-worker ----------------------------------------------------------------------
-- Reads the job it was handed and writes only the LLM stage's output. It cannot touch
-- users or voices at all, and cannot write audio keys.

GRANT SELECT ON jobs, voices TO story2audio_story_worker;
GRANT UPDATE (
    status,
    story_text,
    llm_model,
    llm_output_tokens,
    error_code,
    error_message,
    retry_count,
    writing_at,
    written_at,
    finished_at,
    last_event_seq,
    updated_at
) ON jobs TO story2audio_story_worker;

-- --- tts-worker ------------------------------------------------------------------------
-- Reads the story and the voices it must synthesize with, and writes only the audio
-- stage's output. Notably it has no UPDATE on `prompt` or `story_text`: whatever goes
-- wrong in synthesis, the story the user already read cannot change underneath them.

GRANT SELECT ON jobs, voices TO story2audio_tts_worker;
GRANT UPDATE (
    status,
    audio_key_mp3,
    audio_key_wav,
    audio_duration_seconds,
    segment_count,
    segments_done,
    error_code,
    error_message,
    retry_count,
    synthesizing_at,
    finished_at,
    last_event_seq,
    updated_at
) ON jobs TO story2audio_tts_worker;

-- --- Defaults ----------------------------------------------------------------------------
-- A future table created by the admin role grants nothing to the services by default.
-- That is deliberate: a new table should require a deliberate grant here, so privileges
-- are reviewed rather than inherited.

REVOKE ALL ON ALL TABLES IN SCHEMA public FROM PUBLIC;
