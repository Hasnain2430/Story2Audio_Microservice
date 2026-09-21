"""The voice catalogue.

Two kinds of voice live in one table: built-ins seeded from the repository's reference
pack, visible to everyone, and uploads, visible only to the user who created them. A
`Voice` row is the indirection that replaced v1's client-supplied filesystem path.
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID, uuid4

from sqlalchemy import Select, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.audio import ValidatedVoice, check_content_type, validate_voice_upload
from gateway.ratelimit import RateLimiter
from story2audio_shared.config import LimitSettings
from story2audio_shared.errors import AppError, ErrorCode
from story2audio_shared.ids import uuid7
from story2audio_shared.models import Job, Voice
from story2audio_shared.schemas import VoiceResponse
from story2audio_shared.storage import ObjectStorage, voice_key

_UPLOADS_WINDOW_SECONDS = 3600


async def list_voices(
    session: AsyncSession, *, owner_id: UUID, limit: int, cursor: UUID | None
) -> tuple[Sequence[Voice], bool]:
    """Built-in voices plus the caller's own, newest first."""
    query: Select[tuple[Voice]] = select(Voice).where(
        or_(Voice.is_builtin.is_(True), Voice.owner_id == owner_id)
    )
    if cursor is not None:
        query = query.where(Voice.id < cursor)

    rows = (await session.scalars(query.order_by(Voice.id.desc()).limit(limit + 1))).all()
    has_more = len(rows) > limit
    return rows[:limit], has_more


async def create_voice(
    session: AsyncSession,
    storage: ObjectStorage,
    rate_limiter: RateLimiter,
    *,
    owner_id: UUID,
    name: str,
    data: bytes,
    content_type: str | None,
    limits: LimitSettings,
) -> Voice:
    """Validate an uploaded sample, store it, and register it in the catalogue.

    The upload is decoded rather than measured: that establishes the real duration, proves
    the bytes are audio, and produces one canonical WAV to store. v1 accepted anything at
    least 15000 bytes long -- about a sixth of a second.
    """
    await _assert_upload_quota_available(rate_limiter, owner_id, limits)

    check_content_type(content_type)
    validated: ValidatedVoice = validate_voice_upload(
        data,
        min_duration_seconds=limits.min_voice_duration_seconds,
        max_duration_seconds=limits.max_voice_duration_seconds,
        max_bytes=limits.max_voice_upload_bytes,
        clip_seconds=limits.reference_clip_seconds,
    )

    await _assert_name_available(session, owner_id, name)

    voice_id = uuid7()
    key = voice_key(voice_id)
    # Stored before the row is committed. An orphaned object costs pennies and is swept up
    # by the storage lifecycle rule; a row pointing at a missing object is a broken voice
    # the user can select and fail on.
    storage.put_bytes(key, validated.wav_bytes, content_type="audio/wav")

    voice = Voice(
        id=voice_id,
        owner_id=owner_id,
        name=name,
        storage_key=key,
        duration_seconds=validated.duration_seconds,
        sample_rate=validated.sample_rate,
        is_builtin=False,
    )
    session.add(voice)
    try:
        await session.flush()
    except IntegrityError as exc:
        await session.rollback()
        storage.delete(key)
        raise AppError(ErrorCode.VOICE_NAME_TAKEN, detail=f"name {name!r} already used") from exc

    return voice


async def delete_voice(
    session: AsyncSession, storage: ObjectStorage, *, owner_id: UUID, voice_id: UUID
) -> None:
    """Delete one of the caller's own voices.

    Refused while a job still references it: the job's audio may not have been rendered
    yet, and the database's RESTRICT on the foreign key would reject the delete anyway.
    Catching it here turns a 500 into a clear 409.
    """
    voice = await session.scalar(select(Voice).where(Voice.id == voice_id))
    if voice is None:
        raise AppError(ErrorCode.VOICE_NOT_FOUND, detail=f"voice {voice_id} does not exist")
    if voice.is_builtin:
        raise AppError(ErrorCode.VOICE_FORBIDDEN, detail="built-in voices cannot be deleted")
    if voice.owner_id != owner_id:
        raise AppError(ErrorCode.VOICE_FORBIDDEN, detail="voice belongs to another user")

    referenced = await session.scalar(
        select(Job.id)
        .where(or_(Job.voice_id == voice_id, Job.dialogue_voice_id == voice_id))
        .limit(1)
    )
    if referenced is not None:
        raise AppError(
            ErrorCode.VOICE_FORBIDDEN,
            detail=f"voice {voice_id} is still referenced by job {referenced}",
        )

    await session.delete(voice)
    await session.flush()
    storage.delete(voice.storage_key)


def to_response(
    voice: Voice, storage: ObjectStorage, *, ttl_seconds: int, with_preview: bool = True
) -> VoiceResponse:
    """Render a voice for the API, with a presigned preview URL.

    The preview is what lets the picker play a sample before committing to a ten-minute
    generation -- worth the presign on every row.
    """
    preview_url: str | None = None
    if with_preview:
        preview_url = storage.presign_get(voice.storage_key, ttl_seconds=ttl_seconds).url

    return VoiceResponse(
        id=voice.id,
        name=voice.name,
        is_builtin=voice.is_builtin,
        duration_seconds=voice.duration_seconds,
        sample_rate=voice.sample_rate,
        preview_url=preview_url,
        created_at=voice.created_at,
    )


async def _assert_name_available(session: AsyncSession, owner_id: UUID, name: str) -> None:
    clash = await session.scalar(
        select(Voice.id).where(Voice.owner_id == owner_id, Voice.name == name)
    )
    if clash is not None:
        raise AppError(ErrorCode.VOICE_NAME_TAKEN, detail=f"name {name!r} already used")


async def _assert_upload_quota_available(
    rate_limiter: RateLimiter, owner_id: UUID, limits: LimitSettings
) -> None:
    decision = await rate_limiter.check_sliding_window(
        bucket="uploads",
        subject=owner_id,
        limit=limits.rate_limit_uploads_per_hour,
        window_seconds=_UPLOADS_WINDOW_SECONDS,
        token=uuid4().hex,
    )
    if not decision.allowed:
        raise AppError(
            ErrorCode.RATE_LIMITED,
            detail=f"{decision.used}/{decision.limit} uploads in the last hour",
        )
