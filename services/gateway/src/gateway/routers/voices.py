"""Voice catalogue routes."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, File, Form, Query, Response, UploadFile, status

from gateway.deps import (
    LimitsDep,
    RateLimiterDep,
    SessionDep,
    StateDep,
    StorageDep,
    UserIdDep,
)
from gateway.services import voices as voice_service
from story2audio_shared.errors import AppError, ErrorCode
from story2audio_shared.logging import get_logger
from story2audio_shared.schemas import Page, VoiceName, VoiceResponse

log = get_logger(__name__)

router = APIRouter(prefix="/v1/voices", tags=["voices"])

#: Read the upload in bounded chunks. `Content-Length` is checked by middleware, but a
#: chunked request carries none, so the ceiling is enforced again while reading rather
#: than trusted from a header.
_CHUNK_BYTES = 64 * 1024


@router.get("", response_model=Page[VoiceResponse], summary="List available voices")
async def list_voices(
    user_id: UserIdDep,
    session: SessionDep,
    storage: StorageDep,
    state: StateDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    cursor: Annotated[UUID | None, Query()] = None,
) -> Page[VoiceResponse]:
    """Built-in voices plus the caller's own uploads.

    Each row carries a presigned preview URL, so the picker can play a sample before the
    user commits to a generation.
    """
    rows, has_more = await voice_service.list_voices(
        session, owner_id=user_id, limit=limit, cursor=cursor
    )
    items = [
        voice_service.to_response(voice, storage, ttl_seconds=state.presigned_ttl_seconds)
        for voice in rows
    ]
    return Page[VoiceResponse](
        items=items,
        next_cursor=items[-1].id if items and has_more else None,
        has_more=has_more,
    )


@router.post(
    "",
    response_model=VoiceResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a reference voice",
)
async def create_voice(
    user_id: UserIdDep,
    session: SessionDep,
    storage: StorageDep,
    state: StateDep,
    limits: LimitsDep,
    rate_limiter: RateLimiterDep,
    name: Annotated[VoiceName, Form()],
    file: Annotated[UploadFile, File()],
) -> VoiceResponse:
    """Register a new voice from an uploaded audio sample.

    Multipart, not base64-in-JSON: v1's REST proxy took the sample as a base64 string in
    the request body, which inflates it by a third and pushes the whole thing through the
    JSON parser.
    """
    data = await _read_capped(file, limits.max_voice_upload_bytes)
    voice = await voice_service.create_voice(
        session,
        storage,
        rate_limiter,
        owner_id=user_id,
        name=name,
        data=data,
        content_type=file.content_type,
        limits=limits,
    )
    log.info(
        "voice_created",
        voice_id=str(voice.id),
        duration_seconds=round(voice.duration_seconds, 2),
    )
    return voice_service.to_response(voice, storage, ttl_seconds=state.presigned_ttl_seconds)


@router.delete("/{voice_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete your voice")
async def delete_voice(
    voice_id: UUID,
    user_id: UserIdDep,
    session: SessionDep,
    storage: StorageDep,
) -> Response:
    """Delete one of the caller's own voices. Built-ins are not deletable."""
    await voice_service.delete_voice(session, storage, owner_id=user_id, voice_id=voice_id)
    log.info("voice_deleted", voice_id=str(voice_id))
    return Response(status_code=status.HTTP_204_NO_CONTENT)


async def _read_capped(file: UploadFile, max_bytes: int) -> bytes:
    """Read an upload, refusing to buffer more than the limit.

    Stops at the first chunk that crosses the ceiling instead of reading the whole body
    and measuring afterwards — otherwise the size limit is enforced only after the memory
    has already been spent.
    """
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(_CHUNK_BYTES):
        total += len(chunk)
        if total > max_bytes:
            raise AppError(
                ErrorCode.VOICE_TOO_LARGE,
                detail=f"upload exceeded {max_bytes} bytes while streaming",
            )
        chunks.append(chunk)
    return b"".join(chunks)
