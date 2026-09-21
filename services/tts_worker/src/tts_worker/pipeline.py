"""The TTS stage, independent of Celery.

Kept free of task decorators and broker concerns so it can be driven directly in tests
against a stub engine. `tasks.py` is the thin Celery wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from story2audio_shared.enums import AudioFormat, JobStatus, SegmentKind, VoiceMode
from story2audio_shared.errors import AppError, ErrorCode
from story2audio_shared.events import DoneEvent, JobEvent, ProgressEvent, StatusEvent
from story2audio_shared.logging import get_logger
from story2audio_shared.models import Job, Voice
from story2audio_shared.schemas import AudioAsset
from story2audio_shared.storage import ObjectStorage, audio_key
from story2audio_shared.worker import (
    SyncEventPublisher,
    advance_status,
    current_status,
    is_cancelled,
    load_job,
    session_scope,
)
from tts_worker import audio as audio_ops
from tts_worker.audio import PcmAudio
from tts_worker.engine_client import EngineClient
from tts_worker.segmentation import Segment, split_story
from tts_worker.settings import TtsWorkerSettings

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class StageOutcome:
    """What the stage did."""

    completed: bool
    segment_count: int = 0
    duration_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class _Request:
    """Everything the stage needs, read once so no session stays open during synthesis."""

    story: str
    language: str
    speed: float
    mode: VoiceMode
    narrator_voice_id: UUID
    narrator_key: str
    dialogue_voice_id: UUID | None
    dialogue_key: str | None


def run_tts_stage(
    session_factory: sessionmaker[Session],
    publisher: SyncEventPublisher,
    engine: EngineClient,
    storage: ObjectStorage,
    settings: TtsWorkerSettings,
    job_id: UUID,
) -> StageOutcome:
    """Synthesize, assemble, upload and finish one job.

    Idempotent against its own row, because Celery delivers at least once:

    - already ``done`` or terminal — nothing to do.
    - ``synthesizing`` — a redelivery of an interrupted attempt. Synthesis restarts from
      the beginning: rendered PCM is not persisted, so there is nothing to resume from.
      That is a deliberate limit, and an acceptable one because the *expensive* half is
      already protected — the story is durable at ``written``, so the model is never
      called again.
    - ``written`` — the normal path.
    """
    with session_scope(session_factory) as session:
        job = load_job(session, job_id)
        if job is None:
            log.warning("tts_stage_job_missing", job_id=str(job_id))
            return StageOutcome(completed=False)

        if job.status in {JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED}:
            log.info("tts_stage_skipped_terminal", job_id=str(job_id), status=job.status.value)
            return StageOutcome(completed=False)

        if job.status not in {JobStatus.WRITTEN, JobStatus.SYNTHESIZING}:
            log.info("tts_stage_unexpected_status", job_id=str(job_id), status=job.status.value)
            return StageOutcome(completed=False)

        request = _snapshot_request(session, job)

    segments = split_story(request.story, max_segment_chars=settings.max_segment_chars)
    if not segments:
        raise AppError(ErrorCode.AUDIO_ASSEMBLY_FAILED, detail="story produced no segments")

    if not _begin(session_factory, publisher, job_id, len(segments)):
        return StageOutcome(completed=False)

    rendered = _synthesize_segments(
        session_factory, publisher, engine, storage, settings, job_id, request, segments
    )
    mixed = _assemble(rendered, settings)
    assets = _upload(storage, job_id, mixed, settings)

    with session_scope(session_factory) as session:
        _finish(session, publisher, job_id, mixed=mixed, assets=assets, segments=len(segments))

    return StageOutcome(
        completed=True, segment_count=len(segments), duration_seconds=mixed.duration_seconds
    )


# --- Setup ----------------------------------------------------------------------------


def _snapshot_request(session: Session, job: Job) -> _Request:
    if not job.story_text:
        raise AppError(
            ErrorCode.AUDIO_ASSEMBLY_FAILED, detail="job reached the TTS stage with no story"
        )

    narrator = session.scalar(select(Voice).where(Voice.id == job.voice_id))
    if narrator is None:
        raise AppError(ErrorCode.VOICE_NOT_FOUND, detail=f"voice {job.voice_id} is gone")

    dialogue: Voice | None = None
    if job.dialogue_voice_id is not None:
        dialogue = session.scalar(select(Voice).where(Voice.id == job.dialogue_voice_id))
        if dialogue is None:
            raise AppError(
                ErrorCode.VOICE_NOT_FOUND, detail=f"voice {job.dialogue_voice_id} is gone"
            )

    return _Request(
        story=job.story_text,
        language=job.language.value,
        speed=job.speed,
        mode=job.mode,
        narrator_voice_id=narrator.id,
        narrator_key=narrator.storage_key,
        dialogue_voice_id=dialogue.id if dialogue else None,
        dialogue_key=dialogue.storage_key if dialogue else None,
    )


def _begin(
    session_factory: sessionmaker[Session],
    publisher: SyncEventPublisher,
    job_id: UUID,
    segment_count: int,
) -> bool:
    """Move the job into ``synthesizing``, tolerating a resumed attempt."""
    with session_scope(session_factory) as session:
        status = current_status(session, job_id)
        if status is JobStatus.SYNTHESIZING:
            log.info("tts_stage_restarting", job_id=str(job_id))
            return True

        if status is not JobStatus.WRITTEN:
            return False

        result = advance_status(
            session,
            job_id,
            expected=JobStatus.WRITTEN,
            target=JobStatus.SYNTHESIZING,
            synthesizing_at=datetime.now(UTC),
            segment_count=segment_count,
            segments_done=0,
        )
        if not result.applied:
            log.info("tts_stage_lost_transition", job_id=str(job_id), now=str(result.previous))
            return False

        def build(seq: int, at: datetime) -> JobEvent:
            return StatusEvent(job_id=job_id, seq=seq, at=at, status=JobStatus.SYNTHESIZING)

        publisher.publish(session, job_id, build)
    return True


# --- Synthesis -------------------------------------------------------------------------------


def _synthesize_segments(
    session_factory: sessionmaker[Session],
    publisher: SyncEventPublisher,
    engine: EngineClient,
    storage: ObjectStorage,
    settings: TtsWorkerSettings,
    job_id: UUID,
    request: _Request,
    segments: list[Segment],
) -> list[PcmAudio]:
    """Render each segment, publishing progress as it goes."""
    total = len(segments)
    rendered: list[PcmAudio] = []

    # Reference audio is fetched at most once per voice per job, and only if the engine
    # reports a cache miss.
    loaders = _reference_loaders(storage, request)

    for index, segment in enumerate(segments, start=1):
        if index % settings.cancel_check_every_segments == 0:
            with session_scope(session_factory) as session:
                if is_cancelled(session, job_id):
                    # Stops the GPU mid-job. v1 had no cancellation at all: closing the
                    # tab left the work running to completion.
                    raise AppError(
                        ErrorCode.CANCELLED, detail=f"cancelled after {index - 1}/{total} segments"
                    )

        voice_id, loader = _voice_for(segment, request, loaders)
        result = engine.synthesize(
            segment.text,
            voice_id=str(voice_id),
            reference_loader=loader,
            language=request.language,
            speed=request.speed,
            request_id=f"{job_id}:{index}",
        )

        trimmed = audio_ops.trim_silence(
            result.audio, threshold_dbfs=settings.silence_threshold_dbfs
        )
        rendered.append(audio_ops.apply_fades(trimmed, fade_ms=settings.fade_ms))

        _publish_progress(session_factory, publisher, job_id, done=index, total=total)

    return rendered


def _reference_loaders(storage: ObjectStorage, request: _Request) -> dict[UUID, _CachedLoader]:
    loaders = {request.narrator_voice_id: _CachedLoader(storage, request.narrator_key)}
    if request.dialogue_voice_id is not None and request.dialogue_key is not None:
        loaders[request.dialogue_voice_id] = _CachedLoader(storage, request.dialogue_key)
    return loaders


class _CachedLoader:
    """Fetches a reference sample from storage at most once per job."""

    def __init__(self, storage: ObjectStorage, key: str) -> None:
        self._storage = storage
        self._key = key
        self._data: bytes | None = None

    def __call__(self) -> bytes:
        if self._data is None:
            self._data = self._storage.get_bytes(self._key)
        return self._data


def _voice_for(
    segment: Segment, request: _Request, loaders: dict[UUID, _CachedLoader]
) -> tuple[UUID, _CachedLoader]:
    """Pick the voice for a segment.

    Dialogue uses the second voice only when the job asked for dialogue mode. v1
    hardcoded ``voices/female.wav`` for every spoken line regardless of what the user
    chose.
    """
    use_dialogue = (
        segment.kind is SegmentKind.DIALOGUE
        and request.mode is VoiceMode.NARRATION_WITH_DIALOGUE
        and request.dialogue_voice_id is not None
    )
    voice_id = request.dialogue_voice_id if use_dialogue else request.narrator_voice_id
    assert voice_id is not None  # narrowed by `use_dialogue`
    return voice_id, loaders[voice_id]


def _publish_progress(
    session_factory: sessionmaker[Session],
    publisher: SyncEventPublisher,
    job_id: UUID,
    *,
    done: int,
    total: int,
) -> None:
    with session_scope(session_factory) as session:
        session.query(Job).filter(Job.id == job_id).update({"segments_done": done})

        def build(seq: int, at: datetime) -> JobEvent:
            return ProgressEvent(job_id=job_id, seq=seq, at=at, done=done, total=total)

        publisher.publish(session, job_id, build)


# --- Assembly and delivery -----------------------------------------------------------------------


def _assemble(rendered: list[PcmAudio], settings: TtsWorkerSettings) -> PcmAudio:
    """Join the segments into one track."""
    usable = [segment for segment in rendered if not segment.is_empty]
    if not usable:
        raise AppError(ErrorCode.AUDIO_ASSEMBLY_FAILED, detail="every segment rendered silent")

    sample_rate = usable[0].sample_rate
    if any(segment.sample_rate != sample_rate for segment in usable):
        # Concatenating mismatched rates would play back at the wrong speed rather than
        # failing, which is the kind of bug that reaches a listener before a log.
        raise AppError(
            ErrorCode.AUDIO_ASSEMBLY_FAILED, detail="segments came back at differing sample rates"
        )

    joined = audio_ops.join(
        usable,
        sample_rate=sample_rate,
        pause_ms=settings.segment_pause_ms,
        lead_ms=settings.lead_silence_ms,
    )
    return audio_ops.normalise_peak(joined)


def _upload(
    storage: ObjectStorage, job_id: UUID, mixed: PcmAudio, settings: TtsWorkerSettings
) -> dict[AudioFormat, tuple[str, int]]:
    """Encode and store both formats, returning key and size per format."""
    mp3 = audio_ops.encode_mp3(mixed, bitrate_kbps=settings.mp3_bitrate_kbps)
    wav = audio_ops.encode_wav(mixed)

    assets: dict[AudioFormat, tuple[str, int]] = {}
    for audio_format, data in ((AudioFormat.MP3, mp3), (AudioFormat.WAV, wav)):
        key = audio_key(job_id, audio_format)
        storage.put_audio(key, data, audio_format)
        assets[audio_format] = (key, len(data))
    return assets


def _finish(
    session: Session,
    publisher: SyncEventPublisher,
    job_id: UUID,
    *,
    mixed: PcmAudio,
    assets: dict[AudioFormat, tuple[str, int]],
    segments: int,
) -> None:
    """Mark the job done and announce it."""
    mp3_key, mp3_bytes = assets[AudioFormat.MP3]
    wav_key, wav_bytes = assets[AudioFormat.WAV]

    result = advance_status(
        session,
        job_id,
        expected=JobStatus.SYNTHESIZING,
        target=JobStatus.DONE,
        audio_key_mp3=mp3_key,
        audio_key_wav=wav_key,
        audio_bytes_mp3=mp3_bytes,
        audio_bytes_wav=wav_bytes,
        audio_duration_seconds=mixed.duration_seconds,
        segments_done=segments,
        finished_at=datetime.now(UTC),
    )
    if not result.applied:
        # Cancelled while the last segment was uploading. The user's decision stands.
        raise AppError(
            ErrorCode.CANCELLED,
            detail=f"job left `synthesizing` during assembly (now {result.previous})",
        )

    # The event carries no presigned URL: signing here would bake this worker's clock and
    # TTL into a durable event. The gateway signs on read, where the TTL is meaningful.
    def build(seq: int, at: datetime) -> JobEvent:
        return DoneEvent(
            job_id=job_id,
            seq=seq,
            at=at,
            audio=list[AudioAsset](),
            duration_seconds=mixed.duration_seconds,
        )

    publisher.publish(session, job_id, build)
    log.info(
        "tts_done",
        job_id=str(job_id),
        segments=segments,
        duration_seconds=round(mixed.duration_seconds, 2),
        mp3_bytes=assets[AudioFormat.MP3][1],
    )
