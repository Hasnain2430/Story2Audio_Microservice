"""The TTS stage, independent of Celery.

Kept free of task decorators and broker concerns so it can be driven directly in tests
against a stub engine. `tasks.py` is the thin Celery wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from story2audio_shared.enums import AudioFormat, JobStatus, SegmentKind, VoiceMode
from story2audio_shared.errors import AppError, ErrorCode
from story2audio_shared.events import (
    DoneEvent,
    JobEvent,
    ProgressEvent,
    SegmentReadyEvent,
    StatusEvent,
)
from story2audio_shared.logging import get_logger
from story2audio_shared.models import Job, Voice
from story2audio_shared.schemas import AudioAsset
from story2audio_shared.storage import ObjectStorage, audio_key, segment_audio_key
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
from tts_worker.segmentation import Segment, max_segment_chars, split_story
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
    second_dialogue_voice_id: UUID | None
    second_dialogue_key: str | None


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

    # Capped by what XTTS accepts for this language, not only by what is configured:
    # over the limit it truncates the audio and merely logs about it.
    segments = split_story(
        request.story,
        max_segment_chars=max_segment_chars(request.language, settings.max_segment_chars),
    )
    if not segments:
        raise AppError(ErrorCode.AUDIO_ASSEMBLY_FAILED, detail="story produced no segments")

    if not _begin(session_factory, publisher, job_id, len(segments)):
        return StageOutcome(completed=False)

    assembler = _synthesize_segments(
        session_factory, publisher, engine, storage, settings, job_id, request, segments
    )
    mixed, timeline = assembler.finish()
    assets = _upload(storage, job_id, mixed, settings)

    with session_scope(session_factory) as session:
        _finish(
            session,
            publisher,
            job_id,
            mixed=mixed,
            assets=assets,
            segments=len(segments),
            timeline=timeline,
        )

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
    second = None
    if job.second_dialogue_voice_id is not None:
        second = session.scalar(select(Voice).where(Voice.id == job.second_dialogue_voice_id))
        if second is None:
            raise AppError(
                ErrorCode.VOICE_NOT_FOUND,
                detail=f"voice {job.second_dialogue_voice_id} is gone",
            )

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
        second_dialogue_voice_id=second.id if second else None,
        second_dialogue_key=second.storage_key if second else None,
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
) -> _Assembler:
    """Render each segment, publishing it as it goes.

    Each finished segment is uploaded and announced immediately. The first one is ready
    seconds into a job that takes minutes, and a listener can start on it while the rest
    are still being made -- which is the difference between "the queue means you are not
    blocked" and "you are already listening".
    """
    total = len(segments)
    assembler = _Assembler(settings, request.story)

    # Reference audio is fetched at most once per voice per job, and only if the engine
    # reports a cache miss.
    loaders = _reference_loaders(storage, request)
    cast = _build_cast(segments, request)
    if cast:
        # Full ids, not a prefix. These are UUIDv7 and therefore time-ordered, so voices
        # created in the same seeding run share their leading characters — truncating
        # them printed two different voices as the same string.
        log.info("tts_cast", job_id=str(job_id), cast={k: str(v) for k, v in cast.items()})

    for index, segment in enumerate(segments, start=1):
        if index % settings.cancel_check_every_segments == 0:
            with session_scope(session_factory) as session:
                if is_cancelled(session, job_id):
                    # Stops the GPU mid-job. v1 had no cancellation at all: closing the
                    # tab left the work running to completion.
                    raise AppError(
                        ErrorCode.CANCELLED, detail=f"cancelled after {index - 1}/{total} segments"
                    )

        voice_id, loader = _voice_for(segment, request, loaders, cast)
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
        placed = assembler.place(segment, audio_ops.apply_fades(trimmed, fade_ms=settings.fade_ms))

        if placed is not None:
            _publish_segment(
                session_factory,
                publisher,
                storage,
                job_id,
                placed,
                assembler.count - 1,
                total,
                assembler.timeline(),
            )

        _publish_progress(session_factory, publisher, job_id, done=index, total=total)

    return assembler


def _publish_segment(
    session_factory: sessionmaker[Session],
    publisher: SyncEventPublisher,
    storage: ObjectStorage,
    job_id: UUID,
    placed: _Placed,
    index: int,
    total: int,
    timeline: list[dict[str, Any]],
) -> None:
    """Upload one rendered segment and announce that it can be played.

    Uploaded as WAV rather than MP3 on purpose. MP3 frames carry encoder delay and
    padding, so decoding a run of separately encoded clips and butting them together
    inserts a few milliseconds of silence at every seam -- inaudible in one file,
    a stutter once per segment across a whole story. The assembled download is still
    MP3; this is the copy that exists to be scheduled against a clock.

    A failed upload must not fail the job. The segment is already rendered and will be in
    the final track; losing it here costs progressive playback for that one segment and
    nothing else.
    """
    try:
        storage.put_bytes(
            segment_audio_key(job_id, index),
            audio_ops.encode_wav(placed.audio),
            content_type="audio/wav",
        )
    except Exception as exc:  # noqa: BLE001 - see docstring; never fail a job for this
        log.warning("segment_publish_failed", job_id=str(job_id), index=index, error=str(exc))
        return

    def build(seq: int, at: datetime) -> JobEvent:
        return SegmentReadyEvent(
            job_id=job_id,
            seq=seq,
            at=at,
            index=index,
            total=total,
            kind=placed.segment.kind,
            text=placed.segment.text,
            speaker=placed.segment.speaker,
            start_char=placed.segment.start_char,
            end_char=placed.segment.end_char,
            start_seconds=placed.start_seconds,
            end_seconds=placed.end_seconds,
        )

    with session_scope(session_factory) as session:
        # Persist the timeline so far, not only announce it. Redis pub/sub keeps no
        # history, so a client that reloads or connects late hears about none of the
        # segments already rendered -- and without a stored copy it cannot play them
        # either, which breaks the promise the compose page makes about closing the tab
        # and coming back. One small update per segment buys recovery for free.
        session.query(Job).filter(Job.id == job_id).update({"segment_timeline": timeline})
        publisher.publish(session, job_id, build)


def _reference_loaders(storage: ObjectStorage, request: _Request) -> dict[UUID, _CachedLoader]:
    loaders = {request.narrator_voice_id: _CachedLoader(storage, request.narrator_key)}
    if request.dialogue_voice_id is not None and request.dialogue_key is not None:
        loaders[request.dialogue_voice_id] = _CachedLoader(storage, request.dialogue_key)
    if request.second_dialogue_voice_id is not None and request.second_dialogue_key is not None:
        loaders[request.second_dialogue_voice_id] = _CachedLoader(
            storage, request.second_dialogue_key
        )
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


def _build_cast(segments: list[Segment], request: _Request) -> dict[str, UUID]:
    """Assign a voice to each character who speaks.

    Characters are cast in order of first appearance, which is stable for a given story
    and needs no state: the first person to speak gets the first dialogue voice, the
    second gets the second. Beyond that the voices repeat, because the job carries two.

    A job with only one dialogue voice casts everyone to it — exactly the old behaviour,
    which is right: without a second voice there is nothing better to do, and it is not
    this function's place to decide the narrator should start playing a part.
    """
    voices = [
        voice_id
        for voice_id in (request.dialogue_voice_id, request.second_dialogue_voice_id)
        if voice_id is not None
    ]
    if not voices:
        return {}

    speakers = dict.fromkeys(
        segment.speaker
        for segment in segments
        if segment.kind is SegmentKind.DIALOGUE and segment.speaker
    )
    return {name: voices[index % len(voices)] for index, name in enumerate(speakers)}


def _voice_for(
    segment: Segment, request: _Request, loaders: dict[UUID, _CachedLoader], cast: dict[str, UUID]
) -> tuple[UUID, _CachedLoader]:
    """Pick the voice for a segment.

    Dialogue uses a character's voice when the job asked for dialogue mode. v1 hardcoded
    ``voices/female.wav`` for every spoken line regardless of what the user chose; the
    first version of v2 improved on that only by making the single voice configurable,
    which still read a two-hander in one voice.

    A line whose speaker could not be attributed falls back to the first dialogue voice.
    That is a deliberate choice over guessing: the fallback is merely unremarkable, and a
    wrong voice is something the listener hears immediately.
    """
    if (
        segment.kind is SegmentKind.DIALOGUE
        and request.mode is VoiceMode.NARRATION_WITH_DIALOGUE
        and request.dialogue_voice_id is not None
    ):
        voice_id = cast.get(segment.speaker or "", request.dialogue_voice_id)
    else:
        voice_id = request.narrator_voice_id
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


def _voice_changes(previous: Segment, segment: Segment) -> bool:
    """Whether these two segments are spoken by different people.

    Compares narration-or-whom, not the kind alone: two consecutive lines from the same
    character are one voice continuing, while narration to dialogue is always a change
    even when both would land on the same voice, because the register does.
    """
    return (previous.kind, previous.speaker) != (segment.kind, segment.speaker)


@dataclass(slots=True)
class _Placed:
    """One rendered segment, with the silence in front of it and where it lands."""

    segment: Segment
    audio: PcmAudio
    gap_ms: int
    start_seconds: float
    end_seconds: float


class _Assembler:
    """Places segments on the timeline as they are rendered.

    Assembly used to happen after every segment was finished, which was fine while the
    listener could not hear anything until then anyway. Progressive playback needs each
    segment's position *at the moment it is rendered*, so it can be published and played
    while the rest are still being made.

    Nothing about the arithmetic changes — the gap before a segment depends only on the
    segment before it and the story text, both of which are known as soon as it is
    rendered. What changes is that there is now one object that owns it, so the streamed
    positions and the final track cannot disagree.

    Empty renders are dropped here rather than filtered later: a segment that came back
    silent must not occupy time or consume a gap.
    """

    def __init__(self, settings: TtsWorkerSettings, story: str) -> None:
        self._settings = settings
        self._story = story
        self._placed: list[_Placed] = []
        self._cursor = 0.0

    def place(self, segment: Segment, audio: PcmAudio) -> _Placed | None:
        """Add a rendered segment, returning where it landed, or ``None`` if silent."""
        if audio.is_empty:
            return None

        gap_ms = self._gap_before(segment)
        self._cursor += gap_ms / 1000
        start = self._cursor
        self._cursor += audio.duration_seconds

        placed = _Placed(
            segment=segment,
            audio=audio,
            gap_ms=gap_ms,
            start_seconds=round(start, 3),
            end_seconds=round(self._cursor, 3),
        )
        self._placed.append(placed)
        return placed

    def _gap_before(self, segment: Segment) -> int:
        """How much silence precedes this segment, in milliseconds.

        Three kinds of boundary, and treating them alike is what made the old output
        sound chopped. A single 300 ms pause everywhere put the same gap between two
        halves of one sentence, between two paragraphs, and between a question and its
        answer.

        - **A change of speaker** gets the longest beat. The listener needs a moment to
          register that someone else is talking.
        - **A paragraph break** the author wrote gets an ordinary pause.
        - **Everything else** is a boundary this pipeline invented, because the text was
          longer than one synthesis request. A reader does not pause there, so the gap is
          short — kept non-zero only because two independently rendered clips need a seam.
        """
        if not self._placed:
            return self._settings.lead_silence_ms

        previous = self._placed[-1].segment
        if _voice_changes(previous, segment):
            return self._settings.speaker_change_pause_ms
        if "\n\n" in self._story[previous.end_char : segment.start_char]:
            return self._settings.segment_pause_ms
        return self._settings.continuation_pause_ms

    @property
    def count(self) -> int:
        return len(self._placed)

    def finish(self) -> tuple[PcmAudio, list[dict[str, Any]]]:
        """The assembled track, and the timeline describing it."""
        if not self._placed:
            raise AppError(ErrorCode.AUDIO_ASSEMBLY_FAILED, detail="every segment rendered silent")

        sample_rate = self._placed[0].audio.sample_rate
        if any(placed.audio.sample_rate != sample_rate for placed in self._placed):
            # Concatenating mismatched rates would play back at the wrong speed rather
            # than failing, which is the kind of bug that reaches a listener before a log.
            raise AppError(
                ErrorCode.AUDIO_ASSEMBLY_FAILED,
                detail="segments came back at differing sample rates",
            )

        joined = audio_ops.join(
            [placed.audio for placed in self._placed],
            sample_rate=sample_rate,
            gaps_ms=[placed.gap_ms for placed in self._placed],
        )
        # Peak normalisation scales amplitude only, so the timeline is unaffected by it.
        return audio_ops.normalise_peak(joined), self.timeline()

    def timeline(self) -> list[dict[str, Any]]:
        """Where each segment starts and ends in the assembled track.

        Measured, not estimated: the worker rendered every segment, knows each one's
        exact duration, and inserted every pause itself. It is the same arithmetic that
        produced the audio rather than a second derivation of it, which is what lets a
        player highlight the story in time without anything having to listen afterwards.
        """
        return [
            {
                "index": index,
                "kind": placed.segment.kind.value,
                "text": placed.segment.text,
                "speaker": placed.segment.speaker,
                "start_char": placed.segment.start_char,
                "end_char": placed.segment.end_char,
                "start_seconds": placed.start_seconds,
                "end_seconds": placed.end_seconds,
            }
            for index, placed in enumerate(self._placed)
        ]


# --- Assembly and delivery -----------------------------------------------------------------------


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
    timeline: list[dict[str, Any]],
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
        segment_timeline=timeline,
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
