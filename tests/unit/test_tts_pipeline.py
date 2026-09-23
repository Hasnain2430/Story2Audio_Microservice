"""The TTS stage: segmentation, audio assembly, and the pipeline itself.

Segmentation and audio are pure functions and are tested directly. The pipeline is
driven against a stub engine so every branch — cancellation between segments, a cache
miss, a dialogue job using two voices — is reachable without a GPU.
"""

from __future__ import annotations

import io
import itertools
import json
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import fakeredis
import numpy as np
import pytest
from sqlalchemy import Engine, create_engine, event, select
from sqlalchemy.orm import Session, sessionmaker

from story2audio_shared.enums import (
    AudioFormat,
    Emotion,
    JobStatus,
    Language,
    SegmentKind,
    StoryLength,
    VoiceMode,
)
from story2audio_shared.errors import AppError, ErrorCode
from story2audio_shared.ids import uuid7
from story2audio_shared.models import Base, Job, User, Voice
from story2audio_shared.storage import PresignedUrl
from story2audio_shared.worker import SyncEventPublisher, create_worker_session_factory
from tts_worker import audio as audio_ops
from tts_worker.audio import PcmAudio, from_pcm_bytes
from tts_worker.engine_client import SynthesisResult
from tts_worker.pipeline import run_tts_stage
from tts_worker.segmentation import Segment, clean_segment_text, split_story
from tts_worker.settings import TtsWorkerSettings

SAMPLE_RATE = 24_000

NARRATION_ONLY = (
    "The keeper climbed the stairs. The lamp was cold to the touch. "
    "Outside, the sea kept its own counsel."
)
WITH_DIALOGUE = 'She stopped at the door. "I am not leaving without you," she said. He nodded.'


# --- Segmentation --------------------------------------------------------------------------------


def test_narration_is_split_into_multiple_segments() -> None:
    """v1 handed the whole story to one tts_to_file call, so progress was impossible."""
    story = " ".join(f"Sentence number {index} runs on for a while." for index in range(30))

    segments = split_story(story, max_segment_chars=200)

    assert len(segments) > 1
    assert all(segment.kind is SegmentKind.NARRATION for segment in segments)


def test_segments_respect_the_size_limit() -> None:
    story = " ".join(f"A short sentence {index}." for index in range(50))

    segments = split_story(story, max_segment_chars=120)

    # A single over-long sentence is kept whole; none of these are.
    assert all(len(segment.text) <= 120 for segment in segments)


def test_an_overlong_sentence_is_broken_at_a_clause() -> None:
    """This used to be kept whole, and that was wrong.

    The reasoning was that a hard cut inside a clause sounds worse than one long segment,
    and that the engine would split it further anyway. The engine does not:
    `enable_text_splitting` is off, and past its per-language character limit XTTS
    truncates the audio and merely logs about it. A break at a comma is a breath; a
    truncation is a sentence that stops existing.
    """
    story = (
        "The keeper climbed the stairs while the wind rose, and the sea below turned the "
        "colour of old iron, and somewhere beyond the headland a bell began to ring, slow "
        "and uneven, and still he climbed."
    )

    segments = split_story(story, max_segment_chars=120)

    assert len(segments) > 1
    assert all(len(segment.text) <= 120 for segment in segments)
    # Broken where the sentence already paused, not mid-phrase.
    assert all(not segment.text.endswith(" ") for segment in segments)

    # The pieces divide the sentence's span between them. Review caught this: giving
    # every piece the whole sentence's span produces overlapping spans, and the
    # read-along then clamps each one after the first to zero width -- so the back half
    # of a long sentence is spoken with nothing highlighted, silently.
    assert all(
        earlier.end_char <= later.start_char for earlier, later in itertools.pairwise(segments)
    )
    assert segments[0].start_char == 0
    assert segments[-1].end_char == len(story.rstrip())


def test_a_clause_with_nowhere_to_break_is_split_between_words() -> None:
    """Last resort. Still better than losing the end of the line entirely."""
    story = "word " * 200

    segments = split_story(story, max_segment_chars=100)

    assert len(segments) > 1
    assert all(len(segment.text) <= 100 for segment in segments)
    # No word is cut in half.
    assert all(part == "word" for segment in segments for part in segment.text.split())


SCENE = """Tom polished the lens in silence. His hands were steady.

"I can't stay here forever," Lily said. She looked at her father.

Tom set the cloth aside, his voice steady. "This place has kept us safe for years."

"I love you, but I want to see cities," Lily said.

Tom sighed, the sound mixing with the distant gulls. "Leaving means the light will be alone."

The wind began to howl, rattling the panes.

"Your mother would have wanted you to live fully," he said.

"She taught me to love the world," she said."""


def spoken(story: str) -> list[tuple[str | None, str]]:
    return [
        (segment.speaker, segment.text)
        for segment in split_story(story)
        if segment.kind is SegmentKind.DIALOGUE
    ]


def test_every_line_in_a_scene_is_attributed_to_the_right_character() -> None:
    """The whole point of a second voice.

    This scene is a real generation, and it defeated three earlier versions of the
    attribution code in three different ways: Tom is never introduced by a speech tag
    (only by action beats), two of the lines are tagged with a bare pronoun, and one
    reply is separated from its speaker by a paragraph of narration. Each of those put a
    line in the wrong character's voice, which is immediately audible.
    """
    assert [speaker for speaker, _ in spoken(SCENE)] == [
        "Lily",
        "Tom",
        "Lily",
        "Tom",
        "Tom",
        "Lily",
    ]


def test_a_speaker_is_found_from_an_action_beat() -> None:
    """Prose attributes as often by action as by "said", and a model told to avoid
    adverbial tags does it constantly. Reading only "said" left that character out of
    the cast entirely, so all their lines fell back to the other one's voice."""
    story = 'Tom set the cloth aside, his voice steady. "This place has kept us safe."'

    assert spoken(story) == [("Tom", "This place has kept us safe.")]


def test_a_sentence_opener_is_not_mistaken_for_a_character() -> None:
    """Every sentence starts with a capital, so "The wind rose." looks like a name."""
    story = 'The wind began to howl, rattling the panes. "We should go inside."'

    assert spoken(story) == [(None, "We should go inside.")]


def test_a_pronoun_tag_resolves_by_elimination() -> None:
    """ "he said" names nobody, but it rules somebody out.

    Ignoring it and falling through to alternation put the line in the other
    character's voice — the one person the tag excludes.
    """
    story = (
        'Tom watched the water, his coat wet through. "The tide is turning."\n\n'
        '"We should turn back," Lily said. She shivered.\n\n'
        '"Not yet," he said.'
    )

    assert spoken(story)[-1] == ("Tom", "Not yet,")


def test_a_character_never_named_beside_a_line_cannot_be_cast() -> None:
    """The boundary of what attribution can do, asserted rather than left to be found.

    Every rule here resolves to a name the story attached to a spoken line somewhere. A
    character introduced only in narration that carries no dialogue never enters the
    cast, so a pronoun tag has nobody to resolve to and the line falls back to the first
    voice. That is the intended failure: unremarkable, rather than confidently wrong.
    """
    story = (
        "Tom watched the water. His coat was wet.\n\n"
        '"We should turn back," Lily said. She shivered.\n\n'
        '"Not yet," he said.'
    )

    assert spoken(story)[-1][0] != "Lily", "a 'he' line must never land on her"


def test_an_unattributed_reply_alternates() -> None:
    """A new paragraph of untagged speech after one character is the other replying."""
    story = '"Sell it," Mara said.\n\n"Never."\n\n"Then keep it."'
    speakers = [speaker for speaker, _ in spoken(story)]

    assert speakers == ["Mara", None, "Mara"] or speakers[0] == "Mara"
    # Whatever the unattributed lines resolve to, consecutive turns must not all collapse
    # onto one speaker: that is the failure this rule exists to prevent.
    assert speakers[1] != "Mara"


def test_dialogue_is_separated_from_narration() -> None:
    segments = split_story(WITH_DIALOGUE)

    kinds = [segment.kind for segment in segments]
    assert SegmentKind.DIALOGUE in kinds
    assert SegmentKind.NARRATION in kinds


def test_curly_quotes_are_recognised_as_dialogue() -> None:
    """v1 matched only straight quotes, so a model emitting “ ” lost every spoken line."""
    story = "He waited. “I am here,” she called. Then silence."

    segments = split_story(story)

    dialogue = [segment for segment in segments if segment.kind is SegmentKind.DIALOGUE]
    assert len(dialogue) == 1
    assert "I am here" in dialogue[0].text


def test_segment_order_matches_the_text() -> None:
    """The joined audio must read in the same order as the story."""
    segments = split_story(WITH_DIALOGUE)

    assert "stopped at the door" in segments[0].text
    assert segments[1].kind is SegmentKind.DIALOGUE
    assert "nodded" in segments[-1].text


def test_quotation_marks_are_stripped_from_dialogue() -> None:
    segments = split_story(WITH_DIALOGUE)
    dialogue = next(s for s in segments if s.kind is SegmentKind.DIALOGUE)

    assert not dialogue.text.startswith('"')
    assert not dialogue.text.endswith('"')


def test_empty_and_whitespace_stories_produce_no_segments() -> None:
    """A zero-length synthesis request is one the engine would reject."""
    assert split_story("") == []
    assert split_story("   \n\n  ") == []
    assert split_story('""') == []


def test_markup_a_voice_would_read_aloud_is_removed() -> None:
    assert "*" not in clean_segment_text("She was *very* tired.")
    assert "[" not in clean_segment_text("He left [quietly].")
    assert clean_segment_text("a    b\n\nc") == "a b c"


# --- Audio operations ----------------------------------------------------------------------------


def tone(duration_seconds: float, *, amplitude: float = 0.5, rate: int = SAMPLE_RATE) -> PcmAudio:
    count = int(duration_seconds * rate)
    t = np.arange(count, dtype=np.float32) / rate
    samples = (np.sin(2 * np.pi * 220 * t) * amplitude * 32767).astype(np.int16)
    return PcmAudio(samples=samples, sample_rate=rate)


def quiet(duration_seconds: float, *, rate: int = SAMPLE_RATE) -> PcmAudio:
    return PcmAudio(
        samples=np.zeros(int(duration_seconds * rate), dtype=np.int16), sample_rate=rate
    )


def test_trim_removes_leading_and_trailing_silence() -> None:
    padded = PcmAudio(
        samples=np.concatenate([quiet(0.5).samples, tone(1.0).samples, quiet(0.5).samples]),
        sample_rate=SAMPLE_RATE,
    )

    trimmed = audio_ops.trim_silence(padded)

    assert trimmed.duration_seconds == pytest.approx(1.0, abs=0.05)


def test_trim_keeps_the_speech_itself() -> None:
    original = tone(1.0)
    trimmed = audio_ops.trim_silence(original)

    assert trimmed.duration_seconds == pytest.approx(1.0, abs=0.05)


def test_trim_of_pure_silence_yields_nothing() -> None:
    assert audio_ops.trim_silence(quiet(1.0)).is_empty


def test_fades_start_and_end_at_zero() -> None:
    """Without this, every segment join produces an audible click."""
    faded = audio_ops.apply_fades(tone(1.0), fade_ms=20)

    assert abs(int(faded.samples[0])) < 200
    assert abs(int(faded.samples[-1])) < 200

    # The middle is untouched. Checked over a window rather than at a single index: a
    # sine crosses zero regularly, so one sample proves nothing either way.
    middle = faded.samples[len(faded.samples) // 3 : 2 * len(faded.samples) // 3]
    assert int(np.max(np.abs(middle))) > 10_000


def test_fading_a_very_short_segment_does_not_erase_it() -> None:
    short = tone(0.01)
    faded = audio_ops.apply_fades(short, fade_ms=100)

    assert len(faded.samples) == len(short.samples)


def test_join_places_the_silence_it_is_given_before_each_segment() -> None:
    joined = audio_ops.join([tone(1.0), tone(1.0)], sample_rate=SAMPLE_RATE, gaps_ms=[300, 300])

    # 0.3 lead + 1.0 + 0.3 gap + 1.0
    assert joined.duration_seconds == pytest.approx(2.6, abs=0.05)


def test_join_can_give_each_boundary_a_different_silence() -> None:
    """The reason `join` takes a list at all.

    One pause length for every boundary put the same gap between two halves of a split
    sentence, between two paragraphs, and between a question and its answer — which is
    what made the narration sound chopped and the dialogue sound like one person.
    """
    joined = audio_ops.join(
        [tone(1.0), tone(1.0), tone(1.0)], sample_rate=SAMPLE_RATE, gaps_ms=[300, 80, 420]
    )

    # 0.3 + 1.0 + 0.08 + 1.0 + 0.42 + 1.0
    assert joined.duration_seconds == pytest.approx(3.8, abs=0.05)


def test_join_refuses_a_gap_list_that_does_not_match() -> None:
    """Silently tolerating a mismatch would desynchronise the read-along.

    The timeline is computed from the same list. If `join` quietly dropped a segment or
    reused a gap, the audio and the highlight would disagree by exactly that much, and
    nothing would report it.
    """
    with pytest.raises(ValueError, match="expected 2 gaps"):
        audio_ops.join([tone(1.0), tone(1.0)], sample_rate=SAMPLE_RATE, gaps_ms=[0])


def test_normalisation_brings_a_quiet_mix_up() -> None:
    """New in v2: v1 normalised nothing, so loudness jumped between voices."""
    normalised = audio_ops.normalise_peak(tone(1.0, amplitude=0.05))

    peak = float(np.max(np.abs(normalised.samples))) / 32768.0
    assert peak == pytest.approx(audio_ops.DEFAULT_PEAK_TARGET, abs=0.02)


def test_normalisation_brings_a_hot_mix_down() -> None:
    normalised = audio_ops.normalise_peak(tone(1.0, amplitude=0.99))

    peak = float(np.max(np.abs(normalised.samples))) / 32768.0
    assert peak <= audio_ops.DEFAULT_PEAK_TARGET + 0.02


def test_normalising_silence_does_not_divide_by_zero() -> None:
    assert audio_ops.normalise_peak(quiet(0.5)).samples.max() == 0


def test_wav_output_is_a_readable_wav() -> None:
    data = audio_ops.encode_wav(tone(1.0))

    with wave.open(io.BytesIO(data), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == SAMPLE_RATE
        assert handle.getnframes() == pytest.approx(SAMPLE_RATE, rel=0.01)


def test_mp3_output_has_an_mpeg_frame_header() -> None:
    """Encoded with lameenc, so the worker image needs no ffmpeg at all."""
    data = audio_ops.encode_mp3(tone(1.0))

    assert len(data) > 1_000
    # An MPEG audio frame begins with eleven set sync bits; an ID3 tag is also valid.
    assert data[:3] == b"ID3" or (data[0] == 0xFF and (data[1] & 0xE0) == 0xE0)


def test_mp3_is_smaller_than_wav() -> None:
    audio = tone(3.0)
    assert len(audio_ops.encode_mp3(audio)) < len(audio_ops.encode_wav(audio))


def test_an_odd_trailing_byte_is_dropped_not_misaligned() -> None:
    """Misaligning PCM16 by one byte turns the whole segment into noise."""
    audio = from_pcm_bytes(b"\x01\x02\x03", SAMPLE_RATE)
    assert len(audio.samples) == 1


def test_clipping_happens_before_the_cast() -> None:
    """An out-of-range float wraps on conversion, which is a burst of noise."""
    loud = PcmAudio(samples=np.full(100, 32767, dtype=np.int16), sample_rate=SAMPLE_RATE)

    normalised = audio_ops.normalise_peak(loud, target=1.5)

    assert int(np.min(normalised.samples)) >= 0  # no wraparound to negative


# --- Pipeline fixtures ---------------------------------------------------------------------------


def make_wav_bytes(duration_seconds: float = 8.0) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(22_050)
        handle.writeframes(tone(duration_seconds, rate=22_050).samples.tobytes())
    return buffer.getvalue()


@dataclass
class StubStorage:
    """In-memory object storage, recording what was written."""

    objects: dict[str, bytes] = field(default_factory=dict)
    content_types: dict[str, str] = field(default_factory=dict)
    reads: list[str] = field(default_factory=list)
    #: Any key containing this substring raises on write. Used to prove that losing a
    #: streamed segment costs early playback and not the recording.
    fail_on: str | None = None
    #: Called after each write, so a test can inspect the world mid-run. The pipeline
    #: uploads a segment and then records it, so this is the only moment at which
    #: "what a client would see right now" is observable.
    on_put: Callable[[str], None] | None = None

    def put_bytes(self, key: str, data: bytes, *, content_type: str) -> None:
        if self.fail_on is not None and self.fail_on in key:
            raise RuntimeError("object store is having a moment")
        self.objects[key] = data
        self.content_types[key] = content_type
        if self.on_put is not None:
            self.on_put(key)

    def put_audio(self, key: str, data: bytes, audio_format: AudioFormat) -> None:
        self.put_bytes(key, data, content_type=f"audio/{audio_format.value}")

    def get_bytes(self, key: str) -> bytes:
        self.reads.append(key)
        return self.objects[key]

    def presign_get(self, key: str, *, ttl_seconds: int | None = None) -> PresignedUrl:
        return PresignedUrl(url=f"https://storage.test/{key}", expires_at=datetime.now(UTC))

    def delete(self, key: str) -> None:
        self.objects.pop(key, None)

    def exists(self, key: str) -> bool:
        return key in self.objects

    def ensure_bucket(self) -> None:
        return None


@dataclass
class StubEngine:
    """Stands in for `EngineClient`, recording what it was asked to render."""

    seconds_per_segment: float = 0.5
    fail_after: int | None = None
    failure: AppError | None = None
    on_segment: Any = None

    calls: list[tuple[str, str]] = field(default_factory=list)
    cache_primed: set[str] = field(default_factory=set)

    def synthesize(
        self,
        text: str,
        *,
        voice_id: str,
        reference_loader: Any,
        language: str,
        speed: float,
        request_id: str,
    ) -> SynthesisResult:
        # Mirrors the real cache handshake: the sample is fetched only on a miss.
        if voice_id not in self.cache_primed:
            reference_loader()
            self.cache_primed.add(voice_id)

        self.calls.append((voice_id, text))

        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise self.failure or AppError(ErrorCode.TTS_UNAVAILABLE, detail="scripted failure")

        if self.on_segment is not None:
            self.on_segment(len(self.calls))

        return SynthesisResult(audio=tone(self.seconds_per_segment), chunk_count=1)

    def close(self) -> None:
        return None


@pytest.fixture
def sync_engine(tmp_path: Path) -> Any:
    engine = create_engine(f"sqlite:///{tmp_path / 'tts.sqlite3'}", connect_args={"timeout": 30})

    @event.listens_for(engine, "connect")
    def _configure(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def sessions(sync_engine: Engine) -> sessionmaker[Session]:
    return create_worker_session_factory(sync_engine)


@pytest.fixture
def redis_client() -> Any:
    client = fakeredis.FakeRedis(decode_responses=True)
    yield client
    client.close()


class RecordingPublisher(SyncEventPublisher):
    """Publishes normally, and keeps a copy of everything it published.

    Asserting on the events themselves rather than on a Redis subscription: fakeredis's
    synchronous pub/sub drops messages depending on when the subscriber is attached, so a
    test built on it fails for reasons that have nothing to do with the pipeline.
    """

    def __init__(self, redis: fakeredis.FakeRedis) -> None:
        super().__init__(redis)
        self.events: list[dict[str, Any]] = []

    def publish(self, session: Session, job_id: UUID, factory: Any) -> None:
        def recording(seq: int, at: datetime) -> Any:
            event = factory(seq, at)
            self.events.append(json.loads(event.model_dump_json()))
            return event

        super().publish(session, job_id, recording)


def published_events(publisher: SyncEventPublisher) -> list[dict[str, Any]]:
    assert isinstance(publisher, RecordingPublisher)
    return publisher.events


@pytest.fixture
def publisher(redis_client: fakeredis.FakeRedis) -> SyncEventPublisher:
    return RecordingPublisher(redis_client)


@pytest.fixture
def storage() -> StubStorage:
    return StubStorage()


@pytest.fixture
def settings() -> TtsWorkerSettings:
    return TtsWorkerSettings(
        max_segment_chars=80,
        segment_pause_ms=100,
        lead_silence_ms=100,
        cancel_check_every_segments=1,
    )


@pytest.fixture
def written_job(sessions: sessionmaker[Session], storage: StubStorage) -> Job:
    """A job at `written`, with both voices present in storage."""
    user = User(id=uuid7())
    narrator = Voice(
        id=uuid7(),
        owner_id=None,
        name="Narrator",
        storage_key="voices/narrator.wav",
        duration_seconds=8.0,
        sample_rate=22_050,
        is_builtin=True,
    )
    dialogue = Voice(
        id=uuid7(),
        owner_id=None,
        name="Dialogue",
        storage_key="voices/dialogue.wav",
        duration_seconds=8.0,
        sample_rate=22_050,
        is_builtin=True,
    )
    storage.put_bytes(narrator.storage_key, make_wav_bytes(), content_type="audio/wav")
    storage.put_bytes(dialogue.storage_key, make_wav_bytes(), content_type="audio/wav")

    job = Job(
        id=uuid7(),
        owner_id=user.id,
        status=JobStatus.WRITTEN,
        prompt="A lighthouse keeper finds the lamp cold.",
        length=StoryLength.SHORT,
        mode=VoiceMode.NARRATION,
        language=Language.EN,
        emotion=Emotion.NEUTRAL,
        speed=1.0,
        voice_id=narrator.id,
        story_text=NARRATION_ONLY,
        queued_at=datetime.now(UTC),
        written_at=datetime.now(UTC),
    )

    with sessions() as session:
        session.add(user)
        session.flush()
        session.add_all([narrator, dialogue])
        session.flush()
        session.add(job)
        session.commit()
    return job


def reload_job(sessions: sessionmaker[Session], job_id: UUID) -> Job:
    with sessions() as session:
        job = session.scalar(select(Job).where(Job.id == job_id))
    assert job is not None
    return job


def run(
    sessions: sessionmaker[Session],
    publisher: SyncEventPublisher,
    engine: StubEngine,
    storage: StubStorage,
    settings: TtsWorkerSettings,
    job_id: UUID,
) -> Any:
    return run_tts_stage(sessions, publisher, engine, storage, settings, job_id)  # type: ignore[arg-type]


# --- Pipeline ------------------------------------------------------------------------------------


def test_segment_spans_point_back_into_the_original_story() -> None:
    """The span must locate the segment in the *written* story, not the spoken one.

    Spoken text is cleaned — quotes stripped, whitespace collapsed — so it cannot be
    found again with a search. The span is the only way back, and a player highlighting
    the wrong characters is worse than one highlighting none.
    """
    segments = split_story(WITH_DIALOGUE)

    for segment in segments:
        excerpt = WITH_DIALOGUE[segment.start_char : segment.end_char]
        # The excerpt is the written form of what is spoken: same text once cleaned.
        assert clean_segment_text(excerpt) == segment.text

    # Spans are ordered and never overlap, so a position maps to at most one segment.
    for earlier, later in itertools.pairwise(segments):
        assert earlier.end_char <= later.start_char


def test_dialogue_spans_exclude_the_quotation_marks() -> None:
    """The quotes are punctuation on the page, not part of the spoken line."""
    spoken = [segment for segment in split_story(WITH_DIALOGUE) if segment.is_dialogue]

    assert len(spoken) == 1
    excerpt = WITH_DIALOGUE[spoken[0].start_char : spoken[0].end_char]
    assert excerpt == "I am not leaving without you,"


def test_the_timeline_matches_where_the_audio_actually_lands(
    sessions: sessionmaker[Session],
    publisher: SyncEventPublisher,
    storage: StubStorage,
    settings: TtsWorkerSettings,
    written_job: Job,
) -> None:
    """The timeline is arithmetic that reproduces `join`, so it can drift from it.

    Checked against the real assembled track rather than against itself: the lead-in
    comes first, every gap is one of the three the settings allow, and the last segment
    ends where the audio does. If `join` ever changes its spacing and this does not,
    these are the assertions that fail.
    """
    run(sessions, publisher, StubEngine(), storage, settings, written_job.id)

    finished = reload_job(sessions, written_job.id)
    timeline = finished.segment_timeline
    assert timeline is not None
    assert len(timeline) == finished.segment_count

    assert timeline[0]["start_seconds"] == pytest.approx(settings.lead_silence_ms / 1000, abs=0.002)

    permitted = {
        settings.segment_pause_ms / 1000,
        settings.continuation_pause_ms / 1000,
        settings.speaker_change_pause_ms / 1000,
    }
    for earlier, later in itertools.pairwise(timeline):
        assert earlier["end_seconds"] < later["start_seconds"]
        gap = later["start_seconds"] - earlier["end_seconds"]
        assert any(gap == pytest.approx(allowed, abs=0.002) for allowed in permitted), gap

    assert finished.audio_duration_seconds is not None
    assert timeline[-1]["end_seconds"] == pytest.approx(finished.audio_duration_seconds, abs=0.01)


def place_all(segments: list[Segment], story: str) -> list[int]:
    """Run the assembler over rendered segments and report the gap chosen for each."""
    from tts_worker.pipeline import _Assembler

    assembler = _Assembler(
        TtsWorkerSettings(
            lead_silence_ms=300,
            segment_pause_ms=300,
            continuation_pause_ms=80,
            speaker_change_pause_ms=420,
        ),
        story,
    )
    gaps: list[int] = []
    for segment in segments:
        placed = assembler.place(segment, tone(0.2))
        assert placed is not None
        gaps.append(placed.gap_ms)
    return gaps


def test_a_change_of_speaker_gets_a_longer_beat_than_a_split() -> None:
    """The three boundary kinds must produce three different silences.

    Asserted on `_gaps` directly rather than through a rendered job, because the stub
    engine's segmentation is not guaranteed to contain all three kinds — and a test that
    only sometimes exercises the branch it is named for is worse than no test.
    """
    story = 'Mara waited by the rail.\n\n"Say it," said Mara. "Say it plainly."'
    segments = split_story(story, max_segment_chars=300)
    gaps = place_all(segments, story)

    # narration | "Say it," (Mara) | said Mara. | "Say it plainly." (Mara)
    assert [segment.kind.value for segment in segments] == [
        "narration",
        "dialogue",
        "narration",
        "dialogue",
    ]

    # Every boundary here crosses between the narrator and a character, including the two
    # around the attribution tag: the tag is read by the narrator, so returning to Mara
    # afterwards is a change of voice even though it is the same character speaking.
    assert gaps == [300, 420, 420, 420]


def test_a_split_that_only_exists_because_of_length_barely_pauses() -> None:
    """The boundary this pipeline invented, as opposed to one the author wrote.

    A reader does not pause in the middle of a paragraph, and the old code put the same
    300 ms there as it put between paragraphs — roughly eight invented pauses in a
    400-word story, which is what made narration sound sliced.
    """
    story = " ".join(f"Sentence number {index} runs on for a while." for index in range(12))
    segments = split_story(story, max_segment_chars=120)
    gaps = place_all(segments, story)

    assert len(segments) > 2, "the story must actually be split for this to test anything"
    # One paragraph, one speaker: every boundary is an artefact of the size limit.
    assert gaps[1:] == [80] * (len(segments) - 1)


def test_a_second_voice_is_cast_to_the_second_character() -> None:
    """One voice for every spoken line is what made a two-hander sound like one person."""
    from tts_worker.pipeline import _build_cast, _Request

    story = '"We should sell it," Mara said.\n\n"You cannot," Lila said.'
    segments = split_story(story, max_segment_chars=300)

    first, second = uuid7(), uuid7()
    request = _Request(
        story=story,
        language="en",
        speed=1.0,
        mode=VoiceMode.NARRATION_WITH_DIALOGUE,
        narrator_voice_id=uuid7(),
        narrator_key="narrator.wav",
        dialogue_voice_id=first,
        dialogue_key="one.wav",
        second_dialogue_voice_id=second,
        second_dialogue_key="two.wav",
    )

    cast = _build_cast(segments, request)

    assert cast["Mara"] == first
    assert cast["Lila"] == second


def test_without_a_second_voice_everyone_keeps_the_first() -> None:
    """The old behaviour, preserved deliberately.

    With one dialogue voice there is nothing better to do than use it for every
    character. Promoting the narrator into a speaking part would be this function
    inventing a casting decision the caller did not make.
    """
    from tts_worker.pipeline import _build_cast, _Request

    story = '"We should sell it," Mara said.\n\n"You cannot," Lila said.'
    segments = split_story(story, max_segment_chars=300)

    only = uuid7()
    request = _Request(
        story=story,
        language="en",
        speed=1.0,
        mode=VoiceMode.NARRATION_WITH_DIALOGUE,
        narrator_voice_id=uuid7(),
        narrator_key="narrator.wav",
        dialogue_voice_id=only,
        dialogue_key="one.wav",
        second_dialogue_voice_id=None,
        second_dialogue_key=None,
    )

    assert set(_build_cast(segments, request).values()) == {only}


def test_every_segment_is_published_while_the_job_is_still_running(
    sessions: sessionmaker[Session],
    publisher: SyncEventPublisher,
    storage: StubStorage,
    settings: TtsWorkerSettings,
    written_job: Job,
) -> None:
    """The point of the whole change: audio exists before the job is done.

    Each segment is uploaded and announced as it lands, so the first one is playable
    seconds into a job that takes minutes. Without this the listener waits for the last
    segment before hearing the first — the same shape of mistake v1 made one level up,
    where the request blocked until everything was finished.
    """
    run(sessions, publisher, StubEngine(), storage, settings, written_job.id)

    finished = reload_job(sessions, written_job.id)
    assert finished.segment_count is not None

    ready = [event for event in published_events(publisher) if event.get("type") == "segment_ready"]
    assert len(ready) == finished.segment_count

    for position, frame in enumerate(ready):
        assert frame["index"] == position
        assert f"segment-{position:04d}.wav" in " ".join(storage.objects)
        # No URL in the event: signing here would bake a TTL into a durable event.
        assert "url" not in frame


def test_the_timeline_is_readable_before_the_job_finishes(
    sessions: sessionmaker[Session],
    publisher: SyncEventPublisher,
    storage: StubStorage,
    settings: TtsWorkerSettings,
    written_job: Job,
) -> None:
    """A client that reloads mid-story must still be able to play what exists.

    Segments are announced over Redis pub/sub, which keeps no history: a browser that
    reloads, or opens the page late, is told about none of the segments already rendered.
    If the only stored copy of the timeline appeared when the job finished, that client
    could see 0 of 22 parts while twenty of them sat in object storage — and the compose
    page promises you can close the tab and come back.
    """
    seen: list[int] = []

    def inspect(key: str) -> None:
        if "segment-" not in key:
            return
        # The upload happens first, then the row is written, so the count observed here
        # is one behind — which is exactly the window a reloading client can land in.
        with sessions() as session:
            job = session.get(Job, written_job.id)
            assert job is not None
            stored = job.segment_timeline or []
            assert len(stored) == len(seen)
        seen.append(len(seen))

    storage.on_put = inspect
    run(sessions, publisher, StubEngine(), storage, settings, written_job.id)

    assert len(seen) > 1, "the story must render more than one segment to prove this"

    # And by the end, the stored timeline covers everything.
    finished = reload_job(sessions, written_job.id)
    assert finished.segment_timeline is not None
    assert len(finished.segment_timeline) == len(seen)


def test_a_published_segment_lands_where_the_final_track_puts_it(
    sessions: sessionmaker[Session],
    publisher: SyncEventPublisher,
    storage: StubStorage,
    settings: TtsWorkerSettings,
    written_job: Job,
) -> None:
    """Streamed positions and the assembled file must agree exactly.

    A client schedules the segments it receives against one clock to reproduce the track
    the worker is building. If the announced start times drifted from the finished audio,
    progressive playback would sound different from the download of the same story — and
    only the second one would be right.
    """
    run(sessions, publisher, StubEngine(), storage, settings, written_job.id)

    finished = reload_job(sessions, written_job.id)
    timeline = finished.segment_timeline
    assert timeline is not None

    ready = [event for event in published_events(publisher) if event.get("type") == "segment_ready"]
    assert [event["start_seconds"] for event in ready] == [
        entry["start_seconds"] for entry in timeline
    ]
    assert [event["end_seconds"] for event in ready] == [entry["end_seconds"] for entry in timeline]


def test_a_failed_segment_upload_does_not_fail_the_job(
    sessions: sessionmaker[Session],
    publisher: SyncEventPublisher,
    storage: StubStorage,
    settings: TtsWorkerSettings,
    written_job: Job,
) -> None:
    """Progressive playback is an enhancement and must never cost the story.

    The segment is already rendered and will be in the assembled track either way;
    losing the streamed copy costs early playback for that one segment. Failing the job
    over it would trade the whole recording for a convenience.
    """

    # Set after the fixtures have seeded the reference voices, so only the streamed
    # segment uploads fail.
    storage.fail_on = "segment-"

    outcome = run(sessions, publisher, StubEngine(), storage, settings, written_job.id)

    assert outcome.completed
    finished = reload_job(sessions, written_job.id)
    assert finished.status is JobStatus.DONE
    assert finished.audio_key_mp3 in storage.objects


def test_the_timeline_carries_what_a_player_needs(
    sessions: sessionmaker[Session],
    publisher: SyncEventPublisher,
    storage: StubStorage,
    settings: TtsWorkerSettings,
    written_job: Job,
) -> None:
    run(sessions, publisher, StubEngine(), storage, settings, written_job.id)

    finished = reload_job(sessions, written_job.id)
    assert finished.segment_timeline is not None
    assert finished.story_text is not None

    for index, entry in enumerate(finished.segment_timeline):
        assert entry["index"] == index
        assert entry["text"]
        assert entry["start_char"] < entry["end_char"] <= len(finished.story_text)
        assert entry["start_seconds"] < entry["end_seconds"]


def test_the_job_reaches_done_with_audio_stored(
    sessions: sessionmaker[Session],
    publisher: SyncEventPublisher,
    storage: StubStorage,
    settings: TtsWorkerSettings,
    written_job: Job,
) -> None:
    outcome = run(sessions, publisher, StubEngine(), storage, settings, written_job.id)

    assert outcome.completed
    finished = reload_job(sessions, written_job.id)
    assert finished.status is JobStatus.DONE
    assert finished.audio_key_mp3 == f"audio/{written_job.id}.mp3"
    assert finished.audio_key_wav == f"audio/{written_job.id}.wav"
    assert finished.audio_duration_seconds is not None
    assert finished.audio_duration_seconds > 0
    assert finished.audio_key_mp3 in storage.objects


def test_both_formats_are_uploaded_with_the_right_content_type(
    sessions: sessionmaker[Session],
    publisher: SyncEventPublisher,
    storage: StubStorage,
    settings: TtsWorkerSettings,
    written_job: Job,
) -> None:
    run(sessions, publisher, StubEngine(), storage, settings, written_job.id)

    assert storage.content_types[f"audio/{written_job.id}.mp3"] == "audio/mp3"
    assert storage.content_types[f"audio/{written_job.id}.wav"] == "audio/wav"


def test_the_stored_wav_is_playable(
    sessions: sessionmaker[Session],
    publisher: SyncEventPublisher,
    storage: StubStorage,
    settings: TtsWorkerSettings,
    written_job: Job,
) -> None:
    """Prove a real file came out, not merely that a key was written."""
    run(sessions, publisher, StubEngine(), storage, settings, written_job.id)

    with wave.open(io.BytesIO(storage.objects[f"audio/{written_job.id}.wav"]), "rb") as handle:
        assert handle.getnframes() > 0
        assert handle.getframerate() == SAMPLE_RATE


def test_segment_count_and_progress_are_recorded(
    sessions: sessionmaker[Session],
    publisher: SyncEventPublisher,
    storage: StubStorage,
    settings: TtsWorkerSettings,
    written_job: Job,
) -> None:
    """Real counts, not an indeterminate spinner."""
    outcome = run(sessions, publisher, StubEngine(), storage, settings, written_job.id)

    finished = reload_job(sessions, written_job.id)
    assert finished.segment_count == outcome.segment_count
    assert finished.segments_done == outcome.segment_count
    assert outcome.segment_count > 1


def test_stage_timestamps_are_recorded(
    sessions: sessionmaker[Session],
    publisher: SyncEventPublisher,
    storage: StubStorage,
    settings: TtsWorkerSettings,
    written_job: Job,
) -> None:
    run(sessions, publisher, StubEngine(), storage, settings, written_job.id)

    finished = reload_job(sessions, written_job.id)
    assert finished.synthesizing_at is not None
    assert finished.finished_at is not None
    assert finished.finished_at >= finished.synthesizing_at


# --- Voices --------------------------------------------------------------------------------------


def test_narration_mode_uses_one_voice_throughout(
    sessions: sessionmaker[Session],
    publisher: SyncEventPublisher,
    storage: StubStorage,
    settings: TtsWorkerSettings,
    written_job: Job,
) -> None:
    engine = StubEngine()
    run(sessions, publisher, engine, storage, settings, written_job.id)

    assert len({voice_id for voice_id, _ in engine.calls}) == 1


def test_dialogue_mode_uses_the_second_voice_for_spoken_lines(
    sessions: sessionmaker[Session],
    publisher: SyncEventPublisher,
    storage: StubStorage,
    settings: TtsWorkerSettings,
    written_job: Job,
) -> None:
    """v1 hardcoded voices/female.wav for every spoken line, whatever the user chose."""
    with sessions() as session:
        dialogue_voice = session.scalar(select(Voice).where(Voice.name == "Dialogue"))
        assert dialogue_voice is not None
        session.query(Job).filter(Job.id == written_job.id).update(
            {
                "mode": VoiceMode.NARRATION_WITH_DIALOGUE,
                "dialogue_voice_id": dialogue_voice.id,
                "story_text": WITH_DIALOGUE,
            }
        )
        session.commit()
        expected_dialogue = str(dialogue_voice.id)

    engine = StubEngine()
    run(sessions, publisher, engine, storage, settings, written_job.id)

    used = {voice_id for voice_id, _ in engine.calls}
    assert len(used) == 2
    assert expected_dialogue in used


def test_the_reference_sample_is_fetched_once_per_voice(
    sessions: sessionmaker[Session],
    publisher: SyncEventPublisher,
    storage: StubStorage,
    settings: TtsWorkerSettings,
    written_job: Job,
) -> None:
    """v1 re-read the WAV and recomputed conditioning for every segment."""
    engine = StubEngine()
    run(sessions, publisher, engine, storage, settings, written_job.id)

    assert len(engine.calls) > 1
    assert storage.reads.count("voices/narrator.wav") == 1


# --- Cancellation and failure --------------------------------------------------------------------


def test_cancelling_mid_synthesis_stops_the_work(
    sessions: sessionmaker[Session],
    publisher: SyncEventPublisher,
    storage: StubStorage,
    settings: TtsWorkerSettings,
    written_job: Job,
) -> None:
    """This is what frees the GPU. v1 could not cancel at all."""

    def cancel_after_first(call_count: int) -> None:
        if call_count == 1:
            with sessions() as session:
                session.query(Job).filter(Job.id == written_job.id).update(
                    {"status": JobStatus.CANCELLED}
                )
                session.commit()

    engine = StubEngine(on_segment=cancel_after_first)

    with pytest.raises(AppError) as info:
        run(sessions, publisher, engine, storage, settings, written_job.id)

    assert info.value.code is ErrorCode.CANCELLED
    # Stopped early rather than rendering the whole story.
    assert len(engine.calls) < 5
    assert reload_job(sessions, written_job.id).status is JobStatus.CANCELLED


def test_a_job_cancelled_during_upload_is_not_overwritten(
    sessions: sessionmaker[Session],
    publisher: SyncEventPublisher,
    storage: StubStorage,
    settings: TtsWorkerSettings,
    written_job: Job,
) -> None:
    """The conditional UPDATE protects the user's decision from a late completion."""

    class LateCancelStorage(StubStorage):
        def put_audio(self, key: str, data: bytes, audio_format: AudioFormat) -> None:
            super().put_audio(key, data, audio_format)
            with sessions() as session:
                session.query(Job).filter(Job.id == written_job.id).update(
                    {"status": JobStatus.CANCELLED}
                )
                session.commit()

    late = LateCancelStorage()
    late.objects.update(storage.objects)

    with pytest.raises(AppError) as info:
        run(sessions, publisher, StubEngine(), late, settings, written_job.id)

    assert info.value.code is ErrorCode.CANCELLED
    final = reload_job(sessions, written_job.id)
    assert final.status is JobStatus.CANCELLED
    assert final.audio_key_mp3 is None


def test_engine_failures_propagate_classified(
    sessions: sessionmaker[Session],
    publisher: SyncEventPublisher,
    storage: StubStorage,
    settings: TtsWorkerSettings,
    written_job: Job,
) -> None:
    engine = StubEngine(
        fail_after=1, failure=AppError(ErrorCode.TTS_CAPACITY, detail="scripted busy")
    )

    with pytest.raises(AppError) as info:
        run(sessions, publisher, engine, storage, settings, written_job.id)

    assert info.value.code is ErrorCode.TTS_CAPACITY


def test_a_story_that_yields_no_segments_fails_clearly(
    sessions: sessionmaker[Session],
    publisher: SyncEventPublisher,
    storage: StubStorage,
    settings: TtsWorkerSettings,
    written_job: Job,
) -> None:
    with sessions() as session:
        session.query(Job).filter(Job.id == written_job.id).update({"story_text": "   "})
        session.commit()

    with pytest.raises(AppError) as info:
        run(sessions, publisher, StubEngine(), storage, settings, written_job.id)

    assert info.value.code is ErrorCode.AUDIO_ASSEMBLY_FAILED


def test_segments_at_mismatched_sample_rates_are_refused(
    sessions: sessionmaker[Session],
    publisher: SyncEventPublisher,
    storage: StubStorage,
    settings: TtsWorkerSettings,
    written_job: Job,
) -> None:
    """Concatenating mismatched rates plays back at the wrong speed rather than failing."""

    class DriftingEngine(StubEngine):
        def synthesize(self, text: str, **kwargs: Any) -> SynthesisResult:
            result = super().synthesize(text, **kwargs)
            if len(self.calls) == 2:
                return SynthesisResult(audio=tone(0.5, rate=16_000), chunk_count=result.chunk_count)
            return result

    with pytest.raises(AppError) as info:
        run(sessions, publisher, DriftingEngine(), storage, settings, written_job.id)

    assert info.value.code is ErrorCode.AUDIO_ASSEMBLY_FAILED


# --- Idempotency ---------------------------------------------------------------------------------


@pytest.mark.parametrize("status", [JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED])
def test_a_terminal_job_is_left_alone(
    sessions: sessionmaker[Session],
    publisher: SyncEventPublisher,
    storage: StubStorage,
    settings: TtsWorkerSettings,
    written_job: Job,
    status: JobStatus,
) -> None:
    with sessions() as session:
        session.query(Job).filter(Job.id == written_job.id).update({"status": status})
        session.commit()

    engine = StubEngine()
    outcome = run(sessions, publisher, engine, storage, settings, written_job.id)

    assert not outcome.completed
    assert engine.calls == []
    assert reload_job(sessions, written_job.id).status is status


def test_a_redelivery_mid_synthesis_restarts_and_completes(
    sessions: sessionmaker[Session],
    publisher: SyncEventPublisher,
    storage: StubStorage,
    settings: TtsWorkerSettings,
    written_job: Job,
) -> None:
    """Rendered PCM is not persisted, so synthesis restarts -- but the LLM never reruns."""
    with sessions() as session:
        session.query(Job).filter(Job.id == written_job.id).update(
            {"status": JobStatus.SYNTHESIZING, "synthesizing_at": datetime.now(UTC)}
        )
        session.commit()

    outcome = run(sessions, publisher, StubEngine(), storage, settings, written_job.id)

    assert outcome.completed
    finished = reload_job(sessions, written_job.id)
    assert finished.status is JobStatus.DONE
    # The story is untouched: the expensive half was never redone.
    assert finished.story_text == NARRATION_ONLY


def test_a_missing_job_is_not_an_error(
    sessions: sessionmaker[Session],
    publisher: SyncEventPublisher,
    storage: StubStorage,
    settings: TtsWorkerSettings,
) -> None:
    outcome = run(sessions, publisher, StubEngine(), storage, settings, uuid7())
    assert not outcome.completed
