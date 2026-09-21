"""Splitting a story into synthesis segments.

Ported from v1's `split_into_narration_and_dialogues`, which was one of the things v1 got
right: quoted speech is separated from narration so the two can be rendered in different
voices. Three changes.

**Narration is chunked.** In narration-only mode v1 handed the *entire story* to a single
``tts_to_file`` call. That is why it could never report progress — there was exactly one
unit of work — and why long stories synthesised poorly, since XTTS degrades on very long
inputs. Narration is now split on sentence boundaries into segments of a few hundred
characters, which gives better synthesis, real progress, and a boundary at which
cancellation can take effect.

**Curly quotes count.** v1 matched only `"`. Language models routinely emit `“ ”`, so
every line of dialogue in such a story silently stayed narration and was read in the
narrator's voice.

**Empty fragments are dropped early**, rather than producing a zero-length synthesis
request the engine would reject.

**Every segment keeps its span in the original story.** The text that is spoken is not
the text that was written -- quotes are stripped, asterisks removed, whitespace
collapsed -- so a segment cannot be found again by searching for it. Carrying
``start_char`` and ``end_char`` is what lets the player highlight the story as it is
read: the timeline is built from segment durations, and the span says which characters
that covers. Without them the UI would have to display the spoken text instead of the
written story, which is a subtly different document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from story2audio_shared.enums import SegmentKind

#: Target size for a narration segment. Comfortably inside XTTS's effective input window
#: while still large enough that segment boundaries do not audibly chop the prose.
DEFAULT_MAX_SEGMENT_CHARS = 300

#: Opening/closing quote pairs treated as dialogue delimiters.
_QUOTE_PATTERN = re.compile(r'"([^"]*)"|“([^”]*)”')

#: Sentence boundary: terminal punctuation followed by whitespace. Includes the Hindi
#: danda and full-width stops, since the story may be in any supported language.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…।؟！？。])\s+")


@dataclass(frozen=True, slots=True)
class Segment:
    """One unit of synthesis, and where it came from."""

    kind: SegmentKind
    #: Cleaned text, which is what is actually spoken.
    text: str
    #: Half-open span of the *original* story this segment was taken from. Spoken text
    #: differs from written text, so this is the only reliable way back to it.
    start_char: int
    end_char: int

    @property
    def is_dialogue(self) -> bool:
        return self.kind is SegmentKind.DIALOGUE


def clean_segment_text(text: str) -> str:
    """Normalise a fragment before it is spoken.

    Strips stray quotation marks left at the edges by the split, collapses whitespace,
    and removes characters a TTS voice reads badly. v1's `clean_sentence` did the first
    two; the third is new, because a model that emits an asterisk or a bracketed aside
    makes the narrator say "asterisk".
    """
    cleaned = text.strip()
    cleaned = re.sub(r'^[\'"“”‘’]+', "", cleaned)
    cleaned = re.sub(r'[\'"“”‘’]+$', "", cleaned)
    # Characters that are punctuation on the page and noise in the ear.
    cleaned = re.sub(r"[*_`~<>|\[\]{}]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def split_story(story: str, *, max_segment_chars: int = DEFAULT_MAX_SEGMENT_CHARS) -> list[Segment]:
    """Split a story into ordered narration and dialogue segments.

    Order is preserved exactly, so the joined audio reads in the same sequence as the
    text. A story with no quoted speech yields narration segments only.
    """
    segments: list[Segment] = []
    cursor = 0

    for match in _QUOTE_PATTERN.finditer(story):
        segments.extend(
            _chunk(story[cursor : match.start()], max_segment_chars, SegmentKind.NARRATION, cursor)
        )

        # Exactly one of the two alternation groups matched. The span taken is the
        # group's, not the match's, so the quotation marks stay outside the highlight.
        group = 1 if match.group(1) is not None else 2
        segments.extend(
            _chunk(
                match.group(group) or "",
                max_segment_chars,
                SegmentKind.DIALOGUE,
                match.start(group),
            )
        )

        cursor = match.end()

    segments.extend(_chunk(story[cursor:], max_segment_chars, SegmentKind.NARRATION, cursor))
    return segments


@dataclass(frozen=True, slots=True)
class _Sentence:
    """One sentence, cleaned for speech, still carrying where it was written."""

    text: str
    start: int
    end: int


def _chunk(text: str, max_chars: int, kind: SegmentKind, base: int) -> list[Segment]:
    """Group sentences into segments of at most ``max_chars``.

    Splits on sentence boundaries rather than mid-word, so a segment break lands where a
    reader would pause anyway. A single sentence longer than the limit is kept whole: a
    hard cut inside a clause is worse than one long segment, and the engine splits
    further internally.

    Sentences are cut from the *raw* text and cleaned one at a time, rather than cleaning
    the whole fragment and splitting the result. Same spoken output, but the offsets
    survive it: cleaning is not length-preserving, so doing it first would make every
    position afterwards a guess.
    """
    sentences = _sentences(text, base)
    if not sentences:
        return []

    segments: list[Segment] = []
    current: list[_Sentence] = []
    length = 0

    for sentence in sentences:
        addition = len(sentence.text) + (1 if current else 0)
        if current and length + addition > max_chars:
            segments.append(_join(current, kind))
            current = [sentence]
            length = len(sentence.text)
        else:
            current.append(sentence)
            length += addition

    if current:
        segments.append(_join(current, kind))

    return segments


def _sentences(text: str, base: int) -> list[_Sentence]:
    """Cut ``text`` at sentence boundaries, keeping absolute offsets.

    Spans are tightened past surrounding whitespace, so a highlight starts on a letter
    rather than on the space in front of it.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for boundary in _SENTENCE_BOUNDARY.finditer(text):
        spans.append((start, boundary.start()))
        start = boundary.end()
    spans.append((start, len(text)))

    sentences: list[_Sentence] = []
    for begin, finish in spans:
        raw = text[begin:finish]
        cleaned = clean_segment_text(raw)
        if not cleaned:
            continue
        lead = len(raw) - len(raw.lstrip())
        trail = len(raw) - len(raw.rstrip())
        sentences.append(
            _Sentence(text=cleaned, start=base + begin + lead, end=base + finish - trail)
        )
    return sentences


def _join(sentences: list[_Sentence], kind: SegmentKind) -> Segment:
    return Segment(
        kind=kind,
        text=" ".join(sentence.text for sentence in sentences),
        start_char=sentences[0].start,
        end_char=sentences[-1].end,
    )
