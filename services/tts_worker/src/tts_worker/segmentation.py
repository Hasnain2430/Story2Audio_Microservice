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
    """One unit of synthesis."""

    kind: SegmentKind
    text: str

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
        narration = story[cursor : match.start()]
        segments.extend(_narration_segments(narration, max_segment_chars))

        # Exactly one of the two alternation groups matched.
        spoken = match.group(1) if match.group(1) is not None else match.group(2)
        segments.extend(_dialogue_segments(spoken or "", max_segment_chars))

        cursor = match.end()

    segments.extend(_narration_segments(story[cursor:], max_segment_chars))
    return segments


def _narration_segments(text: str, max_chars: int) -> list[Segment]:
    return [Segment(kind=SegmentKind.NARRATION, text=chunk) for chunk in _chunk(text, max_chars)]


def _dialogue_segments(text: str, max_chars: int) -> list[Segment]:
    return [Segment(kind=SegmentKind.DIALOGUE, text=chunk) for chunk in _chunk(text, max_chars)]


def _chunk(text: str, max_chars: int) -> list[str]:
    """Group sentences into chunks of at most ``max_chars``.

    Splits on sentence boundaries rather than mid-word, so a segment break lands where a
    reader would pause anyway. A single sentence longer than the limit is kept whole: a
    hard cut inside a clause is worse than one long segment, and the engine splits
    further internally.
    """
    cleaned = clean_segment_text(text)
    if not cleaned:
        return []

    sentences = [part for part in _SENTENCE_BOUNDARY.split(cleaned) if part.strip()]
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    length = 0

    for sentence in sentences:
        addition = len(sentence) + (1 if current else 0)
        if current and length + addition > max_chars:
            chunks.append(" ".join(current))
            current = [sentence]
            length = len(sentence)
        else:
            current.append(sentence)
            length += addition

    if current:
        chunks.append(" ".join(current))

    return [chunk for chunk in chunks if chunk.strip()]
