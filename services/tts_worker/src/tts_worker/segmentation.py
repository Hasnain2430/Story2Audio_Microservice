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

**Quoted lines are attributed to a speaker.** v1 sent every quoted line to one hardcoded
voice, so a scene between two people was read by the same person twice. Here the
attribution tag beside each line ("Mara said", "said Mara", "she whispered") is parsed and
the speaker's name travels with the segment, which is what lets the pipeline give each
character a voice and keep it across the story.

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
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Final

from story2audio_shared.enums import SegmentKind

#: Target size for a narration segment. Large enough that segment boundaries do not
#: audibly chop the prose, and then capped per language by :func:`max_segment_chars`.
DEFAULT_MAX_SEGMENT_CHARS = 300

#: Hard per-language input limits, copied from Coqui's own tokenizer
#: (``TTS/tts/layers/xtts/tokenizer.py``, ``VoiceBpeTokenizer.char_limits``). Past these
#: XTTS logs "this might cause truncated audio" and carries on — it does not raise, and
#: the truncation is only discoverable by listening. A 300-character target was over the
#: limit for every language here except French.
XTTS_CHAR_LIMITS: Final[dict[str, int]] = {
    "en": 250,
    "de": 253,
    "fr": 273,
    "es": 239,
    "it": 213,
    "ru": 182,
    "hi": 250,
}

#: Coqui's fallback for a language it has no entry for.
_DEFAULT_CHAR_LIMIT = 250

#: Clause boundaries, used only to break a sentence that is too long to synthesise whole.
#: Ordered by how little damage the break does to the reading.
_CLAUSE_BOUNDARY = re.compile(r"(?<=[;:,—–])\s+")


def max_segment_chars(language: str, configured: int = DEFAULT_MAX_SEGMENT_CHARS) -> int:
    """The largest segment this language can actually synthesise.

    Takes the stricter of what the operator configured and what XTTS will accept, because
    exceeding the model's limit does not fail loudly — it truncates the audio and logs a
    warning nobody reads.
    """
    return min(configured, XTTS_CHAR_LIMITS.get(language, _DEFAULT_CHAR_LIMIT))


#: Opening/closing quote pairs treated as dialogue delimiters.
_QUOTE_PATTERN = re.compile(r'"([^"]*)"|“([^”]*)”')

#: Sentence boundary: terminal punctuation followed by whitespace. Includes the Hindi
#: danda and full-width stops, since the story may be in any supported language.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…।؟！？。])\s+")

#: Verbs that introduce speech. Deliberately short: a longer list buys little and starts
#: matching ordinary narration, which would attribute a line to the wrong character —
#: worse than leaving it unattributed, because a wrong voice is audible and a fallback is
#: merely unremarkable.
_SPEECH_VERBS = (
    r"said|says|asked|asks|replied|replies|answered|answers|added|adds|"
    r"whispered|whispers|murmured|murmurs|called|calls|shouted|shouts|"
    r"offered|offers|insisted|insists|repeated|repeats|told|tells"
)

#: A capitalised name, allowing one particle ("Mara", "Old Tom"). Anchored to the edge of
#: the attribution window so a capitalised word deeper in the sentence is not mistaken for
#: the speaker.
_NAME = r"[A-Z][\w'’-]+(?:\s+[A-Z][\w'’-]+)?"

#: "said Mara" / "whispered Old Tom" — verb first, immediately after the quote.
_TAG_AFTER_VERB_FIRST = re.compile(rf"^[\s,—–-]*(?:{_SPEECH_VERBS})\s+({_NAME})")

#: "Mara said" / "Mara finally said" — name first, then the verb close behind it.
_TAG_AFTER_NAME_FIRST = re.compile(rf"^[\s,—–-]*({_NAME})\s+(?:\w+\s+)?(?:{_SPEECH_VERBS})\b")

#: "Mara said," immediately *before* an opening quote, for lines tagged in front.
_TAG_BEFORE = re.compile(rf"({_NAME})\s+(?:\w+\s+)?(?:{_SPEECH_VERBS})\b[\s,:—–-]*$")

#: How far either side of a quoted line to look for its attribution. One clause, not one
#: sentence: the tag sits immediately beside the line, and a wider window starts finding
#: names belonging to the next sentence.
_ATTRIBUTION_WINDOW = 60

#: Words that open a sentence and are capitalised for that reason alone. Without this an
#: action beat like "The wind began to howl." reads as a character called "The".
_NOT_A_NAME: Final[frozenset[str]] = frozenset(
    {
        "A",
        "An",
        "The",
        "He",
        "She",
        "They",
        "It",
        "We",
        "You",
        "I",
        "His",
        "Her",
        "Their",
        "Its",
        "Our",
        "My",
        "Your",
        "There",
        "Then",
        "That",
        "This",
        "These",
        "Those",
        "Here",
        "When",
        "While",
        "After",
        "Before",
        "As",
        "Once",
        "Now",
        "Later",
        "But",
        "And",
        "So",
        "Yet",
        "For",
        "Still",
        "Even",
        "Only",
        "Just",
        "Outside",
        "Inside",
        "Above",
        "Below",
        "Beyond",
        "Behind",
        "Across",
        "One",
        "Two",
        "Every",
        "Each",
        "Some",
        "No",
        "Neither",
        "Both",
        "In",
        "On",
        "At",
        "By",
        "From",
        "With",
        "Without",
        "Through",
        "Somewhere",
        "Something",
        "Someone",
        "Nothing",
        "Finally",
        "Suddenly",
    }
)

#: A pronoun speech tag: "he said", "she whispered". It names no one, but it does say
#: which of the cast it is not — which is enough when the cast is two people who are
#: referred to differently in the prose.
#: The run these skip over includes the quotation marks themselves. A segment's span
#: covers the words *inside* the quotes, so the text immediately after it starts with the
#: closing mark — and a character class that did not allow it matched nothing at all.
_PRONOUN_TAG_AFTER = re.compile(
    rf"^[\s,.;:!?—–\-\"'“”‘’]*(he|she)\s+(?:\w+\s+)?(?:{_SPEECH_VERBS})\b", re.I
)
_PRONOUN_TAG_BEFORE = re.compile(
    rf"\b(he|she)\s+(?:\w+\s+)?(?:{_SPEECH_VERBS})\b[\s,:—–\-\"'“”‘’]*$", re.I
)

#: Pronouns counted near a name to guess how the story refers to that character.
_MASCULINE = re.compile(r"\b(?:he|him|his)\b", re.I)
_FEMININE = re.compile(r"\b(?:she|her|hers)\b", re.I)

#: How much text after a name to inspect when working out which pronoun follows it.
_GENDER_WINDOW = 120

#: An action beat: the paragraph opens with a character doing something, and the quoted
#: line follows in the same paragraph. "Tom set the cloth aside, his voice steady. '...'"
#: This is how a model writes most of its attribution once it is told to avoid adverbial
#: speech tags, and reading only "said" tags misses it entirely — which leaves that
#: character out of the cast and their lines in somebody else's voice.
_ACTION_BEAT = re.compile(rf"^\s*({_NAME})\s+[a-z]")


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
    #: Who says this, for dialogue whose attribution could be read. ``None`` for narration
    #: and for a line with no usable tag — the caller decides what an unknown speaker
    #: falls back to, because that is a casting decision, not a parsing one.
    speaker: str | None = None

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
                speaker=_attribute(story, match.start(), match.end()),
            )
        )

        cursor = match.end()

    segments.extend(_chunk(story[cursor:], max_segment_chars, SegmentKind.NARRATION, cursor))
    return _resolve_speakers(segments, story)


def _resolve_speakers(segments: list[Segment], story: str) -> list[Segment]:
    """Fill in the speakers that the attribution tags alone could not give.

    Explicit tags cover fewer lines than they look like they should. Real prose writes
    "she said" as often as "Mara said", and a character who speaks twice in one turn only
    gets tagged once. Both leave a line unattributed, and an unattributed line falls back
    to the first character's voice — so in a two-hander roughly half the replies come out
    in the wrong voice, which is precisely what makes a scene sound wrong.

    Three rules, applied in order, and all of them restricted to names that were already
    identified by an explicit tag somewhere in the story. That restriction is what makes
    the lookback safe: prose capitalises the first word of every sentence, so searching
    for "a capitalised word" would happily return "Then" or "Father".

    1. The most recent cast name written just before the line. Covers both "Mara set the
       lantern down. '...' she said" and a second line inside a turn already tagged.
    2. Otherwise, if the line continues the same paragraph as the previous spoken line,
       it belongs to whoever was speaking.
    3. Otherwise, in a two-character scene, alternate. A new paragraph of untagged speech
       following one character is the other one replying; that is the convention the
       prompt asks the model to write in.

    Anything still unresolved stays ``None``, which the pipeline reads as "the first
    character's voice". A fallback is unremarkable; a confidently wrong voice is not.
    """
    cast = [segment.speaker for segment in segments if segment.speaker]
    known = dict.fromkeys(cast)  # ordered, de-duplicated
    if not known:
        return segments

    genders = _infer_genders(story, known)

    resolved: list[Segment] = []
    previous_speaker: str | None = None
    previous_dialogue_end: int | None = None

    for segment in segments:
        if segment.kind is not SegmentKind.DIALOGUE:
            resolved.append(segment)
            continue

        speaker = segment.speaker
        if speaker is None:
            speaker = _paragraph_local_name(story, segment.start_char, known)
        if speaker is None:
            speaker = _speaker_by_pronoun(story, segment, genders, known)
        if speaker is None and previous_speaker is not None and previous_dialogue_end is not None:
            between = story[previous_dialogue_end : segment.start_char]
            if "\n\n" in between:
                speaker = next((name for name in known if name != previous_speaker), None)
            else:
                speaker = previous_speaker

        resolved.append(replace(segment, speaker=speaker))
        previous_speaker = speaker or previous_speaker
        previous_dialogue_end = segment.end_char

    return resolved


def _infer_genders(story: str, known: Mapping[str, None]) -> dict[str, str]:
    """Guess how the story refers to each character, from the pronouns near their name.

    Only the *first* pronoun after each mention is counted. Counting every pronoun in a
    window makes each character look like the other one: in a two-hander the prose moves
    from one to the other within a sentence or two, so a wide window around "Tom" is full
    of "she" and "her". The pronoun immediately following the name is the one that refers
    to it.

    A majority across mentions decides; a tie leaves the character unlabelled, and the
    pronoun rule then falls back rather than guessing.
    """
    genders: dict[str, str] = {}

    for name in known:
        masculine = feminine = 0
        position = story.find(name)
        while position >= 0:
            window = story[position + len(name) : position + len(name) + _GENDER_WINDOW]
            male = _MASCULINE.search(window)
            female = _FEMININE.search(window)
            if male and (female is None or male.start() < female.start()):
                masculine += 1
            elif female:
                feminine += 1
            position = story.find(name, position + 1)

        if masculine > feminine:
            genders[name] = "he"
        elif feminine > masculine:
            genders[name] = "she"

    return genders


def _speaker_by_pronoun(
    story: str, segment: Segment, genders: Mapping[str, str], known: Mapping[str, None]
) -> str | None:
    """Resolve "he said" / "she said" when exactly one character matches the pronoun.

    A pronoun tag names nobody, so an earlier version ignored it and fell through to
    alternation — which put a line tagged "he said" in the other character's voice, the
    one case the tag rules out. Used only when the match is unambiguous.
    """

    after = story[segment.end_char : segment.end_char + _ATTRIBUTION_WINDOW]
    found = _PRONOUN_TAG_AFTER.match(after)
    if found is None:
        before = story[max(0, segment.start_char - _ATTRIBUTION_WINDOW) : segment.start_char]
        found = _PRONOUN_TAG_BEFORE.search(before)
    if found is None:
        return None

    # Eliminate rather than match. A pronoun tag says who it is *not*, and that is the
    # stronger signal: it resolves a two-hander even when only one character's pronouns
    # were clear enough to label.
    pronoun = found.group(1).lower()
    candidates = [name for name in known if genders.get(name, pronoun) == pronoun]
    return candidates[0] if len(candidates) == 1 else None


def _paragraph_local_name(story: str, before: int, known: Mapping[str, None]) -> str | None:
    """The last cast name written in this line's own paragraph, if any.

    Restricted to the paragraph on purpose. An earlier draft searched a fixed window of
    characters backwards, which reached into the previous paragraph and confidently
    attributed a reply to the character who had just finished speaking — the one person
    it could not have been.
    """
    window = _paragraph_before(story, before)
    best: tuple[int, str] | None = None
    for name in known:
        position = window.rfind(name)
        if position >= 0 and (best is None or position > best[0]):
            best = (position, name)
    return best[1] if best else None


@dataclass(frozen=True, slots=True)
class _Sentence:
    """One sentence, cleaned for speech, still carrying where it was written."""

    text: str
    start: int
    end: int


def _attribute(story: str, quote_start: int, quote_end: int) -> str | None:
    """Read the speaker's name out of the attribution beside a quoted line.

    Looks after the line first, because that is where English puts the tag most often, and
    only then before it. Returns ``None`` when nothing matches: an unattributed line is
    normal prose (a reply whose speaker is obvious from the exchange), and guessing would
    put a wrong voice in the listener's ear.
    """
    after = story[quote_end : quote_end + _ATTRIBUTION_WINDOW]
    for pattern in (_TAG_AFTER_VERB_FIRST, _TAG_AFTER_NAME_FIRST):
        found = pattern.match(after)
        if found and _is_name(found.group(1)):
            return found.group(1)

    before = story[max(0, quote_start - _ATTRIBUTION_WINDOW) : quote_start]
    found = _TAG_BEFORE.search(before)
    if found and _is_name(found.group(1)):
        return found.group(1)

    # No speech tag. The paragraph may still open with the speaker acting.
    beat = _ACTION_BEAT.match(_paragraph_before(story, quote_start))
    return beat.group(1) if beat and _is_name(beat.group(1)) else None


def _is_name(candidate: str) -> bool:
    """Reject a capitalised word that is capitalised only for starting a sentence."""
    return candidate.split()[0] not in _NOT_A_NAME


def _paragraph_before(story: str, position: int) -> str:
    """The text from the start of ``position``'s paragraph up to it.

    Attribution lives inside one paragraph. Looking further back finds the name of
    whoever was speaking in the *previous* exchange, which is reliably the wrong one.
    """
    break_at = story.rfind("\n\n", 0, position)
    return story[0 if break_at < 0 else break_at + 2 : position]


def _chunk(
    text: str,
    max_chars: int,
    kind: SegmentKind,
    base: int,
    *,
    speaker: str | None = None,
) -> list[Segment]:
    """Group sentences into segments of at most ``max_chars``.

    Splits on sentence boundaries rather than mid-word, so a segment break lands where a
    reader would pause anyway. A single sentence longer than the limit is kept whole: a
    hard cut inside a clause is worse than one long segment, and the engine splits
    further internally.

    Sentences are cut from the *raw* text and cleaned one at a time, rather than cleaning
    the whole fragment and splitting the result. Same spoken output, but the offsets
    survive it: cleaning is not length-preserving, so doing it first would make every
    position afterwards a guess.

    A sentence longer than the limit used to be kept whole, on the reasoning that a hard
    cut inside a clause is worse than one long segment and that the engine would split it
    further anyway. The engine does not: `enable_text_splitting` is off, so XTTS truncates
    such a sentence and only mentions it in a log line. Being cut at a comma is better
    than being cut off mid-word.
    """
    sentences = [
        piece for sentence in _sentences(text, base) for piece in _fit(sentence, max_chars)
    ]
    if not sentences:
        return []

    segments: list[Segment] = []
    current: list[_Sentence] = []
    length = 0

    for sentence in sentences:
        addition = len(sentence.text) + (1 if current else 0)
        if current and length + addition > max_chars:
            segments.append(_join(current, kind, speaker))
            current = [sentence]
            length = len(sentence.text)
        else:
            current.append(sentence)
            length += addition

    if current:
        segments.append(_join(current, kind, speaker))

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


def _fit(sentence: _Sentence, max_chars: int) -> list[_Sentence]:
    """Break one over-long sentence into pieces the model will accept.

    Clause boundaries first, since a break at a comma reads as a breath. Only if a single
    clause is still too long does it fall back to breaking between words, which is
    audible but still better than the silent truncation it replaces.
    """
    if len(sentence.text) <= max_chars:
        return [sentence]

    chunks = _break_text(sentence.text, max_chars)
    total = sum(len(chunk) for chunk in chunks)
    if total == 0:
        return []

    # The sentence's span is divided between the pieces in proportion to their length.
    # It is an approximation -- cleaned text is not written text, so a position inside it
    # cannot be mapped back exactly -- but it must be an *ordered, non-overlapping* one.
    # Giving every piece the whole sentence's span instead reads as overlapping spans
    # downstream, and the read-along drops the highlight for every piece after the first.
    pieces: list[_Sentence] = []
    span = sentence.end - sentence.start
    cursor = sentence.start
    consumed = 0

    for index, chunk in enumerate(chunks):
        consumed += len(chunk)
        last = index == len(chunks) - 1
        end = sentence.end if last else sentence.start + round(span * consumed / total)
        end = max(end, cursor + 1)
        pieces.append(_Sentence(text=chunk, start=cursor, end=min(end, sentence.end)))
        cursor = min(end, sentence.end)

    return pieces


def _break_text(text: str, max_chars: int) -> list[str]:
    """Split ``text`` into runs of at most ``max_chars``, preferring clause boundaries."""
    parts = [part for part in _CLAUSE_BOUNDARY.split(text) if part.strip()]
    chunks: list[str] = []
    current = ""

    for part in parts:
        candidate = f"{current} {part}".strip() if current else part
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = part
        else:
            current = candidate

    if current:
        chunks.append(current)

    # A single clause can still be too long; break it between words as a last resort.
    final: list[str] = []
    for chunk in chunks:
        while len(chunk) > max_chars:
            cut = chunk.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            final.append(chunk[:cut].strip())
            chunk = chunk[cut:].strip()
        if chunk:
            final.append(chunk)

    return final


def _join(sentences: list[_Sentence], kind: SegmentKind, speaker: str | None) -> Segment:
    return Segment(
        kind=kind,
        text=" ".join(sentence.text for sentence in sentences),
        start_char=sentences[0].start,
        end_char=sentences[-1].end,
        speaker=speaker,
    )
