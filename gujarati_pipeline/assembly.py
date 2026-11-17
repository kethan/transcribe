from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .config import Config
from .types import AlignedWord, DiarInterval

SENTENCE_END_CHARS = set(".?!।॥…")
TRAILING_QUOTES = set("\"'”’)]}›»")


def q3(value: float) -> float:
    return round(float(value), 3)


def merge_intervals_by_speaker(intervals: list[DiarInterval]) -> dict[str, list[tuple[float, float]]]:
    """Union intervals per speaker so overlap is never double-counted."""
    by_speaker: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for interval in intervals:
        by_speaker[interval.speaker_id].append((interval.start, interval.end))
    merged: dict[str, list[tuple[float, float]]] = {}
    for speaker, spans in by_speaker.items():
        spans.sort(key=lambda x: (x[0], x[1]))
        out: list[tuple[float, float]] = []
        for start, end in spans:
            if out and start <= out[-1][1]:
                out[-1] = (out[-1][0], max(out[-1][1], end))
            else:
                out.append((start, end))
        merged[speaker] = out
    return merged


@dataclass
class SpeakerStats:
    no_overlap_words: int = 0
    uncertain_words: int = 0
    assigned_words: int = 0
    unattributed_words: int = 0


def assign_speakers(
    words: list[AlignedWord],
    intervals: list[DiarInterval],
    config: Config,
    warnings: list[str],
) -> SpeakerStats:
    """Deterministic greatest-overlap assignment with explicit uncertainty -> null.

    For each word, overlap is summed over the union of each speaker's intervals.
    Candidates are ranked by (-overlap, earliest interval start, speaker_id). A
    word is assigned only when the winner covers at least
    ``speaker_min_coverage`` of the word and the runner-up is not within
    ``speaker_ambiguity_margin`` of the winner. Otherwise ``speaker_id=null``.
    """
    merged = merge_intervals_by_speaker(intervals)
    stats = SpeakerStats()
    for word in words:
        duration = word.end - word.start
        if not (math.isfinite(duration) and duration > 0):
            raise ValueError(f"Invalid word duration for {word.text!r}: [{word.start}, {word.end}]")
        overlaps: dict[str, tuple[float, float]] = {}
        for speaker, spans in merged.items():
            total = 0.0
            first_start = math.inf
            for start, end in spans:
                overlap = max(0.0, min(word.end, end) - max(word.start, start))
                if overlap > 0:
                    total += overlap
                    first_start = min(first_start, start)
            if total > 0:
                overlaps[speaker] = (total, first_start)
        if not overlaps:
            word.speaker_id = None
            stats.no_overlap_words += 1
            stats.unattributed_words += 1
            continue
        ranked = sorted(
            overlaps.items(),
            key=lambda item: (-item[1][0], item[1][1], item[0]),
        )
        winner, (best_overlap, _first) = ranked[0]
        second_overlap = ranked[1][1][0] if len(ranked) > 1 else 0.0
        coverage = best_overlap / duration
        ambiguous = second_overlap >= best_overlap * (1.0 - config.speaker_ambiguity_margin)
        if coverage < config.speaker_min_coverage or ambiguous:
            word.speaker_id = None
            stats.uncertain_words += 1
            stats.unattributed_words += 1
        else:
            word.speaker_id = winner
            stats.assigned_words += 1
    if stats.no_overlap_words:
        warnings.append(
            f"{stats.no_overlap_words} word(s) had no NeMo interval overlap; speaker_id=null."
        )
    if stats.uncertain_words:
        warnings.append(
            f"{stats.uncertain_words} word(s) had insufficient or ambiguous speaker overlap; speaker_id=null."
        )
    return stats


def is_sentence_end(text: str) -> bool:
    stripped = text.rstrip()
    while stripped and stripped[-1] in TRAILING_QUOTES:
        stripped = stripped[:-1].rstrip()
    return bool(stripped) and stripped[-1] in SENTENCE_END_CHARS


def deterministic_segment_speaker(words: list[AlignedWord]) -> str | None:
    durations: dict[str | None, float] = defaultdict(float)
    for word in words:
        durations[word.speaker_id] += word.end - word.start
    if not durations:
        return None
    if len(durations) == 1:
        return next(iter(durations))
    candidates = [(s, d) for s, d in durations.items() if s is not None]
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[1], item[0]))
    return candidates[0][0]


def _segment_from_words(words: list[AlignedWord]) -> dict[str, Any]:
    start = min(w.start for w in words)
    end = max(w.end for w in words)
    text = " ".join(w.text for w in words)
    return {
        "start": q3(start),
        "end": q3(end),
        "text": text,
        "speaker_id": deterministic_segment_speaker(words),
        "words": [
            {
                "text": w.text,
                "start": q3(w.start),
                "end": q3(w.end),
                "type": "word",
                "speaker_id": w.speaker_id,
            }
            for w in words
        ],
    }


def _group_by_speaker_turn(ordered: list[AlignedWord]) -> list[dict[str, Any]]:
    """One segment per consecutive speaker run, with no duration/pause splitting.

    A run with ``speaker_id=None`` is kept as its own unattributed segment; no
    speaker label is invented for words that diarization could not assign.
    """
    segments: list[dict[str, Any]] = []
    current: list[AlignedWord] = []
    for word in ordered:
        if current and word.speaker_id != current[-1].speaker_id:
            segments.append(_segment_from_words(current))
            current = []
        current.append(word)
    if current:
        segments.append(_segment_from_words(current))
    return segments


def _group_legacy(
    ordered: list[AlignedWord], config: Config
) -> list[dict[str, Any]]:
    """Original finer-grained grouping with pause/sentence/chunk boundaries."""
    segments: list[dict[str, Any]] = []
    current: list[AlignedWord] = []
    for word in ordered:
        if not current:
            current = [word]
            continue
        prev = current[-1]
        gap = word.start - prev.end
        speaker_changed = word.speaker_id != prev.speaker_id
        chunk_changed = word.chunk_key != prev.chunk_key
        unit_changed = (
            word.chunk_index != prev.chunk_index or word.unit_index != prev.unit_index
        )
        meaningful_pause = gap >= config.pause_threshold
        distinct_utterance = is_sentence_end(prev.text)
        too_long = (word.end - current[0].start) > config.max_segment_seconds
        if (
            speaker_changed
            or chunk_changed
            or unit_changed
            or meaningful_pause
            or distinct_utterance
            or too_long
        ):
            segments.append(_segment_from_words(current))
            current = [word]
        else:
            current.append(word)
    if current:
        segments.append(_segment_from_words(current))
    return segments


def group_segments(
    words: list[AlignedWord],
    config: Config,
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Group consecutive words; never create one segment per word.

    The default ``speaker-turn`` mode matches the ElevenLabs reference shape:
    each segment's ``text`` holds the full text of one speaker before the next
    speaker's segment begins. ``legacy`` retains the earlier short-segment rules.
    """
    ordered = sorted(words, key=lambda w: (w.start, w.end, w.original_order))
    if [w.original_order for w in ordered] != [w.original_order for w in words]:
        warnings.append(
            "Words were reordered into chronological timeline order; overlapping-speech "
            "segments may therefore differ from Bodhan's textual order."
        )
    if config.segmentation_mode == "speaker-turn":
        return _group_by_speaker_turn(ordered)
    return _group_legacy(ordered, config)


@dataclass
class AssemblyStats:
    word_count: int
    segment_count: int
    unattributed_words: int
    no_overlap_words: int
    uncertain_words: int


def build_output(
    words: list[AlignedWord],
    intervals: list[DiarInterval],
    config: Config,
    warnings: list[str],
) -> tuple[dict[str, Any], SpeakerStats, AssemblyStats]:
    speaker_stats = assign_speakers(words, intervals, config, warnings)
    segments = group_segments(words, config, warnings)
    output = {"segments": segments}
    assembly_stats = AssemblyStats(
        word_count=len(words),
        segment_count=len(segments),
        unattributed_words=speaker_stats.unattributed_words,
        no_overlap_words=speaker_stats.no_overlap_words,
        uncertain_words=speaker_stats.uncertain_words,
    )
    return output, speaker_stats, assembly_stats
