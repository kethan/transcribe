from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any

from .types import ValidationError

SEGMENT_KEYS = {"start", "end", "text", "speaker_id", "words"}
WORD_KEYS = {"text", "start", "end", "type", "speaker_id"}
SPEAKER_RE = re.compile(r"^speaker_\d+$")


def canonical(text: str) -> str:
    return "".join(unicodedata.normalize("NFC", text).split())


def _segment_expected_speaker(words: list[dict[str, Any]]) -> str | None:
    durations: dict[str | None, float] = defaultdict(float)
    for word in words:
        durations[word["speaker_id"]] += word["end"] - word["start"]
    if not durations:
        return None
    if len(durations) == 1:
        return next(iter(durations))
    candidates = [(s, d) for s, d in durations.items() if s is not None]
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[1], item[0]))
    return candidates[0][0]


def validate_output(
    output: dict[str, Any],
    duration: float,
    source_text: str,
    speaker_ids: list[str],
    expected_word_count: int,
    warnings: list[str],
) -> dict[str, Any]:
    """Validate the final JSON before it is written. Returns summary counts."""
    problems: list[str] = []
    if set(output.keys()) != {"segments"}:
        problems.append(f"Top-level keys must be exactly ['segments'], got {sorted(output.keys())}")
    segments = output.get("segments")
    if not isinstance(segments, list):
        raise ValidationError("segments must be a list")
    allowed_speakers = set(speaker_ids)
    all_word_texts: list[str] = []
    unattributed = 0
    word_count = 0
    max_end = 0.0

    for si, segment in enumerate(segments):
        if not isinstance(segment, dict) or set(segment.keys()) != SEGMENT_KEYS:
            problems.append(f"segment {si}: keys must be exactly {sorted(SEGMENT_KEYS)}")
            continue
        words = segment.get("words")
        if not isinstance(words, list) or not words:
            problems.append(f"segment {si}: words must be a non-empty list")
            continue
        seg_start = segment.get("start")
        seg_end = segment.get("end")
        if not all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in (seg_start, seg_end)):
            problems.append(f"segment {si}: start/end must be finite numbers")
            continue
        if not (0 <= float(seg_start) < float(seg_end) <= duration + 0.001):
            problems.append(
                f"segment {si}: invalid interval [{seg_start}, {seg_end}] outside [0, {duration}]"
            )
        if not isinstance(segment.get("text"), str):
            problems.append(f"segment {si}: text must be a string")
        segment_speaker = segment.get("speaker_id")
        if segment_speaker is not None:
            if not isinstance(segment_speaker, str) or not SPEAKER_RE.match(segment_speaker):
                problems.append(f"segment {si}: malformed speaker_id {segment_speaker!r}")
            elif segment_speaker not in allowed_speakers:
                problems.append(f"segment {si}: unknown speaker_id {segment_speaker!r}")

        prev_start = -math.inf
        word_texts: list[str] = []
        for wi, word in enumerate(words):
            if not isinstance(word, dict) or set(word.keys()) != WORD_KEYS:
                problems.append(f"segment {si} word {wi}: keys must be exactly {sorted(WORD_KEYS)}")
                continue
            start, end = word.get("start"), word.get("end")
            if not all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in (start, end)):
                problems.append(f"segment {si} word {wi}: start/end must be finite numbers")
                continue
            start, end = float(start), float(end)
            if not (0 <= start < end <= duration + 0.001):
                problems.append(
                    f"segment {si} word {wi}: invalid interval [{start}, {end}] outside [0, {duration}]"
                )
            if start + 1e-6 < prev_start:
                problems.append(f"segment {si} word {wi}: start times are not chronological")
            prev_start = start
            max_end = max(max_end, end)
            if word.get("type") != "word":
                problems.append(f"segment {si} word {wi}: type must be 'word'")
            text = word.get("text")
            if not isinstance(text, str) or not text.strip():
                problems.append(f"segment {si} word {wi}: text must be a non-empty string")
                continue
            if "\ufffd" in text:
                problems.append(f"segment {si} word {wi}: Unicode replacement character in text")
            word_texts.append(text)
            all_word_texts.append(text)
            speaker = word.get("speaker_id")
            if speaker is None:
                unattributed += 1
            elif not isinstance(speaker, str) or not SPEAKER_RE.match(speaker):
                problems.append(f"segment {si} word {wi}: malformed speaker_id {speaker!r}")
            elif speaker not in allowed_speakers:
                problems.append(f"segment {si} word {wi}: unknown speaker_id {speaker!r}")
            word_count += 1
        if word_texts:
            joined = " ".join(word_texts)
            if segment.get("text") != joined:
                problems.append(f"segment {si}: text is not the concatenated word text")
            expected_speaker = _segment_expected_speaker(words)
            if segment_speaker != expected_speaker:
                problems.append(
                    f"segment {si}: speaker_id {segment_speaker!r} is inconsistent with its words "
                    f"(expected {expected_speaker!r})"
                )
            if float(seg_start) > min(float(w["start"]) for w in words) + 0.001:
                problems.append(f"segment {si}: start is later than its first word")
            if float(seg_end) + 0.001 < max(float(w["end"]) for w in words):
                problems.append(f"segment {si}: end is earlier than its last word")

    source_canon = canonical(source_text)
    output_canon = canonical("".join(all_word_texts))
    order_changed = False
    if source_canon != output_canon:
        if Counter(source_canon) == Counter(output_canon):
            order_changed = True
            warnings.append(
                "Final word characters match Bodhan exactly but their order changed when "
                "segments were sorted onto the common timeline (overlapping speech)."
            )
        else:
            missing = Counter(source_canon) - Counter(output_canon)
            extra = Counter(output_canon) - Counter(source_canon)
            problems.append(
                "Transcript characters were dropped or added: "
                f"missing={dict(missing.most_common(8))}, extra={dict(extra.most_common(8))}"
            )
    if "\ufffd" in source_canon or "\ufffd" in output_canon:
        problems.append("Unicode replacement character U+FFFD detected in transcript")
    if word_count != expected_word_count:
        warnings.append(
            f"Output has {word_count} word objects vs {expected_word_count} Bodhan whitespace tokens; "
            "this is caused only by explicitly reported merges of unalignable tokens."
        )
    if max_end > duration + 0.001:
        problems.append(f"Word end {max_end} exceeds audio duration {duration}")

    if problems:
        raise ValidationError("Final JSON validation failed:\n- " + "\n- ".join(problems[:30]))
    return {
        "segments": len(segments),
        "words": word_count,
        "words_without_speaker": unattributed,
        "expected_bodhan_word_tokens": expected_word_count,
        "text_order_changed": order_changed,
        "max_word_end": max_end,
    }
