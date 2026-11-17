from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tools.scribe2_reference import DEFAULT_REFERENCE, load_json, write_json

NOTE = (
    "Speaker-turn view of the attached ElevenLabs Scribe 2 JSON. This is a "
    "reference artifact, not Bodhan/NeMo/CTC pipeline output and not the strict "
    "final transcript schema. Original word objects (including spacing, "
    "audio_event, logprob, characters and channel_index) are preserved verbatim."
)


def _speaker(obj: dict[str, Any]) -> str | None:
    value = obj.get("speaker_id")
    return value if isinstance(value, str) else None


def _times(obj: dict[str, Any]) -> tuple[float | None, float | None]:
    try:
        start = float(obj["start"])
        end = float(obj["end"])
    except (KeyError, TypeError, ValueError):
        return None, None
    if start != start or end != end:  # NaN
        return None, None
    return start, end


def _turn_speaker(words: list[dict[str, Any]], current: str | None) -> str | None:
    if current is not None:
        return current
    durations: dict[str, float] = defaultdict(float)
    for obj in words:
        speaker = _speaker(obj)
        if speaker is None or obj.get("type") == "spacing":
            continue
        start, end = _times(obj)
        durations[speaker] += max(0.0, (end or 0.0) - (start or 0.0))
    if not durations:
        return None
    return sorted(durations, key=lambda speaker: (-durations[speaker], speaker))[0]


def _make_turn(words: list[dict[str, Any]], current: str | None) -> dict[str, Any]:
    starts: list[float] = []
    ends: list[float] = []
    for obj in words:
        start, end = _times(obj)
        if start is not None:
            starts.append(start)
        if end is not None:
            ends.append(end)
    return {
        "start": min(starts) if starts else None,
        "end": max(ends) if ends else None,
        "text": "".join(str(obj.get("text", "")) for obj in words),
        "speaker_id": _turn_speaker(words, current),
        "words": words,
    }


def convert_speaker_turns(
    reference: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Group Scribe 2 objects into consecutive full speaker turns.

    A change in a non-spacing object's ``speaker_id`` starts the next turn;
    spacing objects stay with the turn they punctuate. No source object is
    dropped, reordered, retimed or given a new label.
    """
    if set(reference.keys()) != {"segments"} or not isinstance(reference["segments"], list):
        raise ValueError("Reference must be an object with exactly one key, 'segments', containing a list.")
    flat: list[dict[str, Any]] = []
    source_segment_count = len(reference["segments"])
    for si, segment in enumerate(reference["segments"]):
        if not isinstance(segment, dict):
            raise ValueError(f"Reference segment {si} is not an object.")
        words = segment.get("words")
        if not isinstance(words, list):
            raise ValueError(f"Reference segment {si} has no words list.")
        for wi, obj in enumerate(words):
            if not isinstance(obj, dict):
                raise ValueError(f"Reference segment {si} word {wi} is not an object.")
            flat.append(obj)

    turns: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_speaker: str | None = None
    for obj in flat:
        if obj.get("type") != "spacing":
            speaker = _speaker(obj)
            if current and speaker != current_speaker:
                turns.append(_make_turn(current, current_speaker))
                current = []
            current_speaker = speaker
        current.append(obj)
    if current:
        turns.append(_make_turn(current, current_speaker))

    source_text = "".join(str(obj.get("text", "")) for obj in flat)
    output_text = "".join(turn["text"] for turn in turns)
    all_starts: list[float] = []
    all_ends: list[float] = []
    for obj in flat:
        start, end = _times(obj)
        if start is not None:
            all_starts.append(start)
        if end is not None:
            all_ends.append(end)
    speakers = sorted({_speaker(obj) for obj in flat if _speaker(obj) is not None})
    stats: dict[str, Any] = {
        "note": NOTE,
        "source_segments": source_segment_count,
        "speaker_turn_segments": len(turns),
        "source_word_objects": len(flat),
        "output_word_objects": sum(len(turn["words"]) for turn in turns),
        "word_object_type_counts": dict(Counter(str(obj.get("type")) for obj in flat)),
        "speaker_ids": speakers,
        "source_text_matches_output_text": source_text == output_text,
        "source_text_characters": len(source_text),
        "output_text_characters": len(output_text),
        "min_word_start_seconds": min(all_starts) if all_starts else None,
        "max_word_end_seconds": max(all_ends) if all_ends else None,
        "segments_with_null_speaker": sum(1 for turn in turns if turn["speaker_id"] is None),
    }
    return {"segments": turns}, stats


def markdown_report(stats: dict[str, Any], output_path: Path) -> str:
    lines = [
        "# ElevenLabs Scribe 2 speaker-turn reference",
        "",
        stats["note"],
        "",
        f"- **Output:** `{output_path}`",
        f"- **Source reference segments:** {stats['source_segments']}",
        f"- **Speaker-turn segments:** {stats['speaker_turn_segments']}",
        f"- **Word objects preserved:** {stats['output_word_objects']} / {stats['source_word_objects']}",
        f"- **Speaker IDs:** {', '.join(stats['speaker_ids']) or 'none'}",
        f"- **Segments with null speaker:** {stats['segments_with_null_speaker']}",
        f"- **Exact source text preserved:** {stats['source_text_matches_output_text']}",
        f"- **Time range:** {stats['min_word_start_seconds']}–{stats['max_word_end_seconds']} s",
        "",
        "## Source object counts",
        "",
    ]
    lines.extend(
        f"- `{kind}`: {count}"
        for kind, count in sorted(stats["word_object_type_counts"].items())
    )
    lines.extend(
        [
            "",
            "## Schema distinction",
            "",
            "This JSON intentionally keeps ElevenLabs-specific word fields and "
            "`spacing`/`audio_event` objects. The pipeline's final `transcript.json` "
            "remains strict: only `text`, `start`, `end`, `type` and `speaker_id` "
            "are allowed on words, with `type: \"word\"`. The pipeline now uses "
            "speaker-turn grouping for its segments, but does not copy ElevenLabs "
            "text into the final output.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.scribe2_speaker_turns",
        description=(
            "Create a downloadable speaker-turn reference from the attached "
            "ElevenLabs Scribe 2 JSON, preserving all original word fields."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=Path("outputs/scribe2_speaker_turns.json"))
    parser.add_argument(
        "--report-md", type=Path, default=Path("outputs/scribe2_speaker_turns_report.md")
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        reference = load_json(args.reference)
        turns, stats = convert_speaker_turns(reference)
        write_json(args.output, turns)
        args.report_md.parent.mkdir(parents=True, exist_ok=True)
        args.report_md.write_text(
            markdown_report(stats, args.output),
            encoding="utf-8",
        )
        print(f"Wrote Scribe 2 speaker-turn reference: {args.output}")
        print(f"Wrote report: {args.report_md}")
        print(
            f"Segments={stats['speaker_turn_segments']}, "
            f"word_objects={stats['output_word_objects']}, "
            f"speakers={len(stats['speaker_ids'])}, "
            f"text_preserved={stats['source_text_matches_output_text']}"
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
