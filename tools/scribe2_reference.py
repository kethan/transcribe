from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from gujarati_pipeline.types import ValidationError
from gujarati_pipeline.validation import validate_output

DEFAULT_REFERENCE = Path("audio/20261031 CHD 3 GTH 11 12 13 14 15 (51) CHN.json")
DEFAULT_AUDIO = Path("audio/20261031 CHD 3 GTH 11 12 13 14 15 (51) CHN.mp3")
WORD_KEYS = {"text", "start", "end", "type", "speaker_id"}
SEGMENT_KEYS = {"start", "end", "text", "speaker_id", "words"}


def canonical(text: str) -> str:
    return "".join(unicodedata.normalize("NFC", text).split())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def audio_duration(path: Path, override: float | None = None) -> float:
    """Return MP3 duration. mutagen is optional for this offline tool only."""
    if override is not None:
        return float(override)
    try:
        from mutagen.mp3 import MP3
    except ImportError as exc:  # pragma: no cover - optional developer dependency
        raise RuntimeError(
            "mutagen is needed to read MP3 duration automatically. Install it with "
            "`pip install mutagen`, or pass --duration SECONDS."
        ) from exc
    if not path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {path}")
    return float(MP3(str(path)).info.length)


def _next_real_end(
    flat_entries: list[dict[str, Any]],
    index: int,
    start: float,
    duration: float,
) -> tuple[float | None, int | None, bool]:
    """Borrow the next real interval boundary from the full Scribe 2 response.

    Scribe 2 occasionally emits a zero-duration word. We never invent a time:
    the replacement end is an actual ``end`` value from a later object, or the
    measured audio duration as a last-resort real boundary. The search is global
    because a zero-duration token can be the last object in its segment.
    """
    for j in range(index + 1, len(flat_entries)):
        try:
            end = float(flat_entries[j]["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(end) and end > start + 1e-9:
            return end, j, False
    if math.isfinite(duration) and duration > start + 1e-9:
        return duration, None, True
    return None, None, False


def _segment_expected_speaker(words: list[dict[str, Any]]) -> str | None:
    durations: dict[str | None, float] = defaultdict(float)
    for word in words:
        durations[word["speaker_id"]] += float(word["end"]) - float(word["start"])
    candidates = [(s, d) for s, d in durations.items() if s is not None]
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[1], item[0]))
    return candidates[0][0]


def convert_reference(
    reference: dict[str, Any],
    duration: float,
    *,
    include_audio_events: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert an ElevenLabs Scribe 2 sample into the project's strict schema.

    Spacing objects are dropped. ``word`` objects are retained verbatim except
    for exact key filtering. ``audio_event`` objects are retained as
    ``type: "word"`` by default because the strict schema permits no other type;
    this is a schema conversion, not a claim that they are lexical words.
    """
    if set(reference.keys()) != {"segments"} or not isinstance(reference["segments"], list):
        raise ValueError("Reference must be an object with exactly one key, 'segments', containing a list.")
    segments: list[dict[str, Any]] = []
    all_texts: list[str] = []
    warnings: list[str] = []
    stats: dict[str, Any] = {
        "audio_duration_seconds": duration,
        "reference_segments": len(reference["segments"]),
        "include_audio_events": include_audio_events,
        "dropped_spacing_objects": 0,
        "dropped_audio_event_objects": 0,
        "mapped_audio_events_as_words": 0,
        "zero_duration_tokens_repaired": 0,
        "zero_duration_tokens_repaired_from_audio_duration": 0,
        "zero_duration_tokens_dropped": 0,
        "repair_examples": [],
        "dropped_unknown_type_objects": 0,
        "segments_without_words": 0,
    }
    type_counts: Counter[str] = Counter()
    speaker_set: set[str] = set()
    flat_entries: list[dict[str, Any]] = []
    global_index: dict[tuple[int, int], int] = {}
    for si, source_segment in enumerate(reference["segments"]):
        for wi, obj in enumerate(source_segment.get("words", [])):
            global_index[(si, wi)] = len(flat_entries)
            flat_entries.append(obj if isinstance(obj, dict) else {})

    for si, source_segment in enumerate(reference["segments"]):
        source_words = source_segment.get("words")
        if not isinstance(source_words, list):
            raise ValueError(f"Reference segment {si} has no words list.")
        selected: list[dict[str, Any]] = []
        for wi, source_word in enumerate(source_words):
            if not isinstance(source_word, dict):
                stats["dropped_unknown_type_objects"] += 1
                continue
            source_type = source_word.get("type")
            type_counts[str(source_type)] += 1
            if source_type == "spacing":
                stats["dropped_spacing_objects"] += 1
                continue
            if source_type == "audio_event":
                if not include_audio_events:
                    stats["dropped_audio_event_objects"] += 1
                    continue
                stats["mapped_audio_events_as_words"] += 1
            elif source_type != "word":
                stats["dropped_unknown_type_objects"] += 1
                warnings.append(
                    f"segment {si} word {wi}: dropped unknown Scribe 2 type {source_type!r}."
                )
                continue
            try:
                start = float(source_word["start"])
                end = float(source_word["end"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"segment {si} word {wi}: invalid start/end.") from exc
            if not (math.isfinite(start) and math.isfinite(end)):
                raise ValueError(f"segment {si} word {wi}: non-finite start/end.")
            if end <= start:
                borrowed, source_index, from_audio_duration = _next_real_end(
                    flat_entries, global_index[(si, wi)], start, duration
                )
                if borrowed is None or borrowed <= start:
                    stats["zero_duration_tokens_dropped"] += 1
                    warnings.append(
                        f"segment {si} word {wi}: dropped zero-duration token "
                        f"{source_word.get('text')!r}; no later real interval boundary exists."
                    )
                    continue
                end = min(borrowed, duration)
                stats["zero_duration_tokens_repaired"] += 1
                if from_audio_duration:
                    stats["zero_duration_tokens_repaired_from_audio_duration"] += 1
                if len(stats["repair_examples"]) < 30:
                    stats["repair_examples"].append(
                        {
                            "segment": si,
                            "word_index": wi,
                            "text": source_word.get("text"),
                            "start": start,
                            "borrowed_end": end,
                            "boundary_source_global_index": source_index,
                            "boundary_from_audio_duration": from_audio_duration,
                        }
                    )
            if not (0 <= start < end <= duration + 0.001):
                raise ValueError(
                    f"segment {si} word {wi}: interval [{start}, {end}] outside audio duration {duration}."
                )
            text = source_word.get("text")
            if not isinstance(text, str) or not text.strip():
                warnings.append(f"segment {si} word {wi}: dropped empty token text.")
                continue
            speaker = source_word.get("speaker_id")
            if speaker is not None:
                if not isinstance(speaker, str):
                    raise ValueError(f"segment {si} word {wi}: malformed speaker_id {speaker!r}.")
                speaker_set.add(speaker)
            selected.append(
                {
                    "text": text,
                    "start": start,
                    "end": end,
                    "type": "word",
                    "speaker_id": speaker,
                }
            )
        if not selected:
            stats["segments_without_words"] += 1
            warnings.append(f"segment {si}: no non-spacing tokens; segment omitted.")
            continue
        starts = [w["start"] for w in selected]
        if any(b + 1e-6 < a for a, b in zip(starts, starts[1:])):
            raise ValueError(f"segment {si}: non-chronological source word starts.")
        word_texts = [w["text"] for w in selected]
        segments.append(
            {
                "start": min(starts),
                "end": max(w["end"] for w in selected),
                "text": " ".join(word_texts),
                "speaker_id": _segment_expected_speaker(selected),
                "words": selected,
            }
        )
        all_texts.extend(word_texts)

    output = {"segments": segments}
    source_text = " ".join(all_texts)
    speaker_ids = sorted(speaker_set)
    validation_warnings: list[str] = []
    summary = validate_output(
        output,
        duration=duration,
        source_text=source_text,
        speaker_ids=speaker_ids,
        expected_word_count=len(all_texts),
        warnings=validation_warnings,
    )
    stats.update(
        {
            "reference_word_object_type_counts": dict(type_counts),
            "strict_segments": len(segments),
            "strict_words": len(all_texts),
            "speaker_ids": speaker_ids,
            "max_word_end_seconds": max((w["end"] for s in segments for w in s["words"]), default=0.0),
            "reference_canonical_matches_converted": canonical(
                " ".join(str(s.get("text", "")) for s in reference["segments"])
            )
            == canonical(source_text),
            "validation": summary,
            "warnings": warnings + validation_warnings,
        }
    )
    return output, stats


def strict_word_rows(output: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segment in output.get("segments", []):
        for word in segment.get("words", []):
            rows.append(
                {
                    "text": word["text"],
                    "start": float(word["start"]),
                    "end": float(word["end"]),
                    "type": word["type"],
                    "speaker_id": word["speaker_id"],
                }
            )
    return rows


def validate_candidate(candidate: dict[str, Any], duration: float) -> dict[str, Any]:
    rows = strict_word_rows(candidate)
    source_text = " ".join(r["text"] for r in rows)
    speaker_ids = sorted({r["speaker_id"] for r in rows if r["speaker_id"] is not None})
    warnings: list[str] = []
    try:
        summary = validate_output(
            candidate,
            duration=duration,
            source_text=source_text,
            speaker_ids=speaker_ids,
            expected_word_count=len(rows),
            warnings=warnings,
        )
    except ValidationError as exc:
        return {"ok": False, "error": str(exc), "warnings": warnings}
    return {"ok": True, "summary": summary, "warnings": warnings}


def _numeric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "max": None}
    return {
        "count": len(values),
        "mean_seconds": round(statistics.fmean(values), 6),
        "median_seconds": round(statistics.median(values), 6),
        "max_seconds": round(max(values), 6),
    }


def compare_candidate(
    candidate: dict[str, Any],
    reference_strict: dict[str, Any],
    duration: float,
) -> dict[str, Any]:
    ref_rows = strict_word_rows(reference_strict)
    cand_rows = strict_word_rows(candidate)
    ref_keys = [canonical(r["text"]) for r in ref_rows]
    cand_keys = [canonical(r["text"]) for r in cand_rows]
    matcher = SequenceMatcher(None, ref_keys, cand_keys, autojunk=False)
    blocks = [b for b in matcher.get_matching_blocks() if b.size]
    matched = sum(b.size for b in blocks)
    start_deltas: list[float] = []
    end_deltas: list[float] = []
    speaker_checked = 0
    speaker_agreed = 0
    candidate_null_speakers_on_matches = 0
    for block in blocks:
        for k in range(block.size):
            ref = ref_rows[block.a + k]
            cand = cand_rows[block.b + k]
            start_deltas.append(abs(ref["start"] - cand["start"]))
            end_deltas.append(abs(ref["end"] - cand["end"]))
            if cand["speaker_id"] is None:
                candidate_null_speakers_on_matches += 1
            else:
                speaker_checked += 1
                if ref["speaker_id"] == cand["speaker_id"]:
                    speaker_agreed += 1
    ref_canon = canonical("".join(r["text"] for r in ref_rows))
    cand_canon = canonical("".join(r["text"] for r in cand_rows))
    missing = Counter(ref_canon) - Counter(cand_canon)
    extra = Counter(cand_canon) - Counter(ref_canon)
    ref_speakers = sorted({r["speaker_id"] for r in ref_rows if r["speaker_id"] is not None})
    cand_speakers = sorted({r["speaker_id"] for r in cand_rows if r["speaker_id"] is not None})
    return {
        "reference": {
            "provenance": "ElevenLabs Scribe 2 sample JSON attached with the MP3",
            "segments": len(reference_strict["segments"]),
            "words": len(ref_rows),
            "speaker_ids": ref_speakers,
        },
        "candidate": {
            "segments": len(candidate.get("segments", [])),
            "words": len(cand_rows),
            "speaker_ids": cand_speakers,
            "words_without_speaker": sum(1 for r in cand_rows if r["speaker_id"] is None),
            "schema_validation": validate_candidate(candidate, duration),
        },
        "comparison": {
            "canonical_transcripts_identical": ref_canon == cand_canon,
            "word_sequence_identical": ref_keys == cand_keys,
            "word_sequence_similarity": round(
                SequenceMatcher(None, ref_keys, cand_keys, autojunk=False).ratio(), 6
            ),
            "matched_words": matched,
            "matched_word_ratio_of_reference": round(matched / max(1, len(ref_rows)), 6),
            "missing_character_count": sum(missing.values()),
            "extra_character_count": sum(extra.values()),
            "missing_character_examples": dict(missing.most_common(8)),
            "extra_character_examples": dict(extra.most_common(8)),
            "matched_word_start_delta": _numeric_summary(start_deltas),
            "matched_word_end_delta": _numeric_summary(end_deltas),
            "speaker_agreement_on_matched_words": {
                "checked": speaker_checked,
                "agreed": speaker_agreed,
                "ratio": round(speaker_agreed / speaker_checked, 6) if speaker_checked else None,
                "candidate_null_speakers_on_matches": candidate_null_speakers_on_matches,
            },
        },
        "note": (
            "This compares a candidate JSON with a strict-schema conversion of the "
            "attached ElevenLabs Scribe 2 sample. The sample is a reference, not an "
            "independent ground-truth accuracy certification."
        ),
    }


def conversion_markdown(stats: dict[str, Any], output_path: Path) -> str:
    validation = stats["validation"]
    lines = [
        "# ElevenLabs Scribe 2 strict-schema conversion report",
        "",
        "This JSON is a **strict-schema conversion of the attached ElevenLabs Scribe 2 "
        "sample**, not a Bodhan/NeMo/CTC pipeline run and not an independent ASR result.",
        "",
        f"- **Audio duration:** {stats['audio_duration_seconds']:.6f} s",
        f"- **Source reference segments:** {stats['reference_segments']}",
        f"- **Converted segments:** {stats['strict_segments']}",
        f"- **Converted word objects:** {stats['strict_words']}",
        f"- **Detected speaker IDs:** {', '.join(stats['speaker_ids']) or 'none'}",
        f"- **Words without speaker attribution:** {validation['words_without_speaker']}",
        f"- **Max word end:** {stats['max_word_end_seconds']:.6f} s",
        f"- **Output:** `{output_path}`",
        "",
        "## Source object counts",
        "",
    ]
    for key, value in sorted(stats["reference_word_object_type_counts"].items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## Conversion decisions",
            "",
            f"- Dropped spacing objects: {stats['dropped_spacing_objects']}",
            f"- Dropped audio-event objects: {stats['dropped_audio_event_objects']}",
            f"- Mapped audio-event objects to `type: \"word\"`: {stats['mapped_audio_events_as_words']}",
            f"- Zero-duration tokens repaired from the next real source interval boundary: "
            f"{stats['zero_duration_tokens_repaired']}",
            f"- Repairs that used measured audio duration as the boundary: "
            f"{stats['zero_duration_tokens_repaired_from_audio_duration']}",
            f"- Zero-duration tokens dropped: {stats['zero_duration_tokens_dropped']}",
            f"- Source canonical text preserved: {stats['reference_canonical_matches_converted']}",
            "",
            "## Strict validation",
            "",
            f"- Segments: {validation['segments']}",
            f"- Words: {validation['words']}",
            f"- Expected word tokens: {validation['expected_bodhan_word_tokens']}",
            f"- Text order changed: {validation['text_order_changed']}",
            "",
            "## Warnings",
            "",
        ]
    )
    warnings = stats["warnings"]
    if warnings:
        lines.extend(f"- {w}" for w in warnings[:30])
    else:
        lines.append("- No conversion warnings.")
    lines.extend(
        [
            "",
            "## Limitation",
            "",
            "The MP3 was measured for its duration, but no ASR, diarization or forced "
            "alignment model was executed to create this JSON. It exists to verify a "
            "future pipeline output against the attached Scribe 2 sample.",
            "",
        ]
    )
    return "\n".join(lines)


def verification_markdown(report: dict[str, Any]) -> str:
    ref = report["reference"]
    cand = report["candidate"]
    comp = report["comparison"]
    schema = cand["schema_validation"]
    speakers = comp["speaker_agreement_on_matched_words"]
    lines = [
        "# Scribe 2 reference verification",
        "",
        report["note"],
        "",
        f"- **Reference:** {ref['provenance']}",
        f"- **Reference segments / words:** {ref['segments']} / {ref['words']}",
        f"- **Reference speaker IDs:** {', '.join(ref['speaker_ids']) or 'none'}",
        f"- **Candidate segments / words:** {cand['segments']} / {cand['words']}",
        f"- **Candidate words without speaker:** {cand['words_without_speaker']}",
        f"- **Candidate schema valid:** {schema['ok']}",
        "",
        "## Comparison",
        "",
        f"- Canonical transcripts identical: **{comp['canonical_transcripts_identical']}**",
        f"- Word sequence identical: **{comp['word_sequence_identical']}**",
        f"- Word sequence similarity: {comp['word_sequence_similarity']}",
        f"- Matched words: {comp['matched_words']} ({comp['matched_word_ratio_of_reference']} of reference)",
        f"- Missing / extra canonical characters: {comp['missing_character_count']} / {comp['extra_character_count']}",
        f"- Matched-word start deltas: {comp['matched_word_start_delta']}",
        f"- Matched-word end deltas: {comp['matched_word_end_delta']}",
        f"- Speaker agreement on matched words: {speakers['agreed']}/{speakers['checked']} "
        f"(ratio={speakers['ratio']}, candidate null={speakers['candidate_null_speakers_on_matches']})",
        "",
        "## Candidate schema validation",
        "",
    ]
    if schema["ok"]:
        lines.append(f"- Summary: `{schema['summary']}`")
    else:
        lines.append(f"- FAILED: {schema['error']}")
    if schema.get("warnings"):
        lines.extend(f"- Warning: {w}" for w in schema["warnings"][:20])
    lines.extend(["", "## Interpretation", ""])
    if comp["canonical_transcripts_identical"] and comp["word_sequence_identical"]:
        lines.append(
            "The candidate word sequence matches the Scribe 2 reference exactly. Timestamp "
            "and speaker deltas above quantify how closely the two JSONs agree."
        )
    else:
        lines.append(
            "The candidate differs from the Scribe 2 reference. Use the character and word "
            "differences above as a diagnostic; they do not by themselves prove either "
            "transcript is correct."
        )
    lines.append("")
    return "\n".join(lines)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(data).__name__}.")
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.scribe2_reference",
        description=(
            "Convert the attached ElevenLabs Scribe 2 sample to the pipeline's strict "
            "JSON schema, and later verify a pipeline JSON against it."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    convert = sub.add_parser("convert", help="Create a strict-schema reference JSON from the Scribe 2 sample.")
    convert.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    convert.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    convert.add_argument("--duration", type=float, default=None, help="Override MP3 duration in seconds.")
    convert.add_argument("--output", type=Path, default=Path("outputs/scribe2_reference_strict.json"))
    convert.add_argument("--report-json", type=Path, default=Path("outputs/scribe2_conversion_report.json"))
    convert.add_argument("--report-md", type=Path, default=Path("outputs/scribe2_conversion_report.md"))
    convert.add_argument(
        "--drop-audio-events",
        action="store_true",
        help="Exclude audio_event tokens instead of mapping them to type=word.",
    )

    verify = sub.add_parser("verify", help="Compare a future pipeline JSON against the Scribe 2 reference.")
    verify.add_argument("--candidate", type=Path, default=Path("outputs/transcript.json"))
    verify.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    verify.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    verify.add_argument("--duration", type=float, default=None, help="Override MP3 duration in seconds.")
    verify.add_argument("--report-json", type=Path, default=Path("outputs/scribe2_verification_report.json"))
    verify.add_argument("--report-md", type=Path, default=Path("outputs/scribe2_verification_report.md"))
    verify.add_argument(
        "--drop-audio-events",
        action="store_true",
        help="Exclude audio_event tokens from the strict reference conversion.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        duration = audio_duration(args.audio, args.duration)
        reference = load_json(args.reference)
        reference_strict, conversion_stats = convert_reference(
            reference,
            duration,
            include_audio_events=not args.drop_audio_events,
        )
        if args.command == "convert":
            write_json(args.output, reference_strict)
            write_json(args.report_json, conversion_stats)
            args.report_md.parent.mkdir(parents=True, exist_ok=True)
            args.report_md.write_text(
                conversion_markdown(conversion_stats, args.output),
                encoding="utf-8",
            )
            print(f"Wrote strict reference JSON: {args.output}")
            print(f"Wrote conversion reports: {args.report_json}, {args.report_md}")
            print(
                f"Segments={conversion_stats['strict_segments']}, "
                f"words={conversion_stats['strict_words']}, "
                f"speakers={len(conversion_stats['speaker_ids'])}, "
                f"duration={duration:.6f}s"
            )
            return 0
        candidate = load_json(args.candidate)
        report = compare_candidate(candidate, reference_strict, duration)
        report["conversion_stats"] = conversion_stats
        write_json(args.report_json, report)
        args.report_md.parent.mkdir(parents=True, exist_ok=True)
        args.report_md.write_text(verification_markdown(report), encoding="utf-8")
        print(f"Wrote verification reports: {args.report_json}, {args.report_md}")
        print(
            f"canonical_match={report['comparison']['canonical_transcripts_identical']}, "
            f"word_sequence_match={report['comparison']['word_sequence_identical']}, "
            f"matched_words={report['comparison']['matched_words']}"
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
