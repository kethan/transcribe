#!/usr/bin/env python3
"""WER/CER comparison for the CPU-only transcript path.

This tool never calls the Bodhan cloud API. Save a cloud API response or a
manually corrected transcript to a UTF-8 text file, then compare it with the
local ``cpu_transcript.txt``. Lower WER/CER is better; neither number proves
accuracy parity.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any, Callable, Sequence


def normalize_text(
    text: str,
    *,
    ignore_punctuation: bool = True,
    lowercase: bool = False,
) -> str:
    """Normalize whitespace, optionally punctuation and Latin case."""
    if lowercase:
        text = text.casefold()
    if ignore_punctuation:
        # Replace punctuation with a space so "word,word" does not merge.
        text = "".join(
            " " if unicodedata.category(character).startswith("P") else character
            for character in text
        )
    return " ".join(text.split())


def compute_rates(
    reference: str,
    hypothesis: str,
    *,
    ignore_punctuation: bool = True,
    lowercase: bool = False,
    distance_fn: Callable[[Sequence[Any], Sequence[Any]], int] | None = None,
) -> dict[str, Any]:
    """Compute token WER and character CER using Levenshtein distance."""
    if distance_fn is None:
        try:
            from rapidfuzz.distance import Levenshtein
        except ImportError as exc:  # pragma: no cover - install-time error
            raise RuntimeError(
                "rapidfuzz is required for comparison. "
                "Install laptop/requirements.txt first."
            ) from exc
        distance_fn = Levenshtein.distance

    ref = normalize_text(
        reference,
        ignore_punctuation=ignore_punctuation,
        lowercase=lowercase,
    )
    hyp = normalize_text(
        hypothesis,
        ignore_punctuation=ignore_punctuation,
        lowercase=lowercase,
    )
    ref_words = ref.split()
    hyp_words = hyp.split()
    ref_chars = list(ref.replace(" ", ""))
    hyp_chars = list(hyp.replace(" ", ""))

    word_edits = int(distance_fn(ref_words, hyp_words))
    char_edits = int(distance_fn(ref_chars, hyp_chars))
    word_denominator = max(1, len(ref_words))
    char_denominator = max(1, len(ref_chars))

    return {
        "normalization": {
            "ignore_punctuation": ignore_punctuation,
            "lowercase": lowercase,
            "whitespace": "collapsed",
            "cer_spaces": "removed",
        },
        "reference_words": len(ref_words),
        "hypothesis_words": len(hyp_words),
        "reference_characters": len(ref_chars),
        "hypothesis_characters": len(hyp_chars),
        "word_edits": word_edits,
        "character_edits": char_edits,
        "wer": word_edits / word_denominator,
        "cer": char_edits / char_denominator,
    }


def write_reports(
    report: dict[str, Any],
    json_path: Path,
    md_path: Path,
) -> None:
    """Write machine-readable and human-readable reports."""
    json_path = Path(json_path)
    md_path = Path(md_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# CPU ASR comparison",
        "",
        f"- Reference: `{report['reference_file']}`",
        f"- Hypothesis: `{report['hypothesis_file']}`",
        f"- Ignore punctuation: `{report['normalization']['ignore_punctuation']}`",
        f"- Lowercase: `{report['normalization']['lowercase']}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| WER | {report['wer']:.2%} |",
        f"| CER | {report['cer']:.2%} |",
        f"| Word edits | {report['word_edits']} / {report['reference_words']} reference words |",
        f"| Character edits | {report['character_edits']} / {report['reference_characters']} reference characters |",
        "",
        "Lower is better. WER can exceed 100% when insertions outnumber the "
        "reference words. This is a diagnostic, not proof of parity with the "
        "Bodhan cloud API. Use a manually corrected transcript or a saved cloud "
        "response as the reference; ElevenLabs/Scribe 2 output is not ground truth.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute WER/CER between a reference text and the CPU transcript.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--reference", required=True,
        help="UTF-8 reference text (saved cloud response or manually corrected text)",
    )
    parser.add_argument(
        "--hypothesis", required=True,
        help="UTF-8 local CPU transcript text",
    )
    parser.add_argument(
        "--keep-punctuation", action="store_true",
        help="Score punctuation instead of ignoring it",
    )
    parser.add_argument(
        "--lowercase", action="store_true",
        help="Casefold before scoring (mainly for Latin code-mixed words)",
    )
    parser.add_argument(
        "--output-json", default="cpu_compare.json",
        help="Write numeric results as JSON",
    )
    parser.add_argument(
        "--output-md", default="cpu_compare.md",
        help="Write a short Markdown report",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    reference_path = Path(args.reference)
    hypothesis_path = Path(args.hypothesis)
    if not reference_path.is_file():
        parser.error(f"reference file not found: {reference_path}")
    if not hypothesis_path.is_file():
        parser.error(f"hypothesis file not found: {hypothesis_path}")

    reference = reference_path.read_text(encoding="utf-8")
    hypothesis = hypothesis_path.read_text(encoding="utf-8")
    report = compute_rates(
        reference,
        hypothesis,
        ignore_punctuation=not args.keep_punctuation,
        lowercase=args.lowercase,
    )
    report = {
        "reference_file": str(reference_path),
        "hypothesis_file": str(hypothesis_path),
        **report,
    }
    write_reports(report, Path(args.output_json), Path(args.output_md))
    print(f"WER={report['wer']:.4f} CER={report['cer']:.4f}")
    print(f"JSON: {Path(args.output_json).resolve()}")
    print(f"Markdown: {Path(args.output_md).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
