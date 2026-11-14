#!/usr/bin/env python3
"""WER/CER comparison for the CPU transcript. Never calls the cloud API."""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Any, Callable, Sequence


def normalize_text(
    text: str, *, ignore_punctuation: bool = True, lowercase: bool = False
) -> str:
    if lowercase:
        text = text.casefold()
    if ignore_punctuation:
        text = "".join(
            " " if unicodedata.category(ch).startswith("P") else ch for ch in text
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
    if distance_fn is None:
        from rapidfuzz.distance import Levenshtein
        distance_fn = Levenshtein.distance

    ref = normalize_text(
        reference, ignore_punctuation=ignore_punctuation, lowercase=lowercase
    )
    hyp = normalize_text(
        hypothesis, ignore_punctuation=ignore_punctuation, lowercase=lowercase
    )
    ref_words, hyp_words = ref.split(), hyp.split()
    ref_chars = list(ref.replace(" ", ""))
    hyp_chars = list(hyp.replace(" ", ""))
    word_edits = int(distance_fn(ref_words, hyp_words))
    char_edits = int(distance_fn(ref_chars, hyp_chars))
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
        "wer": word_edits / max(1, len(ref_words)),
        "cer": char_edits / max(1, len(ref_chars)),
        "note": (
            "Lower is better. This is a diagnostic, not proof of Bodhan cloud "
            "API parity. Use a saved cloud response or manually corrected text "
            "as the reference; ElevenLabs/Scribe 2 is not ground truth."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute WER/CER between a reference and the CPU transcript."
    )
    parser.add_argument("--reference", required=True, help="UTF-8 reference text")
    parser.add_argument("--hypothesis", required=True, help="UTF-8 CPU transcript")
    parser.add_argument("--keep-punctuation", action="store_true")
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--output-json", default="cpu_compare.json")
    args = parser.parse_args(argv)

    reference_path = Path(args.reference)
    hypothesis_path = Path(args.hypothesis)
    if not reference_path.is_file():
        parser.error(f"reference file not found: {reference_path}")
    if not hypothesis_path.is_file():
        parser.error(f"hypothesis file not found: {hypothesis_path}")

    report = compute_rates(
        reference_path.read_text(encoding="utf-8"),
        hypothesis_path.read_text(encoding="utf-8"),
        ignore_punctuation=not args.keep_punctuation,
        lowercase=args.lowercase,
    )
    report = {
        "reference_file": str(reference_path),
        "hypothesis_file": str(hypothesis_path),
        **report,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"WER={report['wer']:.4f} CER={report['cer']:.4f}")
    print(f"Report: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
