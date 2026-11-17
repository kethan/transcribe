from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from gujarati_pipeline.config import Config

DEFAULT_WORK_DIR = Path("work")
DEFAULT_TXT = Path("outputs/bodhan_transcript.txt")
DEFAULT_JSON = Path("outputs/bodhan_transcript.json")
CACHE_NOTE = (
    "Verbatim Bodhan ASR response text recovered from v2 transcription cache "
    "files. It is a reference artifact only: no NeMo speaker labels, no CTC "
    "word timestamps, and no ElevenLabs Scribe 2 text. It is not the final "
    "strict transcript JSON."
)


def config_digest_from_report(path: Path) -> str:
    """Recompute the transcription cache digest recorded by a pipeline report."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    details = data.get("details") if isinstance(data, dict) else None
    config_dict = details.get("config") if isinstance(details, dict) else None
    if not isinstance(config_dict, dict):
        raise ValueError(f"{path} has no details.config object.")
    field_names = {field.name for field in dataclasses.fields(Config)}
    kwargs = {key: value for key, value in config_dict.items() if key in field_names}
    return Config(**kwargs).transcription_config_digest()


def cache_records(
    cache_dir: Path,
    *,
    backend: str | None = None,
    digest: str | None = None,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Group valid v2 cached Bodhan responses by backend and config digest."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    if not cache_dir.is_dir():
        return groups
    for path in sorted(cache_dir.glob("*.json")):
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(envelope, dict) or envelope.get("version") != 2:
            continue
        response = envelope.get("response")
        if not isinstance(response, dict) or "error" in response:
            continue
        raw_text = response.get("text")
        if not isinstance(raw_text, str):
            continue
        row_backend = str(envelope.get("backend") or "")
        row_digest = str(envelope.get("config_digest") or "")
        if backend and row_backend != backend:
            continue
        if digest and row_digest != digest:
            continue
        span = envelope.get("span") if isinstance(envelope.get("span"), dict) else {}
        try:
            start = float(span.get("start", 0.0))
            end = float(span.get("end", 0.0))
            chunk_index = int(span.get("index", -1))
        except (TypeError, ValueError):
            continue
        groups[(row_backend, row_digest)].append(
            {
                "cache_file": str(path),
                "backend": row_backend,
                "config_digest": row_digest,
                "span_key": envelope.get("span_key"),
                "chunk_index": chunk_index,
                "start": start,
                "end": end,
                "text": raw_text,
                "model": response.get("model"),
                "language": response.get("language"),
            }
        )
    return groups


def select_group(
    groups: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    backend: str | None,
    digest: str | None,
) -> tuple[tuple[str, str], list[dict[str, Any]]]:
    """Select exactly one cache group; never combine different ASR runs."""
    candidates = {
        key: records
        for key, records in groups.items()
        if (not backend or key[0] == backend) and (not digest or key[1] == digest)
    }
    if not candidates:
        raise RuntimeError(
            "No valid v2 Bodhan response cache matched the requested backend/digest. "
            "Run the pipeline first, or pass --config-digest from its processing report."
        )
    if len(candidates) > 1:
        keys = ", ".join(
            f"{key[0]}/{key[1]} ({len(records)} chunks)"
            for key, records in sorted(candidates.items())
        )
        raise RuntimeError(
            "Multiple ASR cache groups found; refusing to mix them. "
            f"Pass --config-digest (or --report). Groups: {keys}"
        )
    return next(iter(candidates.items()))


def build_payload(
    key: tuple[str, str], records: list[dict[str, Any]]
) -> dict[str, Any]:
    ordered = sorted(
        records,
        key=lambda record: (
            record["start"],
            record["end"],
            record["chunk_index"],
            record["cache_file"],
        ),
    )
    text = "\n".join(record["text"].strip() for record in ordered if record["text"].strip())
    indices = sorted({r["chunk_index"] for r in ordered if r["chunk_index"] >= 0})
    missing_indices: list[int] = []
    if indices:
        present = set(indices)
        missing_indices = [i for i in range(indices[0], indices[-1] + 1) if i not in present]
    return {
        "provenance": "Bodhan ASR v2 transcription response cache",
        "note": CACHE_NOTE,
        "source": "cache",
        "backend": key[0],
        "config_digest": key[1],
        "chunk_count": len(ordered),
        "text_characters": len(text),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "missing_chunk_indices_in_cached_range": missing_indices,
        "text": text,
        "chunks": ordered,
    }


def write_outputs(
    payload: dict[str, Any], output_txt: Path, output_json: Path
) -> tuple[Path, Path]:
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text(
        payload["text"] + ("\n" if payload["text"] else ""),
        encoding="utf-8",
    )
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_txt, output_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.bodhan_reference",
        description=(
            "Export the actual Bodhan ASR text from v2 transcription cache files "
            "as downloadable TXT and JSON references. No model inference is run."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="processing_report.json; used to select the matching config digest.",
    )
    parser.add_argument("--backend", default="bodhan-hf-local")
    parser.add_argument("--config-digest", default=None)
    parser.add_argument("--output-txt", type=Path, default=DEFAULT_TXT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument(
        "--list-groups",
        action="store_true",
        help="List cached backend/digest groups and exit without writing files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cache_dir = args.work_dir / "transcription" / "responses"
    try:
        all_groups = cache_records(cache_dir)
        if args.list_groups:
            if not all_groups:
                print(f"No v2 Bodhan response caches found in {cache_dir}")
            for (backend, digest), records in sorted(all_groups.items()):
                print(f"{backend}\t{digest}\t{len(records)} chunks")
            return 0
        digest = args.config_digest
        if digest is None and args.report is not None:
            digest = config_digest_from_report(args.report)
        key, records = select_group(
            all_groups, backend=args.backend or None, digest=digest
        )
        payload = build_payload(key, records)
        txt_path, json_path = write_outputs(payload, args.output_txt, args.output_json)
        print(f"Wrote Bodhan text reference: {txt_path}")
        print(f"Wrote Bodhan JSON reference: {json_path}")
        print(
            f"Backend={key[0]}, digest={key[1]}, chunks={payload['chunk_count']}, "
            f"characters={payload['text_characters']}"
        )
        if payload["missing_chunk_indices_in_cached_range"]:
            print(
                "Warning: cached chunk indices are not contiguous: "
                f"{payload['missing_chunk_indices_in_cached_range']}",
                file=sys.stderr,
            )
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
