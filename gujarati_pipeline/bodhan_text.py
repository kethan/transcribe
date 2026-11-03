from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .config import Config
from .types import jsonable

REFERENCE_NOTE = (
    "Verbatim Bodhan ASR response text from this run, in audio-span order. "
    "It is a reference artifact only: it has no NeMo speaker labels and no CTC "
    "word timestamps, and it is not the final strict transcript JSON. No text "
    "from the ElevenLabs Scribe 2 sample is included."
)


def _raw_response_text(chunk: Any) -> str:
    response = getattr(chunk, "response", None)
    if isinstance(response, dict) and isinstance(response.get("text"), str):
        return response["text"]
    return ""


def _pipeline_text(chunk: Any) -> str:
    text = getattr(chunk, "text", "")
    return text if isinstance(text, str) else ""


def bodhan_reference_records(chunks: Iterable[Any]) -> list[dict[str, Any]]:
    """Return one record per Bodhan chunk in absolute chronological order."""
    ordered = sorted(
        chunks,
        key=lambda chunk: (
            float(getattr(getattr(chunk, "span", None), "start", 0.0)),
            float(getattr(getattr(chunk, "span", None), "end", 0.0)),
            int(getattr(getattr(chunk, "span", None), "index", 0)),
        ),
    )
    records: list[dict[str, Any]] = []
    for chunk in ordered:
        span = getattr(chunk, "span", None)
        response = getattr(chunk, "response", None)
        response_dict = response if isinstance(response, dict) else {}
        raw_text = _raw_response_text(chunk)
        pipeline_text = _pipeline_text(chunk)
        text = raw_text if raw_text.strip() else pipeline_text
        records.append(
            {
                "chunk_index": int(getattr(span, "index", -1)),
                "chunk_key": str(getattr(chunk, "key", "")),
                "span_key": str(getattr(span, "key", "")),
                "start": float(getattr(span, "start", 0.0)),
                "end": float(getattr(span, "end", 0.0)),
                "text": text,
                "response_text": raw_text,
                "pipeline_text": pipeline_text,
                "backend": response_dict.get("backend"),
                "model": response_dict.get("model"),
                "language": response_dict.get("language"),
            }
        )
    return records


def bodhan_reference_payload(chunks: Iterable[Any], config: Config) -> dict[str, Any]:
    """Build a provenance-bearing JSON reference from live Bodhan responses."""
    records = bodhan_reference_records(chunks)
    text = "\n".join(record["text"].strip() for record in records if record["text"].strip())
    return {
        "provenance": "Bodhan ASR responses captured by gujarati_pipeline",
        "note": REFERENCE_NOTE,
        "backend": config.asr_backend,
        "hf_model": config.hf_asr_model if config.asr_backend == "hf-local" else None,
        "cloud_model": config.bodhan_model if config.asr_backend == "cloud" else None,
        "language": config.language,
        "chunk_count": len(records),
        "text_characters": len(text),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "text": text,
        "chunks": records,
    }


def write_bodhan_reference(
    chunks: Iterable[Any],
    config: Config,
    output_dir: Path | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Write downloadable plain-text and JSON Bodhan references.

    Neither file is used as pipeline input. They contain only text emitted by
    the configured ASR backend for this run; no transcript text is hardcoded.
    """
    directory = Path(output_dir) if output_dir is not None else config.output.parent
    directory.mkdir(parents=True, exist_ok=True)
    payload = bodhan_reference_payload(chunks, config)
    text_path = directory / "bodhan_transcript.txt"
    json_path = directory / "bodhan_transcript.json"
    text_path.write_text(
        payload["text"] + ("\n" if payload["text"] else ""),
        encoding="utf-8",
    )
    json_path.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return text_path, json_path, payload
