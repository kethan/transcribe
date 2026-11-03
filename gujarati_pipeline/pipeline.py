from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from .alignment import AlignmentRouter, tokenize_transcript, verify_alignment_compatibility
from .assembly import build_output
from .audio import (
    SAMPLE_RATE,
    is_canonical_wav,
    plan_spans,
    prepare_canonical_audio,
    read_pcm16_wav,
)
from .config import Config
from .transcription import create_transcriber
from .diarization import run_diarization, verify_environment
from .runtime import gpu_info, package_version
from .types import ProcessingReport, StageError, jsonable
from .validation import validate_output


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _source_text(chunks: list[Any]) -> str:
    return " ".join(chunk.text for chunk in chunks if chunk.text.strip())


def _expected_word_count(chunks: list[Any]) -> int:
    return sum(len(tokenize_transcript(chunk.text)) for chunk in chunks)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _markdown_report(report: ProcessingReport) -> str:
    d = report.details
    audio = d.get("audio", {})
    diar = d.get("diarization", {})
    out = d.get("output", {})
    align = d.get("alignment", {})
    lines = [
        "# Processing report",
        "",
        f"- **Status:** {report.status}",
        f"- **Stage:** {report.stage or 'completed'}",
        f"- **Audio duration:** {audio.get('duration_seconds', 'n/a')} s",
        f"- **Detected speakers:** {diar.get('speaker_count', 'n/a')}",
        f"- **Segments:** {out.get('segments', 'n/a')}",
        f"- **Aligned words:** {out.get('words', 'n/a')}",
        f"- **Words without speaker attribution:** {out.get('words_without_speaker', 'n/a')}",
        f"- **ASR chunks:** {d.get('transcription', d.get('bodhan', {})).get('chunks', 'n/a')}",
        f"- **Alignment models used:** {', '.join(align.get('models_used', {}).keys()) or 'none'}",
        f"- **Final JSON:** `{d.get('paths', {}).get('transcript', '')}`",
        "",
        "## Alignment warnings or failures",
        "",
    ]
    failures = list(report.failures) + list(align.get("failures", []))
    if failures:
        lines.extend(f"- FAILURE: {f}" for f in failures)
    else:
        lines.append("- No alignment failures.")
    warn_lines = [
        w
        for w in report.warnings
        if any(
            token in w.lower()
            for token in ("align", "ctc", "asr", "bodhan", "word", "split", "merged", "transcrib")
        )
        or "NeMo" in w
    ]
    if warn_lines:
        lines.extend(f"- WARNING: {w}" for w in warn_lines[:25])
    else:
        lines.append("- No alignment warnings.")
    lines.extend(["", "## All warnings", ""])
    if report.warnings:
        lines.extend(f"- {w}" for w in report.warnings[:50])
    else:
        lines.append("- None.")
    lines.append("")
    return "\n".join(lines)


def _write_report(config: Config, report: ProcessingReport, final_json_written: bool) -> None:
    report.details.setdefault("paths", {})["transcript"] = str(config.output)
    report.details.setdefault("paths", {})["report_json"] = str(config.report_json)
    report.details.setdefault("paths", {})["report_md"] = str(config.report_md)
    report.details["final_json_written"] = final_json_written
    assert config.report_json is not None and config.report_md is not None
    _write_json(config.report_json, report)
    config.report_md.parent.mkdir(parents=True, exist_ok=True)
    config.report_md.write_text(_markdown_report(report), encoding="utf-8")


def _write_failure_report(
    config: Config,
    stage: str,
    error: Exception,
    warnings: list[str],
) -> None:
    report = ProcessingReport(
        status="failed",
        stage=stage,
        warnings=_dedupe(warnings),
        failures=[f"{type(error).__name__}: {error}"],
        details={
            "config": config.public_dict(),
            "gpu": gpu_info(),
            "audio": {"path": str(config.audio) if config.audio else None},
        },
    )
    try:
        _write_report(config, report, final_json_written=False)
    except Exception as exc:  # pragma: no cover - never mask the original error
        print(f"[Report] Could not write failure report: {exc}", file=sys.stderr)


def check(config: Config) -> int:
    """Offline compatibility check; does not download model weights."""
    env = verify_environment(config, None)
    alignment = verify_alignment_compatibility(config)
    ffmpeg = shutil.which("ffmpeg")
    audio_status: dict[str, Any] = {"provided": bool(config.audio)}
    if config.audio:
        audio_status["exists"] = config.audio.is_file()
        audio_status["already_16k_mono_pcm16_wav"] = (
            config.audio.is_file() and is_canonical_wav(config.audio)
        )
    payload = {
        "environment": env,
        "alignment_compatibility": alignment,
        "ffmpeg_available": bool(ffmpeg),
        "ffmpeg_path": ffmpeg,
        "audio": audio_status,
        "asr": {
            "backend": config.asr_backend,
            "hf_model": config.hf_asr_model,
            "hf_model_dir": str(config.hf_asr_model_dir) if config.hf_asr_model_dir else None,
            "hf_token_present": bool(
                os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            ),
            "bodhan_cloud_api_key_present": bool(config.bodhan_api_key),
            "bodhan_cloud_documented_response": {"text": "string", "word_timestamps": "not documented"},
        },
        "bodhan_api_key_present": bool(config.bodhan_api_key),
        "versions": {
            "python": sys.version.split()[0],
            "numpy": package_version("numpy"),
            "requests": package_version("requests"),
            "uroman": package_version("uroman"),
            "huggingface_hub": package_version("huggingface_hub"),
            "sentencepiece": package_version("sentencepiece"),
            "soundfile": package_version("soundfile"),
            "transformers": package_version("transformers"),
            "nemo-toolkit": package_version("nemo-toolkit"),
        },
    }
    print(json.dumps(jsonable(payload), ensure_ascii=False, indent=2))
    if not env["ok"]:
        print("\nCHECK FAILED: resolve the environment errors above.", file=sys.stderr)
        return 1
    if not alignment.get("ok"):
        print(
            "\nCHECK WARNING: alignment model metadata could not be fully verified. "
            "See alignment_compatibility above.",
            file=sys.stderr,
        )
        return 1
    if config.audio and not audio_status.get("already_16k_mono_pcm16_wav", False) and not ffmpeg:
        print(
            "\nCHECK WARNING: input is not 16 kHz mono PCM16 WAV and ffmpeg is missing.",
            file=sys.stderr,
        )
    if (
        config.asr_backend == "hf-local"
        and not config.hf_asr_model_dir
        and not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    ):
        print(
            "\nCHECK NOTE: the local Bodhan HF repo is gated. Accept its license and set "
            "HF_TOKEN, or pass --hf-asr-model-dir to an already-downloaded snapshot.",
            file=sys.stderr,
        )
    print(
        "\nCHECK OK: ASR backend, NeMo class, PyTorch/GPU status, CTC metadata and "
        "audio conversion are available."
    )
    return 0


def run(config: Config) -> dict[str, Any]:
    """Execute all seven stages and write the validated transcript + reports."""
    config.validate()
    warnings: list[str] = []
    failures: list[str] = []
    stage = "environment"
    info = None
    chunks: list[Any] = []
    print("=" * 78)
    print("Gujarati Bodhan / NVIDIA NeMo / CTC alignment pipeline")
    print("=" * 78)
    try:
        print("\n[0/7] Environment")
        env0 = verify_environment(config, None)
        if not env0["ok"]:
            raise StageError("NeMo environment check failed:\n- " + "\n- ".join(env0["errors"]))
        warnings.extend(env0["warnings"])
        gpu = gpu_info()
        print(
            f"[Environment] Python {sys.version.split()[0]}, torch={gpu.get('torch_version')}, "
            f"CUDA={gpu.get('cuda_available')}"
            + (f", GPU={gpu.get('device_name')}" if gpu.get("cuda_available") else "")
        )

        stage = "audio"
        print("\n[1/7] Audio preparation")
        info = prepare_canonical_audio(config)
        samples = read_pcm16_wav(info.canonical_path)
        print(
            f"[Audio] {info.original_path.name} -> {info.canonical_path} "
            f"({info.duration:.3f}s, {info.sample_rate} Hz mono PCM16; {info.conversion})"
        )
        if samples.size != info.frames:
            print(
                f"[Audio] Warning: header says {info.frames} samples, decoded {samples.size}.",
                file=sys.stderr,
            )

        stage = "asr-chunking"
        print("\n[2/7] Silence-aware ASR chunking (Bodhan 30 s limit)")
        spans = plan_spans(info, samples, config, warnings)
        print(f"[ASR] {len(spans)} trimmed speech span(s), each <= {config.bodhan_max_chunk_seconds}s")

        stage = "asr-transcription"
        print(f"\n[3/7] Bodhan transcription ({config.asr_backend})")
        transcriber = create_transcriber(config, samples, warnings)
        chunks = transcriber.transcribe(spans)
        source_text = _source_text(chunks)
        expected_words = _expected_word_count(chunks)
        bodhan_word_chunks = sum(1 for c in chunks if c.has_word_times)
        if not chunks:
            ts_source = "no speech spans / empty transcript"
        elif bodhan_word_chunks == 0:
            ts_source = "CTC forced alignment (ASR response has only text)"
        elif bodhan_word_chunks == len(chunks):
            ts_source = "ASR word timestamps"
        else:
            ts_source = "mixed ASR word timestamps and CTC forced alignment"
        print(
            f"[ASR] {len(chunks)} chunk(s), {expected_words} whitespace token(s); "
            f"word-time source: {ts_source}"
        )

        stage = "nemo-diarization"
        print("\n[4/7] NVIDIA NeMo Sortformer diarization")
        intervals, speaker_ids, diar_meta = run_diarization(info, config, warnings)

        stage = "ctc-alignment"
        print("\n[5/7] Word-level CTC forced alignment")
        router = AlignmentRouter(config, info, warnings)
        alignment_stats = router.align_chunks(
            chunks, force="alignment" in config.force_stages
        )
        failures.extend(alignment_stats.failures)
        print(
            f"[Alignment] {len(alignment_stats.words)} word(s); models={alignment_stats.models_used}; "
            f"cached={alignment_stats.cached_units}, computed={alignment_stats.aligned_units}, "
            f"auto-fallbacks={alignment_stats.fallback_units}"
        )

        stage = "assembly"
        print("\n[6/7] Speaker overlap assignment and segment grouping")
        output, speaker_stats, assembly_stats = build_output(
            alignment_stats.words, intervals, config, warnings
        )
        print(
            f"[Assembly] {assembly_stats.segment_count} segment(s); "
            f"{speaker_stats.unattributed_words} word(s) with speaker_id=null"
        )

        stage = "validation-export"
        print("\n[7/7] Validation and UTF-8 export")
        summary = validate_output(
            output,
            info.duration,
            source_text,
            speaker_ids,
            expected_words,
            warnings,
        )
        assert info is not None
        _write_json(config.output, output)
        report = ProcessingReport(
            status="ok",
            stage="completed",
            warnings=_dedupe(warnings),
            failures=failures,
            details={
                "config": config.public_dict(),
                "gpu": gpu_info(),
                "audio": {
                    "path": str(info.original_path),
                    "canonical_wav": str(info.canonical_path),
                    "audio_hash": info.audio_hash,
                    "duration_seconds": info.duration,
                    "sample_rate": info.sample_rate,
                    "channels": info.channels,
                    "sample_width_bytes": info.sample_width_bytes,
                    "conversion": info.conversion,
                },
                "transcription": {
                    "backend": config.asr_backend,
                    "hf_model": config.hf_asr_model if config.asr_backend == "hf-local" else None,
                    "hf_model_dir": str(config.hf_asr_model_dir) if config.hf_asr_model_dir else None,
                    "cloud_endpoint": (
                        f"{config.bodhan_base_url}/v1/audio/transcriptions"
                        if config.asr_backend == "cloud"
                        else None
                    ),
                    "cloud_model": config.bodhan_model if config.asr_backend == "cloud" else None,
                    "language": config.language,
                    "chunks": len(chunks),
                    "word_timestamp_source": ts_source,
                    "chunks_with_asr_word_times": bodhan_word_chunks,
                    "source_text_characters": len(source_text),
                },
                "diarization": {
                    "speaker_count": len(speaker_ids),
                    "speaker_ids": speaker_ids,
                    "interval_count": len(intervals),
                    "meta": diar_meta,
                },
                "alignment": {
                    "models_used": alignment_stats.models_used,
                    "bodhan_word_units": alignment_stats.bodhan_word_units,
                    "cached_units": alignment_stats.cached_units,
                    "aligned_units": alignment_stats.aligned_units,
                    "fallback_units": alignment_stats.fallback_units,
                    "failures": alignment_stats.failures,
                },
                "speaker_assignment": {
                    "assigned_words": speaker_stats.assigned_words,
                    "no_overlap_words": speaker_stats.no_overlap_words,
                    "uncertain_words": speaker_stats.uncertain_words,
                    "unattributed_words": speaker_stats.unattributed_words,
                    "min_coverage": config.speaker_min_coverage,
                    "ambiguity_margin": config.speaker_ambiguity_margin,
                },
                "output": summary,
            },
        )
        _write_report(config, report, final_json_written=True)
        print(f"[Export] Final JSON: {config.output}")
        print(f"[Export] Report:     {config.report_json}")
        print(f"[Export] Report MD:  {config.report_md}")
        print(
            f"[Summary] duration={info.duration:.2f}s speakers={len(speaker_ids)} "
            f"segments={summary['segments']} words={summary['words']} "
            f"without_speaker={summary['words_without_speaker']}"
        )
        return output
    except Exception as exc:
        _write_failure_report(config, stage, exc, warnings)
        raise
