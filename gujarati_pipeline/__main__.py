from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

from . import __version__
from .config import Config
from .pipeline import check, run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m gujarati_pipeline",
        description=(
            "Gujarati/Hindi/English meeting pipeline: local Bodhan HF ASR by default, "
            "NVIDIA NeMo Sortformer diarization, CTC word alignment, validated JSON."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    io = p.add_argument_group("Input / output")
    io.add_argument("--audio", type=Path, default=None, help="Input WAV (any ffmpeg-readable format also accepted).")
    io.add_argument("--output", type=Path, default=None, help="Final transcript JSON.")
    io.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Convenience directory: transcript.json, processing_report.json, processing_report.md.",
    )
    io.add_argument("--work-dir", type=Path, default=None, help="Organised intermediate/cache directory.")
    io.add_argument("--report-json", type=Path, default=None, help="Detailed processing report JSON.")
    io.add_argument("--report-md", type=Path, default=None, help="Brief processing report Markdown.")
    io.add_argument("--check", action="store_true", help="Verify environment/model compatibility without running inference.")
    io.add_argument(
        "--force",
        default=None,
        help="Comma-separated cache stages to recompute: transcription,diarization,alignment,all,none.",
    )

    asr = p.add_argument_group(
        "Bodhan ASR (default: local Hugging Face model; env: ASR_BACKEND, HF_ASR_MODEL, "
        "HF_ASR_MODEL_DIR, HF_TOKEN, BODHAN_LANGUAGE)"
    )
    asr.add_argument(
        "--asr-backend",
        default=None,
        choices=["hf-local", "cloud"],
        help="hf-local uses bodhan-ai/indic-transcribe-core; cloud uses the hosted Bodhan API.",
    )
    asr.add_argument("--hf-asr-model", default=None, help="Bodhan HF repo id used by hf-local.")
    asr.add_argument(
        "--hf-asr-model-dir",
        type=Path,
        default=None,
        help="Already-downloaded HF snapshot directory; avoids a gated download.",
    )
    asr.add_argument(
        "--asr-compute-dtype",
        default=None,
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Local ASR dtype; auto picks bfloat16 on CUDA, float32 on CPU.",
    )
    asr.add_argument(
        "--language",
        default=None,
        help="ASR language hint, e.g. gu, hi, en; empty string enables model LID.",
    )
    asr.add_argument(
        "--transcription-cache-only",
        action="store_true",
        help="Never run ASR; require valid cached responses for every chunk.",
    )
    asr.add_argument(
        "--bodhan-cache-only",
        action="store_true",
        help="Deprecated alias for --transcription-cache-only.",
    )
    asr.add_argument("--bodhan-chunk-seconds", type=float, default=None, help="Nominal silence-aware chunk length.")
    asr.add_argument("--bodhan-max-chunk-seconds", type=float, default=None, help="Hard max; must stay <= 30 s.")
    asr.add_argument("--bodhan-min-chunk-seconds", type=float, default=None, help="Minimum chunk before a boundary.")
    asr.add_argument("--bodhan-silence-search-seconds", type=float, default=None, help="± window for a quiet boundary.")

    cloud = p.add_argument_group(
        "Optional hosted Bodhan API (only with --asr-backend cloud; env: BODHAN_API_KEY, "
        "BODHAN_BASE_URL, BODHAN_MODEL)"
    )
    cloud.add_argument("--bodhan-base-url", default=None, help="Bodhan API base URL.")
    cloud.add_argument("--bodhan-model", default=None, help="Documented model name: indic-transcribe.")
    cloud.add_argument(
        "--bodhan-workers",
        type=int,
        default=None,
        help="Parallel cloud uploads; keep within Bodhan rate limits. Local HF inference is sequential.",
    )
    cloud.add_argument("--bodhan-max-retries", type=int, default=None, help="Retries on 429/5xx/network errors.")
    cloud.add_argument("--bodhan-timeout", type=float, default=None, help="Read timeout in seconds.")
    cloud.add_argument("--bodhan-retry-base", type=float, default=None, help="Exponential backoff base.")

    nemo = p.add_argument_group("NVIDIA NeMo (env: NEMO_DIAR_MODEL, HF_TOKEN, PIPELINE_DEVICE)")
    nemo.add_argument("--diar-model", default=None, help="HF repo id or local .nemo checkpoint.")
    nemo.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"], help="Inference device.")

    align = p.add_argument_group("CTC alignment (env: ALIGNMENT_MODEL, ALIGNMENT_LANGUAGE, MMS_ALIGN_MODEL, GUJARATI_ALIGN_MODEL)")
    align.add_argument(
        "--alignment-model",
        default=None,
        choices=["mms-uroman", "gujarati-wav2vec2", "auto"],
        help="Default mms-uroman handles Gujarati/Hindi/English via uroman; auto may load both CTC models.",
    )
    align.add_argument("--alignment-language", default=None, help="ISO 639-3 hint for uroman, e.g. guj.")
    align.add_argument("--mms-model", default=None, help="MMS CTC forced-aligner checkpoint.")
    align.add_argument("--gujarati-model", default=None, help="Gujarati Wav2Vec2 CTC checkpoint to evaluate.")
    align.add_argument("--compute-dtype", default=None, choices=["float32", "float16", "bfloat16"])
    align.add_argument("--star-penalty", type=float, default=None, help="Penalty for the CTC <star> edge token.")

    assembly = p.add_argument_group("Speaker assignment / grouping")
    assembly.add_argument(
        "--segmentation-mode",
        default=None,
        choices=["speaker-turn", "legacy"],
        help=(
            "speaker-turn keeps all consecutive words from one speaker in a single segment; "
            "legacy also splits on pauses, sentence ends, ASR chunks and max duration."
        ),
    )
    assembly.add_argument("--pause-threshold", type=float, default=None, help="Meaningful pause in seconds (legacy mode).")
    assembly.add_argument("--max-segment-seconds", type=float, default=None, help="Maximum grouped segment duration (legacy mode).")
    assembly.add_argument("--speaker-min-coverage", type=float, default=None, help="Min winner overlap / word duration.")
    assembly.add_argument("--speaker-ambiguity-margin", type=float, default=None, help="Runner-up tie threshold (0-1).")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args: Any = parser.parse_args(argv)
    output_dir = getattr(args, "output_dir", None)
    if output_dir is not None:
        output_dir = Path(output_dir)
        if args.output is None:
            args.output = output_dir / "transcript.json"
        if args.report_json is None:
            args.report_json = output_dir / "processing_report.json"
        if args.report_md is None:
            args.report_md = output_dir / "processing_report.md"
    try:
        config = Config.from_args(args)
        config.validate()
    except Exception as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    if config.check:
        try:
            return check(config)
        except Exception as exc:
            print(f"Compatibility check failed: {exc}", file=sys.stderr)
            return 1
    try:
        run(config)
    except Exception as exc:
        print(f"\nPipeline failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
