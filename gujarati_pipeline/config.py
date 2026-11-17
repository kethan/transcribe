from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from .types import jsonable

DEFAULT_DIGEST_KEEP = object()


@dataclass
class Config:
    audio: Optional[Path] = None
    output: Path = Path("outputs/transcript.json")
    work_dir: Path = Path("work")
    report_json: Optional[Path] = None
    report_md: Optional[Path] = None
    check: bool = False

    # Transcription. The local Bodhan Hugging Face model is the default; the
    # hosted Bodhan API remains an optional, explicitly selected backend.
    asr_backend: str = "hf-local"  # hf-local | cloud
    hf_asr_model: str = "bodhan-ai/indic-transcribe-core"
    hf_asr_model_dir: Optional[Path] = None
    asr_compute_dtype: str = "auto"  # auto | float32 | float16 | bfloat16
    language: str = "gu"
    transcription_cache_only: bool = False

    # Hosted Bodhan AI. Only used with --asr-backend cloud.
    bodhan_base_url: str = "https://api.bodhan.ai"
    bodhan_model: str = "indic-transcribe"
    bodhan_api_key: Optional[str] = None
    bodhan_chunk_seconds: float = 25.0
    bodhan_max_chunk_seconds: float = 28.0
    bodhan_min_chunk_seconds: float = 5.0
    bodhan_silence_search_seconds: float = 3.0
    bodhan_workers: int = 1
    bodhan_max_retries: int = 5
    bodhan_timeout: float = 120.0
    bodhan_retry_base: float = 1.5

    # NeMo / NVIDIA Nemotron-3-Diarization (Sortformer).
    diar_model: str = "nvidia/Nemotron-3-Diarization"
    device: str = "auto"

    # CTC forced alignment.
    alignment_model: str = "mms-uroman"  # mms-uroman | gujarati-wav2vec2 | auto
    alignment_language: str = "guj"  # ISO 639-3, used for uroman
    mms_model: str = "MahmoudAshraf/mms-300m-1130-forced-aligner"
    gujarati_model: str = "gchhablani/wav2vec2-large-xlsr-gu"
    compute_dtype: str = "float32"
    star_penalty: float = 0.0

    # Assembly / confidence. "speaker-turn" keeps every consecutive run of
    # words from one speaker together, like the ElevenLabs reference sample.
    segmentation_mode: str = "speaker-turn"  # speaker-turn | legacy
    pause_threshold: float = 0.70
    max_segment_seconds: float = 30.0
    speaker_min_coverage: float = 0.50
    speaker_ambiguity_margin: float = 0.10

    # Cache control.
    force: str = "none"
    force_stages: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.audio = Path(self.audio) if self.audio is not None else None
        self.output = Path(self.output)
        self.work_dir = Path(self.work_dir)
        self.report_json = Path(self.report_json) if self.report_json else self.output.with_name(
            "processing_report.json"
        )
        self.report_md = Path(self.report_md) if self.report_md else self.output.with_name(
            "processing_report.md"
        )
        self.hf_asr_model_dir = Path(self.hf_asr_model_dir) if self.hf_asr_model_dir else None
        self.bodhan_base_url = self.bodhan_base_url.rstrip("/")
        self.force_stages = self._parse_force(self.force)

    @staticmethod
    def _parse_force(value: str) -> set[str]:
        if not value or value.lower() == "none":
            return set()
        parts = {p.strip().lower() for p in value.split(",") if p.strip()}
        valid = {"transcription", "diarization", "alignment"}
        if "all" in parts:
            return valid
        unknown = parts - valid
        if unknown:
            raise ValueError(f"Unknown --force stage(s): {', '.join(sorted(unknown))}")
        return parts & valid

    @staticmethod
    def _digest(payload: Any) -> str:
        raw = json.dumps(jsonable(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]

    def transcription_config_digest(self) -> str:
        """Cache key for whichever ASR backend produced the chunk text."""
        return self._digest(
            {
                "version": 2,
                "backend": self.asr_backend,
                "bodhan_base_url": self.bodhan_base_url,
                "bodhan_model": self.bodhan_model,
                "hf_asr_model": self.hf_asr_model,
                "hf_asr_model_dir": str(self.hf_asr_model_dir) if self.hf_asr_model_dir else None,
                "asr_compute_dtype": self.asr_compute_dtype,
                "device": self.device,
                "language": self.language,
                "chunk_seconds": self.bodhan_chunk_seconds,
                "max_chunk_seconds": self.bodhan_max_chunk_seconds,
                "min_chunk_seconds": self.bodhan_min_chunk_seconds,
                "silence_search_seconds": self.bodhan_silence_search_seconds,
            }
        )

    def bodhan_config_digest(self) -> str:
        """Backwards-compatible alias used by older cache/report code."""
        return self.transcription_config_digest()

    def diarization_config_digest(self) -> str:
        return self._digest(
            {
                "version": 1,
                "model": self.diar_model,
                "device": self.device,
                # Offline-style Sortformer configuration from the model card.
                "spkcache_len": 264,
                "fifo_len": 40,
                "chunk_len": 340,
                "chunk_right_context": 40,
                "spkcache_update_period": 300,
            }
        )

    def alignment_config_digest(self) -> str:
        return self._digest(
            {
                "version": 1,
                "strategy": self.alignment_model,
                "language": self.alignment_language,
                "mms_model": self.mms_model,
                "gujarati_model": self.gujarati_model,
                "compute_dtype": self.compute_dtype,
                "star_penalty": self.star_penalty,
            }
        )

    def validate(self) -> None:
        if not self.check and self.audio is None:
            raise ValueError("--audio is required unless --check is used")
        if not self.check and self.audio is not None and not self.audio.is_file():
            raise FileNotFoundError(f"Input audio does not exist: {self.audio}")
        if self.asr_backend not in {"hf-local", "cloud"}:
            raise ValueError("--asr-backend must be hf-local or cloud")
        if self.asr_compute_dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("--asr-compute-dtype must be auto, float32, float16 or bfloat16")
        if self.device not in {"auto", "cuda", "cpu"}:
            raise ValueError("--device must be auto, cuda or cpu")
        if (
            not self.check
            and self.asr_backend == "cloud"
            and not self.bodhan_api_key
            and not self.transcription_cache_only
        ):
            raise ValueError(
                "--asr-backend cloud requires BODHAN_API_KEY in the environment, "
                "or --transcription-cache-only with valid cached responses."
            )
        if not 1.0 <= self.bodhan_max_chunk_seconds <= 30.0:
            raise ValueError(
                "--bodhan-max-chunk-seconds must be in [1, 30] "
                "(Bodhan API limit and local Indic-Transcribe training limit are 30 s)"
            )
        if not 0.5 <= self.bodhan_chunk_seconds <= self.bodhan_max_chunk_seconds:
            raise ValueError("--bodhan-chunk-seconds must be in [0.5, max chunk seconds]")
        if self.bodhan_min_chunk_seconds <= 0 or self.bodhan_min_chunk_seconds > self.bodhan_chunk_seconds:
            raise ValueError("--bodhan-min-chunk-seconds must be > 0 and <= chunk seconds")
        if self.bodhan_workers < 1:
            raise ValueError("--bodhan-workers must be >= 1")
        if self.bodhan_max_retries < 0:
            raise ValueError("--bodhan-max-retries must be >= 0")
        if self.alignment_model not in {"mms-uroman", "gujarati-wav2vec2", "auto"}:
            raise ValueError("--alignment-model must be mms-uroman, gujarati-wav2vec2 or auto")
        if self.compute_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("--compute-dtype must be float32, float16 or bfloat16")
        if not 0.0 <= self.speaker_min_coverage <= 1.0:
            raise ValueError("--speaker-min-coverage must be in [0, 1]")
        if not 0.0 <= self.speaker_ambiguity_margin < 1.0:
            raise ValueError("--speaker-ambiguity-margin must be in [0, 1)")
        if self.segmentation_mode not in {"speaker-turn", "legacy"}:
            raise ValueError("--segmentation-mode must be speaker-turn or legacy")
        if self.pause_threshold < 0:
            raise ValueError("--pause-threshold must be >= 0")
        if self.max_segment_seconds <= 0:
            raise ValueError("--max-segment-seconds must be > 0")

    def public_dict(self) -> dict[str, Any]:
        data = jsonable(asdict(self))
        data.pop("bodhan_api_key", None)
        data["bodhan_api_key_present"] = bool(self.bodhan_api_key)
        data["hf_token_present"] = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
        data["force_stages"] = sorted(self.force_stages)
        return data

    @classmethod
    def from_args(cls, args: Any) -> "Config":
        """Build config from CLI overrides with environment-variable fallbacks."""

        def pick(name: str, env_key: Optional[str], default: Any = DEFAULT_DIGEST_KEEP) -> Any:
            value = getattr(args, name, None)
            if value is not None:
                return value
            if env_key and env_key in os.environ:
                return os.environ[env_key]
            if default is DEFAULT_DIGEST_KEEP:
                return None
            return default

        return cls(
            audio=pick("audio", None),
            output=pick("output", None, Path("outputs/transcript.json")),
            work_dir=pick("work_dir", None, Path("work")),
            report_json=pick("report_json", None),
            report_md=pick("report_md", None),
            check=bool(getattr(args, "check", False)),
            asr_backend=pick("asr_backend", "ASR_BACKEND", "hf-local"),
            hf_asr_model=pick("hf_asr_model", "HF_ASR_MODEL", "bodhan-ai/indic-transcribe-core"),
            hf_asr_model_dir=pick("hf_asr_model_dir", "HF_ASR_MODEL_DIR"),
            asr_compute_dtype=pick("asr_compute_dtype", "ASR_COMPUTE_DTYPE", "auto"),
            language=pick("language", "BODHAN_LANGUAGE", "gu"),
            transcription_cache_only=bool(
                getattr(args, "transcription_cache_only", False)
                or getattr(args, "bodhan_cache_only", False)
            ),
            bodhan_base_url=pick("bodhan_base_url", "BODHAN_BASE_URL", "https://api.bodhan.ai"),
            bodhan_model=pick("bodhan_model", "BODHAN_MODEL", "indic-transcribe"),
            bodhan_api_key=os.environ.get("BODHAN_API_KEY") or None,
            bodhan_chunk_seconds=float(pick("bodhan_chunk_seconds", None, 25.0)),
            bodhan_max_chunk_seconds=float(pick("bodhan_max_chunk_seconds", None, 28.0)),
            bodhan_min_chunk_seconds=float(pick("bodhan_min_chunk_seconds", None, 5.0)),
            bodhan_silence_search_seconds=float(pick("bodhan_silence_search_seconds", None, 3.0)),
            bodhan_workers=int(pick("bodhan_workers", None, 1)),
            bodhan_max_retries=int(pick("bodhan_max_retries", None, 5)),
            bodhan_timeout=float(pick("bodhan_timeout", None, 120.0)),
            bodhan_retry_base=float(pick("bodhan_retry_base", None, 1.5)),
            diar_model=pick("diar_model", "NEMO_DIAR_MODEL", "nvidia/Nemotron-3-Diarization"),
            device=pick("device", "PIPELINE_DEVICE", "auto"),
            alignment_model=pick("alignment_model", "ALIGNMENT_MODEL", "mms-uroman"),
            alignment_language=pick("alignment_language", "ALIGNMENT_LANGUAGE", "guj"),
            mms_model=pick("mms_model", "MMS_ALIGN_MODEL", "MahmoudAshraf/mms-300m-1130-forced-aligner"),
            gujarati_model=pick(
                "gujarati_model", "GUJARATI_ALIGN_MODEL", "gchhablani/wav2vec2-large-xlsr-gu"
            ),
            compute_dtype=pick("compute_dtype", None, "float32"),
            star_penalty=float(pick("star_penalty", None, 0.0)),
            segmentation_mode=pick("segmentation_mode", "SEGMENTATION_MODE", "speaker-turn"),
            pause_threshold=float(pick("pause_threshold", None, 0.70)),
            max_segment_seconds=float(pick("max_segment_seconds", None, 30.0)),
            speaker_min_coverage=float(pick("speaker_min_coverage", None, 0.50)),
            speaker_ambiguity_margin=float(pick("speaker_ambiguity_margin", None, 0.10)),
            force=pick("force", None, "none"),
        )
