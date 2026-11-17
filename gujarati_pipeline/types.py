from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


class PipelineError(Exception):
    """Base class for errors that should produce a readable stage-failure report."""


class StageError(PipelineError):
    """An error attributable to a named pipeline stage."""


class ValidationError(PipelineError):
    """The assembled JSON violates its own contract."""


class AlignmentError(PipelineError):
    """CTC forced alignment could not produce trustworthy word timestamps."""


class AlignmentVocabularyError(AlignmentError):
    """The transcript contains characters the selected CTC model cannot score."""


def jsonable(value: Any) -> Any:
    """Recursively turn dataclasses / Paths into JSON-serialisable values."""
    if hasattr(value, "__dataclass_fields__"):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass
class AudioInfo:
    original_path: Path
    audio_hash: str
    canonical_path: Path
    sample_rate: int
    channels: int
    sample_width_bytes: int
    frames: int
    duration: float
    conversion: str

    @property
    def total_samples(self) -> int:
        return self.frames * self.channels


@dataclass
class AudioSpan:
    index: int
    start_sample: int
    end_sample: int
    start: float
    end: float
    path: Path
    key: str
    audio_hash: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class TimedWord:
    text: str
    start: float
    end: float


@dataclass
class TranscriptUnit:
    """One Bodhan text span tied to an absolute window on the canonical WAV.

    If ``words`` is populated, Bodhan itself supplied reliable word timestamps
    (absolute seconds). Otherwise forced alignment is required.
    """

    chunk_index: int
    unit_index: int
    key: str
    text: str
    start: float
    end: float
    span: AudioSpan
    words: Optional[list[TimedWord]] = None
    source: str = "bodhan.text"


@dataclass
class BodhanChunk:
    span: AudioSpan
    key: str
    response: dict[str, Any]
    units: list[TranscriptUnit]
    text: str
    has_word_times: bool


@dataclass
class DiarInterval:
    start: float
    end: float
    speaker_id: str
    raw_speaker: str


@dataclass
class AlignedWord:
    text: str
    start: float
    end: float
    chunk_index: int
    unit_index: int
    alignment_model: str
    source: str
    chunk_key: str = ""
    speaker_id: Optional[str] = None
    original_order: int = 0


@dataclass
class ProcessingReport:
    status: str = "ok"
    stage: str = ""
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
