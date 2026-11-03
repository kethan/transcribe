from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

from .config import Config
from .runtime import gpu_info, import_torch, package_version, resolve_device
from .types import AudioInfo, DiarInterval, StageError, jsonable

# Offline-style configuration from the Nemotron-3-Diarization model card
# (all values are measured in 80 ms frames).
OFFLINE_SORTFORMER_PARAMS = {
    "spkcache_len": 264,
    "fifo_len": 40,
    "chunk_len": 340,
    "chunk_right_context": 40,
    "spkcache_update_period": 300,
}


def verify_environment(config: Config, info: AudioInfo | None = None) -> dict[str, Any]:
    """Verify NeMo/Python/PyTorch/audio compatibility before any model download."""
    result: dict[str, Any] = {
        "ok": True,
        "errors": [],
        "warnings": [],
        "python_version": sys.version.split()[0],
        "nemo_toolkit_version": package_version("nemo-toolkit"),
        "gpu": gpu_info(),
    }
    if sys.version_info < (3, 12):
        result["warnings"].append(
            "NeMo Speech 3.0 documentation requires Python 3.12+; "
            f"detected Python {sys.version_info.major}.{sys.version_info.minor}."
        )
    gpu = result["gpu"]
    if not gpu.get("torch_available"):
        result["errors"].append(str(gpu.get("error")))
    elif not gpu.get("cuda_available"):
        result["warnings"].append(
            "CUDA GPU not detected. Sortformer is GPU-optimised; CPU inference may be very slow."
        )
    try:
        from nemo.collections.asr.models import SortformerEncLabelModel  # noqa: F401

        result["sortformer_class_available"] = True
    except Exception as exc:
        result["sortformer_class_available"] = False
        result["errors"].append(
            "nemo.collections.asr.models.SortformerEncLabelModel could not be imported. "
            "Install nemo-toolkit[asr]==3.0.0 (or the matching NeMo Speech 3.0 build). "
            f"Import error: {exc}"
        )
    if info is not None:
        if info.sample_rate != 16000 or info.channels != 1 or info.sample_width_bytes != 2:
            result["errors"].append(
                f"Canonical audio must be 16 kHz mono PCM16; got {info.sample_rate} Hz, "
                f"{info.channels} channel(s), {info.sample_width_bytes * 8}-bit."
            )
        result["audio_duration_seconds"] = info.duration
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if "/" in config.diar_model and not Path(config.diar_model).is_file() and not hf_token:
        result["warnings"].append(
            "HF_TOKEN is not set. The Nemotron-3-Diarization model card notes that a "
            "Hugging Face token may be required for from_pretrained()."
        )
    if config.asr_backend == "hf-local":
        for pkg in ("huggingface_hub", "sentencepiece", "soundfile", "torchaudio", "transformers"):
            if package_version(pkg) is None:
                result["warnings"].append(
                    f"{pkg} is not installed; the local Bodhan HF ASR wrapper needs it."
                )
        if not config.hf_asr_model_dir and not hf_token:
            result["warnings"].append(
                "Local Bodhan HF repo is gated and HF_TOKEN is not set. Accept the model "
                "license and set HF_TOKEN, or pass --hf-asr-model-dir."
            )
    result["ok"] = not result["errors"]
    return result


def _parse_segment(item: Any) -> tuple[float, float, str] | None:
    """Parse NeMo 3.0 Sortformer output and a few compatible shapes."""
    try:
        if isinstance(item, str):
            parts = item.split()
            if not parts:
                return None
            if parts[0].upper() == "SPEAKER" and len(parts) >= 8:
                start = float(parts[3])
                end = start + float(parts[4])
                speaker = str(parts[7])
            elif len(parts) >= 3:
                start = float(parts[0])
                end = float(parts[1])
                speaker = str(parts[2])
            else:
                return None
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            start = float(item[0])
            end = float(item[1])
            speaker = str(item[2])
        elif isinstance(item, dict):
            start = float(
                item.get("start", item.get("Start", item.get("begin", item.get("Begin"))))
            )
            end = float(item.get("end", item.get("End", item.get("finish"))))
            speaker = str(
                item.get("speaker", item.get("speaker_id", item.get("Speaker", item.get("Speaker_ID"))))
            )
        else:
            return None
    except (TypeError, ValueError, KeyError):
        return None
    if not (math.isfinite(start) and math.isfinite(end) and start < end):
        return None
    return start, end, speaker


def _flatten_segments(result: Any) -> list[Any]:
    """NeMo 3.0 returns list[list[str]] like '0.50 3.12 speaker_0'."""
    if isinstance(result, tuple):  # include_tensor_outputs=True shape
        result = result[0]
    if not isinstance(result, list):
        raise StageError(f"Unexpected NeMo diarization return type: {type(result).__name__}")
    if result and isinstance(result[0], list):
        result = result[0]
    return result


def _stable_speaker_ids(raw_intervals: list[tuple[float, float, str]]) -> dict[str, str]:
    first_seen: dict[str, float] = {}
    for start, _end, speaker in raw_intervals:
        if speaker not in first_seen or start < first_seen[speaker]:
            first_seen[speaker] = start
    ordered = sorted(first_seen, key=lambda s: (first_seen[s], s))
    return {raw: f"speaker_{i}" for i, raw in enumerate(ordered)}


def _cache_path(config: Config, info: AudioInfo) -> Path:
    digest = config.diarization_config_digest()
    return config.work_dir / "diarization" / info.audio_hash[:20] / f"{digest}.json"


def _load_cache(path: Path, config: Config, info: AudioInfo) -> tuple[list[DiarInterval], list[str], dict[str, Any]] | None:
    if not path.is_file():
        return None
    try:
        cache = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if (
        cache.get("version") != 1
        or cache.get("audio_hash") != info.audio_hash
        or cache.get("config_digest") != config.diarization_config_digest()
        or not isinstance(cache.get("intervals"), list)
    ):
        return None
    intervals = [
        DiarInterval(
            start=float(x["start"]),
            end=float(x["end"]),
            speaker_id=str(x["speaker_id"]),
            raw_speaker=str(x.get("raw_speaker", "")),
        )
        for x in cache["intervals"]
    ]
    speaker_ids = [str(x) for x in cache.get("speaker_ids", [])]
    meta = cache.get("meta", {})
    return intervals, speaker_ids, meta


def run_diarization(
    info: AudioInfo,
    config: Config,
    warnings: list[str],
) -> tuple[list[DiarInterval], list[str], dict[str, Any]]:
    env = verify_environment(config, info)
    if not env["ok"]:
        raise StageError("NeMo environment check failed:\n- " + "\n- ".join(env["errors"]))
    warnings.extend(env["warnings"])
    cache_path = _cache_path(config, info)
    if "diarization" not in config.force_stages:
        cached = _load_cache(cache_path, config, info)
        if cached is not None:
            print(f"[NeMo] Using cached diarization: {len(cached[0])} intervals, {len(cached[1])} speakers")
            return cached

    torch = import_torch()
    device = resolve_device(config.device, torch)
    try:
        from nemo.collections.asr.models import SortformerEncLabelModel
    except Exception as exc:
        raise StageError(f"Cannot import NVIDIA NeMo Sortformer: {exc}") from exc

    print(f"[NeMo] Loading diarization model {config.diar_model} on {device}")
    try:
        model_path = Path(config.diar_model)
        if model_path.is_file() and model_path.suffix == ".nemo":
            model = SortformerEncLabelModel.restore_from(
                restore_path=str(model_path), map_location=torch.device(device), strict=False
            )
        else:
            model = SortformerEncLabelModel.from_pretrained(
                model_name=config.diar_model, map_location=torch.device(device)
            )
        model.eval()
        modules = getattr(model, "sortformer_modules", None)
        if modules is None:
            raise StageError("Loaded model has no sortformer_modules; this is not a Sortformer checkpoint.")
        for key, value in OFFLINE_SORTFORMER_PARAMS.items():
            if not hasattr(modules, key):
                raise StageError(f"Sortformer module is missing {key}; incompatible NeMo/model version.")
            setattr(modules, key, value)
        checker = getattr(model, "_check_streaming_parameters", None) or getattr(
            modules, "_check_streaming_parameters", None
        )
        if checker is None:
            raise StageError("Sortformer model has no _check_streaming_parameters(); incompatible NeMo version.")
        checker()
        print("[NeMo] Running offline Sortformer diarization on the canonical 16 kHz WAV")
        result = model.diarize(
            audio=[str(info.canonical_path)],
            batch_size=1,
            num_workers=0,
            verbose=True,
        )
    except StageError:
        raise
    except Exception as exc:
        raise StageError(
            "NVIDIA NeMo diarization failed. Verify nemo-toolkit[asr]==3.0.0, "
            "a CUDA-enabled torch>=2.7 build, the 16 kHz mono WAV, and HF_TOKEN/access "
            f"to {config.diar_model}. Original error: {exc}"
        ) from exc

    try:
        raw_segments = _flatten_segments(result)
    except StageError as exc:
        raise StageError(f"Cannot interpret NeMo diarization output: {exc}") from exc
    parsed: list[tuple[float, float, str]] = []
    skipped = 0
    for item in raw_segments:
        triple = _parse_segment(item)
        if triple is None:
            skipped += 1
            continue
        start, end, speaker = triple
        start = max(0.0, min(info.duration, start))
        end = max(0.0, min(info.duration, end))
        if start < end:
            parsed.append((start, end, speaker))
    if skipped:
        warnings.append(f"NeMo returned {skipped} unparsable segment line(s); they were ignored.")
    parsed.sort(key=lambda x: (x[0], x[1], x[2]))
    mapping = _stable_speaker_ids(parsed)
    intervals = [
        DiarInterval(start=start, end=end, speaker_id=mapping[speaker], raw_speaker=speaker)
        for start, end, speaker in parsed
    ]
    speaker_ids = sorted(set(mapping.values()), key=lambda s: int(s.split("_")[-1]))
    if not intervals:
        warnings.append("NeMo returned no speaker intervals; all words will receive speaker_id=null.")
    meta = {
        "model": config.diar_model,
        "device": device,
        "offline_params": OFFLINE_SORTFORMER_PARAMS,
        "raw_segment_count": len(raw_segments),
        "parsed_interval_count": len(intervals),
        "skipped_segments": skipped,
        "speaker_mapping": mapping,
        "nemo_version": env.get("nemo_toolkit_version"),
        "output_shape": "list[list[str]] 'start end speaker_N' (NeMo 3.0.0)",
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "version": 1,
                "audio_hash": info.audio_hash,
                "config_digest": config.diarization_config_digest(),
                "duration": info.duration,
                "speaker_ids": speaker_ids,
                "intervals": jsonable(intervals),
                "meta": meta,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"[NeMo] Detected {len(speaker_ids)} speaker(s) and {len(intervals)} interval(s); "
        "overlapping intervals are retained"
    )
    return intervals, speaker_ids, meta
