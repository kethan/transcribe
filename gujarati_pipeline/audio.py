from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .types import AudioInfo, AudioSpan, StageError, jsonable

SAMPLE_RATE = 16_000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 320 samples, same as wav2vec2 stride
SILENCE_PAD_SECONDS = 0.15


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _wav_params(path: Path) -> dict[str, Any]:
    try:
        with wave.open(str(path), "rb") as wf:
            return {
                "channels": wf.getnchannels(),
                "sample_width": wf.getsampwidth(),
                "framerate": wf.getframerate(),
                "frames": wf.getnframes(),
                "comptype": wf.getcomptype(),
            }
    except (wave.Error, OSError) as exc:
        raise StageError(f"Cannot read WAV header of {path}: {exc}") from exc


def is_canonical_wav(path: Path) -> bool:
    try:
        p = _wav_params(path)
    except StageError:
        return False
    return (
        p["channels"] == 1
        and p["sample_width"] == 2
        and p["framerate"] == SAMPLE_RATE
        and p["comptype"] == "NONE"
    )


def read_pcm16_wav(path: Path) -> np.ndarray:
    """Read a mono PCM16 WAV as a little-endian int16 NumPy array."""
    p = _wav_params(path)
    if p["channels"] != 1 or p["sample_width"] != 2:
        raise StageError(f"Expected mono PCM16 WAV, got {p}: {path}")
    with wave.open(str(path), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype="<i2").copy()


def write_pcm16_wav(path: Path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(samples, dtype="<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(arr.tobytes())


def prepare_canonical_audio(config: Config) -> AudioInfo:
    """Create a cached 16 kHz mono PCM16 WAV; the original file is never modified."""
    assert config.audio is not None
    original = config.audio.resolve()
    audio_hash = sha256_file(original)
    audio_dir = config.work_dir / "audio" / audio_hash[:20]
    canonical = audio_dir / "audio_16k_mono.wav"
    meta_path = audio_dir / "meta.json"

    if (
        meta_path.is_file()
        and canonical.is_file()
        and not config.force_stages
    ):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("audio_hash") == audio_hash:
                return AudioInfo(
                    original_path=original,
                    audio_hash=audio_hash,
                    canonical_path=Path(meta["canonical_path"]),
                    sample_rate=int(meta["sample_rate"]),
                    channels=int(meta["channels"]),
                    sample_width_bytes=int(meta["sample_width_bytes"]),
                    frames=int(meta["frames"]),
                    duration=float(meta["duration"]),
                    conversion=str(meta.get("conversion", "cached")),
                )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass  # regenerate a corrupt cache entry

    audio_dir.mkdir(parents=True, exist_ok=True)
    if is_canonical_wav(original):
        shutil.copyfile(original, canonical)
        conversion = "copy (input already 16 kHz mono PCM16 WAV)"
    else:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise StageError(
                "ffmpeg is required to decode/resample non-WAV or non-16 kHz input. "
                "Install it with: apt-get install -y ffmpeg"
            )
        cmd = [
            ffmpeg,
            "-y",
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            str(original),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(canonical),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not canonical.is_file():
            raise StageError(
                "ffmpeg failed to create the 16 kHz mono WAV:\n"
                + (proc.stderr.strip() or "(no stderr)")
            )
        conversion = "ffmpeg decode/downmix/resample to 16 kHz mono PCM16 WAV"

    p = _wav_params(canonical)
    if p["framerate"] != SAMPLE_RATE or p["channels"] != 1 or p["sample_width"] != 2:
        raise StageError(f"Canonical WAV has unexpected parameters: {p}")
    info = AudioInfo(
        original_path=original,
        audio_hash=audio_hash,
        canonical_path=canonical,
        sample_rate=p["framerate"],
        channels=p["channels"],
        sample_width_bytes=p["sample_width"],
        frames=p["frames"],
        duration=p["frames"] / p["framerate"],
        conversion=conversion,
    )
    meta_path.write_text(
        json.dumps(jsonable(info), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return info


def frame_rms(samples: np.ndarray) -> np.ndarray:
    """Short-term RMS in 20 ms frames, normalised to [-1, 1] amplitude."""
    x = np.asarray(samples, dtype=np.float32) / 32768.0
    pad = (-x.size) % FRAME_SAMPLES
    if pad:
        x = np.pad(x, (0, pad))
    frames = x.reshape(-1, FRAME_SAMPLES)
    return np.sqrt(np.mean(frames * frames, axis=1) + 1e-12).astype(np.float32)


def _energy_threshold(rms: np.ndarray) -> float:
    if rms.size == 0:
        return 0.003
    noise = float(np.percentile(rms, 10))
    peak = float(np.percentile(rms, 95))
    threshold = max(noise * 2.5, 0.003)
    # Guard against a recording with almost no silence: keep at least the
    # louder 65% of frames active so we do not declare everything silence.
    return float(min(threshold, max(peak * 0.35, 0.003)))


def _best_boundary_frame(
    rms: np.ndarray,
    start_frame: int,
    end_frame: int,
    target_frame: int,
    search_frames: int,
) -> tuple[int, bool]:
    """Pick the quietest frame near a nominal chunk boundary.

    Returns (frame_index, found_quiet_frame). A hard cut is reported by the
    caller when no frame falls below the energy threshold.
    """
    lo = max(start_frame + 1, target_frame - search_frames)
    hi = min(end_frame, target_frame + search_frames)
    if hi <= lo:
        return max(start_frame + 1, min(end_frame, target_frame)), False
    window = rms[lo : hi + 1]
    # Slight distance penalty makes the result deterministic and keeps chunks
    # close to the requested length when the whole window is equally quiet.
    positions = np.arange(lo, hi + 1, dtype=np.float32)
    distance = np.abs(positions - target_frame) / max(1, search_frames)
    score = window + 0.0005 * distance
    idx = int(np.argmin(score))
    return lo + idx, bool(window[idx] <= np.percentile(rms, 35))


def _span_key(audio_hash: str, start_sample: int, end_sample: int, config: Config, suffix: str = "") -> str:
    payload = f"{audio_hash}|{start_sample}|{end_sample}|{config.transcription_config_digest()}|{suffix}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _make_span(
    *,
    index: int,
    start_sample: int,
    end_sample: int,
    samples: np.ndarray,
    audio_hash: str,
    config: Config,
    suffix: str = "",
) -> AudioSpan:
    start = start_sample / SAMPLE_RATE
    end = end_sample / SAMPLE_RATE
    key = _span_key(audio_hash, start_sample, end_sample, config, suffix)
    path = config.work_dir / "transcription" / "audio" / f"{key}.wav"
    if not path.is_file():
        write_pcm16_wav(path, samples[start_sample:end_sample])
    return AudioSpan(
        index=index,
        start_sample=start_sample,
        end_sample=end_sample,
        start=start,
        end=end,
        path=path,
        key=key,
        audio_hash=audio_hash,
    )


def plan_spans(
    info: AudioInfo,
    samples: np.ndarray,
    config: Config,
    warnings: list[str],
) -> list[AudioSpan]:
    """Silence-aware non-overlapping Bodhan chunks with exact original offsets.

    Each chunk is trimmed of leading/trailing silence for the API, but its
    absolute ``start``/``end`` values still refer to the canonical WAV timeline.
    """
    rms = frame_rms(samples)
    threshold = _energy_threshold(rms)
    active = rms > threshold
    total_samples = samples.size
    if total_samples == 0 or not bool(np.any(active)):
        warnings.append("No speech activity detected by the energy VAD; no Bodhan chunks were submitted.")
        return []

    max_samples = int(config.bodhan_max_chunk_seconds * SAMPLE_RATE)
    target_samples = int(config.bodhan_chunk_seconds * SAMPLE_RATE)
    min_samples = int(config.bodhan_min_chunk_seconds * SAMPLE_RATE)
    search_samples = int(config.bodhan_silence_search_seconds * SAMPLE_RATE)
    pad_samples = int(SILENCE_PAD_SECONDS * SAMPLE_RATE)
    hard_cuts = 0
    silent_skips = 0
    spans: list[AudioSpan] = []

    start_sample = 0
    while start_sample < total_samples:
        remaining = total_samples - start_sample
        if remaining <= max_samples:
            boundary_sample = total_samples
        else:
            target_sample = min(start_sample + target_samples, start_sample + max_samples)
            bf, quiet = _best_boundary_frame(
                rms,
                start_frame=start_sample // FRAME_SAMPLES,
                end_frame=(start_sample + max_samples) // FRAME_SAMPLES,
                target_frame=target_sample // FRAME_SAMPLES,
                search_frames=max(1, search_samples // FRAME_SAMPLES),
            )
            boundary_sample = min(total_samples, max(start_sample + min_samples, bf * FRAME_SAMPLES))
            if not quiet:
                hard_cuts += 1

        boundary_sample = min(boundary_sample, total_samples)
        if boundary_sample <= start_sample:
            break

        f0 = start_sample // FRAME_SAMPLES
        f1 = math.ceil(boundary_sample / FRAME_SAMPLES)
        local_active = active[f0:f1]
        active_positions = np.flatnonzero(local_active)
        if active_positions.size == 0:
            silent_skips += 1
            start_sample = boundary_sample
            continue

        first_frame = f0 + int(active_positions[0])
        last_frame = f0 + int(active_positions[-1])
        trim_start = max(start_sample, first_frame * FRAME_SAMPLES - pad_samples)
        trim_end = min(boundary_sample, (last_frame + 1) * FRAME_SAMPLES + pad_samples)
        if trim_end - trim_start < int(0.05 * SAMPLE_RATE):
            silent_skips += 1
            start_sample = boundary_sample
            continue

        spans.append(
            _make_span(
                index=len(spans),
                start_sample=trim_start,
                end_sample=trim_end,
                samples=samples,
                audio_hash=info.audio_hash,
                config=config,
            )
        )
        start_sample = boundary_sample

    if hard_cuts:
        warnings.append(
            f"{hard_cuts} Bodhan chunk boundary/boundaries had no low-energy point "
            "inside the search window; a word may be split at that boundary."
        )
    if silent_skips:
        warnings.append(f"Skipped {silent_skips} silent chunk(s); no transcript words were dropped.")
    return spans


def split_span(
    span: AudioSpan,
    config: Config,
    warnings: list[str],
    canonical_samples: np.ndarray,
) -> list[AudioSpan]:
    """Split a span after an ASR duration error or repetition-loop retry.

    The split point prefers a low-energy frame near the midpoint; it never
    invents or duplicates audio. Both halves keep absolute original offsets.
    """
    local = read_pcm16_wav(span.path)
    rms = frame_rms(local)
    total = local.size
    if total < 2 * SAMPLE_RATE:
        raise StageError(f"Cannot split a {span.duration:.2f}s ASR span; reduce --bodhan-chunk-seconds.")
    midpoint = total // 2
    boundary_frame, quiet = _best_boundary_frame(
        rms,
        start_frame=0,
        end_frame=total // FRAME_SAMPLES,
        target_frame=midpoint // FRAME_SAMPLES,
        search_frames=max(1, int(2.0 * SAMPLE_RATE) // FRAME_SAMPLES),
    )
    boundary = min(total - 1, max(1, boundary_frame * FRAME_SAMPLES))
    if not quiet:
        warnings.append(f"ASR retry split of {span.key} used a hard midpoint cut; check the boundary text.")
    halves = [(0, boundary), (boundary, total)]
    out: list[AudioSpan] = []
    for i, (a, b) in enumerate(halves):
        abs_start = span.start_sample + a
        abs_end = span.start_sample + b
        out.append(
            _make_span(
                index=span.index * 10 + i,
                start_sample=abs_start,
                end_sample=abs_end,
                samples=canonical_samples,
                audio_hash=span.audio_hash,
                config=config,
                suffix=f"split-{abs_start}-{abs_end}",
            )
        )
    return out


def unit_samples(span: AudioSpan, start: float, end: float) -> np.ndarray:
    """Read an absolute-time sub-window from a chunk WAV."""
    samples = read_pcm16_wav(span.path)
    rel_start = int(round((start - span.start) * SAMPLE_RATE))
    rel_end = int(round((end - span.start) * SAMPLE_RATE))
    if rel_start < 0 or rel_end > samples.size or rel_start >= rel_end:
        raise StageError(
            f"Alignment window [{start:.3f}, {end:.3f}] is outside chunk "
            f"[{span.start:.3f}, {span.end:.3f}] ({samples.size} samples)."
        )
    return samples[rel_start:rel_end]
