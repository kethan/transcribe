#!/usr/bin/env python3
"""CPU-only plain-text transcription with an INT8 ONNX Bodhan derivative.

This is intentionally separate from the GPU/Kaggle pipeline. It produces UTF-8
transcript text (and a provenance JSON), not diarization, word timestamps,
speaker turns, or the strict ``transcript.json`` schema.

Model: schamp007/verbatim-hindi-asr-core-sc, a pinned community re-export of
Bodhan Indic-Transcribe-Core with the same INT8 encoder/weights and a KV-cache
decoder. The exact linked ``adidsh`` port is slower because it recomputes the
whole token prefix at every decoding step.

Attribution required by the model license:
  Built with Indic-Transcribe-Core from Bodhan AI / AI4Bharat.
  Base model: Canary-1B-v2 developed by NVIDIA Corporation (CC-BY-4.0).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# --- Pinned model -----------------------------------------------------------

MODEL_ID = "schamp007/verbatim-hindi-asr-core-sc"
MODEL_REVISION = "e17f007b2906828614334050e4693e1f4029871c"
SOURCE_MODEL = "bodhan-ai/indic-transcribe-core"
BASE_MODEL = "nvidia/canary-1b-v2"

REQUIRED_MODEL_FILES = (
    "encoder.onnx",
    "cross_kv.onnx",
    "decoder_kv.onnx",
    "bodhan_vocab.json",
)

# Hashes are model-file integrity checks, not transcript content.
EXPECTED_SHA256: dict[str, str] = {
    "encoder.onnx": "d25b1a9fcba2888c44428a0df46a2574b17692e937124a8f6328acf330e29de1",
    "cross_kv.onnx": "72bf71daee584dd5b66812c3df38c4de46e481f5acedb29718848b59eb95cce3",
    "decoder_kv.onnx": "98b90cea1096fb0adeabeab4c71a78b0b74d8508422b1e7d4172aa2bbd147325",
    "bodhan_vocab.json": "571fa2fac0efa0cb7c4196cb17e899560cf0c91510f7f2d739243d3766d5fb44",
    "LICENSE": "b39f2a9cb2d5ace3a03f4425cbc6db38ebd9a53ab7a7648d67dc1e0998ad75c2",
    "NOTICE.md": "9a197aafe35f84e96b62d782d545a535423f05ba51b0bb78333c8393316586b2",
    "README.md": "4db42ab1c45c126ee46f98d00b3b72da7e91966081cb239c9fca368685a146da",
}

DOWNLOAD_FILES = tuple(EXPECTED_SHA256) + ("SHA256SUMS",)

# Core model card languages. The KV derivative was only measured on Hindi.
CORE_LANGUAGES: tuple[str, ...] = (
    "as", "bn", "brx", "doi", "gu", "hi", "kn", "ks", "kok", "mai", "ml",
    "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur",
    "en", "bho", "bhb",
)

DEFAULT_LANGUAGE = "hi"
DEFAULT_THREADS = 4
DEFAULT_MAX_NEW_TOKENS = 512
SAMPLE_RATE = 16_000
FRAME_SAMPLES = 320  # 20 ms at 16 kHz
LAYERS = 24
HEADS = 8
HEAD_DIM = 128
VOCAB_SIZE = 7_152
EOS_ID = 3
PAD_ID = 2
MAX_POSITIONS = 1_024


# --- Small utilities --------------------------------------------------------

def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """Return the SHA-256 digest of UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def default_model_dir() -> Path:
    """Keep ~1.57 GB of ONNX weights out of the Git checkout."""
    override = os.environ.get("LAPTOP_ASR_MODEL_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "transcribe-laptop" / "verbatim-hindi-asr-core-sc"


def parse_sha256sums(path: Path) -> dict[str, str]:
    """Parse a standard ``sha256sum`` file."""
    result: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        # ``*filename`` marks binary mode in some sha256sum implementations.
        result[parts[1].lstrip("*")] = parts[0].lower()
    return result


def ensure_model_files(
    model_dir: Path,
    *,
    offline: bool = False,
    force_download: bool = False,
) -> dict[str, str]:
    """Download (or verify) the pinned ONNX model files."""
    if offline and force_download:
        raise ValueError("--offline and --force-download cannot be combined")

    model_dir = Path(model_dir).expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: model_dir / name for name in DOWNLOAD_FILES}

    if force_download:
        for path in paths.values():
            if path.is_file():
                path.unlink()

    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing and offline:
        raise FileNotFoundError(
            f"Offline mode is missing {', '.join(missing)} in {model_dir}. "
            "Run once without --offline or omit --offline for this run."
        )

    if missing:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover - install-time error
            raise RuntimeError(
                "huggingface-hub is required to download the model. "
                "Install laptop/requirements.txt first."
            ) from exc
        try:
            snapshot_download(
                repo_id=MODEL_ID,
                revision=MODEL_REVISION,
                allow_patterns=list(DOWNLOAD_FILES),
                local_dir=str(model_dir),
            )
        except Exception as exc:  # pragma: no cover - network/runtime error
            raise RuntimeError(
                f"Could not download {MODEL_ID} at {MODEL_REVISION}: {exc}"
            ) from exc

    still_missing = [name for name, path in paths.items() if not path.is_file()]
    if still_missing:
        raise FileNotFoundError(
            f"Model download is incomplete: {', '.join(still_missing)} in {model_dir}"
        )

    for name, expected in EXPECTED_SHA256.items():
        actual = sha256_file(paths[name])
        if actual != expected:
            raise RuntimeError(
                f"Checksum mismatch for {paths[name]}. "
                f"Expected {expected}, got {actual}. "
                "Delete that file or rerun with --force-download."
            )

    published = parse_sha256sums(paths["SHA256SUMS"])
    for name, expected in EXPECTED_SHA256.items():
        if name in published and published[name] != expected:
            raise RuntimeError(
                f"Published SHA256SUMS disagrees with the pinned hash for {name}."
            )

    return {name: str(paths[name]) for name in REQUIRED_MODEL_FILES}


# --- Audio and features -----------------------------------------------------

def _ffmpeg_load(path: Path) -> np.ndarray | None:
    """Fallback decoder for MP3s that soundfile/librosa cannot open."""
    command = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-map", "0:a:0", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "f32le", "-",
    ]
    try:
        completed = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
        )
    except FileNotFoundError:
        return None
    if completed.returncode != 0:
        return None
    return np.frombuffer(completed.stdout, dtype="<f4").copy()


def load_audio(path: Path) -> np.ndarray:
    """Decode audio to contiguous 16 kHz mono float32 samples."""
    audio_path = Path(path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    decode_error: Exception | None = None
    try:
        import librosa
        audio, _ = librosa.load(
            str(audio_path), sr=SAMPLE_RATE, mono=True, dtype=np.float32
        )
    except Exception as exc:
        decode_error = exc
        audio = _ffmpeg_load(audio_path)
        if audio is None:
            raise RuntimeError(
                f"Could not decode {audio_path} with librosa ({exc}). "
                "Install ffmpeg and put it on PATH, or convert the file to "
                "16 kHz mono WAV first."
            ) from exc

    array = np.asarray(audio)
    if array.ndim == 2:
        array = array.mean(axis=0)
    array = np.asarray(array, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise ValueError(f"Audio file contains no samples: {audio_path}")
    if not np.isfinite(array).all():
        raise ValueError(f"Audio contains NaN or infinity: {audio_path}")
    if decode_error is not None:
        print(
            "Decoded audio with ffmpeg because the Python decoder failed.",
            file=sys.stderr,
        )
    return np.ascontiguousarray(array, dtype=np.float32)


def mel_features(samples: np.ndarray) -> np.ndarray:
    """Compute the exact front-end described by the KV-cache model card.

    16 kHz mono; <1 s clips are zero-padded to 1 s; pre-emphasis 0.97;
    reflect-pad 256; STFT n_fft=512/hop=160/symmetric Hann(400) centred in
    512; power spectrum; 128-bin Slaney mel (fmin=0, fmax=8000); ln(x+2^-24);
    then per-bin normalization over the clip with unbiased std + 1e-5.
    """
    import librosa

    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if audio.size < SAMPLE_RATE:
        pad = SAMPLE_RATE - audio.size
        left = pad // 2
        right = pad - left
        audio = np.pad(audio, (left, right), mode="constant")

    # NeMo-style pre-emphasis: preserve sample 0, filter the remainder.
    emphasized = np.empty_like(audio, dtype=np.float32)
    emphasized[0] = audio[0]
    emphasized[1:] = audio[1:] - np.float32(0.97) * audio[:-1]

    window = librosa.filters.get_window("hann", 400, fftbins=True)
    stft = librosa.stft(
        emphasized,
        n_fft=512,
        hop_length=160,
        win_length=400,
        window=window,
        center=True,
        pad_mode="reflect",
    )
    power = np.abs(stft) ** 2
    mel_basis = librosa.filters.mel(
        sr=SAMPLE_RATE,
        n_fft=512,
        n_mels=128,
        fmin=0,
        fmax=SAMPLE_RATE / 2,
        htk=False,
        norm="slaney",
    )
    mel = np.dot(mel_basis, power)
    log_mel = np.log(mel + (2.0 ** -24))
    mean = log_mel.mean(axis=1, keepdims=True)
    std = log_mel.std(axis=1, ddof=1, keepdims=True)
    features = (log_mel - mean) / (std + 1e-5)
    return np.ascontiguousarray(features[None, :, :], dtype=np.float32)


def frame_rms(samples: np.ndarray, block_frames: int = 8192) -> np.ndarray:
    """RMS energy for non-overlapping 20 ms frames without a giant temp array."""
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    frame_count = audio.size // FRAME_SAMPLES
    if frame_count == 0:
        return np.zeros(1, dtype=np.float32)

    result = np.empty(frame_count, dtype=np.float32)
    for start in range(0, frame_count, block_frames):
        end = min(start + block_frames, frame_count)
        frames = audio[start * FRAME_SAMPLES:end * FRAME_SAMPLES]
        frames = frames.reshape(-1, FRAME_SAMPLES)
        np.mean(frames * frames, axis=1, out=result[start:end])
        result[start:end] += np.float32(1e-12)
        np.sqrt(result[start:end], out=result[start:end])
    return result


def auto_silence_threshold(rms: np.ndarray) -> float:
    """Conservative RMS threshold; 0.0 in the CLI means use this."""
    if rms.size == 0:
        return 1e-5
    return max(1e-5, float(np.quantile(rms, 0.25)) * 2.0)


def plan_chunks(
    samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    *,
    max_seconds: float = 28.0,
    min_seconds: float = 1.0,
    search_seconds: float = 3.0,
    silence_threshold: float = 0.0,
) -> tuple[list[dict[str, Any]], float]:
    """Split contiguous audio at quiet points into <=30 s ASR windows."""
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"plan_chunks expects {SAMPLE_RATE} Hz audio")
    if not 0 < max_seconds <= 30.0:
        raise ValueError("max_seconds must be in (0, 30]")
    if not 0.5 <= min_seconds < max_seconds:
        raise ValueError("min_seconds must be >= 0.5 and smaller than max_seconds")
    if not 0 <= search_seconds < max_seconds - min_seconds:
        raise ValueError("search_seconds must be in [0, max_seconds-min_seconds)")
    if silence_threshold < 0:
        raise ValueError("silence_threshold must be >= 0 (0 means auto)")

    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    rms = frame_rms(audio)
    threshold = (
        float(silence_threshold)
        if silence_threshold > 0
        else auto_silence_threshold(rms)
    )
    frame_duration = FRAME_SAMPLES / sample_rate
    max_frames = max(1, int(max_seconds / frame_duration))
    min_frames = max(1, int(min_seconds / frame_duration))
    search_frames = int(search_seconds / frame_duration)
    total_samples = audio.size
    total_frames = min(rms.size, total_samples // FRAME_SAMPLES)

    chunks: list[dict[str, Any]] = []
    start_sample = 0
    start_frame = 0

    while start_sample < total_samples:
        remaining_samples = total_samples - start_sample
        if remaining_samples <= max_frames * FRAME_SAMPLES:
            # Final chunk: include any trailing partial frame.
            end_sample = total_samples
            boundary_rms: float | None = None
            cut_at_pause: bool | None = None
        else:
            end_frame_limit = min(start_frame + max_frames, total_frames)
            search_start = max(
                start_frame + min_frames,
                end_frame_limit - search_frames,
            )
            search_end = min(end_frame_limit, total_frames)
            if search_end < search_start:
                search_start = search_end

            boundary_frames = np.arange(
                search_start, search_end + 1, dtype=np.int64
            )
            # Energy of the frame immediately before each candidate boundary.
            before = np.maximum(boundary_frames - 1, 0)
            before = np.minimum(before, max(0, total_frames - 1))
            energies = rms[before] if total_frames > 0 else np.zeros_like(
                boundary_frames, dtype=np.float32
            )
            best = int(np.argmin(energies))
            boundary_frame = int(boundary_frames[best])
            end_sample = min(boundary_frame * FRAME_SAMPLES, total_samples)
            boundary_rms = float(energies[best])
            cut_at_pause = boundary_rms <= threshold

        if end_sample <= start_sample:
            end_sample = min(
                start_sample + max_frames * FRAME_SAMPLES, total_samples
            )
            boundary_rms = None
            cut_at_pause = None

        end_sample = min(end_sample, total_samples)
        chunks.append(
            {
                "start_sample": int(start_sample),
                "end_sample": int(end_sample),
                "start": round(start_sample / sample_rate, 6),
                "end": round(end_sample / sample_rate, 6),
                "boundary_rms": boundary_rms,
                "cut_at_pause": cut_at_pause,
            }
        )
        start_sample = end_sample
        start_frame = start_sample // FRAME_SAMPLES

    return chunks, threshold


def join_texts(texts: Iterable[str]) -> str:
    """Join non-empty chunk outputs with a single space."""
    return " ".join(text.strip() for text in texts if text.strip())


# --- Prompt and tokenizer ---------------------------------------------------

def build_prompt(
    vocab: Sequence[str],
    language: str,
    script: str = "native",
    pnc: str = "yes",
) -> list[int]:
    """Build the 10-token Canary-style prompt used by the ONNX export."""
    if language not in CORE_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")
    if script not in {"native", "mixed"}:
        raise ValueError("script must be native or mixed")
    if pnc not in {"yes", "no"}:
        raise ValueError("pnc must be yes or no")

    language_token = f"<|{language}|>"
    try:
        language_id = list(vocab).index(language_token)
    except ValueError as exc:
        raise RuntimeError(
            f"{language_token} is missing from bodhan_vocab.json"
        ) from exc

    # Token 5 = <|pnc|>, 6 = <|nopnc|>; 9 native script, 8 mixed script.
    pnc_id = 5 if pnc == "yes" else 6
    script_id = 9 if script == "native" else 8
    return [7, 4, 18, language_id, language_id, pnc_id, script_id, 11, 13, 15]


def detokenize(vocab: Sequence[str], token_ids: Sequence[int]) -> str:
    """Detokenize generated ids; filter non-EOS/PAD special tokens defensively."""
    pieces: list[str] = []
    for token_id in token_ids:
        piece = vocab[int(token_id)]
        if piece == "<pad>" or (piece.startswith("<|") and piece.endswith("|>")):
            continue
        pieces.append(piece)
    return "".join(pieces).replace("\u2581", " ").strip()


# --- ONNX Runtime CPU engine ------------------------------------------------

class OnnxASR:
    """Minimal greedy encoder/cross-KV/decoder-KV CPU inference."""

    def __init__(
        self,
        model_files: dict[str, str],
        *,
        language: str,
        script: str,
        pnc: str,
        threads: int,
        max_new_tokens: int,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - install-time error
            raise RuntimeError(
                "onnxruntime is required. Install laptop/requirements.txt first."
            ) from exc

        providers = ort.get_available_providers()
        if "CPUExecutionProvider" not in providers:
            raise RuntimeError(
                f"CPUExecutionProvider is unavailable; providers={providers}"
            )

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(threads))
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        # Growing KV-cache tensors benefit from disabling the CPU arena.
        options.enable_cpu_mem_arena = False

        self.ort_version = ort.__version__
        self.encoder = ort.InferenceSession(
            model_files["encoder.onnx"], sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.cross = ort.InferenceSession(
            model_files["cross_kv.onnx"], sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.decoder = ort.InferenceSession(
            model_files["decoder_kv.onnx"], sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        vocab_path = Path(model_files["bodhan_vocab.json"])
        self.vocab: list[str] = json.loads(vocab_path.read_text(encoding="utf-8"))
        if len(self.vocab) != VOCAB_SIZE:
            raise RuntimeError(
                f"Expected {VOCAB_SIZE} vocabulary pieces, got {len(self.vocab)}"
            )

        self.prompt = build_prompt(self.vocab, language, script, pnc)
        self.language = language
        self.script = script
        self.pnc = pnc
        self.threads = int(threads)
        self.max_new_tokens = int(max_new_tokens)

        self.past_input_names = {
            f"past_self_key.{layer}" for layer in range(LAYERS)
        } | {
            f"past_self_value.{layer}" for layer in range(LAYERS)
        }
        self.cross_input_names = {
            f"cross_key.{layer}" for layer in range(LAYERS)
        } | {
            f"cross_value.{layer}" for layer in range(LAYERS)
        }
        self.present_output_names = {
            f"present_self_key.{layer}" for layer in range(LAYERS)
        } | {
            f"present_self_value.{layer}" for layer in range(LAYERS)
        }
        self._validate_io()

    def _validate_io(self) -> None:
        encoder_inputs = {item.name for item in self.encoder.get_inputs()}
        encoder_outputs = {item.name for item in self.encoder.get_outputs()}
        cross_inputs = {item.name for item in self.cross.get_inputs()}
        cross_outputs = {item.name for item in self.cross.get_outputs()}
        decoder_inputs = {item.name for item in self.decoder.get_inputs()}
        decoder_outputs = {item.name for item in self.decoder.get_outputs()}

        if encoder_inputs != {"feats"}:
            raise RuntimeError(f"Unexpected encoder inputs: {sorted(encoder_inputs)}")
        if "hidden_states" not in encoder_outputs:
            raise RuntimeError("encoder.onnx has no hidden_states output")
        if cross_inputs != {"encoder_hidden_states"}:
            raise RuntimeError(f"Unexpected cross_kv inputs: {sorted(cross_inputs)}")

        missing_cross = self.cross_input_names - cross_outputs
        if missing_cross:
            raise RuntimeError(
                f"cross_kv.onnx is missing outputs: {sorted(missing_cross)[:4]}..."
            )

        expected_decoder_inputs = (
            {"input_ids"} | self.past_input_names | self.cross_input_names
        )
        missing_decoder = expected_decoder_inputs - decoder_inputs
        if missing_decoder:
            raise RuntimeError(
                "decoder_kv.onnx is missing inputs: "
                f"{sorted(missing_decoder)[:4]}..."
            )
        unexpected_decoder = decoder_inputs - expected_decoder_inputs
        if unexpected_decoder:
            raise RuntimeError(
                "decoder_kv.onnx has unexpected inputs: "
                f"{sorted(unexpected_decoder)[:4]}..."
            )
        expected_decoder_outputs = {"logits"} | self.present_output_names
        missing_outputs = expected_decoder_outputs - decoder_outputs
        if missing_outputs:
            raise RuntimeError(
                "decoder_kv.onnx is missing outputs: "
                f"{sorted(missing_outputs)[:4]}..."
            )

    @staticmethod
    def _run_named(session: Any, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        values = session.run(None, feed)
        return {
            output.name: value
            for output, value in zip(session.get_outputs(), values)
        }

    def decode_array(self, samples: np.ndarray) -> tuple[str, list[int], bool]:
        """Greedy-decode one <=30 s audio window."""
        features = mel_features(samples)
        encoder_out = self._run_named(self.encoder, {"feats": features})
        hidden = np.ascontiguousarray(
            encoder_out["hidden_states"], dtype=np.float32
        )

        cross_out = self._run_named(
            self.cross, {"encoder_hidden_states": hidden}
        )
        cross_feed = {
            name: np.ascontiguousarray(cross_out[name], dtype=np.float32)
            for name in self.cross_input_names
        }

        empty_cache = np.zeros(
            (1, HEADS, 0, HEAD_DIM), dtype=np.float32
        )
        feed: dict[str, np.ndarray] = {
            "input_ids": np.asarray([self.prompt], dtype=np.int64),
        }
        for name in self.past_input_names:
            feed[name] = empty_cache
        feed.update(cross_feed)

        outputs = self._run_named(self.decoder, feed)
        logits = np.asarray(outputs["logits"]).reshape(-1)
        next_id = int(np.argmax(logits))
        generated: list[int] = []
        truncated = False

        while next_id not in {EOS_ID, PAD_ID}:
            if len(generated) >= self.max_new_tokens:
                truncated = True
                break
            generated.append(next_id)

            next_feed: dict[str, np.ndarray] = {
                "input_ids": np.asarray([[next_id]], dtype=np.int64),
            }
            for layer in range(LAYERS):
                next_feed[f"past_self_key.{layer}"] = outputs[
                    f"present_self_key.{layer}"
                ]
                next_feed[f"past_self_value.{layer}"] = outputs[
                    f"present_self_value.{layer}"
                ]
            next_feed.update(cross_feed)
            outputs = self._run_named(self.decoder, next_feed)
            logits = np.asarray(outputs["logits"]).reshape(-1)
            next_id = int(np.argmax(logits))

        return detokenize(self.vocab, generated), generated, truncated


# --- Output -----------------------------------------------------------------

def write_outputs(
    txt_path: Path,
    json_path: Path,
    payload: dict[str, Any],
) -> None:
    """Write plain text and provenance JSON in UTF-8."""
    txt_path = Path(txt_path)
    json_path = Path(json_path)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(payload["text"] + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


# --- CLI --------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only plain-text ASR with a pinned INT8 ONNX Bodhan Core "
            "derivative. No diarization, word timestamps, cloud key, or GPU."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "audio", nargs="?", help="Input WAV/MP3/FLAC/etc. audio file"
    )
    parser.add_argument(
        "--language", default=DEFAULT_LANGUAGE, choices=CORE_LANGUAGES,
        help="Native-script language code (metadata lists others; only Hindi was measured)",
    )
    parser.add_argument(
        "--script", choices=("native", "mixed"), default="native",
        help="Native script or mixed Latin/English words (Core favors native)",
    )
    parser.add_argument(
        "--pnc", choices=("yes", "no"), default="yes",
        help="Decoder prompt token for punctuation/capitalization",
    )
    parser.add_argument(
        "--model-dir", default=str(default_model_dir()),
        help="Local directory for the pinned ONNX weights",
    )
    parser.add_argument(
        "--output-txt", default="cpu_transcript.txt",
        help="Plain UTF-8 transcript text path",
    )
    parser.add_argument(
        "--output-json", default="cpu_transcript.json",
        help="Provenance/chunk JSON path (not the strict pipeline schema)",
    )
    parser.add_argument(
        "--threads", type=int, default=DEFAULT_THREADS,
        help="ONNX Runtime intra-op CPU threads",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
        help="Greedy decoding cap per audio chunk",
    )
    parser.add_argument(
        "--max-chunk-seconds", type=float, default=28.0,
        help="Maximum ASR window; must be <= 30",
    )
    parser.add_argument(
        "--min-chunk-seconds", type=float, default=1.0,
        help="Minimum ASR window",
    )
    parser.add_argument(
        "--search-seconds", type=float, default=3.0,
        help="Search this many seconds before the hard limit for a quiet cut",
    )
    parser.add_argument(
        "--silence-threshold", type=float, default=0.0,
        help="RMS pause threshold; 0 means auto",
    )
    parser.add_argument(
        "--limit-chunks", type=int, default=None,
        help="Only decode the first N chunks (partial transcript)",
    )
    parser.add_argument(
        "--list-languages", action="store_true",
        help="Print supported language codes and exit",
    )
    parser.add_argument(
        "--download-only", action="store_true",
        help="Download/verify model files and exit",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Verify model files and CPU ONNX Runtime, then exit",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the chunk plan without downloading weights or decoding",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress per-chunk progress"
    )
    parser.add_argument(
        "--show-text", action="store_true",
        help="Print each chunk's decoded text during the run"
    )

    network = parser.add_mutually_exclusive_group()
    network.add_argument(
        "--offline", action="store_true",
        help="Use only local model files; never download",
    )
    network.add_argument(
        "--force-download", action="store_true",
        help="Delete and redownload the pinned model files",
    )
    return parser


def _validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if not 0 < args.max_chunk_seconds <= 30.0:
        parser.error("--max-chunk-seconds must be in (0, 30]")
    if not 0.5 <= args.min_chunk_seconds < args.max_chunk_seconds:
        parser.error("--min-chunk-seconds must be >= 0.5 and smaller than max")
    if not 0 <= args.search_seconds < args.max_chunk_seconds - args.min_chunk_seconds:
        parser.error(
            "--search-seconds must be >= 0 and smaller than max-min chunk seconds"
        )
    if args.silence_threshold < 0:
        parser.error("--silence-threshold must be >= 0")
    if args.threads < 1:
        parser.error("--threads must be >= 1")
    if not 1 <= args.max_new_tokens <= MAX_POSITIONS - 10:
        parser.error(
            f"--max-new-tokens must be between 1 and {MAX_POSITIONS - 10}"
        )
    if args.limit_chunks is not None and args.limit_chunks < 1:
        parser.error("--limit-chunks must be >= 1 when provided")
    if args.list_languages or args.download_only or args.check:
        return
    if not args.audio:
        parser.error(
            "audio is required unless using --list-languages, --download-only, "
            "or --check; --dry-run also requires audio"
        )
    if args.dry_run and args.force_download:
        parser.error("--force-download has no effect with --dry-run; omit it")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    if args.list_languages:
        print(" ".join(CORE_LANGUAGES))
        print(
            "Note: the KV-cache derivative was measured only on Hindi; "
            "other codes come from the upstream Core model metadata.",
            file=sys.stderr,
        )
        return 0

    model_dir = Path(args.model_dir)
    if args.download_only or args.check:
        files = ensure_model_files(
            model_dir, offline=args.offline, force_download=args.force_download
        )
        print(f"Model directory: {model_dir.resolve()}")
        for name in REQUIRED_MODEL_FILES:
            print(f"  {name}: {files[name]}")
        if args.check:
            import onnxruntime as ort
            print(f"onnxruntime: {ort.__version__}")
            print(f"providers: {', '.join(ort.get_available_providers())}")
        return 0

    audio_path = Path(args.audio)
    audio = load_audio(audio_path)
    chunks, threshold = plan_chunks(
        audio,
        max_seconds=args.max_chunk_seconds,
        min_seconds=args.min_chunk_seconds,
        search_seconds=args.search_seconds,
        silence_threshold=args.silence_threshold,
    )
    duration = audio.size / SAMPLE_RATE

    if args.dry_run:
        print(f"Audio: {audio_path}")
        print(f"Duration: {duration:.3f}s, chunks: {len(chunks)}")
        print(f"Auto RMS threshold: {threshold:.8f}")
        for index, chunk in enumerate(chunks, 1):
            print(
                f"  {index:03d}: {chunk['start']:9.3f}-{chunk['end']:9.3f}s "
                f"cut_at_pause={chunk['cut_at_pause']}"
            )
        return 0

    files = ensure_model_files(
        model_dir, offline=args.offline, force_download=args.force_download
    )
    engine = OnnxASR(
        files,
        language=args.language,
        script=args.script,
        pnc=args.pnc,
        threads=args.threads,
        max_new_tokens=args.max_new_tokens,
    )

    selected = chunks[: args.limit_chunks] if args.limit_chunks else chunks
    partial_run = len(selected) < len(chunks)
    decoded_chunks: list[dict[str, Any]] = []
    warnings: list[str] = []
    hard_cuts = 0
    truncated_chunks = 0

    for index, chunk in enumerate(selected, 1):
        if not args.quiet:
            print(
                f"[{index:03d}/{len(selected):03d}] "
                f"{chunk['start']:.2f}-{chunk['end']:.2f}s",
                file=sys.stderr,
            )
        samples = audio[chunk["start_sample"]:chunk["end_sample"]]
        text, token_ids, truncated = engine.decode_array(samples)
        if not chunk["cut_at_pause"] and chunk["cut_at_pause"] is not None:
            hard_cuts += 1
        if truncated:
            truncated_chunks += 1
        record = {
            "index": index,
            **chunk,
            "text": text,
            "token_count": len(token_ids),
            "text_sha256": sha256_text(text),
            "truncated": truncated,
        }
        decoded_chunks.append(record)
        if args.show_text and text:
            print(text, flush=True)

    text = join_texts(record["text"] for record in decoded_chunks)
    if args.script == "mixed":
        warnings.append(
            "Core is optimized for native script; Indic-Transcribe-Flex may "
            "handle Hinglish/transliteration better."
        )
    if args.pnc == "no":
        warnings.append(
            "Punctuation/capitalization is disabled; cloud API output may differ."
        )
    if partial_run:
        warnings.append(
            f"Partial run: only the first {len(selected)} of {len(chunks)} "
            "chunks were decoded."
        )
    if hard_cuts:
        warnings.append(
            f"{hard_cuts} chunk(s) used a low-energy hard cut because no "
            "quiet point was found in the search window."
        )
    if truncated_chunks:
        warnings.append(
            f"{truncated_chunks} chunk(s) hit --max-new-tokens and may be "
            "incomplete."
        )

    payload: dict[str, Any] = {
        "provenance": {
            "generator": "laptop/transcribe.py",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "source_model": SOURCE_MODEL,
            "base_model": BASE_MODEL,
            "format": "INT8 ONNX with KV-cache decoder",
            "execution_provider": "CPUExecutionProvider",
            "onnxruntime_version": engine.ort_version,
            "attribution": [
                "Built with Indic-Transcribe-Core from Bodhan AI / AI4Bharat.",
                "Base model: Canary-1B-v2 developed by NVIDIA Corporation (CC-BY-4.0).",
            ],
            "accuracy_note": (
                "No measured parity with the Bodhan cloud API. INT8 quantization, "
                "chunking, and greedy CPU decoding can change words."
            ),
        },
        "audio": {
            "path": str(audio_path),
            "sha256": sha256_file(audio_path),
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
            "duration_seconds": round(duration, 6),
        },
        "options": {
            "language": args.language,
            "script": args.script,
            "pnc": args.pnc,
            "prompt": engine.prompt,
            "threads": engine.threads,
            "max_new_tokens": engine.max_new_tokens,
            "max_chunk_seconds": args.max_chunk_seconds,
            "min_chunk_seconds": args.min_chunk_seconds,
            "search_seconds": args.search_seconds,
            "silence_threshold": threshold,
        },
        "coverage": {
            "full_audio": not partial_run,
            "decoded_chunks": len(decoded_chunks),
            "planned_chunks": len(chunks),
            "covered_start": decoded_chunks[0]["start"] if decoded_chunks else 0.0,
            "covered_end": decoded_chunks[-1]["end"] if decoded_chunks else 0.0,
        },
        "text": text,
        "text_sha256": sha256_text(text),
        "text_characters": len(text),
        "chunks": decoded_chunks,
        "warnings": warnings,
        "limitations": [
            "No diarization or speaker IDs.",
            "No word timestamps or CTC alignment.",
            "No strict final transcript.json schema.",
            "Chunks are ASR processing windows, not speaker turns.",
        ],
    }

    txt_path = Path(args.output_txt)
    json_path = Path(args.output_json)
    write_outputs(txt_path, json_path, payload)
    print(f"Transcript text: {txt_path.resolve()}")
    print(f"Provenance JSON: {json_path.resolve()}")
    print(
        f"Characters={len(text)} chunks={len(decoded_chunks)} "
        f"duration={duration:.3f}s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
