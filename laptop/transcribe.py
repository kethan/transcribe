#!/usr/bin/env python3
"""Minimal CPU-only plain-text ASR for a 16 GB laptop.

Pinned INT8 ONNX KV-cache derivative of Bodhan Indic-Transcribe-Core.
No diarization, word timestamps, strict JSON, cloud key, GPU, NeMo, or CTC.

Attribution:
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

MODEL_ID = "schamp007/verbatim-hindi-asr-core-sc"
REVISION = "e17f007b2906828614334050e4693e1f4029871c"
SOURCE_MODEL = "bodhan-ai/indic-transcribe-core"
FILES = ("encoder.onnx", "cross_kv.onnx", "decoder_kv.onnx", "bodhan_vocab.json")
HASHES = {
    "encoder.onnx": "d25b1a9fcba2888c44428a0df46a2574b17692e937124a8f6328acf330e29de1",
    "cross_kv.onnx": "72bf71daee584dd5b66812c3df38c4de46e481f5acedb29718848b59eb95cce3",
    "decoder_kv.onnx": "98b90cea1096fb0adeabeab4c71a78b0b74d8508422b1e7d4172aa2bbd147325",
    "bodhan_vocab.json": "571fa2fac0efa0cb7c4196cb17e899560cf0c91510f7f2d739243d3766d5fb44",
}

LANGUAGES = (
    "as", "bn", "brx", "doi", "gu", "hi", "kn", "ks", "kok", "mai", "ml",
    "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur",
    "en", "bho", "bhb",
)
SAMPLE_RATE = 16_000
FRAME = 320          # 20 ms at 16 kHz
LAYERS, HEADS, HEAD_DIM = 24, 8, 128
VOCAB_SIZE, EOS, PAD = 7_152, 3, 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def default_model_dir() -> Path:
    override = os.environ.get("LAPTOP_ASR_MODEL_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "transcribe-laptop" / "verbatim-hindi-asr-core-sc"


def ensure_model_files(model_dir: Path, offline: bool = False) -> dict[str, str]:
    """Download the four pinned files once, then verify their SHA-256 hashes."""
    model_dir = Path(model_dir).expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: model_dir / name for name in FILES}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing and offline:
        raise FileNotFoundError(
            f"Offline mode is missing {', '.join(missing)} in {model_dir}. "
            "Run once without --offline."
        )
    if missing:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=MODEL_ID,
            revision=REVISION,
            allow_patterns=list(FILES),
            local_dir=str(model_dir),
        )
    for name in FILES:
        if not paths[name].is_file():
            raise FileNotFoundError(f"Missing model file: {paths[name]}")
        actual = sha256_file(paths[name])
        if actual != HASHES[name]:
            raise RuntimeError(
                f"Checksum mismatch for {paths[name]}. "
                f"Expected {HASHES[name]}, got {actual}. Delete it and rerun."
            )
    return {name: str(paths[name]) for name in FILES}


def load_audio(path: Path) -> np.ndarray:
    """Decode to contiguous 16 kHz mono float32; ffmpeg is an MP3 fallback."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        import librosa
        audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True, dtype=np.float32)
    except Exception as exc:
        command = [
            "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0",
            "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-",
        ]
        try:
            result = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
            )
        except FileNotFoundError as ff:
            raise RuntimeError(
                f"Could not decode {path}; install ffmpeg or convert to WAV: {exc}"
            ) from ff
        if result.returncode != 0:
            raise RuntimeError(
                f"Could not decode {path}; install ffmpeg or convert to WAV: {exc}"
            ) from exc
        audio = np.frombuffer(result.stdout, dtype="<f4").copy()

    array = np.asarray(audio)
    if array.ndim == 2:
        array = array.mean(axis=0)
    array = np.ascontiguousarray(array.reshape(-1), dtype=np.float32)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"Invalid audio samples in {path}")
    return array


def mel_features(samples: np.ndarray) -> np.ndarray:
    """Front-end documented by the KV-cache model card: [1, 128, T] float32."""
    import librosa

    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if audio.size < SAMPLE_RATE:
        pad = SAMPLE_RATE - audio.size
        audio = np.pad(audio, (pad // 2, pad - pad // 2))
    emphasized = np.empty_like(audio)
    emphasized[0] = audio[0]
    emphasized[1:] = audio[1:] - np.float32(0.97) * audio[:-1]

    # Model card specifies a symmetric Hann window (not periodic/DFT-even).
    window = librosa.filters.get_window("hann", 400, fftbins=False)
    stft = librosa.stft(
        emphasized, n_fft=512, hop_length=160, win_length=400,
        window=window, center=True, pad_mode="reflect",
    )
    mel = np.dot(
        librosa.filters.mel(
            sr=SAMPLE_RATE, n_fft=512, n_mels=128, fmin=0,
            fmax=SAMPLE_RATE / 2, htk=False, norm="slaney",
        ),
        np.abs(stft) ** 2,
    )
    log_mel = np.log(mel + 2.0 ** -24)
    normalized = (
        log_mel - log_mel.mean(axis=1, keepdims=True)
    ) / (log_mel.std(axis=1, ddof=1, keepdims=True) + 1e-5)
    return np.ascontiguousarray(normalized[None], dtype=np.float32)


def _frame_rms(audio: np.ndarray) -> np.ndarray:
    """20 ms non-overlapping RMS frames, including any final partial frame."""
    count = max(1, (audio.size + FRAME - 1) // FRAME)
    result = np.empty(count, dtype=np.float32)
    block_frames = 8192
    for offset in range(0, count, block_frames):
        stop = min(offset + block_frames, count)
        left = offset * FRAME
        right = min(stop * FRAME, audio.size)
        block = audio[left:right]
        missing = (stop - offset) * FRAME - block.size
        if missing:
            block = np.pad(block, (0, missing))
        frames = block.reshape(stop - offset, FRAME)
        result[offset:stop] = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    return result


def plan_chunks(
    audio: np.ndarray,
    max_seconds: float = 28.0,
    search_seconds: float = 3.0,
) -> tuple[list[dict[str, Any]], float]:
    """Contiguous <=max_seconds windows, cut near quiet points."""
    if not 0 < max_seconds <= 30:
        raise ValueError("max_seconds must be in (0, 30]")
    if not 0 <= search_seconds < max_seconds:
        raise ValueError("search_seconds must be in [0, max_seconds)")
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise ValueError("Audio has no samples")
    rms = _frame_rms(audio)
    threshold = max(1e-5, float(np.quantile(rms, 0.25)) * 2.0)
    frame_seconds = FRAME / SAMPLE_RATE
    max_frames = max(1, int(max_seconds / frame_seconds))
    min_frames = max(1, int(1.0 / frame_seconds))
    search_frames = int(search_seconds / frame_seconds)
    total = audio.size
    chunks: list[dict[str, Any]] = []
    start = 0

    while start < total:
        if total - start <= max_frames * FRAME:
            end = total
            cut_at_pause: bool | None = None
        else:
            start_frame = start // FRAME
            stop_frame = min(start_frame + max_frames, rms.size)
            low = max(start_frame + min_frames, stop_frame - search_frames)
            high = min(stop_frame, rms.size)
            if high < low:
                low = high
            frames = np.arange(low, high + 1, dtype=np.int64)
            before = np.clip(frames - 1, 0, max(0, rms.size - 1))
            energies = rms[before]
            best = int(np.argmin(energies))
            boundary = int(frames[best])
            end = min(boundary * FRAME, total)
            cut_at_pause = bool(energies[best] <= threshold)
        if end <= start:
            end = min(start + max_frames * FRAME, total)
            cut_at_pause = None
        chunks.append({
            "start_sample": int(start),
            "end_sample": int(end),
            "start": round(start / SAMPLE_RATE, 6),
            "end": round(end / SAMPLE_RATE, 6),
            "cut_at_pause": cut_at_pause,
        })
        start = end
    return chunks, threshold


def build_prompt(vocab: Sequence[str], language: str) -> list[int]:
    """Fixed native-script, PnC=yes 10-token Canary prompt."""
    if language not in LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")
    token = f"<|{language}|>"
    try:
        language_id = list(vocab).index(token)
    except ValueError as exc:
        raise RuntimeError(f"{token} missing from bodhan_vocab.json") from exc
    # 5=<|pnc|>, 9=native, 11=<|noromanized|>, 13=<|notimestamp|>, 15=<|nodiarize|>
    return [7, 4, 18, language_id, language_id, 5, 9, 11, 13, 15]


def detokenize(vocab: Sequence[str], ids: Sequence[int]) -> str:
    pieces = []
    for token_id in ids:
        piece = vocab[int(token_id)]
        if piece != "<pad>" and not (piece.startswith("<|") and piece.endswith("|>")):
            pieces.append(piece)
    return "".join(pieces).replace("\u2581", " ").strip()


def run_named(session: Any, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    values = session.run(None, feed)
    return {out.name: value for out, value in zip(session.get_outputs(), values)}


class OnnxASR:
    def __init__(
        self,
        files: dict[str, str],
        language: str,
        threads: int,
        max_new_tokens: int,
    ) -> None:
        import onnxruntime as ort

        if "CPUExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CPUExecutionProvider is unavailable")
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, threads)
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        options.enable_cpu_mem_arena = False
        provider = ["CPUExecutionProvider"]
        self.ort_version = ort.__version__
        self.encoder = ort.InferenceSession(
            files["encoder.onnx"], sess_options=options, providers=provider
        )
        self.cross = ort.InferenceSession(
            files["cross_kv.onnx"], sess_options=options, providers=provider
        )
        self.decoder = ort.InferenceSession(
            files["decoder_kv.onnx"], sess_options=options, providers=provider
        )
        self.vocab = json.loads(Path(files["bodhan_vocab.json"]).read_text("utf-8"))
        if len(self.vocab) != VOCAB_SIZE:
            raise RuntimeError(f"Expected {VOCAB_SIZE} vocabulary pieces")
        self.prompt = build_prompt(self.vocab, language)
        self.max_new_tokens = max_new_tokens
        self.cross_names = [
            f"cross_{kind}.{layer}"
            for layer in range(LAYERS)
            for kind in ("key", "value")
        ]
        self.past_names = [
            f"past_self_{kind}.{layer}"
            for layer in range(LAYERS)
            for kind in ("key", "value")
        ]

    def decode(self, samples: np.ndarray) -> tuple[str, list[int], bool]:
        features = mel_features(samples)
        hidden = np.ascontiguousarray(
            run_named(self.encoder, {"feats": features})["hidden_states"],
            dtype=np.float32,
        )
        cross = run_named(self.cross, {"encoder_hidden_states": hidden})
        cross_feed = {
            name: np.ascontiguousarray(cross[name], dtype=np.float32)
            for name in self.cross_names
        }
        empty = np.zeros((1, HEADS, 0, HEAD_DIM), dtype=np.float32)
        feed: dict[str, np.ndarray] = {
            "input_ids": np.asarray([self.prompt], dtype=np.int64),
        }
        feed.update({name: empty for name in self.past_names})
        feed.update(cross_feed)

        out = run_named(self.decoder, feed)
        next_id = int(np.argmax(np.asarray(out["logits"]).reshape(-1)))
        generated: list[int] = []
        truncated = False
        while next_id not in (EOS, PAD):
            if len(generated) >= self.max_new_tokens:
                truncated = True
                break
            generated.append(next_id)
            feed = {"input_ids": np.asarray([[next_id]], dtype=np.int64)}
            for layer in range(LAYERS):
                feed[f"past_self_key.{layer}"] = out[f"present_self_key.{layer}"]
                feed[f"past_self_value.{layer}"] = out[f"present_self_value.{layer}"]
            feed.update(cross_feed)
            out = run_named(self.decoder, feed)
            next_id = int(np.argmax(np.asarray(out["logits"]).reshape(-1)))
        return detokenize(self.vocab, generated), generated, truncated


def join_texts(texts: Iterable[str]) -> str:
    return " ".join(text.strip() for text in texts if text.strip())


def write_outputs(txt: Path, js: Path, payload: dict[str, Any]) -> None:
    txt.parent.mkdir(parents=True, exist_ok=True)
    js.parent.mkdir(parents=True, exist_ok=True)
    txt.write_text(payload["text"] + "\n", encoding="utf-8")
    js.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CPU-only plain-text ASR with pinned INT8 ONNX Bodhan Core weights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("audio", nargs="?", help="Input WAV/MP3/FLAC audio")
    parser.add_argument("--language", default="hi", choices=LANGUAGES)
    parser.add_argument("--model-dir", default=str(default_model_dir()))
    parser.add_argument("--output-txt", default="cpu_transcript.txt")
    parser.add_argument("--output-json", default="cpu_transcript.json")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-chunk-seconds", type=float, default=28.0)
    parser.add_argument("--search-seconds", type=float, default=3.0)
    parser.add_argument("--limit-chunks", type=int, default=None)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)

    if not 0 < args.max_chunk_seconds <= 30:
        parser.error("--max-chunk-seconds must be in (0, 30]")
    if not 0 <= args.search_seconds < args.max_chunk_seconds:
        parser.error("--search-seconds must be in [0, max-chunk-seconds)")
    if args.threads < 1:
        parser.error("--threads must be >= 1")
    if not 1 <= args.max_new_tokens <= 1014:
        parser.error("--max-new-tokens must be between 1 and 1014")
    if args.limit_chunks is not None and args.limit_chunks < 1:
        parser.error("--limit-chunks must be >= 1")
    if args.download_only:
        files = ensure_model_files(Path(args.model_dir), offline=args.offline)
        for name in FILES:
            print(f"{name}: {files[name]}")
        return 0
    if not args.audio:
        parser.error("audio is required unless --download-only is used")

    audio = load_audio(Path(args.audio))
    chunks, threshold = plan_chunks(
        audio, args.max_chunk_seconds, args.search_seconds
    )
    if args.dry_run:
        print(f"duration={audio.size / SAMPLE_RATE:.3f}s chunks={len(chunks)} threshold={threshold:.8f}")
        for index, chunk in enumerate(chunks, 1):
            print(
                f"{index:03d}: {chunk['start']:.3f}-{chunk['end']:.3f}s "
                f"pause={chunk['cut_at_pause']}"
            )
        return 0

    files = ensure_model_files(Path(args.model_dir), offline=args.offline)
    engine = OnnxASR(
        files, args.language, args.threads, args.max_new_tokens
    )
    selected = chunks[: args.limit_chunks] if args.limit_chunks else chunks
    partial = len(selected) < len(chunks)
    results: list[dict[str, Any]] = []
    hard_cuts = truncated = 0

    for index, chunk in enumerate(selected, 1):
        print(
            f"[{index:03d}/{len(selected):03d}] "
            f"{chunk['start']:.2f}-{chunk['end']:.2f}s",
            file=sys.stderr,
        )
        samples = audio[chunk["start_sample"]:chunk["end_sample"]]
        text, ids, was_truncated = engine.decode(samples)
        if chunk["cut_at_pause"] is False:
            hard_cuts += 1
        if was_truncated:
            truncated += 1
        results.append({
            **chunk,
            "index": index,
            "text": text,
            "token_count": len(ids),
            "text_sha256": sha256_text(text),
            "truncated": was_truncated,
        })

    text = join_texts(item["text"] for item in results)
    warnings = []
    if partial:
        warnings.append(
            f"Partial run: {len(selected)} of {len(chunks)} chunks decoded."
        )
    if hard_cuts:
        warnings.append(f"{hard_cuts} hard cuts; no quiet point was found.")
    if truncated:
        warnings.append(f"{truncated} chunks hit --max-new-tokens.")

    payload = {
        "model": {
            "id": MODEL_ID,
            "revision": REVISION,
            "source": SOURCE_MODEL,
            "format": "INT8 ONNX with KV-cache decoder",
            "onnxruntime": engine.ort_version,
            "accuracy_note": "No measured parity with the Bodhan cloud API.",
        },
        "audio": {
            "path": str(args.audio),
            "sha256": sha256_file(Path(args.audio)),
            "sample_rate": SAMPLE_RATE,
            "duration_seconds": round(audio.size / SAMPLE_RATE, 6),
        },
        "language": args.language,
        "prompt": engine.prompt,
        "text": text,
        "text_sha256": sha256_text(text),
        "text_characters": len(text),
        "chunk_plan": {
            "max_seconds": args.max_chunk_seconds,
            "search_seconds": args.search_seconds,
            "rms_threshold": threshold,
            "planned": len(chunks),
            "decoded": len(results),
            "full_audio": not partial,
        },
        "chunks": results,
        "warnings": warnings,
        "notes": [
            "Text comes only from live ONNX decoder output; no transcript is hardcoded.",
            "No diarization, word timestamps, speaker turns, or strict final JSON.",
            "Chunks are ASR windows, not speaker turns.",
        ],
    }
    write_outputs(Path(args.output_txt), Path(args.output_json), payload)
    print(f"Transcript text: {Path(args.output_txt).resolve()}")
    print(f"Provenance JSON: {Path(args.output_json).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
