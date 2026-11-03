from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .audio import split_span
from .bodhan import (
    BodhanClient,
    BodhanFatalError,
    BodhanPayloadTooLarge,
    parse_bodhan_response,
)
from .config import Config
from .runtime import import_torch, resolve_device, torch_dtype
from .types import (
    AudioSpan,
    BodhanChunk,
    StageError,
    jsonable,
)


class TranscriptionError(StageError):
    """ASR backend configuration or inference failure."""


# Only the files needed by the HF inference wrapper. The same repo also ships a
# duplicate ~4.6 GB NeMo .nemo checkpoint under nemo/; skipping it roughly halves
# the first download. `snapshot_download` supports allow_patterns filtering.
HF_ASR_INFERENCE_FILES = [
    "config.json",
    "configuration_indic_canary.py",
    "feature_extraction_indic_canary.py",
    "feature_extractor.safetensors",
    "generation_config.json",
    "indic_transcribe.py",
    "inference.py",
    "lid.py",
    "long_form.py",
    "model.safetensors",
    "modeling_indic_canary.py",
    "requirements.txt",
    "tokenization_indic_canary.py",
    "tokenizer_config.json",
    "tokenizer_multilingual.model",
    "tokenizer_spl_tokens.model",
]


class BaseTranscriber:
    """Shared cache/parsing/order logic for cloud and local Bodhan ASR."""

    backend_name = "asr"

    def __init__(
        self,
        config: Config,
        canonical_samples: np.ndarray,
        warnings: list[str],
    ):
        self.config = config
        self.canonical_samples = canonical_samples
        self.warnings = warnings

    def _chunk_key(self, span: AudioSpan) -> str:
        return hashlib.sha256(
            f"{span.key}|{self.config.transcription_config_digest()}|{self.backend_name}".encode("utf-8")
        ).hexdigest()[:24]

    def _cache_path(self, span: AudioSpan) -> Path:
        return (
            self.config.work_dir
            / "transcription"
            / "responses"
            / f"{self._chunk_key(span)}.json"
        )

    def _load_cache(self, path: Path, span: AudioSpan) -> Optional[dict[str, Any]]:
        if not path.is_file():
            return None
        try:
            cache = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if (
            cache.get("version") != 2
            or cache.get("backend") != self.backend_name
            or cache.get("span_key") != span.key
            or cache.get("config_digest") != self.config.transcription_config_digest()
            or not isinstance(cache.get("response"), dict)
            or "error" in cache["response"]
        ):
            return None
        return cache["response"]

    def _save_cache(self, path: Path, span: AudioSpan, response: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "version": 2,
                    "backend": self.backend_name,
                    "span_key": span.key,
                    "config_digest": self.config.transcription_config_digest(),
                    "span": jsonable(span),
                    "response": response,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def _make_chunk(self, span: AudioSpan, raw: dict[str, Any]) -> BodhanChunk:
        chunk_key = self._chunk_key(span)
        units, text, has_word_times = parse_bodhan_response(
            raw, span, span.index, chunk_key, self.warnings
        )
        return BodhanChunk(
            span=span,
            key=chunk_key,
            response=raw,
            units=units,
            text=text,
            has_word_times=has_word_times,
        )

    def _cache_miss_error(self, span: AudioSpan) -> TranscriptionError:
        return TranscriptionError(
            f"No valid cached {self.backend_name} transcription for {span.key}. "
            "Disable --transcription-cache-only to run the configured ASR backend."
        )

    def _transcribe_span(self, span: AudioSpan, depth: int) -> list[BodhanChunk]:
        raise NotImplementedError

    def transcribe(self, spans: list[AudioSpan]) -> list[BodhanChunk]:
        results: list[BodhanChunk] = []
        if not spans:
            return results
        # A single loaded local model is not assumed to be thread-safe. Cloud
        # uploads are independent and may use the configured worker count.
        workers = self.config.bodhan_workers if self.backend_name == "bodhan-cloud" else 1
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(self._transcribe_span, span, 0) for span in spans]
                for i, future in enumerate(as_completed(futures), 1):
                    results.extend(future.result())
                    print(f"[{self.backend_name}] {i}/{len(spans)} chunk groups complete")
        else:
            for i, span in enumerate(spans, 1):
                results.extend(self._transcribe_span(span, 0))
                print(
                    f"[{self.backend_name}] {i}/{len(spans)} chunks: "
                    f"{span.start:.2f}-{span.end:.2f}s"
                )
        results.sort(key=lambda c: (c.span.start, c.span.end, c.key))
        return results


class CloudBodhanTranscriber(BaseTranscriber):
    """Optional hosted Bodhan API backend; never used by default."""

    backend_name = "bodhan-cloud"

    def __init__(
        self,
        config: Config,
        canonical_samples: np.ndarray,
        warnings: list[str],
    ):
        super().__init__(config, canonical_samples, warnings)
        self.client: Optional[BodhanClient] = None
        if not config.transcription_cache_only:
            try:
                self.client = BodhanClient(config)
            except BodhanFatalError:
                # A fully cached run may proceed without the API key; a missing
                # cache entry will raise a clear error below.
                self.client = None

    def _transcribe_span(self, span: AudioSpan, depth: int) -> list[BodhanChunk]:
        cache_path = self._cache_path(span)
        force = "transcription" in self.config.force_stages
        if not force:
            cached = self._load_cache(cache_path, span)
            if cached is not None:
                return [self._make_chunk(span, cached)]
        if self.config.transcription_cache_only or self.client is None:
            raise TranscriptionError(
                f"No valid cached Bodhan cloud response for {span.key} and "
                "BODHAN_API_KEY is unavailable (or --transcription-cache-only was set). "
                "Use --asr-backend hf-local to avoid the cloud API."
            )
        try:
            raw = self.client.request(span, span.key)
        except BodhanPayloadTooLarge as exc:
            self.warnings.append(str(exc))
            if depth >= 3:
                raise TranscriptionError(
                    f"Repeated 413/duration errors for {span.key}; reduce --bodhan-chunk-seconds."
                ) from exc
            halves = split_span(span, self.config, self.warnings, self.canonical_samples)
            out: list[BodhanChunk] = []
            for half in halves:
                out.extend(self._transcribe_span(half, depth + 1))
            return out
        self._save_cache(cache_path, span, raw)
        return [self._make_chunk(span, raw)]


class LocalBodhanHFTranscriber(BaseTranscriber):
    """Local Bodhan `Indic-Transcribe-core` Hugging Face model.

    The model ships its own `indic_transcribe.py`; it is not a standard
    `transformers.pipeline()` checkpoint. We snapshot the repository, put it on
    `sys.path`, and call its documented `IndicTranscribe.transcribe()` API on
    each silence-aware <=28 s WAV chunk. No Bodhan cloud API key is used.
    """

    backend_name = "bodhan-hf-local"

    def __init__(
        self,
        config: Config,
        canonical_samples: np.ndarray,
        warnings: list[str],
        asr: Any = None,
        long_form: Any = None,
    ):
        super().__init__(config, canonical_samples, warnings)
        # Injectable for offline unit tests; otherwise loaded lazily on first chunk.
        self._asr = asr
        self._long_form = long_form
        self._device: Optional[str] = None
        self._dtype_name: Optional[str] = None
        self._model_dir: Optional[Path] = None

    def _resolve_dtype(self, device: str) -> str:
        if self.config.asr_compute_dtype == "auto":
            return "bfloat16" if device == "cuda" else "float32"
        return self.config.asr_compute_dtype

    def _load_model(self) -> tuple[Any, Any, str, str]:
        if self._asr is not None:
            return (
                self._asr,
                self._long_form,
                self._device or "test",
                self._dtype_name or "float32",
            )
        torch = import_torch()
        device = resolve_device(self.config.device, torch)
        dtype_name = self._resolve_dtype(device)
        if device == "cpu" and dtype_name == "bfloat16":
            self.warnings.append(
                "ASR bfloat16 requested on CPU; this can be slow. Use --asr-compute-dtype float32."
            )

        if self.config.hf_asr_model_dir:
            model_dir = self.config.hf_asr_model_dir.expanduser().resolve()
        else:
            try:
                from huggingface_hub import snapshot_download
            except Exception as exc:  # pragma: no cover - installation issue
                raise TranscriptionError(
                    "huggingface_hub is required to download the local Bodhan HF model. "
                    "Install it with `pip install -U huggingface_hub`."
                ) from exc
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
            print(
                f"[bodhan-hf-local] downloading only HF inference files from "
                f"{self.config.hf_asr_model} (skipping the duplicate .nemo checkpoint)"
            )
            try:
                model_dir = Path(
                    snapshot_download(
                        repo_id=self.config.hf_asr_model,
                        token=token,
                        allow_patterns=HF_ASR_INFERENCE_FILES,
                    )
                ).resolve()
            except Exception as exc:
                raise TranscriptionError(
                    f"Could not download {self.config.hf_asr_model!r}. The repo is gated: "
                    "accept its license on Hugging Face, then set HF_TOKEN for the same "
                    "account (or pass --hf-asr-model-dir with an already-downloaded "
                    "snapshot). Run `python -m tools.check_hf_access` first. "
                    f"Original error: {exc}"
                ) from exc

        if not (model_dir / "indic_transcribe.py").is_file():
            raise TranscriptionError(
                f"{model_dir} is not a Bodhan Indic-Transcribe snapshot: "
                "indic_transcribe.py is missing."
            )
        sys.path.insert(0, str(model_dir))
        try:
            from indic_transcribe import IndicTranscribe
        except Exception as exc:
            raise TranscriptionError(
                "Could not import IndicTranscribe from the downloaded model directory. "
                "Install torch, torchaudio, transformers, sentencepiece and soundfile. "
                f"Original error: {exc}"
            ) from exc

        try:
            asr = IndicTranscribe.from_pretrained(str(model_dir), device=device)
        except TypeError:
            # Older/newer wrapper revisions may not expose the device argument.
            asr = IndicTranscribe.from_pretrained(str(model_dir))
        if hasattr(asr, "eval"):
            try:
                asr.eval()
            except Exception:
                pass
        if dtype_name != "float32" and hasattr(asr, "to"):
            try:
                asr.to(device=device, dtype=torch_dtype(torch, dtype_name))
            except TypeError:
                try:
                    asr.to(device)
                except Exception:
                    pass

        try:
            import long_form  # type: ignore  # noqa: F401
        except Exception:
            long_form = None

        self._asr = asr
        self._long_form = long_form
        self._device = device
        self._dtype_name = dtype_name
        self._model_dir = model_dir
        print(
            f"[bodhan-hf-local] loaded {self.config.hf_asr_model} "
            f"from {model_dir} on {device} ({dtype_name})"
        )
        return asr, long_form, device, dtype_name

    @staticmethod
    def _coerce_text(result: Any) -> tuple[str, Any]:
        lid: Any = None
        text: Any = result
        if isinstance(result, tuple) and len(result) >= 2:
            text, lid = result[0], result[1]
        elif isinstance(result, list) and result:
            text = result[0]
        if isinstance(text, (list, tuple)) and text:
            text = text[0]
        if not isinstance(text, str):
            raise TranscriptionError(
                f"IndicTranscribe returned a non-string transcript: {type(text).__name__}"
            )
        return text.strip(), lid

    @staticmethod
    def _fallback_repetition(words: list[str]) -> bool:
        if len(words) < 12:
            return False
        for n in (3, 4, 5):
            grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
            if not grams:
                continue
            if max(Counter(grams).values()) >= max(3, len(grams) // 4):
                return True
        return False

    @staticmethod
    def _fallback_collapse(words: list[str]) -> list[str]:
        for n in range(8, 2, -1):
            out: list[str] = []
            i = 0
            changed = False
            while i < len(words):
                gram = words[i : i + n]
                out.extend(gram)
                j = i + n
                while j + n <= len(words) and words[j : j + n] == gram:
                    changed = True
                    j += n
                i = j
            if changed:
                words = out
        return words

    def _is_repetition_loop(self, text: str, long_form: Any) -> bool:
        words = text.split()
        if long_form is not None and hasattr(long_form, "_repetition"):
            try:
                min_words = int(getattr(long_form, "LOOP_MIN_WORDS", 12))
                threshold = float(getattr(long_form, "LOOP_REPETITION", 0.5))
                return len(words) >= min_words and float(long_form._repetition(words)) > threshold
            except Exception:
                pass
        return self._fallback_repetition(words)

    def _collapse_repeats(self, text: str, long_form: Any) -> str:
        words = text.split()
        if long_form is not None and hasattr(long_form, "_collapse_repeats"):
            try:
                return " ".join(long_form._collapse_repeats(words)).strip()
            except Exception:
                pass
        return " ".join(self._fallback_collapse(words)).strip()

    def _decode(self, span: AudioSpan) -> tuple[str, Any, bool, str, str]:
        asr, long_form, device, dtype_name = self._load_model()
        lang = self.config.language.strip() or None
        kwargs: dict[str, Any] = {}
        if lang:
            kwargs["lang"] = lang
        try:
            result = asr.transcribe(str(span.path), return_lid=True, **kwargs)
        except TypeError:
            # Older wrapper revisions may not expose return_lid.
            result = asr.transcribe(str(span.path), **kwargs)
        text, lid = self._coerce_text(result)
        looped = self._is_repetition_loop(text, long_form)
        return text, lid, looped, device, dtype_name

    def _transcribe_span(self, span: AudioSpan, depth: int) -> list[BodhanChunk]:
        cache_path = self._cache_path(span)
        force = "transcription" in self.config.force_stages
        if not force:
            cached = self._load_cache(cache_path, span)
            if cached is not None:
                return [self._make_chunk(span, cached)]
        if self.config.transcription_cache_only:
            raise self._cache_miss_error(span)

        text, lid, looped, device, dtype_name = self._decode(span)
        if looped and depth < 2:
            self.warnings.append(
                f"{span.key}: Indic-Transcribe repetition loop detected; "
                "splitting this chunk at a quiet boundary and decoding both halves."
            )
            halves = split_span(span, self.config, self.warnings, self.canonical_samples)
            out: list[BodhanChunk] = []
            for half in halves:
                out.extend(self._transcribe_span(half, depth + 1))
            return out
        if looped:
            text = self._collapse_repeats(text, self._long_form)
            self.warnings.append(
                f"{span.key}: Indic-Transcribe repetition loop persisted after two splits; "
                "collapsed repeated phrases. Check this span's transcript."
            )

        raw: dict[str, Any] = {
            "text": text,
            "backend": self.backend_name,
            "model": self.config.hf_asr_model,
            "model_dir": str(self._model_dir) if self._model_dir else None,
            "language": self.config.language,
            "language_id": lid,
            "device": device,
            "compute_dtype": dtype_name,
            "repetition_guard": bool(looped),
        }
        self._save_cache(cache_path, span, raw)
        return [self._make_chunk(span, raw)]


def create_transcriber(
    config: Config,
    canonical_samples: np.ndarray,
    warnings: list[str],
) -> BaseTranscriber:
    """Select the ASR backend: local Bodhan HF model by default."""
    if config.asr_backend == "hf-local":
        return LocalBodhanHFTranscriber(config, canonical_samples, warnings)
    if config.asr_backend == "cloud":
        return CloudBodhanTranscriber(config, canonical_samples, warnings)
    raise ValueError(f"Unknown ASR backend: {config.asr_backend!r}")


# Compatibility alias for code that imported the old cloud-only class name.
BodhanTranscriber = CloudBodhanTranscriber
