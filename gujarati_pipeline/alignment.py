from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .audio import SAMPLE_RATE, unit_samples
from .config import Config
from .ctc import CTCAlignment, append_star_column, ctc_forced_align, word_spans_from_path
from .runtime import import_torch, resolve_device, torch_dtype
from .types import (
    AlignedWord,
    AlignmentError,
    AlignmentVocabularyError,
    AudioInfo,
    BodhanChunk,
    StageError,
    TranscriptUnit,
    jsonable,
)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def is_punctuation_only(token: str) -> bool:
    if not token or any(ch.isalnum() for ch in token):
        return False
    return all(ch.isspace() or unicodedata.category(ch).startswith(("P", "S")) for ch in token)


def tokenize_transcript(text: str) -> list[str]:
    """Whitespace word tokens; standalone punctuation stays attached to a word.

    The original characters are never rewritten. This only prevents a bare
    punctuation token from becoming a separate unalignable word.
    """
    words: list[str] = []
    for token in text.split():
        if is_punctuation_only(token) and words:
            words[-1] += token
        else:
            words.append(token)
    return words


def _join_tokens(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    return left + ("" if is_punctuation_only(right) else " ") + right


def merge_unalignable_words(
    raw_words: list[str],
    norm_words: list[str],
    context: str,
) -> tuple[list[str], list[str], int, list[str]]:
    """Fold tokens with no alignable characters into a neighbour.

    This is explicit, reported and character-preserving. It is used for e.g. a
    standalone ASCII number in a Gujarati transcript: uroman removes the digits
    and the MMS CTC alphabet has no digit token. We never invent a timestamp for
    such a token; instead it becomes part of an adjacent aligned word.
    """
    if len(raw_words) != len(norm_words):
        raise AlignmentError("Internal error: word/normalisation length mismatch.")
    texts: list[str] = []
    norms: list[str] = []
    pending = ""
    merges = 0
    for raw, norm in zip(raw_words, norm_words):
        if norm:
            if pending:
                raw = _join_tokens(pending, raw)
                pending = ""
                merges += 1
            texts.append(raw)
            norms.append(norm)
        else:
            merges += 1
            if texts:
                texts[-1] = _join_tokens(texts[-1], raw)
            else:
                pending = _join_tokens(pending, raw)
    if pending:
        if texts:
            texts[-1] = _join_tokens(texts[-1], pending)
        else:
            raise AlignmentError(f"{context}: transcript has no alignable characters.")
    return texts, norms, merges, [
        f"{context}: merged {merges} token(s) with no CTC-alignable characters into adjacent words."
    ] if merges else []


def normalize_romanized_word(word: str, uroman: Any, lcode: str) -> str:
    romanized = uroman.romanize_string(word, lcode=lcode) or ""
    romanized = romanized.lower()
    romanized = re.sub(r"[^a-z' ]", " ", romanized)
    romanized = re.sub(r"\s+", " ", romanized).strip()
    return romanized.replace(" ", "")


def gujarati_word_chars(word: str) -> str:
    """Characters offered to the Gujarati CTC model, preserving Indic marks."""
    out: list[str] = []
    for ch in unicodedata.normalize("NFC", word):
        cat = unicodedata.category(ch)
        if ch.isspace() or cat.startswith(("P", "S")):
            continue
        out.append(ch)
    return "".join(out)


def _load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_id, word_delimiter_token=None)
    except TypeError:
        return AutoTokenizer.from_pretrained(model_id)


def filtered_vocab(tokenizer: Any, vocab_size: int) -> dict[str, int]:
    """Lower-cased vocabulary restricted to real CTC output columns."""
    return {
        str(k).lower(): int(v)
        for k, v in tokenizer.get_vocab().items()
        if 0 <= int(v) < int(vocab_size)
    }


def blank_id_from_vocab(vocab: dict[str, int], model: Any, tokenizer: Any) -> int:
    for key in ("<blank>", "[pad]", "<pad>"):
        if key in vocab:
            return vocab[key]
    value = getattr(tokenizer, "pad_token_id", None)
    if value is None:
        value = getattr(getattr(model, "config", None), "pad_token_id", None)
    if value is None:
        raise AlignmentError("Cannot determine the CTC blank id from tokenizer/model config.")
    return int(value)


class CTCAligner:
    """Shared HF-Wav2Vec2-CTC emissions + custom Viterbi forced alignment."""

    kind = "ctc"
    model_id = ""

    def __init__(self, config: Config, model_id: str, audio_duration: float):
        self.config = config
        self.model_id = model_id
        self.audio_duration = audio_duration
        self.torch: Any = None
        self.model: Any = None
        self.processor: Any = None
        self.tokenizer: Any = None
        self.vocab: dict[str, int] = {}
        self.blank_id: Optional[int] = None
        self.device: str = "cpu"
        self.dtype: Any = None
        self.uroman: Any = None

    def _load(self) -> None:
        if self.model is not None:
            return
        torch = import_torch()
        device = resolve_device(self.config.device, torch)
        dtype = torch_dtype(torch, self.config.compute_dtype)
        if device == "cpu" and dtype != torch.float32:
            print("[Alignment] Non-float32 dtype on CPU is unsupported; using float32.")
            dtype = torch.float32
        from transformers import AutoModelForCTC, AutoProcessor

        print(f"[Alignment] Loading {self.kind} CTC model: {self.model_id} on {device} ({dtype})")
        self.model = AutoModelForCTC.from_pretrained(self.model_id, torch_dtype=dtype).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.tokenizer = _load_tokenizer(self.model_id)
        vocab_size = int(getattr(self.model.config, "vocab_size", 0))
        if vocab_size <= 0:
            raise AlignmentError(f"{self.model_id}: model config has no vocab_size")
        self.vocab = filtered_vocab(self.tokenizer, vocab_size)
        self.blank_id = blank_id_from_vocab(self.vocab, self.model, self.tokenizer)
        if not (0 <= self.blank_id < vocab_size):
            raise AlignmentError(f"{self.model_id}: invalid blank id {self.blank_id}")
        self.torch = torch
        self.device = device
        self.dtype = dtype

    def _load_extra(self) -> None:  # pragma: no cover - trivial subclass hook
        return None

    def _normalize_word(self, word: str) -> str:
        raise NotImplementedError

    def _emissions(self, samples: np.ndarray) -> tuple[np.ndarray, float]:
        self._load()
        self._load_extra()
        torch = self.torch
        waveform = np.asarray(samples, dtype=np.float32) / 32768.0
        original_seconds = waveform.size / SAMPLE_RATE
        if waveform.size < 400:  # wav2vec2 convolutional minimum
            waveform = np.pad(waveform, (0, 400 - waveform.size))
        inputs = self.processor(
            waveform,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        batch: dict[str, Any] = {}
        for key, value in inputs.items():
            if not torch.is_tensor(value):
                continue
            if key == "input_values":
                batch[key] = value.to(device=self.device, dtype=self.dtype)
            else:
                batch[key] = value.to(self.device)
        with torch.inference_mode():
            logits = self.model(**batch).logits
        log_probs = torch.log_softmax(logits[0].float(), dim=-1).cpu().numpy()
        T = int(log_probs.shape[0])
        if T <= 0:
            raise AlignmentError("CTC model returned no frames.")
        frame_duration = (waveform.size / SAMPLE_RATE) / T
        if frame_duration <= 0:
            raise AlignmentError("Invalid CTC frame duration.")
        # Padding was only for the convolutional minimum; never let word times
        # run past the real window.
        self._last_original_seconds = original_seconds
        return log_probs, frame_duration

    def _build_targets(
        self, norm_words: list[str], star_id: int
    ) -> tuple[list[int], list[Optional[int]]]:
        targets: list[int] = [star_id]
        word_index: list[Optional[int]] = [None]
        for widx, norm in enumerate(norm_words):
            for ch in norm:
                key = ch.lower()
                if key not in self.vocab:
                    raise AlignmentVocabularyError(
                        f"{self.model_id}: character {key!r} (U+{ord(key):04X}) is not in the CTC vocabulary."
                    )
                targets.append(self.vocab[key])
                word_index.append(widx)
        targets.append(star_id)
        word_index.append(None)
        if self.blank_id in targets:
            raise AlignmentError("CTC blank id leaked into the target sequence.")
        return targets, word_index

    def align_unit(self, unit: TranscriptUnit, warnings: list[str]) -> list[AlignedWord]:
        raw_words = tokenize_transcript(unit.text)
        if not raw_words:
            return []
        norm_words = [self._normalize_word(w) for w in raw_words]
        word_texts, norm_words, merges, merge_warnings = merge_unalignable_words(
            raw_words, norm_words, f"unit {unit.key}"
        )
        warnings.extend(merge_warnings)
        samples = unit_samples(unit.span, unit.start, unit.end)
        log_probs, frame_duration = self._emissions(samples)
        if max(self.vocab.values(), default=-1) >= log_probs.shape[1]:
            raise AlignmentError(
                f"{self.model_id}: tokenizer has ids outside the {log_probs.shape[1]}-column CTC head."
            )
        lp_star, star_id = append_star_column(log_probs, self.config.star_penalty)
        targets, word_index = self._build_targets(norm_words, star_id)
        alignment: CTCAlignment = ctc_forced_align(lp_star, targets, self.blank_id or 0)
        spans = word_spans_from_path(alignment, word_index, len(word_texts), len(targets))
        max_end = min(unit.end, self.audio_duration)
        aligned: list[AlignedWord] = []
        previous_end = unit.start
        for text, span in zip(word_texts, spans):
            start = unit.start + span.start_frame * frame_duration
            end = unit.start + span.end_frame * frame_duration
            start = max(start, previous_end)
            end = min(end, max_end)
            if not (start < end):
                raise AlignmentError(
                    f"unit {unit.key}: word {text!r} collapsed to zero duration after clipping; "
                    "refusing to invent a timestamp."
                )
            aligned.append(
                AlignedWord(
                    text=text,
                    start=float(start),
                    end=float(end),
                    chunk_index=unit.chunk_index,
                    unit_index=unit.unit_index,
                    alignment_model=self.kind,
                    source=f"{self.kind}:ctc-viterbi",
                    chunk_key=unit.span.key,
                    original_order=0,
                )
            )
            previous_end = end
        if merges:
            warnings.append(
                f"unit {unit.key}: {merges} token(s) were merged; output word count "
                "differs from Bodhan's whitespace token count by design."
            )
        return aligned


class MMSUromanAligner(CTCAligner):
    kind = "mms-uroman"

    def _load_extra(self) -> None:
        if self.uroman is None:
            from uroman import Uroman

            self.uroman = Uroman()

    def _normalize_word(self, word: str) -> str:
        self._load()
        self._load_extra()
        return normalize_romanized_word(word, self.uroman, self.config.alignment_language)


class GujaratiWav2Vec2Aligner(CTCAligner):
    kind = "gujarati-wav2vec2"

    def _normalize_word(self, word: str) -> str:
        self._load()
        norm = gujarati_word_chars(word)
        oov = sorted({ch for ch in norm if ch.lower() not in self.vocab})
        if oov:
            pretty = ", ".join(f"{ch!r} (U+{ord(ch):04X})" for ch in oov)
            raise AlignmentVocabularyError(
                f"{self.model_id} is Gujarati-character-only and cannot align {pretty}. "
                "Use --alignment-model mms-uroman or auto for Hindi/English/punctuation text."
            )
        return norm


@dataclass
class AlignmentStats:
    words: list[AlignedWord] = field(default_factory=list)
    models_used: dict[str, int] = field(default_factory=dict)
    bodhan_word_units: int = 0
    cached_units: int = 0
    aligned_units: int = 0
    fallback_units: int = 0
    merged_tokens: int = 0
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class AlignmentRouter:
    """Choose a verified CTC aligner per transcript unit and cache its output."""

    def __init__(
        self,
        config: Config,
        info: AudioInfo,
        warnings: list[str],
    ):
        self.config = config
        self.info = info
        self.warnings = warnings
        self.aligners: dict[str, CTCAligner] = {}
        self._guj_vocab: Optional[set[str]] = None

    def _gujarati_vocab(self) -> set[str]:
        if self._guj_vocab is None:
            from transformers import AutoConfig

            cfg = AutoConfig.from_pretrained(self.config.gujarati_model)
            tok = _load_tokenizer(self.config.gujarati_model)
            self._guj_vocab = set(filtered_vocab(tok, int(cfg.vocab_size)).keys())
        return self._guj_vocab

    def _unit_needs_mms(self, unit: TranscriptUnit) -> tuple[bool, str]:
        vocab = self._gujarati_vocab()
        oov: set[str] = set()
        for word in tokenize_transcript(unit.text):
            for ch in gujarati_word_chars(word):
                if ch.lower() not in vocab:
                    oov.add(ch)
        if oov:
            pretty = ", ".join(f"{ch!r} (U+{ord(ch):04X})" for ch in sorted(oov)[:8])
            return True, pretty
        return False, ""

    def _choose_kind(self, unit: TranscriptUnit) -> str:
        strategy = self.config.alignment_model
        if strategy == "mms-uroman":
            return "mms-uroman"
        if strategy == "gujarati-wav2vec2":
            return "gujarati-wav2vec2"
        needs_mms, oov = self._unit_needs_mms(unit)
        if needs_mms:
            self.warnings.append(
                f"unit {unit.key}: Gujarati CTC vocabulary lacks {oov}; auto-selected MMS+uroman."
            )
            return "mms-uroman"
        return "gujarati-wav2vec2"

    def _aligner(self, kind: str) -> CTCAligner:
        if kind not in self.aligners:
            if kind == "mms-uroman":
                self.aligners[kind] = MMSUromanAligner(
                    self.config, self.config.mms_model, self.info.duration
                )
            else:
                self.aligners[kind] = GujaratiWav2Vec2Aligner(
                    self.config, self.config.gujarati_model, self.info.duration
                )
        return self.aligners[kind]

    def _cache_path(self, chunk: BodhanChunk, unit: TranscriptUnit, kind: str) -> Path:
        digest = hashlib.sha256(
            (
                f"{chunk.key}|{unit.key}|{unit.start:.6f}|{unit.end:.6f}|"
                f"{text_hash(unit.text)}|{kind}|{self.config.alignment_config_digest()}"
            ).encode("utf-8")
        ).hexdigest()[:24]
        return self.config.work_dir / "alignment" / self.info.audio_hash[:20] / f"{digest}.json"

    @staticmethod
    def _cache_valid(cache: dict[str, Any], unit: TranscriptUnit, kind: str, digest: str) -> bool:
        return (
            cache.get("version") == 1
            and cache.get("model") == kind
            and cache.get("config_digest") == digest
            and cache.get("text") == unit.text
            and cache.get("text_hash") == text_hash(unit.text)
            and isinstance(cache.get("words"), list)
        )

    def _from_cache(self, cache: dict[str, Any], unit: TranscriptUnit, kind: str) -> list[AlignedWord]:
        out: list[AlignedWord] = []
        for raw in cache["words"]:
            word = AlignedWord(
                text=str(raw["text"]),
                start=float(raw["start"]),
                end=float(raw["end"]),
                chunk_index=unit.chunk_index,
                unit_index=unit.unit_index,
                alignment_model=kind,
                source=f"{kind}:cache",
                chunk_key=unit.span.key,
            )
            out.append(word)
        return out

    def _from_bodhan_words(self, unit: TranscriptUnit) -> list[AlignedWord]:
        out: list[AlignedWord] = []
        max_end = min(unit.end, self.info.duration)
        for tw in unit.words or []:
            if not (unit.start <= tw.start < tw.end <= max_end):
                raise AlignmentError(
                    f"unit {unit.key}: Bodhan word timestamp [{tw.start}, {tw.end}] is outside "
                    f"the unit window [{unit.start}, {max_end}]."
                )
            out.append(
                AlignedWord(
                    text=tw.text,
                    start=float(tw.start),
                    end=float(tw.end),
                    chunk_index=unit.chunk_index,
                    unit_index=unit.unit_index,
                    alignment_model="bodhan",
                    source="bodhan.word_timestamps",
                    chunk_key=unit.span.key,
                )
            )
        return out

    def align_chunks(self, chunks: list[BodhanChunk], force: bool) -> AlignmentStats:
        stats = AlignmentStats(warnings=self.warnings)
        total_units = sum(len(c.units) for c in chunks)
        done = 0
        digest = self.config.alignment_config_digest()
        for chunk in chunks:
            for unit in chunk.units:
                done += 1
                if not unit.text.strip():
                    continue
                if done % 10 == 0 or done == total_units:
                    print(f"[Alignment] {done}/{total_units} transcript units")
                try:
                    if unit.words is not None:
                        words = self._from_bodhan_words(unit)
                        stats.bodhan_word_units += 1
                        stats.models_used["bodhan"] = stats.models_used.get("bodhan", 0) + 1
                    else:
                        kind = self._choose_kind(unit)
                        if (
                            self.config.alignment_model == "auto"
                            and kind == "mms-uroman"
                            and self._unit_needs_mms(unit)[0]
                        ):
                            stats.fallback_units += 1
                        cache_path = self._cache_path(chunk, unit, kind)
                        if not force and cache_path.is_file():
                            try:
                                cache = json.loads(cache_path.read_text(encoding="utf-8"))
                            except json.JSONDecodeError:
                                cache = {}
                            if self._cache_valid(cache, unit, kind, digest):
                                words = self._from_cache(cache, unit, kind)
                                stats.cached_units += 1
                            else:
                                cache = {}
                        else:
                            cache = {}
                        if not cache or not self._cache_valid(cache, unit, kind, digest):
                            aligner = self._aligner(kind)
                            words = aligner.align_unit(unit, self.warnings)
                            cache_path.parent.mkdir(parents=True, exist_ok=True)
                            cache_path.write_text(
                                json.dumps(
                                    {
                                        "version": 1,
                                        "model": kind,
                                        "model_id": aligner.model_id,
                                        "config_digest": digest,
                                        "text": unit.text,
                                        "text_hash": text_hash(unit.text),
                                        "unit_start": unit.start,
                                        "unit_end": unit.end,
                                        "words": jsonable(
                                            [
                                                {"text": w.text, "start": w.start, "end": w.end}
                                                for w in words
                                            ]
                                        ),
                                    },
                                    ensure_ascii=False,
                                    indent=2,
                                )
                                + "\n",
                                encoding="utf-8",
                            )
                            stats.aligned_units += 1
                        stats.models_used[kind] = stats.models_used.get(kind, 0) + 1
                except AlignmentError as exc:
                    stats.failures.append(f"unit {unit.key}: {exc}")
                    raise StageError(f"Alignment failed for unit {unit.key}: {exc}") from exc
                for word in words:
                    word.original_order = len(stats.words)
                    stats.words.append(word)
        return stats


def verify_alignment_compatibility(config: Config) -> dict[str, Any]:
    """Inspect tokenizer/config/output shape compatibility without model weights."""
    result: dict[str, Any] = {"ok": False, "checks": {}}
    try:
        from transformers import AutoConfig, __version__ as transformers_version
    except Exception as exc:
        result["error"] = f"transformers is unavailable: {exc}"
        return result
    result["transformers_version"] = transformers_version
    u = None
    try:
        import uroman

        u = uroman.Uroman()
        sample = "આત્મા જ્ઞાતા છે. आत्मा है। Hello 11"
        romanized = normalize_romanized_word(sample, u, config.alignment_language)
        result["uroman_sample"] = romanized
        result["uroman_ok"] = bool(romanized) and all(
            ch in "abcdefghijklmnopqrstuvwxyz'" for ch in romanized
        )
    except Exception as exc:
        result["uroman_ok"] = False
        result["uroman_error"] = str(exc)

    for label, model_id in (
        ("gujarati_wav2vec2", config.gujarati_model),
        ("mms_uroman", config.mms_model),
    ):
        try:
            cfg = AutoConfig.from_pretrained(model_id)
            tok = _load_tokenizer(model_id)
            vocab = filtered_vocab(tok, int(cfg.vocab_size))
            blank = None
            for key in ("<blank>", "[pad]", "<pad>"):
                if key in vocab:
                    blank = vocab[key]
                    break
            if blank is None:
                blank = int(getattr(tok, "pad_token_id", getattr(cfg, "pad_token_id", 0)))
            mixed = "આત્મા જ્ઞાતા છે. आत्मा है। Hello 11"
            if label == "mms_uroman":
                normalized_sample = (
                    normalize_romanized_word(mixed, u, config.alignment_language) if u else ""
                )
                oov = sorted({ch for ch in normalized_sample if ch not in vocab})
                suitable = bool(u) and bool(normalized_sample) and not oov
            else:
                normalized_sample = gujarati_word_chars(mixed)
                oov = sorted({ch for ch in normalized_sample if ch.lower() not in vocab})
                suitable = False
            result["checks"][label] = {
                "model_id": model_id,
                "architecture": getattr(cfg, "architectures", None),
                "vocab_size": int(cfg.vocab_size),
                "ctc_blank_id": blank,
                "pad_token_id": int(getattr(cfg, "pad_token_id", -1)),
                "tokenizer_columns": len(vocab),
                "normalized_sample": normalized_sample,
                "mixed_script_oov_count": len(oov),
                "mixed_script_oov_sample": oov[:10],
                "suitable_for_mixed_transcript": suitable,
            }
        except Exception as exc:
            result["checks"][label] = {"model_id": model_id, "error": str(exc)}
    result["ok"] = all("error" not in v for v in result["checks"].values()) and bool(
        result.get("uroman_ok")
    )
    return result
