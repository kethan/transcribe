from __future__ import annotations

import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from gujarati_pipeline.alignment import (
    gujarati_word_chars,
    merge_unalignable_words,
    normalize_romanized_word,
    tokenize_transcript,
)
from gujarati_pipeline.assembly import group_segments
from gujarati_pipeline.audio import SAMPLE_RATE, plan_spans, read_pcm16_wav, write_pcm16_wav
from gujarati_pipeline.bodhan import canonical_text, parse_bodhan_response
from gujarati_pipeline.config import Config
from gujarati_pipeline.ctc import append_star_column, ctc_forced_align, labels_from_path, word_spans_from_path
from gujarati_pipeline.diarization import nemo_rope_support
from gujarati_pipeline.transcription import (
    HF_ASR_INFERENCE_FILES,
    CloudBodhanTranscriber,
    LocalBodhanHFTranscriber,
    create_transcriber,
)
from gujarati_pipeline.types import AlignedWord, AudioInfo, AudioSpan, ValidationError
from gujarati_pipeline.validation import validate_output


def make_config(root: Path, **overrides) -> Config:
    kwargs = dict(
        audio=root / "input.wav",
        output=root / "out" / "transcript.json",
        work_dir=root / "work",
        device="cpu",
        force="none",
        bodhan_api_key=None,
        bodhan_chunk_seconds=10.0,
        bodhan_max_chunk_seconds=12.0,
        bodhan_min_chunk_seconds=2.0,
        bodhan_silence_search_seconds=2.0,
        pause_threshold=0.7,
        max_segment_seconds=30.0,
        speaker_min_coverage=0.5,
        speaker_ambiguity_margin=0.1,
    )
    kwargs.update(overrides)
    return Config(**kwargs)


def make_info(root: Path, samples: np.ndarray) -> AudioInfo:
    return AudioInfo(
        original_path=root / "input.wav",
        audio_hash="test-hash",
        canonical_path=root / "canonical.wav",
        sample_rate=SAMPLE_RATE,
        channels=1,
        sample_width_bytes=2,
        frames=int(samples.size),
        duration=float(samples.size / SAMPLE_RATE),
        conversion="test",
    )


def make_span(root: Path, start: float = 5.0, duration: float = 2.0, index: int = 0) -> AudioSpan:
    return AudioSpan(
        index=index,
        start_sample=int(start * SAMPLE_RATE),
        end_sample=int((start + duration) * SAMPLE_RATE),
        start=start,
        end=start + duration,
        path=root / f"chunk_{index}.wav",
        key=f"chunk-{index}",
        audio_hash="test-hash",
    )


def make_word(
    text: str,
    start: float,
    end: float,
    speaker: str | None,
    chunk_key: str,
    order: int,
) -> AlignedWord:
    return AlignedWord(
        text=text,
        start=start,
        end=end,
        chunk_index=0,
        unit_index=0,
        alignment_model="test",
        source="test",
        chunk_key=chunk_key,
        speaker_id=speaker,
        original_order=order,
    )


class AudioTests(unittest.TestCase):
    def test_pcm16_wav_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            x = np.array([0, 1000, -1000, 32767, -32768, 123], dtype=np.int16)
            path = root / "a.wav"
            write_pcm16_wav(path, x)
            y = read_pcm16_wav(path)
            self.assertTrue(np.array_equal(x, y))
            self.assertEqual(y.dtype, np.dtype("int16"))

    def test_silence_aware_spans_keep_absolute_offsets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sr = SAMPLE_RATE
            samples = np.zeros(60 * sr, dtype=np.int16)
            t = np.arange(sr, dtype=np.float32) / sr
            tone = (6000 * np.sin(2 * np.pi * 220 * t)).astype(np.int16)
            samples[1 * sr : 20 * sr] = np.tile(tone, 19)
            samples[25 * sr : 50 * sr] = np.tile(tone, 25)
            info = make_info(root, samples)
            cfg = make_config(root)
            warnings: list[str] = []
            spans = plan_spans(info, samples, cfg, warnings)
            self.assertGreaterEqual(len(spans), 3)
            self.assertTrue(all(0 < s.duration <= cfg.bodhan_max_chunk_seconds + 0.2 for s in spans))
            for a, b in zip(spans, spans[1:]):
                self.assertLessEqual(a.end, b.start + 1e-9)
            self.assertTrue(any(abs(s.start - 0.85) < 0.35 for s in spans))
            self.assertTrue(any(abs(s.start - 24.85) < 0.35 for s in spans))
            for span in spans:
                self.assertTrue(span.path.is_file())
                self.assertEqual(read_pcm16_wav(span.path).size, span.end_sample - span.start_sample)


class BodhanParserTests(unittest.TestCase):
    def test_documented_text_only_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            span = make_span(root)
            raw = {"text": "આત્મા જ્ઞાતા છે."}
            warnings: list[str] = []
            units, text, has_times = parse_bodhan_response(raw, span, 0, "c0", warnings)
            self.assertFalse(has_times)
            self.assertEqual(len(units), 1)
            self.assertEqual(units[0].text, raw["text"])
            self.assertIsNone(units[0].words)
            self.assertEqual(units[0].start, 5.0)
            self.assertEqual(units[0].end, 7.0)
            self.assertEqual(text, raw["text"])

    def test_optional_top_level_word_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            span = make_span(root)
            raw = {
                "text": "આત્મા જ્ઞાતા છે.",
                "words": [
                    {"word": "આત્મા", "start": 0.10, "end": 0.70},
                    {"word": "જ્ઞાતા", "start": 0.75, "end": 1.30},
                    {"word": "છે.", "start": 1.35, "end": 1.85},
                ],
            }
            warnings: list[str] = []
            units, text, has_times = parse_bodhan_response(raw, span, 0, "c0", warnings)
            self.assertTrue(has_times)
            self.assertEqual(len(units), 1)
            self.assertEqual(canonical_text(text), canonical_text(raw["text"]))
            self.assertEqual([w.start for w in units[0].words], [5.10, 5.75, 6.35])
            self.assertEqual([w.end for w in units[0].words], [5.70, 6.30, 6.85])

    def test_optional_segmented_json_with_segment_times(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            span = make_span(root)
            raw = {
                "segments": [
                    {"text": "આત્મા", "start": 0.1, "end": 0.8},
                    {"text": "જ્ઞાતા છે.", "start": 0.9, "end": 1.8},
                ]
            }
            warnings: list[str] = []
            units, text, has_times = parse_bodhan_response(raw, span, 0, "c0", warnings)
            self.assertFalse(has_times)
            self.assertEqual(len(units), 2)
            self.assertEqual(units[0].text, "આત્મા")
            self.assertEqual(units[0].start, 5.1)
            self.assertEqual(units[1].end, 6.8)
            self.assertEqual(canonical_text(text), "આત્માજ્ઞાતાછે.")


class CTCTests(unittest.TestCase):
    def test_viterbi_path_and_word_spans(self):
        # blank=0, a=1, b=2, c=3, star=4
        blank, a, b, c, star = 0, 1, 2, 3, 4
        targets = [star, a, a, b, c, star]
        word_index = [None, 0, 0, 1, 1, None]
        desired_tokens = [star, a, blank, a, blank, b, blank, c, blank, star, star, blank]
        T = len(desired_tokens)
        lp = np.full((T, 5), -10.0, dtype=np.float32)
        lp[:, blank] = -1.0
        lp[:, star] = -1.0
        lp[:, a] = -1.0
        lp[:, b] = -1.0
        lp[:, c] = -1.0
        for t, token in enumerate(desired_tokens):
            lp[t, token] = 0.0
        alignment = ctc_forced_align(lp, targets, blank)
        self.assertTrue(
            np.array_equal(labels_from_path(alignment.path, np.asarray(targets), blank), np.asarray(targets))
        )
        spans = word_spans_from_path(alignment, word_index, n_words=2, n_targets=len(targets))
        self.assertEqual(len(spans), 2)
        self.assertLess(spans[0].start_frame, spans[0].end_frame)
        self.assertLessEqual(spans[0].end_frame, spans[1].start_frame)
        self.assertTrue(all(math.isfinite(s.score) for s in spans))

    def test_append_star_column(self):
        lp = np.zeros((4, 3), dtype=np.float32)
        out, star_id = append_star_column(lp, star_penalty=0.25)
        self.assertEqual(out.shape, (4, 4))
        self.assertEqual(star_id, 3)
        self.assertTrue(np.all(out[:, 3] == -0.25))

    def test_romanization_and_unalignable_token_merge(self):
        from uroman import Uroman

        u = Uroman()
        normalized = normalize_romanized_word("આત્મા છે.", u, "guj")
        self.assertTrue(normalized)
        self.assertTrue(all(ch in "abcdefghijklmnopqrstuvwxyz'" for ch in normalized))
        raw_words = tokenize_transcript("આત્મા 11 છે.")
        norm_words = [normalize_romanized_word(w, u, "guj") for w in raw_words]
        self.assertEqual(norm_words[1], "")
        texts, norms, merges, warnings = merge_unalignable_words(raw_words, norm_words, "unit test")
        self.assertGreaterEqual(merges, 1)
        self.assertTrue(warnings)
        self.assertIn("11", "".join(texts))
        self.assertEqual(len(texts), len(norms))
        self.assertTrue(all(norms))

    def test_gujarati_model_rejects_mixed_script_oov(self):
        vocab = {"આ": 0, "ત": 1, "્": 2, "મ": 3, "ા": 4, "છ": 5, "ે": 6}
        mixed = gujarati_word_chars("આત્મા Hello આत्मा છે.")
        oov = {ch for ch in mixed if ch.lower() not in vocab}
        self.assertIn("H", oov)
        self.assertTrue(any("\u0900" <= ch <= "\u097f" for ch in oov))
        self.assertNotIn(".", oov)  # punctuation is stripped for alignment


class AssemblyValidationTests(unittest.TestCase):
    def test_tie_is_null_and_clear_winner_is_assigned(self):
        from gujarati_pipeline.assembly import assign_speakers
        from gujarati_pipeline.types import DiarInterval

        cfg = make_config(Path("/tmp"), speaker_min_coverage=0.5, speaker_ambiguity_margin=0.1)
        words = [
            make_word("a", 0.0, 1.0, None, "c0", 0),
            make_word("b", 1.0, 2.0, None, "c0", 1),
            make_word("tie", 0.5, 1.5, None, "c0", 2),
        ]
        intervals = [
            DiarInterval(0.0, 1.2, "speaker_0", "0"),
            DiarInterval(1.0, 3.0, "speaker_1", "1"),
        ]
        # A 50/50 tie word needs two equal, non-overlapping intervals.
        intervals = [
            DiarInterval(0.0, 1.0, "speaker_0", "0"),
            DiarInterval(1.0, 3.0, "speaker_1", "1"),
        ]
        warnings: list[str] = []
        stats = assign_speakers(words, intervals, cfg, warnings)
        self.assertEqual(words[0].speaker_id, "speaker_0")
        self.assertEqual(words[1].speaker_id, "speaker_1")
        self.assertIsNone(words[2].speaker_id)
        self.assertEqual(stats.unattributed_words, 1)

    def test_default_speaker_turn_grouping_keeps_full_speaker_text(self):
        cfg = make_config(
            Path("/tmp"),
            segmentation_mode="speaker-turn",
            pause_threshold=0.20,
            max_segment_seconds=0.60,
        )
        words = [
            make_word("આત્મા", 0.00, 0.30, "speaker_0", "c0", 0),
            make_word("જ્ઞાતા", 1.00, 1.30, "speaker_0", "c1", 1),
            make_word("છે.", 2.00, 2.30, "speaker_0", "c2", 2),
            make_word("હા", 2.40, 2.60, "speaker_1", "c3", 3),
            make_word("જી", 2.70, 2.90, "speaker_1", "c4", 4),
            make_word("ફરી", 3.00, 3.30, "speaker_0", "c5", 5),
            make_word("અજ્ઞાત", 3.40, 3.60, None, "c6", 6),
        ]
        warnings: list[str] = []
        segments = group_segments(words, cfg, warnings)
        self.assertEqual(len(segments), 4)
        self.assertEqual(
            [segment["speaker_id"] for segment in segments],
            ["speaker_0", "speaker_1", "speaker_0", None],
        )
        self.assertEqual(segments[0]["text"], "આત્મા જ્ઞાતા છે.")
        self.assertEqual(segments[1]["text"], "હા જી")
        # Pause, sentence, chunk and max-duration rules do not apply to turns.
        self.assertEqual(len(segments[0]["words"]), 3)
        self.assertGreater(segments[0]["end"] - segments[0]["start"], cfg.max_segment_seconds)
        legacy = group_segments(
            words,
            make_config(Path("/tmp"), segmentation_mode="legacy", check=True),
            [],
        )
        self.assertGreater(len(legacy), len(segments))
        bad = make_config(Path("/tmp"), segmentation_mode="unknown", check=True)
        with self.assertRaises(ValueError):
            bad.validate()

    def test_grouping_and_full_validation(self):
        cfg = make_config(Path("/tmp"), pause_threshold=0.7, max_segment_seconds=30.0)
        words = [
            make_word("આત્મા", 0.00, 0.50, "speaker_0", "c0", 0),
            make_word("જ્ઞાતા", 0.55, 1.00, "speaker_0", "c0", 1),
            make_word("છે.", 1.05, 1.50, "speaker_0", "c0", 2),
            make_word("હા", 3.00, 3.40, "speaker_1", "c1", 3),
        ]
        warnings: list[str] = []
        segments = group_segments(words, cfg, warnings)
        self.assertGreaterEqual(len(segments), 2)
        self.assertGreater(len(segments[0]["words"]), 1)  # not one segment per word
        output = {"segments": segments}
        summary = validate_output(
            output,
            duration=10.0,
            source_text=" ".join(w.text for w in words),
            speaker_ids=["speaker_0", "speaker_1"],
            expected_word_count=len(words),
            warnings=warnings,
        )
        self.assertEqual(summary["words"], 4)
        self.assertEqual(summary["segments"], len(segments))

    def test_validation_detects_dropped_word(self):
        cfg = make_config(Path("/tmp"))
        words = [
            make_word("આત્મા", 0.00, 0.50, "speaker_0", "c0", 0),
            make_word("જ્ઞાતા", 0.55, 1.00, "speaker_0", "c0", 1),
            make_word("છે.", 1.05, 1.50, "speaker_0", "c0", 2),
        ]
        warnings: list[str] = []
        segments = group_segments(words, cfg, warnings)
        segments[0]["words"].pop()
        segments[0]["text"] = " ".join(w["text"] for w in segments[0]["words"])
        output = {"segments": segments}
        with self.assertRaises(ValidationError):
            validate_output(
                output,
                duration=10.0,
                source_text=" ".join(w.text for w in words),
                speaker_ids=["speaker_0"],
                expected_word_count=len(words),
                warnings=warnings,
            )


class LocalBodhanHFTests(unittest.TestCase):
    class FakeIndicTranscribe:
        def __init__(self, text: str):
            self.text = text
            self.calls: list[tuple[str, str | None, bool]] = []

        def transcribe(self, path: str, lang: str | None = None, return_lid: bool = False, **_: object):
            self.calls.append((path, lang, return_lid))
            if return_lid:
                return self.text, {
                    "lang": lang or "gu",
                    "source": "explicit" if lang else "lid",
                    "topk": [],
                }
            return self.text

    def test_hf_snapshot_filter_skips_duplicate_nemo_checkpoint(self):
        self.assertIn("indic_transcribe.py", HF_ASR_INFERENCE_FILES)
        self.assertIn("model.safetensors", HF_ASR_INFERENCE_FILES)
        self.assertIn("tokenizer_multilingual.model", HF_ASR_INFERENCE_FILES)
        self.assertFalse(any("nemo/" in name or name.endswith(".nemo") for name in HF_ASR_INFERENCE_FILES))
        self.assertNotIn("banner.png", HF_ASR_INFERENCE_FILES)

    def test_local_hf_backend_default_and_no_cloud_key_needed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_pcm16_wav(root / "input.wav", np.zeros(SAMPLE_RATE, dtype=np.int16))
            cfg = make_config(root, asr_backend="hf-local", bodhan_api_key=None)
            cfg.validate()  # must not require BODHAN_API_KEY
            self.assertIsInstance(create_transcriber(cfg, np.zeros(SAMPLE_RATE, dtype=np.int16), []),
                                  LocalBodhanHFTranscriber)

            cloud = make_config(root, asr_backend="cloud", bodhan_api_key=None)
            with self.assertRaises(ValueError):
                cloud.validate()
            self.assertIsInstance(
                create_transcriber(cloud, np.zeros(SAMPLE_RATE, dtype=np.int16), []),
                CloudBodhanTranscriber,
            )

    def test_local_hf_transcription_cache_and_absolute_offsets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            span = make_span(root, start=5.0, duration=2.0, index=0)
            write_pcm16_wav(span.path, np.zeros(2 * SAMPLE_RATE, dtype=np.int16))
            cfg = make_config(
                root,
                asr_backend="hf-local",
                language="gu",
                asr_compute_dtype="float32",
                transcription_cache_only=False,
            )
            warnings: list[str] = []
            fake = self.FakeIndicTranscribe("આત્મા જ્ઞાતા છે.")
            transcriber = LocalBodhanHFTranscriber(
                cfg,
                np.zeros(7 * SAMPLE_RATE, dtype=np.int16),
                warnings,
                asr=fake,
                long_form=None,
            )
            chunks = transcriber.transcribe([span])
            self.assertEqual(len(chunks), 1)
            self.assertFalse(chunks[0].has_word_times)
            self.assertEqual(chunks[0].text, "આત્મા જ્ઞાતા છે.")
            self.assertEqual(chunks[0].units[0].start, 5.0)
            self.assertEqual(chunks[0].units[0].end, 7.0)
            self.assertEqual(fake.calls[0][1], "gu")
            self.assertTrue(fake.calls[0][2])

            cache_files = list((root / "work" / "transcription" / "responses").glob("*.json"))
            self.assertEqual(len(cache_files), 1)
            cached = json.loads(cache_files[0].read_text(encoding="utf-8"))
            self.assertEqual(cached["backend"], "bodhan-hf-local")
            self.assertEqual(cached["response"]["text"], "આત્મા જ્ઞાતા છે.")

            # A second run reuses the valid cache and never calls the model again.
            chunks2 = transcriber.transcribe([span])
            self.assertEqual(len(chunks2), 1)
            self.assertEqual(len(fake.calls), 1)

    def test_local_hf_cache_only_requires_cached_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            span = make_span(root, start=1.0, duration=2.0, index=0)
            write_pcm16_wav(span.path, np.zeros(2 * SAMPLE_RATE, dtype=np.int16))
            cfg = make_config(root, transcription_cache_only=True)
            transcriber = LocalBodhanHFTranscriber(
                cfg,
                np.zeros(3 * SAMPLE_RATE, dtype=np.int16),
                [],
                asr=self.FakeIndicTranscribe("ignored"),
                long_form=None,
            )
            from gujarati_pipeline.transcription import TranscriptionError

            with self.assertRaises(TranscriptionError):
                transcriber.transcribe([span])


class NeMoRoPeSupportTests(unittest.TestCase):
    MODULE = "nemo.collections.asr.modules.transformer_encoder"

    def test_detects_pypi_3_0_0_without_rope(self):
        fake = types.ModuleType(self.MODULE)
        fake._SUPPORTED_SELF_ATTENTION_MODELS = ("abs_pos", "rel_pos", "no_pos")
        with mock.patch.dict(sys.modules, {self.MODULE: fake}):
            supported, note = nemo_rope_support()
        self.assertFalse(supported)
        self.assertIn("abs_pos", note)

    def test_detects_pinned_main_build_with_rope(self):
        fake = types.ModuleType(self.MODULE)
        fake._SUPPORTED_SELF_ATTENTION_MODELS = ("abs_pos", "rel_pos", "no_pos", "rope")
        with mock.patch.dict(sys.modules, {self.MODULE: fake}):
            supported, note = nemo_rope_support()
        self.assertTrue(supported)
        self.assertIn("rope", note)


if __name__ == "__main__":
    unittest.main()
