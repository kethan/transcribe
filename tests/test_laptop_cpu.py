"""Offline tests for the small CPU-only ONNX transcript path."""

from __future__ import annotations

import re
import unittest
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from laptop import compare, transcribe


class _NamedIO:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSession:
    def __init__(
        self,
        input_names: Sequence[str],
        output_names: Sequence[str],
        run_fn: Callable[[dict[str, np.ndarray]], list[np.ndarray]],
    ) -> None:
        self._inputs = [_NamedIO(name) for name in input_names]
        self._outputs = [_NamedIO(name) for name in output_names]
        self._run_fn = run_fn
        self.calls = 0

    def get_inputs(self) -> list[_NamedIO]:
        return self._inputs

    def get_outputs(self) -> list[_NamedIO]:
        return self._outputs

    def run(
        self,
        output_names: Any,
        feed: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        self.calls += 1
        return self._run_fn(feed)


def _levenshtein(
    left: Sequence[Any],
    right: Sequence[Any],
) -> int:
    """Tiny pure-Python distance for injecting into compare.compute_rates."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, 1):
        current = [i]
        for j, right_item in enumerate(right, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


class LaptopCpuTests(unittest.TestCase):
    def test_canary_prompt_matches_the_documented_token_ids(self) -> None:
        vocab = [""] * 200
        vocab[89] = "<|hi|>"
        self.assertEqual(
            transcribe.build_prompt(vocab, "hi"),
            [7, 4, 18, 89, 89, 5, 9, 11, 13, 15],
        )
        with self.assertRaises(ValueError):
            transcribe.build_prompt(vocab, "zz")

    def test_detokenize_filters_special_tokens(self) -> None:
        vocab = [""] * 7_152
        vocab[0] = " "
        vocab[1] = "<|nospeech|>"
        vocab[3] = "<|endoftext|>"
        vocab[1200] = "\u2581hello"
        vocab[1201] = "\u2581world"
        self.assertEqual(
            transcribe.detokenize(vocab, [1200, 1, 1201]),
            "hello world",
        )

    def test_chunk_plan_is_contiguous_and_prefers_quiet_points(self) -> None:
        sr = transcribe.SAMPLE_RATE
        silence = np.zeros(sr, dtype=np.float32)
        tone = (
            0.1
            * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)
        ).astype(np.float32)
        audio = np.concatenate([silence, tone, silence, tone])
        frame_duration = transcribe.FRAME / sr

        chunks, threshold = transcribe.plan_chunks(
            audio,
            max_seconds=1.5,
            search_seconds=0.5,
        )

        self.assertGreater(threshold, 0.0)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0]["start_sample"], 0)
        self.assertEqual(chunks[-1]["end_sample"], audio.size)
        for first, second in zip(chunks, chunks[1:]):
            self.assertEqual(first["end_sample"], second["start_sample"])
        for chunk in chunks:
            self.assertLessEqual(
                chunk["end"] - chunk["start"],
                1.5 + frame_duration + 1e-6,
            )
        self.assertTrue(
            any(chunk["cut_at_pause"] is True for chunk in chunks[:-1])
        )
        self.assertIsNone(chunks[-1]["cut_at_pause"])

    def test_join_texts_skips_empty_chunks(self) -> None:
        self.assertEqual(
            transcribe.join_texts(["  hello ", "", "world  "]),
            "hello world",
        )

    def test_model_files_are_pinned_and_have_checksums(self) -> None:
        self.assertEqual(len(transcribe.REVISION), 40)
        for name in transcribe.FILES:
            self.assertIn(name, transcribe.HASHES)
            self.assertEqual(len(transcribe.HASHES[name]), 64)

    def test_normalize_text_and_rates(self) -> None:
        self.assertEqual(
            compare.normalize_text(
                "Hello,  world!", ignore_punctuation=True, lowercase=True
            ),
            "hello world",
        )
        result = compare.compute_rates(
            "Hello, world!",
            "Hello there world",
            ignore_punctuation=True,
            lowercase=True,
            distance_fn=_levenshtein,
        )
        self.assertEqual(result["reference_words"], 2)
        self.assertEqual(result["hypothesis_words"], 3)
        self.assertEqual(result["word_edits"], 1)
        self.assertAlmostEqual(result["wer"], 0.5)
        self.assertEqual(result["reference_characters"], 10)
        self.assertEqual(result["hypothesis_characters"], 15)
        self.assertEqual(result["character_edits"], 5)
        self.assertAlmostEqual(result["cer"], 0.5)

    def test_kv_cache_decode_loop_with_fake_onnx_sessions(self) -> None:
        vocab = [""] * transcribe.VOCAB_SIZE
        vocab[0] = " "
        vocab[1] = "<|nospeech|>"
        vocab[2] = "<pad>"
        vocab[3] = "<|endoftext|>"
        vocab[1200] = "\u2581hello"

        encoder_outputs = ["hidden_states"]
        cross_outputs: list[str] = []
        for layer in range(transcribe.LAYERS):
            cross_outputs.append(f"cross_key.{layer}")
            cross_outputs.append(f"cross_value.{layer}")
        decoder_outputs = ["logits"]
        for layer in range(transcribe.LAYERS):
            decoder_outputs.append(f"present_self_key.{layer}")
            decoder_outputs.append(f"present_self_value.{layer}")

        def encoder_run(_feed: dict[str, np.ndarray]) -> list[np.ndarray]:
            return [np.zeros((1, 4, 1024), dtype=np.float32)]

        def cross_run(_feed: dict[str, np.ndarray]) -> list[np.ndarray]:
            return [
                np.zeros((1, 8, 4, 128), dtype=np.float32)
                for _ in cross_outputs
            ]

        state = {"calls": 0}

        def decoder_run(feed: dict[str, np.ndarray]) -> list[np.ndarray]:
            state["calls"] += 1
            chosen = 1200 if state["calls"] == 1 else transcribe.EOS
            cache_length = feed["past_self_key.0"].shape[2]
            new_length = cache_length + feed["input_ids"].shape[1]
            logits = np.full(
                (1, transcribe.VOCAB_SIZE), -1.0, dtype=np.float32
            )
            logits[0, chosen] = 10.0
            values: list[np.ndarray] = [logits]
            for _ in range(transcribe.LAYERS):
                values.append(
                    np.zeros((1, 8, new_length, 128), dtype=np.float32)
                )
                values.append(
                    np.zeros((1, 8, new_length, 128), dtype=np.float32)
                )
            return values

        encoder = _FakeSession(["feats"], encoder_outputs, encoder_run)
        cross = _FakeSession(
            ["encoder_hidden_states"], cross_outputs, cross_run
        )
        decoder = _FakeSession(
            ["input_ids"]
            + [f"past_self_{kind}.{layer}" for layer in range(transcribe.LAYERS)
               for kind in ("key", "value")]
            + cross_outputs,
            decoder_outputs,
            decoder_run,
        )

        asr = transcribe.OnnxASR.__new__(transcribe.OnnxASR)
        asr.encoder = encoder
        asr.cross = cross
        asr.decoder = decoder
        asr.vocab = vocab
        asr.prompt = [7, 4, 18, 89, 89, 5, 9, 11, 13, 15]
        asr.max_new_tokens = 512
        asr.past_names = [
            f"past_self_{kind}.{layer}"
            for layer in range(transcribe.LAYERS)
            for kind in ("key", "value")
        ]
        asr.cross_names = list(cross_outputs)

        original = transcribe.mel_features
        transcribe.mel_features = lambda _samples: np.zeros(
            (1, 128, 64), dtype=np.float32
        )
        try:
            text, token_ids, truncated = asr.decode(
                np.zeros(transcribe.SAMPLE_RATE, dtype=np.float32)
            )
        finally:
            transcribe.mel_features = original

        self.assertEqual(text, "hello")
        self.assertEqual(token_ids, [1200])
        self.assertFalse(truncated)
        self.assertEqual(decoder.calls, 2)

    def test_no_indic_transcript_literals_in_laptop_source(self) -> None:
        # The CPU path must derive all text from live model output.
        pattern = re.compile(r"[\u0900-\u097F\u0A80-\u0AFF]")
        laptop_dir = Path(__file__).resolve().parents[1] / "laptop"
        for source in sorted(laptop_dir.glob("*.py")):
            self.assertIsNone(
                pattern.search(source.read_text(encoding="utf-8")),
                f"Unexpected hardcoded Indic-script text in {source.name}",
            )


if __name__ == "__main__":
    unittest.main()
