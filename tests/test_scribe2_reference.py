from __future__ import annotations

import json
import unittest
from typing import Any

from tools import scribe2_reference as s


def synthetic_reference() -> dict[str, Any]:
    def obj(text: str, start: float, end: float, kind: str, speaker: str = "speaker_0") -> dict[str, Any]:
        return {
            "text": text,
            "start": start,
            "end": end,
            "type": kind,
            "speaker_id": speaker,
            "logprob": -0.1,
            "characters": None,
            "channel_index": None,
        }

    return {
        "segments": [
            {
                "text": "નમસ્તે દુનિયા. [खँखार] 11",
                "words": [
                    obj("નમસ્તે", 0.10, 0.50, "word"),
                    obj(" ", 0.50, 0.60, "spacing"),
                    obj("દુનિયા.", 0.60, 0.60, "word"),  # zero duration in the sample
                    obj(" ", 0.60, 0.70, "spacing"),
                    obj("[खँखार]", 0.70, 1.00, "audio_event"),
                    obj(" ", 1.00, 1.10, "spacing"),
                    obj("11", 1.10, 1.30, "word", "speaker_1"),
                ],
            }
        ]
    }


class Scribe2ReferenceTests(unittest.TestCase):
    def test_convert_strict_schema_and_preserve_canonical_text(self):
        reference = synthetic_reference()
        output, stats = s.convert_reference(reference, duration=10.0)
        self.assertTrue(stats["reference_canonical_matches_converted"])
        self.assertEqual(stats["zero_duration_tokens_repaired"], 1)
        self.assertEqual(stats["zero_duration_tokens_dropped"], 0)
        self.assertEqual(stats["strict_segments"], 1)
        self.assertEqual(stats["strict_words"], 4)
        self.assertEqual(set(output.keys()), {"segments"})
        segment = output["segments"][0]
        self.assertEqual(set(segment.keys()), s.SEGMENT_KEYS)
        self.assertEqual(segment["text"], " ".join(w["text"] for w in segment["words"]))
        self.assertEqual(segment["speaker_id"], "speaker_0")
        for word in segment["words"]:
            self.assertEqual(set(word.keys()), s.WORD_KEYS)
            self.assertEqual(word["type"], "word")
            self.assertLess(word["start"], word["end"])
        self.assertEqual(stats["validation"]["words"], 4)

    def test_drop_audio_events(self):
        output, stats = s.convert_reference(
            synthetic_reference(), duration=10.0, include_audio_events=False
        )
        self.assertEqual(stats["strict_words"], 3)
        self.assertEqual(stats["dropped_audio_event_objects"], 1)
        self.assertEqual(len(output["segments"][0]["words"]), 3)

    def test_self_verification_and_extra_key_rejection(self):
        output, _ = s.convert_reference(synthetic_reference(), duration=10.0)
        report = s.compare_candidate(output, output, duration=10.0)
        self.assertTrue(report["candidate"]["schema_validation"]["ok"])
        self.assertTrue(report["comparison"]["canonical_transcripts_identical"])
        self.assertTrue(report["comparison"]["word_sequence_identical"])
        self.assertEqual(report["comparison"]["matched_words"], 4)
        self.assertEqual(report["comparison"]["speaker_agreement_on_matched_words"]["ratio"], 1.0)

        bad = json.loads(json.dumps(output))
        bad["segments"][0]["words"][0]["extra"] = "not allowed"
        result = s.validate_candidate(bad, duration=10.0)
        self.assertFalse(result["ok"])
        self.assertIn("keys must be exactly", result["error"])


if __name__ == "__main__":
    unittest.main()
