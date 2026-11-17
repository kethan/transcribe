from __future__ import annotations

import unittest
from typing import Any

from tools import scribe2_speaker_turns as st


def obj(
    text: str,
    start: float,
    end: float,
    kind: str,
    speaker: str | None,
) -> dict[str, Any]:
    return {
        "text": text,
        "start": start,
        "end": end,
        "type": kind,
        "speaker_id": speaker,
        "logprob": -0.25,
        "characters": None,
        "channel_index": None,
    }


def reference() -> dict[str, Any]:
    return {
        "segments": [
            {
                "text": "hello world. ",
                "words": [
                    obj("hello", 0.0, 0.5, "word", "speaker_0"),
                    obj(" ", 0.5, 0.6, "spacing", "speaker_0"),
                    obj("world.", 0.6, 1.2, "word", "speaker_0"),
                    obj(" ", 1.2, 1.3, "spacing", "speaker_0"),
                ],
            },
            {
                "text": "bye.",
                "words": [obj("bye.", 1.3, 1.8, "word", "speaker_1")],
            },
            {
                "text": "more.",
                "words": [obj("more.", 1.8, 2.2, "word", "speaker_1")],
            },
            {
                "text": "again.",
                "words": [obj("again.", 2.2, 2.8, "word", "speaker_0")],
            },
            {
                "text": "unknown.",
                "words": [obj("unknown.", 2.8, 3.2, "word", None)],
            },
        ]
    }


class Scribe2SpeakerTurnTests(unittest.TestCase):
    def test_groups_full_speaker_turns_and_preserves_all_objects(self):
        source = reference()
        output, stats = st.convert_speaker_turns(source)
        segments = output["segments"]
        self.assertEqual(len(segments), 4)
        self.assertEqual(
            [segment["speaker_id"] for segment in segments],
            ["speaker_0", "speaker_1", "speaker_0", None],
        )
        self.assertEqual(segments[0]["text"], "hello world. ")
        self.assertEqual(segments[1]["text"], "bye.more.")
        self.assertEqual(segments[2]["text"], "again.")
        self.assertEqual(segments[3]["text"], "unknown.")
        source_objects = [w for segment in source["segments"] for w in segment["words"]]
        output_objects = [w for segment in segments for w in segment["words"]]
        self.assertEqual(output_objects, source_objects)
        self.assertEqual(stats["source_text_matches_output_text"], True)
        self.assertEqual(stats["source_word_objects"], stats["output_word_objects"])
        self.assertEqual(stats["word_object_type_counts"], {"word": 6, "spacing": 2})
        self.assertEqual(stats["speaker_ids"], ["speaker_0", "speaker_1"])
        self.assertEqual(stats["segments_with_null_speaker"], 1)
        self.assertEqual(segments[0]["start"], 0.0)
        self.assertEqual(segments[-1]["end"], 3.2)
        for segment in segments:
            self.assertEqual(set(segment), {"start", "end", "text", "speaker_id", "words"})
        # ElevenLabs-specific fields survive in the reference artifact.
        self.assertIn("logprob", output_objects[0])
        self.assertEqual(output_objects[1]["type"], "spacing")


if __name__ == "__main__":
    unittest.main()
