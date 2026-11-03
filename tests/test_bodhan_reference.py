from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from gujarati_pipeline.bodhan_text import write_bodhan_reference
from gujarati_pipeline.config import Config
from gujarati_pipeline.types import AudioSpan
from tools import bodhan_reference as br


def span(index: int, start: float, end: float) -> AudioSpan:
    return AudioSpan(
        index=index,
        start_sample=int(start * 16000),
        end_sample=int(end * 16000),
        start=start,
        end=end,
        path=Path(f"chunk-{index}.wav"),
        key=f"chunk-{index}",
        audio_hash="synthetic-cache-hash",
    )


def chunk(index: int, start: float, end: float, raw: str, parsed: str) -> SimpleNamespace:
    return SimpleNamespace(
        span=span(index, start, end),
        key=f"key-{index}",
        response={
            "text": raw,
            "backend": "bodhan-hf-local",
            "model": "bodhan-ai/indic-transcribe-core",
            "language": "hi",
        },
        text=parsed,
        has_word_times=False,
    )


class BodhanTextReferenceTests(unittest.TestCase):
    def test_live_chunks_write_verbatim_text_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = Config(output=root / "out" / "transcript.json", check=True)
            chunks = [
                chunk(0, 0.0, 2.0, "raw first chunk", "parsed first chunk"),
                chunk(1, 2.0, 4.0, "raw second chunk", "parsed second chunk"),
            ]
            txt_path, json_path, payload = write_bodhan_reference(chunks, cfg)
            self.assertEqual(txt_path.name, "bodhan_transcript.txt")
            self.assertEqual(json_path.name, "bodhan_transcript.json")
            self.assertEqual(
                txt_path.read_text(encoding="utf-8"),
                "raw first chunk\nraw second chunk\n",
            )
            self.assertEqual(payload["chunk_count"], 2)
            self.assertEqual(payload["text_characters"], len("raw first chunk\nraw second chunk"))
            self.assertEqual(payload["backend"], "hf-local")
            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8"))["text"], payload["text"])
            self.assertNotIn("ElevenLabs", payload["provenance"])
            self.assertIn("reference artifact only", payload["note"])

    def test_cache_exporter_selects_one_digest_and_never_mixes_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = Config(output=root / "out" / "transcript.json", check=True)
            digest = cfg.transcription_config_digest()
            responses = root / "work" / "transcription" / "responses"
            responses.mkdir(parents=True)
            for index, (start, end, text) in enumerate(
                [(0.0, 2.0, "cache first chunk"), (2.0, 4.0, "cache second chunk")]
            ):
                (responses / f"{index}.json").write_text(
                    json.dumps(
                        {
                            "version": 2,
                            "backend": "bodhan-hf-local",
                            "span_key": f"chunk-{index}",
                            "config_digest": digest,
                            "span": {
                                "index": index,
                                "start": start,
                                "end": end,
                                "key": f"chunk-{index}",
                            },
                            "response": {
                                "text": text,
                                "backend": "bodhan-hf-local",
                                "model": "bodhan-ai/indic-transcribe-core",
                                "language": "hi",
                            },
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
            report_path = root / "processing_report.json"
            report_path.write_text(
                json.dumps({"details": {"config": cfg.public_dict()}}),
                encoding="utf-8",
            )
            self.assertEqual(br.config_digest_from_report(report_path), digest)
            groups = br.cache_records(responses)
            key, records = br.select_group(
                groups, backend="bodhan-hf-local", digest=digest
            )
            payload = br.build_payload(key, records)
            self.assertEqual(payload["chunk_count"], 2)
            self.assertEqual(payload["text"], "cache first chunk\ncache second chunk")
            txt_path, json_path = br.write_outputs(
                payload, root / "download.txt", root / "download.json"
            )
            self.assertEqual(
                txt_path.read_text(encoding="utf-8"),
                "cache first chunk\ncache second chunk\n",
            )
            self.assertEqual(json.loads(json_path.read_text())["config_digest"], digest)
            groups[("bodhan-hf-local", "another-digest")] = records
            with self.assertRaises(RuntimeError):
                br.select_group(groups, backend="bodhan-hf-local", digest=None)


if __name__ == "__main__":
    unittest.main()
