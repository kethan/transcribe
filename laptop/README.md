# CPU-only laptop transcription

A separate, small path for a 16 GB Intel i7 laptop with no GPU. It writes plain
UTF-8 transcript text and one provenance JSON. It does **not** produce
diarization, speaker IDs, word timestamps, CTC alignment, speaker turns, or the
strict `transcript.json` schema.

It uses ONNX Runtime CPU and a pinned community INT8 ONNX derivative of
`bodhan-ai/indic-transcribe-core`. No cloud key, `torch`, NeMo, CUDA, Kaggle,
or gated Hugging Face access is needed.

> Attribution: **Built with Indic-Transcribe-Core from Bodhan AI / AI4Bharat.**
> **Base model: Canary-1B-v2 developed by NVIDIA Corporation (CC-BY-4.0).**

## Model choice

The exact linked repo
[`adidsh/indic-transcribe-core-int8-onnx`](https://huggingface.co/adidsh/indic-transcribe-core-int8-onnx)
is a ~1.57 GB INT8 ONNX port whose decoder recomputes the whole token prefix
each step. This script uses the pinned
[`schamp007/verbatim-hindi-asr-core-sc`](https://huggingface.co/schamp007/verbatim-hindi-asr-core-sc)
re-export: same INT8 encoder/weights, plus a KV-cache decoder. Its card reports
the same Hindi WER/CER as the no-cache port, ~5.2x faster end-to-end on a
4.6-minute sample, ~2.7 GB peak RAM, and four differing words in 611. Those
numbers were measured on an Apple M4/4 threads, so an Intel i7 may be slower.
Only Hindi native/mixed was measured; other language codes come from the
[upstream Core card](https://huggingface.co/bodhan-ai/indic-transcribe-core).

ONNX was chosen over the GGUF files in the linked repo because this Python/ONNX
Runtime path has documented tensor names and a working x86 CPU decoding loop.

## Accuracy caveat

**Exact “same accuracy as Bodhan cloud API” is not guaranteed.** This is a
community INT8 derivative, not an official cloud-equivalent runtime.
Quantization, 28 s chunking, front-end details, greedy decoding, and any
cloud-side VAD/post-processing can change words. The derivative was compared
with the no-cache INT8 port, not with the cloud API. Core is native-script
oriented and optimized for single-speaker conversational speech; the attached
MP3 is a multi-speaker meeting.

Use `compare.py` to measure WER/CER against a saved cloud response or a
manually corrected reference. Do not use the ElevenLabs/Scribe 2 JSON as ground
truth. If you require the same implementation as the cloud service, use the
cloud API itself; even the original FP32 upstream model can differ.

## Install

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\Activate.ps1      # Windows PowerShell
pip install -r laptop/requirements.txt
```

If MP3 decoding fails, install `ffmpeg` on `PATH` or convert to 16 kHz mono
WAV. WAV/FLAC normally work through soundfile/librosa.

## One-time download

```bash
python -m laptop.transcribe --download-only
```

This downloads ~1.57 GB to
`~/.cache/transcribe-laptop/verbatim-hindi-asr-core-sc` (or
`%USERPROFILE%\.cache\...` on Windows) and verifies SHA-256 hashes. Override
with `--model-dir` or `LAPTOP_ASR_MODEL_DIR`. Use `--offline` on later runs. If
a checksum fails, delete the named file and rerun without `--offline`.

## Transcribe

Default is Hindi/Devanagari, matching the attached MP3:

```bash
python -m laptop.transcribe "audio/20261031 CHD 3 GTH 11 12 13 14 15 (51) CHN.mp3" \
  --language hi \
  --output-txt outputs/cpu_transcript.txt \
  --output-json outputs/cpu_transcript.json
```

Other examples:

```bash
# See chunk plan without downloading weights:
python -m laptop.transcribe "audio/...CHN.mp3" --dry-run

# Two-chunk smoke test (partial transcript):
python -m laptop.transcribe "audio/...CHN.mp3" --limit-chunks 2 \
  --output-txt outputs/cpu_smoke.txt --output-json outputs/cpu_smoke.json

# Gujarati:
python -m laptop.transcribe recording.wav --language gu

# Fewer CPU threads / smaller windows if RAM or heat is an issue:
python -m laptop.transcribe recording.wav --threads 2 --max-chunk-seconds 20
```

The prompt is fixed to native script with `pnc=yes`; `--language` selects the
language token. `--max-chunk-seconds` must be ≤30; default 28. The script cuts
at the quietest point in the last `--search-seconds` (default 3) before the
hard limit and never skips audio. On the attached 75-minute MP3, `--dry-run`
plans **171 contiguous chunks**.

Outputs:

- `cpu_transcript.txt`: chunk outputs joined with spaces.
- `cpu_transcript.json`: model revision, prompt, absolute chunk boundaries,
  per-chunk text, SHA-256 digests, warnings, and notes. Not the strict pipeline
  schema. Chunks are ASR windows, not speaker turns.

No transcript text is hardcoded; it comes only from live ONNX decoder output.

## Compare with cloud/manual reference

```bash
python -m laptop.compare \
  --reference cloud.txt \
  --hypothesis outputs/cpu_transcript.txt \
  --output-json outputs/cpu_compare.json
```

Punctuation is ignored by default. Add `--keep-punctuation` to score it, or
`--lowercase` for Latin code-mixed text. Lower WER/CER is better; WER can
exceed 100% on insertions. `compare.py` never calls the cloud API and the
result is not proof of parity.

## What the script does

1. Decode audio to 16 kHz mono float32.
2. Split into contiguous ≤28 s windows at quiet points.
3. Compute the documented front-end: <1 s zero-pad, pre-emphasis 0.97,
   reflect-pad 256, STFT 512/160 with centred symmetric Hann(400), 128-bin
   Slaney mel, `ln(x + 2^-24)`, per-bin normalization.
4. Run `encoder.onnx`, `cross_kv.onnx`, then greedy autoregressive
   `decoder_kv.onnx` with growing KV caches. Stop on EOS/PAD or 512 new tokens.
5. Build the fixed 10-token prompt `[7,4,18,L,L,5,9,11,13,15]`; Hindi native
   is `[7,4,18,89,89,5,9,11,13,15]`.
6. Detokenize generated IDs and write UTF-8 text plus provenance JSON.

No beam search, LM, punctuation model, diarization, alignment, or cloud
fallback. For speakers/word times, use the main Kaggle/NeMo pipeline.
