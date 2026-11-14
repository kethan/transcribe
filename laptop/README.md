# CPU-only laptop transcription (plain text)

This is a **separate, small path** for a 16 GB RAM Intel Core i7 laptop with no
GPU. It does **not** modify or replace the Kaggle/NeMo pipeline in the rest of
this repository. It produces downloadable UTF-8 transcript text, without
diarization, speaker IDs, word timestamps, CTC alignment, or the strict
`transcript.json` schema.

It runs a pinned community INT8 ONNX derivative of
`bodhan-ai/indic-transcribe-core` with **ONNX Runtime CPU**. No Bodhan cloud
API key, `torch`, NeMo, CUDA, Kaggle, or gated HF access is required.

> Attribution required by the model license:  
> **Built with Indic-Transcribe-Core from Bodhan AI / AI4Bharat.**  
> **Base model: Canary-1B-v2 developed by NVIDIA Corporation (CC-BY-4.0).**

## What was chosen and why

- The exact linked repository,
  [`adidsh/indic-transcribe-core-int8-onnx`](https://huggingface.co/adidsh/indic-transcribe-core-int8-onnx),
  is a 1.45–1.57 GB INT8 ONNX export intended for Android. Its decoder
  recomputes the whole token prefix at every step.
- This script uses
  [`schamp007/verbatim-hindi-asr-core-sc`](https://huggingface.co/schamp007/verbatim-hindi-asr-core-sc),
  a pinned re-export with the **same INT8 encoder/weights** and a KV-cache
  decoder. Its card reports the same Hindi WER/CER as the no-cache port, about
  **5.2x faster end-to-end** on a 4.6-minute Hindi recording, ~2.7 GB peak
  process RAM, and only four differing words in 611 on that sample. Those
  numbers were measured on an Apple M4 with four CPU threads, so an Intel i7
  laptop may be slower.
- ONNX was chosen over the GGUF files in the linked repository because this
  Python/ONNX Runtime path has documented tensor names and a working CPU
  decoding loop. GGUF does not by itself provide Canary ASR decoding on x86.

Sources: [linked INT8 ONNX card](https://huggingface.co/adidsh/indic-transcribe-core-int8-onnx),
[KV-cache derivative card](https://huggingface.co/schamp007/verbatim-hindi-asr-core-sc),
[upstream Bodhan Core card](https://huggingface.co/bodhan-ai/indic-transcribe-core).

## Accuracy expectation (important)

**Exact “same accuracy as Bodhan cloud API” cannot be guaranteed.** This is a
community INT8-quantized derivative, not an official cloud-equivalent runtime.
Quantization, 28 s chunking, feature extraction, greedy decoding, and any
cloud-side long-form VAD/post-processing can change words. The derivative
author measured parity only against the no-cache INT8 port, not against the
Bodhan cloud API; only Hindi native/mixed was measured. The Core card also
says it is optimized for native Indic scripts and single-speaker conversational
speech, while the attached MP3 is a multi-speaker meeting.

What you get instead:

- local text from the same upstream Core model lineage;
- a reproducible, pinned CPU run that needs no cloud key;
- an optional `compare.py` WER/CER report against a saved cloud response or a
  manually corrected reference.

If byte-for-byte or measured-WER parity with the cloud API is a hard
requirement, use the cloud API directly or run the original FP32 upstream model;
this INT8 path cannot promise it. Do not treat the ElevenLabs/Scribe 2 JSON as
ground truth.

## Install

From the repository root:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
pip install -r laptop/requirements.txt
```

`requirements.txt` installs CPU ONNX Runtime, NumPy, librosa, soundfile,
huggingface-hub, and rapidfuzz. If MP3 decoding fails, install `ffmpeg` and put
it on `PATH`; WAV/FLAC normally work through soundfile/librosa directly.

## One-time model download

```bash
python -m laptop.transcribe --download-only
```

Default model location:

- Linux/macOS: `~/.cache/transcribe-laptop/verbatim-hindi-asr-core-sc`
- Windows: `%USERPROFILE%\.cache\transcribe-laptop\verbatim-hindi-asr-core-sc`

Override with `--model-dir` or the `LAPTOP_ASR_MODEL_DIR` environment variable.
The first run downloads about **1.57 GB** and verifies SHA-256 checksums. Use
`--offline` afterwards. Use `--force-download` to repair a checksum mismatch.
Use `--check` to verify files and print the CPU ONNX Runtime provider without
transcribing.

## Transcribe

Hindi/Devanagari is the default, matching the attached MP3:

```bash
python -m laptop.transcribe "audio/20261031 CHD 3 GTH 11 12 13 14 15 (51) CHN.mp3" \
  --language hi \
  --output-txt outputs/cpu_transcript.txt \
  --output-json outputs/cpu_transcript.json \
  --threads 4
```

Quick checks:

```bash
# See the <=28 s chunk plan without downloading weights:
python -m laptop.transcribe "audio/...CHN.mp3" --dry-run

# Decode only the first two chunks to validate the setup:
python -m laptop.transcribe "audio/...CHN.mp3" --limit-chunks 2 \
  --output-txt outputs/cpu_smoke.txt --output-json outputs/cpu_smoke.json

# Gujarati:
python -m laptop.transcribe recording.wav --language gu \
  --output-txt outputs/gu.txt --output-json outputs/gu.json

# No punctuation/capitalization prompt token:
python -m laptop.transcribe recording.wav --pnc no
```

`--limit-chunks` produces a **partial** transcript and records that fact in the
JSON. `--script mixed` switches the prompt token for mixed Latin/English words,
but Core is native-script oriented; Indic-Transcribe-Flex is the better model
for heavy Hinglish/transliteration. `--list-languages` prints the 25 upstream
Core codes.

### Outputs

- `cpu_transcript.txt`: plain UTF-8 text, chunk outputs joined with spaces.
- `cpu_transcript.json`: model revision/hashes, language/prompt, chunk
  boundaries in absolute seconds, per-chunk text, text SHA-256, warnings, and
  limitations. It is **not** the strict pipeline `transcript.json`.

No transcript text is hardcoded. Text comes only from the ONNX decoder's
generated token IDs. Chunks are ASR processing windows, not speaker turns.

## Compare with a cloud/manual reference

Save the Bodhan cloud API text (or a manually corrected transcript) as
`cloud.txt`, then run:

```bash
python -m laptop.compare \
  --reference cloud.txt \
  --hypothesis outputs/cpu_transcript.txt \
  --output-json outputs/cpu_compare.json \
  --output-md outputs/cpu_compare.md
```

Punctuation is ignored by default. Add `--keep-punctuation` to score it, or
`--lowercase` for Latin code-mixed text. WER/CER are lower-is-better; WER can
exceed 100% when insertions outnumber reference words. The report is a
diagnostic, not proof of parity. `compare.py` never calls the cloud API.

## How it works

1. Decode any input audio to 16 kHz mono float32 (ffmpeg fallback for MP3).
2. Measure 20 ms RMS frames and split contiguous audio at the quietest point in
   the last `--search-seconds` before a 28 s hard limit. No audio is skipped.
3. Compute the exact documented front-end: <1 s zero-pad, pre-emphasis 0.97,
   reflect-pad 256, STFT 512/160 with centred symmetric Hann(400), 128-bin
   Slaney mel, `ln(x + 2^-24)`, per-bin unbiased-std normalization.
4. Run `encoder.onnx` once, `cross_kv.onnx` once, then autoregressive
   `decoder_kv.onnx` with growing self-attention KV caches. Greedy argmax stops
   on EOS/PAD; cap is 512 new tokens per chunk.
5. Build the 10-token Canary-style prompt
   `[7,4,18,L,L,PNC,ITN,11,13,15]`. Hindi native is
   `[7,4,18,89,89,5,9,11,13,15]`. Token 5 is `<|pnc|>`; token 9 selects the
   native-script path.
6. Detokenize generated IDs from `bodhan_vocab.json`, replace the SentencePiece
   word-boundary mark with a space, and write text plus provenance JSON.

There is no beam search, external LM, punctuation model, diarization, CTC
alignment, or cloud fallback in this path.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Checksum mismatch | Rerun with `--force-download`, or delete the named file. |
| MP3 cannot be decoded | Install `ffmpeg`, or convert to 16 kHz mono WAV first. |
| High RAM / thermal throttling | Try `--threads 2` and `--max-chunk-seconds 20`. |
| Garbled script | Check `--language`; keep `--script native` unless the audio is genuinely code-mixed. |
| Missing punctuation | Keep the default `--pnc yes`. |
| Very slow full meeting | Use `--dry-run` first; `--limit-chunks 2` for a smoke test. The attached 75-minute MP3 plans 171 contiguous chunks with the default 28 s limit; other audio may differ. |
| Need speakers/word times | Use the main Kaggle/NeMo pipeline instead. |
