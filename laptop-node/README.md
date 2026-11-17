# Bodhan Offline Transcriber (Node.js + ONNX)

A small command-line CPU transcription app for Windows x64 using Bodhan Indic-Transcribe-Core weights in INT8 ONNX format. The inference code uses ONNX Runtime Web's WebAssembly CPU provider, so it does not require Python, CUDA, or a separately installed native ONNX Runtime. The supplied model files are pinned and SHA-256 verified.

**Status:** the Node app, audio decoding, feature extraction, chunking, model-file verification, and ONNX Runtime WASM int8-operator smoke test are implemented. Unit tests pass; JS mel features matched the existing Python/librosa frontend on a one-second tone (maximum absolute difference about 6.3e-5). Full speech inference has **not** been run here because the 1.57 GB ONNX model files are not in the repository. WASM CPU speed and accuracy on the user's Intel laptop are not yet measured.

## Run with Node.js available

Use 64-bit Node.js 20 or 22 on Windows (or Node.js 20+ on Linux for development). In PowerShell:

```powershell
cd laptop-node
npm ci
npm start -- --download-model
npm start -- "C:\Audio\meeting.mp3" --language hi --output "C:\Audio\transcript.txt" --json "C:\Audio\transcript.json"
```

`--download-model` is the only command that accesses the network. It downloads the pinned four-file model from Hugging Face and checks every SHA-256. Transcription itself does not make network requests. You can instead place the four files in `models\bodhan-core-int8-kv` or pass `--model-dir`.

Inputs: WAV, MP3, and FLAC. Bundled JavaScript/WebAssembly decoders and a polyphase resampler convert audio to 16 kHz mono; no FFmpeg executable is needed. Default language is Hindi (`hi`); use `--help` for the supported language codes and options. Output is UTF-8 plain text plus provenance JSON; no diarization, timestamps, or cloud comparison is included.

## Portable Windows folder / ZIP (no Python or installed Node on the target)

Prepare this on a Windows x64 machine that has Node.js 20/22 and internet access:

```powershell
cd laptop-node
npm ci --omit=dev
npm start -- --download-model
.\package-portable.ps1
```

The script verifies the local model files, copies the active `node.exe`, app code, npm runtime packages, and model files into `release\BodhanTranscribe`, and creates `release\BodhanTranscribe-win-x64.zip`. Copy/extract that folder on the offline laptop and run:

```cmd
run.cmd "D:\Audio\meeting.mp3" --language hi
```

The transfer includes the runtime and model, so the destination needs no Python, Node installation, admin rights, or internet. Expect a large transfer: the ONNX model is about 1.57 GB before app/runtime files. This portable-packaging script is written for Windows but has not been executed in a Windows build environment in this workspace.

## Hardware / performance notes

This is CPU-only and uses WebAssembly rather than the faster native ONNX Runtime CPU provider. Four threads are used by default (`--threads 2` can reduce heat/memory pressure). WASM has a different speed/memory profile; do not assume the prior Python/native ONNX timings apply. Try a short clip first. Long recordings are split into contiguous chunks of up to 28 seconds.

The model is a community INT8/KV-cache derivative, not Bodhan's cloud service. There is no measured parity with the cloud API or benchmark on the target laptop. Keep a human review for important transcripts.

## License / attribution

Model attribution:

> Built with Indic-Transcribe-Core from Bodhan AI / AI4Bharat.<br>
> Base model: Canary-1B-v2 developed by NVIDIA Corporation (CC-BY-4.0).

Review the model repository's Indic Open Model License and usage terms before redistribution. `onnxruntime-web`, `fft.js`, `wav-decoder`, `@audio/resample`, and the decoder packages declare MIT licenses. Review the model repository's separate terms before redistribution. This is not legal advice.

## Development checks

```powershell
npm test
node src/cli.js --help
```
