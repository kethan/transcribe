# Bodhan Offline Transcriber (Node.js + ONNX)

A command-line CPU transcription app for Windows x64. It has the original INT8 ONNX/WASM path and an experimental FP32 ONNX/native-CPU path for Indic-Transcribe-Core. Neither path requires Python at runtime. The FP32 export is gated and must be downloaded separately by a user who has accepted its terms.

**Status:** audio decoding, chunking, the original INT8 path, FP32 vocabulary/safetensors parsing, and a JavaScript port of the FP32 frontend are implemented. Eleven Node tests pass, including a tiny native ONNX Runtime CPU graph. The FP32 path has **not** been tested with the speech model weights or compared with reference transcripts, so it is not yet a verified working transcription path and no transcript-quality improvement is claimed. Windows packaging and inference on the user's laptop also remain unverified. A repetition-loop guard can alter token choices and is not an accuracy fix.

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

## Experimental FP32 ONNX route

This route uses the gated `hari31416/indic-transcribe-onnx-fp32` export and native ONNX Runtime CPU. The pipeline was ported from the uploaded reference frontend and runner, but is not yet validated against the ONNX graphs or reference transcripts. Use the [pinned model revision](https://huggingface.co/hari31416/indic-transcribe-onnx-fp32/tree/d9450f0550e9a4dfbded6082be9ba3c7411137cc) for matching vocabulary/frontend files.

After accepting the model terms, download these files from the model repository into `laptop-node/models/indic-transcribe-core-fp32/`:

- `encoder-model.onnx` and `encoder-model.onnx.data`
- `decoder-model.onnx` and `decoder-model.onnx.data`
- `config.json`, `vocab.txt`, and `feature_extractor.safetensors`

The safetensors reader also accepts the uploaded attachment name `feature_extractor.safetensors.txt`; it parses the original binary safetensors bytes, not a text conversion. The frontend asset supplied for development is kept in the ignored local model directory and is not included in Git. Do not rename or edit the ONNX external-data sidecars.

On a Windows x64 machine with internet access, install the native runtime once and try a short clip:

```powershell
cd laptop-node
npm ci
node src/cli.js "C:\Audio\short-clip.mp3" --model-format fp32 --language hi --mode native
```

Then the app can run offline from the local model directory. FP32 is not downloaded automatically because the repository is gated; `--download-model` remains for the separate public INT8 model only. For this community export the publisher estimates roughly 4.59 GB of storage and about 5.15 GB peak inference RAM; leave extra disk/RAM headroom. Those figures are publisher estimates, not measurements from this app. Native-runtime installation and real inference still need verification on Windows.

To package this route for an offline Windows x64 laptop, after all model files and native runtime dependencies are present:

```powershell
npm ci --omit=dev
.\package-portable.ps1 -ModelFormat fp32
```

This copies the FP32 model, Node runtime, app, and Node modules into the portable folder/ZIP. The resulting archive is several gigabytes. This packaging route has not yet been run on Windows.

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

Both paths are CPU-only. The existing INT8 path uses WebAssembly; the experimental FP32 path uses native ONNX Runtime CPU. Four threads are used by default (`--threads 2` can reduce heat/memory pressure). Their speed and memory profiles differ; do not assume Python/native timings apply to the WASM route. Try a short clip first. Long recordings are split into contiguous chunks of up to 28 seconds.

The model is a community INT8/KV-cache derivative, not Bodhan's cloud service. There is no measured parity with the cloud API or benchmark on the target laptop. Keep a human review for important transcripts.

## License / attribution

Model attribution:

> Built with Indic-Transcribe-Core from Bodhan AI / AI4Bharat.<br>
> Base model: Canary-1B-v2 developed by NVIDIA Corporation (CC-BY-4.0).

Review the model repository's Indic Open Model License and usage terms before redistribution. `onnxruntime-web`, `onnxruntime-node`, `fft.js`, `wav-decoder`, `@audio/resample`, and the decoder packages declare MIT licenses. Review each model/runtime's terms before redistribution. This is not legal advice.

## Development checks

```powershell
npm test
node src/cli.js --help
```
