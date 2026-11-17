#!/usr/bin/env node
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { access } from 'node:fs/promises';
import { decodeAudio, planChunks } from './audio.js';
import { OnnxTranscriber, LANGUAGES, transcribeFile } from './inference.js';
import { createIndicCoreFp32Transcriber, INDIC_CORE_FP32_LANGUAGES, verifyIndicCoreFp32Model } from './fp32-model.js';
import { downloadModel, verifyModel } from './model.js';

const APP_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const INT8_MODEL_DIR = path.join(APP_ROOT, 'models', 'bodhan-core-int8-kv');
const FP32_MODEL_DIR = path.join(APP_ROOT, 'models', 'indic-transcribe-core-fp32');
const HELP = `Bodhan offline CPU transcription (Node.js + ONNX Runtime)

Usage:
  node src/cli.js <audio.wav|audio.mp3|audio.flac> [options]
  node src/cli.js --download-model [options]

Options:
  --model-format FORMAT  int8 (default) or fp32
  --language CODE       Language code (default: hi)
  --mode MODE           FP32 prompt mode: native, mixed, or romanized (default: native)
  --model-dir DIR       Folder containing model files
  --download-model      Download/verify the public INT8 model (not available for gated FP32)
  --threads N           CPU worker threads (default: 4)
  --max-chunk-seconds N Audio chunk limit, 1–30 seconds (default: 28)
  --max-new-tokens N    Decoder limit per chunk (default: 512)
  --output FILE         Transcript text output (default: transcript.txt)
  --json FILE           Provenance JSON output (default: transcript.json)
  --help                Show this help

FP32 is a separate, gated model download. See README.md for the required files.
Transcription itself makes no network requests.
`;

function parseArgs(argv) {
  const options = { modelDir: null, modelFormat: 'int8', language: 'hi', mode: 'native', threads: 4, maxChunkSeconds: 28, maxNewTokens: 512, output: 'transcript.txt', json: 'transcript.json', downloadModel: false, audio: null };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--help' || arg === '-h') options.help = true;
    else if (arg === '--download-model') options.downloadModel = true;
    else if (arg === '--language') options.language = argv[++i];
    else if (arg === '--model-format') options.modelFormat = argv[++i];
    else if (arg === '--mode') options.mode = argv[++i];
    else if (arg === '--model-dir') options.modelDir = path.resolve(argv[++i]);
    else if (arg === '--threads') options.threads = Number(argv[++i]);
    else if (arg === '--max-chunk-seconds') options.maxChunkSeconds = Number(argv[++i]);
    else if (arg === '--max-new-tokens') options.maxNewTokens = Number(argv[++i]);
    else if (arg === '--output') options.output = argv[++i];
    else if (arg === '--json') options.json = argv[++i];
    else if (arg.startsWith('-')) throw new Error(`Unknown option: ${arg}`);
    else if (!options.audio) options.audio = arg;
    else throw new Error(`Unexpected argument: ${arg}`);
  }
  return options;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) { console.log(HELP); return; }
  if (!['int8', 'fp32'].includes(args.modelFormat)) throw new Error('--model-format must be int8 or fp32.');
  args.modelDir ??= args.modelFormat === 'fp32' ? FP32_MODEL_DIR : INT8_MODEL_DIR;
  if (args.downloadModel) {
    if (args.modelFormat !== 'int8') throw new Error('The FP32 model is gated and must be downloaded manually from Hugging Face after accepting its terms.');
    await downloadModel(args.modelDir);
    console.log(`\nModel ready and checksum-verified: ${args.modelDir}`);
    return;
  }
  if (!args.audio) throw new Error('Provide an audio path, or use --download-model first.');
  const supportedLanguages = args.modelFormat === 'fp32' ? INDIC_CORE_FP32_LANGUAGES : LANGUAGES;
  if (!supportedLanguages.includes(args.language)) throw new Error(`Unsupported language ${args.language}. Use one of: ${supportedLanguages.join(', ')}`);
  if (args.modelFormat === 'int8' && args.mode !== 'native') throw new Error('The current INT8 decoder supports only --mode native.');
  if (!Number.isInteger(args.threads) || args.threads < 1 || args.threads > 64) throw new Error('--threads must be an integer from 1 to 64.');
  if (!(args.maxChunkSeconds > 0 && args.maxChunkSeconds <= 30)) throw new Error('--max-chunk-seconds must be in (0, 30].');
  if (!Number.isInteger(args.maxNewTokens) || args.maxNewTokens < 1 || args.maxNewTokens > 4096) throw new Error('--max-new-tokens must be an integer from 1 to 4096.');
  const fp32Assets = args.modelFormat === 'fp32' ? await verifyIndicCoreFp32Model(args.modelDir) : null;
  const int8Vocab = args.modelFormat === 'int8' ? await verifyModel(args.modelDir) : null;
  await access(args.audio);
  console.log('Decoding audio to 16 kHz mono...');
  const audio = await decodeAudio(args.audio, { resampler: args.modelFormat === 'fp32' ? 'indic-canary' : 'polyphase' });
  const { chunks } = planChunks(audio, args.maxChunkSeconds);
  console.log(`Audio: ${(audio.length / 16000).toFixed(1)}s; ${chunks.length} chunk(s); language=${args.language}`);
  console.log(`Loading ${args.modelFormat.toUpperCase()} CPU ONNX model (${args.threads} threads)...`);
  let transcriber;
  if (args.modelFormat === 'fp32') {
    transcriber = await createIndicCoreFp32Transcriber(args.modelDir, args.language, args.threads, args.maxNewTokens, args.mode, fp32Assets);
  } else {
    transcriber = await OnnxTranscriber.create(args.modelDir, int8Vocab, args.language, args.threads, args.maxNewTokens);
  }
  const { text } = await transcribeFile(transcriber, args.audio, audio, chunks, args.output, args.json, args.language);
  console.log(`\nTranscript: ${path.resolve(args.output)}`);
  console.log(`Provenance: ${path.resolve(args.json)}`);
  console.log(`Characters: ${text.length}`);
}

main().catch((error) => {
  console.error(`Error: ${error.message}`);
  process.exitCode = 1;
});
