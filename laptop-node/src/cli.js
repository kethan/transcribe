#!/usr/bin/env node
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { access } from 'node:fs/promises';
import { decodeAudio, planChunks } from './audio.js';
import { OnnxTranscriber, LANGUAGES, transcribeFile } from './inference.js';
import { downloadModel, verifyModel } from './model.js';

const APP_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_MODEL_DIR = path.join(APP_ROOT, 'models', 'bodhan-core-int8-kv');
const HELP = `Bodhan offline CPU transcription (Node.js + ONNX Runtime)

Usage:
  node src/cli.js <audio.wav|audio.mp3|audio.flac> [options]
  node src/cli.js --download-model [options]

Options:
  --language CODE       Language code (default: hi). Available: ${LANGUAGES.join(', ')}
  --model-dir DIR       Folder containing the four pinned model files
  --download-model      Download and verify model files (internet used only for this command)
  --threads N           CPU worker threads (default: 4)
  --max-chunk-seconds N Audio chunk limit, 1–30 seconds (default: 28)
  --output FILE         Transcript text output (default: transcript.txt)
  --json FILE           Provenance JSON output (default: transcript.json)
  --help                Show this help

Transcription never makes network requests. For offline transfer, copy this app,
its node_modules/runtime, and all four verified model files together.
`;

function parseArgs(argv) {
  const options = { modelDir: DEFAULT_MODEL_DIR, language: 'hi', threads: 4, maxChunkSeconds: 28, output: 'transcript.txt', json: 'transcript.json', downloadModel: false, audio: null };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--help' || arg === '-h') options.help = true;
    else if (arg === '--download-model') options.downloadModel = true;
    else if (arg === '--language') options.language = argv[++i];
    else if (arg === '--model-dir') options.modelDir = path.resolve(argv[++i]);
    else if (arg === '--threads') options.threads = Number(argv[++i]);
    else if (arg === '--max-chunk-seconds') options.maxChunkSeconds = Number(argv[++i]);
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
  if (args.downloadModel) {
    await downloadModel(args.modelDir);
    console.log(`\nModel ready and checksum-verified: ${args.modelDir}`);
    return;
  }
  if (!args.audio) throw new Error('Provide an audio path, or use --download-model first.');
  if (!LANGUAGES.includes(args.language)) throw new Error(`Unsupported language ${args.language}. Use one of: ${LANGUAGES.join(', ')}`);
  if (!Number.isInteger(args.threads) || args.threads < 1 || args.threads > 64) throw new Error('--threads must be an integer from 1 to 64.');
  if (!(args.maxChunkSeconds > 0 && args.maxChunkSeconds <= 30)) throw new Error('--max-chunk-seconds must be in (0, 30].');
  await access(args.audio);
  const vocab = await verifyModel(args.modelDir);
  console.log('Decoding audio to 16 kHz mono...');
  const audio = await decodeAudio(args.audio);
  const { chunks } = planChunks(audio, args.maxChunkSeconds);
  console.log(`Audio: ${(audio.length / 16000).toFixed(1)}s; ${chunks.length} chunk(s); language=${args.language}`);
  console.log(`Loading CPU ONNX model (${args.threads} threads)...`);
  const transcriber = await OnnxTranscriber.create(args.modelDir, vocab, args.language, args.threads);
  const { text } = await transcribeFile(transcriber, args.audio, audio, chunks, args.output, args.json, args.language);
  console.log(`\nTranscript: ${path.resolve(args.output)}`);
  console.log(`Provenance: ${path.resolve(args.json)}`);
  console.log(`Characters: ${text.length}`);
}

main().catch((error) => {
  console.error(`Error: ${error.message}`);
  process.exitCode = 1;
});
