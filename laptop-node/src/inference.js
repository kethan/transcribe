import * as ort from 'onnxruntime-web';
import { melFeatures } from './features.js';
import { MODEL_ID, REVISION, sha256File } from './model.js';

const EOS = 3;
const PAD = 2;
const LAYERS = 24;
const HEADS = 8;
const HEAD_DIM = 128;
const LANGUAGES = ['as', 'bn', 'brx', 'doi', 'gu', 'hi', 'kn', 'ks', 'kok', 'mai', 'ml', 'mni', 'mr', 'ne', 'or', 'pa', 'sa', 'sat', 'sd', 'ta', 'te', 'ur', 'en', 'bho', 'bhb'];

function buildPrompt(vocab, language) {
  if (!LANGUAGES.includes(language)) throw new Error(`Unsupported language code: ${language}`);
  const langId = vocab.indexOf(`<|${language}|>`);
  if (langId < 0) throw new Error(`Language token <|${language}|> not found in model vocabulary.`);
  // Fixed Bodhan Core native-script, punctuation-enabled Canary prompt.
  return [7, 4, 18, langId, langId, 5, 9, 11, 13, 15];
}

function argmaxLastLogits(tensor, banned = new Set()) {
  const data = tensor.data;
  const dims = tensor.dims;
  const vocabSize = dims[dims.length - 1];
  const start = data.length - vocabSize;
  let bestIndex = -1;
  let bestValue = -Infinity;
  for (let i = 0; i < vocabSize; i++) {
    if (banned.has(i)) continue;
    const value = data[start + i];
    if (value > bestValue) { bestValue = value; bestIndex = i; }
  }
  return bestIndex;
}

function detokenize(vocab, ids) {
  return ids.map((id) => vocab[id])
    .filter((piece) => piece !== '<pad>' && !(piece.startsWith('<|') && piece.endsWith('|>')))
    .join('').replaceAll('\u2581', ' ').trim();
}

// Detect a repeated suffix cycle so the decoder can reject the greedy token
// that would extend it before it becomes an unbounded loop.
export function detectRepetitionLoop(ids, maxPeriod = 16) {
  const limit = Math.min(maxPeriod, Math.floor(ids.length / 3));
  for (let period = 1; period <= limit; period++) {
    const minCycles = period === 1 ? 8 : 4;
    let repeated = 0;
    for (let i = ids.length - 1; i >= period && ids[i] === ids[i - period]; i--) repeated++;
    const cycles = 1 + Math.floor(repeated / period);
    if (cycles < minCycles) continue;
    // Ignore non-primitive periods (e.g. a run of one token also looks like
    // period two) so the shortest meaningful cycle determines the cut point.
    let primitive = true;
    for (let smaller = 1; smaller < period; smaller++) {
      if (period % smaller !== 0) continue;
      let matches = true;
      for (let i = smaller; i < period; i++) {
        if (ids[ids.length - period + i] !== ids[ids.length - period + (i % smaller)]) { matches = false; break; }
      }
      if (matches) { primitive = false; break; }
    }
    if (!primitive) continue;
    return { period, cycles };
  }
  return null;
}

export function chooseTokenWithLoopGuard(logits, history) {
  const greedy = argmaxLastLogits(logits);
  if (greedy < 0 || greedy === EOS || greedy === PAD) {
    return { token: greedy, loopDetected: false, loopPeriod: null, suppressedCount: 0 };
  }
  const firstLoop = detectRepetitionLoop([...history, greedy]);
  if (!firstLoop) return { token: greedy, loopDetected: false, loopPeriod: null, suppressedCount: 0 };

  // Reject only candidate tokens that would extend the detected cycle, then
  // continue with the next-highest logit. If all viable candidates loop, stop
  // and flag the chunk instead of silently writing an unbounded repetition.
  const banned = new Set([greedy]);
  while (banned.size < logits.dims.at(-1)) {
    const alternative = argmaxLastLogits(logits, banned);
    if (alternative < 0 || alternative === EOS || alternative === PAD) break;
    const alternativeLoop = detectRepetitionLoop([...history, alternative]);
    if (!alternativeLoop) {
      return { token: alternative, loopDetected: true, loopPeriod: firstLoop.period, suppressedCount: banned.size };
    }
    banned.add(alternative);
  }
  return { token: null, loopDetected: true, loopPeriod: firstLoop.period, suppressedCount: banned.size };
}

export class OnnxTranscriber {
  static async create(modelDir, vocab, language = 'hi', threads = 4, maxNewTokens = 512) {
    // The WebAssembly execution provider keeps deployment inside npm packages:
    // no Python, native toolchain, GPU, or separate ONNX Runtime download.
    ort.env.wasm.numThreads = threads;
    ort.env.wasm.proxy = false;
    const options = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
    const [encoder, cross, decoder] = await Promise.all([
      ort.InferenceSession.create(`${modelDir}/encoder.onnx`, options),
      ort.InferenceSession.create(`${modelDir}/cross_kv.onnx`, options),
      ort.InferenceSession.create(`${modelDir}/decoder_kv.onnx`, options),
    ]);
    return new OnnxTranscriber(encoder, cross, decoder, vocab, language, maxNewTokens);
  }

  constructor(encoder, cross, decoder, vocab, language, maxNewTokens) {
    this.encoder = encoder;
    this.cross = cross;
    this.decoder = decoder;
    this.vocab = vocab;
    this.language = language;
    this.prompt = buildPrompt(vocab, language);
    this.maxNewTokens = maxNewTokens;
    this.crossNames = Array.from({ length: LAYERS }, (_, layer) => ['key', 'value'].map((kind) => `cross_${kind}.${layer}`)).flat();
    this.pastNames = Array.from({ length: LAYERS }, (_, layer) => ['key', 'value'].map((kind) => `past_self_${kind}.${layer}`)).flat();
  }

  async decode(samples) {
    const { data, dims } = melFeatures(samples);
    const featureTensor = new ort.Tensor('float32', data, dims);
    const encoded = await this.encoder.run({ feats: featureTensor });
    const cross = await this.cross.run({ encoder_hidden_states: encoded.hidden_states });
    const crossFeed = Object.fromEntries(this.crossNames.map((name) => [name, cross[name]]));
    const emptyCache = () => new ort.Tensor('float32', new Float32Array(0), [1, HEADS, 0, HEAD_DIM]);
    const feed = {
      input_ids: new ort.Tensor('int64', BigInt64Array.from(this.prompt, BigInt), [1, this.prompt.length]),
      ...Object.fromEntries(this.pastNames.map((name) => [name, emptyCache()])),
      ...crossFeed,
    };
    let output = await this.decoder.run(feed);
    const tokens = [];
    let choice = chooseTokenWithLoopGuard(output.logits, tokens);
    let truncated = choice.token === null;
    let loopDetected = choice.loopDetected;
    let loopPeriod = choice.loopPeriod;
    let suppressedCount = choice.suppressedCount;
    let stopReason = choice.token === null ? 'repetition_loop_unrecoverable' : 'eos';
    let next = choice.token;
    while (next !== null && next !== EOS && next !== PAD) {
      if (tokens.length >= this.maxNewTokens) { truncated = true; stopReason = 'max_new_tokens'; break; }
      tokens.push(next);
      const nextFeed = { input_ids: new ort.Tensor('int64', BigInt64Array.of(BigInt(next)), [1, 1]), ...crossFeed };
      for (let layer = 0; layer < LAYERS; layer++) {
        nextFeed[`past_self_key.${layer}`] = output[`present_self_key.${layer}`];
        nextFeed[`past_self_value.${layer}`] = output[`present_self_value.${layer}`];
      }
      output = await this.decoder.run(nextFeed);
      choice = chooseTokenWithLoopGuard(output.logits, tokens);
      next = choice.token;
      if (choice.loopDetected) {
        loopDetected = true;
        loopPeriod ??= choice.loopPeriod;
        suppressedCount += choice.suppressedCount;
      }
      if (next === null) { truncated = true; stopReason = 'repetition_loop_unrecoverable'; }
    }
    return {
      text: detokenize(this.vocab, tokens),
      tokenCount: tokens.length,
      truncated,
      loopDetected,
      loopPeriod,
      suppressedCount,
      stopReason,
    };
  }
}

export async function transcribeFile(transcriber, audioPath, audio, chunks, outputPath, jsonPath, language) {
  const entries = [];
  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];
    console.error(`[${String(i + 1).padStart(3, '0')}/${String(chunks.length).padStart(3, '0')}] ${chunk.start.toFixed(2)}–${chunk.end.toFixed(2)}s`);
    const decoded = await transcriber.decode(audio.subarray(chunk.startSample, chunk.endSample));
    if (decoded.loopDetected) {
      const detail = decoded.stopReason === 'repetition_loop_unrecoverable'
        ? 'no safe alternative token was available; this chunk was cut short'
        : `blocked ${decoded.suppressedCount} loop candidate(s) and continued`;
      console.error(`  WARNING: repetition loop (period ${decoded.loopPeriod}); ${detail}. Review this segment.`);
    }
    entries.push({ ...chunk, index: i + 1, ...decoded });
  }
  const text = entries.map((entry) => entry.text).filter(Boolean).join(' ');
  const loopChunks = entries.filter((entry) => entry.loopDetected).map((entry) => entry.index);
  const unrecoverableLoopChunks = entries.filter((entry) => entry.stopReason === 'repetition_loop_unrecoverable').map((entry) => entry.index);
  const truncatedChunks = entries.filter((entry) => entry.stopReason === 'max_new_tokens').map((entry) => entry.index);
  const payload = {
    model: { id: MODEL_ID, revision: REVISION, source: 'bodhan-ai/indic-transcribe-core', format: 'INT8 ONNX with KV-cache decoder', runtime: `ONNX Runtime Web ${ort.env.versions?.web ?? 'unknown'} (WASM CPU)`, accuracyNote: 'No measured parity with the Bodhan cloud API.' },
    audio: { path: audioPath, sha256: await sha256File(audioPath), sampleRate: 16000, durationSeconds: Number((audio.length / 16000).toFixed(6)) },
    language, prompt: transcriber.prompt, text,
    chunks: entries,
    warnings: [
      ...(loopChunks.length ? [`Repetition guard changed the greedy token choice in chunks: ${loopChunks.join(', ')}. Review those segments; this may alter wording.`] : []),
      ...(unrecoverableLoopChunks.length ? [`No non-looping token was available; chunks ended early: ${unrecoverableLoopChunks.join(', ')}.`] : []),
      ...(truncatedChunks.length ? [`Maximum token limit reached in chunks: ${truncatedChunks.join(', ')}.`] : []),
    ],
    notes: ['Transcript is generated by ONNX inference; no transcript is hardcoded.', 'CPU-only path: no cloud calls, GPU, Python, NeMo, or diarization.', 'The repetition guard selects a lower-ranked token when greedy decoding enters a detected loop; it can change words and does not ensure transcript accuracy.'],
  };
  const { mkdir, writeFile } = await import('node:fs/promises');
  const { dirname, resolve } = await import('node:path');
  await mkdir(dirname(resolve(outputPath)), { recursive: true });
  await mkdir(dirname(resolve(jsonPath)), { recursive: true });
  await writeFile(outputPath, `${text}\n`, 'utf8');
  await writeFile(jsonPath, `${JSON.stringify(payload, null, 2)}\n`, 'utf8');
  return { text, payload };
}

export { LANGUAGES };
