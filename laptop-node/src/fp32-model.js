import { readFile, stat } from 'node:fs/promises';
import path from 'node:path';
import { chooseTokenWithLoopGuard } from './inference.js';
import { indicCanaryFeatures } from './fp32-features.js';
import { loadFeatureExtractor } from './safetensors.js';

export const INDIC_CORE_FP32_REVISION = 'd9450f0550e9a4dfbded6082be9ba3c7411137cc';
export const INDIC_CORE_FP32_MODEL_ID = 'hari31416/indic-transcribe-onnx-fp32';

const REQUIRED_FILES = [
  'config.json',
  'encoder-model.onnx',
  'encoder-model.onnx.data',
  'decoder-model.onnx',
  'decoder-model.onnx.data',
  'vocab.txt',
];

export function parseIndicCoreVocabulary(text) {
  const tokens = [];
  const tokenToId = new Map();
  for (const [lineIndex, line] of text.split(/\r?\n/).entries()) {
    if (!line) continue;
    const separator = line.lastIndexOf(' ');
    if (separator < 1) throw new Error(`Invalid vocabulary row ${lineIndex + 1}.`);
    const piece = line.slice(0, separator).replaceAll('\u2581', ' ');
    const idText = line.slice(separator + 1).trim();
    if (!/^\d+$/.test(idText)) throw new Error(`Invalid token id on vocabulary row ${lineIndex + 1}.`);
    const id = Number(idText);
    if (!Number.isSafeInteger(id) || id < 0 || tokens[id] !== undefined) throw new Error(`Duplicate or invalid token id ${idText}.`);
    tokens[id] = piece;
    tokenToId.set(piece, id);
  }
  let hasGap = false;
  for (let id = 0; id < tokens.length; id++) if (tokens[id] === undefined) { hasGap = true; break; }
  if (tokens.length !== 7152 || hasGap) {
    throw new Error(`Expected a contiguous 7,152-token vocabulary; found ${tokens.length} rows.`);
  }
  return { tokens, tokenToId };
}

export async function verifyIndicCoreFp32Model(modelDir) {
  for (const name of REQUIRED_FILES) {
    const file = path.join(modelDir, name);
    let info;
    try { info = await stat(file); } catch { throw new Error(`Missing FP32 model file: ${file}`); }
    if (!info.isFile() || info.size === 0) throw new Error(`FP32 model file is empty or invalid: ${file}`);
  }
  const [configText, vocabText, featureExtractor] = await Promise.all([
    readFile(path.join(modelDir, 'config.json'), 'utf8'),
    readFile(path.join(modelDir, 'vocab.txt'), 'utf8'),
    loadFeatureExtractor(modelDir),
  ]);
  let config;
  try { config = JSON.parse(configText); } catch (error) { throw new Error(`Invalid FP32 config.json: ${error.message}`); }
  if (config.model_type !== 'nemo-conformer-aed' || config.features_size !== 128 || config.vocab_size !== 7152) {
    throw new Error('Unsupported FP32 model configuration; expected Indic Canary AED with 128 features and 7,152 vocabulary entries.');
  }
  const vocabulary = parseIndicCoreVocabulary(vocabText);
  return { modelDir, config, vocabulary, featureExtractor };
}

export async function createIndicCoreFp32Transcriber(modelDir, language = 'hi', threads = 4, maxNewTokens = 512, mode = 'native', verifiedAssets = null) {
  const assets = verifiedAssets ?? await verifyIndicCoreFp32Model(modelDir);
  let ort;
  try {
    ort = await import('onnxruntime-node');
  } catch (error) {
    throw new Error(`FP32 transcription requires the native ONNX Runtime Node package. Install laptop-node dependencies on Windows with internet access, then copy the packaged app offline. Original error: ${error.message}`);
  }

  const options = {
    executionProviders: ['cpu'],
    graphOptimizationLevel: 'all',
    intraOpNumThreads: threads,
    interOpNumThreads: 1,
  };
  // Load sequentially to avoid an avoidable spike in transient memory.
  const encoder = await ort.InferenceSession.create(path.join(modelDir, 'encoder-model.onnx'), options);
  let decoder;
  try {
    decoder = await ort.InferenceSession.create(path.join(modelDir, 'decoder-model.onnx'), options);
  } catch (error) {
    await encoder.release();
    throw error;
  }
  const requiredEncoderInputs = ['audio_signal', 'length'];
  const requiredEncoderOutputs = ['encoder_embeddings', 'encoder_mask'];
  const requiredDecoderInputs = ['input_ids', 'encoder_embeddings', 'encoder_mask', 'decoder_mems'];
  const requiredDecoderOutputs = ['logits', 'decoder_hidden_states'];
  const missingEncoder = requiredEncoderInputs.filter((name) => !encoder.inputNames.includes(name));
  const missingEncoderOutputs = requiredEncoderOutputs.filter((name) => !encoder.outputNames.includes(name));
  const missingDecoder = requiredDecoderInputs.filter((name) => !decoder.inputNames.includes(name));
  const missingDecoderOutputs = requiredDecoderOutputs.filter((name) => !decoder.outputNames.includes(name));
  if (missingEncoder.length || missingEncoderOutputs.length || missingDecoder.length || missingDecoderOutputs.length) {
    await Promise.all([encoder.release(), decoder.release()]);
    throw new Error(`Unexpected FP32 ONNX graph I/O; encoder missing inputs [${missingEncoder}]/outputs [${missingEncoderOutputs}], decoder missing inputs [${missingDecoder}]/outputs [${missingDecoderOutputs}].`);
  }
  const memsMetadata = decoder.inputMetadata.find((item) => item.name === 'decoder_mems');
  const memsShape = memsMetadata?.shape ?? [];
  const transcriber = new IndicCoreFp32Transcriber({
    ort,
    encoder,
    decoder,
    assets,
    language,
    maxNewTokens,
    mode,
    numLayers: Number.isInteger(memsShape[0]) ? memsShape[0] : 24,
    modelWidth: Number.isInteger(memsShape[3]) ? memsShape[3] : 1024,
  });
  return transcriber;
}

const MODES = Object.freeze({
  native: { itn: false, romanized: false },
  mixed: { itn: true, romanized: false },
  romanized: { itn: false, romanized: true },
  romanised: { itn: false, romanized: true },
});

export const INDIC_CORE_FP32_LANGUAGES = Object.freeze([
  'as', 'bgc', 'bhb', 'bho', 'bn', 'brx', 'doi', 'en', 'gu', 'hi', 'hne', 'kn', 'kok',
  'ks', 'mai', 'ml', 'mni', 'mr', 'ne', 'or', 'pa', 'sa', 'sat', 'sd', 'ta', 'te', 'ur',
]);

function tokenId(tokenToId, piece, fallback) {
  return tokenToId.get(piece) ?? fallback;
}

function makePrompt(tokenToId, language, mode) {
  if (!INDIC_CORE_FP32_LANGUAGES.includes(language)) throw new Error(`Unsupported FP32 model language: ${language}`);
  const langId = tokenToId.get(`<|${language}|>`);
  if (langId === undefined) throw new Error(`Language token <|${language}|> is absent from vocab.txt.`);
  const flags = MODES[mode];
  if (!flags) throw new Error(`Unsupported transcription mode '${mode}'. Use native, mixed, or romanized.`);
  return [
    tokenId(tokenToId, '<|startofcontext|>', 7),
    tokenId(tokenToId, '<|startoftranscript|>', 4),
    tokenId(tokenToId, '<|emo:undefined|>', 18),
    langId,
    langId,
    tokenId(tokenToId, '<|pnc|>', 5),
    tokenId(tokenToId, flags.itn ? '<|itn|>' : '<|noitn|>', flags.itn ? 8 : 9),
    tokenId(tokenToId, flags.romanized ? '<|romanized|>' : '<|noromanized|>', flags.romanized ? 10 : 11),
    tokenId(tokenToId, '<|notimestamp|>', 13),
    tokenId(tokenToId, '<|nodiarize|>', 15),
  ];
}

function emptyDecoderMems(ort, layerCount, modelWidth) {
  return new ort.Tensor('float32', new Float32Array(0), [layerCount, 1, 0, modelWidth]);
}

export class IndicCoreFp32Transcriber {
  constructor({ ort, encoder, decoder, assets, language, maxNewTokens, mode, numLayers, modelWidth }) {
    this.ort = ort;
    this.encoder = encoder;
    this.decoder = decoder;
    this.assets = assets;
    this.language = language;
    this.mode = mode;
    this.maxNewTokens = maxNewTokens;
    this.numLayers = numLayers;
    this.modelWidth = modelWidth;
    this.prompt = makePrompt(assets.vocabulary.tokenToId, language, mode);
    this.eosId = assets.vocabulary.tokenToId.get('<|endoftext|>') ?? 3;
    this.padId = assets.vocabulary.tokenToId.get('<pad>') ?? 2;
    this.modelMetadata = {
      id: INDIC_CORE_FP32_MODEL_ID,
      revision: INDIC_CORE_FP32_REVISION,
      source: 'bodhan-ai/indic-transcribe-core',
      format: 'FP32 ONNX encoder/decoder with stateful decoder memory',
      runtime: 'ONNX Runtime Node (native CPU)',
      accuracyNote: 'The model publisher reports reference parity; this Node.js pipeline has not yet been validated against the reference outputs or the user recording.',
    };
  }

  async decode(samples) {
    const features = indicCanaryFeatures(samples, samples.length, this.assets.featureExtractor);
    const featureTensor = new this.ort.Tensor('float32', features.data, features.dims);
    const featureLength = new this.ort.Tensor('int64', features.lengths, [1]);
    const encoded = await this.encoder.run({ audio_signal: featureTensor, length: featureLength });
    if (!encoded.encoder_embeddings || !encoded.encoder_mask) throw new Error('FP32 encoder returned unexpected outputs.');

    let decoderMems = emptyDecoderMems(this.ort, this.numLayers, this.modelWidth);
    let inputIds = new this.ort.Tensor('int64', BigInt64Array.from(this.prompt, BigInt), [1, this.prompt.length]);
    let decoded = await this.decoder.run({
      input_ids: inputIds,
      encoder_embeddings: encoded.encoder_embeddings,
      encoder_mask: encoded.encoder_mask,
      decoder_mems: decoderMems,
    });
    if (!decoded.logits || !decoded.decoder_hidden_states) throw new Error('FP32 decoder returned unexpected outputs.');

    const generatedIds = [];
    let loopDetected = false;
    let loopPeriod = null;
    let suppressedCount = 0;
    let truncated = true;
    let stopReason = 'max_new_tokens';
    for (let step = 0; step < this.maxNewTokens; step++) {
      // Reuse the tested loop guard from the current WASM decoder, operating on
      // the final time-step logits of the native FP32 graph.
      const choice = chooseTokenWithLoopGuard(decoded.logits, generatedIds);
      if (choice.loopDetected) {
        loopDetected = true;
        loopPeriod ??= choice.loopPeriod;
        suppressedCount += choice.suppressedCount;
      }
      const nextToken = choice.token === null ? null : choice.token;
      if (nextToken === null) {
        truncated = true;
        stopReason = 'repetition_loop_unrecoverable';
        break;
      }
      if (nextToken === this.eosId || nextToken === this.padId) {
        truncated = false;
        stopReason = 'eos';
        break;
      }
      generatedIds.push(nextToken);
      decoderMems = decoded.decoder_hidden_states;
      inputIds = new this.ort.Tensor('int64', BigInt64Array.of(BigInt(nextToken)), [1, 1]);
      decoded = await this.decoder.run({
        input_ids: inputIds,
        encoder_embeddings: encoded.encoder_embeddings,
        encoder_mask: encoded.encoder_mask,
        decoder_mems: decoderMems,
      });
      if (!decoded.logits || !decoded.decoder_hidden_states) throw new Error('FP32 decoder returned unexpected outputs.');
    }

    const pieces = [];
    for (const id of generatedIds) {
      if (id === this.eosId) break;
      const token = this.assets.vocabulary.tokens[id] ?? '';
      if (!token.startsWith('<|')) pieces.push(token);
    }
    return {
      text: pieces.join('').trim(),
      tokenCount: generatedIds.length,
      truncated,
      loopDetected,
      loopPeriod,
      suppressedCount,
      stopReason,
    };
  }
}
