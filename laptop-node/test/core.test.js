import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { decodeAudio, planChunks } from '../src/audio.js';
import { melFeatures } from '../src/features.js';
import { chooseTokenWithLoopGuard, detectRepetitionLoop } from '../src/inference.js';
import { indicCanaryFeatures } from '../src/fp32-features.js';
import { parseFloat32Safetensors } from '../src/safetensors.js';
import { parseIndicCoreVocabulary } from '../src/fp32-model.js';
import { torchaudioSincResample } from '../src/torchaudio-resample.js';

function pcmWav(samples, sampleRate = 16000) {
  const dataBytes = samples.length * 2;
  const out = Buffer.alloc(44 + dataBytes);
  out.write('RIFF', 0); out.writeUInt32LE(36 + dataBytes, 4); out.write('WAVE', 8);
  out.write('fmt ', 12); out.writeUInt32LE(16, 16); out.writeUInt16LE(1, 20);
  out.writeUInt16LE(1, 22); out.writeUInt32LE(sampleRate, 24);
  out.writeUInt32LE(sampleRate * 2, 28); out.writeUInt16LE(2, 32);
  out.writeUInt16LE(16, 34); out.write('data', 36); out.writeUInt32LE(dataBytes, 40);
  for (let i = 0; i < samples.length; i++) out.writeInt16LE(Math.round(samples[i] * 32767), 44 + i * 2);
  return out;
}

test('chunk plan is contiguous and covers all samples', () => {
  const audio = new Float32Array(61 * 16000);
  for (let i = 0; i < audio.length; i++) audio[i] = Math.sin(i * 0.01) * 0.1;
  const { chunks } = planChunks(audio);
  assert.ok(chunks.length >= 3);
  assert.equal(chunks[0].startSample, 0);
  assert.equal(chunks.at(-1).endSample, audio.length);
  for (let i = 1; i < chunks.length; i++) assert.equal(chunks[i - 1].endSample, chunks[i].startSample);
  for (const chunk of chunks) assert.ok(chunk.endSample > chunk.startSample && chunk.end - chunk.start <= 30.01);
});

test('mel features have the ONNX shape and finite normalized values', () => {
  const audio = Float32Array.from({ length: 16000 }, (_, i) => Math.sin(2 * Math.PI * 440 * i / 16000) * 0.1);
  const features = melFeatures(audio);
  assert.deepEqual(features.dims, [1, 128, 101]);
  assert.equal(features.data.length, 128 * 101);
  assert.ok(features.data.every(Number.isFinite));
  for (let m = 0; m < 128; m++) {
    let mean = 0;
    for (let t = 0; t < 101; t++) mean += features.data[m * 101 + t];
    assert.ok(Math.abs(mean / 101) < 1e-4);
  }
});

test('repetition guard detects decoder loops without stripping a single repeated phrase', () => {
  assert.equal(detectRepetitionLoop([1, 2, 1, 2, 1, 2]), null);
  assert.equal(detectRepetitionLoop([1, 2, 1, 2, 1, 2, 1, 2])?.period, 2);
  assert.equal(detectRepetitionLoop([7, 7, 7, 7, 7, 7, 7, 7])?.cycles, 8);
  assert.equal(detectRepetitionLoop([1, 2, 1, 2]), null);
});

test('loop guard chooses a non-looping alternative instead of cutting the chunk', () => {
  const history = [5, 6, 5, 6, 5, 6, 5];
  const choice = chooseTokenWithLoopGuard({ dims: [1, 9], data: Float32Array.from([0, 1, 0, 0, 0, 0, 10, 0, 9]) }, history);
  assert.equal(choice.token, 8);
  assert.equal(choice.loopDetected, true);
  assert.equal(choice.loopPeriod, 2);
  assert.equal(choice.suppressedCount, 1);
});

test('loop guard leaves EOS and PAD alone when they are the greedy next token', () => {
  const logits = { dims: [1, 4], data: Float32Array.from([0, 0, 10, 9]) };
  assert.equal(chooseTokenWithLoopGuard(logits, [5, 6]).token, 2);
});

test('FP32 safetensors reader preserves tensor dimensions and float32 payloads', () => {
  const header = Buffer.from(JSON.stringify({
    fb: { dtype: 'F32', shape: [1, 128, 257], data_offsets: [0, 128 * 257 * 4] },
    window: { dtype: 'F32', shape: [400], data_offsets: [128 * 257 * 4, (128 * 257 + 400) * 4] },
  }));
  const payload = Buffer.alloc(8 + header.length + (128 * 257 + 400) * 4);
  payload.writeBigUInt64LE(BigInt(header.length), 0);
  header.copy(payload, 8);
  const dataStart = 8 + header.length;
  payload.writeFloatLE(0.25, dataStart);
  payload.writeFloatLE(0.75, dataStart + 128 * 257 * 4);
  const tensors = parseFloat32Safetensors(payload);
  assert.deepEqual(tensors.fb.dims, [1, 128, 257]);
  assert.deepEqual(tensors.window.dims, [400]);
  assert.equal(tensors.fb.data[0], 0.25);
  assert.equal(tensors.window.data[0], 0.75);
});

test('Indic Canary frontend pads short audio and returns finite normalized features', () => {
  const audio = Float32Array.from({ length: 4000 }, (_, i) => Math.sin(2 * Math.PI * 440 * i / 16000) * 0.1);
  const fb = new Float32Array(128 * 257);
  const window = new Float32Array(400).fill(1);
  for (let mel = 0; mel < 128; mel++) fb[mel * 257 + 40 + (mel % 3)] = 1;
  const result = indicCanaryFeatures(audio, audio.length, { fb, window });
  assert.deepEqual(result.dims, [1, 128, 101]);
  assert.deepEqual(Array.from(result.lengths), [101n]);
  assert.ok(result.data.every(Number.isFinite));
  assert.ok(result.data.some((value) => value !== 0));
});

test('FP32 resampler matches the model frontend target length and preserves an interior tone', () => {
  const sourceRate = 48000;
  const source = Float32Array.from({ length: sourceRate }, (_, i) => Math.sin(2 * Math.PI * 1000 * i / sourceRate) * 0.2);
  const result = torchaudioSincResample(source, sourceRate, 16000);
  assert.equal(result.length, 16000);
  assert.ok(result.every(Number.isFinite));
  const start = 1000, end = 15000;
  let squareSum = 0;
  for (let i = start; i < end; i++) squareSum += result[i] * result[i];
  const rms = Math.sqrt(squareSum / (end - start));
  assert.ok(Math.abs(rms - 0.2 / Math.sqrt(2)) < 0.01, `unexpected RMS ${rms}`);
});

test('FP32 vocabulary parser validates contiguous IDs and special-token lookup', () => {
  const rows = Array.from({ length: 7152 }, (_, id) => `${id === 3 ? '<|endoftext|>' : `piece${id}`} ${id}`);
  const vocab = parseIndicCoreVocabulary(rows.join('\n'));
  assert.equal(vocab.tokens.length, 7152);
  assert.equal(vocab.tokenToId.get('<|endoftext|>'), 3);
  assert.throws(() => parseIndicCoreVocabulary('duplicate 0\nduplicate 0'), /Duplicate or invalid token id/);
  assert.throws(() => parseIndicCoreVocabulary(rows.filter((_, id) => id !== 5).join('\n')), /contiguous 7,152-token/);
});

test('WAV decoder and resampler produce 16 kHz mono float samples', async () => {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'bodhan-node-'));
  const wav = path.join(dir, 'tone.wav');
  const source = Float32Array.from({ length: 8000 }, (_, i) => Math.sin(2 * Math.PI * 220 * i / 8000) * 0.2);
  await writeFile(wav, pcmWav(source, 8000));
  try {
    const decoded = await decodeAudio(wav);
    assert.equal(decoded.length, 16000);
    assert.ok(decoded.some((x) => Math.abs(x) > 0.05));
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
