import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { decodeAudio, planChunks } from '../src/audio.js';
import { melFeatures } from '../src/features.js';
import { chooseTokenWithLoopGuard, detectRepetitionLoop } from '../src/inference.js';

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
