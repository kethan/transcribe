import { readFile } from 'node:fs/promises';
import { polyphase } from '@audio/resample';
import { torchaudioSincResample } from './torchaudio-resample.js';
import { FLACDecoder } from '@wasm-audio-decoders/flac';
import { MPEGDecoder } from 'mpg123-decoder';
import WavDecoder from 'wav-decoder';

const RATE = 16_000;

async function decodeFileBytes(filePath, bytes) {
  const ext = filePath.toLowerCase().split('.').at(-1);
  if (ext === 'wav' || ext === 'wave') {
    return WavDecoder.decode(bytes);
  }
  const data = new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (ext === 'mp3') {
    const decoder = new MPEGDecoder();
    try {
      await decoder.ready;
      const result = decoder.decode(data);
      if (result.errors?.length) {
        throw new Error(`MP3 decoder reported ${result.errors.length} damaged frame(s).`);
      }
      return result;
    } finally {
      decoder.free();
    }
  }
  if (ext === 'flac') {
    const decoder = new FLACDecoder();
    try {
      await decoder.ready;
      const result = await decoder.decodeFile(data);
      if (result.errors?.length) {
        throw new Error(`FLAC decoder reported ${result.errors.length} damaged frame(s).`);
      }
      return result;
    } finally {
      decoder.free();
    }
  }
  throw new Error(`Unsupported audio extension .${ext}; use WAV, MP3, or FLAC.`);
}

export async function decodeAudio(filePath, { resampler = 'polyphase' } = {}) {
  const bytes = await readFile(filePath);
  if (!bytes.length) throw new Error('Audio file is empty.');
  const decoded = await decodeFileBytes(filePath, bytes);
  if (!Number.isFinite(decoded.sampleRate) || decoded.sampleRate < 1 || !decoded.channelData?.length) {
    throw new Error('Audio decoder returned invalid sample rate or channels.');
  }
  const length = Math.min(...decoded.channelData.map((channel) => channel.length));
  if (!length) throw new Error('Audio decoder returned no samples.');
  const mono = new Float32Array(length);
  for (const channel of decoded.channelData) {
    for (let i = 0; i < length; i++) mono[i] += channel[i] / decoded.channelData.length;
  }
  if (mono.some((sample) => !Number.isFinite(sample))) throw new Error('Audio contains invalid samples.');
  if (decoded.sampleRate === RATE) return mono;
  const resampled = resampler === 'indic-canary'
    ? torchaudioSincResample(mono, decoded.sampleRate, RATE)
    : polyphase(mono, { from: decoded.sampleRate, to: RATE });
  if (!resampled.length || resampled.some((sample) => !Number.isFinite(sample))) {
    throw new Error('Audio resampling failed.');
  }
  return resampled;
}

export function planChunks(audio, maxSeconds = 28, searchSeconds = 3) {
  if (!(maxSeconds > 0 && maxSeconds <= 30)) throw new Error('Chunk length must be in (0, 30] seconds.');
  if (!(searchSeconds >= 0 && searchSeconds < maxSeconds)) throw new Error('Search window must be in [0, chunk length).');
  if (!audio.length) throw new Error('Audio has no samples.');
  const frame = 320;
  const count = Math.ceil(audio.length / frame);
  const rms = new Float32Array(count);
  for (let f = 0; f < count; f++) {
    const start = f * frame;
    const end = Math.min(start + frame, audio.length);
    let sum = 0;
    for (let i = start; i < end; i++) sum += audio[i] * audio[i];
    rms[f] = Math.sqrt(sum / frame + 1e-12);
  }
  const sorted = Array.from(rms).sort((a, b) => a - b);
  const quantileIndex = (sorted.length - 1) * 0.25;
  const quantileLow = Math.floor(quantileIndex);
  const quantile = sorted[quantileLow] + (sorted[Math.min(quantileLow + 1, sorted.length - 1)] - sorted[quantileLow]) * (quantileIndex - quantileLow);
  const threshold = Math.max(1e-5, quantile * 2);
  const maxFrames = Math.max(1, Math.floor(maxSeconds * RATE / frame));
  const minFrames = Math.ceil(RATE / frame);
  const searchFrames = Math.floor(searchSeconds * RATE / frame);
  const chunks = [];
  let start = 0;
  while (start < audio.length) {
    let end;
    let cutAtPause = null;
    if (audio.length - start <= maxFrames * frame) {
      end = audio.length;
    } else {
      const startFrame = Math.floor(start / frame);
      const stopFrame = Math.min(startFrame + maxFrames, rms.length);
      let low = Math.max(startFrame + minFrames, stopFrame - searchFrames);
      let high = Math.min(stopFrame, rms.length - 1);
      if (high < low) low = high;
      let best = low;
      for (let f = low; f <= high; f++) {
        if (rms[Math.max(0, f - 1)] < rms[Math.max(0, best - 1)]) best = f;
      }
      end = Math.min(best * frame, audio.length);
      cutAtPause = rms[Math.max(0, best - 1)] <= threshold;
    }
    if (end <= start) { end = Math.min(start + maxFrames * frame, audio.length); cutAtPause = null; }
    chunks.push({ startSample: start, endSample: end, start: start / RATE, end: end / RATE, cutAtPause });
    start = end;
  }
  return { chunks, threshold };
}
