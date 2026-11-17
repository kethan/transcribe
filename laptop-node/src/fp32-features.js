import FFT from 'fft.js';

const RATE = 16_000;
const N_FFT = 512;
const HOP = 160;
const WIN = 400;
const MEL_BINS = 128;
const FFT_BINS = N_FFT / 2 + 1;
const PREEMPHASIS = Math.fround(0.97);
const LOG_GUARD = 2 ** -24;

function reflectIndex(index, length) {
  if (length <= 1) return 0;
  while (index < 0 || index >= length) {
    if (index < 0) index = -index;
    if (index >= length) index = 2 * length - 2 - index;
  }
  return index;
}

/**
 * Port of IndicCanaryFeatureExtractor.forward for one 16 kHz mono waveform.
 * The mel bank and Hann window are read from the model's safetensors file.
 */
export function indicCanaryFeatures(input, sampleLength, { fb, window }) {
  if (!(input instanceof Float32Array) || !input.length) throw new Error('Feature input must be a non-empty Float32Array.');
  if (!fb || fb.length !== MEL_BINS * FFT_BINS || !window || window.length !== WIN) {
    throw new Error('Indic Canary feature tensors have unexpected dimensions.');
  }
  if (!Number.isInteger(sampleLength) || sampleLength < 1 || sampleLength > input.length) {
    throw new Error('Feature sample length is invalid.');
  }

  // Match the model's reference inference: center waveforms shorter than 1 s
  // in exactly one second of zero padding and treat that padded span as valid.
  let audio = input;
  if (sampleLength < RATE) {
    audio = new Float32Array(RATE);
    const offset = Math.round((RATE - sampleLength) / 2);
    audio.set(input.subarray(0, sampleLength), offset);
    sampleLength = RATE;
  } else if (sampleLength !== input.length) {
    audio = input.subarray(0, sampleLength);
  }

  const emphasized = new Float32Array(sampleLength);
  emphasized[0] = audio[0];
  for (let i = 1; i < sampleLength; i++) {
    emphasized[i] = Math.fround(audio[i] - Math.fround(PREEMPHASIS * audio[i - 1]));
  }

  const frameCount = Math.floor(sampleLength / HOP) + 1;
  const logMel = new Float32Array(MEL_BINS * frameCount);
  const fft = new FFT(N_FFT);
  const spectrum = fft.createComplexArray();
  const frame = new Float64Array(N_FFT);
  const power = new Float32Array(FFT_BINS);

  for (let t = 0; t < frameCount; t++) {
    frame.fill(0);
    const center = t * HOP;
    for (let i = 0; i < WIN; i++) {
      const sourceIndex = reflectIndex(center + i - (WIN >> 1), sampleLength);
      frame[(N_FFT - WIN) / 2 + i] = emphasized[sourceIndex] * window[i];
    }
    fft.realTransform(spectrum, frame);
    fft.completeSpectrum(spectrum);
    for (let k = 0; k < FFT_BINS; k++) {
      const real = spectrum[2 * k];
      const imag = spectrum[2 * k + 1];
      const magnitude = Math.fround(Math.sqrt(Math.fround(real * real + imag * imag)));
      power[k] = Math.fround(magnitude * magnitude);
    }
    for (let mel = 0; mel < MEL_BINS; mel++) {
      const filterOffset = mel * FFT_BINS;
      let energy = 0;
      for (let k = 0; k < FFT_BINS; k++) {
        energy = Math.fround(energy + Math.fround(fb[filterOffset + k] * power[k]));
      }
      logMel[mel * frameCount + t] = Math.fround(Math.log(Math.fround(energy + LOG_GUARD)));
    }
  }

  // NeMo's normalize_batch computes an unbiased std over valid frames, adds
  // 1e-5 after sqrt, then masks padded frames to zero. Each chunk is processed
  // independently here, so every extracted frame is valid.
  const features = new Float32Array(logMel.length);
  for (let mel = 0; mel < MEL_BINS; mel++) {
    const offset = mel * frameCount;
    let sum = 0;
    for (let t = 0; t < frameCount; t++) sum = Math.fround(sum + logMel[offset + t]);
    const mean = Math.fround(sum / frameCount);
    let squared = 0;
    for (let t = 0; t < frameCount; t++) {
      const delta = Math.fround(logMel[offset + t] - mean);
      squared = Math.fround(squared + Math.fround(delta * delta));
    }
    const std = Math.fround(Math.sqrt(Math.fround(squared / (frameCount - 1))) + 1e-5);
    for (let t = 0; t < frameCount; t++) {
      features[offset + t] = Math.fround(Math.fround(logMel[offset + t] - mean) / std);
    }
  }

  return {
    data: features,
    dims: [1, MEL_BINS, frameCount],
    lengths: BigInt64Array.of(BigInt(frameCount)),
  };
}

export const INDIC_CANARY_FRONTEND = Object.freeze({ sampleRate: RATE, nFft: N_FFT, hopLength: HOP, winLength: WIN, melBins: MEL_BINS });
