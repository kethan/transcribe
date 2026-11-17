import FFT from 'fft.js';

const RATE = 16_000;
const N_FFT = 512;
const HOP = 160;
const WIN = 400;
const MEL_BINS = 128;
const PREEMPHASIS = 0.97;
const EPSILON = 2 ** -24;

function hzToSlaneyMel(hz) {
  const fSp = 200 / 3;
  const minLogHz = 1000;
  const minLogMel = minLogHz / fSp;
  const logStep = Math.log(6.4) / 27;
  return hz >= minLogHz ? minLogMel + Math.log(hz / minLogHz) / logStep : hz / fSp;
}

function slaneyMelToHz(mel) {
  const fSp = 200 / 3;
  const minLogHz = 1000;
  const minLogMel = minLogHz / fSp;
  const logStep = Math.log(6.4) / 27;
  return mel >= minLogMel ? minLogHz * Math.exp(logStep * (mel - minLogMel)) : mel * fSp;
}

function makeMelFilters() {
  const minMel = hzToSlaneyMel(0);
  const maxMel = hzToSlaneyMel(RATE / 2);
  const points = Array.from({ length: MEL_BINS + 2 }, (_, i) =>
    slaneyMelToHz(minMel + (maxMel - minMel) * i / (MEL_BINS + 1)));
  const filters = Array.from({ length: MEL_BINS }, () => new Float64Array(N_FFT / 2 + 1));
  for (let m = 0; m < MEL_BINS; m++) {
    const left = points[m], center = points[m + 1], right = points[m + 2];
    const areaScale = 2 / (right - left);
    for (let k = 0; k <= N_FFT / 2; k++) {
      const hz = k * RATE / N_FFT;
      const rising = (hz - left) / (center - left);
      const falling = (right - hz) / (right - center);
      filters[m][k] = Math.max(0, Math.min(rising, falling)) * areaScale;
    }
  }
  return filters;
}

const MEL_FILTERS = makeMelFilters();
const HANN = Float64Array.from({ length: WIN }, (_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (WIN - 1)));

function reflectIndex(index, length) {
  if (length <= 1) return 0;
  while (index < 0 || index >= length) {
    if (index < 0) index = -index;
    if (index >= length) index = 2 * length - 2 - index;
  }
  return index;
}

export function melFeatures(input) {
  let audio = input;
  if (audio.length < RATE) {
    const padded = new Float32Array(RATE);
    const left = Math.floor((RATE - audio.length) / 2);
    padded.set(audio, left);
    audio = padded;
  }
  const emphasized = new Float32Array(audio.length);
  if (audio.length) emphasized[0] = audio[0];
  for (let i = 1; i < audio.length; i++) emphasized[i] = audio[i] - PREEMPHASIS * audio[i - 1];

  const frameCount = 1 + Math.floor(audio.length / HOP);
  const values = Array.from({ length: MEL_BINS }, () => new Float64Array(frameCount));
  const fft = new FFT(N_FFT);
  const output = fft.createComplexArray();
  const frame = new Float64Array(N_FFT);
  const power = new Float64Array(N_FFT / 2 + 1);

  for (let t = 0; t < frameCount; t++) {
    frame.fill(0);
    const center = t * HOP;
    for (let i = 0; i < WIN; i++) {
      // Centered 400-sample Hann window in a 512-sample analysis frame.
      const audioIndex = reflectIndex(center + i - (WIN >> 1), emphasized.length);
      frame[(N_FFT - WIN) / 2 + i] = emphasized[audioIndex] * HANN[i];
    }
    fft.realTransform(output, frame);
    fft.completeSpectrum(output);
    for (let k = 0; k <= N_FFT / 2; k++) {
      const re = output[2 * k], im = output[2 * k + 1];
      power[k] = re * re + im * im;
    }
    for (let m = 0; m < MEL_BINS; m++) {
      const filter = MEL_FILTERS[m];
      let sum = 0;
      for (let k = 0; k < power.length; k++) sum += filter[k] * power[k];
      values[m][t] = Math.log(sum + EPSILON);
    }
  }

  // Match librosa's per-mel-bin normalization (sample standard deviation, ddof=1).
  const result = new Float32Array(MEL_BINS * frameCount);
  for (let m = 0; m < MEL_BINS; m++) {
    const row = values[m];
    let mean = 0;
    for (const x of row) mean += x;
    mean /= frameCount;
    let variance = 0;
    for (const x of row) variance += (x - mean) ** 2;
    const std = Math.sqrt(variance / Math.max(1, frameCount - 1));
    const divisor = std + 1e-5;
    for (let t = 0; t < frameCount; t++) result[m * frameCount + t] = (row[t] - mean) / divisor;
  }
  return { data: result, dims: [1, MEL_BINS, frameCount] };
}
