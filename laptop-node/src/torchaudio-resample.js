const LOWPASS_FILTER_WIDTH = 6;
const ROLLOFF = 0.99;
const kernelCache = new Map();

function gcd(a, b) {
  while (b) [a, b] = [b, a % b];
  return a;
}

function phaseKernel(origRate, targetRate) {
  if (!Number.isInteger(origRate) || !Number.isInteger(targetRate) || origRate < 1 || targetRate < 1) {
    throw new Error('Sample rates must be positive integers.');
  }
  const divisor = gcd(origRate, targetRate);
  const source = origRate / divisor;
  const target = targetRate / divisor;
  const key = `${source}:${target}`;
  if (kernelCache.has(key)) return kernelCache.get(key);

  const baseFrequency = Math.min(source, target) * ROLLOFF;
  const width = Math.ceil(LOWPASS_FILTER_WIDTH * source / baseFrequency);
  const radius = LOWPASS_FILTER_WIDTH * source / baseFrequency;
  const phaseWeights = Array.from({ length: target }, (_, phase) => {
    const fraction = phase / target;
    const offsets = [];
    const weights = [];
    for (let offset = -width - 1; offset <= width + 1; offset++) {
      const time = ((offset - fraction) / source) * baseFrequency;
      if (Math.abs(time) >= LOWPASS_FILTER_WIDTH) continue;
      const angle = Math.PI * time;
      const sinc = angle === 0 ? 1 : Math.sin(angle) / angle;
      const window = Math.cos(time * Math.PI / LOWPASS_FILTER_WIDTH / 2) ** 2;
      offsets.push(offset);
      // torchaudio precomputes this kernel in float64 and stores it as float32.
      weights.push(Math.fround(sinc * window * (baseFrequency / source)));
    }
    return { offsets: Int32Array.from(offsets), weights: Float32Array.from(weights) };
  });
  const result = { source, target, radius, phaseWeights };
  if (kernelCache.size >= 8) kernelCache.delete(kernelCache.keys().next().value);
  kernelCache.set(key, result);
  return result;
}

/**
 * CPU port of torchaudio's default sinc_interp_hann resampler (filter width 6,
 * rolloff .99), including zero padding and its ceil-length output. The Bodhan
 * frontend trims/pads that output to round-half-up duration * 16 kHz.
 */
export function torchaudioSincResample(input, origRate, targetRate = 16_000) {
  if (!(input instanceof Float32Array)) throw new Error('Resampler input must be Float32Array.');
  if (origRate === targetRate) return new Float32Array(input);
  const kernel = phaseKernel(origRate, targetRate);
  const outputLength = Math.ceil(input.length * kernel.target / kernel.source);
  const output = new Float32Array(outputLength);

  for (let outIndex = 0; outIndex < outputLength; outIndex++) {
    const numerator = outIndex * kernel.source;
    const baseIndex = Math.floor(numerator / kernel.target);
    const phase = numerator % kernel.target;
    const phaseKernel = kernel.phaseWeights[phase];
    let value = 0;
    for (let tap = 0; tap < phaseKernel.offsets.length; tap++) {
      const sourceIndex = baseIndex + phaseKernel.offsets[tap];
      if (sourceIndex < 0 || sourceIndex >= input.length) continue;
      value = Math.fround(value + Math.fround(input[sourceIndex] * phaseKernel.weights[tap]));
    }
    output[outIndex] = value;
  }

  // The model wrapper follows Lhotse's ROUND_HALF_UP target length, while
  // torchaudio itself returns ceil. Normalize length here to match that wrapper.
  const expectedLength = Math.floor(input.length / origRate * targetRate + 0.5);
  if (output.length === expectedLength) return output;
  if (output.length > expectedLength) return output.subarray(0, expectedLength);
  const padded = new Float32Array(expectedLength);
  padded.set(output);
  return padded;
}
