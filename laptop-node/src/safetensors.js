import { readFile } from 'node:fs/promises';

const SUPPORTED_FLOAT32 = 'F32';

/** Read a contiguous tensor from the small safetensors feature-extractor file. */
export function parseFloat32Safetensors(bytes, required = ['fb', 'window']) {
  const buffer = Buffer.isBuffer(bytes) ? bytes : Buffer.from(bytes);
  if (buffer.length < 10) throw new Error('Safetensors file is truncated.');
  const headerLength = Number(buffer.readBigUInt64LE(0));
  const dataStart = 8 + headerLength;
  if (!Number.isSafeInteger(headerLength) || headerLength < 2 || dataStart > buffer.length) {
    throw new Error('Safetensors header has an invalid length.');
  }

  let header;
  try {
    header = JSON.parse(buffer.toString('utf8', 8, dataStart));
  } catch (error) {
    throw new Error(`Could not parse safetensors header: ${error.message}`);
  }

  const tensors = {};
  for (const name of required) {
    const entry = header[name];
    if (!entry || entry.dtype !== SUPPORTED_FLOAT32 || !Array.isArray(entry.shape) || !Array.isArray(entry.data_offsets)) {
      throw new Error(`Safetensors tensor '${name}' must be contiguous float32 data.`);
    }
    const [start, end] = entry.data_offsets;
    const count = entry.shape.reduce((product, size) => product * size, 1);
    if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || end < start || end - start !== count * 4 || dataStart + end > buffer.length) {
      throw new Error(`Safetensors tensor '${name}' has invalid offsets or shape.`);
    }
    const values = new Float32Array(count);
    for (let i = 0; i < count; i++) values[i] = buffer.readFloatLE(dataStart + start + i * 4);
    if (values.some((value) => !Number.isFinite(value))) throw new Error(`Safetensors tensor '${name}' contains a non-finite value.`);
    tensors[name] = { data: values, dims: [...entry.shape] };
  }
  return tensors;
}

export async function loadFeatureExtractor(modelDir) {
  const { access } = await import('node:fs/promises');
  const { join } = await import('node:path');
  let filePath;
  for (const name of ['feature_extractor.safetensors', 'feature_extractor.safetensors.txt']) {
    const candidate = join(modelDir, name);
    try { await access(candidate); filePath = candidate; break; } catch { /* try the next supported name */ }
  }
  if (!filePath) throw new Error(`Missing feature_extractor.safetensors (or uploaded feature_extractor.safetensors.txt) in ${modelDir}.`);
  const tensors = parseFloat32Safetensors(await readFile(filePath));
  const fb = tensors.fb;
  const window = tensors.window;
  if (fb.dims.join(',') !== '1,128,257' || window.dims.join(',') !== '400') {
    throw new Error(`Unexpected feature extractor tensor shapes: fb=[${fb.dims}], window=[${window.dims}].`);
  }
  return { fb: fb.data, window: window.data, filePath };
}
