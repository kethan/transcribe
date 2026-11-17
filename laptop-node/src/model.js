import { createHash } from 'node:crypto';
import { createWriteStream } from 'node:fs';
import { mkdir, readFile, stat, unlink } from 'node:fs/promises';
import path from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';

export const MODEL_ID = 'schamp007/verbatim-hindi-asr-core-sc';
export const REVISION = 'e17f007b2906828614334050e4693e1f4029871c';
export const MODEL_FILES = {
  'encoder.onnx': 'd25b1a9fcba2888c44428a0df46a2574b17692e937124a8f6328acf330e29de1',
  'cross_kv.onnx': '72bf71daee584dd5b66812c3df38c4de46e481f5acedb29718848b59eb95cce3',
  'decoder_kv.onnx': '98b90cea1096fb0adeabeab4c71a78b0b74d8508422b1e7d4172aa2bbd147325',
  'bodhan_vocab.json': '571fa2fac0efa0cb7c4196cb17e899560cf0c91510f7f2d739243d3766d5fb44',
};

export async function sha256File(file) {
  const hash = createHash('sha256');
  const { createReadStream } = await import('node:fs');
  for await (const block of createReadStream(file)) hash.update(block);
  return hash.digest('hex');
}

export async function verifyModel(modelDir) {
  const vocabPath = path.join(modelDir, 'bodhan_vocab.json');
  const vocab = JSON.parse(await readFile(vocabPath, 'utf8'));
  if (!Array.isArray(vocab) || vocab.length !== 7152) throw new Error('Model vocab must contain exactly 7,152 entries.');
  for (const [name, expected] of Object.entries(MODEL_FILES)) {
    const file = path.join(modelDir, name);
    let info;
    try { info = await stat(file); } catch { throw new Error(`Missing model file: ${file}`); }
    if (!info.isFile() || info.size === 0) throw new Error(`Model file is empty or invalid: ${file}`);
    const actual = await sha256File(file);
    if (actual !== expected) throw new Error(`SHA-256 mismatch for ${name}; expected ${expected}, got ${actual}.`);
  }
  return vocab;
}

export async function downloadModel(modelDir) {
  await mkdir(modelDir, { recursive: true });
  for (const [name, expectedHash] of Object.entries(MODEL_FILES)) {
    const destination = path.join(modelDir, name);
    try {
      const currentHash = await sha256File(destination);
      if (currentHash === expectedHash) {
        console.log(`Verified ${name}`);
        continue;
      }
      await unlink(destination);
    } catch { /* Download missing/unreadable file. */ }
    const url = `https://huggingface.co/${MODEL_ID}/resolve/${REVISION}/${name}?download=true`;
    console.log(`Downloading ${name} from the pinned model revision...`);
    const response = await fetch(url, { redirect: 'follow' });
    if (!response.ok || !response.body) throw new Error(`Download failed for ${name}: HTTP ${response.status}`);
    const temp = `${destination}.partial`;
    try {
      await pipeline(Readable.fromWeb(response.body), createWriteStream(temp));
      const actual = await sha256File(temp);
      if (actual !== expectedHash) throw new Error(`SHA-256 mismatch for downloaded ${name}; expected ${expectedHash}, got ${actual}.`);
      const { rename } = await import('node:fs/promises');
      await rename(temp, destination);
      console.log(`Verified ${name}`);
    } catch (error) {
      await unlink(temp).catch(() => {});
      throw error;
    }
  }
  await verifyModel(modelDir);
}
