import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import * as ort from 'onnxruntime-node';

function varint(value) {
  const bytes = [];
  let number = BigInt(value);
  while (number > 127n) { bytes.push(Number((number & 127n) | 128n)); number >>= 7n; }
  bytes.push(Number(number));
  return Buffer.from(bytes);
}
function fieldVarint(field, value) { return Buffer.concat([varint(field * 8), varint(value)]); }
function fieldString(field, value) {
  const bytes = Buffer.from(value);
  return Buffer.concat([varint(field * 8 + 2), varint(bytes.length), bytes]);
}
function fieldMessage(field, message) {
  return Buffer.concat([varint(field * 8 + 2), varint(message.length), message]);
}
function valueInfo(name) {
  const dimension = fieldVarint(1, 2);
  const shape = fieldMessage(1, dimension);
  const tensorType = Buffer.concat([fieldVarint(1, 1), fieldMessage(2, shape)]); // float[2]
  const type = fieldMessage(1, tensorType);
  return Buffer.concat([fieldString(1, name), fieldMessage(2, type)]);
}
function tinyAddModel() {
  const node = Buffer.concat([
    fieldString(1, 'X'), fieldString(1, 'X'), fieldString(2, 'Y'), fieldString(4, 'Add'),
  ]);
  const graph = Buffer.concat([
    fieldMessage(1, node), fieldString(2, 'native-runtime-smoke'),
    fieldMessage(11, valueInfo('X')), fieldMessage(12, valueInfo('Y')),
  ]);
  const opset = fieldVarint(2, 13);
  return Buffer.concat([
    fieldVarint(1, 9), fieldString(2, 'arena-test'), fieldMessage(7, graph), fieldMessage(8, opset),
  ]);
}

test('onnxruntime-node executes a tiny ONNX graph on the native CPU provider', async () => {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'bodhan-ort-native-'));
  let session;
  try {
    const modelPath = path.join(directory, 'add.onnx');
    await writeFile(modelPath, tinyAddModel());
    session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['cpu'],
      intraOpNumThreads: 1,
    });
    const outputs = await session.run({ X: new ort.Tensor('float32', Float32Array.of(1.25, -3), [2]) });
    assert.deepEqual(Array.from(outputs.Y.data), [2.5, -6]);
  } finally {
    if (session) await session.release();
    await rm(directory, { recursive: true, force: true });
  }
});
