import { readSync, writeSync } from "fs";

function readBytes(fd: number, count: number): Buffer {
  const buf = Buffer.alloc(count);
  let offset = 0;
  while (offset < count) {
    const bytesRead = readSync(fd, buf, offset, count - offset, null);
    if (bytesRead === 0) break;
    offset += bytesRead;
  }
  return buf;
}

function compare(a: number, b: number): number {
  const aNan = Number.isNaN(a);
  const bNan = Number.isNaN(b);
  if (aNan && bNan) return 0;
  if (aNan) return 1;
  if (bNan) return -1;
  if (a < b) return -1;
  if (a > b) return 1;
  // Equal values: distinguish -0.0 from +0.0
  // -0.0 should come before +0.0
  const sa = Object.is(a, -0) || a < 0 ? 1 : 0;
  const sb = Object.is(b, -0) || b < 0 ? 1 : 0;
  return sb - sa;
}

function main(): void {
  const headerBuf = readBytes(0, 4);
  const n = headerBuf.readInt32LE(0);

  const dataBuf = readBytes(0, n * 4);
  const float32Arr = new Float32Array(dataBuf.buffer, dataBuf.byteOffset, n);

  // Copy to regular array for sorting with custom comparator
  const arr = Array.from(float32Arr);
  arr.sort(compare);

  // Write back through Float32Array for proper encoding
  const result = new Float32Array(arr);
  const outBuf = Buffer.from(result.buffer, result.byteOffset, result.byteLength);
  writeSync(1, outBuf);
}

main();
