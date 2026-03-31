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

function main(): void {
  const headerBuf = readBytes(0, 4);
  const n = headerBuf.readInt32LE(0);

  const matrixBytes = n * n * 4;
  const aBuf = readBytes(0, matrixBytes);
  const bBuf = readBytes(0, matrixBytes);

  const a = new Float32Array(aBuf.buffer, aBuf.byteOffset, n * n);
  const b = new Float32Array(bBuf.buffer, bBuf.byteOffset, n * n);
  const c = new Float32Array(n * n);

  // Naive i,j,k loop with Math.fround to match C's float32 accumulation
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let k = 0; k < n; k++) {
        sum = Math.fround(sum + Math.fround(a[i * n + k] * b[k * n + j]));
      }
      c[i * n + j] = sum;
    }
  }

  const outBuf = Buffer.from(c.buffer, c.byteOffset, c.byteLength);
  writeSync(1, outBuf);
}

main();
