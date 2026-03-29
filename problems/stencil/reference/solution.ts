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
  const headerBuf = readBytes(0, 12);
  const w = headerBuf.readInt32LE(0);
  const h = headerBuf.readInt32LE(4);
  const iters = headerBuf.readInt32LE(8);

  const n = w * h;
  const gridBuf = readBytes(0, n * 4);
  const old = new Float32Array(gridBuf.buffer, gridBuf.byteOffset, n);
  const cur = new Float32Array(n);
  cur.set(old);
  const next = new Float32Array(n);
  next.set(old);

  for (let iter = 0; iter < iters; iter++) {
    for (let i = 1; i < h - 1; i++) {
      for (let j = 1; j < w - 1; j++) {
        const idx = i * w + j;
        next[idx] = Math.fround(
          Math.fround(
            Math.fround(
              Math.fround(
                Math.fround(cur[(i - 1) * w + j] + cur[(i + 1) * w + j]) +
                cur[i * w + (j - 1)]
              ) + cur[i * w + (j + 1)]
            ) + cur[idx]
          ) / 5.0
        );
      }
    }
    // Swap: copy next into cur
    cur.set(next);
    // Boundary is already correct in cur since next had them from init/previous copy
    if (iter < iters - 1) {
      // Copy boundary from cur into next for the next iteration
      // Top row
      for (let j = 0; j < w; j++) next[j] = cur[j];
      // Bottom row
      for (let j = 0; j < w; j++) next[(h - 1) * w + j] = cur[(h - 1) * w + j];
      // Left and right columns
      for (let i = 1; i < h - 1; i++) {
        next[i * w] = cur[i * w];
        next[i * w + w - 1] = cur[i * w + w - 1];
      }
    }
  }

  const outBuf = Buffer.from(cur.buffer, cur.byteOffset, cur.byteLength);
  writeSync(1, outBuf);
}

main();
