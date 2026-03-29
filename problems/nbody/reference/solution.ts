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
  const headerBuf = readBytes(0, 12); // N (int32) + steps (int32) + dt (float32)
  const n = headerBuf.readInt32LE(0);
  const steps = headerBuf.readInt32LE(4);
  const dt = headerBuf.readFloatLE(8);

  // Read body data: 7 floats per body
  const bodyBytes = readBytes(0, n * 7 * 4);
  const bodyData = new Float32Array(
    bodyBytes.buffer,
    bodyBytes.byteOffset,
    n * 7,
  );

  const x = new Float32Array(n);
  const y = new Float32Array(n);
  const z = new Float32Array(n);
  const vx = new Float32Array(n);
  const vy = new Float32Array(n);
  const vz = new Float32Array(n);
  const mass = new Float32Array(n);

  for (let i = 0; i < n; i++) {
    x[i] = bodyData[i * 7 + 0];
    y[i] = bodyData[i * 7 + 1];
    z[i] = bodyData[i * 7 + 2];
    vx[i] = bodyData[i * 7 + 3];
    vy[i] = bodyData[i * 7 + 4];
    vz[i] = bodyData[i * 7 + 5];
    mass[i] = bodyData[i * 7 + 6];
  }

  const epsSq = 1e-6;

  for (let step = 0; step < steps; step++) {
    const ax = new Float32Array(n);
    const ay = new Float32Array(n);
    const az = new Float32Array(n);

    // Compute pairwise gravitational accelerations
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const dx = x[j] - x[i];
        const dy = y[j] - y[i];
        const dz = z[j] - z[i];
        const distSq = dx * dx + dy * dy + dz * dz + epsSq;
        const invDist = 1.0 / Math.sqrt(distSq);
        const invDist3 = invDist * invDist * invDist;
        ax[i] += mass[j] * dx * invDist3;
        ay[i] += mass[j] * dy * invDist3;
        az[i] += mass[j] * dz * invDist3;
      }
    }

    // Update velocities
    for (let i = 0; i < n; i++) {
      vx[i] += ax[i] * dt;
      vy[i] += ay[i] * dt;
      vz[i] += az[i] * dt;
    }

    // Update positions
    for (let i = 0; i < n; i++) {
      x[i] += vx[i] * dt;
      y[i] += vy[i] * dt;
      z[i] += vz[i] * dt;
    }
  }

  // Write output: 7 floats per body
  const outBuf = Buffer.alloc(n * 7 * 4);
  for (let i = 0; i < n; i++) {
    outBuf.writeFloatLE(x[i], (i * 7 + 0) * 4);
    outBuf.writeFloatLE(y[i], (i * 7 + 1) * 4);
    outBuf.writeFloatLE(z[i], (i * 7 + 2) * 4);
    outBuf.writeFloatLE(vx[i], (i * 7 + 3) * 4);
    outBuf.writeFloatLE(vy[i], (i * 7 + 4) * 4);
    outBuf.writeFloatLE(vz[i], (i * 7 + 5) * 4);
    outBuf.writeFloatLE(mass[i], (i * 7 + 6) * 4);
  }
  writeSync(1, outBuf);
}

main();
