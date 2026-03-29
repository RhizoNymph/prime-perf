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

function readInt32(fd: number): number {
  const buf = readBytes(fd, 4);
  return buf.readInt32LE(0);
}

function main(): void {
  const nInsert = readInt32(0);

  // Use a JavaScript Map for the naive reference
  const table = new Map<string, Buffer>();

  // Insert phase
  for (let i = 0; i < nInsert; i++) {
    const keyLen = readInt32(0);
    const keyBuf = readBytes(0, keyLen);
    const key = keyBuf.toString("utf-8");

    const valLen = readInt32(0);
    const valBuf = readBytes(0, valLen);

    table.set(key, valBuf); // last write wins for duplicates
  }

  // Lookup phase
  const nLookup = readInt32(0);
  const outParts: Buffer[] = [];

  for (let i = 0; i < nLookup; i++) {
    const keyLen = readInt32(0);
    const keyBuf = readBytes(0, keyLen);
    const key = keyBuf.toString("utf-8");

    const val = table.get(key);
    if (val !== undefined) {
      const header = Buffer.alloc(8);
      header.writeInt32LE(1, 0);        // found = 1
      header.writeInt32LE(val.length, 4); // val_len
      outParts.push(header);
      outParts.push(val);
    } else {
      const header = Buffer.alloc(4);
      header.writeInt32LE(0, 0);         // found = 0
      outParts.push(header);
    }
  }

  const outBuf = Buffer.concat(outParts);
  writeSync(1, outBuf);
}

main();
