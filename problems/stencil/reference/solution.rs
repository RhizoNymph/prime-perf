use std::io::{self, Read, Write};

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut stdin = stdin.lock();
    let mut stdout = stdout.lock();

    let mut buf4 = [0u8; 4];

    stdin.read_exact(&mut buf4).unwrap();
    let w = i32::from_le_bytes(buf4) as usize;

    stdin.read_exact(&mut buf4).unwrap();
    let h = i32::from_le_bytes(buf4) as usize;

    stdin.read_exact(&mut buf4).unwrap();
    let iters = i32::from_le_bytes(buf4) as usize;

    let n = w * h;
    let mut bytes = vec![0u8; n * 4];
    stdin.read_exact(&mut bytes).unwrap();

    let mut old: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    let mut new = old.clone();

    for iter in 0..iters {
        for i in 1..h - 1 {
            for j in 1..w - 1 {
                new[i * w + j] = (old[(i - 1) * w + j]
                    + old[(i + 1) * w + j]
                    + old[i * w + (j - 1)]
                    + old[i * w + (j + 1)]
                    + old[i * w + j])
                    / 5.0f32;
            }
        }
        std::mem::swap(&mut old, &mut new);
        // Copy boundary into new for next iteration
        if iter < iters - 1 {
            // Top row
            new[..w].copy_from_slice(&old[..w]);
            // Bottom row
            let start = (h - 1) * w;
            new[start..start + w].copy_from_slice(&old[start..start + w]);
            // Left and right columns
            for i in 1..h - 1 {
                new[i * w] = old[i * w];
                new[i * w + w - 1] = old[i * w + w - 1];
            }
        }
    }

    let out_bytes: Vec<u8> = old.iter().flat_map(|v| v.to_le_bytes()).collect();
    stdout.write_all(&out_bytes).unwrap();
}
