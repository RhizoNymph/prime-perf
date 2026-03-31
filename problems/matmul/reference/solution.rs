use std::io::{self, Read, Write};

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut stdin = stdin.lock();
    let mut stdout = stdout.lock();

    let mut buf4 = [0u8; 4];
    stdin.read_exact(&mut buf4).unwrap();
    let n = i32::from_le_bytes(buf4) as usize;

    let read_matrix = |reader: &mut io::StdinLock| -> Vec<f32> {
        let mut bytes = vec![0u8; n * n * 4];
        reader.read_exact(&mut bytes).unwrap();
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    };

    let a = read_matrix(&mut stdin);
    let b = read_matrix(&mut stdin);
    let mut c = vec![0.0f32; n * n];

    // Naive i,j,k loop — cache-hostile B access
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0f32;
            for k in 0..n {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }

    let out_bytes: Vec<u8> = c.iter().flat_map(|v| v.to_le_bytes()).collect();
    stdout.write_all(&out_bytes).unwrap();
}
