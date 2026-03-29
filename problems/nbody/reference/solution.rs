use std::io::{self, Read, Write};

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut stdin = stdin.lock();
    let mut stdout = stdout.lock();

    let mut buf4 = [0u8; 4];

    stdin.read_exact(&mut buf4).unwrap();
    let n = i32::from_le_bytes(buf4) as usize;

    stdin.read_exact(&mut buf4).unwrap();
    let steps = i32::from_le_bytes(buf4) as usize;

    stdin.read_exact(&mut buf4).unwrap();
    let dt = f32::from_le_bytes(buf4);

    // Read body data: 7 floats per body
    let mut body_bytes = vec![0u8; n * 7 * 4];
    stdin.read_exact(&mut body_bytes).unwrap();

    let mut x = vec![0.0f32; n];
    let mut y = vec![0.0f32; n];
    let mut z = vec![0.0f32; n];
    let mut vx = vec![0.0f32; n];
    let mut vy = vec![0.0f32; n];
    let mut vz = vec![0.0f32; n];
    let mut mass = vec![0.0f32; n];

    for i in 0..n {
        let base = i * 7 * 4;
        x[i] = f32::from_le_bytes(body_bytes[base..base + 4].try_into().unwrap());
        y[i] = f32::from_le_bytes(body_bytes[base + 4..base + 8].try_into().unwrap());
        z[i] = f32::from_le_bytes(body_bytes[base + 8..base + 12].try_into().unwrap());
        vx[i] = f32::from_le_bytes(body_bytes[base + 12..base + 16].try_into().unwrap());
        vy[i] = f32::from_le_bytes(body_bytes[base + 16..base + 20].try_into().unwrap());
        vz[i] = f32::from_le_bytes(body_bytes[base + 20..base + 24].try_into().unwrap());
        mass[i] = f32::from_le_bytes(body_bytes[base + 24..base + 28].try_into().unwrap());
    }

    let eps_sq: f32 = 1e-6;

    for _step in 0..steps {
        let mut ax = vec![0.0f32; n];
        let mut ay = vec![0.0f32; n];
        let mut az = vec![0.0f32; n];

        // Compute pairwise gravitational accelerations
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let dx = x[j] - x[i];
                let dy = y[j] - y[i];
                let dz = z[j] - z[i];
                let dist_sq = dx * dx + dy * dy + dz * dz + eps_sq;
                let inv_dist = 1.0f32 / dist_sq.sqrt();
                let inv_dist3 = inv_dist * inv_dist * inv_dist;
                ax[i] += mass[j] * dx * inv_dist3;
                ay[i] += mass[j] * dy * inv_dist3;
                az[i] += mass[j] * dz * inv_dist3;
            }
        }

        // Update velocities
        for i in 0..n {
            vx[i] += ax[i] * dt;
            vy[i] += ay[i] * dt;
            vz[i] += az[i] * dt;
        }

        // Update positions
        for i in 0..n {
            x[i] += vx[i] * dt;
            y[i] += vy[i] * dt;
            z[i] += vz[i] * dt;
        }
    }

    // Write output: 7 floats per body
    let mut out_bytes = Vec::with_capacity(n * 7 * 4);
    for i in 0..n {
        out_bytes.extend_from_slice(&x[i].to_le_bytes());
        out_bytes.extend_from_slice(&y[i].to_le_bytes());
        out_bytes.extend_from_slice(&z[i].to_le_bytes());
        out_bytes.extend_from_slice(&vx[i].to_le_bytes());
        out_bytes.extend_from_slice(&vy[i].to_le_bytes());
        out_bytes.extend_from_slice(&vz[i].to_le_bytes());
        out_bytes.extend_from_slice(&mass[i].to_le_bytes());
    }
    stdout.write_all(&out_bytes).unwrap();
}
