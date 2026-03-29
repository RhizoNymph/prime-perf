use std::io::{self, Read, Write};

fn compare(a: &f32, b: &f32) -> std::cmp::Ordering {
    let a_nan = a.is_nan();
    let b_nan = b.is_nan();
    if a_nan && b_nan {
        return std::cmp::Ordering::Equal;
    }
    if a_nan {
        return std::cmp::Ordering::Greater;
    }
    if b_nan {
        return std::cmp::Ordering::Less;
    }
    if a < b {
        return std::cmp::Ordering::Less;
    }
    if a > b {
        return std::cmp::Ordering::Greater;
    }
    // Equal values: distinguish -0.0 from +0.0
    // -0.0 (sign_negative=true) should come before +0.0 (sign_negative=false)
    let sa = a.is_sign_negative() as i32;
    let sb = b.is_sign_negative() as i32;
    sb.cmp(&sa)
}

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut stdin = stdin.lock();
    let mut stdout = stdout.lock();

    let mut buf4 = [0u8; 4];
    stdin.read_exact(&mut buf4).unwrap();
    let n = i32::from_le_bytes(buf4) as usize;

    let mut bytes = vec![0u8; n * 4];
    stdin.read_exact(&mut bytes).unwrap();
    let mut arr: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    arr.sort_by(compare);

    let out_bytes: Vec<u8> = arr.iter().flat_map(|v| v.to_le_bytes()).collect();
    stdout.write_all(&out_bytes).unwrap();
}
