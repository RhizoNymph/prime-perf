use std::io::{self, Read, Write};

/// FNV-1a hash
fn fnv1a(data: &[u8]) -> u32 {
    let mut h: u32 = 2166136261;
    for &b in data {
        h ^= b as u32;
        h = h.wrapping_mul(16777619);
    }
    h
}

/// Next power of 2 >= n
fn next_pow2(n: usize) -> usize {
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

/// Linked-list node for chaining
struct Node {
    key: Vec<u8>,
    val: Vec<u8>,
    next: Option<Box<Node>>,
}

fn read_i32(reader: &mut impl Read) -> i32 {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf).unwrap();
    i32::from_le_bytes(buf)
}

fn read_bytes(reader: &mut impl Read, len: usize) -> Vec<u8> {
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf).unwrap();
    buf
}

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut reader = stdin.lock();
    let mut writer = stdout.lock();

    let n_insert = read_i32(&mut reader) as usize;

    let capacity = next_pow2(if n_insert < 16 { 16 } else { n_insert });
    let mask = capacity - 1;

    let mut buckets: Vec<Option<Box<Node>>> = (0..capacity).map(|_| None).collect();

    // Insert phase
    for _ in 0..n_insert {
        let key_len = read_i32(&mut reader) as usize;
        let key = read_bytes(&mut reader, key_len);
        let val_len = read_i32(&mut reader) as usize;
        let val = read_bytes(&mut reader, val_len);

        let h = fnv1a(&key) as usize & mask;

        // Search for existing key to update
        let mut found = false;
        {
            let mut cur = &mut buckets[h];
            loop {
                match cur {
                    Some(node) if node.key == key => {
                        node.val = val.clone();
                        found = true;
                        break;
                    }
                    Some(node) => {
                        cur = &mut node.next;
                    }
                    None => break,
                }
            }
        }

        if !found {
            let node = Box::new(Node {
                key,
                val,
                next: buckets[h].take(),
            });
            buckets[h] = Some(node);
        }
    }

    // Lookup phase
    let n_lookup = read_i32(&mut reader) as usize;

    for _ in 0..n_lookup {
        let key_len = read_i32(&mut reader) as usize;
        let key = read_bytes(&mut reader, key_len);

        let h = fnv1a(&key) as usize & mask;

        let mut found = false;
        let mut cur = &buckets[h];
        while let Some(node) = cur {
            if node.key == key {
                let one: i32 = 1;
                writer.write_all(&one.to_le_bytes()).unwrap();
                let vlen = node.val.len() as i32;
                writer.write_all(&vlen.to_le_bytes()).unwrap();
                writer.write_all(&node.val).unwrap();
                found = true;
                break;
            }
            cur = &node.next;
        }

        if !found {
            let zero: i32 = 0;
            writer.write_all(&zero.to_le_bytes()).unwrap();
        }
    }
}
