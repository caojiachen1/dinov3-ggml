//! Experiment: do two model instances on one GPU overlap? (Strata-style
//! parallel streams — each GgmlVitModel owns its own backend/stream.)
//!
//! Usage: cargo run --release --features cuda --example multi_inst -- [n]
use std::time::Instant;

use dinov3_ggml::{GgmlVitModel, VitConfig};

fn main() {
    let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(256);
    let mut cfg = VitConfig::vit_small_16();
    cfg.input_height = 256;
    cfg.input_width = 256;
    cfg.max_batch = 32;
    let img_f = 3 * 256 * 256;

    // Preprocessed pseudo-inputs
    let mk_inputs = |seed: usize, count: usize| -> Vec<f32> {
        let mut v = Vec::with_capacity(count * img_f);
        for i in 0..count * img_f {
            v.push(((i * 7 + seed * 13) % 251) as f32 / 251.0 - 0.5);
        }
        v
    };

    let infer_half = |m: &GgmlVitModel, inputs: &[f32], count: usize| {
        for _ in 0..2 {
            m.infer_batch(inputs, count).unwrap();
        }
        let t = Instant::now();
        m.infer_batch(inputs, count).unwrap();
        t.elapsed().as_secs_f64()
    };

    // --- single instance baseline ---
    let mut m1 = GgmlVitModel::new(cfg.clone()).unwrap();
    m1.load_weights(std::path::Path::new("models/dinov3_vits16.bin")).unwrap();
    let inputs = mk_inputs(1, n);
    let secs = infer_half(&m1, &inputs, n);
    println!("[single] {} imgs in {:.3} s -> {:.0} img/s", n, secs, n as f64 / secs);
    drop(m1);

    // --- two instances, two threads, disjoint halves ---
    let mut m1 = GgmlVitModel::new(cfg.clone()).unwrap();
    m1.load_weights(std::path::Path::new("models/dinov3_vits16.bin")).unwrap();
    let mut m2 = GgmlVitModel::new(cfg.clone()).unwrap();
    m2.load_weights(std::path::Path::new("models/dinov3_vits16.bin")).unwrap();
    let m1 = std::sync::Arc::new(m1);
    let m2 = std::sync::Arc::new(m2);

    let half = n / 2;
    let a = mk_inputs(1, half);
    let b = mk_inputs(2, n - half);
    // warm both
    let _ = m1.infer_batch(&a, half).unwrap();
    let _ = m2.infer_batch(&b, n - half).unwrap();

    let m1c = m1.clone();
    let m2c = m2.clone();
    let t = Instant::now();
    let h1 = std::thread::spawn(move || infer_half(&m1c, &a, half));
    let h2 = std::thread::spawn(move || infer_half(&m2c, &b, n - half));
    let (s1, s2) = (h1.join().unwrap(), h2.join().unwrap());
    let wall = t.elapsed().as_secs_f64();
    println!(
        "[dual]   {} imgs wall {:.3} s -> {:.0} img/s (thread rates {:.0} / {:.0})",
        n, wall, n as f64 / wall, half as f64 / s1, (n - half) as f64 / s2
    );
}
