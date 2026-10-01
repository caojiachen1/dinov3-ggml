//! GPU-side profiler: separates pure graph replay time from host transfers,
//! so kernel-level optimizations can be measured without decode/preprocess
//! noise.
//!
//! Usage:
//!   cargo run --release --features cuda --example gpuprof -- [res] [max_batch] [iters]

use std::path::PathBuf;
use std::time::Instant;

use dinov3_ggml::{preprocess_image, GgmlVitModel, VitConfig};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let res: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(256);
    let max_batch: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(32);
    let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(50);

    let mut cfg = VitConfig::vit_small_16();
    cfg.input_height = res;
    cfg.input_width = res;
    std::env::set_var("GGML_VIT_MAX_BATCH", max_batch.to_string());

    let t = Instant::now();
    let mut model = GgmlVitModel::new(cfg.clone()).expect("model create failed");
    model
        .load_weights(&PathBuf::from("models/dinov3_vits16.bin"))
        .expect("weight load failed");
    println!("[load] {} ms", t.elapsed().as_millis());

    // One real batch to put realistic data into the input tensor
    let files: Vec<PathBuf> = std::fs::read_dir("test")
        .expect("test dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|e| e == "jpg").unwrap_or(false))
        .collect();
    let bytes: Vec<Vec<u8>> = files.iter().map(|p| std::fs::read(p).unwrap()).collect();
    let mut flat = Vec::new();
    for b in bytes.iter().take(max_batch) {
        flat.extend_from_slice(&preprocess_image(b, &cfg).unwrap());
    }

    // 1. Pure graph replay (no upload/download)
    let per = model.bench_graph(iters).expect("bench_graph failed");
    let b = model.max_batch();
    println!(
        "[replay] {:.3} ms/graph (B={} -> {:.0} img/s, {:.3} ms/img)",
        per * 1e3,
        b,
        b as f64 / per,
        per * 1e3 / b as f64
    );

    // 2. Full infer_batch_cls path (upload + compute + CLS download)
    let reps = 8;
    let t = Instant::now();
    for _ in 0..reps {
        let _ = model.infer_batch_cls(&flat, b.min(flat.len() / (3 * res * res))).unwrap();
    }
    let secs = t.elapsed().as_secs_f64() / reps as f64;
    println!(
        "[cls-path] {:.3} ms/call incl. transfers ({:.0} img/s)",
        secs * 1e3,
        b as f64 / secs
    );

    // 3. Upload cost alone: infer_batch_cls minus replay ≈ upload+download
    let overhead = secs - per;
    println!(
        "[overhead] {:.3} ms/call (upload+download+launch, {:.1}% of cls-path)",
        overhead * 1e3,
        100.0 * overhead / secs
    );
}
