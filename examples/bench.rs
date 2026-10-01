//! Batch throughput benchmark with per-stage breakdown.
//!
//! Usage:
//!   cargo run --release --features cuda --example bench -- [n_images] [res] [max_batch]
//!
//! Defaults: 200 images, 518px, max_batch=1.
//! All images are loaded into memory first: this measures CPU/GPU pipeline
//! throughput, not disk I/O.
//!
//! Stages reported:
//!   [load]    model load time
//!   [preproc] preprocess-only throughput (decode+resize+normalize, rayon)
//!   [infer]   inference-only throughput (preprocessed buffers in, GPU only)
//!   [batch]   end-to-end extract_batch throughput (pipelined)
//!   [sim]     pairwise sims of the first 3 images — accuracy tracker,
//!             compare this line across code changes at the same resolution

use std::path::PathBuf;
use std::time::Instant;

use dinov3_ggml::{
    cosine_similarity, l2_normalize, preprocess_image, FeatureExtractor, GgmlVitModel, VitConfig,
};
use rayon::prelude::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_images: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let res: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(518);
    let max_batch: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
    // Optional: progressive-JPEG scan limit for the decode fast path
    let scan_limit: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);

    // --- 1. Test images into memory (BENCH_TEST_DIR overrides the default) ---
    let test_dir = PathBuf::from(
        std::env::var("BENCH_TEST_DIR").unwrap_or_else(|_| "test".to_string()),
    );
    let mut files: Vec<PathBuf> = std::fs::read_dir(&test_dir)
        .expect("cannot read test dir (put jpg/png images there)")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            matches!(
                p.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase),
                Some(ref e) if e == "jpg" || e == "jpeg" || e == "png"
            )
        })
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no test images found in test/");
    let image_bytes: Vec<Vec<u8>> = files.iter().map(|p| std::fs::read(p).expect("read image")).collect();
    let total_bytes: usize = image_bytes.iter().map(|b| b.len()).sum();
    println!(
        "[data] {} files, {:.1} MB total from {} | rayon threads: {} | target: {} imgs @ {}x{}, max_batch={}",
        image_bytes.len(),
        total_bytes as f64 / 1e6,
        test_dir.display(),
        rayon::current_num_threads(),
        n_images,
        res,
        res,
        max_batch
    );

    let images: Vec<&[u8]> = (0..n_images)
        .map(|i| image_bytes[i % image_bytes.len()].as_slice())
        .collect();

    // --- 2. Config + raw model (for stage benchmarks) ---
    let mut cfg = VitConfig::vit_small_16();
    cfg.input_height = res;
    cfg.input_width = res;
    cfg.jpeg_scan_limit = scan_limit;
    let hidden_size = cfg.hidden_size;
    std::env::set_var("GGML_VIT_MAX_BATCH", max_batch.to_string());

    let t = Instant::now();
    let mut model = GgmlVitModel::new(cfg.clone()).expect("model create failed");
    model
        .load_weights(&PathBuf::from("models/dinov3_vits16.bin"))
        .expect("weight load failed");
    println!(
        "[load] {} ms (res {}x{}, seq {} tokens)",
        t.elapsed().as_millis(),
        res,
        res,
        cfg.sequence_length()
    );

    // --- 3. Preprocess-only throughput (CPU: decode + resize + normalize) ---
    let n_sample = n_images.min(256);
    let sample: Vec<&[u8]> = images.iter().take(n_sample).cloned().collect();
    let t = Instant::now();
    let pre: Vec<Vec<f32>> = sample
        .par_iter()
        .map(|b| preprocess_image(b, &cfg).expect("preprocess failed"))
        .collect();
    let secs = t.elapsed().as_secs_f64();
    println!(
        "[preproc] {} imgs in {:.2} s -> {:.0} img/s (CPU decode+resize+normalize)",
        n_sample,
        secs,
        n_sample as f64 / secs
    );

    // --- 3b. Preprocess stage split (single-thread, decode vs resize+normalize) ---
    {
        use dinov3_ggml::preprocess_dynamic_image;
        let n_split = n_sample.min(16);
        let (mut t_dec, mut t_rest) = (std::time::Duration::ZERO, std::time::Duration::ZERO);
        for b in sample.iter().take(n_split) {
            let t = Instant::now();
            let img = image::load_from_memory(b).expect("decode failed");
            t_dec += t.elapsed();
            let t = Instant::now();
            let _ = preprocess_dynamic_image(&img, &cfg).expect("resize+norm failed");
            t_rest += t.elapsed();
        }
        let k = n_split as f64;
        println!(
            "[split] decode {:.2} ms | resize+normalize {:.2} ms (per img, single-thread)",
            t_dec.as_secs_f64() * 1000.0 / k,
            t_rest.as_secs_f64() * 1000.0 / k
        );
    }

    // --- 4. Inference-only throughput (preprocessed input, no decode/resize) ---
    let img_floats = 3 * res * res;
    let mut flat: Vec<f32> = Vec::with_capacity(n_sample * img_floats);
    for p in &pre {
        flat.extend_from_slice(p);
    }
    // Warmup (CUDA graph capture happens on first call)
    let _ = model.infer_batch(&flat, n_sample).expect("infer failed");
    let t = Instant::now();
    let _ = model.infer_batch(&flat, n_sample).expect("infer failed");
    let secs = t.elapsed().as_secs_f64();
    println!(
        "[infer] {} imgs in {:.2} s -> {:.0} img/s (GPU only, preprocessed input)",
        n_sample,
        secs,
        n_sample as f64 / secs
    );
    drop(model);

    // --- 5. End-to-end extract_batch (pipelined producer/consumer) ---
    let t = Instant::now();
    let extractor = FeatureExtractor::load("models/dinov3_vits16.bin", cfg).expect("load failed");
    println!("[load2] {} ms", t.elapsed().as_millis());

    let warm: Vec<&[u8]> = images.iter().take(8.min(n_images)).cloned().collect();
    let _ = extractor.extract_batch(&warm).expect("warmup failed");

    let t = Instant::now();
    let feats = extractor.extract_batch(&images).expect("extract_batch failed");
    let secs = t.elapsed().as_secs_f64();
    println!(
        "[batch] {} imgs in {:.2} s -> {:.0} img/s (end-to-end, {:.2} ms/img)",
        n_images,
        secs,
        n_images as f64 / secs,
        secs * 1000.0 / n_images as f64
    );

    // --- 6. Accuracy smoke: pairwise sims of the first 3 images ---
    let mut f: Vec<Vec<f32>> = feats.iter().take(3).cloned().collect();
    for v in &mut f {
        l2_normalize(v);
    }
    if f.len() == 3 {
        println!(
            "[sim] {:.4} {:.4} {:.4} (dim {})",
            cosine_similarity(&f[0], &f[1]),
            cosine_similarity(&f[0], &f[2]),
            cosine_similarity(&f[1], &f[2]),
            feats[0].len()
        );
        // CLS-token view (first hidden_size dims, the global descriptor):
        // similarity structure reference for app-level pooling
        let mut c: Vec<Vec<f32>> = f.iter().map(|v| v[..hidden_size].to_vec()).collect();
        for v in &mut c {
            l2_normalize(v);
        }
        println!(
            "[sim-cls] {:.4} {:.4} {:.4} (dim {})",
            cosine_similarity(&c[0], &c[1]),
            cosine_similarity(&c[0], &c[2]),
            cosine_similarity(&c[1], &c[2]),
            hidden_size
        );
    }
}
