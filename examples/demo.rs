//! DINOv3 GGML demo: extract features and compare images pairwise.
//!
//! Usage:
//!   cargo run --release --example demo [--features cuda] -- <image1> <image2> ...
//!
//! With no arguments, all jpg/png images in ./test are used.
//! Model path: GGML_MODEL_PATH env var, default ./models/dinov3_vits16.bin

use std::path::PathBuf;
use std::time::Instant;

use dinov3_ggml::{cosine_similarity, l2_normalize, FeatureExtractor, VitConfig};

fn main() {
    println!("=== DINOv3 GGML Demo ===\n");

    // 1. Collect images: CLI args, or ./test directory
    let mut images: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
    if images.is_empty() {
        if let Ok(dir) = std::fs::read_dir("test") {
            images = dir
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| {
                    matches!(
                        p.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()),
                        Some(ref e) if e == "jpg" || e == "jpeg" || e == "png"
                    )
                })
                .collect();
            images.sort();
        }
    }
    if images.is_empty() {
        eprintln!("No images. Pass image paths as arguments or put jpg/png files in ./test");
        std::process::exit(1);
    }

    // 2. Load model
    let model_path = std::env::var("GGML_MODEL_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("models/dinov3_vits16.bin"));
    if !model_path.exists() {
        eprintln!("Model not found: {}", model_path.display());
        eprintln!("See README for download & conversion instructions.");
        std::process::exit(1);
    }

    let t = Instant::now();
    let extractor = match FeatureExtractor::load(&model_path, VitConfig::vit_small_16()) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Failed to load model: {e:#}");
            std::process::exit(1);
        }
    };
    println!("Model loaded in {:.2?}\n", t.elapsed());

    // 3. Extract features
    let mut features: Vec<(String, Vec<f32>)> = Vec::new();
    for path in &images {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let data = match std::fs::read(path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("  {name}: cannot read - {e}");
                std::process::exit(1);
            }
        };
        let t = Instant::now();
        match extractor.extract(&data) {
            Ok(mut f) => {
                l2_normalize(&mut f);
                println!(
                    "  {name}: {:.1} ms, {} features",
                    t.elapsed().as_secs_f64() * 1000.0,
                    f.len()
                );
                features.push((name, f));
            }
            Err(e) => {
                eprintln!("  {name}: FAILED - {e:#}");
                std::process::exit(1);
            }
        }
    }

    // 4. Pairwise similarities
    if features.len() > 1 {
        println!("\n--- Pairwise cosine similarities ---");
        for i in 0..features.len() {
            for j in (i + 1)..features.len() {
                let sim = cosine_similarity(&features[i].1, &features[j].1);
                println!("  {} vs {} : {:.4}", features[i].0, features[j].0, sim);
            }
        }
    }

    println!("\n=== Done ===");
}
