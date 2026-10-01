//! Dump full feature vectors for the first N test images (one line per image,
//! space-separated floats) plus the CLS slice, for Gate B accuracy comparison
//! between two builds.
//!
//! Usage: dumpfeat [res] [n] [max_batch]

use std::path::PathBuf;

use dinov3_ggml::{preprocess_image, GgmlVitModel, VitConfig};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let res: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(256);
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
    let mb: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4);
    std::env::set_var("GGML_VIT_MAX_BATCH", mb.to_string());

    let mut cfg = VitConfig::vit_small_16();
    cfg.input_height = res;
    cfg.input_width = res;

    let mut model = GgmlVitModel::new(cfg.clone()).expect("model create failed");
    model
        .load_weights(&PathBuf::from("models/dinov3_vits16.bin"))
        .expect("weight load failed");

    let files: Vec<PathBuf> = std::fs::read_dir("test")
        .expect("test dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|e| e == "jpg").unwrap_or(false))
        .collect();

    let mut flat = Vec::new();
    let mut count = 0;
    for p in files.iter().take(n) {
        let bytes = std::fs::read(p).unwrap();
        let input = preprocess_image(&bytes, &cfg).unwrap();
        flat.extend_from_slice(&input);
        count += 1;
    }

    let feats = model.infer_batch(&flat, count).expect("infer failed");
    for f in &feats {
        let line: Vec<String> = f.iter().map(|v| format!("{:.6}", v)).collect();
        println!("{}", line.join(" "));
    }
}
