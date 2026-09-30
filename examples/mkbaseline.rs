//! Regenerate the test images as baseline (non-progressive) JPEGs, so the
//! benchmark can compare decoder performance on progressive vs baseline
//! scans. Output goes to test_baseline/.
//!
//! Usage: cargo run --release --example mkbaseline

use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from("test_baseline");
    std::fs::create_dir_all(&out_dir).expect("create test_baseline/");

    let mut files: Vec<PathBuf> = std::fs::read_dir("test")
        .expect("cannot read test/")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|e| e.to_str()).is_some())
        .collect();
    files.sort();

    for p in &files {
        let img = image::open(p).expect("decode failed");
        // image's JPEG encoder emits baseline scans
        let out = out_dir.join(p.file_name().unwrap());
        let mut buf = Vec::new();
        img.to_rgb8()
            .write_with_encoder(image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 92))
            .expect("encode failed");
        std::fs::write(&out, &buf).expect("write failed");
        println!("{} -> {} ({} bytes)", p.display(), out.display(), buf.len());
    }
}
