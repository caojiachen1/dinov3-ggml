//! # dinov3-ggml
//!
//! DINOv3 Vision Transformer inference with [GGML](https://github.com/ggml-org/ggml),
//! numerically matching the official Hugging Face `DINOv3ViTModel` implementation
//! (patch embedding, [CLS, registers, patches] token order, DINOv3-style RoPE
//! with normalized patch-center coordinates, LayerScale, exact-erf GELU).
//!
//! Backends: CPU (default) or CUDA (`cuda` feature). Typical usage:
//!
//! ```no_run
//! use dinov3_ggml::{FeatureExtractor, VitConfig, l2_normalize, cosine_similarity};
//!
//! let extractor = FeatureExtractor::load("models/dinov3_vits16.bin", VitConfig::vit_small_16())?;
//! let mut a = extractor.extract(&std::fs::read("a.jpg")?)?;
//! let mut b = extractor.extract(&std::fs::read("b.jpg")?)?;
//! l2_normalize(&mut a);
//! l2_normalize(&mut b);
//! println!("similarity = {}", cosine_similarity(&a, &b));
//! # anyhow::Ok(())
//! ```

pub mod config;
pub mod postprocessing;
pub mod preprocessing;

pub use config::VitConfig;
pub use postprocessing::{cosine_similarity, find_similar, l2_normalize};
pub use preprocessing::{preprocess_dynamic_image, preprocess_image};

use std::ffi::CString;
use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};
use rayon::prelude::*;

/// Raw FFI bindings to the ggml_vit C library (csrc/ggml_vit.c).
#[allow(non_camel_case_types)]
mod ffi {
    use std::ffi::{c_char, c_float, c_int};

    /// Opaque model handle from the C library.
    pub enum ggml_vit_model {}

    /// Model configuration matching the C struct in include/ggml_vit.h.
    #[repr(C)]
    pub struct vit_config_t {
        pub hidden_size: c_int,
        pub num_layers: c_int,
        pub num_heads: c_int,
        pub intermediate_size: c_int,
        pub patch_size: c_int,
        pub num_register_tokens: c_int,
        pub input_height: c_int,
        pub input_width: c_int,
        pub has_cls_token: c_int,
        pub layer_scale_eps: c_float,
        pub rope_freq_base: c_float,
        pub max_batch: c_int,
    }

    extern "C" {
        pub fn ggml_vit_create(config: *const vit_config_t) -> *mut ggml_vit_model;
        pub fn ggml_vit_load_weights(model: *mut ggml_vit_model, path: *const c_char) -> c_int;
        pub fn ggml_vit_infer(
            model: *mut ggml_vit_model,
            input: *const c_float,
            height: c_int,
            width: c_int,
            output: *mut c_float,
            output_size: c_int,
        ) -> c_int;
        pub fn ggml_vit_infer_batch(
            model: *mut ggml_vit_model,
            input: *const c_float,
            n_images: c_int,
            height: c_int,
            width: c_int,
            output: *mut c_float,
            output_size: c_int,
        ) -> c_int;
        pub fn ggml_vit_destroy(model: *mut ggml_vit_model);
        #[allow(dead_code)]
        pub fn ggml_vit_get_output_size(model: *const ggml_vit_model) -> c_int;
    }
}

/// Low-level GGML ViT model.
///
/// Wraps the C ggml_vit_model with safe Rust lifetime management.
/// Inference calls are serialized internally: the C implementation shares a
/// single graph allocator and backend, so concurrent infer() is not safe.
pub struct GgmlVitModel {
    inner: *mut ffi::ggml_vit_model,
    config: VitConfig,
    max_batch: usize,
    infer_lock: Mutex<()>,
}

// SAFETY: The C model is only mutated during inference, which is serialized
// by `infer_lock`; weight loading takes &mut self.
unsafe impl Send for GgmlVitModel {}
unsafe impl Sync for GgmlVitModel {}

impl GgmlVitModel {
    /// Create a new model with the given configuration.
    ///
    /// Uses the CUDA backend when compiled with the `cuda` feature and a GPU
    /// is available (set `GGML_VIT_FORCE_CPU=1` to override), else CPU.
    pub fn new(config: VitConfig) -> Result<Self> {
        // Batched-graph size (GGML_VIT_MAX_BATCH). Default 1: on GPUs already
        // saturated by a single image, batching only adds latency.
        let max_batch = std::env::var("GGML_VIT_MAX_BATCH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v >= 1)
            .unwrap_or(1);
        let ffi_config = ffi::vit_config_t {
            hidden_size: config.hidden_size as i32,
            num_layers: config.num_layers as i32,
            num_heads: config.num_heads as i32,
            intermediate_size: config.intermediate_size as i32,
            patch_size: config.patch_size as i32,
            num_register_tokens: config.num_register_tokens as i32,
            input_height: config.input_height as i32,
            input_width: config.input_width as i32,
            has_cls_token: 1,
            layer_scale_eps: 1.0,
            rope_freq_base: config.rope_freq_base,
            max_batch: max_batch as i32,
        };
        let inner = unsafe { ffi::ggml_vit_create(&ffi_config) };
        if inner.is_null() {
            anyhow::bail!("Failed to create GGML ViT model");
        }
        Ok(Self { inner, config, max_batch, infer_lock: Mutex::new(()) })
    }

    /// Load weights from a raw binary file (see scripts/convert_weights.py).
    pub fn load_weights(&mut self, path: &Path) -> Result<()> {
        // The raw format carries no header: catch config/file mismatches early
        // by checking the exact expected size instead of failing mid-read.
        let meta = std::fs::metadata(path)
            .with_context(|| format!("Cannot access weight file {:?}", path))?;
        let expected = self.config.weight_file_size();
        anyhow::ensure!(
            meta.len() == expected,
            "Weight file {:?} is {} bytes, but the configured model expects {} bytes.\n\
             The file was probably converted for a different architecture \
             (see scripts/convert_weights.py --model).",
            path, meta.len(), expected
        );

        let c_path = CString::new(path.to_str().context("Invalid path encoding")?)
            .context("Path contains null byte")?;
        let ret = unsafe { ffi::ggml_vit_load_weights(self.inner, c_path.as_ptr()) };
        if ret != 0 {
            anyhow::bail!("Failed to load weights from {:?}", path);
        }
        log::info!("Loaded ViT weights from {:?}", path);
        Ok(())
    }

    /// Run inference on a preprocessed NCHW float tensor
    /// ([1, 3, height, width] flattened). Returns output_dim() features.
    ///
    /// `height`/`width` must match the configured input size: the RoPE tables
    /// and token grid are precomputed for it at model creation.
    pub fn infer(&self, input: &[f32], height: i32, width: i32) -> Result<Vec<f32>> {
        // Validate before handing the pointer to C: a short slice would cause
        // an out-of-bounds read, a wrong size silently corrupts the results.
        anyhow::ensure!(
            height as usize == self.config.input_height && width as usize == self.config.input_width,
            "Input size {}x{} does not match the configured model input {}x{}",
            width, height, self.config.input_width, self.config.input_height
        );
        let expected = 3 * self.config.input_height * self.config.input_width;
        anyhow::ensure!(
            input.len() == expected,
            "Input tensor has {} floats, expected {} (3 x {} x {})",
            input.len(), expected, self.config.input_height, self.config.input_width
        );

        let output_size = self.config.output_dim() as i32;
        let mut output = vec![0.0f32; output_size as usize];

        // Serialize access to the shared C-side graph allocator/backend
        let _guard = self.infer_lock.lock()
            .map_err(|_| anyhow::anyhow!("Inference lock poisoned"))?;
        let ret = unsafe {
            ffi::ggml_vit_infer(
                self.inner,
                input.as_ptr(),
                height,
                width,
                output.as_mut_ptr(),
                output_size,
            )
        };

        if ret != 0 {
            anyhow::bail!("ViT inference failed");
        }

        Ok(output)
    }

    /// Run batched inference on `n_images` preprocessed images stored back to
    /// back in `inputs` (each 3*H*W floats). Returns one feature vector per
    /// image. Uses the prebuilt batched compute graph (max_batch per pass).
    pub fn infer_batch(&self, inputs: &[f32], n_images: usize) -> Result<Vec<Vec<f32>>> {
        anyhow::ensure!(n_images > 0, "infer_batch called with 0 images");
        let img_floats = 3 * self.config.input_height * self.config.input_width;
        anyhow::ensure!(
            inputs.len() == n_images * img_floats,
            "Input buffer has {} floats, expected {} ({} images x {})",
            inputs.len(), n_images * img_floats, n_images, img_floats
        );

        let out_dim = self.config.output_dim();
        let mut output = vec![0.0f32; n_images * out_dim];

        let _guard = self.infer_lock.lock()
            .map_err(|_| anyhow::anyhow!("Inference lock poisoned"))?;
        let ret = unsafe {
            ffi::ggml_vit_infer_batch(
                self.inner,
                inputs.as_ptr(),
                n_images as i32,
                self.config.input_height as i32,
                self.config.input_width as i32,
                output.as_mut_ptr(),
                (n_images * out_dim) as i32,
            )
        };
        drop(_guard);

        if ret != 0 {
            anyhow::bail!("ViT batched inference failed");
        }

        Ok(output.chunks_exact(out_dim).map(|c| c.to_vec()).collect())
    }

    /// Images per batched forward pass (GGML_VIT_MAX_BATCH, default 1).
    pub fn max_batch(&self) -> usize {
        self.max_batch
    }

    /// Get the model configuration.
    pub fn config(&self) -> &VitConfig {
        &self.config
    }
}

impl Drop for GgmlVitModel {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe { ffi::ggml_vit_destroy(self.inner) };
            self.inner = std::ptr::null_mut();
        }
    }
}

/// High-level feature extractor: image bytes in, DINOv3 features out.
///
/// Output is the flattened `last_hidden_state` [seq_len, hidden_size]
/// (token order [CLS, registers, patches]), NOT L2-normalized —
/// call [`l2_normalize`] before computing cosine similarities.
pub struct FeatureExtractor {
    model: GgmlVitModel,
}

impl FeatureExtractor {
    /// Create the model and load weights in one step.
    pub fn load(weights_path: impl AsRef<Path>, config: VitConfig) -> Result<Self> {
        let mut model = GgmlVitModel::new(config)?;
        model.load_weights(weights_path.as_ref())?;
        Ok(Self { model })
    }

    /// Extract features from encoded image bytes (JPEG/PNG/...).
    pub fn extract(&self, image_data: &[u8]) -> Result<Vec<f32>> {
        let config = self.model.config();
        let input = preprocess_image(image_data, config)
            .context("Failed to preprocess image")?;
        self.model
            .infer(&input, config.input_height as i32, config.input_width as i32)
            .context("ViT inference failed")
    }

    /// Batch extraction with pipelined parallelism:
    /// image decode + resize (CPU-bound) runs on all cores via rayon a chunk
    /// ahead, while the GPU runs batched inference (max_batch images per
    /// forward pass) on the previous chunk.
    pub fn extract_batch(&self, images: &[&[u8]]) -> Result<Vec<Vec<f32>>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        let config = self.model.config();
        // Preprocessing chunk: large enough to keep all cores busy regardless
        // of the GPU batch size (infer_batch splits into max_batch chunks).
        let chunk_size = self.model.max_batch().max(16);
        let img_floats = 3 * config.input_height * config.input_width;

        let (tx, rx) = std::sync::mpsc::sync_channel::<(usize, Result<Vec<f32>>)>(2);

        std::thread::scope(|s| {
            // Producer: preprocess one chunk at a time (parallel within chunk),
            // at most 2 chunks ahead of the GPU (sync_channel backpressure).
            s.spawn(move || {
                for (ci, chunk) in images.chunks(chunk_size).enumerate() {
                    let pre: Vec<Result<Vec<f32>>> = chunk
                        .par_iter()
                        .map(|img| preprocess_image(img, config).context("Failed to preprocess image"))
                        .collect();

                    let mut flat = Vec::with_capacity(chunk.len() * img_floats);
                    let mut result: Result<Vec<f32>> = Ok(Vec::new());
                    for (i, p) in pre.into_iter().enumerate() {
                        match p {
                            Ok(v) => flat.extend_from_slice(&v),
                            Err(e) => {
                                result = Err(e.context(format!(
                                    "image {}/{}", ci * chunk_size + i + 1, images.len()
                                )));
                                break;
                            }
                        }
                    }
                    if result.is_ok() {
                        result = Ok(flat);
                    }
                    let failed = result.is_err();
                    if tx.send((chunk.len(), result)).is_err() || failed {
                        return; // consumer gone or error forwarded
                    }
                }
            });

            // Consumer: batched GPU inference per chunk.
            let mut features = Vec::with_capacity(images.len());
            for (n, flat) in rx {
                let flat = flat?;
                let mut out = self
                    .model
                    .infer_batch(&flat, n)
                    .with_context(|| format!(
                        "Batched ViT inference failed (images {}..{})",
                        features.len() + 1, features.len() + n
                    ))?;
                features.append(&mut out);
            }
            Ok(features)
        })
    }

    /// Output feature dimension (sequence_length * hidden_size).
    pub fn output_dim(&self) -> usize {
        self.model.config().output_dim()
    }

    /// Get the model configuration.
    pub fn config(&self) -> &VitConfig {
        self.model.config()
    }
}
