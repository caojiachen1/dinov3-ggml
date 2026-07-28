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
        };
        let inner = unsafe { ffi::ggml_vit_create(&ffi_config) };
        if inner.is_null() {
            anyhow::bail!("Failed to create GGML ViT model");
        }
        Ok(Self { inner, config, infer_lock: Mutex::new(()) })
    }

    /// Load weights from a raw binary file (see scripts/convert_weights.py).
    pub fn load_weights(&mut self, path: &Path) -> Result<()> {
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
    pub fn infer(&self, input: &[f32], height: i32, width: i32) -> Result<Vec<f32>> {
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
    /// image decode + resize (CPU-bound) runs on all cores via rayon, while
    /// the GPU/backend inference itself is serialized (single graph allocator).
    pub fn extract_batch(&self, images: &[&[u8]]) -> Result<Vec<Vec<f32>>> {
        let config = self.model.config();

        // Stage 1: parallel preprocessing
        let inputs: Vec<Result<Vec<f32>>> = images
            .par_iter()
            .map(|img| preprocess_image(img, config).context("Failed to preprocess image"))
            .collect();

        // Stage 2: sequential inference (infer() locks internally anyway)
        inputs
            .into_iter()
            .enumerate()
            .map(|(i, input)| {
                let input = input.with_context(|| format!("image {}/{}", i + 1, images.len()))?;
                self.model
                    .infer(&input, config.input_height as i32, config.input_width as i32)
                    .with_context(|| format!("ViT inference failed for image {}/{}", i + 1, images.len()))
            })
            .collect()
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
