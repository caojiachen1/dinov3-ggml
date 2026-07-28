//! Model configuration for DINOv3 Vision Transformer variants.

/// Configuration for a DINOv3 ViT model.
///
/// Contains architecture hyperparameters plus preprocessing constants
/// (input size, ImageNet normalization).
#[derive(Debug, Clone)]
pub struct VitConfig {
    /// Dimensionality of the hidden representations.
    pub hidden_size: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimensionality of the feed-forward intermediate layer.
    pub intermediate_size: usize,
    /// Size of each image patch (16 for ViT-16).
    pub patch_size: usize,
    /// Number of register tokens (4 in DINOv3).
    pub num_register_tokens: usize,
    /// Expected input image height in pixels.
    pub input_height: usize,
    /// Expected input image width in pixels.
    pub input_width: usize,
    /// RoPE frequency base (DINOv3 rope_theta, 100.0).
    pub rope_freq_base: f32,
    /// ImageNet normalization mean for RGB channels.
    pub image_mean: [f32; 3],
    /// ImageNet normalization standard deviation for RGB channels.
    pub image_std: [f32; 3],
}

impl VitConfig {
    /// ViT-S/16: 384 hidden dims, 12 layers, 6 heads, 1536 intermediate.
    pub fn vit_small_16() -> Self {
        Self {
            hidden_size: 384,
            num_layers: 12,
            num_heads: 6,
            intermediate_size: 1536,
            ..Self::base_defaults()
        }
    }

    /// ViT-B/16: 768 hidden dims, 12 layers, 12 heads, 3072 intermediate.
    pub fn vit_base_16() -> Self {
        Self {
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 3072,
            ..Self::base_defaults()
        }
    }

    /// ViT-L/16: 1024 hidden dims, 24 layers, 16 heads, 4096 intermediate.
    pub fn vit_large_16() -> Self {
        Self {
            hidden_size: 1024,
            num_layers: 24,
            num_heads: 16,
            intermediate_size: 4096,
            ..Self::base_defaults()
        }
    }

    fn base_defaults() -> Self {
        Self {
            hidden_size: 0,
            num_layers: 0,
            num_heads: 0,
            intermediate_size: 0,
            patch_size: 16,
            num_register_tokens: 4,
            input_height: 518,
            input_width: 518,
            rope_freq_base: 100.0,
            image_mean: [0.485, 0.456, 0.406],
            image_std: [0.229, 0.224, 0.225],
        }
    }

    /// Number of patch tokens for the configured input size.
    pub fn num_patches(&self) -> usize {
        (self.input_height / self.patch_size) * (self.input_width / self.patch_size)
    }

    /// Total sequence length: CLS + registers + patches.
    pub fn sequence_length(&self) -> usize {
        1 + self.num_register_tokens + self.num_patches()
    }

    /// Output feature dimension: sequence_length * hidden_size.
    pub fn output_dim(&self) -> usize {
        self.sequence_length() * self.hidden_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vit_small_16_dims() {
        let c = VitConfig::vit_small_16();
        // 518 / 16 = 32 patches per dim -> 1024 patches, 1029 tokens
        assert_eq!(c.num_patches(), 1024);
        assert_eq!(c.sequence_length(), 1029);
        assert_eq!(c.output_dim(), 1029 * 384);
    }
}
