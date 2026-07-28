//! Image preprocessing pipeline for DINOv3 inference.
//!
//! Converts raw image bytes into a normalized NCHW tensor:
//! decode -> resize (Lanczos3) -> RGB f32 -> [0,1] -> ImageNet standardization.
//!
//! Must stay bit-identical with the preprocessing used by any reference
//! backend when comparing features across implementations.

use anyhow::{Context, Result};
use image::{imageops::FilterType, DynamicImage};

use crate::config::VitConfig;

/// Preprocess raw image bytes into a normalized NCHW float tensor
/// of shape [1, 3, input_height, input_width] (flattened).
pub fn preprocess_image(image_data: &[u8], config: &VitConfig) -> Result<Vec<f32>> {
    let img = image::load_from_memory(image_data)
        .context("Failed to decode image from bytes")?;
    preprocess_dynamic_image(&img, config)
}

/// Preprocess a decoded [`DynamicImage`] into a normalized NCHW float tensor.
pub fn preprocess_dynamic_image(img: &DynamicImage, config: &VitConfig) -> Result<Vec<f32>> {
    let resized = img.resize_exact(
        config.input_width as u32,
        config.input_height as u32,
        FilterType::Lanczos3,
    );

    let rgb = resized.to_rgb8();
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);
    let pixels = rgb.as_raw();

    // HWC -> NCHW with normalization
    let num_pixels = width * height;
    let mut tensor = vec![0.0f32; 3 * num_pixels];

    for y in 0..height {
        for x in 0..width {
            let src_idx = (y * width + x) * 3;
            let r = pixels[src_idx] as f32 / 255.0;
            let g = pixels[src_idx + 1] as f32 / 255.0;
            let b = pixels[src_idx + 2] as f32 / 255.0;

            let dst_idx = y * width + x;
            tensor[dst_idx] = (r - config.image_mean[0]) / config.image_std[0];
            tensor[num_pixels + dst_idx] = (g - config.image_mean[1]) / config.image_std[1];
            tensor[2 * num_pixels + dst_idx] = (b - config.image_mean[2]) / config.image_std[2];
        }
    }

    Ok(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocess_output_shape() {
        let config = VitConfig::vit_small_16();
        let img = DynamicImage::new_rgb8(100, 100);
        let result = preprocess_dynamic_image(&img, &config).unwrap();
        assert_eq!(result.len(), 3 * 518 * 518);
    }

    #[test]
    fn test_preprocess_normalization() {
        let config = VitConfig::vit_small_16();
        let mut img = DynamicImage::new_rgb8(10, 10);
        for pixel in img.as_mut_rgb8().unwrap().pixels_mut() {
            *pixel = image::Rgb([255, 255, 255]);
        }
        let result = preprocess_dynamic_image(&img, &config).unwrap();
        // White pixel R channel: (1.0 - 0.485) / 0.229
        let expected_r = (1.0 - 0.485) / 0.229;
        assert!((result[0] - expected_r).abs() < 0.01);
    }
}
