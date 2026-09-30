//! Image preprocessing pipeline for DINOv3 inference.
//!
//! Converts raw image bytes into a normalized NCHW tensor:
//! decode (image crate / zune-jpeg) -> resize (Lanczos3, SIMD via
//! fast_image_resize) -> RGB f32 -> [0,1] -> ImageNet standardization.
//!
//! Note: libjpeg-turbo (`turbojpeg`) was benchmarked as a JPEG fast path and
//! turned out SLOWER than zune-jpeg on progressive JPEGs (10.7 ms vs 8.4 ms
//! per 690x920 image), so the image crate stays the decoder. The SIMD resize
//! kernel is not bit-identical to the previous `image`-crate implementation
//! (same Lanczos3 filter, low-order bit differences only).

use anyhow::{Context, Result};
use fast_image_resize::{
    images::Image as FirImage, FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer,
};
use image::DynamicImage;

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
    let rgb = img.to_rgb8();
    let width = rgb.width() as usize;
    let height = rgb.height() as usize;
    preprocess_rgb8(width, height, rgb.into_raw(), config)
}

/// Preprocess an owned packed RGB8 buffer into a normalized NCHW float tensor.
fn preprocess_rgb8(
    src_w: usize,
    src_h: usize,
    pixels: Vec<u8>,
    config: &VitConfig,
) -> Result<Vec<f32>> {
    let dst_w = config.input_width as u32;
    let dst_h = config.input_height as u32;

    // SIMD resize. Lanczos3 matches the filter previously used from the
    // `image` crate; the fast_image_resize implementation is not bit-identical.
    let src_image = FirImage::from_vec_u8(src_w as u32, src_h as u32, pixels, PixelType::U8x3)
        .context("Invalid source image buffer")?;
    let mut dst_image = FirImage::new(dst_w, dst_h, PixelType::U8x3);
    let mut resizer = Resizer::new();
    let options =
        ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3));
    resizer
        .resize(&src_image, &mut dst_image, Some(&options))
        .context("Resize failed")?;

    let width = dst_w as usize;
    let height = dst_h as usize;
    let pixels = dst_image.buffer();

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
    use image::{DynamicImage, Rgb};

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
            *pixel = Rgb([255, 255, 255]);
        }
        let result = preprocess_dynamic_image(&img, &config).unwrap();
        // White pixel R channel: (1.0 - 0.485) / 0.229
        let expected_r = (1.0 - 0.485) / 0.229;
        assert!((result[0] - expected_r).abs() < 0.01);
    }
}
