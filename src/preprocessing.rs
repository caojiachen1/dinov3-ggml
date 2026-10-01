//! Image preprocessing pipeline for DINOv3 inference.
//!
//! Converts raw image bytes into a normalized NCHW tensor:
//! decode -> resize (Lanczos3, SIMD via fast_image_resize) -> RGB f32
//! -> [0,1] -> ImageNet standardization.
//!
//! JPEG fast path (opt-in via [`VitConfig::scaled_decode`], on by default):
//! libjpeg-turbo decodes with the largest DCT scaling (1/8, 1/4, 1/2) that
//! still covers the model input, optionally truncated to the first
//! [`VitConfig::jpeg_scan_limit`] scans of progressive JPEGs. Both are
//! deliberate approximations that cut decode cost several-fold on large
//! photos and break the progressive-JPEG entropy-decode wall; measured
//! impact is quantified by `examples/oracle.rs` (Gate B).
//!
//! The unscaled path and all non-JPEG formats decode via the `image` crate
//! (zune-jpeg), which was benchmarked faster than turbojpeg at full
//! resolution. The SIMD resize kernel is not bit-identical to the previous
//! `image`-crate implementation (same Lanczos3 filter).

use anyhow::{Context, Result};
use fast_image_resize::{
    images::Image as FirImage, FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer,
};
use image::DynamicImage;
use turbojpeg::{Decompressor, PixelFormat, ScalingFactor};

use crate::config::VitConfig;

/// Preprocess raw image bytes into a normalized NCHW float tensor
/// of shape [1, 3, input_height, input_width] (flattened).
pub fn preprocess_image(image_data: &[u8], config: &VitConfig) -> Result<Vec<f32>> {
    // JPEG fast path: DCT-scaled (and optionally scan-limited) decode when a
    // downscaled decode still covers the model input. Falls back to the
    // image crate otherwise (non-JPEG, small images, decode errors).
    if config.scaled_decode
        && image_data.len() >= 2
        && image_data[0] == 0xFF
        && image_data[1] == 0xD8
    {
        if let Some((w, h, pixels)) = decode_jpeg_scaled(
            image_data,
            config.input_width as u32,
            config.input_height as u32,
            config.jpeg_scan_limit,
        ) {
            return preprocess_rgb8(w, h, pixels, config);
        }
    }

    let img = image::load_from_memory(image_data)
        .context("Failed to decode image from bytes")?;
    preprocess_dynamic_image(&img, config)
}

/// Decode a JPEG with the largest supported DCT scaling factor (1/8, 1/4,
/// 1/2) whose scaled dimensions still cover `target_w x target_h`, so the
/// subsequent resize never upscales. `scan_limit` > 0 truncates progressive
/// JPEGs to their first N scans. Returns `(width, height, packed RGB8)`, or
/// None when scaled decoding is not applicable (lossless JPEG, decode
/// error, source too small for any scaling factor).
fn decode_jpeg_scaled(
    image_data: &[u8],
    target_w: u32,
    target_h: u32,
    scan_limit: u32,
) -> Option<(usize, usize, Vec<u8>)> {
    let mut decompressor = Decompressor::new().ok()?;
    let header = decompressor.read_header(image_data).ok()?;
    if header.is_lossless {
        return None;
    }
    // Ascending scale: the first factor that covers the target wins (the
    // cheapest decode that doesn't throw away needed resolution).
    for factor in [
        ScalingFactor::ONE_EIGHTH,
        ScalingFactor::ONE_QUARTER,
        ScalingFactor::ONE_HALF,
    ] {
        let w = factor.scale(header.width);
        let h = factor.scale(header.height);
        if (w as u32) < target_w || (h as u32) < target_h {
            continue;
        }
        decompressor.set_scaling_factor(factor).ok()?;
        // Cheap chroma-upsample fast path (matters at 1/2 scale; no-op when
        // all planes are scaled as at 1/8)
        decompressor.set_fast_upsample(true).ok()?;
        if scan_limit > 0 {
            decompressor.set_scan_limit(scan_limit).ok()?;
        }
        let mut img = turbojpeg::Image {
            pixels: vec![0u8; h * w * 3],
            width: w,
            pitch: w * 3,
            height: h,
            format: PixelFormat::RGB,
        };
        decompressor
            .decompress(image_data, img.as_deref_mut())
            .ok()?;
        return Some((img.width, img.height, img.pixels));
    }
    None
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
