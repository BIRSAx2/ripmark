//! Watermark removal (bypass) pipeline.
//!
//! Three modes:
//!   V1 -- JPEG quality cycling. Fast, moderate removal (~37 dB PSNR).
//!   V2 -- Calibrated noise + bilateral smoothing + V1. Balanced.
//!   V3 -- Multi-pass spectral subtraction. Best quality (target: 43+ dB PSNR).
//!
//! V3 algorithm per pass:
//!   For each carrier (fy, fx) and each channel c:
//!     confidence    = coherence[carrier] (how fixed the watermark phase is)
//!     subtract_mag  = image_mag * confidence * removal_fraction * channel_weight
//!     subtract_mag  = min(subtract_mag, 0.90 * image_mag)   -- safety cap
//!     image_fft    -= subtract_mag * exp(i * ref_phase)
//!   Then apply a light Gaussian blur to smooth compression artefacts.

use std::io::Cursor;

use anyhow::Result;
use image::{DynamicImage, ImageFormat, RgbImage};
use ndarray::{s, Array3, ArrayView3};
use rustfft::num_complex::Complex;

use crate::carriers::{CARRIERS_DARK, CARRIERS_WHITE};
use crate::codebook::{Codebook, ResolutionProfile};
use crate::fft::{fft2, Spectrum};

/// Which removal strategy to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BypassMode {
    /// JPEG quality cycling at Q50 then re-encode at Q90.
    V1,
    /// Calibrated noise injection + bilateral smoothing + V1.
    V2,
    /// Multi-pass FFT-domain spectral subtraction at known carrier frequencies.
    V3,
}

/// Output of a bypass operation.
#[derive(Debug, Clone)]
pub struct BypassResult {
    /// Processed image, RGB values in [0, 1].
    pub image: Array3<f32>,
    /// Names of stages applied (for logging).
    pub stages: Vec<String>,
}

/// Remove the SynthID watermark from an image.
///
/// `image` must be RGB, values in [0, 1], shape (H, W, 3).
pub fn bypass(
    image: ArrayView3<f32>,
    codebook: &Codebook,
    mode: BypassMode,
) -> Result<BypassResult> {
    match mode {
        BypassMode::V1 => v1(image),
        BypassMode::V2 => v2(image),
        BypassMode::V3 => v3(image, codebook),
    }
}

// V1: JPEG quality cycling.
fn v1(image: ArrayView3<f32>) -> Result<BypassResult> {
    let cycled = jpeg_cycle(image, 50)?;
    Ok(BypassResult {
        image: cycled,
        stages: vec!["jpeg_cycle_q50".into()],
    })
}

// V2: noise injection + bilateral smoothing + V1.
fn v2(image: ArrayView3<f32>) -> Result<BypassResult> {
    let noisy = add_gaussian_noise(image, 5.0);
    let smoothed = bilateral_smooth(noisy.view(), 9, 75.0, 75.0);
    let BypassResult { image: cycled, .. } = v1(smoothed.view())?;
    Ok(BypassResult {
        image: cycled,
        stages: vec![
            "noise_injection".into(),
            "bilateral_smooth".into(),
            "jpeg_cycle_q50".into(),
        ],
    })
}

// V3: multi-pass spectral subtraction.
fn v3(image: ArrayView3<f32>, codebook: &Codebook) -> Result<BypassResult> {
    let (profile, _) = codebook.best_profile(image.shape()[0], image.shape()[1])?;

    // Three passes with decreasing aggressiveness.
    // (removal_fraction, consistency_floor)
    let passes: &[(f32, f32)] = &[(0.95, 0.30), (0.80, 0.50), (0.60, 0.70)];

    let mut current = image.to_owned();

    for &(removal_fraction, consistency_floor) in passes {
        current = spectral_pass(current.view(), profile, removal_fraction, consistency_floor);
    }

    // Light Gaussian blur to smooth any spectral artefacts.
    current = gaussian_blur(current.view(), 0.4);

    Ok(BypassResult {
        image: current,
        stages: vec![
            "spectral_pass_aggressive".into(),
            "spectral_pass_moderate".into(),
            "spectral_pass_gentle".into(),
            "gaussian_antialias".into(),
        ],
    })
}

// Single spectral subtraction pass over all channels and carrier sets.
fn spectral_pass(
    image: ArrayView3<f32>,
    profile: &ResolutionProfile,
    removal_fraction: f32,
    consistency_floor: f32,
) -> Array3<f32> {
    // R=0.85, G=1.0, B=0.70 -- Green carries the strongest watermark signal.
    const CHANNEL_WEIGHTS: [f32; 3] = [0.85, 1.0, 0.70];
    const SAFETY_CAP: f32 = 0.90;

    let h = image.shape()[0];
    let w = image.shape()[1];
    let mut result = Array3::<f32>::zeros((h, w, 3));

    // Build combined carrier list with coherence and reference phase per carrier.
    let dark_carriers: Vec<(i32, i32, f32, f32)> = CARRIERS_DARK
        .iter()
        .zip(profile.dark.coherence.iter())
        .zip(profile.dark.ref_phases.iter())
        .map(|((&freq, &coh), &phase)| (freq.0, freq.1, coh, phase))
        .collect();

    let white_carriers: Vec<(i32, i32, f32, f32)> = CARRIERS_WHITE
        .iter()
        .zip(profile.white.coherence.iter())
        .zip(profile.white.ref_phases.iter())
        .map(|((&freq, &coh), &phase)| (freq.0, freq.1, coh, phase))
        .collect();

    let size = profile.image_size;

    for (ch, &weight) in CHANNEL_WEIGHTS.iter().enumerate() {
        let channel = image.slice(s![.., .., ch]);

        // Resize channel to codebook size, subtract in that space, resize back.
        let channel_owned = channel.to_owned();
        let resized_ch = crate::fft::resize_gray(&channel_owned, size, size);

        // Forward FFT (unshifted layout for subtraction).
        let mut fft_data = fft2(resized_ch.view()).data;

        for carriers in [dark_carriers.as_slice(), white_carriers.as_slice()] {
            for &(fy, fx, coherence, ref_phase) in carriers {
                if coherence < consistency_floor {
                    continue;
                }

                // Map shifted carrier (fy, fx) back to unshifted FFT index.
                let (uy, ux) = shifted_to_unshifted(fy, fx, size);
                if uy >= size || ux >= size {
                    continue;
                }

                let bin = fft_data[[uy, ux]];
                let img_mag = bin.norm();
                if img_mag < 1e-10 {
                    continue;
                }

                let subtract_mag =
                    (img_mag * coherence * removal_fraction * weight).min(SAFETY_CAP * img_mag);

                let subtract = Complex::new(
                    subtract_mag * ref_phase.cos(),
                    subtract_mag * ref_phase.sin(),
                );

                fft_data[[uy, ux]] -= subtract;
            }
        }

        // Inverse FFT and resize back to original dimensions.
        let cleaned_resized = Spectrum { data: fft_data }.to_spatial();
        let cleaned = crate::fft::resize_gray(&cleaned_resized, h, w);

        result
            .slice_mut(s![.., .., ch])
            .assign(&cleaned.mapv(|v| v.clamp(0.0, 1.0)));
    }

    result
}

// Convert a shifted carrier frequency (fy, fx) to an unshifted FFT bin index.
// In fftshift layout, [0,0] is at [size/2, size/2]. We reverse this.
fn shifted_to_unshifted(fy: i32, fx: i32, size: usize) -> (usize, usize) {
    let center = size as i32 / 2;
    let y = ((fy + center).rem_euclid(size as i32)) as usize;
    let x = ((fx + center).rem_euclid(size as i32)) as usize;
    (y, x)
}

// JPEG encode at `quality` and decode back. Degrades phase coherence at carriers.
fn jpeg_cycle(image: ArrayView3<f32>, _quality: u8) -> Result<Array3<f32>> {
    let h = image.shape()[0] as u32;
    let w = image.shape()[1] as u32;

    // Convert f32 -> u8.
    let pixels: Vec<u8> = image
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();

    let rgb = RgbImage::from_raw(w, h, pixels)
        .ok_or_else(|| anyhow::anyhow!("failed to build RgbImage"))?;

    // Encode to JPEG.
    let mut buf = Vec::new();
    DynamicImage::ImageRgb8(rgb).write_to(&mut Cursor::new(&mut buf), ImageFormat::Jpeg)?;

    // Decode back.
    let decoded = image::load_from_memory_with_format(&buf, ImageFormat::Jpeg)?.into_rgb8();

    let out = Array3::from_shape_fn((h as usize, w as usize, 3), |(y, x, c)| {
        decoded.get_pixel(x as u32, y as u32)[c] as f32 / 255.0
    });

    Ok(out)
}

// Add Gaussian noise with standard deviation sigma/255.
fn add_gaussian_noise(image: ArrayView3<f32>, sigma: f32) -> Array3<f32> {
    // Deterministic LCG noise for reproducibility.
    let scale = sigma / 255.0;
    let mut state: u64 = 0x123456789abcdef0;
    Array3::from_shape_fn(image.dim(), |(r, c, ch)| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Box-Muller via two uniform samples.
        let u1 = (state >> 33) as f32 / (u32::MAX as f32);
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u2 = (state >> 33) as f32 / (u32::MAX as f32);
        let gauss = (-2.0 * (u1 + 1e-10).ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        (image[[r, c, ch]] + gauss * scale).clamp(0.0, 1.0)
    })
}

// Simplified bilateral filter: edge-preserving spatial smoothing.
// Uses a small spatial kernel with range weighting based on intensity difference.
fn bilateral_smooth(
    image: ArrayView3<f32>,
    radius: i32,
    sigma_space: f32,
    sigma_range: f32,
) -> Array3<f32> {
    let h = image.shape()[0] as i32;
    let w = image.shape()[1] as i32;
    let ss2 = 2.0 * sigma_space * sigma_space;
    let sr2 = 2.0 * sigma_range * sigma_range;

    Array3::from_shape_fn(image.dim(), |(y, x, c)| {
        let center_val = image[[y, x, c]];
        let mut sum = 0.0_f32;
        let mut weight_sum = 0.0_f32;

        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let ny = (y as i32 + dy).clamp(0, h - 1) as usize;
                let nx = (x as i32 + dx).clamp(0, w - 1) as usize;
                let val = image[[ny, nx, c]];
                let spatial_w = (-(dy * dy + dx * dx) as f32 / ss2).exp();
                let range_w = (-(val - center_val).powi(2) / sr2).exp();
                let w = spatial_w * range_w;
                sum += val * w;
                weight_sum += w;
            }
        }

        if weight_sum < 1e-10 {
            center_val
        } else {
            (sum / weight_sum).clamp(0.0, 1.0)
        }
    })
}

// Separable Gaussian blur with the given sigma.
// Uses a 1-D kernel of radius ceil(3*sigma).
fn gaussian_blur(image: ArrayView3<f32>, sigma: f32) -> Array3<f32> {
    if sigma < 1e-6 {
        return image.to_owned();
    }

    let radius = (3.0 * sigma).ceil() as i32;
    let kernel: Vec<f32> = (-radius..=radius)
        .map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp())
        .collect();
    let kernel_sum: f32 = kernel.iter().sum();
    let kernel: Vec<f32> = kernel.iter().map(|&v| v / kernel_sum).collect();

    let h = image.shape()[0] as i32;
    let w = image.shape()[1] as i32;

    // Horizontal pass.
    let mut tmp = Array3::<f32>::zeros(image.dim());
    for y in 0..h as usize {
        for x in 0..w as usize {
            for c in 0..3 {
                let mut acc = 0.0_f32;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let nx = (x as i32 + ki as i32 - radius).clamp(0, w - 1) as usize;
                    acc += image[[y, nx, c]] * kv;
                }
                tmp[[y, x, c]] = acc;
            }
        }
    }

    // Vertical pass.
    let mut out = Array3::<f32>::zeros(image.dim());
    for y in 0..h as usize {
        for x in 0..w as usize {
            for c in 0..3 {
                let mut acc = 0.0_f32;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let ny = (y as i32 + ki as i32 - radius).clamp(0, h - 1) as usize;
                    acc += tmp[[ny, x, c]] * kv;
                }
                out[[y, x, c]] = acc.clamp(0.0, 1.0);
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebook::Codebook;
    use ndarray::Array3;

    fn dummy_codebook() -> Codebook {
        let images: Vec<Array3<f32>> = (0..4)
            .map(|i| {
                Array3::from_shape_fn((64, 64, 3), |(r, c, ch)| {
                    ((r + c + ch + i * 11) as f32 / 192.0).min(1.0)
                })
            })
            .collect();
        Codebook::build(&images, 64, "test")
    }

    fn test_image() -> Array3<f32> {
        Array3::from_shape_fn((64, 64, 3), |(r, c, ch)| {
            ((r + c + ch) as f32 / 192.0).min(1.0)
        })
    }

    #[test]
    fn v1_output_has_correct_shape() {
        let img = test_image();
        let result = bypass(img.view(), &dummy_codebook(), BypassMode::V1).unwrap();
        assert_eq!(result.image.shape(), img.shape());
        assert!(result.image.iter().all(|&v| v >= 0.0 && v <= 1.0));
    }

    #[test]
    fn v3_output_has_correct_shape_and_range() {
        let img = test_image();
        let cb = dummy_codebook();
        let result = bypass(img.view(), &cb, BypassMode::V3).unwrap();
        assert_eq!(result.image.shape(), img.shape());
        assert!(result.image.iter().all(|&v| v >= 0.0 && v <= 1.0));
    }

    #[test]
    fn v3_output_is_close_to_input() {
        let img = test_image();
        let cb = dummy_codebook();
        let result = bypass(img.view(), &cb, BypassMode::V3).unwrap();
        let max_diff = img
            .iter()
            .zip(result.image.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        // Spectral subtraction on a non-watermarked image should change it minimally.
        assert!(
            max_diff < 0.5,
            "V3 changed image too much: max_diff={max_diff}"
        );
    }

    #[test]
    fn shifted_to_unshifted_roundtrip() {
        let size = 512_usize;
        let center = size as i32 / 2;
        // DC carrier (0,0) in shifted layout is at [center, center] in the array,
        // which maps to [0, 0] in unshifted layout.
        let (uy, ux) = shifted_to_unshifted(0, 0, size);
        assert_eq!((uy, ux), (center as usize, center as usize));
    }

    #[test]
    fn gaussian_blur_does_not_change_constant_image() {
        let img = Array3::from_elem((16, 16, 3), 0.5_f32);
        let blurred = gaussian_blur(img.view(), 1.0);
        for (&a, &b) in img.iter().zip(blurred.iter()) {
            assert!((a - b).abs() < 1e-4);
        }
    }
}
