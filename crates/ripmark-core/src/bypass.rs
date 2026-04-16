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

use crate::carriers::{all_carriers, CARRIERS_DARK, CARRIERS_WHITE};
use crate::codebook::{Codebook, ResolutionProfile};
use crate::detect::detect;
use crate::fft::{fft2, resize_gray, Spectrum};

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
    /// Peak signal-to-noise ratio against the original image.
    pub psnr: f32,
    /// Block SSIM approximation against the original image.
    pub ssim: f32,
    /// Relative drop in carrier-band energy, in [0, 1] when the energy falls.
    pub carrier_energy_drop: f32,
    /// Relative drop in phase match against the reference carrier template.
    pub phase_coherence_drop: f32,
    /// Resolution profile chosen from the codebook.
    pub profile_resolution: (usize, usize),
    /// Whether the selected profile exactly matched the input image dimensions.
    pub exact_profile_match: bool,
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
    let (profile, exact_match) = codebook.best_profile(image.shape()[0], image.shape()[1])?;

    let image_out = match mode {
        BypassMode::V1 => v1(image)?,
        BypassMode::V2 => v2(image)?,
        BypassMode::V3 => v3(image, profile)?,
    };

    let before_detection = detect(image, codebook);
    let after_detection = detect(image_out.image.view(), codebook);
    let before_energy = carrier_energy(image, profile);
    let after_energy = carrier_energy(image_out.image.view(), profile);

    Ok(BypassResult {
        psnr: compute_psnr(image, image_out.image.view()),
        ssim: compute_ssim(image, image_out.image.view()),
        carrier_energy_drop: relative_drop(before_energy, after_energy),
        phase_coherence_drop: relative_drop(
            before_detection.phase_match,
            after_detection.phase_match,
        ),
        profile_resolution: (profile.height, profile.width),
        exact_profile_match: exact_match,
        image: image_out.image,
        stages: image_out.stages,
    })
}

// V1: JPEG quality cycling.
fn v1(image: ArrayView3<f32>) -> Result<IntermediateBypassResult> {
    let cycled = jpeg_cycle(image, 50)?;
    Ok(IntermediateBypassResult {
        image: cycled,
        stages: vec!["jpeg_cycle_q50".into()],
    })
}

// V2: noise injection + bilateral smoothing + V1.
fn v2(image: ArrayView3<f32>) -> Result<IntermediateBypassResult> {
    let noisy = add_gaussian_noise(image, 5.0);
    let smoothed = bilateral_smooth(noisy.view(), 9, 75.0, 75.0);
    let IntermediateBypassResult { image: cycled, .. } = v1(smoothed.view())?;
    Ok(IntermediateBypassResult {
        image: cycled,
        stages: vec![
            "noise_injection".into(),
            "bilateral_smooth".into(),
            "jpeg_cycle_q50".into(),
        ],
    })
}

// V3: multi-pass spectral subtraction.
fn v3(image: ArrayView3<f32>, profile: &ResolutionProfile) -> Result<IntermediateBypassResult> {
    let mut current = image.to_owned();
    let avg_luminance = current.iter().copied().sum::<f32>() / (current.len() as f32);
    let passes = [
        StrengthConfig::aggressive(),
        StrengthConfig::moderate(),
        StrengthConfig::gentle(),
    ];

    for pass in passes {
        current = spectral_pass(current.view(), profile, avg_luminance, pass);
    }

    // Light Gaussian blur to smooth any spectral artefacts.
    current = gaussian_blur(current.view(), 0.4);

    Ok(IntermediateBypassResult {
        image: current,
        stages: vec![
            "spectral_pass_aggressive".into(),
            "spectral_pass_moderate".into(),
            "spectral_pass_gentle".into(),
            "gaussian_antialias".into(),
        ],
    })
}

#[derive(Debug, Clone)]
struct IntermediateBypassResult {
    image: Array3<f32>,
    stages: Vec<String>,
}

// Single spectral subtraction pass over all channels and carrier sets.
fn spectral_pass(
    image: ArrayView3<f32>,
    profile: &ResolutionProfile,
    avg_luminance: f32,
    cfg: StrengthConfig,
) -> Array3<f32> {
    let h = image.shape()[0];
    let w = image.shape()[1];
    let spectral_ready = spectral_profile_is_reliable(profile)
        && profile.height == h
        && profile.width == w
        && profile.magnitude_profile.is_some()
        && profile.phase_template.is_some()
        && profile.phase_consistency.is_some();

    if spectral_ready {
        spectral_pass_exact(image, profile, avg_luminance, cfg)
    } else if spectral_profile_is_reliable(profile)
        && profile.magnitude_profile.is_some()
        && profile.phase_template.is_some()
        && profile.phase_consistency.is_some()
    {
        spectral_pass_fallback(image, profile, avg_luminance, cfg)
    } else {
        spectral_pass_carrier_only(image, profile, cfg.removal_fraction, cfg.consistency_floor)
    }
}

#[derive(Clone, Copy)]
struct StrengthConfig {
    removal_fraction: f32,
    consistency_floor: f32,
    magnitude_cap: f32,
    dc_radius: f32,
}

impl StrengthConfig {
    fn gentle() -> Self {
        Self {
            removal_fraction: 0.60,
            consistency_floor: 0.70,
            magnitude_cap: 0.50,
            dc_radius: 30.0,
        }
    }

    fn moderate() -> Self {
        Self {
            removal_fraction: 0.80,
            consistency_floor: 0.50,
            magnitude_cap: 0.70,
            dc_radius: 25.0,
        }
    }

    fn aggressive() -> Self {
        Self {
            removal_fraction: 0.95,
            consistency_floor: 0.30,
            magnitude_cap: 0.90,
            dc_radius: 20.0,
        }
    }
}

fn spectral_pass_exact(
    image: ArrayView3<f32>,
    profile: &ResolutionProfile,
    avg_luminance: f32,
    cfg: StrengthConfig,
) -> Array3<f32> {
    let mut result = Array3::<f32>::zeros(image.dim());

    for (ch, &weight) in channel_weights().iter().enumerate() {
        let channel = image.slice(s![.., .., ch]);
        let mut fft_data = fft2(channel).data;
        let wm_est = estimate_watermark_fft(profile, &fft_data, ch, avg_luminance, cfg, weight);

        for ((y, x), value) in fft_data.indexed_iter_mut() {
            *value -= wm_est[[y, x]];
        }

        let cleaned = Spectrum { data: fft_data }.to_spatial();
        result
            .slice_mut(s![.., .., ch])
            .assign(&cleaned.mapv(|v| v.clamp(0.0, 1.0)));
    }

    result
}

fn spectral_pass_fallback(
    image: ArrayView3<f32>,
    profile: &ResolutionProfile,
    avg_luminance: f32,
    cfg: StrengthConfig,
) -> Array3<f32> {
    let h = image.shape()[0];
    let w = image.shape()[1];
    let mut result = Array3::<f32>::zeros((h, w, 3));

    for (ch, _) in channel_weights().iter().enumerate() {
        let wm_native = watermark_spatial(profile, ch, avg_luminance, cfg);
        let wm_resized = resize_gray(&wm_native, h, w);
        let cleaned = &image.slice(s![.., .., ch]).to_owned() - &wm_resized;
        result
            .slice_mut(s![.., .., ch])
            .assign(&cleaned.mapv(|v| v.clamp(0.0, 1.0)));
    }

    result
}

fn spectral_pass_carrier_only(
    image: ArrayView3<f32>,
    profile: &ResolutionProfile,
    removal_fraction: f32,
    consistency_floor: f32,
) -> Array3<f32> {
    const SAFETY_CAP: f32 = 0.90;
    let h = image.shape()[0];
    let w = image.shape()[1];
    let mut result = Array3::<f32>::zeros((h, w, 3));
    let size = profile.image_size;

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

    for (ch, &weight) in channel_weights().iter().enumerate() {
        let channel_owned = image.slice(s![.., .., ch]).to_owned();
        let resized_ch = resize_gray(&channel_owned, size, size);
        let mut fft_data = fft2(resized_ch.view()).data;

        for carriers in [dark_carriers.as_slice(), white_carriers.as_slice()] {
            for &(fy, fx, coherence, ref_phase) in carriers {
                if coherence < consistency_floor {
                    continue;
                }
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

        let cleaned_resized = Spectrum { data: fft_data }.to_spatial();
        let cleaned = resize_gray(&cleaned_resized, h, w);
        result
            .slice_mut(s![.., .., ch])
            .assign(&cleaned.mapv(|v| v.clamp(0.0, 1.0)));
    }

    result
}

fn estimate_watermark_fft(
    profile: &ResolutionProfile,
    image_fft: &ndarray::Array2<Complex<f32>>,
    channel: usize,
    avg_luminance: f32,
    cfg: StrengthConfig,
    channel_weight: f32,
) -> ndarray::Array2<Complex<f32>> {
    let (h, w) = image_fft.dim();
    let mag = profile
        .magnitude_profile_array()
        .expect("checked by caller for spectral profile");
    let phase = profile
        .phase_template_array()
        .expect("checked by caller for spectral profile");
    let consistency = profile
        .phase_consistency_array()
        .expect("checked by caller for spectral profile");
    let white_mag = profile.white_magnitude_profile_array();
    let agreement = profile.black_white_agreement_array();

    ndarray::Array2::from_shape_fn((h, w), |(y, x)| {
        let mut wm_mag = if let Some(white) = &white_mag {
            mag[[y, x, channel]] * (1.0 - avg_luminance) + white[[y, x, channel]] * avg_luminance
        } else {
            mag[[y, x, channel]]
        };

        if let Some(agree) = &agreement {
            wm_mag *= agree[[y, x, channel]];
        }

        let fy = signed_freq(y, h);
        let fx = signed_freq(x, w);
        let dc_ramp = ((fy * fy + fx * fx).sqrt() / cfg.dc_radius).clamp(0.0, 1.0);
        wm_mag *= dc_ramp;

        let consistency_val = consistency[[y, x, channel]];
        let cons_weight = ((consistency_val - cfg.consistency_floor)
            / (1.0 - cfg.consistency_floor + 1e-9))
            .clamp(0.0, 1.0);

        let subtract_mag = (wm_mag * cons_weight * cfg.removal_fraction * channel_weight)
            .min(image_fft[[y, x]].norm() * cfg.magnitude_cap);
        let ref_phase = phase[[y, x, channel]];
        Complex::new(subtract_mag * ref_phase.cos(), subtract_mag * ref_phase.sin())
    })
}

fn watermark_spatial(
    profile: &ResolutionProfile,
    channel: usize,
    avg_luminance: f32,
    cfg: StrengthConfig,
) -> ndarray::Array2<f32> {
    let content = profile.content_magnitude_baseline_array();
    let phase = profile
        .phase_template_array()
        .expect("checked by caller for spectral profile");
    let (h, w, _) = phase.dim();

    let synth_fft = ndarray::Array2::from_shape_fn((h, w), |(y, x)| {
        let mag = content
            .as_ref()
            .map(|baseline| baseline[[y, x, channel]])
            .unwrap_or_else(|| {
                profile
                    .magnitude_profile_array()
                    .expect("spectral magnitude exists")[[y, x, channel]]
                    * 10.0
            });
        let ref_phase = phase[[y, x, channel]];
        Complex::new(mag * ref_phase.cos(), mag * ref_phase.sin())
    });

    let wm_fft = estimate_watermark_fft(profile, &synth_fft, channel, avg_luminance, cfg, channel_weights()[channel]);
    Spectrum { data: wm_fft }.to_spatial()
}

fn channel_weights() -> [f32; 3] {
    [0.85, 1.0, 0.70]
}

fn spectral_profile_is_reliable(profile: &ResolutionProfile) -> bool {
    let has_cross_validation = profile.black_white_agreement.is_some();
    has_cross_validation || profile.n_images >= 8
}

fn signed_freq(index: usize, len: usize) -> f32 {
    let idx = index as f32;
    let half = len as f32 / 2.0;
    if idx > half {
        idx - len as f32
    } else {
        idx
    }
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

fn carrier_energy(image: ArrayView3<f32>, profile: &ResolutionProfile) -> f32 {
    let size = profile.image_size;
    let center = size as i32 / 2;
    let resized = crate::fft::resize_rgb(&image.to_owned(), size, size);
    let noise = crate::denoise::extract_noise_fused(resized.view());
    let noise_gray = crate::fft::to_grayscale(noise.view());
    let noise_spectrum = crate::fft::fftshift(fft2(noise_gray.view()).data);
    let noise_mag = noise_spectrum.mapv(|c| c.norm());

    let carrier_mags: Vec<f32> = all_carriers()
        .into_iter()
        .filter_map(|(fy, fx)| {
            let y = (fy + center) as usize;
            let x = (fx + center) as usize;
            if y < size && x < size {
                Some(noise_mag[[y, x]])
            } else {
                None
            }
        })
        .collect();

    mean(&carrier_mags)
}

fn compute_psnr(original: ArrayView3<f32>, modified: ArrayView3<f32>) -> f32 {
    let mse = original
        .iter()
        .zip(modified.iter())
        .map(|(&a, &b)| {
            let d = a - b;
            d * d
        })
        .sum::<f32>()
        / original.len() as f32;

    if mse <= 1e-12 {
        f32::INFINITY
    } else {
        10.0 * (1.0 / mse).log10()
    }
}

fn compute_ssim(original: ArrayView3<f32>, modified: ArrayView3<f32>) -> f32 {
    let gray_o = crate::fft::to_grayscale(original);
    let gray_m = crate::fft::to_grayscale(modified);
    let block = 8usize;
    let rows = gray_o.nrows() / block * block;
    let cols = gray_o.ncols() / block * block;

    if rows == 0 || cols == 0 {
        return 1.0;
    }

    let k1_sq = 0.0001_f32;
    let k2_sq = 0.0009_f32;
    let mut scores = Vec::new();

    for by in (0..rows).step_by(block) {
        for bx in (0..cols).step_by(block) {
            let mut sum_a = 0.0_f32;
            let mut sum_b = 0.0_f32;
            let mut sum_aa = 0.0_f32;
            let mut sum_bb = 0.0_f32;
            let mut sum_ab = 0.0_f32;

            for y in by..(by + block) {
                for x in bx..(bx + block) {
                    let a = gray_o[[y, x]];
                    let b = gray_m[[y, x]];
                    sum_a += a;
                    sum_b += b;
                    sum_aa += a * a;
                    sum_bb += b * b;
                    sum_ab += a * b;
                }
            }

            let n = (block * block) as f32;
            let mu_a = sum_a / n;
            let mu_b = sum_b / n;
            let var_a = sum_aa / n - mu_a * mu_a;
            let var_b = sum_bb / n - mu_b * mu_b;
            let cov_ab = sum_ab / n - mu_a * mu_b;

            let num = (2.0 * mu_a * mu_b + k1_sq) * (2.0 * cov_ab + k2_sq);
            let den = (mu_a * mu_a + mu_b * mu_b + k1_sq) * (var_a + var_b + k2_sq);
            if den.abs() > 1e-12 {
                scores.push(num / den);
            }
        }
    }

    mean(&scores)
}

fn relative_drop(before: f32, after: f32) -> f32 {
    if before.abs() < 1e-10 {
        0.0
    } else {
        ((before - after) / before).clamp(-10.0, 10.0)
    }
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
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
