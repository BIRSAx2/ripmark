//! SynthID watermark detection pipeline.
//!
//! Detection is based on two signals:
//!
//! 1. Phase match -- how closely the image's FFT phase at known carrier
//!    frequencies matches the reference phases stored in the codebook.
//!    Calibrated range: watermarked 0.92-0.99, non-watermarked 0.47-0.71.
//!
//! 2. CVR ratio -- carrier-to-random magnitude ratio in the noise residual.
//!    Provides a supporting signal, especially for dark images.
//!
//! Both signals are combined via sigmoid scoring with a 0.50 confidence
//! threshold.

use ndarray::ArrayView3;

use crate::carriers::{all_carriers, CARRIERS_DARK, CARRIERS_WHITE};
use crate::codebook::{Codebook, ResolutionProfile};
use crate::denoise::extract_noise_fused;
use crate::fft::{fft2, fftshift, resize_rgb, to_grayscale};

/// Which carrier set produced the best phase match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BestSet {
    Dark,
    White,
}

/// Full output of the detection pipeline.
#[derive(Debug, Clone)]
pub struct DetectionResult {
    pub is_watermarked: bool,
    /// Combined confidence score in [0, 1].
    pub confidence: f32,
    /// Best phase match across both carrier sets, in [0, 1].
    pub phase_match: f32,
    /// Which carrier set produced the best phase match.
    pub best_set: BestSet,
    /// Phase match for the dark carrier set specifically.
    pub dark_phase_match: f32,
    /// Phase match for the white carrier set specifically.
    pub white_phase_match: f32,
    /// Carrier-to-random magnitude ratio from the noise residual.
    pub cvr_ratio: f32,
    /// Intermediate sigmoid score for phase match.
    pub phase_score: f32,
    /// Intermediate sigmoid score for CVR ratio.
    pub cvr_score: f32,
    /// Correlation between the image noise residual and the reference noise.
    pub correlation: f32,
    /// Ratio of noise standard deviation to average absolute noise magnitude.
    pub structure_ratio: f32,
    /// Mean image-domain FFT magnitude at all known carrier bins.
    pub carrier_strength: f32,
    /// Placeholder for the Python extractor's multi-scale consistency metric.
    pub multi_scale_consistency: f32,
}

/// Detect a SynthID watermark in an image.
///
/// `image` must be RGB, values in [0, 1], shape (H, W, 3).
pub fn detect(image: ArrayView3<f32>, codebook: &Codebook) -> DetectionResult {
    let (profile, _) = codebook
        .best_profile(image.shape()[0], image.shape()[1])
        .expect("codebook has at least one profile");
    detect_with_profile(image, profile)
}

fn detect_with_profile(image: ArrayView3<f32>, profile: &ResolutionProfile) -> DetectionResult {
    let size = profile.image_size;
    let center = size as i32 / 2;

    let resized = resize_rgb(&image.to_owned(), size, size);

    // Compute shifted FFT of grayscale for phase extraction.
    let gray = to_grayscale(resized.view());
    let spectrum = fftshift(fft2(gray.view()).data);
    let img_phase = spectrum.mapv(|c| c.arg());
    let img_mag = spectrum.mapv(|c| c.norm());

    // Phase match for each carrier set.
    let dark_phase_match = phase_match(
        &img_phase,
        CARRIERS_DARK,
        &profile.dark.ref_phases,
        center,
        size,
    );
    let white_phase_match = phase_match(
        &img_phase,
        CARRIERS_WHITE,
        &profile.white.ref_phases,
        center,
        size,
    );

    let (best_phase_match, best_set) = if dark_phase_match >= white_phase_match {
        (dark_phase_match, BestSet::Dark)
    } else {
        (white_phase_match, BestSet::White)
    };

    // CVR ratio from the noise residual.
    let cvr_ratio = carrier_to_random_ratio(resized.view(), &all_carriers(), center, size);

    // Legacy reporting metrics for parity with the Python extractor.
    let noise = extract_noise_fused(resized.view());
    let reference_noise = profile.reference_noise_array();
    let correlation = pearson_correlation(
        noise.iter().copied().collect::<Vec<_>>().as_slice(),
        reference_noise
            .iter()
            .copied()
            .collect::<Vec<_>>()
            .as_slice(),
    );
    let noise_gray = crate::fft::to_grayscale(noise.view());
    let structure_ratio = structure_ratio(&noise_gray);
    let carrier_strength = mean_carrier_strength(&img_mag, &all_carriers(), center, size);
    let multi_scale_consistency = 0.0;

    // Sigmoid scoring. Positive argument = high value gives high score.
    // Threshold 0.78 sits between max non-watermarked (0.71) and min watermarked (0.92).
    let phase_score = sigmoid(20.0 * (best_phase_match - 0.78));
    // CVR threshold 2.0 separates watermarked from non-watermarked noise ratios.
    let cvr_score = sigmoid(2.0 * (cvr_ratio - 2.0));

    let confidence = (0.80 * phase_score + 0.20 * cvr_score).min(1.0);

    DetectionResult {
        is_watermarked: confidence > 0.50,
        confidence,
        phase_match: best_phase_match,
        best_set,
        dark_phase_match,
        white_phase_match,
        cvr_ratio,
        phase_score,
        cvr_score,
        correlation,
        structure_ratio,
        carrier_strength,
        multi_scale_consistency,
    }
}

// Compute the mean phase agreement between the image spectrum and reference phases.
//
// For each carrier at (fy, fx), the agreement is:
//   1 - |angle(exp(i * (img_phase - ref_phase)))| / pi
// which is 1.0 for a perfect match and 0.0 for a perfect anti-match.
fn phase_match(
    img_phase: &ndarray::Array2<f32>,
    carriers: &[(i32, i32)],
    ref_phases: &[f32],
    center: i32,
    size: usize,
) -> f32 {
    let mut matches = Vec::with_capacity(carriers.len());

    for (&(fy, fx), &ref_phase) in carriers.iter().zip(ref_phases.iter()) {
        let y = (fy + center) as usize;
        let x = (fx + center) as usize;
        if y < size && x < size {
            let diff = img_phase[[y, x]] - ref_phase;
            // Wrap to [-pi, pi] then normalise to [0, 1].
            let wrapped = (diff + std::f32::consts::PI).rem_euclid(2.0 * std::f32::consts::PI)
                - std::f32::consts::PI;
            matches.push(1.0 - wrapped.abs() / std::f32::consts::PI);
        }
    }

    if matches.is_empty() {
        0.0
    } else {
        matches.iter().sum::<f32>() / matches.len() as f32
    }
}

// Compute the ratio of mean carrier magnitude to mean random-bin magnitude
// in the noise residual. Elevated ratios indicate structured watermark energy
// at the known carrier positions.
fn carrier_to_random_ratio(
    image: ArrayView3<f32>,
    carriers: &[(i32, i32)],
    center: i32,
    size: usize,
) -> f32 {
    let noise = extract_noise_fused(image);
    let noise_gray = crate::fft::to_grayscale(noise.view());
    let noise_spectrum = fftshift(fft2(noise_gray.view()).data);
    let noise_mag = noise_spectrum.mapv(|c| c.norm());

    let carrier_mags: Vec<f32> = carriers
        .iter()
        .filter_map(|&(fy, fx)| {
            let y = (fy + center) as usize;
            let x = (fx + center) as usize;
            if y < size && x < size {
                Some(noise_mag[[y, x]])
            } else {
                None
            }
        })
        .collect();

    // Sample random bins using a fixed seed for reproducibility.
    let random_mags: Vec<f32> = random_bins(size, carriers.len() * 4)
        .into_iter()
        .map(|(ry, rx)| noise_mag[[ry, rx]])
        .collect();

    let mean_carrier = mean(&carrier_mags);
    let mean_random = mean(&random_mags);

    if mean_random < 1e-10 {
        0.0
    } else {
        mean_carrier / mean_random
    }
}

// Generate random (y, x) bin positions away from DC, deterministically.
fn random_bins(size: usize, n: usize) -> Vec<(usize, usize)> {
    let center = size / 2;
    let mut bins = Vec::with_capacity(n);
    let mut state: u64 = 0xdeadbeef_cafebabe;

    while bins.len() < n {
        // xorshift64
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let ry = 10 + (state >> 32) as usize % (size - 20);
        let rx = 10 + (state & 0xffffffff) as usize % (size - 20);
        // Skip DC neighbourhood.
        if ry.abs_diff(center) < 5 && rx.abs_diff(center) < 5 {
            continue;
        }
        bins.push((ry, rx));
    }
    bins
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f32>() / values.len() as f32
}

fn mean_carrier_strength(
    image_mag: &ndarray::Array2<f32>,
    carriers: &[(i32, i32)],
    center: i32,
    size: usize,
) -> f32 {
    let carrier_mags: Vec<f32> = carriers
        .iter()
        .filter_map(|&(fy, fx)| {
            let y = (fy + center) as usize;
            let x = (fx + center) as usize;
            if y < size && x < size {
                Some(image_mag[[y, x]])
            } else {
                None
            }
        })
        .collect();
    mean(&carrier_mags)
}

fn structure_ratio(noise_gray: &ndarray::Array2<f32>) -> f32 {
    let values: Vec<f32> = noise_gray.iter().copied().collect();
    let mean_abs = values.iter().map(|v| v.abs()).sum::<f32>() / values.len().max(1) as f32;
    let mean = values.iter().sum::<f32>() / values.len().max(1) as f32;
    let variance = values
        .iter()
        .map(|&v| {
            let d = v - mean;
            d * d
        })
        .sum::<f32>()
        / values.len().max(1) as f32;
    variance.sqrt() / (mean_abs + 1e-10)
}

fn pearson_correlation(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let n = a.len() as f32;
    let mean_a = a.iter().sum::<f32>() / n;
    let mean_b = b.iter().sum::<f32>() / n;

    let (mut num, mut den_a, mut den_b) = (0.0_f32, 0.0_f32, 0.0_f32);
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        let da = ai - mean_a;
        let db = bi - mean_b;
        num += da * db;
        den_a += da * da;
        den_b += db * db;
    }

    let denom = (den_a * den_b).sqrt();
    if denom < 1e-10 || !denom.is_finite() {
        0.0
    } else {
        let corr = num / denom;
        if corr.is_finite() {
            corr
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebook::Codebook;
    use ndarray::Array3;

    fn dummy_codebook(size: usize) -> Codebook {
        let images: Vec<Array3<f32>> = (0..8)
            .map(|i| {
                Array3::from_shape_fn((size, size, 3), |(r, c, ch)| {
                    ((r + c + ch + i * 13) as f32 / (size * 3) as f32).min(1.0)
                })
            })
            .collect();
        Codebook::build(&images, size, "test")
    }

    #[test]
    fn detect_returns_valid_confidence() {
        let cb = dummy_codebook(64);
        let image = Array3::from_shape_fn((64, 64, 3), |(r, c, ch)| {
            ((r + c + ch) as f32 / 192.0).min(1.0)
        });
        let result = detect(image.view(), &cb);
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        assert!(result.phase_match >= 0.0 && result.phase_match <= 1.0);
        assert!(result.cvr_ratio >= 0.0);
    }

    #[test]
    fn phase_score_above_threshold_gives_high_confidence() {
        // A phase match of 0.95 (well into watermarked range) should score high.
        let score = sigmoid(20.0 * (0.95_f32 - 0.78));
        assert!(
            score > 0.96,
            "expected high score for 0.95 match, got {score}"
        );
    }

    #[test]
    fn phase_score_below_threshold_gives_low_confidence() {
        // A phase match of 0.60 (non-watermarked range) should score low.
        let score = sigmoid(20.0 * (0.60_f32 - 0.78));
        assert!(
            score < 0.03,
            "expected low score for 0.60 match, got {score}"
        );
    }

    #[test]
    fn sigmoid_midpoint() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
    }
}
