//! Noise extraction via denoising.
//!
//! The SynthID watermark lives in the noise residual: `noise = image − denoised`.
//!
//! Implemented methods (in priority order):
//!   1. Wiener filter (FFT-domain)        — fast, good for coloured noise
//!   2. Haar wavelet soft-threshold       — good for local structure
//!   3. Fused extractor                   — weighted average of both

use ndarray::{s, Array2, Array3, ArrayView2, ArrayView3};
use crate::fft::{fft2, Spectrum};

// ─── Wiener filter ───────────────────────────────────────────────────────────

/// Denoise a single grayscale channel using a Wiener filter in the FFT domain.
///
/// Equivalent to the Python implementation:
/// ```python
/// f = fft2(image)
/// power = np.abs(f) ** 2
/// signal_power = np.maximum(power - noise_variance, 0)
/// wiener_ratio = signal_power / (signal_power + noise_variance + 1e-10)
/// denoised = np.real(ifft2(f * wiener_ratio))
/// ```
pub fn wiener_channel(channel: ArrayView2<f32>) -> Array2<f32> {
    let spectrum = fft2(channel);

    // Estimate noise variance from median power (robust to outliers).
    // High-energy DC bins would skew a mean estimate, so median is better.
    let mut powers: Vec<f32> = spectrum.data.iter().map(|c| c.norm_sqr()).collect();
    powers.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let noise_var = powers[powers.len() / 2];

    // Apply Wiener transfer function: H = S_signal / (S_signal + σ²)
    let filtered_data = spectrum.data.mapv(|c| {
        let p = c.norm_sqr();
        let signal_power = (p - noise_var).max(0.0);
        let ratio = signal_power / (signal_power + noise_var + 1e-10);
        c * ratio
    });

    Spectrum { data: filtered_data }.to_spatial()
}

/// Denoise each RGB channel independently with the Wiener filter.
pub fn wiener_rgb(image: ArrayView3<f32>) -> Array3<f32> {
    let h = image.shape()[0];
    let w = image.shape()[1];
    let mut out = Array3::<f32>::zeros((h, w, 3));
    for c in 0..3 {
        let ch = image.slice(s![.., .., c]);
        out.slice_mut(s![.., .., c]).assign(&wiener_channel(ch));
    }
    out
}

// ─── Haar wavelet soft-threshold ─────────────────────────────────────────────

/// Forward 1-D Haar DWT.
/// Returns `(approximation, detail)`, each of length `n/2`.
/// If `n` is odd, the last sample is appended to approximation with a zero detail.
fn haar1d_forward(data: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n = data.len();
    let half = n / 2;
    let mut approx = Vec::with_capacity(half + (n & 1));
    let mut detail = Vec::with_capacity(half + (n & 1));

    for i in 0..half {
        approx.push((data[2 * i] + data[2 * i + 1]) * 0.5);
        detail.push((data[2 * i] - data[2 * i + 1]) * 0.5);
    }
    if n & 1 == 1 {
        approx.push(data[n - 1]);
        detail.push(0.0);
    }
    (approx, detail)
}

/// Inverse 1-D Haar DWT.
fn haar1d_inverse(approx: &[f32], detail: &[f32]) -> Vec<f32> {
    assert_eq!(approx.len(), detail.len());
    let n = approx.len();
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        out.push(approx[i] + detail[i]); // even sample
        out.push(approx[i] - detail[i]); // odd sample
    }
    out
}

/// Forward single-level 2-D Haar DWT.
///
/// Returns an H×W array with four subbands:
/// ```text
/// ┌──────┬──────┐
/// │  LL  │  LH  │
/// ├──────┼──────┤
/// │  HL  │  HH  │
/// └──────┴──────┘
/// ```
/// where hh = H/2, hw = W/2.
fn haar2d_forward(img: ArrayView2<f32>) -> (Array2<f32>, usize, usize) {
    let h = img.nrows();
    let w = img.ncols();
    let hh = h / 2 + (h & 1);
    let hw = w / 2 + (w & 1);

    let mut temp = Array2::<f32>::zeros((h, w));

    // Transform rows
    for r in 0..h {
        let row: Vec<f32> = img.slice(s![r, ..]).to_vec();
        let (a, d) = haar1d_forward(&row);
        for (c, &v) in a.iter().enumerate() { temp[[r, c]]      = v; }
        for (c, &v) in d.iter().enumerate() { temp[[r, c + hw]] = v; }
    }

    let mut result = Array2::<f32>::zeros((h, w));

    // Transform columns of both halves
    for c in 0..w {
        let col: Vec<f32> = temp.slice(s![.., c]).to_vec();
        let (a, d) = haar1d_forward(&col);
        for (r, &v) in a.iter().enumerate() { result[[r,      c]] = v; }
        for (r, &v) in d.iter().enumerate() { result[[r + hh, c]] = v; }
    }

    (result, hh, hw)
}

/// Inverse single-level 2-D Haar DWT.
fn haar2d_inverse(coeffs: &Array2<f32>, hh: usize, hw: usize) -> Array2<f32> {
    let h = coeffs.nrows();
    let w = coeffs.ncols();

    let mut temp = Array2::<f32>::zeros((h, w));

    // Invert column transform
    for c in 0..w {
        let approx: Vec<f32> = coeffs.slice(s![..hh, c]).to_vec();
        let detail: Vec<f32> = coeffs.slice(s![hh.., c]).to_vec();
        let rec = haar1d_inverse(&approx, &detail);
        for (r, v) in rec.into_iter().enumerate() {
            if r < h { temp[[r, c]] = v; }
        }
    }

    let mut result = Array2::<f32>::zeros((h, w));

    // Invert row transform
    for r in 0..h {
        let approx: Vec<f32> = temp.slice(s![r, ..hw]).to_vec();
        let detail: Vec<f32> = temp.slice(s![r, hw..]).to_vec();
        let rec = haar1d_inverse(&approx, &detail);
        for (c, v) in rec.into_iter().enumerate() {
            if c < w { result[[r, c]] = v; }
        }
    }

    result
}

/// Soft threshold: `sign(x) * max(|x| - λ, 0)`.
#[inline]
fn soft_threshold(x: f32, lambda: f32) -> f32 {
    let ax = x.abs();
    if ax <= lambda { 0.0 } else { x.signum() * (ax - lambda) }
}

/// Denoise a single grayscale channel using Haar wavelet soft-thresholding.
///
/// Matches the Python `wavelet_denoise` logic:
/// - Estimate noise σ from finest detail coefficients via MAD estimator
/// - Threshold = σ × √(2 × ln(N))
/// - Apply soft thresholding to all detail coefficients (not approximation)
pub fn wavelet_channel(channel: ArrayView2<f32>, levels: usize) -> Array2<f32> {
    let mut coeffs = channel.to_owned();
    let mut level_meta: Vec<(usize, usize, usize, usize)> = Vec::new(); // (hh, hw, h, w)

    // Forward multi-level DWT (applied to LL subband each time)
    for _ in 0..levels {
        let h = coeffs.nrows();
        let w = coeffs.ncols();
        if h < 4 || w < 4 { break; }

        let ll_view = coeffs.slice(s![..h, ..w]).to_owned();
        let (new_coeffs, hh, hw) = haar2d_forward(ll_view.view());
        level_meta.push((hh, hw, h, w));
        coeffs = new_coeffs;
    }

    // Estimate noise σ from the finest detail (HH subband of deepest level)
    let sigma = if let Some(&(hh, hw, h, w)) = level_meta.last() {
        let hh_band: Vec<f32> = coeffs
            .slice(s![hh..h, hw..w])
            .iter()
            .map(|&v| v.abs())
            .collect();
        let median = median_f32(&hh_band);
        median / 0.6745
    } else {
        0.01 // fallback
    };

    // Universal threshold: σ√(2·ln(N))
    let n = channel.len() as f32;
    let threshold = sigma * (2.0_f32 * n.ln()).sqrt();

    // Soft-threshold all detail subbands at every level
    for &(hh, hw, h, w) in &level_meta {
        // LH subband: [0..hh, hw..w]
        for r in 0..hh {
            for c in hw..w {
                coeffs[[r, c]] = soft_threshold(coeffs[[r, c]], threshold);
            }
        }
        // HL subband: [hh..h, 0..hw]
        for r in hh..h {
            for c in 0..hw {
                coeffs[[r, c]] = soft_threshold(coeffs[[r, c]], threshold);
            }
        }
        // HH subband: [hh..h, hw..w]
        for r in hh..h {
            for c in hw..w {
                coeffs[[r, c]] = soft_threshold(coeffs[[r, c]], threshold);
            }
        }
    }

    // Inverse multi-level DWT
    for &(hh, hw, h, w) in level_meta.iter().rev() {
        let block = coeffs.slice(s![..h, ..w]).to_owned();
        let rec = haar2d_inverse(&block, hh, hw);
        coeffs.slice_mut(s![..h, ..w]).assign(&rec);
    }

    coeffs
}

/// Denoise each RGB channel independently with Haar wavelet soft-threshold.
pub fn wavelet_rgb(image: ArrayView3<f32>, levels: usize) -> Array3<f32> {
    let h = image.shape()[0];
    let w = image.shape()[1];
    let mut out = Array3::<f32>::zeros((h, w, 3));
    for c in 0..3 {
        let ch = image.slice(s![.., .., c]);
        out.slice_mut(s![.., .., c]).assign(&wavelet_channel(ch, levels));
    }
    out
}

// ─── fused denoiser ──────────────────────────────────────────────────────────

/// Extract noise from an RGB image using a fused denoiser.
///
/// Returns `image − denoised` (the noise residual that contains the watermark).
///
/// Weights (matching Python's fused extractor, simplified to two methods):
///   - Haar wavelet (3 levels): 1.0
///   - Wiener filter:           0.6
pub fn extract_noise_fused(image: ArrayView3<f32>) -> Array3<f32> {
    const W_WAVELET: f32 = 1.0;
    const W_WIENER:  f32 = 0.6;
    const TOTAL: f32     = W_WAVELET + W_WIENER;

    let denoised_wav = wavelet_rgb(image, 3);
    let denoised_wie = wiener_rgb(image);

    // Weighted fused denoised image
    let fused_denoised = Array3::from_shape_fn(image.dim(), |(r, c, ch)| {
        (denoised_wav[[r, c, ch]] * W_WAVELET
            + denoised_wie[[r, c, ch]] * W_WIENER)
            / TOTAL
    });

    // Noise residual
    Array3::from_shape_fn(image.dim(), |(r, c, ch)| {
        image[[r, c, ch]] - fused_denoised[[r, c, ch]]
    })
}

/// Extract noise using only the wavelet denoiser (faster, less accurate fusion).
pub fn extract_noise_wavelet(image: ArrayView3<f32>, levels: usize) -> Array3<f32> {
    let denoised = wavelet_rgb(image, levels);
    Array3::from_shape_fn(image.dim(), |(r, c, ch)| {
        image[[r, c, ch]] - denoised[[r, c, ch]]
    })
}

// ─── internal helpers ─────────────────────────────────────────────────────────

fn median_f32(data: &[f32]) -> f32 {
    if data.is_empty() { return 0.0; }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        (sorted[mid - 1] + sorted[mid]) * 0.5
    }
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    /// Denoising a clean image should return something close to the original.
    #[test]
    fn wiener_preserves_smooth_image() {
        let img = Array2::from_shape_fn((32, 32), |(r, c)| {
            ((r + c) as f32 / 64.0).min(1.0)
        });
        let denoised = wiener_channel(img.view());
        let max_err = img.iter().zip(denoised.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        // Smooth image: denoiser should not distort it much
        assert!(max_err < 0.2, "Wiener distorted smooth image: max_err={max_err}");
    }

    /// Haar DWT round-trip must be lossless.
    #[test]
    fn haar2d_roundtrip() {
        let input = Array2::from_shape_fn((16, 16), |(r, c)| (r * 16 + c) as f32 / 256.0);
        let (coeffs, hh, hw) = haar2d_forward(input.view());
        let recovered = haar2d_inverse(&coeffs, hh, hw);
        for (a, b) in input.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-5, "Haar round-trip error: {a} vs {b}");
        }
    }

    /// Haar DWT of a constant image has zero detail coefficients.
    #[test]
    fn haar2d_constant_has_zero_details() {
        let input = Array2::from_elem((8, 8), 0.5_f32);
        let (coeffs, hh, hw) = haar2d_forward(input.view());
        let h = coeffs.nrows();
        let w = coeffs.ncols();
        // HH detail band
        for r in hh..h {
            for c in hw..w {
                assert!(coeffs[[r, c]].abs() < 1e-6,
                    "Expected zero detail, got {}", coeffs[[r, c]]);
            }
        }
    }

    /// Noise residual of a clean image should be small.
    #[test]
    fn noise_residual_of_smooth_image_is_small() {
        let img = Array3::from_shape_fn((32, 32, 3), |(r, c, ch)| {
            ((r + c + ch) as f32 / 96.0).min(1.0)
        });
        let noise = extract_noise_fused(img.view());
        let rms = (noise.iter().map(|&v| v * v).sum::<f32>() / noise.len() as f32).sqrt();
        assert!(rms < 0.05, "Noise residual too large for smooth image: rms={rms}");
    }

    #[test]
    fn median_f32_correct() {
        assert_eq!(median_f32(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median_f32(&[4.0, 1.0, 3.0, 2.0]), 2.5);
        assert_eq!(median_f32(&[]), 0.0);
    }
}
