//! Research-oriented analysis tools for discovering and visualising the
//! SynthID watermark structure.
//!
//! The main entry point is `analyze_image_set`, which produces a full
//! phase-coherence map across the spectrum and a ranked list of candidate
//! carrier frequencies.

use ndarray::{Array1, Array2, Array3, ArrayView3};

use crate::carriers::{detect_carriers, CarrierInfo};
use crate::denoise::extract_noise_fused;
use crate::fft::{fft2, fftshift, fftshift_real, resize_rgb, to_grayscale};

/// Output of a full-spectrum analysis run.
#[derive(Debug, Clone)]
pub struct AnalysisReport {
    /// Average FFT magnitude across all images at the reference scale.
    /// Shape: (image_size, image_size), zero-frequency at centre.
    pub magnitude_spectrum: Array2<f32>,

    /// Phase coherence at every frequency bin.
    /// Values near 1.0 are candidate watermark carriers.
    /// Shape: (image_size, image_size), zero-frequency at centre.
    pub phase_coherence_map: Array2<f32>,

    /// Candidate carriers ranked by coherence * log-magnitude score.
    pub top_carriers: Vec<CarrierInfo>,

    /// Dominant watermark component extracted by PCA on the noise residuals.
    /// Shape: (image_size, image_size).
    pub pca_watermark: Array2<f32>,

    /// Reference scale at which the analysis was performed.
    pub image_size: usize,
}

/// Analyse a set of images and return the full watermark fingerprint report.
///
/// `images` must be RGB, values in [0, 1]. All images are resized to
/// `image_size x image_size` before processing.
/// `scales` controls which resolutions are used for multi-scale carrier voting.
pub fn analyze_image_set(
    images: &[Array3<f32>],
    image_size: usize,
    scales: &[usize],
) -> AnalysisReport {
    assert!(!images.is_empty(), "need at least one image");

    let resized: Vec<Array3<f32>> = images
        .iter()
        .map(|img| resize_rgb(img, image_size, image_size))
        .collect();

    let (magnitude_spectrum, phase_coherence_map) =
        full_spectrum_stats(&resized, image_size);

    let views: Vec<ArrayView3<f32>> = resized.iter().map(|a| a.view()).collect();
    let top_carriers = detect_carriers(&views, scales, 100);

    let pca_watermark = pca_noise_component(&resized, image_size);

    AnalysisReport {
        magnitude_spectrum,
        phase_coherence_map,
        top_carriers,
        pca_watermark,
        image_size,
    }
}

// Compute per-bin average magnitude and phase coherence across all images.
// Both returned arrays are in fftshift layout (zero-frequency at centre).
fn full_spectrum_stats(
    images: &[Array3<f32>],
    size: usize,
) -> (Array2<f32>, Array2<f32>) {
    use rustfft::num_complex::Complex;

    let n = images.len() as f32;
    let mut mag_sum   = Array2::<f32>::zeros((size, size));
    let mut phase_vec = Array2::<Complex<f32>>::zeros((size, size));

    for img in images {
        let gray = to_grayscale(img.view());
        let spectrum = fftshift(fft2(gray.view()).data);

        for ((r, c), v) in spectrum.indexed_iter() {
            mag_sum[[r, c]] += v.norm();
            let phi = v.arg();
            phase_vec[[r, c]] += Complex::new(phi.cos(), phi.sin());
        }
    }

    let avg_magnitude  = mag_sum.mapv(|v| v / n);
    let phase_coherence = phase_vec.mapv(|v| v.norm() / n);

    (avg_magnitude, phase_coherence)
}

// Extract the most consistent noise component across images using PCA.
// Returns the leading principal component reshaped to (size, size).
fn pca_noise_component(images: &[Array3<f32>], size: usize) -> Array2<f32> {
    // Limit to 50 images for performance.
    let sample: &[Array3<f32>] = &images[..images.len().min(50)];
    let n = sample.len();
    let d = size * size; // grayscale feature dimension

    // Build data matrix X: rows are noise residuals (n x d).
    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(n);
    for img in sample {
        let noise = extract_noise_fused(img.view());
        let gray = to_grayscale(noise.view());
        // Mean-centre this sample.
        let mean = gray.iter().sum::<f32>() / d as f32;
        rows.push(gray.iter().map(|&v| v - mean).collect());
    }

    // Gram matrix G = X X^T  (n x n, tractable when n << d).
    let mut gram = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in i..n {
            let dot: f32 = rows[i].iter().zip(rows[j].iter()).map(|(a, b)| a * b).sum();
            gram[i * n + j] = dot;
            gram[j * n + i] = dot;
        }
    }

    // Leading eigenvector of G via power iteration.
    let alpha = power_iteration(&gram, n, 100);

    // Reconstruct the leading principal component in feature space: v = X^T alpha.
    let mut component = vec![0.0_f32; d];
    for (i, row) in rows.iter().enumerate() {
        let a = alpha[i];
        for (j, &x) in row.iter().enumerate() {
            component[j] += a * x;
        }
    }

    // Normalise.
    let norm = component.iter().map(|&v| v * v).sum::<f32>().sqrt().max(1e-10);
    let component: Vec<f32> = component.iter().map(|&v| v / norm).collect();

    Array2::from_shape_vec((size, size), component)
        .expect("shape matches size*size")
}

// Power iteration to find the leading eigenvector of a symmetric n x n matrix.
fn power_iteration(matrix: &[f32], n: usize, iters: usize) -> Vec<f32> {
    // Start from a non-trivial vector.
    let mut v: Vec<f32> = (0..n).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();

    for _ in 0..iters {
        // w = M v
        let mut w = vec![0.0_f32; n];
        for i in 0..n {
            for j in 0..n {
                w[i] += matrix[i * n + j] * v[j];
            }
        }
        // Normalise.
        let norm = w.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-10);
        for i in 0..n { v[i] = w[i] / norm; }
    }

    v
}

/// Identify the top-N frequency bins by phase coherence, excluding DC.
/// Useful for quickly finding candidate carriers without running full detection.
pub fn top_coherent_bins(
    coherence_map: &Array2<f32>,
    magnitude_map: &Array2<f32>,
    n: usize,
    dc_exclusion_radius: i32,
) -> Vec<((i32, i32), f32)> {
    let h = coherence_map.nrows() as i32;
    let w = coherence_map.ncols() as i32;
    let cy = h / 2;
    let cx = w / 2;

    let mut candidates: Vec<((i32, i32), f32)> = coherence_map
        .indexed_iter()
        .filter(|&((r, c), _)| {
            let dy = r as i32 - cy;
            let dx = c as i32 - cx;
            dy * dy + dx * dx > dc_exclusion_radius * dc_exclusion_radius
        })
        .map(|((r, c), &coh)| {
            let score = coh * magnitude_map[[r, c]].ln_1p();
            ((r as i32 - cy, c as i32 - cx), score)
        })
        .collect();

    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(n);
    candidates
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    fn dummy_images(n: usize, size: usize) -> Vec<Array3<f32>> {
        (0..n)
            .map(|i| {
                Array3::from_shape_fn((size, size, 3), |(r, c, ch)| {
                    ((r + c + ch + i * 7) as f32 / (size * 3) as f32).min(1.0)
                })
            })
            .collect()
    }

    #[test]
    fn analyze_produces_correct_shapes() {
        let images = dummy_images(4, 32);
        let report = analyze_image_set(&images, 32, &[32]);
        assert_eq!(report.magnitude_spectrum.shape(), &[32, 32]);
        assert_eq!(report.phase_coherence_map.shape(), &[32, 32]);
        assert_eq!(report.pca_watermark.shape(), &[32, 32]);
        assert!(!report.top_carriers.is_empty());
    }

    #[test]
    fn coherence_map_values_in_range() {
        let images = dummy_images(4, 32);
        let report = analyze_image_set(&images, 32, &[32]);
        for &v in report.phase_coherence_map.iter() {
            assert!(v >= 0.0 && v <= 1.0 + 1e-5, "coherence out of range: {v}");
        }
    }

    #[test]
    fn power_iteration_finds_dominant_direction() {
        // Matrix with clear leading eigenvector along [1, 0, 0].
        let matrix = vec![
            9.0_f32, 0.0, 0.0,
            0.0,     4.0, 0.0,
            0.0,     0.0, 1.0,
        ];
        let v = power_iteration(&matrix, 3, 50);
        assert!(v[0].abs() > 0.99, "expected leading direction [1,0,0], got {v:?}");
    }

    #[test]
    fn top_coherent_bins_returns_n_results() {
        let images = dummy_images(4, 32);
        let report = analyze_image_set(&images, 32, &[32]);
        let bins = top_coherent_bins(&report.phase_coherence_map, &report.magnitude_spectrum, 10, 2);
        assert_eq!(bins.len(), 10);
    }
}
