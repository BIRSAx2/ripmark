//! SynthID carrier frequency tables and data-driven carrier detection.
//!
//! Carrier frequencies were empirically verified from 291 Gemini-generated
//! images (black, white, nb_pro sets) at 512x512 resolution.
//! Each set has >0.95 intra-set phase coherence.

use ndarray::{Array2, ArrayView3};
use rustfft::num_complex::Complex;

use crate::fft::{fft2, fftshift, resize_gray, to_grayscale};

/// Dark-image carriers: diagonal grid pattern.
/// Extracted from black and nb_pro Gemini images at 512x512.
pub const CARRIERS_DARK: &[(i32, i32)] = &[
    (-5, -3), (5,  3), (-5,  3), (5, -3),
    (-3, -4), (3,  4), (-3,  4), (3, -4),
    (-4, -3), (4,  3), (-4,  3), (4, -3),
    (-5, -1), (5,  1), (-5,  1), (5, -1),
    (-5, -2), (5,  2), (-5,  2), (5, -2),
    (-2, -5), (2,  5), (-2,  5), (2, -5),
    (-1, -5), (1,  5), (-1,  5), (1, -5),
    (-4, -4), (4,  4), (-4,  4), (4, -4),
    (-1, -6), (1,  6), (-3, -5), (3,  5),
];

/// White-image carriers: horizontal axis.
/// Extracted from white Gemini images at 512x512.
pub const CARRIERS_WHITE: &[(i32, i32)] = &[
    (0, -7),  (0,  7),  (0, -8),  (0,  8),
    (0, -9),  (0,  9),  (0, -10), (0, 10),
    (0, -11), (0, 11),  (0, -12), (0, 12),
    (0, -20), (0, 20),  (0, -21), (0, 21),
    (0, -22), (0, 22),  (0, -23), (0, 23),
];

/// Union of both carrier sets (used when image type is unknown).
pub fn all_carriers() -> Vec<(i32, i32)> {
    CARRIERS_DARK.iter().chain(CARRIERS_WHITE.iter()).copied().collect()
}

/// Which carrier set an image belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CarrierSet {
    Dark,
    White,
}

/// A carrier frequency with its associated signal statistics.
#[derive(Debug, Clone)]
pub struct CarrierInfo {
    /// Frequency offset from DC in (fy, fx) at the 512px reference scale.
    pub freq: (i32, i32),
    pub magnitude: f32,
    pub phase: f32,
    /// Phase coherence in [0, 1]. 1.0 means phase is identical across all
    /// images (fixed watermark signal). 0.0 means phase is random (noise).
    pub coherence: f32,
    /// Number of scale levels at which this carrier was independently detected.
    pub votes: u32,
    pub score: f32,
}

/// Per-carrier reference phases for one carrier set.
#[derive(Debug, Clone)]
pub struct CarrierRefs {
    pub set: CarrierSet,
    /// Reference phase for each carrier, in the same order as the carrier table.
    pub phases: Vec<f32>,
}

/// Compute phase coherence at each carrier position across a set of images.
///
/// Phase coherence is the circular-mean magnitude: |sum(exp(i*phase_k))| / N.
/// Returns values in [0, 1] in the same order as `carriers`.
pub fn phase_coherence_at(
    images: &[ArrayView3<f32>],
    carriers: &[(i32, i32)],
    size: usize,
) -> Vec<f32> {
    let n = images.len();
    if n == 0 { return vec![0.0; carriers.len()]; }

    let mut phase_sum: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); carriers.len()];
    let center = size as i32 / 2;

    for img in images {
        let gray = to_grayscale(*img);
        let gray_resized = resize_gray(&gray, size, size);
        let spectrum = fftshift(fft2(gray_resized.view()).data);

        for (k, &(fy, fx)) in carriers.iter().enumerate() {
            let y = (fy + center) as usize;
            let x = (fx + center) as usize;
            if y < size && x < size {
                let phase = spectrum[[y, x]].arg();
                phase_sum[k] += Complex::new(phase.cos(), phase.sin());
            }
        }
    }

    phase_sum.into_iter().map(|s| s.norm() / n as f32).collect()
}

/// Extract the circular-mean reference phase at each carrier position.
pub fn reference_phases(
    images: &[ArrayView3<f32>],
    carriers: &[(i32, i32)],
    size: usize,
) -> Vec<f32> {
    let n = images.len();
    if n == 0 { return vec![0.0; carriers.len()]; }

    let mut phase_sum: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); carriers.len()];
    let center = size as i32 / 2;

    for img in images {
        let gray = to_grayscale(*img);
        let gray_resized = resize_gray(&gray, size, size);
        let spectrum = fftshift(fft2(gray_resized.view()).data);

        for (k, &(fy, fx)) in carriers.iter().enumerate() {
            let y = (fy + center) as usize;
            let x = (fx + center) as usize;
            if y < size && x < size {
                let phase = spectrum[[y, x]].arg();
                phase_sum[k] += Complex::new(phase.cos(), phase.sin());
            }
        }
    }

    phase_sum.into_iter().map(|s| s.arg()).collect()
}

/// Detect carrier frequencies from a set of images at a single scale.
///
/// Returned frequencies are relative to the spectrum centre (fftshift layout).
pub fn detect_carriers_at_scale(
    images: &[ArrayView3<f32>],
    size: usize,
    n_carriers: usize,
) -> Vec<CarrierInfo> {
    let n = images.len();
    if n == 0 { return vec![]; }

    let mut mag_sum   = Array2::<f32>::zeros((size, size));
    let mut phase_vec = Array2::<Complex<f32>>::zeros((size, size));

    for img in images {
        let gray = to_grayscale(*img);
        let gray_rs = resize_gray(&gray, size, size);
        let spectrum_shifted = fftshift(fft2(gray_rs.view()).data);

        for ((r, c), v) in spectrum_shifted.indexed_iter() {
            mag_sum[[r, c]]   += v.norm();
            let phi = v.arg();
            phase_vec[[r, c]] += Complex::new(phi.cos(), phi.sin());
        }
    }

    let avg_mag   = mag_sum.mapv(|v| v / n as f32);
    let coherence = phase_vec.mapv(|v| v.norm() / n as f32);
    let avg_phase = phase_vec.mapv(|v| v.arg());

    // Score = log1p(magnitude) * coherence. DC neighbourhood excluded (r < 5).
    let center = (size / 2) as i32;
    let dc_radius_sq = 25_i32;

    let mut candidates: Vec<(f32, i32, i32)> = (0..size)
        .flat_map(|r| (0..size).map(move |c| (r, c)))
        .filter(|&(r, c)| {
            let dy = r as i32 - center;
            let dx = c as i32 - center;
            dy * dy + dx * dx > dc_radius_sq
        })
        .map(|(r, c)| {
            let score = avg_mag[[r, c]].ln_1p() * coherence[[r, c]];
            (score, r as i32 - center, c as i32 - center)
        })
        .collect();

    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(n_carriers);

    candidates
        .into_iter()
        .map(|(score, fy, fx)| {
            let y = (fy + center) as usize;
            let x = (fx + center) as usize;
            CarrierInfo {
                freq: (fy, fx),
                magnitude: avg_mag[[y, x]],
                phase: avg_phase[[y, x]],
                coherence: coherence[[y, x]],
                votes: 1,
                score,
            }
        })
        .collect()
}

/// Detect carrier frequencies using multi-scale analysis with cross-scale voting.
///
/// Carriers appearing consistently across scales are ranked higher.
/// Falls back to hard-coded carriers if fewer than 5 are detected.
/// All returned frequencies are normalised to the 512px base scale.
pub fn detect_carriers(
    images: &[ArrayView3<f32>],
    scales: &[usize],
    n_carriers: usize,
) -> Vec<CarrierInfo> {
    const BASE_SCALE: usize = 512;
    // Frequencies within 2 bins of each other are treated as the same carrier.
    const BIN: i32 = 2;

    let mut vote_map: std::collections::HashMap<(i32, i32), Vec<CarrierInfo>> =
        std::collections::HashMap::new();

    for &scale in scales {
        for c in detect_carriers_at_scale(images, scale, n_carriers) {
            let nfy = (c.freq.0 as f64 * BASE_SCALE as f64 / scale as f64).round() as i32;
            let nfx = (c.freq.1 as f64 * BASE_SCALE as f64 / scale as f64).round() as i32;
            let key = ((nfy / BIN) * BIN, (nfx / BIN) * BIN);
            vote_map.entry(key).or_default().push(CarrierInfo { freq: (nfy, nfx), ..c });
        }
    }

    let mut result: Vec<CarrierInfo> = vote_map
        .into_values()
        .map(|entries| {
            let votes = entries.len() as u32;
            let avg = |f: fn(&CarrierInfo) -> f32| {
                entries.iter().map(f).sum::<f32>() / votes as f32
            };
            CarrierInfo {
                freq:      entries[0].freq,
                magnitude: avg(|e| e.magnitude),
                phase:     avg(|e| e.phase),
                coherence: avg(|e| e.coherence),
                score:     avg(|e| e.score),
                votes,
            }
        })
        .collect();

    result.sort_by(|a, b| {
        b.votes.cmp(&a.votes)
            .then(b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal))
    });

    if result.len() < 5 {
        eprintln!(
            "warn: only {} carriers detected; appending hard-coded fallbacks",
            result.len()
        );
        let found: std::collections::HashSet<(i32, i32)> =
            result.iter().map(|c| c.freq).collect();

        for &freq in CARRIERS_DARK.iter().chain(CARRIERS_WHITE.iter()) {
            if !found.contains(&freq) {
                result.push(CarrierInfo {
                    freq, magnitude: 1000.0, phase: 0.0, coherence: 0.99,
                    votes: 0, score: 50.0,
                });
            }
        }
    }

    result.truncate(n_carriers);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn all_carriers_is_union() {
        let all = all_carriers();
        for &c in CARRIERS_DARK  { assert!(all.contains(&c)); }
        for &c in CARRIERS_WHITE { assert!(all.contains(&c)); }
        assert_eq!(all.len(), CARRIERS_DARK.len() + CARRIERS_WHITE.len());
    }

    #[test]
    fn phase_coherence_on_flat_image_does_not_panic() {
        let img = Array3::<f32>::zeros((64, 64, 3));
        let views = vec![img.view()];
        let coh = phase_coherence_at(&views, CARRIERS_DARK, 64);
        assert_eq!(coh.len(), CARRIERS_DARK.len());
        assert!(coh.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn detect_carriers_fallback_on_empty_input() {
        let carriers = detect_carriers(&[], &[512], 100);
        let freqs: std::collections::HashSet<(i32, i32)> =
            carriers.iter().map(|c| c.freq).collect();
        for &f in CARRIERS_DARK  { assert!(freqs.contains(&f), "{f:?} missing"); }
        for &f in CARRIERS_WHITE { assert!(freqs.contains(&f), "{f:?} missing"); }
    }
}
