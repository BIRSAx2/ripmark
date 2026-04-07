//! Codebook: extracted SynthID watermark profile used for detection and bypass.
//!
//! Built from a set of Gemini-generated images. Stores reference phases at known
//! carrier frequencies, phase coherence scores, an average noise residual, and
//! detection calibration statistics.
//!
//! Saved to `.ripbook` files (bincode, length-prefixed with magic header).

use std::path::Path;

use anyhow::{Context, Result};
use ndarray::{Array3, ArrayView3};
use serde::{Deserialize, Serialize};

use crate::carriers::{phase_coherence_at, reference_phases, CARRIERS_DARK, CARRIERS_WHITE};
use crate::denoise::extract_noise_fused;
use crate::fft::resize_rgb;

const MAGIC: &[u8; 8] = b"RIPBOOK\x01";

/// Reference phases and coherence scores for one carrier set (dark or white).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CarrierProfile {
    /// Circular-mean phase at each carrier position across the training images.
    /// Indexed in the same order as CARRIERS_DARK / CARRIERS_WHITE.
    pub ref_phases: Vec<f32>,

    /// Phase coherence in [0, 1] at each carrier.
    /// Near 1.0 means the phase is fixed across images (watermark signal).
    /// Near 0.0 means the phase is random (content noise).
    pub coherence: Vec<f32>,
}

/// The complete watermark profile extracted from a set of Gemini images.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Codebook {
    pub version: u32,
    pub source: String,
    /// Square size in pixels at which all processing is done (typically 512).
    pub image_size: usize,
    pub n_images: usize,
    /// Profile for the dark-image carrier set (diagonal grid).
    pub dark: CarrierProfile,
    /// Profile for the white-image carrier set (horizontal axis).
    pub white: CarrierProfile,
    /// Average noise residual across all training images,
    /// flattened from shape (image_size, image_size, 3) in row-major order.
    pub reference_noise: Vec<f32>,
    /// Mean pairwise noise-residual Pearson correlation across the training set.
    pub correlation_mean: f32,
    pub correlation_std: f32,
    /// correlation_mean - 2.5 * correlation_std. Images below this are likely
    /// non-watermarked.
    pub detection_threshold: f32,
}

impl Codebook {
    /// Build a codebook from a set of watermarked images (RGB, values in [0, 1]).
    /// All images are internally resized to image_size x image_size.
    pub fn build(images: &[Array3<f32>], image_size: usize, source: impl Into<String>) -> Self {
        let n = images.len();
        assert!(n > 0, "need at least one image to build a codebook");

        let resized: Vec<Array3<f32>> = images
            .iter()
            .map(|img| resize_rgb(img, image_size, image_size))
            .collect();

        let views: Vec<ArrayView3<f32>> = resized.iter().map(|a| a.view()).collect();

        let dark_phases  = reference_phases(&views, CARRIERS_DARK,  image_size);
        let dark_coh     = phase_coherence_at(&views, CARRIERS_DARK,  image_size);
        let white_phases = reference_phases(&views, CARRIERS_WHITE, image_size);
        let white_coh    = phase_coherence_at(&views, CARRIERS_WHITE, image_size);

        let mut noise_acc = Array3::<f64>::zeros((image_size, image_size, 3));
        for img in &resized {
            for ((r, c, ch), v) in extract_noise_fused(img.view()).indexed_iter() {
                noise_acc[[r, c, ch]] += *v as f64;
            }
        }
        let reference_noise: Vec<f32> = noise_acc
            .iter()
            .map(|&v| (v / n as f64) as f32)
            .collect();

        // Pairwise correlation on up to 50 images to calibrate the detector.
        let sample_size = n.min(50);
        let noises: Vec<Vec<f32>> = resized[..sample_size]
            .iter()
            .map(|img| extract_noise_fused(img.view()).into_iter().collect())
            .collect();

        let mut correlations: Vec<f32> = Vec::new();
        for i in 0..sample_size {
            for j in (i + 1)..sample_size {
                correlations.push(pearson_correlation(&noises[i], &noises[j]));
            }
        }

        let (correlation_mean, correlation_std) = mean_std(&correlations);

        Codebook {
            version: 1,
            source: source.into(),
            image_size,
            n_images: n,
            dark:  CarrierProfile { ref_phases: dark_phases,  coherence: dark_coh },
            white: CarrierProfile { ref_phases: white_phases, coherence: white_coh },
            reference_noise,
            correlation_mean,
            correlation_std,
            detection_threshold: correlation_mean - 2.5 * correlation_std,
        }
    }

    /// Save codebook to a `.ripbook` file.
    pub fn save(&self, path: &Path) -> Result<()> {
        use std::io::Write;

        let payload = bincode::serde::encode_to_vec(self, bincode::config::standard())
            .context("failed to encode codebook")?;

        let mut file = std::fs::File::create(path)
            .with_context(|| format!("cannot create {}", path.display()))?;

        file.write_all(MAGIC).context("write magic")?;
        file.write_all(&(payload.len() as u64).to_le_bytes()).context("write length")?;
        file.write_all(&payload).context("write payload")?;

        Ok(())
    }

    /// Load codebook from a `.ripbook` file.
    pub fn load(path: &Path) -> Result<Self> {
        use std::io::Read;

        let mut file = std::fs::File::open(path)
            .with_context(|| format!("cannot open {}", path.display()))?;

        let mut magic = [0u8; 8];
        file.read_exact(&mut magic).context("read magic")?;
        if &magic != MAGIC {
            anyhow::bail!("not a .ripbook file: {}", path.display());
        }

        let mut len_buf = [0u8; 8];
        file.read_exact(&mut len_buf).context("read length")?;
        let payload_len = u64::from_le_bytes(len_buf) as usize;

        let mut payload = vec![0u8; payload_len];
        file.read_exact(&mut payload).context("read payload")?;

        let (codebook, _): (Codebook, _) =
            bincode::serde::decode_from_slice(&payload, bincode::config::standard())
                .context("failed to decode codebook")?;

        Ok(codebook)
    }

    /// Return the reference noise as an (image_size, image_size, 3) array.
    pub fn reference_noise_array(&self) -> Array3<f32> {
        let s = self.image_size;
        Array3::from_shape_vec((s, s, 3), self.reference_noise.clone())
            .expect("reference_noise length matches image_size")
    }
}

fn pearson_correlation(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let mean_a = a.iter().sum::<f32>() / n;
    let mean_b = b.iter().sum::<f32>() / n;

    let (mut num, mut den_a, mut den_b) = (0.0_f32, 0.0_f32, 0.0_f32);
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        let da = ai - mean_a;
        let db = bi - mean_b;
        num   += da * db;
        den_a += da * da;
        den_b += db * db;
    }

    let denom = (den_a * den_b).sqrt();
    if denom < 1e-10 { 0.0 } else { num / denom }
}

fn mean_std(values: &[f32]) -> (f32, f32) {
    if values.is_empty() { return (0.0, 0.0); }
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let variance = values.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
    (mean, variance.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    use tempfile::NamedTempFile;

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
    fn build_codebook_basic() {
        let images = dummy_images(4, 32);
        let cb = Codebook::build(&images, 32, "test");

        assert_eq!(cb.image_size, 32);
        assert_eq!(cb.n_images, 4);
        assert_eq!(cb.dark.ref_phases.len(),  CARRIERS_DARK.len());
        assert_eq!(cb.white.ref_phases.len(), CARRIERS_WHITE.len());
        assert_eq!(cb.reference_noise.len(),  32 * 32 * 3);
        assert!(cb.dark.coherence.iter().all(|v| v.is_finite()));
        assert!(cb.white.coherence.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn codebook_save_load_roundtrip() {
        let images = dummy_images(3, 32);
        let cb = Codebook::build(&images, 32, "roundtrip-test");

        let tmp = NamedTempFile::new().unwrap();
        cb.save(tmp.path()).expect("save failed");
        let loaded = Codebook::load(tmp.path()).expect("load failed");

        assert_eq!(loaded.version,    cb.version);
        assert_eq!(loaded.n_images,   cb.n_images);
        assert_eq!(loaded.image_size, cb.image_size);

        for (a, b) in cb.dark.ref_phases.iter().zip(loaded.dark.ref_phases.iter()) {
            assert!((a - b).abs() < 1e-5, "phase mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn bad_magic_returns_error() {
        let tmp = NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"NOTRIPBK\x00\x00\x00\x00\x00\x00\x00\x00").unwrap();
        assert!(Codebook::load(tmp.path()).is_err());
    }

    #[test]
    fn pearson_perfect_correlation() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 4.0, 6.0, 8.0];
        assert!((pearson_correlation(&a, &b) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn pearson_anticorrelated() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![4.0, 3.0, 2.0, 1.0];
        assert!((pearson_correlation(&a, &b) + 1.0).abs() < 1e-5);
    }
}
