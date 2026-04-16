//! Codebook: extracted SynthID watermark profiles used for detection and bypass.
//!
//! A single `.ripbook` can store multiple profiles keyed by target image
//! resolution. Each profile contains the carrier reference phases, coherence
//! scores, a reference noise residual, and calibration statistics for one
//! resolution bucket.

use std::path::Path;

use anyhow::{Context, Result};
use ndarray::{Array3, ArrayView3};
use serde::{Deserialize, Serialize};

use crate::carriers::{phase_coherence_at, reference_phases, CARRIERS_DARK, CARRIERS_WHITE};
use crate::denoise::extract_noise_fused;
use crate::fft::{fft2_rgb, resize_rgb};

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

/// Watermark profile for one target resolution bucket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionProfile {
    /// Target image height for this profile.
    pub height: usize,
    /// Target image width for this profile.
    pub width: usize,
    /// Square processing size used internally by the current Rust pipeline.
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
    /// Native-resolution average watermark magnitude estimate.
    /// Flattened from shape (height, width, 3) in row-major order.
    pub magnitude_profile: Option<Vec<f32>>,
    /// Native-resolution circular-mean FFT phase template.
    /// Flattened from shape (height, width, 3) in row-major order.
    pub phase_template: Option<Vec<f32>>,
    /// Native-resolution phase consistency in [0, 1].
    /// Flattened from shape (height, width, 3) in row-major order.
    pub phase_consistency: Option<Vec<f32>>,
    /// Native-resolution average content magnitude baseline.
    /// Flattened from shape (height, width, 3) in row-major order.
    pub content_magnitude_baseline: Option<Vec<f32>>,
    /// Optional white-reference magnitude profile.
    /// Flattened from shape (height, width, 3) in row-major order.
    pub white_magnitude_profile: Option<Vec<f32>>,
    /// Optional black/white agreement confidence.
    /// Flattened from shape (height, width, 3) in row-major order.
    pub black_white_agreement: Option<Vec<f32>>,
}

impl ResolutionProfile {
    /// Return the reference noise as an (image_size, image_size, 3) array.
    pub fn reference_noise_array(&self) -> Array3<f32> {
        let s = self.image_size;
        Array3::from_shape_vec((s, s, 3), self.reference_noise.clone())
            .expect("reference_noise length matches image_size")
    }

    fn profile_array(
        data: &Option<Vec<f32>>,
        height: usize,
        width: usize,
    ) -> Option<Array3<f32>> {
        data.as_ref().map(|values| {
            Array3::from_shape_vec((height, width, 3), values.clone())
                .expect("profile array length matches resolution")
        })
    }

    pub fn magnitude_profile_array(&self) -> Option<Array3<f32>> {
        Self::profile_array(&self.magnitude_profile, self.height, self.width)
    }

    pub fn phase_template_array(&self) -> Option<Array3<f32>> {
        Self::profile_array(&self.phase_template, self.height, self.width)
    }

    pub fn phase_consistency_array(&self) -> Option<Array3<f32>> {
        Self::profile_array(&self.phase_consistency, self.height, self.width)
    }

    pub fn content_magnitude_baseline_array(&self) -> Option<Array3<f32>> {
        Self::profile_array(&self.content_magnitude_baseline, self.height, self.width)
    }

    pub fn white_magnitude_profile_array(&self) -> Option<Array3<f32>> {
        Self::profile_array(&self.white_magnitude_profile, self.height, self.width)
    }

    pub fn black_white_agreement_array(&self) -> Option<Array3<f32>> {
        Self::profile_array(&self.black_white_agreement, self.height, self.width)
    }
}

/// Multi-resolution codebook for detection and bypass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Codebook {
    pub version: u32,
    pub source: String,
    pub profiles: Vec<ResolutionProfile>,
}

impl Codebook {
    /// Build a single-profile codebook from a set of watermarked images.
    ///
    /// The created profile is keyed to `(image_size, image_size)` for backwards
    /// compatibility with the original single-resolution implementation.
    pub fn build(images: &[Array3<f32>], image_size: usize, source: impl Into<String>) -> Self {
        let source = source.into();
        let profile = Self::build_profile(images, image_size, image_size, image_size);
        Self {
            version: 2,
            source,
            profiles: vec![profile],
        }
    }

    /// Build one resolution profile from a set of watermarked images.
    ///
    /// `height` and `width` describe the target image resolution this profile
    /// should be selected for. `image_size` is the square internal processing
    /// size used by the current Rust implementation.
    pub fn build_profile(
        images: &[Array3<f32>],
        height: usize,
        width: usize,
        image_size: usize,
    ) -> ResolutionProfile {
        let n = images.len();
        assert!(n > 0, "need at least one image to build a codebook profile");

        let resized: Vec<Array3<f32>> = images
            .iter()
            .map(|img| resize_rgb(img, image_size, image_size))
            .collect();

        let views: Vec<ArrayView3<f32>> = resized.iter().map(|a| a.view()).collect();

        let dark_phases = reference_phases(&views, CARRIERS_DARK, image_size);
        let dark_coh = phase_coherence_at(&views, CARRIERS_DARK, image_size);
        let white_phases = reference_phases(&views, CARRIERS_WHITE, image_size);
        let white_coh = phase_coherence_at(&views, CARRIERS_WHITE, image_size);

        let mut noise_acc = Array3::<f64>::zeros((image_size, image_size, 3));
        for img in &resized {
            for ((r, c, ch), v) in extract_noise_fused(img.view()).indexed_iter() {
                noise_acc[[r, c, ch]] += *v as f64;
            }
        }
        let reference_noise: Vec<f32> = noise_acc.iter().map(|&v| (v / n as f64) as f32).collect();

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
        let spectral = build_spectral_profile(images, height, width);

        ResolutionProfile {
            height,
            width,
            image_size,
            n_images: n,
            dark: CarrierProfile {
                ref_phases: dark_phases,
                coherence: dark_coh,
            },
            white: CarrierProfile {
                ref_phases: white_phases,
                coherence: white_coh,
            },
            reference_noise,
            correlation_mean,
            correlation_std,
            detection_threshold: correlation_mean - 2.5 * correlation_std,
            magnitude_profile: Some(spectral.magnitude_profile),
            phase_template: Some(spectral.phase_template),
            phase_consistency: Some(spectral.phase_consistency),
            content_magnitude_baseline: Some(spectral.content_magnitude_baseline),
            white_magnitude_profile: None,
            black_white_agreement: None,
        }
    }

    /// Add a profile to the codebook, replacing any existing entry for the same
    /// `(height, width)` bucket.
    pub fn add_profile(&mut self, profile: ResolutionProfile) {
        if let Some(existing) = self
            .profiles
            .iter_mut()
            .find(|p| p.height == profile.height && p.width == profile.width)
        {
            *existing = profile;
        } else {
            self.profiles.push(profile);
        }
    }

    pub fn resolutions(&self) -> Vec<(usize, usize)> {
        self.profiles.iter().map(|p| (p.height, p.width)).collect()
    }

    pub fn primary_profile(&self) -> Option<&ResolutionProfile> {
        self.profiles.first()
    }

    /// Best-matching profile for target `(height, width)`.
    ///
    /// Prefers exact resolution matches, else chooses the closest aspect ratio
    /// and pixel count, following the Python `SpectralCodebook` heuristic.
    pub fn best_profile(&self, height: usize, width: usize) -> Result<(&ResolutionProfile, bool)> {
        if let Some(exact) = self
            .profiles
            .iter()
            .find(|profile| profile.height == height && profile.width == width)
        {
            return Ok((exact, true));
        }

        if self.profiles.is_empty() {
            anyhow::bail!("codebook has no profiles");
        }

        let target_ar = height as f64 / (width.max(1) as f64);
        let target_px = (height * width).max(1) as f64;

        let best = self
            .profiles
            .iter()
            .min_by(|a, b| {
                let score_a = profile_match_score(a, target_ar, target_px);
                let score_b = profile_match_score(b, target_ar, target_px);
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("non-empty profiles");

        Ok((best, false))
    }

    /// Save codebook to a `.ripbook` file.
    pub fn save(&self, path: &Path) -> Result<()> {
        use std::io::Write;

        let payload = bincode::serde::encode_to_vec(self, bincode::config::standard())
            .context("failed to encode codebook")?;

        let mut file = std::fs::File::create(path)
            .with_context(|| format!("cannot create {}", path.display()))?;

        file.write_all(MAGIC).context("write magic")?;
        file.write_all(&(payload.len() as u64).to_le_bytes())
            .context("write length")?;
        file.write_all(&payload).context("write payload")?;

        Ok(())
    }

    /// Load codebook from a `.ripbook` file.
    ///
    /// Supports both the current multi-profile format and the original
    /// single-profile format used earlier in this Rust port.
    pub fn load(path: &Path) -> Result<Self> {
        use std::io::Read;

        let mut file =
            std::fs::File::open(path).with_context(|| format!("cannot open {}", path.display()))?;

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

        if let Ok((codebook, _)) =
            bincode::serde::decode_from_slice::<Codebook, _>(&payload, bincode::config::standard())
        {
            if codebook.version >= 2 && !codebook.profiles.is_empty() {
                return Ok(codebook);
            }
        }

        let (legacy, _): (LegacyCodebook, _) =
            bincode::serde::decode_from_slice(&payload, bincode::config::standard())
                .context("failed to decode codebook")?;

        Ok(legacy.into())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacyCodebook {
    version: u32,
    source: String,
    image_size: usize,
    n_images: usize,
    dark: CarrierProfile,
    white: CarrierProfile,
    reference_noise: Vec<f32>,
    correlation_mean: f32,
    correlation_std: f32,
    detection_threshold: f32,
}

impl From<LegacyCodebook> for Codebook {
    fn from(value: LegacyCodebook) -> Self {
        Self {
            version: value.version.max(2),
            source: value.source,
            profiles: vec![ResolutionProfile {
                height: value.image_size,
                width: value.image_size,
                image_size: value.image_size,
                n_images: value.n_images,
                dark: value.dark,
                white: value.white,
                reference_noise: value.reference_noise,
                correlation_mean: value.correlation_mean,
                correlation_std: value.correlation_std,
                detection_threshold: value.detection_threshold,
                magnitude_profile: None,
                phase_template: None,
                phase_consistency: None,
                content_magnitude_baseline: None,
                white_magnitude_profile: None,
                black_white_agreement: None,
            }],
        }
    }
}

struct SpectralProfileData {
    magnitude_profile: Vec<f32>,
    phase_template: Vec<f32>,
    phase_consistency: Vec<f32>,
    content_magnitude_baseline: Vec<f32>,
}

fn build_spectral_profile(images: &[Array3<f32>], height: usize, width: usize) -> SpectralProfileData {
    let mut mag_sum = Array3::<f64>::zeros((height, width, 3));
    let mut phase_re = Array3::<f64>::zeros((height, width, 3));
    let mut phase_im = Array3::<f64>::zeros((height, width, 3));

    for image in images {
        let spectra = fft2_rgb(image.view());
        for (ch, spectrum) in spectra.iter().enumerate() {
            for ((y, x), bin) in spectrum.data.indexed_iter() {
                let mag = bin.norm() as f64;
                let phase = bin.arg() as f64;
                mag_sum[[y, x, ch]] += mag;
                phase_re[[y, x, ch]] += phase.cos();
                phase_im[[y, x, ch]] += phase.sin();
            }
        }
    }

    let n = images.len() as f64;
    let mut magnitude_profile = Vec::with_capacity(height * width * 3);
    let mut phase_template = Vec::with_capacity(height * width * 3);
    let mut phase_consistency = Vec::with_capacity(height * width * 3);
    let mut content_magnitude_baseline = Vec::with_capacity(height * width * 3);

    for y in 0..height {
        for x in 0..width {
            for ch in 0..3 {
                let avg_mag = mag_sum[[y, x, ch]] / n;
                let re = phase_re[[y, x, ch]] / n;
                let im = phase_im[[y, x, ch]] / n;
                let coherence = (re * re + im * im).sqrt() as f32;
                magnitude_profile.push((avg_mag as f32) * coherence * coherence);
                phase_template.push(im.atan2(re) as f32);
                phase_consistency.push(coherence);
                content_magnitude_baseline.push(avg_mag as f32);
            }
        }
    }

    SpectralProfileData {
        magnitude_profile,
        phase_template,
        phase_consistency,
        content_magnitude_baseline,
    }
}

fn profile_match_score(profile: &ResolutionProfile, target_ar: f64, target_px: f64) -> f64 {
    let profile_ar = profile.height as f64 / (profile.width.max(1) as f64);
    let profile_px = (profile.height * profile.width).max(1) as f64;
    let ar_diff = (profile_ar - target_ar).abs() / target_ar.max(1e-9);
    let px_diff = (profile_px - target_px).abs() / target_px.max(1e-9);
    ar_diff * 2.0 + px_diff
}

fn pearson_correlation(a: &[f32], b: &[f32]) -> f32 {
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
    if denom < 1e-10 {
        0.0
    } else {
        num / denom
    }
}

fn mean_std(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
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

    fn dummy_images(n: usize, h: usize, w: usize) -> Vec<Array3<f32>> {
        (0..n)
            .map(|i| {
                Array3::from_shape_fn((h, w, 3), |(r, c, ch)| {
                    ((r + c + ch + i * 7) as f32 / ((h + w + 3) as f32)).min(1.0)
                })
            })
            .collect()
    }

    #[test]
    fn build_codebook_basic() {
        let images = dummy_images(4, 32, 48);
        let profile = Codebook::build_profile(&images, 32, 48, 32);

        assert_eq!(profile.height, 32);
        assert_eq!(profile.width, 48);
        assert_eq!(profile.image_size, 32);
        assert_eq!(profile.n_images, 4);
        assert_eq!(profile.dark.ref_phases.len(), CARRIERS_DARK.len());
        assert_eq!(profile.white.ref_phases.len(), CARRIERS_WHITE.len());
        assert_eq!(profile.reference_noise.len(), 32 * 32 * 3);
        assert!(profile.dark.coherence.iter().all(|v| v.is_finite()));
        assert!(profile.white.coherence.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn codebook_save_load_roundtrip() {
        let images_a = dummy_images(3, 32, 32);
        let images_b = dummy_images(3, 48, 64);

        let mut cb = Codebook {
            version: 2,
            source: "roundtrip-test".into(),
            profiles: vec![Codebook::build_profile(&images_a, 32, 32, 32)],
        };
        cb.add_profile(Codebook::build_profile(&images_b, 48, 64, 64));

        let tmp = NamedTempFile::new().unwrap();
        cb.save(tmp.path()).expect("save failed");
        let loaded = Codebook::load(tmp.path()).expect("load failed");

        assert_eq!(loaded.version, cb.version);
        assert_eq!(loaded.profiles.len(), 2);
        assert!(loaded.resolutions().contains(&(32, 32)));
        assert!(loaded.resolutions().contains(&(48, 64)));
    }

    #[test]
    fn best_profile_prefers_exact_resolution() {
        let images_a = dummy_images(3, 512, 512);
        let images_b = dummy_images(3, 768, 1024);
        let mut cb = Codebook {
            version: 2,
            source: "exact-match".into(),
            profiles: vec![Codebook::build_profile(&images_a, 512, 512, 32)],
        };
        cb.add_profile(Codebook::build_profile(&images_b, 768, 1024, 32));

        let (profile, exact) = cb.best_profile(768, 1024).unwrap();
        assert!(exact);
        assert_eq!((profile.height, profile.width), (768, 1024));
    }

    #[test]
    fn best_profile_falls_back_to_closest_aspect_ratio() {
        let images_a = dummy_images(3, 512, 512);
        let images_b = dummy_images(3, 768, 1024);
        let mut cb = Codebook {
            version: 2,
            source: "fallback-match".into(),
            profiles: vec![Codebook::build_profile(&images_a, 512, 512, 32)],
        };
        cb.add_profile(Codebook::build_profile(&images_b, 768, 1024, 32));

        let (profile, exact) = cb.best_profile(720, 960).unwrap();
        assert!(!exact);
        assert_eq!((profile.height, profile.width), (768, 1024));
    }

    #[test]
    fn legacy_codebook_loads_as_single_profile() {
        let legacy = LegacyCodebook {
            version: 1,
            source: "legacy".into(),
            image_size: 32,
            n_images: 4,
            dark: CarrierProfile {
                ref_phases: vec![0.0; CARRIERS_DARK.len()],
                coherence: vec![1.0; CARRIERS_DARK.len()],
            },
            white: CarrierProfile {
                ref_phases: vec![0.0; CARRIERS_WHITE.len()],
                coherence: vec![1.0; CARRIERS_WHITE.len()],
            },
            reference_noise: vec![0.0; 32 * 32 * 3],
            correlation_mean: 0.1,
            correlation_std: 0.01,
            detection_threshold: 0.075,
        };

        let payload = bincode::serde::encode_to_vec(&legacy, bincode::config::standard()).unwrap();

        let tmp = NamedTempFile::new().unwrap();
        use std::io::Write;
        let mut file = std::fs::File::create(tmp.path()).unwrap();
        file.write_all(MAGIC).unwrap();
        file.write_all(&(payload.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&payload).unwrap();

        let loaded = Codebook::load(tmp.path()).unwrap();
        assert_eq!(loaded.profiles.len(), 1);
        let profile = loaded.primary_profile().unwrap();
        assert_eq!((profile.height, profile.width), (32, 32));
        assert_eq!(profile.image_size, 32);
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
