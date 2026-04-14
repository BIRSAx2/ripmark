use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use image::{ImageBuffer, Luma, RgbImage};
use ndarray::{Array2, Array3};
use ripmark_core::analysis::{analyze_image_set, top_coherent_bins};
use ripmark_core::bypass::{bypass, BypassMode};
use ripmark_core::codebook::Codebook;
use ripmark_core::detect::{detect, BestSet};
use serde::Serialize;

#[derive(Parser)]
#[command(
    name = "ripmark",
    about = "SynthID watermark analysis toolkit",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Command,

    /// Print detailed scores and timing
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Output machine-readable JSON
    #[arg(long, global = true)]
    json: bool,
}

#[derive(Subcommand)]
enum Command {
    /// Detect SynthID watermark in an image
    Detect {
        image: PathBuf,
        #[arg(long)]
        codebook: PathBuf,
        #[arg(long, default_value = "0.5")]
        threshold: f32,
    },

    /// Build a codebook from a directory of watermarked images
    Extract {
        image_dir: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value = "250")]
        max_images: usize,
        #[arg(long, value_delimiter = ',', default_values = ["256", "512", "1024"])]
        scales: Vec<u32>,
    },

    /// Remove SynthID watermark from an image
    Bypass {
        image: PathBuf,
        #[arg(long)]
        codebook: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value = "v3")]
        mode: String,
    },

    /// Full frequency/phase analysis of an image set
    Analyze {
        image_dir: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
    },
}

#[derive(Debug, Serialize)]
struct DetectOutput {
    image: String,
    codebook: String,
    selected_profile: String,
    exact_profile_match: bool,
    watermarked: bool,
    confidence: f32,
    threshold: f32,
    phase_match: f32,
    best_set: String,
    dark_phase_match: f32,
    white_phase_match: f32,
    cvr_ratio: f32,
    phase_score: f32,
    cvr_score: f32,
}

#[derive(Debug, Serialize)]
struct ExtractOutput {
    image_dir: String,
    output: String,
    profile_resolution: String,
    n_images: usize,
    image_size: usize,
    requested_scales: Vec<u32>,
    codebook_version: u32,
    resolutions: Vec<String>,
    detection_threshold: f32,
    correlation_mean: f32,
    correlation_std: f32,
}

#[derive(Debug, Serialize)]
struct BypassOutput {
    image: String,
    output: String,
    codebook: String,
    mode: String,
    selected_profile: String,
    exact_profile_match: bool,
    psnr: f32,
    ssim: f32,
    carrier_energy_drop: f32,
    phase_coherence_drop: f32,
    stages: Vec<String>,
}

#[derive(Debug, Serialize)]
struct AnalyzeCarrier {
    freq: (i32, i32),
    magnitude: f32,
    phase: f32,
    coherence: f32,
    votes: u32,
    score: f32,
}

#[derive(Debug, Serialize)]
struct AnalyzeTopBin {
    freq: (i32, i32),
    score: f32,
}

#[derive(Debug, Serialize)]
struct AnalyzeSummary {
    image_dir: String,
    output_dir: String,
    n_images: usize,
    image_size: usize,
    overall_phase_coherence: f32,
    top_carriers: Vec<AnalyzeCarrier>,
    top_coherent_bins: Vec<AnalyzeTopBin>,
    files_written: Vec<String>,
}

#[derive(Debug, Serialize)]
struct AnalyzeProfiles {
    vertical_magnitude: Vec<f32>,
    vertical_coherence: Vec<f32>,
    radial_magnitude: Vec<f32>,
    radial_coherence: Vec<f32>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let verbose = cli.verbose;
    let json = cli.json;

    match cli.command {
        Command::Detect {
            image,
            codebook,
            threshold,
        } => run_detect(verbose, json, &image, &codebook, threshold),
        Command::Extract {
            image_dir,
            output,
            max_images,
            scales,
        } => run_extract(verbose, json, &image_dir, &output, max_images, &scales),
        Command::Bypass {
            image,
            codebook,
            output,
            mode,
        } => run_bypass(verbose, json, &image, &codebook, &output, &mode),
        Command::Analyze {
            image_dir,
            output_dir,
        } => run_analyze(verbose, json, &image_dir, &output_dir),
    }
}

fn run_detect(
    verbose: bool,
    json: bool,
    image_path: &Path,
    codebook_path: &Path,
    threshold: f32,
) -> Result<()> {
    let image = load_rgb_image(image_path)?;
    let codebook = Codebook::load(codebook_path)?;
    let (profile, exact_match) = codebook.best_profile(image.dim().0, image.dim().1)?;
    let result = detect(image.view(), &codebook);

    let output = DetectOutput {
        image: image_path.display().to_string(),
        codebook: codebook_path.display().to_string(),
        selected_profile: format!("{}x{}", profile.height, profile.width),
        exact_profile_match: exact_match,
        watermarked: result.confidence > threshold,
        confidence: result.confidence,
        threshold,
        phase_match: result.phase_match,
        best_set: best_set_label(result.best_set).to_string(),
        dark_phase_match: result.dark_phase_match,
        white_phase_match: result.white_phase_match,
        cvr_ratio: result.cvr_ratio,
        phase_score: result.phase_score,
        cvr_score: result.cvr_score,
    };

    if json {
        print_json(&output)?;
    } else {
        println!("watermarked: {}", output.watermarked);
        println!("confidence: {:.3}", output.confidence);
        println!("threshold:  {:.3}", output.threshold);
        println!(
            "profile:    {}{}",
            output.selected_profile,
            if output.exact_profile_match {
                ""
            } else {
                " (fallback)"
            }
        );
        println!(
            "phase_match: {:.3} (best carrier set: {})",
            output.phase_match, output.best_set
        );
        println!("cvr_ratio:   {:.3}", output.cvr_ratio);
        if verbose {
            println!("dark_phase_match:  {:.3}", output.dark_phase_match);
            println!("white_phase_match: {:.3}", output.white_phase_match);
            println!("phase_score:       {:.3}", output.phase_score);
            println!("cvr_score:         {:.3}", output.cvr_score);
        }
    }

    Ok(())
}

fn run_extract(
    verbose: bool,
    json: bool,
    image_dir: &Path,
    output_path: &Path,
    max_images: usize,
    scales: &[u32],
) -> Result<()> {
    let image_paths = collect_image_paths(image_dir, usize::MAX)?;
    let (profile_height, profile_width) = image_dimensions(&image_paths[0])?;
    let filtered_paths: Vec<PathBuf> = image_paths
        .into_iter()
        .filter(|path| {
            image_dimensions(path)
                .map(|dims| dims == (profile_height, profile_width))
                .unwrap_or(false)
        })
        .take(max_images)
        .collect();
    if filtered_paths.is_empty() {
        bail!(
            "no images matching resolution {}x{} found in {}",
            profile_height,
            profile_width,
            image_dir.display()
        );
    }

    let images = load_images(&filtered_paths)?;
    let image_size = choose_extract_size(scales);

    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
    }

    let profile = Codebook::build_profile(&images, profile_height, profile_width, image_size);
    let mut codebook = if output_path.exists() {
        Codebook::load(output_path)?
    } else {
        Codebook {
            version: 2,
            source: image_dir.display().to_string(),
            profiles: Vec::new(),
        }
    };
    codebook.add_profile(profile.clone());
    codebook.save(output_path)?;

    let output = ExtractOutput {
        image_dir: image_dir.display().to_string(),
        output: output_path.display().to_string(),
        profile_resolution: format!("{}x{}", profile.height, profile.width),
        n_images: images.len(),
        image_size,
        requested_scales: scales.to_vec(),
        codebook_version: codebook.version,
        resolutions: codebook
            .resolutions()
            .into_iter()
            .map(|(h, w)| format!("{h}x{w}"))
            .collect(),
        detection_threshold: profile.detection_threshold,
        correlation_mean: profile.correlation_mean,
        correlation_std: profile.correlation_std,
    };

    if json {
        print_json(&output)?;
    } else {
        println!("extracted codebook: {}", output.output);
        println!("profile_resolution: {}", output.profile_resolution);
        println!("images:             {}", output.n_images);
        println!("image_size:         {}", output.image_size);
        println!("requested_scales:   {:?}", output.requested_scales);
        if verbose {
            println!("version:            {}", output.codebook_version);
            println!("resolutions:        {}", output.resolutions.join(", "));
            println!("detection_threshold:{:.4}", output.detection_threshold);
            println!("correlation_mean:   {:.4}", output.correlation_mean);
            println!("correlation_std:    {:.4}", output.correlation_std);
        }
    }

    Ok(())
}

fn run_bypass(
    verbose: bool,
    json: bool,
    image_path: &Path,
    codebook_path: &Path,
    output_path: &Path,
    mode: &str,
) -> Result<()> {
    let image = load_rgb_image(image_path)?;
    let codebook = Codebook::load(codebook_path)?;
    let mode = parse_bypass_mode(mode)?;
    let result = bypass(image.view(), &codebook, mode)?;

    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
    }
    save_rgb_image(output_path, &result.image)?;

    let output = BypassOutput {
        image: image_path.display().to_string(),
        output: output_path.display().to_string(),
        codebook: codebook_path.display().to_string(),
        mode: bypass_mode_label(mode).to_string(),
        selected_profile: format!(
            "{}x{}",
            result.profile_resolution.0, result.profile_resolution.1
        ),
        exact_profile_match: result.exact_profile_match,
        psnr: result.psnr,
        ssim: result.ssim,
        carrier_energy_drop: result.carrier_energy_drop,
        phase_coherence_drop: result.phase_coherence_drop,
        stages: result.stages,
    };

    if json {
        print_json(&output)?;
    } else {
        println!("wrote bypassed image: {}", output.output);
        println!("mode:                {}", output.mode);
        println!(
            "profile:             {}{}",
            output.selected_profile,
            if output.exact_profile_match {
                ""
            } else {
                " (fallback)"
            }
        );
        println!("psnr:                {:.3}", output.psnr);
        println!("ssim:                {:.4}", output.ssim);
        println!("carrier_energy_drop: {:.4}", output.carrier_energy_drop);
        println!("phase_coherence_drop:{:.4}", output.phase_coherence_drop);
        if verbose {
            println!("stages:              {}", output.stages.join(", "));
        }
    }

    Ok(())
}

fn run_analyze(verbose: bool, json: bool, image_dir: &Path, output_dir: &Path) -> Result<()> {
    let image_paths = collect_image_paths(image_dir, usize::MAX)?;
    let images = load_images(&image_paths)?;
    let report = analyze_image_set(&images, 512, &[256, 512, 1024]);

    fs::create_dir_all(output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;

    let report_path = output_dir.join("report.json");
    let summary_path = output_dir.join("summary.json");
    let top_carriers_path = output_dir.join("top_carriers.json");
    let top_bins_path = output_dir.join("top_coherent_bins.json");
    let profiles_path = output_dir.join("profiles.json");
    let magnitude_path = output_dir.join("magnitude_spectrum.png");
    let coherence_path = output_dir.join("phase_coherence_map.png");
    let pca_path = output_dir.join("pca_watermark.png");
    let carrier_mask_path = output_dir.join("carrier_mask.png");

    save_heatmap(&magnitude_path, &report.magnitude_spectrum)?;
    save_heatmap(&coherence_path, &report.phase_coherence_map)?;
    save_heatmap(&pca_path, &report.pca_watermark)?;

    let top_bins = top_coherent_bins(
        &report.phase_coherence_map,
        &report.magnitude_spectrum,
        20,
        5,
    );

    let top_carriers: Vec<AnalyzeCarrier> = report
        .top_carriers
        .iter()
        .take(20)
        .map(|carrier| AnalyzeCarrier {
            freq: carrier.freq,
            magnitude: carrier.magnitude,
            phase: carrier.phase,
            coherence: carrier.coherence,
            votes: carrier.votes,
            score: carrier.score,
        })
        .collect();
    let top_coherent_bins: Vec<AnalyzeTopBin> = top_bins
        .iter()
        .map(|(freq, score)| AnalyzeTopBin {
            freq: *freq,
            score: *score,
        })
        .collect();
    let profiles = AnalyzeProfiles {
        vertical_magnitude: vertical_profile(&report.magnitude_spectrum),
        vertical_coherence: vertical_profile(&report.phase_coherence_map),
        radial_magnitude: radial_profile(&report.magnitude_spectrum),
        radial_coherence: radial_profile(&report.phase_coherence_map),
    };
    let overall_phase_coherence = mean_f32(report.phase_coherence_map.iter().copied());
    save_carrier_mask(&carrier_mask_path, report.image_size, &top_carriers)?;

    let summary = AnalyzeSummary {
        image_dir: image_dir.display().to_string(),
        output_dir: output_dir.display().to_string(),
        n_images: images.len(),
        image_size: report.image_size,
        overall_phase_coherence,
        top_carriers,
        top_coherent_bins,
        files_written: vec![
            report_path.display().to_string(),
            summary_path.display().to_string(),
            top_carriers_path.display().to_string(),
            top_bins_path.display().to_string(),
            profiles_path.display().to_string(),
            magnitude_path.display().to_string(),
            coherence_path.display().to_string(),
            pca_path.display().to_string(),
            carrier_mask_path.display().to_string(),
        ],
    };

    fs::write(&report_path, serde_json::to_vec_pretty(&summary)?)
        .with_context(|| format!("failed to write {}", report_path.display()))?;
    fs::write(&summary_path, serde_json::to_vec_pretty(&summary)?)
        .with_context(|| format!("failed to write {}", summary_path.display()))?;
    fs::write(
        &top_carriers_path,
        serde_json::to_vec_pretty(&summary.top_carriers)?,
    )
    .with_context(|| format!("failed to write {}", top_carriers_path.display()))?;
    fs::write(
        &top_bins_path,
        serde_json::to_vec_pretty(&summary.top_coherent_bins)?,
    )
    .with_context(|| format!("failed to write {}", top_bins_path.display()))?;
    fs::write(&profiles_path, serde_json::to_vec_pretty(&profiles)?)
        .with_context(|| format!("failed to write {}", profiles_path.display()))?;

    if json {
        print_json(&summary)?;
    } else {
        println!("analyzed images: {}", summary.n_images);
        println!("output_dir:      {}", summary.output_dir);
        println!("report:          {}", report_path.display());
        println!("phase_coherence: {:.4}", summary.overall_phase_coherence);
        if verbose {
            println!("files:");
            for file in &summary.files_written {
                println!("  {}", file);
            }
        }
    }

    Ok(())
}

fn collect_image_paths(dir: &Path, max_images: usize) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    let entries =
        fs::read_dir(dir).with_context(|| format!("failed to read directory {}", dir.display()))?;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if !is_supported_image(&path) {
            continue;
        }
        paths.push(path);
    }

    paths.sort();
    if max_images != usize::MAX {
        paths.truncate(max_images);
    }

    if paths.is_empty() {
        bail!("no supported image files found in {}", dir.display());
    }

    Ok(paths)
}

fn load_images(paths: &[PathBuf]) -> Result<Vec<Array3<f32>>> {
    paths.iter().map(|path| load_rgb_image(path)).collect()
}

fn image_dimensions(path: &Path) -> Result<(usize, usize)> {
    let (width, height) = image::image_dimensions(path)
        .with_context(|| format!("failed to read image dimensions for {}", path.display()))?;
    Ok((height as usize, width as usize))
}

fn load_rgb_image(path: &Path) -> Result<Array3<f32>> {
    let img = image::open(path)
        .with_context(|| format!("failed to open image {}", path.display()))?
        .into_rgb8();

    let (w, h) = img.dimensions();
    let raw = img.into_raw();
    let data: Vec<f32> = raw.into_iter().map(|v| v as f32 / 255.0).collect();

    Array3::from_shape_vec((h as usize, w as usize, 3), data)
        .with_context(|| format!("failed to map image array for {}", path.display()))
}

fn save_rgb_image(path: &Path, image: &Array3<f32>) -> Result<()> {
    let (h, w, ch) = image.dim();
    if ch != 3 {
        bail!("expected RGB image with 3 channels, got {}", ch);
    }

    let raw: Vec<u8> = image
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();

    let buffer = RgbImage::from_raw(w as u32, h as u32, raw)
        .ok_or_else(|| anyhow::anyhow!("failed to construct RGB output buffer"))?;

    buffer
        .save(path)
        .with_context(|| format!("failed to save image {}", path.display()))
}

fn save_heatmap(path: &Path, map: &Array2<f32>) -> Result<()> {
    let (h, w) = map.dim();
    let (min, max) = map
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &value| {
            (min.min(value), max.max(value))
        });
    let range = (max - min).max(1e-10);

    let raw: Vec<u8> = map
        .iter()
        .map(|&value| (((value - min) / range).clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();

    let buffer: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::from_raw(w as u32, h as u32, raw)
        .ok_or_else(|| anyhow::anyhow!("failed to construct heatmap buffer"))?;

    buffer
        .save(path)
        .with_context(|| format!("failed to save heatmap {}", path.display()))
}

fn save_carrier_mask(path: &Path, image_size: usize, carriers: &[AnalyzeCarrier]) -> Result<()> {
    let center = image_size as i32 / 2;
    let mut raw = vec![0u8; image_size * image_size];
    for carrier in carriers {
        let y = carrier.freq.0 + center;
        let x = carrier.freq.1 + center;
        if (0..image_size as i32).contains(&y) && (0..image_size as i32).contains(&x) {
            raw[y as usize * image_size + x as usize] = 255;
        }
    }

    let buffer: ImageBuffer<Luma<u8>, Vec<u8>> =
        ImageBuffer::from_raw(image_size as u32, image_size as u32, raw)
            .ok_or_else(|| anyhow::anyhow!("failed to construct carrier mask buffer"))?;
    buffer
        .save(path)
        .with_context(|| format!("failed to save carrier mask {}", path.display()))
}

fn is_supported_image(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_ascii_lowercase()),
        Some(ext) if matches!(ext.as_str(), "png" | "jpg" | "jpeg" | "webp" | "bmp" | "tiff" | "tif")
    )
}

fn choose_extract_size(scales: &[u32]) -> usize {
    scales.iter().copied().max().unwrap_or(512).min(4096) as usize
}

fn parse_bypass_mode(mode: &str) -> Result<BypassMode> {
    match mode.to_ascii_lowercase().as_str() {
        "v1" => Ok(BypassMode::V1),
        "v2" => Ok(BypassMode::V2),
        "v3" => Ok(BypassMode::V3),
        _ => bail!("unsupported bypass mode: {mode}"),
    }
}

fn bypass_mode_label(mode: BypassMode) -> &'static str {
    match mode {
        BypassMode::V1 => "v1",
        BypassMode::V2 => "v2",
        BypassMode::V3 => "v3",
    }
}

fn best_set_label(best_set: BestSet) -> &'static str {
    match best_set {
        BestSet::Dark => "dark",
        BestSet::White => "white",
    }
}

fn print_json<T: Serialize>(value: &T) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

fn vertical_profile(map: &Array2<f32>) -> Vec<f32> {
    let cx = map.ncols() / 2;
    (0..map.nrows()).map(|y| map[[y, cx]]).collect()
}

fn radial_profile(map: &Array2<f32>) -> Vec<f32> {
    let h = map.nrows() as i32;
    let w = map.ncols() as i32;
    let cy = h / 2;
    let cx = w / 2;
    let max_r = cy.min(cx).max(0) as usize;
    let mut sums = vec![0.0_f32; max_r + 1];
    let mut counts = vec![0usize; max_r + 1];

    for y in 0..h {
        for x in 0..w {
            let dy = y - cy;
            let dx = x - cx;
            let r = (((dy * dy + dx * dx) as f64).sqrt().round() as usize).min(max_r);
            sums[r] += map[[y as usize, x as usize]];
            counts[r] += 1;
        }
    }

    sums.into_iter()
        .zip(counts)
        .map(|(sum, count)| if count == 0 { 0.0 } else { sum / count as f32 })
        .collect()
}

fn mean_f32<I>(values: I) -> f32
where
    I: IntoIterator<Item = f32>,
{
    let mut sum = 0.0_f32;
    let mut count = 0usize;
    for value in values {
        sum += value;
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f32
    }
}
