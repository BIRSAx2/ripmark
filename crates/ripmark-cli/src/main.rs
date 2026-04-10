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
    n_images: usize,
    image_size: usize,
    requested_scales: Vec<u32>,
    codebook_version: u32,
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
    top_carriers: Vec<AnalyzeCarrier>,
    top_coherent_bins: Vec<AnalyzeTopBin>,
    files_written: Vec<String>,
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
    let result = detect(image.view(), &codebook);

    let output = DetectOutput {
        image: image_path.display().to_string(),
        codebook: codebook_path.display().to_string(),
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
    let image_paths = collect_image_paths(image_dir, max_images)?;
    let images = load_images(&image_paths)?;
    let image_size = choose_extract_size(scales);

    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
    }

    let codebook = Codebook::build(&images, image_size, image_dir.display().to_string());
    codebook.save(output_path)?;

    let output = ExtractOutput {
        image_dir: image_dir.display().to_string(),
        output: output_path.display().to_string(),
        n_images: images.len(),
        image_size,
        requested_scales: scales.to_vec(),
        codebook_version: codebook.version,
        detection_threshold: codebook.detection_threshold,
        correlation_mean: codebook.correlation_mean,
        correlation_std: codebook.correlation_std,
    };

    if json {
        print_json(&output)?;
    } else {
        println!("extracted codebook: {}", output.output);
        println!("images:             {}", output.n_images);
        println!("image_size:         {}", output.image_size);
        println!("requested_scales:   {:?}", output.requested_scales);
        if verbose {
            println!("version:            {}", output.codebook_version);
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
        stages: result.stages,
    };

    if json {
        print_json(&output)?;
    } else {
        println!("wrote bypassed image: {}", output.output);
        println!("mode:                {}", output.mode);
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

    let summary_path = output_dir.join("summary.json");
    let magnitude_path = output_dir.join("magnitude_spectrum.png");
    let coherence_path = output_dir.join("phase_coherence_map.png");
    let pca_path = output_dir.join("pca_watermark.png");

    save_heatmap(&magnitude_path, &report.magnitude_spectrum)?;
    save_heatmap(&coherence_path, &report.phase_coherence_map)?;
    save_heatmap(&pca_path, &report.pca_watermark)?;

    let top_bins = top_coherent_bins(
        &report.phase_coherence_map,
        &report.magnitude_spectrum,
        20,
        5,
    );

    let summary = AnalyzeSummary {
        image_dir: image_dir.display().to_string(),
        output_dir: output_dir.display().to_string(),
        n_images: images.len(),
        image_size: report.image_size,
        top_carriers: report
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
            .collect(),
        top_coherent_bins: top_bins
            .into_iter()
            .map(|(freq, score)| AnalyzeTopBin { freq, score })
            .collect(),
        files_written: vec![
            summary_path.display().to_string(),
            magnitude_path.display().to_string(),
            coherence_path.display().to_string(),
            pca_path.display().to_string(),
        ],
    };

    fs::write(&summary_path, serde_json::to_vec_pretty(&summary)?)
        .with_context(|| format!("failed to write {}", summary_path.display()))?;

    if json {
        print_json(&summary)?;
    } else {
        println!("analyzed images: {}", summary.n_images);
        println!("output_dir:      {}", summary.output_dir);
        println!("summary:         {}", summary_path.display());
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
