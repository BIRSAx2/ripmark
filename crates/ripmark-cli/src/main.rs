use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

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

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Detect { image, codebook, threshold } => {
            println!("detect: {} (codebook: {}, threshold: {})",
                image.display(), codebook.display(), threshold);
            todo!("ripmark_core::detect")
        }
        Command::Extract { image_dir, output, max_images, scales } => {
            println!("extract: {} → {} (max_images: {}, scales: {:?})",
                image_dir.display(), output.display(), max_images, scales);
            todo!("ripmark_core::codebook")
        }
        Command::Bypass { image, codebook, output, mode } => {
            println!("bypass: {} → {} (mode: {}, codebook: {})",
                image.display(), output.display(), mode, codebook.display());
            todo!("ripmark_core::bypass")
        }
        Command::Analyze { image_dir, output_dir } => {
            println!("analyze: {} → {}",
                image_dir.display(), output_dir.display());
            todo!("ripmark_core::analysis")
        }
    }
}
