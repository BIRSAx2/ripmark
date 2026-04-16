# ripmark

`ripmark` is a Rust port of the Python project `reverse-SynthID`. The goal is the same: inspect, detect, and remove the SynthID image watermark used by Gemini-generated images.

This repository is not a line-for-line port. The CLI and core library are implemented in Rust, the codebook format is `.ripbook` rather than `.npz`/`.pkl`, and some of the Python V3 spectral pipeline is still only partially matched.

## status

What works today:

- `detect` loads an image and a `.ripbook`, selects the best matching profile, and reports confidence and intermediate scores.
- `extract` builds multi-resolution codebooks from directories of reference or watermarked images.
- `bypass` supports `v1`, `v2`, and `v3`, writes an output image, and reports PSNR, SSIM, carrier-energy drop, and phase-coherence drop.
- `analyze` writes JSON summaries and image artifacts such as magnitude spectra, coherence maps, and PCA-derived outputs.
- `ripmark-core` has unit tests for FFT, denoising, codebook handling, detection, analysis, and bypass helpers.

What does not match the Python project yet:

- The Rust V3 path does not yet reproduce the quality of the Python `SpectralCodebook` pipeline.
- The Rust extractor stores native spectral profile data, but its training logic is still simpler than the Python codebook builder.
- `ripmark` cannot read the Python `.npz` spectral codebook directly.
- The benchmark path is present, but the current quality target from `reverse-SynthID` is not met.

## repository layout

```text
crates/ripmark-core/
  FFT, denoising, codebooks, detection, bypass, analysis

crates/ripmark-cli/
  CLI wrapper over ripmark-core

docs/plan.md
  implementation plan and remaining work

scripts/compare_reverse_synthid.py
  benchmark ripmark against a reverse-SynthID checkout
```

## build

```bash
cargo build --release -p ripmark
```

Run the CLI:

```bash
./target/release/ripmark --help
```

Run tests:

```bash
cargo test --workspace
```

## codebook model

The `.ripbook` file is the trained watermark profile used by `detect` and `bypass`.

Each file stores one or more `ResolutionProfile`s. A profile contains:

- target image resolution
- carrier reference phases and coherence values used by detection
- averaged reference noise and correlation statistics
- optional native-resolution spectral data used by the current Rust V3 path:
  - watermark magnitude estimate
  - phase template
  - phase consistency
  - content magnitude baseline

At runtime, `ripmark` selects the best profile for the input image dimensions. Exact resolution matches are preferred; otherwise the nearest aspect ratio and pixel count are used.

## basic usage

Build a codebook from Gemini image directories:

```bash
./target/release/ripmark extract \
  --black /path/to/gemini_black \
  --white /path/to/gemini_white \
  --watermarked /path/to/gemini_random \
  --output artifacts/synthid.ripbook
```

Detect:

```bash
./target/release/ripmark detect image.png \
  --codebook artifacts/synthid.ripbook
```

Bypass:

```bash
./target/release/ripmark bypass image.png \
  --codebook artifacts/synthid.ripbook \
  --output cleaned.png \
  --mode v3
```

Analyze:

```bash
./target/release/ripmark analyze /path/to/images \
  --output-dir artifacts/analysis
```

Use `--json` on any command for machine-readable output.

## benchmarking against reverse-SynthID

The repository includes a comparison script:

```bash
python3 scripts/compare_reverse_synthid.py \
  --reverse-repo /path/to/reverse-SynthID \
  --train-root /path/to/reverse-synthid-dataset \
  --output /tmp/ripmark-vs-reverse.json
```

The script will:

- ensure the reverse-SynthID virtual environment exists
- install Python dependencies if needed
- build `ripmark` in the selected Cargo profile
- train a Rust codebook from the supplied training directories
- evaluate both tools on `validation_images`

Current release-mode result on the checked-in `validation_images` set:

- `ripmark detect`: `208.23 ms`
- `reverse-SynthID detect`: `294.11 ms`
- `ripmark bypass`: `533.74 ms`
- `reverse-SynthID bypass`: `1033.86 ms`
- `ripmark` reference SSIM: `0.9300`
- `reverse-SynthID` reference SSIM: `0.9937`

If no training directories are provided, the script falls back to building the Rust codebook from `validation_images`. That fallback is only useful for smoke testing. It is not a meaningful evaluation setup.

## current limitations

- The current Rust V3 bypass is faster than the Python path in release builds, but it does not match the Python output quality on the checked-in validation set.
- A Rust codebook trained from a small number of images per resolution is not reliable enough for full-spectrum subtraction. The code currently guards against that by falling back to the safer carrier-only path.
- The extractor does not yet reproduce the Python reference builder's black/white cross-validation and compact spectral storage model.
- The codebook format is Rust-specific. `ripmark` writes `.ripbook`; `reverse-SynthID` writes `.npz` and `.pkl`.
- Detection and reporting are close enough to compare, but they are not yet strict parity.

## future work

- import Python `spectral_codebook_v3.npz` into Rust so both implementations can be tested against the same learned profile
- improve Rust-native codebook extraction so it matches the Python spectral builder more closely
- finish the exact-match V3 spectral subtraction path for well-trained profiles
- expand benchmarking on the Hugging Face dataset while keeping `validation_images` as evaluation only
- revisit codebook storage once the profile format stabilizes

## source project

This port is based on the local `reverse-SynthID` repository and the associated Hugging Face dataset used by that project.
