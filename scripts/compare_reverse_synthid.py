#!/usr/bin/env python3
"""
Benchmark and compare reverse-SynthID against ripmark on a shared validation set.

This script can be launched with any Python interpreter. It will ensure the
target reverse-SynthID repository has a usable `.venv`, install dependencies
from `requirements.txt` when needed, then re-exec itself inside that venv.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_RIPMARK_REPO = Path(__file__).resolve().parents[1]
DEFAULT_REVERSE_REPO = DEFAULT_RIPMARK_REPO.parent / "reverse-SynthID"
ENV_BOOTSTRAPPED = "RIPMARK_REVERSE_VENV_READY"

cv2 = None
np = None


@dataclass
class ToolDetect:
    elapsed_ms: float
    is_watermarked: bool
    confidence: float
    phase_match: float
    correlation: float | None
    structure_ratio: float | None
    carrier_strength: float | None


@dataclass
class ToolBypass:
    elapsed_ms: float
    psnr: float
    ssim: float
    phase_drop: float | None
    carrier_drop: float | None
    output_path: str


@dataclass
class ImageComparison:
    image: str
    ripmark_detect: ToolDetect
    reverse_detect: ToolDetect
    ripmark_bypass: ToolBypass
    reverse_bypass: ToolBypass
    ripmark_vs_reference_psnr: float
    ripmark_vs_reference_ssim: float
    reverse_vs_reference_psnr: float
    reverse_vs_reference_ssim: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ripmark against reverse-SynthID.")
    parser.add_argument(
        "--ripmark-repo",
        type=Path,
        default=DEFAULT_RIPMARK_REPO,
        help="Path to the ripmark repo",
    )
    parser.add_argument(
        "--reverse-repo",
        type=Path,
        default=DEFAULT_REVERSE_REPO,
        help="Path to the reverse-SynthID repo",
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=None,
        help="Validation image directory. Defaults to <reverse-repo>/validation_images",
    )
    parser.add_argument(
        "--train-root",
        type=Path,
        default=None,
        help="Optional dataset root used to build the ripmark codebook",
    )
    parser.add_argument(
        "--train-black",
        type=Path,
        default=None,
        help="Explicit black-reference directory for ripmark codebook extraction",
    )
    parser.add_argument(
        "--train-white",
        type=Path,
        default=None,
        help="Explicit white-reference directory for ripmark codebook extraction",
    )
    parser.add_argument(
        "--train-watermarked",
        type=Path,
        action="append",
        default=None,
        help="Explicit watermarked training directory for ripmark codebook extraction; repeatable",
    )
    parser.add_argument(
        "--ripmark-codebook",
        type=Path,
        default=Path("/tmp/ripmark-benchmark.ripbook"),
        help="Path to the generated ripmark benchmark codebook",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/ripmark-vs-reverse.json"),
        help="Where to write the benchmark JSON report",
    )
    parser.add_argument(
        "--cargo-profile",
        choices=["debug", "release"],
        default="release",
        help="Cargo profile to build and benchmark for ripmark",
    )
    return parser.parse_args()


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def run(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def run_streaming(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def ensure_reverse_repo(reverse_repo: Path) -> None:
    if not reverse_repo.exists():
        raise FileNotFoundError(f"reverse-SynthID repo not found: {reverse_repo}")
    if not reverse_repo.is_dir():
        raise NotADirectoryError(f"reverse-SynthID path is not a directory: {reverse_repo}")
    requirements = reverse_repo / "requirements.txt"
    if not requirements.exists():
        raise FileNotFoundError(f"requirements.txt not found in reverse repo: {requirements}")


def ensure_reverse_venv(args: argparse.Namespace) -> None:
    reverse_repo = args.reverse_repo.resolve()
    ensure_reverse_repo(reverse_repo)

    venv_dir = reverse_repo / ".venv"
    python_bin = venv_python(venv_dir)
    requirements = reverse_repo / "requirements.txt"

    if not python_bin.exists():
        print(f"[bootstrap] creating virtualenv at {venv_dir}", file=sys.stderr)
        run_streaming([sys.executable, "-m", "venv", str(venv_dir)], cwd=reverse_repo)
        print(f"[bootstrap] installing dependencies from {requirements}", file=sys.stderr)
        run_streaming([str(python_bin), "-m", "pip", "install", "-r", str(requirements)], cwd=reverse_repo)

    current = Path(sys.executable).resolve()
    if current != python_bin.resolve() or os.environ.get(ENV_BOOTSTRAPPED) != "1":
        env = os.environ.copy()
        env[ENV_BOOTSTRAPPED] = "1"
        cmd = [str(python_bin), str(Path(__file__).resolve()), *sys.argv[1:]]
        os.execve(str(python_bin), cmd, env)


def ensure_runtime_imports() -> None:
    global cv2, np
    if cv2 is not None and np is not None:
        return

    import cv2 as cv2_mod
    import numpy as np_mod

    cv2 = cv2_mod
    np = np_mod


def ripmark_binary(repo: Path, profile: str) -> Path:
    return repo / "target" / profile / "ripmark"


def ensure_ripmark_binary(repo: Path, profile: str) -> Path:
    cmd = ["cargo", "build", "-p", "ripmark"]
    if profile == "release":
        cmd.append("--release")
    run_streaming(cmd, cwd=repo)
    binary = ripmark_binary(repo, profile)
    if not binary.exists():
        raise FileNotFoundError(f"ripmark binary not found at {binary}")
    return binary


def image_dimensions(path: Path) -> tuple[int, int]:
    ensure_runtime_imports()
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    return int(img.shape[0]), int(img.shape[1])


def original_images(validation_dir: Path) -> list[Path]:
    return sorted(path for path in validation_dir.glob("*.png") if path.is_file())


def link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def collect_train_spec(args: argparse.Namespace) -> tuple[Path | None, Path | None, list[Path]]:
    train_black = args.train_black.resolve() if args.train_black else None
    train_white = args.train_white.resolve() if args.train_white else None
    train_watermarked = (
        [path.resolve() for path in args.train_watermarked] if args.train_watermarked else []
    )

    if args.train_root:
        root = args.train_root.resolve()
        candidates = {
            "black": [root / "gemini_black", root / "gemini_black_nb_pro"],
            "white": [root / "gemini_white", root / "gemini_white_nb_pro"],
            "watermarked": [root / "gemini_random", root / "gemini_watermarked"],
        }
        if train_black is None:
            train_black = next((path for path in candidates["black"] if path.exists()), None)
        if train_white is None:
            train_white = next((path for path in candidates["white"] if path.exists()), None)
        if not train_watermarked:
            train_watermarked = [path for path in candidates["watermarked"] if path.exists()]

    return train_black, train_white, train_watermarked


def build_ripmark_codebook(
    ripmark_repo: Path,
    ripmark_bin: Path,
    output_path: Path,
    train_black: Path | None,
    train_white: Path | None,
    train_watermarked: list[Path],
    fallback_validation_dir: Path,
) -> dict[str, Any]:
    if output_path.exists():
        output_path.unlink()

    if train_black or train_white or train_watermarked:
        cmd = [str(ripmark_bin), "extract"]
        if train_black:
            cmd.extend(["--black", str(train_black)])
        if train_white:
            cmd.extend(["--white", str(train_white)])
        for path in train_watermarked:
            cmd.extend(["--watermarked", str(path)])
        cmd.extend(
            [
                "--output",
                str(output_path),
                "--max-images",
                "250",
                "--scales",
                "256,512,1024",
                "--json",
            ]
        )
        proc = run(cmd, cwd=ripmark_repo)
        payload = json.loads(proc.stdout)
        return {
            "mode": payload.get("mode"),
            "profiles_added": payload.get("profiles_added", []),
            "source": "training_dirs",
        }

    groups: dict[tuple[int, int], list[Path]] = {}
    for image in original_images(fallback_validation_dir):
        groups.setdefault(image_dimensions(image), []).append(image)

    with tempfile.TemporaryDirectory(prefix="ripmark-bench-dataset-") as tmp:
        tmp_root = Path(tmp)
        added = []
        for idx, ((_h, _w), paths) in enumerate(sorted(groups.items())):
            group_dir = tmp_root / f"group_{idx}"
            group_dir.mkdir(parents=True, exist_ok=True)
            for path in paths:
                link_or_copy(path, group_dir / path.name)

            cmd = [
                str(ripmark_bin),
                "extract",
                "--watermarked",
                str(group_dir),
                "--output",
                str(output_path),
                "--max-images",
                str(len(paths)),
                "--scales",
                "256,512",
                "--json",
            ]
            proc = run(cmd, cwd=ripmark_repo)
            payload = json.loads(proc.stdout)
            added.extend(payload.get("profiles_added", []))
    return {
        "mode": "watermarked",
        "profiles_added": added,
        "source": "validation_fallback",
    }


def parse_ripmark_detect(ripmark_bin: Path, repo: Path, image: Path, codebook: Path) -> ToolDetect:
    start = time.perf_counter()
    proc = run(
        [str(ripmark_bin), "detect", str(image), "--codebook", str(codebook), "--json"],
        cwd=repo,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    payload = json.loads(proc.stdout)
    return ToolDetect(
        elapsed_ms=elapsed_ms,
        is_watermarked=payload["watermarked"],
        confidence=payload["confidence"],
        phase_match=payload["phase_match"],
        correlation=payload.get("correlation"),
        structure_ratio=payload.get("structure_ratio"),
        carrier_strength=payload.get("carrier_strength"),
    )


def parse_ripmark_bypass(
    ripmark_bin: Path,
    repo: Path,
    image: Path,
    codebook: Path,
    output: Path,
) -> ToolBypass:
    start = time.perf_counter()
    proc = run(
        [
            str(ripmark_bin),
            "bypass",
            str(image),
            "--codebook",
            str(codebook),
            "--output",
            str(output),
            "--mode",
            "v3",
            "--json",
        ],
        cwd=repo,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    payload = json.loads(proc.stdout)
    return ToolBypass(
        elapsed_ms=elapsed_ms,
        psnr=payload["psnr"],
        ssim=payload["ssim"],
        phase_drop=payload.get("phase_coherence_drop"),
        carrier_drop=payload.get("carrier_energy_drop"),
        output_path=payload["output"],
    )


def load_reverse_tools(reverse_repo: Path) -> tuple[Any, Any]:
    sys.path.insert(0, str(reverse_repo / "src" / "extraction"))
    from robust_extractor import RobustSynthIDExtractor
    from synthid_bypass import SpectralCodebook, SynthIDBypass

    extractor = RobustSynthIDExtractor()
    extractor.load_codebook(str(reverse_repo / "artifacts" / "codebook" / "robust_codebook.pkl"))

    codebook = SpectralCodebook()
    codebook.load(str(reverse_repo / "artifacts" / "spectral_codebook_v3.npz"))

    bypass = SynthIDBypass(extractor=extractor)
    return extractor, (bypass, codebook)


def parse_reverse_detect(extractor: Any, image: Path) -> ToolDetect:
    start = time.perf_counter()
    result = extractor.detect(str(image))
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return ToolDetect(
        elapsed_ms=elapsed_ms,
        is_watermarked=bool(result.is_watermarked),
        confidence=float(result.confidence),
        phase_match=float(result.phase_match),
        correlation=float(result.correlation),
        structure_ratio=float(result.structure_ratio),
        carrier_strength=float(result.carrier_strength),
    )


def parse_reverse_bypass(bypass: Any, codebook: Any, image: Path, output: Path) -> ToolBypass:
    start = time.perf_counter()
    result = bypass.bypass_v3_file(
        str(image),
        str(output),
        codebook,
        strength="aggressive",
        verify=True,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    phase_drop = None
    if result.detection_before and result.detection_after:
        phase_drop = float(result.detection_before["phase_match"]) - float(
            result.detection_after["phase_match"]
        )

    return ToolBypass(
        elapsed_ms=elapsed_ms,
        psnr=float(result.psnr),
        ssim=float(result.ssim),
        phase_drop=phase_drop,
        carrier_drop=None,
        output_path=str(output),
    )


def psnr(a: Any, b: Any) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def ssim(a: Any, b: Any) -> float:
    if a.ndim == 3:
        gray_a = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
        gray_b = 0.299 * b[:, :, 0] + 0.587 * b[:, :, 1] + 0.114 * b[:, :, 2]
    else:
        gray_a = a
        gray_b = b

    blk = 8
    rows, cols = gray_a.shape
    rc = (rows // blk) * blk
    cc = (cols // blk) * blk
    if rc == 0 or cc == 0:
        return 1.0

    a_blocks = gray_a[:rc, :cc].reshape(rc // blk, blk, cc // blk, blk)
    a_blocks = a_blocks.transpose(0, 2, 1, 3).reshape(-1, blk, blk)
    b_blocks = gray_b[:rc, :cc].reshape(rc // blk, blk, cc // blk, blk)
    b_blocks = b_blocks.transpose(0, 2, 1, 3).reshape(-1, blk, blk)

    mu_a = a_blocks.mean(axis=(1, 2))
    mu_b = b_blocks.mean(axis=(1, 2))
    var_a = a_blocks.var(axis=(1, 2))
    var_b = b_blocks.var(axis=(1, 2))
    cov_ab = ((a_blocks - mu_a[:, None, None]) * (b_blocks - mu_b[:, None, None])).mean(
        axis=(1, 2)
    )

    k1_sq = 0.0001
    k2_sq = 0.0009
    num = (2.0 * mu_a * mu_b + k1_sq) * (2.0 * cov_ab + k2_sq)
    den = (mu_a * mu_a + mu_b * mu_b + k1_sq) * (var_a + var_b + k2_sq)
    return float(np.mean(num / den))


def load_rgb01(path: Path) -> Any:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    ensure_runtime_imports()

    if args.validation_dir is None:
        args.validation_dir = args.reverse_repo / "validation_images"
    train_black, train_white, train_watermarked = collect_train_spec(args)

    ripmark_bin = ensure_ripmark_binary(args.ripmark_repo, args.cargo_profile)
    train_info = build_ripmark_codebook(
        args.ripmark_repo,
        ripmark_bin,
        args.ripmark_codebook,
        train_black,
        train_white,
        train_watermarked,
        args.validation_dir,
    )

    reverse_extractor, (reverse_bypass, reverse_codebook) = load_reverse_tools(args.reverse_repo)

    comparisons: list[ImageComparison] = []
    with tempfile.TemporaryDirectory(prefix="ripmark-vs-reverse-") as tmp:
        tmp_root = Path(tmp)
        for image in original_images(args.validation_dir):
            reference = args.validation_dir / "cleaned" / image.name
            if not reference.exists():
                continue

            rip_out = tmp_root / f"ripmark_{image.name}"
            rev_out = tmp_root / f"reverse_{image.name}"

            rip_detect = parse_ripmark_detect(ripmark_bin, args.ripmark_repo, image, args.ripmark_codebook)
            rev_detect = parse_reverse_detect(reverse_extractor, image)

            rip_bypass = parse_ripmark_bypass(
                ripmark_bin,
                args.ripmark_repo,
                image,
                args.ripmark_codebook,
                rip_out,
            )
            rev_bypass = parse_reverse_bypass(reverse_bypass, reverse_codebook, image, rev_out)

            ref_img = load_rgb01(reference)
            rip_img = load_rgb01(Path(rip_bypass.output_path))
            rev_img = load_rgb01(Path(rev_bypass.output_path))

            comparisons.append(
                ImageComparison(
                    image=image.name,
                    ripmark_detect=rip_detect,
                    reverse_detect=rev_detect,
                    ripmark_bypass=rip_bypass,
                    reverse_bypass=rev_bypass,
                    ripmark_vs_reference_psnr=psnr(rip_img, ref_img),
                    ripmark_vs_reference_ssim=ssim(rip_img, ref_img),
                    reverse_vs_reference_psnr=psnr(rev_img, ref_img),
                    reverse_vs_reference_ssim=ssim(rev_img, ref_img),
                )
            )

    return {
        "n_images": len(comparisons),
        "cargo_profile": args.cargo_profile,
        "training": {
            "source": train_info["source"],
            "mode": train_info["mode"],
            "black_dir": str(train_black) if train_black else None,
            "white_dir": str(train_white) if train_white else None,
            "watermarked_dirs": [str(path) for path in train_watermarked],
            "profiles_added": train_info["profiles_added"],
        },
        "ripmark": {
            "avg_detect_ms": mean([c.ripmark_detect.elapsed_ms for c in comparisons]),
            "avg_detect_confidence": mean([c.ripmark_detect.confidence for c in comparisons]),
            "avg_detect_phase_match": mean([c.ripmark_detect.phase_match for c in comparisons]),
            "avg_bypass_ms": mean([c.ripmark_bypass.elapsed_ms for c in comparisons]),
            "avg_bypass_psnr": mean([c.ripmark_bypass.psnr for c in comparisons]),
            "avg_bypass_ssim": mean([c.ripmark_bypass.ssim for c in comparisons]),
            "avg_phase_drop": mean(
                [c.ripmark_bypass.phase_drop for c in comparisons if c.ripmark_bypass.phase_drop is not None]
            ),
            "avg_ref_psnr": mean([c.ripmark_vs_reference_psnr for c in comparisons]),
            "avg_ref_ssim": mean([c.ripmark_vs_reference_ssim for c in comparisons]),
        },
        "reverse_synthid": {
            "avg_detect_ms": mean([c.reverse_detect.elapsed_ms for c in comparisons]),
            "avg_detect_confidence": mean([c.reverse_detect.confidence for c in comparisons]),
            "avg_detect_phase_match": mean([c.reverse_detect.phase_match for c in comparisons]),
            "avg_bypass_ms": mean([c.reverse_bypass.elapsed_ms for c in comparisons]),
            "avg_bypass_psnr": mean([c.reverse_bypass.psnr for c in comparisons]),
            "avg_bypass_ssim": mean([c.reverse_bypass.ssim for c in comparisons]),
            "avg_phase_drop": mean(
                [c.reverse_bypass.phase_drop for c in comparisons if c.reverse_bypass.phase_drop is not None]
            ),
            "avg_ref_psnr": mean([c.reverse_vs_reference_psnr for c in comparisons]),
            "avg_ref_ssim": mean([c.reverse_vs_reference_ssim for c in comparisons]),
        },
        "details": [asdict(c) for c in comparisons],
    }


def main() -> int:
    args = parse_args()
    args.ripmark_repo = args.ripmark_repo.resolve()
    args.reverse_repo = args.reverse_repo.resolve()
    if args.validation_dir is not None:
        args.validation_dir = args.validation_dir.resolve()

    ensure_reverse_venv(args)

    summary = benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))

    print(f"images:               {summary['n_images']}")
    print(f"ripmark detect ms:    {summary['ripmark']['avg_detect_ms']:.2f}")
    print(f"reverse detect ms:    {summary['reverse_synthid']['avg_detect_ms']:.2f}")
    print(f"ripmark bypass ms:    {summary['ripmark']['avg_bypass_ms']:.2f}")
    print(f"reverse bypass ms:    {summary['reverse_synthid']['avg_bypass_ms']:.2f}")
    print(f"ripmark ref SSIM:     {summary['ripmark']['avg_ref_ssim']:.4f}")
    print(f"reverse ref SSIM:     {summary['reverse_synthid']['avg_ref_ssim']:.4f}")
    print(f"report:               {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
