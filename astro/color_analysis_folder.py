#!/usr/bin/env python3
"""
Batch color analysis for all images in a folder.

Outputs:
- colors.csv: dominant colors + percentages per image
- palettes/ (optional): palette PNG per image

Usage:
  python color_analysis_folder.py --input ./images --clusters 6 --sample 60000 --write-palettes

Dependencies:
  pip install pillow numpy scikit-learn
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw
from sklearn.cluster import MiniBatchKMeans


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def srgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def rgb_to_hsv_np(rgb01: np.ndarray) -> np.ndarray:
    """
    rgb01: (..., 3) in [0,1]
    returns hsv (..., 3) in [0,1]
    """
    r, g, b = rgb01[..., 0], rgb01[..., 1], rgb01[..., 2]
    mx = np.max(rgb01, axis=-1)
    mn = np.min(rgb01, axis=-1)
    diff = mx - mn

    h = np.zeros_like(mx)
    s = np.zeros_like(mx)
    v = mx

    # saturation
    nonzero = mx > 0
    s[nonzero] = diff[nonzero] / mx[nonzero]

    # hue
    mask = diff > 1e-12
    # when max is r
    idx = mask & (mx == r)
    h[idx] = ((g[idx] - b[idx]) / diff[idx]) % 6
    # when max is g
    idx = mask & (mx == g)
    h[idx] = ((b[idx] - r[idx]) / diff[idx]) + 2
    # when max is b
    idx = mask & (mx == b)
    h[idx] = ((r[idx] - g[idx]) / diff[idx]) + 4

    h = (h / 6.0) % 1.0
    hsv = np.stack([h, s, v], axis=-1)
    return hsv


def load_pixels(image_path: Path, max_side: int = 900) -> np.ndarray:
    """
    Load image, downscale for speed, return pixels as Nx3 uint8 (RGB).
    """
    im = Image.open(image_path).convert("RGBA")

    # Composite on white to handle transparency nicely
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    im = Image.alpha_composite(bg, im).convert("RGB")

    w, h = im.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        im = im.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    arr = np.asarray(im, dtype=np.uint8)
    pixels = arr.reshape(-1, 3)
    return pixels


def sample_pixels(pixels: np.ndarray, n: int, seed: int = 42) -> np.ndarray:
    if pixels.shape[0] <= n:
        return pixels
    rng = np.random.default_rng(seed)
    idx = rng.choice(pixels.shape[0], size=n, replace=False)
    return pixels[idx]


def dominant_colors_kmeans(
    pixels: np.ndarray,
    k: int,
    sample_n: int,
    seed: int = 42
) -> List[Tuple[Tuple[int, int, int], float]]:
    """
    Returns list of (rgb, proportion) sorted by proportion desc.
    """
    X = sample_pixels(pixels, sample_n, seed=seed).astype(np.float32)

    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        batch_size=2048,
        n_init="auto"
    )
    labels = kmeans.fit_predict(X)

    centers = np.clip(np.rint(kmeans.cluster_centers_), 0, 255).astype(np.int32)
    counts = np.bincount(labels, minlength=k).astype(np.float64)
    proportions = counts / counts.sum()

    # Sort by proportion
    order = np.argsort(-proportions)
    result = []
    for i in order:
        rgb = tuple(int(x) for x in centers[i])
        result.append((rgb, float(proportions[i])))
    return result


def avg_color(pixels: np.ndarray) -> Tuple[int, int, int]:
    m = pixels.mean(axis=0)
    return (int(round(m[0])), int(round(m[1])), int(round(m[2])))


def metrics_from_pixels(pixels: np.ndarray) -> Tuple[float, float]:
    """
    returns (avg_brightness, avg_saturation) in [0,1]
    brightness = V in HSV, saturation = S in HSV
    """
    rgb01 = (pixels.astype(np.float32) / 255.0)
    hsv = rgb_to_hsv_np(rgb01)
    avg_sat = float(np.mean(hsv[..., 1]))
    avg_bri = float(np.mean(hsv[..., 2]))
    return avg_bri, avg_sat


def write_palette_png(
    out_path: Path,
    colors: List[Tuple[Tuple[int, int, int], float]],
    width: int = 800,
    height: int = 140
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    x = 0
    for rgb, prop in colors:
        w = max(1, int(round(width * prop)))
        draw.rectangle([x, 0, x + w, height], fill=rgb)
        x += w

    # Fix any rounding gap
    if x < width:
        last_rgb = colors[0][0] if colors else (255, 255, 255)
        draw.rectangle([x, 0, width, height], fill=last_rgb)

    img.save(out_path)


def iter_images(folder: Path) -> List[Path]:
    files = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
    return sorted(files)


def main() -> None:
    ap = argparse.ArgumentParser(description="Color analysis for all images in a folder.")
    ap.add_argument("--input", required=True, help="Input folder containing images.")
    ap.add_argument("--output", default="colors.csv", help="CSV output path.")
    ap.add_argument("--clusters", type=int, default=6, help="Number of dominant colors (k).")
    ap.add_argument("--sample", type=int, default=60000, help="Number of pixels to sample per image.")
    ap.add_argument("--max-side", type=int, default=900, help="Downscale images so max side is this many pixels.")
    ap.add_argument("--write-palettes", action="store_true", help="Write palette PNGs to ./palettes/")
    ap.add_argument("--palette-dir", default="palettes", help="Directory for palette PNGs.")
    args = ap.parse_args()

    in_dir = Path(args.input)
    if not in_dir.exists() or not in_dir.is_dir():
        raise SystemExit(f"Input folder not found or not a folder: {in_dir}")

    images = iter_images(in_dir)
    if not images:
        raise SystemExit(f"No images found in: {in_dir}")

    out_csv = Path(args.output)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # CSV columns: image, avg_color, brightness, saturation, then k*(hex, r,g,b, pct)
    header = ["image", "avg_hex", "avg_r", "avg_g", "avg_b", "avg_brightness", "avg_saturation"]
    for i in range(1, args.clusters + 1):
        header += [f"c{i}_hex", f"c{i}_r", f"c{i}_g", f"c{i}_b", f"c{i}_pct"]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for img_path in images:
            try:
                pixels = load_pixels(img_path, max_side=args.max_side)
                dom = dominant_colors_kmeans(pixels, k=args.clusters, sample_n=args.sample)
                av = avg_color(pixels)
                bri, sat = metrics_from_pixels(pixels)

                row = [
                    str(img_path.relative_to(in_dir)),
                    srgb_to_hex(av), av[0], av[1], av[2],
                    round(bri, 4), round(sat, 4),
                ]

                for rgb, prop in dom:
                    row += [srgb_to_hex(rgb), rgb[0], rgb[1], rgb[2], round(prop * 100, 2)]

                writer.writerow(row)

                if args.write_palettes:
                    palette_path = Path(args.palette_dir) / (img_path.stem + "_palette.png")
                    write_palette_png(palette_path, dom)

                print(f"✓ {img_path.name}")

            except Exception as e:
                print(f"✗ {img_path.name}: {e}")

    print(f"\nDone. Wrote: {out_csv.resolve()}")
    if args.write_palettes:
        print(f"Palette images: {Path(args.palette_dir).resolve()}")


if __name__ == "__main__":
    main()
