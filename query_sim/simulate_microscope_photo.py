#!/usr/bin/env python3
"""
Usage:
    python simulate_microscope_photo.py <wsi_path> [--x X] [--y Y] [--mpp MPP]

Outputs two images to ./result/:
    query_image.png          — raw crop from QueryFromWSI
    augmentation_effects.png — original / individual effects / chained
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from QueryFromWSI import QueryFromWSI
from capture import apply_color, apply_noise, apply_jpeg
from field import apply_field_mask, apply_vignette, apply_stage_shift
from lens import apply_distortion, apply_defocus, apply_chromatic


def _as_rgb_uint8(img) -> np.ndarray:
    """Accept PIL Image or numpy array; return (H, W, 3) uint8 RGB."""
    if isinstance(img, Image.Image):
        return np.array(img.convert('RGB'))
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def simulate_microscope_photo(img):
    img = _as_rgb_uint8(img)
    img = apply_color(img)
    img = apply_noise(img)
    img = apply_jpeg(img)
    # img = apply_field_mask(img)
    img = apply_vignette(img)
    img = apply_stage_shift(img)
    img = apply_distortion(img)
    img = apply_defocus(img)
    img = apply_chromatic(img)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('wsi_path', help='WSI file (.svs / .ndpi / ...)')
    ap.add_argument('--WH_ratio', default='4:3')
    ap.add_argument('--MPixels', type=float, default=12)
    ap.add_argument('--mpp',     type=float, default=0.25)
    ap.add_argument('--x',       type=int,   default=0, dest='x_top_left')
    ap.add_argument('--y',       type=int,   default=0, dest='y_top_left')
    args = ap.parse_args()

    # ── 1. Query image from WSI ──────────────────────────────────────────────
    qwsi = QueryFromWSI(
        args.wsi_path,
        WH_ratio=args.WH_ratio,
        MPixels=args.MPixels,
        mpp=args.mpp,
        x_top_left=args.x_top_left,
        y_top_left=args.y_top_left,
    )
    qwsi.load_query_image()

    if qwsi.query_image is None:
        sys.exit('Error: load_query_image() returned None — check WSI path / mpp.')

    pil_img = qwsi.query_image
    img     = np.array(pil_img)        # (H, W, 3) uint8 RGB

    os.makedirs('result', exist_ok=True)

    # ── 2. Figure 1: raw query image ─────────────────────────────────────────
    pil_img.save('result/query_image.png')
    print('Saved  result/query_image.png')

    # ── 3. Individual effects + chained ─────────────────────────────────────
    panels = [
        ('Original',    img.copy()),
        ('distortion',  apply_distortion(img.copy())),
        ('defocus',     apply_defocus(img.copy())),
        ('chromatic',   apply_chromatic(img.copy())),
        ('field_mask',  apply_field_mask(img.copy())),
        ('vignette',    apply_vignette(img.copy())),
        ('stage_shift', apply_stage_shift(img.copy())),
        ('color',       apply_color(img.copy())),
        ('noise',       apply_noise(img.copy())),
        ('jpeg',        apply_jpeg(img.copy())),
        ('Chained (All)', simulate_microscope_photo(img.copy())),
    ]

    # ── 4. Figure 2: 3×4 subplot grid ───────────────────────────────────────
    N_COLS = 4
    n_rows = (len(panels) + N_COLS - 1) // N_COLS

    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(N_COLS * 4, n_rows * 4))
    axes = axes.flatten()

    for ax, (title, result_img) in zip(axes, panels):
        ax.imshow(result_img)
        is_chained = (title == 'Chained (All)')
        ax.set_title(
            title, fontsize=11,
            fontweight='bold' if is_chained else 'normal',
            color='steelblue' if is_chained else 'black',
        )
        ax.axis('off')
        if is_chained:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('steelblue')
                spine.set_linewidth(2)

    for ax in axes[len(panels):]:
        ax.axis('off')

    fig.suptitle('Microscope Simulation — Augmentation Effects', fontsize=14)
    fig.tight_layout()
    fig.savefig('result/augmentation_effects.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved  result/augmentation_effects.png')


if __name__ == '__main__':
    main()
