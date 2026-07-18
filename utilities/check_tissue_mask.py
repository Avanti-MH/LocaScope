#!/usr/bin/env python3
"""
Visualize and save tissue mask for a WSI.

Usage:
    python check_tissue_mask.py <wsi_path> [--method hsv|otsu] [--sat S]
"""

import argparse
import os

import cv2
import numpy as np
import openslide
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from TissuesRegionsMask import TissuesRegionsMask, _mask_hsv, _mask_otsu


def overlay(rgb, mask, color=(0, 200, 80), alpha=0.45):
    out = rgb.copy()
    out[mask] = (out[mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('wsi_path')
    ap.add_argument('--method', choices=['hsv', 'otsu'], default='hsv')
    ap.add_argument('--sat',    type=int,   default=15)
    ap.add_argument('--out',    default='result/tissue_mask.png')
    ap.add_argument('--save_mask', default=None,
                    help='prefix to save mask (e.g. result/slide_mask)')
    args = ap.parse_args()

    wsi    = openslide.OpenSlide(args.wsi_path)
    vendor = wsi.properties.get('openslide.vendor', 'unknown')
    W0, H0 = wsi.level_dimensions[0]

    # Generate both masks for comparison
    sat = args.sat
    tm_hsv  = TissuesRegionsMask.from_wsi(wsi, method=lambda rgb: _mask_hsv(rgb, sat_thresh=sat))
    tm_otsu = TissuesRegionsMask.from_wsi(wsi, method=_mask_otsu)

    # Use the same get_thumbnail() call so thumb matches the mask shape exactly
    Ht, Wt  = tm_hsv.main_mask.shape[:2]
    thumb   = np.array(wsi.get_thumbnail((Wt, Ht)).convert('RGB'))
    # Ensure exact size match (get_thumbnail may round slightly)
    thumb   = cv2.resize(thumb, (Wt, Ht))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(thumb)
    axes[0].set_title(f'Thumbnail  [{vendor}]  {Wt}×{Ht}', fontsize=11)

    axes[1].imshow(overlay(thumb, tm_hsv.main_mask,  color=(0, 200, 80)))
    axes[1].set_title(
        f'HSV sat>{args.sat}  tissue={tm_hsv.tissue_fraction()*100:.1f}%'
        f'\nmpp={tm_hsv.mask_mpp:.2f}  ds={tm_hsv.mask_ds_x:.1f}', fontsize=11)

    axes[2].imshow(overlay(thumb, tm_otsu.main_mask, color=(30, 120, 255)))
    axes[2].set_title(
        f'Otsu (auto bg)  tissue={tm_otsu.tissue_fraction()*100:.1f}%'
        f'\nmpp={tm_otsu.mask_mpp:.2f}  ds={tm_otsu.mask_ds_x:.1f}', fontsize=11)

    for ax in axes:
        ax.axis('off')

    p1 = mpatches.Patch(color=(0, 200/255, 80/255),  alpha=0.7, label='HSV')
    p2 = mpatches.Patch(color=(30/255, 120/255, 1.0), alpha=0.7, label='Otsu')
    fig.legend(handles=[p1, p2], loc='lower center', ncol=2, fontsize=11)
    fig.suptitle(f'{os.path.basename(args.wsi_path)}  L0: {W0}×{H0}', fontsize=10)
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved  {args.out}')

    # save() not yet implemented on TissuesRegionsMask
    if args.save_mask:
        print(f'[SKIP] --save_mask not yet supported for TissuesRegionsMask')


if __name__ == '__main__':
    main()
