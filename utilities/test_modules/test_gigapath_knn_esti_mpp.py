#!/usr/bin/env python3
"""
Test GigaPathKnnEstiMpp by cropping queries at known MPPs (one per WSI level)
and measuring KNN estimation error.

Usage:
    python utilities/test_modules/test_gigapath_knn_esti_mpp.py <wsi_path> \\
        [--x X] [--y Y] [--tile T] [--samples N] [--k K] [--mpixels M]
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import openslide
import torch

from _paths import job_result_dir, setup_import_paths

setup_import_paths()
from QueryFromWSI import QueryFromWSI
from simulate_microscope_photo import simulate_microscope_photo
from GigaPathKnnEstiMpp import GigaPathKnnEstiMpp
from GigaPathFunc import gigapath_model, gigapath_encode
from TissuesRegionsMask import TissuesRegionsMask


_LEVEL_COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0', '#00BCD4']

def _level_color(level: int) -> str:
    return _LEVEL_COLORS[level % len(_LEVEL_COLORS)]


def _add_border(img: np.ndarray, color_hex: str, width: int = 6) -> np.ndarray:
    rgb = tuple(int(c * 255) for c in mcolors.to_rgb(color_hex))
    out = img.copy()
    out[:width, :]  = rgb
    out[-width:, :] = rgb
    out[:, :width]  = rgb
    out[:, -width:] = rgb
    return out


def plot_reference_bank(est: GigaPathKnnEstiMpp, n_per_level: int = 8, out: str = None):
    """Show sample tiles from each pyramid level in the reference bank."""
    levels = sorted(set(t.level for t in est.sampler.tiles))
    n_levels = len(levels)

    fig, axes = plt.subplots(n_levels, n_per_level,
                              figsize=(n_per_level * 1.6, n_levels * 1.9))
    if n_levels == 1:
        axes = axes[np.newaxis, :]

    for row, lv in enumerate(levels):
        lv_tiles = est.sampler.tiles_at_level(lv)
        color = _level_color(lv)
        mpp = lv_tiles[0].mpp if lv_tiles else 0.0

        for col in range(n_per_level):
            ax = axes[row, col]
            if col < len(lv_tiles):
                img = np.array(est.sampler.read_tile(lv_tiles[col]))
                ax.imshow(_add_border(img, color))
                ax.set_title(f'#{col}', fontsize=6)
            else:
                ax.axis('off')
            ax.set_xticks([]); ax.set_yticks([])

        axes[row, 0].set_ylabel(f'Level {lv}\nMPP={mpp:.3f}', fontsize=8)

    fig.suptitle('Reference Bank Samples (one row per level)', fontsize=11)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {out}')


def plot_knn_debug(est: GigaPathKnnEstiMpp, gt_mpp: float,
                   n_patches: int = 8, out: str = None):
    """
    For each query patch, show the patch and its k nearest reference tiles.
    Border color = reference tile's pyramid level.
    Title color = green (correct vote) / red (wrong vote).
    """
    if est.knn.last_indices is None:
        print('[SKIP] plot_knn_debug: call estimate() first')
        return

    indices      = est.knn.last_indices.numpy()   # [M, k]
    patch_labels = est.knn.last_patch_labels       # [M]
    k = indices.shape[1]

    query_patches = list(est.qc.iter_main())
    M = min(n_patches, len(query_patches))

    fig, axes = plt.subplots(M, 1 + k, figsize=((1 + k) * 1.5, M * 1.7))
    if M == 1:
        axes = axes[np.newaxis, :]

    for row in range(M):
        # ── Query patch ───────────────────────────────────────────────────────
        ax = axes[row, 0]
        ax.imshow(query_patches[row])
        ax.set_xticks([]); ax.set_yticks([])
        voted = patch_labels[row]
        correct = abs(voted - gt_mpp) / gt_mpp < 0.05
        ax.set_title(f'voted\n{voted:.3f}', fontsize=6,
                     color='green' if correct else 'red')
        if row == 0:
            ax.set_xlabel('Query', fontsize=7)

        # ── k nearest reference tiles ─────────────────────────────────────────
        for ci, ref_idx in enumerate(indices[row]):
            ax = axes[row, 1 + ci]
            tile = est.sampler.tiles[ref_idx]
            img  = np.array(est.sampler.read_tile(tile))
            ax.imshow(_add_border(img, _level_color(tile.level)))
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'L{tile.level}\n{tile.mpp:.3f}', fontsize=6)

    # Legend: level → color
    legend_handles = [
        mpatches.Patch(color=_level_color(lv),
                       label=f'Level {lv}  MPP={est.sampler.level_mpps[lv]:.3f}')
        for lv in range(est.wsi.level_count)
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=est.wsi.level_count, fontsize=8,
               bbox_to_anchor=(0.5, 0))

    fig.suptitle(
        f'KNN Debug  |  GT MPP={gt_mpp:.3f}  est={est.result.estimated_mpp:.3f}\n'
        f'Border = ref level  |  title color: green=correct  red=wrong',
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    if out:
        fig.savefig(out, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {out}')


def pct_error(estimated: float, ground_truth: float) -> float:
    return abs(estimated - ground_truth) / ground_truth * 100


def load_query(wsi_path: str, gt_mpp: float, x: int, y: int, mpixels: float):
    qwsi = QueryFromWSI(wsi_path, MPixels=mpixels, mpp=gt_mpp,
                        x_top_left=x, y_top_left=y)
    img = qwsi.load_query_image()
    if img is not None:
        img = simulate_microscope_photo(img)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('wsi_path')
    ap.add_argument('--x',          type=int,   default=31700)
    ap.add_argument('--y',          type=int,   default=33600)
    ap.add_argument('--tile',       type=int,   default=256)
    ap.add_argument('--samples',    type=int,   default=100)
    ap.add_argument('--k',          type=int,   default=11)
    ap.add_argument('--mpixels',    type=float, default=1.475)
    ap.add_argument('--batch-size', type=int,   default=32)
    ap.add_argument('--device',     default='auto')
    args = ap.parse_args()

    device = torch.device(
        args.device if args.device != 'auto'
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )

    wsi = openslide.OpenSlide(args.wsi_path)
    base_mpp = float(wsi.properties.get('openslide.mpp-x', 0))
    if base_mpp == 0:
        sys.exit('Error: WSI has no openslide.mpp-x metadata.')

    test_mpps = [base_mpp * ds for ds in wsi.level_downsamples]

    print(f'\nWSI      : {args.wsi_path}')
    print(f'Base MPP : {base_mpp:.4f}  |  Levels : {wsi.level_count}')
    print(f'Test MPPs: {[f"{m:.3f}" for m in test_mpps]}')
    print(f'Device   : {device}  |  KNN k : {args.k}')

    # ── Build encoder + estimator (ref bank built once) ───────────────────────
    print('\nLoading GigaPath model...')
    _model = gigapath_model(device)
    encoder = lambda patches: gigapath_encode(
        patches, _model, device, batch_size=args.batch_size
    )

    mask = TissuesRegionsMask.from_wsi(wsi)
    est = GigaPathKnnEstiMpp(
        wsi, encoder=encoder, mask=mask,
        tile_size=args.tile, samples_per_level=args.samples, k=args.k,
    )

    print('Building reference bank...')
    est.build_samples()
    est.build_ref_features()
    print(f'  reference tiles: {len(est.ref_mpps)}')

    _job_dir = job_result_dir('KnnEstiMppTest')
    plot_reference_bank(est, n_per_level=8,
                        out=os.path.join(_job_dir, 'knn_mpp__ref_bank.png'))
    print('=' * 65)

    # ── Test each level ───────────────────────────────────────────────────────
    knn_errors: list[float] = []
    gt_list: list[float] = []

    for lv, gt_mpp in enumerate(test_mpps):
        query_img = load_query(args.wsi_path, gt_mpp, args.x, args.y, args.mpixels)
        if query_img is None:
            print(f'\nGT MPP = {gt_mpp:.4f}  [SKIP] failed to load query image')
            continue

        result = est.estimate(query_img)

        if result.query_patch_count == 0:
            print(f'\nGT MPP = {gt_mpp:.4f}  [SKIP] no full query patches')
            continue

        knn_err = pct_error(result.estimated_mpp, gt_mpp)
        gt_list.append(gt_mpp)
        knn_errors.append(knn_err)

        status = 'OK  ' if knn_err < 5 else 'WARN'
        print(f'\nGT MPP = {gt_mpp:.4f}')
        print(
            f'  [{status}] KNN (k={args.k})  '
            f'est={result.estimated_mpp:.4f}  err={knn_err:.1f}%  '
            f'patches={result.query_patch_count}'
        )

        # Per-patch vote distribution (debug)
        patch_mpps = est.knn.last_patch_labels
        unique, counts = np.unique(patch_mpps, return_counts=True)
        votes = '  '.join(f'{m:.3f}×{c}' for m, c in zip(unique, counts))
        print(f'  patch votes: {votes}')

        plot_knn_debug(est, gt_mpp, n_patches=8,
                       out=os.path.join(_job_dir, f'knn_mpp__debug_lv{lv}.png'))

    if not gt_list:
        sys.exit('Error: no valid test cases.')

    print('\n' + '=' * 65)
    print(f'Summary  GigaPath KNN (k={args.k})')
    print(f'  mean error : {np.mean(knn_errors):.1f}%')
    print(f'  max  error : {np.max(knn_errors):.1f}%')

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(gt_list, knn_errors, '^-', label=f'KNN (k={args.k})', linewidth=2)
    ax.axhline(5, color='r', linestyle='--', linewidth=0.8, label='5% threshold')
    ax.set_xlabel('Ground Truth MPP')
    ax.set_ylabel('Estimation Error (%)')
    ax.set_title('GigaPath KNN MPP Estimation Error by Level')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(gt_list, gt_list, 'k--', linewidth=0.8, label='ideal')
    knn_ests = [gt * (1 + e / 100) for gt, e in zip(gt_list, knn_errors)]
    ax.scatter(gt_list, knn_ests, marker='^', s=80, label=f'KNN (k={args.k})')
    ax.set_xlabel('Ground Truth MPP')
    ax.set_ylabel('Estimated MPP')
    ax.set_title('GT vs Estimated MPP')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(_job_dir, 'knn_mpp__accuracy.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {out}')


if __name__ == '__main__':
    main()
