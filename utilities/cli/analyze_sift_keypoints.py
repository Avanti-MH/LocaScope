#!/usr/bin/env python3
"""Why does SIFT find 260k+ keypoints on some WSI crops and not others?

cv2.BFMatcher asserts the train descriptor count stays under
IMGIDX_ONE = 1 << 18 = 262144. On BRACS_1936 level 0 some crops blow past
that while others on the SAME slide at the SAME level match fine, so the
trigger is the tissue under the crop, not the slide or the resolution.

This runs no GigaPath and no retrieval. Synthetic cases read their WSI
counterpart straight from the recorded ground-truth position via
QueryFromWSI, so the query image and the WSI crop are the exact pair SIFT
would have been handed.

Usage:
    python utilities/cli/analyze_sift_keypoints.py \\
        --gt-csv     result/MultiBatch/gt.csv \\
        --images-dir result/MultiBatch/images \\
        --case BRACS_1936_L0_syn00000.png \\
        --case BRACS_1936_L0_syn00007.png \\
        --case S1137178,G7E,110926_L0_syn00000.png \\
        --photo /work/u26130998/datasets/Ki67/S1104360_ki67/1.bmp \\
        --out result/SiftKeypointStudy

Outputs, in --out:
    kp_spatial.png        keypoints drawn on each query / WSI crop pair
    kp_distributions.png  response and scale histograms per case
    kp_vs_level.png       same tissue, every pyramid level
    kp_vs_contrast.png    contrastThreshold sweep
    kp_summary.png        counts and density side by side
    kp_stats.csv          every number behind the plots
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import openslide
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT / 'utilities'))
sys.path.insert(0, str(_ROOT / 'query_sim'))

from source.wsi_query import QueryFromWSI            # noqa: E402

BFMATCHER_LIMIT = 1 << 18       # 262144, the cv2 assertion this study is about


@dataclass
class Case:
    name:      str
    query:     np.ndarray                 # the shot / photo itself
    wsi_crop:  Optional[np.ndarray] = None   # WSI at the ground-truth position
    wsi_path:  Optional[str]        = None
    gt_x:      Optional[int]        = None
    gt_y:      Optional[int]        = None
    level:     Optional[int]        = None
    stain:     str                  = '?'     # HE / IHC, for grouping
    kp_query:  list                 = field(default_factory=list)
    kp_crop:   list                 = field(default_factory=list)


# ── SIFT helpers ──────────────────────────────────────────────────────────────

def detect(img: np.ndarray, contrast: float = 0.04, nfeatures: int = 0):
    """Keypoints only. nfeatures=0 means uncapped, which is the point here."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=contrast)
    return sift.detect(gray, None)


def density_per_mp(n_kp: int, img: np.ndarray) -> float:
    return n_kp / (img.shape[0] * img.shape[1] / 1e6)


# ── case loading ──────────────────────────────────────────────────────────────

def _stain_of(wsi_path: str) -> str:
    return 'IHC (Ki67)' if '/Ki67/' in wsi_path else 'H&E (BRACS)'


def load_gt_cases(gt_csv: str, images_dir: str, names: List[str]) -> List[Case]:
    with open(gt_csv) as f:
        rows = {r['filename']: r for r in csv.DictReader(f)}

    cases = []
    for name in names:
        if name not in rows:
            print(f'[skip] {name} not in {gt_csv}')
            continue
        r = rows[name]
        query = np.array(Image.open(os.path.join(images_dir, name)).convert('RGB'))

        # The WSI counterpart: same shape spec, read at the recorded position.
        # This is the un-augmented content SIFT was asked to match against.
        qfw = QueryFromWSI(r['wsi_path'],
                           wh_ratio=r['wh_ratio'],
                           MPixels=float(r['MPixels']),
                           mpp=float(r['query_mpp']))
        pil = qfw.crop(int(r['gt_x']), int(r['gt_y']))
        crop = np.array(pil) if pil is not None else None
        qfw.wsi.close()

        cases.append(Case(
            name=os.path.splitext(name)[0],
            query=query, wsi_crop=crop,
            wsi_path=r['wsi_path'],
            gt_x=int(r['gt_x']), gt_y=int(r['gt_y']),
            level=int(r['level']),
            stain=_stain_of(r['wsi_path']),
        ))
        print(f'  loaded {name}: query {query.shape[1]}x{query.shape[0]}'
              + (f', wsi crop {crop.shape[1]}x{crop.shape[0]}' if crop is not None
                 else ', wsi crop FAILED'))
    return cases


def load_photo_cases(paths: List[str]) -> List[Case]:
    cases = []
    for p in paths:
        img = np.array(Image.open(p).convert('RGB'))
        cases.append(Case(name='REAL ' + os.path.basename(p), query=img,
                          stain='IHC (Ki67) real photo'))
        print(f'  loaded {p}: {img.shape[1]}x{img.shape[0]}')
    return cases


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_spatial(cases: List[Case], out_path: str) -> None:
    """Where the keypoints land. Uniform saturation vs clustered tells them apart."""
    fig, axes = plt.subplots(len(cases), 2, figsize=(13, 5.5 * len(cases)),
                             squeeze=False)
    for i, c in enumerate(cases):
        for j, (img, kps, tag) in enumerate((
            (c.query,    c.kp_query, 'query / photo'),
            (c.wsi_crop, c.kp_crop,  'WSI at GT position'),
        )):
            ax = axes[i][j]
            if img is None:
                ax.set_title(f'{c.name}\n{tag}: n/a')
                ax.axis('off')
                continue
            small = cv2.drawKeypoints(
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR), kps[:60000], None,
                color=(0, 255, 0),
            )
            ax.imshow(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            over = ' OVER BFMatcher LIMIT' if len(kps) >= BFMATCHER_LIMIT else ''
            ax.set_title(f'{c.name}\n{tag}: {len(kps):,} kp  '
                         f'({density_per_mp(len(kps), img):,.0f}/MP){over}',
                         fontsize=9,
                         color='crimson' if over else 'black')
            ax.axis('off')
    fig.suptitle('SIFT keypoints in place  (green dots, capped at 60k drawn)',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'  {os.path.basename(out_path)}')


def plot_distributions(cases: List[Case], out_path: str) -> None:
    """Response and scale. A flood of weak, fine-scale points means noise."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for c in cases:
        kps = c.kp_crop if c.kp_crop else c.kp_query
        if not kps:
            continue
        resp = np.array([k.response for k in kps])
        size = np.array([k.size for k in kps])
        axes[0].hist(resp, bins=120, histtype='step', lw=1.5,
                     label=f'{c.name} (n={len(kps):,})', density=True)
        axes[1].hist(size, bins=120, histtype='step', lw=1.5,
                     label=f'{c.name}', density=True)
    axes[0].set_xlabel('keypoint response (contrast)')
    axes[0].set_ylabel('density')
    axes[0].set_title('Response\nmass piled at the low end = weak detections')
    axes[0].set_yscale('log')
    axes[0].axvline(0.04, color='gray', ls=':', lw=1)
    axes[0].text(0.04, axes[0].get_ylim()[1], ' default contrastThreshold',
                 fontsize=7, color='gray', va='top')
    axes[1].set_xlabel('keypoint size (scale, px)')
    axes[1].set_title('Scale\nconcentrated at small sizes = fine texture')
    axes[1].set_yscale('log')
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle('What kind of keypoints are these?', fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  {os.path.basename(out_path)}')


def plot_vs_level(cases: List[Case], out_path: str, rows: List[dict]) -> None:
    """Same physical tissue, every pyramid level. Isolates resolution."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for c in cases:
        if c.wsi_path is None or c.gt_x is None:
            continue
        wsi = openslide.OpenSlide(c.wsi_path)
        base_mpp = float(wsi.properties.get(openslide.PROPERTY_NAME_MPP_X, 0.25))
        # Fix the level-0 footprint, then read it at each level.
        w_l0 = c.wsi_crop.shape[1] * (base_mpp * wsi.level_downsamples[c.level]) / base_mpp
        h_l0 = c.wsi_crop.shape[0] * (base_mpp * wsi.level_downsamples[c.level]) / base_mpp
        lvls, counts = [], []
        for lv in range(wsi.level_count):
            ds = wsi.level_downsamples[lv]
            w, h = int(w_l0 / ds), int(h_l0 / ds)
            if w < 32 or h < 32 or w * h > 60e6:
                continue
            img = np.array(wsi.read_region((c.gt_x, c.gt_y), lv, (w, h)).convert('RGB'))
            n = len(detect(img))
            lvls.append(lv)
            counts.append(n)
            rows.append({'case': c.name, 'analysis': 'vs_level', 'level': lv,
                         'mpp': round(base_mpp * ds, 4), 'width': w, 'height': h,
                         'contrast': 0.04, 'n_keypoints': n,
                         'per_MP': round(density_per_mp(n, img), 1)})
        wsi.close()
        ax.plot(lvls, counts, 'o-', label=c.name)
    ax.axhline(BFMATCHER_LIMIT, color='crimson', ls='--', lw=1.2,
               label=f'BFMatcher limit ({BFMATCHER_LIMIT:,})')
    ax.set_yscale('log')
    ax.set_xlabel('pyramid level (same physical area, coarser resolution)')
    ax.set_ylabel('SIFT keypoints')
    ax.set_title('Keypoint count vs resolution')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  {os.path.basename(out_path)}')


def plot_vs_contrast(cases: List[Case], out_path: str, rows: List[dict]) -> None:
    """How far contrastThreshold has to move to get under the limit."""
    thresholds = [0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.24]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for c in cases:
        img = c.wsi_crop if c.wsi_crop is not None else c.query
        counts = []
        for t in thresholds:
            n = len(detect(img, contrast=t))
            counts.append(n)
            rows.append({'case': c.name, 'analysis': 'vs_contrast', 'level': c.level,
                         'mpp': '', 'width': img.shape[1], 'height': img.shape[0],
                         'contrast': t, 'n_keypoints': n,
                         'per_MP': round(density_per_mp(n, img), 1)})
        ax.plot(thresholds, counts, 'o-', label=c.name)
    ax.axhline(BFMATCHER_LIMIT, color='crimson', ls='--', lw=1.2,
               label=f'BFMatcher limit ({BFMATCHER_LIMIT:,})')
    ax.axvline(0.04, color='gray', ls=':', lw=1)
    ax.text(0.04, ax.get_ylim()[1], ' cv2 default', fontsize=7,
            color='gray', va='top')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('contrastThreshold')
    ax.set_ylabel('SIFT keypoints')
    ax.set_title('Keypoint count vs contrastThreshold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  {os.path.basename(out_path)}')


def plot_summary(cases: List[Case], out_path: str) -> None:
    """Query vs WSI crop, counts and per-megapixel density."""
    names = [c.name for c in cases]
    x = np.arange(len(cases))
    q_n = [len(c.kp_query) for c in cases]
    c_n = [len(c.kp_crop) for c in cases]
    q_d = [density_per_mp(len(c.kp_query), c.query) for c in cases]
    c_d = [density_per_mp(len(c.kp_crop), c.wsi_crop) if c.wsi_crop is not None else 0
           for c in cases]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, (a, b, ylab, title) in zip(axes, (
        (q_n, c_n, 'SIFT keypoints', 'Absolute count'),
        (q_d, c_d, 'keypoints per megapixel', 'Density (size-independent)'),
    )):
        ax.bar(x - 0.2, a, width=0.4, label='query / photo', color='steelblue')
        ax.bar(x + 0.2, b, width=0.4, label='WSI at GT position', color='darkorange')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha='right', fontsize=8)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=8)
    axes[0].axhline(BFMATCHER_LIMIT, color='crimson', ls='--', lw=1.2)
    axes[0].text(len(cases) - 0.5, BFMATCHER_LIMIT, ' BFMatcher limit',
                 fontsize=8, color='crimson', va='bottom', ha='right')
    fig.suptitle('Query vs WSI keypoint counts  '
                 '(a large gap between the pair is a domain gap SIFT cannot bridge)',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  {os.path.basename(out_path)}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt-csv',     default=None)
    ap.add_argument('--images-dir', default=None)
    ap.add_argument('--case',  action='append', default=[],
                    help='filename from gt.csv; repeatable')
    ap.add_argument('--photo', action='append', default=[],
                    help='path to a real photo; repeatable')
    ap.add_argument('--out',   default='result/SiftKeypointStudy')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f'Output -> {args.out}\n')

    print('Loading cases ...')
    cases: List[Case] = []
    if args.case:
        if not (args.gt_csv and args.images_dir):
            sys.exit('--case needs --gt-csv and --images-dir')
        cases += load_gt_cases(args.gt_csv, args.images_dir, args.case)
    cases += load_photo_cases(args.photo)
    if not cases:
        sys.exit('no cases loaded')

    print('\nDetecting keypoints (uncapped) ...')
    rows: List[dict] = []
    for c in cases:
        c.kp_query = detect(c.query)
        c.kp_crop  = detect(c.wsi_crop) if c.wsi_crop is not None else []
        over = '  <-- OVER LIMIT' if len(c.kp_crop) >= BFMATCHER_LIMIT else ''
        print(f'  {c.name:44s} query {len(c.kp_query):>8,}   '
              f'wsi {len(c.kp_crop):>8,}{over}')
        for tag, img, kps in (('query', c.query, c.kp_query),
                              ('wsi',   c.wsi_crop, c.kp_crop)):
            if img is None:
                continue
            rows.append({'case': c.name, 'analysis': f'default_{tag}',
                         'level': c.level, 'mpp': '',
                         'width': img.shape[1], 'height': img.shape[0],
                         'contrast': 0.04, 'n_keypoints': len(kps),
                         'per_MP': round(density_per_mp(len(kps), img), 1)})

    print('\nPlotting ...')
    plot_spatial(cases,       os.path.join(args.out, 'kp_spatial.png'))
    plot_distributions(cases, os.path.join(args.out, 'kp_distributions.png'))
    plot_summary(cases,       os.path.join(args.out, 'kp_summary.png'))
    plot_vs_contrast(cases,   os.path.join(args.out, 'kp_vs_contrast.png'), rows)
    plot_vs_level(cases,      os.path.join(args.out, 'kp_vs_level.png'), rows)

    csv_path = os.path.join(args.out, 'kp_stats.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'  kp_stats.csv  ({len(rows)} rows)')


if __name__ == '__main__':
    main()
