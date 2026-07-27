#!/usr/bin/env python3
"""Batch synthesise N FOVs from a WSI, with per-FOV ground truth CSV.

Position sampling uses TissuesRegionsMask (region-first, same pattern as
TileSampler); crop shape is QueryFromWSI (wh_ratio + MPixels + query_mpp).

Usage:
    python query_sim/cli/batch.py <wsi_path> \\
        [--n 300] [--wh-ratio 4:3] [--MPixels 12] [--mpp 0.25] [--seed 0]
        [--tissue-ratio 0.5] [--mask-ds 32]
        [--scale-min 0.9 --scale-max 1.15] [--no-distortion]
        [--no-geometric] [--no-photometric]

Outputs (in result/<SLURM_JOB_NAME or QuerySimBatch>/):
    images/S<slide>_syn00001.png ...
    gt.csv               one row per FOV (see FOVRecord)
"""

from __future__ import annotations

import argparse
import os
import sys

# ── query_sim/ onto sys.path so flat imports work ────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # parent = query_sim/

from cli       import job_result_dir        # noqa: E402
from config    import DomainGapConfig       # noqa: E402
from generator import generate              # noqa: E402


def main():
    ap = argparse.ArgumentParser(description='Generate synthetic microscope FOVs with GT.')
    ap.add_argument('wsi_path', help='WSI file (.svs / .ndpi / ...)')
    ap.add_argument('--n',        type=int,   default=300, help='Number of FOVs to generate')

    # ── QueryFromWSI shape (the microscope-photo spec) ────────────────────────
    ap.add_argument('--wh-ratio', default='4:3',    help='Output aspect ratio, e.g. 4:3')
    ap.add_argument('--MPixels',  type=float, default=12,   help='Total pixel budget (megapixels)')
    ap.add_argument('--mpp',      type=float, default=0.25, help='Target um/px of the output FOV')

    # ── Sampling / tissue mask ────────────────────────────────────────────────
    ap.add_argument('--tissue-ratio', type=float, default=0.5,
                    help='Min mask fraction inside a candidate crop (default 0.5)')
    ap.add_argument('--region-protrusion', type=float, default=0.5,
                    help='Fraction of bounding-square padding allowed to spill '
                         'past a tissue region edge (0=strict, 1=only FoV rect '
                         'must fit). Default 0.5.')
    ap.add_argument('--mask-ds',      type=float, default=32.0,
                    help='TissuesRegionsMask thumb downsample (default 32)')

    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out',  default=None,
                    help='Output dir (default: result/<SLURM_JOB_NAME or QuerySimBatch>)')

    # ── DomainGapConfig overrides most useful for calibration ─────────────────
    ap.add_argument('--scale-min',      type=float, default=None)
    ap.add_argument('--scale-max',      type=float, default=None)
    ap.add_argument('--angle-jitter',   type=float, default=None,
                    help='Max +/- jitter (deg) on top of the 0/90/180/270 choice')
    ap.add_argument('--no-distortion',  action='store_true',
                    help='Force lens distortion k1 range to (0, 0)')
    ap.add_argument('--no-geometric',   action='store_true',
                    help='Skip rotation + scale entirely (photometric only)')
    ap.add_argument('--no-photometric', action='store_true',
                    help='Skip colour / lens / noise entirely (geometric only)')
    args = ap.parse_args()

    cfg = DomainGapConfig(
        wh_ratio  = args.wh_ratio,
        MPixels   = args.MPixels,
        query_mpp = args.mpp,
    )
    if args.scale_min is not None:
        cfg.scale_range = (args.scale_min, cfg.scale_range[1])
    if args.scale_max is not None:
        cfg.scale_range = (cfg.scale_range[0], args.scale_max)
    if args.angle_jitter is not None:
        cfg.angle_jitter_deg = args.angle_jitter
    if args.no_distortion:
        cfg.distortion_k1_range = (0.0, 0.0)
    if args.no_geometric:
        cfg.geometric = False
    if args.no_photometric:
        cfg.photometric = False

    out_dir = args.out or job_result_dir('QuerySimBatch')

    generate(
        wsi_path     = args.wsi_path,
        out_dir      = out_dir,
        n            = args.n,
        cfg          = cfg,
        seed         = args.seed,
        tissue_ratio = args.tissue_ratio,
        region_protrusion_ratio = args.region_protrusion,
        mask_ds      = args.mask_ds,
    )


if __name__ == '__main__':
    main()
