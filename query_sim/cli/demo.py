#!/usr/bin/env python3
"""Single-FOV demo: crop one query from a WSI and show every augment side-by-side.

Usage:
    python query_sim/cli/demo.py <wsi_path> [--x X] [--y Y] [--mpp MPP] [--MPixels M]

Outputs (in result/<SLURM_JOB_NAME or QuerySimDemo>/):
    query_image.png             raw crop (QueryFromWSI.crop)
    augmentation_effects.png    each augment individually + Camera.capture
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# ── query_sim/ onto sys.path so flat imports work ────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # parent = query_sim/

from cli              import job_result_dir                                      # noqa: E402
from config           import DomainGapConfig                                     # noqa: E402
from camera           import Camera                                              # noqa: E402
from source.wsi_query import QueryFromWSI                                        # noqa: E402
from augment.color    import apply_color, apply_color_temp, apply_brightness_contrast, apply_jpeg  # noqa: E402
from augment.field    import apply_vignette, apply_stage_shift                                     # noqa: E402
from augment.lens     import apply_distortion, apply_defocus, apply_chromatic                      # noqa: E402
from augment.geometry import apply_rotation, apply_scale                         # noqa: E402
from augment.noise    import apply_noise                                         # noqa: E402


def _print_capture_params(cfg, params: dict):
    """Pretty-print the sampled augment values for the Camera.capture panel
    alongside the cfg range each was drawn from."""
    if params is None:
        print('\nCamera.capture params: <capture returned None>')
        return
    rot_actual = params['rot_deg'] + params['angle_jitter']
    print('\nCamera.capture params (this shot | cfg range):')
    print(f'  rot_deg       = {params["rot_deg"]:>5}    '
          f'| choices={cfg.rotation_choices}')
    print(f'  angle_jitter  = {params["angle_jitter"]:+.3f} deg  '
          f'| +/- {cfg.angle_jitter_deg} deg   (actual angle = {rot_actual:+.3f} deg)')
    print(f'  scale         = {params["scale"]:.4f}   '
          f'| {cfg.scale_range}')
    print(f'  vignette      = {params["vignette_strength"]:.3f}    '
          f'| {cfg.vignette_range}')
    print(f'  color_temp    = {params["color_temp"]:+.3f}   '
          f'| {cfg.color_temp_range}')
    print(f'  brightness    = {params["brightness"]:+.3f}   '
          f'| {cfg.brightness_range}')
    print(f'  contrast      = {params["contrast"]:+.3f}   '
          f'| {cfg.contrast_range}')
    print(f'  distortion_k1 = {params["distortion_k1"]:+.4f}  '
          f'| {cfg.distortion_k1_range}')
    print(f'  defocus_r     = {params["defocus_radius"]}       '
          f'| fixed cfg.defocus_radius={cfg.defocus_radius}')
    print(f'  chromatic     = {params["chromatic_shift"]} px    '
          f'| fixed cfg.chromatic_shift={cfg.chromatic_shift}')
    print(f'  stage_shift   = ({params["stage_shift_dx"]:+d}, {params["stage_shift_dy"]:+d}) px '
          f'| +/- {cfg.stage_shift_max} px')
    print(f'  noise_sigma   = {params["noise_sigma"]}     '
          f'| fixed cfg.noise_sigma={cfg.noise_sigma}')
    print(f'  jpeg_quality  = {params["jpeg_quality"]}      '
          f'| fixed cfg.jpeg_quality={cfg.jpeg_quality}')
    print(f'  saturation    = {params["saturation"]}     '
          f'| fixed cfg.saturation={cfg.saturation}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('wsi_path', help='WSI file (.svs / .ndpi / ...)')
    ap.add_argument('--wh-ratio', default='4:3')
    ap.add_argument('--MPixels',  type=float, default=12)
    ap.add_argument('--mpp',      type=float, default=0.25)
    ap.add_argument('--x',        type=int,   default=0)
    ap.add_argument('--y',        type=int,   default=0)
    ap.add_argument('--rotation', type=float, default=None,
                    help='Force rotation (deg) on the Camera.capture panel; '
                         'None = cfg-driven random from (0/90/180/270) + jitter')
    ap.add_argument('--seed',     type=int,   default=0)
    args = ap.parse_args()

    out_dir = job_result_dir('QuerySimDemo')

    # ── 1. Raw crop via QFW (the "ideal microscope") ─────────────────────────
    qfw = QueryFromWSI(args.wsi_path,
                       wh_ratio=args.wh_ratio, MPixels=args.MPixels, mpp=args.mpp)
    pil_img = qfw.crop(args.x, args.y)
    if pil_img is None:
        sys.exit(f'Error: crop at ({args.x}, {args.y}) is out of WSI bounds '
                 f'(rect {qfw.rect_w_l0}x{qfw.rect_h_l0}, wsi {qfw.wsi.dimensions}).')

    img = np.array(pil_img)
    pil_img.save(os.path.join(out_dir, 'query_image.png'))
    print(f'Saved  {os.path.join(out_dir, "query_image.png")}')

    # ── 2. Full "microscope photograph" via Camera (raw crop + augment) ──────
    cfg = DomainGapConfig(wh_ratio=args.wh_ratio, MPixels=args.MPixels, query_mpp=args.mpp)
    cam = Camera(qfw.wsi, cfg=cfg, seed=args.seed)   # reuse the same openslide handle
    chained, params = cam.capture_with_gt(args.x, args.y, rotation=args.rotation)
    _print_capture_params(cfg, params)

    # ── 3. Individual effect panels ──────────────────────────────────────────
    panels = [
        ('Original',              img.copy()),
        ('rotation (15 deg)',     apply_rotation(img.copy(), 15)),
        ('scale (1.15x)',         apply_scale(img.copy(), 1.15)),
        ('distortion',            apply_distortion(img.copy())),
        ('defocus',               apply_defocus(img.copy())),
        ('chromatic',             apply_chromatic(img.copy())),
        ('vignette',              apply_vignette(img.copy())),
        ('stage_shift',           apply_stage_shift(img.copy())),
        ('color',                 apply_color(img.copy())),
        ('color_temp (+0.12)',    apply_color_temp(img.copy(), temp=0.12)),
        ('brightness_contrast',   apply_brightness_contrast(img.copy(), 0.1, 0.1)),
        ('noise',                 apply_noise(img.copy())),
        ('jpeg',                  apply_jpeg(img.copy())),
        ('Camera.capture',        chained),
    ]

    N_COLS = 5
    n_rows = (len(panels) + N_COLS - 1) // N_COLS
    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(N_COLS * 4, n_rows * 4))
    axes = np.atleast_2d(axes).flatten()

    for ax, (title, result_img) in zip(axes, panels):
        ax.imshow(result_img)
        is_chained = (title == 'Camera.capture')
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

    fig.suptitle('Microscope Simulation -- Augmentation Effects', fontsize=14)
    fig.tight_layout()
    out_path = os.path.join(out_dir, 'augmentation_effects.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved  {out_path}')


if __name__ == '__main__':
    main()
