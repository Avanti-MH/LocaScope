"""Layer 2: batch synthesise N FOVs from a WSI, with per-FOV ground truth.

Thin loop over `Camera` — Camera owns sampling / QFW / augment; generator owns
file naming + gt.csv writing.
"""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import asdict
from typing import List, Optional

from PIL import Image

# ── utilities/ on sys.path so TissueMask is importable ───────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_UTILITIES = os.path.abspath(os.path.join(_HERE, '..', 'utilities'))
if _UTILITIES not in sys.path:
    sys.path.insert(0, _UTILITIES)

from TissueMask import TissueMask   # noqa: E402

from config  import DomainGapConfig          # noqa: E402
from record  import FOVRecord                # noqa: E402
from camera  import Camera, CameraShot       # noqa: E402


def _slide_tag(wsi_path: str, max_len: int = 12) -> str:
    return os.path.splitext(os.path.basename(wsi_path))[0][:max_len]


def base_mask(wsi, masks) -> TissueMask:
    """The level-INDEPENDENT mask: the recipe's segmentation and region prep.

    `masks` is the caller's `TissueMaskConfig.MaskMaker` -- recipe, device and
    optional cache. Taken as an object and never imported, so query_sim stays
    importable without the segmenters (and torch) behind it.

    Nothing here looks at a Camera, so the result is valid for every pyramid
    level of the slide, and a caller that sweeps levels builds it once.
    """
    return masks.mask(wsi)[0]


def camera_mask(cam: Camera, base: TissueMask) -> TissueMask:
    """The one level-DEPENDENT step: only the regions that can host this
    camera's strict read (FoV + non-protruding padding). A view -- the base is
    untouched, so a level sweep derives each level from the same base."""
    return base.patchable(cam.required_region_side_l0)


def _record_from_shot(
    shot:     CameraShot,
    filename: str,
    wsi_path: str,
    cfg:      DomainGapConfig,
    fov_w:    int,
    fov_h:    int,
    level:    int = 0,
) -> FOVRecord:
    p = shot.params
    return FOVRecord(
        filename      = filename,
        wsi_path      = wsi_path,
        level         = level,
        wh_ratio      = cfg.wh_ratio,
        MPixels       = cfg.MPixels,
        query_mpp     = cfg.query_mpp,
        nominal_mpp   = cfg.query_mpp,
        effective_mpp = float(p['effective_mpp']),
        fov_width     = fov_w,
        fov_height    = fov_h,
        gt_x          = shot.gt_x,
        gt_y          = shot.gt_y,
        rot_deg           = int(p['rot_deg']),
        angle_jitter      = round(float(p['angle_jitter']), 3),
        scale             = round(float(p['scale']), 4),
        vignette_strength = round(float(p['vignette_strength']), 3),
        color_temp        = round(float(p['color_temp']), 3),
        brightness        = round(float(p['brightness']), 3),
        contrast          = round(float(p['contrast']), 3),
        distortion_k1     = round(float(p['distortion_k1']), 4),
        defocus_radius    = int(p['defocus_radius']),
        chromatic_shift   = int(p['chromatic_shift']),
        stage_shift_dx    = int(p['stage_shift_dx']),
        stage_shift_dy    = int(p['stage_shift_dy']),
        noise_sigma       = float(p['noise_sigma']),
        jpeg_quality      = int(p['jpeg_quality']),
    )


def generate(
    wsi_path:      str,
    out_dir:       str,
    n:             int,
    cfg:           Optional[DomainGapConfig] = None,
    seed:          int    = 0,
    tissue_ratio:  float  = 0.3,
    region_protrusion_ratio: float = 0.5,
    *,
    masks,
) -> List[FOVRecord]:
    """Generate `n` synthetic FOVs into `out_dir/images/` + `out_dir/gt.csv`.

    `masks` is a `TissueMaskConfig.MaskMaker`; see `base_mask`."""
    cfg = cfg or DomainGapConfig()

    cam = Camera(wsi_path, cfg=cfg, seed=seed,
                 tissue_ratio=tissue_ratio,
                 region_protrusion_ratio=region_protrusion_ratio)
    print(f'FOV spec  : {cfg.wh_ratio}  {cfg.MPixels}MP  @ mpp={cfg.query_mpp}', flush=True)
    print(f'            output {cam.output_w}x{cam.output_h} px, '
          f'level-0 rect {cam.rect_w_l0}x{cam.rect_h_l0}', flush=True)
    print(f'            bounding square side (level-0) = {cam.bounding_square_side_l0}  '
          f'(rotation-safe read window)', flush=True)
    print(f'            region_protrusion_ratio={cam.region_protrusion_ratio}  '
          f'required_region_side={cam.required_region_side_l0}', flush=True)

    print(f'Building tissue mask ({masks.cfg.seg_id()}) ...', flush=True)
    cam.mask = camera_mask(cam, base_mask(cam.wsi, masks))
    print(f'            tissue_frac={cam.mask.tissue_fraction()*100:.1f}%, '
          f'usable_regions={len(cam.mask.tissue_regions)}', flush=True)

    slide_tag = _slide_tag(wsi_path)
    img_dir   = os.path.join(out_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    gt_path   = os.path.join(out_dir, 'gt.csv')

    records: List[FOVRecord] = []
    for shot in cam:
        if len(records) >= n:
            break
        idx = len(records)
        fname = f'{slide_tag}_syn{idx:05d}.png'
        Image.fromarray(shot.image).save(os.path.join(img_dir, fname))
        records.append(_record_from_shot(
            shot, fname, wsi_path, cfg, cam.output_w, cam.output_h,
        ))
        print(f'  [saved] {len(records)}/{n}  {fname}', flush=True)

    if not records:
        raise RuntimeError('No FOV accepted. Check WSI, tissue_ratio and the '
                           'mask recipe.')

    with open(gt_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))

    print(f'\n{len(records)} synthetic FOVs -> {img_dir}')
    print(f'GT -> {gt_path}')
    return records
