"""Layer 2: batch synthesise N FOVs from a WSI, with per-FOV ground truth.

Thin loop over `Camera` — Camera owns sampling / QFW / augment; generator owns
file naming + gt.csv writing.
"""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import asdict
from typing import Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image

# ── utilities/ on sys.path so TissuesRegionsMask is importable ───────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_UTILITIES = os.path.abspath(os.path.join(_HERE, '..', 'utilities'))
if _UTILITIES not in sys.path:
    sys.path.insert(0, _UTILITIES)

from TissuesRegionsMask import TissuesRegionsMask   # noqa: E402

from config  import DomainGapConfig          # noqa: E402
from record  import FOVRecord                # noqa: E402
from camera  import Camera, CameraShot       # noqa: E402


def _slide_tag(wsi_path: str, max_len: int = 12) -> str:
    return os.path.splitext(os.path.basename(wsi_path))[0][:max_len]


def _build_base_mask(
    wsi,
    mask_ds:          float,
    mask_method,
    min_region_ratio: float,
    max_pixels:       Optional[int] = None,
    read_max_pixels:  Optional[int] = None,
) -> TissuesRegionsMask:
    """The level-INDEPENDENT half of mask prep: build + filter_regions + merge.

    Nothing here looks at a Camera, so the result is valid for every pyramid
    level of this WSI and a caller that sweeps levels should build it once.
    from_wsi is the expensive step -- a full level read plus segmentation,
    and with a DL method (HEST) a forward pass over the whole image -- so
    repeating it per level costs level_count times more for no new
    information. Pushes 2 snapshots onto the region undo stack.
    """
    mask = TissuesRegionsMask.from_wsi(wsi, ds=mask_ds, method=mask_method,
                                       max_pixels=max_pixels,
                                       read_max_pixels=read_max_pixels)
    mask.filter_regions(min_ratio=min_region_ratio)
    mask.merge_overlapping()
    return mask


def _prep_mask_for_camera(
    cam:              Camera,
    mask_ds:          float,
    mask_method,
    min_region_ratio: float,
    max_pixels:       Optional[int] = None,
) -> TissuesRegionsMask:
    """Build mask + apply the three region-prep mutations. Caller undoes them.

    filter_patchable uses the Camera's `required_region_side_l0` (= FoV +
    non-protruding padding) so surviving regions can host at least the
    strict portion of the bounding-square read. It is the only level-dependent
    step; callers sweeping levels on one WSI should call `_build_base_mask`
    once themselves and apply filter_patchable per level instead of calling
    this.
    """
    mask = _build_base_mask(cam.wsi, mask_ds, mask_method, min_region_ratio,
                            max_pixels=max_pixels)
    mask.filter_patchable(tile_size=cam.required_region_side_l0, ds=1.0)
    return mask


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
    mask_ds:       float  = 32.0,
    mask_method            = None,
    min_region_ratio: float = 0.01,
) -> List[FOVRecord]:
    """Generate `n` synthetic FOVs into `out_dir/images/` + `out_dir/gt.csv`."""
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

    print(f'Building tissue mask (ds={mask_ds}) ...', flush=True)
    cam.mask = _prep_mask_for_camera(
        cam,
        mask_ds=mask_ds,
        mask_method=mask_method,
        min_region_ratio=min_region_ratio,
    )
    print(f'            tissue_frac={cam.mask.tissue_fraction()*100:.1f}%, '
          f'usable_regions={len(cam.mask.tissue_regions)}', flush=True)

    slide_tag = _slide_tag(wsi_path)
    img_dir   = os.path.join(out_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    gt_path   = os.path.join(out_dir, 'gt.csv')

    records: List[FOVRecord] = []
    try:
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
    finally:
        cam.mask.regions_undo()  # filter_patchable
        cam.mask.regions_undo()  # merge_overlapping
        cam.mask.regions_undo()  # filter_regions

    if not records:
        raise RuntimeError('No FOV accepted. Check WSI, tissue_ratio, mask_ds.')

    with open(gt_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))

    print(f'\n{len(records)} synthetic FOVs -> {img_dir}')
    print(f'GT -> {gt_path}')
    return records


def generate_iter(
    wsi_path:     str,
    n:            int,
    cfg:          Optional[DomainGapConfig] = None,
    seed:         int    = 0,
    tissue_ratio: float  = 0.3,
    region_protrusion_ratio: float = 0.5,
    mask_ds:      float  = 32.0,
    mask_method            = None,
    min_region_ratio: float = 0.01,
) -> Iterator[Tuple[np.ndarray, FOVRecord]]:
    """In-memory streaming variant (no disk writes). Yields (image, FOVRecord)."""
    cfg = cfg or DomainGapConfig()
    cam = Camera(wsi_path, cfg=cfg, seed=seed,
                 tissue_ratio=tissue_ratio,
                 region_protrusion_ratio=region_protrusion_ratio)
    cam.mask = _prep_mask_for_camera(
        cam,
        mask_ds=mask_ds,
        mask_method=mask_method,
        min_region_ratio=min_region_ratio,
    )

    slide_tag = _slide_tag(wsi_path)

    idx = 0
    try:
        for shot in cam:
            if idx >= n:
                break
            fname = f'{slide_tag}_syn{idx:05d}.png'
            yield shot.image, _record_from_shot(
                shot, fname, wsi_path, cfg, cam.output_w, cam.output_h,
            )
            idx += 1
    finally:
        cam.mask.regions_undo()
        cam.mask.regions_undo()
        cam.mask.regions_undo()
