#!/usr/bin/env python3
"""diag_container_retire -- the read-geometry facts stage 3's level-0
bookkeeping rests on.

    python utilities/cli/diagnostics/diag_container_retire.py \
        --slide bracs/test:BRACS_1413:1 --checks phase
    python utilities/cli/diagnostics/diag_container_retire.py --checks origins

Checks, each chosen with --checks:

    phase     how openslide samples a level at a non-multiple level-0
              location -- the premise every level-0 <-> level-n bookkeeping in
              stage 3 and in the synthetic GT rests on. A is read at a, B_d at
              a + d for d level-0 px, and B_d is predicted from A by each model:
                  floor     level px = floor(x / ds): B_d is A shifted by
                            floor((a+d)/ds) - floor(a/ds) whole px
                  round     the same with round()
                  bilinear  level coordinate x / ds kept fractional: B_d =
                            (1 - t) A + t A[+1 px], t = d / ds
              The model whose residual is near the JPEG noise is how openslide
              reads; the others are far off. No template matching, no fitting:
              a noise-free read-and-compare.
    origins   how far region origins fall from a level's pixel grid in real
              data: frac(region.x / ds) per region and native level, from the
              cached masks of --mask-cache-job (MppRoutingHead by default),
              a sample of slides per dataset. Under the bilinear model the old
              bookkeeping int(region.x / ds) is off by that fraction of a level
              pixel; under floor it is exact. Arithmetic only, no pixel read.

The three checks that compared the retired WsiTissuesContainer with its
replacement -- pixels (tile pixels identical), crops (stage-3 windows) and
localize (the old -frac * ds bias, and the new bookkeeping within 0.25 level
px) -- passed and went with the container; their numbers are in log/TODO.log.

Each --slide is <dataset>:<slide>:<level>, the dataset may be empty for a
name that is unique across datasets. Read-only: nothing is written to
any cache except the mask cache of --mask-cache-job, which only grows.
Output: one block per check, result/<job>/phase.csv and origins.csv.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import _paths                                                       # noqa: E402
_paths.setup_import_paths()

import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402

import Cache                                                        # noqa: E402
from AccessDatasets import locate                                   # noqa: E402
from SafeSlide import SafeSlide                                     # noqa: E402
from TissueMaskConfig import (MASK_RECIPES, MaskMaker,              # noqa: E402
                              add_mask_args, mask_cfg_from_args)
from _paths import job_result_dir                                   # noqa: E402

JOB_NAME = 'DiagContainerRetire'
CHECKS = ('phase', 'origins')


def write_csv(rows, path) -> None:
    if not rows:
        return
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    print(f'  {path}  ({len(rows)} rows)', flush=True)




# ══════════════════════════════════════════════════════════════════════════════
#  phase: how openslide samples a level
# ══════════════════════════════════════════════════════════════════════════════

PHASE_SIZE = 192          # level px, the compared block
PHASE_SPOTS = 6           # textured places per slide/level
#: d as a fraction of ds; d = ceil(f * ds) level-0 px, so a coarse level gets
#: several sub-pixel steps and ds 2 gets its only one (d = 1).
PHASE_FRACTIONS = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)


def _block(a, sx, sy):
    """The PHASE_SIZE block of `a` starting sx px right and sy px down."""
    return a[sy:sy + PHASE_SIZE, sx:sx + PHASE_SIZE]


def check_phase(slide, mask, level, seed, rows_out) -> None:
    ds = float(slide.level_downsamples[level])
    rng = np.random.default_rng(seed + 7)
    margin = (PHASE_SIZE + 4) * ds
    regions = [r for r in mask.tissue_regions if r.w > margin and r.h > margin]
    steps = sorted({max(1, math.ceil(f * ds)) for f in PHASE_FRACTIONS} - {math.ceil(ds)})
    found = 0
    res = {m: [] for m in ('floor', 'round', 'bilinear')}
    print(f'  phase     level {level}  ds {ds:g}  d = {steps} level-0 px', flush=True)
    for _ in range(PHASE_SPOTS * 20):
        if found >= PHASE_SPOTS or not regions:
            break
        r = regions[int(rng.integers(len(regions)))]
        k_x = round((r.x + ds + rng.integers(int(r.w - margin))) / ds)
        k_y = round((r.y + ds + rng.integers(int(r.h - margin))) / ds)
        # a = round(k * ds): A sits as near the level's own grid as a level-0
        # integer allows (exactly on it at an integer ds), so each model predicts
        # B_d from A without a second interpolation of its own
        ax, ay = round(k_x * ds), round(k_y * ds)
        big = (PHASE_SIZE + 2, PHASE_SIZE + 2)
        A = slide.read_region_rgb((ax, ay), level, big).astype(np.float64)
        if _block(A, 0, 0).mean(axis=2).std() < 15:
            continue
        found += 1
        for name in ('x', 'y'):
            a0 = ax if name == 'x' else ay
            for d in steps:
                loc = (ax + d, ay) if name == 'x' else (ax, ay + d)
                B = _block(slide.read_region_rgb(loc, level, big).astype(np.float64), 0, 0)
                pred = {}
                for m, fn in (('floor', math.floor), ('round', round)):
                    s = int(fn((a0 + d) / ds) - fn(a0 / ds))
                    pred[m] = _block(A, s, 0) if name == 'x' else _block(A, 0, s)
                t = d / ds
                A1 = _block(A, 1, 0) if name == 'x' else _block(A, 0, 1)
                pred['bilinear'] = (1 - t) * _block(A, 0, 0) + t * A1
                row = dict(slide=Path(getattr(slide, '_filename', '')).stem,
                           level=level, ds=ds, spot=found, axis=name, d=d,
                           t=round(t, 4))
                for m, p in pred.items():
                    e = float(np.abs(B - p).mean())
                    res[m].append(e)
                    row[f'resid_{m}'] = e
                rows_out.append(row)
    if not found:
        print('    no textured spot found', flush=True)
        return
    med = {m: float(np.median(v)) for m, v in res.items()}
    best = min(med, key=med.get)
    print(f'    {found} spots x 2 axes x {len(steps)} steps   median mean|residual| '
          + '  '.join(f'{m} {v:.2f}' for m, v in med.items())
          + f'   -> {best}', flush=True)


# ══════════════════════════════════════════════════════════════════════════════
#  origins: region origins against each level's pixel grid, from cached masks
# ══════════════════════════════════════════════════════════════════════════════

def check_origins(cache_job, seg, per_dataset, seed, rows_out) -> None:
    cfg = MASK_RECIPES[seg]
    root = Cache.cache_root(cache_job, 'mask') / cfg.seg_id()
    metas = sorted(glob.glob(str(root / '*' / 'mask_meta.json')))
    by_ds = {}
    for m in metas:
        with open(m) as fh:
            path = json.load(fh).get('wsi_path', '')
        if not os.path.exists(path):
            continue
        parts = Path(path).parts
        dataset = parts[parts.index('datasets') + 1] if 'datasets' in parts else '?'
        by_ds.setdefault(dataset, []).append(path)
    rng = np.random.default_rng(seed)
    masks = MaskMaker(cfg, cache_root=Cache.cache_root(cache_job, 'mask'))
    print(f'  origins   {root}  ({len(metas)} masks, {per_dataset} per dataset)',
          flush=True)
    for dataset, paths in sorted(by_ds.items()):
        pick = [paths[i] for i in sorted(rng.choice(len(paths),
                                                     min(per_dataset, len(paths)),
                                                     replace=False))]
        for path in pick:
            slide = SafeSlide(path)
            mask, hit = masks.mask(slide)
            if not hit:
                # MaskMaker segments on a miss; a census must not, so a miss
                # is reported and its regions are not used
                print(f'    {Path(path).name}: not in the cache -- skipped', flush=True)
                slide.close()
                continue
            base_mpp = float(slide.base_mpp)
            for level in range(1, slide.level_count):
                ds = float(slide.level_downsamples[level])
                for r in mask.tissue_regions:
                    fx, fy = (r.x / ds) % 1.0, (r.y / ds) % 1.0
                    rows_out.append(dict(
                        dataset=dataset, slide=Path(path).stem, level=level, ds=ds,
                        region_x=r.x, region_y=r.y, frac_x=fx, frac_y=fy,
                        # the old bookkeeping's error IF openslide is bilinear
                        old_err_um_if_bilinear=max(fx, fy) * ds * base_mpp))
            slide.close()
    masks.close()
    g = {}
    for r in rows_out:
        g.setdefault((r['dataset'], r['level']), []).append(r)
    for (dataset, level), grp in sorted(g.items()):
        f = np.array([max(r['frac_x'], r['frac_y']) for r in grp])
        um = np.array([r['old_err_um_if_bilinear'] for r in grp])
        print(f'    {dataset:22s} L{level}  ds {grp[0]["ds"]:<9.5g} {len(grp):>5d} regions  '
              f'frac > 0.01: {float((f > 0.01).mean()):6.1%}   frac median '
              f'{np.median(f):.3f} max {f.max():.3f}   -> old error if bilinear: '
              f'median {np.median(um):.2f} um, max {um.max():.2f} um', flush=True)


# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--slide', action='append', default=[],
                    help='<dataset>:<slide>:<level>, repeatable (phase)')
    ap.add_argument('--checks', nargs='+', default=list(CHECKS), choices=CHECKS)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--mask-cache-job', default='MppRoutingHead',
                    help="masks come from this job's cache")
    ap.add_argument('--origin-per-dataset', type=int, default=8,
                    help='slides sampled per dataset for origins')
    add_mask_args(ap)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(job_result_dir(JOB_NAME))
    out_dir.mkdir(parents=True, exist_ok=True)

    if 'origins' in args.checks:
        rows = []
        check_origins(args.mask_cache_job, args.seg, args.origin_per_dataset,
                      args.seed, rows)
        write_csv(rows, out_dir / 'origins.csv')

    if 'phase' in args.checks and args.slide:
        masks = MaskMaker(mask_cfg_from_args(args), device=device,
                          cache_root=Cache.cache_root(args.mask_cache_job, 'mask'))
        phase_rows = []
        for spec in args.slide:
            dataset, rest = spec.split(':', 1)
            name, level = rest.rsplit(':', 1)
            level = int(level)
            slide = SafeSlide(locate(name, dataset=dataset or None).path)
            mask, _ = masks.mask(slide)
            print(f'\n== {name}  level {level}  '
                  f'ds {float(slide.level_downsamples[level]):g}', flush=True)
            check_phase(slide, mask, level, args.seed, phase_rows)
            slide.close()
        masks.close()
        write_csv(phase_rows, out_dir / 'phase.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
