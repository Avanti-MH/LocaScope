#!/usr/bin/env python3
"""diag_container_retire -- the new retrieval read path against the container,
and the read-geometry facts its stage-3 bookkeeping depends on.

    python utilities/cli/diagnostics/diag_container_retire.py \
        --slide bracs/test:BRACS_1413:1 --checks pixels crops phase
    python utilities/cli/diagnostics/diag_container_retire.py --checks origins

Checks, each chosen with --checks:

    pixels    SlidingWinSimRot's build: the container read every region
              whole and cut its tiles (old) against SlideReader.read_grid +
              WsiFeaturesMap.from_grid_read (new). PASS needs the same regions,
              the same grids and every tile's PIXELS identical -- compared by
              an exact integer fingerprint, so encoder arithmetic cannot blur
              it. The feature difference is then reported beside the encoder's
              own noise floor (the old tiles encoded twice at two batch sizes),
              for context, not as the verdict.
    crops     stage 3: the window cut out of the container's region image (old,
              the deleted code frozen below) against the window now read on
              demand (new). At an integer ds they are one read of the same
              pixels and must be IDENTICAL. At a non-integer ds they are read
              from different origins, so how they should differ depends on how
              openslide samples -- reported, not judged, until `phase` says.
    localize  stage 3 on a query read at a known level-0 point P, booked the
              old way (crop at int(region / ds) + x0, times ds) and the new
              (read origin + px * ds), against P + (w / 2) * ds. Regions are
              taken by descending frac(region / ds) so the old bias, predicted
              -frac * ds, is exercised; PASS needs the new one within a
              quarter of a level pixel.
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

Each --slide is <dataset>:<slide>:<level>, the dataset may be empty for a
name that is unique across datasets. Read-only: nothing is written to
any cache except the mask cache of --mask-cache-job, which only grows.
Output: one block per check and result/<job>/container_retire.csv (pixels,
crops), phase.csv, origins.csv.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import _paths                                                       # noqa: E402
_paths.setup_import_paths()

import cv2                                                          # noqa: E402
import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402

import Cache                                                        # noqa: E402
from AccessDatasets import locate                                   # noqa: E402
from PatchingLib import (QueryPatchContainer, WsiFeaturesMap,       # noqa: E402
                         WsiTissuesContainer,
                         region_grids)
from SafeSlide import SafeSlide                                     # noqa: E402
from stage3_localization.SIFT_RANSAC import SiftRansacLocalizer                         # noqa: E402
from stage2_retrieval.StageInterface import Candidate, CandidateSet                     # noqa: E402
from SlideReader import SlideReader                                 # noqa: E402
from TileEncoderFunc import encoder_config                          # noqa: E402
from TissueMaskConfig import (MASK_RECIPES, MaskMaker,              # noqa: E402
                              add_mask_args, mask_cfg_from_args)
from _paths import job_result_dir                                   # noqa: E402

JOB_NAME = 'DiagContainerRetire'
#: The query a crop is sized for: CLAUDE.md's real photo.
QUERY_W, QUERY_H = 1440, 1024
CHECKS = ('pixels', 'crops', 'localize', 'phase', 'origins')


def write_csv(rows, path) -> None:
    if not rows:
        return
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    print(f'  {path}  ({len(rows)} rows)', flush=True)


# ══════════════════════════════════════════════════════════════════════════════
#  The OLD crop, frozen: SiftRansacLocalizer.read_wsi_crop as it was before
#  2026-10-05, cutting the window out of the container's region image.
# ══════════════════════════════════════════════════════════════════════════════

def old_crop(container, location, query_w, query_h, padding):
    tpc = container[location.best_region_index]
    ts = container.tile_size
    local_x = location.best_x - tpc.img_origin_x
    local_y = location.best_y - tpc.img_origin_y
    win_w = int(np.ceil(query_w / ts)) * ts
    win_h = int(np.ceil(query_h / ts)) * ts
    x0 = max(0, local_x - padding * ts)
    y0 = max(0, local_y - padding * ts)
    x1 = min(tpc.img.shape[1], local_x + win_w + padding * ts)
    y1 = min(tpc.img.shape[0], local_y + win_h + padding * ts)
    return (tpc.img[y0:y1, x0:x1].copy(),
            tpc.img_origin_x + x0, tpc.img_origin_y + y0)


# ══════════════════════════════════════════════════════════════════════════════
#  pixels
# ══════════════════════════════════════════════════════════════════════════════

#: Fixed integer weights: a tile's fingerprint is its pixels dotted with each
#: map, in int64 -- exact, so two tiles share a fingerprint only if they are
#: equal (bar a collision of four random 64-bit sums).
_FP_WEIGHTS = None


def fingerprint(tiles) -> torch.Tensor:
    """[N, 4] exact integer fingerprints of uint8 tiles, an EncodeFn: it takes
    what an encoder takes (a list of [T, T, 3] arrays or a [N, T, T, 3]
    tensor), so it rides the same two paths the encoder does."""
    global _FP_WEIGHTS
    t = (torch.as_tensor(np.stack(tiles)) if isinstance(tiles, list)
         else torch.as_tensor(tiles)).to(torch.int64)
    if _FP_WEIGHTS is None or _FP_WEIGHTS.shape[1:] != t.shape[1:]:
        g = torch.Generator().manual_seed(0)
        _FP_WEIGHTS = torch.randint(1, 1 << 20, (4,) + tuple(t.shape[1:]),
                                    generator=g, dtype=torch.int64)
    return torch.stack([(t * w).flatten(1).sum(1) for w in _FP_WEIGHTS], dim=1)


def check_pixels(container, mask, encoder, encoder_alt, reader, tile, row) -> bool:
    ds, level = container.ds, container.level
    regions = mask.patchable(tile * ds).tissue_regions
    same_regions = ([(r.x, r.y, r.w, r.h) for r in regions]
                    == [(r.x, r.y, r.w, r.h) for r in container.tissue_regions])
    grids = region_grids(regions, ds=ds, level=level, tile_size=tile, overlap=True)
    same_grids = [len(g) for g in grids] == [len(tp) for tp in container]
    geo = dict(ds=ds, level=level, tile_size=tile, overlap=True)

    def new_path(fn):
        return WsiFeaturesMap.from_grid_read(
            reader.read_grid(regions, grids, ds, tile=tile, offset=True, level=level),
            regions, grids, fn, **geo)

    old_fp, new_fp = container.to_features(fingerprint), new_path(fingerprint)
    n_tiles = old_fp.n_patches()
    n_diff = sum(int((a.features != b.features).any(dim=1).sum())
                 for a, b in zip(old_fp, new_fp))
    ok = same_regions and same_grids and n_diff == 0
    print(f'  pixels    regions same={same_regions} grids same={same_grids}  '
          f'{n_tiles:,} tiles, {n_diff} with different pixels   '
          f'{"PASS" if ok else "FAIL"}', flush=True)
    row.update(n_tiles=n_tiles, same_regions=same_regions, same_grids=same_grids,
               tiles_pixels_differ=n_diff, pixels_pass=ok)

    if encoder is not None:
        t0 = time.time()
        old = container.to_features(encoder)
        old_alt = container.to_features(encoder_alt)
        new = new_path(encoder)
        diff = max(float((a.features.float() - b.features.float()).abs().max())
                   for a, b in zip(old, new))
        floor = max(float((a.features.float() - b.features.float()).abs().max())
                    for a, b in zip(old, old_alt))
        print(f'  features  new vs old max|d| {diff:.2e};  the encoder\'s own noise '
              f'floor (old tiles, batch {encoder.cfg.batch_size} vs '
              f'{encoder_alt.cfg.batch_size}) {floor:.2e}   ({time.time() - t0:.0f}s)',
              flush=True)
        row.update(feat_maxdiff=diff, feat_noise_floor=floor)
    return ok


# ══════════════════════════════════════════════════════════════════════════════
#  crops
# ══════════════════════════════════════════════════════════════════════════════

def _window_candidate_set(container, grids, r, info):
    """The CandidateSet stage 2 would hand over for main window (row, col) of
    region r -- one candidate, rotation 0 -- in the container's frame."""
    c = Candidate(r, 'main', info.row, info.col, 0, 1.0)
    return CandidateSet(candidates=(c,), level=container.level, ds=container.ds,
                        grids=tuple(grids))


def _blank_query(w, h, tile):
    qc = QueryPatchContainer(np.zeros((h, w, 3), np.uint8))
    qc.extract_all(tile, overlap=True)
    return qc


def check_crops(container, reader, grids, level, n, padding, seed, row) -> bool:
    """The frozen old crop against SiftRansacLocalizer's, for the same window:
    same size, the new read origin is the old crop's position on the
    region's own level-0 phase, and -- at an integer ds -- the same pixels."""
    rng = np.random.default_rng(seed)
    ts = container.tile_size
    query = _blank_query(QUERY_W, QUERY_H, ts)
    loc = SiftRansacLocalizer(padding=padding).build(reader)
    integer = abs(container.ds - round(container.ds)) < 1e-9
    diffs, n_done = [], 0
    for _ in range(n):
        r = int(rng.integers(len(grids)))
        grid = grids[r]
        if not len(grid):
            continue
        info = grid.main_patch_infos[int(rng.integers(len(grid.main_patch_infos)))]
        location = SimpleNamespace(best_x=info.x, best_y=info.y,
                                   best_region_index=r, ds=container.ds)
        ref, ox, oy = old_crop(container, location, QUERY_W, QUERY_H, padding)
        cs = _window_candidate_set(container, grids, r, info)
        got = loc.prepare(query, cs, 0).read_wsi_crop()
        want_l0 = grid.local_to_l0(ox - grid.x_offset, oy - grid.y_offset)
        if got.shape != ref.shape or tuple(loc.crop_origin_l0) != tuple(want_l0):
            print(f'  crops     region {r} ({info.row}, {info.col}): shape '
                  f'{got.shape} vs {ref.shape}, read origin {loc.crop_origin_l0} vs '
                  f'{want_l0}   FAIL', flush=True)
            row['crops_pass'] = False
            return False
        diffs.append(float(np.abs(got.astype(np.int16) - ref.astype(np.int16)).mean()))
        n_done += 1
    worst = max(diffs) if diffs else float('nan')
    median = float(np.median(diffs)) if diffs else float('nan')
    ok = (worst == 0.0) if integer else True     # phase: see `phase`
    print(f'  crops     {n_done} windows, same shape and read origin; integer '
          f'ds={integer}  mean|d| worst {worst:.3f}, median {median:.3f}   '
          f'{"PASS" if ok else "FAIL"}', flush=True)
    row.update(n_crops=n_done, crop_mean_absdiff_worst=worst,
               crop_mean_absdiff_median=median, integer_ds=integer, crops_pass=ok)
    return ok


# ══════════════════════════════════════════════════════════════════════════════
#  localize: the old stage-3 bookkeeping against the new, on a query whose
#  level-0 position is known
# ══════════════════════════════════════════════════════════════════════════════

LOCALIZE_W, LOCALIZE_H = 1024, 768     # level px; textured enough for SIFT
LOCALIZE_PER_REGION = 3
#: The new error must be this far under the old one's prediction to call the
#: bias removed; SIFT itself is good to a fraction of a level pixel here.
LOCALIZE_TOL = 0.25                     # level px


def check_localize(slide, container, reader, grids, padding, seed, row) -> bool:
    """A patch read at a known level-0 point P is the query; stage 3 runs on
    the window that holds it, booked the old way (crop at int(region / ds) +
    x0, times ds) and the new way (the read origin plus px * ds). Truth for the
    centre is P + (w / 2) * ds -- openslide's bilinear rule, `phase`.

    Regions are taken by descending frac(region.x / ds), so the bias the old
    bookkeeping is predicted to carry, -frac * ds, is actually exercised."""
    ds, ts = float(container.ds), container.tile_size
    rng = np.random.default_rng(seed + 11)
    w, h = LOCALIZE_W, LOCALIZE_H
    span_x, span_y = (w + 4 * ts) * ds, (h + 4 * ts) * ds
    order = sorted(range(len(grids)), key=lambda i: -max(
        (container.tissue_regions[i].x / ds) % 1, (container.tissue_regions[i].y / ds) % 1))
    loc = SiftRansacLocalizer(padding=padding).build(reader)
    rows_out, worst_new, worst_old_vs_pred = [], 0.0, 0.0
    for r in order:
        region = container.tissue_regions[r]
        if region.w <= span_x or region.h <= span_y:
            continue
        grid = grids[r]
        for _ in range(LOCALIZE_PER_REGION):
            px = int(region.x + 2 * ts * ds + rng.integers(int(region.w - span_x)))
            py = int(region.y + 2 * ts * ds + rng.integers(int(region.h - span_y)))
            patch = slide.read_region_rgb((px, py), container.level, (w, h))
            if cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY).std() < 15:
                continue
            query = QueryPatchContainer(patch)
            query.extract_all(ts, overlap=True)
            # the main window whose top-left tile holds P
            lx, ly = (px - region.x) / ds, (py - region.y) / ds
            info = grid.main_patch_infos[
                min(int(ly // ts), grid.grid_rows - 1) * grid.grid_cols
                + min(int(lx // ts), grid.grid_cols - 1)]
            cs = _window_candidate_set(container, grids, r, info)
            res = loc.localize(query, cs, 0)
            if not res.success:
                continue
            truth = (px + w / 2.0 * ds, py + h / 2.0 * ds)
            # old bookkeeping: the same crop and H, booked at int(region/ds)+x0
            cx_px = (res.center_x0 - loc.crop_origin_l0[0]) / ds
            cy_px = (res.center_y0 - loc.crop_origin_l0[1]) / ds
            ox_ln = grid.x_offset + (round((loc.crop_origin_l0[0] - region.x) / ds))
            oy_ln = grid.y_offset + (round((loc.crop_origin_l0[1] - region.y) / ds))
            old = ((ox_ln + cx_px) * ds, (oy_ln + cy_px) * ds)
            pred = ((int(region.x / ds) - region.x / ds) * ds,
                    (int(region.y / ds) - region.y / ds) * ds)
            e_new = ((res.center_x0 - truth[0]) / ds, (res.center_y0 - truth[1]) / ds)
            e_old = ((old[0] - truth[0]) / ds, (old[1] - truth[1]) / ds)
            off = (e_old[0] - pred[0] / ds, e_old[1] - pred[1] / ds)
            worst_new = max(worst_new, abs(e_new[0]), abs(e_new[1]))
            worst_old_vs_pred = max(worst_old_vs_pred, abs(off[0]), abs(off[1]))
            rows_out.append((r, (region.x / ds) % 1, (region.y / ds) % 1,
                             pred[0] / ds, pred[1] / ds, e_old, e_new, res.inlier_count))
        if len(rows_out) >= 12:
            break
    print(f'  localize  ds {ds:g}  {len(rows_out)} queries  (errors in LEVEL px, '
          f'centre against P + w/2 * ds)', flush=True)
    for r, fx, fy, prx, pry, e_old, e_new, inl in rows_out:
        print(f'    region {r:>3d} frac ({fx:.3f}, {fy:.3f})  predicted old '
              f'({prx:+.3f}, {pry:+.3f})  old ({e_old[0]:+.3f}, {e_old[1]:+.3f})  '
              f'new ({e_new[0]:+.3f}, {e_new[1]:+.3f})  inliers {inl}', flush=True)
    ok = bool(rows_out) and worst_new < LOCALIZE_TOL
    print(f'    new worst |error| {worst_new:.3f} level px (< {LOCALIZE_TOL})  '
          f'old against its prediction {worst_old_vs_pred:.3f}   '
          f'{"PASS" if ok else "FAIL"}', flush=True)
    row.update(localize_n=len(rows_out), localize_new_worst=worst_new,
               localize_old_vs_pred=worst_old_vs_pred, localize_pass=ok)
    return ok


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
                    help='<dataset>:<slide>:<level>, repeatable (pixels, crops, phase)')
    ap.add_argument('--checks', nargs='+', default=list(CHECKS), choices=CHECKS)
    ap.add_argument('--encoder', default='uni2')
    ap.add_argument('--tile', type=int, default=256)
    ap.add_argument('--crops', type=int, default=40)
    ap.add_argument('--padding', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--mask-cache-job', default='MppRoutingHead',
                    help="masks come from this job's cache")
    ap.add_argument('--origin-per-dataset', type=int, default=8,
                    help='slides sampled per dataset for origins')
    add_mask_args(ap)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(job_result_dir(JOB_NAME))
    out_dir.mkdir(parents=True, exist_ok=True)
    ok_all = True

    if 'origins' in args.checks:
        rows = []
        check_origins(args.mask_cache_job, args.seg, args.origin_per_dataset,
                      args.seed, rows)
        write_csv(rows, out_dir / 'origins.csv')

    slide_checks = [c for c in args.checks if c != 'origins']
    if slide_checks and args.slide:
        masks = MaskMaker(mask_cfg_from_args(args), device=device,
                          cache_root=Cache.cache_root(args.mask_cache_job, 'mask'))
        encoder = encoder_alt = None
        if 'pixels' in slide_checks:
            # fp32: under fp16 two batch splits can move a feature by ~1e-3
            # and blur the comparison; the alternate batch size is the noise
            # floor the feature difference is read against.
            encoder = encoder_config(args.encoder).with_model(dtype='fp32').build(device)
            encoder_alt = encoder_config(args.encoder, batch_size=max(
                1, encoder.cfg.batch_size // 3)).with_model(dtype='fp32').build(device)
        rows, phase_rows = [], []
        for spec in args.slide:
            dataset, rest = spec.split(':', 1)
            name, level = rest.rsplit(':', 1)
            level = int(level)
            slide = SafeSlide(locate(name, dataset=dataset or None).path)
            mask, _ = masks.mask(slide)
            ds = float(slide.level_downsamples[level])
            print(f'\n== {name}  level {level}  ds {ds:g}', flush=True)
            if 'phase' in slide_checks:
                check_phase(slide, mask, level, args.seed, phase_rows)
            if {'pixels', 'crops', 'localize'} & set(slide_checks):
                reader = SlideReader(slide, workers=args.workers)
                t0 = time.time()
                container = WsiTissuesContainer.from_ds(slide, ds, tile_size=args.tile,
                                                        overlap=True, mask=mask)
                print(f'  old container read {time.time() - t0:.0f}s', flush=True)
                row = dict(slide=name, level=level, ds=ds)
                grids = region_grids(container.tissue_regions, ds=container.ds,
                                     level=container.level, tile_size=args.tile,
                                     overlap=True)
                if 'pixels' in slide_checks:
                    ok_all &= check_pixels(container, mask, encoder, encoder_alt,
                                           reader, args.tile, row)
                if 'crops' in slide_checks:
                    ok_all &= check_crops(container, reader, grids, container.level,
                                          args.crops, args.padding, args.seed, row)
                if 'localize' in slide_checks:
                    ok_all &= check_localize(slide, container, reader, grids,
                                             args.padding, args.seed, row)
                rows.append(row)
                del container
            slide.close()
        masks.close()
        write_csv(rows, out_dir / 'container_retire.csv')
        write_csv(phase_rows, out_dir / 'phase.csv')

    print(f'\n{"ALL PASS" if ok_all else "FAILED"}')
    return 0 if ok_all else 1


if __name__ == '__main__':
    sys.exit(main())
