#!/usr/bin/env python3
"""How does retrieval's window score decay when a FoV lands OFF the grid?

Retrieval scores a query against a fixed lattice of tile positions: a main grid
at (256i, 256j) and an overlap grid half a tile off it. A real photograph lands
wherever it lands, so its content sits at some displacement from both. The
pooling bench recorded that displacement (`delta`) but never measured what it
costs -- and "the truth was never proposed" is retrieval's largest failure
bucket, so how fast the score falls between grid points is a direct question
about whether the lattice is fine enough.

The shape of the experiment
---------------------------
For one grid point (x, y):

    1. the reference is encoded ONCE and never moves. Two windows, both R x C
       tiles, taken from the WSI:
           main     anchored exactly at (x, y)
           overlap  anchored at (x + 128, y + 128), which is the overlap-grid
                    position production's `_window_xy(..., use_overlap=True)`
                    puts nearest to this main point
    2. the QUERY moves. For every (dx, dy) in [0, 128] x [0, 128] stepping by
       --step, Camera photographs a fresh FoV whose top-left is (x+dx, y+dy),
       and its patches are scored against those same two fixed windows.

So the score is production's own window score -- `SlidingWindowSimilarity` at
ONE window position, `hm.mean(dim=(-2,-1))` -- not a re-implementation, and not
a search. Nothing slides.

Why the two curves have to cross
--------------------------------
At (0, 0) the query is exactly on the main point, so the main score is at its
maximum and the overlap score at its minimum. At (128, 128) the query has walked
onto the overlap point and the two swap. Somewhere between them the curves
cross, and where that happens IS the answer to "does the half-tile overlap grid
earn its keep": if the crossing sits near the middle with both scores still
high, the lattice covers the plane well; if both dip hard before they cross, a
query landing in that hollow is one the lattice describes badly no matter which
grid it is matched to.

That crossing is also what the two gates check, because it is a property the
geometry forces -- if it does not appear, the grid pairing or the coordinates
are wrong and no score in the output means anything.

Reading one tile at a time, not the region
------------------------------------------
`WsiTissuesContainer` reads a whole tissue region in one call, which is correct
for production (it needs every window) and catastrophic here (an unsegmented L0
region is 18.7 Gpx; one such read took 4083 s, see TODO 2026-08-13). This needs
only (R+2) x (C+2) tiles around each grid point, so it reads exactly that
window and builds the same `TissuePatchContainer` over it -- the identical three
lines `WsiTissuesContainer.__init__` runs, handed a region that covers the
expanded footprint instead of the slide.

The one-tile margin is what makes the scan legal: at (dx, dy) = (128, 128) the
query footprint and the overlap window both extend half a tile past the main
window's edge, and without the margin the overlap window would be cut off.

Rotation is fixed at 0
----------------------
This measures displacement sensitivity alone. Rotation has its own test
(test_gigapath_slide_win_sim.py step 5); mixing them would leave one number
answering two questions, and a drop would not say which.

Two axes on top of the displacement scan
----------------------------------------
    preprocess   'none' is RGB as production encodes it; 'grey' is luminance
                 replicated to three channels, applied to BOTH sides. Each needs
                 its own forward pass, so it is a row of the CSV. The FoV is
                 captured once and encoded twice -- capture dominates, so the
                 second is nearly free and both see the identical photograph.

    combiner     how the query's per-tile cosines become one window score.
                 `mean` is additive and is what production ships: strong tiles
                 carry a window even when the rest disagree, an OR. `geomean` is
                 multiplicative, so one bad tile drags the window down whatever
                 the others do -- an AND, and the same change as "multiply
                 instead of add", since a linear combiner cannot be an AND.
                 `min` is the hardest AND. All three read the same tensor, so
                 they are columns and cost nothing.

Outputs, all under result/OffGridScore/
---------------------------------------
    offgrid_scores.csv      one row per (slide, level, point, dx, dy, preprocess)
    offgrid_heatmap.png     2D score maps, both preprocessings, plus ALL
    offgrid_surface.png     3D surfaces per preprocess, where main and overlap
                            trade places
    offgrid_profile.png     along the diagonal, one panel per combiner,
                            mean +- 1 std
    offgrid_gates.csv       the geometry gates, per (slide, level, preprocess),
                            aggregated over points -- see aggregate_gates for
                            why they are not per point
"""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _directory in ('utilities', 'aiNNModel', 'query_sim', '2_retrieval',
                   'utilities/test_modules'):
    _path = str(_ROOT / _directory)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402
import matplotlib                                                   # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                     # noqa: E402
from mpl_toolkits.mplot3d import Axes3D                             # noqa: E402,F401

from PatchingLib import (PatchGrid, QueryPatchContainer,            # noqa: E402
                         TissuePatchContainer)
from SafeSlide import SafeSlide                                     # noqa: E402
from TissuesRegionsMask import TissuesRegionsMask, TissueRegion     # noqa: E402
from TissueSegFunc import HestSegConfig                             # noqa: E402
from TileEncoderFunc import encoder_config, encoder_names      # noqa: E402
from GigaPathSlidingWinSim import SlidingWindowSimilarity           # noqa: E402
from camera import Camera                                           # noqa: E402
from config import DomainGapConfig                                  # noqa: E402
from _paths import job_result_dir                                   # noqa: E402

TILE = 256

#: Half a tile. The overlap grid sits exactly this far from the main grid on
#: both axes, so it is both the scan's upper bound and the displacement at
#: which the two windows trade places.
HALF_TILE = TILE // 2

#: Tiles of margin added on every side of the FoV footprint before the WSI
#: window is read. One is enough: the furthest the query or its overlap window
#: reaches past the main window is HALF_TILE.
MARGIN_TILES = 1

#: The query is cut on the main grid ONLY, and this is not a shortcut -- the
#: query's overlap tiles are never read. `SlidingWindowSimilarity` opens with
#:
#:     q_grid = qFeatureMap.main_feature_grid()   # fixed: always main kernel
#:
#: and scores BOTH the WSI main grid and the WSI overlap grid against that one
#: kernel ("combinations 1+3", GigaPathSlidingWinSim.py:54).
#: `qFeatureMap.overlap_feature_grid()` is never called. Cutting them anyway
#: encoded 12 tiles per FoV -- 37.5% of every forward pass -- and threw the
#: result away. The WSI side keeps overlap=True, because its overlap grid IS
#: consumed and is half of what this bench measures.
QUERY_OVERLAP = False

#: Row dimension. Each needs its own forward pass, so it is a separate
#: measurement rather than a column: 'none' is what production encodes, 'grey'
#: removes colour from both sides. Applied symmetrically -- comparing a grey
#: query against a colour reference would not be describing one thing.
PREPROCESS = ('none', 'grey')

#: Column dimension. All three are reductions of the SAME [R_q, C_q] tensor of
#: per-tile cosines, so adding them costs no capture and no encode.
#:
#:   mean     the arithmetic mean, which is what production scores with. A few
#:            strongly matching tiles carry a window even when the rest disagree
#:            -- an OR over the query's tiles.
#:   geomean  the geometric mean, exp(mean(log(x))). Multiplicative, so one tile
#:            near zero drags the whole window down however good the others are
#:            -- an AND. "Use AND instead of OR" and "multiply instead of add"
#:            are the same change; a linear combiner cannot be an AND.
#:   min      the hardest AND: a single tile can veto the window.
COMBINERS = ('mean', 'geomean', 'min')

#: Floor applied before the log in geomean. Cosines of L2-normalised features
#: are in [-1, 1] and log is undefined at or below zero, so a floor is not
#: optional -- but it is also not free: raising it pulls geomean towards the
#: arithmetic mean, because it stops the worst tile from dominating. Recorded in
#: the definitions rather than left as an invisible constant.
GEOMEAN_FLOOR = 1e-3

DEFINITIONS = [
    ('dx, dy', 'displacement of the query FoV from the grid point, in level-n '
               'pixels so the axis is comparable across levels.'),
    ('score', "mean cosine over the query's R x C patches against the WSI "
              'patches of ONE fixed window. This is production\'s window score '
              '(SlidingWindowSimilarity at a single position), not a search.'),
    ('main', 'the reference window anchored exactly at the grid point (x, y).'),
    ('overlap', 'the reference window anchored at (x+128, y+128) -- the '
                'overlap-grid position nearest that main point.'),
    ('crossover', 'the displacement where the overlap score first exceeds the '
                  'main score. Past it the half-tile grid is the better match.'),
    ('band = +-1 std', 'standard deviation across all grid points, slides and '
                       'levels at that displacement -- spread between places, '
                       'not repeated exposures of one place.'),
    ('ALL', 'every grid point of every slide and level averaged together.'),
    ('preprocess', "what the tile looked like going into the encoder: 'none' is "
                   "RGB as production encodes it, 'grey' is luminance replicated "
                   'to three channels. Applied to BOTH the query and the '
                   'reference, so the two sides always match.'),
    ('mean', "arithmetic mean of the query's per-tile cosines -- production's "
             'window score. Additive, so good tiles compensate for bad ones.'),
    ('geomean', 'geometric mean, exp(mean(log(x))). Multiplicative, so one bad '
                'tile drags the window down regardless of the rest. This is the '
                'AND of the same measurement the mean takes the OR of.'),
    ('min', 'the worst tile in the window. The hardest AND available.'),
    ('geomean floor', f'cosines are floored at {GEOMEAN_FLOOR} before the log, '
                      'since log is undefined at zero. Raising it moves geomean '
                      'towards mean, so it is a real parameter, not a guard.'),
]


# ══════════════════════════════════════════════════════════════════════════════
#  Grid geometry, without reading pixels
# ══════════════════════════════════════════════════════════════════════════════

def grid_positions(mask, level: int, ds: float):
    """Every main-grid position at this level, as level-0 top-left coordinates.

    PatchGrid.from_size takes sizes and offsets only, so this costs nothing even
    when the region is the whole slide.
    """
    xs, ys, regions = [], [], []
    for region_index, region in enumerate(mask.tissue_regions):
        width_n, height_n = int(region.w / ds), int(region.h / ds)
        if width_n < TILE or height_n < TILE:
            continue
        grid = PatchGrid.from_size(
            width_n, height_n, TILE, overlap=False,
            x_offset=int(region.x / ds), y_offset=int(region.y / ds),
            ds=ds, level=level)
        for info in grid.iter_infos():
            xs.append(int(round(info.x * ds)))
            ys.append(int(round(info.y * ds)))
            regions.append(region_index)
    if not xs:
        return None
    return (np.array([xs, ys], dtype=np.int64).T,
            np.array(regions, dtype=np.int64))


def pick_points(mask, level, ds, camera, n_points, white_max, rng):
    """Grid points that are mostly tissue AND leave room for the whole scan.

    Two filters, and the second is not cosmetic: a point close to a region's
    right or bottom edge cannot host the expanded window, and the FoV at
    (x+128, y+128) would read past the region. Rejecting them here keeps the
    output free of rows that only measure where the slide ran out.
    """
    found = grid_positions(mask, level, ds)
    if found is None:
        return []
    positions, region_of = found

    white = mask.white_fractions(positions, level, TILE)
    candidates = np.flatnonzero(white < white_max)
    if not len(candidates):
        return []

    footprint_w, footprint_h = camera.qfw.rect_w_l0, camera.qfw.rect_h_l0
    margin_l0 = MARGIN_TILES * TILE * ds
    usable = []
    for index in candidates:
        x, y = int(positions[index, 0]), int(positions[index, 1])
        region = mask.tissue_regions[int(region_of[index])]
        # Room for: the margin on the near side, the FoV itself, the full
        # HALF_TILE scan, and the margin on the far side.
        needs_x = x - margin_l0
        needs_y = y - margin_l0
        far_x = x + HALF_TILE * ds + footprint_w + margin_l0
        far_y = y + HALF_TILE * ds + footprint_h + margin_l0
        if (needs_x >= region.x and needs_y >= region.y
                and far_x <= region.x + region.w
                and far_y <= region.y + region.h):
            usable.append((x, y, float(white[index])))

    if not usable:
        return []
    chosen = rng.choice(len(usable), size=min(n_points, len(usable)),
                        replace=False)
    return [usable[int(i)] for i in chosen]


# ══════════════════════════════════════════════════════════════════════════════
#  The fixed reference, encoded once per grid point
# ══════════════════════════════════════════════════════════════════════════════

def reference_windows(slide, x, y, level, ds, query_rows, query_cols, encoders):
    """The two windows the query is scored against, encoded once PER preprocess.

    Reads the FoV footprint expanded by MARGIN_TILES on every side and builds a
    TissuePatchContainer over exactly that -- the same construction
    WsiTissuesContainer performs per region, handed a region that is this
    window rather than the whole tissue.

    Returns {preprocess: FeaturesMap}. The tiles are read ONCE and encoded once
    per preprocess -- reading dominates, and the same pixels answer both. Which
    cell of the map is `main` and which is `overlap` is decided by the caller,
    because that indexing is the part a mistake would hide in.
    """
    margin_l0 = MARGIN_TILES * TILE * ds
    origin_x = int(round(x - margin_l0))
    origin_y = int(round(y - margin_l0))
    window_cols = query_cols + 2 * MARGIN_TILES
    window_rows = query_rows + 2 * MARGIN_TILES
    width_l0 = int(round(window_cols * TILE * ds))
    height_l0 = int(round(window_rows * TILE * ds))

    image, _ = slide.read_region_valid(
        (origin_x, origin_y), level,
        (int(window_cols * TILE), int(window_rows * TILE)))

    region = TissueRegion(x=origin_x, y=origin_y, w=width_l0, h=height_l0,
                          index=0)
    container = TissuePatchContainer(image, region=region, img_ds=ds,
                                     is_crop=True, at_level=level)
    container.extract_all(tile_size=TILE, overlap=True)
    return {name: container.to_features(encode)
            for name, encode in encoders.items()}


def _combine(per_tile: torch.Tensor) -> dict:
    """The [R_q, C_q] per-tile cosines of ONE window, reduced three ways.

    All three read the same tensor, which is why they are columns of one row
    rather than separate runs: nothing here costs a capture or a forward pass.
    """
    values = per_tile.flatten().float()
    floored = values.clamp_min(GEOMEAN_FLOOR)
    return {
        'mean': float(values.mean()),
        'geomean': float(torch.exp(torch.log(floored).mean())),
        'min': float(values.min()),
    }


def window_scores(query_features, reference_features) -> dict:
    """Every (window, combiner) score at the ONE position that belongs to the
    grid point. Keys are 'main_mean', 'overlap_geomean', and so on.

    SlidingWindowSimilarity returns every window position; with a margin of one
    tile there are 3x3 of them on the main grid and 2x2 on the overlap grid.
    Index (1, 1) of the main output is the window whose top-left is the grid
    point itself -- the margin shifted the reference origin one tile back, so
    one tile forward is the point again. Index (1, 1) of the overlap output is
    the position half a tile beyond it, which is (x+128, y+128).

    Only the query's MAIN grid reaches this function -- see QUERY_OVERLAP. Both
    outputs below are scored against that single kernel, which is why cutting
    the query's overlap tiles changes nothing here and saves 37.5% of the
    encoding.

    The trailing two axes are the per-tile cosines of that window, so they are
    handed to _combine rather than averaged here: the mean production takes is
    one of three reductions, not the definition of the score.
    """
    out = {f'{w}_{c}': float('nan')
           for w in ('main', 'overlap') for c in COMBINERS}

    main_sim, overlap_sim = SlidingWindowSimilarity(query_features,
                                                    reference_features)
    if main_sim.numel() == 0:
        return out
    for name, value in _combine(
            main_sim[MARGIN_TILES, MARGIN_TILES]).items():
        out[f'main_{name}'] = value

    if overlap_sim.numel() > 0 and (overlap_sim.shape[0] > MARGIN_TILES
                                    and overlap_sim.shape[1] > MARGIN_TILES):
        for name, value in _combine(
                overlap_sim[MARGIN_TILES, MARGIN_TILES]).items():
            out[f'overlap_{name}'] = value
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  One grid point: the scan
# ══════════════════════════════════════════════════════════════════════════════

def scan_point(slide, camera, encoders, x, y, level, ds, anchor_white,
               point_id, wsi_stem, mask, step) -> list:
    """Photograph the FoV at every displacement and score it against the fixed
    reference. Returns one row per (dx, dy, preprocess).

    The FoV is captured ONCE per displacement and encoded once per preprocess.
    Capturing dominates the cost -- reading a bounding square and running the
    whole augmentation chain -- so the second preprocess is nearly free, and
    both see the identical photograph rather than two draws of one."""
    probe, _ = camera.capture_with_gt(x, y, rotation=0)
    if probe is None:
        return []
    probe_container = QueryPatchContainer(probe)
    probe_container.extract_all(TILE, overlap=QUERY_OVERLAP)
    query_rows = probe_container.grid.grid_rows
    query_cols = probe_container.grid.grid_cols
    if query_rows == 0 or query_cols == 0:
        return []

    reference = reference_windows(slide, x, y, level, ds,
                                  query_rows, query_cols, encoders)

    # Background over the whole footprint, recorded rather than filtered on:
    # the anchor tile passing --white-max says nothing about the other 19.
    footprint = np.array(
        [[x + c * TILE * ds, y + r * TILE * ds]
         for r in range(query_rows) for c in range(query_cols)],
        dtype=np.int64)
    fov_white = float(mask.white_fractions(footprint, level, TILE).mean())

    offsets = list(range(0, HALF_TILE + 1, step))
    if offsets[-1] != HALF_TILE:
        offsets.append(HALF_TILE)

    base_mpp = slide.base_mpp  # SafeSlide.base_mpp: mean of mpp-x/y, one definition
    rows = []
    for dy in offsets:
        for dx in offsets:
            shot, _ = camera.capture_with_gt(int(round(x + dx * ds)),
                                             int(round(y + dy * ds)),
                                             rotation=0)
            if shot is None:
                continue
            query_container = QueryPatchContainer(shot)
            query_container.extract_all(TILE, overlap=QUERY_OVERLAP)
            if (query_container.grid.grid_rows != query_rows
                    or query_container.grid.grid_cols != query_cols):
                # A different tile count would compare two different kernels
                # and the scores would not belong on the same axis.
                continue
            for name, encode in encoders.items():
                scores = window_scores(query_container.to_features(encode),
                                       reference[name])
                rows.append(dict(
                    wsi_stem=wsi_stem, level=level, ds=ds,
                    mpp=base_mpp * ds,
                    point_id=point_id, anchor_x=x, anchor_y=y,
                    anchor_white_frac=round(anchor_white, 4),
                    fov_white_frac=round(fov_white, 4),
                    dx=dx, dy=dy, preprocess=name,
                    **{f'score_{k}': v for k, v in scores.items()},
                    n_query_tiles=query_rows * query_cols))
    return rows


def aggregate_gates(rows, combiner: str = 'mean') -> list:
    """The crossing the geometry forces, checked on the AGGREGATE.

    Deliberately not per point. The domain gap is re-randomised on every
    capture -- realistically so, since each displacement is a separate exposure
    -- which means one (0,0) sample and one (128,128) sample differ by a fresh
    draw of colour temperature, vignetting and noise as well as by position. On
    the first run two of ten points "failed" that way while the mean over all
    ten crossed cleanly at exactly d=64: the geometry was never in doubt, the
    single samples were just noisy.

    Averaged over the points of one (slide, level, preprocess) the exposure
    noise cancels and what is left is the displacement, which is what a gate can
    actually test. Scored on `combiner` -- production's mean by default; the
    other reductions are in the CSV and are not gated, since only this one
    defines the score production ships.
    """
    groups = {}
    for row in rows:
        key = (row['wsi_stem'], row['level'], row['preprocess'])
        groups.setdefault(key, []).append(row)

    out = []
    for (stem, level, preprocess), subset in sorted(groups.items()):
        def mean_at(dx, dy, window):
            values = [r[f'score_{window}_{combiner}'] for r in subset
                      if r['dx'] == dx and r['dy'] == dy
                      and np.isfinite(r[f'score_{window}_{combiner}'])]
            return float(np.mean(values)) if values else float('nan')

        main_near = mean_at(0, 0, 'main')
        main_far = mean_at(HALF_TILE, HALF_TILE, 'main')
        overlap_near = mean_at(0, 0, 'overlap')
        overlap_far = mean_at(HALF_TILE, HALF_TILE, 'overlap')

        main_ok = main_near > main_far
        overlap_ok = (overlap_far > overlap_near
                      if np.isfinite(overlap_near) and np.isfinite(overlap_far)
                      else True)
        out.append(dict(
            wsi_stem=stem, level=level, preprocess=preprocess,
            combiner=combiner,
            n_points=len({r['point_id'] for r in subset}),
            main_at_0=main_near, main_at_128=main_far,
            overlap_at_0=overlap_near, overlap_at_128=overlap_far,
            main_decays=bool(main_ok), overlap_rises=bool(overlap_ok),
            passed=bool(main_ok and overlap_ok)))
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  One slide
# ══════════════════════════════════════════════════════════════════════════════

def analyse_slide(wsi_path, args, encoders, hest_method, rng) -> list:
    stem = Path(wsi_path).stem
    print(f'\n{"=" * 78}\n{stem}\n{"=" * 78}', flush=True)
    slide = SafeSlide(str(wsi_path))
    score_rows = []
    try:
        # level_rule='nearest', not openslide's default. BRACS_1228's level 1
        # reports downsample 4.00003, so a strict comparison against a request
        # for 4.0 falls back to level 0 and the mask is segmented over 6.58 Gpx
        # instead of 411 Mpx. The first run of this bench did exactly that --
        # `tiled seg ... input 61197x107568` in log/OffGridScore is level 0 --
        # and paid several hundred seconds per slide for a mask that is only
        # used to pick grid points and count background.
        mask = TissuesRegionsMask.from_wsi(
            slide, ds=args.mask_ds, method=hest_method,
            seg_chunk_px=int(args.seg_chunk_px), stitch_overlap=128,
            level_rule='nearest')
        mask.filter_regions(min_ratio=args.min_region_ratio)
        mask.merge_overlapping()
        print(f'  mask: tissue={mask.tissue_fraction() * 100:.1f}%  '
              f'{len(mask.tissue_regions)} regions', flush=True)
        if not mask.tissue_regions:
            return []

        base_mpp = slide.base_mpp  # SafeSlide.base_mpp: mean of mpp-x/y, one definition
        levels = args.levels if args.levels else range(slide.level_count)
        for level in levels:
            if level >= slide.level_count:
                continue
            ds = float(slide.level_downsamples[level])
            level_mpp = base_mpp * ds

            config = DomainGapConfig(
                wh_ratio=args.fov_ratio, MPixels=args.fov_mpixels,
                query_mpp=level_mpp,
                angle_jitter_deg=0.0, scale_range=(1.0, 1.0),
                query_mpp_jitter=0.0, stage_shift_max=0,
                photometric=args.domain_gap)
            camera = Camera(slide, cfg=config, mask=mask, seed=args.seed)

            points = pick_points(mask, level, ds, camera, args.points,
                                 args.white_max, rng)
            if not points:
                print(f'  L{level}: no grid point has room for the scan '
                      f'-- skipped', flush=True)
                continue
            print(f'  L{level}  mpp={level_mpp:.4f}  {len(points)} points  '
                  f'FoV {camera.qfw.output_w}x{camera.qfw.output_h}', flush=True)

            for point_id, (x, y, white) in enumerate(points):
                rows = scan_point(slide, camera, encoders, x, y, level, ds,
                                  white, point_id, stem, mask, args.step)
                if not rows:
                    print(f'    point {point_id} at ({x},{y}): no shot '
                          f'landed -- skipped', flush=True)
                    continue
                score_rows.extend(rows)
                # Per point the endpoints are two single exposures, so their
                # order says little; the gate runs on the aggregate at the end.
                # Printed here only so a run in progress shows something moving.
                near = [r for r in rows if r['dx'] == 0 and r['dy'] == 0
                        and r['preprocess'] == 'none']
                far = [r for r in rows if r['dx'] == HALF_TILE
                       and r['dy'] == HALF_TILE and r['preprocess'] == 'none']
                shape = ''
                if near and far:
                    shape = (f'  main {near[0]["score_main_mean"]:.4f}->'
                             f'{far[0]["score_main_mean"]:.4f}'
                             f'  overlap {near[0]["score_overlap_mean"]:.4f}->'
                             f'{far[0]["score_overlap_mean"]:.4f}')
                print(f'    point {point_id} at ({x},{y})  '
                      f'{len(rows)} rows{shape}', flush=True)
    finally:
        slide.close()
    return score_rows


# ══════════════════════════════════════════════════════════════════════════════
#  Output
# ══════════════════════════════════════════════════════════════════════════════

def write_csv(rows, path) -> None:
    if not rows:
        print(f'  (nothing to write to {path.name})')
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, restval='')
        writer.writeheader()
        writer.writerows(rows)
    print(f'  {path}  ({len(rows)} rows)')


def _mean_map(rows, key):
    """(offsets, [n, n] mean score) over whatever rows are handed in.

    The caller slices by preprocess before calling; averaging 'none' and 'grey'
    together would report a number no encoder ever produced.
    """
    offsets = sorted({row['dx'] for row in rows} | {row['dy'] for row in rows})
    index = {value: i for i, value in enumerate(offsets)}
    total = np.zeros((len(offsets), len(offsets)))
    count = np.zeros_like(total)
    for row in rows:
        value = row[key]
        if not np.isfinite(value):
            continue
        i, j = index[row['dy']], index[row['dx']]
        total[i, j] += value
        count[i, j] += 1
    with np.errstate(invalid='ignore'):
        return np.array(offsets), np.where(count > 0, total / count, np.nan)


CAPTION = (
    'score = the query FoV\'s per-tile cosines against ONE fixed WSI window, '
    'reduced by the named combiner. Nothing slides.\n'
    'mean = additive (what production ships).    geomean = multiplicative, an '
    'AND: one bad tile drags the window down.    min = the hardest AND.\n'
    'main = window at the grid point (x, y).    overlap = window at '
    '(x+128, y+128).    none = RGB, grey = luminance, applied to BOTH sides.\n'
    'dx, dy are level-n pixels, so the axis means the same thing at every '
    'level. The reference is encoded once and never moves; only the query does.')


def plot_heatmap(rows, path) -> None:
    """Every (slide, level) as its own row of maps, plus ALL at the bottom.

    Four columns: both preprocessings against both windows, so the question
    "does removing colour change the shape of the decay" is answered by looking
    across a row rather than between two files.
    """
    groups = sorted({(row['wsi_stem'], row['level']) for row in rows})
    if not groups:
        return
    columns = [(pre, win) for pre in PREPROCESS for win in ('main', 'overlap')]
    n_rows = len(groups) + 1
    figure, axes = plt.subplots(n_rows, len(columns), squeeze=False,
                                figsize=(3.6 * len(columns), 3.1 * n_rows))

    panels = [(f'{stem}  L{level}',
               [r for r in rows if r['wsi_stem'] == stem and r['level'] == level])
              for stem, level in groups]
    panels.append(('ALL', rows))

    for row_index, (label, subset) in enumerate(panels):
        for col_index, (preprocess, window) in enumerate(columns):
            axis = axes[row_index][col_index]
            sliced = [r for r in subset if r['preprocess'] == preprocess]
            offsets, grid = _mean_map(sliced, f'score_{window}_mean')
            if not np.isfinite(grid).any():
                axis.axis('off')
                continue
            image = axis.imshow(grid, origin='lower', cmap='viridis',
                                extent=(offsets[0], offsets[-1],
                                        offsets[0], offsets[-1]))
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            axis.set_xlabel('dx (level-n px)')
            axis.set_ylabel('dy (level-n px)')
            weight = 'bold' if label == 'ALL' else 'normal'
            axis.set_title(f'{label}   {preprocess} / {window}', fontsize=8,
                           fontweight=weight)
    figure.suptitle('window score (mean combiner) against displacement '
                    'from the grid point', fontsize=11)
    figure.tight_layout(rect=(0, 0.03, 1, 0.985))
    figure.savefig(path, dpi=130)
    plt.close(figure)
    print(f'  {path}')


def plot_surface(rows, path) -> None:
    """One 3D panel per preprocess: main and overlap surfaces, and where they
    trade places. Same z range on both so the two are comparable by eye."""
    figure = plt.figure(figsize=(7.0 * len(PREPROCESS), 5.6))
    drew = False
    for index, preprocess in enumerate(PREPROCESS):
        sliced = [r for r in rows if r['preprocess'] == preprocess]
        if not sliced:
            continue
        offsets, main_grid = _mean_map(sliced, 'score_main_mean')
        _, overlap_grid = _mean_map(sliced, 'score_overlap_mean')
        if not np.isfinite(main_grid).any():
            continue
        drew = True
        axis = figure.add_subplot(1, len(PREPROCESS), index + 1,
                                  projection='3d')
        mesh_x, mesh_y = np.meshgrid(offsets, offsets)
        axis.plot_surface(mesh_x, mesh_y, main_grid, alpha=0.75,
                          color='tab:blue')
        if np.isfinite(overlap_grid).any():
            axis.plot_surface(mesh_x, mesh_y, overlap_grid, alpha=0.75,
                              color='tab:orange')
        axis.set_xlabel('dx (level-n px)')
        axis.set_ylabel('dy (level-n px)')
        axis.set_zlabel('score (mean)')
        axis.set_title(f'{preprocess} -- blue = main, orange = overlap\n'
                       'they trade places across the scan', fontsize=9)
    if not drew:
        plt.close(figure)
        return
    figure.tight_layout(rect=(0, 0.10, 1, 1))
    figure.text(0.01, 0.01, CAPTION, fontsize=7.2, va='bottom',
                family='monospace', linespacing=1.5)
    figure.savefig(path, dpi=130)
    plt.close(figure)
    print(f'  {path}')


def plot_profile(rows, path) -> None:
    """Along the diagonal dx == dy -- the path from the main point to the
    overlap point, which is why the crossing has to be on it.

    One column per combiner, so the additive and multiplicative reductions of
    the very same measurement sit side by side. Solid = none, dashed = grey;
    band is +-1 std over every grid point of every slide and level.
    """
    diagonal = [r for r in rows if r['dx'] == r['dy']]
    if not diagonal:
        return
    offsets = sorted({r['dx'] for r in diagonal})

    figure, axes = plt.subplots(1, len(COMBINERS),
                                figsize=(5.6 * len(COMBINERS), 4.8))
    axes = np.atleast_1d(axes)

    for axis, combiner in zip(axes, COMBINERS):
        for preprocess, style in zip(PREPROCESS, ('-', '--')):
            for window, colour in (('main', 'tab:blue'),
                                   ('overlap', 'tab:orange')):
                key = f'score_{window}_{combiner}'
                means, spreads = [], []
                for offset in offsets:
                    values = [r[key] for r in diagonal
                              if r['dx'] == offset
                              and r['preprocess'] == preprocess
                              and np.isfinite(r[key])]
                    means.append(np.mean(values) if values else np.nan)
                    spreads.append(np.std(values) if values else np.nan)
                means, spreads = np.array(means), np.array(spreads)
                axis.plot(offsets, means, style, color=colour, marker='o',
                          markersize=3,
                          label=f'{window} / {preprocess}')
                axis.fill_between(offsets, means - spreads, means + spreads,
                                  color=colour, alpha=0.12, linewidth=0)
        axis.set_xlabel('displacement along dx = dy (level-n px)')
        axis.set_title(f'{combiner}', fontsize=10)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel('score')
    axes[-1].legend(fontsize=7, loc='best')

    figure.suptitle('additive (mean) against multiplicative (geomean) and the '
                    'hardest AND (min), colour against greyscale', fontsize=11)
    figure.tight_layout(rect=(0, 0.13, 1, 0.95))
    figure.text(0.01, 0.01, CAPTION, fontsize=7.2, va='bottom',
                family='monospace', linespacing=1.5)
    figure.savefig(path, dpi=130)
    plt.close(figure)
    print(f'  {path}')


def read_scores(paths) -> list:
    """Load offgrid_scores.csv files back into the rows the figures expect.

    csv gives strings; the plotting code indexes numerically on dx/dy/level and
    arithmetically on the scores, so the numeric columns are converted here
    rather than at every use. Anything unrecognised stays a string, which is
    what wsi_stem and preprocess want.
    """
    integer_columns = {'level', 'point_id', 'anchor_x', 'anchor_y',
                       'dx', 'dy', 'n_query_tiles'}
    rows = []
    for path in paths:
        with open(path, newline='') as handle:
            for raw in csv.DictReader(handle):
                row = {}
                for key, value in raw.items():
                    if key in integer_columns:
                        row[key] = int(float(value))
                    elif key.startswith('score_') or key in (
                            'ds', 'mpp', 'anchor_white_frac', 'fov_white_frac'):
                        row[key] = float(value) if value != '' else float('nan')
                    else:
                        row[key] = value
                rows.append(row)
        print(f'  read {path}')
    return rows


def encoder_tag(rows, fallback: str = '') -> str:
    """Which encoder these rows came from, read from the rows themselves.

    From the rows and not from args, because --plot-only draws CSVs written by
    an earlier run: argparse's default would name whichever encoder was not
    used, and the figure would carry the wrong model's name.

    Mixed rows stop the run. Every figure below averages over rows, so one
    heatmap drawn from two encoders is a picture of neither -- and it would
    look exactly like a picture of one.

    Empty when the rows predate the column, which leaves the old filenames
    unchanged so --plot-only over an old CSV still redraws what it drew.
    """
    seen = {r.get('encoder', '') for r in rows} - {''}
    if len(seen) > 1:
        raise SystemExit(
            f'these rows mix encoders ({", ".join(sorted(seen))}). Draw them '
            f'separately -- one heatmap cannot mean two models.')
    return seen.pop() if seen else fallback


def render(rows, out_dir: Path, encoder: str = '') -> list:
    """Gates, figures and the definition list. Shared by a real run and by
    --plot-only, so a merged CSV is drawn by exactly the code that drew the
    per-slide ones."""
    gates = aggregate_gates(rows) if rows else []
    tag = encoder_tag(rows, encoder)
    suffix = f'_{tag}' if tag else ''
    write_csv(gates, out_dir / f'offgrid_gates{suffix}.csv')
    # No suffix: the definition list is this bench's own vocabulary and says
    # nothing about which model produced a number.
    write_csv([dict(term=term, means=means) for term, means in DEFINITIONS],
              out_dir / 'offgrid_definitions.csv')
    if rows:
        plot_heatmap(rows, out_dir / f'offgrid_heatmap{suffix}.png')
        plot_surface(rows, out_dir / f'offgrid_surface{suffix}.png')
        plot_profile(rows, out_dir / f'offgrid_profile{suffix}.png')
    return gates


def report_gates(gates) -> list:
    failed = [g for g in gates if not g['passed']]
    if failed:
        print(f'\n{len(failed)}/{len(gates)} GATE FAILURE(S) -- '
              f'averaged over every point, the geometry did not appear:')
        for gate in failed:
            print(f'  {gate["wsi_stem"]} L{gate["level"]} '
                  f'{gate["preprocess"]} (n={gate["n_points"]} points): '
                  f'main_decays={gate["main_decays"]} '
                  f'overlap_rises={gate["overlap_rises"]}')
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(
        description='How does the window score decay off the grid?')
    parser.add_argument('slides', nargs='*',
                        help='WSI paths; with --plot-only these are '
                             'offgrid_scores.csv paths instead')
    parser.add_argument('--plot-only', action='store_true',
                        help='draw the figures from existing offgrid_scores.csv '
                             'files and exit. No GPU, no WSI, no model -- runs '
                             'on a login node in seconds. This is how the array '
                             'run gets ONE set of figures covering every slide: '
                             'each task writes its own CSV, then those are '
                             'merged and redrawn here.')
    parser.add_argument('--step', type=int, default=16,
                        help='displacement step in level-n px (default 16, so '
                             '9x9 = 81 offsets per grid point)')
    parser.add_argument('--points', type=int, default=5,
                        help='grid points per (slide, level)')
    parser.add_argument('--levels', type=int, nargs='*', default=None,
                        help='levels to scan; default every level')
    parser.add_argument('--white-max', type=float, default=0.15,
                        help='anchor tile must be below this background '
                             'fraction')
    parser.add_argument('--fov-ratio', default='45:32')
    parser.add_argument('--fov-mpixels', type=float, default=1.47456)
    parser.add_argument('--domain-gap', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='apply colour/vignette/lens/noise/JPEG to the FoV')
    parser.add_argument('--mask-ds', type=float, default=4.0)
    parser.add_argument('--seg-chunk-px', type=float, default=4_000_000)
    parser.add_argument('--min-region-ratio', type=float, default=0.01)
    parser.add_argument(
        '--encoder', default='gigapath', choices=encoder_names(),
        help='which tile encoder. Only the module for THIS one is imported: '
             'every implementation sets HF_HOME above its own timm import and '
             'setdefault is first-one-wins, so importing all three would point '
             'two of them at the wrong weight cache -- silently. See '
             'TileEncoderFunc._IMPLEMENTATIONS.')
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    if not args.slides:
        parser.error('give at least one WSI path, or CSV paths with --plot-only')

    out_dir = Path(args.out or job_result_dir('OffGridScore'))
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        missing = [p for p in args.slides if not Path(p).exists()]
        if missing:
            parser.error('no such CSV: ' + ', '.join(missing))
        print(f'plot-only: {len(args.slides)} csv -> {out_dir}')
        rows = read_scores(args.slides)
        print(f'  {len(rows)} rows, '
              f'{len({r["wsi_stem"] for r in rows})} slide(s)')
        gates = render(rows, out_dir)
        return 1 if report_gates(gates) else 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}  step={args.step}  points={args.points}  '
          f'domain_gap={args.domain_gap}')
    # One encoder per preprocess, over ONE loaded model. variant() shares the
    # weights and swaps only the pipeline, so the second arm costs a forward
    # pass and not another 4.5 GB. dtype='fp32' keeps what the free function
    # defaulted to before this call site moved to a config.
    #
    # The transform is REPLACED rather than edited, and that carries a hazard
    # worth naming now that --encoder exists: TransformConfig(preprocess=name)
    # takes its other five fields from the dataclass defaults, which are
    # GigaPath's 256/224 bicubic ImageNet numbers. For UNI2 or CONCH that would
    # silently substitute another model's preprocessing -- so it is built off
    # the encoder's OWN transform, with only `preprocess` changed.
    cfg = encoder_config(args.encoder, batch_size=args.batch_size)\
        .with_model(dtype='fp32')
    base = cfg.build(device)
    import dataclasses as _dc
    encoders = {
        name: base.variant(transform=_dc.replace(cfg.transform, preprocess=name))
        for name in PREPROCESS}
    hest_method = HestSegConfig().build(device)
    rng = np.random.default_rng(args.seed)

    all_rows, failed = [], []
    for path in args.slides:
        try:
            all_rows.extend(
                analyse_slide(path, args, encoders, hest_method, rng))
        except Exception as exc:                        # noqa: BLE001
            print(f'\n{Path(path).stem}: {type(exc).__name__}: {exc}')
            traceback.print_exc()
            failed.append(Path(path).stem)

    # Stamped here rather than inside scan_point: one place that cannot miss a
    # row, and scan_point has no reason to know which encoder built the models
    # it was handed. The encoder is in the NAME as well as in every row --
    # the name stops a second encoder's run from overwriting the first one's
    # numbers, the column stops the two from being drawn on one heatmap.
    for row in all_rows:
        row['encoder'] = args.encoder

    print(f'\n{"=" * 78}\nwriting to {out_dir}')
    write_csv(all_rows, out_dir / f'offgrid_scores_{args.encoder}.csv')
    failed_gates = report_gates(render(all_rows, out_dir, args.encoder))
    if failed:
        print(f'\n{len(failed)} slide(s) failed: {", ".join(failed)}')
    return 1 if (failed or failed_gates) else 0


if __name__ == '__main__':
    sys.exit(main())
