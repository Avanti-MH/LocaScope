#!/usr/bin/env python3
"""diag_read_exp -- the read path: how fast it is, flow by flow, and the read
geometry it rests on.

    python utilities/cli/diagnostics/diag_read_exp.py --slides BRACS_1228 ...
    sbatch jobscripts/DiagReadExp.sh

Times the production read path as it stands -- `SlideReader`, `Render`,
`FovSupply`, the training `CameraBank` -- on real slides, so a change to any
of them can be measured before and after against the same numbers:

    grid      SlideReader.read_grid at each of --grid-levels: tiles/s, with
              the CpuBudget's workers (blocks at an integer ds, one read per
              region otherwise -- the BRACS level-1 case)
    capture   render_row over sampler-placed training rows (CAMERA_FULL):
              rows/s, one process
    tiles     SlideReader.read_samples of native reference tiles: tiles/s
    fov       FovSupply over the window bench's still FoV: shots/s
    pca       the UNI2-PCA segmenter's two reads (the fit's scattered tiles,
              the projection's lattice) against a frozen WsiTileLoader at the
              same positions: tiles/s for both, and
              PASS only if every tile is identical -- level 0, the same pixels.
    s1photo   bench_stage1_mpp's photo (Render, placed by render_spec) against
              a frozen bare sensor read + simulate_with_gt, at the same ladder positions: identical
              with no gap (PASS), the side bands a 92 degree turn filled by
              reflection, the corners a 0.9 zoom-out at 3 degrees pads, and
              photos/s for both under the bench's gap.
              pca and s1photo compare; the exit status is their verdict.
    phase     how openslide samples a level at a non-multiple level-0
              location -- the premise every level-0 <-> level-n bookkeeping in
              stage 3 and in the synthetic GT rests on. A is read at a, B_d at
              a + d level-0 px, and B_d is predicted from A by floor / round /
              bilinear (level coordinate x / ds kept fractional); the model
              whose residual is near the JPEG noise is how openslide reads.
              Per slide at --phase-levels, on the hest masks of
              --mask-cache-job. Writes phase.csv.
    origins   frac(region.x / ds) per region and native level over the cached
              masks of --mask-cache-job, --origin-per-dataset slides per
              dataset: how far region origins fall from a level's pixel grid,
              which is what int(region.x / ds) bookkeeping is off by.
              Arithmetic only, no pixel read, no slide list. Writes origins.csv.
Each is run --repeats times and the best kept. Apart from pca and s1photo, nothing is
compared with anything: the correctness of each path is its tests'
(TestReadPath.sh).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from _paths import job_result_dir, setup_import_paths                 # noqa: E402

setup_import_paths()

import Cache                                                          # noqa: E402
from AccessDatasets import locate                                     # noqa: E402
from CpuBudget import CpuBudget                                       # noqa: E402
from ReadGeometry import ReadSpec                                     # noqa: E402
from SafeSlide import SafeSlide                                       # noqa: E402
from SlideReader import SlideReader                                   # noqa: E402
from PatchingLib import region_grids                                  # noqa: E402
from TileSampler import PlanSpec, SamplerConfig, TileSampler          # noqa: E402
from TissueMaskConfig import MASK_RECIPES, MaskMaker                  # noqa: E402
from camera import Render, render_spec                                # noqa: E402
from pipeline import simulate_with_gt                                 # noqa: E402
from ReadGeometry import REAL_PHOTO_SENSOR                            # noqa: E402
from config import DomainGapConfig                                    # noqa: E402
from FovSupply import FovSupply                                       # noqa: E402
import training.MppRoutingHead.Datasets as Datasets                    # noqa: E402

TILE = 256
RUNGS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
#: Timings, the default. pca and s1photo compare; phase and origins measure
#: read geometry.
TIMING_FLOWS = ('grid', 'capture', 'tiles', 'fov')
FLOWS = TIMING_FLOWS + ('pca', 's1photo', 'phase', 'origins')
DATASET_OF = {'BRACS': 'bracs/test', 'S1': 'ki67_with_photo'}


def _dataset_of(name: str) -> str:
    for prefix, ds in DATASET_OF.items():
        if name.startswith(prefix):
            return ds
    raise ValueError(f'no dataset for {name}; add it to DATASET_OF')


def _best(fn, repeats: int) -> float:
    best = float('inf')
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


class Timings:
    def __init__(self):
        self.rows = []

    def add(self, slide, flow, case, units, seconds, note=''):
        rate = units / seconds if seconds > 0 else 0.0
        self.rows.append(dict(slide=slide, flow=flow, case=case, units=units,
                              seconds=round(seconds, 3), per_s=round(rate, 1),
                              note=note))
        print(f'  {flow:<8s} {case:<24s} {units:>8d}  {seconds:8.2f}s  '
              f'{rate:9.1f}/s' + (f'   {note}' if note else ''), flush=True)


def _sample(wsi, mask, spec: ReadSpec, kind: str, n: int, seed: int):
    plan = (PlanSpec('native', camera=spec) if kind == 'native'
            else PlanSpec('ladder', RUNGS, camera=spec))
    return list(TileSampler(wsi, mask, SamplerConfig(n_per_rung=n, seed=seed))
                .sample(plan.plans_for(wsi)))


def flow_grid(t, args, name, wsi, mask, budget):
    reader = SlideReader(wsi, workers=budget.workers)
    for level in args.grid_levels:
        if level >= wsi.level_count:
            continue
        ds = float(wsi.level_downsamples[level])
        regions = mask.patchable(TILE * ds).tissue_regions
        if not regions:
            continue
        grids = region_grids(regions, ds=ds, level=level, tile_size=TILE, overlap=True)
        read = reader.read_grid(regions, grids, ds, tile=TILE,
                                block_rows=args.block_rows, level=level)

        def drain():
            for _ in read:
                pass
        t.add(name, 'grid', f'L{level} ds {ds:.5g}', read.n_tiles,
              _best(drain, args.repeats),
              f'{len(regions)} regions, '
              + ('one read per region' if read.one_read_per_region else 'blocks')
              + f', {budget.workers} workers')


def flow_capture(t, args, dataset, name, wsi, mask):
    spec = render_spec(Datasets.CAMERA_FULL, (TILE, TILE))
    rows = [Datasets.ManifestRow(dataset=dataset, wsi_name=name, x=int(s.meta.x),
                                 y=int(s.meta.y), rung=float(s.meta.ds),
                                 bucket=s.meta.bucket,
                                 footprint_l0=int(s.meta.footprint_l0))
            for s in _sample(wsi, mask, spec, 'ladder', args.n, 11)]
    rc = Datasets.RenderConfig(tile_size=TILE)

    def go():
        bank = Datasets.CameraBank(rc)
        for row in rows:
            Datasets.render_row(bank, row, rc, deterministic=True)
    t.add(name, 'capture', 'render_row CAMERA_FULL', len(rows), _best(go, args.repeats))


def flow_tiles(t, args, name, wsi, mask):
    spec = ReadSpec(TILE, TILE)
    samples = _sample(wsi, mask, spec, 'native', args.n, 31)
    reader = SlideReader(wsi, resize='area')
    t.add(name, 'tiles', 'native reference', len(samples),
          _best(lambda: reader.read_samples(samples, spec), args.repeats))


def flow_fov(t, args, name, wsi, mask):
    still = DomainGapConfig(rotation_choices=(0,), angle_jitter_deg=0.0,
                            scale_range=(1.0, 1.0), query_mpp_jitter=0.0,
                            stage_shift_max=0)
    reader = SlideReader(wsi)
    for level in range(min(2, wsi.level_count)):
        ds = float(wsi.level_downsamples[level])
        cfg = SamplerConfig(n_per_rung=args.n, seed=21 + level)

        def bank():
            cam = Render(reader, REAL_PHOTO_SENSOR, still, ds=ds, seed=21 + level)
            plan = PlanSpec('ladder', (ds,), camera=cam.spec)
            return list(FovSupply(cam, plan, cfg, mask))
        try:
            n = len(bank())
        except RuntimeError:
            continue
        t.add(name, 'fov', f'bank still L{level}', n, _best(bank, args.repeats))


# ── pca: the UNI2-PCA segmenter's reads, against a frozen WsiTileLoader ─────

class _OldWsiTileSet(torch.utils.data.Dataset):
    """FROZEN: utilities/WsiTileLoader.WsiTileSet, reduced to the read. The tile at grid (row, col) is read at level-0
    origin + (col, row) * round(tile * level_ds), one read_region_rgb per tile,
    on a worker that opens the slide itself."""

    def __init__(self, path, origin, tile, positions, level):
        self.path, self.origin, self.tile = str(path), origin, int(tile)
        self.positions, self.level = list(positions), int(level)
        self._wsi = None

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, k):
        if self._wsi is None:
            self._wsi = SafeSlide(self.path, warn=False)
            self._stride = int(round(
                self.tile * float(self._wsi.level_downsamples[self.level])))
        row, col = self.positions[k]
        rgb = self._wsi.read_region_rgb(
            (self.origin[0] + col * self._stride, self.origin[1] + row * self._stride),
            self.level, (self.tile, self.tile))
        return torch.from_numpy(np.ascontiguousarray(rgb)), k


def _old_tiles(path, origin, tile, positions, level, workers):
    """{index: tile} through the frozen reader."""
    loader = torch.utils.data.DataLoader(
        _OldWsiTileSet(path, origin, tile, positions, level),
        batch_size=64, shuffle=False, num_workers=workers)
    out = {}
    for tiles, index in loader:
        for t, k in zip(tiles.numpy(), index.tolist()):
            out[k] = t
    return out


def flow_pca(t, args, name, wsi, budget):
    """The segmenter's two reads -- the fit's scattered tiles (`_read_tiles`,
    SlideReader.read_points) and the projection's whole lattice (`_read_lattice`,
    SlideReader.read_grid) -- against the frozen WsiTileLoader at the same
    grid positions. Level 0, where the segmenter runs, so the two are the same
    pixels or a bug: PASS needs every tile identical and the lattice the same
    shape as WsiTileLoader.grid_positions'. The projection is compared on its
    first --pca-tiles tiles, not the whole slide."""
    from types import SimpleNamespace
    from PatchingLib import PatchGrid
    from TissueMask import TissueRegion
    from TissueSegFunc import scanned_rect
    from Uni2PcaSegFunc import LEVEL, Uni2PcaSegConfig, Uni2PcaSegmenter

    cfg = Uni2PcaSegConfig(workers=budget.workers)
    seg = SimpleNamespace(cfg=cfg)           # the two readers use only .cfg
    level = LEVEL
    ds = float(wsi.level_downsamples[level])
    origin, span = scanned_rect(wsi, cfg.limit_bounds)
    region = TissueRegion(x=int(origin[0]), y=int(origin[1]),
                          w=int(span[0]), h=int(span[1]), index=0)
    grid = PatchGrid.for_region(region, ds, cfg.tile, overlap=False, level=level)
    rows, cols = grid.lattice_dims('main')
    # WsiTileLoader.grid_positions, frozen
    old_shape = (int(span[1] // (cfg.tile * ds)), int(span[0] // (cfg.tile * ds)))
    shape_ok = (rows, cols) == old_shape
    print(f'  pca      lattice {rows}x{cols}  WsiTileLoader {old_shape[0]}x'
          f'{old_shape[1]}  {"same" if shape_ok else "DIFFERENT"}', flush=True)

    # projection: the lattice, first --pca-tiles tiles in row-major order
    n = min(args.pca_tiles, rows * cols)
    t0 = time.perf_counter()
    new = {}
    for tiles, _, index in Uni2PcaSegmenter._read_lattice(
            seg, wsi, region, grid, ds, level):
        for tile, k in zip(tiles, index.tolist()):
            if k < n:
                new[k] = tile
        if len(new) >= n:
            break
    t_new = time.perf_counter() - t0
    positions = [(k // cols, k % cols) for k in range(n)]
    t0 = time.perf_counter()
    old = _old_tiles(wsi._filename, origin, cfg.tile, positions, level, budget.workers)
    t_old = time.perf_counter() - t0
    same_lat = sum(np.array_equal(new.get(k), old[k]) for k in range(n))
    t.add(name, 'pca', 'lattice (read_grid)', n, t_new,
          f'{same_lat}/{n} identical to WsiTileLoader')
    t.add(name, 'pca', 'lattice WsiTileLoader', n, t_old)

    # fit: scattered positions over the whole lattice
    rng = np.random.default_rng(0)
    picks = rng.choice(rows * cols, size=min(args.pca_points, rows * cols),
                       replace=False)
    scattered = [(int(k) // cols, int(k) % cols) for k in picks]
    t0 = time.perf_counter()
    got = {}
    for tiles, _, index in Uni2PcaSegmenter._read_tiles(
            seg, wsi, origin, scattered, level):
        for tile, k in zip(tiles, index.tolist()):
            got[k] = tile
    t_pts = time.perf_counter() - t0
    t0 = time.perf_counter()
    want = _old_tiles(wsi._filename, origin, cfg.tile, scattered, level,
                      budget.workers)
    t_old_pts = time.perf_counter() - t0
    same_pts = sum(np.array_equal(got.get(k), want[k]) for k in range(len(scattered)))
    t.add(name, 'pca', 'points (read_points)', len(scattered), t_pts,
          f'{same_pts}/{len(scattered)} identical to WsiTileLoader')
    t.add(name, 'pca', 'points WsiTileLoader', len(scattered), t_old_pts)
    ok = shape_ok and same_lat == n and same_pts == len(scattered)
    print(f'  pca      {"PASS" if ok else "FAIL"}', flush=True)
    return ok


# ── s1photo: the stage-1 bench's photo, against the read it made before ──────

S1_SENSOR = REAL_PHOTO_SENSOR              # bench_stage1_mpp's --sensor default


def _old_s1photo(reader, pos, cfg, rng, rotation=None):
    """FROZEN: a bare-sensor photo -- the sensor rectangle off the slide, then the domain gap on that rectangle alone."""
    image = reader.read(pos['x'], pos['y'], ReadSpec(*S1_SENSOR), pos['rung'])
    if image is None:
        return None
    return simulate_with_gt(image, cfg=cfg, rng=rng, rotation=rotation)[0]


S1_BORDER = 8          # output px: the band a resize's truncated kernel reaches
S1_CORNER = 24         # output px: the corner square the corners check reads
S1_CORNER_REF = 400    # output px: how much wider the corners reference reads


def _s1photo_same(reader, pos, new, old, spec_new):
    """One no-gap photo against the frozen read, judged by what the two reads
    can be expected to share. Both read the same level with the same filter;
    Render's read (`spec_new`) is the FoV grown by its margin, so it starts
    `shift` level-0 px earlier. Three cases follow from the plans alone:

      exact    the shift is a whole number of level px and nothing is
               resized: the same samples, so every pixel must be equal
      resized  a whole-pixel shift, resized: the same filter on the same grid,
               so the inside must be equal; only the outermost S1_BORDER px may
               differ, where the frozen read's kernel ran off its own edge
      phase    the shift is not a whole number of level px (a level at a
               non-integer ds): openslide samples the level at another
               sub-pixel phase, so every pixel may differ, by far less than
               the same photo moved one pixel -- the decoy."""
    spec_old = ReadSpec(*S1_SENSOR)
    p_old = reader.plan(pos['x'], pos['y'], spec_old, pos['rung'])
    p_new = reader.plan(pos['x'], pos['y'], spec_new, pos['rung'])
    lds = float(reader.level_downsamples[p_new.level])
    shift_level = (p_old.rect.x0 - p_new.rect.x0) / lds
    whole = abs(shift_level - round(shift_level)) < 1e-9
    resized = (tuple(p_new.read_wh) != tuple(p_new.out_wh)
               or tuple(p_old.read_wh) != tuple(p_old.out_wh))
    kind = 'phase' if not whole else ('resized' if resized else 'exact')

    d = np.abs(new.astype(np.int16) - old).max(axis=2)
    b = S1_BORDER
    inner = d[b:-b, b:-b]
    border = d.copy()
    border[b:-b, b:-b] = 0
    decoy = np.abs(new[b:-b, b + 1:-b + 1].astype(np.int16)
                   - old[b:-b, b:-b]).max(axis=2)
    row = dict(rung=pos['rung'], x=pos['x'], y=pos['y'], level=p_new.level,
               lds=lds, resized=resized, shift_level=shift_level, kind=kind,
               max_all=int(d.max()), max_border=int(border.max()),
               max_inner=int(inner.max()), mean_inner=float(inner.mean()),
               mean_decoy=float(decoy.mean()))
    if kind == 'exact':
        good = row['max_all'] == 0
    elif kind == 'resized':
        good = row['max_inner'] == 0
    else:
        good = row['mean_inner'] < row['mean_decoy'] / 10
    row['verdict'] = 'PASS' if good else 'FAIL'
    return row, kind


def flow_s1photo(t, args, name, wsi, mask):
    """bench_stage1_mpp's photos (`Render`, placed by `render_spec`) against the
    frozen read they replaced, at the same positions:

      same     no gap at all, per rung: what the two reads must share given
               their plans (`_s1photo_same`: exact, resized, phase), which is
               what says Render reads the level, filter and rectangle the frozen
               read did. Writes s1photo_same_<slide>.csv and the photo with
               the largest inner difference, new | old | |d| x 20. The verdict.
      bands    a 92 degree turn, geometry only: the frozen read fills the two
               side bands (outside the turned 1024 px) by reflection, Render
               reads them. Mean |new - old| in the bands against the centre,
               where both hold the same tissue -- the centre is the decoy.
      speed    the full gap (`DomainGapConfig`'s own, as the bench uses),
               photos/s for both."""
    gap = DomainGapConfig()
    plan = PlanSpec('ladder', RUNGS, camera=render_spec(gap, S1_SENSOR))
    samples = list(TileSampler(wsi, mask, SamplerConfig(n_per_rung=args.n, seed=41))
                   .sample(plan.plans_for(wsi)))
    positions = [dict(x=int(s.meta.fov_rect[0]), y=int(s.meta.fov_rect[1]),
                      rung=float(s.meta.ds)) for s in samples]
    reader = SlideReader(wsi)

    def rng(k):
        return random.Random(1000 + k)

    # same: no gap
    bare = DomainGapConfig(rotation_choices=(0,),
                           angle_jitter_deg=0.0, stage_shift_max=0,
                           geometric=False, photometric=False)
    still = Render(reader, S1_SENSOR, bare, ds=1.0)
    rows, ok, worst = [], True, None
    for k, pos in enumerate(positions):
        new, _ = still.at(pos['rung']).capture_with_gt(pos['x'], pos['y'], rng=rng(k))
        old = _old_s1photo(reader, pos, bare, rng(k))
        if new is None or old is None or new.shape != old.shape:
            rows.append(dict(slide=name, rung=pos['rung'], x=pos['x'], y=pos['y'],
                             verdict='FAIL', note='a read missing or a shape apart'))
            ok = False
            continue
        row, kind = _s1photo_same(reader, pos, new, old, still.spec)
        row.update(slide=name)
        rows.append(row)
        ok &= row['verdict'] == 'PASS'
        if worst is None or row['max_inner'] > worst[0]:
            worst = (row['max_inner'], pos, new, old)
    for rung in sorted({r['rung'] for r in rows}):
        g = [r for r in rows if r['rung'] == rung]
        r0 = g[0]
        print(f'  s1photo  same   rung {rung:g}: L{r0.get("level", "?")} '
              f'lds {r0.get("lds", float("nan")):.6f}  resized {r0.get("resized")}  '
              f'shift {r0.get("shift_level", float("nan")):.4f} level px  '
              f'{r0.get("kind", "")}  identical {sum(r.get("max_all") == 0 for r in g)}'
              f'/{len(g)}  max|d| border {max(r.get("max_border", -1) for r in g)} '
              f'inner {max(r.get("max_inner", -1) for r in g)}  mean|d| inner '
              f'{max(r.get("mean_inner", -1) for r in g):.3f} vs 1 px decoy '
              f'{min(r.get("mean_decoy", -1) for r in g):.2f}  '
              f'{"PASS" if all(r["verdict"] == "PASS" for r in g) else "FAIL"}',
              flush=True)
    write_csv(rows, Path(args.out_dir) / f's1photo_same_{name}.csv')
    if worst is not None:
        _, pos, new, old = worst
        d = np.abs(new.astype(np.int16) - old).max(axis=2)
        heat = np.clip(d * 20, 0, 255).astype(np.uint8)
        strip = np.concatenate([new, old, np.stack([heat] * 3, axis=2)], axis=1)
        path = Path(args.out_dir) / f's1photo_worst_{name}.png'
        from PIL import Image                                    # noqa: PLC0415
        Image.fromarray(strip).save(path)
        print(f'  s1photo  same   worst inner: rung {pos["rung"]:g} '
              f'({pos["x"]}, {pos["y"]}) -> {path}  (new | old | |d| x 20)',
              flush=True)
    print(f'  s1photo  same   {"PASS" if ok else "FAIL"}', flush=True)

    # bands: geometry only, an explicit 92 degree turn
    geo = DomainGapConfig(scale_range=(1.0, 1.0),
                          stage_shift_max=0, photometric=False)
    turned = Render(reader, S1_SENSOR, geo, ds=1.0)
    w, h = S1_SENSOR
    lo, hi = (w - h) // 2, (w + h) // 2         # the turned rectangle's columns
    band, centre = [], []
    for k, pos in enumerate(positions):
        new, _ = turned.at(pos['rung']).capture_with_gt(
            pos['x'], pos['y'], rotation=92.0, rng=rng(k))
        old = _old_s1photo(reader, pos, geo, rng(k), rotation=92.0)
        if new is None or old is None:
            continue
        d = np.abs(new.astype(np.int16) - old).mean(axis=2)
        band.append(float(np.concatenate([d[:, :lo - 16], d[:, hi + 16:]], 1).mean()))
        centre.append(float(d[:, lo + 48:hi - 48].mean()))
    if band:
        print(f'  s1photo  bands  {len(band)} photos  mean|new-old| side bands '
              f'{np.median(band):.1f}  centre {np.median(centre):.1f} (median)',
              flush=True)

    # corners: the rotating read is the square of the SENSOR's diagonal
    # (FovGeometry.square_out), which holds the sensor at scale 1 at any angle
    # but not a zoom-out (scale < 1 sees 1/scale more) nor the lens margin the
    # scene stage crops to. At 0.9 and 3 degrees the scene needs ~1807 px of a
    # 1768 px read, so `apply_scale` would pad the corners by reflection. The
    # reference renders the same scene from a read with real pixels far beyond
    # it (a plain read grown by S1_CORNER_REF output px, centred alike); the
    # centre, the same tissue in both, is the decoy.
    zoom = DomainGapConfig(scale_range=(0.9, 0.9),
                           angle_jitter_deg=0.0, stage_shift_max=0,
                           photometric=False)
    zoomed = Render(reader, S1_SENSOR, zoom, ds=1.0)
    w, h = S1_SENSOR
    c = S1_CORNER
    corner, middle = [], []
    for k, pos in enumerate(positions):
        cam = zoomed.at(pos['rung'])
        new, _ = cam.capture_with_gt(pos['x'], pos['y'], rotation=3.0, rng=rng(k))
        raw = reader.read(pos['x'], pos['y'],
                          ReadSpec(w, h, rotates=False, margin_out=S1_CORNER_REF),
                          pos['rung'])
        if new is None or raw is None:
            continue
        ref, _ = simulate_with_gt(raw, cfg=cam.cfg, rng=rng(k), rotation=3.0,
                                  output_wh=(w, h))
        d = np.abs(new.astype(np.int16) - ref).mean(axis=2)
        corner.append(float(np.mean([d[:c, :c].mean(), d[:c, -c:].mean(),
                                     d[-c:, :c].mean(), d[-c:, -c:].mean()])))
        middle.append(float(d[h // 2 - c:h // 2 + c, w // 2 - c:w // 2 + c].mean()))
    if corner:
        print(f'  s1photo  corners scale 0.9, 3 deg: {len(corner)} photos  mean|new-ref| '
              f'in the {c} px corners {np.median(corner):.1f}  centre '
              f'{np.median(middle):.1f} (median; ref = the scene from a read '
              f'{S1_CORNER_REF} px wider)', flush=True)

    # speed: the bench's own gap
    def made_new():
        cam = Render(reader, S1_SENSOR, gap, ds=1.0)
        for k, pos in enumerate(positions):
            cam.at(pos['rung']).capture_with_gt(pos['x'], pos['y'], rng=rng(k))

    def made_old():
        for k, pos in enumerate(positions):
            _old_s1photo(reader, pos, None, rng(k))
    t.add(name, 's1photo', 'Render (bench now)', len(positions),
          _best(made_new, args.repeats))
    t.add(name, 's1photo', 'sensor read (frozen)', len(positions),
          _best(made_old, args.repeats))
    return ok


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
                        # int(x / ds) bookkeeping's error IF openslide is bilinear
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



def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--slides', nargs='+',
                    default=['BRACS_1228', 'S1104233,G7E,110208'])
    ap.add_argument('--flows', nargs='+', choices=FLOWS, default=list(TIMING_FLOWS))
    ap.add_argument('--n', type=int, default=6, help='positions per rung / level')
    ap.add_argument('--grid-levels', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--block-rows', type=int, default=8)
    ap.add_argument('--repeats', type=int, default=2)
    ap.add_argument('--pca-tiles', type=int, default=4096,
                    help='pca: lattice tiles compared (the first, row-major)')
    ap.add_argument('--pca-points', type=int, default=500,
                    help='pca: scattered positions compared, as the fit reads')
    ap.add_argument('--phase-levels', type=int, nargs='+', default=[1, 2, 3],
                    help='phase: levels of every slide (a level it lacks is skipped)')
    ap.add_argument('--mask-cache-job', default='MppRoutingHead',
                    help="phase and origins: masks from this job's cache (hest)")
    ap.add_argument('--origin-per-dataset', type=int, default=8,
                    help='origins: slides sampled per dataset')
    ap.add_argument('--seed', type=int, default=0, help='phase and origins')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    out = Path(args.out or job_result_dir('DiagReadExp'))
    out.mkdir(parents=True, exist_ok=True)
    args.out_dir = str(out)
    budget = CpuBudget.for_job(processes=1).apply()
    print(f'diag_read_exp  flows {args.flows}  repeats {args.repeats}\n'
          f'  {budget.line()}\n  out {out}', flush=True)

    if 'origins' in args.flows:
        rows = []
        check_origins(args.mask_cache_job, 'hest', args.origin_per_dataset,
                      args.seed, rows)
        write_csv(rows, out / 'origins.csv')

    t = Timings()
    ok = True
    phase_rows = []
    per_slide = [f for f in args.flows if f != 'origins']
    # phase reads tissue spots off the hest masks of --mask-cache-job, as it
    # always did; the timing flows keep their own hsv masks, so their numbers
    # stay comparable with earlier speed.csv files
    phase_masks = (MaskMaker(MASK_RECIPES['hest'],
                             cache_root=Cache.cache_root(args.mask_cache_job, 'mask'))
                   if 'phase' in per_slide else None)
    with MaskMaker(MASK_RECIPES['hsv']) as masks:
        for name in (args.slides if per_slide else []):
            dataset = _dataset_of(name)
            wsi = SafeSlide(locate(name, dataset=dataset).path)
            print(f'\n======== {name} ({dataset})  levels '
                  + ', '.join(f'{d:.5g}' for d in wsi.level_downsamples)
                  + ' ========', flush=True)
            if 'phase' in per_slide:
                hest, _ = phase_masks.mask(wsi)
                for level in args.phase_levels:
                    if level < wsi.level_count:
                        check_phase(wsi, hest, level, args.seed, phase_rows)
            if set(per_slide) & (set(TIMING_FLOWS) | {'s1photo'}):
                mask, _ = masks.mask(wsi)
            if 'grid' in per_slide:
                flow_grid(t, args, name, wsi, mask, budget)
            if 'capture' in per_slide:
                flow_capture(t, args, dataset, name, wsi, mask)
            if 'tiles' in per_slide:
                flow_tiles(t, args, name, wsi, mask)
            if 'fov' in per_slide:
                flow_fov(t, args, name, wsi, mask)
            if 'pca' in per_slide:
                ok &= flow_pca(t, args, name, wsi, budget)
            if 's1photo' in per_slide:
                ok &= flow_s1photo(t, args, name, wsi, mask)
    if phase_masks is not None:
        phase_masks.close()
    write_csv(phase_rows, out / 'phase.csv')
    write_csv(t.rows, out / 'speed.csv')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
