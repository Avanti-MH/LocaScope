#!/usr/bin/env python3
"""diag_read_exp -- how fast the read path is, flow by flow.

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
    fov       FovSupply.bank() of the window bench's still FoV: shots/s

Each is run --repeats times and the best kept. Nothing is compared with
anything: the correctness of each path is its tests' (TestReadPath.sh).

HISTORY. Until 2026-10-03 this file held the planned refactor written out
whole beside the production code, and ran every flow through both: 566/566
reads identical, 14/14 timings not slower (result/DiagReadExp/DiagReadExp/).
The refactor then moved into production and the old side was deleted, so the
comparison cannot be run again. `diag_render_reads.py`, the exploratory
diagnostic the numbers in ARCHITECTURE.md come from, was deleted at the same
time: everything it compared against is gone.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from _paths import job_result_dir, setup_import_paths                 # noqa: E402

setup_import_paths()
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT / 'training/MppRoutingHead') not in sys.path:
    sys.path.insert(0, str(_ROOT / 'training/MppRoutingHead'))

from AccessDatasets import locate                                     # noqa: E402
from CpuBudget import CpuBudget                                       # noqa: E402
from ReadGeometry import ReadSpec                                     # noqa: E402
from SafeSlide import SafeSlide                                       # noqa: E402
from SlideReader import SlideReader                                   # noqa: E402
from PatchingLib import region_grids                                  # noqa: E402
from TileSampler import PlanSpec, SamplerConfig, TileSampler          # noqa: E402
from TissueMaskConfig import MASK_RECIPES, MaskMaker                  # noqa: E402
from camera import Render, render_spec                                # noqa: E402
from config import DomainGapConfig                                    # noqa: E402
from generator import FovSupply                                       # noqa: E402
import Datasets                                                       # noqa: E402

TILE = 256
RUNGS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
FLOWS = ('grid', 'capture', 'tiles', 'fov')
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
    still = DomainGapConfig(wh_ratio='45:32', MPixels=1.47456,
                            rotation_choices=(0,), angle_jitter_deg=0.0,
                            scale_range=(1.0, 1.0), query_mpp_jitter=0.0,
                            stage_shift_max=0)
    reader = SlideReader(wsi)
    for level in range(min(2, wsi.level_count)):
        ds = float(wsi.level_downsamples[level])
        cfg = SamplerConfig(n_per_rung=args.n, seed=21 + level)

        def bank():
            cam = Render(reader, still, ds=ds, seed=21 + level)
            return FovSupply(cam, mask, cfg).bank()
        try:
            n = len(bank())
        except RuntimeError:
            continue
        t.add(name, 'fov', f'bank still L{level}', n, _best(bank, args.repeats))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--slides', nargs='+',
                    default=['BRACS_1228', 'S1104233,G7E,110208'])
    ap.add_argument('--flows', nargs='+', choices=FLOWS, default=list(FLOWS))
    ap.add_argument('--n', type=int, default=6, help='positions per rung / level')
    ap.add_argument('--grid-levels', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--block-rows', type=int, default=8)
    ap.add_argument('--repeats', type=int, default=2)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    out = Path(args.out or job_result_dir('DiagReadExp'))
    out.mkdir(parents=True, exist_ok=True)
    budget = CpuBudget.for_job(processes=1).apply()
    print(f'diag_read_exp  flows {args.flows}  repeats {args.repeats}\n'
          f'  {budget.line()}\n  out {out}', flush=True)

    t = Timings()
    with MaskMaker(MASK_RECIPES['hsv']) as masks:
        for name in args.slides:
            dataset = _dataset_of(name)
            wsi = SafeSlide(locate(name, dataset=dataset).path)
            mask, _ = masks.mask(wsi)
            print(f'\n======== {name} ({dataset})  levels '
                  + ', '.join(f'{d:.5g}' for d in wsi.level_downsamples)
                  + ' ========', flush=True)
            if 'grid' in args.flows:
                flow_grid(t, args, name, wsi, mask, budget)
            if 'capture' in args.flows:
                flow_capture(t, args, dataset, name, wsi, mask)
            if 'tiles' in args.flows:
                flow_tiles(t, args, name, wsi, mask)
            if 'fov' in args.flows:
                flow_fov(t, args, name, wsi, mask)

    if t.rows:
        with open(out / 'speed.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(t.rows[0]))
            w.writeheader()
            w.writerows(t.rows)
        print(f'\n{out / "speed.csv"}  ({len(t.rows)} rows)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
