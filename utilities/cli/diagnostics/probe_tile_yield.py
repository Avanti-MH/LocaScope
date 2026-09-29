#!/usr/bin/env python3
"""How many tiles does each (slide, tile_size, ds) yield, and in which buckets?

    python utilities/cli/diagnostics/probe_tile_yield.py [--seg uni2_pca] [--mask-cache-job BuildMaskStore] [--n 500]

Outputs (in result/<SLURM_JOB_NAME or ProbeTileYield>/):
    tile_yield.png
    tile_yield_definitions.csv
    tile_yield.csv            one row per cell of the grid

spec.md 12 step 3b. NO MODEL and no encoding -- it reads the masks the store
already holds and runs the same rejection sampling the extraction will, so the
only cost is mask arithmetic.

WHAT THIS DECIDES
-----------------
Three things, which is why it is worth its own step rather than being discovered
during extraction:

    richness floors   whether `bg30_50`'s 50 per cent is reachable. The wall
                      moves left with
                      it, and by how much has never been measured -- the numbers
                      in spec.md 6.5 are from hsv masks at ds 32, and the mask
                      in use now is ds 14, where a small gap in the tissue is
                      resolved rather than averaged away.
    rung balance      `align-min` (every cell takes min(counts)) or
                      `loss-weight` (take what there is, weight the detector CE
                      by 1/count). If the worst cell holds 300 the first is
                      cheap; if it holds 40 it throws away nine tenths.
    reachable rungs   spec.md 6.5's footprint table says three of the eighteen
                      (tile, ds) cells are empty at ratio 0.5. At 0.75 more may
                      be, and `model_512` / `model_1024` lose rungs accordingly.

Doing nothing with the answer is the failure this prevents. Without it, "take
what each cell gives" becomes the policy by not being a decision -- and ds 1
yielding 500 against ds 32 yielding 40 means the detector sees the fine rungs
twelve times more often, with nothing anywhere saying so.

WHAT IT DOES NOT MEASURE
-------------------------
How many rejection TRIES each cell burned. `TileSampler.sample` returns the
tiles and not the attempt count, and adding one would mean changing a module
with existing callers. The yield against the request answers the three questions
above on its own: a cell that returns 500/500 had room to spare, and one that
returns 40/500 after the whole budget did not.

THE PRE-TILE COLUMN
--------------------
`clipped` counts sampled positions whose PRE-TILE runs off the scanned
rectangle. Training tiles are cut from a 3x pre-tile so the homography warp has
real tissue to sample instead of black (spec.md 6.6), and near the edge of the
scanned area that pre-tile is short. Those positions still yield a tile; the
tile just carries some black after warping, and the `valid_mask` assertion in
Homographic Adaptation is what reports it per draw.

It is here because the probe already knows every sampled position and the
scanned bounds, so counting is free -- and because a rung whose positions are
mostly near an edge is a rung whose labels will be quietly worse.

WHICH WSIs
----------
Default (no --dataset, no --wsi): every WSI already in the `--seg` recipe's
mask cache (`utilities/cli/build_cache/build_mask_store.py` writes it) -- this
tool cannot probe a slide with no mask to read, so "everything the store
already has" is this tool's own natural default, the same way
`diag_wsi_scale.py` defaults to every registered dataset. `--dataset`/
`--wsi` (via `WsiSelection`) narrow to a specific set -- entries with no
mask in the store are skipped with a message, not a crash.
"""

from __future__ import annotations

import argparse
import dataclasses
import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, '..', '..'),
          os.path.join(_HERE, '..', '..', '..', 'aiNNModel')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                              # noqa: E402
import matplotlib                                               # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                 # noqa: E402

from _paths import job_result_dir, setup_import_paths          # noqa: E402

setup_import_paths()

from Cache import cache_root, find, read_meta, wsi_stem_of      # noqa: E402
from TissueMaskConfig import MASK_RECIPES                        # noqa: E402
from SafeSlide import SafeSlide                                  # noqa: E402
from TileSampler import (PRE_TILE_FACTOR, OverlapConfig,      # noqa: E402
                         SamplerConfig, TileSampler)
from TissueMask import SlideMask                # noqa: E402
from DsLadder import DEFAULT_RUNGS, DsLadder              # noqa: E402
from WsiSelection import resolve_wsi_paths                       # noqa: E402


#: Where build_mask_store.py pre-warms masks by default.
DEFAULT_MASK_CACHE_JOB = 'BuildMaskStore'


#: Every string a reader will see. Rewritten on every run so the definitions
#: cannot drift from the code that produced them (ClaudeRules section 12).
DEFINITIONS = [
    ('n_requested', 'tiles asked for in this cell'),
    ('n_got', 'tiles the rejection sampler returned before the budget ran out'),
    ('yield', 'n_got / n_requested. 1.0 means the cell had room to spare'),
    ('footprint_l0',
     'tile_size * ds -- the LEVEL-0 square one tile covers, and the quantity '
     'the tissue mask has to accommodate. spec.md 6.5 measured the wall at '
     '8192 (fine), 16384 (0 tiles), 32768 (no region fits)'),
    ('clipped',
     'sampled positions whose 3x PRE-TILE runs off the scanned rectangle. Those '
     'tiles still exist; their warped views carry some black at one edge'),
    ('level', 'the pyramid level this rung reads, via DsLadder'),
    ('read_size', 'LEVEL pixels read per tile, before shrinking to tile_size'),
    ('sampler_id', 'sha256[:8] of every field that decides which tiles came out'),
    ('floor_frame', "'ask' = floors are a share of n_per_rung, 'taken' = of what is achievable"),
    ('n_goal', 'the count the mix was scaled to. Equals n_requested under floor_frame=ask'),
    ('n_got_if_taken', 'what the OTHER frame would have returned -- the cost of the switch'),
    ('n_goal_if_taken', 'and what it would have scaled the rung to'),
    ('n_admissible', 'candidates whose bucket has a non-zero cap -- the old gate'),
    ('n_below_floor', 'tiles a bucket FLOOR asked for and the slide could not supply'),
    ('n_spilled', 'tiles that reached a cap-only bucket because another fell short'),
    ('supply_<bucket>', 'candidates in that bucket, before any cap -- the ceiling on its floor'),
    ('got_<bucket>', 'tiles actually taken from it. supply minus got is the cap biting'),
    ('floor_<bucket>', 'what the floor asked for, in tiles'),
]


class TileYieldProbe:
    """`run(entries)` is the shape every diagnostic in this directory shares
    (see `WsiSelection.py`) -- but this tool ALSO needs a prebuilt mask per
    slide, so an entry with none is skipped rather than probed. See the
    module docstring's "WHICH WSIs" for the default when no entries are
    given at all.
    """

    def __init__(self, seg: str = 'uni2_pca',
                mask_cache_job: str = DEFAULT_MASK_CACHE_JOB,
                tile_sizes=(256, 512, 1024), ds_values=DEFAULT_RUNGS,
                n: int = 500, candidates: str = 'lattice', grid_step: int = 0,
                max_overlap: float = 0.0, overlapping_share: float = 0.0,
                max_tries: int = 2500, seed: int = 0, out_dir=None):
        self.mask_cfg = MASK_RECIPES[seg]
        self.mask_root = cache_root(mask_cache_job, 'mask')
        self.seg_dir = self.mask_root / self.mask_cfg.seg_id()
        self.tile_sizes = list(tile_sizes)
        self.ds_values = list(ds_values)
        self.n = n
        self.candidates = candidates
        self.grid_step = grid_step
        self.max_overlap = max_overlap
        self.overlapping_share = overlapping_share
        self.max_tries = max_tries
        self.seed = seed
        self.out_dir = out_dir or job_result_dir('ProbeTileYield')
        os.makedirs(self.out_dir, exist_ok=True)

    def default_entries(self) -> list:
        """Every WSI the mask store already holds -- this tool's own
        default when the caller names no `--dataset`/`--wsi`."""
        found = [read_meta(p) for p in find(self.seg_dir, '*/mask_meta.json')]
        return [dict(dataset=None, wsi_name=wsi_stem_of(m['wsi_path']),
                     path=m['wsi_path']) for m in found]

    def _probe_cell(self, wsi, trm, tile_size, rung) -> dict:
        """One (tile_size, ds) cell. Returns a row dict.

        `supply_*` is `preflight`'s histogram of the whole candidate pool,
        which is the number the floors have to live within. `got_*` is what
        the fill actually took. The two differ by exactly the caps -- which
        is the point: a bucket short in `got` but plentiful in `supply` was
        held back, and one short in both is a slide that does not have it.

        Goes through `TileSampler` rather than calling `has_tissue`
        directly, because the yield is not only about the tissue fraction:
        the sampler takes a `patchable` view per rung, and that is what
        empties a cell outright when no region can hold the window. A probe that skipped
        that would over-report exactly the cells spec.md 6.5 says are empty.
        """
        plan = DsLadder(rungs=(float(rung),)).plan(wsi.level_downsamples, tile_size)[0]

        # tile_size in LEVEL pixels is `read_size`, so the level-0 footprint
        # the sampler enforces is read_size * level_ds == tile_size * rung.
        # The plan IS the argument now: it already says which level to read,
        # what the footprint is and what must fit.
        #
        # `self.candidates` is the arm. The 216-cell table of 2026-08-26 was
        # cut with 'random', so a re-run at 'lattice' is a DIFFERENT
        # measurement, not a correction of that one.
        cfg = SamplerConfig(
            tile=plan.tile_size, n_per_rung=self.n, seed=self.seed,
            candidates=self.candidates,
            max_tries_per_tile=max(1, self.max_tries // max(self.n, 1)),
            overlap=OverlapConfig(grid_step=self.grid_step,
                                  max_overlap_ratio=self.max_overlap,
                                  overlapping_share=self.overlapping_share))
        sampler = TileSampler(wsi, trm, cfg)
        pre = sampler.preflight([plan])[0]
        sampler.sample([plan])
        tiles = [s.meta for s in sampler]
        rep_ = sampler.reports[plan.rung_ds]
        rich = cfg.richness

        origin, span = _scanned(trm)
        footprint = plan.requested_footprint_l0
        margin = footprint * (PRE_TILE_FACTOR - 1) / 2.0
        clipped = sum(1 for t in tiles
                      if t.x - margin < origin[0]
                      or t.y - margin < origin[1]
                      or t.x + footprint + margin > origin[0] + span[0]
                      or t.y + footprint + margin > origin[1] + span[1])

        row_buckets = {}
        for i, name in enumerate(rich.names):
            row_buckets[f'supply_{name}'] = int(pre.supply.get(name, 0))
            row_buckets[f'got_{name}'] = int(rep_.per_bucket.get(name, 0))
            row_buckets[f'floor_{name}'] = int(round(rich.floors[i] * self.n))

        # BOTH FRAMES, EVERY CELL. `floor_frame` is a switch (RichnessConfig),
        # and the whole point of a switch is that the number deciding it must
        # be in front of whoever flips it. Running the second arm costs one
        # more pass over the same mask arithmetic and no pixels at all, so
        # the CSV carries what the other frame WOULD have returned even while
        # this one is what runs.
        alt_frame = 'taken' if rich.floor_frame == 'ask' else 'ask'
        alt_cfg = dataclasses.replace(
            cfg, richness=dataclasses.replace(rich, floor_frame=alt_frame))
        alt = TileSampler(wsi, trm, alt_cfg)
        alt.sample([plan])
        alt_rep = alt.reports[plan.rung_ds]

        return {'wsi_stem': wsi_stem_of(wsi),
                'sampler_id': cfg.sampler_id(),
                'floor_frame': rich.floor_frame,
                'n_goal': rep_.n_goal,
                f'n_got_if_{alt_frame}': alt_rep.n_taken,
                f'n_goal_if_{alt_frame}': alt_rep.n_goal,
                'n_below_floor': rep_.n_below_floor,
                'n_spilled': rep_.n_spilled,
                'n_admissible': pre.n_admissible,
                **row_buckets,
                'tile_size': tile_size, 'ds': float(rung),
                'level': plan.level, 'level_ds': plan.level_ds,
                'read_size': plan.read_size,
                'footprint_l0': int(footprint),
                'n_requested': self.n, 'n_got': len(tiles),
                'yield': len(tiles) / self.n if self.n else 0.0,
                'clipped': clipped,
                'clipped_frac': clipped / len(tiles) if tiles else 0.0}

    def run_one(self, entry: dict) -> list:
        """One WSI -> its rows across every (tile_size, ds) cell, or `[]`
        with a printed message if the mask store has nothing for it."""
        stem = wsi_stem_of(entry['path'])
        folder = self.seg_dir / stem
        if not (folder / 'mask_meta.json').exists():
            print(f'    no mask under {folder}', flush=True)
            return []
        meta = read_meta(folder / 'mask_meta.json')
        slide_mask = SlideMask.load(folder / 'mask.safetensors')
        print(f'    mask {meta["rows"]}x{meta["cols"]} at ds {meta["mask_ds"]:.0f}, '
              f'tissue {meta["fraction"]:.1%}   ({self.mask_cfg.seg_id()})',
              flush=True)

        rows = []
        with SafeSlide(entry['path']) as wsi:
            # The recipe's regions -- exactly what a sampler is handed; the
            # per-rung `patchable` step is the sampler's own and part of what
            # is being probed.
            trm = self.mask_cfg.regions(wsi, slide_mask)
            print(f'    {len(trm.tissue_regions)} tissue regions', flush=True)
            for tile_size in self.tile_sizes:
                line = []
                for rung in self.ds_values:
                    row = self._probe_cell(wsi, trm, tile_size, rung)
                    row['dataset'] = entry.get('dataset')
                    rows.append(row)
                    flag = '!' if row['n_below_floor'] else ''
                    line.append(f"ds{rung:g}:{row['n_got']}{flag}")
                print(f"    tile={tile_size}   {'  '.join(line)}"
                      f"    (! = a floor went unmet)", flush=True)
        return rows

    def run(self, entries: list) -> list:
        rows = []
        entries = entries or self.default_entries()
        for index, entry in enumerate(entries, 1):
            print(f'\n[{index}/{len(entries)}] {entry["wsi_name"]}', flush=True)
            rows += self.run_one(entry)

        if not rows:
            print('nothing probed')
            return rows

        summary = os.path.join(self.out_dir, 'tile_yield.csv')
        with open(summary, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        figure = self._draw(rows)
        print(f'\nSaved {summary}   ({len(rows)} cells)')
        print(f'Saved {figure}')

        worst = min(rows, key=lambda r: r['n_got'])
        empty = [r for r in rows if r['n_got'] == 0]
        print(f'\nWorst cell: {worst["wsi_stem"]} '
              f'tile={worst["tile_size"]} ds={worst["ds"]:g} -> '
              f'{worst["n_got"]}/{self.n}')
        print(f'Empty cells: {len(empty)} of {len(rows)}')

        unmet = [r for r in rows if r['n_below_floor']]
        print(f'\nCells missing a floor: {len(unmet)} of {len(rows)}')
        for r in sorted(unmet, key=lambda r: -r['n_below_floor'])[:12]:
            short = ' '.join(
                f'{k.replace("floor_", "")}:{r[k] - r[k.replace("floor_", "got_")]}'
                for k in r if k.startswith('floor_')
                and r[k] > r[k.replace('floor_', 'got_')])
            print(f'  {r["wsi_stem"]:28s} tile={r["tile_size"]:4d} '
                  f'ds={r["ds"]:<5g} short {r["n_below_floor"]:4d}   {short}')
        print('\nWhat to do with it:')
        print('  - worst cell in the hundreds -> align-min is affordable')
        print('  - worst cell in the tens     -> loss-weight, and record why')
        print('  - 0.75 empties cells that 0.5 fills -> that is the cost of '
              '0.75, and it is a decision, not a default')
        return rows

    def _draw(self, rows) -> str:
        """One panel per tile_size: yield against ds, a line per slide.

        Which cells are empty is the question, so the floor line must be
        comparable within a panel -- solid against dashed on the same axes,
        not two figures a reader has to hold in their head.
        """
        tiles = sorted({r['tile_size'] for r in rows})
        slides = sorted({r['wsi_stem'] for r in rows})
        colours = plt.cm.tab10(np.linspace(0, 1, max(len(slides), 2)))

        fig, axes = plt.subplots(1, len(tiles), figsize=(6 * len(tiles), 5.5),
                                 sharey=True, squeeze=False)
        for axis, tile in zip(axes[0], tiles):
            for colour, stem in zip(colours, slides):
                pts = sorted((r['ds'], r['yield']) for r in rows
                             if r['tile_size'] == tile and r['wsi_stem'] == stem)
                if pts:
                    axis.plot([p[0] for p in pts], [p[1] for p in pts],
                              '-', color=colour, marker='o', markersize=3,
                              label=stem)
                # The FLOOR line: what share of the rung `bg30_50` is owed.
                # A yield curve above it says nothing on its own -- the floor
                # is about one bucket, not the total -- so the dashed line is
                # drawn against the bucket's own supply share.
                sup = sorted(
                    (r['ds'], (r['supply_bg30_50'] / max(1, r['n_admissible'])))
                    for r in rows
                    if r['tile_size'] == tile and r['wsi_stem'] == stem)
                if sup:
                    axis.plot([q[0] for q in sup], [q[1] for q in sup],
                              '--', color=colour, linewidth=0.9, alpha=0.6)
            axis.set_xscale('log', base=2)
            axis.set_xlabel('ds (rung)')
            axis.set_title(f'tile {tile}\nfootprint = {tile} x ds', fontsize=10)
            axis.axhline(1.0, color='0.8', linewidth=0.8)
            axis.set_ylim(-0.05, 1.05)
        axes[0][0].set_ylabel(f'yield  (n_got / {self.n})')
        axes[0][-1].legend(fontsize=6, loc='lower left', ncol=1)

        fig.suptitle(
            f'Tile yield per (slide, tile_size, ds)   '
            f'budget {self.max_tries} tries for {self.n} tiles\n'
            f'solid = yield, dashed = bg30_50 share of the admissible pool.  A cell '
            f'at yield 0 is a rung that model_{tiles[-1]} cannot have', fontsize=12)
        for axis in axes[0]:
            axis.axhline(0.50, color='crimson', linewidth=0.8, linestyle=':')
        fig.text(0.5, 0.005,
                 'the dotted line is the bg30_50 FLOOR (50 per cent). Where a '
                 "slide's dashed curve falls under it, that rung cannot supply the "
                 'floor and the deficit spills into bg50_70 and bg70_85 -- or the '
                 'rung comes up short. Yield below 1.0 is positions running out, '
                 'not tissue.',
                 ha='center', fontsize=8, color='0.35')
        fig.tight_layout(rect=(0, 0.03, 1, 0.9))

        path = os.path.join(self.out_dir, 'tile_yield.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        with open(os.path.join(self.out_dir, 'tile_yield_definitions.csv'), 'w',
                  newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['term', 'means'])
            writer.writerows(DEFINITIONS)
        return path


def _scanned(trm):
    """(origin, span) in LEVEL-0, off the mask rather than off the slide.

    The mask knows where it starts -- its SlideMask recorded it -- and asking
    the slide again would re-derive `openslide.bounds-*` in a second place.
    On a MIRAX those differ from (0, 0) by tens of thousands of pixels, so
    the two have to be the same number and the cheapest way to guarantee
    that is to have only one.
    """
    origin = (trm.origin_x, trm.origin_y)
    span = (int(round(trm.mask_ds_x * trm.main_mask.shape[1])),
            int(round(trm.mask_ds_y * trm.main_mask.shape[0])))
    return origin, span


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seg', choices=sorted(MASK_RECIPES), default='uni2_pca',
                    help='mask recipe whose cached masks to probe')
    ap.add_argument('--mask-cache-job', default=DEFAULT_MASK_CACHE_JOB,
                    help='the job that made the mask cache: result/cache/'
                         '<this>_mask/ (build_mask_store.py writes there)')
    ap.add_argument('--dataset', nargs='+', default=None,
                    help='narrow to these datasets (via AccessDatasets) -- '
                         'still needs a cached mask per slide')
    ap.add_argument('--wsi', nargs='+', default=None,
                    help='explicit WSI path(s). Default (with no --dataset '
                         'either): every mask already in the cache')
    ap.add_argument('--val-only', action='store_true')
    # NO --tissue-ratio: the gate it swept is gone, and the axis with it.
    # What replaced the question is `supply_<bucket>` against `floor_<bucket>`.
    ap.add_argument('--tissue-ratio', type=float, nargs='+', default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument('--tile-size', dest='tile_sizes', type=int, nargs='+',
                    default=[256, 512, 1024],
                    help='the three models of spec.md 6.5')
    ap.add_argument('--ds', dest='ds_values', type=float, nargs='+',
                    default=list(DEFAULT_RUNGS))
    ap.add_argument('--n', type=int, default=500,
                    help='tiles per cell. 500 is the production number, so the '
                         'yield reads directly as "what extraction will get"')
    ap.add_argument('--candidates', default='lattice',
                    choices=('lattice', 'random'),
                    help="'random' is the sampler this replaced, kept as the "
                         'control arm.')
    ap.add_argument('--grid-step', type=int, default=0,
                    help='lattice step in OUTPUT px. 0 means the tile, i.e. '
                         'disjoint')
    ap.add_argument('--max-overlap', type=float, default=0.0,
                    help='largest area fraction any two tiles of a rung may '
                         'share')
    ap.add_argument('--overlapping-share', type=float, default=0.0,
                    help='largest share of a rung that may overlap anything '
                         'at all. 0 forbids it outright')
    ap.add_argument('--max-tries', type=int, default=2500,
                    help='rejection budget per cell')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    if args.tissue_ratio is not None:
        ap.error(
            '--tissue-ratio is gone with the gate it swept. The question it '
            'used to answer -- how much does a stricter cut cost -- is now '
            'supply_<bucket> against floor_<bucket>, because the cut IS the '
            'richness caps. See RichnessConfig.')

    out_dir = args.out or job_result_dir('ProbeTileYield')

    if args.dataset or args.wsi:
        entries = resolve_wsi_paths(dataset=args.dataset, wsi=args.wsi,
                                    val_only=args.val_only)
    else:
        entries = None   # TileYieldProbe.run() falls back to the mask store

    prober = TileYieldProbe(
        seg=args.seg, mask_cache_job=args.mask_cache_job,
        tile_sizes=args.tile_sizes,
        ds_values=args.ds_values, n=args.n, candidates=args.candidates,
        grid_step=args.grid_step, max_overlap=args.max_overlap,
        overlapping_share=args.overlapping_share, max_tries=args.max_tries,
        seed=args.seed, out_dir=out_dir)

    if entries is None:
        entries = prober.default_entries()
        if not entries:
            print(f'no masks under {prober.seg_dir}. Run '
                  f'utilities/cli/build_cache/build_mask_store.py first.')
            return 1
        print(f'{len(entries)} slides from the mask store', flush=True)

    rows = prober.run(entries)
    return 0 if rows else 1


if __name__ == '__main__':
    sys.exit(main())
