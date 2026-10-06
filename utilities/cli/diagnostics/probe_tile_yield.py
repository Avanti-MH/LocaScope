#!/usr/bin/env python3
"""How many tiles does each (slide, tile_size, ds) yield, and in which buckets?

    python utilities/cli/diagnostics/probe_tile_yield.py [--seg uni2_pca] [--mask-cache-job BuildMaskStore] [--n 500]

Outputs (in result/<SLURM_JOB_NAME or ProbeTileYield>/):
    tile_yield.png
    tile_yield_definitions.csv
    tile_yield.csv            one row per cell of the grid
    feature_map_size.csv      --feature-map on|only: one row per (slide, tile, level)

spec.md 12 step 3b. NO MODEL and no encoding -- it reads the masks the store
already holds and runs the same rejection sampling the extraction will, so the
only cost is mask arithmetic.

THE FEATURE-MAP REPORT  (--feature-map on | only)
-------------------------------------------------
A different question on the same slides and the same masks: not how many tiles a
SAMPLER can draw, but how large the WHOLE-SLIDE feature map is -- what
`FeatureMapCache` stores, one row of features per tile of the grid. For every
pyramid level the slide really has (up to the coarsest --ds):

    tiles      the whole grid, four ways:
                   no mask, main       one region, the scanned rectangle
                   no mask, overlap    ... plus the grid offset by half a tile
                   mask,    main       the recipe's regions that hold a tile
                   mask,    overlap     ... plus the offset grid
               counted by `region_grids`, the same geometry the container builds
               with, so no pixel is read.
    tissue     the mask grid split again by what is IN each tile: a tile has
               tissue when its background fraction (`white_fractions`, the
               score the sampler's richness buckets use) is below --bg-max
               (0.95: more than 5% of the tile is tissue; the production
               sampler stops earlier, at 0.85), else it
               is empty. Reported for main and for main + offset, so
               tissue + empty is exactly the mask tile count. A region is a
               bounding box, and a bounding box holds glass.
    GB         tiles x (slots x dim x bytes + 18) per thing kept: `raw` (the
               model's own output, all slots), `cls`, and the five reduced
               poolings. The 18 is the x, y, region and grid_rc columns.

`--feature-map only` skips the sampler cells (minutes) and answers in seconds.
The bytes per tile need the encoder's `model_spec`, so the encoder is built on
the CPU; `--encoder` / `--head` / `--fp32` say which and at what width.

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
from TissueMaskConfig import MASK_RECIPES, MaskMaker             # noqa: E402
from PatchingLib import region_grids                             # noqa: E402
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


#: `feature_map_size.csv`. Same rule: rewritten every run.
GRID_DEFINITIONS = [
    ('level', 'the pyramid level, from the slide itself'),
    ('ds', "that level's own downsample"),
    ('tile_size', 'output px of one tile'),
    ('regions_<mask>', 'regions that can hold one tile at this level'),
    ('tiles_nomask_main', 'main grid over the scanned rectangle'),
    ('tiles_nomask_overlap', 'main + the grid offset by half a tile: what '
     'FeatureMapCache stores with overlap=True'),
    ('tiles_mask_main', 'main grid over the mask\'s regions'),
    ('tiles_mask_overlap', 'main + offset grid over the mask\'s regions'),
    ('tiles_mask_{main,overlap}_{tissue,empty}',
     'the two above split by the tile\'s own background fraction: tissue when '
     'it is below bg_max, empty otherwise. tissue + empty == the mask tile count'),
    ('bg_max', 'the threshold used for that split'),
    ('gb_mask_{main,overlap}_tissue_<what>',
     'GB to store <what> for ONLY the tiles with tissue in that grid. The '
     'feature-map cache stores the whole grid today, so this is what a '
     'tissue-only cache would need, not what is written'),
    ('gb_<mask>_<grid>_<what>', 'GB to store <what> for that grid: raw is the '
     "model's own output (all slots), 5_arms is cls+cls_avg+cls_std+rings3+"
     'grid2x2 as five files. Each tile also carries 18 bytes of x, y, region '
     'and grid_rc'),
]

#: bytes of x (int32), y (int32), region (int16) and grid_rc (2 x int32) per tile
INDEX_BYTES = 18

#: the reduced poolings the window bench compares, and `tokens`' cousin `raw`
POOLING_NAMES = ('cls', 'cls_avg', 'cls_std', 'rings3', 'grid2x2')


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
                n: int = 500, candidates: str = 'lattice', step: float = 1.0,
                max_overlap: float = 0.0, overlapping_share: float = 0.0,
                max_tries: int = 2500, seed: int = 0, out_dir=None,
                feature_map: str = 'off', encoder: str = 'uni2',
                head: str = '', value_bytes: int = 2, bg_max: float = 0.95):
        if feature_map not in ('off', 'on', 'only'):
            raise ValueError(f"feature_map must be off, on or only, got {feature_map!r}")
        self.mask_cfg = MASK_RECIPES[seg]
        self.mask_root = cache_root(mask_cache_job, 'mask')
        self.seg_dir = self.mask_root / self.mask_cfg.seg_id()
        self.tile_sizes = list(tile_sizes)
        self.ds_values = list(ds_values)
        self.n = n
        self.candidates = candidates
        self.step = step
        self.max_overlap = max_overlap
        self.overlapping_share = overlapping_share
        self.max_tries = max_tries
        self.seed = seed
        self.out_dir = out_dir or job_result_dir('ProbeTileYield')
        os.makedirs(self.out_dir, exist_ok=True)
        self.feature_map = feature_map
        self.encoder = encoder
        self.head = head
        self.value_bytes = int(value_bytes)
        self.bg_max = float(bg_max)
        self._per_tile = None          # {what: bytes per tile}, built on first use
        self.grid_rows = []

    def per_tile_bytes(self) -> dict:
        """`{what: bytes per tile}` for `raw`, each reduced pooling and their
        sum. Read off the encoder's `model_spec`, so the encoder is built -- on
        the CPU, once."""
        if self._per_tile is None:
            import torch                                            # noqa: PLC0415
            from TileEncoderFunc import (admissible_poolings,    # noqa: PLC0415
                                         encoder_config, pool_slots)
            over = {'head': self.head} if self.head else {}
            cfg = encoder_config(self.encoder, **over)
            spec = cfg.build(torch.device('cpu')).model_spec
            kept, dropped = admissible_poolings(cfg, POOLING_NAMES)
            width = spec.dim * self.value_bytes
            per = {'raw': spec.n_tokens() * width + INDEX_BYTES}
            for name in kept:
                per[name] = len(pool_slots(name, spec)[0]) * width + INDEX_BYTES
            per['5_arms'] = sum(per[n] for n in kept)
            self._per_tile = per
            print(f'  feature map: {self.encoder}  dim {spec.dim}  '
                  f'{spec.n_tokens()} slots in raw  {self.value_bytes} B/value  '
                  + '  '.join(f'{k} {v / 1e3:.1f} KB/tile' for k, v in per.items())
                  + (f'  (cannot do {dropped})' if dropped else ''), flush=True)
        return self._per_tile

    def _grid_cell(self, wsi, masks: dict, tile_size: int, level: int) -> dict:
        """One (tile_size, level): how big the whole-slide feature map is,
        four ways. `masks` is `{'nomask': TissueMask, 'mask': TissueMask|None}`;
        a missing one leaves its columns empty rather than zero."""
        ds = float(wsi.level_downsamples[level])
        row = {'wsi_stem': wsi_stem_of(wsi), 'tile_size': tile_size,
               'level': level, 'ds': ds, 'footprint_l0': int(tile_size * ds)}
        per = self.per_tile_bytes()
        for label, trm in masks.items():
            regions = (trm.patchable(tile_size * ds).tissue_regions
                       if trm is not None else None)
            row[f'regions_{label}'] = len(regions) if regions is not None else ''
            for overlap in (False, True):
                grid = 'overlap' if overlap else 'main'
                n = ('' if regions is None else int(sum(len(g) for g in region_grids(
                    regions, ds=ds, level=level, tile_size=tile_size,
                    overlap=overlap))))
                row[f'tiles_{label}_{grid}'] = n
                for what, nbytes in per.items():
                    row[f'gb_{label}_{grid}_{what}'] = (
                        '' if n == '' else round(n * nbytes / 1e9, 4))
        row['bg_max'] = self.bg_max
        row.update(self._tissue_split(masks.get('mask'), tile_size, ds, level))
        # What keeping ONLY the tiles that have tissue would cost. Not something
        # FeatureMapCache can do today -- it insists on the whole grid -- but it
        # is the number that says whether relaxing that is worth it.
        for grid in ('main', 'overlap'):
            n = row[f'tiles_mask_{grid}_tissue']
            for what, nbytes in per.items():
                row[f'gb_mask_{grid}_tissue_{what}'] = (
                    '' if n == '' else round(n * nbytes / 1e9, 4))
        return row

    def _tissue_split(self, trm, tile_size: int, ds: float, level: int) -> dict:
        """The mask grid, split by what is inside each tile.

        Every tile of the grid the feature map would cover -- main, and the
        offset grid -- gets the background fraction of its own footprint from the
        mask (`white_fractions`, level-0 top-left and the tile in level pixels,
        which is how the sampler scores a candidate). Below `bg_max` it has
        tissue. The offset tiles are `overlap_patch_infos`, so main + offset is
        exactly the `overlap=True` grid and tissue + empty adds up to it."""
        keys = ('main_tissue', 'main_empty', 'overlap_tissue', 'overlap_empty')
        if trm is None:
            return {f'tiles_mask_{k}': '' for k in keys}
        regions = trm.patchable(tile_size * ds).tissue_regions
        grids = region_grids(regions, ds=ds, level=level, tile_size=tile_size,
                             overlap=True)
        counts = dict.fromkeys(keys, 0)
        for kind, infos in (('main', lambda g: g.main_patch_infos),
                            ('offset', lambda g: g.overlap_patch_infos)):
            for grid in grids:
                pts = infos(grid)
                if not pts:
                    continue
                xy = np.array([(p.x * ds, p.y * ds) for p in pts], dtype=np.int64)
                background = np.asarray(trm.white_fractions(xy, level, tile_size))
                has = int((background < self.bg_max).sum())
                empty = len(pts) - has
                if kind == 'main':
                    counts['main_tissue'] += has
                    counts['main_empty'] += empty
                counts['overlap_tissue'] += has
                counts['overlap_empty'] += empty
        return {f'tiles_mask_{k}': v for k, v in counts.items()}

    def _grid_rows_for(self, wsi, trm, entry) -> None:
        """Every native level of this slide, every --tile-size."""
        none_mask = MaskMaker(MASK_RECIPES['none']).mask(wsi)[0]
        masks = {'nomask': none_mask, 'mask': trm}
        top = max(self.ds_values)
        levels = [lv for lv, d in enumerate(wsi.level_downsamples)
                  if float(d) <= top * (1 + 1e-3)]
        # One aligned table per slide. The tile counts are `len` of the PatchGrid
        # `region_grids` builds (PatchingLib.PatchGrid.for_region, geometry only);
        # the GB columns are the OVERLAP grid, which is what FeatureMapCache
        # stores with overlap=True. The tissue columns split the mask grid.
        self.per_tile_bytes()          # prints its own line: before the table, not in it
        cw, tw, gw, sep, dw = 10, 10, 11, ' | ', 8
        g1, g2, g3 = 2 * cw + 1, 4 * tw + 3, 3 * gw + 2
        print(f'    + = with tissue, - = without: a tile has tissue when its '
              f'background fraction is below {self.bg_max:g}.  '
              f'all = main + the grid offset by half a tile (what FeatureMapCache '
              f'stores), so all is about twice main')
        print(f'    {"tile":>4} {"L":>2} {"ds":>{dw}}{sep}{"no-mask tiles".center(g1)}'
              f'{sep}{"mask tiles".center(g1)}{sep}'
              f'{"mask tiles, by what is in them".center(g2)}{sep}'
              f'{"raw GB (all = main + offset)".center(g3)}')
        print(f'    {"":>4} {"":>2} {"":>{dw}}{sep}{"main":>{cw}} {"all":>{cw}}'
              f'{sep}{"main":>{cw}} {"all":>{cw}}{sep}'
              f'{"main +":>{tw}} {"main -":>{tw}} {"all +":>{tw}} '
              f'{"all -":>{tw}}{sep}{"no-mask":>{gw}} {"mask":>{gw}} '
              f'{"tissue only":>{gw}}')
        print('    ' + '-' * 4 + ' ' + '-' * 2 + ' ' + '-' * dw + '-+-' + '-' * g1
              + '-+-' + '-' * g1 + '-+-' + '-' * g2 + '-+-' + '-' * g3)

        def count(v, w=cw):
            return f'{v:>{w},}' if v != '' else f'{"-":>{w}}'

        def gigs(v):
            return f'{v:>{gw}.2f}' if v != '' else f'{"-":>{gw}}'

        for tile_size in self.tile_sizes:
            for level in levels:
                row = self._grid_cell(wsi, masks, tile_size, level)
                row['dataset'] = entry.get('dataset')
                self.grid_rows.append(row)
                print(f"    {tile_size:>4d} {level:>2d} {row['ds']:>{dw}.6g}{sep}"
                      f"{count(row['tiles_nomask_main'])} "
                      f"{count(row['tiles_nomask_overlap'])}{sep}"
                      f"{count(row['tiles_mask_main'])} "
                      f"{count(row['tiles_mask_overlap'])}{sep}"
                      f"{count(row['tiles_mask_main_tissue'], tw)} "
                      f"{count(row['tiles_mask_main_empty'], tw)} "
                      f"{count(row['tiles_mask_overlap_tissue'], tw)} "
                      f"{count(row['tiles_mask_overlap_empty'], tw)}{sep}"
                      f"{gigs(row['gb_nomask_overlap_raw'])} "
                      f"{gigs(row['gb_mask_overlap_raw'])} "
                      f"{gigs(row['gb_mask_overlap_tissue_raw'])}", flush=True)

    def _write_grid(self) -> None:
        if not self.grid_rows:
            return
        path = os.path.join(self.out_dir, 'feature_map_size.csv')
        keys = list(dict.fromkeys(k for r in self.grid_rows for k in r))
        with open(path, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=keys, restval='')
            writer.writeheader()
            writer.writerows(self.grid_rows)
        with open(os.path.join(self.out_dir, 'feature_map_size_definitions.csv'),
                  'w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(('column', 'meaning'))
            writer.writerows(GRID_DEFINITIONS)
        print(f'\nSaved {path}   ({len(self.grid_rows)} rows)')
        # Totals: what keeping ONE thing for EVERY probed slide costs, per tile
        # size and grid. `raw` and `5_arms` are the two ways to keep everything.
        per = self.per_tile_bytes()
        for tile_size in self.tile_sizes:
            rows = [r for r in self.grid_rows if r['tile_size'] == tile_size]
            if not rows:
                continue
            tiles = ('tiles_nomask_main', 'tiles_nomask_overlap', 'tiles_mask_main',
                     'tiles_mask_overlap', 'tiles_mask_main_tissue',
                     'tiles_mask_main_empty', 'tiles_mask_overlap_tissue',
                     'tiles_mask_overlap_empty')
            print(f'\ntile {tile_size}: total tiles over '
                  f'{len({r["wsi_stem"] for r in rows})} slides, every level')
            for name in tiles:
                total = sum(r[name] for r in rows if r.get(name, '') != '')
                print(f'  {name:<28}{total:>14,}')
            print(f'\ntile {tile_size}: total over {len({r["wsi_stem"] for r in rows})} '
                  f'slides, every level  (GB)')
            print(f'  {"":<26}' + ''.join(f'{w:>12}' for w in per))
            for name, prefix in (('nomask main', 'gb_nomask_main'),
                                 ('nomask overlap', 'gb_nomask_overlap'),
                                 ('mask main', 'gb_mask_main'),
                                 ('mask overlap', 'gb_mask_overlap'),
                                 ('mask main, tissue only', 'gb_mask_main_tissue'),
                                 ('mask overlap, tissue only', 'gb_mask_overlap_tissue')):
                sums = []
                for what in per:
                    col = f'{prefix}_{what}'
                    sums.append(sum(r[col] for r in rows if r.get(col, '') != ''))
                print(f'  {name:<26}' + ''.join(f'{v:>12,.1f}' for v in sums))

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
        # `self.candidates` is the arm: 'random' and 'lattice' are DIFFERENT
        # measurements.
        cfg = SamplerConfig(
            n_per_rung=self.n, seed=self.seed,
            candidates=self.candidates,
            max_tries_per_tile=max(1, self.max_tries // max(self.n, 1)),
            overlap=OverlapConfig(step=self.step,
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
        have_mask = (folder / 'mask_meta.json').exists()
        if not have_mask:
            print(f'    no mask under {folder}', flush=True)
            if self.feature_map == 'off':
                return []
            print('    (the feature-map report still gives the no-mask columns)',
                  flush=True)
        slide_mask = None
        if have_mask:
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
            trm = (self.mask_cfg.regions(wsi, slide_mask)
                   if slide_mask is not None else None)
            if trm is not None:
                print(f'    {len(trm.tissue_regions)} tissue regions', flush=True)
            if self.feature_map != 'off':
                self._grid_rows_for(wsi, trm, entry)
            if self.feature_map == 'only' or trm is None:
                return rows
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

        self._write_grid()
        if not rows:
            if not self.grid_rows:
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
    ap.add_argument('--step', type=float, default=1.0,
                    help='lattice step as a fraction of the tile; 1.0 is '
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
    ap.add_argument('--feature-map', choices=('off', 'on', 'only'), default='off',
                    help='also report how big the WHOLE-SLIDE feature map is at '
                         'every level -- tiles and GB, with and without the mask, '
                         'with and without the overlap grid. `only` skips the '
                         'sampler cells and takes seconds')
    ap.add_argument('--encoder', default='uni2',
                    help='--feature-map: whose tokens; read off its model_spec '
                         '(built on the CPU)')
    ap.add_argument('--head', default='', help='--feature-map: the encoder head')
    ap.add_argument('--fp32', action='store_true',
                    help='--feature-map: 4 bytes per value instead of 2')
    ap.add_argument('--bg-max', type=float, default=0.95,
                    help='--feature-map: a tile has tissue when its background '
                         'fraction is below this. 0.95 = more than 5%% of the tile '
                         'is tissue. (The production sampler admits up to 0.85; '
                         'pass 0.85 to count what it could draw.)')
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
        step=args.step, max_overlap=args.max_overlap,
        overlapping_share=args.overlapping_share, max_tries=args.max_tries,
        seed=args.seed, out_dir=out_dir, feature_map=args.feature_map,
        encoder=args.encoder, head=args.head,
        value_bytes=4 if args.fp32 else 2, bg_max=args.bg_max)

    if entries is None:
        entries = prober.default_entries()
        if not entries:
            print(f'no masks under {prober.seg_dir}. Run '
                  f'utilities/cli/build_cache/build_mask_store.py first.')
            return 1
        print(f'{len(entries)} slides from the mask store', flush=True)

    rows = prober.run(entries)
    return 0 if (rows or prober.grid_rows) else 1


if __name__ == '__main__':
    sys.exit(main())
