#!/usr/bin/env python3
"""End-to-end LocaScope bench: synthetic FoVs with known positions through the
three stages, every stage's output and by-products kept as cache entries.

    python utilities/bench_modules/bench_locascope.py \\
        --datasets bracs/test ki67_with_photo --split test --n-wsi 5 \\
        --stage1 knn:gigapath --stage2 slidewin:gigapath --stage3 sift:default \\
        --route both

SHOTS: the first --n-wsi slides of each dataset's recorded --split; per slide
one draw over the --fov recipe's levels on the hest masks of
--fov-mask-cache-job, photographed through `FovSupply.cached` (draw and
render in --draw-cache-job / --render-cache-job, default this job).

STAGES: each one is `<method>:<recipe>` from its package's METHODS table, and
every field of the recipe can be replaced by a `--stageN-<field>` flag (the
encoder, the masks and the bank draw are not flags; they are the recipe's or
--seg's). A stage's results for a slide are ONE cache entry, written once the
slide is done (`Cache.Entry.writing`), under this job's tree:

    .../render=<gap>/stage1/{output,neighbours,probs,votes,prototypes_index}_<s1>.csv
    .../render=<gap>/stage1=<s1|oracle>/stage2/
            {output,tile_sims,truth,truth_sim}_<s2>.csv
    .../render=<gap>/stage1=<s1|oracle>/stage2=<s2>/stage3/
            {output,matches}_<s3>.csv

`output` is each stage's output interface, row for row: stage 1's
`EstMppResult`, stage 2's `CandidateSet` (one row per window, with its level
and ds), stage 3's `SiftRansacResult` per verified rank. The rest is what the
stage has beside it: a KNN's neighbours, a voting method's probs and votes,
a prototype method's support tiles (`prototypes_index`, one row per tile as the
bank draw's index holds it -- not per FoV),
the per-tile cosines of every output window (`tile_sims`), the truth window
and its per-tile cosines (`truth`, `truth_sim`), stage 3's point pairs.

`<sN>` is `<method>-<recipe>-<16 hex>`: the hex is the config's identity (for
stage 2 together with the mask it searches), so an overridden field is another
hex under the same label. Every table has an `index` column, the FoV's row in
the draw and in the render CSV, which holds its ground truth. A stage whose
entry is a hit is read back and its model never loaded; a slide whose entries
all hit is not rendered.

--route stage1 routes stage 2 to stage 1's level, oracle to the level the FoV
was placed at (`stage1=oracle/`), both runs stages 2 and 3 once per route.

Nothing here scores or draws. `utilities/cli/plot/plot_locascope.py` joins the
render CSV and these tables and computes every error from them.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # utilities/
import _paths                                                       # noqa: E402
_paths.setup_import_paths()

import Cache                                                           # noqa: E402
from ConfigArgs        import add_config_args, config_from_args          # noqa: E402
from ConfigIdentity    import record, short_id                          # noqa: E402
from CpuBudget         import CpuBudget                                 # noqa: E402
from LocaScopePipeline import LocaScopePipeline, UnusableLevel          # noqa: E402
from PatchingLib       import QueryPatchContainer                       # noqa: E402
from TileSampler       import PlanSpec                                  # noqa: E402
from TissueMaskConfig  import (MASK_RECIPES, MaskMaker, add_mask_args,  # noqa: E402
                               mask_cfg_from_args)
from AccessDatasets    import list_names, locate                        # noqa: E402
from SlideReader       import SlideReader                               # noqa: E402
from camera            import Render                                    # noqa: E402
from FovSupply         import FovSupply, add_fov_args, fov_from_args    # noqa: E402
from stage2_retrieval.StageInterface import Candidate, CandidateSet     # noqa: E402
import stage1_estimation                                                 # noqa: E402
import stage2_retrieval                                                  # noqa: E402
import stage3_localization                                               # noqa: E402


#: The route whose level is the one the FoV was placed at.
ORACLE = 'oracle'

#: Fields a `--stageN-*` flag does not reach: configs of their own, chosen by
#: the recipe (encoder, bank draw) or by --seg (the mask).
_NOT_FLAGS = {1: ('encoder', 'mask_cfg', 'sampler_cfg', 'levels'), 2: ('encoder',),
              3: ()}


# ── the stages: a recipe, its id, its record ─────────────────────────────────

class Stage:
    """One stage's chosen method: its config, the label its entries are filed
    under, and the class it builds -- built on first use, so a stage whose
    entries all hit never loads a model."""

    def __init__(self, n: int, method: str, name: str, cfg, cls, make, *,
                 also_id: tuple = ()):
        self.n, self.method, self.name, self.cfg, self.cls = n, method, name, cfg, cls
        hexid = short_id([cfg.identity_id(), *also_id]) if also_id else cfg.identity_id()
        self.id = f'{method}-{name}-{hexid}'
        self._make, self._obj = make, None

    def obj(self):
        if self._obj is None:
            print(f'  [stage {self.n}] loading {self.id}', flush=True)
            self._obj = self._make()
        return self._obj

    def record(self, **upstream) -> dict:
        """The config's record, the class's code and, for a method with a
        checkpoint, the file's fingerprint: `weights` is no part of the
        config's id, so a retrained file under the same name is seen here."""
        weights = getattr(self.cfg, 'weights', None)
        if weights:
            from ConfigIdentity import file_fingerprint           # noqa: PLC0415
            from stage1_estimation.StageInterface import weights_path  # noqa: PLC0415
            upstream['weights'] = file_fingerprint(weights_path(weights))
        return dict(record(self.cfg, also=(self.cls,), **upstream), label=self.id)


def parse_stage(args, n: int):
    """`(method, recipe, config, class)` of `--stage<n>`, its flags applied."""
    pkg = (stage1_estimation, stage2_retrieval, stage3_localization)[n - 1]
    method, name, cfg, cls = pkg.recipe(getattr(args, f'stage{n}'))
    return method, name, config_from_args(args, cfg, f'stage{n}',
                                          skip=_NOT_FLAGS[n]), cls


# ── table files ──────────────────────────────────────────────────────────────

def write_rows(path, rows: List[dict]) -> None:
    """A CSV of `rows`, columns in first-seen order; None is an empty cell."""
    cols: Dict[str, None] = {}
    for r in rows:
        cols.update(dict.fromkeys(r))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(cols) or ['index'])
        w.writeheader()
        for r in rows:
            w.writerow({k: ('' if v is None else v) for k, v in r.items()})


def read_rows(path) -> List[Dict[str, str]]:
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def by_index(rows) -> Dict[int, List[Dict[str, str]]]:
    out: Dict[int, List[Dict[str, str]]] = {}
    for r in rows:
        out.setdefault(int(r['index']), []).append(r)
    return out


class Tables:
    """One stage entry's tables while a slide runs: rows per role, written
    together when the slide is done."""

    def __init__(self, entry, id: str, rec: dict, roles):
        self.entry, self.id, self.rec = entry, id, rec
        self.rows = {r: [] for r in roles}

    def add(self, role: str, index: int, rows) -> None:
        self.rows[role].extend({'index': index, **r} for r in rows)

    def add_slide(self, role: str, rows) -> None:
        """Rows about the slide rather than one FoV -- a prototype's support,
        built once per slide; their own columns, no FoV `index`."""
        self.rows[role].extend(rows)

    def write(self) -> None:
        with self.entry.writing(self.id, self.rec) as put:
            for role, rows in self.rows.items():
                write_rows(put(role, '.csv'), rows)


def status(entry, id: str, rec: dict) -> str:
    state, diff = entry.status(id, rec)
    if state == 'stale':
        print(f'  [{entry.kind}] {entry.record_path(id)} is stale: '
              + '; '.join(diff), flush=True)
    return state


# ── one FoV through one stage ────────────────────────────────────────────────

#: The vote rules every method that votes over per-patch probabilities
#: (classifier, prototype) is scored under, all from one forward pass:
#: FoV_Vote.md's six with the variants it asks for -- both tie rules of the
#: patch class median, and the quality-weighted mean under each signal in
#: `FoVVote.QUALITY_SIGNALS`. label -> (FoVVote rule, options). Part of the
#: stage-1 record: another set of rules is another `votes` table.
RULES = {
    'mean_probability':         ('mean_probability', {}),
    'hard_majority':            ('hard_majority', {}),
    'patch_class_median:lower': ('patch_class_median', {'tie_rule': 'lower'}),
    'patch_class_median:upper': ('patch_class_median', {'tie_rule': 'upper'}),
    'median_log_rung':          ('median_log_rung', {}),
    'sum_log_probability':      ('sum_log_probability', {}),
    'quality_weighted:tissue':  ('quality_weighted', {'signal': 'tissue'}),
    'quality_weighted:agree':   ('quality_weighted', {'signal': 'agree'}),
}


def _class_of(result, classes) -> int:
    """The class index a result chose -- `estimated_ds` is classes[i] exactly."""
    return min(range(len(classes)),
               key=lambda i: abs(math.log(classes[i] / result.estimated_ds)))


def support_rows(est) -> List[dict]:
    """A prototype method's support tiles for this slide (`prototype_rows`),
    none for a method that has no prototypes."""
    rows = getattr(est, 'prototype_rows', None)
    return rows() if rows is not None else []


def run_stage1(est, img, tables: Tables, index: int) -> Optional[int]:
    """Stage 1 on one FoV, its rows added: `output` (the method's own answer),
    and what the method has beside it -- a KNN's `neighbours`, a voting
    method's per-patch `probs` and every rule of RULES over them (`votes`)."""
    from stage1_estimation.FoVVote import QUALITY_SIGNALS, diagnose  # noqa: PLC0415
    t0 = time.perf_counter()
    try:
        if hasattr(est, 'patch_probs'):
            patches = est.query_patches(img)
            probs = est.patch_probs(img)
            r1 = est.from_probs(probs, est.cfg.vote)
        else:
            r1 = est.estimate(img)
    except Exception as e:                                          # noqa: BLE001
        tables.add('output', index, [dict(error=f'{type(e).__name__}: {e}')])
        return None
    tables.add('output', index, [dict(r1.row(), error='',
                                      t_s=round(time.perf_counter() - t0, 3))])
    rows = getattr(est, 'neighbour_rows', None)
    if rows is not None:
        tables.add('neighbours', index, rows())
    if hasattr(est, 'patch_probs'):
        classes = [float(d) for d in est.classes_ds]
        p = probs.detach().float().cpu()
        tables.add('probs', index, [
            dict(patch=i, class_ds=classes[c], p=float(p[i, c]))
            for i in range(p.shape[0]) for c in range(p.shape[1])])
        weights = {name: QUALITY_SIGNALS[name](probs, patches)
                   for name in {o['signal'] for _, o in RULES.values()
                                if 'signal' in o}}
        votes = []
        for label, (rule, opts) in RULES.items():
            w = weights.get(opts.get('signal'))
            result = est.from_probs(probs, rule, weights=w,
                                    tie_rule=opts.get('tie_rule', 'lower'))
            risk = diagnose(rule, probs, _class_of(result, classes),
                            rungs=classes, weights=w)
            if w is not None:
                risk['risk_w_mean'] = float(w.float().mean())
            votes.append(dict(rule=label, **result.row(), **risk))
        tables.add('votes', index, votes)
    return int(r1.chosen_level)


def matching_rotation(rot_deg) -> int:
    """The stage-2 rotation that turns a photo taken at `rot_deg` back upright:
    `(360 - rot_deg) mod 360`. Stage 2 turns the query with `np.rot90`
    (counter-clockwise) and the camera turned the photo the other way, so 90
    and 270 swap and 0 and 180 stay -- measured, a 270 photo found at rank 1
    under rotation 90."""
    return (-int(round(float(rot_deg)))) % 360


def query_grid_centre(camera, x0: int, y0: int, params: dict, shape,
                      tile: int):
    """The level-0 point the query's tile grid stands for at the photo's own
    rotation: stage 2 turns the photo (`np.rot90`) and cuts whole `tile` px tiles from its top-left, so the grid's centre is
    not the photo's (a 1440 px side holds 5 tiles and 160 px outside them,
    on a side that moves with the turn). The turn is the one that sets the
    photo upright (`matching_rotation`). The grid centre, back in the photo's
    own pixels, through `Render.output_to_level0` with the photo's full angle
    and scale. `(x0, y0)` is the FoV rect's level-0 top-left."""
    H, W = shape[:2]
    k = matching_rotation(params['rot_deg']) // 90
    Hr, Wr = (H, W) if k % 2 == 0 else (W, H)
    xr, yr = (Wr // tile) * tile / 2.0, (Hr // tile) * tile / 2.0
    # np.rot90(a, k) as a map from the turned image back to the photo
    u, v = {0: (xr, yr), 1: (W - yr, xr), 2: (W - xr, H - yr), 3: (yr, H - xr)}[k]
    angle = float(params['rot_deg']) + float(params.get('angle_jitter', 0.0))
    return camera.output_to_level0(x0, y0, u, v, rot_deg=angle,
                                   scale=float(params['scale']))


def run_stage2(pl, img, level: int, truth_at, rot_deg: int,
               tables: Tables, index: int, true_level: int, true_ds: float):
    """`(qc, cs)` of the FoV at `level`, its rows added; None when it failed.
    `truth_at(tile)` is the level-0 point the query's tile grid stands for
    (`query_grid_centre`); `rot_deg` the stage-2 rotation that sets the
    photo upright (`matching_rotation`). The truth window is the one at the
    photo's own level `true_level`: scored only when stage 2 searched that
    level, its box alone (`_truth_box`) when it was sent elsewhere."""
    t0 = time.perf_counter()
    try:
        qc, cs = pl.stage2(img, level)
    except Exception as e:                                          # noqa: BLE001
        tables.add('output', index, [dict(level=level,
                                          error=f'{type(e).__name__}: {e}')])
        return None
    t = round(time.perf_counter() - t0, 3)
    ret = pl.retriever
    # the output interface itself: the CandidateSet, one row per window, in
    # the frame it is expressed in
    tables.add('output', index, [dict(r, level=cs.level, ds=cs.ds, error='', t_s=t)
                                 for r in cs.rows(qc)])
    sims = []
    for rank, c in enumerate(cs, 1):
        sims += _tile_rows(ret, c, rank=rank)
    tables.add('tile_sims', index, sims)
    try:
        if level == true_level:
            truth, tiles = _truth_rows(ret, cs, qc, truth_at(pl.tile_size), rot_deg)
            tables.add('truth_sim', index, tiles)
        else:
            truth = _truth_box(img.shape, rot_deg, pl.tile_size, true_ds,
                               truth_at(pl.tile_size))
        tables.add('truth', index, [dict(truth, truth_level=true_level)])
    except Exception as e:                                          # noqa: BLE001
        tables.add('truth', index, [dict(error=f'{type(e).__name__}: {e}')])
    return qc, cs


def _truth_box(shape, rot_deg: int, tile: int, ds: float, centre) -> dict:
    """The truth row of a FoV stage 2 searched at a level not its own: the
    level-0 box the query's tile grid, turned by `rot_deg`, covers at its own
    level (`ds`), centred on `centre`. No window there was scored, so no
    rank, score or cosines."""
    H, W = shape[:2]
    Hr, Wr = (H, W) if (rot_deg // 90) % 2 == 0 else (W, H)
    w0, h0 = (Wr // tile) * tile * ds, (Hr // tile) * tile * ds
    cx, cy = centre
    return dict(truth_rotation=rot_deg, truth_x0=int(round(cx - w0 / 2)),
                truth_y0=int(round(cy - h0 / 2)), truth_w0=w0, truth_h0=h0)


def _tile_rows(ret, c, **key) -> List[dict]:
    """Window `c`'s per-tile cosines as rows: the query tile (`q_row`,
    `q_col`, on the query turned to `c.rotation`), the reference tile under it
    on the window's lattice, and their cosine."""
    grid = ret.window_tile_sims(c).float().cpu().numpy()
    return [dict(key, q_row=qr, q_col=qcol, ref_row=c.row + qr,
                 ref_col=c.col + qcol, cosine=float(grid[qr, qcol]))
            for qr in range(grid.shape[0]) for qcol in range(grid.shape[1])]


def _truth_rows(ret, cs, qc, centre, rot_deg: int):
    """`(truth row, truth tile rows)`: the window at `rot_deg`, the rotation
    that sets the photo upright, whose centre is nearest `centre` -- the
    level-0 point the query's tile grid stands for, not the photo's centre
    (`query_grid_centre`) -- over both lattices: its score, its level-0 box,
    its distance to `centre` and its per-tile cosines; and the distance to
    the nearest main-lattice window (`d_main`). No rank: where the truth sits
    in stage 2's answer is read off the output (plot_locascope's
    `s2_hit_rank`), the one ranking there is. Needs the similarity maps, so
    it is computed here."""
    cx, cy = centre
    tw, _, dist = ret.nearest_window(cx, cy, rot_deg)
    _, _, mdist = ret.nearest_window(cx, cy, rot_deg, lattices=('main',))
    box = CandidateSet(candidates=(tw,), level=cs.level, ds=cs.ds,
                       grids=cs.grids).rows(qc)[0]
    row = dict(
        truth_region=tw.region_index, truth_lattice=tw.lattice,
        truth_row=tw.row, truth_col=tw.col, truth_rotation=tw.rotation,
        truth_score=tw.score, truth_dist_l0=dist, truth_main_dist_l0=mdist,
        truth_x0=box['x0'], truth_y0=box['y0'], truth_w0=box['w0'],
        truth_h0=box['h0'])
    return row, _tile_rows(ret, tw)


def run_stage3(pl, qc, cs, tables: Tables, index: int) -> None:
    t0 = time.perf_counter()
    try:
        ranks = pl.stage3(qc, cs)
    except Exception as e:                                          # noqa: BLE001
        tables.add('output', index, [dict(error=f'{type(e).__name__}: {e}')])
        return
    t = round(time.perf_counter() - t0, 3)
    # the output interface itself: one SiftRansacResult per verified rank
    tables.add('output', index, [dict(r.row(), error='', t_s=t) for r in ranks]
               or [dict(error='no candidate verified', t_s=t)])
    tables.add('matches', index, [p for r in ranks for p in r.pair_rows()])


def stored_candidates(rows) -> List[Candidate]:
    return [Candidate(int(r['region']), r['lattice'], int(r['row']), int(r['col']),
                      int(r['rotation']), float(r['score']))
            for r in sorted(rows, key=lambda r: int(r['rank']))]


# ── one slide ────────────────────────────────────────────────────────────────

#: The tables of each stage's entry.
ROLES = {1: ('output', 'neighbours', 'probs', 'votes', 'prototypes_index'),
         2: ('output', 'tile_sims', 'truth', 'truth_sim'),
         3: ('output', 'matches')}


#: The bench's own code in the stage-2 entry: how the truth window is placed
#: (`query_grid_centre`, `matching_rotation`, `_truth_rows`). No stage class
#: owns it, so a change to it is bumped here.
TRUTH_VERSION = 3


def stage1_record(s1, base, limit: int = 0) -> dict:
    """The record of stage 1's entry under the render address `base`: the
    method's config and code, the photos it saw, the vote rules."""
    return dict(s1.record(render=dict(base.levels)['render']),
                rules=list(RULES), tables=list(ROLES[1]),
                limit=int(limit))


def stage_entries(base, stages, route: str, mask_cfg, limit: int = 0) -> dict:
    """`{n: (entry, id, record)}` of each stage `route` goes through, under
    `base`, a `render=` address in the stage job's tree. Stage 1 is there only
    on the stage-1 route; the oracle has none."""
    s1, s2, s3 = stages
    lim = {'limit': int(limit)}            # 0 = every FoV; part of every record
    up = s1.id if route == 'stage1' else ORACLE
    a1 = base.at(stage1=up)
    out = {
        2: (a1.entry('stage2'), s2.id,
            dict(s2.record(stage1=up, seg=mask_cfg.seg_id(),
                           region=mask_cfg.region_id()),
                 tables=list(ROLES[2]), truth=TRUTH_VERSION, **lim)),
        3: (a1.at(stage2=s2.id).entry('stage3'), s3.id,
            dict(s3.record(stage2=s2.id), tables=list(ROLES[3]), **lim))}
    if route == 'stage1':
        out[1] = (base.entry('stage1'), s1.id, stage1_record(s1, base, limit))
    return out


def bench_slide(path: str, supply: FovSupply, stages, routes, masks, args,
                own: str, feature_job) -> None:
    s1, s2, s3 = stages
    base = supply.render_address.on(own)

    # every entry this slide needs, and whether it is there
    t1, t2, t3, stored2 = None, {}, {}, {}
    for route in routes:
        ents = stage_entries(base, stages, route, masks.cfg, args.limit)
        if 1 in ents:
            e, sid, rec = ents[1]
            if status(e, sid, rec) != 'hit':
                t1 = Tables(e, sid, rec, ROLES[1])
            else:
                stored1 = by_index(read_rows(e.path('output', sid, '.csv')))
        e, sid, rec = ents[2]
        if status(e, sid, rec) != 'hit':
            t2[route] = Tables(e, sid, rec, ROLES[2])
        e3, sid3, rec3 = ents[3]
        if status(e3, sid3, rec3) != 'hit':
            t3[route] = Tables(e3, sid3, rec3, ROLES[3])
            if route not in t2:
                stored2[route] = by_index(read_rows(e.path('output', sid, '.csv')))
    if t1 is None and not t2 and not t3:
        print('  every stage hit -- nothing to run', flush=True)
        return
    print(f'  to run: ' + ' '.join(
        ([f'stage1'] if t1 else []) + [f'stage2[{r}]' for r in t2]
        + [f'stage3[{r}]' for r in t3]), flush=True)

    pl = LocaScopePipeline(
        path, s1.obj() if t1 else None,
        s2.obj() if (t2 or t3) else None,
        s3.obj() if t3 else None, masks,
        feature_cache_job=feature_job,
        feature_store_mode=args.feature_store_mode,
        bank_cache_job=own).build()
    if t1 is not None:
        t1.add_slide('prototypes_index', support_rows(pl.estimator))

    clock = dict.fromkeys(['photo', 'stage1', 'stage2', 'stage3'], 0.0)
    n, t_p = 0, time.perf_counter()
    shots = supply.shots(workers=args.workers)
    for index, meta, img, params in shots:
        clock['photo'] += time.perf_counter() - t_p
        if args.limit and n >= args.limit:
            break
        n += 1
        geom = supply.geometry(meta)
        rot = int(params['rot_deg'])
        t = time.perf_counter()
        if t1 is not None:
            level1 = run_stage1(pl.estimator, img, t1, index)
        elif 'stage1' in routes:
            cell = stored1[index][0].get('chosen_level', '')
            level1 = int(cell) if cell != '' else None
        clock['stage1'] += time.perf_counter() - t
        line = [f'  [{index:4d}] L{geom["level"]}']
        if 'stage1' in routes:
            line.append(f'route L{level1 if level1 is not None else "-"}')
        for route in routes:
            level = level1 if route == 'stage1' else int(geom['level'])
            t = time.perf_counter()
            got = None
            if route in t2:
                if level is None:
                    t2[route].add('output', index, [dict(error='stage 1 failed')])
                else:
                    got = run_stage2(
                        pl, img, level,
                        lambda tile: query_grid_centre(
                            supply.camera_for(meta.ds), geom['x0'], geom['y0'],
                            params, img.shape, tile),
                        matching_rotation(rot), t2[route], index,
                        int(geom['level']), float(meta.ds))
            clock['stage2'] += time.perf_counter() - t
            t = time.perf_counter()
            if route in t3:
                if got is None and route in stored2:
                    rows = stored2[route].get(index, [])
                    if not rows or rows[0].get('error'):
                        got = None
                    else:
                        try:
                            cs = pl.candidates_at(level, stored_candidates(rows))
                            qc = QueryPatchContainer(img)
                            qc.extract_all(pl.tile_size, overlap=pl.retriever.overlap)
                            got = qc, cs
                        except UnusableLevel:
                            got = None
                if got is None:
                    t3[route].add('output', index, [dict(error='no candidates')])
                else:
                    run_stage3(pl, *got, t3[route], index)
            clock['stage3'] += time.perf_counter() - t
            if got is not None:
                line.append(f'{route}: rank1 s={got[1].best.score:.3f}')
        print('  '.join(line), flush=True)
        t_p = time.perf_counter()
    shots.close()   # an early stop drops the staged render entry here

    for tables in [t1, *t2.values(), *t3.values()]:
        if tables is not None:
            tables.write()
            print(f'  wrote {tables.entry.dir}/*_{tables.id}', flush=True)
    print('  seconds: ' + '  '.join(f'{k} {v:.0f}' for k, v in clock.items())
          + f'  ({n} FoVs)', flush=True)


# ── what a run is: shared with plot_locascope, which reads what it wrote ─────

def add_run_args(ap) -> None:
    """The flags that decide which entries a run writes -- slides, FoVs,
    stages, routes, masks, caches. A reader given the same flags computes the
    same addresses (`run_from_args`, `slide_supplies`, `stage_entries`)."""
    ap.add_argument('--datasets', nargs='+', default=['bracs/test', 'ki67_with_photo'])
    ap.add_argument('--split', default='test', choices=['val', 'test'])
    ap.add_argument('--n-wsi', type=int, default=5, help='slides per dataset')
    ap.add_argument('--stage1', default='knn:gigapath', help='<method>:<recipe>')
    ap.add_argument('--stage2', default='slidewin:gigapath', help='<method>:<recipe>')
    ap.add_argument('--stage3', default='sift:default', help='<method>:<recipe>')
    ap.add_argument('--route', choices=['stage1', 'oracle', 'both'], default='both',
                    help="the level stage 2 searches: stage 1's, the level the "
                         'FoV was placed at, or each in turn')
    # --fov, --max-ds, --sampler-*, --camera-*
    add_fov_args(ap)
    ap.add_argument('--fov-mask-cache-job', default='MppRoutingHead',
                    help='hest masks the FoVs are placed on')
    # --seg: the mask stages 1 and 2 search
    add_mask_args(ap)
    ap.add_argument('--mask-cache-job', default='MppRoutingHead',
                    help="whose mask cache the --seg mask is read from and "
                         'written to')
    ap.add_argument('--draw-cache-job', default=None,
                    help='whose cache the FoV draws are in. Default: the stage '
                         'cache job')
    ap.add_argument('--render-cache-job', default=None,
                    help='whose cache the photo record (and photos) is in. '
                         'Default: the stage cache job')
    ap.add_argument('--limit', type=int, default=0,
                    help='only the first N FoVs of each slide (a smoke run); '
                         'recorded in the entries, so a full run does not hit them')


def parse_run_args(ap, argv=None):
    """`ap.parse_args` with each `--stageN`'s recipe flags added first: which
    flags exist depends on which method was named."""
    pre, _ = ap.parse_known_args(argv)
    for n in (1, 2, 3):
        pkg = (stage1_estimation, stage2_retrieval, stage3_localization)[n - 1]
        add_config_args(ap, pkg.recipe(getattr(pre, f'stage{n}'))[2],
                        f'stage{n}', skip=_NOT_FLAGS[n])
    return ap.parse_args(argv)


def stage1_of(method: str, name: str, cfg, cls, mask_cfg, seg: str,
              device=None) -> Stage:
    """A stage-1 method on the mask the run searches: a recipe that names a
    mask recipe (`mask_cfg`, or `seg` by name) is given the run's, so its
    reference bank and stage 2 agree on what tissue is."""
    if hasattr(cfg, 'mask_cfg'):
        cfg = replace(cfg, mask_cfg=mask_cfg)
    elif hasattr(cfg, 'seg'):
        cfg = replace(cfg, seg=seg)
    return Stage(1, method, name, cfg, cls, lambda: cls(cfg, device=device))


def run_from_args(args, device=None, *, batch_size=None, multi_gpu=False,
                  read_workers=0):
    """`(stages, routes, mask_cfg)`. The stages build their models only when
    asked (`Stage.obj`), so a reader can name them for nothing."""
    mask_cfg = mask_cfg_from_args(args)
    s1 = stage1_of(*parse_stage(args, 1), mask_cfg, args.seg, device)
    m2, n2, c2, k2 = parse_stage(args, 2)
    if batch_size:
        c2 = replace(c2, encoder=replace(c2.encoder, batch_size=batch_size))
    m3, n3, c3, k3 = parse_stage(args, 3)
    s2 = Stage(2, m2, n2, c2, k2,
               lambda: k2(c2, device, multi_gpu=multi_gpu,
                          read_workers=read_workers),
               also_id=(mask_cfg.seg_id(), mask_cfg.region_id()))
    s3 = Stage(3, m3, n3, c3, k3, lambda: k3(c3))
    routes = ['stage1', ORACLE] if args.route == 'both' else [args.route]
    return (s1, s2, s3), routes, mask_cfg


def run_slides(args) -> List[tuple]:
    """`(dataset, name, path)` of every slide of the run: the first `--n-wsi`
    of each dataset's recorded `--split` -- the split is shuffled when it is
    made (`WsiSplit`), so a prefix is a random pick."""
    out = []
    for dataset in args.datasets:
        dataset_split = f'{dataset}#{args.split}'
        names = list_names(dataset=dataset_split)
        if args.n_wsi > len(names):
            raise ValueError(f'n_wsi {args.n_wsi} but {dataset_split} holds '
                             f'{len(names)} slide(s)')
        out += [(dataset, name, str(locate(name, dataset=dataset_split).path))
                for name in names[:args.n_wsi]]
    return out


def supply_for(path: str, fov, fov_masks, args, job: str,
               save_photos: bool = False) -> Optional[FovSupply]:
    """One slide's FoVs: a microscope at ds 1, ONE draw over every rung the
    recipe gives this slide (`FovRecipe.rungs_for`), the draw in the draw cache
    job and the photos' record in the render cache job (`FovSupply.cached`).
    None, said aloud, for a slide with no FoV position at any level."""
    reader = SlideReader(path)
    microscope = Render(reader, fov.sensor, fov.gap, ds=1.0, seed=fov.sampler.seed)
    plan = PlanSpec('ladder', fov.rungs_for(reader.level_downsamples),
                    camera=microscope.spec)
    try:
        supply = FovSupply.cached(
            microscope, plan, fov.sampler, masks=fov_masks,
            draw_job=args.draw_cache_job or job,
            render_job=args.render_cache_job or job, save_photos=save_photos)
    except RuntimeError as exc:
        print(f'  no FoV position at any level -- skipped '
              f'({str(exc).splitlines()[0]})', flush=True)
        return None
    if not len(supply.sampler):
        print('  no FoV position at any level -- skipped', flush=True)
        return None
    return supply


def slide_supplies(args, fov_masks, job: str, save_photos: bool = False,
                   fov=None):
    """`(dataset, name, path, supply)` per slide of the run (`run_slides`,
    `supply_for`). `fov` is the FoV recipe already resolved by a caller with
    flags of its own; default `fov_from_args`."""
    fov = fov if fov is not None else fov_from_args(args)
    for dataset, name, path in run_slides(args):
        print(f'== {name}', flush=True)
        supply = supply_for(path, fov, fov_masks, args, job, save_photos)
        if supply is not None:
            yield dataset, name, path, supply


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    add_run_args(ap)
    ap.add_argument('--save-photos', action='store_true',
                    help='keep every photo beside its record')
    ap.add_argument('--features-cache-job', default=None, metavar='JOB',
                    help="whose cache stage 2's (slide, level) grid features "
                         'are read from and written to. Off when absent.')
    ap.add_argument('--feature-store-mode', choices=['r', 'w', 'rw'], default='rw')
    ap.add_argument('--batch-size', type=int, default=None,
                    help="stage 2 encoder's batch (not identity)")
    ap.add_argument('--multi-gpu', action='store_true')
    ap.add_argument('--device', default='auto')
    args = parse_run_args(ap)

    import torch
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                          if args.device == 'auto' else args.device)
    budget = CpuBudget.for_job(processes=1).apply()
    args.workers = budget.workers
    print(f'device     : {device}   {budget.line()}', flush=True)

    stages, routes, mask_cfg = run_from_args(
        args, device, batch_size=args.batch_size, multi_gpu=args.multi_gpu,
        read_workers=budget.workers)
    for s in stages:
        print(f'stage {s.n}    : {s.id}', flush=True)
    print(f'routes     : {" ".join(routes)}   mask {args.seg} '
          f'{mask_cfg.seg_id()}/{mask_cfg.region_id()}', flush=True)

    own = Cache.job_name('BenchLocaScope')
    fov_masks = MaskMaker(MASK_RECIPES['hest'], args.fov_mask_cache_job, device)
    masks = MaskMaker(mask_cfg, args.mask_cache_job, device)
    print(f'shots      : {" ".join(args.datasets)}  #{args.split}  '
          f'n_wsi={args.n_wsi}  cache job {own}', flush=True)

    t_start = time.time()
    for _, _, path, supply in slide_supplies(args, fov_masks, own,
                                             save_photos=args.save_photos):
        bench_slide(path, supply, stages, routes, masks, args, own,
                    args.features_cache_job)
    print(f'\nTotal wall time: {time.time() - t_start:.1f}s', flush=True)


if __name__ == '__main__':
    main()
