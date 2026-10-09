"""What the benches share: the stages of a run, the tables of a stage's cache
entry, one FoV or one photo through one stage, and the one loop that runs a
slide's shots through the three stages and writes the entries.

Three programs run shots through the stages and keep the same entries:

    bench_locascope.py                    synthetic FoVs (ground truth known), the
                                          three stages, the stage-1 route and the
                                          oracle's
    cli/driver/locate_photo.py            real photos (no ground truth), the
                                          stage-1 route
    bench_modules/bench_stage1_mpp.py     stage 1 only, several methods

and three read them: `cli/plot/plot_locascope.py`, `cli/metrics/
analyze_stage1_metrics.py`, `bench_window_retrieval.py` (its slides and
photos). What they share is here, so none imports another's entry point.

STAGE ENTRIES. Each stage's tables for a slide are ONE cache entry, written
when the slide is done (`Cache.Entry.writing`), under the address of the shots
(`render=<gap>` for synthetic FoVs, `photos=<id>` for real ones):

    <shots>/stage1/{output,neighbours,probs,votes,prototypes_index}_<s1>.csv
    <shots>/stage1=<s1|oracle>/stage2/{output,tile_sims,truth,truth_sim}_<s2>.csv
    <shots>/stage1=<s1|oracle>/stage2=<s2>/stage3/{output,matches}_<s3>.csv

`<sN>` is `<method>-<recipe>-<16 hex>`: the hex is the config's identity (for
stage 2 together with the mask it searches). Every table has an `index`
column, the shot's number. A record holds what makes the entry the one asked
for -- the config, the code's versions, the upstream ids, `limit` -- and NOT
which tables it holds: that is its content (`status`'s `roles`, against the
files the entry lists as its `members`). A table a method does not offer (a
retriever without `tile_sim_rows`) is left empty.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import os
import socket
import math
import sys
import threading
import time
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import (Any, Callable, Dict, Iterable, Iterator, List, Optional,
                    Sequence, Tuple)

import numpy as np
import torch

import Cache                                                           # noqa: E402
from ConfigArgs        import add_config_args, config_from_args          # noqa: E402
from ConfigIdentity    import record, short_id                          # noqa: E402
from LocaScopePipeline import LocaScopePipeline, construct_stage        # noqa: E402
from PatchingLib       import QueryPatchContainer                       # noqa: E402
from TileSampler       import PlanSpec                                  # noqa: E402
from TissueMaskConfig  import (MASK_RECIPES, MaskMaker, add_mask_args,  # noqa: E402
                               mask_cfg_from_args)
from AccessDatasets    import list_names, locate                        # noqa: E402
from SlideReader       import SlideReader                               # noqa: E402
from camera            import Render                                    # noqa: E402
from FovSupply         import FovSupply, add_fov_args, fov_from_args    # noqa: E402
from stage1_estimation.StageInterface import EstMppResult               # noqa: E402
from stage2_retrieval.StageInterface import Candidate, CandidateSet, Retriever  # noqa: E402
from stage3_localization.StageInterface import LocalizationResultSet     # noqa: E402
import stage1_estimation                                                 # noqa: E402
import stage2_retrieval                                                  # noqa: E402
import stage3_localization                                               # noqa: E402


#: The route whose level is the one the FoV was placed at.
ORACLE = 'oracle'

#: Fields a `--stageN-*` flag does not reach: configs of their own, chosen by
#: the recipe (stage 1's encoder, bank draw) or by --seg (the mask). Stage 2's
#: encoder is reached: `--stage2-encoder-pooling cls_avg`, `--stage2-encoder-...`.
_NOT_FLAGS = {1: ('encoder', 'mask_cfg', 'sampler_cfg', 'levels'), 2: (),
              3: ()}


# ── the stages: a recipe, its id, its record ─────────────────────────────────

class Stage:
    """One stage's chosen method: its config, the label its entries are filed
    under, and the class it builds -- built on first use, so a stage whose
    entries all hit never loads a model. `runtime` is how it is built (device,
    multi_gpu, read_workers); `LocaScopePipeline.construct_stage` gives the
    constructor the ones it declares. A loop over slides builds a Stage once and
    hands the object to a pipeline per slide."""

    def __init__(self, n: int, method: str, name: str, cfg: Any, cls: type,
                 runtime: Dict[str, Any], *, also_id: tuple = ()) -> None:
        self.n: int = n
        self.method: str = method
        self.name: str = name
        self.cfg: Any = cfg
        self.cls: type = cls
        hexid: str = (short_id([cfg.identity_id(), *also_id]) if also_id
                      else cfg.identity_id())
        self.id: str = f'{method}-{name}-{hexid}'
        self._runtime: Dict[str, Any] = runtime
        self._obj: Optional[Any] = None

    def obj(self) -> Any:
        if self._obj is None:
            # `construct_stage` hands the constructor only the runtime arguments it
            # declares and drops the rest without a word, so the line says which
            # it took and which it left, to be seen in the log.
            declared: Dict[str, Any] = inspect.signature(self.cls.__init__).parameters
            took: str = ', '.join(f'{k}={v}' for k, v in self._runtime.items()
                                  if k in declared) or 'nothing'
            left: str = ', '.join(k for k in self._runtime if k not in declared)
            print(f'  [stage {self.n}] loading {self.id}  built with {took}'
                  + (f'; NOT taken by its constructor: {left}' if left else ''),
                  flush=True)
            self._obj = construct_stage(self.cls, self.cfg, self._runtime)
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


def status(entry: Any, id: str, rec: Dict[str, Any], roles: Sequence[str] = ()) -> str:
    """`'hit'`, `'miss'` or `'stale'` of variant `id` against `rec`. With
    `roles`, a hit also needs those tables among the files the variant was
    written with (the cache's `members`): which tables an entry holds is what
    it contains, not what it is, so it is not part of `rec`."""
    state: str
    diff: List[str]
    state, diff = entry.status(id, rec)
    if state == 'stale':
        print(f'  [{entry.kind}] {entry.record_path(id)} is stale: '
              + '; '.join(diff), flush=True)
    if state == 'hit' and roles:
        have: set = {p.name for p in entry.members(id)}
        missing: List[str] = [r for r in roles
                              if entry.path(r, id, '.csv').name not in have]
        if missing:
            print(f'  [{entry.kind}] {entry.record_path(id)} lacks {missing}',
                  flush=True)
            return 'miss'
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


def run_stage1(est: Any, img: np.ndarray, tables: Tables,
               index: int) -> Optional[EstMppResult]:
    """Stage 1 on one FoV, its rows added: `output` (the method's own answer),
    and what the method has beside it -- a KNN's `neighbours`, a voting
    method's per-patch `probs` and every rule of RULES over them (`votes`).
    The method's EstMppResult; None when it failed."""
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
    return r1


def estimate_from_row(row: Dict[str, str]) -> Optional[EstMppResult]:
    """The EstMppResult a stage-1 `output` row was written from (its five base
    fields; a method's own are not needed to route stage 2). None for a row
    that holds an error."""
    if row.get('error') or row.get('chosen_level', '') == '':
        return None
    return EstMppResult(
        estimated_ds=float(row['estimated_ds']), estimated_mpp=float(row['estimated_mpp']),
        chosen_ds=float(row['chosen_ds']), chosen_mpp=float(row['chosen_mpp']),
        chosen_level=int(float(row['chosen_level'])))


def placed_estimate(true_ds: float, true_level: int, base_mpp: float) -> EstMppResult:
    """What an oracle stage 1 would say: the level the FoV was placed at, its
    scale exact. Stage 2 is routed by it (route `oracle`)."""
    return EstMppResult(estimated_ds=true_ds, estimated_mpp=base_mpp * true_ds,
                        chosen_ds=true_ds, chosen_mpp=base_mpp * true_ds,
                        chosen_level=true_level)


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


@dataclass
class Truth:
    """Where a synthetic FoV was placed, as stage 2's truth needs it: the level
    and ds it was photographed at, the stage-2 rotation that sets it upright
    (`matching_rotation`), and `centre(tile)`, the level-0 point the query's
    tile grid stands for at that tile size (`query_grid_centre`). A real photo
    has none."""
    level:  int
    ds:     float
    rot:    int
    centre: Callable[[int], Tuple[float, float]]


def _cells(levels: Sequence[int]) -> str:
    """Levels as one table cell, `3 2 1`."""
    return ' '.join(str(lv) for lv in levels)


def _levels(cell: str) -> Tuple[int, ...]:
    """`_cells`, read back."""
    return tuple(int(v) for v in str(cell).split())


def run_stage2(pl: LocaScopePipeline, img: np.ndarray, route: EstMppResult,
               truth: Optional[Truth], tables: Tables,
               index: int) -> Optional[CandidateSet]:
    """The CandidateSet of the FoV, routed by `route`, its rows added; None when
    stage 2 failed or no level could be searched. `truth` places the FoV (a
    synthetic one; a real photo has none): its window is the one at the photo's
    own level, scored only when stage 2 searched that level, its box alone
    (`_truth_box`) when it was sent elsewhere. The methods' optional tables are
    asked for with `getattr` and left out when the retriever has none."""
    t0: float = time.perf_counter()
    try:
        cs: CandidateSet = pl.stage2(img, route)
    except Exception as e:                                          # noqa: BLE001
        tables.add('output', index, [dict(level=route.chosen_level,
                                          error=f'{type(e).__name__}: {e}')])
        return None
    t: float = round(time.perf_counter() - t0, 3)
    ret: Retriever = pl.retriever
    frame: Dict[str, Any] = dict(level=cs.level, ds=cs.ds,
                                 irretrievable_lvl=_cells(cs.irretrievable_lvl),
                                 alter_lvl=_cells(cs.alter_lvl))
    if not len(cs):
        tables.add('output', index, [dict(frame, error='no level holds the query',
                                          t_s=t)])
        return None
    # the query as the retriever cut it: a window's size is its tile grid
    qc: QueryPatchContainer = QueryPatchContainer(img)
    qc.extract_all(ret.tile_size, overlap=ret.overlap)
    # the output interface itself: the CandidateSet, one row per window, in
    # the frame it is expressed in
    tables.add('output', index, [dict(r, **frame, error='', t_s=t)
                                 for r in cs.rows(qc)])
    tile_rows: Optional[Callable[..., List[Dict[str, Any]]]] = getattr(
        ret, 'tile_sim_rows', None)
    if tile_rows is not None:
        tables.add('tile_sims', index, tile_rows(cs.candidates))
    if truth is not None:
        try:
            centre: Tuple[float, float] = truth.centre(ret.tile_size)
            if cs.level == truth.level and hasattr(ret, 'nearest_window'):
                row: Dict[str, Any]
                tiles: List[Dict[str, Any]]
                row, tiles = _truth_rows(ret, cs, qc, centre, truth.rot)
                tables.add('truth_sim', index, tiles)
            else:
                row = _truth_box(img.shape, truth.rot, ret.tile_size, truth.ds, centre)
            tables.add('truth', index, [dict(row, truth_level=truth.level)])
        except Exception as e:                                      # noqa: BLE001
            tables.add('truth', index, [dict(error=f'{type(e).__name__}: {e}')])
    return cs


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


def window_counts(ret: Any, main_score: float) -> Dict[str, int]:
    """How many windows the retriever scored for the query it holds the maps of,
    and where a main window scoring `main_score` stands among the main ones:

        pool_all         every window of every rotation, lattice and region
        pool_main        the main lattice's
        truth_rank_main  1 + the main windows scoring strictly above `main_score`
                         (a tie goes to it, as `nearest_window`'s rank does)

    Read off the retriever's public maps (`sim_maps_by_rot`, one `[H, W, R_q,
    C_q]` per rotation, region and lattice) and its `score`, so the stage's own
    code is not touched. Empty for a retriever that keeps no such maps."""
    maps: Optional[Dict[int, list]] = getattr(ret, 'sim_maps_by_rot', None)
    if not maps:
        return {}
    from stage2_retrieval.SlidingWinSimRot import window_score       # noqa: PLC0415
    kind: str = getattr(ret, 'score', 'mean')
    pool_main: int = 0
    pool_offset: int = 0
    higher: int = 0
    pair: list
    for pair in maps.values():
        main_sim: torch.Tensor
        offset_sim: torch.Tensor
        for main_sim, offset_sim in pair:
            if main_sim.numel():
                scores: torch.Tensor = window_score(main_sim, kind)
                pool_main += scores.numel()
                higher += int((scores > main_score).sum())
            if offset_sim.numel():
                pool_offset += int(np.prod(offset_sim.shape[:-2]))
    return dict(pool_all=pool_main + pool_offset, pool_main=pool_main,
                truth_rank_main=higher + 1)


def _truth_rows(ret: Any, cs: CandidateSet, qc: QueryPatchContainer,
                centre: Tuple[float, float], rot_deg: int
                ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """`(truth row, truth tile rows)`: the window at `rot_deg`, the rotation
    that sets the photo upright, whose centre is nearest `centre` -- the
    level-0 point the query's tile grid stands for, not the photo's centre
    (`query_grid_centre`) -- over both lattices: its score, its level-0 box,
    its distance to `centre` and its per-tile cosines; and the distance to
    the nearest main-lattice window (`d_main`). And where those windows stand
    among ALL the windows the retriever scored (it slides over both lattices,
    every rotation, every region): `truth_rank_all` for the window above,
    `truth_rank_main_in_all` for the nearest main one, `truth_rank_main` for
    that main one among the main windows only, with the pools they are ranks
    in (`pool_all`, `pool_main`; `window_counts`). The retriever scored every
    window, so these cost a count, not another pass. Needs the similarity
    maps, so it is computed here."""
    cx, cy = centre
    tw, rank_all, dist = ret.nearest_window(cx, cy, rot_deg)
    mw, rank_main_in_all, mdist = ret.nearest_window(cx, cy, rot_deg, lattices=('main',))
    box = CandidateSet(candidates=(tw,), level=cs.level, ds=cs.ds,
                       grids=cs.grids, irretrievable_lvl=(),
                       alter_lvl=()).rows(qc)[0]
    row = dict(
        truth_region=tw.region_index, truth_lattice=tw.lattice,
        truth_row=tw.row, truth_col=tw.col, truth_rotation=tw.rotation,
        truth_score=tw.score, truth_dist_l0=dist, truth_main_dist_l0=mdist,
        truth_x0=box['x0'], truth_y0=box['y0'], truth_w0=box['w0'],
        truth_h0=box['h0'], truth_rank_all=rank_all,
        truth_rank_main_in_all=rank_main_in_all, **window_counts(ret, mw.score))
    # the window's own tiles: the retriever's table without the rank column
    tiles: List[Dict[str, Any]] = [
        {k: v for k, v in r.items() if k != 'rank'}
        for r in ret.tile_sim_rows([tw])]
    return row, tiles


def run_stage3(pl: LocaScopePipeline, img: np.ndarray, cs: CandidateSet,
               tables: Tables, index: int) -> Optional[LocalizationResultSet]:
    """Stage 3 on one FoV, its rows added; the LocalizationResultSet, None when
    it failed. `matches` is the localizer's optional `pair_rows()`."""
    t0: float = time.perf_counter()
    try:
        rs: LocalizationResultSet = pl.stage3(img, cs)
    except Exception as e:                                          # noqa: BLE001
        tables.add('output', index, [dict(error=f'{type(e).__name__}: {e}')])
        return None
    t: float = round(time.perf_counter() - t0, 3)
    # the output interface itself: one result per verified rank
    tables.add('output', index, [dict(r.row(), error='', t_s=t) for r in rs]
               or [dict(error='no candidate verified', t_s=t)])
    pairs: List[Dict[str, Any]] = []
    for r in rs:
        pair_rows: Optional[Callable[[], List[Dict[str, Any]]]] = getattr(
            r, 'pair_rows', None)
        if pair_rows is not None:
            pairs += pair_rows()
    tables.add('matches', index, pairs)
    return rs


def stored_candidates(rows: Sequence[Dict[str, str]]) -> List[Candidate]:
    return [Candidate(int(r['region']), r['lattice'], int(r['row']), int(r['col']),
                      int(r['rotation']), float(r['score']))
            for r in sorted(rows, key=lambda r: int(r['rank']))]


def frame_from_rows(retriever: Retriever,
                    rows: Sequence[Dict[str, str]]) -> Optional[CandidateSet]:
    """The CandidateSet stage 2's stored `output` rows of one FoV were written
    from: the candidates read back, in the frame of the level they were found
    at (the retriever's optional `frame(level)`). None when the retriever has no
    `frame`, or the rows hold no level."""
    frame: Optional[Callable[[int], CandidateSet]] = getattr(retriever, 'frame', None)
    if frame is None or rows[0].get('level', '') == '':
        return None
    return replace(frame(int(float(rows[0]['level']))),
                   candidates=tuple(stored_candidates(rows)),
                   irretrievable_lvl=_levels(rows[0].get('irretrievable_lvl', '')),
                   alter_lvl=_levels(rows[0].get('alter_lvl', '')))


def stage2_for_stage3(pl: LocaScopePipeline, img: np.ndarray,
                      est: Optional[EstMppResult],
                      rows: Sequence[Dict[str, str]]) -> Optional[CandidateSet]:
    """The CandidateSet stage 3 runs on when stage 2's entry was a hit: the
    stored rows put back in their frame, or -- for a retriever with no `frame`
    -- stage 2 run again, its tables thrown away. None when there is nothing
    to run stage 3 on (an error row, no estimate to route by)."""
    if not rows or rows[0].get('error'):
        return None
    cs: Optional[CandidateSet] = frame_from_rows(pl.retriever, rows)
    if cs is not None:
        return cs
    if est is None:
        return None
    try:
        again: CandidateSet = pl.stage2(img, est)
    except Exception:                                               # noqa: BLE001
        return None
    return again if len(again) else None


# ── one slide ────────────────────────────────────────────────────────────────

#: The tables of each stage's entry.
ROLES = {1: ('output', 'neighbours', 'probs', 'votes', 'prototypes_index'),
         2: ('output', 'tile_sims', 'truth', 'truth_sim'),
         3: ('output', 'matches')}


#: The bench's own code in the stage-2 entry: how the truth window is placed
#: (`query_grid_centre`, `matching_rotation`, `_truth_rows`). No stage class
#: owns it, so a change to it is bumped here.
TRUTH_VERSION = 3


def stage1_record(s1: Stage, base: Cache.Address, limit: int = 0) -> Dict[str, Any]:
    """The record of stage 1's entry under `base`, the address of the photos it
    saw (a `render=` of synthetic FoVs, or a `photos=` folder of real ones): the
    method's config and code, the photos, the vote rules. Which tables the
    entry holds is its content, not its identity (`status`'s `roles`)."""
    levels: Dict[str, str] = dict(base.levels)
    upstream: Dict[str, str] = ({'render': levels['render']} if 'render' in levels
                                else {'photos': levels['photos']})
    return dict(s1.record(**upstream), rules=list(RULES), limit=int(limit))


def stage_entries(base: Cache.Address, stages: Tuple[Stage, Stage, Stage],
                  route: str, mask_cfg: Any, limit: int = 0,
                  truth: bool = True) -> Dict[int, tuple]:
    """`{n: (entry, id, record)}` of each stage `route` goes through, under
    `base`, a `render=` (or, for real photos, `photos=`) address in the stage
    job's tree. Stage 1 is there only on the stage-1 route; the oracle has
    none. `truth` is whether stage 2 places a truth window (`TRUTH_VERSION` is
    part of its record then); real photos have none."""
    s1, s2, s3 = stages
    lim = {'limit': int(limit)}            # 0 = every FoV; part of every record
    up = s1.id if route == 'stage1' else ORACLE
    a1 = base.at(stage1=up)
    out = {
        2: (a1.entry('stage2'), s2.id,
            dict(s2.record(stage1=up, seg=mask_cfg.seg_id(),
                           region=mask_cfg.region_id()),
                 **({'truth': TRUTH_VERSION} if truth else {}), **lim)),
        3: (a1.at(stage2=s2.id).entry('stage3'), s3.id,
            dict(s3.record(stage2=s2.id), **lim))}
    if route == 'stage1':
        out[1] = (base.entry('stage1'), s1.id, stage1_record(s1, base, limit))
    return out


# ── one slide's shots through the stages ─────────────────────────────────────

@dataclass
class Shot:
    """One query on its way through the stages: its number (`index`, the key of
    every table row), the image, the label its progress line carries, and --
    for a synthetic FoV -- where it was placed (`Truth`); a real photo has no
    placement, and so has no oracle route and no truth tables."""
    index: int
    img:   np.ndarray
    label: str
    truth: Optional[Truth] = None


def require_gpu_if_allocated(device: torch.device, requested: str = 'auto') -> None:
    """A job that was given a GPU and finds none on its node is on a broken one.
    Stop it at once, saying which node: running on, the CPU, it would encode a
    slide's level for hours (a whole day's job) and print nothing wrong. A run
    that asked for the CPU (`requested` not 'auto') or was given no GPU is left
    alone."""
    given: str = ''.join(os.environ.get(v, '') for v in
                         ('SLURM_GPUS_ON_NODE', 'SLURM_JOB_GPUS', 'SLURM_GPUS',
                          'SLURM_GPUS_PER_NODE'))
    if requested == 'auto' and device.type == 'cpu' and given:
        node: str = socket.gethostname()
        raise SystemExit(f'this job was given a GPU but torch finds no CUDA on {node}: '
                         f'not running on the CPU. Resubmit with --exclude={node} added.')


def fmt_time(seconds: float) -> str:
    """`45.0 sec`, or `7.5 min` from two minutes up."""
    return f'{seconds:.1f} sec' if seconds < 120 else f'{seconds / 60:.1f} min'


def fmt_rate(count: float, seconds: float, unit: str) -> str:
    """How many `unit` per second, or per minute when that is under one a second:
    `9.0 photo/sec`, `1.6 photo/min`."""
    if seconds <= 0 or not count:
        return ''
    rate: float = count / seconds
    if rate >= 100:
        return f'{rate:.0f} {unit}/sec'
    if rate >= 1:
        return f'{rate:.1f} {unit}/sec'
    return f'{rate * 60:.2f} {unit}/min'


def photo_time(waited: float, d1: float, d2: Dict[str, float],
               d3: Dict[str, float]) -> str:
    """One photo's time, step by step, under its result line: the wait for the
    photo, stage 1, and stages 2 and 3 once per route that ran them, then the
    photo's total. Every figure is wall-clock seconds."""
    def per_route(d: Dict[str, float]) -> str:
        return ' + '.join(f'{v:.1f} sec [{r}]' for r, v in d.items()) if d else 'read back'
    total: float = waited + d1 + sum(d2.values()) + sum(d3.values())
    return (f'        time: photo {waited:.1f} sec | stage 1 {d1:.1f} sec | '
            f'stage 2 {per_route(d2)} | stage 3 {per_route(d3)} | '
            f'this photo {total:.1f} sec')


def stage3_split(splits: Dict[str, Dict[str, float]]) -> str:
    """TEMPORARY (stage 3 timing split): where one photo's stage 3 went, per route.
    The candidates run on parallel threads, so their steps, summed, add up to
    more than the wall-clock they took. Removed with the lines marked so."""
    lines: List[str] = []
    for route, t in splits.items():
        n: int = int(t.get('candidates', 0))
        if not n:
            continue
        lines.append(
            f'        stage 3 [{route}]: {n} candidates in {t.get("wall", 0.0):.1f} sec '
            f'wall-clock; summed over the candidates (parallel, so more than that): '
            f'read the crop {t.get("read", 0.0):.1f} sec | SIFT on the crops '
            f'{t.get("crop_sift", 0.0):.1f} sec ({int(t.get("crop_kps", 0.0) / n):,} '
            f'keypoints each) | match {t.get("match", 0.0):.1f} sec | RANSAC '
            f'{t.get("ransac", 0.0):.1f} sec | the query: SIFT '
            f'{t.get("query_sift", 0.0):.1f} sec, {int(t.get("query_kps", 0.0)):,} keypoints')
    return '\n'.join(lines)


def fmt_each(count: float, seconds: float, unit: str) -> str:
    """How long one `unit` takes: `12.0 sec/photo`, or `2.1 min/photo` from two
    minutes up."""
    if not count:
        return ''
    each: float = seconds / count
    return (f'{each:.2f} sec/{unit}' if each < 120 else f'{each / 60:.1f} min/{unit}')


def _step_lines(clock: Dict[str, float], n: int, routes: Sequence[str]) -> List[str]:
    """The table of steps shared by one slide's report and the whole run's: each
    step's time, how many photos a second (or a minute) it handles and how long
    one photo takes in it, then the whole pipeline the same way."""
    total: float = sum(clock.values())
    per: str = f'{len(routes)} route{"" if len(routes) == 1 else "s"}'
    steps: List[Tuple[str, str]] = [
        ('photo', 'get the photo (render, or read the file)'),
        ('stage1', 'stage 1  mpp estimate'),
        ('stage2', f'stage 2  retrieval, {per}'),
        ('stage3', f'stage 3  localization, {per}')]
    lines: List[str] = [f'  {"":<44}{"time":>10}{"speed":>18}{"one photo":>16}']
    for key, label in steps:
        sec: float = clock.get(key, 0.0)
        lines.append(f'  {label:<44}{fmt_time(sec):>10}{fmt_rate(n, sec, "photo"):>18}'
                     f'{fmt_each(n, sec, "photo"):>16}')
    lines.append(f'  {"whole pipeline":<44}{fmt_time(total):>10}'
                 f'{fmt_rate(n, total, "photo"):>18}{fmt_each(n, total, "photo"):>16}')
    return lines


def time_report(clock: Dict[str, float], n: int, routes: Sequence[str],
                levels: Dict[int, Tuple[float, int]]) -> str:
    """Where one slide's time went, with the speed of every step. `clock` holds
    `photo` (the wait for the next photo: a synthetic FoV is rendered, a real
    one read from its file), `stage1`, `stage2` and `stage3`, each summed over
    the photos (and, for stages 2 and 3, over the routes); `n` is how many
    photos. `levels` is what the retriever encoded: `{level: (seconds, tiles)}`,
    the whole level once, which is inside stage 2's time, not on top of it."""
    total: float = sum(clock.values())
    lines: List[str] = [f'  this slide: {n} photo in {fmt_time(total)}']
    lines += _step_lines(clock, n, routes)
    if levels:
        lines.append('  encoding a whole level of the slide, once (inside stage 2):')
        for level, (sec, tiles) in sorted(levels.items()):
            lines.append(f'    L{level}  {fmt_time(sec):>9}   {tiles:,} tile, '
                         f'{fmt_rate(tiles, sec, "tile")}')
    return '\n'.join(lines)


class Heartbeat:
    """A line in the log whenever one step has gone on for `interval` seconds
    without ending, so a long step can be told from a stuck run: what it is
    doing and for how long. A step that ends sooner prints nothing."""

    def __init__(self, interval: float = 60.0) -> None:
        self.interval: float = interval
        self._label: str = ''
        self._since: float = time.perf_counter()
        self._stop: threading.Event = threading.Event()
        self._thread: threading.Thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def doing(self, label: str) -> None:
        self._label, self._since = label, time.perf_counter()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            label: str = self._label
            took: float = time.perf_counter() - self._since
            if label and took >= self.interval:
                print(f'      ... still in {label}: {took:.0f} sec so far', flush=True)

    def close(self) -> None:
        self._stop.set()


class RunClock:
    """The time of a whole run: every slide's clock added up, so the end of the
    log says what the run cost step by step and how fast it went."""

    def __init__(self) -> None:
        self.clock: Dict[str, float] = {}
        self.photos: int = 0
        self.slides: int = 0
        #: `{level: [(seconds, tiles), ...]}`, one entry per slide that encoded it
        self.levels: Dict[int, List[Tuple[float, int]]] = {}

    def add(self, clock: Dict[str, float], n: int,
            levels: Dict[int, Tuple[float, int]]) -> None:
        key: str
        for key, sec in clock.items():
            self.clock[key] = self.clock.get(key, 0.0) + sec
        self.photos += n
        self.slides += 1
        level: int
        for level, one in levels.items():
            self.levels.setdefault(level, []).append(one)

    def report(self, routes: Sequence[str]) -> str:
        total: float = sum(self.clock.values())
        lines: List[str] = [
            f'  whole run: {self.slides} slide, {self.photos} photo in '
            f'{fmt_time(total)} (the photos only: not model loading, not the cache '
            f'writes)']
        lines += _step_lines(self.clock, self.photos, routes)
        if self.levels:
            lines.append('  encoding a whole level of a slide, once per slide (inside '
                         'stage 2), averaged over the slides that encoded it:')
            level: int
            for level, got in sorted(self.levels.items()):
                sec: float = sum(g[0] for g in got) / len(got)
                tiles: int = int(sum(g[1] for g in got) / len(got))
                lines.append(f'    L{level}  {fmt_time(sec):>9}   {tiles:,} tile on average, '
                             f'{fmt_rate(sum(g[1] for g in got), sum(g[0] for g in got), "tile")}'
                             f'   ({len(got)} slide)')
        return '\n'.join(lines)


def run_slide_stages(
        path: str, shots: Iterable[Shot], stages: Tuple[Stage, Stage, Stage],
        routes: Sequence[str], masks: MaskMaker, base: Cache.Address,
        roles: Dict[int, Tuple[str, ...]], own: str, limit: int = 0,
        truth: bool = True, before: Optional[Callable[[], None]] = None,
        after_stage3: Optional[Callable[..., None]] = None,
        describe: Optional[Callable[..., str]] = None,
        run_clock: Optional['RunClock'] = None) -> bool:
    """One slide's shots through the stages, each stage's tables one cache
    entry under `base` (the address of the shots), written when the slide is
    done. False when every entry already hits -- nothing is run, no model
    loaded. The bench (synthetic FoVs), locate_photo (real photos) and the
    stage-1 bench share this loop.

    `roles` are the tables each stage keeps (`ROLES`, or fewer for a shot
    source with no truth); `truth` whether stage 2 places a truth window.
    `shots` is not started before something has to run. `limit` stops after N
    shots (a smoke run; recorded in the entries). Hooks: `before()` once
    something is to run, before the models load; `after_stage3(shot, route,
    cs, rs, tables)` after each shot's stage 3 -- `cs` and `rs` None where
    there was nothing to run it on -- for a table of the caller's own;
    `describe(shot, route, cs, rs)` a string to add to the progress line."""
    s1, s2, s3 = stages
    t1: Optional[Tables] = None
    t2: Dict[str, Tables] = {}
    t3: Dict[str, Tables] = {}
    stored1: Dict[int, List[Dict[str, str]]] = {}
    stored2: Dict[str, Dict[int, List[Dict[str, str]]]] = {}
    route: str
    # every entry this slide needs, and whether it is there
    for route in routes:
        ents = stage_entries(base, stages, route, masks.cfg, limit, truth)
        if 1 in ents:
            e, sid, rec = ents[1]
            if status(e, sid, rec, roles[1]) != 'hit':
                t1 = Tables(e, sid, rec, roles[1])
            else:
                stored1 = by_index(read_rows(e.path('output', sid, '.csv')))
        e, sid, rec = ents[2]
        if status(e, sid, rec, roles[2]) != 'hit':
            t2[route] = Tables(e, sid, rec, roles[2])
        e3, sid3, rec3 = ents[3]
        if status(e3, sid3, rec3, roles[3]) != 'hit':
            t3[route] = Tables(e3, sid3, rec3, roles[3])
            if route not in t2:
                stored2[route] = by_index(read_rows(e.path('output', sid, '.csv')))
    if t1 is None and not t2 and not t3:
        print('  every stage hit -- nothing to run', flush=True)
        return False
    print('  to run: ' + ' '.join(
        ([f'stage1'] if t1 else []) + [f'stage2[{r}]' for r in t2]
        + [f'stage3[{r}]' for r in t3]), flush=True)
    if before is not None:
        before()

    pl: LocaScopePipeline = LocaScopePipeline(
        path, masks,
        stage1=s1.obj() if t1 else None,
        stage2=s2.obj() if (t2 or t3) else None,
        stage3=s3.obj() if t3 else None,
        topk=int(getattr(s3.cfg, 'topk', 10)),
        bank_cache_job=own).build()
    if t1 is not None:
        t1.add_slide('prototypes_index', support_rows(pl.estimator))

    clock: Dict[str, float] = dict.fromkeys(['photo', 'stage1', 'stage2', 'stage3'], 0.0)
    n: int = 0
    beat: Heartbeat = Heartbeat()
    beat.doing('waiting for the first photo')
    t_p: float = time.perf_counter()
    shot: Shot
    for shot in shots:
        waited: float = time.perf_counter() - t_p
        clock['photo'] += waited
        if limit and n >= limit:
            break
        n += 1
        d2: Dict[str, float] = {}
        d3: Dict[str, float] = {}
        splits: Dict[str, Dict[str, float]] = {}   # TEMPORARY (stage 3 timing split)
        beat.doing(f'photo {shot.index}, stage 1')
        t: float = time.perf_counter()
        r1: Optional[EstMppResult] = None
        if t1 is not None:
            r1 = run_stage1(pl.estimator, shot.img, t1, shot.index)
        elif 'stage1' in routes:
            rows1: List[Dict[str, str]] = stored1.get(shot.index, [])
            r1 = estimate_from_row(rows1[0]) if rows1 else None
        d1: float = time.perf_counter() - t
        clock['stage1'] += d1
        line: List[str] = [f'  [{shot.index:4d}] {shot.label}']
        if 'stage1' in routes:
            line.append(f'route L{r1.chosen_level if r1 is not None else "-"}')
        for route in routes:
            est: Optional[EstMppResult]
            if route == 'stage1':
                est = r1
            elif shot.truth is None:
                raise ValueError(f'route {route!r} needs the shot placed; shot '
                                 f'{shot.index} has no truth')
            else:
                est = placed_estimate(shot.truth.ds, shot.truth.level, pl.base_mpp)
            beat.doing(f'photo {shot.index}, stage 2 [{route} route]')
            t = time.perf_counter()
            got: Optional[CandidateSet] = None
            if route in t2:
                if est is None:
                    t2[route].add('output', shot.index, [dict(error='stage 1 failed')])
                else:
                    got = run_stage2(pl, shot.img, est, shot.truth, t2[route],
                                     shot.index)
            d2[route] = time.perf_counter() - t
            clock['stage2'] += d2[route]
            beat.doing(f'photo {shot.index}, stage 3 [{route} route]')
            t = time.perf_counter()
            rs: Optional[LocalizationResultSet] = None
            if route in t3:
                if got is None and route in stored2:
                    got = stage2_for_stage3(pl, shot.img, est,
                                            stored2[route].get(shot.index, []))
                if got is None or not len(got):
                    t3[route].add('output', shot.index, [dict(error='no candidates')])
                else:
                    rs = run_stage3(pl, shot.img, got, t3[route], shot.index)
                if after_stage3 is not None:
                    after_stage3(shot, route, got, rs, t3[route])
            d3[route] = time.perf_counter() - t
            clock['stage3'] += d3[route]
            splits[route] = dict(getattr(pl.localizer, 'last_timing', None) or {})  # TEMPORARY
            if got is not None and len(got):
                line.append(f'{route}: rank1 s={got.best.score:.3f}'
                            + (describe(shot, route, got, rs) if describe else ''))
        print('  '.join(line), flush=True)
        print(photo_time(waited, d1, d2, d3), flush=True)
        split_text: str = stage3_split(splits)       # TEMPORARY (stage 3 timing split)
        if split_text:                               # TEMPORARY (stage 3 timing split)
            print(split_text, flush=True)            # TEMPORARY (stage 3 timing split)
        beat.doing('waiting for the next photo')
        t_p = time.perf_counter()
    beat.close()
    close: Optional[Callable[[], None]] = getattr(shots, 'close', None)
    if close is not None:
        close()   # an early stop drops the staged render entry here

    tables: Optional[Tables]
    for tables in [t1, *t2.values(), *t3.values()]:
        if tables is not None:
            tables.write()
            print(f'  wrote {tables.entry.dir}/*_{tables.id}', flush=True)
    encoded: Dict[int, Tuple[float, int]] = dict(
        getattr(pl.retriever, 'level_encodes', {}) or {})
    print(time_report(clock, n, routes, encoded), flush=True)
    if run_clock is not None:
        run_clock.add(clock, n, encoded)
    return True


# ── what a run is: shared with plot_locascope, which reads what it wrote ─────

def add_run_args(ap) -> None:
    """The flags that decide which entries a run writes -- slides, FoVs,
    stages, routes, masks, caches. A reader given the same flags computes the
    same addresses (`run_from_args`, `slide_supplies`, `stage_entries`)."""
    ap.add_argument('--datasets', nargs='+', default=['bracs/test', 'ki67_with_photo'])
    ap.add_argument('--split', default='val', choices=['val', 'test'],
                    help='val: where methods are chosen; test: where the choice is confirmed')
    ap.add_argument('--n-wsi', type=int, default=10, help='slides per dataset')
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
    ap.add_argument('--slide-index', type=int, default=None,
                    help='only the Nth slide of the run (a Slurm array task); the '
                         'slides are the datasets in order, --n-wsi of each')


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
    return Stage(1, method, name, cfg, cls, dict(device=device))


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
               dict(device=device, multi_gpu=multi_gpu, read_workers=read_workers),
               also_id=(mask_cfg.seg_id(), mask_cfg.region_id()))
    s3 = Stage(3, m3, n3, c3, k3, dict(device=device))   # the matching runs on a CUDA device
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
    index: Optional[int] = getattr(args, 'slide_index', None)
    if index is not None:
        if not 0 <= index < len(out):
            print(f'[skip] --slide-index {index} but the run has {len(out)} slides',
                  flush=True)
            return []
        out = [out[index]]
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
