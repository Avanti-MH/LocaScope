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
analyze_stage1_metrics.py`, `bench_stage2_retriever.py` (its slides and
photos). What they share is here, so none imports another's entry point.

STAGE ENTRIES. Each stage's tables for a slide are ONE cache entry, written
when the slide is done (`Cache.Entry.writing`), under the address of the shots
(`render=<gap>` for synthetic FoVs, `photos=<id>` for real ones):

    <shots>/stage1/{output,neighbours,probs,votes,prototypes_index}_<s1>.csv
    <shots>/stage1=<s1|oracle>/stage2/{output,tile_sims,truth,truth_sim,rerank_scores}_<s2>.csv
    <shots>/stage1=<s1|oracle>/stage2=<s2>/stage3/{output,matches}_<s3>.csv

`<sN>` is `<method>-<recipe>-<16 hex>`: the hex is the config's identity (for
stage 2 together with the mask it searches). Every table has an `index`
column, the shot's number. A record holds what makes the entry the one asked
for -- the config, the code's versions, the upstream ids, `limit` -- and NOT
which tables it holds: that is its content (`status`'s `roles`, against the
files the entry lists as its `members`). A table a stage-1 or stage-3 method
does not offer is left empty; a stage-2 phase writes only the tables it holds
(`phase_roles`).
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
import dataclasses
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import (Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional,
                    Sequence, Tuple)

import numpy as np
import torch

import Cache                                                           # noqa: E402
from ConfigArgs        import add_config_args, config_from_args          # noqa: E402
from ConfigIdentity    import file_fingerprint, record, short_id                          # noqa: E402
from LocaScopePipeline import LocaScopePipeline, construct_stage        # noqa: E402
from PatchingLib       import QueryPatchContainer                       # noqa: E402
from TileSampler       import PlanSpec                                  # noqa: E402
from TissueMaskConfig  import (MASK_RECIPES, MaskMaker, add_mask_args,  # noqa: E402
                               mask_cfg_from_args)
from AccessDatasets    import by_source, locate                         # noqa: E402
from SlideReader       import SlideReader                               # noqa: E402
from camera            import Render                                    # noqa: E402
from FovSupply         import FovSupply, add_fov_args, fov_from_args    # noqa: E402
from stage1_estimation.StageInterface import EstMppResult               # noqa: E402
from stage2_retrieval.Context import nearest_window, query_dims          # noqa: E402
from stage2_retrieval.Retriever import RetrievalOutput, Retriever        # noqa: E402
from stage2_retrieval.StageInterface import (Candidate, CandidateSet,  # noqa: E402
                                             RerankedCandidate, ShiftedCandidate)
from stage3_localization.StageInterface import LocalizationResultSet     # noqa: E402
import stage1_estimation                                                 # noqa: E402
import stage2_retrieval                                                  # noqa: E402
import stage3_localization                                               # noqa: E402
from stage2_retrieval.Naming import config_label, entry_name, stage_name   # noqa: E402


#: The route whose level is the one the FoV was placed at.
ORACLE = 'oracle'

#: Fields a `--stageN-*` flag does not reach: configs of their own, chosen by
#: the recipe (stage 1's encoder, bank draw) or by --seg (the mask). A single
#: stage-2 retriever's fields are reached (`--stage2-raw-topk 500`,
#: `--stage2-encoder-pooling cls_avg`); a chain's `phases` (configs of different
#: classes) are chosen by the recipe or the arm: no flag sets a chain.
_NOT_FLAGS = {1: ('encoder', 'mask_cfg', 'sampler_cfg', 'levels'),
              2: ('phases',), 3: ()}


def weights_of_config(cfg: Any) -> List[str]:
    """Every checkpoint file a config reads, nested configs and chains included: each `weights` field that
    names a file, and what `weights_files()` lists. Sorted, once each."""
    found: set = set()

    def walk(obj: Any) -> None:
        if isinstance(obj, (tuple, list)):
            for item in obj:
                walk(item)
        elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            for f in dataclasses.fields(obj):
                value: Any = getattr(obj, f.name)
                if f.name == 'weights' and isinstance(value, str) and value:
                    found.add(value)
                else:
                    walk(value)
            listed: Any = getattr(obj, 'weights_files', None)
            if callable(listed):
                found.update(listed())

    walk(cfg)
    return sorted(found)


def stage_hexid(cfg: Any, also_id: tuple = ()) -> str:
    """The hex id of a stage's entry: the config's identity, `also_id` (the mask ...) and the CONTENT of every
    checkpoint the config reads, so a retrained file under the same name is another entry and a renamed
    file with the same bytes is the same one. A checkpoint that is not there is a part of its own (the
    build fails on it later). A config that reads none keeps the id it had."""
    from stage1_estimation.StageInterface import weights_path    # noqa: PLC0415
    prints: List[str] = []
    for name in weights_of_config(cfg):
        try:
            prints.append(f'weights={file_fingerprint(weights_path(name))}')
        except OSError:
            prints.append(f'weights={name}:missing')
    if not also_id and not prints:
        return cfg.identity_id()
    return short_id([cfg.identity_id(), *also_id, *prints])


def weights_prints(cfg: Any) -> Dict[str, str]:
    """`{'weights[<file>]': fingerprint}` of every checkpoint `cfg` reads, for its record."""
    from stage1_estimation.StageInterface import weights_path    # noqa: PLC0415
    out: Dict[str, str] = {}
    for name in weights_of_config(cfg):
        try:
            out[f'weights[{name}]'] = file_fingerprint(weights_path(name))
        except OSError:
            out[f'weights[{name}]'] = 'missing'
    return out


def phase_ids(cfg: Any, also_id: tuple = ()) -> List[Tuple[str, str]]:
    """`(entry name, own hash)` of every phase of a stage-2 chain: each phase's hash is its config's (with its
    checkpoints, `stage_hexid`), the mask (`also_id`) and the hash of the phase before it, which its name
    carries too (`_after-`). A phase is the same entry in every chain that runs it after the same phases."""
    out: List[Tuple[str, str]] = []
    up: str = ''
    for phase in cfg.phases:
        own: str = stage_hexid(phase, tuple(also_id) + ((up,) if up else ()))
        out.append((entry_name(config_label(phase), own, up), own))
        up = own
    return out


def phase_record(cfg: Any, name: str, after: str, **upstream) -> dict:
    """The record of one phase of a stage-2 chain: that phase's config and code, its checkpoints, and the hash
    of the phase it reads its list from (`after`, '' for the first). The pipeline's and the stage-2 bench's
    entries of one phase are one entry, so both make its record here."""
    return dict(record(cfg, also=(stage2_retrieval.retriever_class(cfg),), after=after, **upstream,
                       **weights_prints(cfg)), label=name)


def as_chain(cfg: Any, cls: type) -> Tuple[Any, type]:
    """A stage-2 config as a chain: a single retriever is a chain of one phase, so every stage 2 is filed the
    same way, phase by phase."""
    from stage2_retrieval.MultiPhases import MultiPhases, MultiPhasesConfig   # noqa: PLC0415
    if isinstance(cfg, MultiPhasesConfig):
        return cfg, cls
    return MultiPhasesConfig(phases=(cfg,)), MultiPhases


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
        hexid: str = stage_hexid(cfg, also_id)
        self.id: str = f'{method}-{name}-{hexid}'
        #: stage 2, a chain (`MultiPhasesConfig`): `(entry name, own hash)` of every phase, each filed after the
        #: one before (`phase_ids`); the stage's id is its last phase's, the list stage 3 reads
        self.phases: List[Tuple[str, str]] = []
        if n == 2 and hasattr(cfg, 'phases'):
            self.phases = phase_ids(cfg, also_id)
            self.id = self.phases[-1][0]
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
        """The config's record, the class's code and the fingerprint of every
        checkpoint it reads (the content is in the id too: `stage_hexid`)."""
        return dict(record(self.cfg, also=(self.cls,), **upstream, **weights_prints(self.cfg)), label=self.id)

    def phase_record(self, i: int, **upstream) -> dict:
        """The record of phase `i` of a stage-2 chain (`phase_record`)."""
        return phase_record(self.cfg.phases[i], self.phases[i][0], self.phases[i - 1][1] if i else '',
                            **upstream)


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
            r1 = est.from_probs(probs, est.cfg.vote, patches=patches)
        else:
            r1 = est.estimate(img)
    except Exception as e:                                          # noqa: BLE001
        print(f'  [stage 1] photo {index} failed: {type(e).__name__}: {e}', flush=True)
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
               truth: Optional[Truth], tables: Dict[int, Tables],
               stored: Dict[int, Dict[int, List[Dict[str, str]]]],
               index: int) -> Optional[CandidateSet]:
    """The list stage 3 reads (the chain's last phase's) for the FoV, routed by
    `route`; None when stage 2 failed or no level could be searched. Stage 2 is a
    chain (`MultiPhases`); `tables` are the phases to write, by phase, and
    `stored` the rows of the phases whose entries hit: such a phase is not run,
    its list goes on to the next (`MultiPhases.run`'s `cached`).

    Each phase written gets its list (`output`), the tables its retriever writes
    of what it offers (`Retriever.tables`), and on the first phase, with `truth`
    (a synthetic FoV), the answers (`truth_rows`): scored and ranked among every
    window when the phase searched the photo's own level, their windows alone
    otherwise."""
    chain: Any = pl.retriever

    def cached(i: int, upstream: Any) -> Optional[CandidateSet]:
        if i not in stored:
            return None
        rows: List[Dict[str, str]] = stored[i].get(index, [])
        return frame_from_rows(chain, rows) if rows and not rows[0].get('error') else None

    t0: float = time.perf_counter()
    try:
        out: RetrievalOutput = pl.stage2(img, route, cached)
    except Exception as e:                                          # noqa: BLE001
        for tab in tables.values():
            tab.add('output', index, [dict(level=route.chosen_level, error=f'{type(e).__name__}: {e}')])
        return None
    t: float = round(time.perf_counter() - t0, 3)
    outs: List[RetrievalOutput] = chain.outputs(out)
    first: Any = chain.cfg.phases[0]
    tile: int = int(first.tile_size)
    overlap: bool = bool(getattr(first, 'overlap', True))
    # the query as the first phase cut it: a window's size is its tile grid
    qc: QueryPatchContainer = QueryPatchContainer(img)
    qc.extract_all(tile, overlap=overlap)
    for i, tab in tables.items():
        write_phase(tab, index, outs[i], chain.phases[i], img, qc, truth, tile, overlap, t)
    last: CandidateSet = outs[-1].candidates
    return last if len(last) else None


def write_phase(tab: Tables, index: int, out: RetrievalOutput, retriever: Any, img: np.ndarray,
                qc: QueryPatchContainer, truth: Optional[Truth], tile: int, overlap: bool, t_s: float) -> None:
    """One FoV's rows of one phase's entry: its list (`output`, in the frame it is expressed in), the tables
    its retriever writes of what it offers (`Retriever.tables`), and, when the entry holds a truth
    (`phase_roles`) and the FoV has one, its answers (`truth_rows`). The one writer of a stage-2 entry, the
    pipeline's and the stage-2 bench's alike."""
    cs: CandidateSet = out.candidates
    frame: Dict[str, Any] = dict(level=cs.level, ds=cs.ds, irretrievable_lvl=_cells(cs.irretrievable_lvl),
                                 alter_lvl=_cells(cs.alter_lvl))
    if not len(cs):
        tab.add('output', index, [dict(frame, error='no level holds the query', t_s=t_s)])
        return
    tab.add('output', index, [dict(r, **frame, n_listed=len(cs), error='', t_s=t_s) for r in cs.rows(qc)])
    for role, rows in retriever.tables(out, qc).items():
        if role in tab.rows:
            tab.add(role, index, rows)
    if truth is not None and 'truth' in tab.rows:
        try:
            row, tiles = truth_rows(out, img, qc, truth, tile, overlap, getattr(retriever, 'frame', None))
            if 'truth_sim' in tab.rows:
                tab.add('truth_sim', index, tiles)
            tab.add('truth', index, [dict(row, truth_level=truth.level)])
        except Exception as e:                                      # noqa: BLE001
            tab.add('truth', index, [dict(error=f'{type(e).__name__}: {e}')])


# ── answers: which windows are a FoV's truth, and where they rank ──────────────
#
# THE ONE DEFINITION of a FoV's answer windows, their ranks in a first phase's scores, and their places in a
# list, for everyone that judges stage 2 -- the pipeline's truth (`truth_rows`), the stage-2 bench (its streamed
# pass and its engine), and the analyses reading the cache back (`answers_of_row`). A second copy of any of
# these is how the bench once named one window and the stage another.

#: A FoV's two answers, by lattice: the main-lattice window and the offset-lattice window whose centre is
#: nearest the point its query's tile grid stands for. The main one is THE answer; a list holding either has
#: the FoV.
ANSWER_LATTICES: Tuple[str, ...] = ('main', 'offset')

#: A window's identity, `Candidate.key()`: `(region, lattice, row, col, rotation)`.
Key = Tuple[int, str, int, int, int]

#: A first phase's score of every window (`window_scores`): one `[rows, cols]` per `(rotation, region, lattice)`.
Maps = Dict[Tuple[int, int, str], torch.Tensor]


@dataclass(frozen=True)
class Answer:
    """One answer of a FoV: the window of `lattice`, at the rotation that sets the photo upright, whose centre is
    nearest the level-0 point the query's tile grid stands for, among the windows the stage scores -- the real
    grid's (`Context.nearest_window`, `PatchGrid.tile_origin_l0`). `box` is its level-0 `(x0, y0, w0, h0)`,
    `dist_l0` its centre's distance to the point; `dx`, `dy` that distance by axis (None for an answer read
    back from a table, which keeps the distance only)."""
    lattice: str
    region: int
    row: int
    col: int
    rotation: int
    box: Tuple[int, int, float, float]
    dist_l0: float
    dx: Optional[float] = None
    dy: Optional[float] = None

    @property
    def key(self) -> Key:
        return (self.region, self.lattice, self.row, self.col, self.rotation)

    @property
    def origin(self) -> Tuple[int, int]:
        return int(self.box[0]), int(self.box[1])


@dataclass(frozen=True)
class Answers:
    """A FoV's answers: `main`, THE answer, and `offset` when there is that lattice."""
    main: Answer
    offset: Optional[Answer] = None

    def __iter__(self) -> Iterator[Answer]:
        yield self.main
        if self.offset is not None:
            yield self.offset

    def main_only(self) -> 'Answers':
        return Answers(self.main)

    def within(self, half_l0: float) -> bool:
        """Every answer is less than `half_l0` from the point on each axis: half a tile is where the lattice
        point nearest the point lies, so a farther one is a window standing in for one the region lacks."""
        return all(a.dx is not None and a.dy is not None and abs(a.dx) <= half_l0 and abs(a.dy) <= half_l0
                   for a in self)


def answers(frame: CandidateSet, dims: Tuple[int, int], centre: Tuple[float, float], rotation: int,
            lattices: Sequence[str] = ANSWER_LATTICES) -> Optional[Answers]:
    """The answers of a query of `dims` (rows, cols of its tile grid at `rotation`) whose grid stands for the
    level-0 `centre`, in `frame` (a level's regions and grids), one per lattice of `lattices` that has a window;
    None when the main lattice has none."""
    cx, cy = centre
    out: Dict[str, Answer] = {}
    for lattice in lattices:
        found: Optional[Tuple[int, str, int, int, float]] = nearest_window(frame, dims, cx, cy, rotation,
                                                                           (lattice,))
        if found is None:
            continue
        region, _, i, j, dist = found
        x0, y0 = frame.origin_l0(Candidate(region, lattice, i, j, int(rotation), 0.0))
        tile: float = float(frame.grids[region].tile_size) * float(frame.ds)
        w0, h0 = dims[1] * tile, dims[0] * tile
        out[lattice] = Answer(lattice=lattice, region=region, row=i, col=j, rotation=int(rotation),
                              box=(x0, y0, w0, h0), dist_l0=float(dist),
                              dx=x0 + w0 / 2.0 - cx, dy=y0 + h0 / 2.0 - cy)
    if 'main' not in out:
        return None
    return Answers(out['main'], out.get('offset'))


def _number(text: Any) -> Optional[float]:
    """A table cell as a number: None for '', None or NaN."""
    try:
        value: float = float(text)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) else value


def answers_of_row(row: Mapping[str, Any]) -> Optional[Answers]:
    """The answers a `truth` row (`truth_row`, read back as text or numbers) names; None when it names none."""
    def one(lattice: str) -> Optional[Answer]:
        p: str = f'truth_{lattice}_'
        cells: List[Optional[float]] = [_number(row.get(p + k)) for k in ('region', 'row', 'col', 'x0', 'y0',
                                                                          'w0', 'h0', 'dist_l0')]
        rotation: Optional[float] = _number(row.get('truth_rotation'))
        if any(v is None for v in cells) or rotation is None:
            return None
        region, r, c, x0, y0, w0, h0, dist = cells
        return Answer(lattice=lattice, region=int(region), row=int(r), col=int(c), rotation=int(rotation),
                      box=(int(x0), int(y0), float(w0), float(h0)), dist_l0=float(dist))

    main: Optional[Answer] = one('main')
    return None if main is None else Answers(main, one('offset'))


@dataclass(frozen=True)
class Pools:
    """How many windows a first phase scored for a query: every region, rotation and lattice."""
    main: int
    offset: int

    @property
    def all(self) -> int:
        return self.main + self.offset


def pools_of(scores: Maps) -> Pools:
    """The pools of a first phase's search, counted off its scores (a region too small for the query has none)."""
    main: int = sum(int(m.numel()) for (_, _, lattice), m in scores.items() if lattice == 'main')
    offset: int = sum(int(m.numel()) for (_, _, lattice), m in scores.items() if lattice == 'offset')
    return Pools(main=main, offset=offset)


def answer_score(scores: Maps, key: Key) -> Optional[float]:
    """The score of the window `key` names; None when no map holds it."""
    region, lattice, row, col, rotation = key
    m: Optional[torch.Tensor] = scores.get((rotation, region, lattice))
    if m is None or m.numel() == 0 or not (0 <= row < m.shape[0] and 0 <= col < m.shape[1]):
        return None
    return float(m[row, col])


def rank_among(scores: Maps, score: float, lattices: Sequence[str]) -> int:
    """1 + the windows of `lattices`, every region and rotation, scoring STRICTLY above `score`: a tie goes
    to the window ranked."""
    return 1 + sum(int((m > score).sum()) for (_, _, lattice), m in scores.items()
                   if lattice in lattices and m.numel())


def _scored(scores: Maps, found: Answers) -> Answers:
    """`found` without an offset answer when no offset window was scored (a main-only phase)."""
    return found if any(k[2] == 'offset' for k in scores) else found.main_only()


def ranked_ranks(scores: Maps, found: Answers, pools: Pools) -> Dict[str, int]:
    """The ranks of the RANKED mode (no suppression, every window): the main answer among the main windows
    (`rank_main`) and among all (`rank_main_in_all`); with an offset answer the same of it (`rank_offset_in_offset`,
    `rank_offset_in_all`). An answer no map holds has the miss value, its pool + 1."""
    def rank(a: Answer, among: Sequence[str], miss: int) -> int:
        s: Optional[float] = answer_score(scores, a.key)
        return miss if s is None else min(miss, rank_among(scores, s, among))

    found = _scored(scores, found)
    out: Dict[str, int] = {'rank_main': rank(found.main, ('main',), pools.main + 1),
                           'rank_main_in_all': rank(found.main, ANSWER_LATTICES, pools.all + 1)}
    if found.offset is not None:
        out.update(rank_offset_in_offset=rank(found.offset, ('offset',), pools.offset + 1),
                   rank_offset_in_all=rank(found.offset, ANSWER_LATTICES, pools.all + 1))
    return out


#: A listed window as the list hands it on: its identity and the level-0 point a read of it starts at (its
#: lattice origin plus any shift a refining phase gave it).
Listed = Tuple[Key, Tuple[int, int]]


def listed_windows(cs: CandidateSet) -> List[Listed]:
    return [(c.key(), cs.origin_l0(c)) for c in cs.candidates]


def listed_places(listed: Sequence[Listed], found: Answers, pools: Pools, tol_px: float) -> Dict[str, int]:
    """The ranks of the LISTED mode: each answer's PLACE in the list. The list was cut by a suppression that
    drops a window when a better one of its rotation lies within `tol_px` (level-0 px: `min_sep_tiles` tiles),
    so the answer's window may be gone, dropped for the one that kept its place: that window stands for it.
    An answer's place is the first window of the list that IS it, or has its rotation and starts nearer than
    `tol_px` to it; with none it is not listed, its pool + 1. `rank_main` is THE answer's (the main one);
    `rank_main_in_all` the same against the whole pool; the offset answer's two when there is one. With `tol_px`
    0 only the window itself counts."""
    def place(a: Answer) -> Optional[int]:
        ox, oy = a.origin
        for n, (key, (x, y)) in enumerate(listed, 1):
            if key == a.key or (key[4] == a.rotation and math.hypot(x - ox, y - oy) < tol_px):
                return n
        return None

    def capped(value: Optional[int], miss: int) -> int:
        return miss if value is None else min(value, miss)

    at_main: Optional[int] = place(found.main)
    out: Dict[str, int] = {'rank_main': capped(at_main, pools.main + 1),
                           'rank_main_in_all': capped(at_main, pools.all + 1)}
    if found.offset is not None:
        at_offset: Optional[int] = place(found.offset)
        out.update(rank_offset_in_offset=capped(at_offset, pools.offset + 1),
                   rank_offset_in_all=capped(at_offset, pools.all + 1))
    return out


def listed_of_rows(rows: Iterable[Mapping[str, Any]]) -> List[Listed]:
    """A stored list (a phase's `output` rows, in rank order, as read back) as `listed_places` reads it."""
    return [((int(float(r['region'])), str(r['lattice']), int(float(r['row'])), int(float(r['col'])),
              int(float(r['rotation']))), (int(float(r['x0'])), int(float(r['y0'])))) for r in rows]


def first_place(listed: Sequence[Listed], found: Answers, tol_px: float) -> Optional[int]:
    """The place of the better of the answers in a list (`listed_places`); None when it holds neither."""
    n: int = len(listed)
    places: Dict[str, int] = listed_places(listed, found, Pools(main=n, offset=n), tol_px)
    best: int = min(places['rank_main_in_all'], places.get('rank_offset_in_all', 2 * n + 1))
    return best if best <= n else None


def truth_row(found: Answers, scores: Optional[Maps]) -> Dict[str, Any]:
    """The `truth` row of a FoV: each answer's window, level-0 box and distance (`truth_<lattice>_*`); with the
    first phase's `scores` (it searched the photo's own level), each scored answer's score, the ranked ranks
    (`truth_` + `ranked_ranks`) and the pools (`pool_main`, `pool_all`; the offset pool is the difference)."""
    row: Dict[str, Any] = dict(truth_rotation=found.main.rotation)
    for a in found:
        p: str = f'truth_{a.lattice}_'
        x0, y0, w0, h0 = a.box
        row.update({f'{p}region': a.region, f'{p}row': a.row, f'{p}col': a.col, f'{p}x0': x0, f'{p}y0': y0,
                    f'{p}w0': w0, f'{p}h0': h0, f'{p}dist_l0': a.dist_l0})
        s: Optional[float] = answer_score(scores, a.key) if scores else None
        if s is not None:
            row[f'{p}score'] = s
    if scores:
        pools: Pools = pools_of(scores)
        row.update({f'truth_{k}': v for k, v in ranked_ranks(scores, found, pools).items()})
        row.update(pool_main=pools.main, pool_all=pools.all)
    return row


def truth_rows(out: RetrievalOutput, img: np.ndarray, qc: QueryPatchContainer, truth: Truth, tile: int,
               overlap: bool, frame_of: Optional[Callable[[int], CandidateSet]] = None
               ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """`(truth row, truth tile rows)` of one phase's output: the FoV's answers (`answers`, on the lattices of the
    photo's own level) as `truth_row` writes them, scored and ranked when the phase scored every window there
    (`window_scores`), and, when it offers per-tile cosines (`tile_sims`), each scored answer's, by its
    `lattice`. A phase that searched another level scored no answer: the windows alone, in the frame of the
    photo's level (`frame_of`, the retriever's `frame`), and without one the rotation alone."""
    cs: CandidateSet = out.candidates
    scored: bool = 'window_scores' in out.extras and cs.level == truth.level
    frame: Optional[CandidateSet] = cs if cs.level == truth.level else None
    if frame is None and frame_of is not None:
        try:
            frame = frame_of(truth.level)
        except Exception:                                           # noqa: BLE001
            frame = None                  # a level the retriever cannot read: no lattice to place them on
    if frame is None:
        return dict(truth_rotation=truth.rot), []
    found: Optional[Answers] = answers(frame, query_dims(img, tile, overlap)[truth.rot], truth.centre(tile),
                                       truth.rot, ANSWER_LATTICES if overlap else ('main',))
    if found is None:
        if scored:
            raise ValueError('no window at this level holds the query')
        return dict(truth_rotation=truth.rot), []
    scores: Optional[Maps] = out.extras['window_scores'].value if scored else None
    tiles: List[Dict[str, Any]] = []
    if scores and 'tile_sims' in out.extras:
        from stage2_retrieval.RetrieverSlidingWinSimRot import tile_sim_rows   # noqa: PLC0415
        for a in _scored(scores, found):
            window: Candidate = Candidate(a.region, a.lattice, a.row, a.col, a.rotation, 0.0)
            # the window's own tiles: the retriever's table without the rank column
            tiles += [dict(lattice=a.lattice, **{k: v for k, v in r.items() if k != 'rank'})
                      for r in tile_sim_rows(out.extras['tile_sims'].value, [window])]
    return truth_row(found, scores), tiles


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


def _number(text: str, default: float = 0.0) -> float:
    return float(text) if text not in ('', None) else default


def stored_candidates(rows: Sequence[Dict[str, str]]) -> List[Candidate]:
    """The candidates of stored `output` rows, in rank order. A row a reranker
    wrote carries `first_score` / `first_rank` (and `dx` / `dy` when it moved
    the window): it comes back as the kind of candidate that holds them, so
    stage 3 reads the same position the rows were written for. What only a
    method's own table needs (votes, inliers) is not read back."""
    out: List[Candidate] = []
    for r in sorted(rows, key=lambda r: int(r['rank'])):
        base: Tuple[int, str, int, int, int, float] = (
            int(r['region']), r['lattice'], int(r['row']), int(r['col']),
            int(r['rotation']), float(r['score']))
        first: Dict[str, Any] = dict(first_score=_number(r.get('first_score', '')),
                                     first_rank=int(_number(r.get('first_rank', ''))))
        if r.get('dx', '') != '' or r.get('dy', '') != '':
            out.append(ShiftedCandidate(*base, **first, shift=(
                int(_number(r.get('dx', ''))), int(_number(r.get('dy', ''))))))
        elif r.get('first_rank', '') != '':
            out.append(RerankedCandidate(*base, **first))
        else:
            out.append(Candidate(*base))
    return out


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
        again: CandidateSet = pl.stage2(img, est).candidates
    except Exception:                                               # noqa: BLE001
        return None
    return again if len(again) else None


# ── one slide ────────────────────────────────────────────────────────────────

#: The tables of each stage's entry. Stage 2's are every table a phase's entry can hold; which of them one
#: holds is that phase's (`phase_roles`).
ROLES = {1: ('output', 'neighbours', 'probs', 'votes', 'prototypes_index'),
         2: ('output', 'tile_sims', 'truth', 'truth_sim', 'rerank_scores'),
         3: ('output', 'matches')}


def roles_of(stages: Tuple[Stage, Stage, Stage]) -> Dict[int, Tuple[str, ...]]:
    """`ROLES`: the same for every method of stages 1 and 3 (what a method does not offer is an empty
    table); stage 2's are each phase's own (`phase_roles`)."""
    return dict(ROLES)


def phase_roles(cls: type, first: bool, truth: bool) -> Tuple[str, ...]:
    """The tables of a stage-2 phase's entry: its list, the tables its retriever offers (`Retriever.TABLES`),
    and with `truth` (a synthetic FoV), on a first phase -- the one that scores every window -- the answers
    (`truth`) and, when it offers per-tile cosines, theirs (`truth_sim`). A reranker scores its list alone, so
    it has no truth of its own: its FoVs' answers are its chain's first phase's. What a phase does not offer
    is not written, and a hit does not ask for it."""
    offered: Tuple[str, ...] = tuple(getattr(cls, 'TABLES', ()))
    roles: Tuple[str, ...] = ('output', *offered)
    if truth and first:
        roles += ('truth',) + (('truth_sim',) if 'tile_sims' in offered else ())
    return roles


#: The bench's own code in the stage-2 entry: how the truth window is placed
#: (`query_grid_centre`, `matching_rotation`, `truth_rows`). No stage class
#: owns it, so a change to it made to be compared is bumped here; a bug fix is not (ConfigIdentity rule 3).
#: 4: both answers (main and offset), each with its window, box, distance, score and ranks; per-tile
#: cosines of both; no truth in a reranker's entry.
TRUTH_VERSION = 4


def stage1_record(s1: Stage, base: Cache.Address, limit: int = 0) -> Dict[str, Any]:
    """The record of stage 1's entry under `base`, the address of the photos it
    saw (a `render=` of synthetic FoVs, or a `photos=` folder of real ones): the
    method's config and code, the photos, the vote rules. Which tables the
    entry holds is its content, not its identity (`status`'s `roles`)."""
    levels: Dict[str, str] = dict(base.levels)
    upstream: Dict[str, str] = ({'render': levels['render']} if 'render' in levels
                                else {'photos': levels['photos']})
    return dict(s1.record(**upstream), rules=list(RULES), limit=int(limit))


def stage2_entries(base: Cache.Address, stages: Tuple[Stage, Stage, Stage], route: str, mask_cfg: Any,
                   limit: int = 0, truth: bool = True) -> List[tuple]:
    """`[(entry, id, record)]` of every phase of stage 2's chain on `route`, first phase first: all in the
    route's `stage2` entry, each under its own name (`Stage.phases`). The last is the list stage 3 reads."""
    s1, s2, _ = stages
    up: str = s1.id if route == 'stage1' else ORACLE
    entry: Any = base.at(stage1=up).entry('stage2')
    extra: Dict[str, Any] = dict({'truth': TRUTH_VERSION} if truth else {}, limit=int(limit))
    return [(entry, name, dict(s2.phase_record(i, stage1=up, seg=mask_cfg.seg_id(), region=mask_cfg.region_id()),
                               **extra))
            for i, (name, _) in enumerate(s2.phases)]


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
        2: stage2_entries(base, stages, route, mask_cfg, limit, truth)[-1],
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

    `roles` are the tables stages 1 and 3 keep (`ROLES`, or more for a shot
    source of its own); stage 2's are each phase's (`phase_roles`). `truth`
    is whether stage 2 places the answers.
    `shots` is not started before something has to run. `limit` stops after N
    shots (a smoke run; recorded in the entries). Hooks: `before()` once
    something is to run, before the models load; `after_stage3(shot, route,
    cs, rs, tables)` after each shot's stage 3 -- `cs` and `rs` None where
    there was nothing to run it on -- for a table of the caller's own;
    `describe(shot, route, cs, rs)` a string to add to the progress line."""
    s1, s2, s3 = stages
    t1: Optional[Tables] = None
    #: stage 2, per route: the phases to write, by phase
    t2: Dict[str, Dict[int, Tables]] = {}
    t3: Dict[str, Tables] = {}
    stored1: Dict[int, List[Dict[str, str]]] = {}
    #: stage 2, per route: the rows of the phases whose entries hit, by phase (a chain runs on from them)
    stored2: Dict[str, Dict[int, Dict[int, List[Dict[str, str]]]]] = {}
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
        e3, sid3, rec3 = ents[3]
        stage3_runs: bool = status(e3, sid3, rec3, roles[3]) != 'hit'
        missing: Dict[int, Tables] = {}
        hits: Dict[int, Dict[int, List[Dict[str, str]]]] = {}
        for i, (e, sid, rec) in enumerate(stage2_entries(base, stages, route, masks.cfg, limit, truth)):
            held: Tuple[str, ...] = phase_roles(stage2_retrieval.retriever_class(s2.cfg.phases[i]), i == 0, truth)
            if status(e, sid, rec, held) != 'hit':
                missing[i] = Tables(e, sid, rec, held)
            else:
                hits[i] = by_index(read_rows(e.path('output', sid, '.csv')))
        if missing:
            t2[route] = missing
        if missing or stage3_runs:
            stored2[route] = hits
        if stage3_runs:
            t3[route] = Tables(e3, sid3, rec3, roles[3])
    if t1 is None and not t2 and not t3:
        print('  every stage hit -- nothing to run', flush=True)
        return False
    print('  to run: ' + ' '.join(
        ([f'stage1'] if t1 else []) + [f'stage2[{r}: phase {",".join(str(i) for i in t2[r])}]' for r in t2]
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
                    for tab in t2[route].values():
                        tab.add('output', shot.index, [dict(error='stage 1 failed')])
                else:
                    got = run_stage2(pl, shot.img, est, shot.truth, t2[route],
                                     stored2.get(route, {}), shot.index)
            d2[route] = time.perf_counter() - t
            clock['stage2'] += d2[route]
            beat.doing(f'photo {shot.index}, stage 3 [{route} route]')
            t = time.perf_counter()
            rs: Optional[LocalizationResultSet] = None
            if route in t3:
                last: int = len(s2.phases) - 1
                if got is None and route not in t2 and last in stored2.get(route, {}):
                    got = stage2_for_stage3(pl, shot.img, est,
                                            stored2[route][last].get(shot.index, []))
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
    for tables in [t1, *(tab for by_phase in t2.values() for tab in by_phase.values()), *t3.values()]:
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
    ap.add_argument('--datasets', nargs='+', default=['@legacy-val'],
                    help='AccessDatasets ids: a registered dataset or a virtual one, `@<name>` (`@legacy-val`, '
                         'where methods are chosen; `@legacy-test`, where the choice is confirmed)')
    ap.add_argument('--n-wsi', type=int, default=10,
                    help='slides of each dataset a --datasets id draws from: the first of its own order')
    ap.add_argument('--stage1', default='knn:gigapath', help='<method>:<recipe>')
    ap.add_argument('--stage2', default='slidewin:gigapath', help='<method>:<recipe>')
    ap.add_argument('--stage3', default='sift:default', help='<method>:<recipe>')
    ap.add_argument('--route', choices=['stage1', 'oracle', 'both'], default='both',
                    help="the level stage 2 searches: stage 1's, the level the "
                         'FoV was placed at, or each in turn')
    # --fov, --max-ds, --sampler-*, --camera-*
    add_fov_args(ap)
    ap.add_argument('--fov-mask-cache-job', default='Masks',
                    help='hest masks the FoVs are placed on')
    # --seg: the mask stages 1 and 2 search
    add_mask_args(ap)
    ap.add_argument('--mask-cache-job', default='Masks',
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
    c2, k2 = as_chain(c2, k2)
    if batch_size:
        c2 = replace(c2, phases=tuple(
            replace(p, encoder=replace(p.encoder, batch_size=batch_size)) if hasattr(p, 'encoder') else p
            for p in c2.phases))
    m3, n3, c3, k3 = parse_stage(args, 3)
    s2 = Stage(2, m2, stage_name(m2, c2, n2), c2, k2,
               dict(device=device, multi_gpu=multi_gpu, read_workers=read_workers),
               also_id=(mask_cfg.seg_id(), mask_cfg.region_id()))
    s3 = Stage(3, m3, n3, c3, k3, dict(device=device))   # the matching runs on a CUDA device
    routes = ['stage1', ORACLE] if args.route == 'both' else [args.route]
    return (s1, s2, s3), routes, mask_cfg


def run_slides(args) -> List[tuple]:
    """`(dataset, name, path)` of every slide of the run: of each `--datasets` id
    (`AccessDatasets.by_source`), the first `--n-wsi` slides of each dataset it
    draws from, in its own order -- a virtual dataset is shuffled when it is drawn,
    so a prefix is a random pick. `dataset` is the one the slide is registered in."""
    out = []
    for dataset_id in args.datasets:
        for dataset, names in by_source(dataset_id, args.n_wsi):
            if len(names) < args.n_wsi:
                raise ValueError(f'n_wsi {args.n_wsi} but {dataset_id} holds {len(names)} slide(s) of '
                                 f'{dataset}')
            out += [(dataset, name, str(locate(name, dataset=dataset).path)) for name in names]
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
