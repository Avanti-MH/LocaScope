#!/usr/bin/env python3
"""Figures, statistics and the demo export of a bench_locascope run, offline:
everything is read from the cache it wrote, nothing runs a stage.

    python utilities/cli/plot/plot_locascope.py <bench_locascope's run flags> \\
        [--stage-cache-job BenchLocaScope] \\
        [--select "level == 0 and s2_hit_rank > 1"] [--sample 20 --seed 0] \\
        [--panels photo,gt,candidates,crop,matches,located,result] \\
        [--plot recall cdf:s3_rank1_err_um confusion scatter:s1_mpp_err_rel,s3_rank1_err_um] \\
        [--by level] [--export demo] [--columns]

THE JOIN TABLE: one row per (FoV, route). The run flags are bench_locascope's
own (`add_run_args`) and give the same addresses, so the same slides, draws,
renders and stage entries are found, never searched for. A row is the render
row (ground truth: level, ds, the level-0 rectangle and centre, rot_deg,
effective_mpp, every photometric draw) joined on `index` with each stage's
`output` -- prefixed `s1_`, `s2_`, `s3_` -- and stage 2's `truth`, and these
computed from them:

    s1_mpp_err_rel        |estimated - effective| / effective
    s1_level_err          chosen level - placed level
    s2_err_um             rank 1's window centre to the truth centre
    s2_rot_ok             rank 1's rotation sets the photo upright (`matching_rotation`)
    s2_hit_rank           the truth window's place in stage 2's output: the
                          first rank at the upright rotation whose centre is
                          within stage 3's search (padding tiles at the
                          FoV's own ds) of the truth window's centre; empty
                          when stage 2 searched another level or no window
                          is that close
    s2_hit_rank_strict    the same within one tile
    s3_rank1_err_um       rank 1's stage-3 centre to the truth centre
    s3_verified_err_um    the first accepted rank's, likewise
    s3_hit_rank           first verified rank whose centre is within the search
    s3_ok, s3_verified_ok error under --tol-um

`--select` is a pandas query on it, `--columns` lists the columns. The table
is written whole (`joined.csv`) and as selected (`selected.csv`).

REAL PHOTOS (--real): the cache of a locate_photo run (job RealTest) instead of
a bench's. A real photo has no truth, so everything that needs one is left out:
the `gt` panel, the truth footprint and "GT = #n" on the others, `windows`'s
truth window, the statistics (--plot) and the demo export. What is shown is
stage 1's neighbours, stage 2's candidates, stage 3's matches and location, and
the ANSWER (the verified candidate with the largest confidence) marked on
`result`. `crop`, `matches` and `located` show the answer's rank unless --rank
names one. The joined table's columns are s1_*, s2_*, s3_* and answer_*, so
--select reads e.g. "answer_confidence < 0.2" or "answer_retrieval_only == 1".

    python utilities/cli/plot/plot_locascope.py --real --stage-cache-job RealTest \\
        --stage3-topk 100 --sample 20 --panels photo,candidates,matches,located,result

PANELS, one row of them per selected FoV: photo, gt (the truth's footprint),
candidates (the first --k-boxes windows and the truth), result (stage 3's
centres) -- these three on one overview read from the slide around the truth
and the boxes -- stage1 (the KNN neighbours' levels), crop (stage 3's crop at
--rank), matches (its point pairs; inliers green, outliers red), located (a
column of three: the query, the place stage 3 found for it -- the crop around
its footprint, outlined -- and a checkerboard of that crop against the query
warped onto it by the homography; where the two agree the board is seamless).
Every coloured mark has a legend.

`windows` is a figure of its own, the demo page's comparison: the truth
window and the first --k-windows candidates, each read from the slide at the
routed level, cut into tiles darkened by their cosine to the query tile over
them and labelled with it, under the photo turned to that window's rotation.

`--export demo` writes the same comparison as the demo page reads it
(`demo_page/index.html` + `data.js` + `img/`, in <out>/demo/): open
index.html from disk. One route (--demo-route), one arm (`cls+mean`).

Output: --out, default result/<SLURM_JOB_NAME or PlotLocaScope>/.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # utilities/
import _paths                                                       # noqa: E402
_paths.setup_import_paths()
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'bench_modules'))

import matplotlib                                                   # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                     # noqa: E402
from matplotlib.lines import Line2D                                 # noqa: E402
from matplotlib.patches import Patch, Polygon, Rectangle            # noqa: E402

import Cache                                                        # noqa: E402
from _paths import job_result_dir                                   # noqa: E402
from AccessDatasets import list_names, locate                       # noqa: E402
from BenchCommon import (ORACLE, matching_rotation, add_run_args, parse_run_args,  # noqa: E402
                             run_from_args, slide_supplies, stage_entries)
from SafeSlide import SafeSlide                                     # noqa: E402
from TissueMaskConfig import MASK_RECIPES, MaskMaker                # noqa: E402

PANELS = ('photo', 'gt', 'candidates', 'stage1', 'crop', 'matches', 'located', 'result',
          'windows')


# ── the join table ───────────────────────────────────────────────────────────

def _read(entry, role: str, sid: str) -> Optional[pd.DataFrame]:
    path = entry.path(role, sid, '.csv')
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    return df if len(df.columns) else None


_ROLES_READ: Tuple[str, ...] = ('output', 'neighbours', 'tile_sims', 'truth', 'truth_sim',
                                'matches', 'answer')


class _Slide:
    """What a slide's runs share: the slide, opened when first read, and its
    thumbnail."""
    path: str
    real: bool = False
    _wsi: Optional[SafeSlide] = None
    _thumb: Optional[Tuple[np.ndarray, float]] = None

    @property
    def wsi(self) -> SafeSlide:
        if self._wsi is None:
            self._wsi = SafeSlide(self.path)
        return self._wsi

    def thumb(self, side: int = 1600):
        if self._thumb is None:
            img = self.wsi.get_thumbnail((side, side))
            self._thumb = (np.asarray(img.convert('RGB')),
                           img.size[0] / self.wsi.dimensions[0])
        return self._thumb


class SlideRun(_Slide):
    """One slide's render rows and stage tables for every route."""

    def __init__(self, dataset, name, path, supply, stages, routes, mask_cfg,
                 job, limit):
        self.dataset, self.name, self.path, self.supply = dataset, name, path, supply
        self.base = supply.render_address.on(job)
        self.tables: Dict[str, Dict[int, Dict[str, pd.DataFrame]]] = {}
        for route in routes:
            got = {}
            for n, (entry, sid, rec) in stage_entries(
                    self.base, stages, route, mask_cfg, limit).items():
                state, diff = entry.status(sid, rec)
                if state != 'hit':
                    print(f'  {name} {route} stage {n}: {state} '
                          f'{"; ".join(diff)}', flush=True)
                    continue
                got[n] = {role: _read(entry, role, sid) for role in _ROLES_READ}
            self.tables[route] = got
        self.rid = supply.microscope.cfg.identity_id()
        self.metas = [s.meta for s in supply.sampler]
        self._wsi = None
        self._thumb = None

    def render_rows(self, indices) -> pd.DataFrame:
        """The render row of each index: the render CSV when the run wrote one
        (a full run), else the geometry and the photo's parameters again --
        the photo is the same at any time (`FovSupply.photo`)."""
        csv_path = self.supply.render_entry.path('render', self.rid, '.csv')
        if csv_path.is_file():
            return pd.read_csv(csv_path)
        rows = []
        for i in sorted(indices):
            _, params = self.supply.photo(self.metas[i])
            rows.append({'index': i, **params, **self.supply.geometry(self.metas[i])})
        return pd.DataFrame(rows)

    def photo(self, index: int) -> np.ndarray:
        photos = self.supply.render_entry.path('photos', self.rid)
        p = photos / f'{index}.png'
        if p.is_file():
            from PIL import Image
            return np.asarray(Image.open(p).convert('RGB'))
        return self.supply.photo(self.metas[index])[0]


#: The dataset a real run (locate_photo) locates the photos of.
DATASET_REAL: str = 'ki67_with_photo'


class RealRun(_Slide):
    """One slide's real photos: the stage tables a locate_photo run wrote under
    `photos=<id>`, and the photo files `shots` names. No truth, no render."""
    real: bool = True

    def __init__(self, name: str, entry: Any, addr: Cache.Address, stages, mask_cfg,
                 limit: int) -> None:
        self.dataset: str = DATASET_REAL
        self.name: str = name
        self.path: str = str(entry.path)
        self.folder: str = entry.related['photos']
        shots: Optional[pd.DataFrame] = _read(addr.entry('shots'), 'index', 'files')
        self.shots: pd.DataFrame = shots if shots is not None else pd.DataFrame(
            columns=['index', 'photo', 'bytes', 'width', 'height'])
        got: Dict[int, Dict[str, Optional[pd.DataFrame]]] = {}
        for n, (stage_entry, sid, rec) in stage_entries(
                addr, stages, 'stage1', mask_cfg, limit, truth=False).items():
            state, diff = stage_entry.status(sid, rec)
            if state != 'hit':
                print(f'  {name} stage {n}: {state} {"; ".join(diff)}', flush=True)
                continue
            got[n] = {role: _read(stage_entry, role, sid) for role in _ROLES_READ}
        self.tables: Dict[str, Dict[int, Dict[str, Optional[pd.DataFrame]]]] = {'stage1': got}

    def photo(self, index: int) -> np.ndarray:
        name: str = str(self.shots[self.shots['index'] == index].iloc[0]['photo'])
        return np.asarray(Image.open(Path(self.folder) / name).convert('RGB'))


def real_runs(args, stages, mask_cfg, job: str) -> List[RealRun]:
    """The slides of `DATASET_REAL` (or `--slides`) that have a locate_photo run
    in `job`'s cache. The photo set is whichever `photos=` the slide's region
    holds -- the newest, when a folder has changed and left two."""
    runs: List[RealRun] = []
    for name in list(args.slides) or list_names(dataset=DATASET_REAL):
        entry = locate(name, dataset=DATASET_REAL)
        if not entry.related.get('photos'):
            continue
        base: Cache.Address = Cache.Address(
            job, slide=Cache.wsi_stem_of(str(entry.path)), seg=mask_cfg.seg_id(),
            region=mask_cfg.region_id())
        ids: List[str] = base.children('photos')
        if not ids:
            print(f'  {name}: nothing in {job}\'s cache', flush=True)
            continue
        if len(ids) > 1:
            print(f'  {name}: {len(ids)} photo sets, using {ids[-1]}', flush=True)
        run: RealRun = RealRun(name, entry, base.at(photos=ids[-1]), stages, mask_cfg,
                               args.limit)
        if len(run.shots) and 3 in run.tables['stage1']:
            runs.append(run)
        else:
            print(f'  {name}: no finished run (shots or stage 3 missing)', flush=True)
    return runs


def join_real(runs: List[RealRun]) -> pd.DataFrame:
    """One row per photo: `s1_*` (stage 1's estimate), `s2_*` (stage 2's first
    candidate), `s3_*` (how many candidates stage 3 verified, how many fitted)
    and `answer_*` (the answer row of stage 3's entry). No truth columns."""
    out: List[Dict[str, Any]] = []
    for run in runs:
        got: Dict[int, Dict[str, Optional[pd.DataFrame]]] = run.tables['stage1']
        per: Dict[int, Dict[str, Optional[Dict[int, pd.DataFrame]]]] = {
            n: {role: (None if df is None else {int(i): g for i, g in df.groupby('index')})
                for role, df in t.items()} for n, t in got.items()}
        for _, shot in run.shots.iterrows():
            i: int = int(shot['index'])
            row: Dict[str, Any] = dict(dataset=run.dataset, slide=run.name, route='stage1',
                                       index=i, photo=shot['photo'], width=shot['width'],
                                       height=shot['height'])
            if 1 in per and per[1]['output'] and i in per[1]['output']:
                row.update({f's1_{k}': v for k, v in per[1]['output'][i].iloc[0].items()
                            if k != 'index'})
            if 2 in per and per[2]['output'] and i in per[2]['output']:
                cands: pd.DataFrame = per[2]['output'][i]
                if 'rank' in cands:
                    cands = cands.sort_values('rank', na_position='last')
                first = cands.iloc[0]
                row.update({f's2_{k}': v for k, v in first.items() if k != 'index'})
                row['s2_routed_level'] = first.get('level')
            if 3 in per and per[3]['output'] and i in per[3]['output']:
                ranks: pd.DataFrame = _ranked(per[3]['output'][i])
                row['s3_verified'] = len(ranks)
                row['s3_fitted'] = (int(ranks['success'].isin([True, 'True']).sum())
                                    if 'success' in ranks else 0)
            if 3 in per and per[3]['answer'] and i in per[3]['answer']:
                row.update({f'answer_{k}': v for k, v in per[3]['answer'][i].iloc[0].items()
                            if k != 'index'})
            out.append(row)
    return pd.DataFrame(out)


def real_summary(df: pd.DataFrame) -> str:
    """Per slide: how many photos, the confidence's median and how many reach
    0.5, and how many have none (the position is the retrieval's)."""
    lines: List[str] = []
    for slide, g in df.groupby('slide'):
        if 'answer_confidence' not in g:
            lines.append(f'{slide:<26} n={len(g):<5} no answers')
            continue
        conf: pd.Series = pd.to_numeric(g['answer_confidence'], errors='coerce')
        lines.append(f'{slide:<26} n={len(g):<5} conf median {conf.median():.2f}'
                     f'  >= 0.5 {(conf >= 0.5).mean():6.1%}'
                     f'  retrieval only {(conf == 0).mean():6.1%}')
    return '\n'.join(lines)


def _first_rank(df: Optional[pd.DataFrame], cx, cy, tol, xcol='x0', ycol='y0',
                centre=False):
    if df is None or not len(df):
        return None
    if centre:
        dx, dy = df['center_x0'] - cx, df['center_y0'] - cy
    else:
        dx = df[xcol] + df['w0'] / 2.0 - cx
        dy = df[ycol] + df['h0'] / 2.0 - cy
    hit = df[np.hypot(dx, dy) <= tol]
    return int(hit['rank'].min()) if len(hit) else None


def join(runs: List[SlideRun], stages, tol_um: float) -> pd.DataFrame:
    tile = stages[1].cfg.tile_size
    padding = stages[2].cfg.padding
    out = []
    for run in runs:
        base_mpp = run.wsi.base_mpp
        indices = set()
        for got in run.tables.values():
            for t in got.values():
                if t['output'] is not None:
                    indices |= set(t['output']['index'])
        if not indices:
            continue
        render = run.render_rows(indices).set_index('index')
        for route, got in run.tables.items():
            per = {n: {role: (None if df is None else
                              {i: g for i, g in df.groupby('index')})
                       for role, df in t.items()} for n, t in got.items()}
            for i in sorted(indices):
                if i not in render.index:
                    continue
                g = render.loc[i]
                row = dict(dataset=run.dataset, slide=run.name, route=route,
                           index=int(i), base_mpp=base_mpp, **g.to_dict())
                cx, cy = float(g['center_x0']), float(g['center_y0'])
                eff = float(g['effective_mpp'])
                if 1 in per and per[1]['output'] and i in per[1]['output']:
                    o = per[1]['output'][i].iloc[0]
                    row.update({f's1_{k}': v for k, v in o.items() if k != 'index'})
                    if pd.notna(o.get('estimated_mpp')):
                        row['s1_mpp_err_rel'] = abs(o['estimated_mpp'] - eff) / eff
                        row['s1_level_err'] = int(o['chosen_level']) - int(g['level'])
                tol_l0 = padding * tile * float(g['ds'])
                if 2 in per and per[2]['output'] and i in per[2]['output']:
                    # the CandidateSet: rank 1 names the FoV's stage 2 answer
                    cands = per[2]['output'][i]
                    if 'rank' in cands:
                        cands = cands.sort_values('rank', na_position='last')
                    o = cands.iloc[0]
                    row.update({f's2_{k}': v for k, v in o.items() if k != 'index'})
                    row['s2_routed_level'] = o.get('level')
                    t = (per[2]['truth'] or {}).get(i)
                    t = None if t is None else t.iloc[0]
                    if t is not None:
                        row.update({k: v for k, v in t.items() if k != 'index'})
                    if pd.notna(o.get('x0')):
                        d = math.hypot(o['x0'] + o['w0'] / 2 - cx,
                                       o['y0'] + o['h0'] / 2 - cy)
                        row['s2_err_um'] = d * base_mpp
                        up = matching_rotation(g['rot_deg'])
                        row['s2_rot_ok'] = int(o['rotation']) == up
                        if (int(o['level']) == int(g['level']) and t is not None
                                and pd.notna(t.get('truth_x0'))):
                            tx = t['truth_x0'] + t['truth_w0'] / 2.0
                            ty = t['truth_y0'] + t['truth_h0'] / 2.0
                            same = cands[cands['rotation'].astype(int) == up]
                            row['s2_hit_rank'] = _first_rank(same, tx, ty, tol_l0)
                            row['s2_hit_rank_strict'] = _first_rank(
                                same, tx, ty, tile * float(g['ds']))
                if 3 in per and per[3]['output'] and i in per[3]['output']:
                    # one SiftRansacResult per verified rank
                    ranks = per[3]['output'][i]
                    if 'rank' in ranks:
                        ranks = ranks.sort_values('rank', na_position='last')
                    row['s3_error'] = ranks.iloc[0].get('error')
                    row['s3_t_s'] = ranks.iloc[0].get('t_s')
                    keep = ('success', 'n_good', 'n_inliers', 'x0', 'y0',
                            'center_x0', 'center_y0')
                    picks = {'rank1': ranks[ranks['rank'] == 1] if 'rank' in ranks else ranks[:0]}
                    ok = ranks[ranks['success'].isin([True, 'True'])] if 'success' in ranks else ranks[:0]
                    picks['verified'] = ok.head(1)
                    row['s3_verified_rank'] = int(ok.iloc[0]['rank']) if len(ok) else None
                    for pick, sel_ in picks.items():
                        if not len(sel_):
                            continue
                        r = sel_.iloc[0]
                        row.update({f's3_{pick}_{k}': r.get(k) for k in keep})
                        if pd.notna(r.get('center_x0')):
                            row[f's3_{pick}_err_um'] = math.hypot(
                                r['center_x0'] - cx, r['center_y0'] - cy) * base_mpp
                    row['s3_ok'] = bool(row.get('s3_rank1_err_um', np.inf) <= tol_um
                                        and row.get('s3_rank1_success') in (True, 'True'))
                    row['s3_verified_ok'] = bool(
                        row.get('s3_verified_err_um', np.inf) <= tol_um)
                    if tol_l0 is not None and 'center_x0' in ranks:
                        row['s3_hit_rank'] = _first_rank(ranks, cx, cy, tol_l0,
                                                         centre=True)
                out.append(row)
    return pd.DataFrame(out)


# ── statistics ───────────────────────────────────────────────────────────────

def _groups(df, by):
    if not by:
        return [('all', df)]
    return [(f'{by}={k}', g) for k, g in df.groupby(by)]


def plot_stats(df: pd.DataFrame, specs, by, out: Path, dpi: int) -> None:
    for spec in specs:
        kind, _, arg = spec.partition(':')
        fig, ax = plt.subplots(figsize=(7, 5))
        if kind == 'cdf':
            for label, g in _groups(df, by):
                v = np.sort(pd.to_numeric(g[arg], errors='coerce').dropna().values)
                if len(v):
                    ax.step(v, np.arange(1, len(v) + 1) / len(g), where='post',
                            label=f'{label} (n={len(g)})')
            ax.set_xscale('log')
            ax.set_xlabel(arg)
            ax.set_ylabel('fraction of FoVs')
        elif kind == 'recall':
            col = arg or 's2_hit_rank'
            k_max = int(np.nanmax(pd.to_numeric(df[col], errors='coerce'))) if \
                df[col].notna().any() else 1
            ks = np.arange(1, max(k_max, 1) + 1)
            for label, g in _groups(df, by):
                r = pd.to_numeric(g[col], errors='coerce')
                ax.plot(ks, [(r <= k).mean() for k in ks], label=f'{label} (n={len(g)})')
            ax.set_xscale('log')
            ax.set_xlabel('K')
            ax.set_ylabel(f'recall@K ({col})')
        elif kind == 'confusion':
            ct = pd.crosstab(df['level'], df['s1_chosen_level'])
            ax.imshow(ct.values, cmap='Blues')
            for (r, c), v in np.ndenumerate(ct.values):
                ax.text(c, r, str(v), ha='center', va='center')
            ax.set_xticks(range(len(ct.columns)), ct.columns)
            ax.set_yticks(range(len(ct.index)), ct.index)
            ax.set_xlabel('stage 1 chosen level')
            ax.set_ylabel('placed level')
        elif kind == 'scatter':
            x, y = arg.split(',')
            for label, g in _groups(df, by):
                ax.scatter(g[x], g[y], s=8, label=label)
            ax.set_xlabel(x)
            ax.set_ylabel(y)
        else:
            raise SystemExit(f'--plot {spec!r}: kinds are cdf:<col>, '
                             f'recall[:<col>], confusion, scatter:<x>,<y>')
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
        name = spec.replace(':', '_').replace(',', '-')
        p = out / f'{name}{"_by-" + by if by else ""}.png'
        fig.tight_layout()
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        print(f'  [plot] {p}', flush=True)


def summary(df: pd.DataFrame, by) -> str:
    lines = []
    for label, g in _groups(df, by):
        for route, r in g.groupby('route'):
            n = len(r)
            line = f'{label:<14} {route:<7} n={n:<5}'
            if 's1_level_err' in r:
                line += f' s1 level ok {(r["s1_level_err"] == 0).mean():6.1%}'
            if 's2_hit_rank' in r:
                h = pd.to_numeric(r['s2_hit_rank'], errors='coerce')
                line += ''.join(f'  R@{k} {(h <= k).mean():6.1%}' for k in (1, 10, 100))
            if 's3_ok' in r:
                line += (f'  s3 ok {r["s3_ok"].mean():6.1%}'
                         f'  verified ok {r["s3_verified_ok"].mean():6.1%}')
            lines.append(line)
    return '\n'.join(lines)


# ── panels ───────────────────────────────────────────────────────────────────

def _quad(ax, cx, cy, w, h, deg, scale, off=(0.0, 0.0), **kw):
    """The photo's footprint: a `w` x `h` level-0 rectangle about (cx, cy)
    turned by `deg`, drawn in a view whose level-0 origin is `off` and whose
    px per level-0 px is `scale`."""
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    pts = [((cx - off[0] + dx * c - dy * s) * scale,
            (cy - off[1] + dx * s + dy * c) * scale)
           for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2),
                          (-w / 2, h / 2))]
    ax.add_patch(Polygon(pts, closed=True, fill=False, **kw))


def _read_box(run: SlideRun, x0, y0, w0, h0, level: int) -> np.ndarray:
    """The level-0 box (x0, y0, w0, h0) read at `level`, RGB."""
    ds = float(run.wsi.level_downsamples[level])
    size = (max(1, int(round(w0 / ds))), max(1, int(round(h0 / ds))))
    return np.asarray(run.wsi.read_region((int(x0), int(y0)), level, size).convert('RGB'))


def _overview(run: SlideRun, boxes, side: int = 1200):
    """`(image, x0, y0, scale)`: the slide around every level-0 box in
    `boxes` -- their union with a margin of one box -- read at the finest
    level that keeps the longer side near `side` px. `scale` is image px per
    level-0 px."""
    xs0 = min(b[0] for b in boxes)
    ys0 = min(b[1] for b in boxes)
    xs1 = max(b[0] + b[2] for b in boxes)
    ys1 = max(b[1] + b[3] for b in boxes)
    pad = max(max(b[2], b[3]) for b in boxes)
    W, H = run.wsi.dimensions
    x0, y0 = max(0, xs0 - pad), max(0, ys0 - pad)
    x1, y1 = min(W, xs1 + pad), min(H, ys1 + pad)
    want = max(x1 - x0, y1 - y0) / side
    lds = list(run.wsi.level_downsamples)
    level = max([i for i, d in enumerate(lds) if d <= max(want, 1.0)] or [0])
    img = _read_box(run, x0, y0, x1 - x0, y1 - y0, level)
    return img, x0, y0, img.shape[1] / max(x1 - x0, 1)


def _tile_grid(ax, shape, tile: int, cos=None, values: bool = True):
    """Lines between `tile` px tiles over an image of `shape`; with `cos`
    ([rows, cols]), each tile darkened by how far its cosine is below 1 and
    labelled with it -- the demo page's drawing."""
    h, w = shape[:2]
    rows = cos.shape[0] if cos is not None else h // tile
    cols = cos.shape[1] if cos is not None else w // tile
    if cos is None and (w > cols * tile or h > rows * tile):
        ax.add_patch(Rectangle((cols * tile, 0), w - cols * tile, h, color='white',
                               alpha=0.55, lw=0))
        ax.add_patch(Rectangle((0, rows * tile), cols * tile, h - rows * tile,
                               color='white', alpha=0.55, lw=0))
    tw, th = (w / cols, h / rows) if cos is not None else (tile, tile)
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(Rectangle((c * tw, r * th), tw, th, fill=False,
                                   edgecolor=(0.08, 0.08, 0.14, 0.6), lw=0.8))
            if cos is not None and np.isfinite(cos[r, c]):
                v = float(cos[r, c])
                ax.add_patch(Rectangle((c * tw, r * th), tw, th, lw=0,
                                       color=(0.06, 0.07, 0.14),
                                       alpha=max(0.0, min(0.65, (1 - v) * 0.9))))
                if values:
                    ax.text((c + 0.5) * tw, (r + 0.5) * th, f'{v:.2f}', color='white',
                            ha='center', va='center', fontsize=9, fontweight='bold')


def _place(row) -> str:
    """Where the truth window is in stage 2's output, as a label."""
    if pd.notna(row.get('s2_hit_rank')):
        return f'#{int(row["s2_hit_rank"])}'
    if pd.notna(row.get('s2_routed_level')) and int(row['s2_routed_level']) != int(row['level']):
        return f'not searched (L{int(row["s2_routed_level"])})'
    return 'not in output'


def _ranked(df: pd.DataFrame) -> pd.DataFrame:
    """The rows of a stage output that are results -- a row with an error
    and no rank is the FoV's failure, not a candidate."""
    return df[df['rank'].notna()] if 'rank' in df else df.iloc[:0]


def _cos_grid(tiles: pd.DataFrame) -> Optional[np.ndarray]:
    if tiles is None or not len(tiles):
        return None
    grid = np.full((int(tiles['q_row'].max()) + 1, int(tiles['q_col'].max()) + 1), np.nan)
    grid[tiles['q_row'].astype(int), tiles['q_col'].astype(int)] = tiles['cosine']
    return grid


def _windows(run: SlideRun, row: pd.Series, k: int):
    """`[(label, box, rotation, level, cosines, is_truth)]`: the truth window
    at the photo's own level, then stage 2's first `k` candidates at the
    level it searched, each with its per-tile cosines. A truth window stage 2
    did not search (sent to another level) has no rank, score or cosines."""
    got = run.tables.get(row['route'], {}).get(2)
    if not got:
        return []
    i = int(row['index'])
    out = []
    t = got.get('truth')
    t = t[t['index'] == i] if t is not None else None
    if t is not None and len(t) and 'truth_x0' in t and pd.notna(t.iloc[0]['truth_x0']):
        r = t.iloc[0]
        level = int(r['truth_level']) if pd.notna(r.get('truth_level')) else int(row['level'])
        box = (r['truth_x0'], r['truth_y0'], r['truth_w0'], r['truth_h0'])
        if pd.notna(r.get('truth_score')):
            tt = got.get('truth_sim')
            out.append((f'GT = {_place(row)}  {r["truth_score"]:.3f}',
                        box, int(r['truth_rotation']), level,
                        _cos_grid(tt[tt['index'] == i] if tt is not None else None), True))
        else:
            out.append((f'GT  L{level}  (not searched)', box,
                        int(r['truth_rotation']), level, None, True))
    c = got.get('output')
    c = None if c is None else _ranked(c)
    ts = got.get('tile_sims')
    if c is not None:
        for _, cand in c[c['index'] == i].sort_values('rank').head(k).iterrows():
            rank = int(cand['rank'])
            sims = (ts[(ts['index'] == i) & (ts['rank'] == rank)] if ts is not None
                    else None)
            out.append((f'#{rank}  {cand["score"]:.3f}',
                        (cand['x0'], cand['y0'], cand['w0'], cand['h0']),
                        int(cand['rotation']), int(cand['level']), _cos_grid(sims), False))
    return out


def draw_windows(run: SlideRun, row: pd.Series, args, out: Path) -> Optional[Path]:
    """The demo page's comparison, one column per window -- the truth window,
    then the first --k-windows candidates. Top: the photo turned to that
    window's rotation, as stage 2 matched it, cut into its tiles. Bottom: the
    window read from the slide at its level -- the truth at the photo's own,
    the candidates at the routed one -- each tile darkened by its cosine to
    the query tile over it and labelled with it."""
    wins = _windows(run, row, args.k_windows)
    if not wins or pd.isna(row.get('s2_routed_level')):
        return None
    i, route = int(row['index']), row['route']
    tile = args.tile_size
    level = int(row['s2_routed_level'])
    photo = run.photo(i)
    true_level: str = f'  true L{int(row["level"])}' if pd.notna(row.get('level')) else ''
    fig, axes = plt.subplots(2, len(wins), figsize=(3.6 * len(wins), 6.4), squeeze=False)
    for col, (label, box, rot, lvl, cos, truth) in enumerate(wins):
        q = np.rot90(photo, k=rot // 90)
        ax = axes[0, col]
        ax.imshow(q)
        _tile_grid(ax, q.shape, tile)
        ax.set_title(f'query  rot {rot}', fontsize=9)
        ax.axis('off')
        ax = axes[1, col]
        win = _read_box(run, *box, lvl)
        ax.imshow(win)
        _tile_grid(ax, win.shape, tile, cos=cos, values=args.tile_values)
        ax.set_title(label, fontsize=10, color='#1f9e5a' if truth else 'black')
        if truth:
            ax.add_patch(Rectangle((1, 1), win.shape[1] - 2, win.shape[0] - 2,
                                   fill=False, edgecolor='#1f9e5a', lw=3, ls='--'))
        ax.axis('off')
    fig.suptitle(f'{run.name}  #{i}  route {route}  searched L{level}{true_level}  darker tile = lower '
                 f'cosine to the query tile', fontsize=11)
    fig.tight_layout()
    p = out / f'{run.name}_{route}_{i}_windows.png'
    fig.savefig(p, dpi=args.dpi)
    plt.close(fig)
    return p


#: A legend under its axes, outside the image, so it never covers the picture.
_BELOW: Dict[str, Any] = dict(loc='upper center', bbox_to_anchor=(0.5, -0.01),
                              fontsize=7, framealpha=0.85)


def _homography(r: pd.Series) -> Optional[np.ndarray]:
    """The 3x3 H of a stage-3 `output` row (query px -> crop px), None when the
    row has none (the fit failed: stage 3 fell back to the candidate window)."""
    vals = [r.get(f'h{a}{b}') for a in range(3) for b in range(3)]
    if any(v is None or pd.isna(v) for v in vals):
        return None
    return np.array(vals, dtype=np.float64).reshape(3, 3)


def _footprint(shape, H: np.ndarray) -> np.ndarray:
    """Where the query's four corners land in the crop: `[4, 2]` px."""
    h, w = shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(corners, H).reshape(4, 2)


def _checker(crop: np.ndarray, query: np.ndarray, H: np.ndarray, n: int = 8) -> np.ndarray:
    """The crop with the query warped onto it (by H) shown in alternate squares
    of a board `n` across the footprint -- one image, the two sources taking
    turns, so a misfit shows as a break at every square's edge."""
    h, w = crop.shape[:2]
    warped = cv2.warpPerspective(query, H, (w, h))
    cover = cv2.warpPerspective(np.full(query.shape[:2], 255, np.uint8), H, (w, h)) > 0
    ys, xs = np.nonzero(cover)
    if not len(xs):
        return crop
    side = max(8, int(round(max(xs.max() - xs.min(), ys.max() - ys.min()) / n)))
    yy, xx = np.indices(cover.shape)
    board = ((yy // side) + (xx // side)) % 2 == 0
    out = crop.copy()
    out[cover & board] = warped[cover & board]
    return out


#: The most pixels `located` reads for the footprint: past this it takes a coarser level.
_LOCATED_MAX_PX: int = 16_000_000


def _located_view(run: SlideRun, r: pd.Series, H: np.ndarray, quad: np.ndarray,
                  ds_s: float, photo_shape) -> Tuple[np.ndarray, np.ndarray]:
    """`(image, H')`: the slide around the query's footprint, read at the query's OWN
    resolution, and the homography onto it.

    Stage 3 matches on a crop read at the level stage 2 searched, which can be much
    coarser than the photo, so the footprint there is a few hundred px and neither
    the outline nor the checkerboard can show whether fine detail lines up. Here the
    footprint plus a margin is read again at the finest level not finer than the
    photo (one query px = `s` crop px = `s * ds_s` level-0 px, `s` from the
    footprint's area), and `H` is carried over: crop px -> new px is a scale
    `k = ds_s / ds_new` and the shift of the new origin."""
    h, w = photo_shape[:2]
    area: float = 0.5 * abs(float(np.dot(quad[:, 0], np.roll(quad[:, 1], -1))
                                  - np.dot(quad[:, 1], np.roll(quad[:, 0], -1))))
    ds_photo: float = math.sqrt(max(area, 1e-9) / (w * h)) * ds_s
    lds: List[float] = [float(d) for d in run.wsi.level_downsamples]
    level: int = max([i for i, d in enumerate(lds) if d <= ds_photo * 1.01] or [0])

    crop_w: float = float(r['crop_w0']) / ds_s
    crop_h: float = float(r['crop_h0']) / ds_s
    pad: float = 0.12 * max(float(np.ptp(quad[:, 0])), float(np.ptp(quad[:, 1])))
    x0c: float = max(0.0, float(quad[:, 0].min()) - pad)
    x1c: float = min(crop_w, float(quad[:, 0].max()) + pad)
    y0c: float = max(0.0, float(quad[:, 1].min()) - pad)
    y1c: float = min(crop_h, float(quad[:, 1].max()) + pad)
    bw0: float = (x1c - x0c) * ds_s
    bh0: float = (y1c - y0c) * ds_s
    while level < len(lds) - 1 and (bw0 / lds[level]) * (bh0 / lds[level]) > _LOCATED_MAX_PX:
        level += 1
    ds_new: float = lds[level]
    ox: int = int(float(r['crop_x0']) + x0c * ds_s)
    oy: int = int(float(r['crop_y0']) + y0c * ds_s)
    img: np.ndarray = _read_box(run, ox, oy, bw0, bh0, level)
    k: float = ds_s / ds_new
    shift: np.ndarray = np.array([[k, 0.0, (float(r['crop_x0']) - ox) / ds_new],
                                  [0.0, k, (float(r['crop_y0']) - oy) / ds_new],
                                  [0.0, 0.0, 1.0]])
    return img, shift @ H


def _draw_located(fig, gs, col: int, run: SlideRun, row: pd.Series, ranks, photo: np.ndarray,
                  args) -> None:
    """The `located` panel, 2.5 times as wide as the others and in three columns: the
    query (narrow), the place stage 3 put it with the footprint outlined, and that place
    against the query warped onto it, as a checkerboard -- each of the last two the full
    height of the figure. The place is read at the query's own resolution
    (`_located_view`), not at the level stage 2 searched."""
    sub = gs[:, col].subgridspec(1, 3, width_ratios=[0.6, 1.0, 1.0], wspace=0.04)
    axes = [fig.add_subplot(sub[0, k]) for k in range(3)]
    axes[0].imshow(photo)
    axes[0].set_title('located: the query', fontsize=9)
    r = None
    if ranks is not None and len(ranks):
        hit = ranks[ranks['rank'] == args.rank]
        r = hit.iloc[0] if len(hit) else None
    H = None if r is None or pd.isna(r.get('crop_x0')) else _homography(r)
    if H is None:
        for ax in axes[1:]:
            ax.text(0.5, 0.5, f'rank {args.rank}: stage 3 found no fit,\nso there is no\n'
                    'location to show', ha='center', va='center', fontsize=9,
                    transform=ax.transAxes)
        for ax in axes:
            ax.axis('off')
        return
    ds_s: float = float(row['s2_ds'])
    quad0: np.ndarray = _footprint(photo.shape, H)
    view, H2 = _located_view(run, r, H, quad0, ds_s, photo.shape)
    quad: np.ndarray = _footprint(photo.shape, H2)
    board: np.ndarray = _checker(view, photo, H2)
    axes[1].imshow(view)
    axes[1].add_patch(Polygon(quad, closed=True, fill=False, edgecolor='magenta', lw=2))
    axes[1].set_title(f'found location  rank {args.rank}  inliers {int(r["n_inliers"])}  '
                      f'conf {float(r.get("confidence", float("nan"))):.2f}\n'
                      'magenta outline = the query\'s footprint (by H)', fontsize=9)
    axes[2].imshow(board)
    axes[2].set_title('checkerboard: slide / query warped onto it\nat the query\'s resolution',
                      fontsize=9)
    for ax in axes:
        ax.axis('off')


def draw_fov(run: SlideRun, row: pd.Series, panels, args, out: Path) -> Path:
    i, route = int(row['index']), row['route']
    real: bool = run.real
    got = run.tables.get(route, {})
    panels = [p for p in panels if p != 'windows' and not (real and p == 'gt')]
    ratios: List[float] = [2.5 if p == 'located' else 1.0 for p in panels]
    fig = plt.figure(figsize=(5.5 * sum(ratios), 5.5))
    gs = fig.add_gridspec(3, len(panels), width_ratios=ratios)
    photo = None
    cands = ranks = None
    if 2 in got and got[2]['output'] is not None:
        cands = _ranked(got[2]['output'][got[2]['output']['index'] == i])
        cands = cands.sort_values('rank').head(args.k_boxes)
    if 3 in got and got[3]['output'] is not None:
        ranks = _ranked(got[3]['output'][got[3]['output']['index'] == i])
    # a real photo's answer: the verified candidate with the largest confidence
    answer: Optional[pd.Series] = None
    if real:
        table = got.get(3, {}).get('answer')
        table = None if table is None else table[table['index'] == i]
        if table is not None and len(table) and pd.notna(table.iloc[0].get('x0')):
            answer = table.iloc[0]
    rank: int = args.rank or (int(answer['rank']) if answer is not None else 1)
    args = argparse.Namespace(**{**vars(args), 'rank': rank})
    # one overview for the slide panels: every box drawn on it, and the truth
    # when there is one (a real photo has none; its answer is in the view)
    cands_l0 = ([tuple(c) for c in cands[['x0', 'y0', 'w0', 'h0']].values]
                if cands is not None else [])
    if real:
        cx = cy = 0.0
        boxes = list(cands_l0)
        if answer is not None and cands_l0:
            w, h = cands_l0[0][2], cands_l0[0][3]
            boxes.append((float(answer['x0']) - w / 2, float(answer['y0']) - h / 2, w, h))
    else:
        cx, cy = float(row['center_x0']), float(row['center_y0'])
        boxes = [(cx - row['w0'] / 2, cy - row['h0'] / 2, row['w0'], row['h0'])] + cands_l0
    view = None
    for col, panel in enumerate(panels):
        if panel in ('photo', 'matches', 'located'):
            photo = run.photo(i) if photo is None else photo
        if panel == 'located':
            _draw_located(fig, gs, col, run, row, ranks, photo, args)
            continue
        ax = fig.add_subplot(gs[:, col])
        ax.set_title(panel)
        if panel == 'photo':
            ax.imshow(photo)
            ax.set_title(f'photo  {row["photo"]}  {int(row["width"])}x{int(row["height"])}'
                         if real else
                         f'photo  L{int(row["level"])} rot {int(row["rot_deg"])}')
        elif panel in ('gt', 'candidates', 'result'):
            if not boxes:
                ax.text(0.5, 0.5, 'stage 2 gave no candidates', ha='center', va='center',
                        transform=ax.transAxes)
                ax.axis('off')
                continue
            if view is None:
                view = _overview(run, boxes)
            img, ox, oy, sc = view
            ax.imshow(img)
            if not real:
                _quad(ax, cx, cy, row['w0'], row['h0'], row['rot_deg'] + row['angle_jitter'],
                      sc, off=(ox, oy), edgecolor='lime', lw=2)
            if panel == 'gt':
                ax.set_title('truth')
            if panel == 'candidates' and cands is not None:
                cmap = plt.get_cmap('autumn')
                for _, c in cands[::-1].iterrows():
                    ax.add_patch(Rectangle(((c['x0'] - ox) * sc, (c['y0'] - oy) * sc),
                                           c['w0'] * sc, c['h0'] * sc, fill=False, lw=1.5,
                                           edgecolor=cmap(c['rank'] / max(args.k_boxes, 1))))
                    ax.text((c['x0'] - ox) * sc, (c['y0'] - oy) * sc, str(int(c['rank'])),
                            color='yellow', fontsize=7, va='bottom')
                ax.set_title(f'candidates top {len(cands)}' if real else
                             f'candidates top {len(cands)}  GT = {_place(row)}')
                cmap = plt.get_cmap('autumn')
                ax.legend(handles=([] if real else [
                    Line2D([0], [0], color='lime', lw=2, label='truth: the FoV\'s true footprint')]) + [
                    Patch(fill=False, edgecolor=cmap(1 / max(args.k_boxes, 1)),
                          label='stage 2 candidate #1 (best)'),
                    Patch(fill=False, edgecolor=cmap(min(len(cands), args.k_boxes) / max(args.k_boxes, 1)),
                          label=f'stage 2 candidate #{len(cands)} (number = rank)')],
                    **_BELOW)
            if panel == 'result' and ranks is not None:
                shown: pd.DataFrame = ranks[ranks['rank'] <= args.k_boxes] if real else ranks
                for _, r in shown.iterrows():
                    if pd.notna(r['center_x0']):
                        ax.plot((r['center_x0'] - ox) * sc, (r['center_y0'] - oy) * sc,
                                'o' if r['success'] else 'x', ms=6,
                                color='cyan' if r['success'] else 'red')
                if answer is not None:
                    ax.plot((float(answer['x0']) - ox) * sc, (float(answer['y0']) - oy) * sc,
                            '*', ms=16, color='gold', mec='black')
                if real:
                    ax.set_title(f'stage 3 answer  rank {rank}  conf '
                                 f'{float(answer["confidence"]):.2f}' if answer is not None
                                 else 'stage 3: no answer')
                else:
                    ax.set_title(f'stage 3  err {row.get("s3_rank1_err_um", float("nan")):.1f} um')
                handles: List[Line2D] = [] if real else [
                    Line2D([0], [0], color='lime', lw=2, label='truth: the FoV\'s true footprint')]
                handles += [
                    Line2D([0], [0], marker='o', color='cyan', lw=0, label='stage 3 located it (a fit)'),
                    Line2D([0], [0], marker='x', color='red', lw=0,
                           label='no fit: the candidate window\'s own centre')]
                if real:
                    handles.append(Line2D([0], [0], marker='*', color='gold', mec='black', lw=0,
                                          ms=12, label='answer (largest confidence)'))
                ax.legend(handles=handles, **_BELOW)
        elif panel == 'stage1':
            nb = got.get(1, {}).get('neighbours') if 1 in got else None
            if nb is not None:
                v = nb[nb['index'] == i]['ref_level'].value_counts().sort_index()
                ax.bar(v.index.astype(str), v.values)
                ax.set_title(f'neighbour levels  chose L{row.get("s1_chosen_level")}' +
                             ('' if real else f' true L{int(row["level"])}'))
        elif panel in ('crop', 'matches') and ranks is not None and len(ranks):
            r = ranks[ranks['rank'] == args.rank]
            if not len(r) or pd.isna(r.iloc[0]['crop_x0']):
                continue
            r = r.iloc[0]
            level, ds = int(row['s2_routed_level']), float(row['s2_ds'])
            crop = _read_box(run, r['crop_x0'], r['crop_y0'], r['crop_w0'], r['crop_h0'],
                             level)
            if panel == 'crop':
                ax.imshow(crop)
                ax.set_title(f'rank {args.rank} crop  inliers {int(r["n_inliers"])}')
            else:
                m = got[3]['matches']
                m = m[(m['index'] == i) & (m['rank'] == args.rank)]
                h = max(photo.shape[0], crop.shape[0])
                canvas = np.full((h, photo.shape[1] + crop.shape[1], 3), 255, np.uint8)
                canvas[:photo.shape[0], :photo.shape[1]] = photo
                canvas[:crop.shape[0], photo.shape[1]:] = crop
                ax.imshow(canvas)
                for _, p in m.iterrows():
                    u = (p['x0'] - r['crop_x0']) / ds + photo.shape[1]
                    v = (p['y0'] - r['crop_y0']) / ds
                    ax.plot([p['qx'], u], [p['qy'], v], lw=0.5,
                            color='lime' if p['inlier'] else 'red', alpha=0.7)
                ax.set_title(f'rank {args.rank} matches  {int(m["inlier"].sum())}/{len(m)}')
                ax.legend(handles=[
                    Line2D([0], [0], color='lime', lw=2, label='inlier (kept by RANSAC)'),
                    Line2D([0], [0], color='red', lw=2, label='outlier (rejected)')],
                    ncol=2, **_BELOW)
        ax.axis('off')
    fig.suptitle(f'{run.name}  {row["photo"]}  #{i}' if real else
                 f'{run.name}  #{i}  route {route}', fontsize=12)
    fig.tight_layout()
    p = out / f'{run.name}_{route}_{i}.png'
    fig.savefig(p, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    return p


#: The page DemoLocaScope serves, with each window's rotation honoured.
DEMO_PAGE = Path(__file__).resolve().parent / 'demo_page' / 'index.html'


def _num(v):
    """A JSON number, or None for a missing one."""
    if v is None:
        return None
    v = float(v)
    return None if math.isnan(v) else v


def _window_key(region, lattice, row, col, rot) -> str:
    return f'{int(region)}:{lattice}:{int(row)}:{int(col)}:{int(rot)}'


def export_demo(runs: dict, sel: pd.DataFrame, stages, args, out: Path) -> Path:
    """The selected FoVs of `--demo-route` as the demo page reads them:
    `out/data.js`, `out/img/*.jpg` and the page itself (`demo_page/index.html`).

    One part per (slide, routed level). Per FoV: the photo, turned to every
    rotation one of its windows was matched at; the truth window and stage 2's
    first --k-windows candidates, each read from the slide at the routed level
    with its level-0 box and its per-tile cosines (`truth_sim`,
    `tile_sims`); the truth's rank, score, the rank-1 score and the margin.
    Stage 2 has one arm, `cls+mean`."""
    from PIL import Image
    from Store import encoder_names
    img_dir = out / 'img'
    img_dir.mkdir(parents=True, exist_ok=True)
    scale, k, tile = float(args.demo_scale), int(args.k_windows), int(args.tile_size)
    arm = 'cls+mean'

    def jpg(arr: np.ndarray, name: str) -> str:
        im = Image.fromarray(np.ascontiguousarray(arr))
        if scale != 1.0:
            im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))),
                           Image.BILINEAR)
        im.save(img_dir / name, quality=85)
        return f'img/{name}'

    def safe(text: str) -> str:
        return ''.join(ch if ch.isalnum() or ch in '-.' else '_' for ch in text)

    parts = {}
    for _, row in sel[sel['route'] == args.demo_route].iterrows():
        if pd.isna(row.get('s2_routed_level')) or pd.isna(row.get('truth_score')):
            continue
        run = runs[row['slide']]
        got = run.tables[row['route']].get(2) or {}
        i, level, ds = int(row['index']), int(row['s2_routed_level']), float(row['s2_ds'])
        photo = run.photo(i)
        part = parts.get((run.name, level))
        if part is None:
            thumb = np.asarray(run.wsi.get_thumbnail((512, 512)).convert('RGB'))
            name = f'{safe(run.name)}_thumb.jpg'
            Image.fromarray(thumb).save(img_dir / name, quality=85)
            part = parts[(run.name, level)] = dict(
                dataset=run.dataset, slide=run.name, level=level, ds=ds,
                rows_q=photo.shape[0] // tile, cols_q=photo.shape[1] // tile, pool=0,
                thumb=f'img/{name}', slide_wh0=list(run.wsi.dimensions),
                windows={}, queries=[])

        def window(key, box, rot, cos):
            if key not in part['windows']:
                pix = _read_box(run, *box, level)
                part['windows'][key] = dict(
                    img=jpg(pix, f'{safe(run.name)}_L{level}_{safe(key)}.jpg'),
                    box0=[_num(v) for v in box], rot=int(rot),
                    rows=int(round(box[3] / (tile * ds))),
                    cols=int(round(box[2] / (tile * ds))))
            return None if cos is None else [_num(v) for v in cos.reshape(-1)]

        t = got['truth'][got['truth']['index'] == i].iloc[0]
        # the truth's exact rank and the pools it is ranked in, when the entry has
        # them (`truth_rank_main`, `pool_main`, `truth_rank_main_in_all`: an entry
        # written before they were kept has none, and the page then shows `-`)
        rank_main = _num(t.get('truth_rank_main'))
        rank_all = _num(t.get('truth_rank_main_in_all'))
        if not part['pool'] and _num(t.get('pool_main')):
            part['pool'] = int(_num(t['pool_main']))
        truth = _window_key(t['truth_region'], t['truth_lattice'], t['truth_row'],
                            t['truth_col'], t['truth_rotation'])
        tt = got.get('truth_sim')
        cos = {truth: {'cls': window(
            truth, (t['truth_x0'], t['truth_y0'], t['truth_w0'], t['truth_h0']),
            t['truth_rotation'], _cos_grid(tt[tt['index'] == i] if tt is not None else None))}}
        cands = _ranked(got['output'])
        cands = cands[cands['index'] == i].sort_values('rank')
        ts = got.get('tile_sims')
        topk = []
        for _, c in cands.head(k).iterrows():
            key = _window_key(c['region'], c['lattice'], c['row'], c['col'], c['rotation'])
            sims = ts[(ts['index'] == i) & (ts['rank'] == c['rank'])] if ts is not None else None
            cos.setdefault(key, {'cls': window(key, (c['x0'], c['y0'], c['w0'], c['h0']),
                                               c['rotation'], _cos_grid(sims))})
            topk.append([key, _num(c['score'])])
        top1 = _num(cands.iloc[0]['score']) if len(cands) else None
        others = [s for key, s in topk if key != truth]
        best_other = max(others) if others else top1
        s = _num(t['truth_score'])
        rots = sorted({w['rot'] for key, w in part['windows'].items() if key in cos})
        imgs = {str(r): jpg(np.rot90(photo, k=r // 90), f'{safe(run.name)}_{i}_r{r}.jpg')
                for r in rots}
        part['queries'].append(dict(
            id=f'{run.name}|L{level}|{i}', fov_id=i,
            img=imgs.get(str(int(t['truth_rotation']))) or next(iter(imgs.values()), ''),
            imgs=imgs,
            d_main=_num(t['truth_main_dist_l0']) / ds if pd.notna(t.get('truth_main_dist_l0')) else 0.0,
            d_overlap=_num(t['truth_dist_l0']) / ds,
            truth=truth,
            arms={arm: dict(rank=rank_main if rank_main is not None else _num(row.get('s2_hit_rank')),
                            s=s, top1=top1,
                            margin=None if s is None or best_other is None else s - best_other,
                            rank_all=rank_all, topk=topk)},
            cos=cos))
    if not parts:
        raise SystemExit(f'no FoV of route {args.demo_route} with stage 2 truth in the selection')
    meta = dict(encoder=encoder_names(stages[1].cfg.encoder)[0], arm_tokens=['cls'],
                scores=['mean'], arms=[arm], arm_base={arm: 'cls'}, baseline=arm,
                top_k=k, tile=tile, image_scale=scale, check=None)
    with open(out / 'data.js', 'w') as handle:
        handle.write('window.WINDOW_DEMO = ')
        json.dump({'meta': meta, 'parts': list(parts.values())}, handle,
                  separators=(',', ':'))
        handle.write(';\n')
    shutil.copyfile(DEMO_PAGE, out / 'index.html')
    return out / 'index.html'


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    add_run_args(ap)
    ap.add_argument('--stage-cache-job', default=None,
                    help='the job whose stage entries are read. Default: '
                         'BenchLocaScope, or RealTest with --real')
    ap.add_argument('--real', action='store_true',
                    help='the real photos of a locate_photo run (no truth: the '
                         'panels and statistics that need one are left out)')
    ap.add_argument('--slides', nargs='*', default=[],
                    help=f'--real: slide names; default every slide of {DATASET_REAL}')
    ap.add_argument('--select', default='all', help='pandas query on the join table')
    ap.add_argument('--sample', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--panels', default='', help=f'comma list of {",".join(PANELS)}')
    ap.add_argument('--plot', nargs='*', default=[],
                    help='cdf:<col> recall[:<col>] confusion scatter:<x>,<y>')
    ap.add_argument('--by', default='', help='group statistics by this column')
    ap.add_argument('--tol-um', type=float, default=100.0)
    ap.add_argument('--rank', type=int, default=0,
                    help="the rank crop/matches/located show; 0 = the answer's (--real) or 1")
    ap.add_argument('--k-boxes', type=int, default=10,
                    help='candidate boxes on the overview')
    ap.add_argument('--k-windows', type=int, default=5,
                    help='candidates in the windows figure, after the truth')
    ap.add_argument('--no-tile-values', dest='tile_values', action='store_false',
                    help='windows figure: shade the tiles, do not print the cosines')
    ap.add_argument('--export', choices=['demo'], default=None,
                    help='demo: the demo page (index.html + data.js + img/) of the '
                         'selected FoVs, in <out>/demo/')
    ap.add_argument('--demo-route', default='stage1', choices=['stage1', 'oracle'],
                    help='which route the demo page shows')
    ap.add_argument('--demo-scale', type=float, default=0.25,
                    help='demo images, relative to level px')
    ap.add_argument('--dpi', type=int, default=200,
                    help='resolution of every figure written (the FoV panels, `windows` and --plot); '
                         'a FoV panel is 5.5 in square, so 200 gives 1100 px')
    ap.add_argument('--columns', action='store_true', help='list the columns and stop')
    ap.add_argument('--out', default=None)
    args = parse_run_args(ap)

    panels = [p for p in args.panels.split(',') if p]
    bad = sorted(set(panels) - set(PANELS))
    if bad:
        raise SystemExit(f'--panels {bad}: the panels are {",".join(PANELS)}')
    out = Path(args.out or job_result_dir('PlotLocaScope'))
    out.mkdir(parents=True, exist_ok=True)

    stages, routes, mask_cfg = run_from_args(args)
    job = args.stage_cache_job or ('RealTest' if args.real else 'BenchLocaScope')
    if args.real:
        if args.export:
            raise SystemExit('--export needs a truth window: not for --real')
        runs = real_runs(args, stages, mask_cfg, job)
        df = join_real(runs)
    else:
        fov_masks = MaskMaker(MASK_RECIPES['hest'], args.fov_mask_cache_job, 'cpu')
        runs = [SlideRun(ds, name, path, supply, stages, routes, mask_cfg, job, args.limit)
                for ds, name, path, supply in slide_supplies(args, fov_masks, job)]
        df = join(runs, stages, args.tol_um)
    if df.empty:
        raise SystemExit('nothing in the cache for these flags')
    if args.columns:
        print('\n'.join(df.columns))
        return
    df.to_csv(out / 'joined.csv', index=False)
    print(real_summary(df) if args.real else summary(df, args.by or None))

    sel = df if args.select == 'all' else df.query(args.select)
    if args.sample and len(sel) > args.sample:
        sel = sel.sample(args.sample, random_state=args.seed)
    sel.to_csv(out / 'selected.csv', index=False)
    print(f'selected {len(sel)} of {len(df)} rows -> {out / "selected.csv"}')

    if args.plot and args.real:
        print('  [plot] --plot needs truth: skipped for --real', flush=True)
    elif args.plot:
        fig_dir = out / 'figures' / 'stats'
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_stats(sel, args.plot, args.by or None, fig_dir, args.dpi)
    by_slide = {r.name: r for r in runs}
    args.tile_size = stages[1].cfg.tile_size
    if panels:
        fig_dir = out / 'figures' / 'fov'
        fig_dir.mkdir(parents=True, exist_ok=True)
        for _, row in sel.iterrows():
            run = by_slide[row['slide']]
            if [p for p in panels if p != 'windows']:
                print(f'  [fov] {draw_fov(run, row, panels, args, fig_dir)}', flush=True)
            if 'windows' in panels:
                p = draw_windows(run, row, args, fig_dir)
                if p:
                    print(f'  [windows] {p}', flush=True)
    if args.export == 'demo':
        page = export_demo(by_slide, sel, stages, args, out / 'demo')
        print(f'  [demo] {page}', flush=True)


if __name__ == '__main__':
    main()
