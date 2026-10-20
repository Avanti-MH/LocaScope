#!/usr/bin/env python3
"""analyze_locascope -- where a bench_locascope run loses its FoVs, stage by stage.

    python utilities/cli/metrics/analyze_locascope.py <bench_locascope's run flags> \\
        [--stage-cache-job BenchLocaScope] [--tol-px 16]
    sbatch jobscripts/AnalyzeLocaScope.sh

Reads the cache the bench wrote (the same flags give the same addresses) and runs nothing. It
needs only the stage-1 route; the oracle route (`bench_locascope --route oracle`, stage 2 given
the level the FoV was placed at) is read when its entries are there, for one extra table, and is
otherwise left out -- the other tables and figures do not change with it.

LOCATED. A FoV is located when the pipeline's final answer (stage 3's verified candidate with the
largest confidence, the better-ranked on a tie: `LocalizationResultSet.best`) lies within --tol-px
of the true centre, in the PHOTO'S OWN PIXELS: the level-0 distance over the photo's ds. A fixed
length in um is a few px of the photo at one scale and hundreds at another; a distance in photo
pixels means the same at every scale.

STAGE 2 IS JUDGED BY A HIT in the candidate list as it was returned: the place of the better of the
FoV's two answers -- the main-lattice and the offset-lattice window, at the rotation that sets the photo
upright, nearest the point the photo's own tile grid stands for, as stage 2's first phase placed them in
its truth row -- or, the suppression having dropped the answer, of the window of its rotation within one
tile of the placed level that kept its place (`BenchCommon.listed_places`, the one rule the stage-2
bench ranks by too). A list stage 2 searched on another level holds no window of the photo's level, so
it has no hit. `s2_reach_rank` is the looser question of stage 3: a window whose centre is within its
search of the main answer's.

Recall@K is the share of FoVs with a hit among the first K candidates; K is as long as the list
the cache holds.

EVERY FoV ENDS IN ONE OUTCOME, charged to the FIRST stage that failed it:

    level not searchable       the placed level is one stage 2 cannot search (`irretrievable_lvl`),
                               or it returned no candidate at all
    stage 1 routed wrong       stage 2 searched a level other than the placed one: stage 1 chose
                               wrongly and stage 2 did not end up at the placed level
    stage 2 missed             the placed level was searched, but no candidate is a hit
    stage 3 not located        a candidate is a hit, but the answer is not within --tol-px
    stage 3 located

Stage 1 choosing a wrong level is not by itself a failure: stage 3 can localize from a window found
at another level, and the report counts stage 1's errors (table 2) apart from the FoVs they cost
(table 1). A level the query is too small for is refused by stage 2 (`irretrievable_lvl`), which goes
finer instead; table 2 says how often that happened to the level stage 1 chose.

TABLES (a csv each, and in report.txt), under result/<job>/:
    1 funnel         the outcomes per dataset, all / native / resampled placements, and by the
                     placed ds; the located rate with a 95% interval resampled over SLIDES
    2 stages         each stage on its own population: stage 1's level accuracy and the direction
                     of its errors; stage 2's recall@1/@10/@K over all FoVs; stage 3's
                     located share on the FoVs stage 2 hit
    3 inliers        stage 3's candidates by their inlier count, and the distribution of inliers,
                     matches (`n_good`) and inliers / matches for candidates located and elsewhere
    4 slides         one row per slide, every share over the slide's FoVs
    5 oracle         the FoVs stage 1 routed wrong, run again at the placed level (only when the
                     oracle route is in the cache)
    6 tree           stage 1 right / too coarse / too fine (and by how many levels), and on each
                     node stage 2's recall and stage 3's located share

FIGURES, a png each, the outcomes in the same colour in all of them:
    funnel.png  recall.png  inliers.png  slides.png  tree_<dataset>.png  oracle.png
    tree_<dataset>.png is the tree itself: boxes joined left to right (all FoVs, stage 1, the levels it
    is off by, stage 2 recall, stage 3 located). In slides.png a circle is stage 1, a square stage 2
    (colour = K),
    a diamond stage 3.

Output: --out, default result/<SLURM_JOB_NAME or AnalyzeLocaScope>/.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # utilities/
import _paths                                                       # noqa: E402
_paths.setup_import_paths()
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'bench_modules'))

import matplotlib                                                   # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                     # noqa: E402
from matplotlib.patches import FancyBboxPatch                       # noqa: E402

from _paths import job_result_dir                                   # noqa: E402
from BenchCommon import (ORACLE, Answers, add_run_args, answers_of_row, first_place, listed_of_rows,  # noqa: E402
                         matching_rotation,
                         parse_run_args, run_from_args, slide_supplies,
                         stage2_entries, stage_entries)
from DsLadder import DEFAULT_RUNGS                                  # noqa: E402
from SafeSlide import SafeSlide                                     # noqa: E402
from TissueMaskConfig import MASK_RECIPES, MaskMaker                # noqa: E402

JOB_NAME: str = 'AnalyzeLocaScope'

#: The outcomes in the order they are charged, best first, and what each is called and drawn as.
#: Colour-blind-safe (Okabe-Ito), one colour per outcome in every figure.
OUTCOMES: Tuple[str, ...] = ('success', 's3_wrong', 's2_miss', 'unsearchable', 'routing_wrong')
LABELS: Dict[str, str] = {
    'success': 'stage 3 located',
    's3_wrong': 'stage 3 not located',
    's2_miss': 'stage 2 missed',
    'unsearchable': 'level not searchable',
    'routing_wrong': 'stage 1 routed wrong'}
COLOURS: Dict[str, str] = {
    'success': '#009E73', 's3_wrong': '#E69F00', 's2_miss': '#56B4E9',
    'unsearchable': '#999999', 'routing_wrong': '#D55E00'}
INK: str = '#0b0b0b'

#: Stage 3's inlier-count bins: [lo, hi) and the label.
INLIER_EDGES: Tuple[float, ...] = (0, 5, 10, 15, 20, 30, 50, math.inf)

#: The tables read of each stage's entry; stage 2's `truth` is its first phase's (`load_route`).
_ROLES: Dict[int, Tuple[str, ...]] = {1: ('output',), 2: ('output',), 3: ('output',)}

Tables = Dict[int, Dict[str, Optional[pd.DataFrame]]]


# ── reading one slide ────────────────────────────────────────────────────────

def read_table(entry: Any, role: str, sid: str) -> Optional[pd.DataFrame]:
    path: Path = entry.path(role, sid, '.csv')
    if not path.is_file():
        return None
    df: pd.DataFrame = pd.read_csv(path)
    return df if len(df.columns) else None


def load_route(base: Any, stages: Any, route: str, mask_cfg: Any, limit: int,
               name: str) -> Optional[Tables]:
    """The route's stage tables of a slide, `None` when any stage of the route is not a hit. Stage 2's
    `truth` is its chain's first phase's, the one that scored every window (a reranker's entry has none)."""
    got: Tables = {}
    state: str
    diff: List[str]
    for n, (entry, sid, rec) in stage_entries(base, stages, route, mask_cfg, limit).items():
        state, diff = entry.status(sid, rec)
        if state != 'hit':
            print(f'  {name} {route} stage {n}: {state} {"; ".join(diff)}', flush=True)
            return None
        got[n] = {role: read_table(entry, role, sid) for role in _ROLES[n]}
    entry, sid, rec = stage2_entries(base, stages, route, mask_cfg, limit)[0]
    state, diff = entry.status(sid, rec)
    if state != 'hit':
        print(f'  {name} {route} stage 2 phase 0: {state} {"; ".join(diff)}', flush=True)
        return None
    got[2]['truth'] = read_table(entry, 'truth', sid)
    return got


def by_index(df: Optional[pd.DataFrame]) -> Dict[int, pd.DataFrame]:
    if df is None or 'index' not in df:
        return {}
    return {int(i): g for i, g in df.groupby('index')}


def ranked(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """The rows of one FoV that are candidates (they have a rank), best first."""
    if df is None or 'rank' not in df:
        return pd.DataFrame()
    out: pd.DataFrame = df[df['rank'].notna()]
    return out.sort_values('rank') if len(out) else out


def first_rank_within(cands: pd.DataFrame, cx: float, cy: float, tol: float) -> Optional[int]:
    """The best rank whose window centre is within `tol` level-0 px of (cx, cy)."""
    if not len(cands):
        return None
    d: pd.Series = np.hypot(cands['x0'] + cands['w0'] / 2.0 - cx,
                            cands['y0'] + cands['h0'] / 2.0 - cy)
    hit: pd.DataFrame = cands[d <= tol]
    return int(hit['rank'].min()) if len(hit) else None


# ── what a stage-2 candidate list is judged against ──────────────────────────

def stage2_hit(cands: pd.DataFrame, found: Optional[Answers], tol: float) -> Optional[int]:
    """The rank (the list's `rank` column) of the FoV in a list stage 2 searched at the photo's own level: the
    better of its two answers' places (`BenchCommon.listed_places`: the answer's window, or, the suppression having dropped it,
    the first window of its rotation starting within `tol` level-0 px of it -- the one that kept its place).
    None when the list holds neither, or there are no answers to judge it on."""
    if found is None or not len(cands):
        return None
    place: Optional[int] = first_place(listed_of_rows(r for _, r in cands.iterrows()), found, float(tol))
    return None if place is None else int(cands.iloc[place - 1]['rank'])


# ── one route of one slide -> a row per FoV, and a row per stage-3 candidate ─

def evaluate(slide: Dict[str, Any], render: pd.DataFrame, tabs: Tables, *, routed: bool,
             tol_px: float, tile: int, padding: int
             ) -> Tuple[List[Dict[str, Any]], List[pd.DataFrame]]:
    """`(FoV rows, candidate frames)` of one route. `routed` is whether stage 1 chose the level
    (the stage-1 route); on the oracle route the level is the placed one by construction. An answer is
    right when its centre is within `tol_px` px of the truth, in the photo's own pixels (level-0 distance
    over the photo's ds): the same distance at every scale, where a fixed length in um is a few px at
    one scale and hundreds at another."""
    base_mpp: float = slide['base_mpp']
    s1: Dict[int, pd.DataFrame] = by_index(tabs[1]['output']) if routed else {}
    s2: Dict[int, pd.DataFrame] = by_index(tabs[2]['output'])
    truth: Dict[int, pd.DataFrame] = by_index(tabs[2]['truth'])
    s3: Dict[int, pd.DataFrame] = by_index(tabs[3]['output'])
    indices: List[int] = sorted(s1) if routed else sorted(s2)
    rows: List[Dict[str, Any]] = []
    cand_frames: List[pd.DataFrame] = []
    for i in indices:
        if i not in render.index:
            continue
        g: pd.Series = render.loc[i]
        placed: int = int(g['level'])
        cx: float = float(g['center_x0'])
        cy: float = float(g['center_y0'])
        ds: float = float(g['ds'])
        tol_l0: float = padding * tile * ds
        row: Dict[str, Any] = dict(
            dataset=slide['dataset'], slide=slide['name'], index=i, level=placed, ds=ds,
            rung=min(DEFAULT_RUNGS, key=lambda r: abs(math.log(ds) - math.log(r))),
            native=bool(abs(math.log(ds / slide['level_ds'][placed])) < math.log(1.01)))

        chosen: Optional[int] = None
        if routed:
            o1: pd.DataFrame = s1[i]
            value: Any = o1.iloc[0].get('chosen_level')
            chosen = int(value) if pd.notna(value) else None
            row['chosen_level'] = chosen
        row['routing_ok'] = (chosen == placed) if routed else True
        row['s1_level_diff'] = (chosen - placed) if chosen is not None else np.nan

        cands: pd.DataFrame = ranked(s2.get(i))
        searched: Optional[int] = int(cands.iloc[0]['level']) if len(cands) else None
        row['searched_level'] = searched
        row['n_cand'] = len(cands)
        row['level_ok'] = searched == placed
        row['moved'] = (chosen is not None and searched is not None and searched != chosen)
        lost: str = str(s2[i].iloc[0].get('irretrievable_lvl', '')) if i in s2 else ''
        row['placed_unsearchable'] = (not len(cands)) or (str(placed) in lost.split())

        t: Optional[pd.Series] = truth[i].iloc[0] if i in truth else None
        # the answers, as stage 2's first phase placed them (`BenchCommon.answers`); THE answer is the main one
        found: Optional[Answers] = answers_of_row(t) if t is not None else None
        row['truth_rank_all'] = t.get('truth_rank_main_in_all') if t is not None else np.nan
        row['pool_all'] = t.get('pool_all') if t is not None else np.nan

        # stage 2: the FoV's place in the list, when it searched the photo's own level (another level's
        # windows are not the photo's: there is no answer among them)
        row['s2_hit_rank'] = (stage2_hit(cands, found, tile * float(slide['level_ds'][placed]))
                              if row['level_ok'] else None)
        # the looser reach: a window of the upright rotation whose centre is within stage 3's search of the
        # main answer's centre
        reach: Optional[int] = None
        if row['level_ok']:
            up: int = matching_rotation(g['rot_deg'])
            same: pd.DataFrame = cands[cands['rotation'].astype(int) == up]
            if found is not None:
                x0, y0, w0, h0 = found.main.box
                tx, ty = x0 + w0 / 2.0, y0 + h0 / 2.0
            else:
                tx, ty = cx, cy
            reach = first_rank_within(same, tx, ty, tol_l0)
        row['s2_reach_rank'] = reach

        r3: pd.DataFrame = ranked(s3.get(i))
        err: float = math.inf
        if len(r3):
            r3 = r3.copy()
            r3['confidence'] = pd.to_numeric(r3['confidence'], errors='coerce').fillna(0.0)
            r3['n_inliers'] = pd.to_numeric(r3.get('n_inliers'), errors='coerce').fillna(0.0)
            r3['n_good'] = pd.to_numeric(r3.get('n_good'), errors='coerce').fillna(0.0)
            r3['err_px'] = (np.hypot(r3['center_x0'] - cx, r3['center_y0'] - cy) / ds).fillna(math.inf)
            r3['ratio'] = np.where(r3['n_good'] > 0, r3['n_inliers'] / r3['n_good'].clip(lower=1.0), np.nan)
            best: pd.Series = r3.sort_values(['confidence', 'rank'], ascending=[False, True]).iloc[0]
            err = float(best['err_px'])
            row.update(answer_rank=int(best['rank']), answer_confidence=float(best['confidence']),
                       answer_n_inliers=float(best['n_inliers']), answer_n_good=float(best['n_good']),
                       answer_ratio=float(best['ratio']), localized=bool(best['confidence'] > 0))
            r3['success_flag'] = r3['success'].astype(str).eq('True') if 'success' in r3 else False
            r3['right'] = r3['err_px'] <= tol_px
            cand_frames.append(r3[['rank', 'n_inliers', 'n_good', 'ratio', 'confidence', 'success_flag',
                                   'right', 'err_px']]
                               .assign(dataset=slide['dataset'], slide=slide['name'], index=i,
                                       level_ok=row['level_ok']))
        row['answer_err_px'] = err if math.isfinite(err) else np.nan
        row['answer_err_um'] = row['answer_err_px'] * ds * base_mpp if math.isfinite(err) else np.nan
        row['success'] = err <= tol_px

        if row['success']:
            outcome: str = 'success'
        elif not row['level_ok']:
            outcome = 'unsearchable' if row['placed_unsearchable'] else 'routing_wrong'
        elif row['s2_hit_rank'] is None:
            outcome = 's2_miss'
        else:
            outcome = 's3_wrong'
        row['outcome'] = outcome
        rows.append(row)
    return rows, cand_frames


# ── numbers ──────────────────────────────────────────────────────────────────

def pct(x: float) -> str:
    return f'{100.0 * x:5.1f}%' if x == x else '   n/a'


def slide_bootstrap(per_slide: pd.DataFrame, draws: int, seed: int) -> Tuple[float, float]:
    """95% interval of the pooled success rate, resampling slides (their FoVs are not
    independent). `per_slide` has `ok` and `n` per slide."""
    if len(per_slide) < 2:
        return math.nan, math.nan
    rng: np.random.Generator = np.random.default_rng(seed)
    ok: np.ndarray = per_slide['ok'].to_numpy(dtype=float)
    n: np.ndarray = per_slide['n'].to_numpy(dtype=float)
    pick: np.ndarray = rng.integers(0, len(ok), size=(draws, len(ok)))
    rate: np.ndarray = ok[pick].sum(axis=1) / n[pick].sum(axis=1)
    return float(np.percentile(rate, 2.5)), float(np.percentile(rate, 97.5))


def funnel_table(fov: pd.DataFrame, by: Sequence[str]) -> pd.DataFrame:
    """Counts and shares of each outcome for every group of `by`."""
    rows: List[Dict[str, Any]] = []
    for key, g in fov.groupby(list(by)):
        key = key if isinstance(key, tuple) else (key,)
        row: Dict[str, Any] = dict(zip(by, key), n=len(g))
        for o in OUTCOMES:
            row[o] = int((g['outcome'] == o).sum())
            row[f'{o}_pct'] = row[o] / len(g)
        rows.append(row)
    return pd.DataFrame(rows)


def recall_at(rank: pd.Series, k: int) -> float:
    """The share of FoVs whose first hit is within the first `k` candidates (no hit: not a hit)."""
    r: pd.Series = pd.to_numeric(rank, errors='coerce')
    return float((r <= k).mean()) if len(r) else math.nan


def stages_table(fov: pd.DataFrame, kmax: int) -> pd.DataFrame:
    """Each stage on the FoVs it is answerable for. Stage 2's recall is over every FoV, whichever level it
    searched: a list at another level holds no answer (`stage2_hit`)."""
    ks: List[int] = sorted({1, 10, kmax})
    rows: List[Dict[str, Any]] = []
    for dataset, g in fov.groupby('dataset'):
        row: Dict[str, Any] = dict(dataset=dataset, n=len(g))
        row['s1_level_ok'] = float(g['routing_ok'].mean())
        wrong: pd.DataFrame = g[~g['routing_ok']]
        row['s1_too_coarse'] = float((wrong['chosen_level'] > wrong['level']).sum() / len(g))
        row['s1_too_fine'] = float((wrong['chosen_level'] < wrong['level']).sum() / len(g))
        row['s1_none'] = float(wrong['chosen_level'].isna().sum() / len(g))
        row['s1_wrong_n'] = len(wrong)
        row['s1_wrong_success'] = int(wrong['success'].sum())
        row['s1_wrong_moved'] = int(wrong['moved'].sum())
        row['s1_wrong_moved_to_placed'] = int((wrong['moved'] & wrong['level_ok']).sum())
        row['s2_n'] = len(g)
        for k in ks:
            row[f's2_hit@{k}'] = recall_at(g['s2_hit_rank'], k)
        searched: pd.DataFrame = g[g['level_ok']]
        row['s2_searched_n'] = len(searched)
        row['s2_searched_hit@k'] = recall_at(searched['s2_hit_rank'], kmax)
        pct_rank: pd.Series = (pd.to_numeric(searched['truth_rank_all'], errors='coerce')
                               / pd.to_numeric(searched['pool_all'], errors='coerce'))
        row['s2_truth_rank_pct_median'] = float(pct_rank.median()) if len(searched) else math.nan
        pop3: pd.DataFrame = g[g['s2_hit_rank'].notna()]
        row['s3_n'] = len(pop3)
        row['s3_located'] = float(pop3['success'].mean()) if len(pop3) else math.nan
        row['s3_fit_found'] = float(pop3['localized'].mean()) if len(pop3) else math.nan
        row['s3_all_located'] = float(g['success'].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def inlier_bin(values: pd.Series) -> pd.Categorical:
    labels: List[str] = [f'{int(lo)}-{int(hi) - 1}' if math.isfinite(hi) else f'{int(lo)}+'
                         for lo, hi in zip(INLIER_EDGES[:-1], INLIER_EDGES[1:])]
    return pd.cut(values, bins=list(INLIER_EDGES), right=False, labels=labels)


def auc(score: np.ndarray, positive: np.ndarray) -> float:
    """P(a positive outscores a negative), ties half: the Mann-Whitney U from average ranks."""
    n_pos: int = int(positive.sum())
    n_neg: int = len(positive) - n_pos
    if n_pos == 0 or n_neg == 0:
        return math.nan
    rank: np.ndarray = pd.Series(score).rank(method='average').to_numpy()
    return float((rank[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def inliers_table(cand: pd.DataFrame, fov: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Per inlier-count bin: the candidates and how many are at the truth, and the FoVs whose answer
    has that many inliers and how many of those are right. Plus the numbers that go beside it."""
    rows: List[Dict[str, Any]] = []
    cb: pd.Series = inlier_bin(cand['n_inliers'])
    answered: pd.DataFrame = fov[fov['answer_n_inliers'].notna()]
    ab: pd.Series = inlier_bin(answered['answer_n_inliers'])
    for label in cb.cat.categories:
        c: pd.DataFrame = cand[cb == label]
        a: pd.DataFrame = answered[ab == label]
        rows.append(dict(inliers=label, candidates=len(c), candidates_right=int(c['right'].sum()),
                         candidates_right_pct=float(c['right'].mean()) if len(c) else math.nan,
                         answers=len(a), answers_right=int(a['success'].sum()),
                         answers_right_pct=float(a['success'].mean()) if len(a) else math.nan))
    flagged: pd.DataFrame = cand[cand['success_flag']]
    extra: Dict[str, float] = dict(
        auc=auc(cand['n_inliers'].to_numpy(), cand['right'].to_numpy(dtype=bool)),
        flag_min_inliers=float(flagged['n_inliers'].min()) if len(flagged) else math.nan,
        flag_precision=float(flagged['right'].mean()) if len(flagged) else math.nan,
        flag_recall=float(cand[cand['right']]['success_flag'].mean()) if cand['right'].any() else math.nan,
        n_right=int(cand['right'].sum()), n_candidates=len(cand))
    return pd.DataFrame(rows), extra


def slides_table(fov: pd.DataFrame, kmax: int) -> pd.DataFrame:
    """One row per slide, every share over all of the slide's FoVs: stage 1 chose the placed level,
    stage 2 hit@10 / @50 / @K, stage 3 located."""
    ks: List[int] = sorted({10, 50, kmax})
    rows: List[Dict[str, Any]] = []
    for (dataset, slide), g in fov.groupby(['dataset', 'slide']):
        row: Dict[str, Any] = dict(dataset=dataset, slide=slide, n=len(g), s1_level_ok=float(g['routing_ok'].mean()))
        for k in ks:
            row[f's2_hit@{k}'] = recall_at(g['s2_hit_rank'], k)
        row['located'] = float(g['success'].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['dataset', 'located']).reset_index(drop=True)





def tree_table(fov: pd.DataFrame, kmax: int) -> pd.DataFrame:
    """The pipeline as a tree, per dataset: every FoV, then by what stage 1 did (right; too coarse, and by how
    many levels; too fine, likewise), and on each node stage 2's recall@1/@10/@K over all of its
    FoVs and stage 3's located share -- over all of them and over those stage 2 hit (within K)."""
    ks: List[int] = sorted({1, 10, kmax})
    rows: List[Dict[str, Any]] = []
    for dataset, g in fov.groupby('dataset'):
        d: pd.Series = g['s1_level_diff']
        nodes: List[Tuple[str, str, int, pd.Series]] = [
            ('all', 'all FoVs', 0, pd.Series(True, index=g.index)),
            ('correct', 'stage 1 right', 1, g['routing_ok']),
            ('coarse', 'stage 1 too coarse', 1, d > 0),
            ('+1', '1 level coarser', 2, d == 1), ('+2', '2 levels coarser', 2, d == 2),
            ('+3', '3+ levels coarser', 2, d >= 3),
            ('fine', 'stage 1 too fine', 1, d < 0),
            ('-1', '1 level finer', 2, d == -1), ('-2', '2 levels finer', 2, d == -2),
            ('-3', '3+ levels finer', 2, d <= -3),
            ('none', 'no estimate', 1, d.isna() & ~g['routing_ok'])]
        for key, label, depth, mask in nodes:
            s: pd.DataFrame = g[mask.fillna(False).to_numpy(dtype=bool)]
            if not len(s):
                continue
            row: Dict[str, Any] = dict(dataset=dataset, node=key, label=label, depth=depth, n=len(s),
                                       share=len(s) / len(g), placed_level_searched=float(s['level_ok'].mean()))
            for k in ks:
                row[f'hit@{k}'] = recall_at(s['s2_hit_rank'], k)
            hit: pd.DataFrame = s[s['s2_hit_rank'].notna()]
            row['located'] = float(s['success'].mean())
            row['located_n'] = int(s['success'].sum())
            row['hit_n'] = len(hit)
            row['located_hit_n'] = int(hit['success'].sum())
            row['located_given_hit'] = float(hit['success'].mean()) if len(hit) else math.nan
            # where the FoVs that were not located are lost: the first stage that failed them (the funnel's
            # outcomes); stage 3's are split into no fit at all and a fit at the wrong place
            oc: pd.Series = s['outcome']
            row['n_located_without_hit'] = int(((oc == 'success') & s['s2_hit_rank'].isna()).sum())
            row['n_fail_s1'] = int((oc == 'routing_wrong').sum())
            row['n_fail_s2'] = int(oc.isin(['s2_miss', 'unsearchable']).sum())
            fail3: pd.DataFrame = s[oc == 's3_wrong']
            row['n_fail_s3_no_fit'] = int((~fail3['localized'].fillna(False).astype(bool)).sum())
            row['n_fail_s3_wrong_place'] = len(fail3) - row['n_fail_s3_no_fit']
            rows.append(row)
    return pd.DataFrame(rows)
def inlier_stats(cand: pd.DataFrame) -> pd.DataFrame:
    """The distribution of what stage 3 counted, per group of candidates -- located within the tolerance of
    the truth or elsewhere: its inliers, its matches (`n_good`, the matches that passed the ratio test) and
    inliers / matches."""
    rows: List[Dict[str, Any]] = []
    for right, name in ((True, 'located'), (False, 'elsewhere')):
        g: pd.DataFrame = cand[cand['right'] == right]
        for column, label in (('n_inliers', 'inliers'), ('n_good', 'matches'), ('ratio', 'inliers / matches')):
            v: pd.Series = pd.to_numeric(g[column], errors='coerce').dropna()
            rows.append(dict(group=name, quantity=label, n=len(v),
                             p10=float(v.quantile(0.1)) if len(v) else math.nan,
                             median=float(v.median()) if len(v) else math.nan,
                             p90=float(v.quantile(0.9)) if len(v) else math.nan,
                             mean=float(v.mean()) if len(v) else math.nan))
    return pd.DataFrame(rows)


def oracle_table(fov: pd.DataFrame, oracle: pd.DataFrame, kmax: int) -> pd.DataFrame:
    """The FoVs lost to stage 1's routing, on the stage-1 route and run again at the placed level."""
    keys: List[str] = ['dataset', 'slide', 'index']
    wrong: pd.DataFrame = fov[fov['outcome'] == 'routing_wrong'][keys + ['success', 'outcome']]
    both: pd.DataFrame = wrong.merge(oracle[keys + ['success', 'level_ok', 's2_hit_rank']],
                                     on=keys, how='inner', suffixes=('_s1', '_oracle'))
    rows: List[Dict[str, Any]] = []
    for dataset, g in both.groupby('dataset'):
        hit: pd.Series = pd.to_numeric(g['s2_hit_rank'], errors='coerce')
        rows.append(dict(dataset=dataset, n=len(g), success_stage1_route=float(g['success_s1'].mean()),
                         success_oracle_route=float(g['success_oracle'].mean()),
                         oracle_searchable=float(g['level_ok'].mean()),
                         oracle_s2_hit_at_10=float((hit <= 10).mean()),
                         oracle_s2_hit_at_k=float(hit.notna().mean())))
    return pd.DataFrame(rows)


# ── report ───────────────────────────────────────────────────────────────────

def report(fov: pd.DataFrame, header: List[str], funnel: pd.DataFrame, by_rung: pd.DataFrame,
           cis: Dict[str, Tuple[float, float]], stages: pd.DataFrame, inl: pd.DataFrame,
           extra: Dict[str, float], istats: pd.DataFrame, slides: pd.DataFrame, tree: pd.DataFrame,
           oracle: Optional[pd.DataFrame], kmax: int, tol_px: float) -> str:
    L: List[str] = list(header)
    bar: str = '=' * 100
    cols: str = ''.join(f'{LABELS[o]:>23}' for o in OUTCOMES)

    L += ['', bar, '1. FUNNEL   every FoV in one outcome (a failure is charged to the first stage that failed it)', bar]
    L.append(f'  {"":<28}{"n":>6}{cols}')
    for dataset, g in funnel[funnel['kind'] == 'all'].groupby('dataset'):
        for kind in ('all', 'native', 'resampled'):
            r: pd.DataFrame = funnel[(funnel['dataset'] == dataset) & (funnel['kind'] == kind)]
            if not len(r):
                continue
            r0: pd.Series = r.iloc[0]
            lab: str = f'{dataset}' if kind == 'all' else f'  {kind}'
            L.append(f'  {lab:<28}{int(r0["n"]):>6}'
                     + ''.join(f'{pct(r0[o + "_pct"]):>23}' for o in OUTCOMES))
        lo, hi = cis[dataset]
        L.append(f'  {"":<28}{"":>6}  located 95% interval over slides: {pct(lo)} - {pct(hi)}')
    L.append(f'  stage 1 chose a wrong level on {int((~fov["routing_ok"]).sum())} FoVs; '
             f'{int((fov["success"] & ~fov["routing_ok"]).sum())} of them are located all the same, '
             f'{int((fov["outcome"] == "routing_wrong").sum())} were lost to it')
    L += ['', '  by the ds the photo was placed at', f'  {"":<28}{"n":>6}{cols}']
    for _, r1 in by_rung.iterrows():
        L.append(f'  {r1["dataset"] + " ds=" + format(r1["rung"], "g"):<28}{int(r1["n"]):>6}'
                 + ''.join(f'{pct(r1[o + "_pct"]):>23}' for o in OUTCOMES))

    ks: List[int] = sorted({1, 10, kmax})
    L += ['', bar, '2. STAGES   each on the FoVs it is answerable for', bar]
    for _, s in stages.iterrows():
        L += [f'  {s["dataset"]}   (n={int(s["n"])})',
              f'    stage 1   level correct {pct(s["s1_level_ok"])}   too coarse {pct(s["s1_too_coarse"])}'
              f'   too fine {pct(s["s1_too_fine"])}   no estimate {pct(s["s1_none"])}',
              f'              {int(s["s1_wrong_n"])} wrong: {int(s["s1_wrong_success"])} located all the same '
              f'(stage 3 localizes from a window at that level); stage 2 refused the level and went finer '
              f'on {int(s["s1_wrong_moved"])}, {int(s["s1_wrong_moved_to_placed"])} of them landing on the placed level',
              f'    stage 2   recall over all {int(s["s2_n"])} FoVs (a hit: the window of either answer, or the one '
              f'of its rotation within a tile that kept its place, in a list of the placed level):',
              '              ' + '   '.join(f'hit@{k} {pct(s[f"s2_hit@{k}"])}' for k in ks),
              f'              on the {int(s["s2_searched_n"])} FoVs whose placed level it searched: hit@{kmax} '
              f'{pct(s["s2_searched_hit@k"])}'
              f'   main answer rank / all windows, median {100 * s["s2_truth_rank_pct_median"]:.2f}%',
              f'    stage 3   on {int(s["s3_n"])} FoVs stage 2 hit: located within {tol_px:g} px '
              f'{pct(s["s3_located"])}   (a fit was found for {pct(s["s3_fit_found"])})',
              f'              over all FoVs: located {pct(s["s3_all_located"])}']
    L.append(f'  (K = {kmax}: every candidate stage 2 returned, which is all stage 3 sees)')

    L += ['', bar, f'3. INLIERS   stage 3 by inlier count (every candidate it verified; located = within {tol_px:g} px)', bar,
          f'  {"inliers":<10}{"candidates":>12}{"located":>10}{"% located":>12}{"answers":>10}{"located":>9}{"% located":>11}']
    for _, b in inl.iterrows():
        L.append(f'  {b["inliers"]:<10}{int(b["candidates"]):>12}{int(b["candidates_right"]):>10}'
                 f'{pct(b["candidates_right_pct"]):>12}{int(b["answers"]):>10}{int(b["answers_right"]):>9}'
                 f'{pct(b["answers_right_pct"]):>11}')
    L += [f'  inlier count separates a candidate located from one that is not: AUC {extra["auc"]:.3f}'
          f'   ({int(extra["n_right"])} of {int(extra["n_candidates"])} candidates are located)',
          f'  the fit counts as success from {extra["flag_min_inliers"]:.0f} inliers: of the candidates it accepts '
          f'{pct(extra["flag_precision"])} are located, and it accepts {pct(extra["flag_recall"])} '
          f'of those that are',
          '  n counts candidates, not FoVs: a FoV contributes every candidate stage 3 verified (up to K), and the',
          '  windows around the truth all localize to the same place, so one FoV adds many located candidates', '',
          f'  {"group":<12}{"quantity":<20}{"n":>9}{"p10":>10}{"median":>10}{"p90":>10}{"mean":>10}']
    for _, q in istats.iterrows():
        L.append(f'  {q["group"]:<12}{q["quantity"]:<20}{int(q["n"]):>9}{q["p10"]:>10.2f}{q["median"]:>10.2f}'
                 f'{q["p90"]:>10.2f}{q["mean"]:>10.2f}')

    L += ['', bar, '4. SLIDES   (sorted by located within a dataset; every share over the slide\'s FoVs)', bar]
    hk: List[int] = sorted({10, 50, kmax})
    L.append(f'  {"slide":<26}{"n":>6}{"stage 1 right":>15}'
             + ''.join(f'{"hit@" + str(k):>10}' for k in hk) + f'{"located":>10}')
    for _, w in slides.iterrows():
        L.append(f'  {w["slide"]:<26}{int(w["n"]):>6}{pct(w["s1_level_ok"]):>15}'
                 + ''.join(f'{pct(w[f"s2_hit@{k}"]):>10}' for k in hk) + f'{pct(w["located"]):>10}')

    L += ['', bar, '5. ORACLE   the FoVs lost to stage 1\'s routing, run again at the placed level', bar]
    if oracle is None or not len(oracle):
        L.append('  not in the cache (bench_locascope --route oracle): left out; nothing above depends on it')
    else:
        L.append(f'  {"":<20}{"n":>6}{"located, stage-1 route":>24}{"located, oracle route":>24}'
                 f'{"searchable":>12}{"hit@10":>9}{"hit@" + str(kmax):>9}')
        for _, o in oracle.iterrows():
            L.append(f'  {o["dataset"]:<20}{int(o["n"]):>6}{pct(o["success_stage1_route"]):>24}'
                     f'{pct(o["success_oracle_route"]):>24}{pct(o["oracle_searchable"]):>12}'
                     f'{pct(o["oracle_s2_hit_at_10"]):>9}{pct(o["oracle_s2_hit_at_k"]):>9}')

    L += ['', bar, f'6. TREE   stage 1 -> stage 2 -> stage 3 (stage 2 recall over the node\'s FoVs; K = {kmax})', bar]
    for dataset, t in tree.groupby('dataset'):
        L += [f'  {dataset}',
              f'  {"":<30}{"n":>6}{"share":>8}' + ''.join(f'{"hit@" + str(k):>9}' for k in ks)
              + f'{"hit n":>8}{"located":>9}{"located/hit":>13}{"lost@1":>8}{"lost@2":>8}{"lost@3":>8}']
        for _, r2 in t.iterrows():
            lab2: str = '    ' * int(r2['depth']) + str(r2['label'])
            L.append(f'  {lab2:<30}{int(r2["n"]):>6}{pct(r2["share"]):>8}'
                     + ''.join(f'{pct(r2[f"hit@{k}"]):>9}' for k in ks)
                     + f'{int(r2["hit_n"]):>8}{pct(r2["located"]):>9}'
                     + f'{str(int(r2["located_hit_n"])) + "/" + str(int(r2["hit_n"])):>13}'
                     + f'{int(r2["n_fail_s1"]):>8}{int(r2["n_fail_s2"]):>8}'
                     + f'{int(r2["n_fail_s3_no_fit"]) + int(r2["n_fail_s3_wrong_place"]):>8}')
    L.append(f'  (hit n = FoVs whose candidate list holds a hit, K = {kmax}; located = the best answer within '
             f'{tol_px:g} px of the truth, in the photo\'s own pixels; located/hit = located among the FoVs with a hit; '
             f'lost@N = FoVs lost at stage N, counted once in the first stage that failed them)')
    return '\n'.join(L)


# ── figures ──────────────────────────────────────────────────────────────────

def _style(ax: Any) -> None:
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK)


def _stack(ax: Any, labels: List[str], frame: pd.DataFrame, title: str) -> None:
    """Horizontal 100% bars, one per row of `frame`, the outcomes in `OUTCOMES` order."""
    y: np.ndarray = np.arange(len(frame))[::-1]
    left: np.ndarray = np.zeros(len(frame))
    for o in OUTCOMES:
        share: np.ndarray = frame[f'{o}_pct'].to_numpy(dtype=float)
        ax.barh(y, share, left=left, color=COLOURS[o], label=LABELS[o], height=0.72,
                edgecolor='white', linewidth=0.6)
        for yi, l0, s in zip(y, left, share):
            if s >= 0.045:
                ax.text(l0 + s / 2, yi, f'{100 * s:.0f}', ha='center', va='center', fontsize=8,
                        color='white' if o in ('success', 'routing_wrong') else INK)
        left = left + share
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ['0', '25', '50', '75', '100'])
    ax.set_xlabel('% of FoVs')
    ax.set_title(title, fontsize=10, loc='left', color=INK)
    _style(ax)


def fig_funnel(funnel: pd.DataFrame, by_rung: pd.DataFrame, head: str, out: Path) -> None:
    order: Dict[str, int] = {'all': 0, 'native': 1, 'resampled': 2}
    left: pd.DataFrame = (funnel.assign(_k=funnel['kind'].map(order))
                          .sort_values(['dataset', '_k']).reset_index(drop=True))
    labels_l: List[str] = [f'{r.dataset}  n={r.n}' if r.kind == 'all' else f'    {r.kind}  n={r.n}'
                           for r in left.itertuples()]
    labels_r: List[str] = [f'{r.dataset}  ds={r.rung:g}  n={r.n}' for r in by_rung.itertuples()]
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.42 * max(len(left), len(by_rung)) + 2.6),
                             gridspec_kw=dict(width_ratios=[1, 1]))
    _stack(axes[0], labels_l, left, 'by dataset, and by native / resampled placement')
    _stack(axes[1], labels_r, by_rung, 'by the ds the photo was placed at')
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='lower center', ncol=5, frameon=False, fontsize=9)
    fig.suptitle(head, x=0.01, ha='left', fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_recall(fov: pd.DataFrame, kmax: int, head: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    palette: List[str] = ['#0072B2', '#CC79A7', '#F0E442', '#000000']
    ks: np.ndarray = np.unique(np.round(np.geomspace(1, max(kmax, 2), 60)).astype(int))
    marks_k: List[int] = sorted({1, 10, kmax})
    for colour, (dataset, g) in zip(palette, fov.groupby('dataset')):
        rank: pd.Series = pd.to_numeric(g['s2_hit_rank'], errors='coerce')
        rate: List[float] = [float((rank <= k).mean()) for k in ks]
        ax.plot(ks, rate, color=colour, lw=2, label=f'{dataset}  (n={len(g)})')
        for k in marks_k:
            ax.scatter([k], [float((rank <= k).mean())], color=colour, zorder=3, s=22)
            ax.annotate(f'{100 * float((rank <= k).mean()):.0f}', (k, float((rank <= k).mean())),
                        textcoords='offset points', xytext=(0, 7), ha='center', fontsize=8, color=colour)
    for k in marks_k:
        ax.axvline(k, color='#c3c2b7', lw=0.8, zorder=0)
    ax.set_xscale('log')
    ax.set_xticks(marks_k, [str(k) for k in marks_k])
    ax.minorticks_off()
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ['0', '25', '50', '75', '100'])
    ax.set_xlabel('K  (stage 2 candidates looked at)')
    ax.set_ylabel('% of FoVs with a hit among them')
    ax.set_title('Stage 2 recall@K over all FoVs\n' + head,
                 fontsize=10, loc='left', color=INK)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _group_hist(ax: Any, cand: pd.DataFrame, column: str, edges: np.ndarray, shift: float, tol_px: float) -> None:
    """Two overlaid step histograms, each normalised to its own group: candidates located within `tol_px` of
    the truth and the rest."""
    for right, colour, name in ((True, COLOURS['success'], f'located (within {tol_px:g} px)'),
                                (False, COLOURS['routing_wrong'], 'elsewhere')):
        v: np.ndarray = pd.to_numeric(cand.loc[cand['right'] == right, column], errors='coerce').dropna().to_numpy() + shift
        if len(v):
            ax.hist(v, bins=edges, weights=np.full(len(v), 1.0 / len(v)), histtype='stepfilled',
                    alpha=0.45, color=colour, edgecolor=colour, label=f'{name}  (n={len(v)} candidates)')


def fig_inliers(cand: pd.DataFrame, inl: pd.DataFrame, extra: Dict[str, float], head: str, tol_px: float,
                out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ticks: List[int] = [0, 1, 5, 10, 50, 100, 500, 1000, 5000]
    for ax, column, name in ((axes[0][0], 'n_inliers', 'inliers'), (axes[0][1], 'n_good', 'matches')):
        top: float = float(max(pd.to_numeric(cand[column], errors='coerce').max(), 2.0))
        edges: np.ndarray = np.geomspace(1, top + 2, 45)       # on n + 1, so that 0 sits at 1
        _group_hist(ax, cand, column, edges, 1.0, tol_px)
        shown: List[int] = [t for t in ticks if t + 1 <= top + 2]
        ax.set_xscale('log')
        ax.set_xticks([t + 1 for t in shown], [str(t) for t in shown])
        ax.set_xlabel(f'stage 3 {name} of a candidate (log scale, 0 at the left)')
        ax.set_ylabel('share of its group')
        _style(ax)
    flag: float = extra['flag_min_inliers']
    if math.isfinite(flag):
        axes[0][0].axvline(flag + 1.0, color=INK, ls='--', lw=1)
        axes[0][0].text(flag + 1.0, axes[0][0].get_ylim()[1] * 0.97, f' fit accepted\n from {flag:.0f} inliers',
                        fontsize=8, va='top', color=INK)
    axes[0][0].set_title(f'inliers: matches that agree with the fitted transform   (AUC {extra["auc"]:.2f})',
                         fontsize=10, loc='left', color=INK)
    axes[0][1].set_title('matches: SIFT matches that passed the ratio test', fontsize=10, loc='left', color=INK)
    axes[0][0].legend(frameon=False, fontsize=8, loc='upper right')

    ax = axes[1][0]
    _group_hist(ax, cand, 'ratio', np.linspace(0, 1, 41), 0.0, tol_px)
    ax.set_xlim(0, 1)
    ax.set_xlabel('inliers / matches of a candidate')
    ax.set_ylabel('share of its group')
    ax.set_title('inliers / matches', fontsize=10, loc='left', color=INK)
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    _style(ax)

    ax = axes[1][1]
    x: np.ndarray = np.arange(len(inl))
    share: np.ndarray = inl['answers_right_pct'].fillna(0).to_numpy(dtype=float)
    ax.bar(x, share, color=COLOURS['success'], width=0.7)
    for xi, s, n in zip(x, share, inl['answers'].to_numpy()):
        ax.text(xi, s + 0.02, f'n={int(n)}', ha='center', fontsize=8, color=INK)
    ax.set_xticks(x, inl['inliers'].astype(str).tolist(), fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ['0', '25', '50', '75', '100'])
    ax.set_xlabel('inliers of the FoV\'s answer')
    ax.set_ylabel(f'% of those answers within {tol_px:g} px')
    ax.set_title('how far to trust an answer (n = FoVs)', fontsize=10, loc='left', color=INK)
    _style(ax)
    fig.text(0.01, 0.005, 'n counts candidates, not FoVs: a FoV adds every candidate stage 3 verified (up to K), '
             'and the windows around the truth all localize to the same place.', fontsize=8, color=INK)
    fig.suptitle(head, x=0.01, ha='left', fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(out, dpi=140)
    plt.close(fig)


#: stage 2's squares, by K: light to dark, so they differ in lightness as well as hue
K_COLOURS: Tuple[str, ...] = ('#BFE1F5', '#56B4E9', '#0B5394')


def fig_slides(slides: pd.DataFrame, kmax: int, tol_px: float, head: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 0.32 * len(slides) + 2.9))
    y: np.ndarray = np.arange(len(slides))[::-1]
    hk: List[int] = sorted({10, 50, kmax})
    cols: List[str] = ['s1_level_ok'] + [f's2_hit@{k}' for k in hk] + ['located']
    for yi, (_, r) in zip(y, slides.iterrows()):
        vals: List[float] = [float(r[c]) for c in cols if r[c] == r[c]]
        ax.plot([min(vals), max(vals)], [yi, yi], color='#c3c2b7', lw=1.5, zorder=1)
    names: List[str] = slides['dataset'].tolist()
    for k in range(1, len(names)):                       # a rule where the dataset changes
        if names[k] != names[k - 1]:
            ax.axhline(y[k] + 0.5, color=INK, lw=0.8)
    ax.scatter(slides['s1_level_ok'], y, color='#D55E00', marker='o', s=40, zorder=3,
               label='stage 1: estimated the placed level (circle)')
    for k, colour in zip(hk, K_COLOURS):
        ax.scatter(slides[f's2_hit@{k}'], y, color=colour, marker='s', s=38, zorder=3, edgecolors='#1b3a57',
                   linewidths=0.5, label=f'stage 2: hit@{k} (square, darker = larger K)')
    ax.scatter(slides['located'], y, color='#009E73', marker='D', s=40, zorder=3,
               label=f'stage 3: located within {tol_px:g} px (diamond)')
    ax.set_yticks(y, [f'{r.slide}' for r in slides.itertuples()], fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ['0', '25', '50', '75', '100'])
    ax.set_xlabel('% of the slide\'s FoVs (every marker over all of them)')
    ax.grid(axis='x', alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=2)
    ax.set_title(head, fontsize=10, loc='left', color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _node(ax: Any, x: float, y: float, w: float, h: float, lines: List[str], face: str, edge: str,
          mono: bool = False, size: float = 9.0) -> None:
    """One box of the tree, `lines` of text centred in it."""
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.02,rounding_size=0.07',
                                facecolor=face, edgecolor=edge, linewidth=1.3, zorder=2))
    ax.text(x, y, '\n'.join(lines), ha='center', va='center', fontsize=size, color=INK, zorder=3,
            family='monospace' if mono else 'sans-serif', linespacing=1.4)


def _link(ax: Any, x0: float, y0: float, x1: float, y1: float) -> None:
    """An elbow from the right edge of a parent (x0, y0) to the left edge of a child (x1, y1)."""
    xm: float = (x0 + x1) / 2.0
    ax.plot([x0, xm, xm, x1], [y0, y0, y1, y1], color='#8A9199', lw=1.4, zorder=1, solid_capstyle='round')


def _leaf(ax: Any, x: float, y: float, w: float, h: float, text: str, failed: bool, empty: bool) -> None:
    """A leaf of the tree: where some of a node's FoVs end up, a failure in orange and a success in green
    (grey when none ended there)."""
    face, edge = ('#FBEBE2', '#D55E00') if failed else ('#E4F4EE', '#009E73')
    if empty:
        face, edge = '#F4F4F2', '#C3C2B7'
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.015,rounding_size=0.05',
                                facecolor=face, edgecolor=edge, linewidth=1.1, zorder=2))
    ax.text(x, y, text, ha='center', va='center', fontsize=7.6, color='#7a7a7a' if empty else INK, zorder=3)


def _drop(ax: Any, x: float, y_top: float, y_bottom: float) -> None:
    """A short vertical branch from a box down to the leaf below it."""
    ax.plot([x, x], [y_top, y_bottom], color='#8A9199', lw=1.3, zorder=1)


def fig_tree(tree: pd.DataFrame, kmax: int, tol_px: float, head: str, out_dir: Path) -> None:
    """The pipeline as a tree, one picture per dataset, read from left to right:

        all FoVs -> what stage 1 did (right / too coarse / too fine) -> how many levels off ->
        stage 2's recall@1 / @10 / @K on that node's FoVs -> stage 3: located or not

    Every box says how many FoVs it holds. The orange LEAVES are the FoVs lost there, each hung under the
    box of the stage that lost it -- stage 1 (it searched a level that is not the photo's), stage 2 (it
    searched the right level and its list has no hit), stage 3 (a hit, but no fit, or a fit at the wrong
    place) -- and the green one, to the right, the FoVs located. A FoV is counted once, in the first stage
    that failed it, so the leaves of a row add up to its stage-1 box. The stage-2 box is the recall of the
    FoVs of the stage-1 box to its left (every FoV of it, whichever level stage 1 sent stage 2 to)."""
    ks: List[int] = sorted({1, 10, kmax})
    groups: List[Tuple[str, List[str]]] = [('correct', []), ('coarse', ['+1', '+2', '+3']),
                                           ('fine', ['-1', '-2', '-3']), ('none', [])]
    xs: List[float] = [0.0, 2.35, 4.7, 7.05, 9.6, 12.55]
    widths: List[float] = [1.45, 1.85, 1.85, 2.45, 2.3, 3.0]
    pitch: float = 2.45
    for dataset, t in tree.groupby('dataset'):
        rows: Dict[str, Dict[str, Any]] = t.set_index('node').to_dict('index')
        plan: List[Tuple[str, Optional[str]]] = []                    # (class node, step node or None), a leaf each
        for cls, steps in groups:
            if cls not in rows:
                continue
            present: List[str] = [s for s in steps if s in rows]
            plan += [(cls, s) for s in present] if present else [(cls, None)]
        n_leaf: int = len(plan)
        fig, ax = plt.subplots(figsize=(15.5, 0.62 * pitch * n_leaf + 1.9))
        ys: Dict[Tuple[str, Optional[str]], float] = {leaf: (n_leaf - 1 - i) * pitch for i, leaf in enumerate(plan)}
        class_y: Dict[str, float] = {}
        for cls in dict.fromkeys(c for c, _ in plan):
            class_y[cls] = float(np.mean([ys[leaf] for leaf in plan if leaf[0] == cls]))
        root_y: float = float(np.mean(list(class_y.values())))
        root: Dict[str, Any] = rows['all']
        _node(ax, xs[0], root_y, widths[0], 0.72, ['all FoVs', f'n={int(root["n"])}'], '#FFFFFF', INK)
        for cls, y in class_y.items():
            r: Dict[str, Any] = rows[cls]
            _link(ax, xs[0] + widths[0] / 2, root_y, xs[1] - widths[1] / 2, y)
            _node(ax, xs[1], y, widths[1], 0.72, [str(r['label']), f'n={int(r["n"])}  ({100 * r["share"]:.1f}%)'],
                  '#E8F0FA' if cls == 'correct' else '#FFFFFF', '#1E5FA8' if cls == 'correct' else '#8A9199')
        for (cls, step), y in ys.items():
            leaf: Dict[str, Any] = rows[step] if step else rows[cls]
            n_all: int = int(leaf['n'])
            s1_x, s1_w = (xs[2], widths[2]) if step else (xs[1], widths[1])
            if step:
                _link(ax, xs[1] + widths[1] / 2, class_y[cls], xs[2] - widths[2] / 2, y)
                _node(ax, xs[2], y, widths[2], 0.72, [str(leaf['label']), f'n={n_all}  ({100 * leaf["share"]:.1f}%)'],
                      '#FFFFFF', '#8A9199')
            _link(ax, s1_x + s1_w / 2, y, xs[3] - widths[3] / 2, y)
            head_line: str = ''.join(f'{"@" + str(k):>9}' for k in ks)
            values: str = ''.join(f'{100 * leaf[f"hit@{k}"]:8.1f}%' for k in ks)
            _node(ax, xs[3], y, widths[3], 0.98, ['stage 2 recall', head_line, values], '#E8F0FA', '#1E5FA8',
                  mono=True, size=9)
            _link(ax, xs[3] + widths[3] / 2, y, xs[4] - widths[4] / 2, y)
            n_hit: int = int(leaf['hit_n'])
            given: str = (f'{int(leaf["located_hit_n"])} / {n_hit} = {pct(float(leaf["located_given_hit"])).strip()}'
                          if n_hit else 'no FoV hit')
            _node(ax, xs[4], y, widths[4], 0.98, ['stage 3', f'hit (stage 2): {n_hit} / {n_all}',
                                                  'located, of the hit:', given], '#E4F4EE', '#009E73', size=8.5)

            def share(k: int) -> str:
                return f'{k}  ({100 * k / n_all:.1f}%)' if n_all else '0'

            f1, f2 = int(leaf['n_fail_s1']), int(leaf['n_fail_s2'])
            f3a, f3b = int(leaf['n_fail_s3_no_fit']), int(leaf['n_fail_s3_wrong_place'])
            located, without = int(leaf['located_n']), int(leaf['n_located_without_hit'])
            h_leaf, gap = 0.34, 0.40
            # under each box, the FoVs that stage lost
            _drop(ax, s1_x, y - 0.36, y - 0.62)
            _leaf(ax, s1_x, y - 0.80, s1_w + 0.3, h_leaf, f'stage 1 failed: {share(f1)}', True, f1 == 0)
            _drop(ax, xs[3], y - 0.49, y - 0.62)
            _leaf(ax, xs[3], y - 0.80, widths[3], h_leaf, f'stage 2 failed: {share(f2)}', True, f2 == 0)
            _drop(ax, xs[4], y - 0.49, y - 0.62)
            _leaf(ax, xs[4], y - 0.80, widths[4] + 0.3, h_leaf, f'stage 3 failed, no fit: {share(f3a)}', True, f3a == 0)
            _leaf(ax, xs[4], y - 0.80 - gap, widths[4] + 0.3, h_leaf, f'stage 3 failed, wrong place: {share(f3b)}',
                  True, f3b == 0)
            _link(ax, xs[4] + widths[4] / 2, y, xs[5] - widths[5] / 2, y)
            _leaf(ax, xs[5], y, widths[5], 0.62, f'located: {share(located)}'
                  + (f'\n{without} of them without a stage-2 hit' if without else ''), False, located == 0)
        top: float = (n_leaf - 1) * pitch + 0.85
        for x, name in zip(xs, ('all', 'stage 1', 'levels off', 'stage 2\nrecall of the FoVs\nin the box to its left',
                                f'stage 3\nlocated = within\n{tol_px:g} px', 'located')):
            ax.text(x, top, name, ha='center', va='bottom', fontsize=9.5, color=INK, fontweight='bold',
                    linespacing=1.25)
        ax.set_xlim(xs[0] - widths[0] / 2 - 0.2, xs[5] + widths[5] / 2 + 0.2)
        ax.set_ylim(-1.35, top + 1.0)
        ax.axis('off')
        fig.suptitle(f'{dataset}   {head}', x=0.01, ha='left', fontsize=11, color=INK)
        fig.text(0.01, 0.005, f'orange leaves: FoVs lost at that stage, each counted once in the first stage that '
                 f'failed it, so a row\'s leaves add up to its stage-1 box. stage 2 hit: the truth window or one within '
                 f'a tile (of the placed level) of it, at the upright rotation; K = {kmax}.  stage 2 failed also holds '
                 f'the FoVs whose placed level cannot be searched.', fontsize=7.5, color=INK)
        fig.tight_layout(rect=(0, 0.03, 1, 0.95))
        fig.savefig(out_dir / f'tree_{dataset.replace("/", "_")}.png', dpi=140)
        plt.close(fig)


def fig_oracle(oracle: pd.DataFrame, kmax: int, head: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    series: Tuple[Tuple[str, str, str], ...] = (
        ('success_stage1_route', 'success, stage-1 route', COLOURS['routing_wrong']),
        ('success_oracle_route', 'success, oracle route', COLOURS['success']),
        ('oracle_s2_hit_at_k', f'stage 2 hit@{kmax}, oracle route', COLOURS['s2_miss']))
    x: np.ndarray = np.arange(len(oracle))
    w: float = 0.26
    for j, (col, name, colour) in enumerate(series):
        v: np.ndarray = oracle[col].to_numpy(dtype=float)
        ax.bar(x + (j - 1) * w, v, width=w, color=colour, label=name)
        for xi, s in zip(x + (j - 1) * w, v):
            ax.text(xi, s + 0.015, f'{100 * s:.0f}', ha='center', fontsize=8, color=INK)
    ax.set_xticks(x, [f'{r.dataset}\n(n={r.n})' for r in oracle.itertuples()])
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ['0', '25', '50', '75', '100'])
    ax.set_ylabel('% of the FoVs lost to stage 1\'s routing')
    ax.set_title('Run again at the placed level\n' + head, fontsize=10, loc='left', color=INK)
    ax.legend(frameon=False, fontsize=9, loc='upper left', bbox_to_anchor=(0, -0.14), ncol=3)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap: argparse.ArgumentParser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    add_run_args(ap)
    ap.add_argument('--stage-cache-job', default='BenchLocaScope',
                    help='the job whose stage entries are read')
    ap.add_argument('--tol-px', type=float, default=16.0,
                    help='the final answer is located when its centre is this close to the truth, in the '
                         'photo\'s own pixels (level-0 distance / the photo\'s ds): the same distance at every scale')
    ap.add_argument('--bootstrap', type=int, default=2000, help='slide resamples for the interval')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    args: argparse.Namespace = parse_run_args(ap)

    out: Path = Path(args.out or job_result_dir(JOB_NAME))
    out.mkdir(parents=True, exist_ok=True)
    stages, _, mask_cfg = run_from_args(args)
    tile: int = stages[1].cfg.tile_size
    padding: int = stages[2].cfg.padding
    job: str = args.stage_cache_job

    fov_masks: MaskMaker = MaskMaker(MASK_RECIPES['hest'], args.fov_mask_cache_job, 'cpu')
    fov_rows: List[Dict[str, Any]] = []
    oracle_rows: List[Dict[str, Any]] = []
    cand_frames: List[pd.DataFrame] = []
    missing: List[str] = []
    oracle_slides: int = 0
    n_slides: int = 0
    for dataset, name, path, supply in slide_supplies(args, fov_masks, job):
        n_slides += 1
        base: Any = supply.render_address.on(job)
        got: Optional[Tables] = load_route(base, stages, 'stage1', mask_cfg, args.limit, name)
        csv_path: Path = supply.render_entry.path('render', supply.microscope.cfg.identity_id(), '.csv')
        if got is None or not csv_path.is_file():
            missing.append(name)
            continue
        wsi: SafeSlide = SafeSlide(path)
        slide: Dict[str, Any] = dict(dataset=dataset, name=name, base_mpp=float(wsi.base_mpp),
                                     level_ds=[float(d) for d in wsi.level_downsamples])
        wsi.close()
        render: pd.DataFrame = pd.read_csv(csv_path).set_index('index')
        rows, frames = evaluate(slide, render, got, routed=True, tol_px=args.tol_px, tile=tile, padding=padding)
        fov_rows += rows
        cand_frames += frames
        oracle_got: Optional[Tables] = load_route(base, stages, ORACLE, mask_cfg, args.limit, name) \
            if (base.at(stage1=ORACLE).entry('stage2').dir.is_dir()) else None
        if oracle_got is not None:
            oracle_slides += 1
            orows, _ = evaluate(slide, render, oracle_got, routed=False, tol_px=args.tol_px,
                                tile=tile, padding=padding)
            oracle_rows += orows
    if not fov_rows:
        raise SystemExit('nothing in the cache for these flags (is the stage-1 route there?)')

    fov: pd.DataFrame = pd.DataFrame(fov_rows)
    cand: pd.DataFrame = (pd.concat(cand_frames, ignore_index=True) if cand_frames else
                          pd.DataFrame(columns=['rank', 'n_inliers', 'n_good', 'ratio', 'confidence',
                                                'success_flag', 'right', 'err_px']))
    oracle_fov: pd.DataFrame = pd.DataFrame(oracle_rows)
    kmax: int = int(fov['n_cand'].max())

    fov['kind'] = np.where(fov['native'], 'native', 'resampled')
    funnel: pd.DataFrame = funnel_table(fov.assign(kind='all'), ['dataset', 'kind'])
    if fov['native'].nunique() > 1:                  # a split of one kind would only repeat the rows above
        funnel = pd.concat([funnel, funnel_table(fov, ['dataset', 'kind'])], ignore_index=True)
    by_rung: pd.DataFrame = funnel_table(fov, ['dataset', 'rung'])
    cis: Dict[str, Tuple[float, float]] = {}
    for dataset, g in fov.groupby('dataset'):
        per: pd.DataFrame = g.groupby('slide').agg(ok=('success', 'sum'), n=('success', 'size'))
        cis[dataset] = slide_bootstrap(per, args.bootstrap, args.seed)
    stages_df: pd.DataFrame = stages_table(fov, kmax)
    inl, extra = inliers_table(cand, fov) if len(cand) else (pd.DataFrame(), {})
    istats: pd.DataFrame = inlier_stats(cand) if len(cand) else pd.DataFrame()
    slides_df: pd.DataFrame = slides_table(fov, kmax)
    tree_df: pd.DataFrame = tree_table(fov, kmax)
    oracle_df: Optional[pd.DataFrame] = (oracle_table(fov, oracle_fov, kmax)
                                         if len(oracle_fov) else None)

    header: List[str] = [
        f'analyze_locascope   cache job {job}   datasets {" ".join(args.datasets)}   located = within {args.tol_px:g} px of the truth '
        f'(photo pixels)   stage 2 returns up to {kmax} candidates',
        f'slides {fov["slide"].nunique()} of {n_slides}   FoVs {len(fov)}'
        + (f'   not in the cache: {", ".join(missing)}' if missing else '')
        + f'   oracle route for {oracle_slides} slides']
    text: str = report(fov, header, funnel, by_rung, cis, stages_df, inl, extra, istats, slides_df, tree_df,
                       oracle_df, kmax, args.tol_px)
    print(text)
    (out / 'report.txt').write_text(text + '\n')

    fov.to_csv(out / 'fov.csv', index=False)
    funnel.to_csv(out / 'funnel.csv', index=False)
    by_rung.to_csv(out / 'funnel_by_rung.csv', index=False)
    stages_df.to_csv(out / 'stages.csv', index=False)
    inl.to_csv(out / 'inliers.csv', index=False)
    slides_df.to_csv(out / 'slides.csv', index=False)
    tree_df.to_csv(out / 'tree.csv', index=False)
    istats.to_csv(out / 'inlier_stats.csv', index=False)
    if oracle_df is not None:
        oracle_df.to_csv(out / 'oracle.csv', index=False)

    head: str = (f'{job}, {" ".join(args.datasets)}, located = within {args.tol_px:g} px of the truth, '
                 f'{len(fov)} FoVs on {fov["slide"].nunique()} slides')
    fig_funnel(funnel, by_rung, head, out / 'funnel.png')
    fig_recall(fov, kmax, head, out / 'recall.png')
    if len(cand):
        fig_inliers(cand, inl, extra, head, args.tol_px, out / 'inliers.png')
    fig_slides(slides_df, kmax, args.tol_px, head, out / 'slides.png')
    fig_tree(tree_df, kmax, args.tol_px, head, out)
    if oracle_df is not None:
        fig_oracle(oracle_df, kmax, head, out / 'oracle.png')
    print(f'\n-> {out}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
