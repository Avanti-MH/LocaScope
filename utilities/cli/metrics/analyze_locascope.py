#!/usr/bin/env python3
"""analyze_locascope -- where a bench_locascope run loses its FoVs, stage by stage.

    python utilities/cli/metrics/analyze_locascope.py <bench_locascope's run flags> \\
        [--stage-cache-job BenchLocaScope] [--tol-um 100]
    sbatch jobscripts/AnalyzeLocaScope.sh

Reads the cache the bench wrote (the same flags give the same addresses) and runs nothing. It
needs only the stage-1 route; the oracle route (`bench_locascope --route oracle`, stage 2 given
the level the FoV was placed at) is read when its entries are there, for one extra table, and is
otherwise left out -- the other four tables and figures do not change with it.

EVERY FoV ENDS IN ONE OUTCOME. A FoV is a success when the pipeline's final answer (stage 3's
verified candidate with the largest confidence, the better-ranked on a tie: `LocalizationResultSet.
best`) lies within --tol-um of the true centre. One that is not is charged to the FIRST stage that
failed it:

    level not searchable       the placed level is one stage 2 cannot search (`irretrievable_lvl`),
                               or it returned no candidate at all
    stage 1 routed wrong       stage 2 searched a level other than the placed one: stage 1 chose
                               wrongly and stage 2 did not end up at the placed level
    stage 2 missed             the placed level was searched, but no candidate window at the
                               upright rotation is within stage 3's search of the truth window
    stage 3 picked wrong       a candidate window covers the truth, but the answer is elsewhere
    success

Stage 1 choosing a wrong level is not by itself a failure: stage 3 can localize from a window found
at another level, and the report counts stage 1's errors (table 2) apart from the FoVs they cost
(table 1). A level the query is too small for is refused by stage 2 (`irretrievable_lvl`), which goes
finer instead; table 2 says how often that happened to the level stage 1 chose.

TABLES (a csv each, and in report.txt), under result/<job>/:
    1 funnel         the outcomes per dataset, all / native / resampled placements, and by the
                     placed ds; the success rate with a 95% interval resampled over SLIDES
    2 stages         each stage on its own population: stage 1's level accuracy and the direction
                     of its errors; stage 2's hit@K and where the truth window ranks among all
                     windows, on the FoVs it could search; stage 3's pick, on the FoVs stage 2 hit
    3 inliers        stage 3's candidates by their inlier count: how many are at the truth, and
                     how often an answer with that many inliers is right
    4 slides         one row per slide
    5 oracle         the FoVs stage 1 routed wrong, run again at the placed level (only when the
                     oracle route is in the cache)

FIGURES, a png each, the five outcomes in the same colour in all of them:
    funnel.png  recall.png  inliers.png  slides.png  oracle.png

Output: --out, default result/<SLURM_JOB_NAME or AnalyzeLocaScope>/.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # utilities/
import _paths                                                       # noqa: E402
_paths.setup_import_paths()
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'bench_modules'))

import matplotlib                                                   # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                     # noqa: E402

from _paths import job_result_dir                                   # noqa: E402
from BenchCommon import (ORACLE, add_run_args, matching_rotation,  # noqa: E402
                         parse_run_args, run_from_args, slide_supplies,
                         stage_entries)
from DsLadder import DEFAULT_RUNGS                                  # noqa: E402
from SafeSlide import SafeSlide                                     # noqa: E402
from TissueMaskConfig import MASK_RECIPES, MaskMaker                # noqa: E402

JOB_NAME: str = 'AnalyzeLocaScope'

#: The outcomes in the order they are charged, best first, and what each is called and drawn as.
#: Colour-blind-safe (Okabe-Ito), one colour per outcome in every figure.
OUTCOMES: Tuple[str, ...] = ('success', 's3_wrong', 's2_miss', 'unsearchable', 'routing_wrong')
LABELS: Dict[str, str] = {
    'success': 'success',
    's3_wrong': 'stage 3 picked wrong',
    's2_miss': 'stage 2 missed',
    'unsearchable': 'level not searchable',
    'routing_wrong': 'stage 1 routed wrong'}
COLOURS: Dict[str, str] = {
    'success': '#009E73', 's3_wrong': '#E69F00', 's2_miss': '#56B4E9',
    'unsearchable': '#999999', 'routing_wrong': '#D55E00'}
INK: str = '#0b0b0b'

#: Stage 3's inlier-count bins: [lo, hi) and the label.
INLIER_EDGES: Tuple[float, ...] = (0, 5, 10, 15, 20, 30, 50, math.inf)

_ROLES: Dict[int, Tuple[str, ...]] = {1: ('output',), 2: ('output', 'truth'), 3: ('output',)}

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
    """The route's stage tables of a slide, `None` when any stage of the route is not a hit."""
    got: Tables = {}
    for n, (entry, sid, rec) in stage_entries(base, stages, route, mask_cfg, limit).items():
        state: str
        diff: List[str]
        state, diff = entry.status(sid, rec)
        if state != 'hit':
            print(f'  {name} {route} stage {n}: {state} {"; ".join(diff)}', flush=True)
            return None
        got[n] = {role: read_table(entry, role, sid) for role in _ROLES[n]}
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


# ── one route of one slide -> a row per FoV, and a row per stage-3 candidate ─

def evaluate(slide: Dict[str, Any], render: pd.DataFrame, tabs: Tables, *, routed: bool,
             tol_um: float, tile: int, padding: int) -> Tuple[List[Dict[str, Any]], List[pd.DataFrame]]:
    """`(FoV rows, candidate frames)` of one route. `routed` is whether stage 1 chose the level
    (the stage-1 route); on the oracle route the level is the placed one by construction."""
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

        cands: pd.DataFrame = ranked(s2.get(i))
        searched: Optional[int] = int(cands.iloc[0]['level']) if len(cands) else None
        row['searched_level'] = searched
        row['n_cand'] = len(cands)
        row['level_ok'] = searched == placed
        row['moved'] = (chosen is not None and searched is not None and searched != chosen)
        lost: str = str(s2[i].iloc[0].get('irretrievable_lvl', '')) if i in s2 else ''
        row['placed_unsearchable'] = (not len(cands)) or (str(placed) in lost.split())

        t: Optional[pd.Series] = truth[i].iloc[0] if i in truth else None
        row['truth_rank_all'] = t.get('truth_rank_all') if t is not None else np.nan
        row['pool_all'] = t.get('pool_all') if t is not None else np.nan
        hit: Optional[int] = None
        if row['level_ok']:
            up: int = matching_rotation(g['rot_deg'])
            same: pd.DataFrame = cands[cands['rotation'].astype(int) == up]
            if t is not None and pd.notna(t.get('truth_x0')):
                tx: float = float(t['truth_x0']) + float(t['truth_w0']) / 2.0
                ty: float = float(t['truth_y0']) + float(t['truth_h0']) / 2.0
            else:
                tx, ty = cx, cy
            hit = first_rank_within(same, tx, ty, tol_l0)
        row['s2_hit_rank'] = hit

        r3: pd.DataFrame = ranked(s3.get(i))
        err: float = math.inf
        if len(r3):
            r3 = r3.copy()
            r3['confidence'] = pd.to_numeric(r3['confidence'], errors='coerce').fillna(0.0)
            r3['n_inliers'] = pd.to_numeric(r3.get('n_inliers'), errors='coerce').fillna(0.0)
            r3['err_um'] = np.hypot(r3['center_x0'] - cx, r3['center_y0'] - cy) * base_mpp
            r3['err_um'] = r3['err_um'].fillna(math.inf)
            best: pd.Series = r3.sort_values(['confidence', 'rank'], ascending=[False, True]).iloc[0]
            err = float(best['err_um'])
            row.update(answer_rank=int(best['rank']), answer_confidence=float(best['confidence']),
                       answer_n_inliers=float(best['n_inliers']), localized=bool(best['confidence'] > 0))
            if row['level_ok']:
                r3['success_flag'] = r3['success'].astype(str).eq('True') if 'success' in r3 else False
                r3['right'] = r3['err_um'] <= tol_um
                cand_frames.append(r3[['rank', 'n_inliers', 'confidence', 'success_flag', 'right']]
                                   .assign(dataset=slide['dataset'], slide=slide['name'], index=i))
        row['answer_err_um'] = err if math.isfinite(err) else np.nan
        row['success'] = err <= tol_um

        if row['success']:
            outcome: str = 'success'
        elif not row['level_ok']:
            outcome = 'unsearchable' if row['placed_unsearchable'] else 'routing_wrong'
        elif hit is None:
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


def stages_table(fov: pd.DataFrame, kmax: int) -> pd.DataFrame:
    """Each stage on the FoVs it is answerable for."""
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
        pop2: pd.DataFrame = g[g['level_ok']]
        row['s2_n'] = len(pop2)
        hit: pd.Series = pd.to_numeric(pop2['s2_hit_rank'], errors='coerce')
        for k in ks:
            row[f's2_hit@{k}'] = float((hit <= k).mean()) if len(pop2) else math.nan
        pct_rank: pd.Series = (pd.to_numeric(pop2['truth_rank_all'], errors='coerce')
                               / pd.to_numeric(pop2['pool_all'], errors='coerce'))
        row['s2_truth_rank_pct_median'] = float(pct_rank.median()) if len(pop2) else math.nan
        pop3: pd.DataFrame = pop2[pop2['s2_hit_rank'].notna()]
        row['s3_n'] = len(pop3)
        row['s3_correct'] = float(pop3['success'].mean()) if len(pop3) else math.nan
        row['s3_localized'] = float(pop3['localized'].mean()) if len(pop3) else math.nan
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


def slides_table(fov: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (dataset, slide), g in fov.groupby(['dataset', 'slide']):
        searched: pd.DataFrame = g[g['level_ok']]
        hit: pd.Series = pd.to_numeric(searched['s2_hit_rank'], errors='coerce')
        rows.append(dict(dataset=dataset, slide=slide, n=len(g), level_ok=float(g['level_ok'].mean()),
                         s2_hit_at_10=(float((hit <= 10).mean()) if len(searched) else math.nan),
                         success=float(g['success'].mean())))
    return pd.DataFrame(rows).sort_values(['dataset', 'success']).reset_index(drop=True)


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
           extra: Dict[str, float], slides: pd.DataFrame, oracle: Optional[pd.DataFrame],
           kmax: int) -> str:
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
        L.append(f'  {"":<28}{"":>6}  success 95% interval over slides: {pct(lo)} - {pct(hi)}')
    L.append(f'  stage 1 chose a wrong level on {int((~fov["routing_ok"]).sum())} FoVs; '
             f'{int((fov["success"] & ~fov["routing_ok"]).sum())} of them are successes all the same, '
             f'{int((fov["outcome"] == "routing_wrong").sum())} were lost to it')
    L += ['', '  by the ds the photo was placed at', f'  {"":<28}{"n":>6}{cols}']
    for _, r1 in by_rung.iterrows():
        L.append(f'  {r1["dataset"] + " ds=" + format(r1["rung"], "g"):<28}{int(r1["n"]):>6}'
                 + ''.join(f'{pct(r1[o + "_pct"]):>23}' for o in OUTCOMES))

    L += ['', bar, '2. STAGES   each on the FoVs it is answerable for', bar]
    for _, s in stages.iterrows():
        L += [f'  {s["dataset"]}   (n={int(s["n"])})',
              f'    stage 1   level correct {pct(s["s1_level_ok"])}   too coarse {pct(s["s1_too_coarse"])}'
              f'   too fine {pct(s["s1_too_fine"])}   no estimate {pct(s["s1_none"])}',
              f'              {int(s["s1_wrong_n"])} wrong: {int(s["s1_wrong_success"])} succeed all the same '
              f'(stage 3 localizes from a window at that level); stage 2 refused the level and went finer '
              f'on {int(s["s1_wrong_moved"])}, {int(s["s1_wrong_moved_to_placed"])} of them landing on the placed level',
              f'    stage 2   on {int(s["s2_n"])} FoVs where it searched the placed level:',
              '              ' + '   '.join(f'hit@{k} {pct(s[c])}' for c, k in
                                            ((c, c.split("@")[1]) for c in stages.columns if c.startswith('s2_hit@')))
              + f'   truth window rank / all windows, median {100 * s["s2_truth_rank_pct_median"]:.2f}%',
              f'    stage 3   on {int(s["s3_n"])} FoVs stage 2 hit:  answer within tolerance {pct(s["s3_correct"])}'
              f'   (a fit was found for {pct(s["s3_localized"])})']
    L.append(f'  (hit@{kmax} = anywhere in the candidates stage 2 returned, which is all stage 3 sees)')

    L += ['', bar, '3. INLIERS   stage 3 by inlier count (candidates of the FoVs whose level was searched)', bar,
          f'  {"inliers":<10}{"candidates":>12}{"at truth":>10}{"% at truth":>12}{"answers":>10}{"right":>8}{"% right":>10}']
    for _, b in inl.iterrows():
        L.append(f'  {b["inliers"]:<10}{int(b["candidates"]):>12}{int(b["candidates_right"]):>10}'
                 f'{pct(b["candidates_right_pct"]):>12}{int(b["answers"]):>10}{int(b["answers_right"]):>8}'
                 f'{pct(b["answers_right_pct"]):>10}')
    L += [f'  inlier count separates a candidate at the truth from one that is not: AUC {extra["auc"]:.3f}'
          f'   ({int(extra["n_right"])} of {int(extra["n_candidates"])} candidates are at the truth)',
          f'  the fit counts as success from {extra["flag_min_inliers"]:.0f} inliers: of the candidates it accepts '
          f'{pct(extra["flag_precision"])} are at the truth, and it accepts {pct(extra["flag_recall"])} '
          f'of those that are']

    L += ['', bar, '4. SLIDES   (sorted by success within a dataset)', bar,
          f'  {"slide":<26}{"n":>6}{"placed level searched":>23}{"stage 2 hit@10":>16}{"success":>10}']
    for _, w in slides.iterrows():
        L.append(f'  {w["slide"]:<26}{int(w["n"]):>6}{pct(w["level_ok"]):>23}{pct(w["s2_hit_at_10"]):>16}'
                 f'{pct(w["success"]):>10}')

    L += ['', bar, '5. ORACLE   the FoVs lost to stage 1\'s routing, run again at the placed level', bar]
    if oracle is None or not len(oracle):
        L.append('  not in the cache (bench_locascope --route oracle): left out; nothing above depends on it')
    else:
        L.append(f'  {"":<20}{"n":>6}{"success, stage-1 route":>24}{"success, oracle route":>24}'
                 f'{"searchable":>12}{"hit@10":>9}{"hit@" + str(kmax):>9}')
        for _, o in oracle.iterrows():
            L.append(f'  {o["dataset"]:<20}{int(o["n"]):>6}{pct(o["success_stage1_route"]):>24}'
                     f'{pct(o["success_oracle_route"]):>24}{pct(o["oracle_searchable"]):>12}'
                     f'{pct(o["oracle_s2_hit_at_10"]):>9}{pct(o["oracle_s2_hit_at_k"]):>9}')
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
    for colour, (dataset, g) in zip(palette, fov.groupby('dataset')):
        pop: pd.DataFrame = g[g['level_ok']]
        if not len(pop):
            continue
        hit: pd.Series = pd.to_numeric(pop['s2_hit_rank'], errors='coerce')
        rate: List[float] = [float((hit <= k).mean()) for k in ks]
        ax.plot(ks, rate, color=colour, lw=2, label=f'{dataset}  (n={len(pop)})')
        for k in sorted({1, 10, kmax}):
            ax.scatter([k], [float((hit <= k).mean())], color=colour, zorder=3, s=26)
            ax.annotate(f'{100 * float((hit <= k).mean()):.0f}', (k, float((hit <= k).mean())),
                        textcoords='offset points', xytext=(0, 7), ha='center', fontsize=8, color=colour)
    for k in sorted({1, 10, kmax}):
        ax.axvline(k, color='#c3c2b7', lw=0.8, zorder=0)
    ax.set_xscale('log')
    marks_k: List[int] = sorted({1, 10, kmax})
    ax.set_xticks(marks_k, [str(k) for k in marks_k])
    ax.minorticks_off()
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ['0', '25', '50', '75', '100'])
    ax.set_xlabel('K  (stage 2 candidates looked at)')
    ax.set_ylabel('% of FoVs whose truth window is among them')
    ax.set_title('Stage 2 recall@K, on the FoVs where it searched the placed level\n' + head,
                 fontsize=10, loc='left', color=INK)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_inliers(cand: pd.DataFrame, inl: pd.DataFrame, extra: Dict[str, float], head: str, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax: Any = axes[0]
    top: float = float(max(cand['n_inliers'].max(), 2.0))
    edges: np.ndarray = np.geomspace(1, top + 2, 45)          # on n + 1, so that 0 inliers sits at 1
    for right, colour, name in ((True, COLOURS['success'], 'at the truth'),
                                (False, COLOURS['routing_wrong'], 'elsewhere')):
        v: np.ndarray = cand.loc[cand['right'] == right, 'n_inliers'].to_numpy() + 1.0
        if len(v):
            ax.hist(v, bins=edges, weights=np.full(len(v), 1.0 / len(v)), histtype='stepfilled',
                    alpha=0.45, color=colour, edgecolor=colour, label=f'{name}  (n={len(v)})')
    flag: float = extra['flag_min_inliers']
    if math.isfinite(flag):
        ax.axvline(flag + 1.0, color=INK, ls='--', lw=1)
        ax.text(flag + 1.0, ax.get_ylim()[1] * 0.97, f' fit counts as success\n from {flag:.0f} inliers',
                fontsize=8, va='top', color=INK)
    ticks: List[int] = [0, 1, 5, 10, 50, 100, 500, 1000, 5000]
    ticks = [t for t in ticks if t + 1 <= top + 2]
    ax.set_xscale('log')
    ax.set_xticks([t + 1 for t in ticks], [str(t) for t in ticks])
    ax.set_xlabel('stage 3 inliers of a candidate (log scale)')
    ax.set_ylabel('share of its group')
    ax.set_title(f'every candidate stage 3 verified   (AUC {extra["auc"]:.2f})', fontsize=10, loc='left', color=INK)
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    _style(ax)

    ax = axes[1]
    x: np.ndarray = np.arange(len(inl))
    share: np.ndarray = inl['answers_right_pct'].fillna(0).to_numpy(dtype=float)
    ax.bar(x, share, color=COLOURS['success'], width=0.7)
    for xi, s, n in zip(x, share, inl['answers'].to_numpy()):
        ax.text(xi, s + 0.02, f'n={int(n)}', ha='center', fontsize=8, color=INK)
    ax.set_xticks(x, inl['inliers'].astype(str).tolist(), fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ['0', '25', '50', '75', '100'])
    ax.set_xlabel('inliers of the FoV\'s answer')
    ax.set_ylabel('% of those answers within tolerance')
    ax.set_title('how far to trust an answer', fontsize=10, loc='left', color=INK)
    _style(ax)
    fig.suptitle(head, x=0.01, ha='left', fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_slides(slides: pd.DataFrame, head: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(slides) + 2.4))
    y: np.ndarray = np.arange(len(slides))[::-1]
    marks: Tuple[Tuple[str, str, str, str], ...] = (
        ('level_ok', 'placed level searched', '#D55E00', 'o'),
        ('s2_hit_at_10', 'stage 2 hit@10', '#56B4E9', 's'),
        ('success', 'success', '#009E73', 'D'))
    for yi, (_, r) in zip(y, slides.iterrows()):
        vals: List[float] = [float(r[c]) for c, _, _, _ in marks if r[c] == r[c]]
        ax.plot([min(vals), max(vals)], [yi, yi], color='#c3c2b7', lw=1.5, zorder=1)
    names: List[str] = slides['dataset'].tolist()
    for k in range(1, len(names)):                       # a rule where the dataset changes
        if names[k] != names[k - 1]:
            ax.axhline(y[k] + 0.5, color=INK, lw=0.8)
    for col, name, colour, marker in marks:
        ax.scatter(slides[col], y, color=colour, marker=marker, s=34, label=name, zorder=3)
    ax.set_yticks(y, [f'{r.slide}' for r in slides.itertuples()], fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ['0', '25', '50', '75', '100'])
    ax.set_xlabel('% of the slide\'s FoVs')
    ax.grid(axis='x', alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    ax.set_title(head, fontsize=10, loc='left', color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
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
    ap.add_argument('--tol-um', type=float, default=100.0,
                    help='the final answer is a success when its centre is this close to the truth')
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
        rows, frames = evaluate(slide, render, got, routed=True, tol_um=args.tol_um, tile=tile, padding=padding)
        fov_rows += rows
        cand_frames += frames
        oracle_got: Optional[Tables] = load_route(base, stages, ORACLE, mask_cfg, args.limit, name) \
            if (base.at(stage1=ORACLE).entry('stage2').dir.is_dir()) else None
        if oracle_got is not None:
            oracle_slides += 1
            orows, _ = evaluate(slide, render, oracle_got, routed=False, tol_um=args.tol_um,
                                tile=tile, padding=padding)
            oracle_rows += orows
    if not fov_rows:
        raise SystemExit('nothing in the cache for these flags (is the stage-1 route there?)')

    fov: pd.DataFrame = pd.DataFrame(fov_rows)
    cand: pd.DataFrame = (pd.concat(cand_frames, ignore_index=True) if cand_frames else
                          pd.DataFrame(columns=['rank', 'n_inliers', 'confidence', 'success_flag', 'right']))
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
    slides_df: pd.DataFrame = slides_table(fov)
    oracle_df: Optional[pd.DataFrame] = (oracle_table(fov, oracle_fov, kmax)
                                         if len(oracle_fov) else None)

    header: List[str] = [
        f'analyze_locascope   cache job {job}   split {args.split}   tolerance {args.tol_um:g} um   '
        f'stage 2 returns up to {kmax} candidates',
        f'slides {fov["slide"].nunique()} of {n_slides}   FoVs {len(fov)}'
        + (f'   not in the cache: {", ".join(missing)}' if missing else '')
        + f'   oracle route for {oracle_slides} slides']
    text: str = report(fov, header, funnel, by_rung, cis, stages_df, inl, extra, slides_df, oracle_df, kmax)
    print(text)
    (out / 'report.txt').write_text(text + '\n')

    fov.to_csv(out / 'fov.csv', index=False)
    funnel.to_csv(out / 'funnel.csv', index=False)
    by_rung.to_csv(out / 'funnel_by_rung.csv', index=False)
    stages_df.to_csv(out / 'stages.csv', index=False)
    inl.to_csv(out / 'inliers.csv', index=False)
    slides_df.to_csv(out / 'slides.csv', index=False)
    if oracle_df is not None:
        oracle_df.to_csv(out / 'oracle.csv', index=False)

    head: str = f'{job}, {args.split}, tolerance {args.tol_um:g} um, {len(fov)} FoVs on {fov["slide"].nunique()} slides'
    fig_funnel(funnel, by_rung, head, out / 'funnel.png')
    fig_recall(fov, kmax, head, out / 'recall.png')
    if len(cand):
        fig_inliers(cand, inl, extra, head, out / 'inliers.png')
    fig_slides(slides_df, head, out / 'slides.png')
    if oracle_df is not None:
        fig_oracle(oracle_df, kmax, head, out / 'oracle.png')
    print(f'\n-> {out}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
