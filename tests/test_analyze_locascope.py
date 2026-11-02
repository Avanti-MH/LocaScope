#!/usr/bin/env python3
"""Tests for the analyses in `utilities/cli/metrics/`: `analyze_locascope.py` -- how a stage-2 candidate
list is judged (a hit), and that the tables, the report and every figure run on a table of FoVs -- and the
figures of `analyze_stage1_metrics.py` (the `s1` tests).

    python utilities/test_modules/test_analyze_locascope.py

No slide, no cache, no GPU: candidate lists are small frames, the FoV table is made up, and the stage-1
table is made-up rows with the columns the real one has (two datasets with the pyramids of BRACS and Ki67,
a knn and a classic baseline, classifier and prototype checkpoints each under every vote rule, the `fov_*`
and `risk_*` numbers) that go through the same functions `main` calls.

WHAT THIS DEFENDS
-----------------
    hit        the window of either answer (the truth row's main and offset), or, the suppression having
               dropped it, a window of its rotation within the tolerance (one tile of the placed level)
               that kept its place -- `BenchCommon.listed_places`, the bench's own rule: a window a tile
               away is not near, one a half-tile diagonal away is; a window of any other rotation is not
               a hit, and with no answers nothing is
    tree       the nodes are stage 1's right / coarse / fine and the levels it is off by, and the leaves of a node
               (located, lost at stage 1, 2 or 3) add up to it; every table, the
               text report and every figure run on a table of FoVs
    s1 levels  one pyramid level is a x4 step of ds on BRACS and x2 on Ki67; an estimate that is 4x off is
               one level too fine on the first and two on the second; a right answer is `right`
    s1 names   a label fits on a figure, the vote rule and the repeated encoder dropped
    s1 methods the methods a figure draws are the best `top` voting checkpoints and every baseline
    s1 figures every figure is written (methods, by rung, levels off, native vs resampled, a confusion
               matrix and a per-slide map per dataset, the vote rules and their pairwise comparison,
               risk-coverage, and per dataset and kind the risk flags, the strata and the patch statistics)
"""

from __future__ import annotations

import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.join(_HERE, '..', 'cli', 'metrics'))
sys.path.insert(0, os.path.join(_HERE, '..', 'bench_modules'))

from _paths import setup_import_paths                            # noqa: E402

setup_import_paths()

import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402

import analyze_locascope as al                                   # noqa: E402
from BenchCommon import Answer, Answers                          # noqa: E402
import analyze_stage1_metrics as s1                              # noqa: E402

_RESULTS: List[Tuple[str, Optional[BaseException]]] = []

TILE: int = 256
DS: float = 2.0
STEP: float = TILE * DS                      # 512 level-0 px: a tile at ds 2
W0: float = 5 * TILE * DS                    # a 5 x 4 tile window at ds 2
H0: float = 4 * TILE * DS
BASE: Tuple[float, float] = (1000.0, 2000.0)


def check(name: str, fn: Callable[[], Optional[str]]) -> None:
    try:
        out: Optional[str] = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   {out}' if out else ''))
    except Exception as e:                                       # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n          {type(e).__name__}: {e}')


def win(rank: int, lattice: str, row: int, col: int, rotation: int = 0, region: int = 0) -> Dict[str, Any]:
    """A candidate on the lattices of `BASE`: the offset lattice sits half a tile down and right."""
    shift: float = STEP / 2.0 if lattice == 'offset' else 0.0
    return dict(rank=rank, region=region, lattice=lattice, row=row, col=col, rotation=rotation,
                x0=BASE[0] + shift + col * STEP, y0=BASE[1] + shift + row * STEP, w0=W0, h0=H0, ds=DS)


def cands(*rows: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def answer(lattice: str, row: int, col: int) -> Answer:
    """An answer on the lattices of `BASE`, at rotation 0, where `win` puts that window."""
    w: Dict[str, Any] = win(0, lattice, row, col)
    return Answer(lattice, 0, row, col, 0, (int(w['x0']), int(w['y0']), W0, H0), 0.0)


#: the answers: the main window (row 2, col 3) and the offset one beside it
FOUND: Answers = Answers(answer('main', 2, 3), answer('offset', 1, 2))


def t_a_a_hit_is_either_answer_or_the_window_that_kept_its_place() -> str:
    far = win(1, 'main', 10, 10)
    exact, near, a_tile = win(5, 'main', 2, 3), win(3, 'offset', 2, 3), win(4, 'main', 2, 5)
    assert al.stage2_hit(cands(far, exact), FOUND, STEP) == 5
    assert al.stage2_hit(cands(far, near, exact), FOUND, STEP) == 3, \
        'a window a half-tile diagonal away (362 px < 512) kept the place of the main answer'
    assert al.stage2_hit(cands(far, a_tile, win(6, 'main', 2, 3)), FOUND, STEP) == 6
    assert al.stage2_hit(cands(far, a_tile), FOUND, STEP) is None, 'two tiles away is not near'
    assert al.stage2_hit(cands(far, win(2, 'main', 3, 3)), FOUND, STEP) is None, \
        'one tile away (512 px, not less) is not near either'
    assert al.stage2_hit(cands(far, win(7, 'offset', 1, 2)), FOUND, STEP) == 7, 'the offset answer is a hit'
    return 'exact, near, the offset answer, and not near'


def t_b_the_tolerance_is_the_one_given() -> str:
    near = win(3, 'offset', 2, 3)                      # 362 px from the main answer
    assert al.stage2_hit(cands(win(1, 'main', 10, 10), near), FOUND, 512.0) == 3
    assert al.stage2_hit(cands(win(1, 'main', 10, 10), near), FOUND, 256.0) is None
    assert al.stage2_hit(cands(win(1, 'main', 10, 10), win(4, 'main', 2, 3)), FOUND, 0.0) == 4, \
        'the answer itself is a hit at any tolerance'
    return '362 px: a hit at 512, not at 256'


def t_c_a_window_of_another_rotation_is_not_a_hit() -> str:
    assert al.stage2_hit(cands(win(1, 'main', 2, 3, rotation=90), win(2, 'main', 10, 10)), FOUND, STEP) is None
    assert al.stage2_hit(cands(win(1, 'main', 2, 3, rotation=90), win(2, 'main', 2, 3)), FOUND, STEP) == 2
    assert al.stage2_hit(pd.DataFrame(), FOUND, STEP) is None
    assert al.stage2_hit(cands(win(1, 'main', 2, 3)), None, STEP) is None, 'no answers: nothing to judge on'
    return 'upright rotation only'


def made_up_fov(n: int = 90) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """A table of FoVs with the columns `evaluate` makes, and a table of stage-3 candidates."""
    rng: np.random.Generator = np.random.default_rng(0)
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        dataset: str = 'bracs/test' if i % 2 == 0 else 'ki67_with_photo'
        diff: int = int(rng.choice([0, 0, 0, 1, 2, 3, -1, -2, -3]))
        hit: Optional[int] = int(rng.integers(1, 100)) if rng.random() < 0.7 else None
        located: bool = hit is not None and rng.random() < 0.8
        level_ok: bool = diff == 0 or rng.random() < 0.2
        outcome: str = ('success' if located else ('routing_wrong' if not level_ok else
                                                   ('s2_miss' if hit is None else 's3_wrong')))
        rows.append(dict(
            dataset=dataset, slide=f'S{i % 4}', index=i, level=1, ds=4.0, rung=4.0, native=True,
            chosen_level=1 + diff, routing_ok=diff == 0, s1_level_diff=float(diff), searched_level=1,
            n_cand=100, level_ok=level_ok, moved=False, placed_unsearchable=False,
            truth_rank_all=float(rng.integers(1, 1000)), pool_all=100000.0, s2_hit_rank=hit,
            s2_reach_rank=hit,
            answer_rank=1, answer_confidence=1.0, answer_n_inliers=30.0, answer_n_good=60.0,
            answer_ratio=0.5, localized=located, answer_err_px=3.0 if located else 900.0, answer_err_um=10.0,
            success=located, outcome=outcome))
    fov: pd.DataFrame = pd.DataFrame(rows)
    m: int = 600
    right: np.ndarray = rng.random(m) < 0.2
    good: np.ndarray = np.where(right, rng.integers(30, 3000, m), rng.integers(0, 40, m)).astype(float)
    inliers: np.ndarray = np.where(right, good * rng.uniform(0.3, 0.9, m), rng.integers(0, 9, m)).astype(float)
    cand: pd.DataFrame = pd.DataFrame(dict(
        rank=rng.integers(1, 100, m), n_inliers=inliers, n_good=good,
        ratio=np.where(good > 0, inliers / np.maximum(good, 1.0), np.nan), confidence=inliers / 100.0,
        success_flag=inliers >= 10, right=right, err_px=np.where(right, 4.0, 700.0)))
    return fov, cand


def t_d_tree_tables_report_and_figures_run() -> str:
    fov, cand = made_up_fov()
    kmax: int = 100
    tree: pd.DataFrame = al.tree_table(fov, kmax)
    assert {'all', 'correct', 'coarse', '+1', '+2', '+3', 'fine', '-1', '-2', '-3'} <= set(tree['node']), set(tree['node'])
    for dataset, t in tree.groupby('dataset'):
        by: pd.Series = t.set_index('node')['n']
        assert by['all'] == int((fov['dataset'] == dataset).sum())
        assert by['correct'] + by['coarse'] + by['fine'] == by['all'], dict(by)
        assert by['+1'] + by['+2'] + by['+3'] == by['coarse'] and by['-1'] + by['-2'] + by['-3'] == by['fine']
    assert not [c for c in tree.columns if 'soft' in c or 'hard' in c], list(tree.columns)
    lost = tree['located_n'] + tree['n_fail_s1'] + tree['n_fail_s2'] + tree['n_fail_s3_no_fit'] \
        + tree['n_fail_s3_wrong_place']
    assert (lost == tree['n']).all(), 'the leaves of a node (located, lost at stage 1, 2, 3) add up to the node'
    assert (tree['n_located_without_hit'] <= tree['located_n']).all()
    assert (tree['hit@100'] >= tree['hit@10']).all() and (tree['hit@10'] >= tree['hit@1']).all()
    stages: pd.DataFrame = al.stages_table(fov, kmax)
    slides: pd.DataFrame = al.slides_table(fov, kmax)
    inl, extra = al.inliers_table(cand, fov)
    istats: pd.DataFrame = al.inlier_stats(cand)
    assert set(istats['quantity']) == {'inliers', 'matches', 'inliers / matches'}
    funnel: pd.DataFrame = al.funnel_table(fov.assign(kind='all'), ['dataset', 'kind'])
    by_rung: pd.DataFrame = al.funnel_table(fov, ['dataset', 'rung'])
    cis: Dict[str, Tuple[float, float]] = {d: (0.1, 0.9) for d in fov['dataset'].unique()}
    text: str = al.report(fov, ['header'], funnel, by_rung, cis, stages, inl, extra, istats, slides, tree, None,
                          kmax, 16.0)
    for part in ('6. TREE', 'hit@100', 'hit n', 'located/hit', 'lost@1', 'inliers / matches', '16 px'):
        assert part in text, part
    assert 'soft' not in text.lower() and 'hard' not in text.lower(), 'nothing is called soft or hard any more'
    with tempfile.TemporaryDirectory() as root:
        out: Path = Path(root)
        al.fig_funnel(funnel, by_rung, 'head', out / 'funnel.png')
        al.fig_recall(fov, kmax, 'head', out / 'recall.png')
        al.fig_inliers(cand, inl, extra, 'head', 16.0, out / 'inliers.png')
        al.fig_slides(slides, kmax, 16.0, 'head', out / 'slides.png')
        al.fig_tree(tree, kmax, 16.0, 'head', out)
        made: List[str] = sorted(p.name for p in out.glob('*.png'))
        assert made == ['funnel.png', 'inliers.png', 'recall.png', 'slides.png', 'tree_bracs_test.png',
                        'tree_ki67_with_photo.png'], made
        assert all((out / m).stat().st_size > 5000 for m in made)
    return f'{len(tree)} tree rows, report {len(text.splitlines())} lines, 6 figures'


# ── analyze_stage1_metrics ────────────────────────────────────────────────────

RULES: Tuple[str, ...] = ('hard_majority', 'mean_probability', 'median_log_rung', 'patch_class_median:lower',
                          'patch_class_median:upper', 'quality_weighted:agree', 'quality_weighted:tissue',
                          'sum_log_probability')
RUNGS_OF: Dict[str, Tuple[float, ...]] = {'bracs/test': (1.0, 4.0, 16.0), 'ki67_with_photo': (1.0, 2.0, 4.0, 8.0, 16.0)}


def made_up_rows() -> List[Dict[str, str]]:
    """Rows in the shape of `stage1_val.csv`, every cell a string."""
    rng = random.Random(0)
    rows: List[Dict[str, str]] = []
    methods: List[Dict[str, Any]] = [
        dict(kind='knn', encoder='gigapath', classifier='', p=0.45, votes=('',)),
        dict(kind='classic', encoder='classic', classifier='', p=0.2, votes=('',))]
    for i, p in enumerate((0.9, 0.8, 0.7)):
        methods.append(dict(kind='classifier', encoder='uni2', classifier=f'uni2_frozen_mlp_{i}_best', p=p,
                            votes=RULES))
    for i, p in enumerate((0.7, 0.55)):
        methods.append(dict(kind='prototype', encoder='uni2', classifier=f'uni2_proto_{i}_native', p=p, votes=RULES))
    for dataset, rungs in RUNGS_OF.items():
        step = rungs[1] / rungs[0]
        for slide in ('A', 'B'):
            for rung in rungs:
                for fov in range(8):
                    base = dict(dataset=dataset, wsi_name=f'{dataset[:3]}_{slide}', x=str(fov), y=str(fov),
                                rung=str(rung), gt_ds=str(rung), gt_mpp=str(0.25 * rung),
                                native=str(rung != 2.0 or fov % 2 == 0), split='val')
                    fov_cols = dict(
                        fov_n_patches='20', fov_n_classes='6', fov_agree_frac=str(rng.random()),
                        fov_n_distinct=str(rng.randint(1, 4)), fov_patch_entropy=str(rng.random()),
                        fov_pooled_entropy=str(rng.random()), fov_pooled_margin=str(rng.random()),
                        fov_argmax_log2_spread=str(rng.random() * 2), fov_gt_reachable='True',
                        fov_gt_prob=str(rng.random()), fov_gt_rank='0', fov_gt_patch_frac=str(rng.random()))
                    for m in methods:
                        for vote in m['votes']:
                            right = rng.random() < m['p'] * (0.9 + 0.2 * RULES.index(vote) / 8 if vote else 1.0)
                            off = 0 if right else rng.choice((-1, 1))
                            est = max(0.5, rung * step ** off)
                            risk = dict(
                                risk_winner_support=str(rng.random()), risk_winner_prob=str(rng.random()),
                                risk_mean_not_mode=str(int(rng.random() < 0.2)), risk_vote_tie=str(int(rng.random() < 0.05)),
                                risk_margin_majority_p50=str(rng.random()), risk_margin_dissent_p50=str(rng.random()),
                                risk_median_split=str(int(rng.random() < 0.1)), risk_top1_p50=str(rng.random()),
                                risk_support_count=str(rng.randint(0, 5)), risk_min_p_winner=str(rng.random() * 0.1),
                                risk_loo_unstable=str(int(rng.random() < 0.1)), risk_flip=str(int(rng.random() < 0.1)),
                                risk_corr_w_e=str(rng.random() * 2 - 1), risk_ess_frac=str(rng.random())) if vote else {}
                            rows.append(dict(base, encoder=m['encoder'], classifier=m['classifier'], reduction='',
                                             loss='', read_level='', kind=m['kind'], vote=vote,
                                             estimator_id=f'{m["kind"]}-{m["classifier"] or m["encoder"]}',
                                             estimated_ds=str(est), estimated_mpp=str(0.25 * est),
                                             chosen_ds=str(est), chosen_mpp=str(0.25 * est), chosen_level='0',
                                             **(fov_cols if vote else {}), **risk))
    return rows


def t_s1_a_a_level_is_one_pyramid_step_on_each_dataset() -> str:
    rows = made_up_rows()
    assert s1.ladder_step(rows, 'bracs/test') == 4.0 and s1.ladder_step(rows, 'ki67_with_photo') == 2.0
    row = dict(rung='4.0', estimated_ds='1.0', gt_ds='4.0')
    assert s1.level_bucket(row, 4.0) == '-1', 'BRACS: 4x too fine is one level'
    assert s1.level_bucket(row, 2.0) == '<=-2', 'Ki67: 4x too fine is two levels'
    assert s1.level_bucket(dict(rung='4.0', estimated_ds='16.0'), 4.0) == '+1'
    assert s1.level_bucket(dict(rung='4.0', estimated_ds='4.0'), 4.0) == 'right'
    assert s1.level_bucket(dict(rung='8.0', estimated_ds='16.0'), 2.0) == '+1'
    return 'x4 on BRACS, x2 on Ki67'


def t_s1_b_labels_fit_and_the_methods_drawn_are_the_best_and_every_baseline() -> str:
    assert s1.short_name('uni2+uni2_frozen_mlp_deep_best@mean_probability') == 'uni2 frozen mlp deep'
    assert s1.short_name('gigapath+pure-train_Ki67--gigapath--frozen--mix_attn_linear--ord_a--best@mean_probability') \
        == 'gigapath frozen mix_attn_linear ord_a'
    assert s1.short_name('gigapath+pure-train_Ki67--gigapath--frozen--mlp--bal--best_unweighted') \
        == 'gigapath frozen mlp bal best_unweighted'
    assert s1.short_name('knn:gigapath') == 'knn:gigapath'
    rows = made_up_rows()
    view3 = s1.overall(rows)
    kinds = {s1.base_method_of(r): s1.kind_of(r) for r in rows}
    picked = s1.pick_methods(view3, kinds, 'bracs/test', top=2)
    voting = [m for m in picked if m['kind'] in ('classifier', 'prototype')]
    baselines = [m for m in picked if m['kind'] in ('knn', 'classic')]
    assert len(voting) == 2 and len(baselines) == 2, [(m['base'], m['kind']) for m in picked]
    accs = [m['acc'] for m in picked]
    assert accs == sorted(accs, reverse=True), accs
    assert all(m['best_acc'] >= m['acc'] - 1e-12 for m in voting)
    return f'{len(picked)} drawn: ' + ', '.join(s1.short_name(m['base'])[:22] for m in picked)


def t_s1_bb_every_dataset_shows_the_same_methods() -> str:
    rows = made_up_rows()
    view3 = s1.overall(rows)
    kinds = {s1.base_method_of(r): s1.kind_of(r) for r in rows}
    shared = s1.pick_shared(view3, kinds, ['bracs/test', 'ki67_with_photo'], top=3)
    sets = {d: {m['base'] for m in ms} for d, ms in shared.items()}
    assert sets['bracs/test'] == sets['ki67_with_photo'], sets
    assert sum(1 for m in shared['bracs/test'] if m['kind'] in ('classifier', 'prototype')) == 3
    assert {m['base'] for m in shared['bracs/test'] if m['kind'] in ('knn', 'classic')} == {'knn:gigapath', 'classic'}
    return 'the same 3 + 2 baselines on both'


def t_s1_c_every_figure_is_written() -> str:
    rows = made_up_rows()
    shown = [r for r in rows if s1.is_default_vote(r)]
    view1, view2, view3 = s1.per_slide_rung(shown), s1.cross_slide_rung(shown), s1.overall(rows)
    paired = s1._paired(rows)
    acc = s1.vote_accuracy(rows)
    tables = dict(acc=acc, summary=s1.vote_summary(acc), pairwise=s1.vote_pairwise(paired),
                  strata=s1.vote_strata(rows, by_method=False), distribution=s1.fov_distribution(rows),
                  rc=s1.risk_coverage(rows), conf=s1.confusion(rows))
    taus = s1.fit_thresholds(rows, 'made_up.csv')['values']
    with tempfile.TemporaryDirectory() as root:
        written = s1.make_figures(rows, view1, view2, view3, tables, taus, root, 'stage1_val', top=4)
        names = sorted(os.path.basename(p) for p in written)
        want = ['methods', 'by_rung', 'levels_off', 'native_vs_resampled', 'votes', 'votes_pairwise', 'risk_coverage']
        for stem in want:
            assert f'stage1_val_fig_{stem}.png' in names, (stem, names)
        for d in ('bracs_test', 'ki67_with_photo'):
            for stem in ('confusion', 'per_slide'):
                assert f'stage1_val_fig_{stem}_{d}.png' in names, (stem, d, names)
            for kind in ('classifier', 'prototype'):
                for stem in ('risk_flags', 'strata', 'patch_stats'):
                    assert f'stage1_val_fig_{stem}_{d}_{kind}.png' in names, (stem, d, kind, names)
        small = [p for p in written if not os.path.isfile(p) or os.path.getsize(p) < 5000]
        assert not small, small
    return f'{len(written)} figures'


def t_s1_d_a_test_set_finds_the_thresholds_its_val_set_wrote() -> str:
    """`--fit-thresholds` on a val set writes `stage1_<set>_thresholds.json`; `--thresholds auto` on the
    test set looks for exactly that file."""
    val: List[str] = ['@legacy-val']
    test: List[str] = ['@legacy-test']
    assert s1.is_val_set(' '.join(val)) and not s1.is_val_set(' '.join(test))
    assert s1.val_set_of(test) == val
    written: str = f'stage1_{s1.set_name(val)}_thresholds.json'
    assert written == 'stage1_legacy-val_thresholds.json', written
    assert s1.val_set_of(['@legacy-test', '@pure-test']) == ['@legacy-val', '@pure-val']
    for no in (['@legacy-val'], ['bracs/test'], ['@legacy-test', 'bracs/test'], []):
        assert s1.val_set_of(no) is None, no
    return f'{test[0]} -> {written}; a set that is not all -test has none'


_TESTS =[t for n, t in sorted(globals().items(), key=lambda kv: kv[0])
          if n.startswith('t_')]


def main() -> int:
    for fn in _TESTS:
        check(fn.__name__[4:].replace('_', ' '), fn)
    failed: List[str] = [n for n, e in _RESULTS if e is not None]
    print(f'\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed')
    if failed:
        print('failed: ' + ', '.join(failed))
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
