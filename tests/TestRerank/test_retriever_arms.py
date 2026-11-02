#!/usr/bin/env python3
"""Tests for the arms of the stage 2 retriever bench that are not the streamed sliding window:
`utilities/bench_modules/RetrieverArms.py` (how the two modes of an arm make the bench's ranks)
and the report side of `bench_stage2_retriever.py` (rows of every family in one CSV, one report
with a block per mode).

    python utilities/test_modules/TestRerank/test_retriever_arms.py

No slide, no model, no GPU: lists and maps are arrays, the "slide" is a pool size.

WHAT THIS DEFENDS
-----------------
    listed     the place of an answer in a list is the position of its window, or of a window of its
               rotation within the suppression radius that stands for it when the suppression dropped
               it (the radius is one tile: nothing farther counts, nor another rotation, nor with
               radius 0 anything but the window); a shifted window counts where it was moved to; an
               answer with none is pool + 1; the record (`rank_main`) is the main answer's (`BenchCommon.listed_places`)
    ranked     1 + the windows of the level scoring strictly higher, among EVERY window of the maps
               (`BenchCommon.ranked_ranks`); a window no map holds is pool + 1
    pools      the level's windows are counted off the first phase's window scores
    operations every operation of a reranker is its own config, column and entry
    engine     the arms are the tree of their phases, a phase shared by the arms that start with it (`@K`
               is a consumer's); every phase is the entry the pipeline names for the same chain; the
               streamed pass owns the sliding window's; two columns of one reranker are one scoring; only
               what an entry to write stands on is run; a chain is judged on its first phase's truth
    rows       an arm's rows have the fields `RetrievalReport` reads, go through a CSV and
               back, and come out of one report with a block per mode, each paired with the
               baseline of its own mode; an older CSV (no `mode`, no `family`) is ranked pooling arms,
               and its `whole` and `production` are the ranked and the listed mode
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'bench_modules'))

from _paths import setup_import_paths                            # noqa: E402

setup_import_paths()

import torch                                                     # noqa: E402

import bench_stage2_retriever as bench                           # noqa: E402
from BenchCommon import (Answer, Answers, Pools, listed_places, listed_windows, pools_of,   # noqa: E402
                         ranked_ranks)
from test_context import TILE, cset                              # noqa: E402
from stage2_retrieval.StageInterface import Candidate, CandidateSet, ShiftedCandidate   # noqa: E402
from dump_function.RetrievalReport import paired_stats, truth_rank   # noqa: E402

_RESULTS: List[Tuple[str, Optional[BaseException]]] = []

Key = Tuple[int, str, int, int, int]

#: the answers: a main window and the offset window beside it, rotation 0, region 0
MAIN: Key = (0, 'main', 3, 4, 0)
OFFSET: Key = (0, 'offset', 3, 3, 0)


def check(name: str, fn: Callable[[], Optional[str]]) -> None:
    try:
        out: Optional[str] = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   {out}' if out else ''))
    except Exception as e:                                       # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n          {type(e).__name__}: {e}')


POOLS: Pools = Pools(main=1000, offset=900)
DS: float = 2.0
#: the suppression radius, one tile: what the stage's `min_sep_tiles` 1.0 means in level-0 px
TOL: float = float(TILE) * DS


def listed_of(*windows: Candidate) -> CandidateSet:
    """A list as a phase hands it on: the windows in order, in a frame whose tiles are TILE px at ds `DS`."""
    return cset(list(windows), ds=DS)


def window(lattice: str, row: int, col: int, rotation: int = 0, region: int = 0) -> Candidate:
    return Candidate(region, lattice, row, col, rotation, 0.0)


def answer_at(key: Key, frame: CandidateSet) -> Answer:
    """The answer the window `key` is, where `frame` puts it."""
    x, y = frame.origin_l0(Candidate(*key, 0.0))
    return Answer(key[1], key[0], key[2], key[3], key[4], (x, y, 0.0, 0.0), 0.0)


#: the two answers, in the frame every list below is in
ANSWERS: Answers = Answers(answer_at(MAIN, cset([], ds=DS)), answer_at(OFFSET, cset([], ds=DS)))


def ranks_of(listed: CandidateSet, tol: float = TOL) -> Dict[str, int]:
    return listed_places(listed_windows(listed), ANSWERS, POOLS, tol)


def t_a_the_places_of_the_answers_in_a_list() -> str:
    both: CandidateSet = listed_of(window('main', 0, 0), window('offset', 9, 9), window('main', 3, 4),
                                   window('offset', 3, 3))
    r: Dict[str, int] = ranks_of(both, tol=100.0)          # a radius too small for one to stand for the other
    assert r['rank_main'] == 3, r                           # THE answer is the main one: its place
    assert r['rank_main_in_all'] == 3 and r['rank_offset_in_all'] == 4, r
    wide: Dict[str, int] = ranks_of(both)
    assert wide['rank_main_in_all'] == 3 and wide['rank_offset_in_all'] == 3, 'one window stands for both'
    only_main: Dict[str, int] = listed_places(listed_windows(both), ANSWERS.main_only(), POOLS, 100.0)
    assert set(only_main) == {'rank_main', 'rank_main_in_all'}, 'a main-only arm has no offset answer'
    return str(r)


def t_b_a_window_that_stands_for_a_dropped_answer_counts() -> str:
    # the main answer was dropped; the offset window one half-tile away (362 px < 512) kept the place
    listed: CandidateSet = listed_of(window('main', 0, 0), window('offset', 3, 3), window('main', 9, 9))
    r: Dict[str, int] = ranks_of(listed)
    assert r['rank_main'] == 2 and r['rank_main_in_all'] == 2, r
    exact: Dict[str, int] = ranks_of(listed, tol=0.0)
    assert exact['rank_main'] == POOLS.main + 1 and exact['rank_main_in_all'] == POOLS.all + 1, \
        'with no suppression radius only the window itself is the answer'
    assert exact['rank_offset_in_all'] == 2, 'the offset answer is in the list'
    return 'the offset window stands for the main answer: place 2; with radius 0 it is not listed'


def t_c_a_window_of_another_rotation_or_too_far_does_not() -> str:
    other_rotation: CandidateSet = listed_of(window('main', 3, 4, rotation=90), window('offset', 3, 3, rotation=180))
    assert ranks_of(other_rotation)['rank_main_in_all'] == POOLS.all + 1, 'a turned window is another hypothesis'
    far: CandidateSet = listed_of(window('main', 3, 8), window('main', 9, 4))
    r: Dict[str, int] = ranks_of(far)
    assert r['rank_main'] == POOLS.main + 1 and r['rank_offset_in_all'] == POOLS.all + 1, r
    other_region: CandidateSet = listed_of(window('main', 3, 4, region=1))
    assert ranks_of(other_region)['rank_main_in_all'] == POOLS.all + 1, 'another region is another place'
    return 'rotation, distance and region are all kept'


def t_d_a_shifted_window_counts_where_it_was_moved_to() -> str:
    # (3, 5) is a whole tile (512 px) from the answer: not nearer than the radius ...
    plain: CandidateSet = listed_of(window('main', 3, 5))
    assert ranks_of(plain)['rank_main_in_all'] == POOLS.all + 1
    # ... until a reranker moves it 100 level px (200 px at ds 2) toward it
    moved: CandidateSet = listed_of(ShiftedCandidate(0, 'main', 3, 5, 0, 0.0, shift=(-100, 0)))
    assert ranks_of(moved)['rank_main_in_all'] == 1
    return 'origin_l0 includes the shift'


def t_e_an_empty_list_and_the_miss_values() -> str:
    r: Dict[str, int] = ranks_of(cset([], ds=DS))
    assert r == {'rank_main': 1001, 'rank_offset_in_offset': 901, 'rank_main_in_all': 1901,
                 'rank_offset_in_all': 1901}, r
    return 'nothing listed: pool + 1 everywhere'


def t_f_pools_are_counted_off_the_window_scores() -> str:
    maps: Dict[Tuple[int, int, str], torch.Tensor] = {
        (0, 0, 'main'): torch.zeros(4, 5), (0, 0, 'offset'): torch.zeros(3, 4),
        (90, 0, 'main'): torch.zeros(5, 4), (90, 0, 'offset'): torch.zeros(4, 3)}
    pools: Pools = pools_of(maps)
    assert pools.main == 4 * 5 + 5 * 4 and pools.offset == 3 * 4 + 4 * 3, pools
    assert pools.all == 64
    return f'main {pools.main}, offset {pools.offset}, all {pools.all}'


def t_g_ranked_ranks_count_strictly_higher_windows() -> str:
    maps: Dict[Tuple[int, int, str], torch.Tensor] = {
        (0, 0, 'main'): torch.tensor([[1.0, 5.0], [3.0, 5.0]]),
        (90, 0, 'main'): torch.tensor([[2.0]]),
        (0, 0, 'offset'): torch.tensor([[4.0]])}
    pools: Pools = Pools(main=5, offset=1)

    def at(main: Key, offset: Key) -> Answers:
        return Answers(Answer(main[1], main[0], main[2], main[3], main[4], (0, 0, 0.0, 0.0), 0.0),
                       Answer(offset[1], offset[0], offset[2], offset[3], offset[4], (0, 0, 0.0, 0.0), 0.0))

    r: Dict[str, int] = ranked_ranks(maps, at((0, 'main', 0, 1, 0), (0, 'offset', 0, 0, 0)), pools)
    assert r['rank_main'] == 1, r                      # the other 5.0 ties: optimistic
    assert r['rank_offset_in_offset'] == 1, r
    assert r['rank_main_in_all'] == 1 and r['rank_offset_in_all'] == 3, r     # two 5.0 above the 4.0
    gone: Dict[str, int] = ranked_ranks(maps, at((0, 'main', 9, 9, 0), (3, 'offset', 0, 0, 0)), pools)
    assert gone == {'rank_main': 6, 'rank_offset_in_offset': 2, 'rank_main_in_all': 7,
                    'rank_offset_in_all': 7}, gone
    main_only = {k: v for k, v in maps.items() if k[2] == 'main'}
    assert set(ranked_ranks(main_only, at((0, 'main', 0, 1, 0), (0, 'offset', 0, 0, 0)), pools)) == \
        {'rank_main', 'rank_main_in_all'}, 'no offset window scored: no offset rank'
    return 'ties optimistic, a window no map holds is pool + 1'


BASE: str = 'uni2:SlidingWindow::cls+mean'
METHOD: str = 'uni2:BagOfTiles::cls_avg+gap'
RERANK: str = 'uni2:BagOfTiles::cls_avg+gap::Dual::cls_avg+dual'


def fake_rows(arm: str, family: str, mode: str, ranks: List[Dict[str, int]], extra: Dict[str, Any]) -> List[dict]:
    """The rows the bench writes: one slide, one level, a FoV per entry of `ranks`."""
    out: List[dict] = []
    for fov, r in enumerate(ranks):
        out.append(dict(
            encoder='uni2', dataset='bracs/test', slide='S', level=0, ds=1.0, fov_id=fov,
            pool=POOLS.main, d_main=10.0, d_overlap=40.0, white_frac=0.1, bucket='b',
            rot_deg=0.0, scale=1.0, source='x', grid='all', family=family, mode=mode, arm=arm,
            rank_main=r['rank_main'], rank_overlap=r['rank_main'], truth_rot=0,
            truth_score='', rank_offset_in_offset=r['rank_offset_in_offset'],
            pool_offset=POOLS.offset, rank_main_in_all=r['rank_main_in_all'],
            rank_offset_in_all=r['rank_offset_in_all'], pool_all=POOLS.all, **extra))
    return out


def sample_rows() -> List[dict]:
    listed: Dict[str, int] = dict(rank_main=3, rank_offset_in_offset=4, rank_main_in_all=3, rank_offset_in_all=4)
    by_column: Dict[str, int] = dict(rank_main=1, rank_offset_in_offset=2, rank_main_in_all=1, rank_offset_in_all=2)
    missing: Dict[str, int] = dict(rank_main=1001, rank_offset_in_offset=901, rank_main_in_all=1901,
                                   rank_offset_in_all=1901)
    full: Dict[str, int] = dict(rank_main=7, rank_offset_in_offset=9, rank_main_in_all=11, rank_offset_in_all=12)
    near: Dict[str, int] = dict(rank_main=2, rank_offset_in_offset=3, rank_main_in_all=3, rank_offset_in_all=4)
    rows: List[dict] = fake_rows(BASE, 'pooling', 'ranked', [full] * 3, {})
    rows += fake_rows(BASE, 'pooling', 'listed', [missing] * 3, dict(list_len=2))
    rows += fake_rows(METHOD, 'method', 'ranked', [near] * 3, {})
    rows += fake_rows(METHOD, 'method', 'listed', [listed, by_column, missing], dict(list_len=4))
    rows += fake_rows(RERANK, 'reranker', 'listed', [by_column, by_column, listed], dict(list_len=4))
    return rows


def t_h_the_two_modes_pair_each_with_its_own_baseline() -> str:
    rows: List[dict] = sample_rows()
    for mode, want in ((bench.MODE_RANKED, [2, 2, 2]), (bench.MODE_LISTED, [3, 1, 1001])):
        subset: List[dict] = [r for r in rows if bench.mode_of(r) == mode]
        paired: List[dict] = bench.attach_baseline(subset, BASE)
        assert len(paired) == len(subset), 'every row finds the baseline of its mode'
        mine: List[dict] = [r for r in paired if r['arm'] == METHOD]
        assert [truth_rank(r) for r in mine] == want, (mode, [truth_rank(r) for r in mine])
    ranked: Dict[str, Any] = paired_stats([r for r in bench.attach_baseline(
        [r for r in rows if bench.mode_of(r) == bench.MODE_RANKED], BASE) if r['arm'] == METHOD])
    assert (ranked['W'], ranked['L'], ranked['T']) == (3, 0, 0), ranked     # 2 beats the baseline's 7
    return 'W/L/T 3/0/0 ranked'


def t_i_a_report_runs_over_both_modes_and_a_csv() -> str:
    rows: List[dict] = sample_rows()
    with tempfile.TemporaryDirectory() as tmp:
        path: Path = Path(tmp) / 'stage2_retriever.csv'
        bench.write_csv(rows, path)
        back: List[dict] = bench.read_rows([path], quiet=True)
    assert len(back) == len(rows)
    assert all(isinstance(r['list_len'], int) for r in back if r['list_len'] != ''), 'list_len is a number'
    assert bench.report_arms(back) == [BASE, METHOD, RERANK]
    assert bench.pick_baseline(back) == BASE
    assert bench.check_pools(back, BASE, emit=lambda s: None) == 0
    lines: List[str] = []
    bench.tabulate(back, False, emit=lines.append)
    text: str = '\n'.join(lines)
    assert 'RANKED (NMS off)' in text and 'LISTED (NMS on)' in text, 'one block per mode'
    assert text.index('RANKED (NMS off)') < text.index('LISTED (NMS on)')
    assert METHOD in text and 'recall_all' in text and 'listed -- how much of the answer' in text
    assert 'recall_all@K of a listed arm' in text, 'the curve over list length'
    assert 'rerankers against the list they were given' in text and RERANK in text
    return f'{len(lines)} lines'


def t_j_an_older_csv_is_read_in_the_grammar() -> str:
    old: List[dict] = [{k: v for k, v in r.items() if k not in ('mode', 'family')} for r in
                       fake_rows('cls+mean', 'pooling', 'whole', [dict(rank_main=7, rank_offset_in_offset=9,
                                                                       rank_main_in_all=11, rank_offset_in_all=12)] * 2, {})]
    assert all(bench.mode_of(r) == bench.MODE_RANKED and bench.family_of(r) == 'pooling' for r in old)
    assert bench.mode_of({'mode': 'whole'}) == bench.MODE_RANKED and bench.mode_of({'mode': 'production'}) == \
        bench.MODE_LISTED, 'the older names of the modes'
    lines: List[str] = []
    bench.tabulate(old, False, emit=lines.append)
    assert old[0]['arm'] == BASE, 'cls+mean of an encoder-uni2 row is uni2:SlidingWindow::cls+mean'
    assert old[0]['mode'] == bench.MODE_RANKED, 'the report writes the mode in its own name'
    assert BASE in '\n'.join(lines)
    return 'cls+mean -> ' + old[0]['arm'] + '; whole / production -> ranked / listed'


def t_k_pools_that_disagree_are_counted() -> str:
    rows: List[dict] = sample_rows()
    for r in rows:
        if r['arm'] == RERANK and r['fov_id'] == 0:
            r['pool_all'] += 1
    assert bench.check_pools(rows, BASE, emit=lambda s: None) == 1, 'one FoV of the reranker arm'
    return 'a FoV ranked in another pool is not silently compared'


def t_l_the_listed_share_is_the_ceiling_of_a_list() -> str:
    rows: List[dict] = [r for r in sample_rows() if bench.mode_of(r) == bench.MODE_LISTED]
    lines: List[str] = []
    bench.listed_report(rows, [BASE, METHOD, RERANK], emit=lines.append)
    row: str = next(line for line in lines if line.startswith(f'| {METHOD}'))
    assert '67%' in row, row                  # 2 of the 3 FoVs have the main answer listed
    base: str = next(line for line in lines if line.startswith(f'| {BASE}'))
    assert ' 0%' in base, base                # its lists hold neither answer
    assert any('miss at every k' in line for line in lines), 'the note on what an unlisted answer is'
    return row.strip()


def t_m_the_curve_never_counts_an_unlisted_answer_as_a_hit() -> str:
    """K past the pool is the whole level, but an answer that is not listed is pool + 1, which a large K
    must not swallow."""
    rows: List[dict] = [r for r in sample_rows() if bench.mode_of(r) == bench.MODE_LISTED and r['arm'] == BASE]
    for r in rows:
        r['pool_all'] = 50
        r['rank_main_in_all'] = r['rank_offset_in_all'] = 51      # not listed in a pool of 50
    lines: List[str] = []
    bench.curve_report(rows, [BASE], emit=lines.append)
    row: str = next(line for line in lines if line.startswith(f'| {BASE}'))
    assert ' 0%' in row and '100%' not in row, row
    return row.strip()


def t_n_every_operation_of_a_reranker_is_its_own_config_column_and_entry() -> str:
    """The bug this defends: the engine cached one reranker per (method, pooling, encoder) and read the
    column of the FIRST operation to be built for every later one, so Corr+diag, +contrast, +mnn and
    +sinkhorn (and Dual, CandGraph) came out as one list four times. Now an operation is the config's: each
    is its own config, orders by its own column and is its own entry."""
    from stage2_retrieval import retriever_class
    from stage2_retrieval.ArmSpec import parse_arm, phase_configs
    from stage2_retrieval.Naming import config_label
    groups: Dict[str, List[str]] = {
        'Corr::cls_avg': ['diag', 'contrast', 'mnn', 'sinkhorn'],
        'Dual::cls_avg': ['dual', 'cos_main'],
        'CandGraph::raw': ['nbr_support', 'prominence', 'rot_margin'],
        'CSLS::grid2x2': ['csls', 'rank']}
    columns: Dict[str, List[str]] = {}
    for head, ops in groups.items():
        cfgs = [phase_configs(parse_arm(f'gigapath:SlidingWindow::cls+mean::{head}+{op}'))[-1] for op in ops]
        cols: List[str] = [retriever_class(c)(c, 'cpu').column for c in cfgs]
        assert len(set(cols)) == len(ops), (head, cols)
        assert len({c.identity_id() for c in cfgs}) == len(ops) and len({config_label(c) for c in cfgs}) == len(ops)
        columns[head] = cols
    assert columns['Corr::cls_avg'] == groups['Corr::cls_avg']
    assert columns['CSLS::grid2x2'] == ['csls', 'rankpct'], 'a mode of CSLS is a setting; its column follows'
    shift = phase_configs(parse_arm('gigapath:SlidingWindow::cls+mean::Shift::cls_avg'))[-1]
    assert retriever_class(shift)(shift, 'cpu').column == 'shift', 'a reranker with one answer orders by it'
    return 'Corr 4, Dual 2, CandGraph 3, CSLS 2: each its own config, column and name'


def t_o_the_arms_a_run_makes_come_prefix_first_and_leave_out_a_plain_window() -> str:
    from RetrieverArms import arm_texts
    from stage2_retrieval.ArmSpec import parse_arm
    specs = [parse_arm(t, 'uni2') for t in (
        'uni2:SlidingWindow::cls+mean::Corr::cls_avg+diag',
        'uni2:SlidingWindow::cls+mean::Corr::cls_avg+sinkhorn',
        'uni2:TileVote::cls_avg+exact',
        'uni2:SlidingWindow::cls+mean@100::Dual::cls_avg+dual')]
    got: List[str] = arm_texts(specs)
    assert got == ['uni2:SlidingWindow::cls+mean::Corr::cls_avg+diag',
                   'uni2:SlidingWindow::cls+mean::Corr::cls_avg+sinkhorn',
                   'uni2:TileVote::cls_avg+exact',
                   'uni2:SlidingWindow::cls+mean@100',
                   'uni2:SlidingWindow::cls+mean@100::Dual::cls_avg+dual'], got
    assert 'uni2:SlidingWindow::cls+mean' not in got, 'a plain sliding window is the streamed pass\'s'
    assert got.index('uni2:SlidingWindow::cls+mean@100') < got.index('uni2:SlidingWindow::cls+mean@100::Dual::cls_avg+dual')
    return f'{len(got)} arms; the cut window comes before its reranker'


def t_p_the_engine_is_the_tree_of_the_phases_and_files_them_as_the_pipeline_does() -> str:
    from BenchCommon import phase_ids
    from RetrieverArms import ArmEngine, _sibling_key
    from stage2_retrieval.ArmSpec import config_of_arm, parse_arm
    from stage2_retrieval.Retriever import EncoderPool
    chain = 'uni2:SlidingWindow::cls+mean::Corr::cls_avg+sinkhorn'
    other = 'uni2:SlidingWindow::cls+mean::Corr::cls_avg+diag'
    vote = 'uni2:TileVote::cls_avg+exact'
    cut = 'uni2:SlidingWindow::cls+mean@100::Dual::cls_avg+dual'
    engine: ArmEngine = ArmEngine([parse_arm(t) for t in (chain, other, vote, cut)], raw_topk=1000, min_sep=1.0,
                                  device='cpu', encoders=EncoderPool('cpu'), seg='s', region='r')
    # one sliding window under every arm that starts with it, cut or not: @K is a consumer's
    assert len(engine.roots) == 2, list(engine.roots)
    slide = next(n for n in engine.roots.values() if not n.after and 'SlidingWin' in type(n.cfg).__name__)
    assert len(slide.children) == 3, 'Corr+sinkhorn, Corr+diag and Dual hang off the one window'
    assert engine.entry_of('uni2:SlidingWindow::cls+mean@100') == slide.entry
    # the entries are the pipeline's: the same name as a chain of the same configs
    want = phase_ids(config_of_arm(parse_arm(chain)), ('s', 'r'))
    assert engine.entry_of(chain) == want[-1][0] and slide.entry == want[0][0]
    # the streamed pass writes the sliding window's entry; the engine every other
    assert slide.entry not in engine.wanted() and engine.entry_of(chain) in engine.wanted()
    assert engine.entry_of(vote) in engine.wanted()
    rec = engine.wanted()[engine.entry_of(chain)]
    assert rec['upstream']['after'] == want[0][1] and rec['label'] == engine.entry_of(chain)
    # two columns of one Corr are one scoring
    corr = [n for n in slide.children.values() if 'Corr' in type(n.cfg).__name__]
    assert len(corr) == 2 and _sibling_key(corr[0].cfg) == _sibling_key(corr[1].cfg)
    # what is run for an entry to write: the chain and what it stands on, not its sibling
    class Sink:
        def has(self, entry: str) -> bool:
            return entry == engine.entry_of(chain)

    engine._sink = Sink()
    assert engine._needed(slide) and not engine._needed(engine._of_text[other])
    assert not engine._needed(engine._of_text[vote])
    # a chain's rows are judged on its first phase's truth
    assert engine.first_entry_of(chain) == slide.entry and engine.first_entry_of(vote) == engine.entry_of(vote)
    assert engine.first_entry_of(cut) == slide.entry
    return '2 first phases, 3 rerankers on the window; the pipeline\'s names; siblings one scoring'


def t_q_methods_at_their_own_baselines_are_not_one_phase() -> str:
    """The bug this defends: a config's id was its fields against its class's baseline and no more, so
    BagOfTiles+gap, TileVote+exact, ClusterBow and Dual+dual on one pooling -- each at its own baseline,
    differing from it in `encoder` only -- had one id. The engine ran them all as the first one it built
    ('RetrieverBagOfTiles eats EstMppResult, not RetrievalOutput' from a Dual arm), and a reranker after
    two of them would have been one entry with one `_after-`."""
    from BenchCommon import phase_ids
    from RetrieverArms import ArmEngine
    from stage2_retrieval.ArmSpec import config_of_arm, parse_arm, phase_configs
    from stage2_retrieval.Retriever import EncoderPool
    firsts: List[str] = ['uni2:BagOfTiles::cls_avg+gap', 'uni2:TileVote::cls_avg+exact', 'uni2:ClusterBow::cls_avg']
    cfgs = [phase_configs(parse_arm(t))[0] for t in firsts]
    cfgs.append(phase_configs(parse_arm('uni2:SlidingWindow::cls+mean::Dual::cls_avg+dual'))[-1])
    assert len({c.identity_id() for c in cfgs}) == len(cfgs), [c.identity_parts() for c in cfgs]
    chains: List[str] = [f'{t}::Dual::cls_avg+dual' for t in firsts]
    entries: List[str] = [phase_ids(config_of_arm(parse_arm(t)), ('s', 'r'))[-1][0] for t in chains]
    assert len(set(entries)) == len(chains), entries
    engine: ArmEngine = ArmEngine([parse_arm(t) for t in firsts + chains], raw_topk=1000, min_sep=1.0,
                                  device='cpu', encoders=EncoderPool('cpu'), seg='s', region='r')
    assert len(engine.roots) == len(firsts), list(engine.roots)
    # one object per config: the three roots apart, and one Dual for the three lists it reorders
    keys: Dict[str, str] = {type(n.cfg).__name__: n.retriever for n in engine._nodes.values()}
    assert len(set(keys.values())) == len(keys) == 4, keys
    assert {n.retriever for n in engine._nodes.values()} == set(keys.values()), 'one config, one object'
    for text in firsts:
        got: str = type(engine._of_text[text].cfg).__name__
        assert got == f'Retriever{parse_arm(text).retriever.method}Config', (text, got)
    return f'{len(cfgs)} configs, {len(entries)} Dual entries, {len(engine.roots)} roots: all apart'


_TESTS = [t for n, t in sorted(globals().items(), key=lambda kv: kv[0])
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
