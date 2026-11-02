#!/usr/bin/env python3
"""Tests for the stage 2 cache entries a phase writes, the pipeline's and the stage 2 bench's alike:
`utilities/bench_modules/BenchCommon.py` (`write_phase`, `truth_rows`, `phase_ids`, `phase_record`,
`stage_hexid`) and `utilities/bench_modules/ArmTables.py` (a slide's sink, completeness).

    python utilities/test_modules/TestRerank/test_arm_tables.py

No slide, no model, no GPU: lists are arrays, the geometry is a slide with no mask, the cache a temporary
directory.

WHAT THIS DEFENDS
-----------------
    rows       a phase's list is the `output` table (rank first, the level, the box and `n_listed` of each
               window); a first phase that scored every window has both answers in its `truth` row -- the
               main and the offset window nearest the point, each with its box, distance, score and ranks
               among its lattice's windows and among all of them -- their per-tile cosines in `truth_sim`
               and the top of its list in `tile_sims`; one that searched another level has the answers'
               windows alone, placed in the frame of the photo's level; a reranker has no truth and its
               `rerank_scores` are its windows and its own columns; an empty list is one error row and no
               truth; a phase writes the tables it holds and no other
    entry      a phase is named and recorded by what decides it: the same reranker after another list, or
               reading another number of it, is another entry; the bench and the pipeline make one record
    weights    a checkpoint is in the id by its content
    complete   an entry holds every level of its slide: a missing entry, or one whose list stops short of
               a level holding a FoV the arms are scored on, is not complete and the slide is run again; an
               entry that got no rows is not written; a sink holds only the entries it was made for
"""

from __future__ import annotations

import dataclasses
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', '..'))
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'bench_modules'))

from _paths import setup_import_paths                            # noqa: E402

setup_import_paths()

import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402

import Cache                                                     # noqa: E402
from ArmTables import SlideTables, entries_complete, entry_levels   # noqa: E402
from BenchCommon import (Stage, Tables, Truth, phase_ids, phase_record, phase_roles,   # noqa: E402
                         stage_hexid, weights_of_config, write_phase)
from PatchingLib import QueryPatchContainer                      # noqa: E402
from stage2_retrieval.ArmSpec import config_of_arm, parse_arm    # noqa: E402
from stage2_retrieval.Context import SlideLevels, nearest_window  # noqa: E402
from stage2_retrieval.MultiPhases import MultiPhases             # noqa: E402
from stage2_retrieval.RerankerScorers import ColumnRanker        # noqa: E402
from stage2_retrieval.Retriever import Extra, RetrievalOutput    # noqa: E402
from stage2_retrieval.RetrieverSlidingWinSimRot import RetrieverSlidingWinSimRot   # noqa: E402
from stage2_retrieval.StageInterface import Candidate, CandidateSet, RerankedCandidate   # noqa: E402
from test_context import TILE, _Slide                            # noqa: E402

_RESULTS: List[Tuple[str, Optional[BaseException]]] = []


def check(name: str, fn: Callable[[], Optional[str]]) -> None:
    try:
        out: Optional[str] = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   {out}' if out else ''))
    except Exception as e:                                       # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n          {type(e).__name__}: {e}')


# ── a phase's output, in a real frame ─────────────────────────────────────────

#: a query of 3 x 4 tiles
IMG: np.ndarray = np.zeros((3 * TILE, 4 * TILE, 3), np.uint8)
LEVELS: SlideLevels = SlideLevels(_Slide(), None, TILE, True)
FRAME: CandidateSet = LEVELS.frame(0)
DIMS: Dict[int, Tuple[int, int]] = LEVELS.query_dims(IMG)


def qc_of(img: np.ndarray = IMG) -> QueryPatchContainer:
    qc: QueryPatchContainer = QueryPatchContainer(img)
    qc.extract_all(TILE, overlap=True)
    return qc


def window_scores(seed: int = 0) -> Dict[Tuple[int, int, str], torch.Tensor]:
    """A score for every window of the frame at every rotation, both lattices."""
    g: torch.Generator = torch.Generator().manual_seed(seed)
    grid: Any = FRAME.grids[0]
    out: Dict[Tuple[int, int, str], torch.Tensor] = {}
    for rot in (0, 90, 180, 270):
        rows, cols = DIMS[rot]
        for lattice, (gr, gc) in (('main', (grid.grid_rows, grid.grid_cols)),
                                  ('offset', (grid.overlap_rows, grid.overlap_cols))):
            out[(rot, 0, lattice)] = torch.rand(gr - rows + 1, gc - cols + 1, generator=g)
    return out


def first_output(listed: List[Candidate], scores: Dict[Tuple[int, int, str], torch.Tensor],
                 tile_sims: bool = True) -> RetrievalOutput:
    """A first phase's output: its list and the score of every window (and, for a sliding window, the per-tile
    cosines: a constant 0.5 under every window)."""
    extras: Dict[str, Extra] = {'window_scores': Extra('window_scores', ('k',), scores)}
    if tile_sims:
        sims: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = {
            rot: [(torch.full(tuple(scores[(rot, 0, 'main')].shape) + DIMS[rot], 0.5),
                   torch.full(tuple(scores[(rot, 0, 'offset')].shape) + DIMS[rot], 0.5))]
            for rot in (0, 90, 180, 270)}
        extras['tile_sims'] = Extra('tile_sims', ('k',), sims)
    return RetrievalOutput(candidates=dataclasses.replace(FRAME, candidates=tuple(listed)), extras=extras)


class _Writer:
    """What `write_phase` asks of a retriever: the tables of what it offers. A sliding window's."""

    def tables(self, out: RetrievalOutput, qc: Any) -> Dict[str, List[Dict[str, Any]]]:
        return RetrieverSlidingWinSimRot.tables(self, out, qc)       # type: ignore[arg-type]


class _Framed(_Writer):
    """A sliding window that can say where a level's windows are (`frame`), as the real one does."""

    def frame(self, level: int) -> CandidateSet:
        return LEVELS.frame(level)


#: the tables of a sliding window's first phase with a truth, and of a reranker's
SLIDING: Tuple[str, ...] = phase_roles(RetrieverSlidingWinSimRot, True, True)
RERANKER: Tuple[str, ...] = phase_roles(ColumnRanker, False, True)


def truth_at(region: int, lattice: str, row: int, col: int, rotation: int = 0) -> Truth:
    """A FoV whose tile grid stands for the centre of that window, placed at level 0."""
    x, y = FRAME.grids[region].tile_origin_l0(lattice, row, col)
    rows, cols = DIMS[rotation]
    centre: Tuple[float, float] = (x + cols * TILE / 2.0, y + rows * TILE / 2.0)
    return Truth(level=0, ds=1.0, rot=rotation, centre=lambda tile: centre)


def tables(roles: Tuple[str, ...] = SLIDING) -> Tables:
    return Tables(None, 'x', {}, roles)


def above(scores: Dict[Tuple[int, int, str], torch.Tensor], score: float, lattices: Tuple[str, ...]) -> int:
    """The windows of `lattices` scoring strictly above `score`, by brute force."""
    return sum(int((m > score).sum()) for (_, _, lat), m in scores.items() if lat in lattices)


def t_a_a_first_phase_writes_its_list_and_both_answers_ranked_among_every_window() -> str:
    scores = window_scores()
    best = max(((k, divmod(int(m.argmax()), m.shape[1])) for k, m in scores.items()),
               key=lambda kv: float(scores[kv[0]][kv[1]]))
    (rot, region, lattice), (i, j) = best
    listed: List[Candidate] = [Candidate(region, lattice, i, j, rot, float(scores[best[0]][i, j])),
                               Candidate(0, 'main', 0, 0, 0, 0.1)]
    tab: Tables = tables()
    write_phase(tab, 7, first_output(listed, scores), _Writer(), IMG, qc_of(), truth_at(region, lattice, i, j, rot),
                TILE, True, 0.5)
    rows = tab.rows
    assert [r['rank'] for r in rows['output']] == [1, 2] and all(r['n_listed'] == 2 for r in rows['output'])
    assert all(r['level'] == 0 and r['error'] == '' and r['index'] == 7 for r in rows['output'])
    assert {'x0', 'y0', 'w0', 'h0', 'score', 'region', 'lattice', 'row', 'col', 'rotation'} <= set(rows['output'][0])
    (truth,) = rows['truth']
    other: str = 'offset' if lattice == 'main' else 'main'
    p: str = f'truth_{lattice}_'
    assert (truth[f'{p}row'], truth[f'{p}col'], truth[f'{p}dist_l0']) == (i, j, 0.0), truth
    assert truth[f'truth_rank_{lattice}_in_all'] == 1 and truth['truth_level'] == 0 and truth['truth_rotation'] == rot
    # the other answer: half a tile away on both axes, and its ranks are the brute-force count
    assert abs(truth[f'truth_{other}_dist_l0'] - (2 * (TILE / 2) ** 2) ** 0.5) < 1e-6, truth
    for lat, own in (('main', 'truth_rank_main'), ('offset', 'truth_rank_offset_in_offset')):
        s: float = truth[f'truth_{lat}_score']
        assert s == float(scores[(rot, truth[f'truth_{lat}_region'], lat)][truth[f'truth_{lat}_row'],
                                                                          truth[f'truth_{lat}_col']])
        assert truth[own] == 1 + above(scores, s, (lat,)), (lat, truth)
        assert truth[f'truth_rank_{lat}_in_all'] == 1 + above(scores, s, ('main', 'offset')), (lat, truth)
        assert truth[f'truth_{lat}_w0'] > 0 and truth[f'truth_{lat}_h0'] > 0
    assert not {'truth_lattice', 'truth_rank_all', 'truth_score', 'truth_dist_l0'} & set(truth), 'the older columns'
    assert truth['pool_all'] == sum(m.numel() for m in scores.values())
    assert truth['pool_main'] == sum(m.numel() for (_, _, lat), m in scores.items() if lat == 'main')
    assert len(rows['truth_sim']) == 2 * DIMS[rot][0] * DIMS[rot][1] and all(r['cosine'] == 0.5 for r in rows['truth_sim'])
    assert sorted({r['lattice'] for r in rows['truth_sim']}) == ['main', 'offset'], 'both answers\' tiles'
    assert len(rows['tile_sims']) == DIMS[rot][0] * DIMS[rot][1] + DIMS[0][0] * DIMS[0][1], \
        'the per-tile cosines of the two listed windows'
    return f'output 2 windows, the {lattice} answer first of {truth["pool_all"]}, the {other} one by brute force'


def t_b_a_reranker_writes_its_windows_and_columns_and_no_truth() -> str:
    scores = window_scores(1)
    up: RetrievalOutput = first_output([Candidate(0, 'main', 2, 3, 0, 0.9), Candidate(0, 'main', 5, 5, 0, 0.8)],
                                       scores, tile_sims=False)
    from stage2_retrieval.Context import ColumnTable, ScoreTable, reranked_output   # noqa: PLC0415
    table: ScoreTable = ScoreTable(2)
    for name, values in (('in', [0.9, 0.8]), ('first', [0.9, 0.8]), ('first_rank', [1.0, 2.0]), ('dual', [0.1, 0.7])):
        table.put(name, torch.tensor(values))
    out: RetrievalOutput = reranked_output(up, up.candidates, table, 'dual')
    assert RERANKER == ('output', 'rerank_scores'), RERANKER
    tab: Tables = tables(RERANKER)
    write_phase(tab, 3, out, ColumnRanker.__new__(ColumnRanker), IMG, qc_of(), truth_at(0, 'main', 2, 3), TILE,
                True, 0.1)
    rows = tab.rows
    assert set(rows) == {'output', 'rerank_scores'}, 'no truth: it scored its list alone'
    assert [r['row'] for r in rows['output']] == [5, 2], 'ordered by its column'
    scored: List[Dict[str, Any]] = rows['rerank_scores']
    assert [(r['row'], r['col']) for r in scored] == [(2, 3), (5, 5)], 'in the order it read them'
    assert [r['dual'] for r in scored] == table['dual'].tolist()
    assert not {'in_rank', 'out_rank', 'score', 'in', 'first', 'first_rank', 'x0'} & set(scored[0]), scored[0]
    return 'rerank_scores: the window and its own columns; no truth'


def t_ba_a_phase_that_searched_another_level_places_the_answers_and_scores_none() -> str:
    truth: Truth = truth_at(0, 'main', 2, 3)                               # placed at level 0
    out: RetrievalOutput = RetrievalOutput(candidates=dataclasses.replace(       # searched level 1
        LEVELS.frame(1), candidates=(Candidate(0, 'main', 0, 0, 0, 0.1),)))
    tab: Tables = tables()
    write_phase(tab, 4, out, _Framed(), IMG, qc_of(), truth, TILE, True, 0.1)
    (row,) = tab.rows['truth']
    assert (row['truth_main_row'], row['truth_main_col'], row['truth_main_dist_l0']) == (2, 3, 0.0), row
    assert 'truth_offset_row' in row and row['truth_level'] == 0
    assert not [k for k in row if 'score' in k or 'rank' in k or 'pool' in k], 'nothing was scored there'
    assert tab.rows['truth_sim'] == []
    bare: Tables = tables()
    write_phase(bare, 4, out, _Writer(), IMG, qc_of(), truth, TILE, True, 0.1)
    assert bare.rows['truth'] == [dict(index=4, truth_rotation=0, truth_level=0)], 'no frame: the rotation alone'
    return 'the windows of level 0 from a level-1 search, no score, rank or pool'


def t_c_an_empty_list_is_one_error_row_and_no_truth() -> str:
    tab: Tables = tables()
    empty: RetrievalOutput = RetrievalOutput(candidates=CandidateSet(candidates=(), level=2, ds=16.0, grids=(),
                                                                     irretrievable_lvl=(1, 2), alter_lvl=(1, 0)))
    write_phase(tab, 0, empty, _Writer(), IMG, qc_of(), truth_at(0, 'main', 0, 0), TILE, True, 0.0)
    rows = tab.rows
    assert len(rows['output']) == 1 and rows['output'][0]['error'] == 'no level holds the query', rows['output']
    assert rows['output'][0]['irretrievable_lvl'] == '1 2' and rows['truth'] == []
    return 'error row, the levels it tried'


def t_d_a_phase_is_named_and_recorded_by_what_decides_it_the_same_for_the_bench_and_the_pipeline() -> str:
    arm = 'uni2:BagOfTiles::cls_avg+gap::CSLS::grid2x2+csls'
    cfg = config_of_arm(parse_arm(arm))
    (bag, bag_own), (csls, csls_own) = phase_ids(cfg, ('hest-aa', 'rr'))
    assert re.fullmatch(r'bagoftiles-uni2-cls-avg-gap-[0-9a-f]{16}', bag), bag
    assert re.fullmatch(r'csls-uni2-grid2x2-csls-[0-9a-f]{16}_after-' + bag_own, csls), csls
    # the same reranker after another list, or reading another number of it, is another entry
    after_quad = phase_ids(config_of_arm(parse_arm('uni2:BagOfTiles::cls_avg+quad::CSLS::grid2x2+csls')),
                           ('hest-aa', 'rr'))[1][0]
    read_100 = phase_ids(config_of_arm(parse_arm('uni2:BagOfTiles::cls_avg+gap@100::CSLS::grid2x2+csls')),
                         ('hest-aa', 'rr'))[1][0]
    assert len({csls, after_quad, read_100}) == 3 and '-k100-' in read_100, (csls, after_quad, read_100)
    # the first phase run alone is the same entry as in the chain
    assert phase_ids(config_of_arm(parse_arm('uni2:BagOfTiles::cls_avg+gap')), ('hest-aa', 'rr'))[0][0] == bag
    # the pipeline's stage and the bench make one record
    stage: Stage = Stage(2, 'multiphases', 'x', cfg, MultiPhases, {}, also_id=('hest-aa', 'rr'))
    assert stage.id == csls and [n for n, _ in stage.phases] == [bag, csls]
    up: Dict[str, str] = dict(stage1='oracle', seg='hest-aa', region='rr')
    assert stage.phase_record(1, **up) == phase_record(cfg.phases[1], csls, bag_own, **up)
    assert stage.phase_record(1, **up)['upstream']['after'] == bag_own
    return csls


def t_e_a_checkpoint_is_in_the_id_by_its_content() -> str:
    base = config_of_arm(parse_arm('uni2:SlidingWindow::cls_avg+mean::TileGT::cls_avg+tilegt'))
    with tempfile.TemporaryDirectory() as root:
        a, b = os.path.join(root, 'a.pt'), os.path.join(root, 'b.pt')
        with open(a, 'wb') as f:
            f.write(b'one')
        shutil.copy(a, b)
        cfg_a = dataclasses.replace(base, phases=(base.phases[0], dataclasses.replace(base.phases[1], weights=a)))
        cfg_b = dataclasses.replace(base, phases=(base.phases[0], dataclasses.replace(base.phases[1], weights=b)))
        assert weights_of_config(cfg_a) == [a]
        before = phase_ids(cfg_a, ('s', 'r'))
        assert phase_ids(cfg_b, ('s', 'r')) == before, 'a renamed file with the same bytes is the same entry'
        with open(a, 'wb') as f:
            f.write(b'fine-tuned')
        after = phase_ids(cfg_a, ('s', 'r'))
        assert after[0] == before[0] and after[1] != before[1], 'the same name with other bytes is another entry'
        gone = dataclasses.replace(base.phases[1], weights=os.path.join(root, 'x.pt'))
        stage_hexid(gone, ('s', 'r'))                 # a missing file does not raise
    plain = base.phases[0]
    assert weights_of_config(plain) == [] and stage_hexid(plain) == plain.identity_id()
    return 'rename: same; new bytes: another; no weights: unchanged'


# ── a slide's entries ─────────────────────────────────────────────────────────

def rec_of(entry_id: str) -> Dict[str, Any]:
    return dict(id=entry_id, label=entry_id, parts=['p'], versions={}, upstream={})


def entry_in(root: str) -> Any:
    Cache.RESULT_DIR = root
    address = Cache.Address('TestJob', slide='S_1', seg='hest-ab12', region='r1', plan='p', draw='d', render='g1')
    return address.at(stage1='oracle').entry('stage2')


def fill(sink: SlideTables, entry_id: str, level: int) -> None:
    """One FoV's rows of a first phase at `level`: its list in that level's frame (scored at level 0, the frame
    the made-up scores are of)."""
    listed: List[Candidate] = [Candidate(0, 'main', 1, 1, 0, 0.5)]
    out: RetrievalOutput = (first_output(listed, window_scores(level)) if level == 0 else
                            RetrievalOutput(candidates=dataclasses.replace(LEVELS.frame(level), candidates=tuple(listed))))
    write_phase(sink.tables(entry_id), 7, out, _Writer(), IMG, qc_of(),
                dataclasses.replace(truth_at(0, 'main', 1, 1), level=level), TILE, True, 0.1)


def t_g_an_entry_is_complete_only_when_it_reaches_every_level_of_the_slide() -> str:
    old: str = Cache.RESULT_DIR
    try:
        with tempfile.TemporaryDirectory() as root:
            entry: Any = entry_in(root)
            wanted: Dict[str, Dict[str, Any]] = {'arm-x': rec_of('arm-x')}
            held: Dict[str, Tuple[str, ...]] = {'arm-x': SLIDING}
            assert not entries_complete(entry, wanted, {0}, held)               # nothing written
            sink: SlideTables = SlideTables(entry, wanted, held)
            fill(sink, 'arm-x', 0)
            written, empty = sink.flush()
            assert (written, empty) == (1, []), (written, empty)
            assert entry_levels(entry, 'arm-x') == {0}
            assert entries_complete(entry, wanted, {0}, held) and entries_complete(entry, wanted, set(), held)
            assert not entries_complete(entry, wanted, {0, 1}, held), \
                'a level-0-only entry must not pass for a slide with level 1'
            assert not entries_complete(entry, wanted, {0}, {'arm-x': SLIDING + ('rerank_scores',)}), \
                'an entry without a table it should hold is not complete'
            second: SlideTables = SlideTables(entry, wanted, held)
            fill(second, 'arm-x', 0)
            fill(second, 'arm-x', 1)
            second.flush()
            assert entry_levels(entry, 'arm-x') == {0, 1} and entries_complete(entry, wanted, {0, 1}, held)
            roles: List[str] = sorted(p.name[:-len('_arm-x.csv')] for p in entry.dir.glob('*.csv'))
            assert roles == sorted(SLIDING), 'the tables it holds and no other: ' + str(roles)
    finally:
        Cache.RESULT_DIR = old
    return 'level 0 only -> run again; both levels -> complete'


def t_h_an_entry_with_no_rows_is_not_written_and_a_sink_holds_only_its_own() -> str:
    old: str = Cache.RESULT_DIR
    try:
        with tempfile.TemporaryDirectory() as root:
            entry: Any = entry_in(root)
            sink: SlideTables = SlideTables(entry, {i: rec_of(i) for i in ('arm-x', 'arm-y')},
                                            {'arm-x': SLIDING, 'arm-y': SLIDING})
            assert sink.has('arm-x') and sink.has('arm-y') and not sink.has('arm-z')
            fill(sink, 'arm-x', 0)
            written, empty = sink.flush()
            assert written == 1 and empty == ['arm-y'], (written, empty)
            assert entry.record_path('arm-x').exists() and not entry.record_path('arm-y').exists()
    finally:
        Cache.RESULT_DIR = old
    return 'arm-y (failed) left for the next run'


def t_i_an_arms_rows_are_read_off_its_entries() -> str:
    """The bench keeps nothing but entries: a first phase's ranked row is its truth, a listed row the main
    answer's place in the list (the first phase's, a reranker's, one cut at @K), judged on the first phase's
    answers; the FoV's columns are the draw's and the render table's."""
    import types
    import bench_stage2_retriever as bench                            # noqa: PLC0415
    from stage2_retrieval.Context import ScoreTable, reranked_output  # noqa: PLC0415
    old: str = Cache.RESULT_DIR
    try:
        with tempfile.TemporaryDirectory() as root:
            entry: Any = entry_in(root)
            scores = window_scores(3)
            scores[(0, 0, 'main')][2, 3] = 2.0              # the answer the best window of all
            listed: List[Candidate] = [Candidate(0, 'main', 2, 3, 0, 2.0), Candidate(0, 'main', 6, 6, 0, 0.9)]
            up: RetrievalOutput = first_output(listed, scores)
            table: ScoreTable = ScoreTable(2)
            for name, values in (('in', [2.0, 0.9]), ('first', [2.0, 0.9]), ('first_rank', [1.0, 2.0]),
                                 ('dual', [0.1, 0.7])):
                table.put(name, torch.tensor(values))
            sink: SlideTables = SlideTables(entry, {'first': rec_of('first'), 'second': rec_of('second')},
                                            {'first': SLIDING, 'second': RERANKER})
            write_phase(sink.tables('first'), 7, up, _Writer(), IMG, qc_of(), truth_at(0, 'main', 2, 3), TILE,
                        True, 0.5)
            write_phase(sink.tables('second'), 7, reranked_output(up, up.candidates, table, 'dual'),
                        ColumnRanker.__new__(ColumnRanker), IMG, qc_of(), truth_at(0, 'main', 2, 3), TILE, True, 0.1)
            sink.flush()
            render: Path = Path(root) / 'render.csv'
            render.write_text('index,rot_deg,angle_jitter,scale\n7,0,0,1\n')
            supply = types.SimpleNamespace(
                render_entry=types.SimpleNamespace(path=lambda role, rid, ext: render),
                microscope=types.SimpleNamespace(cfg=types.SimpleNamespace(identity_id=lambda: 'g')),
                sampler=[types.SimpleNamespace(meta=types.SimpleNamespace(score=0.25, bucket='b'))] * 8)
            arms = [bench.ArmRows('win', 'first', 'first', 'pooling', 'stream', 'all', None, True),
                    bench.ArmRows('rr', 'first', 'second', 'reranker', 'retriever', 'all', None, False),
                    bench.ArmRows('cut', 'first', 'first', 'method', 'retriever', 'all', 1, False),
                    bench.ArmRows('win-M', 'first', 'first', 'pooling', 'stream', 'main', None, True)]
            rows, differ = bench.slide_rows(entry, arms, encoder='uni2', dataset='d', stem='S', supply=supply,
                                            min_sep=1.0)
    finally:
        Cache.RESULT_DIR = old
    assert differ == 0 and len(rows) == 6, (differ, [(r['arm'], r['mode']) for r in rows])
    by = {(r['arm'], r['mode']): r for r in rows}
    ranked, win = by[('win', 'ranked')], by[('win', 'listed')]
    assert ranked['rank_main'] == ranked['rank_main_in_all'] == 1, ranked
    assert ranked['pool'] == sum(m.numel() for (_, _, lat), m in scores.items() if lat == 'main')
    assert win['rank_main'] == 1 and win['list_len'] == 2 and win['truth_score'] == 2.0, win
    assert by[('rr', 'listed')]['rank_main'] == 2, 'the reranker put the answer second'
    assert by[('cut', 'listed')]['list_len'] == 1 and by[('cut', 'listed')]['rank_main'] == 1
    main_only = by[('win-M', 'listed')]
    assert main_only['rank_offset_in_all'] == '' and main_only['pool_all'] == '', 'no offset answer on -M'
    assert (win['fov_id'], win['level'], win['white_frac'], win['bucket'], win['d_main']) == (7, 0, 0.25, 'b', 0.0)
    assert ('rr', 'ranked') not in by and ('cut', 'ranked') not in by, 'a list has a listed mode only'
    return 'ranked from the truth; listed: window 1, reranked 2, cut at 1; -M without the offset columns'


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
