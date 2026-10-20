#!/usr/bin/env python3
"""Tests for the stage-2 retrievers without a network: the rerankers that read features
(`stage2_retrieval/RerankerScorers.py`: Dual, CSLS, Corr, CandGraph, and `ColumnRanker`, what every reranker
does around its scoring) and the first-phase methods' scoring (`RetrieverBagOfTiles`, `RetrieverTileVote`,
`RetrieverClusterBow`).

    python utilities/test_modules/TestRerank/test_rerank.py

No slide, no model, no GPU. Features are random unit vectors, so every expected value is computed here, one
tile at a time, in plain loops, and compared with what the vectorised code returns. A reranker is handed its
upstream's features through the bridge lock (`ported_upstream`), as a reranker after the sliding window of the
same features is in a run.

WHAT THIS DEFENDS
-----------------
    reranker   the list is ordered by the config's column, every candidate a reranked one remembering the
               first phase's score and rank (through a chain too); `topk_in` is what it reads, the
               consumer's `topk` a prefix of what it hands back; the columns go downstream (`columns`)
    reorder    another column of one scoring (`Context.reordered`) is what scoring again with that column
               gives
    kinds      the columns choose the kind of candidate: a shift makes it a `ShiftedCandidate`, inliers a
               `TokenMatchCandidate`; with no features a window's shape is cut from the query
    bridge     features under another key are refused; with no reuse declared none are taken
    dual       the query's half-tile tiles are matched with the right reference tiles, on both lattices
    csls/corr  the formulas, against loops
    candgraph  neighbours, prominence, and the rotation margin of a place
    methods    BagOfTiles' box means, against loops; a block planted in the reference is the top window of
               every first-phase method, of its whole ranking and of the list cut from it; the exact tile
               index is the true nearest, the graph agrees
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', '..'))

from _paths import setup_import_paths                            # noqa: E402

setup_import_paths()

import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402
import torch.nn.functional as F                                  # noqa: E402

import ConfigIdentity as CI                                      # noqa: E402
from test_context import TILE, cset, make_source, unit           # noqa: E402
from TileEncoderFunc import encoder_config                       # noqa: E402
from stage2_retrieval import RetrieverBagOfTiles as RB           # noqa: E402
from stage2_retrieval import RetrieverClusterBow as RC           # noqa: E402
from stage2_retrieval import RetrieverTileVote as RT             # noqa: E402
from stage2_retrieval.Context import (GridSource, LevelFeatures, MemorySource, QueryFeatures,   # noqa: E402
                                      ScoringInput, WindowBatch, feature_key, incoming_table, reordered,
                                      top_candidates)
from stage2_retrieval.RerankerScorers import (ColumnRanker, RerankerCandGraph, RerankerCandGraphConfig,   # noqa: E402
                                              RerankerCorr, RerankerCorrConfig, RerankerCsls,
                                              RerankerCslsConfig, RerankerDual, RerankerDualConfig)
from stage2_retrieval.Retriever import BridgeError, Extra, RetrievalOutput, RetrieverConfig   # noqa: E402
from stage2_retrieval.StageInterface import (Candidate, CandidateSet, RerankedCandidate,  # noqa: E402
                                             ShiftedCandidate, TokenMatchCandidate, VoteCandidate)

_RESULTS: List[Tuple[str, Optional[BaseException]]] = []


def check(name: str, fn: Callable[[], Optional[str]]) -> None:
    try:
        out: Optional[str] = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   {out}' if out else ''))
    except Exception as e:                                       # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n          {type(e).__name__}: {e}')


# ── a first phase's output, and some candidates ───────────────────────────────

ENC: Any = encoder_config('uni2', pooling='cls').with_model(dtype='fp16')
REUSE: Tuple[str, ...] = ('level_features',)
#: a query of 4 x 5 tiles: the shape of the source's rotation-0 query grid
QUERY: np.ndarray = np.zeros((4 * TILE, 5 * TILE, 3), np.uint8)


def grid_source(src: MemorySource, ref: Dict[Tuple[int, str], torch.Tensor], n_regions: int = 2) -> GridSource:
    """`src`'s reference and query grids as a first phase offers them."""
    qmaps: Dict[int, object] = {
        rot: SimpleNamespace(main_feature_grid=lambda rot=rot: src.queries[(rot, 'main')],
                             overlap_feature_grid=lambda rot=rot: src.queries[(rot, 'offset')])
        for rot in (0, 90, 180, 270)}
    return GridSource(LevelFeatures(level=1, ds=2.0, regions=[], grids=[], features=None,
                                    tensors=[(ref[(r, 'main')], ref[(r, 'offset')]) for r in range(n_regions)]),
                      QueryFeatures(containers={}, features=qmaps), src.pooling)


def ported_upstream(src: MemorySource, ref: Dict[Tuple[int, str], torch.Tensor], cs: CandidateSet,
                    enc: object = ENC, n_regions: int = 2) -> RetrievalOutput:
    """A first phase's output over `src`'s tensors: the list `cs` and, as its `level_features`, a `GridSource`
    over the same reference and query grids, keyed as `enc` at 256 px with the offset lattice."""
    return RetrievalOutput(candidates=cs, extras={
        'level_features': Extra('level_features', feature_key(enc, TILE, True), grid_source(src, ref, n_regions))})


def bound(r: Any) -> Any:
    """`r` as if built on a slide: the features it reads come from the upstream, nothing is read."""
    r.wsi = object()
    return r


def random_candidates(ref: Dict[Tuple[int, str], torch.Tensor], n: int, seed: int = 0,
                      qshape: Tuple[int, int] = (4, 5), regions: int = 2) -> List[Candidate]:
    rng: np.random.Generator = np.random.default_rng(seed)
    out: Dict[Tuple[int, str, int, int, int], Candidate] = {}
    while len(out) < n:
        rot: int = int(rng.choice([0, 90, 180, 270]))
        qr, qc = qshape if rot in (0, 180) else (qshape[1], qshape[0])
        region: int = int(rng.integers(0, regions))
        lattice: str = str(rng.choice(['main', 'offset']))
        R, C = ref[(region, lattice)].shape[:2]
        c: Candidate = Candidate(region, lattice, int(rng.integers(0, R - qr + 1)),
                                 int(rng.integers(0, C - qc + 1)), rot, float(rng.random()))
        out.setdefault(c.key(), c)                               # every window once
    return sorted(out.values(), key=lambda c: -c.score)


def scoring_input(src: MemorySource, ref: Dict[Tuple[int, str], torch.Tensor], cands: List[Candidate],
                  n_regions: int = 2) -> ScoringInput:
    cs: CandidateSet = cset(cands, n_regions=n_regions)
    return ScoringInput(QUERY, cs, incoming_table(cs), grid_source(src, ref, n_regions), TILE)


# ── what every reranker does ──────────────────────────────────────────────────

def t_a_reranker_orders_by_its_column_and_remembers_the_first_phase() -> str:
    src, ref = make_source()
    cands: List[Candidate] = random_candidates(ref, 40)
    cs: CandidateSet = cset(cands)
    up: RetrievalOutput = ported_upstream(src, ref, cs)
    out: RetrievalOutput = bound(RerankerDual(RerankerDualConfig(encoder=ENC, order_by='dual', reuse=REUSE),
                                              'cpu')).retrieve(up, QUERY)
    listed: CandidateSet = out.candidates
    assert len(listed) == 40 and listed.level == cs.level and listed.grids == cs.grids
    assert all(type(c) is RerankedCandidate for c in listed)
    scores: List[float] = [c.score for c in listed]
    dual: List[float] = out.extras['columns'].value.cols['dual']
    assert scores == sorted(dual, reverse=True), 'ordered by the column, scored by it'
    for c in listed:
        i: int = next(j for j, o in enumerate(cands) if o.key() == c.key())
        assert c.first_rank == i + 1 and c.first_score == cands[i].score
    assert out.parent is up and 'level_features' not in out.extras, 'it read the upstream\'s, made none'
    return '40 reordered, the first phase remembered, the columns offered'


def t_topk_in_is_what_it_reads_and_topk_a_prefix_of_what_it_hands_back() -> str:
    src, ref = make_source()
    cands: List[Candidate] = random_candidates(ref, 40, seed=3)
    up: RetrievalOutput = ported_upstream(src, ref, cset(cands))
    r: RerankerDual = bound(RerankerDual(RerankerDualConfig(encoder=ENC, topk_in=7, order_by='dual', reuse=REUSE),
                                         'cpu'))
    seven: RetrievalOutput = r.retrieve(up, QUERY)
    assert sorted(c.key() for c in seven.candidates) == sorted(c.key() for c in cands[:7])
    three: RetrievalOutput = r.retrieve(up, QUERY, topk=3)
    assert [c.key() for c in three.candidates] == [c.key() for c in seven.candidates][:3]
    try:
        r.retrieve(up, QUERY, topk=8)
        raise AssertionError('a topk longer than topk_in was accepted')
    except ValueError:
        pass
    return 'reads 7, hands 3 of them on, refuses 8'


def t_a_chain_keeps_the_first_phases_numbers() -> str:
    src, ref = make_source()
    cands: List[Candidate] = random_candidates(ref, 40, seed=1)
    first: RetrievalOutput = bound(RerankerDual(RerankerDualConfig(encoder=ENC, order_by='cos_main', reuse=REUSE),
                                                'cpu')).retrieve(ported_upstream(src, ref, cset(cands)), QUERY)
    second: RetrievalOutput = bound(RerankerDual(RerankerDualConfig(encoder=ENC, topk_in=20, order_by='dual',
                                                                    reuse=REUSE), 'cpu')).retrieve(first, QUERY,
                                                                                                    topk=5)
    assert len(second.candidates) == 5 and second.parent is first
    for c in second.candidates:
        i: int = next(j for j, o in enumerate(cands) if o.key() == c.key())
        assert c.first_rank == i + 1, 'the rank is the FIRST phase\'s, not the first reranker\'s'
        assert c.first_score == cands[i].score
    return 'two rerankers, the second reading the first one\'s 20'


def t_another_column_of_one_scoring_is_scoring_again() -> str:
    src, ref = make_source()
    up: RetrievalOutput = ported_upstream(src, ref, cset(random_candidates(ref, 30, seed=8)))
    done: RetrievalOutput = bound(RerankerCorr(RerankerCorrConfig(encoder=ENC, order_by='sinkhorn', reuse=REUSE),
                                               'cpu')).retrieve(up, QUERY)
    for col in ('diag', 'contrast', 'mnn'):
        again: RetrievalOutput = bound(RerankerCorr(RerankerCorrConfig(encoder=ENC, order_by=col, reuse=REUSE),
                                                    'cpu')).retrieve(up, QUERY)
        cut: RetrievalOutput = reordered(up, done, col)
        assert [(c.key(), c.score, c.first_rank) for c in cut.candidates] == \
            [(c.key(), c.score, c.first_rank) for c in again.candidates], col
        assert cut.extras['columns'].value.cols == again.extras['columns'].value.cols
    return 'diag, contrast, mnn off the sinkhorn scoring: the same lists and columns'


@CI.register('t_fake_cols')
@dataclass(frozen=True)
class _FakeColsConfig(RetrieverConfig):
    kind: str = 'shift'
    tile_size: int = 256
    topk_in: int = 1000
    order_by: str = 'fake'

    BASELINE = {'kind': 'shift', 'tile_size': 256, 'topk_in': 1000, 'order_by': 'fake'}


class _FakeCols(ColumnRanker):
    """Writes the columns that make a shifted / token-matched candidate, with the candidates' reverse order as
    the score. Reads no features."""
    READS_FEATURES = False

    def encoder_needs(self) -> Dict[str, Any]:
        return {}

    def score(self, inp: ScoringInput) -> Dict[str, torch.Tensor]:
        k: int = len(inp.cs)
        out: Dict[str, torch.Tensor] = {'fake': torch.arange(k, 0, -1, dtype=torch.float32)}
        if self.cfg.kind == 'shift':
            out.update(shift_dx=torch.full((k,), 32.0), shift_dy=torch.full((k,), -64.0))
        else:
            out.update(n_inliers=torch.arange(k).float(), inlier_ratio=torch.linspace(0, 1, k))
        return out


def t_the_columns_choose_the_kind_of_candidate_and_a_lean_window_is_cut_from_the_query() -> str:
    _, ref = make_source()
    cs: CandidateSet = cset(random_candidates(ref, 6, seed=2))
    up: RetrievalOutput = RetrievalOutput(candidates=cs)
    for kind, want in (('shift', ShiftedCandidate), ('tokens', TokenMatchCandidate)):
        out: RetrievalOutput = bound(_FakeCols(_FakeColsConfig(kind=kind), 'cpu')).retrieve(up, QUERY, topk=4)
        assert all(type(c) is want for c in out.candidates), (kind, [type(c).__name__ for c in out.candidates])
        assert [c.key() for c in out.candidates] == [c.key() for c in cs.candidates[:4]], \
            'the fake score is the reverse of the order: the first stays first'
    shifted: RetrievalOutput = bound(_FakeCols(_FakeColsConfig(kind='shift'), 'cpu')).retrieve(up, QUERY)
    c0: ShiftedCandidate = shifted.candidates.candidates[0]           # type: ignore[assignment]
    assert c0.shift == (32, -64) and c0.shift_px() == (32, -64)
    ox, oy = cs.origin_l0(cs.candidates[0])
    assert shifted.candidates.origin_l0(c0) == (ox + 64, oy - 128), 'level px times ds = 2'
    lean: ScoringInput = ScoringInput(np.zeros((4 * TILE + 5, 5 * TILE + 9, 3), np.uint8), cs, incoming_table(cs),
                                      None, TILE)
    assert lean.shape_of(0) == (4, 5) and lean.shape_of(180) == (4, 5)
    assert lean.shape_of(90) == (5, 4) and lean.shape_of(270) == (5, 4)
    return 'ShiftedCandidate and TokenMatchCandidate; the window of a query of 4 x 5 tiles'


def t_the_bridge_lock_refuses_features_of_another_key_and_takes_none_unasked() -> str:
    src, ref = make_source()
    cs: CandidateSet = cset(random_candidates(ref, 10, seed=5))
    other: RetrievalOutput = RetrievalOutput(candidates=cs, extras={'level_features': Extra(
        'level_features', ('another trunk', 'cls', TILE, True), None)})
    try:
        bound(RerankerDual(RerankerDualConfig(encoder=ENC, order_by='dual', reuse=REUSE), 'cpu')).retrieve(
            other, QUERY)
        raise AssertionError('features of another encoder were reused')
    except BridgeError:
        pass
    try:
        bound(RerankerDual(RerankerDualConfig(encoder=ENC, order_by='dual', reuse=()), 'cpu')).retrieve(
            ported_upstream(src, ref, cs), QUERY)
        raise AssertionError('features were taken with no reuse declared')
    except RuntimeError as e:
        assert 'build' in str(e), e                          # it went to encode its own level
    return 'another key: BridgeError; no reuse: it encodes its own'


# ── S0 dual ───────────────────────────────────────────────────────────────────

def t_dual_matches_the_loops() -> str:
    src, ref = make_source()
    cands: List[Candidate] = random_candidates(ref, 50, seed=3)
    got: Dict[str, torch.Tensor] = RerankerDual(RerankerDualConfig(encoder=ENC), 'cpu').score(
        scoring_input(src, ref, cands))
    for k, c in enumerate(cands):
        qm: torch.Tensor = src.queries[(c.rotation, 'main')]
        qo: torch.Tensor = src.queries[(c.rotation, 'offset')]
        main: torch.Tensor = ref[(c.region_index, c.lattice)]
        other: torch.Tensor = ref[(c.region_index, 'offset' if c.lattice == 'main' else 'main')]
        s: int = 0 if c.lattice == 'main' else 1
        sm: float = sum(float(qm[i, j] @ main[c.row + i, c.col + j])
                        for i in range(qm.shape[0]) for j in range(qm.shape[1]))
        so: float = sum(float(qo[i, j] @ other[c.row + i + s, c.col + j + s])
                        for i in range(qo.shape[0]) for j in range(qo.shape[1]))
        n_m, n_o = qm.shape[0] * qm.shape[1], qo.shape[0] * qo.shape[1]
        assert abs(float(got['dual'][k]) - (sm + so) / (n_m + n_o)) < 1e-5, k
        assert abs(float(got['cos_main'][k]) - sm / n_m) < 1e-5
        assert abs(float(got['cos_offset'][k]) - so / n_o) < 1e-5
    return '50 windows: 20 main + 12 offset tiles each'


# ── S1 csls ───────────────────────────────────────────────────────────────────

def _brute(src: MemorySource, ref: Dict[Tuple[int, str], torch.Tensor], c: Candidate, k: int
           ) -> Tuple[float, float]:
    """(csls, rankpct) of one window by loops over every tile of the level."""
    every: torch.Tensor = torch.cat([g.reshape(-1, g.shape[-1]) for g in ref.values()])
    qg: torch.Tensor = src.queries[(c.rotation, 'main')]
    g: torch.Tensor = ref[(c.region_index, c.lattice)]
    csls: List[float] = []
    pct: List[float] = []
    for i in range(qg.shape[0]):
        for j in range(qg.shape[1]):
            t: torch.Tensor = g[c.row + i, c.col + j]
            cos: float = float(qg[i, j] @ t)
            rho: float = float((every @ qg[i, j]).topk(k).values.mean())
            hub: float = float((every @ t).topk(k + 1).values[1:].mean())
            csls.append(2 * cos - rho - hub)
            pct.append(1.0 - float(((every @ qg[i, j]) > cos).sum()) / len(every))
    return float(np.mean(csls)), float(np.mean(pct))


def t_csls_and_rank_match_the_loops() -> str:
    src, ref = make_source()
    cands: List[Candidate] = random_candidates(ref, 12, seed=4)
    big: int = 10 ** 9
    csls: Dict[str, torch.Tensor] = RerankerCsls(RerankerCslsConfig(
        encoder=ENC, mode='csls', n_ref=big, k=3, seed=0, chunk=50), 'cpu').score(scoring_input(src, ref, cands))
    rank: Dict[str, torch.Tensor] = RerankerCsls(RerankerCslsConfig(
        encoder=ENC, mode='rank', n_ref=big, k=3, seed=0, chunk=50), 'cpu').score(scoring_input(src, ref, cands))
    for k, c in enumerate(cands):
        want_csls, want_pct = _brute(src, ref, c, 3)
        assert abs(float(csls['csls'][k]) - want_csls) < 1e-4, (k, float(csls['csls'][k]), want_csls)
        # a tile's cosine with itself is found again by the matrix product, to the last bit
        # or not: the count may be one off, one tile in a few hundred
        assert abs(float(rank['rankpct'][k]) - want_pct) < 5e-3, (k, float(rank['rankpct'][k]), want_pct)
    assert 'csls' not in rank and 'rankpct' not in csls
    assert RerankerCsls(RerankerCslsConfig(encoder=ENC, mode='rank'), 'cpu').column == 'rankpct'
    return 'csls and rank, chunked'


# ── S2 corr ───────────────────────────────────────────────────────────────────

def t_corr_on_an_identity_window_and_a_flat_one() -> str:
    dim: int = 32
    basis: torch.Tensor = torch.eye(dim)[:20]                         # 20 orthonormal tile vectors
    ref: Dict[Tuple[int, str], torch.Tensor] = {
        (0, 'main'): torch.cat([basis.reshape(4, 5, dim), basis[:1].expand(8, 5, dim)]),   # rows 0-3: the query; rows 4-11: one vector
        (0, 'offset'): unit(11, 13, dim, seed=5)}
    qs: Dict[Tuple[int, str], torch.Tensor] = {(0, 'main'): basis.reshape(4, 5, dim),
                                               (0, 'offset'): unit(3, 4, dim, seed=6)}
    for rot in (90, 180, 270):
        qs[(rot, 'main')] = unit(5, 4, dim, seed=rot) if rot == 90 else unit(4, 5, dim, seed=rot)
        qs[(rot, 'offset')] = unit(4, 3, dim, seed=rot + 1) if rot == 90 else unit(3, 4, dim, seed=rot + 1)
    src: MemorySource = MemorySource(ref, qs, 'cls')
    cands: List[Candidate] = [Candidate(0, 'main', 0, 0, 0, 1.0), Candidate(0, 'main', 6, 0, 0, 0.5)]
    cols: Dict[str, torch.Tensor] = RerankerCorr(RerankerCorrConfig(encoder=ENC, tau=0.05, iters=20), 'cpu').score(
        scoring_input(src, ref, cands, n_regions=1))
    assert abs(float(cols['diag'][0]) - 1.0) < 1e-6 and abs(float(cols['contrast'][0]) - 1.0) < 1e-6
    assert float(cols['mnn'][0]) == 1.0 and float(cols['sinkhorn'][0]) > 0.99
    # the flat window: every tile is the same vector, so C[i, j] depends on i only
    assert abs(float(cols['contrast'][1])) < 1e-6, 'every window tile is equally good: no contrast'
    assert float(cols['sinkhorn'][1]) < 0.2, float(cols['sinkhorn'][1])
    return 'identity: 1, 1, 1, >0.99; flat: contrast 0'


# ── S6 candgraph ──────────────────────────────────────────────────────────────

def t_candgraph_neighbours_prominence_and_rotation_margin() -> str:
    # 0 and 1: one place at two rotations (same centre); 2: next to 0 (one tile right); 3: far away
    cands: List[Candidate] = [Candidate(0, 'main', 2, 3, 0, 0.9), Candidate(0, 'offset', 1, 3, 90, 0.5),
                              Candidate(0, 'main', 2, 4, 0, 0.7), Candidate(0, 'main', 9, 9, 0, 0.3)]
    cs: CandidateSet = cset(cands, n_regions=1)
    inp: ScoringInput = ScoringInput(QUERY, cs, incoming_table(cs), None, TILE)
    cols: Dict[str, torch.Tensor] = RerankerCandGraph(RerankerCandGraphConfig(on='first', radius=2.0),
                                                      'cpu').score(inp)
    n_nbr: List[float] = cols['n_nbr'].tolist()
    assert n_nbr == [2.0, 2.0, 2.0, 0.0], n_nbr
    assert abs(float(cols['nbr_support'][0]) - (0.5 + 0.7) / 2) < 1e-6
    assert abs(float(cols['prominence'][0]) - (0.9 - 0.6)) < 1e-6 and float(cols['prominence'][3]) == 0.0
    rm: List[float] = cols['rot_margin'].tolist()
    assert abs(rm[0] - (0.9 - 0.5)) < 1e-6 and abs(rm[1] - (0.5 - 0.9)) < 1e-6
    assert abs(rm[3] - (0.3 - 0.3)) < 1e-6, 'a place with no other rotation: own minus the lowest'
    return 'n_nbr, support, prominence, rot_margin'


# ── the first-phase methods ───────────────────────────────────────────────────

def planted(dim: int = 16) -> Tuple[MemorySource, Dict[Tuple[int, str], torch.Tensor], GridSource]:
    """One region; the query (rotation 0) is a block of the main grid, so that window is a perfect match."""
    src, ref = make_source(dim=dim, n_regions=1)
    ref[(0, 'main')] = unit(40, 40, dim, seed=50)
    ref[(0, 'offset')] = unit(39, 39, dim, seed=52)
    src.queries[(0, 'main')] = ref[(0, 'main')][3:7, 4:9].clone()
    src.queries[(0, 'offset')] = unit(3, 4, dim, seed=51)
    src.reference = ref
    return src, ref, grid_source(src, ref, n_regions=1)


def t_box_means_equals_the_loops() -> str:
    grid: torch.Tensor = unit(9, 10, 6, seed=7)
    query: torch.Tensor = unit(4, 5, 6, seed=8)
    for cells in ([(0, 4, 0, 5)], [(0, 2, 0, 2), (0, 2, 2, 5), (2, 4, 0, 2), (2, 4, 2, 5)]):
        got = RB.box_means(grid, query, cells, channels=4)
        for cell, g in zip(cells, got):
            a0, a1, b0, b1 = cell
            mq: torch.Tensor = query[a0:a1, b0:b1].mean(dim=(0, 1))
            for i in range(9 - 4 + 1):
                for j in range(10 - 5 + 1):
                    mw: torch.Tensor = grid[i + a0:i + a1, j + b0:j + b1].mean(dim=(0, 1))
                    want: float = float(F.cosine_similarity(mw, mq, dim=0))
                    assert abs(float(g[i, j]) - want) < 1e-5, (cell, i, j)
    return 'one cell and four, channels in slices of 4'


def _method_maps(source: GridSource) -> Dict[str, Tuple[Dict[Tuple[int, int, str], torch.Tensor], Callable[..., Candidate]]]:
    """Every first-phase method's whole ranking of the planted level, with the candidate it lists."""
    vote_cfg = RT.RetrieverTileVoteConfig(encoder=ENC, backend='exact', m=6, tau=0.02, smooth=0, hnsw_m=32,
                                          ef_search=128, nlist=1024, nprobe=16, chunk=64)
    bow_cfg = RC.RetrieverClusterBowConfig(encoder=ENC, k1=4, k2=6, iters=8, fit_tiles=100000, buckets=256,
                                           bigram_weight=0.5, seed=0, channels=16)
    return {
        'bagoftiles gap': (RB.window_maps(source, 1, 'gap', 8), Candidate),
        'bagoftiles quad': (RB.window_maps(source, 1, 'quad', 8), Candidate),
        'tilevote': (RT.window_maps(source, RT.TileIndex(RT.level_grids(source, 1), vote_cfg), vote_cfg),
                     RT.vote_candidate),
        'clusterbow': (RC.window_maps(source, RC.LevelBag(RT.level_grids(source, 1), bow_cfg), bow_cfg), Candidate)}


def t_every_first_phase_method_finds_the_planted_block_first_in_its_ranking_and_its_list() -> str:
    sys.path.insert(0, os.path.join(_HERE, '..', '..', 'bench_modules'))
    from BenchCommon import Answer, Answers, pools_of, ranked_ranks                     # noqa: PLC0415
    _, _, source = planted()
    frame: CandidateSet = cset([], n_regions=1)
    main: Tuple[int, str, int, int, int] = (0, 'main', 3, 4, 0)
    offset: Tuple[int, str, int, int, int] = (0, 'offset', 3, 3, 0)

    def answers_at(m: Tuple[int, str, int, int, int], o: Tuple[int, str, int, int, int]) -> Answers:
        return Answers(Answer(m[1], m[0], m[2], m[3], m[4], (0, 0, 0.0, 0.0), 0.0),
                       Answer(o[1], o[0], o[2], o[3], o[4], (0, 0, 0.0, 0.0), 0.0))

    for name, (maps, make) in _method_maps(source).items():
        assert maps and all(m.dim() == 2 for m in maps.values()), name
        listed: CandidateSet = top_candidates(maps, frame, 5, 1.0, make)
        assert len(listed) >= 1 and listed.candidates[0].key() == main, (name, listed.candidates[0])
        ranks = ranked_ranks(maps, answers_at(main, offset), pools_of(maps))
        assert ranks['rank_main'] == 1 and ranks['rank_main_in_all'] == 1, (name, ranks)
        assert 1 <= ranks['rank_offset_in_offset'] <= pools_of(maps).offset, (name, ranks)
        missing = ranked_ranks(maps, answers_at((7, 'main', 0, 0, 0), offset), pools_of(maps))
        assert missing['rank_main'] == pools_of(maps).main + 1, 'a window no map holds is the miss value'
        if name == 'tilevote':
            top = listed.candidates[0]
            assert isinstance(top, VoteCandidate) and top.votes == top.score
    return 'bag (gap, quad), vote, cluster bag: window (3, 4), first of the whole ranking and of the list'


def t_tile_index_exact_is_the_true_nearest_and_the_graph_agrees() -> str:
    _, ref, _ = planted()
    grids = [(0, 'main', ref[(0, 'main')]), (0, 'offset', ref[(0, 'offset')])]
    cfg = lambda backend: RT.RetrieverTileVoteConfig(                                  # noqa: E731
        encoder=ENC, backend=backend, m=6, tau=0.05, smooth=1, hnsw_m=16, ef_search=64, nlist=8, nprobe=4,
        chunk=32)
    exact: RT.TileIndex = RT.TileIndex(grids, cfg('exact'))
    every: torch.Tensor = torch.cat([g.reshape(-1, g.shape[-1]) for _, _, g in grids])
    q: torch.Tensor = unit(7, 16, seed=60)
    vals, gid, flat = exact.search(q, 5)
    want = (q @ every.T).topk(5, dim=1)
    assert torch.allclose(vals, want.values, atol=1e-6)
    glob = torch.tensor(exact.edges)[gid] + flat
    assert torch.equal(glob, want.indices), 'the hits are the right tiles'
    try:
        import faiss  # noqa: F401, PLC0415
    except ImportError:
        return 'exact only (faiss not importable here)'
    hnsw: RT.TileIndex = RT.TileIndex(grids, cfg('hnsw'))
    hv, hg, hf = hnsw.search(q, 5)
    top1 = torch.tensor(hnsw.edges)[hg[:, 0]] + hf[:, 0]
    assert (top1 == want.indices[:, 0]).float().mean() >= 0.85, 'the graph finds the nearest almost always'
    return 'exact == brute force; hnsw top-1 agrees'


_TESTS = [t for n, t in sorted(globals().items(), key=lambda kv: kv[0])
          if n.startswith('t_')]


def main() -> int:
    for fn in _TESTS:
        check(fn.__name__[2:].replace('_', ' '), fn)
    failed: List[str] = [n for n, e in _RESULTS if e is not None]
    print(f'\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed')
    if failed:
        print('failed: ' + ', '.join(failed))
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
