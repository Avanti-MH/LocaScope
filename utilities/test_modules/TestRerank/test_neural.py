#!/usr/bin/env python3
"""Tests for the learned part of stage 2: the networks (`aiNNModel/models/RerankHeads.py`), their
checkpoints, the retrievers that load them (`RerankerNeural.py`: TileGT, CorrNet, Stack;
`RetrieverWinEmbed.WindowEmbeddings`), and the training code's pieces (`training/RerankHead`: labels, losses,
metrics, steps, the embedding's loss).

    python utilities/test_modules/TestRerank/test_neural.py

No slide, no encoder, no GPU. The networks are tiny and untrained, so what is
tested is what must be true of ANY weights -- shapes, that a checkpoint rebuilds
the network it saved, that the tensors a network is given are the window's tiles
in the order the retriever compared them, that a window of the slide and a query
laid out the same way embed alike -- and that the training can learn the one
thing a toy example offers: a planted window whose tiles equal the query's.
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', '..'))

from _paths import setup_import_paths                            # noqa: E402

setup_import_paths()

import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402

from test_context import cset, make_source, unit                 # noqa: E402
from test_rerank import ENC, QUERY, REUSE, bound, planted, ported_upstream, random_candidates   # noqa: E402
from Checkpoints import build_rerank_from_checkpoint, save_rerank_checkpoint   # noqa: E402
from RerankChoices import build_rerank                           # noqa: E402
from stage2_retrieval import RerankerNeural as RN, RerankerScorers as RS   # noqa: E402
from stage2_retrieval.Context import GridSource, ScoreTable, WindowBatch, top_candidates   # noqa: E402
from stage2_retrieval.RerankerNeural import (N_GLOBAL, N_PAIR, N_SCALAR, NeuralInputs,   # noqa: E402
                                             neural_inputs, pair_features, stack_features)
from stage2_retrieval.RetrieverWinEmbed import WindowEmbeddings                           # noqa: E402
from stage2_retrieval.StageInterface import (Candidate, CandidateSet, RerankedCandidate,   # noqa: E402
                                             ShiftedCandidate)

from training.RerankHead.Embed import EmbedFov, EmbedVisit, _iou_and_disjoint, embed_loss  # noqa: E402
from training.RerankHead.Labels import (Boxes, LabelConfig, Labels, choose_subset,        # noqa: E402
                                        label_windows)
from training.RerankHead.Losses import (bce_loss, listwise_loss, loc_loss, pool_softmax_loss,   # noqa: E402
                                        rerank_loss, rot_loss, topk_hinge_loss)
from training.RerankHead.Metrics import first_hit_rank, recall_at                         # noqa: E402
from training.RerankHead.Steps import build_head, forward_list, train_step                # noqa: E402
from training.RerankHead.Visits import FovData                                            # noqa: E402

_RESULTS: List[Tuple[str, Optional[BaseException]]] = []


def check(name: str, fn: Callable[[], Optional[str]]) -> None:
    try:
        out: Optional[str] = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   {out}' if out else ''))
    except Exception as e:                                       # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n          {type(e).__name__}: {e}')


DIM = 16
TILES = 20


def small_head(arch: str, dropout: float = 0.0) -> torch.nn.Module:
    return build_head(arch, DIM, TILES, dim=32, layers=1, attn_heads=4, dropout=dropout, device='cpu')


def inputs_of(src, cands: List[Candidate], pooling: str = 'cls') -> NeuralInputs:
    cs: CandidateSet = cset(cands, n_regions=1)
    batch: WindowBatch = WindowBatch.build(cs, src, pooling)
    return neural_inputs(batch, torch.tensor([c.score for c in cands]), torch.arange(1, len(cands) + 1),
                         [c.rotation for c in cands], [c.lattice for c in cands])


# ── the tensors a network is given ───────────────────────────────────────────

def t_neural_inputs_are_the_windows_tiles_in_order() -> str:
    src, ref = make_source(dim=DIM, n_regions=1)
    cands: List[Candidate] = random_candidates(ref, 24, seed=11, regions=1)
    ni: NeuralInputs = inputs_of(src, cands)
    assert ni.q.shape == (24, TILES, DIM) and ni.w.shape == ni.q.shape
    assert ni.pos.shape == (24, TILES, 2) and ni.scalars.shape == (24, TILES, N_SCALAR)
    assert ni.corr.shape == (24, TILES, TILES) and ni.glob.shape == (24, N_GLOBAL)
    for k, c in enumerate(cands):
        qg: torch.Tensor = src.queries[(c.rotation, 'main')]
        rows, cols = qg.shape[:2]
        g: torch.Tensor = ref[(c.region_index, c.lattice)]
        assert ni.shapes[k] == (rows, cols)
        for i in range(rows):
            for j in range(cols):
                t: int = i * cols + j
                assert torch.equal(ni.q[k, t], qg[i, j]) and torch.equal(ni.w[k, t], g[c.row + i, c.col + j])
                assert ni.pos[k, t].tolist() == [i, j]
        assert torch.allclose(ni.corr[k], ni.q[k] @ ni.w[k].T, atol=1e-6)
        assert torch.allclose(ni.scalars[k, :, 0], torch.diagonal(ni.corr[k]), atol=1e-6)
    assert torch.isfinite(ni.glob).all() and ni.glob[:, 3:7].sum(dim=1).eq(1).all(), 'rotation one-hot'
    return '24 windows, both shapes, tile i of query and window aligned'


def t_pair_and_stack_features() -> str:
    centres: torch.Tensor = torch.tensor([[0.0, 0.0], [512.0, 0.0], [9000.0, 0.0]])
    rot: torch.Tensor = torch.tensor([0, 90, 0])
    p: torch.Tensor = pair_features(centres, rot, 512.0)
    assert p.shape == (3, 3, N_PAIR)
    assert p[0, 1, 0] == 1.0 / 8.0 and p[0, 1, 2] == 0.0 and p[0, 1, 3] == 1.0
    assert p[0, 2, 0] == 1.0 and p[0, 2, 2] == 1.0 and p[0, 2, 3] == 0.0, 'clipped, same rotation, far'
    t: ScoreTable = ScoreTable(4)
    t.put('a', torch.tensor([3.0, 1.0, 2.0, 4.0]))
    f: torch.Tensor = stack_features(t, ['a'])
    assert f.shape == (4, 2) and np.allclose(f[:, 0].tolist(), [2 / 3, 0.0, 1 / 3, 1.0]), f[:, 0].tolist()
    assert abs(float(f[:, 1].mean())) < 1e-6
    return 'pairs and ranks'


# ── the networks ─────────────────────────────────────────────────────────────

def t_every_head_runs_and_has_a_gradient() -> str:
    src, ref = make_source(dim=DIM, n_regions=1)
    cands: List[Candidate] = random_candidates(ref, 12, seed=12, regions=1)
    ni: NeuralInputs = inputs_of(src, cands)
    centres: torch.Tensor = torch.rand(12, 2) * 4000
    pair: torch.Tensor = pair_features(centres, torch.tensor([c.rotation for c in cands]), 512.0)
    for arch in ('tilegt', 'tilegt_ctx', 'corrnet'):
        model: torch.nn.Module = small_head(arch)
        out: Dict[str, torch.Tensor] = forward_list(arch, model, ni, pair)
        assert out['logit'].shape == (12,) and torch.isfinite(out['logit']).all(), arch
        if arch != 'corrnet':
            assert out['rot'].shape == (12,) and out['loc'].shape == (12, 2)
        out['logit'].sum().backward()
        grads: List[torch.Tensor] = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum() > 0 for g in grads), arch
    return 'tilegt, tilegt_ctx, corrnet'


def t_a_checkpoint_rebuilds_the_network() -> str:
    src, ref = make_source(dim=DIM, n_regions=1)
    cands: List[Candidate] = random_candidates(ref, 8, seed=13, regions=1)
    ni: NeuralInputs = inputs_of(src, cands)
    pair: torch.Tensor = torch.rand(8, 8, N_PAIR)
    with tempfile.TemporaryDirectory() as d:
        for arch in ('tilegt', 'tilegt_ctx', 'corrnet'):
            model: torch.nn.Module = small_head(arch)
            path: str = save_rerank_checkpoint(os.path.join(d, f'{arch}.pt'), model=model, arch=arch, pooling='cls',
                                               encoder='uni2', epoch=3, val={'R@30': 0.5}, run_args={'x': 1},
                                               extra={'note': 'a'}).as_posix()
            back, ckpt = build_rerank_from_checkpoint(path, 'cpu')
            assert ckpt['arch'] == arch and ckpt['pooling'] == 'cls' and ckpt['extra'] == {'note': 'a'}
            model.eval()
            a = forward_list(arch, model, ni, pair)['logit']
            b = forward_list(arch, back, ni, pair)['logit']
            assert torch.allclose(a, b, atol=1e-6), arch
        emb = build_rerank('winembed', dict(in_dim=DIM, dim=24, shapes=((4, 5), (5, 4))))
        p2: str = save_rerank_checkpoint(os.path.join(d, 'e.pt'), model=emb, arch='winembed', pooling='cls',
                                         encoder='uni2', epoch=1, val={}, run_args={}, extra={}).as_posix()
        back_e, _ = build_rerank_from_checkpoint(p2, 'cpu')
        grid: torch.Tensor = unit(9, 10, DIM, seed=1)
        assert torch.allclose(emb.eval().windows(grid, (4, 5)), back_e.windows(grid, (4, 5)), atol=1e-6)
        stack = build_rerank('stack', dict(n_features=4, hidden=8, dropout=0.0))
        p3: str = save_rerank_checkpoint(os.path.join(d, 's.pt'), model=stack, arch='stack', pooling='cls',
                                         encoder='uni2', epoch=1, val={}, run_args={},
                                         extra={'columns': ['first', 'dual']}).as_posix()
        assert build_rerank_from_checkpoint(p3, 'cpu')[1]['extra']['columns'] == ['first', 'dual']
    return 'tilegt, tilegt_ctx, corrnet, winembed, stack'


def t_a_window_of_the_slide_and_a_query_embed_alike() -> str:
    model = build_rerank('winembed', dict(in_dim=DIM, dim=24, shapes=((4, 5), (5, 4)))).eval()
    grid: torch.Tensor = unit(11, 13, DIM, seed=2)
    with torch.no_grad():
        for shape in ((4, 5), (5, 4)):
            w: torch.Tensor = model.windows(grid, shape)
            assert w.shape == (11 - shape[0] + 1, 13 - shape[1] + 1, 24)
            assert torch.allclose(w.norm(dim=-1), torch.ones(w.shape[:2]), atol=1e-5)
            for r, c in ((0, 0), (3, 4), (11 - shape[0], 13 - shape[1])):
                q: torch.Tensor = model.query(grid[r:r + shape[0], c:c + shape[1]])
                assert torch.allclose(w[r, c], q, atol=1e-5), (shape, r, c)
    return 'windows(grid)[r, c] == query(grid[r:r+a, c:c+b])'


# ── the rerankers and the retriever that load them ───────────────────────────

def t_the_learned_rerankers_order_by_their_column_and_a_stacker_reads_the_chains_columns() -> str:
    src, ref = make_source(dim=DIM, n_regions=1)
    cands: List[Candidate] = random_candidates(ref, 20, seed=14, regions=1)
    up = ported_upstream(src, ref, cset(cands, n_regions=1), n_regions=1)
    with tempfile.TemporaryDirectory() as d:
        paths: Dict[str, str] = {}
        for arch in ('tilegt', 'tilegt_ctx', 'corrnet'):
            paths[arch] = save_rerank_checkpoint(os.path.join(d, f'{arch}.pt'), model=small_head(arch).eval(),
                                                 arch=arch, pooling='cls', encoder='uni2', epoch=1, val={},
                                                 run_args={}, extra={}).as_posix()
        stack = build_rerank('stack', dict(n_features=4, hidden=8, dropout=0.0)).eval()
        paths['stack'] = save_rerank_checkpoint(os.path.join(d, 's.pt'), model=stack, arch='stack', pooling='cls',
                                                encoder='uni2', epoch=1, val={}, run_args={},
                                                extra={'columns': ['first', 'dual']}).as_posix()
        done: List[str] = []
        for new, col, kind in (
                (RN.RerankerTileGT(RN.RerankerTileGTConfig(encoder=ENC, weights=paths['tilegt'], shift_loc=False,
                                                           batch=7, reuse=REUSE), 'cpu'), 'tilegt', RerankedCandidate),
                (RN.RerankerTileGT(RN.RerankerTileGTConfig(encoder=ENC, weights=paths['tilegt'], shift_loc=False,
                                                           order_by='tilegt_rot', batch=7, reuse=REUSE), 'cpu'),
                 'tilegt_rot', RerankedCandidate),
                (RN.RerankerTileGT(RN.RerankerTileGTConfig(encoder=ENC, weights=paths['tilegt_ctx'], shift_loc=True,
                                                           batch=7, reuse=REUSE), 'cpu'), 'tilegt', ShiftedCandidate),
                (RN.RerankerCorrNet(RN.RerankerCorrNetConfig(encoder=ENC, weights=paths['corrnet'], batch=7,
                                                             reuse=REUSE), 'cpu'), 'corrnet', RerankedCandidate)):
            out = bound(new).retrieve(up, QUERY)
            v: List[float] = out.extras['columns'].value.cols[col]
            assert len(v) == 20 and np.isfinite(v).all(), col
            assert [c.score for c in out.candidates] == sorted(v, reverse=True), col
            assert all(type(c) is kind for c in out.candidates), (col, kind)
            done.append(f'{col}{" (shift_loc)" if kind is ShiftedCandidate else ""}')
        dual = bound(RS.RerankerDual(RS.RerankerDualConfig(encoder=ENC, order_by='dual', reuse=REUSE), 'cpu'))
        stacker = bound(RN.RerankerStack(RN.RerankerStackConfig(weights=paths['stack']), 'cpu'))
        after = dual.retrieve(up, QUERY)
        out = stacker.retrieve(after, QUERY)
        with torch.no_grad():
            want = stack(stack_features_of(after, ['first', 'dual'])).flatten().tolist()
        assert np.allclose(sorted(want, reverse=True), [c.score for c in out.candidates], atol=1e-5)
        done.append('dual -> stack')
        try:
            bound(RN.RerankerStack(RN.RerankerStackConfig(weights=paths['stack']), 'cpu')).retrieve(up, QUERY)
            raise AssertionError('a stacker ran with no reranker before it')
        except KeyError:
            done.append('no dual: refused')
    return ', '.join(done)


def stack_features_of(out: Any, columns: List[str]) -> torch.Tensor:
    """The stacker's input for the list `out` hands on, built by hand: its incoming table and the columns the
    rerankers upstream wrote, by window."""
    from stage2_retrieval.Context import incoming_table                              # noqa: PLC0415
    table: ScoreTable = incoming_table(out.candidates)
    keys = [c.key() for c in out.candidates.candidates]
    for name in columns:
        if name not in table:
            table.put(name, torch.tensor(out.extras['columns'].value.column(name, keys), dtype=torch.float32))
    return stack_features(table, columns)


def t_a_learned_reranker_refuses_features_it_was_not_trained_on() -> str:
    src, ref = make_source(dim=DIM, n_regions=1)
    up = ported_upstream(src, ref, cset(random_candidates(ref, 6, seed=15, regions=1), n_regions=1), n_regions=1)
    with tempfile.TemporaryDirectory() as d:
        path: str = save_rerank_checkpoint(os.path.join(d, 'a.pt'), model=small_head('tilegt').eval(), arch='tilegt',
                                           pooling='cls_avg', encoder='uni2', epoch=1, val={}, run_args={},
                                           extra={}).as_posix()
        try:
            bound(RN.RerankerTileGT(RN.RerankerTileGTConfig(encoder=ENC, weights=path, batch=8, reuse=REUSE),
                                    'cpu')).retrieve(up, QUERY)
        except ValueError as e:
            assert 'cls_avg' in str(e)
            return 'the message names the pooling'
    raise AssertionError('a checkpoint trained on cls_avg scored cls features')


def t_the_embedding_scores_are_dot_products_and_banding_is_exact() -> str:
    src, ref, source = planted(dim=DIM)
    model = build_rerank('winembed', dict(in_dim=DIM, dim=24, shapes=((4, 5), (5, 4)))).eval()
    with tempfile.TemporaryDirectory() as d:
        path: str = save_rerank_checkpoint(os.path.join(d, 'e.pt'), model=model, arch='winembed', pooling='cls',
                                           encoder='uni2', epoch=1, val={}, run_args={}, extra={}).as_posix()
        banded: WindowEmbeddings = WindowEmbeddings(path, 7)
        maps = banded.window_maps(source, 1, 1)
    cs: CandidateSet = top_candidates(maps, cset([], n_regions=1), 6, 1.0, Candidate)
    assert len(cs) >= 1
    grid: torch.Tensor = ref[(0, 'main')]
    with torch.no_grad():
        full: torch.Tensor = model.windows(grid, (4, 5))
        assert torch.allclose(banded.embed(model, grid, (4, 5)).float(), full, atol=2e-3), 'bands == one pass'
        best: Candidate = cs.candidates[0]
        e_q: torch.Tensor = model.query(src.queries[(best.rotation, 'main')])
        e_w: torch.Tensor = model.windows(ref[(0, best.lattice)],
                                          tuple(src.queries[(best.rotation, 'main')].shape[:2]))[best.row, best.col]
    assert abs(best.score - float(e_w @ e_q)) < 5e-3, (best.score, float(e_w @ e_q))
    try:
        WindowEmbeddings(path, 7).window_maps(GridSource(source.level, source.query, 'cls_avg'), 1, 1)
        raise AssertionError('features of another pooling were embedded')
    except (ValueError, KeyError, FileNotFoundError):
        pass
    return 'score = embedding dot product (fp16 rounding); another pooling refused'


# ── the training's pieces ────────────────────────────────────────────────────

def t_labels_pos_neg_and_the_rest_ignored() -> str:
    tile_px: float = 512.0
    cfg: LabelConfig = LabelConfig(pos_iou=0.5, hit_iou=0.5, loc_min_iou=0.1, fov_margin=1.2)
    truth: np.ndarray = np.array([4000.0, 4000.0, 4000.0 + 5 * tile_px, 4000.0 + 4 * tile_px])
    fov: np.ndarray = np.array([3000.0, 3000.0, 8000.0, 8000.0])
    where: Boxes = Boxes(truth=truth, fov=fov, rotation=0, centre=(float(truth[0] + truth[2]) / 2, float(truth[1] + truth[3]) / 2))

    def at(dx: float, dy: float, w: float = 5 * tile_px, h: float = 4 * tile_px) -> List[float]:
        return [truth[0] + dx, truth[1] + dy, truth[0] + dx + w, truth[1] + dy + h]

    boxes: np.ndarray = np.array([at(0, 0), at(tile_px, 0), at(2 * tile_px, 0), at(50000, 0), at(0, 0),
                                  at(0, 0, 4 * tile_px, 5 * tile_px)])
    rot: np.ndarray = np.array([0, 0, 0, 0, 90, 90])
    lab: Labels = label_windows(boxes, rot, where, tile_px, cfg)
    assert lab.pos.tolist() == [True, True, False, False, False, False], lab.pos
    assert lab.neg.tolist() == [False, False, False, True, True, True], lab.neg      # far; wrong rotation twice
    assert lab.ignore.tolist() == [False, False, True, False, False, False], lab.ignore
    assert abs(lab.iou[1] - 0.6666666666666666) < 1e-9 and abs(lab.iou[2] - 0.42857142857142855) < 1e-9
    assert lab.loc_mask.tolist() == [True, True, True, False, True, True]
    assert abs(lab.loc[1, 0] - (-1.0)) < 1e-9 and abs(lab.loc[1, 1]) < 1e-9, 'the window is a tile right of the truth'
    assert lab.hit.tolist() == [True, True, False, False, False, False]
    return 'pos / wrong-rotation neg / overlap ignored / far neg'


def t_choose_subset_keeps_every_positive_and_the_hardest_negatives() -> str:
    rng: np.random.Generator = np.random.default_rng(0)
    k: int = 400
    pos: np.ndarray = np.zeros(k, bool)
    pos[[7, 90]] = True
    neg: np.ndarray = ~pos
    neg[[3, 4, 5]] = False
    ignore: np.ndarray = ~(pos | neg)
    lab: Labels = Labels(iou=np.zeros(k), pos=pos, neg=neg, ignore=ignore, rot_ok=np.ones(k, bool), loc=np.zeros((k, 2)),
                         loc_mask=ignore, hit=pos)
    sel: np.ndarray = choose_subset(lab, 64, 1.0, rng, n_loc=2)
    assert sel.tolist() == sorted(set(sel.tolist())) and len(sel) <= 64
    assert {7, 90} <= set(sel.tolist()), 'every positive'
    assert set(sel.tolist()) & {3, 4, 5} <= {3, 4, 5} and len(set(sel.tolist()) & {3, 4, 5}) <= 2
    hard: List[int] = [i for i in sel.tolist() if neg[i]]
    assert hard[:10] == [i for i in range(k) if neg[i]][:10], 'with hard_share 1 the negatives are the earliest ones'
    return f'{len(sel)} of {k}'


def t_losses_mean_what_they_say() -> str:
    logit: torch.Tensor = torch.tensor([2.0, 0.0, 1.0, -1.0, 5.0])
    pos: torch.Tensor = torch.tensor([True, False, False, False, False])
    neg: torch.Tensor = torch.tensor([False, True, True, True, False])      # index 4: ignored, however high
    want: float = -(2.0 - float(torch.logsumexp(torch.tensor([2.0, 0.0, 1.0, -1.0]), 0)))
    assert abs(float(listwise_loss(logit, pos, neg)) - want) < 1e-6
    assert float(listwise_loss(logit, torch.zeros(5, dtype=torch.bool), neg)) == 0.0, 'no positive: no loss'
    # the best negative is 1.0: 1 + 1 - 2 = 0 at margin 1, and 1 at margin 2
    assert abs(float(topk_hinge_loss(logit, pos, neg, k=1, margin=1.0))) < 1e-9
    assert abs(float(topk_hinge_loss(logit, pos, neg, k=1, margin=2.0)) - 1.0) < 1e-9
    # the third best (here the lowest) is -1.0: it takes a margin of 3 to reach the positive
    assert abs(float(topk_hinge_loss(logit, pos, neg, k=3, margin=4.0)) - 1.0) < 1e-9
    assert float(bce_loss(logit, pos, neg, pos_weight=2.0)) > 0
    assert float(rot_loss(torch.tensor([3.0, -3.0]), torch.tensor([True, True]), torch.tensor([True, False]))) < 0.1
    assert float(loc_loss(torch.zeros(2, 2), torch.ones(2, 2), torch.tensor([False, False]))) == 0.0
    flat_pos, flat_keep = pos, pos | neg
    assert abs(float(pool_softmax_loss(logit, flat_pos, flat_keep, 1.0)) - want) < 1e-6
    out = {'logit': logit.clone().requires_grad_(), 'rot': torch.zeros(5, requires_grad=True),
           'loc': torch.zeros(5, 2, requires_grad=True)}
    terms = rerank_loss(out, pos, neg, pos | neg, torch.ones(5, dtype=torch.bool), torch.zeros(5, 2),
                        torch.ones(5, dtype=torch.bool), dict(topk=0.5, bce=0.1, rot=0.5, loc=0.5), 1.0, 2, 1.0, 5.0)
    terms['total'].backward()
    assert out['logit'].grad is not None and out['logit'].grad[4] == 0.0, 'an ignored window gets no gradient'
    return 'listwise, hinge, bce, rot, loc, pool; ignored window: no gradient'


def t_metrics() -> str:
    assert first_hit_rank(np.array([0.1, 0.9, 0.5]), np.array([False, False, True])) == 2.0
    assert first_hit_rank(np.array([0.5, 0.5]), np.array([False, True])) == 1.0, 'a tie goes to the hit'
    assert first_hit_rank(np.array([0.5]), np.array([False])) == float('inf')
    r = recall_at([1.0, 3.0, 40.0, float('inf')], ks=(1, 5, 100))
    assert r['R@1'] == 0.25 and r['R@5'] == 0.5 and r['R@100'] == 0.75 and r['ceiling'] == 0.75 and r['n'] == 4
    return 'rank, ties, ceiling'


def planted_list(n_other: int = 29, seed: int = 20) -> FovData:
    """A list whose first candidate is the query itself (its tiles equal the
    window's) and the rest are random windows: a head can learn `cos_ii`."""
    src, ref, _ = planted(dim=DIM)
    rng: np.random.Generator = np.random.default_rng(seed)
    cands: List[Candidate] = [Candidate(0, 'main', 3, 4, 0, 1.0)]
    while len(cands) < n_other + 1:
        rot: int = int(rng.choice([0, 90, 180, 270]))
        qr, qc = (4, 5) if rot in (0, 180) else (5, 4)
        c: Candidate = Candidate(0, 'main', int(rng.integers(0, 40 - qr + 1)), int(rng.integers(0, 40 - qc + 1)), rot, 0.5)
        if c.key() not in {x.key() for x in cands}:
            cands.append(c)
    ni: NeuralInputs = inputs_of(src, cands)
    k: int = len(cands)
    pos: torch.Tensor = torch.zeros(k, dtype=torch.bool)
    pos[0] = True
    return FovData(ni=ni, pos=pos, neg=~pos, covers=pos.clone(), rot_ok=torch.ones(k, dtype=torch.bool),
                   loc=torch.zeros(k, 2), loc_mask=pos.clone(), pair=torch.zeros(k, k, N_PAIR),
                   slide='s', level=0, index=0, n_total=k)


def t_a_head_learns_the_planted_window() -> str:
    torch.manual_seed(0)
    lists: List[FovData] = [planted_list(seed=s) for s in range(4)]
    weights: Dict[str, float] = dict(topk=0.5, bce=0.1, rot=0.0, loc=0.0)
    for arch in ('tilegt', 'corrnet'):
        model: torch.nn.Module = small_head(arch)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        first: float = train_step(arch, model, opt, lists, weights, 1.0, 5, 1.0, 5.0, 1.0)['list']
        for _ in range(60):
            last: float = train_step(arch, model, opt, lists, weights, 1.0, 5, 1.0, 5.0, 1.0)['list']
        assert last < 0.5 * first, (arch, first, last)
        with torch.no_grad():
            d: FovData = lists[0]
            logit: torch.Tensor = forward_list(arch, model.eval(), d.ni, d.pair)['logit']
        assert int(logit.argmax()) == 0, (arch, 'the planted window is not on top')
    return 'tilegt and corrnet: listwise loss falls by half, the planted window ranks first'


def t_the_embedding_loss_learns_the_planted_block() -> str:
    torch.manual_seed(1)
    f_dim, tile_l0 = DIM, 512.0
    main, off = unit(30, 30, f_dim, seed=70), unit(29, 29, f_dim, seed=71)
    block: torch.Tensor = main[3:7, 4:9].clone()                     # 4 x 5 tiles at (row 3, col 4)
    queries: Dict[int, torch.Tensor] = {0: block, 90: unit(5, 4, f_dim, seed=72), 180: unit(4, 5, f_dim, seed=73),
                                        270: unit(5, 4, f_dim, seed=74)}
    xs_m, ys_m = torch.arange(30, dtype=torch.float64) * tile_l0, torch.arange(30, dtype=torch.float64) * tile_l0
    xs_o, ys_o = xs_m[:29] + tile_l0 / 2, ys_m[:29] + tile_l0 / 2
    visit: EmbedVisit = EmbedVisit(grids={(0, 'main'): main, (0, 'offset'): off},
                                   origin={(0, 'main'): (xs_m, ys_m), (0, 'offset'): (xs_o, ys_o)},
                                   tile_l0=tile_l0, fovs=[])
    truth: np.ndarray = np.array([4 * tile_l0, 3 * tile_l0, 9 * tile_l0, 7 * tile_l0])
    cx, cy = (truth[0] + truth[2]) / 2, (truth[1] + truth[3]) / 2
    fov_box: np.ndarray = np.array([cx - 3000, cy - 3000, cx + 3000, cy + 3000])
    where: Boxes = Boxes(truth=truth, fov=fov_box, rotation=0, centre=(float(cx), float(cy)))
    fov: EmbedFov = EmbedFov(queries=queries, where=where, slide='s', level=0, index=0)
    model = build_rerank('winembed', dict(in_dim=f_dim, dim=24, shapes=((4, 5), (5, 4))))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first: float = float(embed_loss(model, visit, [fov], 0.5, 0.1))
    for _ in range(80):
        opt.zero_grad()
        loss: torch.Tensor = embed_loss(model, visit, [fov], 0.5, 0.1)
        loss.backward()
        opt.step()
    assert float(loss) < 0.5 * first, (first, float(loss))
    with torch.no_grad():
        w = model.eval().windows(main, (4, 5))
        top = (w @ model.query(block)).reshape(-1).argmax().item()
    row, col = divmod(top, int(w.shape[1]))
    assert abs(row - 3) <= 1 and abs(col - 4) <= 1, (row, col)      # a window that covers the truth: a positive
    return f'loss {first:.2f} -> {float(loss):.3f}; the best window of the pool covers the planted block'


def t_embed_geometry_matches_the_box_code() -> str:
    from stage2_retrieval.Context import intersection, iou_matrix                  # noqa: PLC0415
    xs, ys = torch.arange(7, dtype=torch.float64) * 300.0, torch.arange(6, dtype=torch.float64) * 300.0
    truth: np.ndarray = np.array([700.0, 500.0, 2300.0, 1700.0])
    fov: np.ndarray = np.array([400.0, 300.0, 2600.0, 2000.0])
    w, h = 1200.0, 900.0
    iou, disjoint = _iou_and_disjoint(xs[None, :], ys[:, None], w, h, truth, fov)
    boxes: np.ndarray = np.array([[x, y, x + w, y + h] for y in ys.tolist() for x in xs.tolist()])
    want_iou: np.ndarray = iou_matrix(boxes, truth[None, :])[:, 0].reshape(6, 7)
    want_dis: np.ndarray = (intersection(boxes, fov[None, :])[:, 0] <= 0).reshape(6, 7)
    assert np.allclose(iou.numpy(), want_iou) and (disjoint.numpy() == want_dis).all()
    return 'IoU and disjointness by broadcasting == the box functions'


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
