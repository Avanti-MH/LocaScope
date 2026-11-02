#!/usr/bin/env python3
"""Tests for GloRet and LogLoRet (`stage2_retrieval/proposal.md`): the networks (`aiNNModel/models/GloRet.py`), the
retrievers that load them (`stage2_retrieval/RetrieverGloRet.py`), and their training (`training/GloRet`: the labels,
the samples, the losses, the steps, the patch tokens of TilePool `attnpool` and their stages, the resource meter),
and the small tools that came with it (the wandb plan, the earlier wandb runs of an arm).

    python utilities/test_modules/TestGloRet/test_gloret.py

No slide, no encoder, no GPU: the networks are tiny and untrained, the level is a planted one (the query is a block
of the main grid, so that window is a perfect match), the tile reads are stand-ins. It sits apart from
`TestRerank` (the stage 2 re-rankers and RerankHead), so a run of one never overwrites the log of the other:
`jobscripts/GloRet/TestGloRet.sh`.
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'TestRerank'))      # test_context, test_rerank: the stand-in level
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', '..'))

from _paths import setup_import_paths                            # noqa: E402

setup_import_paths()

import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402

from test_context import cset, make_source, unit                 # noqa: E402
from test_rerank import planted                                  # noqa: E402
from Checkpoints import build_rerank_from_checkpoint, save_rerank_checkpoint   # noqa: E402
from RerankChoices import build_rerank                           # noqa: E402
from stage2_retrieval.Context import GridSource, WindowBatch, top_candidates   # noqa: E402
from stage2_retrieval.StageInterface import Candidate, CandidateSet            # noqa: E402

_RESULTS: List[Tuple[str, Optional[BaseException]]] = []


def check(name: str, fn: Callable[[], Optional[str]]) -> None:
    try:
        out: Optional[str] = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   {out}' if out else ''))
    except (Exception, SystemExit) as e:                         # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n          {type(e).__name__}: {e}')


DIM = 16


def small_cfg(**kw: Any) -> Any:
    from GloRet import NetCfg                                                     # noqa: PLC0415
    base = dict(in_dim=16, d=32, d_r=24, layers=1, heads=4, pos_hidden=8, dropout=0.0, d_u=8, score_hidden=16)
    base.update(kw)
    return NetCfg(**base)


def t_gloret_every_context_and_global_en_that_exists_runs_on_any_window_shape() -> str:
    from GloRet import CONTEXT_EN, GLOBAL_EN, GloRetNet, LogLoRetNet, combo_problem  # noqa: PLC0415
    torch.manual_seed(0)
    made: Dict[int, int] = {1: 0, 2: 0}
    for arch, net_class in ((1, GloRetNet), (2, LogLoRetNet)):
        for context in CONTEXT_EN:
            for global_en in GLOBAL_EN:
                cfg = small_cfg(context=context, global_en=global_en)
                if combo_problem(context, global_en, arch) is not None:
                    try:
                        net_class(cfg)
                        raise AssertionError(f'Arch {arch} accepted {context}-{global_en}')
                    except ValueError:
                        continue
                net = net_class(cfg).eval()
                made[arch] += 1
                for shape in ((4, 5), (5, 4), (3, 3)):
                    n: int = shape[0] * shape[1]
                    with torch.no_grad():
                        if arch == 1:
                            q, w = net.encode(torch.randn(2, n, 16), shape, 'query'), net.encode(torch.randn(7, n, 16), shape)
                            want = (7, n, 32) if global_en == 'Align' else (7, 24)
                            assert tuple(w.shape) == want, (context, global_en, shape, tuple(w.shape))
                            assert tuple(net.score(q, w).shape) == (2, 7)
                        else:
                            out = net(torch.randn(1, n, 16), torch.randn(7, n, 16), shape)
                            assert tuple(out['score'].shape) == (7,) and torch.isfinite(out['score']).all()
    assert made == {1: 18, 2: 14}, made
    return 'Arch 1: 18 methods, Arch 2: 14, on shapes 4x5, 5x4 and 3x3; the others refused with a reason'


def t_a_window_vector_does_not_depend_on_the_other_windows_in_its_batch() -> str:
    from GloRet import GloRetNet                                                  # noqa: PLC0415
    torch.manual_seed(1)
    for context, global_en in (('SelfAttn', 'AttnPool'), ('GNN', 'GraphPool'), ('CLSTrans', 'Identity'),
                               ('SptSelfAttn', 'CNN'), ('Identity', 'MaskedMean'), ('SelfAttn', 'Align')):
        net = GloRetNet(small_cfg(context=context, global_en=global_en)).eval()
        w = torch.randn(6, 20, 16)
        with torch.no_grad():
            together = net.encode(w, (4, 5))
            alone = torch.cat([net.encode(w[i:i + 1], (4, 5)) for i in range(6)])
        assert torch.allclose(together, alone, atol=1e-5), (context, global_en)
    return 'a batch of windows encodes as each one alone (the vectors can be made once and kept)'


def t_align_compares_tile_to_tile_and_a_pooled_vector_does_not_see_the_order() -> str:
    from GloRet import GloRetNet                                                  # noqa: PLC0415
    torch.manual_seed(2)
    feats = torch.randn(1, 20, 16)
    shuffled = feats[:, torch.randperm(20)]
    aligned = GloRetNet(small_cfg(context='Identity', global_en='Align')).eval()
    pooled = GloRetNet(small_cfg(context='Identity', global_en='MaskedMean')).eval()
    with torch.no_grad():
        same = aligned.score(aligned.encode(feats, (4, 5), 'query'), aligned.encode(feats, (4, 5)))
        moved = aligned.score(aligned.encode(feats, (4, 5), 'query'), aligned.encode(shuffled, (4, 5)))
        assert abs(float(same) - 1.0 / float(aligned.tau)) < 1e-3, 'every tile is its own tile: cosine 1'
        assert float(moved) < float(same) - 1.0, 'the same tiles in another order are another place'
        a = pooled.score(pooled.encode(feats, (4, 5), 'query'), pooled.encode(feats, (4, 5)))
        b = pooled.score(pooled.encode(feats, (4, 5), 'query'), pooled.encode(shuffled, (4, 5)))
    assert abs(float(a) - float(b)) < 1e-4, 'a mean over tiles cannot tell the order'
    return 'Align: same tiles, other order -> far lower; masked mean of the same: identical'


def t_logloret_matches_sum_to_one_a_sinkhorn_plan_keeps_its_marginals_and_the_query_is_shared() -> str:
    from GloRet import LogLoRetNet, sinkhorn_plan                                 # noqa: PLC0415
    torch.manual_seed(3)
    fq, fw = torch.randn(1, 20, 16), torch.randn(5, 20, 16)
    scores = {}
    for mode in ('dustbin', 'sinkhorn', 'sigmoid'):
        net = LogLoRetNet(small_cfg(no_match=mode)).eval()
        with torch.no_grad():
            out = net(fq, fw, (4, 5), maps=True)
            twice = net(fq.expand(5, -1, -1), fw, (4, 5))
        assert tuple(out['A_fw'].shape) == (5, 4, 20, 21) and tuple(out['A_wf'].shape) == (5, 4, 20, 21)
        assert torch.allclose(out['score'], twice['score'], atol=1e-5), f'{mode}: the shared query is the same query'
        if mode != 'sigmoid':
            assert torch.allclose(out['A_fw'].sum(-1), torch.ones(5, 4, 20), atol=1e-4), mode
        scores[mode] = out['score']
    plan = sinkhorn_plan(torch.randn(2, 3, 6, 6), torch.tensor(0.5), 60)
    assert torch.allclose(plan[..., :6, :].sum(-1), torch.ones(2, 3, 6), atol=1e-3), 'a tile gives one unit in all'
    assert torch.allclose(plan[..., :, :6].sum(-2), torch.ones(2, 3, 6), atol=1e-3)
    assert torch.allclose(plan[..., 6, :].sum(-1), torch.full((2, 3), 6.0), atol=1e-2), 'the dustbin takes N'
    return 'dustbin and sinkhorn rows sum to 1; the plan: tile rows/columns 1, dustbin N; a shared query = N copies'


def t_logloret_runs_with_every_aggregation_gets_a_gradient_everywhere_and_the_two_losses_prefer_the_diagonal() -> str:
    from GloRet import LogLoRetNet, correspondence_loss, cycle_loss               # noqa: PLC0415
    torch.manual_seed(4)
    for agg in ('mean', 'attn', 'robust'):
        net = LogLoRetNet(small_cfg(evidence_agg=agg))
        out = net(torch.randn(1, 20, 16), torch.randn(6, 20, 16), (4, 5), s_up=torch.randn(6), maps=True)
        loss = out['score'].sum() + correspondence_loss(out['A_fw'], out['A_wf']) + cycle_loss(out['A_fw'], out['A_wf'])
        loss.backward()
        missing = [n for n, p in net.named_parameters() if p.grad is None and n != 'dustbin']
        assert not missing, (agg, missing)
    n = 20
    eye = torch.eye(n)
    sharp = torch.cat([eye, torch.zeros(n, 1)], dim=1).expand(2, 4, n, n + 1)
    flat = torch.full((2, 4, n, n + 1), 1.0 / (n + 1))
    assert float(correspondence_loss(sharp, sharp)) < float(correspondence_loss(flat, flat))
    assert float(cycle_loss(sharp, sharp)) < float(cycle_loss(flat, flat)) and float(cycle_loss(sharp, sharp)) < 1e-6
    return 'mean, attn, robust: all parameters trained; correspondence and cycle are lowest on the diagonal'


def t_gloret_and_logloret_checkpoints_rebuild_the_network_through_the_registry() -> str:
    from GloRet import GloRetNet, LogLoRetNet                                      # noqa: PLC0415
    torch.manual_seed(5)
    w, q = torch.randn(4, 20, 16), torch.randn(1, 20, 16)
    with tempfile.TemporaryDirectory() as tmp:
        for arch, net in (('gloret', GloRetNet(small_cfg(context='SelfAttn', global_en='AttnPool')).eval()),
                          ('logloret', LogLoRetNet(small_cfg(context='GNN', global_en='MaskedMean',
                                                             no_match='sinkhorn')).eval())):
            path = os.path.join(tmp, f'{arch}.pt')
            save_rerank_checkpoint(path, model=net, arch=arch, pooling='cls_avg', encoder='gigapath', epoch=1,
                                   val={}, run_args={})
            again, ckpt = build_rerank_from_checkpoint(path, 'cpu')
            assert ckpt['arch'] == arch and type(again) is type(net) and again.cfg == net.cfg
            with torch.no_grad():
                if arch == 'gloret':
                    assert torch.equal(net.encode(w, (4, 5)), again.encode(w, (4, 5)))
                else:
                    assert torch.equal(net(q, w, (4, 5))['score'], again(q, w, (4, 5))['score'])
    return 'both networks come back from a checkpoint with the same config and the same outputs'


def t_gloret_gives_a_window_the_same_score_in_both_interfaces_and_logloret_orders_its_list() -> str:
    from dataclasses import replace                                               # noqa: PLC0415
    from GloRet import GloRetNet, LogLoRetNet                                      # noqa: PLC0415
    from test_context import _enc_cfg, make_source                                 # noqa: PLC0415
    from stage2_retrieval.Context import WindowBatch                               # noqa: PLC0415
    from stage2_retrieval.Retriever import Extra, RetrievalOutput                  # noqa: PLC0415
    from stage2_retrieval.RetrieverGloRet import (RerankerGloRetConfig, RerankerLogLoRet,   # noqa: PLC0415
                                                  RerankerLogLoRetConfig, RetrieverGloRet,
                                                  RetrieverGloRetConfig)
    torch.manual_seed(7)
    src, _ = make_source(dim=16, n_regions=1)
    src.pooling = 'cls'
    cands = [Candidate(0, 'main', r, c, rot, 0.0) for rot in (0, 90) for r, c in ((0, 0), (3, 5), (7, 8))]
    cands += [Candidate(0, 'offset', 2, 3, 180, 0.0), Candidate(0, 'offset', 5, 6, 270, 0.0)]
    cs = cset(cands, n_regions=1)
    enc = _enc_cfg('uni2', 'cls')
    with tempfile.TemporaryDirectory() as tmp:
        for context, global_en in (('SelfAttn', 'AttnPool'), ('GNN', 'MaskedMean'), ('SelfAttn', 'Align'),
                                   ('CLSTrans', 'Identity')):
            net = GloRetNet(small_cfg(context=context, global_en=global_en)).eval()
            path = os.path.join(tmp, f'{context}-{global_en}.pt')
            save_rerank_checkpoint(path, model=net, arch='gloret', pooling='cls', encoder='uni2', epoch=1, val={},
                                   run_args={})
            first = RetrieverGloRet(RetrieverGloRetConfig(encoder=enc, weights=path, method=f'GloRet-{context}-{global_en}', band=3), 'cpu')
            first.amp = False
            maps = first.window_maps(src, 1, 0)
            later = RetrieverGloRet(RerankerGloRetConfig(encoder=enc, weights=path, method=f'GloRet-{context}-{global_en}', topk_in=100, batch=4), 'cpu')
            later.amp = False
            by_list = later.scores_of(WindowBatch.build(cs, src, 'cls'), 4)
            by_grid = torch.stack([maps[(c.rotation, 0, c.lattice)][c.row, c.col] for c in cands])
            # the grid's windows are kept in fp16, so the two agree to what half precision keeps of a cosine / tau
            assert torch.allclose(by_list, by_grid, atol=0.15), (context, global_en, (by_list - by_grid).abs().max())
            assert not torch.allclose(by_list, by_list[0].expand_as(by_list)), 'the windows score differently'
        net = LogLoRetNet(small_cfg(context='SelfAttn', global_en='AttnPool')).eval()
        path = os.path.join(tmp, 'logloret.pt')
        save_rerank_checkpoint(path, model=net, arch='logloret', pooling='cls', encoder='uni2', epoch=1, val={},
                               run_args={})
        ranker = RerankerLogLoRet(RerankerLogLoRetConfig(encoder=enc, weights=path, method='LogLoRet-SelfAttn-AttnPool', no_match='dustbin', topk_in=100, reuse=('level_features',),
                                                         batch=4), 'cpu')
        ranker.amp = False
        from stage2_retrieval.Context import feature_key                           # noqa: PLC0415
        upstream = RetrievalOutput(candidates=cs, extras={'level_features': Extra(
            'level_features', feature_key(enc, 256, True), src)})
        out = ranker.run(upstream, None)
        scores = [c.score for c in out.candidates.candidates]
        assert scores == sorted(scores, reverse=True) and len(scores) == len(cands)
        direct = ranker.scores_of(WindowBatch.build(cs, src, 'cls'), 4)
        assert sorted(scores) == sorted(direct.tolist()), 'the list carries the scores of the windows'
    return 'GloRet: grid and list agree on every window (4 networks); LogLoRet: the list comes back ordered by its score'


def t_a_formal_training_deletes_the_earlier_wandb_runs_of_its_arm_and_never_a_running_one() -> str:
    from types import SimpleNamespace                                                 # noqa: PLC0415
    from training.GloRet import Runtime as GloRuntime                                 # noqa: PLC0415
    from training.RerankHead import Runtime as RerankRuntime                          # noqa: PLC0415
    for runtime in (GloRuntime, RerankRuntime):
        deleted: list = []
        asked: list = []

        class Run:
            def __init__(self, rid: str, state: str) -> None:
                self.id, self.state = rid, state

            def delete(self, delete_artifacts: bool = True) -> None:
                assert delete_artifacts is False, 'the artifacts stay'
                deleted.append(self.id)

        class Api:
            default_entity = 'me'

            def runs(self, path: str, filters: dict) -> list:
                asked.append((path, filters))
                return [Run('old1', 'finished'), Run('old2', 'crashed'), Run('live', 'running')]

        fake = SimpleNamespace(Api=Api)
        assert runtime.delete_previous(fake, 'proj', 'arm-x') == 2
        assert deleted == ['old1', 'old2'] and asked == [('me/proj', {'display_name': 'arm-x'})], (deleted, asked)

        class Broken:
            def Api(self) -> None:
                raise RuntimeError('no network')

        assert runtime.delete_previous(Broken(), 'proj', 'arm-x') == 0, 'a failure is printed, not raised'
    return 'the earlier runs of the name go, a running one stays, their artifacts stay, a failure does not stop the training'


def t_the_wandb_plan_keeps_the_newest_run_of_each_arm_and_every_running_one() -> str:
    import importlib.util                                                             # noqa: PLC0415
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'cli', 'wandb_prune.py')
    spec = importlib.util.spec_from_file_location('wandb_prune_under_test', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod                                                      # a dataclass looks its module up here
    spec.loader.exec_module(mod)                                                      # wandb itself is imported only in main
    R = mod.RunRow
    rows = [R('p', 'a1', 'armA', 'finished', '2026-10-01T10:00:00'), R('p', 'a2', 'armA', 'finished', '2026-10-03T10:00:00'),
            R('p', 'a3', 'armA', 'crashed', '2026-10-02T10:00:00'), R('p', 'b1', 'armB', 'failed', '2026-10-01T09:00:00'),
            R('q', 'a4', 'armA', 'finished', '2026-10-09T10:00:00'), R('p', 'x1', '', 'finished', '2026-10-01T10:00:00'),
            R('p', 'x2', '', 'finished', '2026-10-02T10:00:00'), R('p', 'a0', 'armA', 'running', '2026-09-01T10:00:00')]
    out = mod.plan(rows, 1)
    kept = {r.id: r.keep for r in out}
    assert kept == {'a1': False, 'a2': True, 'a3': False, 'b1': True, 'a4': True, 'x1': True, 'x2': True, 'a0': True}, kept
    # the same arm name in another project is another arm; a run with no name is an arm of its own; a running one stays
    assert [r.id for r in mod.plan(rows, 2) if not r.keep] == ['a1']
    # formal runs only: a smoke / test run goes, and the newest of the arm is of the rest; a dropped project goes whole
    assert mod.is_smoke({'slurm_job_name': 'SmokeRerankHead'}) and mod.is_smoke({'slurm_job_name': 'TestX'})
    assert mod.is_smoke({'max_visits': 3}) and mod.is_smoke({'dry_run': True})
    assert not mod.is_smoke({'slurm_job_name': 'RerankHead', 'max_visits': 0}) and not mod.is_smoke({})
    formal = [R('p', 's1', 'armA', 'finished', '2026-10-09T10:00:00', smoke=True),
              R('p', 'f1', 'armA', 'finished', '2026-10-05T10:00:00'), R('p', 'f2', 'armA', 'finished', '2026-10-04T10:00:00'),
              R('p', 's2', 'armB', 'finished', '2026-10-09T10:00:00', smoke=True),
              R('m', 'm1', 'old', 'finished', '2026-10-01T10:00:00'), R('m', 'm2', 'old', 'running', '2026-10-02T10:00:00')]
    out = mod.plan(formal, 1, drop=['m'], formal_only=True)
    assert {r.id: r.keep for r in out} == {'s1': False, 'f1': True, 'f2': False, 's2': False, 'm1': False, 'm2': True}
    assert {r.id: r.why for r in out}['s1'] == 'smoke / test' and {r.id: r.why for r in out}['m1'] == 'project dropped'
    # a resumed or continued run is the newest and the short one: --prefer longest keeps the one that got farthest
    resumed = [R('p', 'old', 'armA', 'finished', '2026-10-01T10:00:00', steps=900),
               R('p', 'tail', 'armA', 'finished', '2026-10-05T10:00:00', steps=40)]
    assert {r.id: r.keep for r in mod.plan(resumed, 1)} == {'old': False, 'tail': True}
    assert {r.id: r.keep for r in mod.plan(resumed, 1, prefer='longest')} == {'old': True, 'tail': False}
    # one arm under the naming generations of the prototypical runs, a smoke run by its name, and a run that did not finish
    assert mod.arm_of('360897_identity_identity_mean_cosine_tau-uni2-mean-cosine_tau-none', 'x', True) == \
        mod.arm_of('PrototypicalRoutingHead-uni2-mean-cosine_tau-none', 'y', True) == 'uni2-mean-cosine_tau-none'
    assert mod.arm_of('uni2-tilegt', 'x', False) == 'uni2-tilegt' and mod.arm_of('', 'abc', True) == 'abc'
    assert mod.is_smoke({}, 'PrototypicalRoutingHeadSmoke-uni2-mean-cosine_tau-none') and not mod.is_smoke({}, 'uni2-tilegt')
    gens = [R('proto', 'g1', '360897_identity_identity_mean_cosine_tau-uni2-mean-cosine_tau-none', 'finished', '2026-09-25T10:00:00'),
            R('proto', 'g2', 'PrototypicalRoutingHead-uni2-mean-cosine_tau-none', 'finished', '2026-10-11T10:00:00'),
            R('proto', 'g3', 'PrototypicalRoutingHead-ord_a-uni2-mean-cosine_tau-none', 'finished', '2026-10-12T10:00:00'),
            R('rr', 'r1', 'uni2-tilegt', 'failed', '2026-10-13T10:00:00'), R('rr', 'r2', 'gigapath-tilegt', 'finished', '2026-10-14T10:00:00'),
            R('rr', 'r3', 'gigapath-tilegt', 'crashed', '2026-10-15T10:00:00')]
    out = mod.plan(gens, 1, strip=['proto'], finished_only=True)
    assert {r.id: r.keep for r in out} == {'g1': False, 'g2': True, 'g3': True, 'r1': False, 'r2': True, 'r3': False}
    assert {r.id: r.why for r in out}['r1'] == 'not finished (failed)', 'a failed run is no result; the encoders stay apart'
    # only the latest round of one project: --since drops what was created before it there, and only there
    rounds = [R('proto', 'old', 'PrototypicalRoutingHead-ord_a-uni2-mean', 'finished', '2026-10-12T08:00:00'),
              R('proto', 'new', 'PrototypicalRoutingHead-baseline-gigapath', 'finished', '2026-10-20T19:00:00'),
              R('cls', 'c', 'ClassifyingRoutingHead-b2-uni2', 'finished', '2026-10-12T08:00:00')]
    out = mod.plan(rounds, 1, since='2026-10-20', since_projects=['proto'])
    assert {r.id: r.keep for r in out} == {'old': False, 'new': True, 'c': True}
    assert {r.id: r.why for r in out}['old'] == 'created before 2026-10-20'
    return ('newest (or longest) of each (project, arm) stays, a running run and an unnamed one always; smoke runs, '
            'unfinished runs and dropped projects go; the naming generations of one arm are one arm; --since cuts a round')




# ── training/GloRet: the labels, the samples, the losses and the steps ────────

GLO_T: float = 512.0           # a tile's side in level-0 px of the stand-in grids (256 px tile, ds 2)


def glo_result(cands: List[Candidate], seed: int = 0, rot: int = 0, dim: int = DIM) -> Any:
    """A `FovResult` on the planted level: the query (rotation 0) is the block of the main grid at (row 3, col 4), so
    the answers are the main window there and the offset window nearest it. `dim`: the width of a grid entry."""
    from BenchCommon import Answer, Answers                                           # noqa: PLC0415
    from training.GloRet.Supply import FovResult                                      # noqa: PLC0415
    src, ref, _ = planted(dim=dim)
    cs: CandidateSet = cset(cands, n_regions=1)
    x0, y0 = cs.origin_l0(Candidate(0, 'main', 3, 4, 0, 0.0))
    ox, oy = cs.origin_l0(Candidate(0, 'offset', 3, 3, 0, 0.0))
    w0, h0 = 5 * GLO_T, 4 * GLO_T
    centre: Tuple[float, float] = (x0 + w0 / 2.0, y0 + h0 / 2.0)
    main = Answer('main', 0, 3, 4, 0, (x0, y0, w0, h0), 0.0, 0.0, 0.0)
    offset = Answer('offset', 0, 3, 3, 0, (ox, oy, w0, h0), 362.0, -256.0, -256.0)
    queries: Dict[int, torch.Tensor] = {r: src.query_grid(r, 'cls').detach().clone() for r in (0, 90, 180, 270)}
    return FovResult(slide='s', dataset='d', level=1, index=seed, cs=cs, source=src, found=Answers(main, offset),
                     rot=rot, centre=centre, tile_l0=GLO_T, tol_px=GLO_T, queries=queries)


def glo_list(n_other: int = 29, seed: int = 20, with_answer: bool = True) -> List[Candidate]:
    rng: np.random.Generator = np.random.default_rng(seed)
    cands: List[Candidate] = [Candidate(0, 'main', 3, 4, 0, 1.0)] if with_answer else []
    while len(cands) < n_other + int(with_answer):
        rot: int = int(rng.choice([0, 90, 180, 270]))
        qr, qc = (4, 5) if rot in (0, 180) else (5, 4)
        c: Candidate = Candidate(0, 'main', int(rng.integers(0, 40 - qr + 1)), int(rng.integers(0, 40 - qc + 1)), rot, 0.5)
        if c.key() not in {x.key() for x in cands} and not (rot == 0 and abs(c.row - 3) < 2 and abs(c.col - 4) < 2):
            cands.append(c)
    return cands


def t_a_window_is_an_answer_ignored_or_hard_and_the_wrong_rotations_are_ignored_or_negative_as_set() -> str:
    from training.GloRet.Supply import Placed, key_code, mark                         # noqa: PLC0415
    t = 100.0
    main_key, offset_key = (0, 'main', 5, 5, 0), (0, 'offset', 5, 4, 0)
    answers = [(key_code(main_key), (500.0, 500.0)), (key_code(offset_key), (450.0, 500.0))]
    size = np.tile(np.array([[200.0, 200.0]]), (8, 1))
    # 0 the main answer; 1 next to it; 2 a window below it, on the edge of the zone; 3 its place at another rotation;
    # 4 a window of the zone at another rotation; 5 far; 6 the offset answer; 7 just outside the zone
    code = np.array([key_code(main_key), key_code((0, 'main', 5, 6, 0)), key_code((0, 'main', 6, 5, 0)),
                     key_code(main_key), key_code((0, 'main', 5, 6, 180)), key_code((0, 'main', 9, 9, 0)),
                     key_code(offset_key), key_code((0, 'main', 7, 5, 0))])
    rot = np.array([0, 0, 0, 90, 180, 0, 0, 0])
    origin = np.array([[500, 500], [550, 500], [500, 600], [500, 500], [550, 500], [900, 900], [450, 500], [500, 601]],
                      dtype=float)
    placed = Placed(rot=0, centre=(550.0, 450.0), tol_px=t, answers=answers, zone=(100.0, 100.0))
    m = mark(code, rot, origin, size, placed, t)
    assert m.pos.tolist() == [True, False, False, False, False, False, True, False], 'the two answers, at the right rotation'
    assert m.ignore.tolist() == [False, True, True, True, True, False, False, False], \
        'the zone around an answer (a box, edge included), the wrong rotations in it too (the answer\'s own place as well)'
    assert m.neg.tolist() == [False, False, False, False, False, True, False, True], 'every other window is hard'
    assert abs(m.dist[0] - np.hypot(50.0, 150.0) / t) < 1e-9 and m.right.tolist() == [True, True, True, False, False, True, True, True]
    # wrong rotations as negatives: only the right rotation's zone is ignored
    n = mark(code, rot, origin, size, Placed(rot=0, centre=placed.centre, tol_px=t, answers=answers, zone=(100.0, 100.0),
                                              wrong_rotation_ignored=False), t)
    assert n.ignore.tolist() == [False, True, True, False, False, False, False, False]
    assert n.neg.tolist() == [False, False, False, True, True, True, False, True]
    # a list that lacks the main answer: the windows that kept its place are the positives (the benches' listed hit)
    idx = [1, 2, 4, 5]
    m2 = mark(code[idx], rot[idx], origin[idx], size[:4], placed, t, listed=True)
    assert m2.pos.tolist() == [True, False, False, False], 'the stand-in is the positive'
    assert m2.ignore.tolist() == [False, True, True, False], 'the rest of the zone is ignored'
    m3 = mark(code[idx], rot[idx], origin[idx], size[:4], placed, t, listed=False)
    assert not m3.pos.any() and m3.ignore.tolist() == [True, True, True, False], 'in a universe the answer is there: no stand-in'
    return 'pos / ignore (a box, the wrong rotations as set) / hard; a listed stand-in is positive'


def t_the_zone_is_half_the_fovs_window_each_way_and_the_loss_is_the_one_hot_cross_entropy() -> str:
    from training.GloRet.Losses import fov_loss, listwise_loss                        # noqa: PLC0415
    from training.GloRet.Supply import fov_of                                         # noqa: PLC0415
    res = glo_result(glo_list())                      # the query is 4 rows x 5 columns of 512 px tiles
    fov = fov_of(res)
    assert fov.placed.zone == (0.5 * 5 * 512.0, 0.5 * 4 * 512.0), 'half the width left and right, half the height up and down'
    assert fov.placed.wrong_rotation_ignored
    assert fov_of(res, expand=1.0, wrong_rotation_ignored=False).placed.zone == (5 * 512.0, 4 * 512.0)
    assert not fov_of(res, wrong_rotation_ignored=False).placed.wrong_rotation_ignored
    torch.manual_seed(0)
    logits = torch.randn(9)
    pos = torch.zeros(9, dtype=torch.bool)
    pos[3] = True
    ignore = torch.zeros(9, dtype=torch.bool)
    ignore[[1, 5]] = True
    right = torch.ones(9, dtype=torch.bool)
    out = fov_loss(logits, pos, ignore, right, torch.rand(9), 1.0, 0.0)
    manual = -torch.log_softmax(logits[~ignore], dim=0)[int(pos[~ignore].nonzero()[0])]
    assert torch.allclose(out['loss'], manual, atol=1e-6) and float(out['geo']) == 0.0 and torch.allclose(out['loss'], out['list'])
    assert torch.allclose(listwise_loss(logits, pos, ~ignore), manual, atol=1e-6), 'one positive: the cross-entropy of its class'
    return 'zone = (w/2, h/2) tiles; with no geometry term the loss is the one-hot cross-entropy over the non-ignored windows'


def t_the_windows_of_a_level_are_numbered_and_their_features_are_the_grid_blocks() -> str:
    from training.GloRet.Supply import code_of, decode_code, level_of                 # noqa: PLC0415
    res = glo_result(glo_list())
    level = level_of(res, 'cls')
    assert sorted(level.grids) == [(0, 'main'), (0, 'offset')] and level.tile_l0 == GLO_T
    uni = level.universe((4, 5))
    assert uni.total == (40 - 4 + 1) * (40 - 5 + 1) + (39 - 4 + 1) * (39 - 5 + 1)
    ids = np.array([0, 5, uni.total - 1, 36 * 33 + 7])
    part, rows, cols = uni.locate(ids)
    for i, p, r, c in zip(ids, part, rows, cols):
        assert uni.id_of(uni.parts[p], int(r), int(c)) == i
    assert uni.id_of((0, 'main'), 40, 0) == -1 and uni.id_of((0, 'main'), 0, 36) == -1 and uni.id_of((1, 'main'), 0, 0) == -1
    feats = uni.features(ids)
    for k, (p, r, c) in enumerate(zip(part, rows, cols)):
        want = level.grids[uni.parts[p]][r:r + 4, c:c + 5].reshape(20, -1)
        assert torch.equal(feats[k], want)
    origin, code = uni.origins(ids)
    for k, (p, r, c) in enumerate(zip(part, rows, cols)):
        region, lattice = uni.parts[p]
        assert tuple(origin[k]) == res.cs.grids[region].tile_origin_l0(lattice, int(r), int(c))
        assert decode_code(int(code[k])) == (uni.parts[p], int(r), int(c)) and code[k] == code_of(region, lattice == 'offset', r, c)
    assert level.universe((5, 4)).total == (40 - 5 + 1) * (40 - 4 + 1) + (39 - 5 + 1) * (39 - 4 + 1)
    return 'ids <-> (grid, row, col), the features are the blocks of the grids, the origins the grids\' own'


def t_the_losses_count_the_positives_leave_the_ignored_out_and_give_the_near_windows_the_soft_targets() -> str:
    from training.GloRet.Losses import fov_loss, listwise_loss, near_targets            # noqa: PLC0415
    torch.manual_seed(0)
    logits = torch.randn(10)
    pos = torch.zeros(10, dtype=torch.bool)
    pos[[1, 4]] = True
    ignore = torch.zeros(10, dtype=torch.bool)
    ignore[[2, 7, 8]] = True
    keep = ~ignore
    want = -torch.log(torch.exp(logits[pos]).sum() / torch.exp(logits[keep]).sum())
    assert torch.allclose(listwise_loss(logits, pos, keep), want, atol=1e-6)
    moved = logits.clone()
    moved[ignore] += 50.0
    assert torch.allclose(listwise_loss(moved, pos, keep), want, atol=1e-5), 'an ignored window is nowhere in it'
    dist = torch.tensor([3.0, 0.2, 0.9, 2.0, 0.4, 5.0, 1.0, 0.5, 1.5, 2.5])
    right = torch.ones(10, dtype=torch.bool)
    right[[5, 8]] = False                                  # window 8 is ignored at a wrong rotation: no target
    off = fov_loss(logits.clone().requires_grad_(), pos, ignore, right, dist, 1.0, 0.0)
    assert float(off['geo']) == 0.0 and torch.allclose(off['loss'], want, atol=1e-6), 'w_geo 0: the plain list loss'
    # on: windows 2 and 7 (ignored, right rotation, not positive) are near, with the targets alpha * exp(-d^2 / 2 sigma^2)
    z = logits.clone().requires_grad_()
    on = fov_loss(z, pos, ignore, right, dist, 1.0, 1.0, alpha=0.3)
    near = torch.zeros(10, dtype=torch.bool)
    near[[2, 7]] = True
    t = 0.3 * torch.exp(-dist[near] ** 2 / 2.0)
    assert torch.allclose(near_targets(dist[near], 0.3, 1.0), t, atol=1e-7)
    inside = ~ignore | near
    log_all = torch.logsumexp(logits[inside], dim=0)
    list_on = log_all - torch.logsumexp(logits[pos], dim=0)
    want_on = (list_on - (t * (logits[near] - log_all)).sum()) / (1.0 + t.sum())
    assert torch.allclose(on['loss'], want_on, atol=1e-5), (float(on['loss']), float(want_on))
    on['loss'].backward()
    assert float(z.grad[near].abs().sum()) > 0.0, 'the near windows are in the softmax and pulled up'
    assert float(z.grad[8].abs()) == 0.0, 'a window ignored at a wrong rotation is still out of it'
    # a weight of 2 doubles the targets
    two = fov_loss(logits, pos, ignore, right, dist, 1.0, 2.0, alpha=0.3)
    t2 = 2.0 * t
    want_two = (list_on - (t2 * (logits[near] - log_all)).sum()) / (1.0 + t2.sum())
    assert torch.allclose(two['loss'], want_two, atol=1e-5)
    return 'list loss with the ignored out; w_geo above 0: the zone at the right rotation gets alpha*exp(-d^2/2 sigma^2) and is in the softmax'

def t_a_head_is_named_as_the_proposal_names_it_and_its_weights_are_where_an_arm_reads_them() -> str:
    from stage2_retrieval.ArmSpec import learned_weights                             # noqa: PLC0415
    from training.GloRet.cli.train import parse_head, parse_args, weights_file        # noqa: PLC0415
    a = parse_head('GloRet-SelfAttn-AttnPool')
    b = parse_head('LogLoRet-GNN-GraphPool')
    c = parse_head('LogLoRet-SelfAttn-AttnPool+sinkhorn')
    assert (a.arch, a.method, a.op, a.name) == ('gloret', 'GloRet-SelfAttn-AttnPool', '', 'GloRet-SelfAttn-AttnPool')
    assert (b.arch, b.op, b.name) == ('logloret', 'dustbin', 'LogLoRet-GNN-GraphPool+dustbin')
    assert (c.op, c.name) == ('sinkhorn', 'LogLoRet-SelfAttn-AttnPool+sinkhorn')
    for bad in ('GloRet-SelfAttn-Identity', 'LogLoRet-SelfAttn-Align', 'GloRet-CLSTrans-AttnPool', 'GloRet-GNN-GraphPool+hnsw',
                'LogLoRet-SelfAttn-AttnPool+nope', 'WinEmbed', 'GloRet-GNN-Foo'):
        try:
            parse_head(bad)
            raise AssertionError(f'{bad} was accepted')
        except ValueError:
            pass
    args = parse_args(['--encoder', 'gigapath', '--out', '/x', '--top-p', '1.0'])
    assert weights_file(args, a, 'best') == '/x/' + learned_weights('gigapath', 'cls_avg', a.method, '', 'pure-train')
    assert weights_file(args, c, 'last') == '/x/GloRet/weights/pure-train--gigapath--cls_avg--LogLoRet-SelfAttn-AttnPool--sinkhorn--last.pt'
    args.tag = 'smoke'
    assert weights_file(args, a, 'best').endswith('pure-train_smoke--gigapath--cls_avg--GloRet-SelfAttn-AttnPool--best.pt')
    return 'names, the refusals with a reason, and the weights file the arm\'s `learned_weights` names (a tag moves it aside)'


def t_the_ranks_validation_reads_are_the_benches_places() -> str:
    from training.GloRet.Steps import recall_at as glo_recall, upstream_rank          # noqa: PLC0415
    inf = float('inf')
    lone = glo_list()
    assert upstream_rank(glo_result(lone)) == 1.0
    # the offset-lattice window (3, 4) starts 362 px from the main answer's origin, less than the radius (512)
    standing = [Candidate(0, 'main', 20, 20, 0, 0.9), Candidate(0, 'offset', 3, 4, 0, 0.8)] + lone[1:]
    assert upstream_rank(glo_result(standing)) == 2.0, 'a window nearer than the radius stands for the answer (the listed hit)'
    assert upstream_rank(glo_result(glo_list(with_answer=False))) == inf
    wrong = [Candidate(0, 'main', 3, 4, 180, 0.9)] + glo_list(with_answer=False)
    assert upstream_rank(glo_result(wrong)) == inf, 'the right place at the wrong rotation is no hit'
    r = glo_recall([1.0, 3.0, 40.0, 400.0, inf])
    assert r['R@1'] == 0.2 and r['R@5'] == 0.4 and r['R@100'] == 0.6 and r['R@500'] == 0.8 and r['ceiling'] == 0.8 and r['n'] == 5
    return 'the answer, a stand-in one tile over, nothing, the wrong rotation; recall up to 500'


def t_attnpool_reads_a_tiles_tokens_flattened_as_it_reads_them_apart_and_is_trained() -> str:
    from GloRet import GloRetNet, LogLoRetNet                                         # noqa: PLC0415
    torch.manual_seed(5)
    p, f = 3, 16
    tokens = torch.randn(4, 20, p, f)
    flat = tokens.reshape(4, 20, p * f)
    for net in (GloRetNet(small_cfg(tile_pool='attnpool', in_dim=f)).eval(),
                LogLoRetNet(small_cfg(tile_pool='attnpool', in_dim=f)).eval()):
        with torch.no_grad():
            if isinstance(net, GloRetNet):
                assert torch.allclose(net.encode(tokens, (4, 5)), net.encode(flat, (4, 5)), atol=1e-6)
            else:
                a = net(tokens[:1], tokens, (4, 5))['score']
                b = net(flat[:1], flat, (4, 5))['score']
                assert torch.allclose(a, b, atol=1e-5)
    net = GloRetNet(small_cfg(tile_pool='attnpool', in_dim=f))
    net.encode(flat, (4, 5)).sum().backward()
    assert net.embed.pool[0].weight.grad is not None and float(net.embed.pool[0].weight.grad.abs().sum()) > 0, \
        'the pooling over a tile\'s patches gets a gradient'
    try:
        net.encode(torch.randn(2, 20, p * f + 1), (4, 5))
        raise AssertionError('a width that is not whole tokens was accepted')
    except ValueError:
        pass
    # the pooled kinds still refuse tokens
    try:
        GloRetNet(small_cfg()).encode(tokens, (4, 5))
        raise AssertionError('cls_avg read four dimensions')
    except ValueError:
        pass
    return 'tokens [B,N,P,F] and flattened [B,N,P*F] give the same vectors and scores; the pool trains; odd widths refused'


def t_the_token_encoder_drops_the_prefix_and_a_patch_source_offers_the_patch_grids() -> str:
    from types import SimpleNamespace                                                 # noqa: PLC0415
    from stage2_retrieval.Context import GridSource                                   # noqa: PLC0415
    from training.GloRet.Patches import POOLING, PatchSource, TokenEncoder            # noqa: PLC0415
    p, f, prefix = 4, 8, 2

    class FakeEncoder:
        model_spec = SimpleNamespace(num_prefix=prefix)

        def tokens(self, tiles):
            n = len(tiles)
            return torch.arange(n * (prefix + p) * f, dtype=torch.float32).reshape(n, prefix + p, f)

    enc = TokenEncoder(FakeEncoder())
    out = enc([0, 1, 2])
    assert out.dtype == torch.float16 and tuple(out.shape) == (3, p * f)
    want = FakeEncoder().tokens([0, 1, 2])[:, prefix:, :].reshape(3, -1).half()
    assert torch.equal(out, want), 'the CLS and registers are dropped, the patches kept in order'
    ps = PatchSource(lambda: FakeEncoder(), 'cpu')
    frame = cset([], n_regions=1)
    level, qf = object(), object()
    ps.search = SimpleNamespace(run=lambda routed, q, scorer: (scorer(level, qf, frame), level, qf), level_encodes={})
    got = ps.retrieve(SimpleNamespace(chosen_level=1), None, topk=1)
    assert len(got.candidates) == 1, 'a placeholder: it ranks nothing'
    src = got.find('level_features')[-1].value
    assert isinstance(src, GridSource) and src.pooling == POOLING == 'attnpool'
    ps.search = SimpleNamespace(run=lambda routed, q, scorer: (frame, None, None), level_encodes={})
    assert not len(ps.retrieve(SimpleNamespace(chosen_level=1), None).candidates), 'no level searched: nothing offered'
    return 'tokens [N,T,F] -> [N,P*F] fp16 without the prefix; the source offers a GridSource read as attnpool'


def t_a_frozen_attention_pool_turns_a_tiles_tokens_into_the_vector_the_network_would_have_pooled() -> str:
    from types import SimpleNamespace                                                 # noqa: PLC0415
    from GloRet import GloRetNet, LogLoRetNet                                         # noqa: PLC0415
    from stage2_retrieval.Context import GridSource                                   # noqa: PLC0415
    from training.GloRet.Patches import (FROZEN_POOLING, PatchSource, PooledTokenEncoder,   # noqa: PLC0415
                                         load_pool)
    torch.manual_seed(8)
    p, f, prefix = 5, 16, 2
    net = GloRetNet(small_cfg(tile_pool='attnpool', in_dim=f)).eval()

    class FakeEncoder:
        model_spec = SimpleNamespace(num_prefix=prefix)

        def tokens(self, tiles):
            g = torch.Generator().manual_seed(len(tiles))
            return torch.randn(len(tiles), prefix + p, f, generator=g)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'a.pt')
        save_rerank_checkpoint(path, model=net, arch='gloret', pooling='attnpool', encoder='gigapath', epoch=1, val={},
                               run_args={})
        pool = load_pool(path, 'cpu')
        assert all(not q.requires_grad for q in pool.parameters()), 'frozen'
        enc = PooledTokenEncoder(FakeEncoder(), pool)
        got = enc([0, 1, 2])
        tokens = FakeEncoder().tokens([0, 1, 2])[:, prefix:, :]
        with torch.no_grad():
            beta = torch.softmax(net.embed.pool(tokens).squeeze(-1), dim=-1)
            want = (beta.unsqueeze(-1) * tokens).sum(dim=1)
        assert got.dtype == torch.float16 and tuple(got.shape) == (3, f)
        assert torch.allclose(got.float(), want, atol=2e-2), 'the network\'s own pooling of the tokens'
        # a checkpoint of a network that does not pool tokens has no pool to freeze
        pooled_net = GloRetNet(small_cfg()).eval()
        save_rerank_checkpoint(os.path.join(tmp, 'b.pt'), model=pooled_net, arch='gloret', pooling='cls',
                               encoder='gigapath', epoch=1, val={}, run_args={})
        try:
            load_pool(os.path.join(tmp, 'b.pt'), 'cpu')
            raise AssertionError('a pool was taken from a network without one')
        except ValueError:
            pass
        # a source with the pool offers grids of one vector a tile, read as `attnpool_frozen`
        ps = PatchSource(lambda: FakeEncoder(), 'cpu', pool=pool)
        assert ps.pooling == FROZEN_POOLING == 'attnpool_frozen' and isinstance(ps.tokens, PooledTokenEncoder)
        frame = cset([], n_regions=1)
        ps.search = SimpleNamespace(run=lambda routed, q, scorer: (scorer(object(), object(), frame), object(), object()),
                                    level_encodes={})
        src = ps.retrieve(SimpleNamespace(chosen_level=1), None).find('level_features')[-1].value
        assert isinstance(src, GridSource) and src.pooling == 'attnpool_frozen'
    # the networks trained on such grids read [B, N, F] and have no pool of their own
    for cls in (GloRetNet, LogLoRetNet):
        frozen = cls(small_cfg(tile_pool='attnpool_frozen', in_dim=f))
        assert frozen.embed.pool is None
    out = GloRetNet(small_cfg(tile_pool='attnpool_frozen', in_dim=f)).encode(torch.randn(3, 20, f), (4, 5))
    assert tuple(out.shape) == (3, 24)
    return 'the frozen pool gives the vector TileEmbed pools; a network without one is refused; the source reads attnpool_frozen'


def lazy_world(dim: int = 48) -> Tuple[Any, Any, Any]:
    """The planted level as `(res, the grid-based Level, a TileAccess)`: the access reads a tile as its level-0 origin
    and encodes it to the grid's own vector, so what is gathered on demand can be set against the grid."""
    from training.GloRet.Lazy import TileAccess                                       # noqa: PLC0415
    from training.GloRet.Supply import level_of                                       # noqa: PLC0415
    res = glo_result(glo_list(), dim=dim)
    g = res.cs.grids[0]
    g.grid_rows, g.grid_cols, g.overlap_rows, g.overlap_cols = 40, 40, 39, 39
    full = level_of(res, 'cls')
    table: Dict[Tuple[int, int], torch.Tensor] = {}
    for part, grid in full.grids.items():
        for r in range(int(grid.shape[0])):
            for c in range(int(grid.shape[1])):
                table[(int(full.xs[part][c]), int(full.ys[part][r]))] = grid[r, c]

    def read(points: Any) -> Any:
        pts = torch.tensor(points)
        for s in range(0, len(points), 7):                       # batches of 7: the index puts each tile back
            yield pts[s:s + 7], torch.arange(s, min(s + 7, len(points)))

    def encode(batch: Any) -> torch.Tensor:
        return torch.stack([table[(int(x), int(y))] for x, y in batch.tolist()]).half()

    return res, full, TileAccess(read, encode, dim)


def t_tokens_on_demand_gather_the_windows_a_levels_grid_would() -> str:
    from training.GloRet.Lazy import TileTokens, lazy_level_of                         # noqa: PLC0415
    res, full, access = lazy_world()
    lazy = lazy_level_of(res)
    assert sorted(lazy.grids) == sorted(full.grids) and all(g.shape[2] == 0 for g in lazy.grids.values()), \
        'the geometry only'
    assert all(np.array_equal(lazy.xs[k], full.xs[k]) and np.array_equal(lazy.ys[k], full.ys[k]) for k in full.grids)
    tokens = TileTokens(lazy, access, 'cpu')
    rng = np.random.default_rng(0)
    uni = lazy.universe((4, 5))
    ids = rng.integers(0, uni.total, size=40)
    part, rows, cols = uni.locate(ids)
    parts = [uni.parts[p] for p in part]
    keys = [(parts[k], int(rows[k]) + i, int(cols[k]) + j) for k in range(len(ids)) for i in range(4) for j in range(5)]
    n = tokens.fill(keys)
    assert n == len(set(keys)) == len(tokens) and tokens.fill(keys) == 0, 'each tile once'
    got = tokens.gather(parts, rows, cols, (4, 5))
    want = full.window_features(parts, rows, cols, (4, 5))
    assert got.shape == want.shape == (40, 20, 48) and torch.allclose(got.float(), want, atol=2e-3)
    lazy.tokens = tokens
    assert torch.equal(lazy.window_features(parts, rows, cols, (4, 5)), got)
    try:
        TileTokens(lazy, access, 'cpu').gather(parts[:1], rows[:1], cols[:1], (4, 5))
        raise AssertionError('a tile that was never read was gathered')
    except KeyError:
        pass
    return 'fill reads each tile once; the windows gathered are the grid\'s (fp16); a tile not read is refused'


def t_a_source_that_encodes_on_demand_offers_the_query_tokens_and_how_to_read_a_tile() -> str:
    from types import SimpleNamespace                                                 # noqa: PLC0415
    from training.GloRet.Lazy import QuerySource, TileAccess                          # noqa: PLC0415
    from training.GloRet.Patches import PatchSource                                   # noqa: PLC0415
    p, f = 3, 8

    class FakeEncoder:
        model_spec = SimpleNamespace(num_prefix=1, feat_hw=(1, p), dim=f)

        def tokens(self, tiles):
            return torch.ones(len(tiles), 1 + p, f)

    seen: list = []

    class FakeReader:
        def read_points(self, points, spec, ds, level, batch):
            seen.append((list(points), spec, ds, level, batch))
            yield torch.zeros(len(points), 2, 2, 3, dtype=torch.uint8), torch.arange(len(points))

    class FakeLevels:
        def query_dims(self, query):
            return {0: (2, 2)}

        def frame(self, level):
            return cset([], n_regions=1)

        def route(self, routed, dims, prepare, search):
            assert routed == 1
            prepare(routed)
            return search(routed)

    ps = PatchSource(lambda: FakeEncoder(), 'cpu', lazy=True)
    ps.search.levels, ps.search.reader = FakeLevels(), FakeReader()
    got = ps.retrieve(SimpleNamespace(chosen_level=1), np.random.default_rng(0).integers(0, 255, (512, 512, 3), dtype=np.uint8))
    assert len(got.candidates) == 1
    source = got.find('level_features')[-1].value
    access = got.find('tiles')[-1].value
    assert isinstance(source, QuerySource) and isinstance(access, TileAccess) and access.width == p * f
    assert tuple(source.query_grid(0, 'attnpool').shape) == (2, 2, p * f), 'the query is encoded: its tokens'
    try:
        source.reference_grid(0, 'main', 'attnpool')
        raise AssertionError('a level grid was offered')
    except RuntimeError:
        pass
    (tiles, index), = list(access.read([(0, 0), (512, 0), (0, 512)]))
    assert len(index) == 3 and seen[-1][0] == [(0, 0), (512, 0), (0, 512)] and seen[-1][2] == 2.0 and seen[-1][4] == 256
    assert tuple(access.encode(tiles).shape) == (3, p * f)
    return 'the level is not encoded: the query\'s tokens and a way to read and encode tiles are what is offered'


def t_the_usage_meter_reads_the_counters_and_says_what_could_be_handed_back() -> str:
    from training.GloRet import Usage                                                 # noqa: PLC0415
    gpus = Usage.parse_gpus('0, 85, 40000, 143771\n1, 5, 2000, 143771\nbad line\n2, [N/A], 1, 2\n')
    assert [g[0] for g in gpus] == [0, 1] and abs(gpus[0][2] - 41.94304) < 1e-3 and abs(gpus[0][3] - 150.7) < 0.1
    stat = ('cpu  100 0 100 800 0 0 0 0\ncpu0 10 0 10 80 0 0 0 0\ncpu1 30 0 30 20 20 0 0 0\ncpu2 1 0 1 1 1 0 0 0\n')
    busy, total, waiting = Usage.parse_cpu_times(stat, [0, 1])
    assert total == 200.0 and busy == 80.0 and waiting == 20.0, (busy, total, waiting)   # cpu1: 60 busy, 20 waiting for a read, 20 idle
    used, node_total, shm = Usage.node_memory()
    assert node_total > 0 and 0 <= used <= node_total and shm >= 0
    meter = Usage.UsageMeter(interval=3600.0)
    meter.sample()
    summary = meter.summary()
    assert summary['samples'] == 1 and summary['cpus'] >= 1 and summary['mem_peak'] > 0
    assert Usage.UsageMeter(interval=3600.0).summary()['samples'] == 0, 'a new period starts empty'
    import time                                                                       # noqa: PLC0415
    meter.sample()
    meter.sample()
    assert meter.peek(0.0)['samples'] == 2 and meter.peek(time.time() + 10)['samples'] == 0, 'peek ends no period'
    reports: list = []
    running = Usage.UsageMeter(interval=0.05, report_every=0.2, on_report=reports.append).start()
    time.sleep(1.0)
    running.stop()
    assert reports and reports[0]['samples'] >= 1, 'a long run reports as it goes'
    from training.RerankHead import Usage as RerankUsage                                # noqa: PLC0415
    assert RerankUsage.UsageMeter(interval=3600.0).summary()['samples'] == 0, 'RerankHead has its own copy'
    idle = {'host': 'n', 'seconds': 100.0, 'samples': 5, 'cpus': 96, 'cpu_mean': 10.0, 'cpu_peak': 20.0,
            'mem_peak': 100.0, 'mem_limit': 1600.0, 'shm_peak': 1.0,
            'gpus': {0: {'util_mean': 12.0, 'util_max': 50.0, 'mem_peak': 20.0, 'mem_total': 140.0}}}
    hints = Usage.advice(idle)
    assert len(hints) == 4 and any('GPUs are busy 12%' in h for h in hints) and any('CPUs peak at 20%' in h for h in hints) \
        and any('memory peak 100 of 1600' in h for h in hints), hints
    full = dict(idle, cpu_peak=95.0, mem_peak=1500.0, gpus={0: {'util_mean': 90.0, 'util_max': 100.0, 'mem_peak': 100.0, 'mem_total': 140.0}})
    assert Usage.advice(full) == [], 'a busy node has nothing to hand back'
    slow = Usage.advice(dict(full, io_mean=40.0, io_peak=70.0))
    assert len(slow) == 1 and 'wait for a read or write 40%' in slow[0] and 'more of either would not help' in slow[0]
    assert 'waiting for a read or write mean 40%' in ' '.join(Usage.usage_lines(dict(full, io_mean=40.0, io_peak=70.0)))
    meter2 = Usage.UsageMeter(interval=3600.0)
    meter2.sample()
    assert 'io_mean' in meter2.summary(), 'the meter reports the share of time the CPUs waited for a read'
    assert any('GPU' in l for l in Usage.usage_lines(idle))
    return 'nvidia-smi text, /proc/stat busy share, memory; a meter sample and period; advice for idle and for busy nodes'


def t_the_gloret_trainer_reports_host_memory_and_a_process_that_fails_leaves_at_once() -> str:
    import subprocess                                                              # noqa: PLC0415
    from training.GloRet.cli import train as gl_train                              # noqa: PLC0415
    text: str = gl_train.host_memory()
    assert 'host RSS' in text and '/dev/shm' in text, text
    code: str = ('import sys, os\n'
                 f'sys.path[:0] = {[p for p in sys.path if p]!r}\n'
                 'from training.GloRet.cli import train as t\n'
                 'def boom():\n    raise ValueError("planted")\n'
                 't.main = boom\n'
                 'try:\n    t.run()\nexcept ValueError:\n    print("RAISED")\n    sys.exit(0)\n')
    for world, want_code, want_text in (('2', 1, 'planted'), ('1', 0, 'RAISED')):
        done = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                              env={**os.environ, 'WORLD_SIZE': world})
        assert done.returncode == want_code and want_text in done.stdout + done.stderr, (world, done.returncode, done.stderr[-300:])
    return 'host RSS and /dev/shm in the log line; WORLD_SIZE 2: traceback and exit 1; alone: raised'


def t_a_narrowing_gets_a_decoder_only_when_it_narrows_and_a_mode_is_set() -> str:
    from GloRet import Decoded, GloRetNet, narrows, take_aux                          # noqa: PLC0415
    assert narrows(16, 8, 'ae') and narrows(16, 8, 'vae')
    assert not narrows(8, 8, 'ae') and not narrows(8, 16, 'vae') and not narrows(16, 8, '')
    assert Decoded(8, 8, 'ae').dec is None and Decoded(16, 8, '').dec is None and Decoded(8, 16, 'vae').dec is None
    ae, vae = Decoded(16, 8, 'ae'), Decoded(16, 8, 'vae')
    assert ae.dec is not None and ae.logvar is None and vae.dec is not None and vae.logvar is not None
    # in_dim 16 -> d 32 widens, d 32 -> d_r 24 narrows: the decoder is on w_q / w_k only
    net = GloRetNet(small_cfg(recon='ae'))
    assert net.embed.proj.dec is None and net.w_q.dec is not None and net.w_k.dec is not None
    same = GloRetNet(small_cfg(recon='ae', d_r=32))
    assert same.w_q.dec is None and same.w_k.dec is None, 'd_r == d: nothing compressed, no decoder'
    none = GloRetNet(small_cfg())
    w = torch.randn(3, 20, 16)
    none.train()
    none.encode(w, (4, 5))
    assert take_aux(none) == (None, None), 'no mode: no loss'
    return 'a decoder only where the dimension drops and a mode is set; d_r == d or no mode: none, and no aux loss'


def t_the_decoders_loss_is_made_in_training_only_has_a_gradient_and_leaves_the_output_alone() -> str:
    from GloRet import Decoded, GloRetNet, take_aux                                   # noqa: PLC0415
    torch.manual_seed(0)
    x = torch.randn(5, 16)
    for mode in ('ae', 'vae'):
        d = Decoded(16, 8, mode)
        plain = Decoded(16, 8, '')
        plain.proj.load_state_dict(d.proj.state_dict())
        d.eval()
        with torch.no_grad():
            out = d(x)
        assert not d._recon and not d._kl, 'evaluation: no loss'
        d.train()
        with torch.no_grad():
            d(x)
        assert not d._recon, 'no gradient: no loss'
        z = d(x)
        assert torch.equal(z, plain(x)), 'the result is the projection whatever the mode'
        recon, kl = d.take()
        assert recon is not None and 0.0 <= float(recon) <= 2.0 and (kl is not None) == (mode == 'vae')
        assert d.take() == (None, None), 'taken once'
        d(x)
        recon, kl = d.take()
        assert recon is not None
        recon.backward(retain_graph=kl is not None)
        assert d.dec.weight.grad is not None and d.proj.weight.grad is not None
        assert float(d.dec.weight.grad.abs().sum()) > 0.0
        if mode == 'vae':
            assert float(kl) >= 0.0
            kl.backward()
            assert d.logvar.weight.grad is not None
    # the decoder learns to rebuild: the loss falls
    d = Decoded(16, 8, 'ae').train()
    opt = torch.optim.Adam(d.parameters(), lr=1e-2)
    first = None
    for _ in range(300):
        opt.zero_grad()
        d(x)
        r, _ = d.take()
        first = float(r) if first is None else first
        r.backward()
        opt.step()
    assert float(r) < first, (first, float(r))
    # the whole network sums its narrowings
    net = GloRetNet(small_cfg(recon='vae')).train()
    net.encode(torch.randn(3, 20, 16), (4, 5))
    recon, kl = take_aux(net)
    assert recon is not None and kl is not None
    return 'ae / vae: loss only when training with grad; output equals the plain projection; the decoder learns; take_aux sums'


def with_teacher(res: Any, level: Any, seed: int = 0) -> Any:
    """`res` with the first phase's score of every window set (random numbers), as `FovResult.scores` has them."""
    gen = torch.Generator().manual_seed(seed)
    scores: Dict[Tuple[int, int, str], torch.Tensor] = {}
    for rot in (0, 90, 180, 270):
        uni = level.universe((int(res.queries[rot].shape[0]), int(res.queries[rot].shape[1])))
        for p, part in enumerate(uni.parts):
            ho = int(uni.starts[p + 1] - uni.starts[p]) // uni.wide[p]
            scores[(rot, part[0], part[1])] = torch.rand(ho, uni.wide[p], generator=gen)
    res.scores = scores
    return res


def episode_world(answer_at: Optional[int] = 0, n_other: int = 29, teacher: bool = True) -> Tuple[Any, Any, Any]:
    """`(res, level, fov)` of the planted level with a list whose answer stands at place `answer_at` (None: the list
    holds none) and, with `teacher`, a first phase's score for every window."""
    from training.GloRet.Supply import fov_of, level_of                               # noqa: PLC0415
    cands = glo_list(n_other=n_other, with_answer=answer_at is not None)
    if answer_at:
        cands.insert(answer_at, cands.pop(0))
    res = glo_result(cands)
    level = level_of(res, 'cls')
    if teacher:
        with_teacher(res, level)
    return res, level, fov_of(res, level=level)


def t_top_p_is_a_share_or_a_count_and_the_pool_has_a_floor_and_three_regimes() -> str:
    from training.GloRet.Pool import TopP, parse_top_p                                # noqa: PLC0415
    share, count = parse_top_p(['0.05', '2000'])
    assert share == TopP(share=0.05) and count == TopP(count=2000)
    assert parse_top_p(['1.0'])[0] == TopP(share=1.0) and parse_top_p(['1'])[1] == TopP(count=1), 'a point makes a share'
    for bad in ([], ['0.1', '0.2'], ['3', '4']):
        try:
            parse_top_p(bad)
        except ValueError:
            continue
        raise AssertionError(f'--top-p {bad} was accepted')
    for wrong in (dict(share=0.0), dict(share=1.5), dict(count=0), dict()):
        try:
            TopP(**wrong)
        except ValueError:
            continue
        raise AssertionError(f'TopP{wrong} was accepted')
    s = TopP(share=0.03)
    assert s.size(1000, 256) == 256 and s.size(100000, 256) == 3000 and s.size(100, 256) == 100, 'a floor, never more than there is'
    assert TopP(count=2000).size(1500, 256) == 1500 and TopP(count=300).size(1500, 256) == 300, 'a count is what was asked'
    assert TopP(share=1.0).regime(1000, 1000) == 'all' and TopP(share=0.7).regime(1000, 700) == 'each step'
    assert TopP(share=0.5).regime(1000, 500) == 'episodes' and TopP(share=0.01).regime(1000, 256) == 'episodes'
    assert TopP(share=0.01).regime(100, 100) == 'all', 'a floor that reaches every item is all of them'
    assert TopP(count=10).regime(30, 10) == 'episodes' and TopP(count=20).regime(30, 20) == 'each step'
    return 'a point is a share, an integer a count; the floor, the cap; all / each step (above 0.5) / episodes (0.5 or less)'


def t_the_pool_holds_the_answers_the_negatives_that_beat_them_and_then_the_best_of_the_rest() -> str:
    from training.GloRet.Pool import Memory, make_pool                                # noqa: PLC0415
    n = 40
    rng = np.random.default_rng(0)
    scores = rng.normal(size=n)
    pos = np.zeros(n, dtype=bool)
    pos[[7]] = True
    ignore = np.zeros(n, dtype=bool)
    ignore[[8, 9]] = True
    scores[7], scores[8], scores[9] = 0.0, 5.0, 5.0       # an ignored window may score high: it is never chosen
    eligible = ~pos & ~ignore
    always = pos.copy()
    ahead = np.flatnonzero(eligible & (scores > 0.0))
    mem = Memory()
    pool, n_beat = make_pool(scores, pos, eligible, always, 6, mem)
    assert n_beat == len(ahead) and 7 in pool and not ({8, 9} & set(pool.tolist()))
    top_ahead = ahead[np.argsort(-scores[ahead])][:6]
    if len(ahead) >= 6:
        assert set(pool.tolist()) == set(top_ahead.tolist()) | {7}, 'the nearest negatives that beat the answer, all of K'
    assert set(mem.idx.tolist()) == set(ahead.tolist()), 'the memory is the negatives that beat the answer'
    assert np.allclose(np.sort(mem.margin), np.sort(scores[ahead] - 0.0))
    # a pool of 30 holds all that beat the answer, then the best-scoring of the others
    pool, _ = make_pool(scores, pos, eligible, always, 30, mem)
    rest = np.setdiff1d(np.flatnonzero(eligible), ahead)
    want = set(ahead.tolist()) | set(rest[np.argsort(-scores[rest])][:30 - len(ahead)].tolist()) | {7}
    assert set(pool.tolist()) == want and len(pool) == 31
    # the answer now beats some: they are struck off the memory
    scores2 = scores.copy()
    scores2[7] = float(np.sort(scores[ahead])[len(ahead) // 2]) + 1e-6
    pool, n_beat2 = make_pool(scores2, pos, eligible, always, 6, mem)
    assert n_beat2 < len(ahead) and set(mem.idx.tolist()) == set(np.flatnonzero(eligible & (scores2 > scores2[7])).tolist())
    # the near windows are in whatever
    zone = np.zeros(n, dtype=bool)
    zone[[8]] = True
    pool, _ = make_pool(scores, pos, eligible, pos | zone, 3, None)
    assert 8 in pool and 7 in pool
    return 'answers and near windows always; the negatives that beat the answer first, the best of the rest after; the memory follows the ranking'


def t_the_first_phases_scores_of_every_window_come_as_arrays_in_the_order_of_the_universe() -> str:
    from training.GloRet.Supply import fov_of, level_of, teacher_scores              # noqa: PLC0415
    res = glo_result(glo_list())
    level = level_of(res, 'cls')
    assert teacher_scores(level, res) is None and fov_of(res, level=level).teacher is None, 'no scores: none'
    with_teacher(res, level, seed=5)
    got = teacher_scores(level, res)
    assert got is not None and set(got) == {0, 90, 180, 270}
    for rot in (0, 90, 180, 270):
        uni = level.universe((int(res.queries[rot].shape[0]), int(res.queries[rot].shape[1])))
        assert len(got[rot]) == uni.total and got[rot].dtype == np.float32
        for i in (0, 17, uni.total // 2, uni.total - 1):
            part, row, col = uni.locate(np.array([i]))
            key = (rot, uni.parts[int(part[0])][0], uni.parts[int(part[0])][1])
            assert abs(float(res.scores[key][int(row[0]), int(col[0])]) - got[rot][i]) < 1e-7
    f = fov_of(res, level=level)
    assert f.teacher is not None and all(np.array_equal(f.teacher[r], got[r]) for r in got)
    return 'per rotation a float32 array over the windows of its shape, window i of the universe at place i'


def direct_pass(net: Any, level: Any, fov: Any, pool: Dict[int, np.ndarray], cache: Dict[Any, Any], step: Any,
                items: Any = None) -> Any:
    """The loss of one FoV over the windows of `pool`, every window with a graph at once (what a gradient cache must equal)."""
    from training.GloRet.Items import window_marks                                    # noqa: PLC0415
    from training.GloRet.Losses import fov_loss                                       # noqa: PLC0415
    from training.GloRet.Steps import net_input                                       # noqa: PLC0415
    logit, rows = [], {'pos': [], 'ignore': [], 'right': [], 'dist': []}
    for rot in (0, 90, 180, 270):
        ids = pool[rot]
        if not len(ids):
            continue
        shape = fov.shape(rot)
        uni = level.universe(shape)
        k = net.encode(net_input(uni.features(ids)), shape, 'window')
        q = net.encode(fov.queries[rot].reshape(1, -1, fov.queries[rot].shape[-1]).float(), shape, 'query')
        logit.append(net.score(q, k)[0])
        mk = items.dense_marks(rot) if items is not None else window_marks(level, fov, rot, cache)
        for name in rows:
            rows[name].append(getattr(mk, name)[ids])
    cat = {n: torch.as_tensor(np.concatenate(v)) for n, v in rows.items()}
    return fov_loss(torch.cat(logit), cat['pos'], cat['ignore'], cat['right'], cat['dist'], step.sigma, step.w_geo,
                    step.near_alpha, step.soft_target)['loss']


def t_a_gradient_cache_step_over_a_pool_gives_the_gradient_of_one_pass_over_it() -> str:
    from GloRet import GloRetNet                                                      # noqa: PLC0415
    from training.GloRet.Full import FullCfg, glo_step                                # noqa: PLC0415
    from training.GloRet.Items import level_items                                     # noqa: PLC0415
    from training.GloRet.Steps import StepCfg                                         # noqa: PLC0415
    from training.GloRet.Supply import fov_of, level_of                               # noqa: PLC0415
    cases = ((dict(context='SelfAttn', global_en='AttnPool'), 16), (dict(context='Identity', global_en='MaskedMean'), 16),
             (dict(context='Identity', global_en='Align'), 16),
             (dict(context='SelfAttn', global_en='AttnPool', tile_pool='attnpool', in_dim=16), 48))   # 3 patch tokens of 16
    for kw, dim in cases:
        for take_all in (True, False):
            res = glo_result(glo_list(with_answer=False), dim=dim)
            level, fov = level_of(res, 'cls'), fov_of(res)
            cache: dict = {}
            items = level_items(level, fov, cache)
            rng = np.random.default_rng(1)
            flat = np.arange(items.n) if take_all else np.unique(np.concatenate(
                [np.flatnonzero(items.pos), rng.choice(items.n, size=items.n // 3, replace=False)]))
            pool = items.pool_of(flat)
            torch.manual_seed(0)
            net = GloRetNet(small_cfg(dropout=0.0, **kw)).train()
            net.zero_grad()
            want = direct_pass(net, level, fov, pool, cache, StepCfg())
            want.backward()
            direct = {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}
            opt = torch.optim.SGD(net.parameters(), lr=0.0)
            out = glo_step(net, opt, level, [fov], [items], None if take_all else [pool], 0, FullCfg(chunk=97), StepCfg(), 1e9, None)
            tag = (kw['context'], kw['global_en'], kw.get('tile_pool', 'cls'), 'all' if take_all else 'a pool')
            assert out is not None and abs(out['loss'] - float(want)) < 1e-4 * max(1.0, abs(float(want))), (tag, out, float(want))
            assert {'t_keys', 't_loss', 't_back', 'windows'} <= set(out)
            for n, p in net.named_parameters():
                if n in direct:
                    assert p.grad is not None and torch.allclose(p.grad, direct[n], rtol=1e-3, atol=1e-3 * float(direct[n].abs().max()) + 1e-6), (
                        tag, n, float((p.grad - direct[n]).abs().max()), float(direct[n].abs().max()))
                else:
                    assert p.grad is None or float(p.grad.abs().sum()) == 0.0, (tag, n, 'a gradient the direct pass has none of')
    return 'SelfAttn-AttnPool, Identity-MaskedMean, Identity-Align, attnpool tokens: every window, and a pool of a third: the loss and every gradient equal the direct pass'


def _split_worker(rank: int, world: int, size: int, gradcache: bool, store: str, out: str) -> None:
    """One process of a group of `size`: the step over a share of the windows of a pool; its gradients go to a file."""
    import torch.distributed as dist                                                  # noqa: PLC0415
    from GloRet import GloRetNet                                                      # noqa: PLC0415
    from training.GloRet.Full import FullCfg, glo_step, make_split                    # noqa: PLC0415
    from training.GloRet.Items import level_items                                     # noqa: PLC0415
    from training.GloRet.Steps import StepCfg                                         # noqa: PLC0415
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{store}', rank=rank, world_size=world)
    _, level, fov = episode_world(answer_at=None)
    items = level_items(level, fov, {})
    pool = items.pool_of(np.unique(np.concatenate([np.flatnonzero(items.pos), np.arange(0, items.n, 3)])))
    torch.manual_seed(0)
    net = GloRetNet(small_cfg(context='SelfAttn', global_en='AttnPool', dropout=0.0)).train()
    split = make_split(world, rank, size)
    opt = torch.optim.SGD(net.parameters(), lr=0.0)
    res = glo_step(net, opt, level, [fov], [items], [pool], 0, FullCfg(chunk=50, gradcache=gradcache), StepCfg(), 1e9, None, split)
    assert res is not None
    torch.save({'loss': res['loss'], 'grads': {n: p.grad for n, p in net.named_parameters() if p.grad is not None}},
               f'{out}.{rank}')
    dist.destroy_process_group()


def t_a_window_split_over_processes_gives_the_same_gradient_with_a_cache_or_without() -> str:
    from GloRet import GloRetNet                                                      # noqa: PLC0415
    from training.GloRet.Items import level_items                                     # noqa: PLC0415
    from training.GloRet.Steps import StepCfg                                         # noqa: PLC0415
    _, level, fov = episode_world(answer_at=None)
    cache: dict = {}
    items = level_items(level, fov, cache)
    pool = items.pool_of(np.unique(np.concatenate([np.flatnonzero(items.pos), np.arange(0, items.n, 3)])))
    torch.manual_seed(0)
    net = GloRetNet(small_cfg(context='SelfAttn', global_en='AttnPool', dropout=0.0)).train()
    want = direct_pass(net, level, fov, pool, cache, StepCfg())
    want.backward()
    direct = {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}
    with tempfile.TemporaryDirectory() as tmp:
        # 4 processes in 2 groups of 2, each share a gradient cache; 5 in one group of 5 (the windows do not divide),
        # each share encoded with its graph at once
        for world, size, cache_on in ((4, 2, True), (5, 5, False)):
            store, out = os.path.join(tmp, f'store{world}'), os.path.join(tmp, f'out{world}')
            torch.multiprocessing.spawn(_split_worker, args=(world, size, cache_on, store, out), nprocs=world, join=True)
            for rank in range(world):
                got = torch.load(f'{out}.{rank}')
                assert abs(got['loss'] - float(want)) < 1e-4 * max(1.0, abs(float(want))), (world, rank, got['loss'], float(want))
                for n, g in direct.items():
                    assert torch.allclose(got['grads'][n], g, rtol=1e-3, atol=1e-3 * float(g.abs().max()) + 1e-6), (world, size, cache_on, rank, n)
    return 'groups of 2 (cache) and of 5 (no cache, uneven shares): every process has the loss and gradient of the one pass over the pool'


def t_the_network_ranks_every_window_as_it_scores_them_one_by_one() -> str:
    from GloRet import GloRetNet                                                      # noqa: PLC0415
    from training.GloRet.Full import FullCfg, rank_windows                            # noqa: PLC0415
    from training.GloRet.Items import level_items                                     # noqa: PLC0415
    from training.GloRet.Steps import net_input                                       # noqa: PLC0415
    torch.manual_seed(1)
    _, level, fov = episode_world(answer_at=None)
    items = level_items(level, fov, {})
    net = GloRetNet(small_cfg(context='SelfAttn', global_en='AttnPool')).train()
    got = rank_windows(net, level, [fov, fov], [items, None], 0, FullCfg(chunk=97))
    assert got[1] is None and got[0] is not None and len(got[0]) == items.n and net.training, 'the mode is put back'
    net.eval()
    for rot in (0, 90, 180, 270):
        shape = fov.shape(rot)
        uni = level.universe(shape)
        q = net.encode(fov.queries[rot].reshape(1, -1, fov.queries[rot].shape[-1]).float(), shape, 'query')
        for i in (0, 5, uni.total - 1):
            k = net.encode(net_input(uni.features(np.array([i]))), shape, 'window')
            want = float(net.score(q, k)[0, 0])
            assert abs(want - float(got[0][int(np.flatnonzero(items.rot == rot)[0]) + i])) < 1e-3, (rot, i)
    return 'the score of window i at rotation r is the net\'s one-by-one score, at place offset(r) + i'


def t_the_rerankers_gradient_cache_gives_the_gradient_of_one_pass_over_its_pool() -> str:
    from GloRet import LogLoRetNet, correspondence_loss, cycle_loss                    # noqa: PLC0415
    from training.GloRet.Losses import fov_loss                                       # noqa: PLC0415
    from training.GloRet.Items import list_items                                      # noqa: PLC0415
    from training.GloRet.Rerank import RerankCfg, log_step, rerank_scores, score_pairs   # noqa: PLC0415
    from training.GloRet.Steps import StepCfg, net_input                              # noqa: PLC0415
    torch.manual_seed(2)
    _, level, fov = episode_world(answer_at=3)
    items = list_items(level, fov, {})
    assert items is not None and items.n == len(fov.listing)
    pool = np.unique(np.concatenate([np.flatnonzero(items.pos), np.arange(0, items.n, 2)]))
    step = StepCfg(w_corr=0.1, w_cycle=0.1)
    net = LogLoRetNet(small_cfg(dropout=0.0, context='SelfAttn', global_en='AttnPool')).train()
    # the direct pass: every pair of the pool at once
    net.zero_grad()
    scores = torch.zeros(len(pool))
    a_fw, a_wf = [], []
    for rot in (0, 90, 180, 270):
        where = np.flatnonzero(items.cands.rot[pool] == rot)
        if not len(where):
            continue
        sub = pool[where]
        shape = fov.shape(rot)
        feats = level.window_features([items.cands.part[j] for j in sub], items.cands.row[sub], items.cands.col[sub], shape)
        out = net(fov.queries[rot].reshape(1, shape[0] * shape[1], -1).float(), net_input(feats), shape, None, None, maps=True)
        scores = scores.index_put((torch.as_tensor(where),), out['score'].float())
        hit = torch.as_tensor(items.pos[sub])
        if bool(hit.any()):
            a_fw.append(out['A_fw'][hit].float())
            a_wf.append(out['A_wf'][hit].float())
    m = items.take(pool)
    want = fov_loss(scores, torch.as_tensor(m.pos), torch.as_tensor(m.ignore), torch.as_tensor(m.right),
                    torch.as_tensor(m.dist), 1.0, 0.0)['loss']
    want = want + 0.1 * correspondence_loss(torch.cat(a_fw), torch.cat(a_wf)) + 0.1 * cycle_loss(torch.cat(a_fw), torch.cat(a_wf))
    want.backward()
    direct = {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}
    opt = torch.optim.SGD(net.parameters(), lr=0.0)
    out = log_step(net, opt, level, [fov], [items], [pool], 0, RerankCfg(chunk=5), step, 1e9, None)
    assert out is not None and {'corr', 'cycle', 'candidates'} <= set(out)
    for n, p in net.named_parameters():
        if n in direct:
            assert p.grad is not None and torch.allclose(p.grad, direct[n], rtol=1e-3, atol=1e-3 * float(direct[n].abs().max()) + 1e-6), n
    # the scores of the whole list, in chunks, are the one-by-one scores
    net.eval()
    got = rerank_scores(net, level, [fov], [items], 0, RerankCfg(chunk=7))[0]
    one = [float(score_pairs(net, level, fov, items.cands, np.array([i]), 0, 'cpu')[0][0]) for i in (0, 4, items.n - 1)]
    assert np.allclose(got[[0, 4, items.n - 1]], one, atol=1e-4)
    # no cache: the same step, the activations of the pool kept
    net.train()
    other = LogLoRetNet(small_cfg(dropout=0.0, context='SelfAttn', global_en='AttnPool'))
    other.load_state_dict(net.state_dict())
    opt2 = torch.optim.SGD(other.parameters(), lr=0.0)
    out2 = log_step(other, opt2, level, [fov], [items], [pool], 0, RerankCfg(chunk=5, gradcache=False), step, 1e9, None)
    assert out2 is not None and abs(out2['loss'] - out['loss']) < 1e-4
    for n, p in other.named_parameters():
        if n in direct:
            assert torch.allclose(p.grad, direct[n], rtol=1e-3, atol=1e-3 * float(direct[n].abs().max()) + 1e-6), ('no cache', n)
    return 'the cache and the no-cache step both give the loss (list, correspondence, cycle) and gradients of one pass over the pool'


def t_a_visit_takes_its_steps_in_each_regime_and_a_list_without_an_answer_gets_the_answer_added() -> str:
    from GloRet import GloRetNet, LogLoRetNet                                         # noqa: PLC0415
    from training.GloRet.Episodes import PoolCfg, run_visit                          # noqa: PLC0415
    from training.GloRet.Full import FullCfg                                          # noqa: PLC0415
    from training.GloRet.Pool import Memory, TopP                                     # noqa: PLC0415
    from training.GloRet.Rerank import RerankCfg                                      # noqa: PLC0415
    from training.GloRet.Steps import StepCfg                                         # noqa: PLC0415
    torch.manual_seed(0)
    res, level, fov = episode_world(answer_at=None)
    rng = np.random.default_rng(0)
    total = None
    for share, epoch, want in ((1.0, 0, 'all'), (0.7, 0, 'each step'), (0.2, 0, 'teacher'), (0.2, 5, 'student')):
        net = GloRetNet(small_cfg(context='Identity', global_en='MaskedMean'))
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
        pc = PoolCfg(top=TopP(share=share), min_candidates=0, episodes=2, warmup_epochs=2, steps=4, fovs_per_step=1)
        mem: Dict[int, Memory] = {}
        out = run_visit('gloret', 'level', net, opt, level, [fov], epoch, pc, FullCfg(chunk=64), RerankCfg(), StepCfg(), rng, {}, 1.0,
                        None, None, mem)
        assert out is not None and all(np.isfinite(v) for v in out.values()), (share, epoch, out)
        if total is None:
            total = out['windows']
        if want in ('teacher', 'student'):
            assert out['windows'] < total and 'beat' in out and len(mem) == 1, (want, out)
            if want == 'student':       # the network scored the items once per episode (2): the pool seconds are split into that and the rest
                assert 0.0 < out['t_pool_first'] and out['t_pool_score'] <= out['t_pool'], out
                assert abs(out['t_pool_score'] - out['t_pool_first'] - out['t_pool_last']) < 1e-6, out
            else:
                assert 't_pool_score' not in out, 'the teacher\'s scores are not scored by the network'
        elif want == 'each step':
            assert out['windows'] < total
    assert total is not None and total == sum(level.universe(fov.shape(r)).total for r in (0, 90)), 'all: every window of both shapes'
    # a count is a count of items, for a GloRet head too
    out = run_visit('gloret', 'level', net, opt, level, [fov], 0, PoolCfg(top=TopP(count=10), min_candidates=0, steps=2,
                    fovs_per_step=1), FullCfg(), RerankCfg(), StepCfg(), rng, {}, 1.0, None, None, {})
    assert out is not None and out['windows'] < total, 'a count of items'
    # a reranker: a count of the list
    _, level2, fov2 = episode_world(answer_at=3, teacher=False)
    for count, epoch in ((30, 0), (20, 0), (10, 0), (10, 5)):
        net2 = LogLoRetNet(small_cfg(context='Identity', global_en='MaskedMean'))
        opt2 = torch.optim.AdamW(net2.parameters(), lr=1e-3)
        pc = PoolCfg(top=TopP(count=count), min_candidates=0, episodes=2, warmup_epochs=2, steps=3, fovs_per_step=1)
        out = run_visit('logloret', 'list', net2, opt2, level2, [fov2], epoch, pc, FullCfg(), RerankCfg(chunk=8), StepCfg(), rng,
                        {}, 1.0, None, None, {})
        assert out is not None and all(np.isfinite(v) for v in out.values()), (count, epoch, out)
        assert count <= out['candidates'] <= count + 6, (count, out)
        assert out['injected'] == 0.0, 'the list holds the answer'
    # a list that holds no answer: the answer is added to the items, the FoV is trained on
    from training.GloRet.Supply import fov_of, level_of                               # noqa: PLC0415
    res3 = glo_result(glo_list(with_answer=False))
    level3, fov3 = level_of(res3, 'cls'), fov_of(res3)
    out = run_visit('logloret', 'list', net2, opt2, level3, [fov3], 0, pc, FullCfg(), RerankCfg(), StepCfg(), rng, {}, 1.0, None,
                    None, {})
    assert out is not None and out['injected'] == 1.0 and np.isfinite(out['loss']), out
    # the same FoV, a GloRet head over the list, and a LogLoRet head over the level (from the teacher)
    g = GloRetNet(small_cfg(context='Identity', global_en='MaskedMean'))
    out = run_visit('gloret', 'list', g, torch.optim.AdamW(g.parameters(), lr=1e-3), level3, [fov3], 0,
                    PoolCfg(top=TopP(count=10), min_candidates=0, steps=2, fovs_per_step=1), FullCfg(chunk=64), RerankCfg(),
                    StepCfg(), rng, {}, 1.0, None, None, {})
    assert out is not None and out['injected'] == 1.0 and out['windows'] <= 10 + 4, out
    _, level4, fov4 = episode_world(answer_at=None)
    out = run_visit('logloret', 'level', net2, opt2, level4, [fov4], 0,
                    PoolCfg(top=TopP(share=0.2), min_candidates=0, episodes=2, warmup_epochs=2, steps=2, fovs_per_step=1),
                    FullCfg(), RerankCfg(chunk=16), StepCfg(), rng, {}, 1.0, None, None, {})
    assert out is not None and 'beat' in out and np.isfinite(out['loss']), out
    # a share above half is not ranked by the pair scorer at each step (nor a count that is more than half of a small level):
    # its pool is the teacher's, in episodes
    out = run_visit('logloret', 'level', net2, opt2, level4, [fov4], 9, PoolCfg(top=TopP(share=0.7), min_candidates=0, episodes=2,
                    warmup_epochs=0, steps=2, fovs_per_step=1), FullCfg(), RerankCfg(chunk=16), StepCfg(), rng, {}, 1.0, None, None, {})
    assert out is not None and 'beat' in out and np.isfinite(out['loss']), out
    return 'all / each step / episodes for a retriever; a count for a reranker; a list without the answer gets it added (a GloRet head over the list too); a pair scorer over a level takes pools once per episode (the teacher\'s in the warmup epochs, its own after)'


def t_a_lazy_level_reads_the_tiles_of_a_window_when_asked_and_a_visit_on_it_trains_like_the_full_one() -> str:
    import copy                                                                       # noqa: PLC0415
    from GloRet import GloRetNet, LogLoRetNet                                         # noqa: PLC0415
    from training.GloRet import Episodes                                              # noqa: PLC0415
    from training.GloRet.Episodes import PoolCfg, run_visit                           # noqa: PLC0415
    from training.GloRet.Full import FullCfg                                          # noqa: PLC0415
    from training.GloRet.Lazy import gloret_rerank_rank_lazy, lazy_level_of, logloret_rank_lazy   # noqa: PLC0415
    from training.GloRet.Pool import TopP                                             # noqa: PLC0415
    from training.GloRet.Rerank import RerankCfg                                      # noqa: PLC0415
    from training.GloRet.Steps import StepCfg, gloret_rerank_rank, logloret_rank      # noqa: PLC0415
    from training.GloRet.Supply import fov_of                                         # noqa: PLC0415
    torch.manual_seed(4)
    res, full, access = lazy_world()
    lazy = lazy_level_of(res, 'cpu')
    lazy.access = access
    uni = lazy.universe((4, 5))
    part, rows, cols = uni.locate(np.array([0, 7, 100, uni.total - 1]))
    parts = [uni.parts[p] for p in part]
    want = full.window_features(parts, rows, cols, (4, 5))
    # asked for: read, and kept; or read for the call and dropped
    lazy.streaming = True
    assert torch.allclose(lazy.window_features(parts, rows, cols, (4, 5)).float(), want, atol=2e-3) and lazy.tokens is None
    lazy.streaming = False
    assert torch.allclose(lazy.window_features(parts, rows, cols, (4, 5)).float(), want, atol=2e-3)
    assert lazy.tokens is not None and 20 <= len(lazy.tokens) <= 80
    lazy.release()
    assert lazy.tokens is None and lazy.lazy and not full.lazy
    # a visit on it: the same loss as on the full level, in the teacher's epochs and after (the network ranks the list)
    for arch, source, net_a in (('logloret', 'list', LogLoRetNet(small_cfg(in_dim=48, dropout=0.0))),
                                ('gloret', 'list', GloRetNet(small_cfg(in_dim=48, dropout=0.0)))):
        for epoch in (0, 9):
            net_b = copy.deepcopy(net_a)
            opt_a, opt_b = torch.optim.AdamW(net_a.parameters(), lr=0.0), torch.optim.AdamW(net_b.parameters(), lr=0.0)
            fov_a, fov_b = fov_of(res), fov_of(res)
            lz = lazy_level_of(res, 'cpu')
            lz.access = access
            pc = PoolCfg(top=TopP(count=12), min_candidates=0, episodes=1, warmup_epochs=5, steps=1, fovs_per_step=1)
            ta = run_visit(arch, source, net_a, opt_a, full, [fov_a], epoch, pc, FullCfg(chunk=64), RerankCfg(chunk=5), StepCfg(),
                           np.random.default_rng(3), {}, 1.0, None, None, {})
            tb = run_visit(arch, source, net_b, opt_b, lz, [fov_b], epoch, pc, FullCfg(chunk=64), RerankCfg(chunk=5), StepCfg(),
                           np.random.default_rng(3), {}, 1.0, None, None, {})
            assert ta is not None and tb is not None and abs(ta['list'] - tb['list']) < 2e-2, (arch, epoch, ta, tb)
            assert lz.streaming is False and lz.tokens is not None, 'the pool\'s tokens are kept for the steps'
            assert 'injected' in tb and tb['injected'] == 0.0
    # the ranks validation reads, from the tiles of only the first windows
    net = LogLoRetNet(small_cfg(in_dim=48, dropout=0.0)).eval()
    assert logloret_rank_lazy(net, res, access, 50) == logloret_rank(net, res, 'cls', 50)
    g = GloRetNet(small_cfg(in_dim=48, dropout=0.0)).eval()
    assert gloret_rerank_rank_lazy(g, res, access, 50) == gloret_rerank_rank(g, res, 'cls', 50)
    # a list with no positive is not read for an answer's tokens: on a lazy level it is left out
    res2 = glo_result(glo_list(with_answer=False), dim=48)
    res2.cs.grids[0].grid_rows = 40
    out = run_visit('logloret', 'list', net.train(), torch.optim.AdamW(net.parameters(), lr=0.0), lz, [fov_of(res2)], 0,
                    PoolCfg(top=TopP(count=12), min_candidates=0, steps=1, fovs_per_step=1), FullCfg(), RerankCfg(), StepCfg(),
                    np.random.default_rng(0), {}, 1.0, None, None, {})
    assert out is None
    # every item at once cannot be held
    saved = Episodes.LAZY_ALL_LIMIT
    Episodes.LAZY_ALL_LIMIT = 5
    try:
        run_visit('logloret', 'list', net, torch.optim.AdamW(net.parameters(), lr=0.0), lz, [fov_of(res)], 0,
                  PoolCfg(top=TopP(count=10000), min_candidates=0, steps=1, fovs_per_step=1), FullCfg(), RerankCfg(), StepCfg(),
                  np.random.default_rng(0), {}, 1.0, None, None, {})
    except ValueError:
        pass
    else:
        raise AssertionError('a lazy level was asked for the tokens of every item')
    finally:
        Episodes.LAZY_ALL_LIMIT = saved
    return 'tiles read when a window is asked for (kept, or dropped when streaming); a visit on a lazy level has the loss of the full one in the teacher\'s epochs and after; ranks agree; no answer: left out; all items at once: refused'


def t_a_gloret_step_learns_the_planted_window_and_ranks_it_first() -> str:
    from GloRet import GloRetNet                                                      # noqa: PLC0415
    from stage2_retrieval.RetrieverGloRet import WindowEncodings                      # noqa: PLC0415
    from training.GloRet.Full import FullCfg, glo_step                                # noqa: PLC0415
    from training.GloRet.Items import level_items                                     # noqa: PLC0415
    from training.GloRet.Steps import StepCfg, gloret_rank                            # noqa: PLC0415
    torch.manual_seed(0)
    res, level, fov = episode_world(answer_at=None)
    items = level_items(level, fov, {})
    for context, global_en in (('SelfAttn', 'AttnPool'), ('Identity', 'MaskedMean'), ('Identity', 'Align')):
        net = GloRetNet(small_cfg(context=context, global_en=global_en))
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
        first = glo_step(net, opt, level, [fov], [items], None, 0, FullCfg(chunk=64), StepCfg(), 1.0, None)
        assert first is not None
        for n in range(1, 60):
            last = glo_step(net, opt, level, [fov], [items], None, n, FullCfg(chunk=64), StepCfg(), 1.0, None)
        assert last is not None
        if global_en != 'Align':       # Align compares tile to tile: the planted window is a match before any training
            assert last['list'] < 0.5 * first['list'], (context, global_en, first, last)
        assert all(np.isfinite(v) for v in last.values()), last
        enc = WindowEncodings(net.eval(), 'cpu', 8, False)
        assert gloret_rank(enc, res, 100) <= 5.0, (context, global_en, 'the planted window is not near the top')
    return 'three networks: the loss over every window halves (Align starts at a match); the planted window is in the first 5 of the net\'s own list'


def t_a_logloret_step_learns_the_planted_window_and_reorders_the_list() -> str:
    from GloRet import LogLoRetNet                                                    # noqa: PLC0415
    from training.GloRet.Items import list_items                                      # noqa: PLC0415
    from training.GloRet.Rerank import RerankCfg, log_step                            # noqa: PLC0415
    from training.GloRet.Steps import StepCfg, logloret_rank, upstream_rank           # noqa: PLC0415
    torch.manual_seed(1)
    res, level, fov = episode_world(answer_at=12, teacher=False)
    items = list_items(level, fov, {})
    assert items is not None
    pool = np.arange(items.n)
    net = LogLoRetNet(small_cfg(context='SelfAttn', global_en='AttnPool'))
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
    first = log_step(net, opt, level, [fov], [items], [pool], 0, RerankCfg(chunk=16), StepCfg(), 1.0, None)
    assert first is not None
    for n in range(1, 40):
        last = log_step(net, opt, level, [fov], [items], [pool], n, RerankCfg(chunk=16), StepCfg(), 1.0, None)
    assert last is not None and last['list'] < 0.5 * first['list'] and all(np.isfinite(v) for v in last.values()), (first, last)
    assert upstream_rank(res) == 13.0 and logloret_rank(net.eval(), res, 'cls', 50) <= 2.0, 'the answer was 13th; it is now first or second'
    return 'the loss over the pool halves; the answer, 13th in the list, is now first or second'


def t_a_gloret_and_a_logloret_step_run_on_a_levels_flattened_patch_tokens() -> str:
    from GloRet import GloRetNet, LogLoRetNet                                         # noqa: PLC0415
    from stage2_retrieval.RetrieverGloRet import WindowEncodings                      # noqa: PLC0415
    from training.GloRet.Full import FullCfg, glo_step                                # noqa: PLC0415
    from training.GloRet.Items import level_items, list_items                         # noqa: PLC0415
    from training.GloRet.Rerank import RerankCfg, log_step                            # noqa: PLC0415
    from training.GloRet.Steps import StepCfg, gloret_rank, logloret_rank              # noqa: PLC0415
    from training.GloRet.Supply import fov_of, level_of                               # noqa: PLC0415
    torch.manual_seed(3)
    p, f = 3, 16
    res = glo_result(glo_list(), dim=p * f)
    level = level_of(res, 'cls')
    fov = fov_of(res)
    assert level.universe((4, 5)).features(np.array([0, 7])).shape == (2, 20, p * f)
    g = GloRetNet(small_cfg(tile_pool='attnpool', in_dim=f))
    opt = torch.optim.AdamW(g.parameters(), lr=1e-3)
    terms = glo_step(g, opt, level, [fov], [level_items(level, fov, {})], None, 0, FullCfg(chunk=16), StepCfg(), 1.0, None)
    assert terms is not None and all(np.isfinite(v) for v in terms.values()), terms
    assert gloret_rank(WindowEncodings(g.eval(), 'cpu', 1, False), res, 50) >= 1.0
    l = LogLoRetNet(small_cfg(tile_pool='attnpool', in_dim=f))
    opt = torch.optim.AdamW(l.parameters(), lr=1e-3)
    items = list_items(level, fov, {})
    assert items is not None
    for n in range(3):
        terms = log_step(l, opt, level, [fov], [items], [np.arange(items.n)], n, RerankCfg(chunk=16), StepCfg(), 1.0, None)
    assert terms is not None and all(np.isfinite(v) for v in terms.values()), terms
    assert logloret_rank(l.eval(), res, 'cls', 30) >= 1.0
    return 'both architectures take steps, rank and validate on grids of P*F numbers a tile (P=3, F=16)'


def t_the_steps_add_the_decoders_losses_by_their_weights_and_log_them() -> str:
    from GloRet import GloRetNet, LogLoRetNet                                         # noqa: PLC0415
    from training.GloRet.Full import FullCfg, glo_step                                # noqa: PLC0415
    from training.GloRet.Items import level_items, list_items                         # noqa: PLC0415
    from training.GloRet.Rerank import RerankCfg, log_step                            # noqa: PLC0415
    from training.GloRet.Steps import StepCfg                                         # noqa: PLC0415
    torch.manual_seed(0)
    _, level, fov = episode_world(answer_at=2)
    items = list_items(level, fov, {})
    assert items is not None
    for recon, w_recon, w_kl in (('', 0.0, 0.0), ('ae', 0.0, 0.0), ('ae', 0.5, 0.0), ('vae', 0.5, 0.1)):
        step = StepCfg(chunk=32, w_recon=w_recon, w_kl=w_kl)
        net = GloRetNet(small_cfg(recon=recon))
        logs = glo_step(net, torch.optim.AdamW(net.parameters(), lr=1e-3), level, [fov], [level_items(level, fov, {})], None, 0, FullCfg(chunk=32),
                        step, 1.0, None)
        assert logs is not None and ('recon' in logs) == bool(recon) and ('kl' in logs) == (recon == 'vae'), (recon, logs)
        assert all(np.isfinite(v) for v in logs.values())
        net2 = LogLoRetNet(small_cfg(recon=recon))
        logs2 = log_step(net2, torch.optim.AdamW(net2.parameters(), lr=1e-3), level, [fov], [items], [np.arange(items.n)], 0,
                         RerankCfg(chunk=8), step, 1.0, None)
        assert logs2 is not None and all(np.isfinite(v) for v in logs2.values()), (recon, logs2)
    return 'a GloRet and a LogLoRet step log recon / kl when the net has the decoders; without a mode or a narrowing nothing is logged'


def t_a_head_takes_its_items_and_top_after_its_name_and_its_role_is_how_it_is_trained() -> str:
    from training.GloRet.cli.train import parse_head                                   # noqa: PLC0415
    g = parse_head('GloRet-SelfAttn-AttnPool')
    assert (g.items, g.role, g.top, g.variant, g.name) == ('level', 'retriever', None, '', 'GloRet-SelfAttn-AttnPool')
    same = parse_head('GloRet-SelfAttn-AttnPool@items=level')
    assert same.variant == '' and same.name == g.name, 'what is the default is not a change'
    h = parse_head('GloRet-SelfAttn-AttnPool@items=list@top=500')
    assert (h.items, h.role, h.variant) == ('list', 'reranker', 'items-list_top-500')
    assert h.top is not None and h.top.count == 500 and h.name == 'GloRet-SelfAttn-AttnPool@items=list@top=500'
    l = parse_head('LogLoRet-SelfAttn-AttnPool+sinkhorn@items=level@top=0.05')
    assert (l.arch, l.op, l.items, l.role) == ('logloret', 'sinkhorn', 'level', 'retriever'), 'trained over the level: a retriever'
    assert l.top is not None and l.top.share == 0.05 and l.variant == 'items-level_top-0.05'
    r = parse_head('LogLoRet-SelfAttn-AttnPool')
    assert (r.items, r.role, r.variant) == ('list', 'reranker', ''), 'a LogLoRet head is by default trained over the list'
    for bad in ('LogLoRet-SelfAttn-AttnPool@role=retriever', 'GloRet-SelfAttn-AttnPool@items=cloud', 'GloRet-SelfAttn-AttnPool@x=1',
                'GloRet-SelfAttn-AttnPool@top', 'GloRet-SelfAttn-AttnPool@role=judge', 'GloRet-SelfAttn-AttnPool@top=0.1 0.2'):
        try:
            parse_head(bad)
        except ValueError:
            continue
        raise AssertionError(f'{bad} was accepted')
    return 'defaults per architecture; options after @; a changed option is part of the name and the weights\' variant; the role comes from the items, whichever the architecture'


def t_a_list_without_a_positive_gets_the_answer_added_and_a_list_with_one_does_not() -> str:
    from training.GloRet.Items import level_items, list_items                          # noqa: PLC0415
    from training.GloRet.Supply import fov_of, level_of                               # noqa: PLC0415
    res, level, fov = episode_world(answer_at=3)
    items = list_items(level, fov, {})
    assert items is not None and items.injected == 0 and items.n == len(fov.listing) and items.pos.any()
    res2 = glo_result(glo_list(with_answer=False))
    level2, fov2 = level_of(res2, 'cls'), fov_of(res2)
    items2 = list_items(level2, fov2, {})
    assert items2 is not None and items2.injected >= 1 and items2.n == len(fov2.listing) + items2.injected
    assert items2.pos[-items2.injected:].all() and not items2.pos[:-items2.injected].any(), 'the added ones are the positives'
    assert (items2.rot[-items2.injected:] == fov2.placed.rot).all() and items2.teacher is not None
    assert items2.teacher[-1] < items2.teacher[:-items2.injected].min(), 'and the teacher ranks them last'
    assert list_items(level2, fov2, {}, inject=False) is None, 'a level that cannot name its windows leaves the FoV out'
    # a level's items: every window; they name themselves as Cands in their own place
    lv = level_items(level2, fov2, {})
    assert lv is not None and lv.source == 'level' and lv.injected == 0
    pick = np.array([0, 5, lv.n // 2, lv.n - 1])
    c = lv.cands_of(pick)
    assert np.array_equal(c.rot, lv.rot[pick]) and len(c) == 4
    for k, i in enumerate(pick):
        uni = level2.universe(fov2.shape(int(lv.rot[i])))
        assert uni.id_of(c.part[k], int(c.row[k]), int(c.col[k])) == int(lv.window_ids()[i])
    # and the pool of a few items is those windows, per rotation, sorted
    pool = lv.pool_of(pick)
    assert sum(len(v) for v in pool.values()) == 4 and all((np.diff(v) > 0).all() for v in pool.values())
    return 'no positive in the list: the answers are added (last, the teacher\'s worst); with one: nothing added; level items name their windows'


def t_a_gloret_step_over_the_entries_of_a_list_is_one_pass_over_them_and_ranks_them() -> str:
    from GloRet import GloRetNet                                                      # noqa: PLC0415
    from training.GloRet.Full import FullCfg, glo_step, rank_windows                  # noqa: PLC0415
    from training.GloRet.Items import list_items                                      # noqa: PLC0415
    from training.GloRet.Steps import StepCfg                                         # noqa: PLC0415
    from training.GloRet.Supply import fov_of, level_of                               # noqa: PLC0415
    res = glo_result(glo_list(with_answer=False))
    level, fov = level_of(res, 'cls'), fov_of(res)
    cache: dict = {}
    items = list_items(level, fov, cache)
    assert items is not None and items.injected >= 1
    pool = items.pool_of(np.arange(items.n))
    assert sum(len(v) for v in pool.values()) == items.n, 'one window an item'
    torch.manual_seed(0)
    net = GloRetNet(small_cfg(dropout=0.0, context='SelfAttn', global_en='AttnPool')).train()
    net.zero_grad()
    want = direct_pass(net, level, fov, pool, cache, StepCfg(), items)
    want.backward()
    direct = {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}
    opt = torch.optim.SGD(net.parameters(), lr=0.0)
    out = glo_step(net, opt, level, [fov], [items], [pool], 0, FullCfg(chunk=7), StepCfg(), 1e9, None)
    assert out is not None and abs(out['loss'] - float(want)) < 1e-4 * max(1.0, abs(float(want))), (out, float(want))
    assert out['windows'] <= items.n, 'only the windows of the list were encoded'
    for n, p in net.named_parameters():
        if n in direct:
            assert torch.allclose(p.grad, direct[n], rtol=1e-3, atol=1e-3 * float(direct[n].abs().max()) + 1e-6), n
    net.eval()
    got = rank_windows(net, level, [fov], [items], 0, FullCfg(chunk=7))[0]
    assert got is not None and len(got) == items.n and np.isfinite(got).all()
    j = items.n - 1                                           # the last item: an added answer
    rot = int(items.rot[j])
    shape = fov.shape(rot)
    uni = level.universe(shape)
    k = net.encode(net_input_of(uni.features(items.window_ids()[j:j + 1])), shape, 'window')
    q = net.encode(fov.queries[rot].reshape(1, -1, fov.queries[rot].shape[-1]).float(), shape, 'query')
    assert abs(float(net.score(q, k)[0, 0]) - float(got[j])) < 1e-3
    return 'the step over the entries of a list (an answer added) is the loss and gradient of one pass over those windows; the network scores each item as one by one'


def net_input_of(x: Any) -> Any:
    from training.GloRet.Steps import net_input                                       # noqa: PLC0415
    return net_input(x)


def t_a_head_with_flags_of_its_own_trains_on_its_settings_and_its_own_labels() -> str:
    from training.GloRet.cli.train import Visit, head_settings, parse_args, parse_head, relabelled   # noqa: PLC0415
    from training.GloRet.Items import level_items                                     # noqa: PLC0415
    args = parse_args(['--top-p', '1.0'])
    runs = [parse_head('GloRet-SelfAttn-AttnPool'),
            parse_head('GloRet-SelfAttn-AttnPool@w-geo=1.0@soft-target=exponential@sigma=2@ignore-expand=0.25@wrong-rotation=negative@dim=64')]
    head_settings(runs, args)
    a, b = runs
    assert a.relabel is None and a.step.w_geo == 0.0 and a.args.dim == args.dim
    assert (b.step.w_geo, b.step.soft_target, b.step.sigma, b.args.dim, b.relabel) == (1.0, 'exponential', 2.0, 64, (0.25, False))
    assert args.dim != 64 and args.w_geo == 0.0, 'the job\'s own are untouched'
    assert b.name.endswith('@dim=64') and 'w-geo-1.0' in b.variant and 'ignore-expand-0.25' in b.variant
    for bad in ('@soft-target=nope', '@sigma=x', '@wrong-rotation=maybe'):
        try:
            head_settings([parse_head('GloRet-SelfAttn-AttnPool' + bad)], args)
        except SystemExit:
            continue
        raise AssertionError(f'{bad} was accepted')
    # the same photos with other labels, made once
    _, level, fov = episode_world(answer_at=None)
    level.fovs.append(fov)
    visit = Visit(key=('s', 0), level=level)
    out = relabelled(visit, 0.25, False)
    rows, cols = fov.shape(fov.placed.rot)
    assert out[0].placed.zone == (0.25 * cols * level.tile_l0, 0.25 * rows * level.tile_l0) and not out[0].placed.wrong_rotation_ignored
    assert fov.placed.zone != out[0].placed.zone and out[0].queries is fov.queries and relabelled(visit, 0.25, False) is out
    old, new = level_items(level, fov, {}), level_items(level, out[0], {})
    assert old is not None and new is not None and new.ignore.sum() <= old.ignore.sum() and new.pos.sum() == old.pos.sum()
    return 'flags after a head\'s name are its own args, loss settings and labels; the job\'s are untouched; a bad value stops the job'


def t_the_hard_negatives_of_a_visit_are_counted_from_its_pools() -> str:
    from training.GloRet.Episodes import hard_negative_stats                           # noqa: PLC0415
    # three pools of 100 negatives asked for: none beats the answer, 50 do, 200 do (the pool holds the first 100 of them)
    seen = [(1000, 100, 0, 100, -1.0), (1000, 100, 50, 100, 0.8), (1000, 100, 200, 100, 0.6)]
    out = hard_negative_stats(seen, [2, 2, 4])
    assert (out['hn/items'], out['hn/k'], out['hn/pool']) == (1000.0, 100.0, 100.0)
    assert abs(out['hn/ahead'] - 250 / 3) < 1e-9 and abs(out['hn/ratio_med'] - 0.5) < 1e-9 and abs(out['hn/ratio_p90'] - 1.7) < 1e-9, out
    assert abs(out['hn/first'] - 1 / 3) < 1e-9 and abs(out['hn/full'] - 1 / 3) < 1e-9, 'one pool with no hard negative, one with all hard'
    assert abs(out['hn/padded'] - 0.5) < 1e-9, 'of the 300 asked for, 150 did not beat the answer'
    assert abs(out['hn/kept'] - 0.7) < 1e-9, 'the first pool of a FoV has no earlier one to be kept from'
    assert abs(out['hn/uses'] - 8 / 3) < 1e-9
    none = hard_negative_stats([], [])
    assert none == {'hn/uses': 0.0}, none
    assert 'hn/kept' not in hard_negative_stats([(10, 5, 1, 5, -1.0)], [1]), 'no earlier pool: no share kept'
    return 'N\' / K, the share of pools with no or all hard negatives, the padded share, the share kept and the uses of a FoV'


def t_a_visit_with_few_fovs_takes_fewer_of_them_in_a_step_so_that_none_is_seen_again_and_again() -> str:
    from dataclasses import replace                                                    # noqa: PLC0415
    from training.GloRet.Episodes import PoolCfg, fovs_each_step                       # noqa: PLC0415
    from training.GloRet.Pool import TopP                                              # noqa: PLC0415
    pc = PoolCfg(top=TopP(share=0.1), steps=16, fovs_per_step=8, max_fov_uses=2)
    assert [fovs_each_step(pc, n) for n in (64, 1000, 30, 8, 1)] == [8, 8, 4, 1, 1], 'ceil(uses * FoVs / steps), at most fovs-per-step, at least 1'
    off = replace(pc, max_fov_uses=0)
    assert [fovs_each_step(off, n) for n in (64, 3, 1)] == [8, 8, 8], '0: no limit'
    assert fovs_each_step(replace(pc, max_fov_uses=1), 64) == 4
    return 'few FoVs: fewer in a step (the steps stay as many); 64 or more: as before; 0: off'


def t_the_levels_kept_between_epochs_stay_on_the_card_within_its_budget_and_on_the_host_beyond_it() -> str:
    from types import SimpleNamespace                                                  # noqa: PLC0415
    from training.GloRet.Supply import CardBudget, MemoryFeatureStore                  # noqa: PLC0415

    class Feat:
        def __init__(self, n: int, card: bool) -> None:
            self.n, self.is_cuda = n, card

        def numel(self) -> int:
            return self.n

        def element_size(self) -> int:
            return 2

    class Found:
        def __init__(self, level: int, n: int, card: bool) -> None:
            self.regions = [SimpleNamespace(x=0, y=0, w=1, h=1)]
            self.ds, self.level, self.tile_size, self.overlap = 1.0, level, 224, True
            self.maps = [SimpleNamespace(features=Feat(n, card))]
            self.on_host: bool = False

        def to(self, where: str) -> 'Found':
            out = Found(self.level, self.maps[0].features.n, False)
            out.on_host = where == 'cpu'
            return out

    def read(store: MemoryFeatureStore, level: int) -> Any:
        return store.load([SimpleNamespace(x=0, y=0, w=1, h=1)], ds=1.0, level=level, tile_size=224, overlap=True)

    budget = CardBudget(1000)
    assert budget.take(600) and not budget.take(500) and budget.used == 600
    budget.give(600)
    budget.give(5)
    assert budget.used == 0, 'giving back more than was taken leaves none'
    store = MemoryFeatureStore(budget)
    a, b = Found(0, 300, True), Found(1, 300, True)          # 600 bytes each, 1000 to spend
    store.save(a)
    store.save(b)
    assert read(store, 0) is a and store.card_bytes() == 600 and budget.used == 600, 'the first fits: no copy, still on the card'
    held = read(store, 1)
    assert held is not b and held.on_host and not held.maps[0].features.is_cuda, 'the second does not fit: a copy on the host'
    store.save(Found(0, 100, True))                             # the same level again, smaller: the 600 are given back first
    assert store.card_bytes() == 200 and budget.used == 200
    store.save(Found(2, 10, False))                             # already on the host: never counted
    assert budget.used == 200 and store.nbytes() == 200 + 600 + 20, store.nbytes()
    plain = MemoryFeatureStore()
    plain.save(Found(0, 10, True))
    assert read(plain, 0).on_host and plain.card_bytes() == 0, 'no budget: every level goes to the host as before'
    return 'within the budget a level stays where it is, beyond it goes to the host; a level saved again hands its bytes back'


def t_the_memory_marks_read_zero_without_a_card_and_do_not_fail() -> str:
    from training.GloRet.Steps import begin_visit_peak, held_gb, mark_peak, peak_gb, visit_peak_gb   # noqa: PLC0415
    begin_visit_peak('cpu')
    mark_peak('cpu')
    assert (peak_gb('cpu'), held_gb('cpu'), visit_peak_gb('cpu')) == (0.0, 0.0, 0.0)
    return 'on the CPU the GPU memory readings are 0 and nothing breaks'


def t_the_log_prints_the_hard_negatives_the_memory_and_the_seconds_of_the_pools() -> str:
    from training.GloRet.LogLayout import hard_line, hard_negatives_block, memory_line, steps_block   # noqa: PLC0415
    got = {'items': 1000.0, 'k': 100.0, 'ahead': 83.3, 'ratio_med': 0.5, 'ratio_p90': 1.7, 'first': 0.33, 'full': 0.33,
           'padded': 0.5, 'pool': 100.0, 'uses': 2.7}
    text = hard_line(got)
    assert 'pool 100 of 1000 items (K 100)' in text and 'padded 50%' in text and 'trained 2.7' in text and 'last episode' not in text, text
    assert 'kept from the last episode 70%' in hard_line(dict(got, kept=0.7))
    name = 'GloRet-SelfAttn-AttnPool'
    losses = {f'{name}/hn/{k}': v for k, v in got.items()}
    block = hard_negatives_block([name], losses)
    assert len(block) == 5 and block[0].lstrip().startswith('hard negatives'), block
    assert "N'/K p90" in block[1] and 'kept' not in block[1] and '1.70' in block[3] and block[3].lstrip().startswith('GloRet'), block
    assert hard_negatives_block([name], {}) == [], 'a job with no pools to count prints nothing'
    assert memory_line({'m_held': 40.0, 'm_back': 12.34}) == 'GPU peak GB by phase  held before 40.0 · back 12.3'
    steps = steps_block([name], {f'{name}/t_pool': 250.0, f'{name}/t_keys': 1.0, f'{name}/m_held': 40.0, f'{name}/m_keys': 12.0})
    joined = '\n'.join(steps)
    assert '250.0' in joined and 'highest GPU memory' in joined and '40.0' in joined and '12.0' in joined, joined
    from types import SimpleNamespace                                                  # noqa: PLC0415
    from training.GloRet.cli.train import visit_report                                 # noqa: PLC0415
    visit = SimpleNamespace(key=('slide', 0), level=SimpleNamespace(fovs=[0, 1]))
    seconds = {f'{name}/t_pool': 141.1, f'{name}/t_pool_score': 130.2, f'{name}/t_pool_first': 90.0, f'{name}/t_pool_last': 20.1,
               f'{name}/t_scores': 1.4}
    text = visit_report(1, visit, seconds, [name], 195.0, None)
    assert 'pool 141.10 [scoring 130.20: first 90.00, last 20.10; rest 10.90] scores 1.40' in text, text
    bare = visit_report(1, visit, {f'{name}/t_pool': 1.5, f'{name}/t_keys': 0.5}, [name], 2.0, None)
    assert 'seconds: pool 1.50 keys 0.50' in bare and 'scoring' not in bare, bare
    return 'the hard-negative line and table, the memory line, the seconds / memory tables of the steps and the visit line with the pool split'


def t_scoring_a_level_shares_the_gathered_windows_and_gives_the_same_scores() -> str:
    from GloRet import LogLoRetNet                                                     # noqa: PLC0415
    from stage2_retrieval.Context import gather_windows                               # noqa: PLC0415
    from training.GloRet.Items import level_items                                     # noqa: PLC0415
    from training.GloRet.Rerank import RerankCfg, rerank_scores, score_pairs          # noqa: PLC0415
    torch.manual_seed(5)
    _, level, fov = episode_world(answer_at=3)
    a, b = level_items(level, fov, {}), level_items(level, fov, {})                 # two Items of one shape, as two FoVs of a visit
    assert a is not None and b is not None and a is not b and a.n > 0
    net = LogLoRetNet(small_cfg(dropout=0.1, context='SelfAttn', global_en='AttnPool')).eval()
    # the reference: every FoV alone, every chunk gathering its own windows
    want = []
    with torch.no_grad():
        for it in (a, b):
            parts = [score_pairs(net, level, fov, it.cands_of(np.arange(c, min(c + 7, it.n))),
                                 np.arange(min(c + 7, it.n) - c), 0, 'cpu')[0].float() for c in range(0, it.n, 7)]
            want.append(torch.cat(parts).numpy())
    state = torch.random.get_rng_state().clone()
    got = rerank_scores(net, level, [fov, fov], [a, b], 0, RerankCfg(chunk=7))
    assert torch.equal(torch.random.get_rng_state(), state), 'scoring in evaluation mode draws no random numbers'
    for g, w in zip(got, want):
        assert g is not None and np.array_equal(g, w), 'the shared gather gives bit-for-bit the same scores'
    only = rerank_scores(net, level, [fov, fov], [a, None], 0, RerankCfg(chunk=7))
    assert only[1] is None and np.array_equal(only[0], want[0])
    # one grid of windows: the gathered block is returned as it is, as the copy through an empty tensor would give it
    shape = fov.shape(0)
    c = a.cands_of(np.arange(min(a.n, 9)))
    sel = np.flatnonzero(c.rot == 0)
    if len(sel) and len(set(c.part[j] for j in sel)) == 1:
        feats = level.window_features([c.part[j] for j in sel], c.row[sel], c.col[sel], shape)
        block, _ = gather_windows(level.grids[c.part[sel[0]]], torch.as_tensor(c.row[sel]), torch.as_tensor(c.col[sel]), *shape)
        assert torch.equal(feats, block.reshape(len(sel), shape[0] * shape[1], -1))
    # the candidates made without a Python step per window are the ones the loop made, and the blocks made from the ids are
    # the blocks made from the candidates
    from training.GloRet.Rerank import level_blocks, window_blocks                    # noqa: PLC0415
    from training.GloRet.Supply import ROTATIONS                                      # noqa: PLC0415
    flat = np.arange(a.n)[::3]
    made = a.cands_of(flat)
    wid, rot = a.window_ids()[flat], a.rot[flat]
    for r in ROTATIONS:
        at = np.flatnonzero(rot == r)
        if not len(at):
            continue
        uni = level.universe(fov.shape(r))
        p, rr, cc = uni.locate(wid[at])
        assert [made.part[int(s)] for s in at] == [uni.parts[int(k)] for k in p]
        assert np.array_equal(made.row[at], rr) and np.array_equal(made.col[at], cc)
    ids = np.arange(len(flat))
    by_ids, by_cands = level_blocks(level, fov, a, flat), window_blocks(level, fov, made, ids)
    assert [(r, w.tolist()) for r, w, _ in by_ids] == [(r, w.tolist()) for r, w, _ in by_cands]
    assert all(torch.equal(x, y) for (_, _, x), (_, _, y) in zip(by_ids, by_cands))
    return 'FoVs of one shape gather a chunk once; the scores equal the one-by-one ones bit for bit and no random number is drawn'


def _old_logloret_forward(net: Any, fq: Any, fw: Any, shape: Any, s_up: Any = None, valid: Any = None, maps: bool = False) -> Any:
    """`LogLoRetNet.forward` as it was before it was split in `query_side`, `window_side` and `pair_side` (a copy, the reference)."""
    import math                                                                      # noqa: PLC0415
    import torch.nn.functional as F                                                  # noqa: PLC0415
    b, n = fw.shape[0], fw.shape[1]
    xq, gq = net.context(net.embed(fq), shape, None)
    xw, gw = net.context(net.embed(fw), shape, valid)
    if xq.shape[0] != b:
        xq = xq.expand(b, -1, -1)
        gq = gq.expand(b, -1) if gq is not None else None
    g_q, g_w = net.global_en(xq, shape, None, gq), net.global_en(xw, shape, valid, gw)
    s_global = F.cosine_similarity(g_q, g_w, dim=-1)
    s0_fw, s_fw, v_w = net.cross.logits(xq, xw, shape)
    s0_wf, s_wf, v_q = net.cross.logits(xw, xq, shape)
    a_fw, a_wf = net.match(s_fw, s_wf)
    c = net.cross.read(a_fw[..., :n], v_w)
    diag = torch.arange(n, device=fw.device)
    base = math.log(n + 1) if net.cfg.no_match != 'sigmoid' else math.log(n)
    log_n = base if net.cfg.log_n else 0.0
    s_ii = s0_fw[..., diag, diag]
    off = s0_fw.masked_fill(torch.eye(n, dtype=torch.bool, device=fw.device), float('-inf'))
    margin = s_ii - torch.logsumexp(off, dim=-1)
    bi = 0.5 * (a_fw[..., diag, diag].clamp(min=1e-9).log() + a_wf[..., diag, diag].clamp(min=1e-9).log()) + log_n
    rho_fw, rho_wf = a_fw[..., n], a_wf[..., n]
    tile_a = a_fw[..., :n] / a_fw[..., :n].sum(dim=-1, keepdim=True).clamp(min=1e-9)
    entropy = -(tile_a * tile_a.clamp(min=1e-9).log()).sum(dim=-1) / math.log(max(n, 2))
    per_tile = torch.stack([s_ii, margin, bi, rho_fw, rho_wf, entropy], dim=-1).permute(0, 2, 1, 3)
    agg = net.agg_tiles(per_tile, valid)
    best = a_fw[..., :n].argmax(dim=-1)
    taken = torch.zeros(b, best.shape[1], n, device=fw.device, dtype=agg.dtype).scatter_(-1, best, 1.0)
    cover = taken.sum(dim=-1) / n
    collapse = a_fw[..., :n].sum(dim=-2).amax(dim=-1) / n
    head_ev = torch.stack([agg[..., 0], agg[..., 1], agg[..., 2], cover, collapse, agg[..., 3], agg[..., 4], agg[..., 5]], dim=-1)
    u = net.match_mlp(torch.cat([xq, c, xq * c, (xq - c).abs()], dim=-1))
    u_win = net.agg_match(u.unsqueeze(2), valid).squeeze(1)
    e_pair = net.norm(torch.cat([s_global.unsqueeze(-1), head_ev.reshape(b, -1), u_win], dim=-1))
    up = net.up(s_up.reshape(b, 1).to(e_pair.dtype)) if s_up is not None else torch.zeros(b, 1, device=fw.device, dtype=e_pair.dtype)
    out = {'score': net.score_mlp(torch.cat([e_pair, up], dim=-1)).squeeze(-1), 's_global': s_global}
    if maps:
        out.update(A_fw=a_fw, A_wf=a_wf)
    return out


def t_the_pair_scorer_split_in_query_window_and_pair_sides_gives_the_old_forward() -> str:
    from GloRet import LogLoRetNet, take_aux                                          # noqa: PLC0415
    from training.GloRet import Profile                                               # noqa: PLC0415
    shape, b = (2, 3), 5
    for recon in ('', 'vae'):
        for dropout in (0.0, 0.1):
            torch.manual_seed(7)
            net = LogLoRetNet(small_cfg(in_dim=48, dropout=dropout, recon=recon, context='SelfAttn', global_en='AttnPool'))
            fq, fw = torch.randn(1, 6, 48), torch.randn(b, 6, 48)
            # the forward pass in training mode: the same random draws in the same order, the same numbers, and gradients
            # that agree (the sums into a tensor with several readers may add in another order: to the last digit of float32)
            grads = []
            outs = []
            for run in (_old_logloret_forward, lambda n, *a, **k: n(*a, **k)):
                net.train()
                net.zero_grad()
                take_aux(net)
                torch.manual_seed(3)
                out = run(net, fq, fw, shape, None, None, True)
                (out['score'].sum() + out['A_fw'].clamp(min=1e-9).log().mean() + out['A_wf'].clamp(min=1e-9).log().mean()).backward()
                outs.append(out)
                grads.append({k: p.grad.clone() for k, p in net.named_parameters() if p.grad is not None})
            for key in ('score', 's_global', 'A_fw', 'A_wf'):
                assert torch.equal(outs[0][key], outs[1][key]), (recon, dropout, key)
            assert set(grads[0]) == set(grads[1])
            for key in grads[0]:
                assert torch.allclose(grads[0][key], grads[1][key], rtol=1e-5, atol=1e-6), (recon, dropout, key)
    # evaluation: the window side made once serves every query, bit for bit as the whole forward gives it
    net.eval()
    with torch.no_grad():
        ws = net.window_side(fw, shape)
        for seed in (1, 2):
            torch.manual_seed(seed)
            q = torch.randn(1, 6, 48)
            want = net(q, fw, shape, maps=True)
            got = net.pair_side(net.query_side(q, shape, b), ws, shape, maps=True)
            assert all(torch.equal(want[k], got[k]) for k in want), 'the shared window side gives the same scores'
            one = net.pair_side(net.query_side(q, shape), ws, shape, maps=True)         # the query kept as one row
            assert torch.allclose(one['score'], want['score'], rtol=1e-5, atol=1e-6)
    # the parts timed by name (`--profile`): the first pass profiled, the second timed by part, the methods put back
    with tempfile.TemporaryDirectory() as tmp:
        Profile.start(tmp)
        try:
            for _ in range(2):
                with Profile.profiled('t-pool', net), torch.no_grad():
                    ws = net.window_side(fw, shape)
                    net.pair_side(net.query_side(fq, shape, b), ws, shape)
            text = open(os.path.join(tmp, 'profile', 't-pool-modules.txt')).read()
        finally:
            Profile.start(None)
        assert os.path.exists(os.path.join(tmp, 'profile', 't-pool.txt'))
    for part in ('window_side', 'query_side', 'pair_side', 'window_side/embed', 'window_side/context', 'window_side/cross.answer',
                 'pair_side/match_mlp', 'pair_side/cross.score'):
        assert part in text, part
    assert 'window_side' not in net.__dict__ and 'score' not in net.cross.__dict__, 'the methods are put back'
    return 'the split forward equals the old one (outputs bit for bit, gradients to float rounding), the window side is shared, the parts are timed'


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
