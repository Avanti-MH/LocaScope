#!/usr/bin/env python3
"""Tests for the name of an arm: `stage2_retrieval/ArmSpec.py`.

    python utilities/test_modules/TestRerank/test_arm_spec.py

Pure text: no slide, no model, no GPU.

WHAT THIS DEFENDS
-----------------
    grammar    `[encoder:]Method::pooling[+op][@K]` repeated, joined by `::`: what is read is what is meant
    rules      a method with no operation has no `+`; one that takes no pooling writes `raw`; a retriever
               is first and only first, a reranker after it
    canonical  one arm has one text: the retriever's encoder is always written, a reranker's only when
               it is another one; parsing the text of an arm gives the arm
    product    k retrievers x h rerankers are k x h arms, a reranker with no encoder takes its retriever's
    config     an arm is the `MultiPhasesConfig` of its chain: a segment's `@K` is the next phase's
               `topk_in`, a reranker reuses the features of a phase before it exactly when they are the ones
               it would make, a `-M` pooling is the main lattice only, a chain that reads more than it is
               handed is refused
    chain      `MultiPhases` runs its phases in order, each on the output before it; a phase the cache has
               is not run and its list goes on with no extras
"""

from __future__ import annotations

import os
import sys
from typing import Callable, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))

from _paths import setup_import_paths                            # noqa: E402

setup_import_paths()

from stage2_retrieval.ArmSpec import (ArmSpec, ArmSyntaxError, Segment, expand,  # noqa: E402
                                      is_arm_text, parse_arm, parse_segment)

_RESULTS: List[Tuple[str, Optional[BaseException]]] = []


def check(name: str, fn: Callable[[], Optional[str]]) -> None:
    try:
        out: Optional[str] = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   {out}' if out else ''))
    except Exception as e:                                       # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n          {type(e).__name__}: {e}')


def refused(text: str, default: str = 'uni2') -> str:
    try:
        parse_arm(text, default)
    except ArmSyntaxError as e:
        return str(e)
    raise AssertionError(f'{text!r} should have been refused')


def t_a_retriever_and_a_reranker_with_their_encoders() -> str:
    spec: ArmSpec = parse_arm('uni2:BagOfTiles::cls_avg+gap::gigapath:CSLS::grid2x2+csls@100')
    assert spec.retriever == Segment('BagOfTiles', 'cls_avg', 'gap', 'uni2', 0)
    assert spec.rerankers == (Segment('CSLS', 'grid2x2', 'csls', 'gigapath', 100),)
    assert spec.text() == 'uni2:BagOfTiles::cls_avg+gap::gigapath:CSLS::grid2x2+csls@100'
    return spec.text()


def t_the_default_encoder_is_the_retrievers_and_the_text_says_it() -> str:
    spec: ArmSpec = parse_arm('SlidingWindow::cls+mean::Dual::cls_avg+dual', 'uni2')
    assert spec.retriever.encoder == 'uni2' and spec.rerankers[0].encoder == 'uni2'
    assert spec.text() == 'uni2:SlidingWindow::cls+mean::Dual::cls_avg+dual', 'a reranker of the same encoder is not marked'
    other: ArmSpec = parse_arm('uni2:SlidingWindow::cls+mean::gigapath:Dual::cls+dual')
    assert other.text().endswith('::gigapath:Dual::cls+dual'), 'another encoder is'
    assert parse_arm(other.text()) == other and parse_arm(spec.text()) == spec
    return 'one arm, one text'


def t_a_method_without_operations_has_no_plus() -> str:
    assert parse_arm('uni2:ClusterBow::cls_avg').retriever.op == ''
    assert 'no operation' in refused('uni2:ClusterBow::cls_avg+x')
    assert 'needs one of its operations' in refused('uni2:BagOfTiles::cls_avg')
    assert 'has no operation' in refused('uni2:BagOfTiles::cls_avg+mean')
    assert 'needs one of its operations' in refused('uni2:SlidingWindow::cls_avg')
    assert parse_arm('uni2:TileVote::cls_avg+hnsw').retriever.op == 'hnsw'
    return 'ops are checked against the method'


def t_a_method_with_no_pooling_writes_raw() -> str:
    spec: ArmSpec = parse_arm('uni2:SlidingWindow::cls+mean::TokenMatch::raw::CandGraph::raw+prominence')
    assert [s.method for s in spec.rerankers] == ['TokenMatch', 'CandGraph']
    assert 'takes no pooling' in refused('uni2:SlidingWindow::cls+mean::TokenMatch::cls_avg')
    assert 'takes no pooling' in refused('uni2:SlidingWindow::cls+mean::CandGraph::grid2x2+prominence')
    return 'raw for TokenMatch and CandGraph'


def t_a_retriever_is_first_and_a_reranker_is_after_it() -> str:
    assert 'cannot be an arm' in refused('uni2:Dual::cls+dual')
    assert 'cannot follow' in refused('uni2:SlidingWindow::cls+mean::BagOfTiles::cls_avg+gap')
    assert 'not a method' in refused('uni2:Nope::cls+mean')
    return 'order is enforced'


def t_the_cut_is_a_whole_number_after_the_pooling_and_op() -> str:
    seg: Segment = parse_arm('uni2:SlidingWindow::cls+mean@500').retriever
    assert seg.cut == 500 and seg.text() == 'uni2:SlidingWindow::cls+mean@500'
    assert parse_arm('uni2:SlidingWindow::cls+mean::Shift::cls_avg@30').rerankers[0].cut == 30
    assert '@K needs a whole number' in refused('uni2:SlidingWindow::cls+mean@x')
    assert '@K needs a whole number' in refused('uni2:SlidingWindow::cls+mean@0')
    return '@500 and @30'


def t_a_text_without_pairs_or_an_encoder_is_refused() -> str:
    assert 'repeated' in refused('uni2:SlidingWindow')
    assert 'repeated' in refused('uni2:SlidingWindow::cls+mean::Dual')
    assert 'no encoder' in refused('SlidingWindow::cls+mean', '')
    assert is_arm_text('uni2:SlidingWindow::cls+mean') and not is_arm_text('cls_avg+geomean')
    return 'the older bare tokens are told apart'


def t_k_retrievers_times_h_rerankers() -> str:
    retrievers: List[ArmSpec] = [parse_arm('uni2:SlidingWindow::cls+mean'), parse_arm('uni2:BagOfTiles::rings3+quad')]
    rerankers: List[Segment] = [parse_segment('Dual::cls_avg+dual'), parse_segment('gigapath:CSLS::grid2x2+csls@50'),
                                parse_segment('TokenMatch::raw')]
    arms: List[ArmSpec] = expand(retrievers, rerankers)
    assert len(arms) == 6
    assert arms[0].text() == 'uni2:SlidingWindow::cls+mean::Dual::cls_avg+dual'
    assert arms[1].text() == 'uni2:SlidingWindow::cls+mean::gigapath:CSLS::grid2x2+csls@50'
    assert arms[5].rerankers[0].encoder == 'uni2', 'a reranker with no encoder takes its retriever\'s'
    assert len({a.text() for a in arms}) == 6
    return '2 x 3 = 6 arms'


def t_a_prefix_is_the_list_after_those_segments() -> str:
    spec: ArmSpec = parse_arm('uni2:SlidingWindow::cls+mean::Dual::cls_avg+dual::CandGraph::raw+prominence@20')
    assert spec.prefix(1).text() == 'uni2:SlidingWindow::cls+mean'
    assert spec.prefix(2).text() == 'uni2:SlidingWindow::cls+mean::Dual::cls_avg+dual'
    assert spec.prefix(3) == spec
    return 'prefixes are the arms a chain passes through'


def t_an_arm_is_the_config_of_its_chain() -> str:
    import dataclasses                                           # noqa: PLC0415
    from stage2_retrieval.ArmSpec import config_of_arm, arm_topk   # noqa: PLC0415
    from stage2_retrieval.MultiPhases import MultiPhasesConfig   # noqa: PLC0415
    cfg = config_of_arm(parse_arm('uni2:SlidingWindow::cls_avg+mean@100::Dual::cls_avg+dual::'
                                  'CSLS::grid2x2+csls@30::gigapath:Corr::cls_avg+mnn@7'), raw_topk=1000)
    kinds = [type(p).__name__ for p in cfg.phases]
    assert kinds == ['RetrieverSlidingWinSimRotConfig', 'RerankerDualConfig', 'RerankerCslsConfig',
                     'RerankerCorrConfig'], kinds
    assert [getattr(p, 'topk_in', None) for p in cfg.phases] == [None, 100, 100, 30], 'the @K before is topk_in'
    assert [getattr(p, 'reuse', None) for p in cfg.phases[1:]] == [('level_features',), (), ()], \
        'only Dual reads what the window encoded (CSLS another pooling, Corr another encoder)'
    assert cfg.phases[0].raw_topk == 1000 and cfg.phases[0].overlap
    assert arm_topk(parse_arm('uni2:SlidingWindow::cls_avg+mean::Corr::cls_avg+mnn@7')) == 7
    after = config_of_arm(parse_arm('uni2:SlidingWindow::cls_avg+mean::CSLS::grid2x2+csls::Dual::grid2x2+dual'))
    assert after.phases[2].reuse == ('level_features',), 'Dual reads what CSLS encoded itself'
    main = config_of_arm(parse_arm('uni2:SlidingWindow::cls-M+mean::Dual::cls+dual'))
    assert not main.phases[0].overlap and main.phases[1].reuse == (), 'the main lattice only is another key'
    try:
        config_of_arm(parse_arm('uni2:SlidingWindow::raw+mean'))
        raise AssertionError('a raw sliding window was made')
    except NotImplementedError:
        pass
    try:
        MultiPhasesConfig(phases=(cfg.phases[0], dataclasses.replace(cfg.phases[1], topk_in=2000)))
        raise AssertionError('a phase reading more than it is handed was accepted')
    except ValueError:
        pass
    try:
        MultiPhasesConfig(phases=(cfg.phases[1],))
        raise AssertionError('a reranker first was accepted')
    except ValueError:
        pass
    return 'topk_in 100/100/30, reuse only where the key is the same, raw and bad chains refused'


def t_multiphases_runs_its_phases_and_a_cached_phase_is_not_run() -> str:
    from types import SimpleNamespace                           # noqa: PLC0415
    from stage2_retrieval.ArmSpec import config_of_arm           # noqa: PLC0415
    from stage2_retrieval.MultiPhases import MultiPhases         # noqa: PLC0415
    from stage2_retrieval.StageInterface import Candidate, CandidateSet, RerankedCandidate   # noqa: PLC0415
    from test_context import cset                                # noqa: PLC0415
    import numpy as np                                           # noqa: PLC0415
    cfg = config_of_arm(parse_arm('uni2:SlidingWindow::cls_avg+mean::CandGraph::raw+prominence@3'))
    chain: MultiPhases = MultiPhases(cfg, 'cpu')
    chain.wsi = object()
    for p in chain.phases:
        p.wsi = object()
    listed: CandidateSet = cset([Candidate(0, 'main', 0, i, 0, 1.0 - 0.1 * i) for i in range(6)])
    asked = []

    def cached(i: int, upstream: object):
        asked.append(i)
        return listed if i == 0 else None

    est = SimpleNamespace(chosen_level=1)
    out = chain.run(est, np.zeros((4 * 256, 5 * 256, 3), np.uint8), cached=cached)
    assert asked == [0, 1], 'every phase is asked for'
    assert out.parent is not None and out.parent.candidates is listed and not out.parent.extras, \
        'the cached list goes on, with no extras'
    assert len(out.candidates) == 6 and all(isinstance(c, RerankedCandidate) for c in out.candidates)
    assert [o.candidates for o in chain.outputs(out)] == [listed, out.candidates], 'first phase first'
    return 'phase 0 from the cache, phase 1 run on it'


_TESTS = [t for n, t in sorted(globals().items(), key=lambda kv: kv[0]) if n.startswith('t_')]


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
