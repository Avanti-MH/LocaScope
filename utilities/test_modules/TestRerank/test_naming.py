#!/usr/bin/env python3
"""Tests for `stage2_retrieval/Naming.py`: one name for a stage 2 entry, whoever writes it.

    python utilities/test_modules/TestRerank/test_naming.py

No slide, no model, no GPU.

WHAT THIS DEFENDS
-----------------
    label      `{method}-{encoder}-{pooling}[-{operation}][-k{K}]`, made from the CONFIG: the encoder always
               written (none for a method that reads none), a pooling's underscore a dash, `-M` for the main
               lattice only, `-k{K}` only when the list is not 1000 long; every method has its word
    arm        an arm is named by its last phase; a reranker that names no encoder reads with the retriever's;
               the `@K` of a segment is its consumer's, so it is in the NEXT phase's name and in nobody's
               when it is the last one's
    entry      the own hash after the label, `_after-` and the upstream's hash for an entry that reads another
               list; the own hash comes back out of the name
    stage      a recipe of the pipeline and the bench's arm of the same config are one config, so one name and
               one id
"""

from __future__ import annotations

import dataclasses
import os
import sys
from typing import Callable, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))

from _paths import setup_import_paths                            # noqa: E402

setup_import_paths()

from TileEncoderFunc import encoder_config                       # noqa: E402
from stage2_retrieval import METHODS as STAGE_METHODS, recipe    # noqa: E402
from stage2_retrieval.ArmSpec import METHODS, config_of_arm, parse_arm   # noqa: E402
from stage2_retrieval.Naming import (METHOD_SLUGS, arm_label, config_label, entry_name, hash_of,   # noqa: E402
                                     label, slug, stage_name)
from stage2_retrieval.RetrieverSlidingWinSimRot import (WINDOW_SCORES,   # noqa: E402
                                                        RetrieverSlidingWinSimRotConfig)

_RESULTS: List[Tuple[str, Optional[BaseException]]] = []


def check(name: str, fn: Callable[[], Optional[str]]) -> None:
    try:
        out: Optional[str] = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   {out}' if out else ''))
    except Exception as e:                                       # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n          {type(e).__name__}: {e}')


def t_a_every_method_has_its_word_and_the_labels_read_as_written() -> str:
    assert set(METHODS) <= set(METHOD_SLUGS), set(METHODS) - set(METHOD_SLUGS)
    assert slug('cls_avg') == 'cls-avg' and slug('raw-M') == 'raw-M' and slug('nbr_support') == 'nbr-support'
    got = [arm_label(parse_arm(t)) for t in (
        'gigapath:SlidingWindow::cls_avg+geomean',
        'gigapath:TileVote::rings3+ivf',
        'gigapath:ClusterBow::cls_avg',
        'gigapath:SlidingWindow::cls+mean::Corr::cls_avg+sinkhorn',
        'uni2:BagOfTiles::cls_avg+gap::CSLS::grid2x2+csls@100',
        'uni2:BagOfTiles::cls_avg+gap@100::CSLS::grid2x2+csls',
        'uni2:SlidingWindow::cls+mean::CandGraph::raw+nbr_support',
        'uni2:SlidingWindow::cls_avg-M+min')]
    assert got == ['slidewin-gigapath-cls-avg-geomean', 'tilevote-gigapath-rings3-ivf',
                   'clusterbow-gigapath-cls-avg', 'corr-gigapath-cls-avg-sinkhorn',
                   'csls-uni2-grid2x2-csls', 'csls-uni2-grid2x2-csls-k100',
                   'candgraph-raw-nbr-support', 'slidewin-uni2-cls-avg-M-min'], got
    return ', '.join(got[4:7])


def t_b_a_reranker_that_names_no_encoder_reads_with_the_retrievers() -> str:
    assert arm_label(parse_arm('gigapath:SlidingWindow::cls+mean::Dual::cls_avg+dual')) == 'dual-gigapath-cls-avg-dual'
    assert arm_label(parse_arm('gigapath:SlidingWindow::cls+mean::uni2:Dual::cls_avg+dual')) == 'dual-uni2-cls-avg-dual'
    return 'dual-gigapath-cls-avg-dual / dual-uni2-cls-avg-dual'


def t_c_an_entry_name_carries_its_own_hash_and_the_upstreams() -> str:
    own, up = 'a' * 16, 'b' * 16
    plain = entry_name('slidewin-gigapath-cls-mean', own)
    chained = entry_name('corr-gigapath-cls-avg-diag', own, up)
    assert plain == f'slidewin-gigapath-cls-mean-{own}', plain
    assert chained == f'corr-gigapath-cls-avg-diag-{own}_after-{up}', chained
    assert hash_of(plain) == own and hash_of(chained) == own
    return chained


def t_d_a_sliding_window_config_is_named_by_what_it_spells() -> str:
    enc = encoder_config('gigapath', pooling='cls_avg').with_model(dtype='fp16')
    base = RetrieverSlidingWinSimRotConfig(encoder=enc, tile_size=256, overlap=True, raw_topk=1000,
                                           min_sep_tiles=1.0, score='geomean')
    assert stage_name('slidewin', base, 'x') == 'gigapath-cls-avg-geomean'
    main_only = dataclasses.replace(base, overlap=False, score='min')
    assert stage_name('slidewin', main_only, 'x') == 'gigapath-cls-avg-M-min', stage_name('slidewin', main_only, 'x')
    assert config_label(dataclasses.replace(base, raw_topk=500)) == 'slidewin-gigapath-cls-avg-geomean-k500'
    assert 'min' in WINDOW_SCORES, 'a pipeline run can be set to `min`'
    assert stage_name('knn', object(), 'gigapath') == 'gigapath', 'another stage keeps its recipe name'
    return 'gigapath-cls-avg-geomean / gigapath-cls-avg-M-min / -k500 when not 1000'


def t_e_the_label_of_a_main_lattice_window_keeps_the_marker() -> str:
    assert label('SlidingWindow', 'gigapath', 'raw', 'mean', 0, True) == 'slidewin-gigapath-raw-M-mean'
    assert label('SlidingWindow', 'uni2', 'cls_avg', 'min', 0, True) == 'slidewin-uni2-cls-avg-M-min'
    assert label('CandGraph', '', 'raw', 'prominence') == 'candgraph-raw-prominence'
    return 'slidewin-gigapath-raw-M-mean'


def t_f_a_recipe_and_the_arm_of_its_config_are_one_config() -> str:
    pairs = (('slidewin:uni2-avg', 'uni2:SlidingWindow::cls_avg+mean'),
             ('bagoftiles:uni2-avg-gap', 'uni2:BagOfTiles::cls_avg+gap'),
             ('tilevote:uni2-avg-exact', 'uni2:TileVote::cls_avg+exact'),
             ('clusterbow:uni2-avg', 'uni2:ClusterBow::cls_avg'),
             ('winembed:uni2-avg', 'uni2:WinEmbed::cls_avg'),
             ('multiphases:uni2-avg-dual', 'uni2:SlidingWindow::cls_avg+mean::Dual::cls_avg+dual'))
    got: List[str] = []
    for name, arm in pairs:
        assert name.split(':')[0] in STAGE_METHODS
        _, _, cfg, _ = recipe(name)
        mine = config_of_arm(parse_arm(arm))
        phases = cfg.phases if hasattr(cfg, 'phases') else (cfg,)
        assert [p.identity_id() for p in phases] == [p.identity_id() for p in mine.phases], (name, arm)
        assert config_label(cfg) == arm_label(parse_arm(arm)), (name, config_label(cfg))
        got.append(config_label(cfg))
    via_arm = recipe('arm:uni2:SlidingWindow::cls_avg+mean')
    assert via_arm[0] == 'multiphases' and via_arm[1] == 'uni2:SlidingWindow::cls_avg+mean'
    return ', '.join(got)


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
