#!/usr/bin/env python3
"""Tests for utilities/Cache.py -- the tree, addresses and entries, atomic
writes, the sidecar, the slide key. No torch, no slide, temp dirs.

    python utilities/test_modules/test_cache.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
import _paths                                                    # noqa: E402
_paths.setup_import_paths()

import Cache                                                     # noqa: E402
from Cache import (CacheMismatch, atomic_file, cache_root,          # noqa: E402
                   check_source, source_key, wsi_stem_of)

_RESULTS = []


def check(name, fn):
    try:
        out = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   {out}' if out else ''))
    except Exception as e:                                       # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n          {type(e).__name__}: {e}')


def t_root_is_made_by_then_object():
    root = cache_root('MppRoutingHead', 'sampler')
    assert root.name == 'MppRoutingHead_sampler' and root.parent.name == 'cache', root
    try:
        cache_root('', 'sampler')
    except ValueError:
        return str(root)
    raise AssertionError('an empty made_by was accepted')


def t_job_name_prefers_slurm():
    old = os.environ.pop('SLURM_JOB_NAME', None)
    try:
        assert Cache.job_name('Default') == 'Default'
        os.environ['SLURM_JOB_NAME'] = 'FromSlurm'
        assert Cache.job_name('Default') == 'FromSlurm'
    finally:
        os.environ.pop('SLURM_JOB_NAME', None)
        if old is not None:
            os.environ['SLURM_JOB_NAME'] = old
    return 'SLURM_JOB_NAME wins, default otherwise'


def t_atomic_file_leaves_nothing_on_failure():
    with tempfile.TemporaryDirectory() as root:
        target = Path(root) / 'a.json'
        try:
            with atomic_file(target) as tmp:
                tmp.write_text('half')
                raise RuntimeError('killed')
        except RuntimeError:
            pass
        assert not target.exists(), 'a failed write left the target'
        assert list(Path(root).iterdir()) == [], list(Path(root).iterdir())
    return 'no target, no temp'


def t_slide_key_survives_a_remount_and_splits_a_stem():
    a = '/work/data/Group_A/SLIDE_1.svs'
    assert wsi_stem_of(a) == 'SLIDE_1'
    assert source_key(a) == source_key('/mnt/other/Group_A/SLIDE_1.svs')
    assert source_key(a) != source_key('/work/data/Group_B/SLIDE_1.svs')
    meta = {'source': source_key(a)}
    check_source(meta, '/mnt/other/Group_A/SLIDE_1.svs', 'x')
    try:
        check_source(meta, '/work/data/Group_B/SLIDE_1.svs', 'x')
    except CacheMismatch:
        return 'remount accepted, same stem elsewhere refused'
    raise AssertionError('a different slide with the same stem was accepted')


# ── the tree ──────────────────────────────────────────────────────────────────

class _UnderTemp:
    """Cache.RESULT_DIR pointed at a temp dir for the length of a test."""

    def __enter__(self) -> Path:
        self._tmp = tempfile.TemporaryDirectory()
        self._old, Cache.RESULT_DIR = Cache.RESULT_DIR, self._tmp.name
        return Path(self._tmp.name)

    def __exit__(self, *exc):
        Cache.RESULT_DIR = self._old
        self._tmp.cleanup()


def _refused(fn, *args, **kw) -> str:
    try:
        fn(*args, **kw)
    except ValueError as e:
        return str(e)
    raise AssertionError(f'{fn.__name__}{args}{kw} was accepted')


def _rec(id_, **parts):
    return {'id': id_, 'parts': [f'{k}={v}' for k, v in sorted(parts.items())],
            'upstream': {}, 'versions': {}, 'env': {}}


def t_every_entry_level_is_a_level_and_every_parent_a_kind():
    assert all(p is None or p in Cache.TREE for p in Cache.TREE.values())
    assert all(lv in Cache.TREE for lvs in Cache.ENTRIES.values() for lv in lvs)
    return f'{len(Cache.TREE)} levels, {len(Cache.ENTRIES)} entry kinds'


def t_address_orders_by_tree_not_by_argument():
    with _UnderTemp() as res:
        a = Cache.Address('Job', region='r1', slide='S_1', seg='hest-ab12')
        assert a.dir == res / 'cache' / 'Job' / 'slide=S_1' / 'seg=hest-ab12' / 'region=r1', a.dir
        assert a == Cache.Address('Job', slide='S_1', seg='hest-ab12', region='r1')
        assert a.at(plan='p').leaf == 'plan' and a.leaf == 'region'
    return 'slide/seg/region whatever the kwargs order'


def t_address_refuses_what_the_tree_does_not_have():
    msgs = [
        _refused(Cache.Address, 'J', slide='s', region='r'),          # no seg
        _refused(Cache.Address, 'J', slide='s', seg='g', region='r',
                 grid='t256-o1', plan='p'),                            # two branches
        _refused(Cache.Address, 'J', slide='s', colour='x'),           # unknown kind
        _refused(Cache.Address, 'J', slide='a/b'),                     # two components
        _refused(Cache.Address, 'J', slide='a=b'),
        _refused(Cache.Address, 'J', slide=''),
        _refused(Cache.Address('J', slide='s').at, slide='t'),         # twice
    ]
    assert 'seg' in msgs[0], msgs[0]
    return f'{len(msgs)} refused'


def t_on_keeps_the_levels_and_moves_the_root():
    with _UnderTemp() as res:
        a = Cache.Address('Reader', slide='s', seg='g')
        b = a.on('Writer')
        assert b.levels == a.levels and b.dir == res / 'cache' / 'Writer' / 'slide=s' / 'seg=g'
        assert _refused(lambda: Cache.Address(slide='s').dir)
    return 'same levels, other root; no job is no root'


def t_a_dataset_id_is_one_path_component():
    assert Cache.dataset_key('bracs/test') == 'bracs_test'
    assert Cache.dataset_key('ki67_with_photo') == 'ki67_with_photo'
    _refused(Cache.dataset_key, 'bracs/test#val')     # a split is not a dataset
    # a Ki67 MRXS stem: commas are part of real slide names
    assert Cache.Address('J', slide='S1104233,G7E,110208').leaf == 'slide'
    return 'bracs/test -> bracs_test; a #split refused; a Ki67 stem accepted'


def t_entry_sits_only_where_entries_says():
    a = Cache.Address('J', slide='s')
    assert a.entry('mask').dir.name == 'mask'
    _refused(a.entry, 'stage1')
    _refused(a.entry, 'nonsense')
    _refused(a.entry('mask').path, 'mask', 'id', 'safetensors')       # no dot
    return 'mask at slide; stage1 at slide refused'


def t_write_puts_the_record_last_and_a_failure_leaves_nothing():
    with _UnderTemp():
        e = Cache.Address('J', slide='s').entry('mask')
        want = _rec('hest-1', a=1)
        with e.writing('hest-1', want) as put:
            put('mask', '.bin').write_bytes(b'x')
            assert e.status('hest-1', want) == ('miss', []), 'record before the files'
        assert e.status('hest-1', want) == ('hit', [])
        assert sorted(p.name for p in e.dir.iterdir()) == ['mask_hest-1.bin',
                                                         'record_hest-1.json']
        try:
            with e.writing('hest-2', _rec('hest-2')) as put:
                put('mask', '.bin').write_bytes(b'half')
                raise RuntimeError('killed')
        except RuntimeError:
            pass
        assert e.ids() == ['hest-1'] and sorted(p.name for p in e.dir.iterdir()) == [
            'mask_hest-1.bin', 'record_hest-1.json'], list(e.dir.iterdir())
        try:
            with e.writing('hest-3', _rec('hest-3')) as put:
                put('mask', '.bin')                                    # never written
        except RuntimeError:
            pass
        assert e.ids() == ['hest-1']
    return 'miss while writing, hit after; killed and empty writes leave nothing'


def t_stale_names_the_difference_and_a_rewrite_drops_old_roles():
    with _UnderTemp():
        e = Cache.Address('J', slide='s', seg='g', region='r', plan='p',
                          draw='d', render='g1').entry('stage1')
        with e.writing('knn-1', _rec('knn-1', k=5)) as put:
            put('output', '.csv').write_text('old')
            put('probs', '.csv').write_text('old')
            put('photos').mkdir()
        state, diff = e.status('knn-1', _rec('knn-1', k=7))
        assert state == 'stale' and any('k=7' in d for d in diff), diff
        with e.writing('knn-1', _rec('knn-1', k=7)) as put:
            put('output', '.csv').write_text('new')
        names = sorted(p.name for p in e.dir.iterdir())
        assert names == ['output_knn-1.csv', 'record_knn-1.json'], names
        assert (e.dir / 'output_knn-1.csv').read_text() == 'new'
        assert e.status('knn-1', _rec('knn-1', k=7)) == ('hit', [])
    return 'stale with the part named; probs and photos of the old write gone'


def t_drop_touches_no_other_variant_even_one_sharing_a_suffix():
    with _UnderTemp():
        e = Cache.Address('J', slide='s', seg='g', region='r', plan='p',
                          draw='d', render='g1', stage1='s1').entry('stage2')
        for id_ in ('x', 'sims_x'):
            with e.writing(id_, _rec(id_)) as put:
                put('tile_sims', '.csv').write_text(id_)
        e.drop('x')
        assert e.ids() == ['sims_x'] and (e.dir / 'tile_sims_sims_x.csv').exists()
        assert not (e.dir / 'tile_sims_x.csv').exists()
    return "dropping 'x' kept 'sims_x'"


def t_members_is_the_cache_s_own_key():
    with _UnderTemp():
        e = Cache.Address('J', slide='s').entry('mask')
        try:
            with e.writing('a', {**_rec('a'), Cache.MEMBERS: []}):
                pass
        except ValueError:
            return 'a caller record carrying members refused'
    raise AssertionError('a record with members was accepted')


_TESTS = [t for n, t in sorted(globals().items()) if n.startswith('t_')]


def main():
    for fn in _TESTS:
        check(fn.__name__[2:].replace('_', ' '), fn)
    failed = [n for n, e in _RESULTS if e is not None]
    print(f'\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed')
    if failed:
        print('failed: ' + ', '.join(failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
