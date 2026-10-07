#!/usr/bin/env python3
"""Tests for utilities/Cache.py -- layout, atomic writes, the sidecar, the slide
key. Standard library only, like the module: no torch, no slide, temp dirs.

    python utilities/test_modules/test_cache.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
import _paths                                                    # noqa: E402
_paths.setup_import_paths()

import Cache                                                     # noqa: E402
from Cache import (CacheMismatch, atomic_dir, atomic_file,        # noqa: E402
                   cache_root, check_source, find, read_meta,
                   source_key, wsi_stem_of, write_meta)

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


def t_atomic_dir_first_writer_wins():
    """Two jobs finishing the same entry: the second must discard its copy,
    not merge into or replace the first one that readers may already hold."""
    with tempfile.TemporaryDirectory() as root:
        target = Path(root) / 'entry'
        with atomic_dir(target) as tmp:
            (tmp / 'who').write_text('first')
        with atomic_dir(target) as tmp:
            (tmp / 'who').write_text('second')
        assert (target / 'who').read_text() == 'first'
        assert [p.name for p in Path(root).iterdir()] == ['entry']
    return 'first kept, second discarded, no temp left'


def t_atomic_dir_leaves_nothing_on_failure():
    with tempfile.TemporaryDirectory() as root:
        target = Path(root) / 'entry'
        try:
            with atomic_dir(target) as tmp:
                (tmp / 'x').write_text('half')
                raise RuntimeError('killed')
        except RuntimeError:
            pass
        assert not target.exists() and list(Path(root).iterdir()) == []
    return 'no entry, no temp'


def t_read_meta_refuses_a_mismatch():
    with tempfile.TemporaryDirectory() as root:
        path = write_meta(Path(root) / 'm.json', {'seg_id': 'hest-1', 'n': 3})
        assert read_meta(path, require={'seg_id': 'hest-1'})['n'] == 3
        try:
            read_meta(path, require={'seg_id': 'hsv-2'})
        except CacheMismatch as e:
            assert 'seg_id' in str(e), str(e)
            return 'refused, and named the field'
    raise AssertionError('a mismatched require was accepted')


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


def t_find_filters_on_meta_and_sorts():
    with tempfile.TemporaryDirectory() as root:
        for name, method in (('B', 'hest'), ('A', 'hest'), ('C', 'hsv')):
            write_meta(Path(root) / name / 'mask_meta.json', {'method': method})
        got = find(root, '*/mask_meta.json', method='hest')
        assert [p.parent.name for p in got] == ['A', 'B'], got
        assert json.loads(got[0].read_text())['method'] == 'hest'
    return 'A, B'


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
