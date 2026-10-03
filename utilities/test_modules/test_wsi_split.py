#!/usr/bin/env python3
"""Tests for utilities/WsiSplit.py -- the split's specs, its three sets, and what
it does to a split that is already recorded.

    python utilities/test_modules/test_wsi_split.py

No slide, no model, no GPU: slide names are made up and every file lives in a
temporary directory. Nothing here reads or writes a real `result/cache/*_split/`.

WHAT THIS DEFENDS
-----------------
    reproduces    the spec `val=10,test=rest` cuts exactly what the split every
                  package already reads was cut by -- checked against the OLD
                  algorithm written out here, not against `WsiSplit` itself
    compatible    with `train` added, val is unchanged, test is a PREFIX of the
                  old test (so every `test_names[:n]` still reads the same
                  slides) and train is its tail
    existing wins a recorded split is never re-derived; a spec that contradicts it
                  is reported, and `upgrade_split` refuses it
    one original  the backup made by an upgrade is the record as it was, and a
                  later upgrade does not overwrite it
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'))

from _paths import setup_import_paths                        # noqa: E402

setup_import_paths()

import WsiSplit as W                                           # noqa: E402

_RESULTS = []

NAMES = [f'w{i:03d}' for i in range(66)]          # the size of bracs/test's pool


def check(name, fn):
    try:
        out = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   {out}' if out else ''))
    except Exception as e:                                   # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n          {type(e).__name__}: {e}')


def _legacy(names, n_val, seed=42):
    """The split as it was made before there was a spec: sorted, shuffled, the
    first `n_val` are val and the rest are test. Written out here on purpose."""
    names = sorted(names)
    random.Random(seed).shuffle(names)
    return names[:n_val], names[n_val:]


def _refuses(fn, exc=ValueError, contains=''):
    try:
        fn()
    except exc as e:
        assert contains in str(e), e
        return
    raise AssertionError('was not refused')


def _vt(path):
    """`(val, test)` of a split file: what a reader that asks for those two gets."""
    sets = W.read_sets(path)
    return sets['val'], sets['test']


def _recorded(tmp, val, test):
    path = Path(tmp) / 'wsi_split.csv'
    W._write({'val': val, 'test': test}, path)
    return path


# ── specs ────────────────────────────────────────────────────────────────────

def t_a_spec_is_a_count_a_share_or_rest():
    assert W.parse_spec('val=10,test=rest') == {'val': 10, 'test': 'rest'}
    assert W.parse_spec('train=0.6,val=0.2,test=rest') == \
        {'train': 0.6, 'val': 0.2, 'test': 'rest'}
    assert W.parse_spec('val=1.0') == {'val': 1.0}, 'a decimal point makes a share'
    assert W.parse_spec(' val = 10 , test = REST ') == {'val': 10, 'test': 'rest'}


def t_a_spec_that_says_something_else_is_refused():
    for bad in ('', 'val', 'dev=3', 'val=0', 'val=1.5', 'val=ten', 'val=3,val=4',
                'val=rest,test=rest', 'val=-2'):
        _refuses(lambda bad=bad: W.parse_spec(bad))


def t_sizes_count_share_and_rest():
    assert W.resolve_sizes({'val': 10, 'test': 'rest'}, 66) == {'val': 10, 'test': 56}
    assert W.resolve_sizes({'train': 0.5, 'val': 10, 'test': 'rest'}, 66) == \
        {'train': 33, 'val': 10, 'test': 23}
    assert W.resolve_sizes({'val': 0.001, 'test': 'rest'}, 66)['val'] == 1, \
        'a share that rounds to nothing is still one slide'
    assert W.resolve_sizes({'val': 5, 'test': 5}, 66) == {'val': 5, 'test': 5}


def t_sizes_that_do_not_fit_are_refused():
    _refuses(lambda: W.resolve_sizes({'val': 60, 'test': 10}, 66), contains='ask for')
    _refuses(lambda: W.resolve_sizes({'val': 66, 'test': 'rest'}, 66),
             contains='empty')


# ── the cut ──────────────────────────────────────────────────────────────────

def t_val_and_rest_is_the_split_already_in_use():
    sets = W.split_sets('x', {'val': 10, 'test': 'rest'}, candidate_names=NAMES)
    val, test = _legacy(NAMES, 10)
    assert sets['val'] == val and sets['test'] == test
    assert W.split_wsi_names('x', 10, candidate_names=NAMES) == (val, test)
    assert set(sets) == {'val', 'test'}


def t_the_pool_is_sorted_before_the_shuffle():
    reordered = list(reversed(NAMES))
    assert W.split_sets('x', {'val': 10, 'test': 'rest'},
                        candidate_names=reordered)['val'] == _legacy(NAMES, 10)[0]


def t_train_is_the_tail_of_the_old_test_and_val_does_not_move():
    val, old_test = _legacy(NAMES, 10)
    sets = W.split_sets('x', {'val': 10, 'test': 20, 'train': 'rest'},
                        candidate_names=NAMES)
    assert sets['val'] == val
    assert sets['test'] == old_test[:20] and sets['train'] == old_test[20:]
    every = sets['val'] + sets['test'] + sets['train']
    assert sorted(every) == sorted(NAMES), 'every slide in exactly one set'


def t_a_prefix_of_test_reads_the_same_slides_before_and_after():
    _, old_test = _legacy(NAMES, 10)
    new = W.split_sets('x', {'val': 10, 'test': 20, 'train': 'rest'},
                       candidate_names=NAMES)['test']
    for n in (1, 5, 10, 20):
        assert new[:n] == old_test[:n], n


def t_a_spec_without_rest_leaves_the_others_in_no_set():
    sets = W.split_sets('x', {'val': 10, 'test': 10, 'train': 10},
                        candidate_names=NAMES)
    assert {k: len(v) for k, v in sets.items()} == {'val': 10, 'test': 10, 'train': 10}


# ── the file ─────────────────────────────────────────────────────────────────

def t_a_file_with_train_reads_back_set_for_set():
    sets = W.split_sets('x', {'val': 10, 'test': 20, 'train': 'rest'},
                        candidate_names=NAMES)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'wsi_split.csv'
        W._write(sets, path)
        assert W.read_sets(path) == sets
        assert _vt(path) == (sets['val'], sets['test'])


# ── what a spec does to a record ─────────────────────────────────────────────

def t_check_says_same_for_the_spec_that_made_the_record():
    val, test = _legacy(NAMES, 10)
    with tempfile.TemporaryDirectory() as tmp:
        path = _recorded(tmp, val, test)
        assert W.check_split('x', W.parse_spec('val=10,test=rest'), path,
                             candidate_names=NAMES) == ('same', [])


def t_check_says_upgradable_when_the_spec_only_adds_train():
    val, test = _legacy(NAMES, 10)
    with tempfile.TemporaryDirectory() as tmp:
        path = _recorded(tmp, val, test)
        status, why = W.check_split(
            'x', W.parse_spec('val=10,test=20,train=rest'), path,
            candidate_names=NAMES)
        assert status == 'upgradable', (status, why)


def t_check_says_mismatch_when_the_spec_moves_val():
    val, test = _legacy(NAMES, 10)
    with tempfile.TemporaryDirectory() as tmp:
        path = _recorded(tmp, val, test)
        status, why = W.check_split('x', W.parse_spec('val=11,test=rest'), path,
                                    candidate_names=NAMES)
        assert status == 'mismatch' and any(r.startswith('val:') for r in why), why


def t_a_changed_pool_is_a_mismatch_which_is_why_the_record_exists():
    val, test = _legacy(NAMES, 10)
    with tempfile.TemporaryDirectory() as tmp:
        path = _recorded(tmp, val, test)
        grown = NAMES + ['w999']
        status, _ = W.check_split('x', W.parse_spec('val=10,test=rest'), path,
                                  candidate_names=grown)
        assert status == 'mismatch', 'one new slide reshuffled the whole split'


def t_check_says_missing_without_a_record():
    with tempfile.TemporaryDirectory() as tmp:
        assert W.check_split('x', W.parse_spec('val=10,test=rest'),
                             Path(tmp) / 'nope.csv',
                             candidate_names=NAMES)[0] == 'missing'


def t_a_recorded_split_is_never_rederived():
    val, test = _legacy(NAMES, 10)
    with tempfile.TemporaryDirectory() as tmp:
        path = _recorded(tmp, val, test)
        got = W.make_split_spec('x', W.parse_spec('val=5,test=rest'), path,
                                candidate_names=NAMES)
        assert got == {'val': val, 'test': test}
        assert _vt(path) == (val, test), 'the file was rewritten'


def t_a_missing_record_is_written_once():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'wsi_split.csv'
        sets = W.make_split_spec('x', W.parse_spec('val=10,test=rest'), path,
                                 candidate_names=NAMES)
        assert W.read_sets(path) == sets
        assert (sets['val'], sets['test']) == _legacy(NAMES, 10)


# ── upgrading ────────────────────────────────────────────────────────────────

def t_an_upgrade_adds_train_keeps_the_original_and_moves_nothing_read_so_far():
    val, test = _legacy(NAMES, 10)
    spec = W.parse_spec('val=10,test=20,train=rest')
    with tempfile.TemporaryDirectory() as tmp:
        path = _recorded(tmp, val, test)
        before = _vt(path)
        assert W.upgrade_split('x', spec, path, candidate_names=NAMES) == 'upgraded'
        backup = Path(str(path) + '.orig')
        assert _vt(backup) == before, 'the backup is the old record'
        assert W.read_sets(backup).get('train') is None
        after = W.read_sets(path)
        assert after['val'] == val and after['train'] == test[20:]
        for n in (1, 5, 10, 20):
            assert _vt(path)[1][:n] == before[1][:n], n
        assert W.check_split('x', spec, path, candidate_names=NAMES) == ('same', [])


def t_a_second_upgrade_changes_nothing():
    val, test = _legacy(NAMES, 10)
    spec = W.parse_spec('val=10,test=20,train=rest')
    with tempfile.TemporaryDirectory() as tmp:
        path = _recorded(tmp, val, test)
        W.upgrade_split('x', spec, path, candidate_names=NAMES)
        first = path.read_bytes()
        assert W.upgrade_split('x', spec, path, candidate_names=NAMES) == 'same'
        assert path.read_bytes() == first


def t_an_upgrade_refuses_a_spec_that_contradicts_the_record_and_touches_nothing():
    val, test = _legacy(NAMES, 10)
    with tempfile.TemporaryDirectory() as tmp:
        path = _recorded(tmp, val, test)
        before = path.read_bytes()
        _refuses(lambda: W.upgrade_split(
            'x', W.parse_spec('val=11,test=20,train=rest'), path,
            candidate_names=NAMES), contains='mismatch')
        assert path.read_bytes() == before
        assert not Path(str(path) + '.orig').exists(), 'a backup of a refusal'


def t_an_upgrade_never_overwrites_the_original():
    val, test = _legacy(NAMES, 10)
    spec = W.parse_spec('val=10,test=20,train=rest')
    with tempfile.TemporaryDirectory() as tmp:
        path = _recorded(tmp, val, test)
        Path(str(path) + '.orig').write_text('the first original')
        before = path.read_bytes()
        _refuses(lambda: W.upgrade_split('x', spec, path, candidate_names=NAMES),
                 exc=FileExistsError)
        assert Path(str(path) + '.orig').read_text() == 'the first original'
        assert path.read_bytes() == before


_SECTIONS = {
    'spec': ['t_a_spec_is_a_count_a_share_or_rest',
             't_a_spec_that_says_something_else_is_refused',
             't_sizes_count_share_and_rest',
             't_sizes_that_do_not_fit_are_refused'],
    'cut': ['t_val_and_rest_is_the_split_already_in_use',
            't_the_pool_is_sorted_before_the_shuffle',
            't_train_is_the_tail_of_the_old_test_and_val_does_not_move',
            't_a_prefix_of_test_reads_the_same_slides_before_and_after',
            't_a_spec_without_rest_leaves_the_others_in_no_set'],
    'file': ['t_a_file_with_train_reads_back_set_for_set'],
    'record': ['t_check_says_same_for_the_spec_that_made_the_record',
               't_check_says_upgradable_when_the_spec_only_adds_train',
               't_check_says_mismatch_when_the_spec_moves_val',
               't_a_changed_pool_is_a_mismatch_which_is_why_the_record_exists',
               't_check_says_missing_without_a_record',
               't_a_recorded_split_is_never_rederived',
               't_a_missing_record_is_written_once'],
    'upgrade': [
        't_an_upgrade_adds_train_keeps_the_original_and_moves_nothing_read_so_far',
        't_a_second_upgrade_changes_nothing',
        't_an_upgrade_refuses_a_spec_that_contradicts_the_record_and_touches_nothing',
        't_an_upgrade_never_overwrites_the_original'],
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', choices=sorted(_SECTIONS))
    args = ap.parse_args()
    for section in (args.only or list(_SECTIONS)):
        print(f'\n[{section}]')
        for name in _SECTIONS[section]:
            check(name[2:].replace('_', ' '), globals()[name])
    failed = [n for n, e in _RESULTS if e is not None]
    print(f'\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed')
    if failed:
        print('failed: ' + ', '.join(failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
