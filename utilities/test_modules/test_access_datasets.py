#!/usr/bin/env python3
"""Tests for `utilities/AccessDatasets.py`.

    python utilities/test_modules/test_access_datasets.py

`AccessDatasets.py` is used by more than one package (SuperPathPoint's cli/
today, potentially others later), which is why this test sits in the flat
`test_modules/` directory rather than under any one package's `Test<Package>/`
subdirectory -- filing it under one consumer's jobscript would say something
false about who it belongs to (CLAUDE.md's own rule for this).

2026-09-11: `AccessDatasets.py` moved from one hand-written `WsiEntry` per
WSI to one hand-written naming-convention rule per DATASET (`_DATASETS`,
`locate_fn`/`list_fn` pairs) -- `list_names()`/`locate()` are themselves now
directory reads, not table lookups, which is why almost nothing here is
"pure" any more: even the section that used to just check the registry's own
internal consistency now touches disk, because there IS no registry to check
without touching disk. Only the `disk` section's NAME survives to mark the
distinction that mattered before (portable vs this-cluster-only) -- what
still separates the sections is whether a test needs a SPECIFIC real name to
exist (not portable) or just needs `list_names()` to return something
self-consistent, whatever it currently holds (still runs anywhere the roots
are reachable, but says nothing if they are empty).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'))

from _paths import setup_import_paths                        # noqa: E402

setup_import_paths()

import AccessDatasets as AD                                    # noqa: E402

_RESULTS = []


def check(name, fn):
    try:
        out = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   {out}' if out else ''))
    except Exception as e:                                   # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n          {type(e).__name__}: {e}')


# ── 1. datasets ──────────────────────────────────────────────────────────────

def t_list_names_has_no_duplicates_across_datasets():
    """Two registered datasets returning the SAME name would make `locate`
    silently prefer whichever is registered first -- the second dataset's
    entry for that name would be unreachable and nothing would say so."""
    names = AD.list_names()
    if len(names) != len(set(names)):
        dupes = {n for n in names if names.count(n) > 1}
        raise AssertionError(f'duplicate names across datasets: {dupes}')
    return f'{len(names)} names, all unique across every registered dataset'


def t_stains_covers_every_dataset():
    found = AD.stains()
    if set(found) != {d.stain for d in AD._DATASETS}:
        raise AssertionError(f'stains() {found} does not match the set of '
                             f'stains actually registered')
    return f'stains {found}'


# ── 2. locate ────────────────────────────────────────────────────────────────

def t_locate_returns_the_registered_entry():
    entry = AD.locate('BRACS_1228')
    if entry.stain != 'HE' or not entry.path.endswith('BRACS_1228.svs'):
        raise AssertionError(f'unexpected entry: {entry}')
    return f'{entry.name}: stain={entry.stain}'


def t_locate_unknown_name_raises_and_lists_known_names():
    try:
        AD.locate('not_a_real_slide')
    except KeyError as e:
        if 'BRACS_1228' not in str(e):
            raise AssertionError('KeyError should list known names') from e
        return 'KeyError names the known slides, does not guess'
    raise AssertionError('expected KeyError, got no exception')


def t_locate_ki67_with_photo_finds_a_photos_companion():
    """`Ki67_with_photo`'s whole distinguishing feature: a leftover entry
    in the container besides the `.mrxs` file and the pyramid dir resolves
    to `related['photos']`, whichever of the two historical spellings it is."""
    names = AD.list_names(stain='Ki67')
    with_photo_names = [n for n in names
                        if AD._KI67_ROOT in AD.locate(n).path]
    if not with_photo_names:
        raise AssertionError('no Ki67_with_photo names found at all -- '
                             'cannot check the photos companion')
    entry = AD.locate(with_photo_names[0])
    if 'photos' not in entry.related:
        raise AssertionError(f'{entry.name}: expected a photos companion, '
                             f'related={entry.related}')
    return f'{entry.name}: photos -> {entry.related["photos"]}'


def t_locate_ki67_pure_has_no_photos_companion():
    """`Ki67_pure` names should never carry a `photos` key -- that IS what
    'pure' names (no real-photo companion shipped with this batch)."""
    names = AD.list_names(stain='Ki67')
    pure_names = [n for n in names if AD._KI67_PURE_ROOT in AD.locate(n).path]
    if not pure_names:
        raise AssertionError('no Ki67_pure names found at all -- cannot '
                             'check the no-photos property')
    entry = AD.locate(pure_names[0])
    if 'photos' in entry.related:
        raise AssertionError(f'{entry.name}: unexpected photos companion '
                             f'{entry.related["photos"]!r} in a "pure" dataset')
    return f'{entry.name}: related={entry.related} (no photos key)'


def t_locate_with_dataset_kwarg_scopes_the_search():
    entry = AD.locate('BRACS_1228', dataset='bracs/test')
    if entry.name != 'BRACS_1228':
        raise AssertionError(f'unexpected entry: {entry}')
    try:
        AD.locate('BRACS_1228', dataset='ki67_pure')
    except KeyError:
        return 'dataset= scopes the search to just that one dataset'
    raise AssertionError('expected KeyError -- BRACS_1228 is not in ki67_pure')


def t_locate_group_type_hint_skips_the_glob_and_still_finds_the_slide():
    """`group_type` is `bracs/test`'s own narrowing hint -- goes straight
    to the exact path instead of globbing the whole split, but must land on
    the SAME entry the unhinted search finds."""
    hinted = AD.locate('BRACS_1228', dataset='bracs/test',
                       group_type='Group_AT/Type_ADH')
    unhinted = AD.locate('BRACS_1228', dataset='bracs/test')
    if hinted.path != unhinted.path:
        raise AssertionError(f'hinted path {hinted.path} != unhinted path '
                             f'{unhinted.path}')
    return f'group_type hint resolved to the same path: {hinted.path}'


def t_locate_wrong_group_type_hint_is_a_miss_not_a_crash():
    """A narrowing hint that does not actually match `name`'s real location
    is just a miss (KeyError, same as any unfound name) -- it is a search
    shortcut, not a second identity `name` has to also satisfy."""
    try:
        AD.locate('BRACS_1228', dataset='bracs/test',
                 group_type='Group_BT/Type_N')
    except KeyError:
        return 'wrong group_type resolves to not-found, not an exception'
    raise AssertionError('expected KeyError for a group_type that does not '
                         'actually hold BRACS_1228')


def t_locate_kwargs_without_dataset_raises():
    """`group_type` is meaningless without knowing which dataset it is a
    hint FOR -- passing it without `dataset=` must not be silently
    ignored."""
    try:
        AD.locate('BRACS_1228', group_type='Group_AT/Type_ADH')
    except ValueError:
        return 'kwargs without dataset= correctly rejected, not ignored'
    raise AssertionError('expected ValueError')


def t_locate_finds_a_bracs_train_slide():
    """`bracs/train` (`BRACS_WSI/train/`, 2026-09-11) -- registered the same
    way as `bracs/test`, a name found there resolves and is NOT also found
    unscoped-ambiguous with `bracs/test` (the two splits' stems were
    checked disjoint before registering `train` at all)."""
    train_ds = next(d for d in AD._DATASETS if d.id == 'bracs/train')
    names = train_ds.list_fn()
    if not names:
        raise AssertionError('bracs/train has no names at all -- '
                             'BRACS_WSI/train/ unreachable or empty?')
    entry = AD.locate(names[0], dataset='bracs/train')
    if entry.stain != 'HE':
        raise AssertionError(f'unexpected entry: {entry}')
    AD.locate(names[0])   # must also resolve unscoped -- no test/train collision
    return f'{entry.name}: {len(names)} bracs/train slides found, path={entry.path}'


def t_locate_unknown_dataset_id_raises():
    try:
        AD.locate('BRACS_1228', dataset='not_a_real_dataset')
    except KeyError as e:
        if 'not a registered dataset id' not in str(e):
            raise AssertionError(f'wrong KeyError: {e}')
        return 'unknown dataset id raises, does not silently fall through'
    raise AssertionError('expected KeyError')


def t_locate_raises_when_name_is_ambiguous_across_datasets():
    """The same 'no fuzzy match' principle `locate`'s own docstring states
    for typos applies to a genuine collision too: if two registered
    datasets both resolve the same name, `locate` must refuse to silently
    pick one. No real collision exists today (`t_list_names_has_no_
    duplicates_across_datasets` checks that), so this deliberately
    registers a second dataset pointed at the SAME root as the real BRACS
    one -- not a claim that dataset is real, just the cheapest way to
    exercise the ambiguity path -- and unregisters it in `finally`."""
    real_bracs = next(d for d in AD._DATASETS if d.id == 'bracs/test')
    dup = AD._Dataset(id='duplicate_bracs_for_test', stain='HE',
                      locate_fn=real_bracs.locate_fn,
                      list_fn=real_bracs.list_fn)
    AD._DATASETS.append(dup)
    try:
        AD.locate('BRACS_1228')
    except KeyError as e:
        if 'more than one dataset' not in str(e):
            raise AssertionError(f'wrong KeyError: {e}')
        return 'ambiguous name correctly refused rather than silently picked'
    else:
        raise AssertionError('expected KeyError for an ambiguous name')
    finally:
        AD._DATASETS.remove(dup)


def t_dataset_ids_covers_every_registered_dataset():
    ids = AD.dataset_ids()
    if len(ids) != len(AD._DATASETS) or len(set(ids)) != len(ids):
        raise AssertionError(f'dataset_ids() {ids} does not match the '
                             f'{len(AD._DATASETS)} registered datasets, or '
                             f'has duplicates')
    return f'dataset ids: {ids}'


# ── 3. list_names ────────────────────────────────────────────────────────────

def t_list_names_stain_filter_agrees_with_locate_for_every_name():
    ki67 = AD.list_names(stain='Ki67')
    mismatched = [n for n in ki67 if AD.locate(n).stain != 'Ki67']
    if mismatched:
        raise AssertionError(f"list_names(stain='Ki67') returned names "
                             f"locate() disagrees with: {mismatched}")
    return f'{len(ki67)} Ki67 names, each independently confirmed by locate()'


def t_list_names_n_caps_but_never_raises_past_the_end():
    total = len(AD.list_names())
    exact = AD.list_names(n=3)
    if len(exact) != min(3, total):
        raise AssertionError(f'n=3 returned {len(exact)}, expected '
                             f'{min(3, total)}')
    huge = AD.list_names(n=1_000_000)
    if len(huge) != total:
        raise AssertionError(f'n=1000000 returned {len(huge)}, expected '
                             f'every name ({total}) with no error')
    return f'n=3 -> {len(exact)}, n=1000000 -> {len(huge)} (capped at {total})'


# ── 4. disk (NOT portable -- this cluster's filesystem, on purpose) ─────────

def t_every_name_from_list_names_resolves_and_exists():
    names = AD.list_names()
    if not names:
        raise AssertionError('list_names() found nothing at all -- every '
                             'dataset root unreachable?')
    missing = []
    n_related = 0
    for name in names:
        entry = AD.locate(name)
        if not os.path.exists(entry.path):
            missing.append((name, 'path', entry.path))
        for role, path in entry.related.items():
            n_related += 1
            if not os.path.exists(path):
                missing.append((name, role, path))
    if missing:
        raise AssertionError(f'{len(missing)} missing: {missing[:5]}'
                             f'{" ..." if len(missing) > 5 else ""}')
    return f'{len(names)} names, {n_related} related files, all exist'


# ── 5. a recorded split as a dataset: `<id>#<split>` ─────────────────────────
#
# The split file is made up (real slide names, written to a temporary file) and
# `AD.split_file` is pointed at it, so nothing here reads or writes a real
# `result/cache/*_split/`.

@contextlib.contextmanager
def _split_file_of(rows, dataset='ki67_with_photo'):
    """Write `rows` (`(split, name)`) as a `wsi_split.csv` and make
    `AD.split_file` return it for `dataset` (any other id: a missing file)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'wsi_split.csv'
        with open(path, 'w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(('split', 'wsi_name'))
            writer.writerows(rows)
        real = AD.split_file
        AD.split_file = lambda d, job=None: (path if d == dataset
                                             else Path(tmp) / 'absent' / d)
        try:
            yield path
        finally:
            AD.split_file = real


def _ki67(k):
    names = AD.list_names(dataset='ki67_with_photo')
    if len(names) < k:
        raise AssertionError(f'ki67_with_photo has {len(names)} names, need {k}')
    return names[:k]


def _rows(val, test, train=()):
    return ([('val', n) for n in val] + [('test', n) for n in test]
            + [('train', n) for n in train])


def t_split_id_lists_the_recorded_names_in_file_order():
    names = _ki67(6)
    val, test = [names[4], names[0], names[2]], [names[5], names[1]]
    with _split_file_of(_rows(val, test)):
        assert AD.list_names(dataset='ki67_with_photo#val') == val
        assert AD.list_names(dataset='ki67_with_photo#test') == test
        assert AD.list_names(dataset='ki67_with_photo#val', n=2) == val[:2]
        assert AD.list_names(dataset='ki67_with_photo#val', n=99) == val
    return 'val and test, in file order, capped by n'


def t_split_id_locate_accepts_a_member_and_refuses_anything_else():
    names = _ki67(4)
    with _split_file_of(_rows(names[:2], names[2:])):
        got = AD.locate(names[0], dataset='ki67_with_photo#val')
        assert got.path == AD.locate(names[0]).path, 'a member resolves to its own path'
        for wrong in (names[2], names[3]):
            try:
                AD.locate(wrong, dataset='ki67_with_photo#val')
            except KeyError as exc:
                assert 'another split' in str(exc), exc
            else:
                raise AssertionError(f'{wrong!r} is a test slide; #val took it')
    return 'a member resolves, a test slide under #val is a KeyError'


def t_split_id_unknown_set_and_unknown_dataset_raise():
    names = _ki67(2)
    with _split_file_of(_rows(names[:1], names[1:])):
        for bad in ('ki67_with_photo#dev', 'ki67_with_photo#'):
            try:
                AD.list_names(dataset=bad)
            except KeyError:
                pass
            else:
                raise AssertionError(f'{bad!r} did not raise')
        try:
            AD.list_names(dataset='no_such_dataset#val')
        except KeyError:
            pass
        else:
            raise AssertionError('an unregistered base id did not raise')


def t_split_id_missing_file_names_the_writer():
    with _split_file_of([], dataset='some_other_dataset'):
        try:
            AD.list_names(dataset='ki67_with_photo#val')
        except FileNotFoundError as exc:
            assert 'make_split.py' in str(exc), exc
        else:
            raise AssertionError('a missing split file did not raise')


def t_split_id_absent_set_lists_what_the_file_holds():
    names = _ki67(3)
    with _split_file_of(_rows(names[:1], names[1:])):
        try:
            AD.list_names(dataset='ki67_with_photo#train')
        except KeyError as exc:
            assert "'val'" in str(exc) and "'test'" in str(exc), exc
        else:
            raise AssertionError('#train of a file with no train rows did not raise')


def t_split_id_train_rows_leave_val_and_test_as_they_were():
    names = _ki67(6)
    val, test, train = names[:2], names[2:4], names[4:]
    with _split_file_of(_rows(val, test)):
        old = (AD.list_names(dataset='ki67_with_photo#val'),
               AD.list_names(dataset='ki67_with_photo#test'))
    with _split_file_of(_rows(val, test, train)):
        new = (AD.list_names(dataset='ki67_with_photo#val'),
               AD.list_names(dataset='ki67_with_photo#test'))
        assert AD.list_names(dataset='ki67_with_photo#train') == train
    assert old == new == (val, test)


def t_split_job_without_a_hash_is_an_error_not_ignored():
    for call in (lambda: AD.list_names(dataset='ki67_with_photo', split_job='X'),
                 lambda: AD.list_names(split_job='X'),
                 lambda: AD.locate(_ki67(1)[0], dataset='ki67_with_photo',
                                   split_job='X')):
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError('split_job= with no #split was accepted')


def t_split_file_is_read_again_after_it_is_rewritten():
    names = _ki67(4)
    with _split_file_of(_rows(names[:1], names[1:])) as path:
        first = AD.list_names(dataset='ki67_with_photo#val')
        with open(path, 'w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(('split', 'wsi_name'))
            writer.writerows(_rows(names[:2], names[2:]))
        stamp = path.stat().st_mtime + 5
        os.utime(path, (stamp, stamp))
        second = AD.list_names(dataset='ki67_with_photo#val')
    assert first == names[:1] and second == names[:2], (first, second)


def t_split_id_stain_that_is_not_the_datasets_gives_nothing():
    names = _ki67(2)
    with _split_file_of(_rows(names[:1], names[1:])):
        assert AD.list_names(dataset='ki67_with_photo#val', stain='HE') == []
        assert AD.list_names(dataset='ki67_with_photo#val', stain='Ki67') == names[:1]


def t_wsi_split_reads_the_same_file_the_same_way():
    """The writer's reader (`WsiSplit.read_sets`) and the dataset id agree on one
    file -- and `WsiSplit.split_path` is `AD.split_file`, not a second copy of the
    path."""
    import WsiSplit                                            # noqa: PLC0415
    names = _ki67(6)
    val, test, train = names[:2], names[2:4], names[4:]
    with _split_file_of(_rows(val, test, train)) as path:
        assert WsiSplit.split_path('AnyJob', 'ki67_with_photo') == path
        sets = WsiSplit.read_sets(path)
        assert sets == {'val': val, 'test': test, 'train': train}, sets
        assert (AD.list_names(dataset='ki67_with_photo#val'),
                AD.list_names(dataset='ki67_with_photo#test'),
                AD.list_names(dataset='ki67_with_photo#train')) == \
            (sets['val'], sets['test'], sets['train'])
    assert WsiSplit.SPLIT_JOB == AD.SPLIT_JOB


# ── 6. pick_wsi_names ────────────────────────────────────────────────────────

def t_pick_wsi_names_is_seeded_ordered_and_capped():
    names = [f'w{i:03d}' for i in range(100)]
    a = AD.pick_wsi_names(names, 10, seed=42)
    assert len(a) == 10 and len(set(a)) == 10
    assert a == AD.pick_wsi_names(names, 10, seed=42), 'same seed, different slides'
    assert a != AD.pick_wsi_names(names, 10, seed=43), 'the seed does nothing'
    assert a == sorted(a, key=names.index), 'the order the names had is kept'
    assert set(a) <= set(names)


def t_pick_wsi_names_a_cap_of_all_changes_nothing():
    names = [f'w{i}' for i in range(20)]
    for cap in (20, 21, 999, None):
        assert AD.pick_wsi_names(names, cap, seed=1) == names, cap


def t_pick_wsi_names_is_the_one_datasets_py_uses():
    """`Datasets.py` used to define its own; the two must be the one function."""
    import training.MppRoutingHead.Datasets as D               # noqa: PLC0415
    assert D.pick_wsi_names is AD.pick_wsi_names


_SECTIONS = {
    'datasets': ['t_list_names_has_no_duplicates_across_datasets',
                't_stains_covers_every_dataset',
                't_dataset_ids_covers_every_registered_dataset'],
    'locate':   ['t_locate_returns_the_registered_entry',
                't_locate_unknown_name_raises_and_lists_known_names',
                't_locate_ki67_with_photo_finds_a_photos_companion',
                't_locate_ki67_pure_has_no_photos_companion',
                't_locate_with_dataset_kwarg_scopes_the_search',
                't_locate_group_type_hint_skips_the_glob_and_still_finds_the_slide',
                't_locate_wrong_group_type_hint_is_a_miss_not_a_crash',
                't_locate_kwargs_without_dataset_raises',
                't_locate_finds_a_bracs_train_slide',
                't_locate_unknown_dataset_id_raises',
                't_locate_raises_when_name_is_ambiguous_across_datasets'],
    'list':     ['t_list_names_stain_filter_agrees_with_locate_for_every_name',
                't_list_names_n_caps_but_never_raises_past_the_end'],
    'disk':     ['t_every_name_from_list_names_resolves_and_exists'],
    'split':    ['t_split_id_lists_the_recorded_names_in_file_order',
                't_split_id_locate_accepts_a_member_and_refuses_anything_else',
                't_split_id_unknown_set_and_unknown_dataset_raise',
                't_split_id_missing_file_names_the_writer',
                't_split_id_absent_set_lists_what_the_file_holds',
                't_split_id_train_rows_leave_val_and_test_as_they_were',
                't_split_job_without_a_hash_is_an_error_not_ignored',
                't_split_file_is_read_again_after_it_is_rewritten',
                't_split_id_stain_that_is_not_the_datasets_gives_nothing',
                't_wsi_split_reads_the_same_file_the_same_way'],
    'pick':     ['t_pick_wsi_names_is_seeded_ordered_and_capped',
                't_pick_wsi_names_a_cap_of_all_changes_nothing',
                't_pick_wsi_names_is_the_one_datasets_py_uses'],
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
