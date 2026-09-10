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
import os
import sys

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
