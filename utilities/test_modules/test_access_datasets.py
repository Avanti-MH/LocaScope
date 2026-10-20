#!/usr/bin/env python3
"""Tests for `utilities/AccessDatasets.py`.

    python utilities/test_modules/test_access_datasets.py

`AccessDatasets.py` is used by more than one package (SuperPathPoint's cli/
today, potentially others later), which is why this test sits in the flat
`test_modules/` directory rather than under any one package's `Test<Package>/`
subdirectory -- filing it under one consumer's jobscript would say something
false about who it belongs to (CLAUDE.md's own rule for this).

`AccessDatasets.py` holds one naming-convention rule per DATASET, so
`list_names()`/`locate()` are directory reads, not table lookups, and almost
nothing here is pure. What separates the sections is whether a test needs a
SPECIFIC real name to exist (`disk`, not portable) or just needs
`list_names()` to return something self-consistent, whatever it currently
holds (runs anywhere the roots are reachable, but says nothing if they are
empty).
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
    """`bracs/train` (`BRACS_WSI/train/`) -- registered the same
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


# ── 5. pick_wsi_names ────────────────────────────────────────────────────────

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
    """`Datasets.py` and AccessDatasets share one function."""
    import training.ClassifyingRoutingHead.Datasets as D               # noqa: PLC0415
    assert D.pick_wsi_names is AD.pick_wsi_names


# ── 6. virtual datasets: the catalog, selections, `@<name>` ─────────────────

from typing import Any, Dict, Iterator, List, Optional, Tuple       # noqa: E402

import Cache                                                         # noqa: E402


@contextlib.contextmanager
def _own_cache() -> Iterator[Path]:
    """AccessDatasets' cache in a temporary directory, for the length of a test."""
    old: str = Cache.RESULT_DIR
    with tempfile.TemporaryDirectory() as tmp:
        Cache.RESULT_DIR = tmp
        try:
            yield Path(tmp)
        finally:
            Cache.RESULT_DIR = old


class _Header:
    """What the catalog reads of a slide, made up from its file name: base mpp 0.25 or 0.5 by its characters'
    sum, a 4x pyramid of three levels. One name (`_BROKEN`) cannot be opened."""

    def __init__(self, path: str) -> None:
        if os.path.splitext(os.path.basename(path))[0] == _BROKEN[0]:
            raise OSError('unreadable header')
        self.base_mpp: float = 0.25 if sum(map(ord, os.path.basename(path))) % 2 else 0.5
        self.level_downsamples: Tuple[float, ...] = (1.0, 4.0, 16.0)
        self.level_dimensions: Tuple[Tuple[int, int], ...] = ((4096, 3072), (1024, 768), (256, 192))

    def close(self) -> None:
        pass


#: the slide `_Header` refuses to open (filled with a real name by the test that needs one)
_BROKEN: List[str] = ['\0']


def _photo_names() -> List[str]:
    names: List[str] = AD.list_names(dataset='ki67_with_photo')
    if len(names) < 8:
        raise AssertionError(f'ki67_with_photo has {len(names)} names, need 8')
    return names


def t_virtual_ids_are_an_at_and_a_plain_name() -> str:
    assert AD.is_virtual('@val') and not AD.is_virtual('bracs/test') and not AD.is_virtual('bracs/test#val')
    assert AD._virtual_name('@test63') == 'test63' and AD._virtual_name('@Ki-HE_2') == 'Ki-HE_2'
    for bad in ('val', '@', '@a/b', '@#x', '@a b', '@_x', '@x.y'):
        try:
            AD._virtual_name(bad)
        except ValueError:
            continue
        raise AssertionError(f'{bad!r} was taken for a virtual dataset')
    return '@<letters, digits, _ and ->'


def t_a_selection_combines_keeping_the_left_ones_order() -> str:
    names: List[str] = _photo_names()
    a: AD.Selection = AD.Selection([('ki67_with_photo', n) for n in names[:4]], 'a')
    b: AD.Selection = AD.Selection([('ki67_with_photo', n) for n in names[2:6]], 'b')
    assert (a | b).names() == names[:6] and (b | a).names() == names[2:6] + names[:2]
    assert (a & b).names() == names[2:4] and (a - b).names() == names[:2]
    assert len(AD.Selection(list(a) + list(a), 'twice')) == 4, 'a slide is in a selection once'
    assert (a - b).how == '(a) - (b)'
    return 'union, intersection, difference; each slide once'


def t_a_split_is_disjoint_complete_and_seeded() -> str:
    pool: AD.Selection = AD.select(dataset='ki67_with_photo')
    n: int = len(pool)
    parts: Tuple[AD.Selection, ...] = pool.split(3, 0.25, 'rest', seed=1)
    assert [len(p) for p in parts] == [3, int(0.25 * n + 0.5), n - 3 - int(0.25 * n + 0.5)], [len(p) for p in parts]
    assert set().union(*(set(p) for p in parts)) == set(pool), 'every slide is in a part'
    assert all(not (set(p) & set(q)) for i, p in enumerate(parts) for q in parts[i + 1:]), 'no slide in two'
    again: Tuple[AD.Selection, ...] = pool.split(3, 0.25, 'rest', seed=1)
    assert [p.members for p in again] == [p.members for p in parts], 'the same seed, the same parts'
    assert pool.split(3, 0.25, 'rest', seed=2)[2].members != parts[2].members, 'another seed, another order'
    assert pool.sample(5, seed=1).members == pool.split(5, seed=1)[0].members
    assert len(pool.split(3, seed=1)) == 1, 'no rest: the slides left over are in no part'
    for sizes in ((n + 1,), ('rest', 'rest'), (0,), (1.5,), (n, 'rest'), (True,)):
        try:
            pool.split(*sizes, seed=0)
        except ValueError:
            continue
        raise AssertionError(f'split{sizes} of {n} was accepted')
    return f'{[len(p) for p in parts]} of {n}; seeded; refused: too many, two rests, 0, 1.5, an empty rest'


def t_select_narrows_by_dataset_and_stain() -> str:
    both: AD.Selection = AD.select(dataset=['ki67_with_photo', 'ki67_pure'])
    assert both.names() == AD.list_names(dataset='ki67_with_photo') + AD.list_names(dataset='ki67_pure')
    assert {d for d, _ in both} == {'ki67_with_photo', 'ki67_pure'}
    assert not len(AD.select(dataset=['ki67_with_photo', 'ki67_pure'], stain='HE'))
    he: AD.Selection = AD.select(stain='HE')
    assert {d for d, _ in he} <= {'bracs/test', 'bracs/train'} and len(he)
    try:
        AD.select(dataset='ki67_with_photo#val')
        raise AssertionError('an id that is no dataset was drawn from')
    except KeyError:
        pass
    return f'{len(both)} Ki67, {len(he)} HE; an unknown id refused'


def t_the_catalog_is_read_once_per_slide_and_read_back_without_opening() -> str:
    names: List[str] = _photo_names()
    opened: List[str] = []

    def opener(path: str) -> _Header:
        opened.append(path)
        return _Header(path)

    _BROKEN[0] = names[3]
    try:
        with _own_cache():
            try:
                AD.catalog(['ki67_with_photo'])
                raise AssertionError('a dataset with no catalog was read')
            except FileNotFoundError as e:
                assert 'make_dataset.py --catalog' in str(e), e
            AD.build_catalog(['ki67_with_photo'], opener=opener, log=lambda s: None)
            assert len(opened) == len(names)
            AD.build_catalog(['ki67_with_photo'], opener=opener, log=lambda s: None)
            assert len(opened) == len(names) + 1, 'only the unreadable slide is opened again'
            rows: List[AD.SlideInfo] = AD.catalog(['ki67_with_photo'])
            assert [r.name for r in rows] == names and all(r.stain == 'Ki67' for r in rows)
            broken: AD.SlideInfo = next(r for r in rows if r.name == names[3])
            assert broken.error and broken.base_mpp is None and not broken.native_at((1.0,))
            good: AD.SlideInfo = rows[0]
            assert good.level_downsamples == (1.0, 4.0, 16.0) and good.dimensions == (4096, 3072)
            assert good.n_levels == 3 and good.native_at((1.0, 4.0, 16.0)) and not good.native_at((1.0, 2.0))
            assert good.has_photos, 'Ki67_with_photo slides have their photos'
    finally:
        _BROKEN[0] = '\0'
    return f'{len(names)} headers once, the unreadable one again; levels, mpp, native, photos read back'


def t_where_narrows_by_the_header() -> str:
    names: List[str] = _photo_names()
    with _own_cache():
        AD.build_catalog(['ki67_with_photo'], opener=_Header, log=lambda s: None)
        fine: AD.Selection = AD.select(dataset='ki67_with_photo', where=lambda s: s.base_mpp == 0.25,
                                       label='base_mpp 0.25')
        want: List[str] = [n for n in names
                           if _Header(AD.locate(n, dataset='ki67_with_photo').path).base_mpp == 0.25]
        assert fine.names() == want and want, (fine.names(), want)
        assert fine.how.endswith('where base_mpp 0.25'), fine.how
        try:
            AD.select(dataset='ki67_pure', where=lambda s: True)
            raise AssertionError('a dataset with no catalog was narrowed')
        except FileNotFoundError:
            pass
    return f'{len(want)} of {len(names)}; a dataset with no catalog refused'


def t_a_virtual_dataset_is_read_and_located_like_a_dataset() -> str:
    names: List[str] = _photo_names()
    chosen: List[str] = [names[5], names[0], names[3]]
    with _own_cache():
        path: Path = AD.save_virtual('@t1', AD.Selection([('ki67_with_photo', n) for n in chosen], 'by hand'),
                                     recipe='test')
        assert path == AD.vdataset_file('@t1') and path.name == 'list_t1.csv'
        assert AD.list_names(dataset='@t1') == chosen and AD.list_names(dataset='@t1', n=2) == chosen[:2]
        assert AD.list_names(dataset='@t1', stain='HE') == [] and AD.list_names(dataset='@t1', stain='Ki67') == chosen
        assert AD.locate(chosen[1], dataset='@t1') == AD.locate(chosen[1], dataset='ki67_with_photo')
        try:
            AD.locate(names[1], dataset='@t1')
            raise AssertionError('a slide not in @t1 was located in it')
        except KeyError:
            pass
        assert AD.virtual_ids() == ['@t1'] and '@t1' not in AD.dataset_ids()
        rec: Dict[str, Any] = AD.virtual_record('@t1')
        assert (rec['how'], rec['recipe'], rec['n'], rec['sources']) == ('by hand', 'test', 3, ['ki67_with_photo'])
        assert AD.select(dataset='@t1').names() == chosen, 'a virtual dataset is a source of a selection too'
    return 'list_names, locate, stain, select: in its order'


def t_a_virtual_dataset_is_written_once_and_a_bad_list_is_refused() -> str:
    names: List[str] = _photo_names()
    one: AD.Selection = AD.Selection([('ki67_with_photo', names[0])], 'one')
    with _own_cache():
        AD.save_virtual('@once', one)
        refused: List[str] = []
        for vid, sel, err in (('@once', one, FileExistsError),
                              ('@empty', AD.Selection([], 'none'), ValueError),
                              ('@gone', AD.Selection([('ki67_with_photo', 'no-such-slide')], 'gone'),
                               FileNotFoundError),
                              ('@two', AD.Selection([('ki67_with_photo', names[0]), ('ki67_pure', names[0])],
                                                    'two'), ValueError),
                              ('plain', one, ValueError)):
            try:
                AD.save_virtual(vid, sel)
            except err:
                refused.append(vid)
                continue
            raise AssertionError(f'{vid} was written')
        assert AD.virtual_ids() == ['@once'], 'nothing refused left a trace'
        try:
            AD.list_names(dataset='@never')
            raise AssertionError('an unwritten virtual dataset was read')
        except KeyError as e:
            assert '@once' in str(e), 'the error names what is written'
    return f'refused: {refused}'


def t_a_virtual_dataset_with_a_slide_gone_is_an_error_not_a_shorter_list() -> str:
    names: List[str] = _photo_names()
    with _own_cache():
        AD.save_virtual('@t2', AD.Selection([('ki67_with_photo', n) for n in names[:2]], 'two'))
        with open(AD.vdataset_file('@t2'), 'a', newline='') as handle:
            csv.writer(handle).writerow(('ki67_with_photo', 'moved-away'))
        try:
            AD.list_names(dataset='@t2')
            raise AssertionError('a list with a slide gone was read')
        except FileNotFoundError as e:
            assert 'moved-away' in str(e), e
    return 'FileNotFoundError naming the slide'


def t_the_cache_is_access_datasets_own_whatever_the_job() -> str:
    old: Optional[str] = os.environ.get('SLURM_JOB_NAME')
    os.environ['SLURM_JOB_NAME'] = 'SomeOtherJob'
    try:
        assert AD.vdataset_entry().address.made_by == AD.DATASETS_JOB
        assert AD.catalog_entry().address.made_by == AD.DATASETS_JOB
    finally:
        if old is None:
            os.environ.pop('SLURM_JOB_NAME', None)
        else:
            os.environ['SLURM_JOB_NAME'] = old
    return f'result/cache/{AD.DATASETS_JOB}/ under any job name'


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
    'pick':     ['t_pick_wsi_names_is_seeded_ordered_and_capped',
                't_pick_wsi_names_a_cap_of_all_changes_nothing',
                't_pick_wsi_names_is_the_one_datasets_py_uses'],
    'virtual':  ['t_virtual_ids_are_an_at_and_a_plain_name',
                't_a_selection_combines_keeping_the_left_ones_order',
                't_a_split_is_disjoint_complete_and_seeded',
                't_select_narrows_by_dataset_and_stain',
                't_the_catalog_is_read_once_per_slide_and_read_back_without_opening',
                't_where_narrows_by_the_header',
                't_a_virtual_dataset_is_read_and_located_like_a_dataset',
                't_a_virtual_dataset_is_written_once_and_a_bad_list_is_refused',
                't_a_virtual_dataset_with_a_slide_gone_is_an_error_not_a_shorter_list',
                't_the_cache_is_access_datasets_own_whatever_the_job'],
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
