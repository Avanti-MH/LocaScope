"""Where a WSI actually is, and what else belongs with it -- by naming
convention, checked against disk, ONE RULE PER DATASET, not one entry per WSI.

    entry = locate('S1104233,G7E,110208')
    entry.path                          # the slide itself, absolute
    entry.related['photos']             # a companion dir, absolute (if any)

    locate(name, dataset='ki67_pure')   # only if `name` might be ambiguous
                                         # across registered datasets
    locate(name, dataset='bracs/test', group_type='Group_AT/Type_ADH')
                                         # dataset='...' + that dataset's own
                                         # narrowing hints (skips the search)
    list_names(stain='Ki67', n=5)       # first 5 currently-found Ki67 names
    stains()                            # every stain currently registered
    dataset_ids()                       # every dataset id `dataset=` accepts

ONE RULE PER DATASET, NOT ONE ENTRY PER WSI. Adding a new WSI to an already-
registered dataset (a new specimen dropped into `Ki67_pure/`, say) needs NO
change here: `locate(name)` derives the candidate path from `name` using
that dataset's own known naming convention and checks it against disk --
a name that resolves to a real file is found, one that does not is a
`KeyError`. Only a genuinely NEW dataset (a new root, a layout none of the
`_locate_*` functions below already knows) needs a new rule written by hand,
once, here -- not once per WSI it contains.

Checking a name against a known, hand-written template resolves the ambiguity
a blind directory scan would have (`Ki67_with_photo` spells one companion role
two ways, `{specimen}_ki67/` and a bare `{specimen}/`) once, in the template
itself (`_locate_ki67`'s docstring), not per file.

THE REGISTRY MOVES WHEN THE DATA MOVES, AND THAT IS THE ONE THING THIS FILE
EXISTS TO CENTRALISE. A caller that hardcodes a dataset path breaks the day the
dataset is renamed; one that calls `AccessDatasets.locate(...)` does not.

`locate()` RAISES ON AN UNKNOWN NAME, NOT ON A CLOSE GUESS. A caller who
mistypes a name and gets back some OTHER slide's path silently analyses the
wrong tissue; the error lists every name each registered dataset currently
holds, so the typo is obvious.

A RECORDED SPLIT IS A DATASET TOO. `<dataset>#<split>` -- `bracs/test#val`,
`ki67_with_photo#test` -- names the slides one recorded split of a dataset holds,
and `list_names()` / `locate()` take it wherever they take a dataset id:

    list_names(dataset='bracs/test#val')              # the val slides
    locate(name, dataset='bracs/test#val')            # KeyError if it is not in val
    list_names(dataset='bracs/test#val', split_job='OtherJob')

The split is READ from `<split_job>`'s `dataset=<dataset>/split/split_recorded.csv`
(`split,wsi_name`; `split_job` defaults to `SPLIT_JOB`), never written here:
`utilities/cli/build_cache/make_split.py` is the one writer, and a missing file
is a refusal that names it. `dataset_ids()` lists the real datasets only.
"""

from __future__ import annotations

import csv
import glob
import os
import random
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from Cache import Address, Entry, dataset_key
from _paths import DATASETS_DIR

_BRACS_TEST_ROOT = os.path.join(DATASETS_DIR, 'histoimage.na.icar.cnr.it/BRACS_WSI/test')
_BRACS_TRAIN_ROOT = os.path.join(DATASETS_DIR, 'histoimage.na.icar.cnr.it/BRACS_WSI/train')
_KI67_ROOT = os.path.join(DATASETS_DIR, 'Ki67_with_photo')
_KI67_PURE_ROOT = os.path.join(DATASETS_DIR, 'Ki67_pure')


@dataclass(frozen=True)
class WsiEntry:
    """One resolved WSI. `related` is keyed by ROLE, not by filename, so a
    caller asks for `entry.related['photos']` rather than parsing a path to
    figure out which sibling is which.
    """
    name: str
    stain: str
    path: str
    related: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class _Dataset:
    """One registered dataset: a naming convention, not a table of files.

    `id` is the disambiguator `locate(name, dataset=...)` picks by -- NOT
    the same axis as `stain`, on purpose: `Ki67_with_photo` and `Ki67_pure`
    share a stain but are two different datasets a caller may need to tell
    apart (`Ki67_pure` never has a `photos` companion; a name that happens
    to exist in both would otherwise be ambiguous by stain alone).

    `locate_fn(name, **kwargs)` returns the `WsiEntry` if `name` resolves to
    a real file under this dataset's own layout, else `None` -- checked
    against disk, not guessed. `**kwargs` are THIS dataset's own optional
    narrowing hints (e.g. `_locate_bracs`'s `group_type`) -- each
    `_locate_*` factory below declares whichever ones its own layout
    supports; passing one a dataset does not accept raises `TypeError`
    exactly as calling any Python function with an unknown keyword does,
    which is the right error (module docstring: no fuzzy anything here).
    `list_fn()` enumerates every name CURRENTLY found (a fresh directory
    read, not a cached table -- `list_names`'s own docstring).
    """
    id: str
    stain: str
    locate_fn: Callable[..., Optional[WsiEntry]]
    list_fn: Callable[[], List[str]]


def _mrxs_name_in(container: str) -> Optional[str]:
    """The full identifier a `{...}_mrxs/` container actually holds --
    read off the `.mrxs` FILENAME inside, the one thing in there that is
    never ambiguous (`organize_mrxs.py`'s own `plan_renames` reads the same
    thing for the same reason). `None` if the container does not hold
    exactly one `.mrxs` file -- a container that is mid-transfer or
    malformed is not a name this file will guess at.
    """
    try:
        entries = os.listdir(container)
    except OSError:
        return None
    mrxs_files = [f for f in entries if f.endswith('.mrxs')]
    if len(mrxs_files) != 1:
        return None
    return mrxs_files[0][:-len('.mrxs')]


def _locate_ki67(dataset_id: str, root: str, stain: str) -> _Dataset:
    """Ki67-family datasets (`Ki67_with_photo`, `Ki67_pure`): a WSI `name`
    (e.g. `S1104233,G7E,110208`) lives at
    `{root}/{name with ',' -> '_'}_mrxs/{name}.mrxs`
    (`organize_mrxs.py`'s own container-naming convention), with a
    same-named pyramid directory (`{name}/`, no `.mrxs` suffix) next to it.

    Whatever THIRD entry -- if any -- sits in the container besides those
    two known names is the `photos` companion: `Ki67_with_photo` has one
    (either a `{specimen}_ki67` or a bare `{specimen}` spelling; both
    resolve the same way here, as "whichever name is left over" -- the
    ambiguity between the two spellings never has to be guessed because
    only one of them can be sitting in any given container). `Ki67_pure`
    never has a third entry, so `related` never gets a `'photos'` key for
    it -- matching what "pure" names.
    """
    def _locate(name: str) -> Optional[WsiEntry]:
        container = os.path.join(root, name.replace(',', '_') + '_mrxs')
        mrxs_path = os.path.join(container, name + '.mrxs')
        if not os.path.isfile(mrxs_path):
            return None
        related: Dict[str, str] = {}
        pyramid_dir = os.path.join(container, name)
        if os.path.isdir(pyramid_dir):
            related['pyramid_dir'] = pyramid_dir
        leftovers = [e for e in os.listdir(container)
                    if e not in (name + '.mrxs', name)]
        if leftovers:
            related['photos'] = os.path.join(container, leftovers[0])
        return WsiEntry(name=name, stain=stain, path=mrxs_path,
                        related=related)

    def _list() -> List[str]:
        if not os.path.isdir(root):
            return []
        names = []
        for entry in sorted(os.listdir(root)):
            if not entry.endswith('_mrxs'):
                continue
            container = os.path.join(root, entry)
            if not os.path.isdir(container):
                continue
            name = _mrxs_name_in(container)
            if name is not None:
                names.append(name)
        return names

    return _Dataset(id=dataset_id, stain=stain, locate_fn=_locate, list_fn=_list)


def _locate_bracs(dataset_id: str, root: str, stain: str) -> _Dataset:
    """BRACS: `{stem}.svs` lives under `{root}/<Group>/<Type>/`, and
    `<Group>/<Type>` is not derivable from `stem` alone -- but `stem` is
    unique across the whole split, so a glob for it is a LOOKUP BY NAME, not
    a guess at structure: it either finds exactly one file or none, and
    finding MORE than one is treated as an error (stems were assumed
    unique; if that assumption ever breaks, silently picking one would
    analyse the wrong tissue) rather than silently picked.

    `group_type`, if given (e.g. `'Group_AT/Type_ADH'`), IS `<Group>/<Type>`
    -- skips the glob entirely and checks that one exact path, for a caller
    who already knows where its own slide sits and wants to avoid searching
    the whole split for it. Wrong or nonexistent -- unlike a plain `name`
    that is not in this dataset at all -- still just resolves to `None`,
    same as any other miss; it is a narrowing hint, not a second identity
    the name has to also match.
    """
    def _locate(name: str, *, group_type: Optional[str] = None
               ) -> Optional[WsiEntry]:
        if group_type is not None:
            path = os.path.join(root, group_type, name + '.svs')
            return (WsiEntry(name=name, stain=stain, path=path)
                   if os.path.isfile(path) else None)
        matches = glob.glob(os.path.join(root, '*', '*', name + '.svs'))
        if not matches:
            return None
        if len(matches) > 1:
            raise ValueError(
                f'{name!r} matches more than one BRACS path: {matches} -- '
                f'stems were assumed unique across the dataset')
        return WsiEntry(name=name, stain=stain, path=matches[0])

    def _list() -> List[str]:
        paths = glob.glob(os.path.join(root, '*', '*', '*.svs'))
        return sorted(os.path.basename(p)[:-len('.svs')] for p in paths)

    return _Dataset(id=dataset_id, stain=stain, locate_fn=_locate, list_fn=_list)


#: THE REGISTRY. One entry per DATASET (a root + naming convention), not one
#: per WSI -- add an entry by hand only when a genuinely new layout shows up;
#: a new WSI inside an already-registered dataset needs nothing here.
#:
#: `id` SPELLING: `parent/child` only when `child` really is a subdirectory
#: of a larger dataset (`bracs/test` -- `BRACS_WSI/test/`, `bracs/train` --
#: `BRACS_WSI/train/`); a flat `name` when it is its own independent root
#: with no such parent (`Ki67_pure` and `Ki67_with_photo` are siblings on
#: disk, not one inside the other, so `ki67_pure`/`ki67_with_photo`, not
#: `ki67/pure`).
_DATASETS: List[_Dataset] = [
    _locate_bracs('bracs/test', _BRACS_TEST_ROOT, 'HE'),
    _locate_bracs('bracs/train', _BRACS_TRAIN_ROOT, 'HE'),
    _locate_ki67('ki67_with_photo', _KI67_ROOT, 'Ki67'),
    _locate_ki67('ki67_pure', _KI67_PURE_ROOT, 'Ki67'),
]


#: The job whose recorded split `<dataset>#<split>` reads by default. The one
#: writer, `make_split.py`, writes under this name unless told otherwise, and
#: `WsiSplit.SPLIT_JOB` is this same value.
SPLIT_JOB = 'MakeSplit'

#: `<dataset>#<split>`: the separator, and the sets a split can hold.
SPLIT_SEP = '#'
SPLITS = ('train', 'val', 'test')


#: The one variant of a dataset's `split` entry. A split is a record, never
#: re-derived from a config, so it has no config id to be named by.
SPLIT_ID = 'recorded'


def split_entry(dataset_id: str, split_job: Optional[str] = None) -> Entry:
    """The `split` entry at `dataset=<dataset>/` in `split_job`'s cache."""
    return Address(split_job or SPLIT_JOB,
                   dataset=dataset_key(dataset_id)).entry('split')


def split_file(dataset_id: str, split_job: Optional[str] = None) -> Path:
    """`dataset=<dataset>/split/split_recorded.csv`: where one dataset's
    recorded split lives. The only place that path is spelled."""
    return split_entry(dataset_id, split_job).path('split', SPLIT_ID, '.csv')


def parse_dataset_id(dataset_id: str) -> Tuple[str, Optional[str]]:
    """`'bracs/test#val'` -> `('bracs/test', 'val')`; `'bracs/test'` ->
    `('bracs/test', None)`. A set that is not train / val / test is an error
    here, not an empty list a caller could take for an empty set."""
    base, sep, split = dataset_id.partition(SPLIT_SEP)
    if not sep:
        return dataset_id, None
    if split not in SPLITS:
        raise KeyError(f'{dataset_id!r}: {split!r} is not a split set; after '
                       f'{SPLIT_SEP!r} put one of {list(SPLITS)}')
    return base, split


@lru_cache(maxsize=None)
def _read_split(path: str, mtime_ns: int) -> Dict[str, Tuple[str, ...]]:
    """`{set: names}` in file order. Keyed on the modification time as well, so
    a rewritten file is read again and an unchanged one is read once."""
    out: Dict[str, List[str]] = {}
    with open(path, newline='') as handle:
        for row in csv.DictReader(handle):
            out.setdefault(row['split'], []).append(row['wsi_name'])
    return {k: tuple(v) for k, v in out.items()}


def _split_names(base: str, split: str, split_job: Optional[str]) -> List[str]:
    path = split_file(base, split_job)
    if not path.exists():
        raise FileNotFoundError(
            f'{path} does not exist. The split of {base!r} is written once, by '
            f'utilities/cli/build_cache/make_split.py -- run it first, so every '
            f'package scores on the same held-out slides')
    by_split = _read_split(str(path), path.stat().st_mtime_ns)
    if split not in by_split:
        raise KeyError(f'{base}{SPLIT_SEP}{split}: the split recorded at {path} '
                       f'holds {sorted(by_split)} only')
    return list(by_split[split])


def pick_wsi_names(names: Sequence[str], max_wsi: Optional[int], seed: int
                   ) -> List[str]:
    '''`max_wsi` of `names`, chosen AT RANDOM (seeded), returned in the order
    `names` had. `None`, or a cap at or above `len(names)`, returns every name
    unchanged -- so a cap of "all of them" changes nothing, not even the order.

    Random and not the first N: a dataset's names are sorted, and the first N
    of a sorted list are one contiguous run of ids -- for BRACS one stretch of
    slides whose stain and pathology mix nobody has checked. The same
    (names, cap, seed) always picks the same slides, so a rerun, the
    supply-only pass and the real run agree.
    '''
    names = list(names)
    if max_wsi is None or max_wsi >= len(names):
        return names
    keep = set(random.Random(seed).sample(range(len(names)), int(max_wsi)))
    return [n for i, n in enumerate(names) if i in keep]


def locate(name: str, *, dataset: Optional[str] = None,
           split_job: Optional[str] = None, **kwargs) -> WsiEntry:
    """The resolved entry for `name`. Raises `KeyError` naming every name
    each registered dataset currently holds if `name` resolves in NONE of
    them -- there is no fuzzy match, because a wrong guess here means every
    downstream read is silently the wrong tissue.

    THE SAME PRINCIPLE APPLIES WHEN `name` RESOLVES IN MORE THAN ONE
    DATASET: `locate` does not silently prefer whichever is registered
    first -- it raises, naming which dataset `id`s it found `name` in, and
    the caller passes `dataset=<one of those ids>` to pick. `dataset_ids()`
    lists every registered id; `list_names()`/`locate()` without `dataset=`
    are still the right call for the common case where a name is unique
    across every registered dataset, which is true of everything currently
    registered (`test_access_datasets.py`'s
    `t_list_names_has_no_duplicates_across_datasets` checks it stays true).

    `**kwargs` ONLY MAKE SENSE WITH `dataset=` GIVEN TOO -- they are that
    ONE dataset's own optional narrowing hints (`_locate_bracs`'s
    `group_type`, say), meaningless without knowing which dataset's
    convention they belong to; passing any without `dataset=` raises rather
    than being silently ignored or guessed at.

    `dataset` may be a recorded split, `<id>#<split>`: the slide is resolved by
    the real dataset and must be in that split, else `KeyError` -- a test slide
    asked for under `#val` is a mistake, not a path to hand back. `split_job`
    picks whose split file is read (default `SPLIT_JOB`); giving it without a
    `#` in `dataset` is an error, not something silently ignored.
    """
    if split_job is not None and (dataset is None or SPLIT_SEP not in dataset):
        raise ValueError('split_job= only means something with dataset='
                         '<id>#<split>')
    if kwargs and dataset is None:
        raise ValueError(
            f'kwargs {list(kwargs)} were given without dataset= -- they are '
            f'one specific dataset\'s own narrowing hints and are meaningless '
            f'without knowing which dataset they belong to')

    if dataset is not None:
        base, split = parse_dataset_id(dataset)
        chosen = [d for d in _DATASETS if d.id == base]
        if not chosen:
            raise KeyError(f'{dataset!r} is not a registered dataset id. '
                           f'Known dataset ids: {dataset_ids()}')
        if split is not None and name not in _split_names(base, split, split_job):
            raise KeyError(f'{name!r} is not in {dataset!r}; it is either in '
                           f'another split of {base!r} or not in {base!r}')
        entry = chosen[0].locate_fn(name, **kwargs)
        if entry is None:
            raise KeyError(f'{name!r} is not found in dataset {dataset!r}.')
        return entry

    matches = [(d.id, d.locate_fn(name)) for d in _DATASETS]
    matches = [(dataset_id, entry) for dataset_id, entry in matches
              if entry is not None]
    if not matches:
        raise KeyError(f'{name!r} is not found in any registered dataset. '
                       f'Known names: {sorted(list_names())}')
    if len(matches) > 1:
        ids = [dataset_id for dataset_id, _ in matches]
        raise KeyError(f'{name!r} exists in more than one dataset: {ids} '
                       f'-- pass dataset=<one of these> to locate() to pick')
    return matches[0][1]


def dataset_ids() -> List[str]:
    """Every registered dataset `id`, in registration order -- what
    `locate(name, dataset=...)` picks by. The real datasets only: a recorded
    split is `<id>#<split>` and is not listed.
    """
    return [d.id for d in _DATASETS]


def list_names(*, stain: Optional[str] = None, dataset: Optional[str] = None,
              n: Optional[int] = None, split_job: Optional[str] = None
              ) -> List[str]:
    """Every name currently found across every registered dataset,
    optionally narrowed to one `stain` and/or one `dataset` id, optionally
    capped at the first `n` (dataset registration order, then each dataset's
    own sorted order -- not globally sorted, so this stays stable as a
    dataset gains names).

    `dataset=` exists for the case `stain=` cannot resolve: `Ki67_with_photo`
    and `Ki67_pure` share a stain (both `'Ki67'`) but are two different
    datasets a caller may need enumerated separately -- e.g. a training split
    that draws every WSI in `Ki67_pure` and nothing from its `photos`-bearing
    sibling. Raises on an unregistered id, same as `locate(name,
    dataset=...)`, rather than silently returning an empty list a caller
    could mistake for "this dataset really is empty".

    `dataset='<id>#<split>'` gives the slides that recorded split holds, in the
    order the file has them (`split_job=` picks whose file, default
    `SPLIT_JOB`); a stain that is not that dataset's gives an empty list.

    A FRESH SCAN EVERY CALL, ON PURPOSE: this is the one function whose
    whole job is "what is actually there right now", not a lookup in a
    static table -- a name this returns is guaranteed findable by `locate`
    at the same moment (modulo a concurrent transfer).
    """
    if split_job is not None and (dataset is None or SPLIT_SEP not in dataset):
        raise ValueError('split_job= only means something with dataset='
                         '<id>#<split>')
    if dataset is not None:
        base, split = parse_dataset_id(dataset)
        if base not in dataset_ids():
            raise KeyError(f'{dataset!r} is not a registered dataset id. '
                           f'Known dataset ids: {dataset_ids()}')
        if split is not None:
            entry = next(d for d in _DATASETS if d.id == base)
            names = ([] if stain is not None and entry.stain != stain
                     else _split_names(base, split, split_job))
            return names if n is None else names[:n]
    names: List[str] = []
    for entry in _DATASETS:
        if stain is not None and entry.stain != stain:
            continue
        if dataset is not None and entry.id != dataset:
            continue
        names.extend(entry.list_fn())
    return names if n is None else names[:n]


def stains() -> List[str]:
    """Every distinct stain currently registered, in registration order."""
    seen: List[str] = []
    for dataset in _DATASETS:
        if dataset.stain not in seen:
            seen.append(dataset.stain)
    return seen
