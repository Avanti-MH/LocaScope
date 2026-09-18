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

An earlier version of this file registered one `WsiEntry` per WSI by hand,
reasoning that "a directory listing cannot tell a companion from an
unrelated thing that happens to sit next to it" (true of `Ki67_with_photo`'s
two spellings for the same role, `{specimen}_ki67/` and a bare
`{specimen}/`). That is true of a BLIND scan guessing at structure it has
never been told; it is not true of checking a name against a known,
hand-written template, which is what every `_locate_*` function below does
-- the ambiguity is resolved once, in the template itself (`_locate_ki67`'s
docstring), not re-guessed per file.

THE REGISTRY MOVES WHEN THE DATA MOVES, AND THAT IS THE ONE THING THIS FILE
EXISTS TO CENTRALISE. `/work/u26130998/datasets/Ki67` was renamed to
`Ki67_with_photo` on 2026-09-06 (after `organize_mrxs.py` folded every
slide's scattered pieces into one `{specimen}_mrxs/` directory each) and the
rename broke 23 files that had the old root hardcoded. Every one of them
should have been reading `AccessDatasets.locate(...)` instead -- this file
is that single place; callers that still hardcode a dataset path are exactly
the next version of that same rename cost.

`locate()` RAISES ON AN UNKNOWN NAME, NOT ON A CLOSE GUESS. A caller who
mistypes a name and gets back some OTHER slide's path silently analyses the
wrong tissue; the error lists every name each registered dataset currently
holds, so the typo is obvious.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

_BRACS_TEST_ROOT = '/work/u26130998/datasets/histoimage.na.icar.cnr.it/BRACS_WSI/test'
_BRACS_TRAIN_ROOT = '/work/u26130998/datasets/histoimage.na.icar.cnr.it/BRACS_WSI/train'
_KI67_ROOT = '/work/u26130998/datasets/Ki67_with_photo'
_KI67_PURE_ROOT = '/work/u26130998/datasets/Ki67_pure'


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


def locate(name: str, *, dataset: Optional[str] = None, **kwargs) -> WsiEntry:
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
    """
    if kwargs and dataset is None:
        raise ValueError(
            f'kwargs {list(kwargs)} were given without dataset= -- they are '
            f'one specific dataset\'s own narrowing hints and are meaningless '
            f'without knowing which dataset they belong to')

    if dataset is not None:
        chosen = [d for d in _DATASETS if d.id == dataset]
        if not chosen:
            raise KeyError(f'{dataset!r} is not a registered dataset id. '
                           f'Known dataset ids: {dataset_ids()}')
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
    `locate(name, dataset=...)` picks by.
    """
    return [d.id for d in _DATASETS]


def list_names(*, stain: Optional[str] = None, dataset: Optional[str] = None,
              n: Optional[int] = None) -> List[str]:
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

    A FRESH SCAN EVERY CALL, ON PURPOSE: this is the one function whose
    whole job is "what is actually there right now", not a lookup in a
    static table -- a name this returns is guaranteed findable by `locate`
    at the same moment (modulo a concurrent transfer).
    """
    if dataset is not None and dataset not in dataset_ids():
        raise KeyError(f'{dataset!r} is not a registered dataset id. '
                       f'Known dataset ids: {dataset_ids()}')
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
