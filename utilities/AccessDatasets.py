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

A VIRTUAL DATASET IS A NAMED LIST OF SLIDES, `@<name>` -- `@train`, `@val`,
`@test63`, `@KiHE` -- drawn from any registered datasets, and it is read through
the same two calls:

    list_names(dataset='@val')                # its slides, in its order
    locate(name, dataset='@val')              # KeyError if it is not in @val

No slide is copied: a virtual dataset is one list, `dataset,name`, in the
`DATASETS_JOB` cache (`vdataset/list_<name>.csv`, with a record of how it was
drawn), and every slide resolves through its own dataset's rule. It is drawn
with `select()`, which narrows the registry by dataset, by stain and by what
each slide's header says (`where=`, a test of its `SlideInfo` in the CATALOG,
`catalog()`), into a `Selection` that combines (`| & -`), samples and splits
into disjoint parts; `save_virtual` writes one, once.
`utilities/cli/build_cache/make_dataset.py` builds the catalog and runs the
recipes that write them. `virtual_ids()` lists them.
"""

from __future__ import annotations

import csv
import glob
import os
import random
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

from Cache import Address, Entry, dataset_key
from ReadGeometry import is_native, level_for
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
    `info_fn(entry)`, when the dataset has one, gives the labels its own
    layout carries for a slide (BRACS's `group` and `type`, from the path),
    which the catalog keeps beside the slide's header (`SlideInfo.labels`).
    """
    id: str
    stain: str
    locate_fn: Callable[..., Optional[WsiEntry]]
    list_fn: Callable[[], List[str]]
    info_fn: Optional[Callable[[WsiEntry], Dict[str, str]]] = None


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

    def _info(entry: WsiEntry) -> Dict[str, str]:
        """`<Group>/<Type>/`, the two directories the slide sits in."""
        type_dir: str = os.path.dirname(entry.path)
        return {'group': os.path.basename(os.path.dirname(type_dir)), 'type': os.path.basename(type_dir)}

    return _Dataset(id=dataset_id, stain=stain, locate_fn=_locate, list_fn=_list, info_fn=_info)


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


# ── the catalog: what each slide's header says ────────────────────────────────

#: The cache root of the catalog and of every virtual dataset: AccessDatasets' own, never a job's. Every job --
#: a smoke run, a bench, a training -- reads the same ones, whatever it is called, and none writes another.
DATASETS_JOB = 'Datasets'

#: What a catalog row holds; another set of columns is another version, and an older catalog is refused.
CATALOG_VERSION = 1


@dataclass(frozen=True)
class SlideInfo:
    """One slide of the catalog: where it is registered, its stain, what its header says, and the labels its
    dataset's layout gives it (`_Dataset.info_fn`: BRACS's `group` and `type`). `error` is why its header
    could not be read; such a slide has no mpp and no levels."""
    dataset: str
    name: str
    stain: str
    base_mpp: Optional[float]
    level_downsamples: Tuple[float, ...]
    level_dimensions: Tuple[Tuple[int, int], ...]
    has_photos: bool
    labels: Dict[str, str] = field(default_factory=dict)
    error: str = ''

    @property
    def n_levels(self) -> int:
        return len(self.level_downsamples)

    @property
    def dimensions(self) -> Tuple[int, int]:
        """Level 0's `(width, height)`."""
        return self.level_dimensions[0] if self.level_dimensions else (0, 0)

    def native_at(self, rungs: Sequence[float]) -> bool:
        """Every rung comes off a level of the slide's own with no resampling: the level `level_for` reads it
        from is that very downsample (`ReadGeometry.is_native`). False for a slide whose header was not read."""
        if not self.level_downsamples:
            return False
        return all(is_native(self.level_downsamples[level_for(self.level_downsamples, r)], r) for r in rungs)


def catalog_entry() -> Entry:
    return Address(DATASETS_JOB).entry('catalog')


def _dataset(dataset_id: str) -> _Dataset:
    for d in _DATASETS:
        if d.id == dataset_id:
            return d
    raise KeyError(f'{dataset_id!r} is not a registered dataset id. Known dataset ids: {dataset_ids()}')


def _cells(values: Iterable[Any]) -> str:
    return ' '.join(str(v) for v in values)


def _catalog_row(ds: _Dataset, name: str, opener: Callable[[str], Any]) -> Dict[str, Any]:
    """The catalog row of one slide: its header, read through `opener` (a slide handle with `base_mpp`,
    `level_downsamples`, `level_dimensions`, `close`), and its labels. A header that cannot be read is a row
    with the error, not a stopped catalog."""
    entry: Optional[WsiEntry] = ds.locate_fn(name)
    if entry is None:
        return dict(name=name, error='not found')
    row: Dict[str, Any] = dict(name=name, has_photos=int('photos' in entry.related))
    if ds.info_fn is not None:
        row.update({f'label_{k}': v for k, v in ds.info_fn(entry).items()})
    try:
        wsi: Any = opener(entry.path)
        try:
            row.update(base_mpp=float(wsi.base_mpp), level_downsamples=_cells(wsi.level_downsamples),
                       level_dimensions=_cells(f'{w}x{h}' for w, h in wsi.level_dimensions))
        finally:
            wsi.close()
    except Exception as e:                                              # noqa: BLE001
        row['error'] = f'{type(e).__name__}: {e}'
    return row


def _write_catalog(dataset_id: str, rows: List[Dict[str, Any]]) -> Path:
    entry: Entry = catalog_entry()
    key: str = dataset_key(dataset_id)
    rec: Dict[str, Any] = dict(id=key, parts=[f'dataset={dataset_id}'], versions={'catalog': CATALOG_VERSION},
                               upstream={}, env={}, n=len(rows), updated_at=time.strftime('%Y-%m-%dT%H:%M:%S'))
    cols: List[str] = list(dict.fromkeys(k for r in rows for k in r))
    with entry.writing(key, rec) as put:
        with open(put('slides', '.csv'), 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=cols or ['name'], restval='')
            writer.writeheader()
            writer.writerows(rows)
    return entry.path('slides', key, '.csv')


@lru_cache(maxsize=None)
def _read_catalog(path: str, mtime_ns: int) -> Tuple[Dict[str, str], ...]:
    with open(path, newline='') as handle:
        return tuple(csv.DictReader(handle))


def _catalog_rows(dataset_id: str) -> Optional[Tuple[Dict[str, str], ...]]:
    """The stored rows of one dataset's catalog; None when it has none. A catalog of another version is an
    error: its columns are not the ones `SlideInfo` reads."""
    entry: Entry = catalog_entry()
    key: str = dataset_key(dataset_id)
    stored: Optional[Dict[str, Any]] = entry.stored(key)
    if stored is None:
        return None
    if stored.get('versions', {}).get('catalog') != CATALOG_VERSION:
        raise ValueError(f'the catalog of {dataset_id!r} is version {stored.get("versions")}, this code reads '
                         f'{CATALOG_VERSION}: run make_dataset.py --catalog --refresh')
    path: Path = entry.path('slides', key, '.csv')
    return _read_catalog(str(path), path.stat().st_mtime_ns)


def build_catalog(datasets: Optional[Sequence[str]] = None, *, refresh: bool = False,
                  opener: Optional[Callable[[str], Any]] = None,
                  log: Callable[[str], None] = print) -> Dict[str, Tuple[int, int]]:
    """Write the catalog of each dataset (every registered one by default): `{dataset: (headers read, slides)}`.
    Only a slide the catalog does not hold yet is opened -- a header, not a pixel -- unless `refresh`, and a
    slide no longer on disk leaves it. `opener` opens a slide (`SafeSlide`, whose `base_mpp` is the one
    definition); a test passes its own."""
    if opener is None:
        from SafeSlide import SafeSlide                                 # noqa: PLC0415
        opener = SafeSlide
    out: Dict[str, Tuple[int, int]] = {}
    for dataset_id in (datasets or dataset_ids()):
        ds: _Dataset = _dataset(dataset_id)
        names: List[str] = ds.list_fn()
        old: Dict[str, Dict[str, str]] = ({} if refresh else
                                          {r['name']: dict(r) for r in (_catalog_rows(dataset_id) or ())})
        rows: List[Dict[str, Any]] = []
        read: int = 0
        for name in names:
            if name in old and not old[name].get('error'):
                rows.append(old[name])
                continue
            rows.append(_catalog_row(ds, name, opener))
            read += 1
        if read or set(old) != set(names):
            _write_catalog(dataset_id, rows)
        bad: int = sum(1 for r in rows if r.get('error'))
        log(f'  {dataset_id:17s} {len(rows)} slides, {read} headers read'
            + (f', {bad} unreadable' if bad else ''))
        out[dataset_id] = (read, len(rows))
    return out


def _floats(cell: str) -> Tuple[float, ...]:
    return tuple(float(v) for v in str(cell or '').split())


def _sizes_of(cell: str) -> Tuple[Tuple[int, int], ...]:
    return tuple((int(w), int(h)) for w, h in (v.split('x') for v in str(cell or '').split()))


def catalog(datasets: Optional[Sequence[str]] = None) -> List[SlideInfo]:
    """The catalog of each dataset (every registered one by default), a `SlideInfo` per slide. Read only: a
    dataset with no catalog is a refusal that names its writer, never a header opened here."""
    out: List[SlideInfo] = []
    for dataset_id in (datasets or dataset_ids()):
        ds: _Dataset = _dataset(dataset_id)
        rows: Optional[Tuple[Dict[str, str], ...]] = _catalog_rows(dataset_id)
        if rows is None:
            raise FileNotFoundError(f'{dataset_id!r} has no catalog in {catalog_entry().dir}: run '
                                    f'utilities/cli/build_cache/make_dataset.py --catalog')
        for r in rows:
            mpp: str = r.get('base_mpp', '')
            out.append(SlideInfo(
                dataset=dataset_id, name=r['name'], stain=ds.stain, base_mpp=float(mpp) if mpp else None,
                level_downsamples=_floats(r.get('level_downsamples', '')),
                level_dimensions=_sizes_of(r.get('level_dimensions', '')),
                has_photos=r.get('has_photos', '') == '1',
                labels={k[len('label_'):]: v for k, v in r.items() if k.startswith('label_') and v},
                error=r.get('error', '') or ''))
    return out


# ── selections: how a virtual dataset is drawn ───────────────────────────────

#: One slide of the registry: `(dataset id, name)`.
Member = Tuple[str, str]

#: How big a part of a split is: a count of slides, a share of the selection, or `'rest'`.
Size = Union[int, float, str]


def _resolve_sizes(sizes: Sequence[Size], n: int) -> List[int]:
    """Slides per part for a selection of `n`: a count as it is, a share rounded to the nearest slide (at least
    one), `'rest'` (at most one part) what the others leave, at least one. Asking for more than there is is an
    error, not a short part."""
    if list(sizes).count('rest') > 1:
        raise ValueError(f'{list(sizes)}: only one part can be the rest')
    fixed: List[Optional[int]] = []
    for size in sizes:
        if size == 'rest':
            fixed.append(None)
        elif isinstance(size, bool) or not isinstance(size, (int, float)):
            raise ValueError(f'{size!r}: a part is a count (10), a share (0.2) or \'rest\'')
        elif isinstance(size, int):
            if size < 1:
                raise ValueError(f'{size}: a part needs at least one slide')
            fixed.append(size)
        else:
            if not 0.0 < size <= 1.0:
                raise ValueError(f'{size}: a share is in (0, 1]')
            fixed.append(max(1, int(size * n + 0.5)))
    used: int = sum(s for s in fixed if s is not None)
    if used > n:
        raise ValueError(f'{list(sizes)} ask for {used} slides of {n}')
    if None in fixed and n - used < 1:
        raise ValueError(f'{list(sizes)}: the rest of {n} would be empty')
    return [n - used if s is None else s for s in fixed]


class Selection:
    """Slides of the registry, `(dataset, name)` in an order, each once, and how they were drawn (`how`, the
    record a virtual dataset keeps). `| & -` combine two (the left one's order first); `where` narrows by the
    catalog; `split` cuts disjoint parts and `sample` one, from the slides sorted and shuffled by a seed."""

    def __init__(self, members: Iterable[Member], how: str) -> None:
        self.members: Tuple[Member, ...] = tuple(dict.fromkeys((str(d), str(n)) for d, n in members))
        self.how: str = how

    def __len__(self) -> int:
        return len(self.members)

    def __iter__(self) -> Iterator[Member]:
        return iter(self.members)

    def __contains__(self, member: object) -> bool:
        return member in self.members

    def __repr__(self) -> str:
        return f'Selection({len(self)} slides: {self.how})'

    def names(self) -> List[str]:
        return [n for _, n in self.members]

    def __or__(self, other: 'Selection') -> 'Selection':
        return Selection(self.members + other.members, f'({self.how}) | ({other.how})')

    def __and__(self, other: 'Selection') -> 'Selection':
        keep: set = set(other.members)
        return Selection([m for m in self.members if m in keep], f'({self.how}) & ({other.how})')

    def __sub__(self, other: 'Selection') -> 'Selection':
        drop: set = set(other.members)
        return Selection([m for m in self.members if m not in drop], f'({self.how}) - ({other.how})')

    def where(self, test: Callable[[SlideInfo], bool], label: str = '') -> 'Selection':
        """The slides whose catalog row passes `test`; `label` says what it tests, for the record (a lambda has
        no name worth keeping). A slide the catalog does not hold is an error: run the catalog first."""
        info: Dict[Member, SlideInfo] = {(s.dataset, s.name): s
                                         for s in catalog(sorted({d for d, _ in self.members}))}
        missing: List[Member] = [m for m in self.members if m not in info]
        if missing:
            raise KeyError(f'the catalog holds no {missing[:5]}{" ..." if len(missing) > 5 else ""}: run '
                           f'make_dataset.py --catalog')
        said: str = label or getattr(test, '__name__', 'test')
        return Selection([m for m in self.members if test(info[m])], f'{self.how} where {said}')

    def sorted(self) -> 'Selection':
        return Selection(sorted(self.members), f'{self.how} sorted')

    def split(self, *sizes: Size, seed: int) -> Tuple['Selection', ...]:
        """Disjoint parts, cut in the order given from the slides sorted and shuffled by `seed`. Each part keeps
        that shuffled order, so a prefix of it is a random sample of it too. Parts that leave slides over
        (no `'rest'`) leave them in no part."""
        pool: List[Member] = sorted(self.members)
        random.Random(seed).shuffle(pool)
        counts: List[int] = _resolve_sizes(sizes, len(pool))
        out: List[Selection] = []
        at: int = 0
        for i, count in enumerate(counts):
            out.append(Selection(pool[at:at + count],
                                 f'{self.how} split({", ".join(map(repr, sizes))}, seed={seed})[{i}]'))
            at += count
        return tuple(out)

    def sample(self, size: Size, *, seed: int) -> 'Selection':
        """`size` of the slides, at random by `seed` (a count or a share), in that random order."""
        (part,) = self.split(size, seed=seed)
        return Selection(part.members, f'{self.how} sample({size!r}, seed={seed})')


def select(*, dataset: Union[None, str, Sequence[str]] = None, stain: Union[None, str, Sequence[str]] = None,
           where: Optional[Callable[[SlideInfo], bool]] = None, label: str = '') -> Selection:
    """The registry's slides as a `Selection`: those of `dataset` -- one id or several, a registered dataset or
    a virtual one `@<name>`; every registered dataset when None -- of `stain` (one or several), passing
    `where`, a test of each slide's catalog row (`Selection.where`). In each dataset's own order: sorted for a
    registered one, the recorded order for a virtual one."""
    ids: List[str] = ([dataset] if isinstance(dataset, str) else list(dataset)) if dataset is not None \
        else dataset_ids()
    stains_wanted: Optional[set] = None if stain is None else ({stain} if isinstance(stain, str) else set(stain))
    members: List[Member] = []
    for dataset_id in ids:
        if is_virtual(dataset_id):
            members += list(_virtual_members(dataset_id))
        else:
            members += [(dataset_id, n) for n in _dataset(dataset_id).list_fn()]
    if stains_wanted is not None:
        members = [m for m in members if _dataset(m[0]).stain in stains_wanted]
    how: str = f'select(dataset={ids!r}' + (f', stain={sorted(stains_wanted)!r}' if stains_wanted else '') + ')'
    chosen: Selection = Selection(members, how)
    return chosen.where(where, label) if where is not None else chosen


def by_source(dataset_id: str, n: Optional[int] = None,
              stain: Union[None, str, Sequence[str]] = None) -> List[Tuple[str, List[str]]]:
    """The slides of a dataset id -- a registered dataset or a virtual one `@<name>` -- by the dataset each is
    registered in: `[(dataset, names)]`, the datasets in the order their first slide comes, each one's names in
    the id's own order, the first `n` of each when given, only the slides of `stain` (one or several) when
    given. What a caller that draws a dataset at a time (a manifest, a sampler) iterates; a virtual dataset is
    shuffled when it is drawn, so a prefix is a random pick."""
    out: Dict[str, List[str]] = {}
    for source, name in select(dataset=dataset_id, stain=stain):
        names: List[str] = out.setdefault(source, [])
        if n is None or len(names) < n:
            names.append(name)
    return list(out.items())


# ── virtual datasets ─────────────────────────────────────────────────────────

#: A virtual dataset's id is `@<name>`. The `@` tells it from a registered dataset at a glance and means
#: nothing to a shell or a Makefile (a word starting with `#` would be a comment, silently dropped).
VIRTUAL_PREFIX = '@'
_VIRTUAL_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]*\Z')


def is_virtual(dataset_id: str) -> bool:
    return str(dataset_id).startswith(VIRTUAL_PREFIX)


def _virtual_name(dataset_id: str) -> str:
    """`'@val'` -> `'val'`; anything that is not `@` and letters, digits, `_`, `-` is refused."""
    name: str = str(dataset_id)[len(VIRTUAL_PREFIX):]
    if not is_virtual(dataset_id) or not _VIRTUAL_NAME.match(name):
        raise ValueError(f'{dataset_id!r}: a virtual dataset is {VIRTUAL_PREFIX}<name>, the name letters, digits, '
                         f'`_` and `-`, starting with a letter or digit')
    return name


def vdataset_entry() -> Entry:
    return Address(DATASETS_JOB).entry('vdataset')


def vdataset_file(dataset_id: str) -> Path:
    """`vdataset/list_<name>.csv` (`dataset,name`): where one virtual dataset's slides are. The only place that
    path is spelled."""
    return vdataset_entry().path('list', _virtual_name(dataset_id), '.csv')


def virtual_ids() -> List[str]:
    """Every virtual dataset written, `@<name>`, sorted."""
    return [VIRTUAL_PREFIX + i for i in vdataset_entry().ids()]


def virtual_record(dataset_id: str) -> Dict[str, Any]:
    """How a virtual dataset was drawn: its record (`how`, `recipe`, `n`, `sources`, `created_at`)."""
    stored: Optional[Dict[str, Any]] = vdataset_entry().stored(_virtual_name(dataset_id))
    if stored is None:
        raise KeyError(f'{dataset_id!r} is not a virtual dataset; written: {virtual_ids()} '
                       f'(utilities/cli/build_cache/make_dataset.py writes them)')
    return stored


@lru_cache(maxsize=None)
def _read_members(path: str, mtime_ns: int) -> Tuple[Member, ...]:
    with open(path, newline='') as handle:
        return tuple((r['dataset'], r['name']) for r in csv.DictReader(handle))


def _virtual_members(dataset_id: str) -> Tuple[Member, ...]:
    """The slides of a virtual dataset, in its order, each checked to be on disk now: a slide moved or gone is
    an error naming it, never a shorter list. One directory read per dataset it draws from."""
    virtual_record(dataset_id)
    path: Path = vdataset_file(dataset_id)
    members: Tuple[Member, ...] = _read_members(str(path), path.stat().st_mtime_ns)
    gone: List[str] = []
    for source in sorted({d for d, _ in members}):
        if source not in dataset_ids():
            gone += [f'{source}:{n}' for d, n in members if d == source]
            continue
        have: set = set(_dataset(source).list_fn())
        gone += [f'{source}:{n}' for d, n in members if d == source and n not in have]
    if gone:
        raise FileNotFoundError(f'{dataset_id}: {len(gone)} of its {len(members)} slides are not found now: '
                                f'{gone[:5]}{" ..." if len(gone) > 5 else ""}')
    return members


def save_virtual(dataset_id: str, selection: Selection, *, recipe: str = '') -> Path:
    """Write `selection` as the virtual dataset `dataset_id` (`@<name>`): its slides in its order, and a record
    of how they were drawn (`selection.how`, `recipe`). Written ONCE: one that exists is refused, never
    overwritten -- a checkpoint chosen on `@val` must not be scored on a `@val` that has changed since. Delete
    its entry to write another under that name. An empty selection, a slide not on disk, and one name from two
    datasets (`locate(name, dataset='@...')` could not tell them apart) are refused too."""
    name: str = _virtual_name(dataset_id)
    entry: Entry = vdataset_entry()
    if entry.stored(name) is not None:
        raise FileExistsError(f'{dataset_id} exists ({entry.record_path(name)}); a virtual dataset is written once')
    if not len(selection):
        raise ValueError(f'{dataset_id}: the selection is empty ({selection.how})')
    names: List[str] = selection.names()
    twice: List[str] = sorted({n for n in names if names.count(n) > 1})
    if twice:
        raise ValueError(f'{dataset_id}: {twice} are in it from two datasets')
    for source in sorted({d for d, _ in selection}):
        have: set = set(_dataset(source).list_fn())
        absent: List[str] = [n for d, n in selection if d == source and n not in have]
        if absent:
            raise FileNotFoundError(f'{dataset_id}: {source} holds no {absent[:5]}')
    rec: Dict[str, Any] = dict(id=name, parts=[f'recipe={recipe}'] if recipe else [], versions={}, upstream={},
                               env={}, how=selection.how, recipe=recipe, n=len(selection),
                               sources=sorted({d for d, _ in selection}),
                               created_at=time.strftime('%Y-%m-%dT%H:%M:%S'))
    with entry.writing(name, rec) as put:
        with open(put('list', '.csv'), 'w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(('dataset', 'name'))
            writer.writerows(selection.members)
    return vdataset_file(dataset_id)


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

    `dataset` may be a virtual dataset, `@<name>`: the slide must be in its list
    and resolves by the dataset it was drawn from (`**kwargs` are that one's).
    """
    if dataset is not None and is_virtual(dataset):
        # a virtual dataset: the slide must be in its list, and resolves by the dataset it was drawn from
        sources: List[str] = [d for d, n in _virtual_members(dataset) if n == name]
        if not sources:
            raise KeyError(f'{name!r} is not in {dataset!r}')
        return locate(name, dataset=sources[0], **kwargs)
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
    `locate(name, dataset=...)` picks by. The real datasets only: a virtual
    dataset is `virtual_ids()`'s.
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

    `dataset='@<name>'` gives a virtual dataset's slides in its order, those of
    `stain` only when given; one of them not on disk now is an error.

    A FRESH SCAN EVERY CALL, ON PURPOSE: this is the one function whose
    whole job is "what is actually there right now", not a lookup in a
    static table -- a name this returns is guaranteed findable by `locate`
    at the same moment (modulo a concurrent transfer).
    """
    if dataset is not None and is_virtual(dataset):
        listed: List[str] = [name for d, name in _virtual_members(dataset)
                             if stain is None or _dataset(d).stain == stain]
        return listed if n is None else listed[:n]
    if dataset is not None:
        if dataset not in dataset_ids():
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
