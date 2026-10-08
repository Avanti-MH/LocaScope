"""Where a cache lives, and how it is written so a killed job cannot corrupt it.

    addr  = Address(job_name('BenchLocaScope'), slide=s, seg=seg_id,
                    region=region_id, plan=plan, draw=sampler_id, render=gap_id)
    stage = addr.entry('stage1')
    if stage.status(s1_id, record)[0] != 'hit':
        with stage.writing(s1_id, record) as put:
            write_csv(put('output', '.csv'))
            write_csv(put('probs', '.csv'))

THE TREE
---------
    result/cache/<made_by>/<kind>=<id>/<kind>=<id>/.../<entry kind>/
                                                       <role>_<id>.<ext>
                                                       record_<id>.json

`TREE` is the one definition of which level nests under which, and `ENTRIES`
of the level each entry kind sits in. An `Address` can only grow along `TREE`:
a level without its parent, two branches at once or an unknown kind is refused
when the address is made, so no caller can spell a path the tree does not have.
A `<kind>=<id>/` directory is one chosen input; a `<kind>/` directory holds the
sibling variants computed from that same input, one `record_<id>.json` per
variant and every file of the variant named `<role>_<id>.<ext>`.

The record is the proof that a variant is complete and is the one asked for:
`Entry.writing` stages every file, moves them into place, and writes the record
last, so a killed write reads as a miss. `Entry.status` compares the stored
record with the one the caller would write (`ConfigIdentity.record_diff`): a hit,
a miss, or stale with the lines that differ. A write over a stale variant drops
every file of that id first, so a role the new write no longer produces cannot
survive from the old one.

The levels of an address are addresses only. A job that reads another job's
mask builds the same levels `.on()` that job's root: the levels above an entry
may hold no files of their own in the reader's tree.

WHO MADE IT
-----------
`made_by` is the job that PRODUCED the cache, not a job that reads it: every
CLI defaults to its own job name, and a reader that wants another job's cache
names it explicitly (`--mask-cache-job`, `--draw-cache-job`, ...). Every cache
is in this tree -- masks, draws, renders, splits, features, stage results,
pre-tiles, keypoint labels, chain-stack tiles -- and this module is the only
one that spells a `<kind>=` path (test_config_identity's lint).

WHAT IS HERE AND WHAT IS NOT
-----------------------------
The mechanism every cache shares: the tree, the roots, atomic writes, the
record, the JSON sidecar, the slide key and the guard against two slides with
one stem. What a cache HOLDS -- the payload format, the domain fields in its
meta -- stays with the module that knows what it means (`TileSampler`,
`TissueMaskConfig`). No hashing here: every id a caller passes comes from its
own config (`sampler_id`, `seg_id`, ...), so this module needs nothing past the
standard library and `ConfigIdentity`, and importing it cannot pull torch into
`TileSampler` or `TissueMask`.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import sys
import uuid
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from _paths import RESULT_DIR, job_name                          # noqa: F401
from ConfigIdentity import record_diff                           # noqa: E402


class CacheMismatch(RuntimeError):
    """A cache entry is not the one the caller asked for. Never a fallback:
    reading the wrong entry is sampling from the wrong regions and finding
    out from a training curve that merely looks disappointing."""


# ── the tree ──────────────────────────────────────────────────────────────────

#: Each level kind and the level it nests under; None is the job's root.
TREE: Dict[str, Optional[str]] = {
    'dataset': None,
    'slide':   None,
    'seg':     'slide',
    'region':  'seg',
    'grid':    'region',
    'plan':    'region',
    'draw':    'plan',
    'pretile': 'draw',
    'ds':      'pretile',
    'render':  'draw',
    'stage1':  'render',
    'stage2':  'stage1',
}

#: Each entry kind and the levels whose directory may hold its `<kind>/`.
ENTRIES: Dict[str, Tuple[str, ...]] = {
    'split':      ('dataset',),
    'mask':       ('slide',),
    'features':   ('grid', 'draw', 'render'),
    'draw':       ('plan',),
    'render':     ('draw',),
    'stage1':     ('render',),
    'stage2':     ('stage1',),
    'stage3':     ('stage2',),
    'tiles':      ('ds',),
    'labels':     ('ds',),
    'chainstack': ('slide',),
}

#: The key of an entry record that lists the variant's files; Cache's own.
MEMBERS = 'members'

#: A level value, an id or a role: one path component that cannot be read as
#: two -- no separator, no `=`, nothing a shell glob would expand. A comma is
#: allowed: the Ki67 MRXS slides are named `S1104233,G7E,110208`.
_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9._,-]*\Z')


def _name(what: str, value) -> str:
    value = str(value)
    if not _NAME.match(value):
        raise ValueError(f'{what} {value!r} is not one path component of '
                         f'[A-Za-z0-9._,-] starting with a letter or digit')
    return value


def dataset_key(dataset_id: str) -> str:
    """The `dataset=` level of a dataset id: `bracs/test` -> `bracs_test`."""
    return _name('dataset id', dataset_id.replace('/', '_'))


def _chain(kind: str) -> Tuple[str, ...]:
    """The kinds from the root down to `kind`, inclusive."""
    out = []
    while kind is not None:
        out.append(kind)
        kind = TREE[kind]
    return tuple(reversed(out))


class Address:
    """A point in `TREE` under one job's root: `<made_by>/<kind>=<id>/...`.

    `levels` may be given in any order; the path orders them by `TREE`. They
    must be exactly one chain from the root, so a level is never missing its
    parent and two branches (`grid` and `plan`) are never mixed."""

    __slots__ = ('made_by', 'levels')

    def __init__(self, made_by: Optional[str] = None, **levels):
        unknown = sorted(set(levels) - set(TREE))
        if unknown:
            raise ValueError(f'no level kind {unknown} in the cache tree; the '
                             f'kinds are {sorted(TREE)}')
        if made_by is not None:
            _name('made_by', made_by)
        chain: Tuple[str, ...] = ()
        if levels:
            chain = max((_chain(k) for k in levels), key=len)
            if set(levels) != set(chain):
                raise ValueError(
                    f'levels {sorted(levels)} are not one chain of the cache '
                    f'tree: below the root, {chain[-1]!r} needs exactly '
                    f'{list(chain)}')
        self.made_by = made_by
        self.levels: Tuple[Tuple[str, str], ...] = tuple(
            (k, _name(f'{k} id', levels[k])) for k in chain)

    def at(self, **more) -> 'Address':
        """This address grown by `more` levels."""
        clash = sorted(set(more) & {k for k, _ in self.levels})
        if clash:
            raise ValueError(f'{self!r} already has {clash}')
        return Address(self.made_by, **dict(self.levels), **more)

    def on(self, made_by: str) -> 'Address':
        """The same levels under another job's root."""
        return Address(made_by, **dict(self.levels))

    @property
    def leaf(self) -> Optional[str]:
        return self.levels[-1][0] if self.levels else None

    @property
    def root(self) -> Path:
        """`result/cache/<made_by>/`. Not created here: a reader pointed at
        another job's cache should find it missing, not find it freshly empty."""
        if self.made_by is None:
            raise ValueError(f'{self!r} names no job, so it is under no root')
        return Path(RESULT_DIR) / 'cache' / self.made_by

    @property
    def dir(self) -> Path:
        return self.root.joinpath(*(f'{k}={v}' for k, v in self.levels))

    def entry(self, kind: str) -> 'Entry':
        return Entry(self, kind)

    def children(self, kind: str) -> List[str]:
        """The values of the `<kind>=` directories directly under this
        address, sorted: what one known level holds, one level down -- the
        rungs of one slide's corpus, say. Not a search: nothing below that
        level is looked at."""
        if TREE.get(kind) != self.leaf:
            raise ValueError(f'{kind!r} does not nest under {self.leaf!r} in the '
                             f'cache tree')
        if not self.dir.is_dir():
            return []
        prefix = f'{kind}='
        return sorted(p.name[len(prefix):] for p in self.dir.glob(f'{prefix}*')
                      if p.is_dir())

    def __eq__(self, other) -> bool:
        return (isinstance(other, Address) and self.made_by == other.made_by
                and self.levels == other.levels)

    def __hash__(self) -> int:
        return hash((self.made_by, self.levels))

    def __repr__(self) -> str:
        inner = ', '.join(f'{k}={v!r}' for k, v in self.levels)
        return f'Address({self.made_by!r}' + (f', {inner})' if inner else ')')


class Entry:
    """The `<kind>/` directory at one address: sibling variants, each one
    `record_<id>.json` plus files `<role>_<id>.<ext>` (a role may be a
    directory, `<role>_<id>/`)."""

    def __init__(self, address: Address, kind: str):
        if kind not in ENTRIES:
            raise ValueError(f'no entry kind {kind!r}; the kinds are '
                             f'{sorted(ENTRIES)}')
        if address.leaf not in ENTRIES[kind]:
            raise ValueError(f'a {kind!r} entry sits at {list(ENTRIES[kind])}, '
                             f'and {address!r} ends at {address.leaf!r}')
        self.address, self.kind = address, kind

    @property
    def dir(self) -> Path:
        return self.address.dir / self.kind

    def path(self, role: str, id: str, ext: str = '') -> Path:
        """`<role>_<id><ext>` in this entry; `ext` includes its dot."""
        if ext and not (ext.startswith('.') and _NAME.match(ext[1:])):
            raise ValueError(f'extension {ext!r} must be a dot and a name')
        return self.dir / f'{_name("role", role)}_{_name("id", id)}{ext}'

    def record_path(self, id: str) -> Path:
        return self.path('record', id, '.json')

    def stored(self, id: str) -> Optional[Dict]:
        """The record variant `id` was written with; None when there is none."""
        path = self.record_path(id)
        return json.loads(path.read_text()) if path.is_file() else None

    def status(self, id: str, record: Dict) -> Tuple[str, List[str]]:
        """`('hit', [])`, `('miss', [])`, or `('stale', <the lines that
        differ>)` for variant `id` against the record a write would carry."""
        stored = self.stored(id)
        if stored is None:
            return 'miss', []
        diff = record_diff(stored, record)
        return ('stale', diff) if diff else ('hit', [])

    def ids(self) -> List[str]:
        """Every complete variant here, sorted."""
        if not self.dir.is_dir():
            return []
        return sorted(p.name[len('record_'):-len('.json')]
                      for p in self.dir.glob('record_*.json'))

    def members(self, id: str) -> List[Path]:
        """The files and directories variant `id` was written with, as its
        record lists them -- by name, never by pattern: a role and an id may
        both hold `_`, so `tile_sims_<id>` and `sims_<id>` are not told apart
        by their ends."""
        stored = self.stored(id) or {}
        return [self.dir / n for n in stored.get(MEMBERS, ())]

    def drop(self, id: str) -> List[Path]:
        """Remove variant `id`: the record first, so a reader that races the
        removal sees a miss and never a variant with files missing."""
        return self._remove(id, ())

    def _remove(self, id: str, also) -> List[Path]:
        record = self.record_path(id)
        gone = sorted(set(self.members(id)) | {self.dir / n for n in also})
        if record.exists():
            record.unlink()
        for p in gone:
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
        return [record] + gone

    @contextlib.contextmanager
    def writing(self, id: str, record: Dict) -> Iterator[Callable[..., Path]]:
        """Yield `put(role, ext='')`, the staged path for one file (or, written
        as a directory, one directory) of variant `id`. On success the old
        variant is dropped, every staged item moved into place, and the record
        written last, listing what was written under `MEMBERS`; on failure
        nothing of the write is left."""
        _name('id', id)
        if MEMBERS in record:
            raise ValueError(f'{MEMBERS!r} is the key the record lists its '
                             f'files under; the caller cannot set it')
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / f'.{id}.{uuid.uuid4().hex[:8]}.tmp'
        tmp.mkdir()
        staged: Dict[str, Path] = {}

        def put(role: str, ext: str = '') -> Path:
            name = self.path(role, id, ext).name
            if name in staged:
                raise ValueError(f'{name} is staged twice')
            staged[name] = tmp / name
            return staged[name]

        try:
            yield put
            missing = [n for n, p in staged.items() if not p.exists()]
            if missing:
                raise RuntimeError(f'{self.dir}: staged {missing} but wrote '
                                   f'nothing there')
            self._remove(id, staged)
            for name, p in staged.items():
                os.replace(p, self.dir / name)
            write_meta(self.record_path(id), {**record, MEMBERS: sorted(staged)})
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)


# ── where ─────────────────────────────────────────────────────────────────────

def wsi_stem_of(wsi_or_path) -> str:
    """The slide's name without directory or extension -- the `<slide>` level
    of every per-slide cache. One definition, because it appears in a directory
    name and in a query, and two spellings of a key is a query that silently
    matches nothing."""
    path = (wsi_or_path if isinstance(wsi_or_path, (str, Path))
            else getattr(wsi_or_path, '_filename', '') or '')
    return Path(str(path)).stem


def source_key(wsi_or_path) -> str:
    """`<parent dir>/<file name>` -- what a per-slide entry records about where
    its slide came from.

    Not the full path: a slide that moved between mounts must still hit, so
    the mount prefix cannot be part of it. Not the stem alone either: two
    datasets can hold different slides under one stem, and a key made of the
    stem would hand one slide's regions to the other. The last two components
    survive a move and still tell those two apart."""
    path = (wsi_or_path if isinstance(wsi_or_path, (str, Path))
            else getattr(wsi_or_path, '_filename', '') or '')
    p = Path(str(path))
    return f'{p.parent.name}/{p.name}'


def check_source(meta: Dict, wsi_or_path, where) -> None:
    """Refuse an entry recorded for a different slide with the same stem."""
    want, got = source_key(wsi_or_path), meta.get('source', '')
    if got != want:
        raise CacheMismatch(
            f'{where} was written for {got!r}, and this slide is {want!r}. Two '
            f'slides share the stem {wsi_stem_of(wsi_or_path)!r}; their cache '
            f'entries cannot share a directory')


# ── how ───────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def atomic_file(path) -> Iterator[Path]:
    """Yield a temporary path beside `path`; on success rename it into place.

    Jobs here get killed by walltime, and a truncated file that still loads is
    worse than no file. `os.replace` is atomic on one filesystem, so a reader
    sees the old file, the new file, or nothing -- never half of one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{uuid.uuid4().hex[:8]}.tmp')
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_meta(path, meta: Dict) -> Path:
    """The JSON sidecar, written atomically."""
    with atomic_file(path) as tmp:
        tmp.write_text(json.dumps(meta, indent=2, sort_keys=True, default=str))
    return Path(path)
