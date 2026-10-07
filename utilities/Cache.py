"""Where a cache lives, and how it is written so a killed job cannot corrupt it.

    root = cache_root(job_name('MppRoutingHead'), 'sampler')
    with atomic_dir(root / key / slide) as tmp:
        ...write files into tmp...
    meta = read_meta(path, require={'seg_id': ...})

THE LAYOUT
-----------
    result/cache/<made_by>_<object>/<config level>/<config level>/...

`made_by` is the job that PRODUCED the cache, not a job that reads it: every
CLI defaults to its own job name, and a reader that wants another job's cache
names it explicitly (`--sampler-cache-job`). Below the root there is one
directory per config the content depends on, upstream above downstream, each
named only by the parameters that level's content actually depends on. The
sampler cache is the worked example -- see `TileSampler.cached`:

    <made_by>_sampler/<seg_id>/<slide>/mask.safetensors       (seg only)
                                      <region>_<sampler>_<plan>/   (+ sampling)

Nesting by dependency is what makes invalidation a single `rm -rf`: dropping a
segmentation recipe drops every sample drawn from it, and a sample directory
can never outlive the mask it was drawn from.

WHAT IS HERE AND WHAT IS NOT
-----------------------------
The mechanism every cache shares: the root, atomic writes, the JSON sidecar, the
slide key and the guard against two slides with one stem. What a cache HOLDS --
the payload format, the domain fields in its meta -- stays with the module that
knows what it means (`TileSampler`, `TissueMaskConfig`). No hashing here: every
key a caller builds comes from its own config's id (`sampler_id`, `seg_id`,
...), so this module needs nothing past the standard library, and importing it
cannot pull torch into `TileSampler` or `TissueMask`.
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from _paths import RESULT_DIR, job_name                          # noqa: F401


class CacheMismatch(RuntimeError):
    """A cache entry is not the one the caller asked for. Never a fallback:
    reading the wrong entry is sampling from the wrong regions and finding
    out from a training curve that merely looks disappointing."""


# ── where ─────────────────────────────────────────────────────────────────────

def cache_root(made_by: str, obj: str) -> Path:
    """`result/cache/<made_by>_<obj>/`. Not created here: a reader pointed at
    another job's cache should find it missing, not find it freshly empty."""
    if not made_by or not obj:
        raise ValueError(f'cache_root needs both names, got {made_by!r}, {obj!r}')
    return Path(RESULT_DIR) / 'cache' / f'{made_by}_{obj}'


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


@contextlib.contextmanager
def atomic_dir(path) -> Iterator[Path]:
    """Yield a temporary directory beside `path`; on success rename it into
    place. If `path` already exists -- another job finished the same entry
    first -- ours is discarded: both were computed from the same key, so they
    hold the same thing, and the first one is already being read."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{uuid.uuid4().hex[:8]}.tmp')
    tmp.mkdir()
    try:
        yield tmp
        if not path.exists():
            try:
                os.rename(tmp, path)
            except OSError:
                if not path.exists():
                    raise
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)


def write_meta(path, meta: Dict) -> Path:
    """The JSON sidecar, written atomically."""
    with atomic_file(path) as tmp:
        tmp.write_text(json.dumps(meta, indent=2, sort_keys=True, default=str))
    return Path(path)


def read_meta(path, *, require: Optional[Dict[str, object]] = None) -> Dict:
    """The JSON sidecar, refused outright when `require` misses."""
    meta = json.loads(Path(path).read_text())
    if require:
        bad = {k: (v, meta.get(k, '<absent>')) for k, v in require.items()
               if meta.get(k) != v}
        if bad:
            raise CacheMismatch(
                f'{path}: ' + ', '.join(f'{k} wanted {w!r}, found {g!r}'
                                        for k, (w, g) in bad.items()))
    return meta


def find(root, pattern: str, **eq) -> List[Path]:
    """Every sidecar under `root` matching the glob `pattern` whose meta has
    each `eq` field equal to the value given. Sorted, so two runs list the same
    entries in the same order."""
    out = []
    for path in sorted(Path(root).glob(pattern)):
        meta = json.loads(path.read_text())
        if all(meta.get(k) == v for k, v in eq.items()):
            out.append(path)
    return out
