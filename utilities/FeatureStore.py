"""Read and write encoded tile features, keyed by what makes a tile a tile.

One file per (wsi, level, pooling, config). The config half of that key is a
hash, so two runs that differ in encoder or tissue mask produce two files rather
than one silently-wrong file.

Why this module exists at all
-----------------------------
A feature store is easy to write and easy to misread. The failure that matters
is not a crash, it is a store that loads cleanly and means something other than
the caller assumed:

  * a store written from a SAMPLE of a slide, read by code that assumes it
    covers the slide -- retrieval then searches 2% of the tissue and reports no
    match, with nothing in the logs to say why;
  * a store written by one encoder, read after the encoder changed;
  * `slots` out of step with the feature tensor's middle dimension, so slot 3
    means something different to the writer and the reader.

None of those raise on their own. So `save()` validates aggressively and
`load(require=...)` refuses rather than falls back. Both are the point of the
module, not decoration around it.

Deliberately NOT imported here
------------------------------
Nothing from this project. Not PatchingLib, not the pipeline. Both the bench and
(later) LocaScopePipeline depend on this file, so whatever it imports becomes
their dependency too. Rebuilding a FeaturesMap from a store is the caller's job;
the four coordinate tensors are here so that it can be done without re-running a
mask.

Layout
------
    result/cache/features/<wsi_stem>__L<level>__<pooling>__<cfg8>.safetensors

      features   [N, n, D] fp16    n = len(slots); n = 1 still uses 3 dims so
                                   readers never branch on shape
      x, y       [N] int32         level-0 top-left of each tile
      region     [N] int16         which tissue region it came from
      grid_rc    [N, 2] int32      (row, col) within that region's grid

    plus any `extra` tensors the caller passes (the query store carries the
    answer indices into its reference store this way, rather than in a sidecar
    that can go missing).

`result/cache/` rather than `result/<job>/` is deliberate: this is reused across
jobs, and `make clean-job JOB=cache` is then the one obvious way to purge it.
See ClaudeRules section 5.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

SCHEMA_VERSION = '1'

CORE_TENSORS = ('features', 'x', 'y', 'region', 'grid_rc')

#: Fields that decide what a tile IS. Two stores that agree on these hold
#: comparable vectors; two that disagree do not, whatever else they share.
#: `level` and `pooling` are excluded because they are already in the filename,
#: and `n_tiles` / `created_at` / `sample_seed` because they do not change the
#: meaning of a tile.
#:
#: Three rules hold this set together. They are here rather than in a design
#: document because this tuple is the only thing a future reader has to get
#: right, and the cost of each mistake is different.
#:
#: 1. APPEND ONLY, AND NEVER REPURPOSE A NAME.
#:    Adding a field re-hashes every existing store, but does not break them:
#:    find() matches on metadata rather than filename, load(require=) still
#:    validates, and save() writes a new name instead of overwriting. Old files
#:    become orphans, which is a recompute, not a wrong answer. Giving an
#:    existing name a new meaning is the one change that fails silently -- the
#:    old and new stores hash the same and hold different things.
#:
#: 2. ONLY QUANTITIES WHOSE DEFINITION IS STABLE.
#:    The values vary freely; that is the point. What must not vary is what the
#:    NAME means. base_mpp used to be here and failed this twice: it decides
#:    nothing about a tile (given `level`, which is in the filename, no part of
#:    the read path touches it), and its definition lives in SafeSlide.base_mpp,
#:    which changed on 2026-08-13 from mpp-x alone to the mean of mpp-x and
#:    mpp-y. Every slide in this project has mpp-x != mpp-y, so that commit gave
#:    a new hash to every store while the tiles and the vectors were unchanged.
#:    ds replaces it and passes: it comes from the slide file
#:    (level_downsamples), not from code of ours. It is redundant with `level`
#:    today and deliberately so -- once a caller can encode at a downsample that
#:    is not a pyramid level, one level resampled to two scales would be two
#:    different sets of tiles under one filename. `level` stays in the filename
#:    for the other half of that: ds=2 resampled from L0 and ds=2 native at L1
#:    are not the same pixels.
#:
#: 3. THE HASH SEPARATES FILES; IT DOES NOT PROVE CORRECTNESS.
#:    Every id here is a string some caller composed, so the hash is exactly as
#:    good as the weakest of them -- mask_id in particular has to cover the
#:    segmentation method, its ds, the region filter and whether merging ran, or
#:    two different masks collide on one name. Getting the hash wrong costs a
#:    recompute; getting a CHECK wrong costs a wrong answer. So the checks are
#:    elsewhere and are made of the thing itself rather than its label: recompute
#:    the grid geometry and compare it against the stored x / y / region /
#:    grid_rc, and re-encode a handful of tiles and compare the vectors.
_IDENTITY_FIELDS = ('ds', 'tile_size', 'overlap', 'encoder_id', 'mask_id',
                    'sampler_id')

_COVERAGE = ('all', 'sample')


class StoreMismatch(RuntimeError):
    """A store exists but is not the one the caller asked for."""


# ── metadata codec ────────────────────────────────────────────────────────────
#
# safetensors metadata is Dict[str, str] and nothing else, so every field needs
# an explicit encode/decode. Dispatch is on the declared annotation rather than
# a hand-kept list: a field whose type has no codec raises at import-time use,
# which is how a newly added field fails loudly instead of being dropped.

def _enc_float(v: float) -> str:
    # Fixed precision, not repr(): cfg_hash feeds on these strings and must come
    # out the same on every machine and every Python.
    return f'{v:.12g}'


def _enc_slots(v: Tuple[str, ...]) -> str:
    for s in v:
        if ',' in s:
            raise ValueError(f'slot name may not contain a comma: {s!r}')
    return ','.join(v)


_CODECS = {
    'str':                       (lambda v: v,                  lambda s: s),
    'int':                       (str,                          int),
    'float':                     (_enc_float,                   float),
    'bool':                      (lambda v: '1' if v else '0',  lambda s: s == '1'),
    'Optional[int]':             (lambda v: '' if v is None else str(v),
                                  lambda s: None if s == '' else int(s)),
    'Tuple[str,...]':            (_enc_slots,
                                  lambda s: tuple(s.split(',')) if s else ()),
    'Optional[Tuple[int,int]]':  (lambda v: '' if v is None else f'{v[0]}x{v[1]}',
                                  lambda s: None if s == ''
                                            else tuple(int(p) for p in s.split('x'))),
}


def _codec(annotation: str):
    key = annotation.replace(' ', '')
    try:
        return _CODECS[key]
    except KeyError:
        raise TypeError(
            f'StoreMeta field annotated {annotation!r} has no metadata codec. '
            f'Add one to _CODECS -- silently dropping the field would make '
            f'require= unable to check it.'
        ) from None


@dataclass(frozen=True)
class StoreMeta:
    """Everything needed to decide whether a store is the one you wanted."""

    wsi_stem:    str
    wsi_path:    str
    level:       int
    ds:          float
    mpp:         float
    base_mpp:    float
    tile_size:   int
    overlap:     bool

    # what is in `features`
    pooling:     str                        # 'tokens' | 'cls' | 'cls_avg' | ...
    slots:       Tuple[str, ...]            # len == n
    slot_layout: str                        # 'none' | 'grid:2x2' | 'ring:3' | ...
    dim:         int                        # D
    token_grid:  Optional[Tuple[int, int]]  # the model's patch grid, if relevant
    num_prefix:  int                        # CLS + registers; the slice point

    # what produced it
    encoder_id:  str                        # e.g. 'prov-gigapath@fp16'
    mask_id:     str                        # e.g. 'hest@ds4' | 'none@ds4'

    # how much of the slide it covers
    coverage:    str                        # 'all' | 'sample'
    n_available: int                        # tiles the grid offers at this level
    sample_seed: Optional[int]
    n_tiles:     int                        # N

    #: Identity of the rule that CHOSE these tiles, not of the encoder that made
    #: them -- ReferenceSampler.SamplerConfig.sampler_id(). Empty for stores
    #: written before quotas existed, which is honest: they record no sampling
    #: rule because they had none beyond a seeded uniform draw.
    #:
    #: It is in _IDENTITY_FIELDS because without it two runs with different
    #: quotas, different thresholds or a different seed produce the same
    #: filename, and the second silently replaces the first.
    sampler_id:  str = ''

    created_at:  str = ''
    schema_version: str = SCHEMA_VERSION

    # ── identity ─────────────────────────────────────────────────────────────

    def cfg_hash(self) -> str:
        parts = []
        by_name = {f.name: f for f in dataclasses.fields(self)}
        for name in sorted(_IDENTITY_FIELDS):
            enc, _ = _codec(by_name[name].type)
            parts.append(f'{name}={enc(getattr(self, name))}')
        return hashlib.sha256('|'.join(parts).encode()).hexdigest()[:8]

    def filename(self) -> str:
        return (f'{self.wsi_stem}__L{self.level}__{self.pooling}'
                f'__{self.cfg_hash()}.safetensors')

    # ── string round-trip ────────────────────────────────────────────────────

    def to_strings(self) -> Dict[str, str]:
        out = {}
        for f in dataclasses.fields(self):
            enc, _ = _codec(f.type)
            out[f.name] = enc(getattr(self, f.name))
        return out

    @classmethod
    def from_strings(cls, d: Dict[str, str]) -> 'StoreMeta':
        kwargs = {}
        for f in dataclasses.fields(cls):
            if f.name not in d:
                # A field added after a store was written falls back to its
                # default rather than making every existing file unreadable. A
                # field WITHOUT a default still raises: that one is not a later
                # addition, it is a store that never carried something it should
                # have, and guessing it would be worse than refusing.
                if f.default is not dataclasses.MISSING:
                    kwargs[f.name] = f.default
                    continue
                raise StoreMismatch(
                    f'store metadata has no {f.name!r} '
                    f'(schema_version in file: {d.get("schema_version")!r})')
            _, dec = _codec(f.type)
            kwargs[f.name] = dec(d[f.name])
        return cls(**kwargs)


# ── validation ────────────────────────────────────────────────────────────────

def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f'FeatureStore.save: {msg}')


def _validate(features, x, y, region, grid_rc, meta: StoreMeta, extra) -> None:
    """Refuse at write time. Every one of these is silent if it reaches a reader."""
    _check(features.ndim == 3,
           f'features must be [N, n, D], got {tuple(features.shape)}')
    n_tiles, n_slots, dim = features.shape

    _check(features.dtype == torch.float16,
           f'features must be fp16, got {features.dtype}')
    _check(len(meta.slots) == n_slots,
           f'{len(meta.slots)} slot names for {n_slots} slots: {meta.slots}')
    _check(meta.dim == dim, f'meta.dim={meta.dim} but features have D={dim}')
    _check(meta.n_tiles == n_tiles,
           f'meta.n_tiles={meta.n_tiles} but features have N={n_tiles}')

    for name, t, want_dtype, want_shape in (
        ('x',       x,       torch.int32, (n_tiles,)),
        ('y',       y,       torch.int32, (n_tiles,)),
        ('region',  region,  torch.int16, (n_tiles,)),
        ('grid_rc', grid_rc, torch.int32, (n_tiles, 2)),
    ):
        _check(tuple(t.shape) == want_shape,
               f'{name} must be {want_shape}, got {tuple(t.shape)}')
        _check(t.dtype == want_dtype,
               f'{name} must be {want_dtype}, got {t.dtype}')

    _check(meta.coverage in _COVERAGE,
           f'coverage must be one of {_COVERAGE}, got {meta.coverage!r}')
    if meta.coverage == 'sample':
        _check(meta.sample_seed is not None,
               "coverage='sample' needs a sample_seed, or the draw cannot be redone")
        # The count that must not exceed the grid is the ON-GRID one, not the
        # total. A sampler may DISPLACE a tile to fill a bucket the grid cannot,
        # and an inherited tile is carried from another level; neither sits on
        # this level's grid, so a store legitimately holds more tiles than the
        # grid offers and the naive `n_tiles <= n_available` fires on normal
        # operation.
        #
        # Scoped this way it still catches what it was for -- the two ways the
        # tiles and the count come from different places: a level index left
        # over from the previous iteration, so L0's three thousand are checked
        # against L2's five hundred; and an accumulator not cleared between
        # levels, so the count doubles while n_available does not. Both surface
        # far from where they happen -- this repo has already paid for one, when
        # an int32 overflow emptied tissue_regions and only failed two stages
        # later as an empty torch.cat.
        origin = (extra or {}).get('origin')
        if origin is not None:
            n_grid = int((origin == 0).sum())
            _check(n_grid <= meta.n_available,
                   f'{n_grid} tiles came from the grid but it offers only '
                   f'{meta.n_available}')
        else:
            # No provenance recorded, so every tile is assumed to be a grid
            # position -- which is what stores written before origin existed
            # actually were.
            _check(n_tiles <= meta.n_available,
                   f'sampled {n_tiles} of {meta.n_available} available')
    else:
        _check(n_tiles == meta.n_available,
               f"coverage='all' but n_tiles={n_tiles} != n_available="
               f'{meta.n_available}')

    if meta.slot_layout.startswith('grid:'):
        gh, gw = (int(v) for v in meta.slot_layout[5:].split('x'))
        want = gh * gw + (1 if 'cls' in meta.slots else 0)
        _check(want == n_slots,
               f'slot_layout {meta.slot_layout!r} implies {want} slots, got {n_slots}')
    elif meta.slot_layout.startswith('ring:'):
        want = int(meta.slot_layout[5:]) + (1 if 'cls' in meta.slots else 0)
        _check(want == n_slots,
               f'slot_layout {meta.slot_layout!r} implies {want} slots, got {n_slots}')

    for k in (extra or {}):
        _check(k not in CORE_TENSORS,
               f'extra tensor {k!r} collides with a core tensor name')


# ── write ─────────────────────────────────────────────────────────────────────

def save(root, *, features, x, y, region, grid_rc,
         meta: StoreMeta, extra: Optional[Dict[str, torch.Tensor]] = None) -> Path:
    """Validate, then write atomically to root/<meta.filename()>.

    Atomic because jobs here get killed by walltime -- twice in one afternoon,
    on record. A truncated store that still loads is worse than no store: it
    quietly drops half the distractors and every recall number comes out high.
    """
    _validate(features, x, y, region, grid_rc, meta, extra)

    meta = dataclasses.replace(meta, created_at=meta.created_at or
                               time.strftime('%Y-%m-%dT%H:%M:%S'))

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / meta.filename()

    tensors = {'features': features.contiguous(),
               'x': x.contiguous(), 'y': y.contiguous(),
               'region': region.contiguous(), 'grid_rc': grid_rc.contiguous()}
    for k, v in (extra or {}).items():
        tensors[k] = v.contiguous()

    tmp = path.with_suffix('.tmp')
    save_file(tensors, str(tmp), metadata=meta.to_strings())
    os.replace(tmp, path)
    return path


# ── read ──────────────────────────────────────────────────────────────────────

def load_meta(path) -> StoreMeta:
    """Metadata only. Reads the header, not the tensors, so this stays instant
    on a 30 GB store."""
    with safe_open(str(path), framework='pt') as f:
        md = f.metadata()
    if md is None:
        raise StoreMismatch(f'{path} has no metadata -- not a FeatureStore file')
    return StoreMeta.from_strings(md)


def load(path, *, require: Optional[Dict[str, object]] = None,
         keys: Optional[Iterable[str]] = None):
    """Return (tensors, meta), refusing outright when `require` does not match.

    `require` is the whole reason a caller should come through here rather than
    calling safetensors directly. It never falls back: a store that is not the
    one you asked for is an error, because the alternative is using the wrong
    features and finding out from a number that looks merely disappointing.
    """
    meta = load_meta(path)

    if require:
        bad = {k: (v, getattr(meta, k, '<no such field>'))
               for k, v in require.items() if getattr(meta, k, None) != v}
        if bad:
            lines = '\n'.join(f'  {k}: wanted {w!r}, store has {g!r}'
                              for k, (w, g) in sorted(bad.items()))
            raise StoreMismatch(f'{path}\n{lines}')

    if keys is None:
        tensors = load_file(str(path))
    else:
        tensors = {}
        with safe_open(str(path), framework='pt') as f:
            for k in keys:
                tensors[k] = f.get_tensor(k)
    return tensors, meta


def find(root, *, wsi_stem: Optional[str] = None, level: Optional[int] = None,
         pooling: Optional[str] = None, **eq) -> list:
    """Stores under `root` matching the given fields, by metadata not filename.

    Reading each header beats parsing names: the filename carries a config hash
    that callers would otherwise have to recompute, and recomputing an identity
    rule in a second place is how the rule drifts.
    """
    want = {k: v for k, v in
            (('wsi_stem', wsi_stem), ('level', level), ('pooling', pooling))
            if v is not None}
    want.update(eq)

    hits = []
    for p in sorted(Path(root).glob('*.safetensors')):
        try:
            meta = load_meta(p)
        except Exception:
            continue                      # not ours; leave other files alone
        if all(getattr(meta, k, None) == v for k, v in want.items()):
            hits.append(p)
    return hits


def find_one(root, *, what: str = 'store', **eq) -> Path:
    """The single store matching `eq`, or an error naming the ones that did.

    find() returns a list because a query can legitimately match several, and
    almost every caller then wants exactly one. Taking hits[0] is the tempting
    shape and the wrong one: a root can hold stores built by different rules --
    a quota-sampled reference and an older uniform draw differ in sampler_id,
    so they have different filenames and coexist happily -- and picking whichever
    sorted first would score an experiment against a reference nobody chose,
    silently and plausibly.

    The fields that separate such stores are all queryable, so the fix when this
    raises is to name one: sampler_id='' for a pre-quota draw, or a hash for one
    quota config.
    """
    hits = find(root, **eq)
    if not hits:
        asked = ', '.join(f'{k}={v!r}' for k, v in sorted(eq.items()))
        raise FileNotFoundError(f'no {what} under {root} matching {asked}')
    if len(hits) > 1:
        names = '\n  '.join(h.name for h in hits)
        asked = ', '.join(f'{k}={v!r}' for k, v in sorted(eq.items()))
        raise StoreMismatch(
            f'{len(hits)} stores under {root} match {asked}:\n  {names}\n'
            f'They were built by different rules. Add a field that separates '
            f'them -- sampler_id and mask_id are the usual ones.')
    return hits[0]
