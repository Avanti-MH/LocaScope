"""Pre-tiles on disk: the context a homography is allowed to sample from.

    meta = PreTileMeta.of(wsi, plan, tile=256, sampler_id=cfg.sampler_id(), seed=0,
                          segmenter_id=seg_id)
    folder = PreTileStore.create(root, meta)
    PreTileStore.save_tile(folder, record, image)      # per sampled position
    PreTileStore.write_index(folder, records)          # once, at the end

    records = PreTileStore.load_index(folder)
    pre = PreTileStore.read_tile(folder, records[0])   # pre_px x pre_px RGB
    tile = PreTileStore.centre_crop(pre, meta.tile)    # what a model sees

WHAT A PRE-TILE IS AND WHY THE STORE HOLDS THAT RATHER THAN THE TILE
---------------------------------------------------------------------
`warp_image` fills anything sampled from outside its input with pure black
(`BORDER_CONSTANT, borderValue=0`), and a production homography needs 1.78x the
source it is given -- measured, spec.md 6.6, `valid 67.8%`. A third of every
warped training image would be black.

Black is not a blank here. It is a straight maximum-contrast edge with two
perfect right angles, which is the stimulus a corner detector is built to fire
on. Those points score high, get warped back by `H_inv`, and land in the HA
average as if they were tissue.

A photograph has no answer to this -- there is nothing outside its frame.
A WSI does: real tissue continues past any tile boundary. So the store holds a
pre-tile of `tile * PRE_TILE_FACTOR` on a side, centred on the tile, the warp
runs on the whole pre-tile, and the centre `tile x tile` is cropped out of the
result. Every output pixel then comes from real tissue.

    read    pre_px x pre_px, centred on the target tile
    warp    the whole pre-tile
    crop    the central tile x tile

THE FACTOR IS 3, DERIVED, NOT CALIBRATED
-----------------------------------------
The four sampling ops multiply. With the tile side as 1: quad `patch_ratio`
0.85 -> perspective 1.25 -> scaling 1.50 -> rotation 2.12 -> frame/quad 2.49.
So `sqrt(2)` -- the rotation term alone -- is not enough, and 3 is.

It is a MAXIMUM and not a percentile because the pre-tile is read ONCE per
position and all N = 100 homographies of that position warp from it. What has to
fit is the largest of 100 draws, and at N = 100 the 99th percentile of one draw
and the worst case are the same thing. The worst case is derivable, so there is
no calibration run -- `cli/demo_homography.py --calibrate` is a CHECK that no
draw exceeds 3, not the thing that chose 3.

THE TILE IS ALWAYS THE CENTRE CROP, INCLUDING AT THE SLIDE EDGE
----------------------------------------------------------------
A position near the edge of the scanned rectangle cannot have its full pre-tile.
The tempting fix is to slide the pre-tile inward until it fits; this store does
not, because that silently moves the tile away from the centre and every crop
downstream would be of the wrong place. Instead the read is taken where it is,
whatever the slide returns at the edge is what is stored, and `clip_px` on the
record says how far out it reached.

That keeps ONE invariant true everywhere -- `centre_crop(pre, tile)` is the
tile -- and turns the edge case into a number a reader can filter on rather than
a geometry that differs per record. spec.md 6.6's free assertion (`valid_mask`
must be all-True over the central tile) is what catches the rest, and on a
clipped record it is expected to fail: that is the record telling the truth.

WHY PNG FILES AND NOT ONE PACKED TENSOR
----------------------------------------
`FeatureStore` and `MaskStore` are single safetensors files because their unit
of use is the whole array: a retrieval run wants every feature, a mask is one
slide. A training epoch wants ONE pre-tile at a time, in an order the sampler
chooses, from several worker processes. One file per record is random access
for free, and it is what prov-gigapath's preprocessing writes for the same
reason.

PNG rather than raw: lossless, so it cannot fabricate the gradients a keypoint
detector is looking at, and roughly half the bytes. JPEG is excluded on the
first of those -- ringing at an 8x8 block edge is a corner.

`index.csv` and `meta.json` beside the images carry what a filename cannot: the
level-0 position of every tile, which is what Stage B needs to line the same
place up across rungs, and the identity of the extraction that produced them.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

#: Bumped when a field changes MEANING. Adding one with a default does not need
#: it; renaming one or redefining what it holds does. Same rule as
#: ConfigIdentity's first and `MaskStore.SCHEMA_VERSION`.
SCHEMA_VERSION = '1'

#: spec.md 6.6. The derived bound is 2.49; 3 is that rounded up, and the slack
#: pays for the one approximation in the derivation (a projective map is not
#: exactly a scaling about the centre, so the 1/patch_ratio term is nominal).
#:
#: `utilities/cli/probe_tile_yield.py` imports it from here rather than holding
#: its own 3. The probe reports how many sampled positions have their PRE-TILE
#: running off the scanned rectangle, and that number is only about this store
#: if it is about the same factor.
PRE_TILE_FACTOR = 3

#: The root is the CALLER's. This module never derives one: `_paths` lives in
#: `utilities/` and is only importable after `setup_import_paths()`, which is a
#: CLI's job. `cli/extract_pretiles.py` resolves it to result/cache/tiles/ --
#: result/cache/ and not result/<job>/ because several jobs read these and they
#: are the byproduct of none of them (`make clean-job JOB=cache` purges them).

_INDEX_NAME = 'index.csv'
_META_NAME = 'meta.json'

#: What makes two extractions different. Everything else in `PreTileMeta` is
#: determined by these plus the slide, so hashing it too would hide a
#: disagreement rather than catch one -- `MaskStore._IDENTITY_FIELDS`'s reason.
#:
#: `segmenter_id` is in here because the mask decides which positions were
#: reachable at all. Two tile sets sampled through two different masks are two
#: different datasets even at the same seed, and without this field they would
#: share a directory name.
#:
#: `sampler_id` REPLACED `tissue_ratio` on 2026-08-27. The gate is gone from
#: the sampler -- a zero cap on the top richness buckets says the same thing
#: once instead of twice -- and a store keyed on a parameter that no longer
#: exists would key two different corpora to one name. `sampler_id` covers all
#: three sampling axes, which `tissue_ratio` never did: two runs differing only
#: in their bucket floors used to collide.
_IDENTITY_FIELDS = ('wsi_stem', 'tile', 'pre_tile_factor', 'ds',
                    'sampler_id', 'seed', 'segmenter_id')


class PreTileMismatch(RuntimeError):
    """A store that is not the one asked for, or one that is incomplete."""


# ── geometry ─────────────────────────────────────────────────────────────────

def pre_tile_px(tile: int, factor: int = PRE_TILE_FACTOR) -> int:
    """The pre-tile side, in tile-resolution pixels.

    Refuses an odd margin. `centre_crop` has to remove `(pre - tile) / 2` from
    each side, and a half-pixel there would put the tile off centre by a
    different amount on the two sides -- which is exactly the failure the store
    is arranged to make impossible. An odd factor times an even tile is always
    fine; the check is here for the day one of those stops being true.
    """
    pre = int(tile) * int(factor)
    if (pre - int(tile)) % 2:
        raise ValueError(
            f'tile {tile} x factor {factor} = {pre} leaves an odd margin of '
            f'{pre - tile} px, so the tile cannot sit exactly in the centre. '
            f'Use an even tile size or an odd factor')
    return pre


def centre_margin(tile: int, factor: int = PRE_TILE_FACTOR) -> int:
    """Pixels removed from each side to get the tile back out of the pre-tile."""
    return (pre_tile_px(tile, factor) - int(tile)) // 2


def centre_crop(image: np.ndarray, tile: int) -> np.ndarray:
    """The central `tile x tile` of `image`. The ONE definition of the crop.

    Extraction, training and the demo all need it, and three spellings of a
    centre crop is three chances for an off-by-one that shifts the labels
    against the pixels by a pixel and looks like a slightly worse model.
    """
    h, w = np.asarray(image).shape[:2]
    if h != w:
        raise ValueError(f'pre-tile must be square, got {h}x{w}')
    if h < tile:
        raise ValueError(f'pre-tile is {h} px, smaller than the {tile} px tile')
    if (h - tile) % 2:
        raise ValueError(
            f'{h} px pre-tile and {tile} px tile leave an odd margin; see '
            f'pre_tile_px()')
    off = (h - tile) // 2
    return image[off:off + tile, off:off + tile]


def footprint_l0(tile: int, ds: float) -> float:
    """Level-0 pixels one TILE covers. Not the pre-tile -- see `PreTileMeta`."""
    return float(tile) * float(ds)


# ── identity ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PreTileMeta:
    """One extraction: one slide, one rung, one tile size.

    The reading facts (`level`, `shrink`, `read_size`) are carried rather than
    re-derived so that a reader can see what was actually read without
    reconstructing a `RungPlan` from a pyramid that may have been remounted.
    They are NOT identity: they are determined by `ds` and the slide.
    """
    wsi_stem:        str
    ds:              float        # the ladder rung, not a native level
    tile:            int          # what the model sees
    sampler_id:      str          # SamplerConfig.sampler_id(): all three axes
    seed:            int
    segmenter_id:    str          # which mask the positions were drawn through
    pre_tile_factor: int = PRE_TILE_FACTOR

    #: Provenance, not identity -- a slide that moved between mounts must not
    #: invalidate a cache. Same rule as `MaskStore.MaskMeta.wsi_path`.
    wsi_path:        str = ''

    #: How the pre-tile was read. `read_size` is in LEVEL pixels and is the
    #: pre-tile's, not the tile's: it is `pre_px` after `shrink`.
    level:           int = 0
    level_ds:        float = 1.0
    shrink:          float = 1.0
    read_size:       int = 0

    #: Facts about what came out.
    n_tiles:         int = 0
    n_clipped:       int = 0     # records whose pre-tile ran past the scan
    n_requested:     int = 0     # what the sampler was asked for
    n_tries:         int = 0     # what rejection sampling spent getting there

    created_at:      str = ''
    schema_version:  str = SCHEMA_VERSION

    # ── derived, so nobody recomputes them differently ──

    @property
    def pre_px(self) -> int:
        return pre_tile_px(self.tile, self.pre_tile_factor)

    @property
    def margin_px(self) -> int:
        """Tile-resolution pixels from the pre-tile edge to the tile edge."""
        return centre_margin(self.tile, self.pre_tile_factor)

    @property
    def tile_footprint_l0(self) -> float:
        """Level-0 px one tile covers. What the richness scorer measured.

        The pre-tile's footprint is `pre_tile_factor` times this and is
        DELIBERATELY not what the sampler gates on: requiring the pre-tile to be
        tissue would multiply the rejection-sampling footprint by 3 and, at
        tile 256, take the reachable ladder from ds 32 down to ds 11 -- losing
        the two coarsest rungs, which are where Stage C's relative-survival
        labels carry the most information. spec.md 6.6 has the arithmetic.
        The pre-tile is warp CONTEXT; it has to be readable, not tissue.
        """
        return footprint_l0(self.tile, self.ds)

    @property
    def margin_l0(self) -> float:
        """Level-0 px from the pre-tile's top-left to the tile's."""
        return self.tile_footprint_l0 * (self.pre_tile_factor - 1) / 2.0

    # ── naming ──

    def cfg_hash(self) -> str:
        parts = [f'{n}={getattr(self, n)}' for n in sorted(_IDENTITY_FIELDS)]
        return hashlib.sha256('|'.join(parts).encode()).hexdigest()[:8]

    def dirname(self) -> str:
        """`<stem>__ds<d>__t<tile>__<cfg8>`.

        The three human-readable fields are in the name because a directory
        listing is how anyone finds these; the hash is there because those three
        do not determine the contents -- `sampler_id`, the seed and the mask
        do too, and two runs that differ only in those must not collide.
        """
        return (f'{self.wsi_stem}__ds{self.ds:g}__t{self.tile}__'
                f'{self.cfg_hash()}')

    def to_json(self) -> Dict[str, object]:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, d: Dict[str, object]) -> 'PreTileMeta':
        # Unknown keys are dropped rather than raised on: a store written by a
        # later build with an extra field stays readable by this one, and
        # SCHEMA_VERSION is what says when it should not be. No casting --
        # JSON round-trips int, float and str as themselves, which is why the
        # meta is JSON here and strings in `MaskStore` (safetensors metadata is
        # str-to-str and has no choice).
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def of(cls, wsi, plan, *, tile: int, sampler_id: str, seed: int,
           segmenter_id: str, factor: int = PRE_TILE_FACTOR,
           **counts) -> 'PreTileMeta':
        """Assemble from the three things that determine it, `MaskMeta.of`-style.

        `plan` is the `RungPlan` for the PRE-TILE (`DsLadder.plan(...,
        tile_size=pre_px)`), because the pre-tile is what gets read. Passing the
        tile's plan would record a `read_size` three times too small, and the
        first thing to notice would be a pre-tile that does not fit its own
        `pre_px` -- caught by `save_tile`, but one call too late to say why.
        """
        from MaskStore import wsi_stem_of        # utilities/, on sys.path

        pre = pre_tile_px(tile, factor)
        if plan.tile_size != pre:
            raise ValueError(
                f'plan is for a {plan.tile_size} px output but the pre-tile is '
                f'{pre} px. Build it with DsLadder.plan(..., tile_size='
                f'pre_tile_px({tile})) -- the PRE-tile is what is read')
        return cls(
            wsi_stem=wsi_stem_of(wsi), ds=float(plan.rung_ds), tile=int(tile),
            sampler_id=str(sampler_id), seed=int(seed),
            segmenter_id=str(segmenter_id), pre_tile_factor=int(factor),
            wsi_path=str(getattr(wsi, '_filename', '') or ''),
            level=int(plan.level), level_ds=float(plan.level_ds),
            shrink=float(plan.shrink), read_size=int(plan.read_size),
            **counts)


@dataclass(frozen=True)
class PreTileRecord:
    """One stored pre-tile.

    `x` and `y` are the TILE's top-left in level-0 coordinates -- the sampler's
    own output, unmodified. The pre-tile's own top-left is `x - meta.margin_l0`,
    derived rather than stored so the two can never disagree about where the
    centre is.
    """
    index:   int
    x:       int      # level-0, the TILE's top-left
    y:       int

    #: How far the pre-tile ran past the scanned rectangle, level-0. **0 on
    #: every record cut under the current contract**: `TileSampler` is handed
    #: the pre-tile as `reserve_l0`, so the lattice never offers a position
    #: that would clip, and `extract_pretiles` asserts this rather than filling
    #: it in. The field stays because a store cut before 2026-08-27 carries
    #: real values -- 295 of 500 at ds 32 on one slide -- and a reader has to
    #: be able to tell the two contracts apart.
    clip_px: int = 0

    # ── the three sampling axes, carried from `TileSampler.SampleMeta` ──
    #
    # Recorded rather than recomputed. Every one of them is a property of the
    # RUN that placed this tile: `bucket` depends on the scorer and the edges,
    # `overlap_max` on what else that rung took, `inherit_id` on a set fixed
    # before any rung was filled. None can be derived from (x, y) afterwards,
    # and a reader that tried would be answering a different question with the
    # same name.

    #: Richness: which bucket, and the raw score it came from.
    bucket: str = 'mid'
    score:  float = 0.0

    #: Overlap: the largest overlap ratio with any other tile of the same rung.
    #: 0.0 for a lattice position, which is every position under a disjoint
    #: lattice -- so a non-zero value here is the budget having been spent.
    overlap_max: float = 0.0

    #: Inheritance: the chain this tile belongs to, or -1. A chain is the same
    #: level-0 centre at every rung, which is what Stage B measures survival
    #: along, so `stack_kind` in `meta.json` says WHICH stack -- 'F' (the
    #: footprint grows with ds) or 'R' (it is held and the tile is degraded).
    inherit_id: int = -1

    #: 'grid' | 'jitter' | 'inherit', and a jittered tile's parent. Kept
    #: because the top-up share is capped and a reader has to be able to check
    #: it was, rather than trusting that it was.
    origin:   str = 'grid'
    parent_x: int = -1
    parent_y: int = -1

    @property
    def filename(self) -> str:
        return f'{self.index:06d}.png'

    def centre_l0(self, meta: PreTileMeta) -> Tuple[float, float]:
        """The tile's centre in level-0. What Stage B matches across rungs.

        The centre and not the corner, for CLAUDE.md's reason: under rotation
        the recorded corner and a predicted corner are different corners of the
        same footprint, and one shot's error read 901 um that way when the true
        centre error was 0.8 um.
        """
        half = meta.tile_footprint_l0 / 2.0
        return (self.x + half, self.y + half)

    def pre_origin_l0(self, meta: PreTileMeta) -> Tuple[float, float]:
        return (self.x - meta.margin_l0, self.y - meta.margin_l0)


# ── write ────────────────────────────────────────────────────────────────────

def create(root, meta: PreTileMeta, *, overwrite: bool = False) -> Path:
    """Make the directory and write `meta.json`. Returns the directory.

    Refuses an existing COMPLETE store rather than adding to it: a directory
    with an `index.csv` is something a training run may already have read, and
    appending to it would change a dataset out from under a run that believes it
    knows how big it is. An interrupted one (no index) is resumable, which is
    what a walltime kill leaves behind.
    """
    folder = Path(root) / meta.dirname()
    if (folder / _INDEX_NAME).exists() and not overwrite:
        raise PreTileMismatch(
            f'{folder} already holds a finished extraction '
            f'({len(load_index(folder))} tiles). Pass overwrite=True to '
            f'replace it, or read it with load_index()')
    folder.mkdir(parents=True, exist_ok=True)
    meta = dataclasses.replace(
        meta, created_at=meta.created_at or time.strftime('%Y-%m-%dT%H:%M:%S'))
    _write_atomic(folder / _META_NAME,
                  json.dumps(meta.to_json(), indent=2, sort_keys=True).encode())
    return folder


def save_tile(folder, record: PreTileRecord, image: np.ndarray,
              meta: Optional[PreTileMeta] = None) -> Path:
    """Write one pre-tile as PNG, atomically.

    `meta` is optional and is only used to check the size. Pass it: a pre-tile
    of the wrong side length is the one error that survives everything
    downstream -- `centre_crop` would still return a square, of the wrong place.
    """
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f'expected HxWx3 RGB, got shape {image.shape}')
    if image.shape[0] != image.shape[1]:
        raise ValueError(f'pre-tile must be square, got {image.shape[:2]}')
    if meta is not None and image.shape[0] != meta.pre_px:
        raise ValueError(
            f'pre-tile is {image.shape[0]} px, meta says {meta.pre_px} '
            f'({meta.tile} x {meta.pre_tile_factor}). The read size or the '
            f'shrink is wrong, and every crop from this store would be of the '
            f'wrong place')

    path = Path(folder) / record.filename
    ok, buf = cv2.imencode('.png', image[:, :, ::-1])     # RGB -> BGR
    if not ok:
        raise RuntimeError(f'cv2 failed to encode {path}')
    _write_atomic(path, buf.tobytes())
    return path


#: `index.csv`'s columns, in order, with `file` last. One definition, read by
#: the writer and by nothing else -- the reader goes through `csv.DictReader`
#: and takes what it needs by name, so a column added here reaches disk without
#: the reader having to be told, and an OLD store missing one still loads.
INDEX_COLUMNS = ('index', 'x', 'y', 'clip_px', 'bucket', 'score',
                 'overlap_max', 'inherit_id', 'origin', 'parent_x', 'parent_y',
                 'file')


def write_index(folder, records: Iterable[PreTileRecord],
                meta: Optional[PreTileMeta] = None) -> Path:
    """Write `index.csv` LAST. Its presence is what marks the store complete.

    Written once at the end rather than appended per tile, so a job killed at
    walltime leaves a directory with no index -- which `find` skips and `create`
    resumes, instead of a half index that reads as a small dataset.
    """
    records = list(records)
    path = Path(folder) / _INDEX_NAME
    lines = [','.join(INDEX_COLUMNS)]
    lines += [','.join(str(getattr(r, c)) for c in INDEX_COLUMNS[:-1])
              + f',{r.filename}' for r in records]
    _write_atomic(path, ('\n'.join(lines) + '\n').encode())

    if meta is not None:
        # Merged onto what `create` WROTE, not onto the caller's copy. Two
        # reasons, and the first one is a bug this had:
        #
        #   `create` stamps `created_at` on a local copy, so the caller still
        #   holds one with an empty field -- rewriting from it silently
        #   un-stamped every finished store, and the only sign was a meta.json
        #   with created_at="".
        #
        #   and the identity on disk is the one the DIRECTORY NAME was derived
        #   from. Taking it from the caller would let a meta with a different
        #   cfg_hash land inside a directory named after another one.
        #
        # So only the two counts come from here.
        try:
            meta = load_meta(folder)
        except PreTileMismatch:
            pass                          # no meta.json: fall back to the arg
        meta = dataclasses.replace(
            meta, n_tiles=len(records),
            n_clipped=sum(1 for r in records if r.clip_px > 0))
        _write_atomic(Path(folder) / _META_NAME,
                      json.dumps(meta.to_json(), indent=2,
                                 sort_keys=True).encode())
    return path


def _write_atomic(path: Path, payload: bytes) -> None:
    """tmp + os.replace. Jobs here get killed by walltime, and a truncated PNG
    that still decodes is worse than a missing one."""
    tmp = Path(str(path) + '.tmp')
    with open(tmp, 'wb') as handle:
        handle.write(payload)
    os.replace(tmp, path)


# ── read ─────────────────────────────────────────────────────────────────────

def load_meta(folder) -> PreTileMeta:
    path = Path(folder) / _META_NAME
    if not path.exists():
        raise PreTileMismatch(f'{folder} has no {_META_NAME} -- not a PreTileStore')
    with open(path) as handle:
        return PreTileMeta.from_json(json.load(handle))


def load_index(folder) -> List[PreTileRecord]:
    path = Path(folder) / _INDEX_NAME
    if not path.exists():
        raise PreTileMismatch(
            f'{folder} has no {_INDEX_NAME}, so the extraction that wrote it '
            f'did not finish. Re-run it -- create() resumes an unfinished '
            f'directory')
    with open(path, newline='') as handle:
        # `.get` with a default on the axis columns, plain `[...]` on the four
        # that have always been there. A store cut before 2026-08-27 has no
        # axis columns, and reading it must give the defaults rather than a
        # KeyError -- but a missing `x` is a corrupt index and should raise.
        return [PreTileRecord(
            index=int(row['index']), x=int(row['x']), y=int(row['y']),
            clip_px=int(row['clip_px']),
            bucket=row.get('bucket', 'mid') or 'mid',
            score=float(row.get('score') or 0.0),
            overlap_max=float(row.get('overlap_max') or 0.0),
            inherit_id=int(row.get('inherit_id') or -1),
            origin=row.get('origin', 'grid') or 'grid',
            parent_x=int(row.get('parent_x') or -1),
            parent_y=int(row.get('parent_y') or -1))
            for row in csv.DictReader(handle)]


def read_tile(folder, record) -> np.ndarray:
    """One pre-tile as HxWx3 RGB uint8. `record` may be a record or an index."""
    if not isinstance(record, PreTileRecord):
        record = PreTileRecord(index=int(record), x=0, y=0)
    path = Path(folder) / record.filename
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise PreTileMismatch(
            f'{path} is missing or unreadable. The index lists it, so either '
            f'the extraction was interrupted between the PNG and the index, or '
            f'files have been deleted under the store')
    return image[:, :, ::-1]


def find(root, **eq) -> List[Path]:
    """Directories under `root` whose meta matches every keyword.

    By metadata and not by filename, `MaskStore.find`'s reason: the name carries
    a hash a caller would otherwise have to recompute, and an identity rule
    recomputed elsewhere is one that drifts.
    """
    hits = []
    for candidate in sorted(Path(root).glob('*')):
        if not candidate.is_dir():
            continue
        try:
            meta = load_meta(candidate)
        except Exception:                                        # noqa: BLE001
            continue                      # not ours; leave other things alone
        if not (candidate / _INDEX_NAME).exists():
            continue                      # unfinished; not a dataset yet
        if all(getattr(meta, k, None) == v for k, v in eq.items()):
            hits.append(candidate)
    return hits


def find_one(root, **eq) -> Path:
    """The single store matching `eq`, or an error naming the ones that did.

    `find()[0]` is the wrong shape for `MaskStore.find_one`'s reason: a root can
    legitimately hold two extractions of the same slide and rung that differ in
    seed or sampler_id, and picking whichever sorted first would train on a
    dataset nobody chose.
    """
    hits = find(root, **eq)
    if len(hits) == 1:
        return hits[0]
    query = ', '.join(f'{k}={v!r}' for k, v in sorted(eq.items())) or '(no filter)'
    if not hits:
        raise PreTileMismatch(
            f'no finished pre-tile store under {root} matching {query}. '
            f'Present: {[p.name for p in sorted(Path(root).glob("*")) if p.is_dir()]}')
    raise PreTileMismatch(
        f'{len(hits)} stores under {root} match {query}, and this call needs '
        f'one. Narrow it with seed=... or sampler_id=... :\n' +
        '\n'.join(f'  {p.name}' for p in hits))


# ── the pre-tile as warp context ─────────────────────────────────────────────
#
# Two functions, and they are here rather than in `Homography.py` because what
# they encode is the pre-tile/tile geometry -- the same thing `centre_crop` and
# `centre_margin` encode -- composed with a warp. Homography.py knows nothing
# about pre-tiles and should not have to.
#
# THREE CALLERS, WHICH IS WHY THEY ARE FUNCTIONS. Homographic Adaptation warps
# `num` views per tile; the training pair dataset warps one per sample; and
# `demo_homography --calibrate` checks that no draw exceeds the factor. All
# three must compose the translation the SAME way, because composing it on the
# right instead of the left shifts the output frame rather than the input one --
# it warps a different part of the tile, and it looks entirely reasonable.

def translate(offset: int) -> np.ndarray:
    """The 3x3 that carries TILE coordinates into PRE-TILE coordinates."""
    matrix = np.eye(3, dtype=np.float64)
    matrix[0, 2] = float(offset)
    matrix[1, 2] = float(offset)
    return matrix


def warp_from_pretile(pre: np.ndarray, matrix: np.ndarray, margin: int,
                      out_shape: Tuple[int, int]) -> np.ndarray:
    """One warped TILE-sized view, sampled out of the pre-tile.

    `matrix` is output(tile) -> input(tile), the matrix `sample_homography`
    returned for the TILE's shape. Every coordinate recorded, warped or inverted
    anywhere else stays in that frame; `translate(margin)` is applied on the
    LEFT, so it changes only which pixels cv2 reads.

    The network therefore still sees `out_shape`, not the pre-tile. Warping the
    whole pre-tile and cropping afterwards would cost `factor**2` -- nine times,
    at factor 3 -- for the same result.
    """
    from common.Homography import warp_image                      # noqa: PLC0415

    return warp_image(pre, translate(margin) @ matrix, out_size=out_shape,
                      interpolation='linear', border_value=0)


def pretile_valid_mask(pre: np.ndarray, matrix: np.ndarray, margin: int,
                       out_shape: Tuple[int, int],
                       erosion_radius: int = 0) -> np.ndarray:
    """Which output pixels came from inside the PRE-tile.

    NOT `valid_mask(out_shape, matrix)`: that answers the question for a source
    the size of the tile and would call two thirds of a legitimate view invalid
    (spec.md 6.6 measured `valid 67.8%` for exactly that call). Built instead by
    warping an all-ones array of the pre-tile's own extent through the same
    composed matrix, so validity is asked of the pixels that were available.

    With factor 3 the result is all-True apart from the eroded rim, and that is
    spec.md 6.6's free assertion: a False in the interior means the draw ran off
    the pre-tile.
    """
    from common.Homography import erode_valid, warp_image         # noqa: PLC0415

    ones = np.ones(np.asarray(pre).shape[:2], np.uint8)
    mask = warp_image(ones, translate(margin) @ matrix, out_size=out_shape,
                      interpolation='nearest', border_value=0)
    return erode_valid(mask, erosion_radius)
