'''Training/eval data for MppRoutingHead. See spec.md's "Camera: train vs
eval", "Caching" and "Handle count" sections for the reasoning; this file is
the implementation of all three.

    rows = build_manifest('ki67_pure', n_per_rung=100)      # positions only

    # TRAIN -- fresh augmentation every access, WSI-batched so open handles
    # stay bounded (see group_by_wsi/iterate_epoch)
    for batch in iterate_epoch(rows, wsi_group_size=8, batch_size=64,
                               num_workers=8, epoch_seed=epoch):
        ...

    # VAL / TEST -- the SAME loop, a different manifest, and split='eval' so
    # each position renders the same pixels every time it is scored
    for batch in iterate_epoch(val_rows, wsi_group_size=8, batch_size=64,
                               num_workers=8, split='eval'):
        ...

NOTHING RENDERED IS EVER WRITTEN TO DISK. An earlier draft materialised the
eval corpus into `.pt` shards and cached a frozen encoder's output on top of
them. Both are gone: a row is one patch now rather than twenty, so re-rendering
is cheap, and the feature cache in particular was storing the un-reduced exit
of every patch -- 197 x 1536 fp16 each -- to save a forward pass. The ONLY
thing persisted is the manifest, which is positions and no pixels.

`split='eval'` is a SEED, not a cache: `_render_row` derives its rng from the
row's own identity, so the same position always renders the same photo without
anything being stored.

Every split renders through `Camera` (`query_sim/camera.py`) via `_render_row`
-- ONE place that turns a `ManifestRow` into `(patch, label, native)`, so train
and eval cannot come to mean different pixels.

ONE ROW IS ONE PATCH. The camera's sensor is `RenderConfig.tile_size` square,
which is also the sampler's window and the encoder's input -- see the
`TILE_WH_RATIO` note below for why all three being one number is the point
rather than a simplification.
'''
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _dir in ('utilities', 'aiNNModel', '1_estimate_query_mpp', 'query_sim'):
    _path = str(_ROOT / _dir)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402
import torch.utils.data                                             # noqa: E402

from AccessDatasets import locate, list_names                        # noqa: E402
from SafeSlide import SafeSlide                                     # noqa: E402
from TissuesRegionsMask import TissuesRegionsMask                    # noqa: E402
from TileSampler import (OverlapConfig, RichnessConfig, SamplerConfig,  # noqa: E402
                         TileSampler)
from DsLadder import DEFAULT_RUNGS, DsLadder                        # noqa: E402
from camera import Camera                                            # noqa: E402
from config import DomainGapConfig                                   # noqa: E402


#: The label space every baseline classifies into -- rung VALUE, not a
#: per-WSI level index, so the class id means the same thing on BRACS (4x
#: pyramid) and Ki67 (2x pyramid) -- DsLadder.py's own stated reason for
#: existing.
RUNGS: Tuple[float, ...] = DEFAULT_RUNGS
RUNG_TO_CLASS: Dict[float, int] = {r: i for i, r in enumerate(RUNGS)}
NUM_CLASSES = len(RUNGS)

#: THE CAMERA'S SENSOR IS ONE TILE, not a whole field of view. Both this
#: package's sampler and its camera then measure the SAME window, which is the
#: single fact everything below depends on:
#:
#:     TileSampler   footprint_l0 = tile_size * rung     (DsLadder.plan)
#:     Camera        rect_w_l0    = output_w * query_mpp / base_mpp
#:                                = tile_size * rung     (query_mpp = base*rung)
#:
#: Rendering CLAUDE.md's real-photo frame (1440x1024, 45:32, 1.475 MPixels)
#: and cutting it up instead makes the two windows disagree by 5.6x: the
#: sampler certifies a `256*rung` window as tissue while the camera reads
#: `1440*rung`, so every coarse rung draws positions that `capture` then
#: refuses (off-slide) -- silently, because
#: `collate_routing_batch` drops a None. `DsLadder.reachable`'s own measured
#: numbers say where that lands: footprint 8192 sampled 100/100, 16384 sampled
#: 0/100, 32768 had no region that could hold it. At 1440 the ladder's top
#: three rungs are 11520/23040/46080 -- i.e. classes 3, 4 and 5 were quietly
#: near-empty. At 256 the top rung is 8192, the case measured 100/100, so all
#: six rungs exist.
#:
#: What this is NOT: a claim that a 256-px photograph is realistic. It is the
#: unit the ENCODER consumes either way -- a 1440x1024 frame reaches the model
#: only as 20 separate 256 patches -- so rendering the tile directly is the
#: same input by a shorter route, with two exceptions handled in
#: `_CameraBank.camera_for`.
TILE_WH_RATIO = '1:1'

#: How often the two FRAME-REFERENCED optics (vignette, lens distortion) are
#: present on a shot -- see `_CameraBank.camera_for` for why they cannot simply
#: be always-on at this sensor size, and `DomainGapConfig.vignette_p` for the
#: mechanism. 0.5 is a starting value, not a measured one: it says a tile is as
#: likely to have come from a frame's lit centre as from its darkened edge.
OPTICS_P = 0.5

#: This package's richness contract: `bg50_70` and `bg70_85` capped at 0 on top
#: of `RichnessConfig`'s own zeros for `bg85_95`/`bg95_100`, so NO tile above 50
#: per cent background is ever sampled. The label here is SCALE, and a mostly
#: blank field carries no scale cue at all -- it is not a hard example, it is an
#: unanswerable one, and at coarse rungs it is exactly what the supply degrades
#: into.
#:
#: READ THE COST BEFORE CHANGING n_per_rung. Those two buckets are where a
#: SHORTFALL goes: `RichnessConfig`'s own docstring records the 475/500 episode,
#: where caps that summed to exactly 1.0 left a short rung with nowhere to take
#: its remainder from and the shortage was read as a property of the slides.
#: These caps sum to exactly 1.0 again -- deliberately this time. With
#: `floor_frame='ask'` the consequence is that a rung which cannot supply 60 per
#: cent `bg30_50` comes back SHORT rather than being topped up with background,
#: so `n_per_rung` is a ceiling and not a promise.
#:
#: That is why the training loop prints a per-rung `trained on` line: the
#: shortfall is now a number on screen every epoch instead of an assumption.
RICHNESS = RichnessConfig(caps=(0.15, 0.25, 0.60, 0.0, 0.0, 0.0, 0.0))


# ══════════════════════════════════════════════════════════════════════════
#  manifest -- position identity only, no pixels, no Camera
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ManifestRow:
    '''One drawn position: which WSI, where, at which rung. Everything past
    this point (`Camera`, patches, labels) is DERIVED from a row, never
    stored on it -- same reason `RungPlan`/`CameraShot`/`SampleMeta` elsewhere
    in this repo separate "where" from "what got read there".'''
    dataset: str
    wsi_name: str
    x: int
    y: int
    rung: float

    #: `SampleMeta.bucket` -- which of `RichnessConfig`'s seven background-
    #: fraction buckets this position fell in (`bg00_15` ... `bg95_100`).
    #: Carried rather than recomputed because it is what the LATTICE honoured:
    #: the sampler's quotas are per bucket, so it is the axis along which a
    #: result can be stratified to ask "is the head only right on the busy
    #: tiles". Recomputing it later from the mask could disagree with the
    #: number that actually placed the tile.
    bucket: str = 'mid'

    @property
    def label(self) -> int:
        return RUNG_TO_CLASS[self.rung]


_MANIFEST_FIELDS = tuple(f.name for f in fields(ManifestRow))


def _atomic_write(path, write_body) -> Path:
    '''Write via a temp file in the SAME directory, then `os.replace` --
    atomic on POSIX, so a reader never sees a half-written file.

    Exists because two of this package's cache files (a manifest, `wsi_split.
    csv`) can legitimately be built by TWO PROCESSES that have never
    coordinated: `jobscripts/MppRoutingHead/MppRoutingHead.sh`'s `PARALLEL`
    mode runs baseline 2 and baseline 3 as separate `train.py` invocations
    sharing one `cache_root`, and both call `train_rows`/`val_rows`
    independently. On the first run ever, both can see the file missing and
    both build it -- deterministically, from the same seed, so the CONTENT
    they compute agrees. What a plain `open(path, 'w')` from each would risk
    is not disagreement, it is INTERLEAVING: two writers to one path can
    produce a file that is neither writer's version, and a reader mid-write
    sees a truncated one. `PID` in the temp name means two writers never
    collide on the temp file either, so the cost of the race is redundant
    computation, never a corrupt file.'''
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.name}.tmp{os.getpid()}')
    write_body(tmp)
    os.replace(tmp, path)
    return path


def split_wsi_names(dataset_id: str, n_val: int, *,
                    seed: int = 42) -> Tuple[List[str], List[str]]:
    '''`(val_names, test_names)` -- a WSI-LEVEL split, so no position from one
    slide can appear on both sides. The val half is the first `n_val` of a
    seeded shuffle.

    `sorted()` before the shuffle on purpose: `list_names` returns whatever
    order the registry happens to hold, so shuffling it directly would make
    the split depend on something nobody controls. Sorted first, the seed is
    the only input, and the same seed gives the same twenty slides on any
    machine.

    Write the answer down (`write_wsi_split`) rather than re-deriving it at
    each use. A split that exists only as "whatever `--seed 42` produces" is
    one library version away from silently becoming a different experiment.
    '''
    names = sorted(list_names(dataset=dataset_id))
    if not 0 < n_val < len(names):
        raise ValueError(f'n_val={n_val}: dataset {dataset_id!r} has '
                         f'{len(names)} WSIs, need 0 < n_val < that')
    random.Random(seed).shuffle(names)
    return names[:n_val], names[n_val:]


def write_wsi_split(val_names: Sequence[str], test_names: Sequence[str],
                    path) -> Path:
    '''`split,wsi_name` rows -- the record of which slides were held out. Row
    ORDER is part of the record: `split_wsi_names` shuffles before it splits,
    so a prefix of the `test` rows is already a random sample and callers take
    one instead of drawing again. Written atomically -- see `_atomic_write`.'''
    def body(tmp):
        with open(tmp, 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(('split', 'wsi_name'))
            writer.writerows(('val', n) for n in val_names)
            writer.writerows(('test', n) for n in test_names)
    return _atomic_write(path, body)


def read_wsi_split(path) -> Tuple[List[str], List[str]]:
    '''`(val_names, test_names)` as recorded, in file order.'''
    with open(Path(path), newline='') as fh:
        rows = list(csv.DictReader(fh))
    return ([r['wsi_name'] for r in rows if r['split'] == 'val'],
            [r['wsi_name'] for r in rows if r['split'] == 'test'])


def wsi_split(dataset_id: str, n_val: int, path, *,
              seed: int = 42) -> Tuple[List[str], List[str]]:
    '''The recorded split if `path` exists, a fresh one written there if not.

    EXISTING WINS, always. Re-deriving on every run would mean that adding or
    removing one slide from a dataset silently reshuffles which ten are held
    out -- and then checkpoints selected on the old val split get scored
    against a test set containing some of it, with nothing anywhere saying so.
    Deleting the file is how you ask for a new split, and doing that
    invalidates every checkpoint beside it.
    '''
    path = Path(path)
    if path.exists():
        return read_wsi_split(path)
    val_names, test_names = split_wsi_names(dataset_id, n_val, seed=seed)
    write_wsi_split(val_names, test_names, path)
    return val_names, test_names


def build_manifest(dataset_id: str, *, tile_size: int = 256,
                   rungs: Sequence[float] = RUNGS, n_per_rung: int = 100,
                   seed: int = 42, max_wsi: Optional[int] = None,
                   wsi_names: Optional[Sequence[str]] = None,
                   ) -> List[ManifestRow]:
    '''One row per drawn position. Over `wsi_names` when given (that is how the
    val/test halves of one dataset get separate manifests -- see
    `split_wsi_names`), otherwise over every WSI `list_names(dataset=
    dataset_id)` currently finds. Cheap -- `TileSampler.sample()` consults
    only the mask, no pixel reads -- so this is fast enough to rebuild
    whenever the sampling recipe changes rather than invalidated by hand.

    Identical for every split; `Camera` (with its own randomness / determinism
    concerns) only enters at render time, in `_render_row` below.
    '''
    rows: List[ManifestRow] = []
    names = list(wsi_names) if wsi_names is not None else list_names(dataset=dataset_id)
    if max_wsi is not None:
        names = names[:max_wsi]
    ladder = DsLadder(rungs=tuple(sorted(rungs)))
    for name in names:
        entry = locate(name, dataset=dataset_id)
        wsi = SafeSlide(entry.path)
        try:
            from TissueSegFunc import TissueSegConfig                # noqa: PLC0415
            mask = TissuesRegionsMask.from_wsi(
                wsi, method=TissueSegConfig('hsv').build())
            plans = ladder.plan_for(wsi, tile_size)
            cfg = SamplerConfig(tile=tile_size, n_per_rung=n_per_rung,
                                seed=seed, richness=RICHNESS,
                                overlap=OverlapConfig())
            sampler = TileSampler(wsi, mask, cfg)
            sampler.sample(plans)
            for s in sampler:
                rows.append(ManifestRow(dataset=dataset_id, wsi_name=name,
                                        x=int(s.meta.x), y=int(s.meta.y),
                                        rung=float(s.meta.ds),
                                        bucket=str(s.meta.bucket)))
        finally:
            wsi.close()
    return rows


def class_weights(rows: List[ManifestRow], device) -> torch.Tensor:
    '''Inverse-frequency weights for `F.cross_entropy(..., weight=...)`, from
    the TRAINING manifest's own rung distribution -- not per-epoch render
    counts. Those drop a handful of positions near a region edge and vary
    trivially epoch to epoch (see `_render_row`); the manifest is what the
    sampler was actually asked to produce, and it is the same for the whole
    run, so the weight is computed once and reused every epoch rather than
    recomputed from a moving target.

    sklearn's 'balanced' formula: `w_c = N / (K * n_c)`, so a class weight
    reflects how UNDER-supplied it is relative to an even K-way split, and
    `sum_c(n_c * w_c) = N` -- the total loss magnitude one epoch sees is
    unchanged, only its distribution across classes. This is the direct fix
    for the failure mode `HeadConfig`'s own docstring already names from the
    OTHER direction (fp16-under-Adam): even with a correct, finite loss,
    `RICHNESS`'s own shortfall at coarse rungs (spec.md: "n_per_rung is a
    ceiling, not a promise") means rung 32 supplies roughly 1 percent of what
    rung 1 does, so an unweighted mean lets the common rungs' gradient drown
    the rare ones out -- which is what the 2026-09-16 full run's val numbers
    showed (near-perfect accuracy wherever the true class was the majority
    rung, near-zero everywhere else).

    A rung with ZERO training positions gets weight 0, not `inf`: there is
    nothing to reweight if nothing was ever sampled for it, and
    `cross_entropy` will never see that class index as a target anyway.
    '''
    counts = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    for row in rows:
        counts[row.label] += 1
    total = float(counts.sum())
    if total <= 0:
        raise ValueError('class_weights: rows is empty, nothing to weight')
    weights = torch.where(counts > 0,
                          total / (NUM_CLASSES * counts.clamp_min(1.0)),
                          torch.zeros_like(counts))
    return weights.float().to(device)


def manifest_parts(*, dataset_id: str, tile_size: int, n_per_rung: int,
                   seed: int, rungs: Sequence[float] = RUNGS,
                   n_wsi: Optional[int] = None,
                   max_wsi: Optional[int] = None,
                   wsi_names: Optional[Sequence[str]] = None) -> List[str]:
    '''Everything a manifest's content depends on, as the `key=value` strings
    `ConfigIdentity.short_id` hashes.

    `richness` is spelled out BY VALUE, not by class name. It was by name while
    this package used `RichnessConfig()` unchanged; `RICHNESS` now caps
    `bg50_70`/`bg70_85` at 0, and a key that said only "RichnessConfig" would
    hash the same before and after that change -- i.e. it would hand back the
    old manifest for the new contract. `overlap` stays by name because
    `OverlapConfig()` really is the default here; that line has to grow the
    same way if it ever stops being.

    `seg=hsv` for the same reason: `build_manifest` hardcodes
    `TissueSegConfig('hsv')`, and a different mask is a different set of
    positions.
    '''
    return [
        f'dataset={dataset_id}',
        f'tile={int(tile_size)}',
        f'rungs={",".join(f"{float(r):g}" for r in rungs)}',
        f'n_per_rung={int(n_per_rung)}',
        f'seed={int(seed)}',
        f'n_wsi={n_wsi}',
        f'max_wsi={max_wsi}',
        # NOT the literal string "all". `list_names(dataset=dataset_id)` is
        # read and hashed here, not just recorded as a name for the default
        # case -- a manifest is reused whenever its hash exists, so if this
        # said "all" as a constant, a WSI added or removed from the dataset
        # (as happened on 2026-09-16: five holed slides deleted from
        # ki67_pure) would leave the hash UNCHANGED, and a cache built before
        # the deletion would go on being served after it, silently, with
        # positions on slides that no longer exist on disk.
        f'wsi={",".join(wsi_names) if wsi_names is not None else ",".join(sorted(list_names(dataset=dataset_id)))}',
        'seg=hsv',
        f'richness_scorer={RICHNESS.scorer}',
        f'richness_edges={",".join(f"{e:g}" for e in RICHNESS.edges)}',
        f'richness_floors={",".join(f"{f:g}" for f in RICHNESS.floors)}',
        f'richness_caps={",".join(f"{c:g}" for c in RICHNESS.caps)}',
        f'richness_frames={RICHNESS.bucket_frame},{RICHNESS.floor_frame}',
        'overlap=OverlapConfig()',
    ]


def manifest_path(ddir, split: str, parts: Sequence[str]) -> Path:
    '''`<ddir>/<split>_<8 hex>.csv`, with a `.json` beside it saying what the
    eight characters cover.

    A HASH, not the parameters spelled out. The readable-name version --
    `test_w5_r50.csv` -- keyed on the two flags that happened to be on the
    command line and silently reused a stale manifest when `--tile` or `--seed`
    changed, which are just as much part of what the positions are. A name that
    is ALMOST a key is worse than either a full one or an opaque one, because
    it looks like it is protecting you.

    The sidecar is what keeps the hash from being the opaque kind CLAUDE.md
    objects to: `ls` shows eight characters, and the JSON next to it says
    exactly which eleven facts they stand for.
    '''
    from ConfigIdentity import short_id                            # noqa: PLC0415
    ddir = Path(ddir)
    stem = f'{split}_{short_id(list(parts))}'
    return ddir / f'{stem}.csv'


def write_manifest_key(path, parts: Sequence[str]) -> Path:
    '''The sidecar for `manifest_path`. Written next to the CSV, same stem,
    atomically -- see `_atomic_write`.'''
    path = Path(path).with_suffix('.json')
    def body(tmp):
        with open(tmp, 'w') as fh:
            json.dump(list(parts), fh, indent=2)
    return _atomic_write(path, body)


def write_manifest(rows: List[ManifestRow], path) -> Path:
    '''Written atomically -- see `_atomic_write`.'''
    def body(tmp):
        with open(tmp, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=_MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(row.__dict__ for row in rows)
    return _atomic_write(path, body)


def read_manifest(path) -> List[ManifestRow]:
    with open(path, newline='') as fh:
        return [ManifestRow(dataset=r['dataset'], wsi_name=r['wsi_name'],
                            x=int(r['x']), y=int(r['y']), rung=float(r['rung']),
                            bucket=r['bucket'])
               for r in csv.DictReader(fh)]


# ══════════════════════════════════════════════════════════════════════════
#  rendering: one ManifestRow -> (patches, label), the ONE place this happens
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RenderConfig:
    '''What every render needs besides the row itself -- one object threaded
    through `_CameraBank`/`RoutingHeadDataset`/`iterate_epoch` instead of
    separate keyword arguments repeated at each of them (the first draft's
    shape: easy for one call site to pass an inconsistent value and nothing
    would catch it).

    `tile_size` is now ONE number doing ONE job -- the camera's sensor side,
    the sampler's window, and the encoder's input are all it (see the
    TILE_WH_RATIO note above). It used to do two unrelated jobs at once:
    `build_manifest`'s sampler window AND the patch side a rendered frame was
    cut into, which is how the two could be 5.6x apart without anything
    saying so.

    `stage_shift_max=0` turns OFF the mechanical stage-jitter augmentation
    (`DomainGapConfig`'s own default is 3). Two independent reasons, and the
    first one stands on its own:

    1. This package's label is SCALE. A +/-3 px translation carries no scale
       information, so the augmentation buys this task nothing -- it is not
       a diversity source worth having here.
    2. It also avoids a real bug in `query_sim/pipeline.py`: `_sample_params`
       draws `stage_shift_dx/dy` from the caller's `rng` and records them in
       `params`, but `_apply_params` (pipeline.py:158) calls
       `apply_stage_shift(img, max_shift=cfg.stage_shift_max)` -- which
       ignores them and draws its own pair from the GLOBAL `np.random`
       (`augment/field.py:107-108`). The recorded ground truth is therefore
       not what was applied, and the applied shift is not reachable by any
       `rng` a caller passes -- which would break `_render_row`'s
       `deterministic=True` guarantee for the eval split. At 0 all three
       layers agree (the ternary at pipeline.py:46-47, the guard at :153,
       and field.py:103's own early return), so no divergence is possible.

    Turning it off here does NOT fix that bug for anything else --
    `query_sim/generator.py` still writes the mismatched value into every
    synthetic corpus's GT CSV. Flagged, not fixed, 2026-09-16.
    '''
    tile_size: int = 256
    stage_shift_max: int = 0

    @property
    def mpixels(self) -> float:
        '''What `DomainGapConfig.MPixels` has to be for `QueryFromWSI` to
        arrive back at `tile_size`: it computes `output_w = int(sqrt(MPixels *
        1e6 / (w_r * h_r)) * w_r)`, which at 1:1 is `int(sqrt(MPixels*1e6))`.
        Derived rather than typed so the two cannot drift; `_CameraBank`
        asserts the round trip anyway, because `int()` on a float square root
        is exactly the kind of step that lands one px low without complaint.'''
        return (self.tile_size ** 2) / 1e6


class _CameraBank:
    '''Lazily-built `{(dataset, wsi_name, rung): Camera}` cache, one `SafeSlide`
    handle opened per WSI (not per rung -- the six rungs of one WSI share it;
    `Camera` accepts an already-open handle and does not reopen). Built fresh
    inside whichever process first uses it -- a `RoutingHeadDataset` under
    `DataLoader(num_workers>0)` gets one `_CameraBank` PER WORKER this way,
    never one shared across a fork/spawn boundary (an `openslide` handle is
    not safely shared across processes; see spec.md's "Camera: train vs
    eval"). Never evicts -- see spec.md's "Handle count": the caller bounds
    how many WSIs one bank ever sees by bounding `rows`, not by eviction here.

    `query_mpp` is baked into a `Camera` at construction (`DomainGapConfig.
    query_mpp` -> `QueryFromWSI(..., mpp=query_mpp)`, fixed for that Camera's
    lifetime) -- so a Camera is keyed by rung as well as WSI, not just WSI:
    one WSI's six rungs are six different microscopes, in `Camera`'s own
    terms.
    '''

    def __init__(self, cfg: RenderConfig):
        self.cfg = cfg
        self._wsi: Dict[Tuple[str, str], SafeSlide] = {}
        self._camera: Dict[Tuple[str, str, float], Camera] = {}

    def _wsi_for(self, dataset_id: str, wsi_name: str) -> SafeSlide:
        key = (dataset_id, wsi_name)
        wsi = self._wsi.get(key)
        if wsi is None:
            entry = locate(wsi_name, dataset=dataset_id)
            wsi = SafeSlide(entry.path)
            self._wsi[key] = wsi
        return wsi

    def camera_for(self, dataset_id: str, wsi_name: str, rung: float) -> Camera:
        key = (dataset_id, wsi_name, rung)
        cam = self._camera.get(key)
        if cam is None:
            wsi = self._wsi_for(dataset_id, wsi_name)
            gap_cfg = DomainGapConfig(
                wh_ratio=TILE_WH_RATIO,
                MPixels=self.cfg.mpixels,
                query_mpp=wsi.base_mpp * rung,
                stage_shift_max=self.cfg.stage_shift_max,
                # THE TWO FRAME-REFERENCED OPS, PRESENT ONLY SOMETIMES. Every
                # other op in `query_sim/augment/` is per-pixel or local
                # (colour, colour temperature, brightness/contrast, defocus,
                # chromatic shift, noise, JPEG) or a scene transform (rotation,
                # scale), and so means the same thing whatever the frame is.
                # These two do not: `apply_vignette`'s falloff and
                # `apply_distortion`'s k1 are both normalised to the SENSOR's
                # half-width (`pipeline._apply_params` says so, and records the
                # episode where taking them from the oversized read left the
                # photo seeing only the central 53% of the falloff curve).
                #
                # The sensor here is one tile, so a tile always carrying a
                # complete centred vignette would be a cue the deployed input
                # does not have: a real photograph's tiles are slices of ONE
                # vignette, and a tile from the centre of the frame has none at
                # all. A probability is what models both kinds -- widening the
                # strength range only ever models the first, more weakly.
                vignette_p=OPTICS_P,
                distortion_p=OPTICS_P,
            )
            # seed=None (the default): determinism, where wanted, is handled
            # per-call via capture(..., rng=...) in _render_row -- a fixed
            # Camera-level seed here would apply to every caller alike and
            # give neither train nor eval what it actually needs.
            cam = Camera(wsi, cfg=gap_cfg)
            got = (cam.qfw.output_w, cam.qfw.output_h)
            if got != (self.cfg.tile_size, self.cfg.tile_size):
                raise RuntimeError(
                    f'sensor is {got[0]}x{got[1]}, not '
                    f'{self.cfg.tile_size}x{self.cfg.tile_size}: '
                    f'RenderConfig.mpixels did not round-trip through '
                    f'QueryFromWSI. Everything downstream assumes the camera, '
                    f'the sampler window and the encoder input are one number')
            self._camera[key] = cam
        return cam


def _eval_seed(row: ManifestRow) -> int:
    '''Deterministic, stable ACROSS PROCESSES (unlike Python's built-in
    `hash()`, which is randomised per-process for strings unless
    `PYTHONHASHSEED` is pinned) -- derived from the sample's own identity,
    not from worker id or call order, so the same query always renders the
    same photo regardless of which DataLoader worker handles it or when.'''
    key = f'{row.dataset}|{row.wsi_name}|{row.x}|{row.y}|{row.rung}'
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def _render_row(bank: _CameraBank, row: ManifestRow, cfg: RenderConfig, *,
                deterministic: bool) -> Optional[Tuple[np.ndarray, int, bool]]:
    '''The one place a `ManifestRow` becomes pixels: `Camera.capture` (fresh
    randomness if `deterministic=False`, `train`'s own case; seeded from the
    row's identity if `deterministic=True`, `eval`'s case -- see spec.md's
    "Camera: train vs eval"). One row is ONE patch, `[tile, tile, 3]` uint8:
    the camera's sensor IS the tile, so there is nothing left to cut up.

    `None` if the read ran off the slide, same as `Camera.capture` returns.
    That is now a rare, per-position event rather than a whole rung's worth:
    the sampler certified `tile*rung` and the camera reads at most the
    rotation bounding square around it, `ceil(sqrt(2)) * tile * rung`, so only
    positions near a region edge fail. `collate_routing_batch` drops them, and
    the training loop's per-rung `trained on` line is where the total shows up.

    The third element is `QueryFromWSI.reads_natively`: whether this rung came
    off a pyramid level at the requested mpp, or off a finer level that was
    then LANCZOS-resampled down. Carried per example rather than derived later,
    because it is a property of (slide pyramid, rung) and the camera is the only
    thing that already knows it. `score()` splits its accuracy on it -- on a
    2x pyramid (Ki67) every rung is native, on a 4x one (BRACS) the odd rungs
    are resampled, so the extra resampling's signature correlates with the
    CLASS instead of averaging out, and a head can score on it rather than on
    scale.

    Reached only through `RoutingHeadDataset.__getitem__`, for every split --
    the first draft of this file wrote this logic out twice, once for training
    and once for building a cached eval corpus; this is the one copy.
    '''
    cam = bank.camera_for(row.dataset, row.wsi_name, row.rung)
    rng = random.Random(_eval_seed(row)) if deterministic else None
    patch = cam.capture(row.x, row.y, rng=rng)
    if patch is None:
        return None
    return patch, row.label, bool(cam.qfw.reads_natively)


# ══════════════════════════════════════════════════════════════════════════
#  the Dataset -- every split, rendered live, nothing written to disk
# ══════════════════════════════════════════════════════════════════════════

class RoutingHeadDataset(torch.utils.data.Dataset):
    '''One manifest row -> one rendered patch, via `_render_row`.

    `split='train'` renders with fresh randomness every access, which is the
    augmentation and the point of it. `split='eval'` renders deterministically
    -- `_render_row` seeds from the row's own identity -- which is what a val
    curve and a reported test number both need. Same class, same render path,
    same everything else: the only difference is where the rng comes from.

    `rows` is normally ONE WSI-BATCH's worth of manifest rows (from
    `group_by_wsi`), not the whole manifest -- see `iterate_epoch` below and
    spec.md's "Handle count: WSI-batching, not LRU eviction". Nothing in
    this class enforces that; it iterates whatever `rows` it is given, and
    the bound on how many WSIs get touched (and so how many `Camera`/
    `SafeSlide` handles this instance's `_CameraBank` ends up holding) comes
    entirely from how big a `rows` list `iterate_epoch` hands it.

    Returns `None` for a position whose read runs off the slide --
    `collate_routing_batch` drops these.
    '''

    def __init__(self, rows: List[ManifestRow], *, split: str = 'train',
                cfg: RenderConfig = RenderConfig()):
        if split not in ('train', 'eval'):
            raise ValueError(f"split must be 'train' or 'eval', got {split!r}")
        self.rows = rows
        self.split = split
        self.cfg = cfg
        # Built lazily in __getitem__, NOT here -- __init__ runs in the main
        # process before DataLoader forks/spawns workers, and a _CameraBank
        # holds open WSI handles that must not cross that boundary.
        self._bank: Optional[_CameraBank] = None

    def __len__(self) -> int:
        return len(self.rows)

    def _bank_for_this_process(self) -> _CameraBank:
        if self._bank is None:
            self._bank = _CameraBank(self.cfg)
        return self._bank

    def __getitem__(self, i: int):
        '''`(row, patch, label, native)`, or `None` if the read ran off the
        slide. The ROW rides along because `iterate_epoch` shuffles and
        regroups: after that, batch position says nothing about which manifest
        line produced an example, and a per-tile prediction CSV needs
        `wsi_name`/`x`/`y`/`bucket` back. `ManifestRow` is a frozen dataclass
        of plain scalars, so it pickles across the worker boundary as it is.'''
        row = self.rows[i]
        bank = self._bank_for_this_process()
        rendered = _render_row(bank, row, self.cfg,
                               deterministic=self.split == 'eval')
        return None if rendered is None else (row, *rendered)


def collate_routing_batch(items):
    '''`items`: a list of `(row, patch[tile,tile,3], label, native)` or `None`
    (dropped off-slide positions). One patch, one label, one example -- so the
    tensors here are a stack and nothing more.

    Nothing downstream regroups patches: `AttentionPoolHead` pools over one
    tile's own TOKENS (`grid_view`), never across tiles.

    `rows` stays a LIST of `ManifestRow`, not a tensor: it is identity
    (`wsi_name`, `bucket`) rather than a quantity, and `default_collate` would
    have turned each field into its own parallel sequence, which is the shape
    `_ShardWriter` existed to avoid.

    `native` is not used by the loss. It rides along so that a run can report
    the composition of what it actually saw (`_render_row` says what it is);
    the training set being all-native is a fact worth being able to check
    rather than assume from the pyramid.
    '''
    items = [it for it in items if it is not None]
    if not items:
        raise RuntimeError('collate_routing_batch: every item in this batch '
                           'ran off its slide -- nothing to train on')
    rows, patches, labels, native = zip(*items)
    return dict(
        rows=list(rows),                                  # [B] ManifestRow
        patches=torch.from_numpy(np.stack(patches)),      # [B, tile, tile, 3]
        labels=torch.tensor(labels, dtype=torch.int64),   # [B]
        native=torch.tensor(native, dtype=torch.bool),    # [B]
    )


# ══════════════════════════════════════════════════════════════════════════
#  Handle count: WSI-batching, not LRU eviction
# ══════════════════════════════════════════════════════════════════════════
#
# A `_CameraBank` never evicts -- it does not need to, as long as whoever
# builds one only ever hands it rows drawn from a BOUNDED number of WSIs.
# That bound is structural, not a cache-size guess: `iterate_epoch` gives
# each WSI-batch its OWN `RoutingHeadDataset` (and so its own `_CameraBank`
# per worker) and its OWN `DataLoader`, exhausts it, and lets it go out of
# scope before building the next one -- workers exit, `SafeSlide`/`Camera`
# handles close with them, and the next WSI-batch starts from zero. At any
# moment, at most `wsi_group_size` WSIs are open per worker, full stop.

def group_by_wsi(rows: List[ManifestRow], group_size: int, *,
                 seed: Optional[int] = None) -> List[List[ManifestRow]]:
    '''Chunk manifest rows into groups of `group_size` WSIs each -- every row
    for one WSI (all its rungs) stays in the SAME group, so that group's own
    `RoutingHeadDataset`/`DataLoader` never touches more than `group_size`
    distinct WSIs.

    `seed` reshuffles which WSIs land in which group, and the groups' own
    order -- `iterate_epoch` passes a different one each epoch (`epoch_seed`)
    so the same WSIs are not always batched together, which is what recovers
    some of the cross-WSI diversity a single global shuffle would have given
    each mini-batch (see spec.md's "Handle count" section for the trade-off
    this is managing).
    '''
    by_wsi: Dict[Tuple[str, str], List[ManifestRow]] = {}
    for row in rows:
        by_wsi.setdefault((row.dataset, row.wsi_name), []).append(row)
    keys = list(by_wsi.keys())
    if seed is not None:
        random.Random(seed).shuffle(keys)
    return [
        [row for key in keys[start:start + group_size] for row in by_wsi[key]]
        for start in range(0, len(keys), group_size)
    ]


def iterate_epoch(rows: List[ManifestRow], *, wsi_group_size: int,
                  batch_size: int, num_workers: int = 0, split: str = 'train',
                  cfg: RenderConfig = RenderConfig(),
                  epoch_seed: Optional[int] = None) -> Iterator[dict]:
    '''One epoch: every row in `rows` trained on exactly once, WSI-batch by
    WSI-batch. Each group gets its own `RoutingHeadDataset` + `DataLoader`,
    which is exhausted and torn down (workers exit, handles close) before
    the next group's `DataLoader` is built -- the handle bound is structural
    (see the section above), no eviction logic anywhere.

    `epoch_seed` should differ every epoch (the caller's training loop
    passes e.g. the epoch number) -- see `group_by_wsi`.
    '''
    for group_rows in group_by_wsi(rows, wsi_group_size, seed=epoch_seed):
        dataset = RoutingHeadDataset(group_rows, split=split, cfg=cfg)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=collate_routing_batch,
            persistent_workers=False)
        yield from loader
