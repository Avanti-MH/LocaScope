'''Training/eval data for MppRoutingHead. See spec.md's "Camera: train vs
eval", "Caching" and "Handle count" sections for the reasoning; this file is
the implementation of all three.

    rows = build_manifest('ki67_pure', masks=MaskMaker(MASK_RECIPES['hest'], job),
                          draw_job=job, n_per_rung=100)   # positions only

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
things persisted are the per-slide mask and draw (`TileSampler.cached`), which
are positions and no pixels.

`split='eval'` is a SEED, not a cache: `render_row` derives its rng from the
row's own identity, so the same position always renders the same photo without
anything being stored.

Every split renders through `Render` (`query_sim/camera.py`) via `render_row`
-- ONE place that turns a `ManifestRow` into `(patch, label, native)`, so train
and eval cannot come to mean different pixels.

ONE ROW IS ONE PATCH. The camera's sensor is `RenderConfig.tile_size` square,
which is also the sampler's window and the encoder's input -- see the
sensor note below for why all three being one number is the point
rather than a simplification.
'''
from __future__ import annotations

import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402
import torch.utils.data                                             # noqa: E402

from AccessDatasets import locate, list_names, pick_wsi_names        # noqa: E402
from SafeSlide import SafeSlide                                     # noqa: E402
from Cache import job_name                                           # noqa: E402
from TileSampler import PlanSpec, TileSampler                       # noqa: E402
from DsLadder import DEFAULT_RUNGS                                  # noqa: E402
from ReadGeometry import finer_levels                               # noqa: E402
from TissueMaskConfig import MASK_RECIPES, MaskMaker                 # noqa: E402
from WsiSplit import SPLIT_JOB                                       # noqa: E402
from camera import Render, photo_rng, render_spec                    # noqa: E402
from SlideReader import SlideReader                                  # noqa: E402
from FovSupply import FOV_RECIPES                                    # noqa: E402


#: The label space every baseline classifies into -- rung VALUE, not a
#: per-WSI level index, so the class id means the same thing on BRACS (4x
#: pyramid) and Ki67 (2x pyramid) -- DsLadder.py's own stated reason for
#: existing.
RUNGS: Tuple[float, ...] = DEFAULT_RUNGS
RUNG_TO_CLASS: Dict[float, int] = {r: i for i, r in enumerate(RUNGS)}
NUM_CLASSES = len(RUNGS)

# THE CAMERA'S SENSOR IS ONE TILE, not a whole field of view. Both this
# package's sampler and its camera then measure the SAME window, which is the
# single fact everything below depends on:
#
#     TileSampler   footprint_l0 = tile_size * rung     (DsLadder.plan)
#     Render        rect_w_l0    = output_w * ds
#                                = tile_size * rung     (Render(..., ds=rung))
#
# Rendering the real-photo frame (`REAL_PHOTO_SENSOR`, 1440x1024)
# and cutting it up instead makes the two windows disagree by 5.6x: the
# sampler certifies a `256*rung` window as tissue while the camera reads
# `1440*rung`, so every coarse rung draws positions that `capture` then
# refuses (off-slide) -- silently, because
# `collate_routing_batch` drops a None. `DsLadder.reachable`'s own measured
# numbers say where that lands: footprint 8192 sampled 100/100, 16384 sampled
# 0/100, 32768 had no region that could hold it. At 1440 the ladder's top
# three rungs are 11520/23040/46080 -- i.e. classes 3, 4 and 5 were quietly
# near-empty. At 256 the top rung is 8192, the case measured 100/100, so all
# six rungs exist.
#
# What this is NOT: a claim that a 256-px photograph is realistic. It is the
# unit the ENCODER consumes either way -- a 1440x1024 frame reaches the model
# only as 20 separate 256 patches -- so rendering the tile directly is the
# same input by a shorter route, with two exceptions handled in
# `CameraBank.camera_for`. `CameraBank` builds every Render with the sensor
# `(tile_size, tile_size)`.

#: The two camera templates and the draw are `FovSupply.FOV_RECIPES`
#: 'routing-query' -- every query tile: the full gap, the two frame-referenced
#: optics present half the time, no stage shift -- and 'routing-support-native'
#: -- `--support-native`: rotation only, every photometric channel at its no-op,
#: which is what a stage-1 reference tile read straight off the WSI looks like.
#: Both share one sampler and DsLadder's six rungs. `CameraBank.camera_for`
#: replaces only `stage_shift_max` (from `RenderConfig`); the sensor is the
#: tile, `(tile_size, tile_size)`.
#:
#: The sampler's caps sum to exactly 1.0, so with `floor_frame='ask'` a rung
#: that cannot supply 60 per cent `bg30_50` comes back SHORT rather than being
#: topped up with background: `n_per_rung` is a ceiling and not a promise, and
#: the training loop prints a per-rung `trained on` line so the shortfall is a
#: number on screen.
QUERY_RECIPE = 'routing-query'
SUPPORT_NATIVE_RECIPE = 'routing-support-native'


def routing_camera(tile_size: int):
    """The `ReadSpec` every manifest row is rendered with: a tile-sized
    sensor under the routing heads' camera. The query and the native support
    templates both rotate, so they read the same square and one spec covers
    both -- checked, so a template that stopped rotating could not leave the
    manifest reserving the wrong read."""
    full = render_spec(FOV_RECIPES[QUERY_RECIPE].gap, (tile_size, tile_size))
    native = render_spec(FOV_RECIPES[SUPPORT_NATIVE_RECIPE].gap,
                         (tile_size, tile_size))
    if full != native:
        raise RuntimeError(f'the two camera templates read differently ({full} '
                           f'vs {native}); a manifest serves both, so they must not')
    return full


#: The code between a manifest row and its tile in this package (`render_row`,
#: `CameraBank`, `choose_read_level`) -- ConfigIdentity rule 3.
DATA_VERSION = 0


def data_record(tile_size: int, seg: str) -> Dict:
    """What a routing head's tiles and labels are made of, apart from WHICH
    slides and how many positions: the sampler's distribution (its config
    without `n_per_rung` and `seed`), the plan, both camera templates, the
    rungs, the mask recipe, every VERSION on the way and the environment.
    A checkpoint carries the one it was trained on (`extra['data']`) and
    `cli/evaluate.py` refuses to score it on data whose record differs, so a
    changed render cannot be read as a changed model."""
    from ConfigIdentity import enc, environment, parts_of, short_id, versions_of
    camera_full = FOV_RECIPES[QUERY_RECIPE].gap
    camera_native = FOV_RECIPES[SUPPORT_NATIVE_RECIPE].gap
    sampler = FOV_RECIPES[QUERY_RECIPE].sampler
    mask = MASK_RECIPES[seg]
    plan = PlanSpec('ladder', tuple(RUNGS), camera=routing_camera(tile_size))
    parts = ([f'sampler.{p}' for p in parts_of(sampler, exclude=('n_per_rung', 'seed'))]
             + [f'plan={enc(plan.key())}', f'rungs={enc(RUNGS)}',
                f'camera_full={enc(camera_full.identity_id())}',
                f'camera_native={enc(camera_native.identity_id())}',
                f'seg_id={enc(mask.seg_id())}', f'region_id={enc(mask.region_id())}',
                f'version={enc(DATA_VERSION)}'])
    versions = {}
    for owner in (sampler, mask, camera_full, SlideReader):
        versions.update(versions_of(owner))
    return {'id': short_id(parts), 'parts': parts,
            'versions': dict(sorted(versions.items())), 'upstream': {},
            'env': environment()}


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

    #: `SampleMeta.footprint_l0` -- the level-0 side this position covers.
    #: Carried for the same reason as `bucket`: it is what the sampler placed,
    #: and an overlap test between two positions (`Episodes`' support/query
    #: rule) needs it without re-deriving a rung's footprint from a tile size
    #: that might not be the one this row was cut at.
    footprint_l0: int = 0

    @property
    def label(self) -> int:
        return RUNG_TO_CLASS[self.rung]


def add_cache_args(ap) -> None:
    """The four flags that say where positions come from -- the same four in
    every CLI that draws a manifest, so they cannot drift apart."""
    ap.add_argument('--seg', choices=sorted(MASK_RECIPES), default='hest',
                    help='tissue-mask recipe the positions are drawn inside '
                         '(TissueMaskConfig.MASK_RECIPES)')
    ap.add_argument('--mask-cache-job', default=None,
                    help='whose cache the masks are read from and written to: '
                         'result/cache/<this>/. Default: this job')
    ap.add_argument('--draw-cache-job', default=None,
                    help='whose cache the draws are read from and written to: '
                         'result/cache/<this>/. Default: this job')
    ap.add_argument('--split-cache-job', default=None,
                    help=f'whose val/test split to read: result/cache/<this>/. '
                         f'Default: {SPLIT_JOB}. Written only by make_split.py')


@dataclass
class Caches:
    """What `add_cache_args` resolved to. `masks` owns the segmenter: call
    `masks.close()` once the manifests are drawn, or it stays on the card."""
    masks: MaskMaker
    draw_job: str
    split_job: str


def open_caches(args, job: str, device) -> Caches:
    """`job` is the calling package's default job name. Each cache defaults to
    THIS job's (`Cache.job_name`, SLURM_JOB_NAME first); the split defaults to
    its one writer's."""
    made_by = job_name(job)
    return Caches(
        masks=MaskMaker(MASK_RECIPES[args.seg], args.mask_cache_job or made_by,
                        device),
        draw_job=args.draw_cache_job or made_by,
        split_job=args.split_cache_job or SPLIT_JOB)


def cache_jobs(args, job: str, caches: Caches) -> dict:
    """The three `--*-cache-job` flags as `open_caches` RESOLVED them. An unset
    flag is None in `vars(args)`, which names no cache and is nothing to filter a
    wandb config on; this is the job it actually read."""
    return dict(mask_cache_job=caches.masks.made_by,
                draw_cache_job=caches.draw_job,
                split_cache_job=caches.split_job)


def build_manifest(dataset_id: str, *, masks, draw_job: str,
                   report_dir=None, tile_size: int = 256,
                   rungs: Sequence[float] = RUNGS, n_per_rung: int = 100,
                   seed: int = 42, max_wsi: Optional[int] = None,
                   wsi_names: Optional[Sequence[str]] = None,
                   ) -> List[ManifestRow]:
    '''One row per drawn position. Over `wsi_names` when given (that is how the
    val/test halves of one dataset get separate manifests -- see
    `WsiSplit`), otherwise over every WSI `list_names(dataset=
    dataset_id)` currently finds.

    Every slide goes through `TileSampler.cached`: a slide already drawn under
    this recipe, config and plan is read back from `draw_job`'s cache without
    being opened; one that is not is opened, its mask taken from `masks` (a
    `TissueMaskConfig.MaskMaker`: the recipe, the device and the mask cache),
    and its draw written back. There is no manifest-level cache: the
    per-slide one is finer (a different WSI list or `max_wsi` still reuses
    every slide it shares) and it is the same cache every other sampler uses.

    The recipe HAS NO DEFAULT. The caller names one (`MASK_RECIPES[--seg]`)
    and owns the MaskMaker, so a segmentation model loaded for a miss is
    dropped when the caller's `with` block ends rather than staying on the
    card through training.

    `report_dir`, when given, receives one `sampler_report_<slide>.md` and
    `samples_<slide>.csv` per slide.

    Identical for every split; `Camera` (with its own randomness / determinism
    concerns) only enters at render time, in `render_row` below.
    '''
    rows: List[ManifestRow] = []
    names = list(wsi_names) if wsi_names is not None else list_names(dataset=dataset_id)
    names = pick_wsi_names(names, max_wsi, seed)
    cfg = replace(FOV_RECIPES[QUERY_RECIPE].sampler,
                  n_per_rung=n_per_rung, seed=seed)
    # The camera that renders every row: its footprint is the tile, and the
    # sampler reserves exactly what it reads, so no position is offered that
    # the Camera cannot read.
    plan = PlanSpec('ladder', tuple(rungs), camera=routing_camera(tile_size))
    for name in names:
        sampler = TileSampler.cached(
            locate(name, dataset=dataset_id).path, cfg, plan, draw_job,
            masks=masks, report_dir=report_dir)
        for s in sampler:
            rows.append(ManifestRow(dataset=dataset_id, wsi_name=name,
                                    x=int(s.meta.x), y=int(s.meta.y),
                                    rung=float(s.meta.ds),
                                    bucket=str(s.meta.bucket),
                                    footprint_l0=int(s.meta.footprint_l0)))
    return rows


def class_weights(rows: List[ManifestRow], device) -> torch.Tensor:
    '''Inverse-frequency weights for `F.cross_entropy(..., weight=...)`, from
    the TRAINING manifest's own rung distribution -- not per-epoch render
    counts. Those drop a handful of positions near a region edge and vary
    trivially epoch to epoch (see `render_row`); the manifest is what the
    sampler was actually asked to produce, and it is the same for the whole
    run, so the weight is computed once and reused every epoch rather than
    recomputed from a moving target.

    sklearn's 'balanced' formula: `w_c = N / (K * n_c)`, so a class weight
    reflects how UNDER-supplied it is relative to an even K-way split, and
    `sum_c(n_c * w_c) = N` -- the total loss magnitude one epoch sees is
    unchanged, only its distribution across classes. This is the direct fix
    for the failure mode `HeadConfig`'s own docstring already names from the
    OTHER direction (fp16-under-Adam): even with a correct, finite loss,
    the routing sampler's own shortfall at coarse rungs (spec.md: "n_per_rung is a
    ceiling, not a promise") means rung 32 supplies roughly 1 percent of what
    rung 1 does, so an unweighted mean lets the common rungs' gradient drown
    the rare ones out (near-perfect accuracy wherever the true class was the majority
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


# ══════════════════════════════════════════════════════════════════════════
#  rendering: one ManifestRow -> (patches, label), the ONE place this happens
# ══════════════════════════════════════════════════════════════════════════

READ_LEVELS = ('pyramid', 'resampled', 'mixed')
RESAMPLE_FROM = ('finer', 'l0')


@dataclass(frozen=True)
class RenderConfig:
    '''What every render needs besides the row itself -- one object threaded
    through `CameraBank`/`RoutingHeadDataset`/`iterate_epoch` instead of
    separate keyword arguments repeated at each of them, so no call site can
    pass an inconsistent value.

    `tile_size` is ONE number doing ONE job -- the camera's sensor side,
    the sampler's window, and the encoder's input are all it (see the
    sensor note above).

    `stage_shift_max=0` turns OFF the mechanical stage-jitter augmentation
    (`DomainGapConfig`'s own default is 3): this package's label is SCALE,
    and a +/-3 px translation carries no scale information.
    '''
    tile_size: int = 256
    stage_shift_max: int = 0

    #: WHICH PYRAMID LEVEL A TRAINING TILE IS READ FROM. See spec.md's 詞彙
    #: (read_level, resample_from, max_resample_factor, resampled_share) and
    #: `choose_read_level`.
    #:
    #:     pyramid    the nearest-level rule: a
    #:                rung with a native level reads it, one without (BRACS's
    #:                2 and 8) reads the next finer level and resamples
    #:     resampled  every rung that has a finer level reads one of them,
    #:                chosen uniformly, and resamples down
    #:     mixed      `resampled` for a share of the tiles, `pyramid` otherwise
    #:
    #: Training only: val and test keep `pyramid`, so every score stays
    #: comparable.
    #:
    #: THESE FOUR DEFAULTS ARE FROZEN. A run at all four defaults gets no
    #: `read_tag`, no file-name segment and no resume-identity key, so every
    #: weight, resume file and CSV row without one stands for `pyramid`. Change
    #: a default and those all start to stand for something they were not
    #: trained with (the same reason `Checkpoints.weight_filename` gives `bal`
    #: no segment).
    read_level: str = 'pyramid'
    #: `finer`: any level finer than the rung's. `l0`: level 0 only.
    resample_from: str = 'finer'
    #: The largest rung / level-downsample a candidate level may need, or None.
    #: A cost bound: the read side is `tile * factor`, 8192 px at 32.
    max_resample_factor: Optional[float] = None
    #: `mixed` only: the share of tiles read `resampled`.
    resampled_share: float = 0.5

    def __post_init__(self):
        if self.read_level not in READ_LEVELS:
            raise ValueError(f'read_level {self.read_level!r}: one of {READ_LEVELS}')
        if self.resample_from not in RESAMPLE_FROM:
            raise ValueError(f'resample_from {self.resample_from!r}: one of '
                             f'{RESAMPLE_FROM}')
        if self.max_resample_factor is not None:
            if self.max_resample_factor <= 1:
                raise ValueError(f'max_resample_factor {self.max_resample_factor}: '
                                 f'a resampled read is at least 2x, so the bound '
                                 f'must be above 1')
            if self.resample_from == 'l0':
                raise ValueError('max_resample_factor with resample_from l0: l0 '
                                 'names the level, so a bound on it either '
                                 'changes nothing or removes the only candidate')
        if not 0.0 < self.resampled_share <= 1.0:
            raise ValueError(f'resampled_share {self.resampled_share}: in (0, 1]')
        # A setting that does nothing in this mode would still reach the run's
        # arguments and its resume identity, so a pyramid run would stop
        # matching its own resume file. Refused instead of ignored.
        defaults = RenderConfig.__dataclass_fields__
        idle = []
        if self.read_level == 'pyramid':
            idle = ['resample_from', 'max_resample_factor', 'resampled_share']
        elif self.read_level == 'resampled':
            idle = ['resampled_share']
        changed = [f for f in idle if getattr(self, f) != defaults[f].default]
        if changed:
            raise ValueError(f'read_level {self.read_level} does not use '
                             f'{", ".join(changed)}; leave them at their defaults')

    @property
    def read_tag(self) -> str:
        """The read mode as one string, for every name it has to keep apart:
        weight and resume file names, the wandb run, the `read_level` CSV
        column, the evaluation label. Empty at the defaults (see above).

            resampled-finer        mixed-l0
            resampled-finer-x8     mixed-finer-p0.3"""
        if self.read_level == 'pyramid':
            return ''
        tag = f'{self.read_level}-{self.resample_from}'
        if self.max_resample_factor is not None:
            tag += f'-x{self.max_resample_factor:g}'
        if self.read_level == 'mixed' and self.resampled_share != 0.5:
            tag += f'-p{self.resampled_share:g}'
        return tag

    @property
    def read_label(self) -> str:
        """`read_tag`, or `pyramid` where the tag is empty: the CSV value."""
        return self.read_tag or 'pyramid'


class CameraBank:
    '''Lazily-built `{(dataset, wsi_name, rung, ...): Render}` cache, one
    `SlideReader` per WSI (not per rung -- the six rungs of one WSI share its
    handle; a `Render` takes the reader and does not reopen). Built fresh
    inside whichever process first uses it -- a `RoutingHeadDataset` under
    `DataLoader(num_workers>0)` gets one `CameraBank` PER WORKER this way,
    never one shared across a fork/spawn boundary (an `openslide` handle is
    not safely shared across processes; see spec.md's "Camera: train vs
    eval"). Never evicts -- see spec.md's "Handle count": the caller bounds
    how many WSIs one bank ever sees by bounding `rows`, not by eviction here.

    The magnification is baked into a `Render` at construction (`ds=rung`,
    fixed for its lifetime; `Render.at` is the same thing cached on it) -- so
    a camera is keyed by rung as well as WSI, not just WSI: one WSI's six
    rungs are six different objectives.
    '''

    def __init__(self, cfg: RenderConfig):
        self.cfg = cfg
        self._reader: Dict[Tuple[str, str], SlideReader] = {}
        self._camera: Dict[tuple, Render] = {}

    def reader_for(self, dataset_id: str, wsi_name: str) -> SlideReader:
        """The WSI's one `SlideReader` in this process."""
        key = (dataset_id, wsi_name)
        reader = self._reader.get(key)
        if reader is None:
            entry = locate(wsi_name, dataset=dataset_id)
            reader = self._reader[key] = SlideReader(SafeSlide(entry.path))
        return reader

    def _wsi_for(self, dataset_id: str, wsi_name: str) -> SafeSlide:
        """The handle `choose_read_level` reads the pyramid off."""
        return self.reader_for(dataset_id, wsi_name).slide

    def camera_for(self, dataset_id: str, wsi_name: str, rung: float, *,
                   native: bool = False,
                   read_level: Optional[int] = None) -> Render:
        '''`native=False` (default): the 'routing-query' gap -- simulates a
        real photograph, every rotation/scale/colour/vignette/distortion
        channel active. `native=True`: 'routing-support-native' -- only
        rotation, every other channel pinned to its own no-op value -- for a
        caller that wants a tile closer to what reading straight off the WSI
        gives: a real Stage 1 reference/support tile, unlike a query, is never
        actually photographed.

        `native` is part of the cache KEY, not just the config: the same
        (WSI, rung) needs up to two DIFFERENT `Camera` instances now, one
        per style, since a caller (`Episodes.render_episode`'s own
        `support_native` switch) may ask for both across one run.

        THE TWO FRAME-REFERENCED OPS, PRESENT ONLY SOMETIMES (on the query
        gap; always off on the native support one). Every other op in
        `query_sim/augment/` is per-pixel or local (colour, colour
        temperature, brightness/contrast, defocus, chromatic shift, noise,
        JPEG) or a scene transform (rotation, scale), and so means the same
        thing whatever the frame is. These two do not: `apply_vignette`'s
        falloff and `apply_distortion`'s k1 are both normalised to the
        SENSOR's half-width (`pipeline._apply_params` says so, and records
        the episode where taking them from the oversized read left the
        photo seeing only the central 53% of the falloff curve). The sensor
        here is one tile, so a tile always carrying a complete centred
        vignette would be a cue the deployed input does not have: a real
        photograph's tiles are slices of ONE vignette, and a tile from the
        centre of the frame has none at all. A probability is what models
        both kinds -- widening the strength range only ever models the
        first, more weakly.

        `read_level` (None: `level_for`) is part of the key for the same
        reason `native` is: a camera's level is fixed when it is built, so a
        rung read from three different levels is three cameras.
        '''
        key = (dataset_id, wsi_name, rung, native, read_level)
        cam = self._camera.get(key)
        if cam is None:
            reader = self.reader_for(dataset_id, wsi_name)
            template = FOV_RECIPES[SUPPORT_NATIVE_RECIPE if native
                                   else QUERY_RECIPE].gap
            gap_cfg = replace(template, stage_shift_max=self.cfg.stage_shift_max)
            # seed=None (the default): determinism, where wanted, is handled
            # per-call via capture(..., rng=...) in render_row -- a fixed
            # camera-level seed here would apply to every caller alike and
            # give neither train nor eval what it actually needs.
            tile = int(self.cfg.tile_size)
            cam = Render(reader, (tile, tile), gap_cfg, ds=rung,
                         read_level=read_level)
            self._camera[key] = cam
        return cam


def read_label_of(run_args: Dict) -> str:
    '''A checkpoint's training read mode as the `read_level` column writes it,
    from the `args` it saved. A checkpoint saved before the read mode existed
    has none of the four keys, and reads `pyramid`, which is what it was
    trained with.'''
    run_args = run_args or {}
    return RenderConfig(
        read_level=run_args.get('read_level', 'pyramid'),
        resample_from=run_args.get('resample_from', 'finer'),
        max_resample_factor=run_args.get('max_resample_factor'),
        resampled_share=run_args.get('resampled_share', 0.5)).read_label


def choose_read_level(wsi, rung: float, cfg: RenderConfig,
                      rng) -> Optional[int]:
    '''The pyramid level one tile is read from under `cfg`'s read mode, or
    None for the nearest-level rule.

    `pyramid` returns None WITHOUT TOUCHING `rng`. Drawing first and ignoring
    the draw would shift every later draw by one, and a default run would no
    longer render the photos it rendered before the read modes existed.

    The candidates are the levels FINER than the rung -- past the slack
    `ReadGeometry.level_for` allows before it calls a level native -- and, under
    `l0`, level 0 alone; `max_resample_factor` drops the ones that would need
    more than that much downsampling. One is drawn uniformly. A rung with no
    finer level (rung 1 is level 0) keeps the rule.'''
    if cfg.read_level == 'pyramid':
        return None
    if cfg.read_level == 'mixed' and rng.random() >= cfg.resampled_share:
        return None
    ds = wsi.level_downsamples
    finer = finer_levels(ds, rung)
    if cfg.resample_from == 'l0':
        finer = [lv for lv in finer if lv == 0]
    if cfg.max_resample_factor is not None:
        finer = [lv for lv in finer
                 if rung / ds[lv] <= cfg.max_resample_factor * (1 + 1e-6)]
    return rng.choice(finer) if finer else None


def _eval_rng(row: ManifestRow) -> random.Random:
    '''The eval photo's rng, from the sample's own identity (`camera.photo_rng`)
    -- not from worker id or call order, so the same query always renders the
    same photo regardless of which DataLoader worker handles it or when.'''
    return photo_rng(row.dataset, row.wsi_name, row.x, row.y, row.rung)


def render_row(bank: CameraBank, row: ManifestRow, cfg: RenderConfig, *,
                deterministic: bool, native: bool = False,
                rng: Optional[random.Random] = None
                ) -> Optional[Tuple[np.ndarray, int, bool]]:
    '''The one place a `ManifestRow` becomes pixels: `Camera.capture` (fresh
    randomness if `deterministic=False`, `train`'s own case; seeded from the
    row's identity if `deterministic=True`, `eval`'s case -- see spec.md's
    "Camera: train vs eval"). One row is ONE patch, `[tile, tile, 3]` uint8:
    the camera's sensor IS the tile, so there is nothing left to cut up.

    `native` (default `False`): passed straight through
    to `CameraBank.camera_for`'s own switch of the same name -- see that
    method's docstring. `MppRoutingHead`'s own callers never pass this; it
    exists for `Episodes.render_episode`'s
    `support_native` switch.

    `rng` (training only, ignored when `deterministic`): the generator this
    one capture's augmentation is drawn from. None (every existing caller)
    keeps the camera's own unseeded generator. `PrototypicalRoutingHead`
    passes one derived from its episode sampler, so a run resumed from a saved
    RNG state renders the same photos it would have rendered uninterrupted.

    `None` if the read ran off the slide, same as `Camera.capture` returns.
    That is now a rare, per-position event rather than a whole rung's worth:
    the sampler certified `tile*rung` and the camera reads at most the
    rotation bounding square around it, `ceil(sqrt(2)) * tile * rung`, so only
    positions near a region edge fail. `collate_routing_batch` drops them, and
    the training loop's per-rung `trained on` line is where the total shows up.

    The third element is `Render.reads_natively`: whether this rung came
    off a pyramid level at the requested mpp, or off a finer level that was
    then LANCZOS-resampled down. Carried per example rather than derived later,
    because it is a property of (slide pyramid, rung) and the camera is the only
    thing that already knows it. `score()` splits its accuracy on it -- on a
    2x pyramid (Ki67) every rung is native, on a 4x one (BRACS) the odd rungs
    are resampled, so the extra resampling's signature correlates with the
    CLASS instead of averaging out, and a head can score on it rather than on
    scale.

    Reached only through `RoutingHeadDataset.__getitem__`, for every split:
    the one copy of this logic.

    `cfg`'s read mode picks the level (`choose_read_level`), from the same
    generator the capture draws from; with no `rng`, training draws the level
    from the worker's `random`, which DataLoader seeds per worker. At the
    default `pyramid` nothing is drawn, so the capture is what it always was.
    '''
    if deterministic:
        rng = _eval_rng(row)
    level = None
    if cfg.read_level != 'pyramid':
        level = choose_read_level(bank._wsi_for(row.dataset, row.wsi_name),
                                  row.rung, cfg, rng or random)
    cam = bank.camera_for(row.dataset, row.wsi_name, row.rung, native=native,
                          read_level=level)
    patch = cam.capture(row.x, row.y, rng=rng)
    if patch is None:
        return None
    return patch, row.label, bool(cam.reads_natively)


# ══════════════════════════════════════════════════════════════════════════
#  the Dataset -- every split, rendered live, nothing written to disk
# ══════════════════════════════════════════════════════════════════════════

class RoutingHeadDataset(torch.utils.data.Dataset):
    '''One manifest row -> one rendered patch, via `render_row`.

    `split='train'` renders with fresh randomness every access, which is the
    augmentation and the point of it. `split='eval'` renders deterministically
    -- `render_row` seeds from the row's own identity -- which is what a val
    curve and a reported test number both need. Same class, same render path,
    same everything else: the only difference is where the rng comes from.

    `rows` is normally ONE WSI-BATCH's worth of manifest rows (from
    `group_by_wsi`), not the whole manifest -- see `iterate_epoch` below and
    spec.md's "Handle count: WSI-batching, not LRU eviction". Nothing in
    this class enforces that; it iterates whatever `rows` it is given, and
    the bound on how many WSIs get touched (and so how many `Camera`/
    `SafeSlide` handles this instance's `CameraBank` ends up holding) comes
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
        # process before DataLoader forks/spawns workers, and a CameraBank
        # holds open WSI handles that must not cross that boundary.
        self._bank: Optional[CameraBank] = None

    def __len__(self) -> int:
        return len(self.rows)

    def _bank_for_this_process(self) -> CameraBank:
        if self._bank is None:
            self._bank = CameraBank(self.cfg)
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
        rendered = render_row(bank, row, self.cfg,
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
    the composition of what it actually saw (`render_row` says what it is);
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
# A `CameraBank` never evicts -- it does not need to, as long as whoever
# builds one only ever hands it rows drawn from a BOUNDED number of WSIs.
# That bound is structural, not a cache-size guess: `iterate_epoch` gives
# each WSI-batch its OWN `RoutingHeadDataset` (and so its own `CameraBank`
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
