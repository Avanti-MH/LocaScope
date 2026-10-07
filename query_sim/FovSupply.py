"""FovSupply -- a FoV's whole pixel flow, strung from the base modules:

    position   TileSampler over a PlanSpec, drawn here or through its cache
    read       the objective's SlideReader, at the level the plan placed it for
    render     Render.capture_with_gt, the domain gap drawn from `photo_rng`

        fov = FOV_RECIPES['bench']
        microscope = Render(SlideReader(wsi), fov.sensor, fov.gap, ds=1.0)
        plan = PlanSpec('ladder', fov.rungs_for(wsi.level_downsamples),
                        camera=microscope.spec)
        for meta, image, params in FovSupply(microscope, plan, fov.sampler, mask):
            ...   # meta: SampleMeta; image: uint8 RGB; params: the gap drawn

        supply = FovSupply.cached(microscope, plan, fov.sampler, masks=masks,
                                  draw_job=job, render_job=job)
        for index, meta, image, params in supply.shots(workers=7):
            ...   # index: the position's row in the draw, what results join on

One supply covers every rung of its plan: each position is photographed
through the objective at its own ds (`microscope.at(meta.ds)`), at the stack
the sampler placed it for (`meta.stack_kind`). The microscope's own ds does
not matter. Nothing here decides anything a base module does not: where is
the sampler's (`SamplerConfig`, `PlanSpec`), how big and what is read is the
camera's (`ReadSpec`), how it looks is the Render's. The one choice of its own
is the photo's rng, `photo_rng(seed, x, y, ds, 0)` -- the draw's seed and the
position -- so a FoV is the same picture in any order, in any subset, on any
thread, in any process.
"""

from __future__ import annotations

import csv
import os
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from itertools import islice
from typing import Callable, Dict, Iterator, Optional, Tuple, Union

import numpy as np

from Cache import Entry                                          # noqa: E402
from ConfigArgs import add_config_args, config_from_args         # noqa: E402
from ConfigIdentity import record                                # noqa: E402
from ReadGeometry import levels_up_to                            # noqa: E402
from SlideReader import SlideReader                              # noqa: E402
from TissueMask import TissueMask                                # noqa: E402
from TileSampler import (SAMPLER_RECIPES, InheritConfig,          # noqa: E402
                         OverlapConfig, PlanSpec,
                         RichnessConfig, SamplerConfig, TileSampler)

from camera import Render, photo_rng                               # noqa: E402
from config import DomainGapConfig                                 # noqa: E402


# ── the recipes -- one name, one corpus of FoVs, wherever it is asked for ─────

@dataclass(frozen=True)
class FovRecipe:
    """What decides which pictures a corpus of FoVs holds, apart from the
    slide and its mask: the sensor, the domain gap, the draw and the rungs.

    `rungs` is 'native' -- each slide's own pyramid levels, up to `max_ds` --
    or the ds values themselves. Each part keeps its own id (`gap`, `sampler`;
    the rungs and the sensor are in the plan's key), so a recipe adds no id of
    its own: two recipes with equal parts are the same FoVs."""
    sensor: Tuple[int, int]
    gap: DomainGapConfig
    sampler: SamplerConfig
    rungs: Union[str, Tuple[float, ...]]
    max_ds: Optional[float]

    def __post_init__(self):
        if isinstance(self.rungs, str) and self.rungs != 'native':
            raise ValueError(f"rungs is 'native' or a tuple of ds, got "
                             f"{self.rungs!r}")
        if not isinstance(self.rungs, str) and self.max_ds is not None:
            raise ValueError(f'max_ds {self.max_ds} bounds the native levels; '
                             f'with explicit rungs {self.rungs} it would bound '
                             f'nothing')

    def rungs_for(self, level_downsamples) -> Tuple[float, ...]:
        """The ds of every rung on a slide with these `level_downsamples`:
        its own levels up to `max_ds` when the recipe is 'native'."""
        if self.rungs != 'native':
            return tuple(float(r) for r in self.rungs)
        return tuple(float(level_downsamples[lv])
                     for lv in levels_up_to(level_downsamples, self.max_ds))


#: `--fov <name>` resolves here. Every field of every config is written out
#: (test_config_identity's recipe lint); a caller `dataclasses.replace`s what
#: its own arguments set, which is a different config and so a different id.
#:
#: bench                   every benchmark's FoVs: the real photos' sensor, the
#:                         full domain gap with 5 per cent mpp jitter, 50 per
#:                         native level up to ds 16, a third of a level's
#:                         budget allowed to overlap so a coarse level whose
#:                         disjoint lattice runs out still fills.
#: plain                   the class defaults written out: the real photos'
#:                         sensor, the full gap with no mpp jitter, the plain
#:                         lattice draw, every native level. What a diagnostic
#:                         switches one part of.
#: routing-query           MppRoutingHead's query tiles: a tile-sized sensor
#:                         (`--tile` replaces it), the full gap with the two
#:                         frame-referenced optics present half the time and no
#:                         stage shift, over DsLadder's six rungs. The half is a
#:                         starting value, not a measured one: a tile is as
#:                         likely to come from a frame's lit centre as from its
#:                         darkened edge. No tile above 50 per cent background:
#:                         the label is scale, and a mostly blank field carries
#:                         no scale cue at all.
#: routing-support-native  its support tiles read as stage 1 reads them at
#:                         inference: rotation only, no photometric gap.
FOV_RECIPES: Dict[str, FovRecipe] = {
    'bench': FovRecipe(
        sensor=(1440, 1024),
        gap=DomainGapConfig(
            rotation_choices=(0, 90, 180, 270), angle_jitter_deg=3.0,
            scale_range=(0.90, 1.15), query_mpp_jitter=0.05,
            brightness_range=(-0.08, 0.08), contrast_range=(-0.08, 0.08),
            saturation=1.0, color_temp_range=(-0.12, 0.12),
            vignette_range=(0.15, 0.45), stage_shift_max=3,
            distortion_k1_range=(-0.04, 0.04), distortion_k2=0.0,
            vignette_p=1.0, distortion_p=1.0, defocus_radius=2,
            chromatic_shift=2, noise_sigma=3.0, jpeg_quality=85,
            photometric=True, geometric=True),
        sampler=SamplerConfig(
            n_per_rung=50, seed=0,
            richness=RichnessConfig(
                scorer='background',
                edges=(0.15, 0.30, 0.50, 0.70, 0.85, 0.95),
                floors=(0.05, 0.15, 0.50, 0.0, 0.0, 0.0, 0.0),
                caps=(0.15, 0.25, 0.60, 0.20, 0.20, 0.0, 0.0),
                bucket_frame='per_rung', floor_frame='ask'),
            overlap=OverlapConfig(
                step=0.5, max_overlap_ratio=0.5, overlapping_share=1 / 3,
                jitter_offsets=((0.25, 1.0), (1.0, 0.25), (0.75, 1.0),
                                (1.0, 0.75), (1.25, 1.25)),
                jitter_cap=0.0),
            inherit=InheritConfig(stack_kind='F', share=0.0, source_rung=None,
                                  on_incomplete='drop'),
            candidates='lattice', max_tries_per_tile=5),
        rungs='native', max_ds=16.0),
    'plain': FovRecipe(
        sensor=(1440, 1024),
        gap=DomainGapConfig(
            rotation_choices=(0, 90, 180, 270), angle_jitter_deg=3.0,
            scale_range=(0.90, 1.15), query_mpp_jitter=0.0,
            brightness_range=(-0.08, 0.08), contrast_range=(-0.08, 0.08),
            saturation=1.0, color_temp_range=(-0.12, 0.12),
            vignette_range=(0.15, 0.45), stage_shift_max=3,
            distortion_k1_range=(-0.04, 0.04), distortion_k2=0.0,
            vignette_p=1.0, distortion_p=1.0, defocus_radius=2,
            chromatic_shift=2, noise_sigma=3.0, jpeg_quality=85,
            photometric=True, geometric=True),
        sampler=SAMPLER_RECIPES['lattice'],
        rungs='native', max_ds=None),
    'routing-query': FovRecipe(
        sensor=(256, 256),
        gap=DomainGapConfig(
            rotation_choices=(0, 90, 180, 270), angle_jitter_deg=3.0,
            scale_range=(0.90, 1.15), query_mpp_jitter=0.0,
            brightness_range=(-0.08, 0.08), contrast_range=(-0.08, 0.08),
            saturation=1.0, color_temp_range=(-0.12, 0.12),
            vignette_range=(0.15, 0.45), stage_shift_max=0,
            distortion_k1_range=(-0.04, 0.04), distortion_k2=0.0,
            vignette_p=0.5, distortion_p=0.5, defocus_radius=2,
            chromatic_shift=2, noise_sigma=3.0, jpeg_quality=85,
            photometric=True, geometric=True),
        sampler=SamplerConfig(
            n_per_rung=500, seed=0,
            richness=RichnessConfig(
                scorer='background',
                edges=(0.15, 0.30, 0.50, 0.70, 0.85, 0.95),
                floors=(0.05, 0.15, 0.50, 0.0, 0.0, 0.0, 0.0),
                caps=(0.15, 0.25, 0.60, 0.0, 0.0, 0.0, 0.0),
                bucket_frame='per_rung', floor_frame='ask'),
            overlap=OverlapConfig(
                step=0.5, max_overlap_ratio=0.5, overlapping_share=1.0,
                jitter_offsets=((0.25, 1.0), (1.0, 0.25), (0.75, 1.0),
                                (1.0, 0.75), (1.25, 1.25)),
                jitter_cap=0.25),
            inherit=InheritConfig(stack_kind='F', share=0.0, source_rung=None,
                                  on_incomplete='drop'),
            candidates='lattice', max_tries_per_tile=5),
        rungs=(1.0, 2.0, 4.0, 8.0, 16.0, 32.0), max_ds=None),
    'routing-support-native': FovRecipe(
        sensor=(256, 256),
        gap=DomainGapConfig(
            rotation_choices=(0, 90, 180, 270), angle_jitter_deg=3.0,
            scale_range=(1.0, 1.0), query_mpp_jitter=0.0,
            brightness_range=(0.0, 0.0), contrast_range=(0.0, 0.0),
            saturation=1.0, color_temp_range=(0.0, 0.0),
            vignette_range=(0.0, 0.0), stage_shift_max=0,
            distortion_k1_range=(0.0, 0.0), distortion_k2=0.0,
            vignette_p=0.0, distortion_p=0.0, defocus_radius=0,
            chromatic_shift=0, noise_sigma=0.0, jpeg_quality=100,
            photometric=False, geometric=True),
        sampler=SamplerConfig(
            n_per_rung=500, seed=0,
            richness=RichnessConfig(
                scorer='background',
                edges=(0.15, 0.30, 0.50, 0.70, 0.85, 0.95),
                floors=(0.05, 0.15, 0.50, 0.0, 0.0, 0.0, 0.0),
                caps=(0.15, 0.25, 0.60, 0.0, 0.0, 0.0, 0.0),
                bucket_frame='per_rung', floor_frame='ask'),
            overlap=OverlapConfig(
                step=0.5, max_overlap_ratio=0.5, overlapping_share=1.0,
                jitter_offsets=((0.25, 1.0), (1.0, 0.25), (0.75, 1.0),
                                (1.0, 0.75), (1.25, 1.25)),
                jitter_cap=0.25),
            inherit=InheritConfig(stack_kind='F', share=0.0, source_rung=None,
                                  on_incomplete='drop'),
            candidates='lattice', max_tries_per_tile=5),
        rungs=(1.0, 2.0, 4.0, 8.0, 16.0, 32.0), max_ds=None),
}


def add_fov_args(ap, default: str = 'bench') -> None:
    """`--fov <recipe>`, one `--sampler-*` flag per field of its draw, one
    `--camera-*` per field of its gap and `--max-ds`: the same flags in every
    tool that photographs FoVs, so any recipe one tool uses another can name.
    A flag given replaces that field of whichever recipe `--fov` names."""
    ap.add_argument('--fov', choices=sorted(FOV_RECIPES), default=default,
                    help='FoV recipe (FovSupply.FOV_RECIPES): sensor, domain '
                         'gap, draw and levels')
    ap.add_argument('--max-ds', type=float, default=None,
                    help="leave out levels coarser than this; default the "
                         "recipe's own")
    add_config_args(ap, FOV_RECIPES[default].sampler, 'sampler')
    add_config_args(ap, FOV_RECIPES[default].gap, 'camera')


def fov_from_args(args) -> FovRecipe:
    """The recipe `--fov` names with every flag given applied. A flag makes a
    different config and so a different id: an override is never served a
    draw or a photo made without it."""
    recipe = FOV_RECIPES[args.fov]
    return replace(
        recipe, sampler=config_from_args(args, recipe.sampler, 'sampler'),
        gap=config_from_args(args, recipe.gap, 'camera'),
        max_ds=args.max_ds if args.max_ds is not None else recipe.max_ds)


class FovSupply:
    """`(meta, image, params)` for every position of one draw over `plan`.

    `microscope` is the camera: its config (sensor, gap) and its reader. The
    plan must place for exactly this camera (`plan.camera == microscope.spec`),
    or the positions would be legal for another read. `cfg` is WHERE (count,
    seed, richness, overlap, inheritance): a recipe's sampler or a replace of one.
    """

    #: The code between a position and its photo: the read through the
    #: objective at the position's ds, the domain gap drawn from `photo_rng`
    #: of the draw's seed and the position (ConfigIdentity rule 3).
    VERSION = 0

    def __init__(self, microscope: Render, plan: PlanSpec,
                 cfg: SamplerConfig,
                 mask: Optional[TissueMask] = None):
        self._check(microscope, plan, cfg)
        self.microscope, self.plan = microscope, plan
        self.cfg = cfg
        self.mask = mask
        self._sampler: Optional[TileSampler] = None
        self._render: Optional[Entry] = None
        self.save_photos = False

    @classmethod
    def cached(cls, microscope: Render, plan: PlanSpec, cfg: SamplerConfig, *,
               masks, draw_job: str, render_job: Optional[str] = None,
               save_photos: bool = False, report_dir=None) -> 'FovSupply':
        """The same, with the draw from `TileSampler.cached` in `draw_job`'s
        cache (`masks` is the MaskMaker it takes the mask from) and, when
        `render_job` is given, the photos' record in that job's cache:

            .../plan=<plan>/draw=<sampler_id>/render/
                render_<gap_id>.csv     index + every drawn parameter
                record_<gap_id>.json
                photos_<gap_id>/<index>.png     with `save_photos`

        The plan's key carries the sensor, so the gap's id names the variant.
        See `shots` for what a hit is and when the entry is written."""
        out = cls(microscope, plan, cfg)
        out._sampler = TileSampler.cached(
            microscope.reader.path, cfg, plan, draw_job, masks=masks,
            report_dir=report_dir)
        if render_job is not None:
            out._render = TileSampler.draw_address(
                render_job, out._sampler.slide, masks.cfg, plan).at(
                    draw=cfg.identity_id()).entry('render')
        out.save_photos = bool(save_photos)
        return out

    @staticmethod
    def _check(microscope, plan, cfg) -> None:
        if not isinstance(plan, PlanSpec):
            raise TypeError(f'FovSupply takes a PlanSpec, got {type(plan).__name__}')
        if plan.camera != microscope.spec:
            raise ValueError(f'the plan places for {plan.camera} and this '
                             f'microscope reads {microscope.spec}: positions '
                             f'legal for one read are not for the other')
        if not isinstance(cfg, SamplerConfig):
            raise TypeError(f'FovSupply takes a SamplerConfig, got '
                            f'{type(cfg).__name__}')

    @property
    def sampler(self) -> TileSampler:
        """The draw: made on first use. A draw with no position at all says
        what the sampler saw."""
        if self._sampler is None:
            sampler = TileSampler(self.microscope.wsi, self.mask, self.cfg).sample(
                self.plan.plans_for(self.microscope.wsi))
            if not len(sampler):
                raise RuntimeError(
                    'No FoV position for this plan. What the sampler saw:\n'
                    + '\n'.join(r.line() for r in sampler.reports.values()))
            self._sampler = sampler
        return self._sampler

    def camera_for(self, ds: float) -> Render:
        """The objective a position at `ds` is photographed through -- also
        what maps its photo back to level 0 (`output_to_level0`). `Render.at`
        caches its objectives, so the same ds is the same Render."""
        return self.microscope.at(float(ds))

    def photo(self, meta, rotation: Optional[float] = None
              ) -> Tuple[np.ndarray, dict]:
        """`(image, params)` of one position of the draw -- the same picture
        whether it is taken alone, in a subset, in order or on another thread.

        `rotation` fixes the angle instead of drawing it
        (`Render.capture_with_gt`). The rng is the position's either way, so
        two rotations of one position share every other draw of the gap."""
        cam = self.camera_for(meta.ds)
        if meta.stack_kind == 'F' and cam.level != meta.level:
            raise RuntimeError(f'{meta.slide} ds {meta.ds:g}: placed for level '
                               f'{meta.level}, the objective reads level '
                               f'{cam.level}')
        x0, y0 = meta.fov_rect[0], meta.fov_rect[1]
        image, params = cam.capture_with_gt(
            x0, y0, rotation=rotation, stack=meta.stack_kind,
            rng=photo_rng(self.sampler.cfg.seed, meta.x, meta.y, f'{meta.ds:g}', 0))
        if image is None:
            # the sampler only offers positions whose reserve is on the slide;
            # a read off it is a placement bug, not a FoV to skip
            raise RuntimeError(f'{meta.slide} ds {meta.ds:g} ({x0}, {y0}): the '
                               f'read runs off the slide')
        return image, params

    def __iter__(self) -> Iterator[Tuple[object, np.ndarray, dict]]:
        for _, meta, image, params in self.shots():
            yield meta, image, params

    # ── many photos ─────────────────────────────────────────────────────────

    def shots(self, *, workers: int = 1,
              skip: Optional[Callable[[object], bool]] = None
              ) -> Iterator[Tuple[int, object, np.ndarray, dict]]:
        """`(index, meta, image, params)` for every position, in draw order,
        rendered `workers` at a time. `index` is the position's row in the
        draw (`TileSampler.write`), what everything made from a photo joins
        on. `skip(meta)` true passes a position over before it is rendered --
        a resumed run's done shots.

        With a render entry (`cached(..., render_job=)`):

            hit, photos stored    the photo is read back, nothing is rendered
            hit, no photos        rendered, and its parameters checked against
                                  the record cell for cell -- a drift is an
                                  error, not a new photo under the old name
            miss or stale         rendered, and the entry written once every
                                  position has been (never under `skip`, and
                                  never when the caller stops early: the write
                                  is staged and dropped). `save_photos` with
                                  a hit that has no photos is a miss.
        """
        todo = [(i, s.meta) for i, s in enumerate(self.sampler)
                if skip is None or not skip(s.meta)]
        entry, rid = self._render, self.microscope.cfg.identity_id()
        if entry is None:
            yield from self._rendered(todo, workers)
            return
        want = self.render_record()
        state, stale = entry.status(rid, want)
        photos = entry.path('photos', rid)
        if state == 'hit' and not (self.save_photos and not photos.is_dir()):
            stored = _read_render(entry.path('render', rid, '.csv'))
            if photos.is_dir():
                for i, m in todo:
                    yield i, m, _read_png(photos / f'{i}.png'), _params(stored[i])
                return
            for i, m, image, params in self._rendered(todo, workers):
                _check_cells(entry, rid, i, stored[i], params)
                yield i, m, image, params
            return
        if state == 'stale':
            print(f'  [render] {entry.record_path(rid)} is stale, rendering '
                  f'again: ' + '; '.join(stale), flush=True)
        if skip is not None:
            yield from self._rendered(todo, workers)
            return
        rows = []
        with entry.writing(rid, dict(want, photos=self.save_photos)) as put:
            folder = put('photos') if self.save_photos else None
            if folder is not None:
                folder.mkdir()
            for i, m, image, params in self._rendered(todo, workers):
                rows.append(_cells(i, params))
                if folder is not None:
                    _write_png(folder / f'{i}.png', image)
                yield i, m, image, params
            _write_render(put('render', '.csv'), rows)

    def render_record(self) -> dict:
        """The identity record of this draw's photos: the gap and every
        VERSION between a position and its pixels, the draw upstream."""
        return record(self.microscope.cfg, also=(SlideReader, FovSupply),
                      draw=self.cfg.identity_id())

    def _rendered(self, todo, workers: int):
        """`(index, meta, image, params)` of `todo`'s `(index, meta)` in order,
        `workers` rendering at once; at most twice that many photos are held
        ahead of the reader."""
        if workers <= 1:
            for i, m in todo:
                yield (i, m, *self.photo(m))
            return
        with ThreadPoolExecutor(max_workers=workers) as pool:
            ahead, pending = iter(todo), deque()
            for i, m in islice(ahead, 2 * workers):
                pending.append((i, m, pool.submit(self.photo, m)))
            while pending:
                i, m, future = pending.popleft()
                image, params = future.result()
                nxt = next(ahead, None)
                if nxt is not None:
                    pending.append((*nxt, pool.submit(self.photo, nxt[1])))
                yield i, m, image, params


# ── the render entry's files ─────────────────────────────────────────────────

def _cell(value) -> str:
    """One parameter as the CSV holds it: exact for a float (`repr` round-
    trips), empty for None."""
    return '' if value is None else repr(value) if isinstance(value, float) else str(value)


def _value(cell: str):
    """`_cell` undone: None, a bool, an int or a float, else the string."""
    if cell == '':
        return None
    if cell in ('True', 'False'):
        return cell == 'True'
    for kind in (int, float):
        try:
            return kind(cell)
        except ValueError:
            pass
    return cell


def _params(row: Dict[str, str]) -> dict:
    """A render row's parameters as the photo's own `params` had them."""
    return {k: _value(v) for k, v in row.items() if k != 'index'}


def _cells(index: int, params: dict) -> Dict[str, str]:
    return {'index': str(int(index)), **{k: _cell(v) for k, v in params.items()}}


def _write_render(path, rows) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, restval='')
        writer.writeheader()
        writer.writerows(rows)


def _read_render(path) -> Dict[int, Dict[str, str]]:
    """`{index: row}`; a row's cells stay strings, compared as `_cell` wrote
    them, and read back as the values they spell."""
    with open(path, newline='') as handle:
        return {int(r['index']): r for r in csv.DictReader(handle)}


def _check_cells(entry, rid, index, stored, params) -> None:
    now = _cells(index, params)
    bad = sorted(k for k in set(now) | set(stored) if now.get(k, '') != stored.get(k, ''))
    if bad:
        raise RuntimeError(
            f'{entry.path("render", rid, ".csv")}: index {index} was rendered '
            f'with other parameters than recorded ({", ".join(bad)}). The same '
            f'position under the same gap must be the same photo; something '
            f'between them changed without a VERSION')


def _write_png(path, image: np.ndarray) -> None:
    from PIL import Image                                         # noqa: PLC0415
    Image.fromarray(np.ascontiguousarray(image)).save(path, compress_level=1)


def _read_png(path) -> np.ndarray:
    from PIL import Image                                         # noqa: PLC0415
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB'))
