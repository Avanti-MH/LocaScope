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

        FovSupply.cached(microscope, plan, cfg, sampler_root, masks=masks)

One supply covers every rung of its plan: each position is photographed
through the objective at its own ds (`microscope.at(meta.ds)`), at the stack
the sampler placed it for (`meta.stack_kind`). The microscope's own ds does
not matter. Nothing here decides anything a base module does not: where is
the sampler's (`SamplerConfig`, `PlanSpec`), how big and what is read is the
camera's (`ReadSpec`), how it looks is the Render's. The one choice of its own
is the photo's rng, `photo_rng(seed, x, y, ds, 0)` -- the draw's seed and the
position -- so a FoV is the same picture in any order, in any subset, in any
process.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace
from typing import Dict, Iterator, Optional, Tuple, Union

import numpy as np

from ConfigArgs import add_config_args, config_from_args         # noqa: E402
from ReadGeometry import levels_up_to                            # noqa: E402
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

    def __init__(self, microscope: Render, plan: PlanSpec,
                 cfg: SamplerConfig,
                 mask: Optional[TissueMask] = None):
        self._check(microscope, plan, cfg)
        self.microscope, self.plan = microscope, plan
        self.cfg = cfg
        self.mask = mask
        self._sampler: Optional[TileSampler] = None

    @classmethod
    def cached(cls, microscope: Render, plan: PlanSpec, cfg: SamplerConfig,
               sampler_root, *, masks, report_dir=None) -> 'FovSupply':
        """The same, with the draw from `TileSampler.cached` -- its arguments,
        passed through: a hit reads the positions back, a miss draws and
        writes them (`masks` is the MaskMaker it takes the mask from)."""
        out = cls(microscope, plan, cfg)
        out._sampler = TileSampler.cached(
            microscope.reader.path, cfg, plan, sampler_root, masks=masks,
            report_dir=report_dir)
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
        whether it is taken alone, in a subset or in order.

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
        for s in self.sampler:
            yield (s.meta, *self.photo(s.meta))
