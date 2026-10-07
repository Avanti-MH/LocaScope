"""FovSupply -- a FoV's whole pixel flow, strung from the base modules:

    position   TileSampler over a PlanSpec, drawn here or through its cache
    read       the objective's SlideReader, at the level the plan placed it for
    render     Render.capture_with_gt, the domain gap drawn from `photo_rng`

        microscope = Render(SlideReader(wsi), REAL_PHOTO_SENSOR, DomainGapConfig(...), ds=1.0)
        plan = PlanSpec('ladder', (1.0, 4.00003, 16.0017), camera=microscope.spec)
        for meta, image, params in FovSupply(microscope, plan, SamplerConfig(...), mask):
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
from typing import Iterator, Optional, Tuple

import numpy as np

from TissueMask import TissueMask                                # noqa: E402
from TileSampler import PlanSpec, SamplerConfig, TileSampler      # noqa: E402

from camera import Render, photo_rng                               # noqa: E402


class FovSupply:
    """`(meta, image, params)` for every position of one draw over `plan`.

    `microscope` is the camera: its config (sensor, gap) and its reader. The
    plan must place for exactly this camera (`plan.camera == microscope.spec`),
    or the positions would be legal for another read. `cfg` is WHERE (count,
    seed, richness, overlap, inheritance); None is `SamplerConfig()`.
    """

    def __init__(self, microscope: Render, plan: PlanSpec,
                 cfg: Optional[SamplerConfig] = None,
                 mask: Optional[TissueMask] = None):
        self._check(microscope, plan, cfg)
        self.microscope, self.plan = microscope, plan
        self.cfg = cfg if cfg is not None else SamplerConfig()
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
        if cfg is not None and not isinstance(cfg, SamplerConfig):
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
