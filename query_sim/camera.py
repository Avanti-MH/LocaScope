"""Render — an imperfect microscope's rendering of a WSI.

    SlideReader = the slide under an ideal objective (raw read)
    Render      = the slide seen through THIS microscope (read + domain gap)

A camera is its SENSOR (whole px), its domain gap (`cfg`: vignette, colour
temp, distortion, noise floor ...) and its objective (`ds`). Different `seed`
= different random exposures. The instance is called `camera` everywhere.

    reader = SlideReader(wsi)
    camera = Render(reader, REAL_PHOTO_SENSOR, DomainGapConfig(), ds=4.0, seed=42)

    img         = camera.capture(x, y)            # np.ndarray | None
    img, params = camera.capture_with_gt(x, y)    # (np.ndarray, dict) | (None, None)
    camera.spec                                   # ReadSpec: what a sampler
                                                  # needs to place this camera
    camera.at(4.0)                                # the same microscope through
                                                  # another objective (ds 4)

A Render decides what a FoV LOOKS like: the domain gap, its reproducibility,
and the ground truth that inverts its geometry (`output_to_level0`). It reads
nothing itself -- the pixels come from its `SlideReader`, at its spec -- and
knows no mask. WHERE the FoVs are is a `TileSampler` draw; `FovSupply(camera,
plan, cfg, mask)` strings the draw, the read and this render together. A read with
no effects is not a renderer's job: `reader.read(..., ReadSpec(t, t))`
(reference tiles, pre-tiles, support tiles).

MAGNIFICATION IS A DOWNSAMPLE. A Render is built at `ds` (relative to the
slide's own level 0), required, and is at that one magnification for life -- `rect_w_l0`, `spec` and
`output_to_level0` all depend on it. Another magnification is another
objective on the same microscope: `camera.at(ds)`, which shares the reader
and the config and is cached, so asking twice is one Render.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import sys
from typing import Dict, Optional, Tuple

import numpy as np

from config     import DomainGapConfig                   # noqa: E402
from pipeline    import read_reach, simulate_with_gt      # noqa: E402
from ReadGeometry import FovGeometry, ReadSpec           # noqa: E402
from SafeSlide   import SafeSlide                        # noqa: E402
from SlideReader import SlideReader                      # noqa: E402


def rotates_for(cfg: DomainGapConfig, rotation: Optional[float] = None) -> bool:
    """Will an exposure under `cfg` turn the slide at all? Decided before the
    read, from what is fixed: an explicit `rotation`, else every angle the cfg
    could draw. Any jitter means the angle is unknown, so the answer is yes.
    Conservative on purpose: saying "no" when the shot does rotate would crop
    content away; saying "yes" when it does not only costs the wider read."""
    if cfg.angle_jitter_deg:
        return True
    if rotation is not None:
        return float(rotation) % 360.0 != 0.0
    return any(float(a) % 360.0 != 0.0 for a in cfg.rotation_choices)


def read_margin(cfg: DomainGapConfig, sensor: Tuple[int, int],
                rotation: Optional[float] = None) -> int:
    """The fewest output px to read around the sensor rectangle -- or around
    its bounding square, if the exposure turns -- so that everything one
    exposure can sample (`pipeline.read_reach`) was read."""
    w, h = (int(v) for v in sensor)
    ex, ey = read_reach(cfg, (w, h), rotation)
    if rotates_for(cfg, rotation):
        side = math.ceil(math.hypot(w, h))
        return max(0, math.ceil(max(ex, ey) - (side - 1) / 2.0))
    return max(0, math.ceil(ex - (w - 1) / 2.0), math.ceil(ey - (h - 1) / 2.0))


def render_spec(cfg: DomainGapConfig, sensor: Tuple[int, int],
                rotation: Optional[float] = None) -> ReadSpec:
    """The `ReadSpec` a renderer with this domain gap and this sensor reads:
    what a sampler needs to place it, before any slide is open."""
    return ReadSpec(int(sensor[0]), int(sensor[1]),
                    rotates=rotates_for(cfg, rotation),
                    margin_out=read_margin(cfg, sensor, rotation))


def photo_rng(*key) -> random.Random:
    """The rng of ONE photo, from that photo's own identity: the same key is
    the same domain gap, in any process, in any order, on any worker.

        camera.capture_with_gt(x, y, rng=photo_rng(seed, x, y, f'{ds:g}', 0))

    The key's parts are joined with '|'; a caller writes each the way it wants
    it compared (a ds as `:g`, say). sha256 and not `hash()`, which Python
    randomises per process for a str. THE ONE DEFINITION: `FovSupply` and
    MppRoutingHead's eval render (`Datasets.render_row`) both draw through it."""
    text = '|'.join(str(k) for k in key)
    return random.Random(int(hashlib.sha256(text.encode()).hexdigest()[:16], 16))


class Render:
    def __init__(self, reader: SlideReader, sensor: Tuple[int, int],
                 cfg: Optional[DomainGapConfig] = None, *, ds: float,
                 seed: Optional[int] = None, read_level: Optional[int] = None):
        """
        `reader` is the slide's `SlideReader`; its filter is the camera's
        ('lanczos' for a microscope). `sensor` is `(w, h)` output px.
        `cfg` is the domain gap, None its defaults. `seed` seeds the sequential
        draw of `capture` when no per-call `rng` is given.

        `ds` is the magnification (downsample from level 0). The nominal mpp a
        photo records is `ds * base_mpp`.

        `read_level` forces the pyramid level the read comes from
        (`SlideReader.level_of` checks it is not coarser than ds); None keeps
        `level_for`. Like `ds` it is a constructor argument and not a
        `DomainGapConfig` field on purpose: that config is hashed into run and
        corpus identities, and a new field would move every one of them.
        """
        if not isinstance(reader, SlideReader):
            raise TypeError(f'Render takes a SlideReader, got {type(reader).__name__}'
                            f' (SlideReader(wsi) wraps a SafeSlide or a path)')
        w, h = (int(v) for v in sensor)
        if w <= 0 or h <= 0:
            raise ValueError(f'sensor must be positive, got {sensor}')
        if ds is None or float(ds) <= 0:
            raise ValueError(f'ds must be a positive downsample, got {ds}')
        self.reader = reader
        self.cfg = cfg or DomainGapConfig()
        self.output_w, self.output_h = w, h
        self.ds = float(ds)
        self.read_level = read_level
        self.level = reader.level_of(self.ds, read_level)
        self.geometry = FovGeometry.of(self.output_w, self.output_h, self.ds)
        self._seed = seed
        # Every objective of this microscope, by ds -- shared by all of them,
        # so `a.at(4).at(1)` is `a` and nothing is built twice.
        self._objectives: Dict[float, 'Render'] = {self.ds: self}
        self._py_rng = random.Random(seed)
        # No `np.random.seed` here: the whole augment chain draws off this
        # object's own generators, and a constructor that seeded the global
        # state would decide the noise of every other Render.

    # ── what it is ─────────────────────────────────────────────────────────

    @property
    def spec(self) -> ReadSpec:
        """What a sampler needs to place this camera: `render_spec` of its gap
        and sensor."""
        return render_spec(self.cfg, self.sensor)

    @property
    def wsi(self) -> SafeSlide:
        return self.reader.slide

    @property
    def base_mpp(self) -> float:
        return self.reader.base_mpp

    @property
    def mpp(self) -> float:
        """The nominal mpp of this objective: `ds * base_mpp`."""
        return self.ds * self.reader.base_mpp

    @property
    def sensor(self) -> Tuple[int, int]:
        return self.output_w, self.output_h

    @property
    def rect_w_l0(self) -> int:
        return self.geometry.rect_w_l0

    @property
    def rect_h_l0(self) -> int:
        return self.geometry.rect_h_l0

    @property
    def bounding_square_side_l0(self) -> int:
        return self.geometry.square_l0

    @property
    def reads_natively(self) -> bool:
        """Does this magnification come off its level with no resampling?
        (`SlideReader.native`; the routing heads split accuracy on it.)"""
        return self.reader.native(self.ds, self.read_level)

    def at(self, ds: float) -> 'Render':
        """This microscope through another objective: the same reader, sensor,
        config and filter at downsample `ds`. Cached -- the same ds is the same
        Render, from whichever objective it is asked.

        Each objective has its own generator, seeded from this microscope's
        seed and the ds (None stays None), so what one objective shoots does
        not depend on what another shot first. A forced `read_level` is not
        carried over: it names a level for one magnification."""
        ds = float(ds)
        cam = self._objectives.get(ds)
        if cam is None:
            seed = (None if self._seed is None else int(hashlib.sha256(
                f'{self._seed}|{ds!r}'.encode()).hexdigest()[:8], 16))
            cam = Render(self.reader, self.sensor, self.cfg, ds=ds, seed=seed)
            cam._seed = self._seed
            cam._objectives = self._objectives
            self._objectives[ds] = cam
        return cam

    # ── the exposure ───────────────────────────────────────────────────────
    # `rotation` optional override: caller decides angle for a single shot;
    # without it the cfg draws one.

    def capture(self, x: int, y: int, rotation: Optional[float] = None,
                rng: Optional[random.Random] = None, stack: str = 'F'
                ) -> Optional[np.ndarray]:
        img, _ = self.capture_with_gt(x, y, rotation=rotation, rng=rng, stack=stack)
        return img

    def capture_with_gt(self, x: int, y: int, rotation: Optional[float] = None,
                        rng: Optional[random.Random] = None, stack: str = 'F'
                        ) -> Tuple[Optional[np.ndarray], Optional[dict]]:
        """The read at (x, y) -- the FoV rectangle's level-0 top-left -- then
        the domain gap, cropped to the sensor. `(None, None)` off the slide.

        What is read is decided BEFORE it, from what is fixed (`render_spec`):
        the bounding square if the exposure can turn (`rotates_for`), else the
        rectangle, grown by everything the exposure can sample (`read_margin`).

        `rng`, if given, is used INSTEAD of this camera's own generator for
        this one call. For a caller that needs the SAME (x, y) to always render
        the SAME photo regardless of call order (`training/MppRoutingHead/
        spec.md`, "Camera: train vs eval"): pass `rng=random.Random(derived
        from the sample's own identity)` instead of building a fresh Render.

        `stack='R'` reads the ds 1 field and degrades it to this ds
        (`SlideReader.read`)."""
        spec = render_spec(self.cfg, self.sensor, rotation)
        raw = self.reader.read(x, y, spec, self.ds, stack=stack,
                               level=self.read_level)
        if raw is None:
            return None, None
        # The centre crop to the sensor happens inside `pipeline._apply_params`,
        # straight after the scene geometry: the vignette's falloff and the
        # lens distortion's normalisation are measured against the frame they
        # are handed, and that has to be the sensor, not the bounding square.
        return simulate_with_gt(raw, cfg=self.cfg, rng=rng or self._py_rng,
                                rotation=rotation,
                                output_wh=(self.output_w, self.output_h),
                                mpp=self.mpp)

    # ── Where did this output pixel come from? ───────────────────────────────

    def output_to_level0(
        self, x: int, y: int, u: float, v: float,
        rot_deg: float = 0.0, scale: float = 1.0,
    ) -> Tuple[float, float]:
        """Level-0 coordinate that output pixel (u, v) was taken from.

        `x, y` is the level-0 top-left of the FoV rect that was passed to
        `capture_with_gt`; `rot_deg` and `scale` come back in its params dict.

        A shot is built as: read a bounding square centred on the FoV centre ->
        rotate about that centre -> centre-crop to (output_w, output_h). Every
        step is about the same centre, so inverting it is one rotation and one
        scale about that point, with no translation bookkeeping:

            C      = FoV centre at level 0
            s      = rect_w_l0 / output_w      level-0 px per output px
            (du,dv)= (u,v) - output centre     offset in the ROTATED frame
            source = C + (s / scale) * R(-rot) . (du, dv)

        Exact for rot in {0, 90, 180, 270}; `angle_jitter`, lens distortion
        and the mechanical STAGE SHIFT are NOT inverted, so a caller that
        leaves them on gets a position off by their magnitude rather than an
        error. The stage shift is not inverted ON PURPOSE: it models the
        jitter a real operator cannot see, so a localiser is meant to eat it
        as irreducible error. `params` records it so the same seed reproduces
        the same shot, not so anyone can correct a coordinate with it.
        test_camera.py (section `map`) pins the whole thing against pixels
        rather than against this derivation -- a sign error in R is invisible
        at 0 and 180.
        """
        cx = float(x) + self.rect_w_l0 / 2.0
        cy = float(y) + self.rect_h_l0 / 2.0
        s = (self.rect_w_l0 / float(self.output_w)) / float(scale)
        du = float(u) - self.output_w / 2.0
        dv = float(v) - self.output_h / 2.0
        th = math.radians(float(rot_deg))
        cos_t, sin_t = math.cos(th), math.sin(th)
        # R(+rot), not R(-rot), even though this inverts the augment's rotation.
        # apply_rotation calls positive angles counter-clockwise per cv2, but
        # image y points DOWN, so inverting in that frame flips one sign back and
        # the two cancel. Determined by test_camera.py (section `map`): the first
        # version used R(-rot) and lost to the point-reflected candidate 40/40
        # times at 90 and 270 degrees while passing 0 and 180.
        du_s = cos_t * du - sin_t * dv
        dv_s = sin_t * du + cos_t * dv
        return cx + s * du_s, cy + s * dv_s

    def output_tile_origins(self, x: int, y: int, tile_size: int,
                            rot_deg: float = 0.0, scale: float = 1.0):
        """Every whole `tile_size` tile of one shot, with its level-0 centre.
        Yields (row, col, u, v, cx_l0, cy_l0), (u, v) the tile's top-left in
        output px. Partial tiles at the right/bottom edge are skipped: a
        1440x1024 output at 256 gives a clean 5x4."""
        for r in range(self.output_h // tile_size):
            for c in range(self.output_w // tile_size):
                u, v = c * tile_size, r * tile_size
                cx, cy = self.output_to_level0(
                    x, y, u + tile_size / 2.0, v + tile_size / 2.0,
                    rot_deg=rot_deg, scale=scale)
                yield r, c, u, v, cx, cy
