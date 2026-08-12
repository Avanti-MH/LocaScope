"""Camera — an imperfect microscope on top of a WSI.

    QueryFromWSI  = the slide under an ideal objective (raw crop)
    Camera        = the slide seen through THIS microscope (crop + augment)

Different `cfg` = different camera (vignette strength, colour temp, distortion,
noise floor ...). Different `seed` = different random exposures.

    cam = Camera(wsi_or_path, cfg=DomainGapConfig(), mask=mask, seed=42)

    # Explicit position (demo / benchmark)
    img         = cam.capture(x, y)              # np.ndarray | None
    img, params = cam.capture_with_gt(x, y)      # (np.ndarray, dict) | (None, None)

    # Random exposures via mask region-first sampling
    for shot in cam:                             # infinite iterator (caller breaks)
        shot.image, shot.gt_x, shot.gt_y, shot.params
"""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple, Union

import numpy as np
import openslide

# utilities/ so TissuesRegionsMask is importable when Camera is used alone
_HERE = os.path.dirname(os.path.abspath(__file__))
_UTILITIES = os.path.abspath(os.path.join(_HERE, '..', 'utilities'))
if _UTILITIES not in sys.path:
    sys.path.insert(0, _UTILITIES)

from TissuesRegionsMask import TissuesRegionsMask   # noqa: E402

from config           import DomainGapConfig      # noqa: E402
from pipeline         import simulate_with_gt     # noqa: E402
from source.wsi_query import QueryFromWSI         # noqa: E402


@dataclass
class CameraShot:
    image:  np.ndarray   # uint8 RGB, augmented
    gt_x:   int          # level-0 top-left of the raw crop
    gt_y:   int
    params: dict         # augment values used for THIS shot


class Camera:
    def __init__(
        self,
        wsi_or_path:  Union[str, openslide.OpenSlide],
        cfg:          Optional[DomainGapConfig]   = None,
        mask:         Optional[TissuesRegionsMask] = None,
        seed:         Optional[int]                = None,
        tissue_ratio: float                        = 0.3,
        region_protrusion_ratio: float             = 0.5,
        max_pos_tries: int                        = 20,
        max_consecutive_fail: int                 = 50,
    ):
        """
        `region_protrusion_ratio` (0.0 - 1.0): how far the bounding square is
        allowed to protrude past the tissue region on each side, as a fraction
        of the bounding-square padding (padding = (square - rect) / 2).
          0.0 = the whole bounding square must lie inside the region (strict)
          0.5 = up to half the padding may spill into non-tissue / blank glass
          1.0 = only the FoV rect itself needs to be inside the region
        Non-rotation captures (0/90/180/270 are lossless) don't sample from the
        padding at all, so protrusion is free in that mode. Non-90-degree
        rotations pull from padding — protruded corners may show non-tissue.

        `max_consecutive_fail`: stop iterating after this many rejected draws
        in a row. Counted since the last successful shot, and reset by every
        yield, so it measures "stuck", not "slow". 0 disables the fuse and
        restores the old unbounded loop. See __iter__ for why this exists.
        """
        if not 0.0 <= region_protrusion_ratio <= 1.0:
            raise ValueError(f'region_protrusion_ratio must be in [0, 1]; got {region_protrusion_ratio}')

        self.cfg = cfg or DomainGapConfig()
        self.qfw = QueryFromWSI(
            wsi_or_path,
            wh_ratio = self.cfg.wh_ratio,
            MPixels  = self.cfg.MPixels,
            mpp      = self.cfg.query_mpp,
        )
        self.mask = mask
        self.tissue_ratio            = tissue_ratio
        self.region_protrusion_ratio = region_protrusion_ratio
        self.max_pos_tries           = max_pos_tries
        self.max_consecutive_fail    = int(max_consecutive_fail)

        # Bounding-square padding in level-0 (used for _sample_position + mask
        # fit-check so the square read never falls off the WSI or region).
        self._pad_x_l0 = (self.qfw.bounding_square_side_l0 - self.qfw.rect_w_l0) // 2
        self._pad_y_l0 = (self.qfw.bounding_square_side_l0 - self.qfw.rect_h_l0) // 2

        # Padding that must remain INSIDE the region (relaxed by protrusion_ratio).
        # The rest may spill past the region edge into (usually non-tissue) space.
        self._req_pad_x_l0 = int(self._pad_x_l0 * (1 - region_protrusion_ratio))
        self._req_pad_y_l0 = int(self._pad_y_l0 * (1 - region_protrusion_ratio))

        self._py_rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        # augment fns still using np.random.* are seeded once for reproducibility
        if seed is not None:
            np.random.seed(seed)

    # ── Attribute forwards to QFW ────────────────────────────────────────────
    @property
    def wsi(self) -> openslide.OpenSlide:
        return self.qfw.wsi

    @property
    def output_w(self) -> int:
        return self.qfw.output_w

    @property
    def output_h(self) -> int:
        return self.qfw.output_h

    @property
    def rect_w_l0(self) -> int:
        return self.qfw.rect_w_l0

    @property
    def rect_h_l0(self) -> int:
        return self.qfw.rect_h_l0

    @property
    def bounding_square_side_l0(self) -> int:
        return self.qfw.bounding_square_side_l0

    @property
    def required_region_side_l0(self) -> int:
        """Minimum side length a tissue region must have to be a valid host.

        = rect_side + 2 * required_padding, i.e. FoV + as much padding as we
        insist must stay inside the region. Used by
        `generator._prep_mask_for_camera` to drive `filter_patchable`.
        """
        req_w = self.qfw.rect_w_l0 + 2 * self._req_pad_x_l0
        req_h = self.qfw.rect_h_l0 + 2 * self._req_pad_y_l0
        return max(req_w, req_h)

    # ── Explicit-position capture ────────────────────────────────────────────
    # `rotation` optional override: caller decides angle for a single shot; the
    # cfg-driven rotation still governs `__iter__` (random exposures).
    def capture(
        self, x: int, y: int, rotation: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        img, _ = self.capture_with_gt(x, y, rotation=rotation)
        return img

    def capture_with_gt(
        self, x: int, y: int, rotation: Optional[float] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[dict]]:
        raw = self.qfw.crop_bounding_square(x, y)
        if raw is None:
            return None, None
        arr, params = simulate_with_gt(
            raw, cfg=self.cfg, rng=self._py_rng, rotation=rotation,
        )
        return self._center_crop_to_output(arr), params

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

        Exact for rot in {0, 90, 180, 270}, which is what this experiment uses;
        `angle_jitter` and lens distortion are NOT inverted here, so a caller
        that leaves them on gets a position off by their magnitude rather than
        an error. test_camera_output_to_level0.py pins the whole thing against
        pixels rather than against this derivation -- the sign convention of
        `R(-rot)` is the part most likely to be wrong, and a sign error is
        invisible at 0 and 180.
        """
        import math

        q = self.qfw
        cx = float(x) + q.rect_w_l0 / 2.0
        cy = float(y) + q.rect_h_l0 / 2.0
        s = (q.rect_w_l0 / float(q.output_w)) / float(scale)

        du = float(u) - q.output_w / 2.0
        dv = float(v) - q.output_h / 2.0

        th = math.radians(float(rot_deg))
        cos_t, sin_t = math.cos(th), math.sin(th)
        # R(+rot), not R(-rot), even though this inverts the augment's rotation.
        # apply_rotation calls positive angles counter-clockwise per cv2, but
        # image y points DOWN, so inverting in that frame flips one sign back and
        # the two cancel. Determined by test_camera_output_to_level0.py, not by
        # reading cv2's docs: the first version used R(-rot) and lost to the
        # point-reflected candidate 40/40 times at 90 and 270 degrees (MAD 5.2 vs
        # 47.5) while passing 0 and 180, where the two forms coincide.
        du_s = cos_t * du - sin_t * dv
        dv_s = sin_t * du + cos_t * dv

        return cx + s * du_s, cy + s * dv_s

    def output_tile_origins(
        self, x: int, y: int, tile_size: int,
        rot_deg: float = 0.0, scale: float = 1.0,
    ):
        """Every whole `tile_size` tile of one shot, with its level-0 centre.

        Yields (row, col, u, v, cx_l0, cy_l0) where (u, v) is the tile's
        top-left in output pixels. Partial tiles at the right/bottom edge are
        skipped: a 1440x1024 output at 256 gives a clean 5x4.
        """
        q = self.qfw
        for r in range(q.output_h // tile_size):
            for c in range(q.output_w // tile_size):
                u, v = c * tile_size, r * tile_size
                cx, cy = self.output_to_level0(
                    x, y, u + tile_size / 2.0, v + tile_size / 2.0,
                    rot_deg=rot_deg, scale=scale)
                yield r, c, u, v, cx, cy

    def _center_crop_to_output(self, arr: np.ndarray) -> np.ndarray:
        """Center-crop a bounding-square augment output back to (output_h, output_w)."""
        cw, ch = self.qfw.output_w, self.qfw.output_h
        h, w = arr.shape[:2]
        if (w, h) == (cw, ch):
            return arr
        x0 = (w - cw) // 2
        y0 = (h - ch) // 2
        return arr[y0:y0 + ch, x0:x0 + cw]

    # ── Random-exposure iterator (requires mask) ─────────────────────────────
    def __iter__(self) -> Iterator[CameraShot]:
        """Yield shots forever, unless sampling gets stuck.

        The two rejection tests here can both be unsatisfiable rather than
        merely unlucky, and the loop used to have no way to say so. On a coarse
        level the FoV rect grows with the level (4x per level on a 4x pyramid),
        until a level-0 window of 23042x16385 is asked to be tissue_ratio
        tissue on a slide that is 16 percent tissue overall -- no position can
        pass, and one BRACS camera spun for hours before the job was killed.
        The caller's own skip path could not fire either: it tests `not
        records` AFTER the for loop, and the for loop only ends by break, which
        needs a success. Both exits were shut at once.

        So the fuse RETURNS rather than raises: a return ends the generator,
        the caller's for loop finishes normally, and its existing `if not
        records` branch writes skips.csv and draws the diagnosis figure with no
        change on that side.

        The counters reset on every yield, so this measures being stuck, not
        being slow -- a camera that is producing shots can run as long as it
        likes. They are kept separate because they fail for unrelated reasons:
        pos_fail is about tissue content, crop_fail is about the bounding
        square hitting the WSI edge, and the reason has to say which.
        """
        if self.mask is None:
            raise RuntimeError(
                'Camera.__iter__ needs a mask. Pass mask=TissuesRegionsMask.from_wsi(cam.wsi, ...) '
                'and prep with filter_regions / merge_overlapping / filter_patchable before iterating.'
            )
        if not self.mask.tissue_regions:
            raise RuntimeError(
                f'Mask has no usable tissue regions for a {self.rect_w_l0}x{self.rect_h_l0} '
                f'level-0 rect. Reduce MPixels/mpp, or check mask prep.'
            )

        pos_fail = 0
        crop_fail = 0
        WARN_EVERY = 20   # every N failed retries between yields, print a warn
        while True:
            pos = self._sample_position()
            if pos is None:
                pos_fail += 1
                if pos_fail % WARN_EVERY == 0:
                    print(f'  [Camera warn] _sample_position None x{pos_fail} '
                          f'(no (x,y) passing region+has_tissue in {self.max_pos_tries} tries)',
                          flush=True)
                if self.max_consecutive_fail and pos_fail >= self.max_consecutive_fail:
                    print(f'  [Camera] GIVE UP after {pos_fail} consecutive position '
                          f'failures: no (x,y) in {len(self.mask.tissue_regions)} region(s) '
                          f'reaches tissue_ratio={self.tissue_ratio} with a '
                          f'{self.qfw.rect_w_l0}x{self.qfw.rect_h_l0} level-0 rect',
                          flush=True)
                    return
                continue
            x, y = pos
            raw = self.qfw.crop_bounding_square(x, y)
            if raw is None:
                crop_fail += 1
                if crop_fail % WARN_EVERY == 0:
                    print(f'  [Camera warn] crop_bounding_square out-of-WSI x{crop_fail} '
                          f'(bounding square hitting WSI edge at last sampled (x,y))',
                          flush=True)
                if self.max_consecutive_fail and crop_fail >= self.max_consecutive_fail:
                    print(f'  [Camera] GIVE UP after {crop_fail} consecutive out-of-WSI '
                          f'crops: every accepted (x,y) puts a '
                          f'{self.qfw.bounding_square_side_l0}px bounding square off the '
                          f'WSI edge', flush=True)
                    return
                continue
            arr, params = simulate_with_gt(raw, cfg=self.cfg, rng=self._py_rng)
            pos_fail = 0
            crop_fail = 0
            yield CameraShot(
                image  = self._center_crop_to_output(arr),
                gt_x   = x,
                gt_y   = y,
                params = params,
            )

    # ── Region-first sampling (same pattern as TileSampler._sample_level) ────
    # (x, y) is the ORIGINAL FoV top-left; sampling leaves
    # (1 - region_protrusion_ratio) of the bounding-square padding on every side
    # inside the region so the square read still fits mostly within tissue.
    # The FoV rect itself is still gated by has_tissue_l0.
    def _sample_position(self) -> Optional[Tuple[int, int]]:
        rw, rh = self.qfw.rect_w_l0, self.qfw.rect_h_l0
        rpx, rpy = self._req_pad_x_l0, self._req_pad_y_l0
        for _ in range(self.max_pos_tries):
            region = self._np_rng.choice(self.mask.tissue_regions)
            lo_x = region.x + rpx
            hi_x = region.x + region.w - rw - rpx + 1
            lo_y = region.y + rpy
            hi_y = region.y + region.h - rh - rpy + 1
            if hi_x <= lo_x or hi_y <= lo_y:
                continue   # region too tight even under the relaxed check
            x = int(self._np_rng.integers(lo_x, hi_x))
            y = int(self._np_rng.integers(lo_y, hi_y))
            if self.mask.has_tissue_l0(x, y, rw, rh, self.tissue_ratio):
                return x, y
        return None
