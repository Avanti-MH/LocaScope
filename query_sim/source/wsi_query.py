"""QueryFromWSI — read one FoV-shaped crop from a WSI at a target MPP.

Photograph metaphor: this is the *slide-side* primitive. It says "hand me the
raw region under the objective", not "what the camera sees". Pair with
`query_sim.camera.Camera` when you want lens + sensor + noise on top.

Everything WSI-derived (base_mpp, chosen_level, output pixel dims, level-0
rectangle) is computed ONCE in `__init__` and exposed as attributes. The
per-call hot path is `.crop(x, y)`.

    qfw = QueryFromWSI(wsi_or_path, wh_ratio='4:3', MPixels=12, mpp=0.25)
    qfw.wsi                        # openslide.OpenSlide
    qfw.base_mpp                   # WSI-native um/px
    qfw.chosen_level               # pyramid level actually read from
    qfw.chosen_mpp                 # um/px at that level
    qfw.output_w, qfw.output_h     # final output pixel size
    qfw.rect_w_l0, qfw.rect_h_l0   # bounding rectangle in level-0 coords
    img = qfw.crop(x, y)           # PIL RGB, or None if x/y out of bounds
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Union

import openslide
from PIL import Image

# utilities/ so SafeSlide is importable when this module is used on its own.
_UTILITIES = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                 'utilities'))
if _UTILITIES not in sys.path:
    sys.path.insert(0, _UTILITIES)

from SafeSlide import SafeSlide                                   # noqa: E402


class QueryFromWSI:
    def __init__(
        self,
        wsi_or_path: Union[str, openslide.OpenSlide],
        wh_ratio:    str   = '4:3',
        MPixels:     float = 12,
        mpp:         float = 0.25,
    ):
        # Accept both a path and an already-open handle so callers can share.
        # Opening goes through SafeSlide, not openslide.OpenSlide: it is a
        # subclass, so every consumer that type-checks against OpenSlide is
        # unaffected, and it brings two things this class's callers need --
        # a read that survives a MIRAX hole, and `nearest_level_for_downsample`,
        # which WsiTissuesContainer.resolve_scale asks of whatever slide it is
        # handed. `test_gigapath_slide_win_sim` passes `qfw.wsi` straight into
        # GigaPathSlidingWinSimRot, so a raw handle here would surface there.
        if isinstance(wsi_or_path, openslide.OpenSlide):
            if not isinstance(wsi_or_path, SafeSlide):
                raise TypeError(
                    'QueryFromWSI needs a SafeSlide, not a bare '
                    'openslide.OpenSlide. It reads `base_mpp` off the handle, '
                    'and downstream (WsiTissuesContainer.resolve_scale) reads '
                    '`nearest_level_for_downsample` -- neither exists on the '
                    'base class. Open with SafeSlide(path); it subclasses '
                    'OpenSlide, so nothing else about the handle changes.')
            self.wsi_path = getattr(wsi_or_path, '_filename', '<open-handle>')
            self.wsi      = wsi_or_path
        else:
            self.wsi_path = wsi_or_path
            self.wsi      = SafeSlide(wsi_or_path)

        self.wh_ratio = wh_ratio
        self.MPixels  = MPixels
        self.mpp      = mpp

        # ── Output pixel size (WH_ratio + MPixels) ────────────────────────────
        w_r, h_r = (int(v) for v in wh_ratio.split(':'))
        factor   = (MPixels * 1e6 / (w_r * h_r)) ** 0.5
        self.output_w = int(factor * w_r)
        self.output_h = int(factor * h_r)

        # ── WSI base MPP ──────────────────────────────────────────────────────
        # SafeSlide.base_mpp is THE definition. This class used to carry its own
        # copy -- the mean of x and y with an aperio fallback -- while
        # LocaScopePipeline and WsiTissuesContainer each carried a different one
        # that read mpp-x alone. Every slide here has mpp-x != mpp-y, so those
        # two answers differed on every slide.
        self.base_mpp = self.wsi.base_mpp

        # ── Nearest pyramid level (matches load-time selection heuristic) ─────
        chosen_lv:  Optional[int]   = None
        chosen_mpp: Optional[float] = None
        for lv, ds in enumerate(self.wsi.level_downsamples):
            mpp_lv = ds * self.base_mpp
            if abs(mpp_lv - mpp) / mpp_lv < 0.05:
                chosen_lv, chosen_mpp = lv, mpp_lv
                break
            if mpp_lv > mpp:
                prev = max(0, lv - 1)
                chosen_lv  = prev
                chosen_mpp = self.wsi.level_downsamples[prev] * self.base_mpp
                break
        if chosen_lv is None:   # target mpp is above every level's mpp -> lowest-res
            chosen_lv  = len(self.wsi.level_downsamples) - 1
            chosen_mpp = self.wsi.level_downsamples[chosen_lv] * self.base_mpp

        self.chosen_level = chosen_lv
        self.chosen_mpp   = float(chosen_mpp)

        # ── Level-N read window (in chosen-level pixels) ──────────────────────
        w_um = self.output_w * mpp   # physical FoV width  in µm
        h_um = self.output_h * mpp   # physical FoV height in µm
        self._read_w = int(w_um / self.chosen_mpp)
        self._read_h = int(h_um / self.chosen_mpp)

        # ── Level-0 bounding rectangle (for tissue mask fit / sampling) ──────
        self.rect_w_l0 = int(w_um / self.base_mpp)
        self.rect_h_l0 = int(h_um / self.base_mpp)

        # ── Bounding SQUARE (side = FoV diagonal) ─────────────────────────────
        # Any rotation of the FoV rectangle about its centre stays inside this
        # square, so caller can augment on the square and centre-crop back to
        # the rectangle without ever needing BORDER_REFLECT margin.
        import math as _math
        self.bounding_square_side_l0 = int(_math.ceil(
            _math.hypot(self.rect_w_l0, self.rect_h_l0)
        ))
        self.bounding_square_side_out = int(_math.ceil(
            _math.hypot(self.output_w, self.output_h)
        ))
        # Level-N read side matches the level-0 square scaled by chosen_mpp/base_mpp
        self._sq_read_side = int(_math.ceil(
            self.bounding_square_side_l0 * self.base_mpp / self.chosen_mpp
        ))

    # ── Convenience aliases ──────────────────────────────────────────────────
    @property
    def query_image_width(self) -> int:
        return self.output_w

    @property
    def query_image_height(self) -> int:
        return self.output_h

    @property
    def query_FoV(self) -> tuple:
        return self.output_w * self.mpp, self.output_h * self.mpp

    # ── Main hot path ────────────────────────────────────────────────────────
    def crop(self, x: int, y: int) -> Optional[Image.Image]:
        """Return one FoV-shaped PIL RGB crop at level-0 (x, y).

        Returns None if the requested rectangle would fall off the WSI bounds.
        """
        wsi_w, wsi_h = self.wsi.dimensions   # level-0 (W, H)
        if x < 0 or y < 0 or x + self.rect_w_l0 > wsi_w or y + self.rect_h_l0 > wsi_h:
            return None
        # read_region_rgb, not `.convert('RGB')`. This image BECOMES a
        # synthetic query -- it is photographed, augmented and matched -- and
        # convert merely drops the alpha, so a scanner hole would arrive as a
        # pure black rectangle whose border is a perfect corner. Back to PIL
        # because the resize below and every caller expect one.
        img = Image.fromarray(self.wsi.read_region_rgb(
            (int(x), int(y)),
            self.chosen_level,
            (self._read_w, self._read_h)))
        if img.size != (self.output_w, self.output_h):
            img = img.resize((self.output_w, self.output_h), Image.LANCZOS)
        return img

    def crop_padded(self, x: int, y: int,
                    margin: int = 0) -> Optional[Image.Image]:
        """The FoV rect grown by `margin` OUTPUT px on every side.

        This is what a shot needs when it does not rotate. The bounding square
        exists only so that a rotation about the FoV centre keeps the rectangle
        inside it; at angle 0 that headroom is 2.12x the area for nothing, paid
        on the read and again on every op that runs before the crop.

        The margin is still real: `apply_defocus`, `apply_chromatic` and
        `apply_distortion` read a neighbourhood, and without it the sensor's own
        edge pixels would fall back on a border rule. `pipeline.SENSOR_MARGIN`
        is what the Camera passes here.

        Returns None if the padded rectangle would fall off the WSI, matching
        `crop` and `crop_bounding_square` -- the caller treats all three the
        same way.
        """
        if margin <= 0:
            return self.crop(x, y)

        scale_l0 = self.rect_w_l0 / self.output_w      # level-0 px per output px
        margin_l0 = int(round(margin * scale_l0))
        margin_n = int(round(margin * self._read_w / self.output_w))

        wsi_w, wsi_h = self.wsi.dimensions
        origin_x = int(x) - margin_l0
        origin_y = int(y) - margin_l0
        if (origin_x < 0 or origin_y < 0
                or origin_x + self.rect_w_l0 + 2 * margin_l0 > wsi_w
                or origin_y + self.rect_h_l0 + 2 * margin_l0 > wsi_h):
            return None

        img = Image.fromarray(self.wsi.read_region_rgb(
            (origin_x, origin_y),
            self.chosen_level,
            (self._read_w + 2 * margin_n, self._read_h + 2 * margin_n)))
        target = (self.output_w + 2 * margin, self.output_h + 2 * margin)
        if img.size != target:
            img = img.resize(target, Image.LANCZOS)
        return img

    def crop_bounding_square(self, x: int, y: int) -> Optional[Image.Image]:
        """Read a SQUARE (side = FoV diagonal) centred on the FoV at (x, y).

        (x, y) is the level-0 top-left of the ORIGINAL FoV rectangle. The
        returned PIL is a square of side `bounding_square_side_out` px whose
        centre coincides with the FoV centre — so any rotation of the FoV
        about its centre still lies inside this square. Caller augments on
        the square and centre-crops back to (output_w, output_h).

        Returns None if the bounding square would fall off the WSI bounds.
        """
        wsi_w, wsi_h = self.wsi.dimensions
        pad_x_l0 = (self.bounding_square_side_l0 - self.rect_w_l0) // 2
        pad_y_l0 = (self.bounding_square_side_l0 - self.rect_h_l0) // 2
        sq_x = int(x) - pad_x_l0
        sq_y = int(y) - pad_y_l0
        if (sq_x < 0 or sq_y < 0
                or sq_x + self.bounding_square_side_l0 > wsi_w
                or sq_y + self.bounding_square_side_l0 > wsi_h):
            return None
        img = Image.fromarray(self.wsi.read_region_rgb(
            (sq_x, sq_y),
            self.chosen_level,
            (self._sq_read_side, self._sq_read_side)))
        target = self.bounding_square_side_out
        if img.size != (target, target):
            img = img.resize((target, target), Image.LANCZOS)
        return img
