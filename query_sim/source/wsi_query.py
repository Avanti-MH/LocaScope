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

from typing import Optional, Union

import openslide
from PIL import Image


class QueryFromWSI:
    def __init__(
        self,
        wsi_or_path: Union[str, openslide.OpenSlide],
        wh_ratio:    str   = '4:3',
        MPixels:     float = 12,
        mpp:         float = 0.25,
    ):
        # Accept both a path and an already-open handle so callers can share.
        if isinstance(wsi_or_path, openslide.OpenSlide):
            self.wsi_path = getattr(wsi_or_path, '_filename', '<open-handle>')
            self.wsi      = wsi_or_path
        else:
            self.wsi_path = wsi_or_path
            self.wsi      = openslide.OpenSlide(wsi_or_path)

        self.wh_ratio = wh_ratio
        self.MPixels  = MPixels
        self.mpp      = mpp

        # ── Output pixel size (WH_ratio + MPixels) ────────────────────────────
        w_r, h_r = (int(v) for v in wh_ratio.split(':'))
        factor   = (MPixels * 1e6 / (w_r * h_r)) ** 0.5
        self.output_w = int(factor * w_r)
        self.output_h = int(factor * h_r)

        # ── WSI base MPP ──────────────────────────────────────────────────────
        props = self.wsi.properties
        mx = props.get('openslide.mpp-x')
        my = props.get('openslide.mpp-y')
        if (mx is None or my is None) and 'aperio.MPP' in props:
            mx = my = props['aperio.MPP']
        if mx is None or my is None:
            raise RuntimeError(f'{self.wsi_path}: WSI has no openslide.mpp-x / aperio.MPP')
        self.base_mpp = (float(mx) + float(my)) / 2

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
        img = self.wsi.read_region(
            (int(x), int(y)),
            self.chosen_level,
            (self._read_w, self._read_h),
        ).convert('RGB')
        if img.size != (self.output_w, self.output_h):
            img = img.resize((self.output_w, self.output_h), Image.LANCZOS)
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
        img = self.wsi.read_region(
            (sq_x, sq_y),
            self.chosen_level,
            (self._sq_read_side, self._sq_read_side),
        ).convert('RGB')
        target = self.bounding_square_side_out
        if img.size != (target, target):
            img = img.resize((target, target), Image.LANCZOS)
        return img
