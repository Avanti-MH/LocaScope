'''What one read covers at level 0 -- the one definition the sampler and the
reader both use.

A sampler reserves room around a position; `SlideReader` then reads around it.
The reserve is derived from the read with the same integers, so every position
the sampler offers is one the reader can read:

    FovGeometry.of(output_w, output_h, ds)              the FoV at level 0
    .read_rect(x, y, rotates, margin_out)               what is read for (x, y)
    ReadSpec(sensor, rotates, margin).place(fp, ds)      what a sampler reserves
    ReadRect.overhang(box)                              how far it leaves a box
    reserve_margin(read, box)                           what to reserve on
                                                        each side of the box
    level_for(level_downsamples, ds)                    which level to read
    level_px(out_px, ds, level_ds)                      how many of its px

Everything is in `ds`, a downsample relative to the slide's own level 0. An
mpp enters once, where a caller has one (`ds = mpp / base_mpp`), and is not
converted back.

One level rule: read the coarsest level not coarser than asked (never
upsample), and round the level px.

Pure arithmetic, no IO, no imports from the rest of the project: `TileSampler`
and `SlideReader` both depend on this, and a sampler must know what a read
covers without opening a slide, so it lives at the bottom.
'''
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

#: Pyramid downsamples are derived from rounded level dimensions, so a "4x"
#: level reports 4.00003 as readily as 4.0, and an exact comparison lands a
#: level away over a part in 1e5. THE ONE DEFINITION: SafeSlide imports it.
LEVEL_REL_TOL = 1e-3


#: The real microscope photographs' sensor, `(w, h)` px: 1440x1024 BMP. A
#: sensor is given in whole pixels, never derived from a ratio and a pixel
#: count, so a sampler placing for a camera and the camera cannot disagree.
REAL_PHOTO_SENSOR = (1440, 1024)

#: The encoder's input tile side, px -- what a reference tile, a query tile and
#: a routing-head sensor all are.
TILE_PX = 256



def level_for(level_downsamples: Sequence[float], ds: float) -> int:
    """The level to READ for downsample `ds`: the coarsest whose native
    downsample is at most `ds` (within `LEVEL_REL_TOL`), so the rest of the
    way is a shrink and never a blow-up.

    Raises when `ds` is finer than level 0: reaching it would be upsampling,
    and the caller almost certainly meant something else.
    """
    if ds <= 0:
        raise ValueError(f'downsample must be positive, got {ds}')
    downsamples = [float(d) for d in level_downsamples]
    threshold = float(ds) * (1.0 + LEVEL_REL_TOL)
    candidates = [i for i, d in enumerate(downsamples) if d <= threshold]
    if not candidates:
        raise ValueError(
            f'ds {ds} is finer than level 0 (ds {downsamples[0]:.6g}); '
            f'reaching it would mean upsampling, which is refused. '
            f'Available: {[round(d, 4) for d in downsamples]}')
    return max(candidates)


def _ceil(x: float) -> int:
    """ceil, forgiving floating-point dust: 363 * 2.0 is 726, and a product
    that lands at 726.0000000001 is 726 too, not 727."""
    return int(math.ceil(x - 1e-6))


def level_px(out_px: int, ds: float, level_ds: float) -> int:
    """Level px to read for `out_px` output px at downsample `ds` off a level
    of downsample `level_ds`. Rounded, not truncated: a level at 4.00014 for
    ds 4 is 255.99 px for a 256 px tile, and that is 256 px read, not 255
    read and blown up. Computed from OUTPUT px, so a native read (ds equal to
    level_ds within rounding) is exactly the output size and needs no resize."""
    return int(round(int(out_px) * float(ds) / float(level_ds)))


@dataclass(frozen=True)
class ReadRect:
    """A level-0 rectangle: top-left (x0, y0), size (w, h)."""
    x0: int
    y0: int
    w: int
    h: int

    def overhang(self, x0: int, y0: int, x1: int, y1: int) -> Tuple[int, int, int, int]:
        """(left, top, right, bottom): how far this rect runs outside the box
        [x0, x1) x [y0, y1), each 0 when it does not."""
        return (max(0, x0 - self.x0), max(0, y0 - self.y0),
                max(0, self.x0 + self.w - x1), max(0, self.y0 + self.h - y1))

    def inside(self, x0: int, y0: int, x1: int, y1: int) -> bool:
        return not any(self.overhang(x0, y0, x1, y1))

    def shifted(self, dx: int, dy: int) -> 'ReadRect':
        return ReadRect(self.x0 + dx, self.y0 + dy, self.w, self.h)


@dataclass(frozen=True)
class FovGeometry:
    """A camera's field of view at level 0, from the three numbers that fix it.

        rect_l0    int(output * ds)                      the FoV rectangle
        square_out ceil(hypot(output_w, output_h))       what a rotating
                                                          exposure reads, sensor px
        square_l0  ceil(square_out * ds)                 the same, level-0 px

    `square_l0` is derived from `square_out`, not from the level-0 rectangle's
    own diagonal, because the read is `level_px(square_out, ...)` px: at ds 2,
    `ceil(hypot(rect_l0))` is 725 while 363 x 2 = 726 are read. It holds any
    rotation: `rect_l0 <= output * ds`, so its diagonal is at most
    `ds * square_out`.
    """
    output_w: int
    output_h: int
    rect_w_l0: int
    rect_h_l0: int
    square_l0: int
    square_out: int
    ds: float

    @classmethod
    def of(cls, output_w: int, output_h: int, ds: float) -> 'FovGeometry':
        rect_w = int(output_w * float(ds))
        rect_h = int(output_h * float(ds))
        square_out = int(math.ceil(math.hypot(output_w, output_h)))
        return cls(output_w=int(output_w), output_h=int(output_h),
                   rect_w_l0=rect_w, rect_h_l0=rect_h,
                   square_l0=_ceil(square_out * float(ds)),
                   square_out=square_out, ds=float(ds))

    def margin_l0(self, margin_out: int) -> int:
        """A sensor margin of `margin_out` output px, in level-0 px."""
        return int(round(margin_out * (self.rect_w_l0 / self.output_w)))

    def read_out(self, rotates: bool, margin_out: int = 0) -> Tuple[int, int]:
        """`(w, h)` output px a read covers: the bounding square or the FoV
        rectangle, grown by `margin_out` output px on every side."""
        m = int(margin_out)
        if rotates:
            return self.square_out + 2 * m, self.square_out + 2 * m
        return self.output_w + 2 * m, self.output_h + 2 * m

    def read_rect(self, x: int, y: int, rotates: bool,
                  margin_out: int = 0) -> ReadRect:
        """What is read for a FoV whose level-0 top-left is (x, y): the
        bounding square centred on the FoV if it rotates, else the FoV, each
        grown by `margin_out` output px on every side."""
        if rotates:
            side = _ceil((self.square_out + 2 * int(margin_out)) * self.ds)
            return ReadRect(int(x) - (side - self.rect_w_l0) // 2,
                            int(y) - (side - self.rect_h_l0) // 2, side, side)
        if margin_out <= 0:
            return ReadRect(int(x), int(y), self.rect_w_l0, self.rect_h_l0)
        m = self.margin_l0(margin_out)
        return ReadRect(int(x) - m, int(y) - m, self.rect_w_l0 + 2 * m,
                        self.rect_h_l0 + 2 * m)


@dataclass(frozen=True)
class ReadSpec:
    """What a read covers, as far as anyone placing it needs to know: its
    sensor in output px, whether it rotates (then the bounding square is
    read), and the margin read around that, in output px. No pixels, no slide.

    The sampler takes its footprint and its reserve from here -- it decides
    WHERE, the spec decides how big and what is read (`SlideReader.read`) --
    so the two cannot describe different reads. `place` is the one
    computation both sides use.

        ReadSpec(256, 256, rotates=True)              a routing-head tile
        ReadSpec(1440, 1024, rotates=True, margin_out=m)  a microscope FoV; m from
                                                      `camera.render_spec`
        ReadSpec(256, 256)                            a plain tile, nothing around it
        ReadSpec(256, 256, margin_out=256)            a 3x pre-tile around a tile

    A square sensor's FoV starts at the sample's own top-left (the routing
    heads render `meta.x, meta.y` directly); a rectangular one is centred in
    the square of its long side, which is the footprint the sampler places.
    """
    sensor_w: int
    sensor_h: int
    rotates: bool = False
    margin_out: int = 0

    def __post_init__(self):
        if self.sensor_w <= 0 or self.sensor_h <= 0:
            raise ValueError(f'sensor must be positive, got {self.sensor_w}x'
                             f'{self.sensor_h}')
        if self.margin_out < 0:
            raise ValueError(f'margin_out must be >= 0, got {self.margin_out}')

    @property
    def long_side(self) -> int:
        """The footprint's side in output px: the sensor's long side."""
        return max(self.sensor_w, self.sensor_h)

    @property
    def square(self) -> bool:
        return self.sensor_w == self.sensor_h

    def key(self) -> str:
        """A readable name for a cache path: every field that changes a read."""
        base = f'cam{self.sensor_w}x{self.sensor_h}' + ('-rot' if self.rotates else '')
        return base + (f'-m{self.margin_out}' if self.margin_out else '')

    def geometry(self, ds: float) -> FovGeometry:
        return FovGeometry.of(self.sensor_w, self.sensor_h, ds)

    def fov_offset(self, box: int, fov_w: int, fov_h: int) -> Tuple[int, int]:
        """Where the FoV's top-left sits inside a footprint box of side `box`:
        (0, 0) for a square sensor, centred for a rectangular one."""
        if self.square:
            return 0, 0
        return (box - fov_w) // 2, (box - fov_h) // 2

    def place(self, footprint_l0: float, ds: float) -> Tuple[int, int, int]:
        """`(fov_w_l0, fov_h_l0, reserve_l0)` for a footprint of `footprint_l0`
        level-0 px at downsample `ds`.

        The FoV is the camera's own rectangle at `ds`; the reserve is the
        footprint box grown by the largest overhang of what the camera READS
        (`read_rect`) beyond it, the same integers the camera will use."""
        geo = self.geometry(ds)
        box = int(footprint_l0)
        fw, fh = geo.rect_w_l0, geo.rect_h_l0
        ox, oy = self.fov_offset(box, fw, fh)
        read = geo.read_rect(ox, oy, self.rotates, margin_out=self.margin_out)
        return fw, fh, box + 2 * reserve_margin(read, 0, 0, box)


def reserve_margin(read: ReadRect, box_x: int, box_y: int, box_side: int) -> int:
    """Level-0 px to reserve on EACH side of a square box so that `read` stays
    inside the box grown by it: the largest of the four overhangs. A sampler
    whose reserve is `box_side + 2 * this` (`TileSampler._margin_of` halves the
    difference back) offers only positions this read fits around."""
    return max(read.overhang(box_x, box_y, box_x + box_side, box_y + box_side))
