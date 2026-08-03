"""An OpenSlide that survives the holes a MIRAX slide is allowed to have.

A MIRAX scanner photographs only the grid cells its pre-scan flagged, so some
positions have no decodable JPEG and openslide answers with

    OpenSlideError: Not a JPEG file: starts with 0x00 0x00

Two properties of that failure make it worse than an ordinary exception.

1. THE HANDLE DIES, PERMANENTLY. openslide-python installs `_check_error` as the
   ctypes errcheck on every wrapped C function (lowlevel.py:284, and the default
   at lowlevel.py:341), and it calls openslide_get_error() after each one. The C
   library latches that flag, so once a slide is in an error state every later
   call raises the same error -- level_count and level_downsamples included,
   though they read no pixels. There is no API to clear it. Reopening is the
   only recovery, which is what makes this a class rather than a try/except at
   each call site: the pipeline hands ONE slide object to TissuesRegionsMask,
   TileSampler and WsiTissuesContainer, so healing has to happen inside the
   object they all share, not in a helper that would hand back a new one and
   leave every existing holder pointing at the corpse.

2. THE USUAL RGB CONVERSION TURNS HOLES BLACK. read_region returns RGBA with
   alpha 0 where there is no image, and the RGB channels there are 0 too, so
   `.convert('RGB')` yields pure black -- indistinguishable from densely stained
   tissue. A segmentation model then marks the emptiness as tissue and regions
   appear over pixels that were never photographed. openslide itself never does
   this: get_thumbnail (openslide/__init__.py:172) composites onto
   openslide.background-color using alpha as the mask. read_region_rgb below
   does the same, and exists so that callers stop reaching for .convert('RGB').

A blank returned because a read FAILED is alpha 0, exactly like a region
openslide knew was never scanned. Downstream cannot tell those apart, which is
correct: in both cases there is no image there.

Usage:
    from SafeSlide import SafeSlide

    wsi = SafeSlide('/path/to/slide.mrxs')
    rgb = wsi.read_region_rgb((x, y), level, (w, h))   # holes come back white
    ...
    print(wsi.hole_summary())

SCOPE. This recovers from a failed read; it does not make the read smaller. One
bad tile still costs the whole requested rect, so asking for a 2816x2608 region
that contains a single hole returns 2816x2608 of blank. Reading in chunks and
subdividing on failure is a separate concern for the caller -- see the
safe_read_region entry in log/TODO.log.
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple, Union

import numpy as np
import openslide
from openslide import lowlevel
from openslide.lowlevel import OpenSlideError
from PIL import Image

# openslide.lowlevel.Filename lives in a TYPE_CHECKING block, so it cannot be
# imported at runtime; this is the same set of accepted types.
Filename = Union[str, os.PathLike]
Location = Tuple[int, int]
Size = Tuple[int, int]

# Colour openslide itself falls back to when a slide declares no background.
_DEFAULT_BACKGROUND = 'ffffff'


class SafeSlide(openslide.OpenSlide):
    """OpenSlide that replaces its own handle when a read fails.

    Subclassing rather than wrapping: every accessor on OpenSlide reads
    `self._osr` at call time (level_count, level_dimensions, level_downsamples,
    properties, read_region), and `properties` is rebuilt on each access rather
    than cached, so reassigning `self._osr` heals all of them at once. It also
    keeps `isinstance(x, openslide.OpenSlide)` true for any code that checks.

    Only read_region needs overriding. Failures originate there and nowhere
    else, and get_thumbnail calls `self.read_region`, so it inherits the
    recovery without a second override.

    Args:
        record_holes: keep every failed position in `holes`. Off saves memory on
            a slide with very many holes; `reopens` still counts them.
        warn: print each failure to stderr. On by default -- substituting blank
            for real data is not something to do quietly.
        warn_limit: stop printing after this many, so one bad slide cannot bury
            a log. The totals stay available on hole_summary().

    Attributes:
        reopens: how many times the handle had to be replaced, which equals the
            number of failed reads.
        holes:   (x, y, level, w, h) of every read that failed, in the order
                 they were attempted. Level-0 coordinates, as read_region takes.
    """

    def __init__(self, filename: Filename, record_holes: bool = True,
                 warn: bool = True, warn_limit: int = 10) -> None:
        super().__init__(filename)
        self.reopens: int = 0
        self.holes: List[Tuple[int, int, int, int, int]] = []
        self._record_holes = record_holes
        self._warn = warn
        self._warn_limit = warn_limit

    # ── recovery ─────────────────────────────────────────────────────────────

    def _report(self, err: BaseException, location: Location, level: int,
                size: Size) -> None:
        """Say on stderr that real data was replaced by a blank.

        Counted off self.reopens, which _heal has already incremented, so the
        cap holds even when record_holes is off.
        """
        if not self._warn:
            return
        if self.reopens <= self._warn_limit:
            print(f'[WARN] SafeSlide: read failed, returning blank  '
                  f'level={level} x={location[0]} y={location[1]} '
                  f'size={size[0]}x{size[1]}  ({type(err).__name__}: {err})',
                  file=sys.stderr, flush=True)
        if self.reopens == self._warn_limit:
            print(f'[WARN] SafeSlide: {self._warn_limit} failures reported; '
                  f'further ones are silent -- see .hole_summary()',
                  file=sys.stderr, flush=True)

    def _heal(self) -> None:
        """Replace the dead handle. Everything reading self._osr recovers."""
        try:
            lowlevel.close(self._osr)
        except Exception:
            # Closing a slide already in an error state can itself raise; the
            # handle is being discarded either way.
            pass
        self._osr = lowlevel.open(self._filename)
        self.reopens += 1

    def read_region(self, location: Location, level: int,
                    size: Size) -> Image.Image:
        """As OpenSlide.read_region, but a failure yields a blank instead.

        The blank is fully transparent, so it is indistinguishable from the area
        openslide returns for parts of the canvas the scanner never covered --
        which is correct, because in both cases there is no image.

        Each failure is reported on stderr, because silently swapping real data
        for blank is exactly the kind of thing that should not happen quietly.
        Output is capped at `warn_limit`: a slide can have hundreds of holes and
        the totals are on `.hole_summary()` anyway.
        """
        try:
            return super().read_region(location, level, size)
        except OpenSlideError as e:
            self._heal()
            self._report(e, location, level, size)
            if self._record_holes:
                self.holes.append((int(location[0]), int(location[1]),
                                   int(level), int(size[0]), int(size[1])))
            return Image.new('RGBA', (int(size[0]), int(size[1])), (0, 0, 0, 0))

    # ── reading without the .convert('RGB') trap ─────────────────────────────

    @property
    def background_color(self) -> str:
        """openslide.background-color as '#rrggbb', white when unset."""
        return '#' + self.properties.get(
            openslide.PROPERTY_NAME_BACKGROUND_COLOR, _DEFAULT_BACKGROUND
        )

    def read_region_rgb(self, location: Location, level: int,
                        size: Size) -> np.ndarray:
        """RGB uint8 (H, W, 3), transparency composited onto the background.

        Use instead of `np.array(read_region(...).convert('RGB'))`. That call
        merely drops the alpha channel, and since unphotographed pixels carry
        RGB 0 it paints every hole black -- which is what made a segmentation
        model claim tissue over areas the scanner skipped. Compositing is what
        openslide does in get_thumbnail.
        """
        tile = self.read_region(location, level, size)
        flat = Image.new('RGB', tile.size, self.background_color)
        flat.paste(tile, None, tile)          # alpha as mask
        return np.array(flat)

    def read_region_valid(self, location: Location, level: int,
                          size: Size) -> Tuple[np.ndarray, np.ndarray]:
        """(rgb, valid): the composited image plus where it is real.

        `valid` is True exactly where openslide returned a pixel, covering holes
        and never-scanned canvas alike since a failed read is filled with the
        same fully transparent blank.

        Nothing in the pipeline consumes this yet -- gating a segmentation mask
        on it was considered and set aside. It is here because the alpha channel
        already carries the information and read_region_rgb throws it away.
        """
        tile = self.read_region(location, level, size)
        valid = np.array(tile.getchannel('A')) > 0
        flat = Image.new('RGB', tile.size, self.background_color)
        flat.paste(tile, None, tile)
        return np.array(flat), valid

    # ── reporting ────────────────────────────────────────────────────────────

    def hole_summary(self) -> str:
        if not self.holes:
            return 'no failed reads'
        area = sum(w * h for _, _, _, w, h in self.holes)
        levels = sorted({lv for _, _, lv, _, _ in self.holes})
        return (f'{len(self.holes)} failed reads on levels {levels}, '
                f'{area} px requested, handle reopened {self.reopens}x')
