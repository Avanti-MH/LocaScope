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

3. A FAILED READ USED TO COST THE WHOLE RECTANGLE. openslide fails the entire
   requested rect when any tile inside it is missing, and WsiTissuesContainer
   reads one rect per tissue region. On S1137178 that blanked a 32500x15232 px
   region -- 7.9 x 3.7 mm of tissue -- at both level 0 and level 1, and since
   retrieval scores windows on those features, it could never propose anywhere
   inside it. None of that slide's 189 predictions land in that rectangle.
   read_region now halves the rect and re-reads each side on failure, down to
   `min_chunk`, so only the parts with genuinely no image come back blank. It
   costs nothing on a slide without holes, because it is on the failure path.

A blank returned because a read FAILED is alpha 0, exactly like a region
openslide knew was never scanned. Downstream cannot tell those apart, which is
correct: in both cases there is no image there.

Usage:
    from SafeSlide import SafeSlide

    wsi = SafeSlide('/path/to/slide.mrxs')
    rgb = wsi.read_region_rgb((x, y), level, (w, h))   # holes come back white
    ...
    print(wsi.hole_summary())

SCOPE. Subdividing bounds the DATA lost, not the memory used: the returned
image is still the full requested rect, so a read too large to hold is still
too large. Reading a region without ever materialising it whole is a separate
concern for the caller.
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
        min_chunk: stop subdividing a failed read once its longer side is down
            to this, and blank what is left. The floor is what bounds the cost:
            every abandoned rect reopens the handle, and reopening a MIRAX
            re-parses its index. Lower recovers more tissue and reopens more
            often. 0 disables subdivision entirely, restoring the behaviour
            where one bad tile blanks the whole requested rect.

    Attributes:
        reopens: how many times the handle had to be replaced, which equals the
            number of failed reads.
        holes:   (x, y, level, w, h) of every read that failed, in the order
                 they were attempted. Level-0 coordinates, as read_region takes.
    """

    def __init__(self, filename: Filename, record_holes: bool = True,
                 warn: bool = True, warn_limit: int = 10,
                 min_chunk: int = 64) -> None:
        super().__init__(filename)
        self.reopens: int = 0
        self.holes: List[Tuple[int, int, int, int, int]] = []
        self._record_holes = record_holes
        self._warn = warn
        self._warn_limit = warn_limit
        self._reports = 0
        self._min_chunk = int(min_chunk)

    # ── recovery ─────────────────────────────────────────────────────────────

    def _report(self, err: BaseException, location: Location, level: int,
                size: Size) -> None:
        """Say on stderr that real data was replaced by a blank.

        Called only where a rect is actually given up on -- a leaf of the
        subdivision -- so the lines describe what was lost, not the path taken
        to find out. Capped on its own counter rather than on reopens, because
        subdividing reopens the handle many times per rect abandoned and the
        cap would otherwise be spent before the first line printed.
        """
        if not self._warn:
            return
        self._reports += 1
        if self._reports <= self._warn_limit:
            print(f'[WARN] SafeSlide: read failed, returning blank  '
                  f'level={level} x={location[0]} y={location[1]} '
                  f'size={size[0]}x{size[1]}  ({type(err).__name__}: {err})',
                  file=sys.stderr, flush=True)
        if self._reports == self._warn_limit:
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
        """As OpenSlide.read_region, but a failure costs a hole, not the rect.

        openslide fails the WHOLE requested rectangle when any tile inside it is
        missing, so a single bad 128 px tile used to blank an entire tissue
        region -- 32500x15232 px on one Ki67 slide, 7.9 x 3.7 mm of tissue that
        retrieval could then never propose because its features were all blank.
        On failure this splits the rect and re-reads each half, so only the
        parts that genuinely have no image come back blank.

        Splitting happens only on the failure path: a slide with no holes never
        pays for it, and 8 of the 10 Ki67 slides have none at any level.

        Recursion goes back through this method rather than through
        super().read_region, because each half needs the same healing: the
        handle latches its error flag, so after one bad half every later read
        would fail too. It stops when the rect is down to `min_chunk`, and that
        floor is what bounds the cost -- each abandoned rect reopens the handle,
        and reopening a MIRAX re-parses its index.

        The blank is fully transparent, indistinguishable from the area
        openslide returns for canvas the scanner never covered -- correct,
        because in both cases there is no image there.
        """
        try:
            return super().read_region(location, level, size)
        except OpenSlideError as e:
            self._heal()
            if self._min_chunk and max(int(size[0]), int(size[1])) > self._min_chunk:
                return self._read_halved(location, level, size)
            self._report(e, location, level, size)
            if self._record_holes:
                self.holes.append((int(location[0]), int(location[1]),
                                   int(level), int(size[0]), int(size[1])))
            return Image.new('RGBA', (int(size[0]), int(size[1])), (0, 0, 0, 0))

    def _read_halved(self, location: Location, level: int,
                     size: Size) -> Image.Image:
        """Split the longer side in two and re-read each half.

        Halving the longer side rather than tiling on a fixed grid keeps the
        two halves square-ish, so the descent narrows in on a hole from every
        direction at once; it is also the shape TissuesRegionsMask._tiled_apply
        already uses for the same kind of problem.

        The odd pixel goes to the first half so the two always sum back to the
        requested size -- an off-by-one here would leave a one-pixel transparent
        seam through otherwise good tissue.

        `location` is level-0 while `size` and the offsets are level-n, so the
        offset has to be scaled by the downsample on its way into location and
        must NOT be on its way into paste(). Rounding rather than truncating,
        because a MIRAX level_downsample is a float near but not equal to 2.
        """
        ds = self.level_downsamples[level]
        w, h = int(size[0]), int(size[1])
        out = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        if w >= h:
            parts = ((0, 0, w - w // 2, h), (w - w // 2, 0, w // 2, h))
        else:
            parts = ((0, 0, w, h - h // 2), (0, h - h // 2, w, h // 2))
        for ox, oy, pw, ph in parts:
            if pw <= 0 or ph <= 0:
                continue
            loc = (int(location[0]) + int(round(ox * ds)),
                   int(location[1]) + int(round(oy * ds)))
            out.paste(self.read_region(loc, level, (pw, ph)), (ox, oy))
        return out

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
