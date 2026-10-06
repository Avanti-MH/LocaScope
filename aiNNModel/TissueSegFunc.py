"""What every tissue segmenter in this project has to be, and the ones that
need no model at all.

    seg  = HestSegConfig().build(device)          # a network
    seg  = PlaneSegConfig('hsv').build()          # colour thresholds, no model
    seg  = PlaneSegConfig('').build()             # nothing runs; see below
    mask = seg.segment_slide(wsi)                 # -> SlideMask, the whole contract

A segmenter turns a SLIDE into a `SlideMask`, and that is the whole contract.
How it gets there is the segmenter's business, and there are two shapes:

    PLANE segmenters    read one pyramid level and threshold it image by image.
    (hsv, otsu, hest)   The reading -- which level, which rectangle, in how many
                        tiles, stitched how -- is the same for all of them and
                        lives in `PlaneSegmenter`. A subclass supplies only
                        `__call__(rgb) -> mask`.
    SLIDE segmenters    need the whole slide before they can threshold any part
    (uni2_pca)          of it -- a PCA fitted across the slide -- so they read it
                        themselves and override `segment_slide`.

The producer produces and the product only holds: this module imports `SlideMask`
from `TissueMask`, and nothing in `TissueMask` knows a segmenter exists.

    method   weights   what it is
    'hest'   yes       DeepLabV3 + ResNet-50, in HestSegFunc
    'hsv'    no        saturation and value thresholds, then open/close
    'otsu'   no        Otsu on grayscale, excluding near-black
    ''       no        no segmentation. One region over the whole plane, and
                       nothing is read to arrive at it.

Everything is tissue, without paying for it
-------------------------------------------
'' reads nothing: a mask of all ones costs a full read of the level and a
full-size array to hold a constant -- 411 MB at mask_ds=4, 6.6 GB at
mask_ds=1, twenty minutes of reading.

Why it is worth having at all, rather than just segmenting: stage 2 scores a
window by the mean cosine over the query's tiles, so a window sitting on blank
glass loses on its own merits and the tissue mask there is an optimisation, not
a correctness requirement. One region also buys back what the optimisation
costs -- find_best takes a global maximum over every placement in every region,
so a region with an order of magnitude more placements wins comparisons on
sample count alone. Measured at about 0.016 of uniform uplift on S1137178,
enough to displace fifteen matches SIFT had verified with 218 to 915 inliers.
With one region there is nothing to compare across.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, NamedTuple, Optional, Tuple, Union

_HERE = Path(__file__).resolve().parent
for _d in (_HERE, _HERE.parent / 'utilities'):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import cv2                                                  # noqa: E402
import numpy as np                                          # noqa: E402
from PIL import Image                                       # noqa: E402

from ConfigIdentity import (IdentifiedBuild, IdentifiedConfig,  # noqa: E402
                            register)
from Cache import wsi_stem_of                               # noqa: E402
from TissueMask import SlideMask                            # noqa: E402

# Type hints only: hsv and otsu run no network, and a torch import at the top
# made the model-free segmenters -- and everything that names a mask recipe --
# unimportable without it. The segmenters that need torch import it themselves.
if TYPE_CHECKING:
    import torch


#: Methods that need no model. Their names are identity -- changing method
#: always changes which pixels are tissue.
NO_MODEL = ('', 'hsv', 'otsu')


# ── the methods with no model ─────────────────────────────────────────────────

def mask_hsv(rgb: np.ndarray, sat_thresh: int = 15,
             val_min: int = 30, val_max: int = 240) -> np.ndarray:
    """Saturation and value thresholds. Per-pixel, so tiling cannot change it."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    mask = ((sat > sat_thresh) & (val > val_min) & (val < val_max)).astype(np.uint8)
    k = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask.astype(bool)


def mask_otsu(rgb: np.ndarray, black_thresh: int = 20) -> np.ndarray:
    """Otsu on grayscale, excluding near-black.

    NOT tiling-safe: the threshold is derived from the histogram of whatever it
    is shown, so a tile of pure background and a tile of dense tissue get
    different thresholds and the stitched result has seams. That is why
    `PlaneSegConfig('otsu')` refuses a chunk budget.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    valid = gray[gray > black_thresh]
    if valid.size == 0:
        return np.zeros(gray.shape, dtype=bool)
    thr, _ = cv2.threshold(valid.reshape(-1, 1), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = ((gray > black_thresh) & (gray < int(thr))).astype(np.uint8)
    k = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask.astype(bool)


_FUNCS = {'hsv': mask_hsv, 'otsu': mask_otsu}


# ── where a slide is, and which of it to read ─────────────────────────────────

def scanned_rect(wsi, limit_bounds: bool = True) -> Tuple[Tuple[int, int],
                                                          Tuple[int, int]]:
    """`((x, y), (w, h))` in level 0: the scanned rectangle, or the canvas.

    A MIRAX canvas is the stage travel range, not the slide: on Ki67 the
    scanned area is 16 percent of it and the rest has no image data, which
    read_region returns as transparent. `limit_bounds` reads only
    `openslide.bounds-*`. Formats without the property (SVS) get the canvas,
    so the crop costs them nothing.
    """
    w0, h0 = wsi.level_dimensions[0]
    if not limit_bounds:
        return (0, 0), (int(w0), int(h0))
    p = wsi.properties
    return ((int(p.get('openslide.bounds-x', 0)), int(p.get('openslide.bounds-y', 0))),
            (int(p.get('openslide.bounds-width', w0)),
             int(p.get('openslide.bounds-height', h0))))


def nearest_level(wsi, ds: float) -> int:
    """The level whose downsample is closest to `ds` by RATIO, either side.

    Not openslide's `get_best_level_for_downsample`, whose rule is "the last
    level whose downsample does not exceed ds" -- a strict comparison, so a
    level reporting 4.00003 loses to a request for 4.0 and segmentation happens
    one level finer. On BRACS_1228 that was 6.58 Gpx instead of 411 Mpx: 646 s
    instead of about 40. It stayed the default for a while so old masks could
    be reproduced; there is no old mask left to reproduce.

    By ratio because pyramid levels are geometric: an absolute metric would
    call 1-vs-4 nearer than 16-vs-64 though both are one level apart. Computed
    here rather than through SafeSlide so any handle works.
    """
    if ds <= 0:
        raise ValueError(f'downsample must be positive, got {ds}')
    downsamples = [float(d) for d in wsi.level_downsamples]
    return min(range(len(downsamples)),
               key=lambda i: abs(np.log(downsamples[i] / float(ds))))


class PlaneGeometry(NamedTuple):
    """Which level to read, and which rectangle of it.

    origin and span are LEVEL-0; shape is that span expressed at `level`, which
    is also the shape of the mask that comes out. Keeping the two systems in
    separately named fields is the whole reason this is its own type: a length
    in one system silently used in the other is the bug class this code has
    already paid for twice.
    """
    level: int
    level_ds: float
    origin: Tuple[int, int]         # (x, y) level-0
    span: Tuple[int, int]           # (w, h) level-0
    shape: Tuple[int, int]          # (rows, cols) at `level`


def plane_geometry(wsi, ds: float, limit_bounds: bool = True) -> PlaneGeometry:
    level = nearest_level(wsi, ds)
    origin, span = scanned_rect(wsi, limit_bounds)
    level_ds = float(wsi.level_downsamples[level])
    return PlaneGeometry(level=level, level_ds=level_ds, origin=origin, span=span,
                         shape=(max(1, int(span[1] / level_ds)),
                                max(1, int(span[0] / level_ds))))


def tiled_apply(method: Callable[[np.ndarray], np.ndarray], H: int, W: int,
                get_tile: Callable[[int, int, int, int], np.ndarray],
                budget_px: int, stitch_overlap: int,
                source: str = 'seg', name: str = '') -> np.ndarray:
    """Tile-and-stitch, decoupled from where the pixels come from.

    `get_tile(y0, x0, y1, x1) -> (h, w, 3) uint8` supplies one expanded tile.
    Slicing an array already in memory is one implementation of that; reading
    the rect straight off the WSI is the other, and the second is what makes
    mask_ds=1 affordable: materialising the level whole first would scale the
    peak with the slide (16 bytes per level-0 pixel measured, 299 GB on the
    largest MRXS here) no matter how small the segmentation budget is.

    The grid halves the longer side until every tile fits `budget_px`. Each
    tile is read `stitch_overlap` px wider on every side and trimmed after the
    method runs, so a convolutional method never sees a tile edge where the
    stitched mask has a seam. `source` and `name` (the slide) only label the
    log line.
    """
    n_h = n_w = 1
    while (H // n_h) * (W // n_w) > budget_px:
        if H // n_h >= W // n_w:
            n_h *= 2
        else:
            n_w *= 2

    tile_h = H // n_h
    tile_w = W // n_w
    print(f'  tiled {source}{f" [{name}]" if name else ""}: '
          f'{n_h}x{n_w} = {n_h * n_w} tiles at '
          f'~{tile_h}x{tile_w} each (input {H}x{W}, budget '
          f'{budget_px / 1e6:.1f}M px, stitch_overlap={stitch_overlap})', flush=True)

    result = np.zeros((H, W), dtype=np.uint8)
    for i in range(n_h):
        for j in range(n_w):
            y0 = i * tile_h
            x0 = j * tile_w
            y1 = H if i == n_h - 1 else (i + 1) * tile_h
            x1 = W if j == n_w - 1 else (j + 1) * tile_w

            y0e = max(0, y0 - stitch_overlap)
            x0e = max(0, x0 - stitch_overlap)
            y1e = min(H, y1 + stitch_overlap)
            x1e = min(W, x1 + stitch_overlap)

            tile_mask = method(get_tile(y0e, x0e, y1e, x1e))

            trim_t = y0 - y0e
            trim_l = x0 - x0e
            result[y0:y1, x0:x1] = tile_mask[
                trim_t : trim_t + (y1 - y0),
                trim_l : trim_l + (x1 - x0),
            ]
    return result


# ── configuration ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TissueSegConfig(IdentifiedConfig):
    """What every segmenter config has. Not buildable on its own.

    Unregistered on purpose: a registered name is something a flag or a stored
    json can ask for, and there is nothing to build from this.
    """
    method: str = ''

    #: Restrict everything to `openslide.bounds-*`. See `scanned_rect`.
    limit_bounds: bool = True

    def build(self, device: Optional['torch.device'] = None) -> 'TissueSegmenter':
        raise TypeError(
            f'{type(self).__name__} is the base; build a PlaneSegConfig, '
            f'HestSegConfig or Uni2PcaSegConfig')

    def weights_key(self) -> str:
        """What a cache key has to add for weights its fields cannot show.
        '' when the config itself fixes the weights."""
        return ''


@register('plane-seg')
@dataclass(frozen=True)
class PlaneSegConfig(TissueSegConfig):
    """A segmenter applied to one pyramid level, image by image -- and complete
    for the model-free methods. HestSegConfig adds a network.

    `ds` is the SEGMENTATION resolution, not an encoding scale. The nearest
    level to it is read (`nearest_level`).

    The two chunk fields are identity although they look like memory knobs.
    The grid is `min(read_chunk_px, seg_chunk_px)`, so they decide where the
    tile boundaries fall and therefore what a non-per-pixel method produces at
    the seams. `read_chunk_px` bounds HOST memory (the level is never held
    whole) and `seg_chunk_px` bounds one forward pass; None switches that bound
    off. Bounded by default: unbounded, BRACS_1844's level was 27.5 GB of RGB
    in one array (job 349425, OOM at 400G).
    """
    method: str = 'hsv'
    ds: float = 4.0
    seg_chunk_px: Optional[int] = 4_000_000
    read_chunk_px: Optional[int] = 4_000_000
    stitch_overlap: int = 128

    def __post_init__(self):
        # Refused rather than documented: a tiled otsu is a mask with seams that
        # looks like a slightly worse mask.
        if self.method == 'otsu' and (self.seg_chunk_px or self.read_chunk_px):
            raise ValueError(
                "otsu is not tiling-safe; use PlaneSegConfig('otsu', "
                'seg_chunk_px=None, read_chunk_px=None), which reads the level '
                'whole')

    def build(self, device: Optional['torch.device'] = None) -> 'PlaneSegmenter':
        if self.method not in NO_MODEL:
            raise ValueError(
                f'{type(self).__name__} handles {NO_MODEL}; {self.method!r} '
                f'needs its own config class -- see HestSegFunc')
        return PlaneSegmenter(self, device)


# ── the segmenters ────────────────────────────────────────────────────────────

class TissueSegmenter(IdentifiedBuild):
    """A slide in, a `SlideMask` out. Subclasses implement `segment_slide`."""

    def segment_slide(self, wsi) -> SlideMask:
        raise NotImplementedError


#: The zero point for a built plane segmenter's identity.
_PLANE_BASELINE = {'method': 'hsv', 'limit_bounds': True, 'ds': 4.0,
                   'seg_chunk_px': 4_000_000, 'read_chunk_px': 4_000_000,
                   'stitch_overlap': 128}


class PlaneSegmenter(TissueSegmenter):
    """Reads one level of the slide and hands it to `__call__`, tile by tile.

    `model` is None for every method here, which is a first-class state and not
    a placeholder: weights_id comes out '' because there are no weights to
    record.
    """

    BASELINE = _PLANE_BASELINE

    def __init__(self, cfg: PlaneSegConfig, device=None):
        self.cfg = cfg
        self.device = device
        self.model = None
        self._weights_id = None

    @property
    def runs(self) -> bool:
        """False when there is nothing to run, so the read is skipped."""
        return self.cfg.method != ''

    def __call__(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """One RGB image -> [H, W] bool, True = tissue."""
        if not self.runs:
            raise RuntimeError(
                "method='' has nothing to run; segment_slide skips the read "
                'instead of calling this. Fabricating ones here would be the '
                'mask_all this replaced -- same answer, plus a full read of the '
                'level and a full-size array to hold a constant')
        rgb = np.asarray(image.convert('RGB')) if isinstance(image, Image.Image) \
            else image
        return _FUNCS[self.cfg.method](rgb)

    def segment_slide(self, wsi) -> SlideMask:
        g = plane_geometry(wsi, self.cfg.ds, self.cfg.limit_bounds)
        # A segmenter that does not run gets neither the read nor the array.
        # broadcast_to gives the shape and the values with no allocation, and it
        # is read-only, which is right: nothing may write into a mask.
        if not self.runs:
            mask = np.broadcast_to(True, g.shape)
        else:
            mask = self._segment_plane(wsi, g).astype(bool)
        # The span is the level-0 extent the mask actually covers, not the
        # canvas: pairing the canvas width with a cropped mask width would
        # inflate mask_ds by 1/crop_fraction and silently misplace everything.
        return SlideMask(mask=mask, origin=g.origin, span=g.span,
                         mask_ds=float(g.span[0]) / mask.shape[1])

    def _segment_plane(self, wsi, g: PlaneGeometry) -> np.ndarray:
        """The (rows, cols) uint8 mask of level `g.level`, however it has to be
        got. With `read_chunk_px` the level is read and segmented tile by tile
        and never held whole; without it the level is read in one call first.
        `tiled_apply` takes a get_tile callable precisely so both are the same
        code."""
        rows, cols = g.shape
        ox, oy = g.origin

        def read(y0: int, x0: int, y1: int, x1: int) -> np.ndarray:
            """One (y0:y1, x0:x1) rect of the level, as RGB.

            read_region takes a LEVEL-0 location but a level-n size, so the
            offset is scaled on its way into the location and must NOT be on its
            way into the size -- the same asymmetry SafeSlide._read_halved
            documents. Rounding rather than truncating, because a MIRAX
            level_downsample is a float near but not equal to 2 and a truncated
            offset would drift the grid by a pixel every few tiles.

            read_region_rgb, not .convert('RGB'), whenever the handle offers it.
            convert() merely drops the alpha channel, and unphotographed pixels
            carry RGB 0, so every MIRAX hole comes out pure black. HSV and Otsu
            happen to reject black; a segmentation model has no such rule and
            will call a black field tissue. Compositing onto the background
            colour makes those pixels what they physically are, blank glass.
            """
            loc = (ox + int(round(x0 * g.level_ds)), oy + int(round(y0 * g.level_ds)))
            size = (x1 - x0, y1 - y0)
            if hasattr(wsi, 'read_region_rgb'):
                return wsi.read_region_rgb(loc, g.level, size)
            return np.array(wsi.read_region(loc, g.level, size).convert('RGB'))

        seg_px, read_px = self.cfg.seg_chunk_px, self.cfg.read_chunk_px
        overlap = self.cfg.stitch_overlap
        if read_px and rows * cols > read_px:
            # One grid serves both bounds, so a heavy method just makes the
            # tiles smaller -- at the cost of more seams.
            budget = min(read_px, seg_px) if seg_px else read_px
            return tiled_apply(self, rows, cols, read, budget, overlap,
                               source='read+seg', name=wsi_stem_of(wsi))
        img = read(0, 0, rows, cols)
        if seg_px and rows * cols > seg_px:
            return tiled_apply(self, rows, cols,
                               lambda y0, x0, y1, x1: img[y0:y1, x0:x1],
                               seg_px, overlap, source='seg',
                               name=wsi_stem_of(wsi))
        return self(img)


# Implementations register themselves on import, and a registry that fills by
# side effect is empty until something imports the module. Listing them here
# means one file answers "what segmenters exist" without anyone having to guess
# which import made a name appear.
#
# The guard is for the cycle, not for style. HestSegFunc imports this module for
# its base classes, so entering through HestSegFunc runs this file to the bottom
# while HestSegFunc is still on line thirty -- and the names below do not exist
# yet. Skipping when it is already in flight is correct: it finishes on its own
# and registers itself either way.
if 'HestSegFunc' not in sys.modules:
    from HestSegFunc import HestSegConfig, HestSegmenter   # noqa: E402,F401

if 'Uni2PcaSegFunc' not in sys.modules:
    from Uni2PcaSegFunc import (Uni2PcaSegConfig,          # noqa: E402,F401
                                Uni2PcaSegmenter)
