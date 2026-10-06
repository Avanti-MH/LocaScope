"""TissueMask -- a slide's tissue mask, the regions found in it, and the
questions both can answer. Plain data: nothing here segments.

    slide_mask = segmenter.segment_slide(wsi)      # TissueSegFunc: the producer
    mask = TissueMask(wsi, slide_mask)             # raw connected components
    mask = mask.filtered(0.01).merged()            # a recipe's region prep
    view = mask.patchable(256 * 4)                 # regions that fit one tile

Normally all of that is `TissueMaskConfig`'s -- `MaskMaker(cfg).mask(wsi)` --
and a caller never builds one by hand.

Coordinate system
-----------------
All public methods take and return LEVEL-0 coordinates. Internally a level-0
coordinate minus the mask's origin, divided by mask_ds_x / mask_ds_y, is a
mask pixel index:

  mask_col = floor((x0 - origin_x) / mask_ds_x)
  mask_row = floor((y0 - origin_y) / mask_ds_y)

ds_x and ds_y are kept separately because a segmenter can return a mask whose
aspect ratio differs slightly from the span it covers.

Regions are views, not state
----------------------------
`filtered`, `merged`, `patchable` and `raw` each return a NEW TissueMask that
shares the raster and owns its region list. A view cannot leak into the mask it
came from, so there is nothing to undo.
"""

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from Cache import CacheMismatch, atomic_file
from ReadGeometry import ReadSpec


# Ceiling on what may be handed to cv2.connectedComponentsWithStats in one call.
# Above it the call does not raise, it segfaults: OpenSlide-sized masks made four
# runs die at exit 139 with the fault inside _search_tissue_regions. The observed
# boundary was between 8.40 Gpx (survived) and 12.34 Gpx (died), consistent with
# a label table of total/4 entries sized in a signed int, so the real limit is
# probably 2**33 pixels. 2**31 is used instead because that theory is inferred
# from the boundary rather than read off OpenCV, and every candidate overflow is
# proportional to rows*cols, so the lower ceiling is safe under all of them.
_CC_DECIMATE_ABOVE_PX = 1 << 31

#: Above this many mask pixels the summed-area table behind white_fractions is
#: built on a decimated view. Deliberately the same shape as the ceiling above:
#: both are the point at which a whole-mask array stops fitting comfortably, and
#: both respond by losing resolution rather than by failing.
#:
#: 1 << 28 is 268 Mpx, so the int32 table stays near 1 GB. Without it this class
#: would put back the last thing the tiled read got rid of -- an array that scales with
#: the slide -- at four bytes per mask pixel: BRACS_1228's level 1 is 411 Mpx,
#: and an MRXS level 1 is four times that again.
#:
#: Nothing measurable is lost. A background fraction is an average over a
#: footprint of 256*ds level-0 pixels, which is tens to thousands of mask pixels
#: across, so a stride of 2 or 4 moves it by far less than the segmentation's own
#: error. int32 rather than float64 for the same reason it is exact: the mask is
#: 0/1, so the running sum is an integer and float32 would silently lose
#: precision past ~1.7e7 -- as a drift in every fraction at once, which is the
#: kind of wrong that never raises.
_INTEGRAL_DECIMATE_ABOVE_PX = 1 << 28


# ── the raw mask, before any region is searched for ──────────────────────────

@dataclass(frozen=True)
class SlideMask:
    """A whole-slide tissue mask, with the two numbers that let it be placed.

    What a segmenter produces and what a cache stores. `TissueMask` is
    what it becomes once regions are searched for (`TissueMask(wsi, it)`); the
    split exists because the search is cheap and depends on parameters the
    segmentation does not, so the expensive half can be stored and the cheap
    half redone.

    `mask` alone is not enough to build a `TissueMask`:

        origin   where the mask's top-left sits in LEVEL-0 coordinates. Zero on
                 an SVS; on a MIRAX it is `openslide.bounds-*`, because the
                 canvas around the scanned rectangle holds no image data at all.
        span     the LEVEL-0 extent the mask COVERS, which is not the extent that
                 was asked for. A tiler that drops partial tiles at the right and
                 bottom edge covers less: 276 tiles of 224 is 61,824 level-0 px
                 against a 61,879 px scanned rectangle, and dividing the larger
                 by the mask width gives ds 14.012 rather than 14 -- 0.09 percent,
                 and 55 level-0 px of drift by the far edge.

    `mask_ds` is carried rather than derived so a reader does not have to do that
    division to find out how blocky the mask is. `report` is whatever the
    segmenter wanted to say about the fit that produced it.
    """
    mask:    np.ndarray               # [rows, cols] bool or uint8, 1 = tissue
    origin:  Tuple[int, int]          # (x, y) level-0
    span:    Tuple[int, int]          # (w, h) level-0 actually covered
    mask_ds: float                    # level-0 px per mask px
    report:  Optional[dict] = None

    #: [rows, cols, k] float16 -- what the mask is a THRESHOLD of, kept beside
    #: the bit it produced. A threshold sweep over a stored mask is seconds with
    #: these and another 3.5 to 6 minutes of GPU per slide without them.
    #: 581 to 814 MB per slide for UNI2-PCA, so a reader that only wants the
    #: mask does not pay for them: `load(with_components=False)`.
    components: Optional[np.ndarray] = None

    @property
    def fraction(self) -> float:
        return float(np.asarray(self.mask).astype(bool).mean())

    @property
    def shape(self) -> Tuple[int, int]:
        return tuple(np.asarray(self.mask).shape[:2])

    def geometry(self) -> dict:
        """Everything but the arrays -- what a sidecar records, so a reader can
        check placement without loading the mask."""
        rows, cols = self.shape
        return dict(origin_x=int(self.origin[0]), origin_y=int(self.origin[1]),
                    span_w=int(self.span[0]), span_h=int(self.span[1]),
                    mask_ds=float(self.mask_ds), rows=int(rows), cols=int(cols),
                    fraction=self.fraction,
                    n_components=(0 if self.components is None
                                  else int(np.asarray(self.components).shape[-1])))

    def save(self, path) -> Path:
        """One safetensors file, written atomically: the mask, and the components
        when there are any. Same file rather than a sidecar because they are the
        same grid at the same instant -- a sidecar can be deleted or rebuilt on
        its own, and a sweep would then read one slide's components against
        another's mask."""
        mask = np.ascontiguousarray(np.asarray(self.mask).astype(np.uint8))
        if mask.ndim != 2:
            raise ValueError(f'mask must be 2-D, got shape {mask.shape}')
        tensors = {'mask': mask}
        if self.components is not None:
            components = np.ascontiguousarray(
                np.asarray(self.components, dtype=np.float16))
            if components.shape[:2] != mask.shape:
                raise ValueError(
                    f'components are {components.shape[:2]} cells, mask is '
                    f'{mask.shape}. They are two views of the same grid, so a '
                    f'disagreement means one of them was cropped or transposed')
            tensors['components'] = components
        metadata = {k: str(v) for k, v in self.geometry().items()}
        metadata['report_json'] = json.dumps(self.report or {}, sort_keys=True,
                                             default=str)
        with atomic_file(path) as tmp:
            save_file(tensors, str(tmp), metadata=metadata)
        return Path(path)

    @classmethod
    def load(cls, path, *, with_components: bool = False) -> 'SlideMask':
        """Read back what `save` wrote. `with_components=True` on a file that has
        none is an error rather than a None: a sweep that silently found nothing
        to sweep would report a flat curve."""
        with safe_open(str(path), framework='numpy') as handle:
            md = handle.metadata() or {}
            if with_components and int(md.get('n_components', 0)) == 0:
                raise CacheMismatch(
                    f'{path} holds no components -- its segmenter has none')
            # safe_open rather than load_file: load_file reads every tensor,
            # 500-800 MB of components for a reader that asked for the mask.
            mask = handle.get_tensor('mask')
            components = (handle.get_tensor('components') if with_components
                          else None)
        report = json.loads(md.get('report_json') or '{}')
        return cls(mask=mask,
                   origin=(int(md['origin_x']), int(md['origin_y'])),
                   span=(int(md['span_w']), int(md['span_h'])),
                   mask_ds=float(md['mask_ds']), report=report or None,
                   components=components)


class TissueRegion:
    """Bounding box of one tissue region, always in level-0 coordinates."""
    def __init__(self, x: int, y: int, w: int, h: int, index: int = -1):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.index = index


class TissueMask:
    """A `SlideMask` placed on its slide, plus the regions found in it."""

    def __init__(self, wsi, slide_mask: SlideMask):
        # No copy when it is already bool: a read-only broadcast (method '')
        # and a 771 MB plane both pass through as they are.
        main_mask = np.asarray(slide_mask.mask)
        if main_mask.dtype != bool:
            main_mask = main_mask.astype(bool)
        rows, cols = main_mask.shape[:2]
        span_w, span_h = slide_mask.span
        wsi_mpp_x = float(wsi.properties.get('openslide.mpp-x', 0))
        wsi_mpp_y = float(wsi.properties.get('openslide.mpp-y', 0))

        self.slide_mask = slide_mask
        self.main_mask = main_mask
        # Per axis, from the span the mask COVERS -- see SlideMask for why that
        # is not the span that was asked for.
        self.mask_ds_x = span_w / cols
        self.mask_ds_y = span_h / rows
        #: One mask pixel in um. Derived, never passed, so it cannot contradict
        #: wsi_mpp * mask_ds -- which it once did in two test fixtures 4x apart.
        self.mask_mpp = ((wsi_mpp_x + wsi_mpp_y) / 2
                         * (self.mask_ds_x + self.mask_ds_y) / 2)
        self.wsi_width, self.wsi_height = (int(v) for v in wsi.level_dimensions[0])
        self.wsi_mpp_x = wsi_mpp_x
        self.wsi_mpp_y = wsi_mpp_y
        self.wsi_level_downsamples = [float(d) for d in wsi.level_downsamples]
        # Level-0 coordinate of main_mask[0, 0]. Non-zero on a MIRAX, where the
        # mask covers only openslide.bounds-*. TissueRegion.x/y stay ABSOLUTE
        # level-0 coordinates regardless, so every consumer outside this class
        # is unaffected by the crop; only indexing into main_mask needs the
        # offset removed -- use to_mask_xy() / region_box().
        self.origin_x, self.origin_y = (int(v) for v in slide_mask.origin)
        self.tissue_regions: List[TissueRegion] = self._search_tissue_regions(
            main_mask, self.mask_ds_x, self.mask_ds_y,
            origin_x=self.origin_x, origin_y=self.origin_y)

    def __len__(self):
        return len(self.tissue_regions)

    def __getitem__(self, index):
        return self.tissue_regions[index]

    def __iter__(self):
        return iter(self.tissue_regions)

    def tissue_fraction(self) -> float:
        """Tissue as a fraction of what the mask covers -- the scanned area
        when the mask was cropped to it, not the canvas."""
        return float(self.main_mask.mean())

    # ── region views ────────────────────────────────────────────────────────

    def _with(self, regions: Sequence[TissueRegion]) -> 'TissueMask':
        """A view: same raster (and its summed-area table), its own regions.
        `copy.copy` shares every attribute; only the region list is replaced,
        and it is replaced by rebinding, so the source keeps its own."""
        view = copy.copy(self)
        view.tissue_regions = [TissueRegion(r.x, r.y, r.w, r.h, r.index)
                               for r in regions]
        return view

    def raw(self) -> 'TissueMask':
        """Every connected component the mask holds, before any recipe."""
        return self._with(self._search_tissue_regions(
            self.main_mask, self.mask_ds_x, self.mask_ds_y,
            origin_x=self.origin_x, origin_y=self.origin_y))

    def filtered(self, min_ratio: float) -> 'TissueMask':
        '''Drop regions that are too small or fully contained by another.

        1. Regions with area < min_ratio * max_region_area
        2. Regions fully contained within another region

        AREA ONLY, and deliberately so. This does not answer "can this region
        host a tile": a 10000x5 strip has a large area, survives any min_ratio,
        and then yields no patches at all -- which reaches the encoder as an
        empty batch, from a stack that names neither the region nor the level.
        `patchable` is the one that checks both side lengths, and it has to be,
        because the answer depends on the tile's footprint, which is not known
        when a mask is built:

            filtered     small, and contained by another. A property of the
                         segmentation, decided once, by the recipe.
            patchable    can host at least one tile. A property of the SCALE,
                         decided per footprint, by whoever tiles.

        Neither subsumes the other and running one is not running the other.
        '''
        if not self.tissue_regions:
            return self._with([])
        max_area = max(r.w * r.h for r in self.tissue_regions)
        kept = [r for r in self.tissue_regions if r.w * r.h >= max_area * min_ratio]

        def contained_by_other(r):
            for o in kept:
                if o is r:
                    continue
                if (o.x <= r.x and o.y <= r.y
                        and o.x + o.w >= r.x + r.w and o.y + o.h >= r.y + r.h):
                    return True
            return False

        return self._with([r for r in kept if not contained_by_other(r)])

    def patchable(self, side_l0: float) -> 'TissueMask':
        '''Only the regions whose level-0 width AND height are >= `side_l0` --
        the ones that can host at least one square of that footprint. A tile of
        `tile` px at downsample `ds` has `side_l0 = tile * ds`.

        Monotone in the footprint, so derive every scale from the SAME base
        mask, never from the previous view: a coarse pass taken first would
        keep a finer one from ever seeing the regions it dropped.
        '''
        return self._with([r for r in self.tissue_regions
                           if r.w >= side_l0 and r.h >= side_l0])

    def merged(self) -> 'TissueMask':
        '''Merge regions whose bboxes PARTIALLY overlap -- intersection > 0 and
        neither contains the other. Chained overlaps (A-B, B-C) collapse into
        one region with the union bbox; indices are renumbered 0..N-1.

        ORDER MATTERS: this is incomplete on its own, by design. Nested and
        identical boxes are skipped on the assumption that `filtered` already
        removed them, so merge-then-filter is not the same recipe as
        filter-then-merge. `TissueMaskConfig.regions` is the one place that
        applies them, in that order.
        '''
        regs = self.tissue_regions
        n = len(regs)
        if n < 2:
            return self._with(regs)

        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        def contains(A, B):
            return (A.x <= B.x and A.y <= B.y
                    and A.x + A.w >= B.x + B.w
                    and A.y + A.h >= B.y + B.h)

        def partial_overlap(A, B):
            ix = max(0, min(A.x + A.w, B.x + B.w) - max(A.x, B.x))
            iy = max(0, min(A.y + A.h, B.y + B.h) - max(A.y, B.y))
            if ix * iy <= 0:
                return False
            return not contains(A, B) and not contains(B, A)

        for i in range(n):
            for j in range(i + 1, n):
                if partial_overlap(regs[i], regs[j]):
                    union(i, j)

        groups = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)

        new_regs = []
        for k, members in enumerate(groups.values()):
            x0 = min(regs[i].x for i in members)
            y0 = min(regs[i].y for i in members)
            x1 = max(regs[i].x + regs[i].w for i in members)
            y1 = max(regs[i].y + regs[i].h for i in members)
            new_regs.append(TissueRegion(x=x0, y=y0, w=x1 - x0, h=y1 - y0, index=k))
        return self._with(new_regs)

    # ── level-0 <-> mask coordinates ─────────────────────────────────────────

    def to_mask_xy(self, x0: float, y0: float) -> Tuple[int, int]:
        """Absolute level-0 (x, y) -> column/row in main_mask."""
        return (int((x0 - self.origin_x) / self.mask_ds_x),
                int((y0 - self.origin_y) / self.mask_ds_y))

    def _to_mask_len(self, w_l0: float, h_l0: float) -> Tuple[int, int]:
        """Level-0 LENGTHS -> mask pixels. No origin: a width is not anchored
        anywhere, and pushing one through `to_mask_xy` shifts every size by the
        crop offset."""
        return int(w_l0 / self.mask_ds_x), int(h_l0 / self.mask_ds_y)

    def region_box(self, r: TissueRegion) -> Tuple[int, int, int, int]:
        """A region bbox in mask coords: (x, y, w, h). For drawing."""
        mx, my = self.to_mask_xy(r.x, r.y)
        return (mx, my) + self._to_mask_len(r.w, r.h)

    def get_local_mask(self, coord_0: Tuple[float, float], tile_size: int,
                       *, transfer2ds: float) -> np.ndarray:
        """One tile's own footprint of `main_mask`, resampled to exactly
        `(tile_size, tile_size)`.

            local = mask.get_local_mask((x0, y0), 256, transfer2ds=4.0)  # [256,256] bool

        `coord_0`: the tile's top-left in ABSOLUTE level-0 pixels.
        `transfer2ds`: the ds the tile was read at (a DsLadder rung, not a
        level index); `tile_size * transfer2ds` is the level-0 footprint.

        NEAREST, not bilinear: this decodes a coarser cell's binary decision
        back onto finer pixels, and bilinear would invent fractional tissue
        along every cell boundary the source mask never claimed. A caller that
        downsamples THIS again (ComplementaryLoss._prepare_tissue_mask) is doing
        the opposite resample -- many pixels into few, where a soft coverage
        fraction is the more informative answer -- which is why that step stays
        bilinear.
        """
        x0, y0 = coord_0
        footprint = float(tile_size) * float(transfer2ds)
        col0, row0 = self.to_mask_xy(x0, y0)
        col1, row1 = self.to_mask_xy(x0 + footprint, y0 + footprint)
        height, width = self.main_mask.shape
        col0, col1 = sorted((max(0, min(col0, width)), max(0, min(col1, width))))
        row0, row1 = sorted((max(0, min(row0, height)), max(0, min(row1, height))))
        if col1 <= col0 or row1 <= row0:
            raise ValueError(
                f'tile at level-0 {tuple(coord_0)} sized {footprint:g} px '
                f'falls entirely outside main_mask (shape {self.main_mask.shape}, '
                f'origin ({self.origin_x}, {self.origin_y})) -- this tile '
                f'should not have passed whatever selected it for training')
        crop = self.main_mask[row0:row1, col0:col1].astype(np.uint8)
        resized = cv2.resize(crop, (int(tile_size), int(tile_size)),
                             interpolation=cv2.INTER_NEAREST)
        return resized.astype(bool)

    def read_matching_rgb(self, wsi) -> np.ndarray:
        """The slide image covering exactly what main_mask covers, same shape.

        Use this instead of wsi.get_thumbnail(main_mask.shape) as a backdrop for
        anything drawn in mask coordinates. get_thumbnail always spans the whole
        canvas, so on a cropped mask it squeezes the entire slide into the frame
        the mask uses for the scanned rectangle alone -- the picture looks
        plausible and every overlay is wrong.

        Read by `SlideReader` at the mask's ds -- the smaller of its two axes,
        so the read stays inside the span the mask covers (they differ by less
        than a mask px) even where that span is the whole slide. The level is `level_for` (never upsampled), and the read is resampled to
        the mask's shape with INTER_AREA -- anything else invents
        high-frequency texture under a mask being judged against the tissue.
        At ds 14 (UNI2's patch grid, not a pyramid step) the read is level px,
        not mask px, so the resample is what keeps the backdrop at the mask's
        scale.
        """
        from SlideReader import SlideReader                    # noqa: PLC0415
        H, W = self.main_mask.shape
        reader = SlideReader(getattr(wsi, 'slide', wsi), resize='area')
        ds = min(self.mask_ds_x, self.mask_ds_y)
        img = reader.read(self.origin_x, self.origin_y, ReadSpec(W, H), ds)
        if img is None:
            raise RuntimeError(f'the mask\'s span runs off the slide at ds {ds:g}')
        return img

    # ── tissue queries ──────────────────────────────────────────────────────

    def has_tissue_l0(self, x: int, y: int, w: int, h: int,
                      tissue_ratio: float = 0.5) -> bool:
        """Is at least `tissue_ratio` of this LEVEL-0 rect tissue?

        Clipped to the mask. With a cropped mask a position left of or above the
        crop converts to a negative mask coordinate, and main_mask[y:y+h, x:x+w]
        would wrap around from the far edge and answer about entirely the wrong
        part of the slide.

        Area outside the mask counts as background rather than being dropped:
        the tissue count is divided by the FULL requested area, so a rect hanging
        half off the scanned region cannot score 100 percent on the half that
        happens to land on tissue.
        """
        mx, my = self.to_mask_xy(x, y)
        mw, mh = self._to_mask_len(w, h)
        if mw <= 0 or mh <= 0:
            return False
        H, W = self.main_mask.shape
        x0, y0 = max(0, mx), max(0, my)
        x1, y1 = min(W, mx + mw), min(H, my + mh)
        if x1 <= x0 or y1 <= y0:
            return False
        return float(self.main_mask[y0:y1, x0:x1].sum()) / (mw * mh) >= tissue_ratio

    def _summed_area_table(self) -> tuple:
        """Cached (table, step) for white_fractions, padded so a rect is four
        lookups. Decimated by `step` above _INTEGRAL_DECIMATE_ABOVE_PX, so every
        lookup afterwards has to be in decimated coordinates -- which is why the
        stride comes back with the table."""
        if getattr(self, '_integral_cache', None) is None:
            step = 1
            while ((self.main_mask.shape[0] // step)
                   * (self.main_mask.shape[1] // step)
                   > _INTEGRAL_DECIMATE_ABOVE_PX):
                step *= 2
            small = self.main_mask[::step, ::step]
            table = np.zeros((small.shape[0] + 1, small.shape[1] + 1),
                             dtype=np.int32)
            table[1:, 1:] = small.astype(np.int32).cumsum(0).cumsum(1)
            self._integral_cache = (table, step)
        return self._integral_cache

    def white_fractions(self, xy: np.ndarray, level: int,
                        tile: int) -> np.ndarray:
        """Background fraction of each tile's footprint, from the mask alone.

        `xy` is [N, 2] level-0 top-left coordinates of `tile`-px tiles read at
        pyramid `level`; the answer is [N] float32. Area outside the mask counts
        as background, as in `has_tissue_l0`.

        Vectorised through a summed-area table because "is this level even
        fillable" has to be answerable before any pixel is read, and a level can
        offer 200,000 candidates.
        """
        S, step = self._summed_area_table()
        H, W = S.shape[0] - 1, S.shape[1] - 1        # in DECIMATED mask pixels

        side = tile * self.wsi_level_downsamples[level]
        mw, mh = self._to_mask_len(side, side)
        # In decimated units the step**2 that would scale both the tissue count
        # and the area cancels in the ratio -- but the footprint must be at
        # least one cell, or a tile smaller than the stride divides by zero.
        mw = max(1, mw // step)
        mh = max(1, mh // step)

        mx0 = np.empty(len(xy), dtype=np.int64)
        my0 = np.empty(len(xy), dtype=np.int64)
        for i, (x, y) in enumerate(xy):
            cx, cy = self.to_mask_xy(int(x), int(y))
            mx0[i], my0[i] = cx // step, cy // step

        x0 = np.clip(mx0, 0, W)
        y0 = np.clip(my0, 0, H)
        x1 = np.clip(mx0 + mw, 0, W)
        y1 = np.clip(my0 + mh, 0, H)

        tissue = (S[y1, x1] - S[y0, x1] - S[y1, x0] + S[y0, x0])
        return (1.0 - tissue / float(mw * mh)).astype(np.float32)

    @staticmethod
    def _search_tissue_regions(mask: np.ndarray,
                               mask_ds_x: float, mask_ds_y: float,
                               min_area_px: int = 100,
                               origin_x: int = 0,
                               origin_y: int = 0) -> list[TissueRegion]:
        """Find connected tissue blobs; return ABSOLUTE level-0 bounding boxes.

        origin_* is the level-0 position of mask[0, 0] and is added back so the
        boxes stay in whole-slide coordinates even when the mask covers only a
        sub-rect. Widths and heights are offset-free, being lengths.

        Decimated first when the mask is too large for cv2 to index (see
        _CC_DECIMATE_ABOVE_PX). The stride is derived, not fixed: a mask_ds=32 mask is
        a few megapixels and gets stride 1, which is every caller that existed
        before mask_ds=1, so their results do not move at all. At mask_ds=1 the
        stride lands on 2 or 4, and a box then quantises to that many level-0
        pixels -- against a required_region_side_l0 of 1602 to 51320 that is
        four orders of magnitude below anything that reads these boxes.

        Nothing is lost by it. The decomposition at mask_ds=1 is already far
        finer than it is used at: HEST found 18035 raw blobs on BRACS_1228 and
        `filtered` kept 15. Decimating also drops the int32 label plane
        cv2 allocates from 4 bytes per pixel to 4/stride**2, which was the last
        thing in mask building still scaling with the slide.
        """
        step = 1
        while (mask.shape[0] // step) * (mask.shape[1] // step) > _CC_DECIMATE_ABOVE_PX:
            step *= 2
        small = mask[::step, ::step]          # stride view; the copy is astype's
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            small.astype(np.uint8), connectivity=8
        )
        regions = []
        for label in range(1, n_labels):      # 0 is background
            # stats are in `small` pixels: an area scales by step**2, a length
            # by step. min_area_px stays in full-resolution mask pixels so the
            # threshold means the same thing at every stride.
            #
            # int() BEFORE the multiply, not after. cv2 returns stats as int32,
            # and a blob covering much of a mask_ds=1 plane has an area in the
            # billions: times step**2 that wraps negative, every region then
            # compares below min_area_px, and tissue_regions comes back empty.
            # It surfaced as `torch.cat(): expected a non-empty list` two stages
            # later, when the mpp bank had no tiles to encode. Python ints do
            # not overflow, so the cast is the whole fix.
            if int(stats[label, cv2.CC_STAT_AREA]) * step * step < min_area_px:
                continue
            mx = int(stats[label, cv2.CC_STAT_LEFT])   * step
            my = int(stats[label, cv2.CC_STAT_TOP])    * step
            mw = int(stats[label, cv2.CC_STAT_WIDTH])  * step
            mh = int(stats[label, cv2.CC_STAT_HEIGHT]) * step
            regions.append(TissueRegion(
                x=int(mx * mask_ds_x) + origin_x,
                y=int(my * mask_ds_y) + origin_y,
                w=int(mw * mask_ds_x),
                h=int(mh * mask_ds_y),
                index=len(regions),
            ))
        return regions
