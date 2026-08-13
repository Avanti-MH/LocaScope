"""
TissueMask — stores tissue regions as a binary mask.

Format on disk:
    <prefix>.npy   — 2D bool array (H, W)
    <prefix>.json  — ds_x, ds_y, mpp, level

Coordinate system
-----------------
All public methods take / return level-0 coordinates.
Internally, level-0 coords are divided by ds_x (x-axis) and ds_y (y-axis)
to get mask pixel indices.

  mask_col = floor(x0 / ds_x)
  mask_row = floor(y0 / ds_y)

ds_x and ds_y are stored separately to handle the case where the method
returns a mask whose aspect ratio differs slightly from the WSI.
"""

import json
import os
from typing import Union

import cv2
import numpy as np
import openslide

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
#: would put back the last thing from_wsi got rid of -- an array that scales with
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


class TissueRegion:
    """Bounding box of one tissue region, always in level-0 coordinates."""
    def __init__(self, x: int, y: int, w: int, h: int, index: int = -1):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.index = index


class TissuesRegionsMask:
    def __init__(self, main_mask: np.ndarray, mask_ds_x: float, mask_ds_y: float,
                 mask_mpp: float, tissue_regions: list[TissueRegion],
                 wsi_width: int, wsi_height: int,
                 wsi_mpp_x: float, wsi_mpp_y: float,
                 wsi_level_downsamples: list[float],
                 origin_x: int = 0, origin_y: int = 0):
        self.main_mask = main_mask
        self.mask_ds_x = mask_ds_x
        self.mask_ds_y = mask_ds_y
        self.mask_mpp = mask_mpp
        self.tissue_regions: list[TissueRegion] = tissue_regions
        self.wsi_width = wsi_width
        self.wsi_height = wsi_height
        self.wsi_mpp_x = wsi_mpp_x
        self.wsi_mpp_y = wsi_mpp_y
        self.wsi_level_downsamples = list(wsi_level_downsamples)
        # Level-0 coordinate of main_mask[0, 0]. Non-zero when from_wsi read a
        # sub-rect instead of the whole canvas (MIRAX openslide.bounds-*), where
        # the scanned area can be a sixth of the canvas and the rest holds no
        # image at all. Zero for SVS and for any whole-canvas read, which makes
        # every conversion below reduce to what it was before.
        #
        # Invariant: TissueRegion.x/y stay ABSOLUTE level-0 coordinates, so every
        # consumer outside this class (WsiTissuesContainer, TileSampler, SIFT)
        # is unaffected by the crop. Only indexing into main_mask needs the
        # offset removed -- use to_mask_xy() / region_box() rather than dividing
        # by mask_ds_* by hand.
        self.origin_x = int(origin_x)
        self.origin_y = int(origin_y)
        # Undo stack for regions mutations (filter_regions / filter_patchable /
        # merge_overlapping snapshot tissue_regions here before modifying).
        self._regions_history: list[list[TissueRegion]] = []

    def __len__(self):
        return len(self.tissue_regions)

    def __getitem__(self, index):
        return self.tissue_regions[index]

    def __iter__(self):
        return iter(self.tissue_regions)

    def tissue_fraction(self) -> float:
        """Tissue as a fraction of what the mask covers.

        Note the denominator follows the mask, not the slide: when from_wsi
        cropped to openslide.bounds-* this is a fraction of the scanned area,
        not of the canvas, so the number jumps versus an uncropped run of the
        same slide. That makes it comparable across slides, but not against
        older logs.
        """
        return float(self.main_mask.mean())

    # ── level-0 <-> mask coordinates ─────────────────────────────────────────

    def to_mask_xy(self, x0: float, y0: float) -> tuple[int, int]:
        """Absolute level-0 (x, y) -> column/row in main_mask."""
        return (int((x0 - self.origin_x) / self.mask_ds_x),
                int((y0 - self.origin_y) / self.mask_ds_y))

    def region_box(self, r: TissueRegion) -> tuple[int, int, int, int]:
        """A region bbox in mask coords: (x, y, w, h). For drawing."""
        mx, my = self.to_mask_xy(r.x, r.y)
        return (mx, my,
                int(r.w / self.mask_ds_x), int(r.h / self.mask_ds_y))

    def read_matching_rgb(self, wsi) -> np.ndarray:
        """The slide image covering exactly what main_mask covers, same shape.

        Use this instead of wsi.get_thumbnail(main_mask.shape) as a backdrop for
        anything drawn in mask coordinates. get_thumbnail always spans the whole
        canvas, so on a cropped mask it squeezes the entire slide into the frame
        the mask uses for the scanned rectangle alone -- the picture looks
        plausible and every overlay is wrong.
        """
        H, W = self.main_mask.shape
        lv = wsi.get_best_level_for_downsample(self.mask_ds_x)
        return np.array(
            wsi.read_region((self.origin_x, self.origin_y), lv, (W, H))
               .convert('RGB')
        )

    # ── Regions mutation history ─────────────────────────────────────────────

    def _snapshot(self) -> None:
        """Push a deep copy of tissue_regions onto the undo stack.

        Deep-copying protects the history against any future code that
        mutates a TissueRegion's fields in place — the snapshot stays
        pristine regardless of what happens to the current list.
        """
        self._regions_history.append([
            TissueRegion(r.x, r.y, r.w, r.h, r.index)
            for r in self.tissue_regions
        ])

    def regions_resume(self) -> None:
        """
        Re-run _search_tissue_regions on main_mask, restoring the pristine
        connected-component regions. Clears the undo history.
        """
        self.tissue_regions = self._search_tissue_regions(
            self.main_mask, self.mask_ds_x, self.mask_ds_y,
            origin_x=self.origin_x, origin_y=self.origin_y,
        )
        self._regions_history.clear()

    def regions_undo(self) -> bool:
        """
        Undo the most recent mutation of tissue_regions (filter_regions,
        filter_patchable, or merge_overlapping). Returns True if a snapshot
        was popped, False if the history was empty (silent no-op).
        """
        if not self._regions_history:
            return False
        self.tissue_regions = self._regions_history.pop()
        return True

    @staticmethod
    def _tiled_apply(method:     callable,
                     H:          int,
                     W:          int,
                     get_tile:   callable,
                     seg_chunk_px: int,
                     overlap:    int,
                     source:     str = 'seg') -> np.ndarray:
        """Tile-and-stitch, decoupled from where the pixels come from.

        `get_tile(y0, x0, y1, x1) -> (h, w, 3) uint8` supplies one expanded
        tile. Slicing an array already in memory is one implementation of that;
        reading the rect straight off the WSI is the other, and the second is
        what makes mask_ds=1 affordable -- from_wsi used to materialise the
        whole level first, so the peak scaled with the slide (16 bytes per
        level-0 pixel measured, 299 GB on the largest MRXS here) no matter how
        small the segmentation budget was. --seg-chunk-px only ever bounded the
        GPU; nothing bounded the host until this split.

        `source` only labels the log line, so a tiled read is distinguishable
        from a tiled segmentation of an already-read image.
        """
        n_h = n_w = 1
        while (H // n_h) * (W // n_w) > seg_chunk_px:
            if H // n_h >= W // n_w:
                n_h *= 2
            else:
                n_w *= 2

        tile_h = H // n_h
        tile_w = W // n_w
        print(f'  tiled {source}: {n_h}x{n_w} = {n_h * n_w} tiles at '
              f'~{tile_h}x{tile_w} each (input {H}x{W}, budget '
              f'{seg_chunk_px / 1e6:.1f}M px, overlap={overlap})', flush=True)

        result = np.zeros((H, W), dtype=np.uint8)
        for i in range(n_h):
            for j in range(n_w):
                y0 = i * tile_h
                x0 = j * tile_w
                y1 = H if i == n_h - 1 else (i + 1) * tile_h
                x1 = W if j == n_w - 1 else (j + 1) * tile_w

                y0e = max(0, y0 - overlap)
                x0e = max(0, x0 - overlap)
                y1e = min(H, y1 + overlap)
                x1e = min(W, x1 + overlap)

                tile_mask = method(get_tile(y0e, x0e, y1e, x1e))

                trim_t = y0 - y0e
                trim_l = x0 - x0e
                result[y0:y1, x0:x1] = tile_mask[
                    trim_t : trim_t + (y1 - y0),
                    trim_l : trim_l + (x1 - x0),
                ]
        return result

    @staticmethod
    def _resolve_geometry(wsi, ds: float, level, limit_bounds: bool,
                          level_rule: str = 'best') -> tuple:
        """Which level to read, and which rectangle of it.

        Returns (lv, ds_lv, origin_x, origin_y, span_w, span_h, rw, rh).
        origin_* and span_* are LEVEL-0; rw and rh are that span expressed at
        level lv, which is also the shape of the mask that comes out. Keeping
        the two apart is the whole reason this is its own function: a length in
        one system silently used in the other is the bug class this module has
        already paid for twice.
        """
        p = wsi.properties
        w0, h0 = wsi.level_dimensions[0]

        n_levels = len(wsi.level_dimensions)
        if level is not None:
            lv = level if level >= 0 else n_levels + level
        elif level_rule == 'nearest':
            lv = wsi.nearest_level_for_downsample(ds)
        elif level_rule == 'best':
            lv = wsi.get_best_level_for_downsample(ds)
        else:
            raise ValueError(f"level_rule must be 'best' or 'nearest', "
                             f'got {level_rule!r}')

        if limit_bounds:
            origin_x = int(p.get('openslide.bounds-x', 0))
            origin_y = int(p.get('openslide.bounds-y', 0))
            span_w   = int(p.get('openslide.bounds-width',  w0))
            span_h   = int(p.get('openslide.bounds-height', h0))
        else:
            origin_x = origin_y = 0
            span_w, span_h = w0, h0

        ds_lv = wsi.level_downsamples[lv]
        rw = max(1, int(span_w / ds_lv))
        rh = max(1, int(span_h / ds_lv))
        return lv, ds_lv, origin_x, origin_y, span_w, span_h, rw, rh

    @classmethod
    def _segment_plane(cls, wsi, lv: int, ds_lv: float,
                       origin_x: int, origin_y: int, rw: int, rh: int,
                       method, seg_chunk_px, read_chunk_px, overlap) -> np.ndarray:
        """The (rh, rw) uint8 mask of level `lv`, however it has to be got.

        Two ways in, and the difference is only where the pixels come from:
        read_chunk_px reads and segments tile by tile so the level is never in
        memory whole, otherwise the level is read in one call first. _tiled_apply
        takes a get_tile callable precisely so both are the same code.
        """
        def _read(y0: int, x0: int, y1: int, x1: int) -> np.ndarray:
            """One (y0:y1, x0:x1) rect of level `lv`, as RGB.

            read_region takes a LEVEL-0 location but a level-lv size, so the
            offset is scaled on its way into the location and must NOT be on its
            way into the size -- the same asymmetry SafeSlide._read_halved
            documents. Rounding rather than truncating, because a MIRAX
            level_downsample is a float near but not equal to 2 and a truncated
            offset would drift the grid by a pixel every few tiles.

            read_region_rgb, not .convert('RGB'), whenever the handle offers it.
            convert() merely drops the alpha channel, and unphotographed pixels
            carry RGB 0, so every MIRAX hole and every never-scanned corner comes
            out pure black. HSV and Otsu happen to reject black -- sat 0 fails
            `sat > 15`, and _mask_otsu excludes gray <= 20 -- so this was
            harmless while those were the only methods. A segmentation model has
            no such rule and will call a black field tissue, which puts
            tissue_regions over areas that have no image at all. Compositing onto
            openslide.background-color (white when unset) makes those pixels what
            they physically are, blank glass, which all three methods reject.

            SafeSlide's stricter read_region_valid is deliberately not used: at
            ds=1 the validity plane is another full-size array, and after
            compositing "no image" and "white glass" are the same thing to a
            tissue mask. If a hole surrounded by tissue ever does get bridged by
            a seg model, SafeSlide.holes records where to check and
            `main_mask &= valid` is the answer then.
            """
            loc = (origin_x + int(round(x0 * ds_lv)),
                   origin_y + int(round(y0 * ds_lv)))
            size = (x1 - x0, y1 - y0)
            if hasattr(wsi, 'read_region_rgb'):
                return wsi.read_region_rgb(loc, lv, size)
            return np.array(wsi.read_region(loc, lv, size).convert('RGB'))

        if read_chunk_px and rw * rh > read_chunk_px:
            # The budget is the smaller of two constraints that are not the same
            # thing: seg_chunk_px is VRAM per forward pass, read_chunk_px is host
            # RAM per read. One grid serves both, so a heavy method just makes
            # the tiles smaller -- at the cost of more seams, which is a design
            # question this collapse currently hides.
            budget = min(read_chunk_px, seg_chunk_px) if seg_chunk_px else read_chunk_px
            return cls._tiled_apply(method, rh, rw, _read, budget, overlap,
                                    source='read+seg')

        # img is local to this function, so it is gone by the time the caller
        # runs the connected-component pass; no del needed.
        img = _read(0, 0, rh, rw)
        if seg_chunk_px is not None and img.shape[0] * img.shape[1] > seg_chunk_px:
            return cls._tiled_apply(method, rh, rw,
                                    lambda y0, x0, y1, x1: img[y0:y1, x0:x1],
                                    seg_chunk_px, overlap, source='seg')
        return method(img)

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
        filter_regions kept 15. Decimating also drops the int32 label plane
        cv2 allocates from 4 bytes per pixel to 4/stride**2, which was the last
        thing in from_wsi still scaling with the slide.
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

    # POSITION converters subtract origin_*; LENGTH converters must not, because
    # a width is not anchored anywhere. has_tissue_l0 / levelloc / mpploc used to
    # push both through the same function, which is harmless only while the mask
    # starts at level-0 (0, 0). With a cropped mask it would shift every size by
    # the crop offset, so the two cases are now named apart.

    def _mppLength_converter(self, w: float, h: float,
                             mpp: Union[float, tuple[float, float]]) -> tuple[int, int]:
        mpp_x, mpp_y = mpp if isinstance(mpp, tuple) else (mpp, mpp)
        return int(w * mpp_x / self.mask_mpp), int(h * mpp_y / self.mask_mpp)

    def _mppCoordinate_converter(self, x: float, y: float,
                                 mpp: Union[float, tuple[float, float]]) -> tuple[int, int]:
        mpp_x, mpp_y = mpp if isinstance(mpp, tuple) else (mpp, mpp)
        base_x = self.wsi_mpp_x or self.mask_mpp
        base_y = self.wsi_mpp_y or self.mask_mpp
        x0 = x * mpp_x / base_x        # -> absolute level-0 px
        y0 = y * mpp_y / base_y
        return self.to_mask_xy(x0, y0)

    def _levelLength_converter(self, w: float, h: float, level: int) -> tuple[int, int]:
        ds = self.wsi_level_downsamples[level]
        return int(w * ds / self.mask_ds_x), int(h * ds / self.mask_ds_y)

    def _levelCoordinate_converter(self, x: float, y: float, level: int) -> tuple[int, int]:
        ds = self.wsi_level_downsamples[level]
        return self.to_mask_xy(x * ds, y * ds)

    # @classmethod
    # def from_wsi(cls, wsi: openslide.OpenSlide,
    #              thumb_size: tuple[int, int] = (2048, 2048),
    #              method: callable = None) -> 'TissuesRegionsMask':
    #     wsi_width  = wsi.level_dimensions[0][0]
    #     wsi_height = wsi.level_dimensions[0][1]
    #     wsi_mpp_x  = float(wsi.properties.get('openslide.mpp-x', 0))
    #     wsi_mpp_y  = float(wsi.properties.get('openslide.mpp-y', 0))
    #     wsi_level_downsamples = wsi.level_downsamples
    #     if method is None:
    #         method = _mask_hsv
    #     thumbnail = np.array(wsi.get_thumbnail(thumb_size).convert('RGB'))
    #     mask      = method(thumbnail)
    #     main_mask = mask.astype(bool)
    #     mask_ds_x = wsi_width  / mask.shape[1]
    #     mask_ds_y = wsi_height / mask.shape[0]
    #     mask_mpp  = (wsi_mpp_x + wsi_mpp_y) / 2 * (mask_ds_x + mask_ds_y) / 2
    #     tissue_regions = cls._search_tissue_regions(main_mask, mask_ds_x, mask_ds_y)
    #     return cls(main_mask=main_mask, mask_ds_x=mask_ds_x, mask_ds_y=mask_ds_y,
    #                mask_mpp=mask_mpp, tissue_regions=tissue_regions,
    #                wsi_width=wsi_width, wsi_height=wsi_height,
    #                wsi_mpp_x=wsi_mpp_x, wsi_mpp_y=wsi_mpp_y,
    #                wsi_level_downsamples=wsi_level_downsamples)

    @classmethod
    def from_wsi(cls, wsi: openslide.OpenSlide,
                 ds: float = 32.0,
                 level: int = None,
                 method: callable = None,
                 seg_chunk_px: int = None,
                 overlap: int = 128,
                 limit_bounds: bool = True,
                 read_chunk_px: int = None,
                 level_rule: str = 'best') -> 'TissuesRegionsMask':
        '''
        Args:
            ds:         Target downsample factor (level-0 px / output px).
                        The closest WSI level with native downsample <= ds is
                        selected per `level_rule` and read in full.
                        Default 32 gives thumbnail-like resolution for Otsu/HSV.
                        Use a smaller value (e.g. 8) for deep-learning seg models
                        that need higher resolution.  Ignored when level is given.
            level:      WSI level to read directly.  Supports negative indexing:
                        -1 = last (lowest-resolution) level, -2 = second to last.
                        Overrides ds when specified.
            method:     callable(img: np.ndarray) -> np.ndarray (uint8 or bool).
                        Receives the RGB level image; returns a binary tissue mask
                        of the same spatial size.  Defaults to HSV thresholding.
            seg_chunk_px: If set and the level image exceeds it, adaptively halve
                        the longer side of the current tile grid until every tile
                        fits within seg_chunk_px; apply `method` to each tile and
                        stitch.  Useful when `method` is a heavy seg model that
                        would OOM on the whole image (e.g. HEST on a large MRXS).
                        None = single call on the whole image (backward compat).
            overlap:    Per-tile margin (px), trimmed after inference to avoid
                        seam artifacts.  Only used when seg_chunk_px is active.
                        Default 128 covers a typical DeepLabV3 receptive field.
            level_rule: How `ds` picks a level. 'best' keeps openslide's
                        get_best_level_for_downsample, whose rule is "the last
                        level whose downsample does not exceed ds" -- a strict
                        comparison, so a level reporting 4.00003 loses to a
                        request for 4.0 and segmentation happens one level
                        finer. On BRACS_1228 that is 6.58 Gpx instead of
                        411 Mpx: 646 s instead of about 40.
                        'nearest' picks the closest level by ratio instead.
                        Default stays 'best' so existing results remain
                        reproducible; new callers should pass 'nearest' and
                        record which they used, because the two produce
                        different region boundaries and the masks are not
                        interchangeable.
            limit_bounds: Read only openslide.bounds-* (the scanned rectangle)
                        instead of the whole canvas.  A MIRAX canvas is the
                        stage travel range, not the slide: on Ki67 the scanned
                        area is 16 percent of it and the rest has no image data,
                        which read_region returns as transparent.  Passing a
                        SafeSlide handle now composites that onto the background
                        colour rather than leaving it black, so the crop is
                        about cost rather than correctness -- but it still cuts
                        the work by that same factor, and a plain OpenSlide
                        handle keeps the old black.
                        Formats without the property (SVS) are
                        unaffected: the crop becomes the full canvas.
                        The mask array then starts at (bounds-x, bounds-y), kept
                        in self.origin_*; tissue_regions stay in absolute
                        level-0 coordinates regardless.
            read_chunk_px: If set and the level exceeds it, read the level in
                        tiles instead of in one call, segmenting each tile as it
                        arrives so the whole level is never in memory. This is
                        the only bound on HOST memory: seg_chunk_px caps the GPU
                        per forward pass, but the read before it was one array
                        whose size scaled with the slide -- 16 bytes per level-0
                        pixel measured end to end, which is 299 GB on the
                        largest MRXS here at ds=1. Tiling drops the peak to the
                        mask itself plus one tile, and what is left is the
                        connected-component pass at roughly 5 bytes/px.
                        Default None keeps the single read, so every existing
                        caller behaves exactly as before.
                        NOT safe with a method that needs the whole image:
                        _mask_otsu derives its threshold globally, and per tile
                        it would threshold blank glass against its own noise.
                        _mask_hsv is per-pixel and HEST is fully convolutional,
                        so both are unaffected -- the same constraint
                        _tiled_apply already documents.
        '''
        wsi_width  = wsi.level_dimensions[0][0]
        wsi_height = wsi.level_dimensions[0][1]
        wsi_mpp_x  = float(wsi.properties.get('openslide.mpp-x', 0))
        wsi_mpp_y  = float(wsi.properties.get('openslide.mpp-y', 0))
        wsi_level_downsamples = wsi.level_downsamples

        if method is None:
            method = _mask_hsv

        lv, ds_lv, origin_x, origin_y, span_w, span_h, rw, rh = \
            cls._resolve_geometry(wsi, ds, level, limit_bounds, level_rule)

        mask = cls._segment_plane(wsi, lv, ds_lv, origin_x, origin_y, rw, rh,
                                  method, seg_chunk_px, read_chunk_px, overlap)

        main_mask = mask.astype(bool)
        del mask          # 1 byte/px, and main_mask is the copy that is kept

        # The numerator has to be the level-0 span the mask actually covers, not
        # the canvas: pairing the canvas width with a cropped mask width would
        # inflate mask_ds by 1/crop_fraction and silently misplace everything.
        # Without a crop span_* IS the canvas, so this is the old formula.
        mask_ds_x = span_w / main_mask.shape[1]
        mask_ds_y = span_h / main_mask.shape[0]
        mask_mpp  = (wsi_mpp_x + wsi_mpp_y) / 2 * (mask_ds_x + mask_ds_y) / 2

        tissue_regions = cls._search_tissue_regions(
            main_mask, mask_ds_x, mask_ds_y,
            origin_x=origin_x, origin_y=origin_y,
        )

        return cls(main_mask=main_mask,
                   mask_ds_x=mask_ds_x,
                   mask_ds_y=mask_ds_y,
                   mask_mpp=mask_mpp,
                   tissue_regions=tissue_regions,
                   wsi_width=wsi_width,
                   wsi_height=wsi_height,
                   wsi_mpp_x=wsi_mpp_x,
                   wsi_mpp_y=wsi_mpp_y,
                   wsi_level_downsamples=wsi_level_downsamples,
                   origin_x=origin_x,
                   origin_y=origin_y)

    def loc(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        return self.main_mask[y:y+h, x:x+w]
    
    def mpploc(self, x: int, y: int, w: int, h: int, mpp: Union[float, tuple[float, float]]) -> np.ndarray:
        x, y = self._mppCoordinate_converter(x, y, mpp)
        w, h = self._mppLength_converter(w, h, mpp)
        return self.main_mask[y:y+h, x:x+w]

    def levelloc(self, x: int, y: int, w: int, h: int, level: int) -> np.ndarray:
        x, y = self._levelCoordinate_converter(x, y, level)
        w, h = self._levelLength_converter(w, h, level)
        return self.main_mask[y:y+h, x:x+w]

    def has_tissue_l0(self, x: int, y: int, w: int, h: int, tissue_ratio: float = 0.5) -> bool:
        x, y = self._levelCoordinate_converter(x, y, 0)
        w, h = self._levelLength_converter(w, h, 0)
        return self.has_tissue(x, y, w, h, tissue_ratio)

    def has_tissue(self, x: int, y: int, w: int, h: int, tissue_ratio: float = 0.5) -> bool:
        """Tissue fraction of a rect given in MASK coords, clipped to the mask.

        Clipping is not cosmetic. With a cropped mask a level-0 position left of
        or above the crop converts to a negative mask coordinate, and
        main_mask[y:y+h, x:x+w] would wrap around from the far edge and answer
        about entirely the wrong part of the slide.

        Area outside the mask counts as background rather than being dropped:
        the inside-tissue count is divided by the FULL requested area, so a rect
        hanging half off the scanned region cannot score 100 percent on the half
        that happens to land on tissue.
        """
        if w <= 0 or h <= 0:
            return False
        H, W = self.main_mask.shape
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            return False
        return float(self.main_mask[y0:y1, x0:x1].sum()) / (w * h) >= tissue_ratio

    def _summed_area_table(self) -> tuple:
        """Cached (table, step) for white_fractions, padded so a rect is four
        lookups.

        The mask is decimated by `step` first when it is larger than
        _INTEGRAL_DECIMATE_ABOVE_PX, and every lookup afterwards has to be in
        decimated coordinates -- which is why the stride comes back with the
        table instead of being recoverable from its shape alone.

        The stride is derived from the mask, not fixed, so a small mask gets
        stride 1 and nothing about the existing behaviour moves.
        """
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

        `xy` is [N, 2] level-0 top-left coordinates; the answer is [N] float32.

        main_mask holds 1 where the segmenter found tissue, so background is its
        complement. Area outside the mask counts as background rather than being
        dropped -- the same choice has_tissue documents, and for the same reason:
        a tile hanging half off the scanned region must not score as pure tissue
        on the half that happens to land on some.

        Vectorised through a summed-area table because "is this level even
        fillable" has to be answerable before any pixel is read, and a level can
        offer 200,000 candidates. has_tissue answers the same question one rect
        at a time and returns a threshold verdict; this returns the fraction
        itself for every candidate at once, which is what a quota needs.
        """
        S, step = self._summed_area_table()
        H, W = S.shape[0] - 1, S.shape[1] - 1        # in DECIMATED mask pixels

        mw, mh = self._levelLength_converter(tile, tile, level)
        # Everything below is in decimated units. The step**2 that would scale
        # both the tissue count and the requested area cancels in the ratio, so
        # it never has to appear -- but the footprint does have to be at least
        # one cell, or a tile smaller than the stride would divide by zero.
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

    def filter_regions(self, min_ratio: float = 0.05) -> None:
        '''Remove tissue_regions that are too small or fully contained by another.

        1. Regions with area < min_ratio * max_region_area
        2. Regions fully contained within another region
        Modifies self.tissue_regions in place; pushes a snapshot onto the
        undo stack (see regions_undo).
        '''
        if not self.tissue_regions:
            return
        self._snapshot()
        max_area = max(r.w * r.h for r in self.tissue_regions)
        threshold = max_area * min_ratio
        kept = [r for r in self.tissue_regions if r.w * r.h >= threshold]

        def contained_by_other(r):
            for o in kept:
                if o is r:
                    continue
                if o.x <= r.x and o.y <= r.y and o.x + o.w >= r.x + r.w and o.y + o.h >= r.y + r.h:
                    return True
            return False

        self.tissue_regions = [r for r in kept if not contained_by_other(r)]

    def filter_patchable(self, tile_size: int, ds: float) -> None:
        '''Remove tissue_regions that cannot produce even one tile at the given level.

        A region is patchable when both its level-0 width and height are >= tile_size * ds.
        Modifies self.tissue_regions in place; pushes a snapshot onto the
        undo stack (see regions_undo).

        Args:
            tile_size: patch size in level-N pixels
            ds:        downsample factor of the target level (level-0 px / level-N px)
        '''
        self._snapshot()
        tile_l0 = tile_size * ds
        self.tissue_regions = [
            r for r in self.tissue_regions if r.w >= tile_l0 and r.h >= tile_l0
        ]

    def merge_overlapping(self) -> None:
        '''Merge tissue_regions whose bboxes partially overlap.

        Criterion (partial overlap only):
          - bbox intersect area > 0
          - AND neither region's bbox fully contains the other
        This means identical / nested cases are left alone (they are
        already handled by filter_regions); only genuine partial overlaps
        get merged.

        Union-find propagates chained overlaps: if A overlaps B and
        B overlaps C, all three collapse into one merged region.
        Each merged region gets the union bbox of its members and a fresh
        index 0..N-1. Modifies self.tissue_regions in place; pushes a
        snapshot onto the undo stack (see regions_undo).
        '''
        regs = self.tissue_regions
        n = len(regs)
        if n < 2:
            return
        self._snapshot()

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
            xs  = [regs[i].x for i in members]
            ys  = [regs[i].y for i in members]
            xes = [regs[i].x + regs[i].w for i in members]
            yes = [regs[i].y + regs[i].h for i in members]
            x0, y0 = min(xs), min(ys)
            x1, y1 = max(xes), max(yes)
            new_regs.append(TissueRegion(
                x=x0, y=y0, w=x1 - x0, h=y1 - y0, index=k,
            ))
        self.tissue_regions = new_regs


# ── Internal mask functions ───────────────────────────────────────────────────

def mask_all(rgb: np.ndarray) -> np.ndarray:
    """Everything is tissue. One blob, so one region covering the whole plane.

    For stage 2. The retriever scores a window by the mean cosine over the
    query's tiles, so a window sitting on blank glass loses on its own merits;
    the tissue mask there is an optimisation, not a correctness requirement.
    Handing it a single region buys something back that the optimisation costs:
    find_best takes a global maximum over every placement in every region, so a
    region with an order of magnitude more placements wins comparisons on
    sample count alone. Measured at about 0.016 of uniform uplift on S1137178,
    enough to displace fifteen matches that SIFT had verified with 218 to 915
    inliers. With one region there is nothing to compare across.

    Public, unlike _mask_hsv and _mask_otsu, because it is meant to be passed
    in by a caller: TissuesRegionsMask.from_wsi(wsi, method=mask_all).

    Tiling-safe by construction -- no global statistic, no neighbourhood, no
    dependence on position.

    WASTEFUL, and knowingly so: from_wsi still reads every pixel of the level
    to hand them to a function that ignores them. At mask_ds=1 that is twenty
    minutes and a full-size array to produce a constant. A constructor that
    fabricates the all-ones mask from the bounds properties alone would do the
    same job with no I/O at all; see log/TODO.log.
    """
    return np.ones(rgb.shape[:2], dtype=bool)


def _mask_hsv(rgb: np.ndarray, sat_thresh: int = 15,
              val_min: int = 30, val_max: int = 240) -> np.ndarray:
    hsv  = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    mask = ((sat > sat_thresh) & (val > val_min) & (val < val_max)).astype(np.uint8)
    k    = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    return mask.astype(bool)


def _mask_otsu(rgb: np.ndarray, black_thresh: int = 20) -> np.ndarray:
    gray  = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    valid = gray[gray > black_thresh]
    if valid.size == 0:
        return np.zeros(gray.shape, dtype=bool)
    thr, _ = cv2.threshold(valid.reshape(-1, 1), 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = ((gray > black_thresh) & (gray < int(thr))).astype(np.uint8)
    k    = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    return mask.astype(bool)

