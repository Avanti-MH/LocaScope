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
                 wsi_level_downsamples: list[float]):
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

    def __len__(self):
        return len(self.tissue_regions)
    
    def __getitem__(self, index):
        return self.tissue_regions[index]
    
    def __iter__(self):
        return iter(self.tissue_regions)

    def tissue_fraction(self) -> float:
        return float(self.main_mask.mean())

    @staticmethod
    def _adaptive_apply(method: callable,
                        img: np.ndarray,
                        max_pixels: int,
                        overlap: int) -> np.ndarray:
        """
        Tile-and-stitch wrapper so heavy segmentation models can run on WSI
        thumbnails larger than the GPU can process in one pass.

        Algorithm
        ---------
        1. Adaptive halving:
             n_h = n_w = 1
             while (H // n_h) * (W // n_w) > max_pixels:
                 halve whichever current tile side is longer
           Grid ends up as 2^i x 2^j — 1x1, 1x2, 2x2, 2x4, 4x4, 4x8, ...
           A 6000x12000 image with max_pixels=4M lands on 4x8 = 32 tiles.

        2. Per-tile inference with overlap:
           Each tile's core rect is expanded by `overlap` pixels on every
           side (clipped to image bounds) so `method` sees enough context
           around the seam.  The expanded rect is passed to `method`.

        3. Trim and stitch:
           The `overlap` margin is trimmed from the returned mask before
           writing the tile's core rect into the preallocated (H, W) result.
           Adjacent tiles' cores meet exactly at the seam — no blending
           needed for a binary class output.

        Constraint
        ----------
        Only valid when `method` is spatially invariant: fully-convolutional
        segmentation networks with no positional embedding (e.g. DeepLabV3
        used by HEST).  A method that depends on absolute coordinates would
        produce different outputs on tiled vs whole-image inference.

        Args
        ----
        method:     img -> mask callable.  Same signature as passed to from_wsi.
        img:        (H, W, 3) uint8 RGB level image read by from_wsi.
        max_pixels: budget per tile (H_tile * W_tile <= max_pixels after halving).
                    Trade-off: larger = fewer tiles, more GPU memory per pass.
                    ~4M px keeps DeepLabV3 under ~5 GB VRAM.
        overlap:    context margin (px) per tile side, trimmed after inference.
                    Should exceed the model's receptive field to remove seams.
                    128 covers DeepLabV3's ASPP receptive field.

        Returns
        -------
        (H, W) uint8 mask, same shape as `img` spatially, values in {0, 1}.

        Notes
        -----
        Prints the chosen tile grid before running so the caller can verify
        the memory budget is sensible.
        """
        H, W = img.shape[:2]

        n_h = n_w = 1
        while (H // n_h) * (W // n_w) > max_pixels:
            if H // n_h >= W // n_w:
                n_h *= 2
            else:
                n_w *= 2

        tile_h = H // n_h
        tile_w = W // n_w
        print(f'  tiled seg: {n_h}x{n_w} = {n_h * n_w} tiles at '
              f'~{tile_h}x{tile_w} each (input {H}x{W}, budget '
              f'{max_pixels / 1e6:.1f}M px, overlap={overlap})', flush=True)

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

                tile_mask = method(img[y0e:y1e, x0e:x1e])

                trim_t = y0 - y0e
                trim_l = x0 - x0e
                result[y0:y1, x0:x1] = tile_mask[
                    trim_t : trim_t + (y1 - y0),
                    trim_l : trim_l + (x1 - x0),
                ]
        return result

    @staticmethod
    def _search_tissue_regions(mask: np.ndarray,
                               mask_ds_x: float, mask_ds_y: float,
                               min_area_px: int = 100) -> list[TissueRegion]:
        """Find connected tissue blobs; return level-0 bounding boxes."""
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        regions = []
        for label in range(1, n_labels):      # 0 is background
            if stats[label, cv2.CC_STAT_AREA] < min_area_px:
                continue
            mx = int(stats[label, cv2.CC_STAT_LEFT])
            my = int(stats[label, cv2.CC_STAT_TOP])
            mw = int(stats[label, cv2.CC_STAT_WIDTH])
            mh = int(stats[label, cv2.CC_STAT_HEIGHT])
            regions.append(TissueRegion(
                x=int(mx * mask_ds_x),
                y=int(my * mask_ds_y),
                w=int(mw * mask_ds_x),
                h=int(mh * mask_ds_y),
                index=len(regions),
            ))
        return regions

    def _mppCoordinate_converter(self, x: float, y: float, mpp: Union[float, tuple[float, float]]) -> tuple[int, int]:
        if isinstance(mpp, tuple):
            mpp_x, mpp_y = mpp
        else:
            mpp_x = mpp_y = mpp
        return int(x * mpp_x / self.mask_mpp), int(y * mpp_y / self.mask_mpp)

    def _levelCoordinate_converter(self, x: float, y: float, level: int) -> tuple[int, int]:
        return (int(x * self.wsi_level_downsamples[level] / self.mask_ds_x),
                int(y * self.wsi_level_downsamples[level] / self.mask_ds_y))

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
                 max_pixels: int = None,
                 overlap: int = 128) -> 'TissuesRegionsMask':
        '''
        Args:
            ds:         Target downsample factor (level-0 px / output px).
                        The closest WSI level with native downsample <= ds is
                        selected via get_best_level_for_downsample and read in full.
                        Default 32 gives thumbnail-like resolution for Otsu/HSV.
                        Use a smaller value (e.g. 8) for deep-learning seg models
                        that need higher resolution.  Ignored when level is given.
            level:      WSI level to read directly.  Supports negative indexing:
                        -1 = last (lowest-resolution) level, -2 = second to last.
                        Overrides ds when specified.
            method:     callable(img: np.ndarray) -> np.ndarray (uint8 or bool).
                        Receives the RGB level image; returns a binary tissue mask
                        of the same spatial size.  Defaults to HSV thresholding.
            max_pixels: If set and the level image exceeds it, adaptively halve
                        the longer side of the current tile grid until every tile
                        fits within max_pixels; apply `method` to each tile and
                        stitch.  Useful when `method` is a heavy seg model that
                        would OOM on the whole image (e.g. HEST on a large MRXS).
                        None = single call on the whole image (backward compat).
            overlap:    Per-tile margin (px), trimmed after inference to avoid
                        seam artifacts.  Only used when max_pixels is active.
                        Default 128 covers a typical DeepLabV3 receptive field.
        '''
        wsi_width  = wsi.level_dimensions[0][0]
        wsi_height = wsi.level_dimensions[0][1]
        wsi_mpp_x  = float(wsi.properties.get('openslide.mpp-x', 0))
        wsi_mpp_y  = float(wsi.properties.get('openslide.mpp-y', 0))
        wsi_level_downsamples = wsi.level_downsamples

        n_levels = len(wsi.level_dimensions)
        if level is not None:
            lv = level if level >= 0 else n_levels + level
        else:
            lv = wsi.get_best_level_for_downsample(ds)
        W, H     = wsi.level_dimensions[lv]
        img      = np.array(wsi.read_region((0, 0), lv, (W, H)).convert('RGB'))

        if method is None:
            method = _mask_hsv
        if max_pixels is not None and img.shape[0] * img.shape[1] > max_pixels:
            mask = cls._adaptive_apply(method, img, max_pixels, overlap)
        else:
            mask = method(img)
        main_mask = mask.astype(bool)
        mask_ds_x = wsi_width  / mask.shape[1]
        mask_ds_y = wsi_height / mask.shape[0]
        mask_mpp  = (wsi_mpp_x + wsi_mpp_y) / 2 * (mask_ds_x + mask_ds_y) / 2
        tissue_regions = cls._search_tissue_regions(main_mask, mask_ds_x, mask_ds_y)

        return cls(main_mask=main_mask,
                   mask_ds_x=mask_ds_x,
                   mask_ds_y=mask_ds_y,
                   mask_mpp=mask_mpp,
                   tissue_regions=tissue_regions,
                   wsi_width=wsi_width,
                   wsi_height=wsi_height,
                   wsi_mpp_x=wsi_mpp_x,
                   wsi_mpp_y=wsi_mpp_y,
                   wsi_level_downsamples=wsi_level_downsamples)

    def loc(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        return self.main_mask[y:y+h, x:x+w]
    
    def mpploc(self, x: int, y: int, w: int, h: int, mpp: Union[float, tuple[float, float]]) -> np.ndarray:
        x, y = self._mppCoordinate_converter(x, y, mpp)
        w, h = self._mppCoordinate_converter(w, h, mpp)
        return self.main_mask[y:y+h, x:x+w]
    
    def levelloc(self, x: int, y: int, w: int, h: int, level: int) -> np.ndarray:
        x, y = self._levelCoordinate_converter(x, y, level)
        w, h = self._levelCoordinate_converter(w, h, level)
        return self.main_mask[y:y+h, x:x+w]
    
    def has_tissue_l0(self, x: int, y: int, w: int, h: int, tissue_ratio: float = 0.5) -> bool:
        x, y = self._levelCoordinate_converter(x, y, 0)
        w, h = self._levelCoordinate_converter(w, h, 0)
        return self.has_tissue(x, y, w, h, tissue_ratio)

    def has_tissue(self, x: int, y: int, w: int, h: int, tissue_ratio: float = 0.5) -> bool:
        return w * h > 0 and self.main_mask[y:y+h, x:x+w].mean() >= tissue_ratio

    def filter_regions(self, min_ratio: float = 0.05) -> None:
        '''Remove tissue_regions that are too small or fully contained by another.

        1. Regions with area < min_ratio * max_region_area
        2. Regions fully contained within another region
        Modifies self.tissue_regions in place.
        '''
        if not self.tissue_regions:
            return
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
        Modifies self.tissue_regions in place.

        Args:
            tile_size: patch size in level-N pixels
            ds:        downsample factor of the target level (level-0 px / level-N px)
        '''
        tile_l0 = tile_size * ds
        self.tissue_regions = [
            r for r in self.tissue_regions if r.w >= tile_l0 and r.h >= tile_l0
        ]


# ── Internal mask functions ───────────────────────────────────────────────────

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

