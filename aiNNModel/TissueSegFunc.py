"""What every tissue segmenter in this project has to be, and the ones that
need no model at all.

    seg = HestSegConfig().build(device)      # a network
    seg = TissueSegConfig('hsv').build()     # colour thresholds, no model
    seg = TissueSegConfig('').build()        # nothing runs; see below
    binary = seg(rgb)                        # [H, W] uint8, 1 = tissue

A segmenter turns one RGB image into a binary tissue mask, and that is the
whole contract. Everything else -- whether there are weights, whether a GPU is
involved -- varies, and a base class that assumed otherwise would be unusable
by the first method that has no network.

    method   weights   what it is
    'hest'   yes       DeepLabV3 + ResNet-50, in HestSegFunc
    'hsv'    no        saturation and value thresholds, then open/close
    'otsu'   no        Otsu on grayscale, excluding near-black
    ''       no        no segmentation. One region over the whole plane, and
                       nothing is read to arrive at it.

Everything is tissue, without paying for it
-------------------------------------------
'' replaces what used to be a method called mask_all -- a function that took
every pixel of the level and returned np.ones of the same shape. The output was
identical to this one; the cost was not. from_wsi read the whole level to hand
it to a function that ignored it, then allocated the result: 411 MB at
mask_ds=4, 6.6 GB at mask_ds=1, twenty minutes of reading, every element True.

Why it is worth having at all, rather than just segmenting: stage 2 scores a
window by the mean cosine over the query's tiles, so a window sitting on blank
glass loses on its own merits and the tissue mask there is an optimisation, not
a correctness requirement. One region also buys back what the optimisation
costs -- find_best takes a global maximum over every placement in every region,
so a region with an order of magnitude more placements wins comparisons on
sample count alone. Measured at about 0.016 of uniform uplift on S1137178,
enough to displace fifteen matches SIFT had verified with 218 to 915 inliers.
With one region there is nothing to compare across.

Why '' and not None
-------------------
`TissuesRegionsMask.from_wsi(method=None)` used to mean "default to HSV". A
caller asking for no segmentation and silently receiving HSV is the worst
available outcome, so the absence is spelled '' and None is not a value here.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

_HERE = Path(__file__).resolve().parent
for _d in (_HERE, _HERE.parent / 'utilities'):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import cv2                                                  # noqa: E402
import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
from PIL import Image                                       # noqa: E402

from ConfigIdentity import (IdentifiedBuild, IdentifiedConfig,  # noqa: E402
                            ModelConfig, register)


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
    different thresholds and the stitched result has seams. from_wsi's tiled
    path is only sound for per-pixel methods and fully convolutional ones.
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


# ── configuration ─────────────────────────────────────────────────────────────

@register('tissue-seg')
@dataclass(frozen=True)
class TissueSegConfig(IdentifiedConfig):
    """Base for every segmenter config, and complete for the model-free ones.

    An implementation with weights adds a ModelConfig field and its own
    baseline; see HestSegFunc.
    """
    method: str = 'hsv'

    def build(self, device: Optional[torch.device] = None) -> 'TissueSegmenter':
        if self.method not in NO_MODEL:
            raise ValueError(
                f'{type(self).__name__} handles {NO_MODEL}; {self.method!r} '
                f'needs its own config class -- see HestSegFunc')
        return TissueSegmenter(self, device)


# ── the segmenter ─────────────────────────────────────────────────────────────

class TissueSegmenter(IdentifiedBuild):
    """Callable, so it goes straight where `method=` used to.

        TissuesRegionsMask.from_wsi(wsi, method=TissueSegConfig('hsv').build())

    `model` is None for every method here, which is a first-class state and not
    a placeholder: weights_id comes out '' because there are no weights to
    record, the same honesty as an empty sampler_id.
    """

    BASELINE = {'method': 'hsv'}

    def __init__(self, cfg: TissueSegConfig, device: Optional[torch.device] = None):
        self.cfg = cfg
        self.device = device
        self.model = None
        self._weights_id = None

    @property
    def runs(self) -> bool:
        """False when there is nothing to run, so a caller can skip the read.

        The whole point of method='': a caller that segments tile by tile should
        ask this BEFORE reading the slide, not call a function that returns ones
        for every pixel it was handed.
        """
        return self.cfg.method != ''

    def fit(self, wsi, level: Optional[int] = None) -> 'TissueSegmenter':
        """Look at the whole slide, once, before any tile is segmented.

        A no-op here and for HEST, because their answer does not depend on the
        slide: hsv and otsu carry their rule in a constant, and HEST's fit
        happened at MahmoodLab. Returning self so `cfg.build(dev).fit(wsi)`
        reads as one expression.

        It exists because one segmenter genuinely needs it. Uni2PcaSegFunc fits
        a PCA on the slide's own features, and where that fit happens decides
        whether the segmenter is tiling-safe at all:

            fitted HERE      __call__ becomes a pure transform, so every tile
                             gets the same basis and from_wsi's seg_chunk_px /
                             read_chunk_px paths are sound
            fitted in
            __call__         each tile gets its own basis. This is exactly the
                             failure from_wsi already documents for _mask_otsu
                             -- "per tile it would threshold blank glass
                             against its own noise" -- and worse, because
                             MinMaxScaler maps a tile's own extremes to 0 and 1,
                             so an all-tissue tile lands its threshold inside
                             tissue

        So the split is: `fit` is where segmenters differ, `__call__` is where
        they are the same. A caller that always calls `fit` before `from_wsi`
        works with every method and pays nothing for the ones that ignore it.

        `level` is the pyramid level the masking pass will read, when the
        implementation needs its fit to come from the same magnification it will
        be applied at. None means "decide from the config".
        """
        return self

    def __call__(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        if not self.runs:
            raise RuntimeError(
                "method='' has nothing to run. A caller asking for no "
                "segmentation should check .runs and skip the pass; "
                "TissuesRegionsMask.from_wsi does. Fabricating ones here would "
                "be the mask_all this replaced -- same answer, plus a full read "
                "of the level and a full-size array to hold a constant")
        rgb = np.asarray(image.convert('RGB')) if isinstance(image, Image.Image) \
            else image
        return _FUNCS[self.cfg.method](rgb)


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
    # Cheap to import: it holds its `from TileEncoderFunc import encoder_config`
    # inside build(), so naming it here costs no timm and touches no HF_HOME.
    # That laziness is the reason it can be listed alongside HEST at all.
    from Uni2PcaSegFunc import (Uni2PcaSegConfig,          # noqa: E402,F401
                                Uni2PcaSegmenter)
