"""The recipe that produces a TissuesRegionsMask, as one value.

    cfg  = TissueMaskConfig(seg=HestSegConfig(), ds=4.0)
    mask = cfg.build(wsi, device)
    mask.mask_id()          # what a store records

Separate from TissuesRegionsMask, and the reason is the dependency direction
rather than file size. This module imports the segmenter; the mask must not.
A product does not import its producer, and the same split already holds one
level over:

    EncoderConfig    -> GigaPathEncoder  -> WsiFeaturesMap
    TissueMaskConfig -> TissueSegmenter  -> TissuesRegionsMask

WsiFeaturesMap lives in PatchingLib and knows nothing about GigaPathFunc. Merging
this into TissuesRegionsMask inverted that -- and made a module whose
dependencies were cv2, numpy and openslide pull torch and torchvision, which was
the visible symptom of the inversion rather than the problem itself.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
for _d in (_HERE, _HERE.parent / 'aiNNModel'):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from ConfigIdentity import (IdentifiedConfig, register,      # noqa: E402
                            parts_against, short_id)
from TissueSegFunc import TissueSegConfig                    # noqa: E402
from TissuesRegionsMask import TissuesRegionsMask            # noqa: E402


# Three things used to be spread across every caller and recorded nowhere:
#
#     TissuesRegionsMask.from_wsi(wsi, ds=..., method=..., seg_chunk_px=...,
#                                 stitch_overlap=..., level_rule=...)
#     mask.filter_regions(min_ratio=...)
#     mask.merge_overlapping()
#
# Fifteen call sites wrote some subset of those, in an order that matters, and
# the mask that came out could not say which subset it was. That is the whole
# problem: a feature store keyed on `mask_id` is only as good as that string,
# and 'hest@ds4' is the same string whether min_region_ratio was 0.10 or 0.30.
#
# THE ORDER IS FIXED HERE because merge_overlapping is incomplete on its own by
# design: it skips nested and identical boxes on the assumption that
# filter_regions has already removed them (see its docstring). Run it first and
# every nested region survives. The dependency is one way and has no assertion
# behind it, so it belongs in one place that always does it the same way rather
# than in two lines every caller writes.
#
# filter_patchable is NOT here. filter_regions and merge_overlapping are choices
# about the segmentation, made once; filter_patchable is a consequence of the
# SCALE -- which regions can host a tile at a given ds -- and
# WsiTissuesContainer.from_ds already applies it, to a regions_view() so the mask
# itself is never narrowed. Its result is visible in a store's geometry and ds is
# already an identity field, so putting it here would record one fact twice and
# let the two disagree.

#: The zero point. LocaScopePipeline's defaults, so a pipeline-built mask hashes
#: to "all baseline" and anything else says how it differs. Editing this
#: invalidates every mask id ever written, on purpose; editing a dataclass
#: DEFAULT does not -- it splits new from old instead.
_MASK_BASELINE = {
    'seg': TissueSegConfig('hsv'),
    'ds': 4.0,
    'level_rule': 'best',
    'limit_bounds': True,
    'seg_chunk_px': 4_000_000,
    'stitch_overlap': 128,
    'read_chunk_px': None,
    'min_region_ratio': 0.01,
    'merge': True,
}


@register('tissue-mask')
@dataclass(frozen=True)
class TissueMaskConfig(IdentifiedConfig):
    """Everything that decides which regions a slide has.

    Every field is identity. That is unusual here -- most configs have a
    NOT_IDENTITY set -- and it is because there is no performance knob among
    them: each one moves the region list, and the region list is what a stored
    feature map is indexed by.

    read_chunk_px included on purpose despite looking like one. It is nominally
    about memory, but the tiling is min(read_chunk_px, seg_chunk_px), so it can
    change where the tile boundaries fall and therefore what a non-per-pixel
    method produces at the seams.
    """
    seg: TissueSegConfig = field(default_factory=lambda: TissueSegConfig('hsv'))

    #: Segmentation resolution. Not the encoding scale -- that is the
    #: container's ds and lives in the store's own identity.
    ds: float = 4.0
    level_rule: str = 'best'
    limit_bounds: bool = True

    seg_chunk_px: int = 4_000_000
    stitch_overlap: int = 128
    read_chunk_px: Optional[int] = None

    min_region_ratio: float = 0.01
    merge: bool = True

    def identity_parts(self, baseline=None):
        return parts_against(self, _MASK_BASELINE if baseline is None else baseline)

    def mask_id(self) -> str:
        """The short name a FeatureStore records under mask_id.

        Its only job is to keep different masks in different files. Getting it
        wrong does not produce a wrong answer -- WsiFeaturesMapStore recomputes
        the region geometry and compares it against the stored coordinates -- it
        produces two configurations overwriting each other's file in turn, a
        cache that never hits. Visible, and cheap to fix; unlike the silent kind.
        """
        return short_id(self.identity_parts())

    def build(self, wsi, device=None) -> TissuesRegionsMask:
        """Segment, filter, merge -- in that order, once."""
        segmenter = self.seg.build(device)
        mask = TissuesRegionsMask.from_wsi(
            wsi,
            ds=self.ds,
            method=segmenter,
            seg_chunk_px=self.seg_chunk_px,
            stitch_overlap=self.stitch_overlap,
            read_chunk_px=self.read_chunk_px,
            limit_bounds=self.limit_bounds,
            level_rule=self.level_rule,
        )
        mask.filter_regions(min_ratio=self.min_region_ratio)
        if self.merge:
            mask.merge_overlapping()

        # The mask can now say how it was made. Nothing in TissuesRegionsMask
        # reads this -- it is carried so that a caller holding only a mask can
        # answer the question a mask could not answer before, which is the
        # question a cache has to ask.
        mask.cfg = self
        return mask
