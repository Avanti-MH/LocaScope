"""The recipe that produces a TissueMask, as one value -- and where masks come
from, cached or not.

    cfg = MASK_RECIPES['hest']
    mask = cfg.build(wsi, device)                          # one slide, no cache

    with MaskMaker(cfg, cache_root, device) as masks:      # a loop over slides
        for wsi in slides:
            mask, hit = masks.mask(wsi)                    # segmentation cached

Three modules, three jobs, one direction:

    TissueMaskConfig  ->  TissueSegFunc (producer)  ->  TissueMask (product)
    recipe + cache        slide -> SlideMask            SlideMask + regions

The product imports neither of the others, the way WsiFeaturesMap knows nothing
about the encoder that filled it. Merging this module into TissueMask would
invert that: the product would import every segmenter.

TWO IDENTITIES, BECAUSE TWO STAGES, AND THE CONFIG'S SHAPE SAYS WHICH. `seg`
is everything the segmentation depends on -- the method and how it reads the
slide both live on the segmenter's own config -- and `seg_id` hashes exactly
that; it names the directory a raw mask is cached under. The fields beside it
are the region prep after it, and `region_id` hashes those. Segmentation is
minutes of GPU per slide and the region prep is milliseconds, so changing
`min_region_ratio` changes `region_id` and never resegments. A field cannot be
on the wrong side, because the side is the nesting.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from Cache import (check_source, read_meta, source_key,          # noqa: E402
                   wsi_stem_of, write_meta)
from ConfigIdentity import (IdentifiedConfig, enc, parts_of,     # noqa: E402
                            record, record_diff,
                            register, short_id)
from HestSegFunc import HestSegConfig                            # noqa: E402
from TissueMask import SlideMask, TissueMask                     # noqa: E402
from TissueSegFunc import PlaneSegConfig, TissueSegConfig        # noqa: E402
from Uni2PcaSegFunc import Uni2PcaSegConfig                      # noqa: E402


@register('tissue-mask')
@dataclass(frozen=True)
class TissueMaskConfig(IdentifiedConfig):
    """Everything that decides which regions a slide has.

    Every field is identity: each one moves the region list, and there is no
    performance knob among them. The segmenter's own config carries the ones
    that move the SEGMENTATION (see TissueSegFunc); the two here only move
    the regions found in it.

    `seg` HAS NO DEFAULT, so no caller gets a segmentation without naming it:
    `MASK_RECIPES['hest']`.

    The order of the region prep is fixed here -- `filtered`, then `merged` --
    because `merged` is incomplete on its own by design: it skips nested and
    identical boxes on the assumption that `filtered` already removed them.
    `patchable` is NOT here. It is a consequence of the tile's footprint,
    decided by whoever tiles, on a view.
    """
    seg: TissueSegConfig
    min_region_ratio: float = 0.01
    merge: bool = True

    BASELINE = {'seg': 'HestSegConfig', 'min_region_ratio': 0.01, 'merge': True}

    def seg_parts(self) -> List[str]:
        """What the segmentation depends on: the segmenter config's own parts,
        and the content of any weights it names by path
        (`TissueSegConfig.weights_key`) -- a key has to see a finetune
        overwritten in place, and a cache hit is exactly when no model is
        built to hash its parameters."""
        weights = self.seg.weights_key()
        return (self.seg.identity_parts()
                + ([f'weights_sha={enc(weights)}'] if weights else []))

    def seg_id(self) -> str:
        """`<method>-<id>`: the directory a raw mask is cached under."""
        return f'{self.seg.method or "none"}-{short_id(self.seg_parts())}'

    def region_id(self) -> str:
        """The id of the region prep after the segmentation: this config's own
        fields, the segmenter's left to `seg_id` one directory up."""
        return short_id(parts_of(self, exclude=('seg',)))

    def regions(self, wsi, slide_mask: SlideMask) -> TissueMask:
        """Search, filter, merge -- in that order, once. The cheap half."""
        mask = TissueMask(wsi, slide_mask).filtered(self.min_region_ratio)
        return mask.merged() if self.merge else mask

    def build(self, wsi, device=None) -> TissueMask:
        """One slide, no cache, segmenter built and dropped with the call. A
        loop over slides wants `MaskMaker`, which builds it once."""
        with MaskMaker(self, device=device) as masks:
            return masks.mask(wsi)[0]


class MaskMaker:
    """Where masks come from: one recipe, one device, optionally one cache.

    Owns the segmenter's lifetime: it is built on the first MISS, never on a
    hit, and dropped when the `with` block ends, so a HEST or UNI2 model does
    not stay on the GPU.

    `cache_root` is `Cache.cache_root(<made_by>, 'mask')`; None means every
    call segments. A slide's raw mask lives at `<cache_root>/<seg_id>/<slide>/`
    as `mask.safetensors` and `mask_meta.json`, the sidecar written LAST: a job
    killed between the two leaves a mask with no sidecar, which reads as a miss
    and is redone.
    """

    def __init__(self, cfg: TissueMaskConfig, cache_root=None, device=None):
        self.cfg = cfg
        self.cache_root = Path(cache_root) if cache_root is not None else None
        self.device = device
        self._segmenter = None

    # ── lifetime ────────────────────────────────────────────────────────────

    @property
    def segmenter(self):
        if self._segmenter is None:
            self._segmenter = self.cfg.seg.build(self.device)
        return self._segmenter

    def close(self) -> None:
        if self._segmenter is None:
            return
        self._segmenter = None
        try:
            import torch                                            # noqa: PLC0415
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def __enter__(self) -> 'MaskMaker':
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ── masks ───────────────────────────────────────────────────────────────

    def slide_dir(self, slide: str) -> Path:
        if self.cache_root is None:
            raise ValueError('this MaskMaker has no cache_root')
        return self.cache_root / self.cfg.seg_id() / slide

    def slide_mask(self, wsi, *, with_components: bool = False
                   ) -> Tuple[SlideMask, bool]:
        """The raw mask, cached when there is a cache. `(slide_mask, hit)`."""
        if self.cache_root is None:
            return self._segment(wsi, with_components), False

        folder = self.slide_dir(wsi_stem_of(wsi))
        data, meta_path = folder / 'mask.safetensors', folder / 'mask_meta.json'
        want = self.record()
        if meta_path.exists():
            meta = read_meta(meta_path)
            check_source(meta, wsi, meta_path)
            stale = record_diff(meta.get('identity'), want)
            if not stale:
                return SlideMask.load(data, with_components=with_components), True
            print(f'  [mask] {meta_path} is stale, segmenting again: '
                  + '; '.join(stale), flush=True)

        slide_mask = self._segment(wsi, with_components)
        slide_mask.save(data)
        write_meta(meta_path, dict(
            seg_id=self.cfg.seg_id(),
            identity=want,
            segmenter_id=self.segmenter.identity_id(),
            source=source_key(wsi),
            wsi_path=str(getattr(wsi, '_filename', '') or ''),
            created_at=time.strftime('%Y-%m-%dT%H:%M:%S'),
            **slide_mask.geometry()))
        return slide_mask, False

    def record(self) -> Dict:
        """The identity record of this recipe's raw masks: the segmenter
        config, its VERSIONs and the reader's, the weights' content and the
        environment."""
        from SlideReader import SlideReader                         # noqa: PLC0415
        return record(self.cfg.seg, also=(SlideReader,),
                      weights=self.cfg.seg.weights_key())

    def mask(self, wsi) -> Tuple[TissueMask, bool]:
        """`slide_mask`, then the region prep -- which is never cached: it is
        cheap, and it depends on `region_id`, which the raw mask does not.
        `hit` means the segmentation was not run."""
        slide_mask, hit = self.slide_mask(wsi)
        return self.cfg.regions(wsi, slide_mask), hit

    def _segment(self, wsi, with_components: bool) -> SlideMask:
        slide_mask = self.segmenter.segment_slide(wsi)
        if with_components and slide_mask.components is None:
            raise ValueError(
                f'{type(self.segmenter).__name__} produces no components')
        return slide_mask


# ── the recipes -- one name, one mask, wherever it is asked for ───────────────

#: `--seg <name>` everywhere resolves here.
#:
#: none      one region per scanned rectangle; nothing is read. Stage 2's
#:           whole-slide search, where blank glass loses on its own merits.
#: hsv       colour thresholds at ds 4.
#: hest      DeepLabV3 at ds 4. The baseline.
#: uni2_pca  a slide segmenter; its resolution is UNI2's patch grid (ds 14).
MASK_RECIPES: Dict[str, TissueMaskConfig] = {
    'none': TissueMaskConfig(seg=PlaneSegConfig('')),
    'hsv': TissueMaskConfig(seg=PlaneSegConfig('hsv')),
    'hest': TissueMaskConfig(seg=HestSegConfig()),
    'uni2_pca': TissueMaskConfig(seg=Uni2PcaSegConfig()),
}


def add_mask_args(ap, default: str = 'hest') -> None:
    """`--seg` and the plane-read overrides, the same flags in every CLI that
    builds a mask, so every recipe a tool uses is one any other tool can
    name."""
    ap.add_argument('--seg', choices=sorted(MASK_RECIPES), default=default,
                    help='tissue-mask recipe (TissueMaskConfig.MASK_RECIPES)')
    ap.add_argument('--mask-ds', type=float, default=None,
                    help="override the recipe's segmentation ds (plane "
                         'segmenters only: none / hsv / hest)')
    ap.add_argument('--seg-chunk-px', type=float, default=None,
                    help="override the recipe's pixels per forward pass "
                         '(plane segmenters only; 0 = unbounded)')
    ap.add_argument('--read-chunk-px', type=float, default=None,
                    help="override the recipe's pixels per slide read "
                         '(plane segmenters only; 0 = read the level whole)')
    ap.add_argument('--min-region-ratio', type=float, default=None,
                    help="override the recipe's region filter")


def mask_cfg_from_args(args, base: Optional[TissueMaskConfig] = None
                       ) -> TissueMaskConfig:
    """The recipe `--seg` names, with any override applied. The result is a
    different config and therefore a different `seg_id` / `region_id`, so an
    override can never be served a cached mask made without it.

    `base` is a config the caller keeps in one visible place (a bench's CONFIG
    block). It is what a tool uses when `--seg` was not given -- which needs
    `add_mask_args(ap, default=None)` -- and `--seg` still names a recipe over it.
    Without a `base`, no `--seg` means the hest recipe, as always."""
    if args.seg is not None:
        cfg = MASK_RECIPES[args.seg]
    else:
        cfg = base if base is not None else MASK_RECIPES['hest']
    seg_over = {}
    if args.mask_ds is not None:
        seg_over['ds'] = float(args.mask_ds)
    for name in ('seg_chunk_px', 'read_chunk_px'):
        value = getattr(args, name)
        if value is not None:
            seg_over[name] = int(value) or None
    if seg_over:
        if not isinstance(cfg.seg, PlaneSegConfig):
            raise ValueError(
                f'{args.seg or "the base config"} reads the slide itself; '
                f'{", ".join(sorted(seg_over))} do not apply to it')
        cfg = replace(cfg, seg=replace(cfg.seg, **seg_over))
    if args.min_region_ratio is not None:
        cfg = replace(cfg, min_region_ratio=float(args.min_region_ratio))
    return cfg
