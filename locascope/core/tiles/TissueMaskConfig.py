"""The recipe that produces a TissueMask, as one value -- and where masks come
from, cached or not.

    cfg = MASK_RECIPES['hest']
    mask = cfg.build(wsi, device)                          # one slide, no cache

    with MaskMaker(cfg, job_name('MyJob'), device) as masks:   # a loop over slides
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
from typing import Dict, List, Optional, Tuple

from Cache import (Address, Entry, check_source, source_key,   # noqa: E402
                   wsi_stem_of)
from ConfigIdentity import (IdentifiedConfig, ModelConfig, enc,  # noqa: E402
                            parts_of, record,
                            register, short_id)
from HestSegFunc import HEST_ARCH, HestSegConfig                 # noqa: E402
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

    `made_by` is the job whose cache the masks are read from and written to;
    None means every call segments. A slide's raw mask is the `mask` entry at
    `slide=<slide>/` (`entry`): `mask_<seg_id>.safetensors`, and its record
    written LAST, so a job killed in between leaves a miss that is redone.
    """

    def __init__(self, cfg: TissueMaskConfig, made_by: Optional[str] = None,
                 device=None):
        self.cfg = cfg
        self.made_by = made_by
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

    def entry(self, slide: str) -> Entry:
        """The `mask` entry of `slide` in this maker's cache."""
        if self.made_by is None:
            raise ValueError('this MaskMaker has no cache')
        return Address(self.made_by, slide=slide).entry('mask')

    def stored(self, slide: str) -> Optional[Dict]:
        """What the cached mask of `slide` was written with, or None."""
        return self.entry(slide).stored(self.cfg.seg_id())

    def slide_mask(self, wsi, *, with_components: bool = False
                   ) -> Tuple[SlideMask, bool]:
        """The raw mask, cached when there is a cache. `(slide_mask, hit)`."""
        if self.made_by is None:
            return self._segment(wsi, with_components), False

        entry, seg_id = self.entry(wsi_stem_of(wsi)), self.cfg.seg_id()
        want = self.record()
        state, stale = entry.status(seg_id, want)
        if state != 'miss':
            check_source(entry.stored(seg_id), wsi, entry.record_path(seg_id))
        if state == 'hit':
            return SlideMask.load(entry.path('mask', seg_id, '.safetensors'),
                                  with_components=with_components), True
        if state == 'stale':
            print(f'  [mask] {entry.record_path(seg_id)} is stale, segmenting '
                  f'again: ' + '; '.join(stale), flush=True)

        slide_mask = self._segment(wsi, with_components)
        with entry.writing(seg_id, dict(
                want, seg_id=seg_id,
                segmenter_id=self.segmenter.identity_id(),
                source=source_key(wsi),
                wsi_path=str(getattr(wsi, '_filename', '') or ''),
                created_at=time.strftime('%Y-%m-%dT%H:%M:%S'),
                **slide_mask.geometry())) as put:
            slide_mask.save(put('mask', '.safetensors'))
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
#:
#: Every field of every config is written out (test_config_identity's recipe
#: lint): a recipe reads as the mask it makes, and a class whose default moves
#: does not move a recipe with it.
MASK_RECIPES: Dict[str, TissueMaskConfig] = {
    'none': TissueMaskConfig(
        seg=PlaneSegConfig(
            method='', limit_bounds=True, ds=4.0, seg_chunk_px=4_000_000,
            read_chunk_px=4_000_000, stitch_overlap=128),
        min_region_ratio=0.01, merge=True),
    'hsv': TissueMaskConfig(
        seg=PlaneSegConfig(
            method='hsv', limit_bounds=True, ds=4.0, seg_chunk_px=4_000_000,
            read_chunk_px=4_000_000, stitch_overlap=128),
        min_region_ratio=0.01, merge=True),
    'hest': TissueMaskConfig(
        seg=HestSegConfig(
            method='hest', limit_bounds=True, ds=4.0, seg_chunk_px=4_000_000,
            read_chunk_px=4_000_000, stitch_overlap=128,
            model=ModelConfig(source='torchvision', arch=HEST_ARCH,
                              dtype='fp32', weights=None)),
        min_region_ratio=0.01, merge=True),
    'uni2_pca': TissueMaskConfig(
        seg=Uni2PcaSegConfig(
            method='uni2-pca-seg', encoder='uni2', tile=224, components=16,
            background_threshold=0.5, larger_pca_as_fg=True, morph_kernel=7,
            feature_norm=False, fp16=True, fit_tiles=1000, fit_bins=10,
            fit_seed=0, fit_ds=32.0, limit_bounds=True, batch_tiles=64,
            workers=8),
        min_region_ratio=0.01, merge=True),
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
