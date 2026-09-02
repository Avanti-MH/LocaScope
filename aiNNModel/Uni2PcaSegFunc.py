"""Tissue segmentation from UNI2 patch features and one PCA per slide.

    seg  = Uni2PcaSegConfig().build(device)
    mask = seg.mask_wsi(wsi)                       # fit + project, level 0, no args
    trm  = TissuesRegionsMask.from_mask(wsi, mask.mask, mask.origin, mask.span)

THIS IS A SLIDE SEGMENTER, NOT AN IMAGE SEGMENTER
--------------------------------------------------
    hsv / otsu / hest        image  ->  mask
    Uni2PcaSeg               slide  ->  mask

That difference is the whole design, and pretending otherwise is what every
earlier shape of this file was arranged around -- a `plane_ds` field nothing
could check, a level argument that had to be passed twice, an invariant held
together by a docstring, and a `level` that never reached `identity_id`. All of
them were downstream of forcing a slide-shaped operation through `method=`.

It cannot be an image segmenter because it fits a PCA ACROSS the slide before it
can threshold any part of it. `TissuesRegionsMask.from_wsi` reads a plane and
hands it to a callable; there is no plane it could hand this that would let it
do its job. So the mask is made first and `from_mask` does the rest --
`_search_tissue_regions`, the origin bookkeeping, `mask_ds` from the shape.

`__call__` survives as the base-class contract and as an escape hatch for a
caller holding a fitted basis and one level-0 plane. It is not the path.

Ported from `utilities/test_modules/test_EoMT.py`, which took it from
prov-gigapath's own `demo/gigapath_pca_visualization_timm.ipynb`. That file is a
spike and says so; this is the half of it that works.

WHICH HALF OF test_EoMT.py THIS IS, AND WHICH HALF IT IS NOT
-------------------------------------------------------------
That file has two paths that produce a mask and only one of them means anything:

    segment()          test_EoMT.py:218   EoMT's mask + class heads. The heads
                       are randomly initialised and nothing in the repo trains
                       them, which the file states at :40-42 -- "every mask is
                       noise". Not ported.
    slide_pca_mask()   test_EoMT.py:504   PCA on UNI2 patch features,
                       mask = PC1 vs a threshold. Ported here.

The distinction is not a preference. EoMT's queries, class head and mask head
are trained JOINTLY with a backbone; every released weight is for DINOv2 or
DINOv3 on COCO / ADE20K / Cityscapes (`/work/u26130998/eomt/model_zoo/` holds
exactly `dinov2.md` and `dinov3.md`), so none of them attaches to UNI2's feature
space -- upstream even flags the class head as dataset-bound with
`--model.load_ckpt_class_head False`. Training it ourselves would need
tissue/background ground truth, which is what we wanted it to produce.

PCA needs no training BY CONSTRUCTION. It is fitted on this slide's own
features, unsupervised, and on a WSI the largest direction of variance is
tissue against glass. The sign of a principal component is arbitrary, so the
polarity really is a coin toss -- but here it is an explicit, hashed config
field (`larger_pca_as_fg`) rather than an initialisation.

ALIGNED WITH test_EoMT.encode_batch, NOT WITH TileEncoder.spatial()
--------------------------------------------------------------------
`spatial()` (`TileEncoderFunc.py:913`) is this project's dense-feature exit and
would be the obvious reuse. It is deliberately NOT used, because it differs from
the notebook path -- which `test_EoMT.encode_batch:302` follows and whose masks
are the ones anyone here has actually looked at -- in two places. Everything
else is the same call:
`forward_intermediates(indices=1, output_fmt='NCHW', intermediates_only=True)`.

1. THE TRANSFORM. `_run:754` puts every image through `cfg.transform`: for UNI2
   that is `Resize(224)` then `CenterCrop(224)` then Normalize
   (`Uni2Func.py:251`). On an exactly-224 square input the geometry is the
   identity, so at the default tile size this difference does not bite.

   At any other tile size it does, and not subtly: a single-int `Resize` scales
   the SHORTER SIDE keeping aspect, and the centre crop takes a square out of
   the middle. `test_EoMT.py:250-253` records the same hazard in its milder
   form -- the notebook's `Resize(256) + CenterCrop(224)` is a 1.14x zoom
   discarding the outer 12 percent, and "the discarded border of every tile is
   the interior of its neighbour, so the stitched mask would be missing a band
   along every seam."

   `_cells` below therefore normalises the tile itself and feeds it AS READ, so
   `tile` is free to be any multiple of the patch size and no pixel of a tile
   is thrown away at its border.

2. THE FINAL LAYERNORM. `_vit_spatial_forward:965` hard-codes `norm=True`; the
   notebook path passes `norm=False`. That docstring names the discrepancy
   itself -- "prov-gigapath's own PCA notebook takes the default and gets the
   un-normed version" -- and `test_EoMT.patch_features` says from the other side
   that it "is not a free choice".

   It is not cosmetic. timm's `self.norm` is a LayerNorm per token across the
   1536 channels, so it removes the two quantities PC1 is most likely to be
   finding: the per-token MEAN (roughly how strongly this patch responds, which
   on a WSI tracks tissue against glass) and the per-token MAGNITUDE (blank
   glass and dense tissue pulled to one scale). Then a learned per-channel
   affine, and PCA is not invariant to a diagonal rescale -- that moves the
   eigenvectors of the covariance.

   So it is subtracting the candidate signal before the PCA looks for it.
   `feature_norm` is a config field defaulting to False, which is
   `encode_batch`'s value; True reproduces what `spatial()` would have returned,
   for whoever wants to measure the difference rather than argue about it.

WHAT IS STILL TAKEN FROM THE ENCODER LAYER
-------------------------------------------
Everything except those two: `encoder_config(name)` supplies the timm recipe,
the weights, and the `mean`/`std` that `_cells` normalises with. Retyping any of
those is the failure that whole table exists to prevent, and the mean/std are
the sharpest case -- CONCH's were once transcribed from memory as ImageNet's
when `factory.py:71` overrides them to OpenAI CLIP's, and nothing would have
raised.

`identity_parts` folds the encoder's own `identity_id()` in for the same reason:
`weights_id` hashes the state dict, so architecture and weights are covered, but
`transform.mean` is not -- and a changed mean changes every feature the PCA
reads.

WHAT THE MASK'S RESOLUTION ACTUALLY IS
---------------------------------------
One cell is `tile / cells_per_side` pixels -- 14 for UNI2, whose 224 tile comes
back as a 16x16 grid. Everything reads at level 0 (see `LEVEL`), so the mask
lands at **ds 14** and that is not a choice anyone makes; it is what the patch
grid does.

14 is FINER than the ds 32 hsv masks this project has been using, so the
granularity worries that a coarser level would have raised do not arise: a 256 px
tile at ds 1 spans 18 cells, and `tissue_ratio` has room to be a knob.

It is also small. A ds 14 mask of BRACS_1003691 is 4416 x 5040 = 22 MB; the same
mask upsampled to level-0 resolution would be 4.4 GB, and the connected-component
pass over it about 22 GB. That factor is `cell_px` squared, and it is the same
196 that makes `components_wsi` stream rather than hold a plane.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

_HERE = Path(__file__).resolve().parent
for _d in (_HERE, _HERE.parent / 'utilities'):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import cv2                                                  # noqa: E402
import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
from PIL import Image                                       # noqa: E402

from ConfigIdentity import register                         # noqa: E402
from MaskStore import SlideMask                            # noqa: E402
from TissueSegFunc import TissueSegConfig, TissueSegmenter  # noqa: E402


#: The pyramid level everything here reads at, and it is not a parameter.
#:
#: The algorithm has no opinion about magnification: a WSI comes in, it is cut
#: into `tile` squares, each becomes `(tile / patch)^2` feature cells, 1000
#: tiles' worth of cells fit a PCA, and every tile is projected and stitched.
#: Nowhere in that does a level appear. `test_EoMT.TileSet.__getitem__:391`
#: passes the literal 0 for the same reason.
#:
#: A level parameter WAS here, and it existed only because fit and apply had
#: been split apart to serve `TissuesRegionsMask.from_wsi(method=...)` -- which
#: made "the two must share a magnification" an invariant that nothing could
#: check. `mask_wsi` puts them back in one call and the parameter has nothing
#: left to do.
#:
#: The cost that seemed to justify a coarser level is measured and small. From
#: the EoMTest run of 2026-08-24, whole slide, every tile:
#:
#:     BRACS_1003691   276 x 315 = 86,940 tiles    ~410 tiles/s    ~3.5 min
#:     S1103520 Ki67   435 x 622 = 270,570 tiles   ~783 tiles/s    ~5.8 min
#:
#: So the mask lands at ds = patch_size = 14, which is FINER than the ds 32 hsv
#: masks this project has been using, and it is a derived fact rather than a
#: choice -- `slide_pca_mask`'s own words: "nothing here chooses a resolution,
#: the patch grid does".
LEVEL = 0

#: The zero point. Editing this invalidates every mask id ever written, on
#: purpose; editing a dataclass DEFAULT does not -- it splits new from old.
#: ConfigIdentity rule 1.
_UNI2_PCA_BASELINE = {
    'method': 'uni2-pca-seg',
    'encoder': 'uni2',
    'tile': 224,
    'components': 16,
    'background_threshold': 0.5,
    'larger_pca_as_fg': False,
    'morph_kernel': 7,
    'feature_norm': False,
    'fp16': True,
    'fit_tiles': 1000,
    'fit_bins': 10,
    'fit_seed': 0,
    'fit_ds': 32.0,
    'limit_bounds': True,
}


# ── reading the slide ─────────────────────────────────────────────────────────

def scanned_bounds(wsi, limit_bounds: bool = True) -> Tuple[Tuple[int, int],
                                                            Tuple[int, int]]:
    """(origin, span) in LEVEL-0 pixels: the rectangle the scanner covered.

    A third copy of the same six lines -- `SlideWinSift.py:248-255` and
    `test_EoMT.slide_bounds:412` are the others, and `TissuesRegionsMask.
    _resolve_geometry` computes it again as part of a larger answer. Copied
    rather than imported because the alternatives are worse: reaching into a
    private staticmethod of `TissuesRegionsMask` from `aiNNModel/` would point
    this layer sideways, and importing that module pulls openslide and the
    connected-component machinery in for six lines of arithmetic.

    Why it matters here specifically: on a MIRAX the canvas outside
    `openslide.bounds-*` is stage travel range holding no image data. Tiles over
    it read nothing, and a ViT handed a blank tile has only its POSITIONAL
    EMBEDDING varying between patches -- so a PCA fitted on those finds
    position, and the mask comes out in bands. That is not a hypothetical; it is
    what banded the earlier figure in test_EoMT.
    """
    props = wsi.properties
    width, height = wsi.level_dimensions[0]
    if not limit_bounds:
        return (0, 0), (width, height)
    return ((int(props.get('openslide.bounds-x', 0)),
             int(props.get('openslide.bounds-y', 0))),
            (int(props.get('openslide.bounds-width', width)),
             int(props.get('openslide.bounds-height', height))))


def tile_saturation(wsi, origin, span, tile: int, ds: float = 32.0) -> np.ndarray:
    """Mean saturation per tile position, from ONE thumbnail read.

    `test_EoMT.tile_saturation:431`. The cheap content signal: one read_region
    rather than one per tile.

    Saturation rather than luminance because glass and a pale section differ far
    more in colour than in brightness -- a fatty area is nearly white and is
    still tissue, and `TissueSegFunc.mask_hsv` thresholds the same quantity.

    Only a proxy, and only used to CHOOSE what the model looks at. Nothing it
    decides reaches the mask.
    """
    level = wsi.get_best_level_for_downsample(ds)
    got = float(wsi.level_downsamples[level])
    level_w = max(1, int(span[0] / got))
    level_h = max(1, int(span[1] / got))
    small = _read_rgb(wsi, origin, level, (level_w, level_h)).astype(np.float32)

    high, low = small.max(axis=2), small.min(axis=2)
    saturation = np.where(high > 0, (high - low) / np.maximum(high, 1e-6), 0.0)

    n_x, n_y = span[0] // tile, span[1] // tile
    step = tile / got                       # tile side in thumbnail pixels
    out = np.zeros((n_y, n_x), dtype=np.float32)
    for j in range(n_y):
        y0 = int(j * step)
        y1 = max(y0 + 1, int((j + 1) * step))
        for i in range(n_x):
            x0 = int(i * step)
            x1 = max(x0 + 1, int((i + 1) * step))
            out[j, i] = saturation[y0:min(y1, level_h), x0:min(x1, level_w)].mean()
    return out


def stratified_positions(saturation: np.ndarray, n: int, bins: int = 10,
                         seed: int = 0) -> List[Tuple[int, int]]:
    """`n` tile positions spread over saturation AND over the slide.

    `test_EoMT.stratified_positions:462`, unchanged.

    Spatial sampling alone gives a sample proportional to AREA, which is not a
    sample of both classes. Measured on this project's seven slides, tissue
    covers 38.2 / 24.4 / 23.1 percent of the three BRACS and 9.2 / 5.9 / 4.6 /
    3.5 percent of the four Ki67 -- so a uniform 1000 over the last one is about
    35 tiles of tissue against 965 of glass, and the largest variance in that
    sample is the variance AMONG BLANK TILES. See `scanned_bounds` for what a
    PCA of blank tiles finds.

    Quantile bins rather than a high half and a low half: the interesting cells
    are the ones in between -- a section edge, fat, a fold -- and a two-way
    split throws exactly those away.

    Jittered inside each bin rather than centred, because scan grids and TMA
    layouts are periodic and a fixed offset can align with them.

    WHAT THIS DOES NOT DO, because the paragraph above reads as if it might.
    It does not oversample the rare class. Equal picks per QUANTILE bin has the
    same expected composition as uniform sampling: tissue covering 3.5 percent
    of the area lands entirely in the top bin, is 35 percent OF that bin, and
    that bin takes a tenth of the picks -- 3.5 percent of the sample either way.
    Measured at 3.5 against uniform's 2.7 by a test that had asserted otherwise.

    What it does give is COVERAGE at small n: every band contributes, where a
    uniform draw of the same size leaves bands empty by chance. With 15 asked
    for over 10 bins, this returns 10 positions in 10 distinct bands and a
    uniform draw of 10 covers about 6.5. That is the property worth having and
    the one `test_uni2_pca_seg` now checks.
    """
    rng = np.random.default_rng(seed)
    flat = saturation.ravel()
    order = np.argsort(flat, kind='stable')
    per_bin = max(1, n // bins)

    picked: List[int] = []
    for b in range(bins):
        lo = b * len(order) // bins
        hi = (b + 1) * len(order) // bins
        idx = order[lo:hi]
        if len(idx) == 0:
            continue
        step = max(1, len(idx) // per_bin)
        picked.extend(idx[min(len(idx) - 1, k * step + rng.integers(0, step))]
                      for k in range(min(per_bin, len(idx))))

    unique = np.unique(np.array(picked, dtype=np.int64))
    return [(int(k // saturation.shape[1]), int(k % saturation.shape[1]))
            for k in unique]


def _read_rgb(wsi, location, level, size) -> np.ndarray:
    """`SafeSlide.read_region_rgb`, and a sentence if the handle is not one.

    `.convert('RGB')` merely drops alpha, and pixels the scanner never wrote
    carry RGB 0 -- so every hole becomes a black rectangle. A ViT reading a
    black rectangle sees a hard edge that exists in no tissue, and a PCA fitted
    with those in the sample spends a component on them.
    """
    reader = getattr(wsi, 'read_region_rgb', None)
    if reader is None:
        raise TypeError(
            f'{type(wsi).__name__} has no read_region_rgb; open the slide with '
            f'utilities/SafeSlide.SafeSlide. A plain OpenSlide handle returns '
            f'unphotographed pixels as transparent, and dropping the alpha '
            f'paints them black -- see SafeSlide.read_region_rgb')
    return reader(location, level, size)


# ── configuration ─────────────────────────────────────────────────────────────

@register('uni2-pca-seg')
@dataclass(frozen=True)
class Uni2PcaSegConfig(TissueSegConfig):
    """Which encoder, at what magnification, and how the PCA is fitted.

    Every field is identity. That is not caution for its own sake: this
    segmenter has no weights of its own, so `weights_id` covers only the
    ENCODER, and two fits differing in `fit_tiles` or `background_threshold`
    produce different masks. A field left out of the hash would give them one
    name -- the single failure ConfigIdentity rule 1 calls silent.
    """

    method: str = 'uni2-pca-seg'

    #: Name into `TileEncoderFunc.encoder_config`, so the timm recipe, the
    #: normalisation constants and the weights all come from the one table that
    #: knows them. Not 'uni2' by assumption -- gigapath and conch_vit are ViTs
    #: with a patch grid too, and swapping is one flag.
    encoder: str = 'uni2'

    #: NO plane_ds, AND THAT IS THE POINT. It used to hold "the downsample the
    #: masking pass will read at", which had to equal the `ds` the caller passed
    #: to `from_wsi` -- and nothing could check it. `__call__` receives an
    #: ndarray and cannot know its magnification, so a basis fitted at ds 8 and
    #: applied to a ds 32 plane produced a mask of the right shape, silently
    #: wrong. An invariant held together by a docstring is not an invariant.
    #:
    #: The magnification now enters exactly once, as `fit(wsi, level)`, and the
    #: caller passes the SAME level to `from_wsi(level=...)`. That is still two
    #: places, but they are two arguments of the same kind at one call site
    #: rather than a config field and a call argument in different files.

    #: Side of the square handed to the encoder, in plane pixels. Fed as read --
    #: `cfg.transform`'s resize and centre crop are bypassed -- so this only has
    #: to be a multiple of the patch size. 224 is what UNI2 was trained at, and
    #: a ViT run far from that size is running with a positional embedding
    #: interpolated to a magnification it was never fitted for.
    tile: int = 224

    #: How many principal components to keep. Only PC1 reaches the mask, so this
    #: is NOMINALLY free -- and it is not, because sklearn picks the randomized
    #: SVD solver at this sample size and its result depends on how many
    #: components were asked for. Hashed rather than argued about.
    components: int = 16

    #: PC1, MinMax-scaled to [0, 1], against this. The sign of a principal
    #: component is arbitrary, which is what the second field is for -- the
    #: notebook's own flag, kept rather than replaced by a rule.
    #:
    #: The POLARITY is settled. `InspectPcaSeg` of 2026-08-26 drew both masks on
    #: BRACS_1228 (H&E) and S1104233 (Ki67) at four magnifications; True is the
    #: side that lies on the tissue in all eight panels, and False is the side
    #: that lies on the glass and puts a tissue-shaped HOLE in itself -- which
    #: is exactly what it looks like when read as a mask, and did get read that
    #: way once. The default moved with the finding: editing a dataclass default
    #: splits new ids from old rather than invalidating them, rule 1 above.
    #:
    #: The THRESHOLD is NOT settled to the same standard, and 0.5 is still the
    #: notebook's number. PC1's histogram is bimodal on Ki67 (valley near 0.2)
    #: and unimodal on BRACS, so no single value can be read off both, and a
    #: mask cannot answer a question about the field it was thresholded FROM.
    #: That is why `mask_wsi` hands its components to `MaskStore`: with them on
    #: disk a sweep over candidate thresholds costs seconds, and without them it
    #: costs another 3.5 to 6 minutes of GPU per slide per candidate.
    background_threshold: float = 0.5
    larger_pca_as_fg: bool = True

    #: Morphological close-then-open on the thresholded mask, in CELLS. 0 is off.
    #:
    #: The threshold is decided per cell and nothing makes neighbouring cells
    #: agree, so what comes out is salt and pepper: on BRACS_1228 at level 0,
    #: `larger_pca_as_fg=True` selected 12.8 percent of the slide as scattered
    #: specks INSIDE a section that covers about half of it. Handing that to
    #: `_search_tissue_regions` gives thousands of one-cell regions, and
    #: `filter_regions` then deletes almost all of them.
    #:
    #: `TissueSegFunc.mask_hsv:80-84` does the same close-then-open with the same
    #: 7 for the same reason, which is why the number is 7 rather than a new
    #: guess: a per-pixel or per-cell decision needs a spatial prior and this is
    #: the one already in use here.
    #:
    #: At ds 14 a 7-cell kernel spans 98 level-0 px, so it fills gaps inside a
    #: section and removes specks smaller than that. It does NOT rescue a badly
    #: placed threshold -- it makes the chosen side connected, and if the chosen
    #: side is the glass it will make the glass connected.
    morph_kernel: int = 7

    #: Apply the trunk's final LayerNorm before the PCA. False is
    #: `test_EoMT.encode_batch:326` and prov-gigapath's notebook; True is what
    #: `TileEncoder.spatial()` returns. They are different features, not two
    #: spellings of one -- see the module docstring.
    feature_norm: bool = False

    #: fp16 autocast for the encoder forward, `encode_batch`'s own default. It
    #: moves the features by about 1e-3, which principal directions are
    #: insensitive to -- insensitive, not identical, which is why it is hashed
    #: rather than sitting in NOT_IDENTITY.
    fp16: bool = True

    #: The fit sample: how many tile positions, over how many saturation
    #: quantiles, with which seed, and the downsample the saturation thumbnail
    #: is read at. `fit_seed` also seeds sklearn's randomized SVD -- see
    #: `Uni2PcaSegmenter.fit`.
    fit_tiles: int = 1000
    fit_bins: int = 10
    fit_seed: int = 0
    fit_ds: float = 32.0

    #: Restrict everything to `openslide.bounds-*`. See `scanned_bounds`.
    limit_bounds: bool = True

    #: Tiles per encoder forward. Cannot change a single tile's features (a ViT
    #: normalises per sample), so it splits no cache -- the same argument
    #: TileEncoderConfig makes for its own batch_size.
    batch_tiles: int = 64

    #: DataLoader workers for reading tiles. Splits no cache either: the loader
    #: runs with `shuffle=False` and carries each tile's index in the batch, so
    #: results are placed by index and the worker count cannot reorder a sample.
    #: 0 reads in the parent, which is what a test wants.
    workers: int = 8

    NOT_IDENTITY = ('batch_tiles', 'workers')

    def build(self, device: Optional[torch.device] = None) -> 'Uni2PcaSegmenter':
        return Uni2PcaSegmenter(self, device or torch.device('cpu'))


# ── the projection ────────────────────────────────────────────────────────────


class _GpuProjection:
    """sklearn's PCA and MinMaxScaler as two device matmuls.

    `test_EoMT.GpuProjection:332`. `transform` is `(x - mean_) @ components_.T`
    and MinMax is `(z - data_min_) / data_range_` clipped -- both are arithmetic,
    and doing them where the features already are keeps 1536 floats per cell off
    the bus.

    The FIT is the part that needs an algorithm and stays in sklearn, on the
    host, once per slide. This is the part that runs per tile.
    """

    def __init__(self, pca, scaler, device: torch.device):
        def to(array):
            return torch.as_tensor(array, dtype=torch.float32, device=device)
        self.mean = to(pca.mean_)
        self.components = to(pca.components_).T          # [D, k]
        self.low = to(scaler.data_min_)
        self.span = to(scaler.data_range_)

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        z = (features - self.mean) @ self.components
        return ((z - self.low) / self.span).clamp_(0., 1.)


# ── the segmenter ─────────────────────────────────────────────────────────────

class Uni2PcaSegmenter(TissueSegmenter):
    """UNI2 features, one PCA per slide, PC1 thresholded.

        seg = Uni2PcaSegConfig().build(device)
        seg.fit(wsi)                       # REQUIRED; see TissueSegmenter.fit
        mask = seg(rgb_plane)              # [H, W] uint8, 1 = tissue
    """

    BASELINE = _UNI2_PCA_BASELINE

    def __init__(self, cfg: Uni2PcaSegConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self._weights_id = None

        # Imported here rather than at module scope so that importing this file
        # costs nothing but numpy and torch. Every encoder module does
        # os.environ.setdefault('HF_HOME', ...) above its own `import timm`, and
        # huggingface_hub freezes that into module constants when IT is
        # imported -- so an encoder import at the top of a segmenter would make
        # the SEGMENTER decide where the weights live. TileEncoderFunc's
        # _IMPLEMENTATIONS table documents the same hazard.
        from TileEncoderFunc import encoder_config

        encoder = encoder_config(cfg.encoder).build(device)
        trunk = getattr(encoder.model, 'module', encoder.model)
        if not hasattr(trunk, 'forward_intermediates'):
            raise TypeError(
                f'{cfg.encoder!r} builds a {type(trunk).__name__}, which has no '
                f'forward_intermediates. This segmenter needs a dense patch '
                f'grid; a model without one cannot supply per-cell features')

        self.encoder = encoder
        #: `model` is what IdentifiedBuild hashes into weights_id. The encoder's
        #: weights are the only weights involved: the PCA is fitted per slide
        #: and is not part of what this CONFIGURATION is.
        self.model = trunk.eval()

        patch = trunk.patch_embed.patch_size
        self._cell_px = int(patch[0]) if isinstance(patch, (tuple, list)) \
            else int(patch)
        if cfg.tile % self._cell_px:
            raise ValueError(
                f'tile {cfg.tile} is not a multiple of the patch size '
                f'{self._cell_px}; the cell grid would not tile the plane evenly')
        self._cells_per_side = cfg.tile // self._cell_px

        #: The tile is fed as read, so the resize and centre crop of
        #: cfg.transform are bypassed and this module applies the normalisation
        #: itself -- from the encoder's own config, never retyped. See the
        #: module docstring, "WHAT IS STILL TAKEN FROM THE ENCODER LAYER".
        self._pixel_mean = torch.tensor(encoder.cfg.transform.mean,
                                        device=device).view(1, 3, 1, 1)
        self._pixel_std = torch.tensor(encoder.cfg.transform.std,
                                       device=device).view(1, 3, 1, 1)

        self._project: Optional[_GpuProjection] = None
        self._fit_report: Optional[dict] = None

    def identity_parts(self) -> list:
        """This config, plus the ENCODER's whole identity.

        `IdentifiedBuild.identity_parts` gives cfg + weights_id, and weights_id
        hashes the state dict -- so architecture and weights are covered, but
        the encoder's dtype, transform, head and pooling are not. Those live in
        a config this object merely holds, and two segmenters differing in the
        encoder's autocast dtype would otherwise share one id.

        Folding in the encoder's own short id rather than copying its fields:
        one name for one thing, and it stays correct when the encoder grows a
        field this module has never heard of.
        """
        return super().identity_parts() + [
            f'encoder_identity={self.encoder.identity_id()}']

    # ── what the mask's resolution really is ────────────────────────────────

    @property
    def mask_ds(self) -> float:
        """Level-0 pixels per genuinely independent mask cell.

        `cell_px * level_ds`, and since everything reads at level 0 (see
        `LEVEL`) that is just `cell_px` -- 14 for UNI2. The multiplication is
        still written out because `fit` can be handed another level by the
        escape hatch, and a constant here would then be a lie.

        Before a fit there is nothing to look up and the answer is the same
        constant, so it does not raise. It did raise while a level was still a
        parameter, because then the pre-fit answer would have been a guess at
        which level a later call would pick.
        """
        if self._fit_report is None:
            return float(self._cell_px)
        return self._cell_px * float(self._fit_report['level_ds'])

    @property
    def fitted(self) -> bool:
        return self._project is not None

    @property
    def runs(self) -> bool:
        return True

    @property
    def fit_report(self) -> Optional[dict]:
        """What the fit saw. None before `fit`.

        Kept because the fit is the part with no ground truth: explained
        variance and the fraction of sampled cells that came out foreground are
        the only signals that the PCA found tissue rather than position or
        scanner banding.
        """
        return self._fit_report

    # ── fit ─────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def fit(self, wsi, level: int = LEVEL) -> 'Uni2PcaSegmenter':
        """One PCA basis for the whole slide, from a stratified sample.

        Called by `mask_wsi`, which is the entry point. Public only because
        `TissueSegmenter.fit` is, and because `__call__` needs a basis before it
        can do anything -- see its docstring for when that is the path you want.

        `level` DEFAULTS TO 0 AND SHOULD NOT BE PASSED. It is an argument at all
        so `__call__` on a non-level-0 plane is reachable by someone who knows
        what they are doing; every other caller wants the default, because the
        algorithm has no opinion about magnification and the mask's resolution
        falls out of the patch grid rather than being chosen. `LEVEL` above
        carries the reasoning and the measurement.

        One basis for the whole slide, not one per tile. `test_EoMT.
        slide_pca_mask:516-519` gives the reason: MinMaxScaler maps that tile's
        own extremes to 0 and 1, so a tile holding only tissue has its threshold
        land inside tissue.
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import MinMaxScaler

        cfg = self.cfg
        origin, span = scanned_bounds(wsi, cfg.limit_bounds)
        level = int(level)
        level_ds = float(wsi.level_downsamples[level])

        # The tile side in LEVEL-0 pixels, so positions can be addressed in the
        # coordinate system read_region actually takes.
        tile_l0 = int(round(cfg.tile * level_ds))
        saturation = tile_saturation(wsi, origin, span, tile_l0, cfg.fit_ds)
        positions = stratified_positions(saturation, cfg.fit_tiles,
                                         cfg.fit_bins, cfg.fit_seed)
        if not positions:
            raise RuntimeError(
                f'no fit positions: the scanned rectangle {span} holds fewer '
                f'than one {tile_l0} px tile at level {level} (ds {level_ds:g})')

        batches = []
        for tiles, _, _ in self._read_tiles(wsi, origin, positions, level):
            batches.append(self._cells(tiles).cpu().numpy())
        sample = np.concatenate(batches, axis=0).reshape(-1, batches[0].shape[-1])
        del batches

        # random_state, which upstream does not set (test_EoMT.py:569). At this
        # sample size -- order 250k cells of 1536 features -- sklearn's
        # svd_solver='auto' selects the RANDOMIZED SVD, which is stochastic. So
        # without a seed the same config on the same slide gives a different
        # basis and the SAME identity_id: one name, two things, the failure
        # ConfigIdentity exists to prevent.
        pca = PCA(n_components=cfg.components,
                  random_state=cfg.fit_seed).fit(sample)
        scaler = MinMaxScaler(clip=True).fit(pca.transform(sample))
        self._project = _GpuProjection(pca, scaler, self.device)

        scaled_pc1 = scaler.transform(pca.transform(sample))[:, 0]
        foreground = (scaled_pc1 > cfg.background_threshold
                      if cfg.larger_pca_as_fg
                      else scaled_pc1 < cfg.background_threshold)
        self._fit_report = {
            'level': int(level),
            'level_ds': level_ds,
            'tile_l0': tile_l0,
            'positions': len(positions),
            'cells': int(sample.shape[0]),
            'explained_variance_top3':
                float(pca.explained_variance_ratio_[:3].sum()),
            'foreground_fraction_in_sample': float(foreground.mean()),
            'mask_ds': self._cell_px * level_ds,
        }
        return self

    # ── apply ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def __call__(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """One RGB plane -> [H, W] uint8, 1 = tissue. Same contract as HEST.

        Tiles internally at `cfg.tile` and batches, so any input size is
        accepted without the caller having to set `seg_chunk_px`. That is only
        sound because the PCA is already fitted: every tile is projected through
        the SAME basis, which is what makes this method a pure function of its
        input the way `mask_hsv` is and `mask_otsu` is not.

        The cell grid is upsampled back to the input's pixel size before
        returning. `TissuesRegionsMask._tiled_apply:307-311` slices the returned
        mask with the INPUT tile's pixel offsets and assigns into a full-size
        array, so a cell-resolution mask raises there on broadcast -- while the
        single-call path would have accepted it and derived a coarser ds from
        the shape. Returning coarse would therefore work until someone passed
        `seg_chunk_px`. Upsampling is unconditional for that reason; it changes
        no information, and `mask_ds` reports what the granularity really is.
        """
        if not self.fitted:
            raise RuntimeError(
                'the PCA has not been fitted. Call seg.fit(wsi) before handing '
                'this to from_wsi. Fitting inside here instead would give every '
                'tile its own basis, which is the failure from_wsi already '
                'documents for _mask_otsu -- and worse, because MinMaxScaler '
                "maps a tile's own extremes to 0 and 1, so an all-tissue tile "
                'lands its threshold inside tissue')

        components = self.components(image)
        pc1 = components[..., 0]
        tissue = (pc1 > self.cfg.background_threshold
                  if self.cfg.larger_pca_as_fg
                  else pc1 < self.cfg.background_threshold)

        # NO _clean HERE, and the asymmetry with mask_wsi is deliberate.
        #
        # `morph_kernel` is a spatial operation: a close near the edge of this
        # plane sees no neighbours, so the result would depend on WHERE the
        # caller cut the plane. That is exactly the property `from_wsi`'s tiled
        # paths require this method not to have -- and the one
        # `test_uni2_pca_seg` pins by segmenting a plane whole and in quadrants
        # and demanding they agree pixel for pixel.
        #
        # `mask_wsi` holds the whole mask, so there is no cut for morphology to
        # depend on, and it cleans there.

        rgb = np.asarray(image.convert('RGB')) if isinstance(image, Image.Image) \
            else np.asarray(image)
        height, width = rgb.shape[:2]

        # INTER_NEAREST: this is a decode of a cell decision back to the pixels
        # it covers, not a resampling of a continuous field. Bilinear would
        # invent half-tissue pixels along every cell boundary, and the caller's
        # `tissue_ratio` would then read them as real.
        full = cv2.resize(tissue.astype(np.uint8),
                          (tissue.shape[1] * self._cell_px,
                           tissue.shape[0] * self._cell_px),
                          interpolation=cv2.INTER_NEAREST)
        return full[:height, :width].astype(np.uint8)

    @torch.no_grad()
    def components(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """One RGB plane -> [cells_h, cells_w, k] float32, the projected components.

        What `__call__` thresholds, before it thresholds. Kept as a public exit
        for the reason `test_EoMT.slide_pca_mask:520-522` already gives: "The
        components are kept, not just the threshold. One bit per cell throws
        away everything the second and third components hold, and re-deriving
        them costs the whole encode again -- hours, against a few hundred MB."

        A diagnostic asking which side of PC1 is tissue cannot answer it from a
        binary mask -- the mask IS the answer being questioned. It needs the
        continuous field and its distribution, which is this.

        The grid covers the PADDED extent, so it is a whole number of cells and
        may reach past the input by up to one tile. `__call__` crops after
        upsampling; a caller reading this directly gets the padding too, and
        the padding is edge-replicated rather than zero.

        Memory: k float32 per cell, and a cell is `cell_px**2` plane pixels --
        so at k=16 and cell 14 it is about 1/12 byte per plane pixel. A 30k x
        20k plane is ~200 MB.
        """
        rgb = np.asarray(image.convert('RGB')) if isinstance(image, Image.Image) \
            else np.asarray(image)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f'expected an RGB image, got shape {rgb.shape}')
        if not self.fitted:
            raise RuntimeError(
                'the PCA has not been fitted. Call seg.fit(wsi) first; see '
                '__call__ for why fitting lazily here would be wrong')
        height, width = rgb.shape[:2]

        tile = self.cfg.tile
        n_rows = (height + tile - 1) // tile
        n_cols = (width + tile - 1) // tile
        # Edge-replicated rather than zero-padded: a zero pad is black, and
        # black is a hard edge that exists in no tissue. It would land in the
        # cells along the right and bottom margins of every plane.
        padded = cv2.copyMakeBorder(rgb, 0, n_rows * tile - height,
                                    0, n_cols * tile - width,
                                    cv2.BORDER_REPLICATE)

        side = self._cells_per_side
        grid = np.zeros((n_rows * side, n_cols * side, self.cfg.components),
                        dtype=np.float32)

        coordinates = [(r, c) for r in range(n_rows) for c in range(n_cols)]
        for start in range(0, len(coordinates), self.cfg.batch_tiles):
            chunk = coordinates[start:start + self.cfg.batch_tiles]
            batch = np.stack([padded[r * tile:(r + 1) * tile,
                                     c * tile:(c + 1) * tile] for r, c in chunk])
            projected = self._project(self._cells(batch))       # [B, cells, k]
            projected = projected.reshape(len(chunk), side, side, -1)
            projected = projected.float().cpu().numpy()
            for k, (r, c) in enumerate(chunk):
                grid[r * side:(r + 1) * side, c * side:(c + 1) * side] = projected[k]
        return grid

    # ── reading tiles off the slide ─────────────────────────────────────────

    def _read_tiles(self, wsi, origin, positions, level, cells=None):
        """Batches of `[B, tile, tile, 3]` uint8, on worker processes.

        `utilities/WsiTileLoader.py`, which is `test_EoMT`'s reader lifted into
        the shared layer. The cost of a fit or a whole-slide projection is the
        READ, not the model: a serial reader in the parent cannot feed a ViT
        that does 650 tiles/s, and a MIRAX decodes every rect on the way out.

        Takes the slide's PATH from the handle rather than the handle itself.
        An OpenSlide handle carried across a fork is one handle used by several
        processes -- it does not raise, it returns pixels from whatever region
        another process last asked for. Each worker opens its own, lazily.
        """
        from WsiTileLoader import wsi_tile_loader

        path = getattr(wsi, '_filename', None) or getattr(wsi, 'filename', None)
        if path is None:
            raise TypeError(
                f'{type(wsi).__name__} does not expose the file it opened, so '
                f'the reader cannot hand a path to its workers. SafeSlide does')

        loader = wsi_tile_loader(path, origin, self.cfg.tile, positions,
                                 level=level, cells=cells,
                                 batch=self.cfg.batch_tiles,
                                 workers=self.cfg.workers)
        for tiles, small, index in loader:
            # [B, 3, H, W] -> [B, H, W, 3], which is what _cells takes and what
            # the notebook path feeds. The permute is here rather than in the
            # worker because uint8 crossing the queue should stay contiguous.
            yield (tiles.permute(0, 2, 3, 1).numpy(), small.numpy(),
                   index.numpy())

    @torch.no_grad()
    def mask_wsi(self, wsi) -> 'SlideMask':
        """A whole slide in, a tissue mask out. The entry point; no parameters.

            seg  = Uni2PcaSegConfig().build(device)
            mask = seg.mask_wsi(wsi)
            trm  = TissuesRegionsMask.from_mask(wsi, mask.mask,
                                                mask.origin, mask.span)

        Fits and projects in ONE call, which is what makes the two impossible to
        disagree about -- and disagreeing about magnification was the failure
        every earlier shape of this class was arranged around. `slide_pca_mask`
        has the same shape for the same reason.

        The two passes inside are PCA's, not a design choice: principal
        directions have to be computed before anything can be projected onto
        them, and they cannot be computed on everything -- 86,940 tiles is 22
        million cells of 1536 floats, 136 GB. So 1000 tiles fit the basis and
        every tile is projected through it.

        Returns a `SlideMask` rather than a bare array because a mask without
        its origin cannot be placed: on a MIRAX the scanned rectangle starts at
        `openslide.bounds-*`, not at zero. The span is there for the same reason
        -- see `SlideMask`.

        The components ride along in that `SlideMask`. They are already in hand
        here and cost 3.5 to 6 minutes of GPU to get back, and the one parameter
        this class cannot derive -- `background_threshold` -- is a question about
        them, not about the bit they produced. Threshold sweeps read them; the
        mask store writes them; every other reader leaves them on disk.
        """
        components, _ = self.components_wsi(wsi)
        pc1 = components[..., 0].astype(np.float32)
        tissue = (pc1 > self.cfg.background_threshold
                  if self.cfg.larger_pca_as_fg
                  else pc1 < self.cfg.background_threshold)
        tissue = self._clean(tissue)

        origin, _ = scanned_bounds(wsi, self.cfg.limit_bounds)
        covered = (tissue.shape[1] * self._cell_px, tissue.shape[0] * self._cell_px)
        return SlideMask(mask=tissue, origin=origin, span=covered,
                         mask_ds=float(self._cell_px), report=self.fit_report,
                         components=components)

    def _clean(self, tissue: np.ndarray) -> np.ndarray:
        """Close then open, in cells. See `Uni2PcaSegConfig.morph_kernel`.

        CLOSE FIRST, which is the order `mask_hsv` uses and the order that
        matters: close fills the holes inside a section, open then removes the
        specks left outside it. Doing open first would delete the isolated
        cells that close was going to join, and the section would come back
        thinner than it is.
        """
        size = int(self.cfg.morph_kernel)
        if size <= 0:
            return tissue
        kernel = np.ones((size, size), np.uint8)
        mask = tissue.astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask.astype(bool)

    @torch.no_grad()
    def components_wsi(self, wsi, level: int = LEVEL, *,
                       thumbnail: bool = False):
        """The whole scanned rectangle's components, WITHOUT a plane in memory.

        Returns `(components, thumb)` at CELL resolution -- `[rows*gh, cols*gw,
        k]` float16, and either the per-cell mean colour or None.

        This exists because reading the plane as one array does not scale and
        was never necessary. Ki67 at level 1 is ds 2, so its scanned rectangle
        is 3.35 Gpx -- 10 GB of RGB. The COMPONENTS of the same rectangle are
        17.1 Mcells, and at 16 components in float16 that is 547 MB. A factor of
        196, which is `cell_px` squared, and it is why
        `test_EoMT.slide_pca_mask` streams tiles and assembles only the cell
        grid.

        `__call__` remains the `method=` contract for `TissuesRegionsMask`,
        where the plane is the caller's to read and `read_chunk_px` is the
        caller's way to stream it. This is the other direction: for a diagnostic
        that wants the continuous field over a whole slide, and later for
        anything that wants the components rather than one bit per cell.

        Partial tiles at the right and bottom edge are dropped, matching
        `grid_positions` and `slide_pca_mask`.
        """
        from WsiTileLoader import grid_positions

        if not self.fitted:
            self.fit(wsi, level)

        cfg = self.cfg
        origin, span = scanned_bounds(wsi, cfg.limit_bounds)
        level_ds = float(wsi.level_downsamples[level])
        positions, (n_rows, n_cols) = grid_positions(span, cfg.tile, level_ds)
        if not positions:
            raise RuntimeError(
                f'the scanned rectangle {span} holds no whole {cfg.tile} px '
                f'tile at level {level} (ds {level_ds:g})')

        side = self._cells_per_side
        components = np.zeros((n_rows * side, n_cols * side, cfg.components),
                              dtype=np.float16)
        thumb = (np.zeros((n_rows * side, n_cols * side, 3), dtype=np.uint8)
                 if thumbnail else None)

        cells = (side, side) if thumbnail else None
        done = 0
        for tiles, small, index in self._read_tiles(wsi, origin, positions,
                                                    level, cells):
            projected = self._project(self._cells(tiles))
            projected = projected.reshape(len(index), side, side, -1)
            projected = projected.to(torch.float16).cpu().numpy()
            for b, k in enumerate(index.tolist()):
                row, col = positions[k]
                ys = slice(row * side, (row + 1) * side)
                xs = slice(col * side, (col + 1) * side)
                components[ys, xs] = projected[b]
                if thumbnail:
                    thumb[ys, xs] = small[b]
            done += len(index)
            print(f'\r        projecting {done}/{len(positions)} tiles',
                  end='', flush=True)
        print(flush=True)
        return components, thumb

    # ── the one encoder call ────────────────────────────────────────────────

    def _cells(self, tiles_u8: np.ndarray) -> torch.Tensor:
        """[B, tile, tile, 3] uint8 on the host -> [B, cells, D] fp32 on device.

        `test_EoMT.encode_batch:302`, which is the notebook's path. NOT
        `TileEncoder.spatial()` -- the module docstring gives the two reasons,
        and this is where both of them live.

        uint8 across the bus and float on the far side: a 224 tile is 150 KB as
        bytes and 600 KB as float32, so at batch 64 that is 9.6 MB instead of
        38 MB per forward of pure transfer.

        Left on the DEVICE on purpose. What follows is a projection to a handful
        of numbers, so bringing 1536 floats per cell to the host first would be
        the slowest step in the file.

        `forward_intermediates` rather than a hand-rolled reshape, for the three
        reasons `TileEncoderFunc._vit_spatial_forward` gives: it strips
        `num_prefix_tokens` itself -- nine for UNI2, and dropping the wrong
        number would still divide cleanly and give a plausible grid -- it
        derives the grid from the INPUT rather than from `patch_embed.grid_size`,
        and it hands back NCHW.
        """
        from contextlib import nullcontext

        x = torch.from_numpy(np.ascontiguousarray(tiles_u8)).permute(0, 3, 1, 2)
        x = x.to(self.device, non_blocking=True).float().div_(255.)
        x = (x - self._pixel_mean) / self._pixel_std

        autocast = (torch.autocast(device_type=self.device.type,
                                   dtype=torch.float16)
                    if self.cfg.fp16 and self.device.type == 'cuda'
                    else nullcontext())
        with torch.no_grad(), autocast:
            features = self.model.forward_intermediates(
                x, indices=1, norm=self.cfg.feature_norm, output_fmt='NCHW',
                intermediates_only=True)[-1]
        batch, dim = features.shape[0], features.shape[1]
        return features.permute(0, 2, 3, 1).reshape(batch, -1, dim).float()
