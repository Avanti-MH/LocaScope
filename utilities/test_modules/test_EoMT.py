#!/usr/bin/env python3
"""EoMT with a UNI2 backbone: everything in one file, on purpose, for now.

    python utilities/test_modules/test_EoMT.py
    python utilities/test_modules/test_EoMT.py --img 518 --num-q 100
    python utilities/test_modules/test_EoMT.py --image /path/to/tile.png

A SPIKE, not a settled test. It sits in test_modules because that is where it
was asked for; it does not test one module, it stands up a pipeline across two
repos to find out whether they fit. When the answer is known it splits into a
real EoMTSegFunc and a real test, and this file goes away.

What is imported and what is copied
-----------------------------------
The rule is one line: code that needs no change is imported, code that needs a
change is copied here and changed, with the original line numbers so a reader
can diff rather than trust. Same convention as ConchVitFunc.

    IMPORTED   models/eomt.py        204 lines, unchanged. It reaches the
                                     backbone only through a list of attribute
                                     names, and a timm ViT has every one.
               models/scale_block.py imported transitively by eomt.py.

    COPIED     models/vit.py         68 lines, four deviations, marked below.
                                     Its whole job is to hand EoMT a backbone,
                                     and every assumption it makes is about
                                     DINOv2.

Why vit.py cannot be used as it is
----------------------------------
UNI2's hub config carries six keys: architecture, num_classes, num_features,
global_pool, and pretrained_cfg's input_size / mean / std. Eight of the twelve
kwargs UNI2 needs are absent -- reg_tokens, mlp_layer, act_layer, init_values,
no_embed_class, depth, num_heads, mlp_ratio -- and `vit.py` has nowhere to put
them. timm's pretrained loader filters shape-mismatched keys rather than
refusing them, so the result is a 1408-wide 40-block tower where UNI2 is 1536
wide and 24 deep, most of it freshly random, running to completion and emitting
plausible vectors. The first check below builds it both ways and prints the two.

The head is randomly initialised. Nothing here trains anything, so every mask
is noise. What is being checked is that the shapes line up, that the backbone
is the one we think, and what it costs.
"""

import argparse
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

#: BEFORE torch. huggingface_hub freezes HF_HOME into module constants at its
#: own import and something reaches it during `import torch`, so a setdefault
#: further down is too late and 2.7 GB comes down again into $HOME.
os.environ.setdefault(
    'HF_HOME', os.environ.get('LOCASCOPE_MODEL_WEIGHTS',
                              '/work/u26130998/model_weights'))

#: The eomt clone, so `from models.eomt import EoMT` resolves -- eomt.py itself
#: does `from models.scale_block import ScaleBlock`, so its root has to be here
#: rather than its models/ directory.
_EOMT = Path(os.environ.get('EOMT_ROOT', '/work/u26130998/eomt'))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'))
for _d in (_EOMT,):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import numpy as np                                               # noqa: E402
import timm                                                      # noqa: E402
import torch                                                     # noqa: E402
import torch.nn as nn                                            # noqa: E402

from _paths import (encoder_tag, job_result_dir,                 # noqa: E402
                    setup_import_paths)
setup_import_paths()

from TileEncoderFunc import encoder_config                       # noqa: E402
from models.eomt import EoMT                                     # noqa: E402


# ══ copied from eomt/models/vit.py, four deviations ═══════════════════════════
#
# Original is 68 lines and the class is :15-45 plus transformers_to_timm at
# :47-68. What changed, and why:
#
#  1  :25   the dispatch. `if "/" in backbone_name` sends anything with a slash
#           to AutoModel.from_pretrained, and UNI2's arch is
#           'hf-hub:MahmoodLab/UNI2-h' -- a timm model on the hub, not a
#           transformers one. It has a slash and would take the wrong branch.
#           Gone: this only ever builds timm models.
#
#  2  :33-39 the kwargs. create_model got img_size, patch_size and num_classes
#           and nothing else. **timm_kwargs is the hole UNI2 falls through
#           without; see the module docstring.
#
#  3  :41-42 pixel_mean / pixel_std were the ImageNet constants, written out.
#           That is correct for DINOv2, which is vit.py's default backbone, and
#           for UNI2, which follows the same recipe. It is wrong for CONCH,
#           which descends from OpenCLIP: means agree to three decimals, stds
#           differ by 15 to 22 percent. Parameters now, defaulting to nothing.
#
#  4  :47-68 transformers_to_timm is dropped. It renames a HuggingFace model's
#           attributes to timm's, for the branch removed in 1. Keeping it would
#           mean importing `transformers` at module level for a path never
#           taken -- and vit.py does exactly that at :12.
#
# Everything else is theirs.

class ViT(nn.Module):
    """What EoMT calls `encoder`. It reads exactly two things off this object
    and the rest off `self.backbone`; see t_backbone_meets_the_contract."""

    def __init__(
        self,
        img_size: tuple,
        backbone_name: str,
        mean,
        std,
        patch_size: Optional[int] = None,
        pretrained: bool = True,
        **timm_kwargs,
    ):
        super().__init__()

        kwargs = dict(timm_kwargs)
        kwargs['img_size'] = img_size
        if patch_size is not None:
            # Their default is 16 while their default backbone is patch 14, so
            # vit.py re-patchifies DINOv2 on purpose: timm resamples
            # patch_embed.proj, and patch 16 buys int(log2(16))-2 = 2 upscale
            # blocks instead of 1, plus a size that divides 512. Left as None
            # here so the backbone keeps its own unless a caller insists.
            kwargs['patch_size'] = patch_size

        self.backbone = timm.create_model(backbone_name, pretrained=pretrained,
                                          num_classes=0, **kwargs)

        pixel_mean = torch.tensor(mean).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor(std).reshape(1, -1, 1, 1)

        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)


# ══ building it ═══════════════════════════════════════════════════════════════

def build_encoder(name: str, img: int, pretrained: bool = True,
                  recipe: bool = True, patch_size: Optional[int] = None) -> ViT:
    """A ViT wrapper around one of this project's encoders, at `img`.

    The recipe and the normalisation both come from the encoder's own config
    rather than being retyped. That table is the one thing in this repo that
    knows what UNI2 has to be built with, and a second copy here would be the
    copy that goes stale -- which is the failure the whole file is about.

    recipe=False is the decoy: the same arch built the way vit.py would build
    it, to show what is lost.
    """
    cfg = encoder_config(name)
    kwargs = dict(cfg.timm_kwargs) if recipe else {}

    # img_size and patch_size belong to ViT's signature, and the recipe carries
    # both, so they have to come out of the kwargs or the call has two values
    # for one name. They are not the same kind of number:
    #
    #   img_size    the recipe says 224, the size the weights were TRAINED at.
    #               The caller is choosing the size to RUN at, and the whole
    #               point here is that they differ.
    #   patch_size  the recipe says 14, which is the backbone's own. Kept
    #               unless the caller overrides, because overriding resamples
    #               patch_embed.proj -- see the --patch-size help.
    kwargs.pop('img_size', None)
    recipe_patch = kwargs.pop('patch_size', None)
    if patch_size is None:
        patch_size = recipe_patch

    return ViT(img_size=img, backbone_name=cfg.model.arch,
               mean=cfg.transform.mean, std=cfg.transform.std,
               patch_size=patch_size, pretrained=pretrained, **kwargs)


def load_image(path: str, img: int, dev) -> torch.Tensor:
    """One RGB image as [1, 3, img, img] in [0, 1].

    NOT normalised. eomt.py:151 does (x - pixel_mean) / pixel_std itself, so
    handing it a normalised tensor would run and be wrong by one affine step.
    """
    from PIL import Image
    pil = Image.open(path).convert('RGB').resize((img, img), Image.BICUBIC)
    x = torch.from_numpy(np.array(pil)).permute(2, 0, 1).float() / 255.0
    return x[None].to(dev)


# ══ inference ═════════════════════════════════════════════════════════════════

def to_per_pixel_logits_semantic(mask_logits, class_logits):
    """lightning_module.py:668-675, verbatim.

    Imported would be better, but importing it drags LightningModule and with
    it lightning, wandb and their dataset classes for four lines. Copied, and
    it is four lines.

    The [..., :-1] drops the no-object logit. What comes out is [B, K, H, W],
    and the queries are gone -- which is what makes their windowed inference
    sound for semantic segmentation: two windows never have to agree about
    which query is which, only about what class 1 means.
    """
    return torch.einsum(
        "bqhw, bqc -> bchw",
        mask_logits.sigmoid(),
        class_logits.softmax(dim=-1)[..., :-1],
    )


def segment(net, x, num_classes: int):
    """[B, 3, H, W] in [0,1] -> [B, H, W] uint8 class map, at input resolution.

    The last of the per-layer predictions is the one to read: eomt.py appends
    one per masked-attention block (:175) and one after the loop (:197), and
    the earlier ones exist for deep supervision during training.
    """
    with torch.no_grad():
        mask_per_layer, class_per_layer = net(x)
    per_pixel = to_per_pixel_logits_semantic(mask_per_layer[-1],
                                             class_per_layer[-1])
    per_pixel = torch.nn.functional.interpolate(
        per_pixel.float(), size=x.shape[-2:], mode='bilinear', align_corners=False)
    return per_pixel.argmax(1).to(torch.uint8)


# ══ the PCA mask, from prov-gigapath's own demo ═══════════════════════════════
#
# demo/gigapath_pca_visualization_timm.ipynb, applied to one tile and then to
# every tile of a slide. Two PCAs, both fitted on the tile in hand:
#
#     pca_features = MinMaxScaler(clip=True).fit_transform(PCA(3).fit_transform(f))
#     fg_indices   = pca_features[:, 0] > background_threshold
#     fg_features  = PCA(3).fit_transform(f[fg_indices])
#     result[fg_indices] = scaler.transform(fg_features)
#
# The first PCA only separates foreground from background; the second colours
# the foreground, and doing it on the foreground alone is why the colours are
# not washed out by a large flat background.
#
# ONE deliberate deviation, and only one. The notebook transform is
# Resize(256, BICUBIC) then CenterCrop(224), which on a 224 input is a 1.14x
# zoom that throws away the outer 12 percent. Harmless for two loose tiles and
# wrong for tiling a slide: the discarded border of every tile is the interior
# of its neighbour, so the stitched mask would be missing a band along every
# seam. The tile is fed at 224 as read.

def demo_pca(features: np.ndarray, background_threshold: float = 0.5,
             larger_pca_as_fg: bool = False):
    """(rgb, fg) for one tile's patch features, exactly as the notebook does it.

    Fitted on THIS tile and nothing else. The notebook fits on its batch, which
    is the two tiles it loads; per tile is the same operation with a batch of
    one. 256 samples for 1536 features is fine -- sklearn caps the components
    at min(n_samples, n_features) and three is well under.

    larger_pca_as_fg exists because the sign of a principal component is
    arbitrary. It is the notebook's own flag, kept rather than replaced by a
    rule, because replacing it is a decision this file has not earned.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler

    pca, scaler = PCA(n_components=3), MinMaxScaler(clip=True)
    pca_features = scaler.fit_transform(pca.fit_transform(features))
    fg = (pca_features[:, 0] > background_threshold if larger_pca_as_fg
          else pca_features[:, 0] < background_threshold)

    rgb = np.zeros((len(features), 3), dtype=np.float32)
    if fg.any():
        # scaler.fit then .transform, not the fit left over from the first
        # PCA. The second PCA runs on the foreground alone and its output has
        # a different range, so the stale fit would clip or squash the colours.
        fg_features = pca.fit_transform(features[fg])
        scaler.fit(fg_features)
        rgb[fg] = scaler.transform(fg_features)
    return rgb, fg


def patch_features(enc, tile: np.ndarray, dev) -> np.ndarray:
    """One HxWx3 uint8 tile -> [n_patches, D] float32, prefix tokens gone.

    forward_intermediates rather than a hand-rolled reshape: it strips
    num_prefix_tokens itself -- nine for UNI2, and dropping the wrong number
    would still divide -- derives the grid from the INPUT rather than from
    patch_embed.grid_size, and applies the final norm when asked. norm=False
    because that is what the demo takes -- it passes no norm at all, so its PCA
    reads the block output before the last LayerNorm. That norm subtracts a
    per-token mean and rescales, which is a change to the covariance structure
    PCA is reading, so it is not a free choice.
    """
    x = torch.from_numpy(tile).permute(2, 0, 1)[None]        # uint8, host
    return encode_batch(enc, x, dev)[0].cpu().numpy()


def encode_batch(enc, x_u8: torch.Tensor, dev, fp16: bool = True):
    """[B, 3, H, W] uint8 on the host -> [B, cells, D] float32 ON THE DEVICE.

    uint8 across the loader queue and the PCIe bus, float on the far side: a
    224 tile is 150 KB as bytes and 600 KB as float32, and at batch 1024 that
    is 154 MB against 616 MB per batch of pure transfer.

    fp16 autocast because the same model reaches 650 tiles/s in
    bench_slidewin_pooling under it and this file was doing about a fortieth of
    that in fp32 at batch 1. It moves the features by ~1e-3, which principal
    directions are insensitive to -- insensitive, not identical, so --no-fp16
    is there for a run that has to match exactly.

    Left on the device on purpose. What follows is a projection to sixteen
    numbers, so bringing 1536 of them to the host first would be the slowest
    step in the file.
    """
    x = x_u8.to(dev, non_blocking=True).float().div_(255.)
    x = (x - enc.pixel_mean) / enc.pixel_std
    m = enc.backbone
    ctx = (torch.autocast(device_type=dev.type, dtype=torch.float16)
           if fp16 and dev.type == 'cuda' else nullcontext())
    with torch.no_grad(), ctx:
        f = m.forward_intermediates(x, indices=1, norm=False, output_fmt='NCHW',
                                    intermediates_only=True)[-1]
    b, d = f.shape[0], f.shape[1]
    return f.permute(0, 2, 3, 1).reshape(b, -1, d).float()


class GpuProjection:
    """sklearn's PCA and MinMaxScaler as two device matmuls.

    transform is (x - mean_) @ components_.T and MinMax is (z - data_min_) /
    data_range_ clipped -- both are arithmetic, and doing them where the
    features already are keeps 1536 floats per cell off the bus. Per batch of
    1024 tiles that is 8 MB crossing instead of 1.6 GB.

    Fitted by sklearn on the sample, applied here. The fit is the part that
    needs an algorithm; this is the part that needs to be fast.
    """

    def __init__(self, pca, scaler, dev):
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)
        self.mean = t(pca.mean_)
        self.comp = t(pca.components_).T          # [D, k]
        self.lo = t(scaler.data_min_)
        self.rng = t(scaler.data_range_)

    def __call__(self, f: torch.Tensor) -> torch.Tensor:
        z = (f - self.mean) @ self.comp
        return ((z - self.lo) / self.rng).clamp_(0., 1.)


# ══ a whole slide, tile by tile ═══════════════════════════════════════════════

class TileSet(torch.utils.data.Dataset):
    """Tiles at given grid positions, read on a worker.

    The slide is opened LAZILY, inside __getitem__, and never in the parent.
    DataLoader workers are forked, and an OpenSlide handle carried across a
    fork is one handle used by several processes -- so each worker reaches this
    with _wsi still None and opens its own. Opening it in __init__ would look
    identical and corrupt reads under load.

    Returns uint8 and the per-cell mean colour. The mean is computed here
    because the worker already holds the pixels and the parent would otherwise
    keep the whole tile alive just to shrink it; it is what the figure draws,
    so the slide and the mask are the same array shape and cannot slip.
    """

    def __init__(self, path, origin, tile, positions, cells):
        self.path, self.origin, self.tile = path, origin, tile
        self.positions, self.cells = positions, cells
        self._wsi = None

    def _slide(self):
        if self._wsi is None:
            from SafeSlide import SafeSlide
            self._wsi = SafeSlide(self.path, warn=False)
        return self._wsi

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, k):
        j, i = self.positions[k]
        ox, oy = self.origin
        rgb = self._slide().read_region_rgb(
            (ox + i * self.tile, oy + j * self.tile), 0, (self.tile, self.tile))
        gh, gw = self.cells
        ph, pw = self.tile // gh, self.tile // gw
        small = (rgb.reshape(gh, ph, gw, pw, 3).mean(axis=(1, 3))
                 .astype(np.uint8))
        return (torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1),
                torch.from_numpy(small), k)


def tile_loader(path, origin, tile, positions, cells, batch, workers):
    return torch.utils.data.DataLoader(
        TileSet(path, origin, tile, positions, cells),
        batch_size=batch, shuffle=False, num_workers=workers,
        pin_memory=True, drop_last=False,
        # Order matters: results are placed by the index the worker returns, so
        # shuffle stays off and the index rides along rather than being assumed.
        prefetch_factor=2 if workers else None,
        persistent_workers=False)



def slide_bounds(wsi, limit_bounds: bool = True):
    """(origin, span) in LEVEL-0 pixels: the rectangle the scanner covered.

    SlideWinSift:248-255 does the same, and says why: openslide.bounds-* is the
    photographed rectangle, and on a MIRAX the surrounding canvas is stage
    travel range holding no image data at all. Tiles over it read nothing --
    and a ViT handed a blank tile has only its positional embedding varying
    between patches, so a PCA of one finds POSITION, which comes out as bands.
    """
    p = wsi.properties
    w0, h0 = wsi.level_dimensions[0]
    if not limit_bounds:
        return (0, 0), (w0, h0)
    return ((int(p.get('openslide.bounds-x', 0)),
             int(p.get('openslide.bounds-y', 0))),
            (int(p.get('openslide.bounds-width', w0)),
             int(p.get('openslide.bounds-height', h0))))


def tile_saturation(wsi, origin, span, tile: int, ds: float = 32.0):
    """Mean saturation per tile position, read once from a thumbnail.

    The cheap content signal, and it costs one read_region rather than one per
    tile. Saturation and not luminance because glass and a pale section differ
    far more in colour than in brightness -- a fatty area is nearly white and is
    still tissue, and TissueSegFunc's own hsv method thresholds the same thing.

    Only a proxy, and only used to CHOOSE what the model looks at. Nothing it
    decides reaches the mask.
    """
    level = wsi.get_best_level_for_downsample(ds)
    got = wsi.level_downsamples[level]
    lw = max(1, int(span[0] / got))
    lh = max(1, int(span[1] / got))
    small = wsi.read_region_rgb(origin, level, (lw, lh)).astype(np.float32)

    mx, mn = small.max(axis=2), small.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)

    nx, ny = span[0] // tile, span[1] // tile
    step = tile / got                       # tile side in thumbnail pixels
    out = np.zeros((ny, nx), dtype=np.float32)
    for j in range(ny):
        y0, y1 = int(j * step), max(int(j * step) + 1, int((j + 1) * step))
        for i in range(nx):
            x0, x1 = int(i * step), max(int(i * step) + 1, int((i + 1) * step))
            out[j, i] = sat[y0:min(y1, lh), x0:min(x1, lw)].mean()
    return out


def stratified_positions(sat: np.ndarray, n: int, bins: int = 10, seed: int = 0):
    """`n` tile positions, spread over saturation AND over the slide.

    Spatial sampling alone gives a sample proportional to AREA, which is not
    the same thing as a sample of both classes. Measured on these seven slides,
    tissue covers 38.2, 24.4 and 23.1 percent of the three BRACS and 9.2, 5.9,
    4.6 and 3.5 percent of the four Ki67 -- so a uniform 1024 over the last one
    is about 36 tiles of tissue against 988 of glass, and the largest variance
    in that sample is the variance AMONG BLANK TILES. A blank tile's patch
    tokens differ only by their positional embedding, so a PCA of them finds
    position, which is what banded the earlier figure.

    Quantile bins rather than a high half and a low half: the interesting cells
    are the ones in between -- a section edge, fat, a fold -- and they are
    exactly what a two-way split throws away.

    Spread inside each bin, so the sample is not clustered on one lump of
    tissue. Jittered rather than centred, because scan grids and TMA layouts
    are periodic and a fixed offset can align with them.
    """
    rng = np.random.default_rng(seed)
    flat = sat.ravel()
    order = np.argsort(flat, kind='stable')
    per = max(1, n // bins)

    picked = []
    for b in range(bins):
        lo, hi = b * len(order) // bins, (b + 1) * len(order) // bins
        idx = order[lo:hi]
        if len(idx) == 0:
            continue
        # Spread inside the bin by walking it in stable saturation order and
        # taking evenly spaced entries, then jittering within each step.
        step = max(1, len(idx) // per)
        take = [idx[min(len(idx) - 1, k * step + rng.integers(0, step))]
                for k in range(min(per, len(idx)))]
        picked.extend(take)

    picked = np.unique(np.array(picked, dtype=np.int64))
    return [(int(k // sat.shape[1]), int(k % sat.shape[1])) for k in picked]


def slide_pca_mask(enc, path: str, args, dev):
    """Fit one PCA for the slide, then project every tile of the bounds rect.

    Three stages, and the first two are cheap:

      thumbnail   one read at ds 32, mean saturation per tile position
      fit         `--fit-tiles` positions drawn across saturation quantiles,
                  encoded, PCA(--components) fitted on all their cells at once
      project     every tile encoded and transformed, never fitted

    One basis for the whole slide, so a colour means the same thing in every
    tile and the components can be compared, clustered or thresholded later.
    Per tile it could not be: MinMaxScaler maps that tile's own extremes to 0
    and 1, so a tile holding only tissue has its threshold land inside tissue.

    The components are kept, not just the threshold. One bit per cell throws
    away everything the second and third components hold, and re-deriving them
    costs the whole encode again -- hours, against a few hundred MB.

    Read tile by tile rather than as one array: the bounds rectangle at level 0
    is over 3 GB of uint8 on a MIRAX before anything is copied. SafeSlide,
    because two of these slides have cells the scanner never wrote.

    The mask lands at ds = patch_size: a 224 tile is 16x16 cells of 14 level-0
    pixels each, so nothing here chooses a resolution -- the patch grid does.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler
    from SafeSlide import SafeSlide

    wsi = SafeSlide(path)
    (ox, oy), (sw, sh) = slide_bounds(wsi, args.limit_bounds)
    ph, pw = enc.backbone.patch_embed.patch_size
    gh, gw = args.tile // ph, args.tile // pw
    nx, ny = sw // args.tile, sh // args.tile
    cw, ch = wsi.level_dimensions[0]

    print(f'        level-0 canvas {cw}x{ch}, bounds {sw}x{sh} at ({ox}, {oy}) '
          f'-- {100.0 * sw * sh / (cw * ch):.1f}% of it')
    print(f'        {nx} x {ny} = {nx * ny} tiles of {args.tile}, '
          f'mask {nx * gw} x {ny * gh} at ds={ph}')

    # ── 1. thumbnail ────────────────────────────────────────────────────────
    sat = tile_saturation(wsi, (ox, oy), (sw, sh), args.tile)
    picks = stratified_positions(sat, args.fit_tiles, args.fit_bins)
    s = sat.ravel()
    print(f'        saturation {s.min():.3f} .. {s.max():.3f}, '
          f'median {np.median(s):.3f}; fitting on {len(picks)} tiles across '
          f'{args.fit_bins} quantiles')
    wsi.close()          # the workers open their own; see TileSet._slide

    def run(positions, what):
        loader = tile_loader(path, (ox, oy), args.tile, positions, (gh, gw),
                             args.batch_tiles, args.workers)
        done, t0 = 0, time.time()
        for x_u8, small, idx in loader:
            yield encode_batch(enc, x_u8, dev, args.fp16), small, idx
            done += len(idx)
            rate = done / max(1e-9, time.time() - t0)
            print(f'        {what} {done}/{len(positions)}  {rate:.0f} tiles/s'
                  f'  eta {(len(positions) - done) / max(rate, 1e-9) / 60:.1f} min',
                  end='\r', flush=True)
        print()

    # ── 2. fit ──────────────────────────────────────────────────────────────
    sample = np.concatenate([f.reshape(-1, f.shape[-1]).cpu().numpy()
                             for f, _, _ in run(picks, 'fit  ')], axis=0)
    pca = PCA(n_components=args.components).fit(sample)
    scaler = MinMaxScaler(clip=True).fit(pca.transform(sample))
    project = GpuProjection(pca, scaler, dev)
    print(f'        PCA({args.components}) on {len(sample)} cells; '
          f'explained variance {pca.explained_variance_ratio_[:3].sum():.1%} '
          f'in the first three')
    del sample

    # ── 3. project ──────────────────────────────────────────────────────────
    grid = [(j, i) for j in range(ny) for i in range(nx)]
    comps = np.zeros((ny * gh, nx * gw, args.components), dtype=np.float16)
    thumb = np.zeros((ny * gh, nx * gw, 3), dtype=np.uint8)
    for f, small, idx in run(grid, 'tiles'):
        c = project(f).reshape(len(idx), gh, gw, args.components)
        c = c.to(torch.float16).cpu().numpy()
        small = small.numpy()
        for b, k in enumerate(idx.tolist()):
            j, i = grid[k]
            ys, xs = slice(j * gh, (j + 1) * gh), slice(i * gw, (i + 1) * gw)
            comps[ys, xs] = c[b]
            thumb[ys, xs] = small[b]

    pc1 = comps[..., 0].astype(np.float32)
    mask = ((pc1 > args.background_threshold) if args.larger_pca_as_fg
            else (pc1 < args.background_threshold)).astype(np.uint8)
    return thumb, mask, comps


def draw(thumb, mask, comps, out: Path, title: str):
    """Slide, the first three components as RGB, the mask, the overlay.

    The RGB panel is the whole point of a shared basis: a colour means the same
    thing everywhere on the slide, so regions can be read off it. Per tile it
    would not -- each tile would have its own axes and the same colour would
    name different things a tile apart.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    over = thumb.copy()
    sel = mask.astype(bool)
    over[sel] = (over[sel] * 0.55 + np.array([0, 200, 80]) * 0.45).astype(np.uint8)
    rgb3 = np.clip(comps[..., :3].astype(np.float32), 0, 1)

    fig, ax = plt.subplots(1, 4, figsize=(22, 6))
    for a, img, t in ((ax[0], thumb, 'slide'),
                      (ax[1], rgb3, 'PC1-3 as RGB, one basis for the slide'),
                      (ax[2], mask * 255, f'PC1 threshold  {mask.mean():.1%}'),
                      (ax[3], over, 'over the slide')):
        a.imshow(img, cmap=None if img.ndim == 3 else 'gray')
        a.set_title(t)
        a.axis('off')
    fig.suptitle(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'        wrote {out}')


def draw_tiles(tiles, rgb, fg, gh, gw, out: Path, title: str,
               alpha: float = 0.3):
    """The notebook figure: one row per image, four panels each.

    ONE PCA over all the tiles, split back for drawing -- which is what the
    notebook does and why it loads two images rather than one. The reason is
    the figure itself: principal directions come out of the data, so fitting
    each tile separately would make the same colour a different direction in
    each row, and two rows side by side would invite a comparison the axes do
    not support. Fitted together, a colour means one thing across the figure.

    (Their `image_paths` has two entries and a later cell still says
    `for i in range(4)`, so the count is whatever was loaded that day. What
    matters is that they share a fit.)

    The overlay is create_overlay_image from the notebook, verbatim:

        result_resized = cv2.resize(result, (original.shape[1], original.shape[0]))
        overlay = (alpha * original + (1 - alpha) * result_resized * 255)

    cv2.resize defaults to bilinear, which is why a 16x16 grid comes back as a
    smooth wash rather than blocks -- the softness in their published figure is
    that interpolation and not anything the model produced. The PCA panel keeps
    nearest, so the two together show both the cells and the blend.

    The mask panel is not the notebook's. It is what the slide loop stitches,
    drawn beside the colours it comes from.
    """
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_cells = gh * gw
    fig, ax = plt.subplots(len(tiles), 4, figsize=(18, 5 * len(tiles)),
                           squeeze=False)
    for k, tile in enumerate(tiles):
        res = rgb[k * n_cells:(k + 1) * n_cells].reshape(gh, gw, 3)
        m = fg[k * n_cells:(k + 1) * n_cells]
        res_up = cv2.resize(res, (tile.shape[1], tile.shape[0]))
        overlay = (alpha * tile + (1 - alpha) * res_up * 255).astype(np.uint8)

        for a, img, t in ((ax[k][0], tile, 'original'),
                          (ax[k][1], res, 'foreground-only PCA'),
                          (ax[k][2], overlay, f'overlay  alpha={alpha}'),
                          (ax[k][3], m.reshape(gh, gw) * 255,
                           f'mask  {m.mean():.1%}')):
            a.imshow(img, cmap=None if img.ndim == 3 else 'gray',
                     interpolation='nearest')
            a.set_title(t)
            a.axis('off')
    fig.suptitle(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'        wrote {out}')


# ══ checks ════════════════════════════════════════════════════════════════════

_RESULTS = []


def check(name, fn):
    try:
        out = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   {out}' if out else ''))
    except Exception as e:                                       # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n          {type(e).__name__}: {e}')


def t_recipe_changes_the_architecture(args):
    """Without the twelve kwargs, timm builds a different tower and says nothing.

    Both sides are pretrained=False, so this costs no download and no load: the
    shape is decided before any weight is read. If the two ever come out equal,
    either the hub config grew the missing keys or _TIMM_KWARGS lost them, and
    both are worth being told about.
    """
    good = build_encoder(args.encoder, args.img, pretrained=False, recipe=True)
    bare = build_encoder(args.encoder, args.img, pretrained=False, recipe=False)

    got = (good.backbone.embed_dim, len(good.backbone.blocks),
           good.backbone.num_prefix_tokens)
    bad = (bare.backbone.embed_dim, len(bare.backbone.blocks),
           bare.backbone.num_prefix_tokens)

    assert got != bad, (
        'the hub config now carries the recipe, so vit.py would build the '
        'right model and this check is moot')
    return f'recipe {got}  vs  hub config alone {bad}   (dim, depth, prefix)'


def t_backbone_meets_the_contract(enc):
    """Every attribute eomt.py reaches for, with the line that reaches for it.

    This list IS the contract: models/vit.py's transformers_to_timm exists to
    fake these names for a HuggingFace model, so what it renames is what EoMT
    needs. A timm ViT has all of them, which is why UNI2 needs no adapter --
    only the kwargs vit.py has no way to pass.
    """
    m = enc.backbone
    need = [('embed_dim', 'eomt:35,37,40-44,52'),
            ('num_prefix_tokens', 'eomt:60,79,138'),
            ('blocks', 'eomt:165'),
            ('norm', 'eomt:175,197'),
            ('patch_embed', 'eomt:47,62,131,157')]
    missing = [f'{a} ({w})' for a, w in need if not hasattr(m, a)]
    assert not missing, 'backbone is missing ' + ', '.join(missing)

    blk = m.blocks[0]
    for a in ('norm1', 'norm2', 'mlp'):
        assert hasattr(blk, a), f'block has no {a} (eomt:185,191)'
    assert hasattr(blk, 'attn') or hasattr(blk, 'attention'), \
        'block has no attn/attention (eomt:181-184)'
    assert hasattr(blk, 'ls1') and hasattr(blk, 'ls2'), \
        'block has no ls1/ls2 (eomt:186-195)'
    for a in ('qkv', 'num_heads', 'head_dim', 'q_norm', 'k_norm', 'scale',
              'fused_attn', 'attn_drop', 'proj', 'proj_drop'):
        assert hasattr(blk.attn, a), f'attn has no {a} (eomt:96-117)'

    # Optional, and both absent here. rope: UNI2 interpolates a learned
    # absolute pos_embed instead, so eomt:154 leaves rope=None and _attn takes
    # the plain path. _pos_embed: present, and it runs at :160 BEFORE the
    # queries are concatenated at :167 -- which is why the queries never
    # receive a positional embedding, and why UNI2's no_embed_class=True is
    # handled entirely inside timm.
    return (f'{len(m.blocks)} blocks, {blk.attn.num_heads} heads, '
            f'rope={hasattr(m, "rope_embeddings")}, '
            f'_pos_embed={hasattr(m, "_pos_embed")}')


def t_grid_is_what_the_input_implies(enc, args):
    """patch_embed.grid_size must be img // patch, because eomt.py reads it.

    _predict:62 reshapes the token axis with grid_size and _attn_mask:131
    interpolates to it. Both are right only while the model was CONSTRUCTED at
    the size it is fed -- which is what passing img_size does, and what breaks
    for a caller relying on dynamic_img_size instead.
    """
    m = enc.backbone
    ph, pw = m.patch_embed.patch_size
    assert args.img % ph == 0, (
        f'img {args.img} is not a multiple of patch {ph}. 512 does not divide '
        f'by 14; use 504 = 36x14 or 518 = 37x14.')
    want = (args.img // ph, args.img // pw)
    got = tuple(m.patch_embed.grid_size)
    assert got == want, f'grid_size {got}, expected {want}'
    return f'{got[0]}x{got[1]} = {got[0] * got[1]} patches, patch {ph}'


def t_forward_gives_the_expected_shapes(net, enc, args, dev):
    """One forward, and the five numbers that have to line up across two repos.

    num_upscale = max(1, int(log2(patch)) - 2), so patch 14 gets ONE ScaleBlock
    and patch 16 gets two: the mask comes back at 2x the grid here and 4x on a
    patch-16 backbone. That is why EoMT re-patchifies DINOv2 from 14 to 16.
    """
    m = enc.backbone
    x = torch.rand(args.batch, 3, args.img, args.img, device=dev)

    if dev.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        mask_per_layer, class_per_layer = net(x)

    gh, gw = m.patch_embed.grid_size
    up = max(1, int(math.log2(max(m.patch_embed.patch_size))) - 2)
    mh, mw = gh * 2 ** up, gw * 2 ** up

    n_pred = args.num_blocks + 1
    assert len(mask_per_layer) == n_pred, (
        f'{len(mask_per_layer)} predictions, expected {n_pred} -- one per '
        f'masked-attention block (eomt:175) plus the final one (eomt:197)')

    msk, cls = mask_per_layer[-1], class_per_layer[-1]
    assert tuple(msk.shape) == (args.batch, args.num_q, mh, mw), (
        f'mask {tuple(msk.shape)}, expected '
        f'{(args.batch, args.num_q, mh, mw)}')
    assert tuple(cls.shape) == (args.batch, args.num_q, args.num_classes + 1), (
        f'class {tuple(cls.shape)}, expected '
        f'{(args.batch, args.num_q, args.num_classes + 1)}')

    n_tok = gh * gw + m.num_prefix_tokens + args.num_q
    peak = (torch.cuda.max_memory_allocated() / 2 ** 30
            if dev.type == 'cuda' else 0.0)
    return (f'{n_pred} predictions; mask {tuple(msk.shape)} = {gh}x{gw} x2^{up}; '
            f'class {tuple(cls.shape)}; {n_tok} tokens in attention; '
            f'peak {peak:.2f} GiB')


def t_segment_gives_a_class_map(net, args, dev):
    """The whole inference path, ending where TissueSegmenter's contract does.

    A [H, W] uint8 map at the input's resolution, 1 = tissue -- the same shape
    HestSegmenter returns. The values are noise because the head is random; the
    shape and the dtype are what is under test.
    """
    x = torch.rand(args.batch, 3, args.img, args.img, device=dev)
    out = segment(net, x, args.num_classes)
    assert tuple(out.shape) == (args.batch, args.img, args.img), tuple(out.shape)
    assert out.dtype == torch.uint8, out.dtype
    assert int(out.max()) < args.num_classes, out.max()
    frac = float((out == 1).float().mean())
    return (f'{tuple(out.shape)} {out.dtype}; class-1 fraction {frac:.3f} '
            f'(random head, so meaningless)')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--encoder', default='uni2',
                    help='which backbone config to take the recipe and the '
                         'normalisation from')
    ap.add_argument('--img', type=int, default=504,
                    help='square input. Must divide by the patch size: 504 = '
                         '36x14 and 518 = 37x14, while 512 does NOT divide 14.')
    ap.add_argument('--patch-size', type=int, default=None,
                    help='override the backbone patch size, resampling '
                         'patch_embed.proj. EoMT does this to DINOv2 (14 -> '
                         '16) for a rounder grid and two upscale blocks. '
                         'Untested for UNI2, so off by default.')
    ap.add_argument('--num-q', type=int, default=16,
                    help='their semantic config uses 100, for 150 ADE20K '
                         'classes. Two classes need far fewer, and num_q rides '
                         'on the attention sequence, which costs N^2.')
    ap.add_argument('--num-classes', type=int, default=2,
                    help='background and tissue')
    ap.add_argument('--num-blocks', type=int, default=4,
                    help="EoMT's default: the last four blocks see the queries")
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--image', default=None,
                    help='run the class map on a real image as well')
    ap.add_argument('--wsi', nargs='*', default=None,
                    help='slide paths. One PCA mask per slide: every tile of '
                         'the bounds rectangle at level 0, the demo run on '
                         'each, stitched.')
    ap.add_argument('--tile', type=int, default=224,
                    help="the demo's tile, and UNI2's own training size. The "
                         'mask lands at ds = patch_size, so this chooses how '
                         'many tiles, not how fine the mask is.')
    ap.add_argument('--background-threshold', type=float, default=0.5,
                    help="the demo's own cut, after MinMaxScaler")
    ap.add_argument('--larger-pca-as-fg', action='store_true',
                    help="the demo's own flag: which side of PC1 is foreground")
    ap.add_argument('--limit-bounds', action=argparse.BooleanOptionalAction,
                    default=True,
                    help='tile only openslide.bounds-*, the rectangle the '
                         'scanner covered. Off, a MIRAX is mostly stage travel '
                         'range with no image in it.')
    ap.add_argument('--tile-figure', nargs='*', default=None,
                    help='tile PNGs for the demo figure. Give TWO, as the '
                         'notebook does: they share one PCA fit, which is what '
                         'makes a colour mean the same thing in both rows.')
    ap.add_argument('--fit-tiles', type=int, default=1000,
                    help='tiles the slide PCA is fitted on, drawn across '
                         'saturation quantiles so both glass and tissue are '
                         'represented whatever the slide is made of')
    ap.add_argument('--fit-bins', type=int, default=10,
                    help='saturation quantiles to draw the fit sample from. '
                         'Ten rather than two, because the cells that decide '
                         'the boundary -- a section edge, fat, a fold -- are '
                         'the ones in between.')
    ap.add_argument('--batch-tiles', type=int, default=1024,
                    help='tiles per forward. bench_slidewin_pooling reaches '
                         '650 tiles/s on this model at 2048; at 1 it was doing '
                         'about a fortieth of that.')
    ap.add_argument('--workers', type=int, default=8,
                    help='loader processes reading tiles. Each opens its own '
                         'slide handle -- see TileSet._slide -- so reads '
                         'overlap the forward instead of taking turns with it.')
    ap.add_argument('--fp16', action=argparse.BooleanOptionalAction, default=True,
                    help='autocast the forward. Moves features by ~1e-3, which '
                         'principal directions are insensitive to; --no-fp16 '
                         'for a run that has to match fp32 exactly.')
    ap.add_argument('--components', type=int, default=16,
                    help='components kept per cell. Three would draw the '
                         'figure; the rest are what a later clustering or a '
                         'different cut would need, and re-deriving them costs '
                         'the whole encode again.')
    args = ap.parse_args()

    # Every figure and every .npz below is this backbone's: the PCA runs on its
    # patch features, so a second encoder writes a different picture under the
    # same name. No --head here, because build_encoder takes the recipe from
    # the config rather than a built model's exit.
    enc_tag = encoder_tag(args.encoder)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={dev}  encoder={args.encoder}  img={args.img}  '
          f'num_q={args.num_q}  num_blocks={args.num_blocks}  '
          f'batch={args.batch}\n')

    print('the hole vit.py has no way to fill')
    check('recipe changes the architecture',
          lambda: t_recipe_changes_the_architecture(args))

    print('\nbackbone')
    enc = build_encoder(args.encoder, args.img,
                        patch_size=args.patch_size).to(dev).eval()
    check("meets eomt.py's attribute contract",
          lambda: t_backbone_meets_the_contract(enc))
    check('grid is what the input implies',
          lambda: t_grid_is_what_the_input_implies(enc, args))

    print('\neomt')
    net = EoMT(encoder=enc, num_classes=args.num_classes, num_q=args.num_q,
               num_blocks=args.num_blocks).to(dev).eval()
    n_head = sum(p.numel() for n, p in net.named_parameters()
                 if not n.startswith('encoder.'))
    n_back = sum(p.numel() for n, p in net.named_parameters()
                 if n.startswith('encoder.'))
    print(f'        head {n_head:,} trainable if the backbone is frozen; '
          f'backbone {n_back:,}')
    check('forward gives the expected shapes',
          lambda: t_forward_gives_the_expected_shapes(net, enc, args, dev))
    check('segment gives a class map',
          lambda: t_segment_gives_a_class_map(net, args, dev))

    if args.image:
        print('\nreal image')
        x = load_image(args.image, args.img, dev)
        out = segment(net, x, args.num_classes)
        print(f'        {args.image}  ->  {tuple(out.shape)}  '
              f'class-1 fraction {float((out == 1).float().mean()):.3f}')

    if args.tile_figure:
        print('\ntile')
        from PIL import Image
        tiles = [np.array(Image.open(p).convert('RGB')
                          .resize((args.tile, args.tile), Image.BICUBIC))
                 for p in args.tile_figure]
        # One fit over all of them, the way the notebook stacks its images into
        # one batch and reshapes to (-1, 1536) before the PCA. Concatenated
        # here rather than batched through the model because two tiles is not
        # a throughput problem and the arithmetic is what has to match.
        feats = np.concatenate([patch_features(enc, t, dev) for t in tiles],
                               axis=0)
        rgb, fg = demo_pca(feats, args.background_threshold,
                           args.larger_pca_as_fg)
        ph, pw = enc.backbone.patch_embed.patch_size
        stem = '__'.join(Path(p).stem for p in args.tile_figure)
        out = Path(job_result_dir('EoMT', encoder=enc_tag)) / f'tile__{stem}.png'
        draw_tiles(tiles, rgb, fg, args.tile // ph, args.tile // pw, out,
                   f'{len(tiles)} tiles, one PCA over both   {args.encoder}   '
                   f'{args.tile}px -> {args.tile // ph}x{args.tile // pw}')

    for path in (args.wsi or []):
        stem = Path(path).stem
        print(f'\nslide  {stem}')
        thumb, mask, comps = slide_pca_mask(enc, path, args, dev)
        ph, _ = enc.backbone.patch_embed.patch_size
        root = Path(job_result_dir('EoMT', encoder=enc_tag))
        draw(thumb, mask, comps, root / f'{stem}__pca_ds{ph}.png',
             f'{stem}   level 0, tile {args.tile}   {args.encoder}   '
             f'mask ds={ph}')
        # The components, not just the threshold. Re-deriving them costs the
        # whole encode again; keeping them costs a few hundred MB and leaves
        # every later question -- another cut, a clustering, a weak label --
        # answerable without the GPU.
        npz = root / f'{stem}__pca_ds{ph}.npz'
        np.savez_compressed(npz, components=comps, thumb=thumb)
        print(f'        wrote {npz}  ({npz.stat().st_size / 2**20:.0f} MB, '
              f'{comps.shape[2]} components)')

    bad = [n for n, e in _RESULTS if e is not None]
    print(f'\n{len(_RESULTS) - len(bad)}/{len(_RESULTS)} passed')
    if bad:
        print('failed: ' + ', '.join(bad))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
