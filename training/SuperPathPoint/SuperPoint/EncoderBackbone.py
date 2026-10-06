"""The foundation-model trunk as a `common.Interfaces.Backbone`. spec.md 5.3.

    cfg = TileEncoderBackboneConfig(encoder='conch_vit', tile_size=256)
    backbone = cfg.build(device)          # [N, 3, 256, 256] -> [N, 768, 16, 16]

WHY THIS IS NOT IN `Backbones.py`
----------------------------------
It was planned there (spec.md 14) and it does not belong there, for a reason
that has nothing to do with tidiness. `TileEncoderFunc._IMPLEMENTATIONS` imports
one encoder module at a time BECAUSE each of them does
`os.environ.setdefault('HF_HOME', ...)` above its own `import timm`, and
huggingface_hub freezes HF_HOME into module constants when it is imported --
first one wins. A top-level import of `aiNNModel` inside `Backbones.py` would
put that side effect on the import path of `KeypointNet.py`, which is imported
by the VGG student, by the loss tests and by every CPU-only check in
`test_superpathpoint.py`. None of them want timm, a weights directory, or the
five seconds.

So this file imports `TileEncoderFunc` at the top -- which is only the base
class and the registry, no HF -- and calls `encoder_config(name)` inside
`build()`, where the one implementation module is imported and nowhere else.

WHY IT HOLDS AN ENCODER RATHER THAN CALLING `spatial()`
--------------------------------------------------------
`TileEncoder.spatial()` is an INFERENCE api. It goes
through `_run`, which converts each image to PIL, applies the config's transform
-- `Resize(256)` then `CenterCrop(224)` -- runs under `torch.no_grad()`, and
brings the result back to the host. A training loop that feeds 256 px tiles and
wants the map on the device can use none of that: the resize alone would put
the feature grid on a different pixel lattice than the labels were splatted onto
and nothing would raise.

What is wanted is `TileEncoder.features_with_grad(batch)` -- batch in, map
out, no transform, no host round trip, gradients intact. It is a public method
on the base, so this file builds the encoder the ordinary way and calls it;
nothing here re-parents, wraps or subclasses an encoder.

WHAT IS LEFT HERE, THEN
------------------------
The Protocol adapter, and only that. `common.Interfaces.Backbone` is
SuperPoint's interface, not a general one: `stride` is paired with `cell` (see
that file -- `cell` is the detector head's prediction block, and
`UpsampleDecoder`'s rung count is `log2(stride / cell)`), and the forward
contract is `[0, 1]` images rather than a normalised batch. Those are facts
about what a keypoint detector needs, so they are stated here rather than
pushed down into `aiNNModel/`, which has no reason to know what a detector
cell is.

WHICH ENCODERS CAN DO THIS, AND AT WHAT TILE SIZE
---------------------------------------------------
ONE fact about the model decides it, and it is read off the model rather than
assumed here: `patch_embed.patch_size`. The tile has to be a multiple of it.

    gigapath    16   224 native, FIXED    tile 256 works
    conch_vit   16   448 native, dynamic  tile 256 works
    uni2        14   224 native, dynamic  no tile size works -- see below

`gigapath` is the trap and it is worth writing down. Its arch is called
`vit_giant_patch14_dinov2`, and prov-gigapath's own `config.json` then overrides
`model_args.patch_size` to **16** -- so the name says 14, the model is 16, and
its 14x14 grid at 224 is 224/16 rather than a patch of 14. Reading the patch off
the arch NAME gets this backwards for the only encoder whose name mentions a
patch size. `_patch_stride` below reads `patch_embed.patch_size`, which is why
the code was right about it while the prose was not.

THE FIXED INPUT SIZE IS NOT A REFUSAL, IT IS ONE CALL
-------------------------------------------------------
`GigaPathFunc` builds with no `dynamic_img_size`, and prov-gigapath's
`pretrained_cfg` says `fixed_input_size: true`: `PatchEmbed.forward` then asserts
the input is exactly 224 (`timm/layers/patch_embed.py:120`). That is a property
of how the model was CONSTRUCTED, not of the weights, and timm has the public
call that changes it -- `VisionTransformer.set_input_size(img_size=...)`
(`timm/models/vision_transformer.py:1013`), which updates `patch_embed.img_size`
and `grid_size` and resamples `pos_embed` onto the new grid, once.

So `__init__` tells every trunk its input size at construction. For a
`dynamic_img_size` model this is the same arithmetic that would otherwise run on
every forward -- `_pos_embed` calls the same `resample_abs_pos_embed` -- done
once instead of per batch, so the two paths agree by construction rather than by
luck. For a fixed one it is the difference between working and asserting.

It MUTATES the loaded model. That is recorded, because `tile_size` is a hashed
field of the config above and the resampled `pos_embed` is part of what
`weights_id` hashes -- so a trunk read at 256 and the same encoder read at 224 in
the retrieval pipeline correctly get different ids. It also means a trunk handed
in from outside is mutated in place, which is fine for `build()` (it makes its
own) and is the reason the tests build their own too.

`check_shapes` at the real tile size is what makes all of this evidence instead
of a claim: it is the forward that proves the resample landed.

THE TILE IS WHAT THE TRUNK IS FED. NOTHING IS RESIZED OR CROPPED
------------------------------------------------------------------
`TransformConfig.build()` is `Resize -> CenterCrop -> ToTensor -> Normalize`,
and the training path takes the last two and neither of the first two. Both
omissions are deliberate and they are not the same omission.

The CROP is the one that would be silent. For retrieval it is free -- one vector
comes out, and gigapath's `crop_pct` of 0.875 discarding the outer ring of a
tile costs a little context. For a DENSE task the discarded ring has labels in
it and no prediction to compare them against: the loss simply never sees them,
falls normally, and the model is trained on the middle 76 per cent of every
tile. (uni2 and conch_vit have `crop_size == scale_size`, so their crop is
already a no-op; only gigapath's removes anything.)

The RESIZE is unnecessary once the crop is gone, because the tile size is not
handed down from anywhere -- the tile is a centre crop of a 768 px pre-tile
(`utilities/Store.py`), so asking for a different one costs nothing. Feed
the trunk the tile, at its own pixel grid, and `stride` is the patch exactly.

PATCH 14 CANNOT REACH CELL 8, AND NO TILE SIZE CHANGES THAT
-------------------------------------------------------------
The decoder has to climb from the feature grid to the cell grid, and that ratio
is

    (tile / cell) / (tile / patch) = patch / cell

with the tile cancelling out. So it is 16/8 = 2 for gigapath and conch_vit --
which is what `UpsampleDecoder`'s stack of stride-2 transposed convolutions
expresses -- and 14/8 = 1.75 for uni2, at every tile size there is. The tile
must also be divisible by BOTH, so uni2's legal tiles are the multiples of
lcm(14, 8) = 56: 112, 168, 224, 280, ... and every one of them is 1.75.

There are exactly three ways out and none of them is a tile size:

    resize the image      256 -> 224 makes the feature grid 16 against a cell
                          grid of 32, i.e. ratio 2. Costs 12.5 per cent of the
                          linear resolution before the trunk sees anything
    resize the features    interpolate 16x16 -> 28x28 inside the decoder, which
                          is the 'or this decoder has to grow a resize' that
                          `UpsampleDecoderConfig` already names. No image
                          resolution lost; an interpolation in feature space
                          instead
    give uni2 its own cell no resize anywhere. `stride / cell` has to be a power
                          of two, and 14 has exactly two divisors that give one:
                          14 (ratio 1) and 7 (ratio 2)

THE THIRD IS CHOSEN, AT cell = 7. Not 14. Both satisfy the decoder, and 7 is
better on every axis that matters:

    cell 14   ratio 1, no upsampling at all. One prediction per 196 px, which
              is 3.06x COARSER than cell 8 -- a lower ceiling on keypoint
              density inside a comparison about keypoints
    cell  7   ratio 2, one `ConvTranspose2d(stride=2)` rung -- the SAME
              machinery gigapath uses at 16/8. One prediction per 49 px, which
              is 1.31x FINER than cell 8. A 50-way softmax against 65

And the shapes come out identical to gigapath's, which is the real argument:

    gigapath  tile 256  stride 16 -> 16x16 features -> 32x32 cells  65 channels
    uni2      tile 224  stride 14 -> 16x16 features -> 32x32 cells  50 channels

Same tensor shapes end to end, same number of predictions, same border fraction
in cells. Only the pixel PITCH differs -- 14 against 16 per feature, 7 against 8
per cell -- which is the difference between the two trunks and is what the
comparison is supposed to be about. 224 is also uni2's native input size, so its
position embedding is not resampled at all.

It needs no new code: `KeypointNetConfig.wired(cell=7)` builds it and
`UpsampleDecoderConfig(cell=7, stride=14)` passes its own power-of-two check.

The label store is the one real cost, and it is avoidable. HA labels are cut per
tile size, so tile 224 and tile 256 are two stores and `make_ha_labels` is the
step measured in GPU-hours. They do not have to be two RUNS: a 224 tile is the
centre crop of a 256 one, so the points transform by subtracting 16 from each
coordinate, dropping what falls outside, and re-cutting `border=4` on the new
frame. Exact for the interior; the outermost 4 px ring is the only part where a
native 224 run would have differed.

WHAT IS FROZEN AND WHY THE CHECKPOINT DOES NOT HOLD IT
-------------------------------------------------------
`trainable = False` (spec.md 5.3): GigaPath's trunk is 1.1B parameters and
unfreezing it is not one card's training run. The trunk is therefore NOT an
`nn.Module` child of this backbone -- it is held in a list so `parameters()`,
`state_dict()` and `to()` do not reach it. A checkpoint that carried 1.1B frozen
floats would be several GB per epoch of exactly the bytes the encoder can
reload from its own weights, and `encoder_id` below is what says which ones.

The gray/RGB axis (spec.md 13) is NOT `in_channels` here. A foundation trunk was
pretrained on three channels and takes three; the grayscale arm is
`TransformConfig.preprocess='grey'`, which is luminance replicated to three
channels, already a hashed field, and already what the encoder does at
inference. Setting `in_channels=1` would be a different patch embedding, i.e. a
different model.

STILL OPEN: WIRING IT INTO `KeypointNet`
------------------------------------------
`KeypointNetConfig.backbone` is annotated `VggBackboneConfig` and `wired()`
constructs one. Both would have to widen, and `wired()` cannot compute
`out_channels` without loading the weights -- `VggBackboneConfig` gets it from
`channels[-2]` and there is no such list here. That is step 8 of spec.md 12 and
is deliberately not done: it is a decision about whether `KeypointNet` accepts a
prebuilt backbone or this file carries a width table, and writing it now would
answer it by accident. Everything in this file is testable without it.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
import torch
import torch.nn as nn

from ConfigIdentity import IdentifiedConfig, register
from TileEncoderFunc import TileEncoder, encoder_config, encoder_names

from common.Interfaces import check_shapes


#: The zero point (ConfigIdentity rule 2).
_TILE_ENCODER_BASELINE = {
    'method': 'tile_encoder',
    'encoder': 'conch_vit',
    'head': '',
    'dtype': 'fp16',
    'preprocess': 'none',
    'tile_size': 256,
}


@register('tile_encoder')
@dataclass(frozen=True)
class TileEncoderBackboneConfig(IdentifiedConfig):
    """Which encoder, read at what size. These fields DETERMINE the encoder.

    `encoder` names a registered implementation and everything else about it --
    architecture, weights, crop, mean and std -- comes from that implementation's
    own baseline. Only the three things this file varies are fields, so the
    hash of this config plus the encoder name is a complete description of the
    trunk. `TileEncoderBackbone.encoder_id` reports the encoder's own
    `identity_id()` alongside, which is the same shape as
    `HaConfig.identity_parts` appending the teacher's.
    """
    method: str = 'tile_encoder'

    #: `TileEncoderFunc.encoder_names()`. Checked at build, not here: naming
    #: them in a validator would duplicate a registry that already exists.
    encoder: str = 'conch_vit'

    #: Which of the model's exits. '' is the model's own; CONCH's tower and its
    #: attentional pooler are 768-d and 512-d vectors in different spaces, which
    #: is why the field is on `TileEncoderConfig` at all. For a feature MAP the
    #: pooler exit does not exist, so this is '' for every encoder today -- kept
    #: because it is part of the encoder's identity and of `encoder_tag`.
    head: str = ''

    #: The trunk is frozen, so fp16 costs nothing that is being trained. It is
    #: `TileEncoderFunc`'s own default for all three encoders and went to
    #: production at cos=0.99995 against fp32 (log/TODO.log).
    dtype: str = 'fp16'

    #: 'none' keeps RGB. 'grey' is luminance replicated to three channels --
    #: the grayscale arm of spec.md 13 for a trunk that takes three channels.
    preprocess: str = 'none'

    #: The size the student is trained at. A field because the refusals below
    #: depend on it: it decides whether the patch grid divides the tile at all.
    tile_size: int = 256

    #: A foundation trunk takes three channels. Not a field -- see the module
    #: docstring. Present because `Backbone`'s callers read it off the config
    #: the way they read `VggBackboneConfig.in_channels`.
    in_channels = 3
    BASELINE = _TILE_ENCODER_BASELINE

    def build(self, device=None) -> 'TileEncoderBackbone':
        if self.encoder not in encoder_names():
            raise ValueError(
                f'no encoder called {self.encoder!r}. Known: '
                f'{", ".join(encoder_names())}')
        enc_cfg = encoder_config(self.encoder, head=self.head)
        enc_cfg = enc_cfg.with_model(dtype=self.dtype)
        enc_cfg = replace(enc_cfg,
                          transform=replace(enc_cfg.transform,
                                            preprocess=self.preprocess))
        device = torch.device(device or 'cpu')
        return TileEncoderBackbone(self, enc_cfg.build(device))


class TileEncoderBackbone(nn.Module):
    """`[N, 3, H, W]` in [0, 1] -> `[N, out_channels, H/stride, W/stride]`.

    Satisfies `common.Interfaces.Backbone`. `check_shapes` is run at
    construction at the REAL tile size, not at the 2x-stride square
    `KeypointNet` uses: for a ViT the interesting failure is a position
    embedding that does not interpolate, and it only shows at a size the model
    was not built at.
    """

    trainable = False

    def __init__(self, cfg: TileEncoderBackboneConfig, trunk: TileEncoder):
        super().__init__()
        self.cfg = cfg
        # A list, so nn.Module's __setattr__ does not register 1.1B parameters
        # as children of a model whose checkpoint is meant to hold a decoder.
        self._trunk = [trunk]

        # `build()` writes cfg.preprocess into the encoder's transform, so the
        # two are one value. A backbone assembled by hand -- which the tests do,
        # and which is the only way to get one without weights -- can set them
        # apart, and then the identity would name a preprocessing the trunk is
        # not doing. Two names for one fact is the bug ConfigIdentity exists to
        # prevent, so it is refused rather than resolved in favour of either.
        if trunk.cfg.transform.preprocess != cfg.preprocess:
            raise ValueError(
                f'cfg.preprocess is {cfg.preprocess!r} and the trunk was built '
                f'with transform.preprocess='
                f'{trunk.cfg.transform.preprocess!r}. The identity would record '
                f'one and the forward would do the other')

        model = trunk.model                 # never a wrapper (TileEncoder._set_model)
        self.stride = _patch_stride(model, cfg.encoder)
        self.out_channels = int(trunk.model_spec.dim)
        self.device = trunk.device

        tile = int(cfg.tile_size)
        if tile % self.stride:
            raise ValueError(
                f'{cfg.encoder} has a patch size of {self.stride} and '
                f'tile_size is {tile}; {tile} % {self.stride} = '
                f'{tile % self.stride}, so the patch grid does not divide the '
                f'tile. The tiles this backbone accepts are the multiples of '
                f'{self.stride}: {self.stride * (tile // self.stride)} and '
                f'{self.stride * (tile // self.stride + 1)} are the two '
                f'nearest. The DECODER adds a second constraint that this '
                f'class cannot check, because a backbone does not know the '
                f'cell: the tile must divide by the cell as well, and '
                f'stride/cell must be a power of two. See '
                f'UpsampleDecoderConfig, and the module docstring here for why '
                f'patch 14 needs cell 7 rather than cell 8. Padding is not a '
                f'way out: it moves the cell grid off the pixel grid the '
                f'labels were splatted onto, and nothing would raise')
        _fit_input_size(model, tile, cfg.encoder)

        # The cheap assertion in front of the expensive run. One forward at the
        # real size, at construction: it catches the declared stride against
        # the real one, which changes nothing that raises and leaves the cell
        # grid a different size than the labels were splatted onto.
        check_shapes(self, image_size=tile, channels=3, device=self.device)

    @property
    def trunk(self) -> TileEncoder:
        return self._trunk[0]

    @property
    def encoder_id(self) -> str:
        """The encoder's own identity, for a CLI's `extra_identity`.

        This config says which encoder and how it is read; that id says which
        weights and which preprocessing. Both are wanted in a checkpoint, and
        only the first can be a dataclass field.
        """
        return self.trunk.identity_id()

    def normalise(self, images: torch.Tensor) -> torch.Tensor:
        """[0, 1] RGB -> what the trunk was trained on. The transform, minus geometry.

        `TransformConfig.build()` is `Resize -> CenterCrop -> [Grayscale] ->
        ToTensor -> Normalize`. The first two are geometry and are exactly what
        must NOT happen here; the last two are a distribution and must. Doing
        them as tensor ops keeps the batch on the device and differentiable,
        neither of which survives a PIL round trip.
        """
        if images.shape[1] != 3:
            raise ValueError(
                f'this backbone takes 3 channels and got {images.shape[1]}. '
                f'The grayscale arm is preprocess=\'grey\' on a 3-channel '
                f'batch, not a 1-channel input -- the patch embedding takes '
                f'three')
        if self.cfg_transform.preprocess == 'grey':
            # torchvision's Grayscale, which is ITU-R BT.601 luma -- the same
            # three coefficients as Backbones.to_model_channels, replicated
            # back to three channels because the patch embedding takes three.
            w = images.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            images = (images * w).sum(1, keepdim=True).expand(-1, 3, -1, -1)
        t = self.cfg_transform
        mean = images.new_tensor(t.mean).view(1, 3, 1, 1)
        std = images.new_tensor(t.std).view(1, 3, 1, 1)
        return (images - mean) / std

    @property
    def cfg_transform(self):
        """The encoder's `TransformConfig`. Named so `normalise` reads once."""
        return self.trunk.cfg.transform

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        h, w = images.shape[-2:]
        if h % self.stride or w % self.stride:
            raise ValueError(
                f'{h}x{w} is not a multiple of the stride {self.stride}. The '
                f'patch grid would cover less than the image while every label '
                f'is still splatted onto the full extent')
        batch = self.normalise(images)
        ctx = torch.no_grad() if not self.trainable else nullcontext()
        with ctx:
            return self.trunk.features_with_grad(batch, exit_name='spatial')


def _fit_input_size(model, tile: int, encoder: str) -> None:
    """Tell the trunk what size it is about to be fed. Once, at construction.

    `set_input_size` resamples `pos_embed` onto the new grid and updates
    `patch_embed.img_size`, so the `strict_img_size` assertion that guards a
    fixed-size model (`timm/layers/patch_embed.py:120`) now passes at the tile
    size instead of at 224. For a `dynamic_img_size` model it moves the same
    resample out of the per-batch path.

    Refuses rather than skipping when the model has no such method: silently
    doing nothing here works for the dynamic models and asserts, deep inside
    timm, for the fixed ones -- which is the failure this call exists to remove.
    """
    resize = getattr(model, 'set_input_size', None)
    if resize is None:
        raise TypeError(
            f'{encoder} has no set_input_size(), so it cannot be told to take a '
            f'{tile} px input. timm has had it since 1.0 '
            f'(VisionTransformer.set_input_size); a trunk that is not a timm '
            f'ViT has to say how it resizes')
    resize(img_size=(int(tile), int(tile)))


def _patch_stride(model, encoder: str) -> int:
    """The trunk's stride, read off the model. One number, or a refusal.

    NOT from `model_spec.feat_hw`. That is `crop_size // patch_size` -- the grid
    the encoder produces at INFERENCE, on a 224 px centre crop -- and this
    backbone is deliberately not doing the centre crop. Dividing the tile by it
    would be wrong by exactly the ratio between the tile and the crop, quietly.
    """
    try:
        ph, pw = (int(v) for v in model.patch_embed.patch_size)
    except AttributeError:
        raise TypeError(
            f'{encoder} has no patch_embed.patch_size, so its stride cannot be '
            f'read. Every implementation registered today is a timm ViT; a CNN '
            f'trunk would answer this differently and has to say how') from None
    if ph != pw:
        raise ValueError(
            f'{encoder} has a non-square patch ({ph}, {pw}) and Backbone.stride '
            f'is one number. Splitting it into two is a change to the protocol '
            f'and to every caller that divides by it')
    return ph
