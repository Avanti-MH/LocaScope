"""What every tile encoder in this project has to be, and what it may also be.

    encoder = SomeEncoderConfig().build(device)
    feats   = encoder(tiles)                 # [N, D] -- always
    toks    = encoder.tokens(tiles)          # [N, T, D] -- only some models

A tile encoder turns a list of 256 px images into one vector each. That much is
universal. Everything past it is not, and pretending otherwise is how a base
class stops being usable by the second implementation:

    kind        model output          who
    'vector'    [N, D]                a CNN with num_classes=0
    'tokens'    [N, T, D]             a ViT with global_pool=''
    'spatial'   [N, C, H, W]          a U-Net, or a CNN before its pooling

So `features()` is required and `tokens()` / `spatial()` / `pooled()` are
capabilities. A model that lacks one says so with a sentence naming what it IS,
rather than an AttributeError about embed_dim from four frames down.

features() is not one operation
-------------------------------
It returns [N, D] for every kind, but by three different routes, and only two of
them are arithmetic:

    'vector'    the output already is one. No choice.
    'spatial'   global average pool. A convention, and a defensible one.
    'tokens'    WHICH TOKEN, OR WHAT REDUCTION OF THEM -- a choice, and the
                model's authors usually made it. GigaPath's is the CLS, pinned
                to the reference tensor its own demo ships.

The first two are implemented here. The third is left abstract on purpose: a
base class that guessed would be answering a question it cannot know, and the
answer would be invisible in every number downstream.
"""

from __future__ import annotations

import sys
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Tuple

_HERE = Path(__file__).resolve().parent
for _d in (_HERE, _HERE.parent / 'utilities'):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
import torch.nn.functional as F                             # noqa: E402
from PIL import Image                                       # noqa: E402
from torchvision import transforms                          # noqa: E402

from ConfigIdentity import (IdentifiedBuild, IdentifiedConfig,  # noqa: E402
                            ModelConfig)


_INTERPOLATION = {
    'bicubic': transforms.InterpolationMode.BICUBIC,
    'bilinear': transforms.InterpolationMode.BILINEAR,
    'nearest': transforms.InterpolationMode.NEAREST,
}

KINDS = ('vector', 'tokens', 'spatial')


def _to_pil(img) -> Image.Image:
    return img if isinstance(img, Image.Image) else Image.fromarray(img)


# ── what a model produces ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class OutputSpec:
    """The shape of what a model hands back, read off the model, never assumed.

    `grid` means the patch grid for 'tokens' and the feature-map size for
    'spatial'; it is None for 'vector'. `num_prefix` is the slice point before
    the patch tokens -- 1 for a CLS, 5 for the DINOv2 _reg4 variants, and
    getting it wrong averages four register tokens in as if they were image
    content, silently, showing up only as a slightly worse score.
    """
    kind: str
    dim: int
    grid: Optional[Tuple[int, int]] = None
    num_prefix: int = 0

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f'kind must be one of {KINDS}, got {self.kind!r}')
        if self.kind == 'tokens' and self.grid is None:
            raise ValueError("kind='tokens' needs a patch grid")
        if self.kind == 'vector' and self.grid is not None:
            raise ValueError("kind='vector' has no grid")

    def n_tokens(self) -> int:
        if self.kind != 'tokens':
            raise ValueError(f'{self.kind} output has no token count')
        return self.num_prefix + self.grid[0] * self.grid[1]


# ── configuration ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TransformConfig(IdentifiedConfig):
    """The tile -> tensor pipeline, as parameters rather than as code.

    scale_size and crop_size are two numbers rather than timm's single crop_pct,
    because crop_pct is a ratio you have to divide to recover them and the
    division is where the mistakes live. crop_pct() reports it for comparison
    with other people's configs; nothing here consumes it.

    No defaults that mean anything: the numbers belong to whichever model was
    validated with them, so an implementation puts them in ITS baseline. A
    shared shape with a shared zero point would quietly impose one model's
    preprocessing on every other.
    """
    scale_size:    int = 256
    crop_size:     int = 224
    interpolation: str = 'bicubic'
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std:  Tuple[float, float, float] = (0.229, 0.224, 0.225)

    #: 'none' keeps RGB as given. 'grey' is luminance replicated to three
    #: channels, because a patch embedding takes three; colour leaves the
    #: problem entirely, and both sides of a comparison have to use it or the
    #: two are not describing the same thing.
    preprocess: str = 'none'

    def crop_pct(self) -> float:
        return self.crop_size / self.scale_size

    def build(self) -> transforms.Compose:
        if self.preprocess not in ('none', 'grey'):
            raise ValueError(
                f"preprocess must be 'none' or 'grey', got {self.preprocess!r}")
        if self.interpolation not in _INTERPOLATION:
            raise ValueError(
                f'interpolation must be one of {sorted(_INTERPOLATION)}, '
                f'got {self.interpolation!r}')
        steps = [
            transforms.Resize(self.scale_size,
                              interpolation=_INTERPOLATION[self.interpolation]),
            transforms.CenterCrop(self.crop_size),
        ]
        if self.preprocess == 'grey':
            steps.append(transforms.Grayscale(num_output_channels=3))
        steps += [
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ]
        return transforms.Compose(steps)


@dataclass(frozen=True)
class TileEncoderConfig(IdentifiedConfig):
    """Base for every encoder config. Implementations add their own fields.

    batch_size is excluded from identity and that is not a convenience: a ViT
    normalises per sample -- LayerNorm, no cross-sample coupling -- so batching
    cannot change a single vector. It is also the most-tuned knob in the repo,
    and hashing it would throw away every cached result on a throughput
    experiment.
    """
    model:     ModelConfig     = field(default_factory=ModelConfig)
    transform: TransformConfig = field(default_factory=TransformConfig)
    batch_size: int = 128

    NOT_IDENTITY = ('batch_size',)

    def with_model(self, **over) -> 'TileEncoderConfig':
        """A copy with some ModelConfig fields changed.

        `dtype` lives on the nested ModelConfig, so changing it would otherwise
        mean respelling `arch` at every call site -- and a caller who has to
        retype the architecture to switch precision will eventually retype it
        wrong. One source of truth for what the model is, one short way to vary
        the parts that are not the model.

            GigaPathEncoderConfig(batch_size=1024).with_model(dtype='fp32')
        """
        import dataclasses as _dc
        return _dc.replace(self, model=_dc.replace(self.model, **over))

    def build(self, device: torch.device, **kw) -> 'TileEncoder':
        raise NotImplementedError(
            f'{type(self).__name__} must implement build()')


# ── the encoder ───────────────────────────────────────────────────────────────

class TileEncoder(IdentifiedBuild):
    """Base for every tile encoder. Subclasses set cfg, device, model, spec.

    Provides the batch loop, the identity surface, `variant`, and features() for
    the two kinds whose reduction is arithmetic. A token model must say which
    token, because nothing here can know.
    """

    #: Fields a variant may change. Everything else decides what gets BUILT, so
    #: changing one has to go through a real construction or the object would
    #: describe a model it does not have.
    _VARIABLE = ('dtype', 'batch_size', 'transform')

    cfg: TileEncoderConfig
    spec: OutputSpec

    # ── capability gates ─────────────────────────────────────────────────────

    def _require(self, kind: str, what: str) -> None:
        if self.spec.kind != kind:
            raise TypeError(
                f'{what} needs a {kind!r} model and this one is '
                f'{self.spec.kind!r} (dim {self.spec.dim}'
                + (f', grid {self.spec.grid}' if self.spec.grid else '')
                + f'). features() works for every kind.')

    # ── encoding ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _run(self, images, reduce: Optional[Callable]) -> torch.Tensor:
        """One batch loop. `reduce` decides what crosses to the host.

        Order is fixed here and not left to the callback: .float() runs BEFORE
        reduce. Under autocast the output arrives fp16, and a reduction means
        .mean() or .std() over a couple of hundred vectors -- in fp16 that
        quietly degrades the averaged parts while leaving a single selected
        token exactly right. The symptom is "averaging poolings do not help",
        which reads as a result rather than an error.

        reduce=None moves everything to the host, which is the only safe
        default: [N, 197, 1536] fp32 is 1.21 MB per tile, so 68k tiles would be
        82 GB held on the device. A caller who wants it to stay there passes
        reduce=lambda t: t and owns that decision. The choice can only be made
        HERE -- by the time _run returns, torch.cat has already built the whole
        thing wherever it was going to live.
        """
        dtype = self.cfg.model.torch_dtype()
        ctx = (torch.autocast(device_type=self.device.type, dtype=dtype)
               if dtype is not torch.float32 else nullcontext())
        out = []
        for start in range(0, len(images), self.cfg.batch_size):
            batch = torch.stack([
                self._transform(_to_pil(img))
                for img in images[start:start + self.cfg.batch_size]
            ]).to(self.device)
            with ctx:
                raw = self.model(batch)
            raw = raw.float()
            out.append(raw.cpu() if reduce is None else reduce(raw))
        return torch.cat(out, dim=0)

    def _vector_from(self, raw: torch.Tensor) -> torch.Tensor:
        """One [B, ...] batch of model output -> [B, D], L2-normalised.

        Implemented for the two kinds where the reduction is not a choice. A
        token model overrides -- see the note in the module docstring.
        """
        if self.spec.kind == 'vector':
            return F.normalize(raw, dim=-1)
        if self.spec.kind == 'spatial':
            return F.normalize(raw.mean(dim=(-2, -1)), dim=-1)
        raise NotImplementedError(
            f'{type(self).__name__} produces {self.spec.kind!r} output, so '
            f'features() depends on which reduction its authors validated. '
            f'Override _vector_from.')

    def features(self, images) -> torch.Tensor:
        """[N, D] L2-normalised. Every encoder has this.

        The reduction runs inside the batch loop, so what crosses to the host is
        a few KB per tile rather than the full output.
        """
        return self._run(images, lambda t: self._vector_from(t).cpu())

    def tokens(self, images, reduce: Optional[Callable] = None) -> torch.Tensor:
        """[N, T, D] fp32, NOT normalised. Token models only.

        Normalising is left to whoever pools: mean-then-normalise and
        normalise-then-mean are different vectors, and only the first is what a
        pooled descriptor means.
        """
        self._require('tokens', 'tokens()')
        return self._run(images, reduce)

    def spatial(self, images, reduce: Optional[Callable] = None) -> torch.Tensor:
        """[N, C, H, W] fp32. Spatial models only."""
        self._require('spatial', 'spatial()')
        return self._run(images, reduce)

    def pooled(self, images, mode: str):
        """(features, slots, layout) for one reduction of a token set."""
        self._require('tokens', 'pooled()')
        raise NotImplementedError(
            f'{type(self).__name__} must implement pooled() -- the set of '
            f'modes belongs to whoever knows the token layout')

    def __call__(self, images) -> torch.Tensor:
        return self.features(images)

    # ── variants ─────────────────────────────────────────────────────────────

    def variant(self, **over) -> 'TileEncoder':
        """A second encoder over the SAME loaded model, configured differently.

        Sweeps are what this is for: batch size x precision, or two
        preprocessing arms. Building a second config would reload the weights to
        change a number, which in a throughput benchmark is most of what is
        being measured.

        Only _VARIABLE may change; anything else raises rather than handing back
        an encoder whose config disagrees with its weights. `dtype` lives on the
        nested ModelConfig, so it is rewritten there.
        """
        import dataclasses as _dc
        bad = set(over) - set(self._VARIABLE)
        if bad:
            raise ValueError(
                f'variant() cannot change {sorted(bad)} -- those decide what is '
                f'built. Construct a config instead. Changeable: '
                f'{list(self._VARIABLE)}')

        cfg = self.cfg
        if 'dtype' in over:
            cfg = _dc.replace(cfg, model=_dc.replace(cfg.model,
                                                     dtype=over.pop('dtype')))
        cfg = _dc.replace(cfg, **over) if over else cfg

        clone = object.__new__(type(self))
        clone.__dict__.update(self.__dict__)
        clone.cfg = cfg
        if 'transform' in over:
            clone._transform = cfg.transform.build()
        return clone
