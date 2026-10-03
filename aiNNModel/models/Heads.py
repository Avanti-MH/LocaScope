'''Classifier head components -- parameterised over `in_dim` so one class
serves any backbone's width (768 for ConvNeXt V2, 1536 for GigaPath/UNI2, or
anything else `TileEncoder.model_spec.dim` reports), and over precision
(`HeadConfig.dtype`) the same way `TileEncoderFunc`'s `ModelConfig.dtype` does
for a trunk.

    LinearHead          `in_dim -> num_classes`, one Linear layer. The main
                        line for both a frozen-feature and a fine-tuned-trunk
                        setup, and the classification stage every other head
                        here ultimately hands off to.
    MlpHead              Linear -> GELU -> Dropout -> Linear. An alternative
                        with more head capacity than the main line, kept
                        deliberately separate so a comparison can ask whether
                        it helps or hurts, rather than being the default.
    ArcFaceHead          Cosine classifier with an angular margin (Deng et
                        al., ArcFace CVPR 2019). See its own docstring for the
                        geometry and for why `s`/`m` are constants, not
                        learned parameters.
    AttentionPoolHead    Genuine LEARNABLE pooling over an un-pooled grid
                        (patch tokens or spatial cells), not more capacity
                        stacked on an already-pooled vector -- a REDUCTION,
                        used as `common.Head.Head`'s `reduction='attn'` stage,
                        never a classifier by itself.
    ResidualMlpBlock     `[D]->[D]`, `MlpHead`'s own transform half without
                        its final `num_classes` projection -- never a
                        classifier by itself either, shared by `training/
                        PrototypicalRoutingHead`'s Stage 2/3 arms that need
                        a per-member/per-query non-linear transform.

`common/Head.py` composes these into a runnable (reduction, classifier) pair;
this file has no opinion about that composition, only about what each
building block computes on its own. Moved here from
`training/MppRoutingHead/Models.py` on 2026-09-17 -- none of it was ever
specific to routing a tile to an mpp rung, and `1_estimate_query_mpp/
ClassifierEstMpp.py` needs the same classes a training loop does, without
importing a training package to get them.
'''
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════
#  precision -- same convention as ConfigIdentity.ModelConfig.dtype, for a
#  module that has no ModelConfig of its own to hang it off
# ══════════════════════════════════════════════════════════════════════════

def torch_dtype(dtype: str) -> torch.dtype:
    '''Same mapping, same error, as `ConfigIdentity.ModelConfig.torch_dtype`
    -- duplicated rather than imported, because a head has no `arch`/
    `source`/`weights` identity to hang a `ModelConfig` off, but the two must
    never disagree about what `'fp16'` means.'''
    try:
        return {'fp16': torch.float16, 'fp32': torch.float32}[dtype]
    except KeyError:
        raise ValueError(f"dtype must be 'fp16' or 'fp32', got {dtype!r}") from None


@dataclass(frozen=True)
class HeadConfig:
    '''What every head needs to know about its input and its precision.
    `num_classes` is unused by `AttentionPoolHead` (it only pools, it does
    not classify) -- one shared config shape across heads that need
    different subsets of it, same as `TileEncoderConfig.HEADS`/`POOLINGS`
    being read differently per subclass.

    `dtype` IS FP32, AND THAT IS NOT AN OVERSIGHT COPIED FROM SOMEWHERE ELSE.
    Inference precision and training precision are different questions. A
    FROZEN encoder in fp16 is correct: one forward pass, no gradients, and the
    measured cos against fp32 is 0.99995 (log/TODO.log). Parameters that an
    OPTIMIZER updates are not that case -- in pure fp16 the gradients
    underflow to zero or overflow to inf, Adam propagates the inf, and every
    logit afterwards is NaN.

    Real mixed precision keeps fp32 MASTER weights and scales the loss
    (`torch.autocast` + `GradScaler`). That is the upgrade if a head ever
    costs enough to matter; none defined here do -- the largest,
    `AttentionPoolHead`, is ~9.4M parameters, tens of MB in fp32.

    THIS DEFAULT ALONE DOES NOT FIX ANYTHING if a caller forwards a training
    loop's own precision flag straight through on top of it -- an explicit
    keyword always beats a dataclass default. `training/MppRoutingHead/
    cli/train.py` hit exactly this on 2026-09-16: it constructed every
    `HeadConfig` as `HeadConfig(..., dtype=args.dtype)`, and `--dtype`'s own
    CLI default was still `'fp16'`, so the fp16-under-Adam NaN never stopped
    happening until the call sites hardcoded `dtype='fp32'` instead of
    forwarding a flag. This default is the fallback for a caller that omits
    `dtype` entirely, not the actual guard -- a caller that trains a head
    still has to pass `dtype='fp32'` itself.
    '''
    in_dim: int
    num_classes: int = 6
    dtype: str = 'fp32'

    #: MlpHead ONLY -- every other classifier ignores these four, same as
    #: `AttentionPoolHead` already ignores `num_classes` above. Defaults
    #: reproduce the ORIGINAL single-hidden-layer MlpHead exactly
    #: (`mlp_width=None` -> `in_dim`), so `HeadConfig(**ckpt['head_cfg'])`
    #: on a checkpoint saved before these existed rebuilds the same
    #: architecture it was trained with, not a different one.
    mlp_depth: int = 1
    mlp_width: int | None = None
    mlp_residual: bool = False
    mlp_dropout: float = 0.1

    #: Which encoder blocks' layer tokens the head mixes, as ABSOLUTE 0-based
    #: block indices -- already resolved by `resolve_encoder_layers`, never
    #: the fractions a registry writes. `()` (the default) is the encoder's
    #: own last-layer `tokens()`, which is what every head trained before this
    #: field existed read, so `HeadConfig(**ckpt['head_cfg'])` on such a
    #: checkpoint rebuilds the same module.
    encoder_layers: tuple = ()


def resolve_encoder_layers(spec, depth: int) -> tuple:
    '''`encoder_layers` as a registry writes it -> absolute 0-based block
    indices, sorted.

    An INT is absolute and 0-based, negatives counted from the end, as timm's
    `indices` (`-1` and `depth - 1` are the same last block). A FLOAT is
    relative, in (0, 1]: block `round(r * depth) - 1`, at least 0, so `1.0` is
    the last block and `0.25` of 24 is block 5. `1` and `1.0` are different
    blocks on purpose.

    A list or tuple only: a bare int reads as timm/DINOv2's "the last n
    blocks", which is not what an int means here. Refused too: a bool, an index
    outside the model, and two entries that land on one block.'''
    if isinstance(spec, (int, float)) or not isinstance(spec, (list, tuple)):
        raise ValueError(
            f'encoder_layers {spec!r}: give a list or tuple of block indices '
            f'(int, 0-based) or depth fractions (float in (0, 1])')
    out = []
    for item in spec:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f'encoder_layers entry {item!r}: int or float only')
        if isinstance(item, int):
            if not -depth <= item < depth:
                raise ValueError(
                    f'encoder_layers entry {item}: the encoder has {depth} '
                    f'blocks, so an absolute index is in {-depth}..{depth - 1}')
            out.append(item % depth)
        else:
            if not 0.0 < item <= 1.0:
                raise ValueError(
                    f'encoder_layers entry {item}: a fraction is in (0, 1]')
            out.append(max(0, round(item * depth) - 1))
    if len(set(out)) != len(out):
        raise ValueError(
            f'encoder_layers {tuple(spec)} lands on blocks {out} of {depth}: '
            f'two entries name one block')
    return tuple(sorted(out))


# ══════════════════════════════════════════════════════════════════════════
#  Linear
# ══════════════════════════════════════════════════════════════════════════

class LinearHead(nn.Module):
    '''`in_dim -> num_classes`, one Linear layer. Matches the shape ConvNeXt
    V2's own released checkpoint head reduces to (`NormMlpClassifierHead` with
    `hidden_size` unset: Norm + one Linear, no hidden layer) and the standard
    linear-probe convention (SimCLR/CLIP) for measuring representation
    quality without head capacity masking it.
    '''

    #: See `ArcFaceHead.NEEDS_TARGET` -- this head's logits do not depend on
    #: the label, so `common.Head.Head` calls it with the features alone.
    NEEDS_TARGET = False

    def __init__(self, cfg: HeadConfig):
        super().__init__()
        self.cfg = cfg
        self.fc = nn.Linear(cfg.in_dim, cfg.num_classes,
                            dtype=torch_dtype(cfg.dtype))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        '''`features`: `[N, in_dim]`, already pooled (a CLS token, a GAP
        vector, or `AttentionPoolHead`'s output).'''
        return self.fc(features.to(self.fc.weight.dtype))


# ══════════════════════════════════════════════════════════════════════════
#  MLP (an alternative, not a default)
# ══════════════════════════════════════════════════════════════════════════

class MlpHead(nn.Module):
    '''`cfg.mlp_depth` hidden layers, each `cfg.mlp_width` wide (ONE width
    for all of them, not one per layer -- a sweep names two numbers instead
    of `mlp_depth` of them). `cfg.mlp_width=None` defaults to `cfg.in_dim`,
    same as before this was configurable, so this one class still serves
    768-d (ConvNeXt V2) and 1536-d (GigaPath/UNI2) input without retyping a
    size per backbone.

    `cfg.mlp_depth=1, mlp_width=None, mlp_residual=False` (the defaults)
    reproduce the ORIGINAL architecture exactly: `Linear(in_dim, in_dim) ->
    GELU -> Dropout -> Linear(in_dim, num_classes)`.

    `cfg.mlp_residual` wraps `x = x + block(x)` around every WIDTH-TO-WIDTH
    hidden layer (the 2nd through `mlp_depth`-th) -- NOT the first
    (`in_dim -> width`: shapes only match when `width == in_dim`) and NOT
    the final classification layer (`width -> num_classes`: a residual
    around the logits is not what "residual" means here). At `mlp_depth=1`
    there is no width-to-width layer to wrap, so `mlp_residual=True` is
    accepted but has nothing to apply to.
    '''

    NEEDS_TARGET = False

    def __init__(self, cfg: HeadConfig):
        super().__init__()
        self.cfg = cfg
        depth = max(1, cfg.mlp_depth)
        width = cfg.mlp_width or cfg.in_dim
        self.residual = cfg.mlp_residual
        dtype = torch_dtype(cfg.dtype)

        self.in_proj = nn.Linear(cfg.in_dim, width, dtype=dtype)
        # depth-1 width-to-width hidden layers -- depth=1 (the default)
        # leaves this empty, matching the original's single Linear->GELU->
        # Dropout->Linear with no room for a residual at all.
        self.hidden = nn.ModuleList([
            nn.Sequential(nn.GELU(), nn.Dropout(cfg.mlp_dropout),
                         nn.Linear(width, width, dtype=dtype))
            for _ in range(depth - 1)
        ])
        self.out_act = nn.GELU()
        self.out_drop = nn.Dropout(cfg.mlp_dropout)
        self.out_proj = nn.Linear(width, cfg.num_classes, dtype=dtype)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(features.to(self.in_proj.weight.dtype))
        for block in self.hidden:
            x = x + block(x) if self.residual else block(x)
        return self.out_proj(self.out_drop(self.out_act(x)))


# ══════════════════════════════════════════════════════════════════════════
#  ArcFace: a cosine classifier with an additive ANGULAR margin
# ══════════════════════════════════════════════════════════════════════════

class ArcFaceHead(nn.Module):
    '''`in_dim -> num_classes` through ANGLE alone, with a training-only
    margin on the true class. Deng et al., ArcFace (CVPR 2019).

    THE GEOMETRY. Both `weight` and the input are L2-normalised, so each of
    the `num_classes` weight ROWS is a direction on the unit sphere -- a
    learned prototype -- and the logit is the cosine between the feature and
    that direction. The decision boundary between two classes is the plane
    that bisects their angle.

    WHY `s`. `cos` lives in [-1, 1], and a softmax over classes in that range
    cannot be confident no matter how well separated they are -- cross-entropy
    stalls. `s` re-expands the range.

    WHY THE MARGIN, AND WHY IT IS A CONSTANT. Plain cosine stops pushing as
    soon as the right class wins, so the prototypes can sit arbitrarily close
    together. The margin demands that the true class win BY m radians:
    `s*cos(theta_y + m)` is deliberately HARDER than the task being scored,
    which is why `m` cannot itself be learned -- dL/dm > 0 always, so gradient
    descent would drive it to zero, and buying it back needs a counter-reward
    term whose own weight is then the hyperparameter (AdaptiveFace). `s` has
    no such contradiction but drifts upward until it saturates, so it is a
    constant here too. Sweep them; do not learn them.

    `forward` takes the target because of all this, and applies the margin
    ONLY in training mode: evaluation scores plain `s*cos(theta)`, which is
    what `argmax` reduces to anyway -- the margin is a training device, not
    part of the classifier.
    '''

    #: A caller composing heads reads this to decide whether to pass the
    #: label down. A flag rather than an isinstance check so a later head
    #: with the same need (CosFace, a margin-based prototype head) needs no
    #: edit at the call site.
    NEEDS_TARGET = True

    def __init__(self, cfg: HeadConfig, s: float = 64.0, m: float = 0.5):
        super().__init__()
        self.cfg = cfg
        self.s, self.m = float(s), float(m)
        self.weight = nn.Parameter(
            torch.empty(cfg.num_classes, cfg.in_dim, dtype=torch_dtype(cfg.dtype)))
        nn.init.xavier_uniform_(self.weight)
        # cos(theta+m) stops being monotonic in theta once theta+m > pi, so
        # past that point the margin would REWARD a worse angle. ArcFace's own
        # fix: fall back to a linear penalty there, continuous at the join.
        self._cos_m, self._sin_m = math.cos(self.m), math.sin(self.m)
        self._threshold = math.cos(math.pi - self.m)
        self._mm = self._sin_m * self.m

    def forward(self, features: torch.Tensor,
                target: torch.Tensor | None = None) -> torch.Tensor:
        '''`features`: `[N, in_dim]`. `target`: `[N]` class indices, required
        while training and ignored in eval mode.

        Computed in fp32 regardless of `cfg.dtype`: `sqrt(1 - cos^2)` loses
        most of its precision in fp16 exactly where it matters (`cos` near
        +-1, i.e. the confident samples), and `s=64` puts the logits an order
        of magnitude closer to fp16's range than the other heads' do. The
        PARAMETERS stay at `cfg.dtype` -- this is the arithmetic, not the
        storage.
        '''
        x = F.normalize(features.float(), dim=-1)
        w = F.normalize(self.weight.float(), dim=-1)
        cos = (x @ w.t()).clamp(-1.0, 1.0)              # [N, num_classes]
        if not self.training or target is None:
            return self.s * cos

        # cos(theta+m) without acos: it is the unstable step, and the angle is
        # never needed for its own sake.
        sin = (1.0 - cos.pow(2)).clamp_min(0.0).sqrt()
        cos_tm = cos * self._cos_m - sin * self._sin_m
        cos_tm = torch.where(cos > self._threshold, cos_tm, cos - self._mm)
        onehot = F.one_hot(target, self.cfg.num_classes).bool()
        return self.s * torch.where(onehot, cos_tm, cos)


# ══════════════════════════════════════════════════════════════════════════
#  AttentionPoolHead: learnable pooling over the UN-pooled grid
# ══════════════════════════════════════════════════════════════════════════

class AttentionPoolHead(nn.Module):
    '''Learnable pooling: `[N, L, D]` -> `[N, D]` via one learned query
    attending over the L positions, where L is patch tokens (CLS excluded --
    mixing the encoder's own pretrained summary token into a pooler meant to
    learn a TASK-specific aggregation would muddy which one a comparison is
    testing) or spatial cells for a CNN's own feature map. Outputs the pooled
    `[N, D]` vector, NOT logits -- chain a classifier after it
    (`common.Head.Head` pairs the two).

    Same role as CONCH's own `AttentionalPooler` (`aiNNModel/ConchVitFunc.py`),
    reimplemented small and generic here rather than reused: CONCH's version
    is one specific pretrained module with `d_model`/`context_dim`/`n_head`
    tied to ITS checkpoint's weights, and this one has to work at whatever
    `in_dim` a given backbone hands it, trained fresh here rather than loaded.

    Parameter count depends on `in_dim`/`n_head`, NOT on L -- unlike
    flattening the grid into one Linear, which for GigaPath's 14x14x1536 grid
    would be a first layer of ~231M parameters alone (measured 2026-09-15),
    several orders above what a small fine-tuning set can support without
    overfitting.
    '''

    def __init__(self, cfg: HeadConfig, n_head: int = 8):
        super().__init__()
        self.cfg = cfg
        dtype = torch_dtype(cfg.dtype)
        self.query = nn.Parameter(torch.randn(1, cfg.in_dim, dtype=dtype))
        self.attn = nn.MultiheadAttention(cfg.in_dim, n_head, batch_first=True,
                                          dtype=dtype)
        self.ln = nn.LayerNorm(cfg.in_dim, dtype=dtype)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        '''`grid`: `[N, L, D]` -- patch tokens (CLS already stripped by the
        caller) or spatial cells, whichever the backbone provides.'''
        grid = self.ln(grid.to(self.query.dtype))
        n = grid.shape[0]
        query = self.query.unsqueeze(0).expand(n, -1, -1)   # [N, 1, D]
        pooled, _ = self.attn(query, grid, grid, need_weights=False)
        return pooled.squeeze(1)                              # [N, D]


class ResidualMlpBlock(nn.Module):
    '''`[.., in_dim] -> [.., in_dim]`, the WHOLE transform wrapped in an
    outer residual (`x + transform(x)`) -- the pure set-MEMBER-transform
    half of `MlpHead` (same `in_proj`/hidden/`out_act`/`out_drop`/
    `out_proj` shape), WITHOUT that class's own final `num_classes`
    projection. `MlpHead` is a CLASSIFIER (fixed output width); this is a
    general-purpose `[D]->[D]` building block, never a classifier by
    itself -- same role `AttentionPoolHead` above already plays for
    pooling, not more capacity stacked onto an already-pooled vector.

    Added 2026-09-22 specifically so `training/PrototypicalRoutingHead`'s
    `PrototypeGenerators.SharedMlpPrototype` and `PrototypeRoutingHeads.
    AttnScoreHead`'s own `MLPq`/`MLPk` share ONE definition instead of two
    near-identical ones (caught the second copy before it shipped).

    `width` (defaults to `in_dim`) is the INNER width -- `SharedMlpPrototype`'s
    own `width_mult` axis stays expressible (pass `width=round(in_dim*
    width_mult)`); a caller with no width axis of its own just leaves it
    at the default.
    '''

    def __init__(self, in_dim: int, width: int = None, depth: int = 2,
                dropout: float = 0.1):
        super().__init__()
        width = width or in_dim
        depth = max(1, depth)
        self.in_proj = nn.Linear(in_dim, width)
        # depth-1 width-to-width hidden layers, each its OWN inner residual
        # -- `in_proj`/`out_proj` are not wrapped the same way `MlpHead`'s
        # own docstring explains: `in_proj` changes shape whenever `width
        # != in_dim`, `out_proj`'s shape is fixed regardless, so neither is
        # a width-to-width layer a residual could wrap without a mismatch.
        self.hidden = nn.ModuleList([
            nn.Sequential(nn.GELU(), nn.Dropout(dropout), nn.Linear(width, width))
            for _ in range(depth - 1)
        ])
        self.out_act = nn.GELU()
        self.out_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(width, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for block in self.hidden:
            h = h + block(h)
        return x + self.out_proj(self.out_drop(self.out_act(h)))


#: A REGISTERED NAME per classifier, same convention `TileEncoderFunc`'s
#: `_IMPLEMENTATIONS`/`encoder_names`/`encoder_config` use for encoders --
#: chosen by us and stable, rather than `type(obj).__name__`, which is a
#: Python implementation detail that moves the moment a class is renamed or
#: split. `common/Checkpoints.py` stores and reads a classifier by this name,
#: not by its class name, so a checkpoint's `classifier` field survives a
#: refactor of `LinearHead` the same way `encoder` surviving a refactor of
#: `GigaPathFunc` already does. `AttentionPoolHead` is NOT here -- it is a
#: REDUCTION (`common.Head.Head`'s other axis), never a `classifier` value.
_CLASSIFIER_REGISTRY = {
    'linear':  LinearHead,
    'mlp':     MlpHead,
    'arcface': ArcFaceHead,
}


def classifier_names() -> list:
    """The names a `classifier` field accepts. For argparse `choices`."""
    return sorted(_CLASSIFIER_REGISTRY)


def classifier_class(name: str) -> type:
    """The class registered as `name`."""
    try:
        return _CLASSIFIER_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f'no classifier called {name!r}. Known: '
            f'{", ".join(classifier_names())}') from None


def classifier_name(cls: type) -> str:
    """The reverse of `classifier_class`: what `save_checkpoint` writes for a
    built classifier's own type."""
    for name, registered in _CLASSIFIER_REGISTRY.items():
        if registered is cls:
            return name
    raise KeyError(
        f'{cls.__name__} is not registered under any name. Known: '
        f'{", ".join(classifier_names())}')
