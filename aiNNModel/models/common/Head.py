'''`Head`: one (reduction, classifier) pair, composed into a runnable module
that turns an encoder's un-reduced exit into class logits.

TWO AXES, not a list of alternatives. The classifier of a fixed-CLS head and
of a learned-attention-pooled head can be the SAME `LinearHead`; what differs
between them is the REDUCTION -- the encoder's own fixed CLS/GAP against a
learned `AttentionPoolHead`. Naming variants "linear" and "attnpool" side by
side hides the axis that is actually changing.

NOT NAMED `Arm`. It was, while this lived inside `training/MppRoutingHead/`
and its whole point was several variants trained SIDE BY SIDE for comparison
-- "arm" the way a clinical trial has arms. Moved here on 2026-09-17 for a
different use (`1_estimate_query_mpp/ClassifierEstMpp.py` loads exactly ONE
trained head to run inference with, not several to compare), where "arm"
reads as a comparison that is not happening. `Head` is the standard word for
"the task-specific top of a network" and is what the task-specific registries
that USE this class (still living with their tasks, e.g. `training/
MppRoutingHead/Runtime.py`'s `HEAD_CHOICES`) already name their entries.
'''
from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F
import torch.nn as nn

from Features import LayerTokens                                    # noqa: E402
from Heads import AttentionPoolHead, HeadConfig                      # noqa: E402

#: Every reduction a `Head` knows. `fixed` and `attn` are the two every
#: checkpoint before 2026-10-01 carries; `clsattn` was added then.
REDUCTIONS = ('fixed', 'attn', 'clsattn')


def pooled_view(raw: torch.Tensor, num_prefix: int) -> torch.Tensor:
    '''CLS (`num_prefix>=1`, `tokens[:, 0]`) or GAP (`num_prefix==0`,
    `mean(dim=1)`) view for a `reduction='fixed'` `Head` -- L2-normalised and
    cast to fp32, matching what `TileEncoder.features()` would have handed
    back directly (bypassed here because the caller holds the raw, un-reduced
    exit so that one forward pass serves both this and `grid_view`).'''
    vec = raw[:, 0] if num_prefix else raw.mean(dim=1)
    return F.normalize(vec.float(), dim=-1)


def grid_view(raw: torch.Tensor, num_prefix: int) -> torch.Tensor:
    '''Patch/spatial-cell grid for a `reduction='attn'` `Head`, CLS AND any
    register tokens excluded via `num_prefix` -- NOT a hardcoded `[:, 1:]`,
    which would be right for a single-CLS ViT and silently wrong for one with
    register tokens (num_prefix > 1: those would be folded in as if they were
    patches).'''
    return raw[:, num_prefix:].float()


class Head(nn.Module):
    """One (reduction, classifier) pair, from `[N, L, D]` to `[N, classes]`.

    `reduction='fixed'` takes the encoder's own answer -- `pooled_view`, which
    is `tokens[:, 0]` for a prefixed model and the cell mean for one without,
    L2-normalised. `reduction='attn'` hands the patch cells to a learned query
    instead (`grid_view` drops the prefix by `num_prefix`; `AttentionPoolHead`
    reduces what is left). `reduction='clsattn'` keeps both: the last block's
    CLS and the attention-pooled patches, each L2-normalised, concatenated and
    LayerNormed, so the classifier sees `2 * in_dim`. It needs a CLS, so a
    model with no prefix (a CNN) is refused.

    `cfg.encoder_layers` non-empty MIXES those blocks' layer tokens before the
    reduction: each block through its own LayerNorm, then a softmax-weighted
    sum (the layer weights, one scalar per block). Defined for `attn` and
    `clsattn` -- the mix is a grid of patches, which is what they pool; the CLS
    of `clsattn` stays the last block's. The input is then a `LayerTokens`.

    Both reduce WITHIN one tile, so every configuration scores on the same
    unit and the only thing that varies between two configurations is the one
    axis their names name.

    A head with `fixed`/`attn` and no `encoder_layers` builds exactly the
    modules it built before either existed, so an old checkpoint's state dict
    loads.

    `target` is threaded through for classifiers that declare `NEEDS_TARGET`
    -- ArcFace's margin applies to the TRUE class, so its training-time logits
    depend on the label. The flag keeps that one classifier's requirement
    from becoming a signature every classifier has to honour, and keeps the
    label from silently reaching one that would ignore it.
    """

    def __init__(self, cfg: HeadConfig, reduction: str, classifier):
        super().__init__()
        if reduction not in REDUCTIONS:
            raise ValueError(f'reduction must be one of {REDUCTIONS}, got '
                             f'{reduction!r}')
        layers = tuple(cfg.encoder_layers)
        if layers and reduction == 'fixed':
            raise ValueError(
                f'encoder_layers {layers} with reduction fixed: a mix is a grid '
                f'of patches and fixed reads no patches. Use attn or clsattn')
        self.reduction = reduction
        self.layers = layers
        self.pool = AttentionPoolHead(cfg) if reduction != 'fixed' else None
        if layers:
            self.mix_norms = nn.ModuleList(nn.LayerNorm(cfg.in_dim)
                                           for _ in layers)
            self.mix_logits = nn.Parameter(torch.zeros(len(layers)))
        if reduction == 'clsattn':
            self.cat_norm = nn.LayerNorm(2 * cfg.in_dim)
            # The classifier reads both halves; its hidden width stays the
            # encoder's own unless the config names one.
            cfg = replace(cfg, in_dim=2 * cfg.in_dim,
                          mlp_width=cfg.mlp_width or cfg.in_dim)
        self.classify = classifier(cfg)

    @property
    def layer_weights(self) -> torch.Tensor:
        """The softmax over the mixed blocks, in `self.layers` order."""
        return torch.softmax(self.mix_logits, dim=0)

    def _grid(self, raw, num_prefix: int) -> torch.Tensor:
        if not self.layers:
            # A run with any mix_ head hands EVERY head a LayerTokens; one that
            # mixes nothing reads the last block, exactly what it got before.
            last = raw.last if isinstance(raw, LayerTokens) else raw
            return grid_view(last, num_prefix)
        if not isinstance(raw, LayerTokens):
            raise TypeError(f'this head mixes encoder layers {self.layers}; pass '
                            f'encode_raw(..., layers=...)\'s LayerTokens')
        missing = [b for b in self.layers if b not in raw.layers]
        if missing:
            raise KeyError(f'LayerTokens holds blocks {sorted(raw.layers)}, not '
                           f'{missing}')
        w = self.layer_weights
        return sum(w[k] * norm(grid_view(raw.layers[b], num_prefix))
                   for k, (b, norm) in enumerate(zip(self.layers, self.mix_norms)))

    def forward(self, raw, num_prefix: int,
                target: torch.Tensor | None = None) -> torch.Tensor:
        last = raw.last if isinstance(raw, LayerTokens) else raw
        if self.reduction == 'fixed':
            features = pooled_view(last, num_prefix)
        elif self.reduction == 'attn':
            features = self.pool(self._grid(raw, num_prefix))
        else:
            if num_prefix < 1:
                raise ValueError('clsattn reads the CLS token and this input has '
                                 'no prefix (a CNN, or a fine-tuned spatial exit)')
            cls = F.normalize(last[:, 0].float(), dim=-1)
            pooled = F.normalize(self.pool(self._grid(raw, num_prefix)).float(),
                                 dim=-1)
            features = self.cat_norm(torch.cat([cls, pooled], dim=-1))
        if getattr(self.classify, 'NEEDS_TARGET', False):
            return self.classify(features, target)
        return self.classify(features)
