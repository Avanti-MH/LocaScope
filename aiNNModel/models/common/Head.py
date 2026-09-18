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

import torch
import torch.nn.functional as F
import torch.nn as nn

from Heads import AttentionPoolHead, HeadConfig                      # noqa: E402


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
    reduces what is left).

    Both reduce WITHIN one tile, so every configuration scores on the same
    unit and the only thing that varies between two configurations is the one
    axis their names name.

    `target` is threaded through for classifiers that declare `NEEDS_TARGET`
    -- ArcFace's margin applies to the TRUE class, so its training-time logits
    depend on the label. The flag keeps that one classifier's requirement
    from becoming a signature every classifier has to honour, and keeps the
    label from silently reaching one that would ignore it.
    """

    def __init__(self, cfg: HeadConfig, reduction: str, classifier):
        super().__init__()
        if reduction not in ('fixed', 'attn'):
            raise ValueError(f"reduction must be 'fixed' or 'attn', got "
                             f'{reduction!r}')
        self.reduction = reduction
        self.pool = AttentionPoolHead(cfg) if reduction == 'attn' else None
        self.classify = classifier(cfg)

    def forward(self, raw: torch.Tensor, num_prefix: int,
                target: torch.Tensor | None = None) -> torch.Tensor:
        features = (pooled_view(raw, num_prefix) if self.pool is None
                    else self.pool(grid_view(raw, num_prefix)))
        if getattr(self.classify, 'NEEDS_TARGET', False):
            return self.classify(features, target)
        return self.classify(features)
