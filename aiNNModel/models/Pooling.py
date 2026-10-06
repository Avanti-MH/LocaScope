'''Shared support/query pooling -- Cls / Avg / Max / Attn / passthrough.

    pool = Pooling(PoolingConfig(kind='attn', in_dim=1536))
    support_vecs = pool(support_raw, num_prefix)     # [N_support, D]
    query_vec    = pool(query_raw, num_prefix)        # SAME instance, SAME weights

spec.md's Stage 2 is explicit about why this is ONE class called twice
rather than two: support and query must go through the SAME pooling choice
with the SAME weights, or they land in two different embedding spaces and
no distance computed after that means anything. That was a real gap in the
first draft of the whole design (caught in review before any code existed)
-- one `Pooling` instance, held by whichever generator/estimator builds it
and passed both a support batch and a query batch, is what makes "the same
f_theta" true by construction instead of by two call sites remembering to
agree.

`Avg`/`Max` are new here -- `common.Head.Head` only ever needed CLS-or-GAP
(`pooled_view`) and learned attention (`AttentionPoolHead`); this project's
Stage 2 main line (`SetTransformerPrototype`) wants the whole per-tile
token grid, unreduced, which is `passthrough`.

`Cls` takes the backbone's own pretrained prefix token
(`raw[:, 0]`) instead of reducing the patch grid at all -- the simplest of
the five, no learned parameters and no grid computation, a floor to compare
`avg`/`max`/`attn` against the same way `TrivialPrototype` is Stage 2's own
generator-side floor. See `forward`'s own docstring for why it does not
reuse `common.Head.pooled_view` outright.
'''
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from Head import grid_view
from Heads import AttentionPoolHead, HeadConfig

POOLING_KINDS = ('avg', 'max', 'attn', 'passthrough', 'cls')


@dataclass(frozen=True)
class PoolingConfig:
    kind: str = 'cls'
    in_dim: int = 1536
    #: AttentionPoolHead's own default, read only when kind='attn'.
    n_head: int = 8


class Pooling(nn.Module):
    '''One (kind, weights) pair. `forward(raw, num_prefix)`:

        avg / max     -> [N, D]    mean or channel-max over the L non-prefix
                                   positions, CLS/registers dropped first
        attn          -> [N, D]    AttentionPoolHead, one learned query
        passthrough   -> [N, L, D] CLS/registers dropped, nothing pooled --
                                   for a generator that wants the whole grid

    `avg` is NOT `common.Head.pooled_view`'s CLS shortcut. `pooled_view`
    returns `tokens[:, 0]` (the backbone's OWN pretrained summary token)
    whenever `num_prefix>=1`, which is a different vector from the mean of
    the PATCH tokens -- deliberately excluded here for the same reason
    `AttentionPoolHead`'s own docstring excludes CLS from its attention
    grid: mixing the encoder's own pretrained summary into a pooling choice
    meant to be compared apples-to-apples across Avg/Max/Attn would muddy
    which one a comparison is actually testing. So `Pooling`'s "Avg" always
    means "mean over the patch grid", identically for a prefixed backbone
    (GigaPath, UNI2) and one without -- one definition, not one that
    silently changes meaning by encoder the way `pooled_view`'s does on
    purpose for a different job.
    '''

    def __init__(self, cfg: PoolingConfig):
        super().__init__()
        if cfg.kind not in POOLING_KINDS:
            raise ValueError(
                f'kind must be one of {POOLING_KINDS}, got {cfg.kind!r}')
        self.cfg = cfg
        self.attn = (AttentionPoolHead(HeadConfig(in_dim=cfg.in_dim),
                                       n_head=cfg.n_head)
                    if cfg.kind == 'attn' else None)

    def forward(self, raw: torch.Tensor, num_prefix: int) -> torch.Tensor:
        '''`raw`: `[N, L+num_prefix, D]`, an encoder's un-reduced exit for a
        batch of tiles (support OR query -- this function does not know or
        care which). `num_prefix`: the backbone's own CLS+register count
        (`model_spec.num_prefix`), read off the SAME encoder the caller
        built, never hardcoded -- see `common.Head.grid_view`'s own
        docstring for why a literal `[:, 1:]` would silently fold UNI2's 8
        register tokens in as if they were patches.

        `cls` is the one kind that reads the PREFIX, not the patch grid --
        `raw[:, 0]`, the backbone's own pretrained summary token. It is NOT
        L2-normalised here, same contract `avg`/`max`/`attn` already keep:
        normalisation happens exactly once, at whichever routing head
        actually needs unit vectors (`CosineTauHead.forward`), not partway
        through pooling. That is also why this is not simply a call to
        `common.Head.pooled_view` -- that helper bakes the normalise+fp32
        cast in, which would make `cls` the one kind with a different
        output contract from the other four.
        '''
        if self.cfg.kind == 'cls':
            if num_prefix == 0:
                raise ValueError(
                    'cls pooling needs a prefix token; this encoder has '
                    'num_prefix=0 (no CLS/register tokens to take)')
            return raw[:, 0]
        grid = grid_view(raw, num_prefix)              # [N, L, D]
        if self.cfg.kind == 'passthrough':
            return grid
        if self.cfg.kind == 'avg':
            return grid.mean(dim=1)
        if self.cfg.kind == 'max':
            return grid.max(dim=1).values
        return self.attn(grid)
