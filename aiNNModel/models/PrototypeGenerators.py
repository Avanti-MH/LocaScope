'''`training/PrototypicalRoutingHead/spec.md`'s Stage 2: generators that
turn a support SET into one prototype vector. Every class here shares one
shape:

    prototype = generator(support)       # support: [K, D], one rung's own
                                         # pooled support-tile vectors ->
                                         # [D], one vector

-- not a per-tile pooling (that is `aiNNModel/models/Pooling.py`'s job,
done once per tile BEFORE any of these ever run), a per-SET one. RUNS ONE
RUNG AT A TIME: support tiles of DIFFERENT rungs must never attend to each
other (spec.md is explicit about this), so the caller loops the rungs and
calls a generator once per rung, never once for the whole support pool.
Nothing here enforces that; it is a calling-convention invariant, the same
kind `Pooling`'s "same instance for support and query" is.

ALL Stage 2 arms built so far live HERE, in ONE file, the same
organisation `PrototypeRoutingHeads.py` already uses for Stage 3 -- before
2026-09-21 they were three separate files (`SetTransformerPrototype.py`/
`TrivialPrototype.py`/`SharedMlpPrototype.py`), which was the odd one out
rather than a deliberate choice: nothing about Stage 2 needed three files
any more than Stage 3 needed one per routing head. `AttnPoolPrototype`
(2026-09-22) joined the same way -- no reason for it to be a fourth file
either.

Lives in `aiNNModel/models/`, not the training package, because it is
encoder/head plumbing (a module that turns tile embeddings into a
prototype), not training logic -- the same line CLAUDE.md's repo-layout
section already draws for `Heads.py`/`common/Head.py`.
'''
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from Heads import AttentionPoolHead, HeadConfig, ResidualMlpBlock


# ══════════════════════════════════════════════════════════════════════════
#  main line: self-attention over the support set, then collapse
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SetTransformerConfig:
    in_dim: int = 1536
    n_head: int = 8
    n_layer: int = 2
    #: `nn.TransformerEncoderLayer`'s own convention (4x d_model) when unset.
    dim_feedforward: Optional[int] = None
    dropout: float = 0.1


class SetTransformerPrototype(nn.Module):
    '''`forward(support) -> prototype`. `support`: `[K, D]`, one rung's own
    pooled support-tile vectors (`K` = however many support tiles that rung
    drew, not fixed across rungs or episodes). Returns `[D]`.

    Collapse-after-attention is an `AttentionPoolHead` (2026-09-21, replacing
    a plain mean over the transformed set) -- ONE learned query attends over
    the K transformed members and picks out which of them matter most for
    this prototype, rather than weighting all K equally the way a mean does.
    A SEPARATE instance from any `Pooling(kind='attn')` the caller also
    built: that one attends over a token GRID within one tile; this one
    attends over a MEMBER SET across K support tiles -- same class, two
    different questions, so no weights are shared between them.
    '''

    def __init__(self, cfg: SetTransformerConfig):
        super().__init__()
        self.cfg = cfg
        ff = cfg.dim_feedforward or 4 * cfg.in_dim
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.in_dim, nhead=cfg.n_head, dim_feedforward=ff,
            dropout=cfg.dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layer)
        self.attn_pool = AttentionPoolHead(HeadConfig(in_dim=cfg.in_dim),
                                           n_head=cfg.n_head)

    def forward(self, support: torch.Tensor) -> torch.Tensor:
        # [K, D] is ONE sequence of length K, not K sequences of length 1 --
        # batch_first's batch dim is 1 here on purpose, so self-attention
        # runs ACROSS the K tiles rather than each tile attending only to
        # itself.
        x = self.encoder(support.unsqueeze(0))          # [1, K, D]
        # `AttentionPoolHead.forward` expects `[N, L, D]` and collapses L --
        # N=1 (one sequence), L=K (the K transformed support members).
        return self.attn_pool(x).squeeze(0)                # [D]


# ══════════════════════════════════════════════════════════════════════════
#  attention pool, no cross-member self-attention first
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AttnPoolConfig:
    in_dim: int = 1536
    n_head: int = 8


class AttnPoolPrototype(nn.Module):
    '''`forward(support) -> prototype`, `AttentionPoolHead` applied directly
    to the K support members with NO prior self-attention among them (that
    is `SetTransformerPrototype`'s own extra step -- see its docstring's
    "Collapse-after-attention" note). Isolates whether letting support
    members inform each other BEFORE collapsing buys anything over just
    collapsing the raw pooled vectors directly -- the direct comparison
    point for `SetTransformerPrototype`, the way `SharedMlpPrototype` is
    for whether cross-member attention specifically (as opposed to a
    per-member transform) matters.

    2026-09-22: this is what `CrossAttentionMatch.collapse` used to
    reimplement from scratch. That reimplementation added nothing over
    `AttentionPoolHead` itself -- `CrossAttentionMatch`'s Stage 2 and
    Stage 3 registry entries were ALREADY separate instances with no
    shared weights (`Runtime.py`'s own builders each `to(device)` a fresh
    one), so there was no actual sharing for reimplementing it to
    preserve, only a second copy of the same six lines able to drift from
    this one. `collapse` was deleted the same day; this class is its
    replacement, and `CrossAttentionMatch` is Stage-3-only from here on.
    '''

    def __init__(self, cfg: AttnPoolConfig):
        super().__init__()
        self.cfg = cfg
        self.attn_pool = AttentionPoolHead(HeadConfig(in_dim=cfg.in_dim),
                                           n_head=cfg.n_head)

    def forward(self, support: torch.Tensor) -> torch.Tensor:
        # `AttentionPoolHead.forward` expects `[N, L, D]` and collapses L --
        # N=1 (one sequence), L=K (the K support members, UNTRANSFORMED).
        return self.attn_pool(support.unsqueeze(0)).squeeze(0)     # [D]


# ══════════════════════════════════════════════════════════════════════════
#  zero-parameter floor
# ══════════════════════════════════════════════════════════════════════════

TRIVIAL_KINDS = ('mean', 'median')


@dataclass(frozen=True)
class TrivialPrototypeConfig:
    kind: str = 'mean'


class TrivialPrototype(nn.Module):
    '''`pooled.mean(dim=0)` or `pooled.median(dim=0).values`, no learned
    parameters at all. The FLOOR for the Stage 2 generator axis -- "does
    SetTransformerPrototype's self-attention over the support set do
    anything at all" needs a zero-parameter alternative to beat, not just
    another learned one (the same role Stage 4's own Baseline arm was
    meant to play one level up, for episodic training itself -- `cli/
    train_baseline.py`, deleted 2026-09-22, its own inference-time
    positioning never having settled).
    '''

    def __init__(self, cfg: TrivialPrototypeConfig):
        super().__init__()
        if cfg.kind not in TRIVIAL_KINDS:
            raise ValueError(f'kind must be one of {TRIVIAL_KINDS}, got {cfg.kind!r}')
        self.cfg = cfg

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        '''`pooled`: `[K_support, D]` -> `[D]`, same shape contract as
        `SetTransformerPrototype.forward`.'''
        return (pooled.mean(dim=0) if self.cfg.kind == 'mean'
               else pooled.median(dim=0).values)


# ══════════════════════════════════════════════════════════════════════════
#  Deep Sets: per-member transform, no cross-member attention
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SharedMlpConfig:
    in_dim: int = 1536
    depth: int = 2
    width_mult: float = 1.0
    dropout: float = 0.1


class SharedMlpPrototype(nn.Module):
    '''Deep-Sets-style generator (Zaheer et al. 2017): the SAME small MLP
    applied to each support member INDEPENDENTLY -- no cross-member
    interaction at all, unlike `SetTransformerPrototype`'s self-attention
    across the whole support set -- then mean-pooled. Isolates whether
    attention ACROSS support members buys anything over a per-member
    learned transform: the standard Deep Sets vs Set Transformer ablation
    in the permutation-invariant-set literature.

    `depth`/`width_mult` default to 2/1.0, not narrower -- a narrower MLP
    (e.g. `width_mult=0.5`) additionally confounds "less capacity" with "no
    attention" against `SetTransformerPrototype`'s own `n_layer=2`/
    `n_head=8`/`dim_feedforward=4*in_dim`. A narrower variant is a
    legitimate SECOND question ("how cheap can this get without losing
    much"), answered once THIS config has a real number -- a separate
    registry entry, the same reason `MppRoutingHead.HEAD_CHOICES`'
    `mlp`/`mlp_narrow` are two names rather than one guessed default.

    BOTH residuals are unconditional, not a knob: the OUTER one
    (`Heads.ResidualMlpBlock`'s own `x + transform(x)`, applied before the
    mean pool here) and the INNER one (each width-to-width hidden block,
    same pattern `Heads.MlpHead`'s `mlp_residual=True` uses).
    `SetTransformerPrototype`'s `nn.TransformerEncoderLayer` has residual
    connections around both its
    self-attention and feedforward sublayers built in -- a standard
    Transformer block -- which anchors its output near its input at
    initialisation. An MLP with no residual starts as an arbitrary,
    unrelated transform of the input at init instead, so a score
    difference against it would partly measure "residual-anchored vs not",
    not "attention vs not" -- keeping both on is what isolates the ONE
    axis this arm exists to test. `in_proj`/`out_proj` are NOT wrapped in
    the inner residual, same reason `MlpHead`'s are not: `in_proj` changes
    shape whenever `width_mult != 1`, and `out_proj`'s shape is fixed
    (`hidden -> in_dim`) regardless of `width_mult`, so neither is a
    width-to-width layer a residual could wrap without a shape mismatch.
    The OUTER residual is unaffected by `width_mult` either way, since it
    only requires the TRANSFORM's input and output to match (`in_dim ->
    in_dim`), which the final `out_proj` guarantees regardless of what
    `hidden` is internally.
    '''

    def __init__(self, cfg: SharedMlpConfig):
        super().__init__()
        self.cfg = cfg
        width = max(1, round(cfg.in_dim * cfg.width_mult))
        # ResidualMlpBlock (Heads.py, 2026-09-22) bakes BOTH residuals in
        # (inner, per hidden block; outer, around the whole transform) --
        # this class no longer wraps a second outer `pooled + ...` around
        # it, that would double the residual. Same math as before this
        # refactor, one definition instead of two near-identical ones
        # (`PrototypeRoutingHeads.AttnScoreHead`'s own MLPq/MLPk share it).
        self.block = ResidualMlpBlock(cfg.in_dim, width=width, depth=cfg.depth,
                                      dropout=cfg.dropout)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        '''`pooled`: `[K_support, D]` -> `[D]`, same shape contract as
        `SetTransformerPrototype.forward`.'''
        return self.block(pooled).mean(dim=0)


# ══════════════════════════════════════════════════════════════════════════
#  Collapse switched off
# ══════════════════════════════════════════════════════════════════════════

class PassthroughCollapse(nn.Module):
    '''`--collapse off`. `forward(support) -> support`, unchanged --
    byte-for-byte `nn.Identity`, but kept as its OWN class rather than
    reused from `ContextEncoders.py`'s F off-switch or a bare
    `nn.Identity()` (2026-09-22): this one carries `COLLAPSES = False`,
    the flag `episode_forward` reads to decide whether `prototypes` comes
    out as a collapsed `[K,D]` stack or the per-rung `{rung:[K_r,D]}` dict
    -- monkey-patching that flag onto PyTorch's own `nn.Identity` class
    would be a global mutation of a stdlib type, affecting every OTHER
    use of `nn.Identity` anywhere in the process, not a property of one
    registry entry.
    '''

    COLLAPSES = False

    def forward(self, support: torch.Tensor) -> torch.Tensor:
        return support
