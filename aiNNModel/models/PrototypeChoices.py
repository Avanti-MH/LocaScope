'''The registry of every selectable Stage 2/Stage 3 arm for `training/
PrototypicalRoutingHead`'s metric-based main line -- moved here from that
package's own `Runtime.py` (2026-09-22) specifically so the GENERIC
checkpoint layer (`aiNNModel/models/common/Checkpoints.py`'s
`build_prototype_from_checkpoint`) can read it too, without importing a
training package into `aiNNModel/models/` -- the same reason `1_estimate_
query_mpp/ClassifierEstMpp.py` reads `Heads._CLASSIFIER_REGISTRY`/
`TileEncoderFunc.encoder_config` rather than anything under `training/`.
`Runtime.py` re-exports these four names unchanged, so `cli/train.py`'s
own `from Runtime import (...)` did not need to change at all.

Stage 2 is the pipeline `pool -> G (support-context) -> F (query-context)
-> Collapse -> Stage 3` (`training/PrototypicalRoutingHead/spec.md`), each
of G/F/Collapse an INDEPENDENTLY switchable, independently instanced
sub-stage:

    SUPPORT_CONTEXT_CHOICES   G, `--support-context`
    QUERY_CONTEXT_CHOICES     F, `--query-context`
    COLLAPSE_CHOICES          Collapse, `--collapse` (renamed from
                              `--generator`/`GENERATOR_CHOICES` 2026-09-22:
                              "generator" stopped being accurate once this
                              stage was ONE OF THREE Stage 2 sub-stages
                              rather than all of Stage 2 -- it does not
                              generate anything G/F do not already touch,
                              it specifically COLLAPSES a support set to
                              one prototype, or (`off`) does not)

`identity` is the off switch for G and F (`aiNNModel/models/
ContextEncoders.py`); `COLLAPSE_CHOICES` has a matching `off` entry
(Collapse skipped, support stays per-member). `ROUTING_HEAD_CHOICES`
(Stage 3, all THREE now in `aiNNModel/models/PrototypeRoutingHeads.py`)
has: `cosine_tau` (pairs with a collapsed `[K,D]` prototype stack),
`cosine_logsumexp` (pairs ONLY with `--collapse off`'s per-rung,
un-collapsed `{rung:[K_r,D]}` -- see `PrototypeRoutingHeads.
CosineLogSumExpHead`'s own docstring for why it produces the SAME kind of
output `cosine_tau` does -- unnormalised logits -- despite the different
mechanism, so both share `Losses.compute_loss` unchanged), and
`attn_score` (`PrototypeRoutingHeads.AttnScoreHead`, learned multi-head
Q/K projections, pairs with EITHER shape -- its own `ACCEPTS_EITHER` flag
skips `cli/train.py`'s usual Collapse/head compatibility check).

`matching_net`/`matching_net_prototype`/`matching_net_raw` (2026-09-22,
same day as they were added) are GONE: forcing G+F+the Stage-3 kernel
into one bespoke "Matching Net" class was the wrong shape once G/F became
their own sub-stages with their own instances -- "Matching Net" is now a
COMPOSITION (`--support-context bilstm --query-context attnlstm --collapse
off --routing-head cosine_logsumexp`), not a name in any registry.
`matching_net_prototype` specifically degenerated to exactly `cosine_tau`
and added nothing under any name.

`mean` is the zero-parameter floor (`PrototypeGenerators.TrivialPrototype`
-- `median` cut from the comparison surface, see `COLLAPSE_CHOICES`'s own
comment); `shared_mlp` (`SharedMlpPrototype`) is the Deep-Sets-style
per-member-transform-then-pool arm, the direct comparison point for "does
`set_transformer`'s cross-member self-attention buy anything" -- see that
class's own docstring for why it is built with the same depth/residual
structure as `SetTransformerPrototype` rather than a narrower/cheaper
one, on the first pass. Every Stage 2 entry (G, F, Collapse) is a BUILDER
FUNCTION, `(in_dim, device) -> (nn.Module, config-or-None)`; Stage 3 is
`(in_dim, device) -> nn.Module` -- not a `(class, config_class)` tuple the
way `MppRoutingHead.HEAD_CHOICES` is: what config fields a future arm
needs is not always known ahead of a second entry to test the shape
against -- the same calibration-not-guessing reasoning `ClaudeRules.md`
states for thresholds. A builder function is free to read whatever CLI
args or literal hyperparameters it needs.
'''
from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch.nn as nn

from PrototypeGenerators import (SetTransformerPrototype, SetTransformerConfig,  # noqa: E501
                                 TrivialPrototype, TrivialPrototypeConfig,
                                 SharedMlpPrototype, SharedMlpConfig,
                                 AttnPoolPrototype, AttnPoolConfig,
                                 PassthroughCollapse)
from PrototypeRoutingHeads import CosineTauHead, CosineLogSumExpHead, AttnScoreHead
from ContextEncoders import (IdentityQueryContext,
                             BiLstmSupportContext, BiLstmContextConfig,
                             AttnLstmQueryContext, AttnLstmContextConfig)


# ══════════════════════════════════════════════════════════════════════════
#  Stage 2, Collapse sub-stage
# ══════════════════════════════════════════════════════════════════════════

def _build_set_transformer(in_dim: int, device) -> Tuple[nn.Module, SetTransformerConfig]:
    cfg = SetTransformerConfig(in_dim=in_dim)
    return SetTransformerPrototype(cfg).to(device), cfg


def _build_attn_pool(in_dim: int, device) -> Tuple[nn.Module, AttnPoolConfig]:
    cfg = AttnPoolConfig(in_dim=in_dim)
    return AttnPoolPrototype(cfg).to(device), cfg


def _build_trivial(kind: str):
    def _builder(in_dim: int, device) -> Tuple[nn.Module, TrivialPrototypeConfig]:
        cfg = TrivialPrototypeConfig(kind=kind)
        return TrivialPrototype(cfg).to(device), cfg
    return _builder


def _build_shared_mlp(in_dim: int, device) -> Tuple[nn.Module, SharedMlpConfig]:
    cfg = SharedMlpConfig(in_dim=in_dim)
    return SharedMlpPrototype(cfg).to(device), cfg


def _build_collapse_off(in_dim: int, device) -> Tuple[nn.Module, None]:
    '''Collapse switched off -- support leaves this stage exactly as G left
    it, per rung, `[K_r,D]`. `PassthroughCollapse.COLLAPSES = False` (on
    the class itself, `PrototypeGenerators.py`) is how `episode_forward`
    tells this apart from every other `COLLAPSE_CHOICES` entry, which all
    default to `COLLAPSES = True` via `getattr`.
    '''
    return PassthroughCollapse().to(device), None


#: `cli/train.py`'s `--collapse` (renamed from `--generator` 2026-09-22 --
#: see this module's own docstring). `median`: dropped from the
#: comparison surface, not from `PrototypeGenerators.TrivialPrototype`/
#: `TRIVIAL_KINDS` itself -- the class still supports it, just not
#: reachable via `--collapse` -- the arm count was cut to what can
#: actually be run and compared.
COLLAPSE_CHOICES: Dict[str, Callable[[int, object], Tuple[nn.Module, object]]] = {
    'set_transformer': _build_set_transformer,   # main line
    'mean': _build_trivial('mean'),               # zero-parameter floor
    'shared_mlp': _build_shared_mlp,               # Deep Sets, no attention
    'attn_pool': _build_attn_pool,                 # AttentionPoolHead, no cross-member attn first
    'off': _build_collapse_off,                    # stays per-member -- pairs with --routing-head cosine_logsumexp
}


# ══════════════════════════════════════════════════════════════════════════
#  Stage 2, G sub-stage (support-context)
# ══════════════════════════════════════════════════════════════════════════

def _build_context_identity_support(in_dim: int, device) -> Tuple[nn.Module, None]:
    return nn.Identity().to(device), None


def _build_context_bilstm(in_dim: int, device) -> Tuple[nn.Module, BiLstmContextConfig]:
    cfg = BiLstmContextConfig(in_dim=in_dim)
    return BiLstmSupportContext(cfg).to(device), cfg


#: Stage 2, G sub-stage: support set -> support set, same shape, members
#: informed by each other or not. `cli/train.py`'s `--support-context`.
SUPPORT_CONTEXT_CHOICES: Dict[str, Callable[[int, object], Tuple[nn.Module, object]]] = {
    'identity': _build_context_identity_support,   # off -- default
    'bilstm': _build_context_bilstm,                 # Vinyals et al.'s own g(x_i,S)
}


# ══════════════════════════════════════════════════════════════════════════
#  Stage 2, F sub-stage (query-context)
# ══════════════════════════════════════════════════════════════════════════

def _build_context_identity_query(in_dim: int, device) -> Tuple[nn.Module, None]:
    return IdentityQueryContext().to(device), None


def _build_context_attnlstm(in_dim: int, device) -> Tuple[nn.Module, AttnLstmContextConfig]:
    cfg = AttnLstmContextConfig(in_dim=in_dim)
    return AttnLstmQueryContext(cfg).to(device), cfg


#: Stage 2, F sub-stage: query, informed by reading the (G-contextualised)
#: support set K times, or not. `cli/train.py`'s `--query-context`.
QUERY_CONTEXT_CHOICES: Dict[str, Callable[[int, object], Tuple[nn.Module, object]]] = {
    'identity': _build_context_identity_query,      # off -- default
    'attnlstm': _build_context_attnlstm,              # Vinyals et al.'s own f(x-hat,S)
}


# ══════════════════════════════════════════════════════════════════════════
#  Stage 3
# ══════════════════════════════════════════════════════════════════════════

def _build_cosine_tau(in_dim: int, device) -> nn.Module:
    return CosineTauHead().to(device)


def _build_cosine_logsumexp(in_dim: int, device) -> nn.Module:
    '''Pairs with `--collapse off` -- see `PrototypeRoutingHeads.
    CosineLogSumExpHead`'s own docstring for the logsumexp identity that makes
    it degenerate exactly to `cosine_tau` when every rung has one member,
    and mathematically equal to Vinyals et al.'s own global-softmax
    classification rule otherwise.
    '''
    return CosineLogSumExpHead().to(device)


def _build_attn_score(in_dim: int, device) -> nn.Module:
    '''Pairs with ANY `--collapse` choice, including `off` -- see
    `PrototypeRoutingHeads.AttnScoreHead`'s own docstring for why it is
    NOT registered with a `NEEDS_RAW_SUPPORT` flag the way `cosine_
    logsumexp` is: its own `forward` accepts either a collapsed `[K,D]`
    stack or the raw per-rung `{rung:[K_r,D]}` dict.
    '''
    return AttnScoreHead(in_dim=in_dim).to(device)


#: Stage 3: query vs prototypes (or raw per-rung support) -> logits.
#: `cli/train.py`'s `--routing-head`.
ROUTING_HEAD_CHOICES: Dict[str, Callable[[int, object], nn.Module]] = {
    'cosine_tau': _build_cosine_tau,       # main line -- pairs with a collapsed prototype stack
    'cosine_logsumexp': _build_cosine_logsumexp,      # pairs with --collapse off's un-collapsed support
    'attn_score': _build_attn_score,        # learned multi-head Q/K -- pairs with EITHER shape
}
