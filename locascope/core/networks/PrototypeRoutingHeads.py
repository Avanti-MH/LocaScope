'''`training/PrototypicalRoutingHead/spec.md`'s Stage 3: routing heads that
compare a QUERY vector against a set of PROTOTYPES. Every class here shares
one shape:

    logits = head(query, prototypes)     # query: [N, D], prototypes: [K, D] -> [N, K]

-- because what actually varies between the arms spec.md lists is the
distance/similarity function, never the shape of that computation. This is
the prototype-routing analogue of `Heads.py`'s single-tile classifiers:
`LinearHead`/`MlpHead`/`ArcFaceHead` take features ALONE because their own
"prototypes" are a fixed learned weight matrix; these take prototypes as an
explicit argument because Stage 2 (`SetTransformerPrototype`, or a simpler
arm) computes a NEW set every episode, not once at construction time.

THREE arms: `CosineTauHead` (main line), `CosineLogSumExpHead` (pairs with
`--collapse off`'s un-collapsed `{rung:[K_r,D]}` support), and `AttnScoreHead`
(learned multi-head Q/K projections, `--routing-head attn_score` --
dynamic-K-safe the same way `CosineLogSumExpHead` is, see its own docstring).
The non-learnable block (negative Euclidean, multi-prototype soft-min) and the
bias-routing arm are plan.md step 4.2 -- still not built. Any
`ROUTING_HEAD_CHOICES` entry accepts an arbitrary `[K, D]` tensor as
`prototypes`, learned or computed.
'''
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from Heads import ResidualMlpBlock


class CosineTauHead(nn.Module):
    '''Main line: L2-normalise both sides, `logit_k = tau * cos(q, p_k)`,
    `tau` a LEARNED scalar.

    Same geometry `Heads.ArcFaceHead` already uses (`s * cos(theta)`), with
    one deliberate difference: ArcFace's `s` is a fixed CONSTANT, because
    its own docstring shows learning it fights the margin term (`dL/dm > 0`
    always, so gradient descent drives a learned margin to zero without a
    counter-reward term ArcFace does not have). `tau` here has no margin to
    fight it, so nothing stops it from being learned, and there is no
    reason to guess a fixed value instead.
    '''

    NEEDS_TARGET = False

    def __init__(self, tau_init: float = 10.0):
        super().__init__()
        # Parameterised in LOG space so gradient descent cannot push tau
        # negative -- a negative temperature would flip which class wins.
        # Guards the PARAMETERISATION rather than clamping the value after
        # the fact, same reasoning `HeadConfig`'s dtype guard applies to a
        # different failure mode.
        self.log_tau = nn.Parameter(torch.tensor(float(tau_init)).log())

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp()

    def forward(self, query: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        '''`query`: `[N, D]`. `prototypes`: `[K, D]`, K = number of rungs
        Stage 2 produced a prototype for (normally all 6, but a caller
        scoring a partial episode is not this class's concern). Returns
        `[N, K]` logits.
        '''
        q = F.normalize(query.float(), dim=-1)
        p = F.normalize(prototypes.float(), dim=-1)
        return self.tau * (q @ p.t())


class CosineLogSumExpHead(nn.Module):
    '''Pairs with `--collapse off` -- compares a query against the RAW,
    un-collapsed support members directly, no prototype ever gets built.
    `logit_r = logsumexp_{i in rung r}( tau * cos(query, member_i) )`.

    Mathematically IDENTICAL to the literal Matching Networks
    classification rule (Vinyals et al. 2016) -- "one softmax attention
    distribution over every support member across every rung at once, a
    rung's probability is the sum of its own members' attention weights"
    -- NOT an approximation of it:

        softmax_r(logit)  =  exp(logit_r) / sum_r' exp(logit_r')
                           =  sum_{i in r} exp(tau*sim_i) / sum_j exp(tau*sim_j)
                           =  sum_{i in r} a(query, member_i)             (a = the paper's own kernel)
                           =  P(rung = r)                                  (the paper's own prediction)

    `sum_j` above runs over EVERY member of EVERY rung -- the same global
    normalisation the paper uses, recovered here because `Losses.
    compute_loss` (unchanged) applies its own softmax ACROSS the `num_
    rungs` logits this returns. Computing it this way instead of literally
    doing `softmax` then a per-rung `sum` keeps this head's output the same
    KIND of number `CosineTauHead` already returns (unnormalised logits),
    so both heads share one loss code path with no branching on whether a
    given head "already normalised" -- and `torch.logsumexp` is the
    numerically stable way to do it (no separate softmax-then-log step,
    which is where precision would otherwise be lost for a sharply peaked
    distribution).

    DEGENERATES to `CosineTauHead` exactly when every rung has ONE member:
    `logsumexp` of a single value is that value itself, so `logit_r =
    tau*cos(query, prototype)`, identical to `CosineTauHead.forward`. Same
    "mode (a) is a strict superset" property `AttnScoreHead` below has,
    for the same reason.

    HOW THIS DIFFERS FROM Vinyals et al.'s OWN FORMULA: the paper's own
    `a`/`g`/`f` can ALSO contextualise support/query via biLSTM/attLSTM
    before this comparison ever runs (`ContextEncoders.py`'s G/F) -- this
    class is only the FINAL comparison step, agnostic to whether G/F ran.
    This class ALSO runs a SEPARATE attention call PER RUNG (never
    concatenated across rungs first, see `forward`'s own docstring) rather
    than one global softmax computed directly -- mathematically equivalent
    (proved above), just computed in a shape that reuses `episode_
    forward`'s existing per-rung structure.
    '''

    NEEDS_RAW_SUPPORT = True

    def __init__(self, tau_init: float = 10.0):
        super().__init__()
        self.log_tau = nn.Parameter(torch.tensor(float(tau_init)).log())

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp()

    def forward(self, query: torch.Tensor, support_by_rung) -> torch.Tensor:
        q = F.normalize(query.float(), dim=-1)                # [N, D]
        logits = []
        for members in support_by_rung.values():
            m = F.normalize(members.float(), dim=-1)             # [K_r, D]
            sim = q @ m.t()                                        # [N, K_r]
            logits.append(torch.logsumexp(self.tau * sim, dim=-1))   # [N]
        return torch.stack(logits, dim=1)                          # [N, num_rungs]


class AttnScoreHead(nn.Module):
    '''Learned multi-head Q/K projections + per-rung `logsumexp` pooling --
    a strictly more expressive generalisation of `CosineLogSumExpHead`
    (which is this class with `MLPq`/`MLPk` fixed to the identity and
    `n_head=1`), asked for specifically because `cosine_tau`/`cosine_
    logsumexp` never learn how to WEIGH or ROTATE the embedding space
    before comparing -- cosine similarity treats every dimension as
    equally informative, always.

    NO fixed-size classifier anywhere in this class -- every tensor whose
    size depends on K (how many rungs THIS episode has) is produced by
    evaluating the SAME per-pair computation once per prototype/member,
    never by a layer whose own weight shape bakes in a specific K. A
    classifier layer has a FIXED output width, which breaks the moment an
    episode is N-way for a different N, exactly the failure mode `cosine_tau`/`cosine_logsumexp` were built to avoid by using a
    K-agnostic dot product/logsumexp instead of a class-count-shaped
    layer. `MLPq`/`MLPk` below are safe for the same reason `Context
    Encoders.BiLstmSupportContext`'s biLSTM is: applied per-member/
    per-query, `[D]->[D]`, never `[K,...]->[K,...]`.

        query [N,D] --MLPq--> Q [N,D] --split H heads--> [N,H,d_h]
        member [D]  --MLPk--> K [D]   --split H heads--> [H,d_h]
        score[n,m,h] = Q[n,h] . K[m,h] / sqrt(d_h)
        score[n,m]   = sum_h score[n,m,h]
        logit[n,rung] = logsumexp_{m in rung}( tau * score[n,m] )

    `MLPq`/`MLPk` are `Heads.ResidualMlpBlock` -- the SAME class
    `PrototypeGenerators.SharedMlpPrototype` builds its own per-member
    transform from, not a parallel reimplementation of the same six lines.

    Pairs with EITHER a collapsed `[K,D]` prototype stack (mode a --
    degenerates exactly to a learned-projection version of `CosineTauHead`
    when every rung has one member) OR the raw per-rung `{rung:[K_r,D]}`
    Collapse=`off` leaves behind (mode b, same as `CosineLogSumExpHead`) --
    `NEEDS_RAW_SUPPORT` is NOT set on this class for that reason: unlike
    `CosineLogSumExpHead`, which `main()`'s startup check requires pairing
    with `--collapse off` specifically, this head's own `forward` accepts
    either shape (`support`'s own `isinstance` check, same normalisation
    `CosineLogSumExpHead`'s predecessor used), so it is compatible with
    ANY `--collapse` choice.
    '''

    #: `cli/train.py`'s `main()` reads this to SKIP the strict Collapse/
    #: head shape check `cosine_tau`/`cosine_logsumexp` are both held to
    #: (see `episode_forward`'s own docstring) -- this head's `forward`
    #: genuinely accepts either shape, so refusing a pairing here would be
    #: wrong, not just over-cautious.
    ACCEPTS_EITHER = True

    def __init__(self, in_dim: int = 1536, n_head: int = 8,
                mlp_depth: int = 2, tau_init: float = 10.0):
        super().__init__()
        if in_dim % n_head != 0:
            raise ValueError(f'in_dim={in_dim} must divide evenly by '
                             f'n_head={n_head}')
        self.in_dim = in_dim
        self.n_head = n_head
        self.mlpq = ResidualMlpBlock(in_dim, depth=mlp_depth)
        self.mlpk = ResidualMlpBlock(in_dim, depth=mlp_depth)
        self.log_tau = nn.Parameter(torch.tensor(float(tau_init)).log())

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp()

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [..., D] -> [..., H, d_h]
        return x.view(*x.shape[:-1], self.n_head, self.in_dim // self.n_head)

    def forward(self, query: torch.Tensor, support) -> torch.Tensor:
        by_rung = ({i: support[i:i + 1] for i in range(support.shape[0])}
                  if isinstance(support, torch.Tensor) else support)
        d_h = self.in_dim // self.n_head
        q = self._split_heads(self.mlpq(query))                # [N, H, d_h]
        logits = []
        for members in by_rung.values():
            k = self._split_heads(self.mlpk(members))             # [K_r, H, d_h]
            # score[n,m,h] = q[n,h,:] . k[m,h,:] / sqrt(d_h)
            score = torch.einsum('nhd,mhd->nmh', q, k) / (d_h ** 0.5)
            score = score.sum(dim=-1)                                # [N, K_r]
            logits.append(torch.logsumexp(self.tau * score, dim=-1))   # [N]
        return torch.stack(logits, dim=1)                          # [N, num_rungs]
