'''The losses of `proposal.md` §9.2, for one FoV's candidates.

    listwise     multi-positive: -log( sum_{c in P} e^{s_c} / sum_{c in P u N} e^{s_c} )          both architectures
                 -- with one class of answers, the one-hot cross-entropy of the softmax over the candidates
    soft target  (`--w-geo` above 0) the windows of the ignore zone at the right rotation are not left out but are
                 near windows, in the softmax's denominator, with a target that falls with their distance d (in tiles)
                 from the truth's centre. `--soft-target` says how:
        gaussian     each near window gets t = w * alpha * exp(-d^2 / (2 sigma^2)) against the answers' 1 (the
                     targets are not a distribution):
                         L = ( L_list - sum_near t_c * log softmax_c ) / ( 1 + sum_near t_c )
                     (the form RerankHead's WinEmbed trains with)
        exponential  the target is a distribution, proportional to exp(-d / sigma) over the windows of the softmax at the
                     right rotation (the answers, the near windows, the other windows), and the loss is
                         L = L_list + w * CE( target, softmax )
    correspondence, cycle    `GloRet.correspondence_loss` and `cycle_loss` on the positive pairs   LogLoRet only

With the soft target off an ignored candidate (`Supply.mark`) is in none of them: it is not in the softmax. With it on,
only the ignored candidates that are not near (at a wrong rotation) are left out. A candidate at a rotation other than
the photo's is in the softmax as a negative, or is ignored (`--wrong-rotation`), and never gets a soft target.
'''
from __future__ import annotations

from typing import Dict

import torch

SOFT_TARGETS = ('gaussian', 'exponential')


def masked_logsumexp(logits: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """`log sum exp` of the entries of `logits` `[K]` where `keep`; `-inf` when none."""
    return torch.logsumexp(logits.masked_fill(~keep, float('-inf')), dim=0)


def listwise_loss(logits: torch.Tensor, pos: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """The multi-positive listwise loss over `logits` `[K]`; `pos` the positives, `valid` the candidates in the
    softmax (positives and negatives: not the ignored)."""
    logits = logits.float()
    return masked_logsumexp(logits, valid) - masked_logsumexp(logits, pos & valid)


def near_targets(dist: torch.Tensor, alpha: float, sigma: float) -> torch.Tensor:
    """`[K]` the gaussian soft target `alpha * exp(-d^2 / (2 sigma^2))` of a near window at distance `dist` tiles."""
    return alpha * torch.exp(-dist.float().square() / (2.0 * sigma ** 2))


def exponential_target(dist: torch.Tensor, right: torch.Tensor, valid: torch.Tensor, sigma: float) -> torch.Tensor:
    """`[K]` the target distribution `exp(-d / sigma)` over the windows at the right rotation that are in the softmax,
    0 elsewhere, summing to 1."""
    weight: torch.Tensor = torch.exp(-dist.float() / sigma) * (right & valid).float()
    return weight / weight.sum().clamp(min=1e-12)


def fov_loss(logits: torch.Tensor, pos: torch.Tensor, ignore: torch.Tensor, right: torch.Tensor, dist: torch.Tensor,
             sigma: float, w_geo: float, alpha: float = 0.3, kind: str = 'gaussian') -> Dict[str, torch.Tensor]:
    """`{"list", "geo", "loss"}` of one FoV: `logits` `[K]` of all its candidates and their marks. With `w_geo` 0 the
    loss is the list loss alone (the one-hot cross-entropy) and `geo` is 0. Above 0 the ignored windows at the right
    rotation are near windows with soft targets of `kind` (see the module docstring); `geo` is the cross-entropy of
    the near windows against their targets (gaussian: per unit of target), `loss` the combined loss."""
    if kind not in SOFT_TARGETS:
        raise ValueError(f'soft target {kind!r}: one of {SOFT_TARGETS}')
    near: torch.Tensor = (ignore & right & ~pos) if w_geo else torch.zeros_like(pos)
    valid: torch.Tensor = ~ignore | near
    listwise: torch.Tensor = listwise_loss(logits, pos, valid)
    if not w_geo or not bool(near.any()):
        return {'list': listwise, 'geo': torch.zeros((), device=logits.device), 'loss': listwise}
    z: torch.Tensor = logits.float()
    log_all: torch.Tensor = masked_logsumexp(z, valid)
    if kind == 'gaussian':
        t: torch.Tensor = w_geo * near_targets(dist[near], alpha, sigma)
        soft: torch.Tensor = -(t * (z[near] - log_all)).sum()
        return {'list': listwise, 'geo': soft / t.sum().clamp(min=1e-12), 'loss': (listwise + soft) / (1.0 + t.sum())}
    target: torch.Tensor = exponential_target(dist, right, valid, sigma)
    logp: torch.Tensor = (z - log_all).masked_fill(target == 0, 0.0)
    cross: torch.Tensor = -(target * logp).sum()
    return {'list': listwise, 'geo': cross, 'loss': listwise + w_geo * cross}


__all__ = ['SOFT_TARGETS', 'exponential_target', 'fov_loss', 'listwise_loss', 'masked_logsumexp', 'near_targets']
