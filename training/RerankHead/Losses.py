'''The losses of the second phase. All take one query's candidates (a "list"),
logits `[K]` and boolean masks `[K]` (the label table in `Labels.py`).

    listwise_loss     -log of the share of the softmax the answers get, over the answers, the
                      negatives and the near windows; each near window also has the share its soft
                      target says (the truth beats it by more the farther it is)
    topk_hinge_loss   a positive must beat the k-th best negative by a margin:
                      recall@k itself, as a hinge
    bce_loss          each positive/negative alone, positives weighted up
    rot_loss          among the windows that cover the truth, which are at the
                      right rotation
    loc_loss          where the truth is from this window (Huber), for every
                      window with something to learn
    pool_softmax_loss the same listwise loss over a whole pool of windows -- the
                      embedding's loss, whose pool is every window of the slide
                      at every rotation
'''
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def listwise_loss(logit: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor, temperature: float = 1.0,
                  near: Optional[torch.Tensor] = None, near_target: Optional[torch.Tensor] = None) -> torch.Tensor:
    """`-log( sum_pos exp(l/T) / sum_all exp(l/T) )` over the positives, negatives and near windows; 0 with no
    positive. With near windows each adds `-t_i log( exp(l_i/T) / sum_all )` (`near_target` `t`, the soft target
    against the answers' 1), and the sum is divided by `1 + sum t`: the answers as a group must hold the most
    mass, each near window the share its target says. With none it is the plain multi-positive loss."""
    if not bool(pos.any()):
        return logit.sum() * 0.0
    z: torch.Tensor = logit / temperature
    use_near: bool = near is not None and near_target is not None and bool(near.any())
    keep: torch.Tensor = pos | neg | near if use_near else pos | neg
    log_all: torch.Tensor = torch.logsumexp(z[keep], dim=0)
    total: torch.Tensor = -(torch.logsumexp(z[pos], dim=0) - log_all)
    if not use_near:
        return total
    t: torch.Tensor = near_target[near]
    return (total - (t * (z[near] - log_all)).sum()) / (1.0 + t.sum())


def topk_hinge_loss(logit: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor, k: int,
                    margin: float = 1.0) -> torch.Tensor:
    """Mean over positives of `max(0, margin + (k-th largest negative) - positive)`:
    zero when every positive has fewer than `k` negatives above it (by the margin).
    With fewer than `k` negatives the lowest one is used."""
    if not bool(pos.any()) or not bool(neg.any()):
        return logit.sum() * 0.0
    negatives: torch.Tensor = logit[neg]
    kth: torch.Tensor = negatives.topk(min(k, len(negatives))).values[-1]
    return F.relu(margin + kth - logit[pos]).mean()


def bce_loss(logit: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor,
             pos_weight: float = 1.0) -> torch.Tensor:
    keep: torch.Tensor = pos | neg
    if not bool(keep.any()):
        return logit.sum() * 0.0
    target: torch.Tensor = pos[keep].float()
    weight: torch.Tensor = torch.where(pos[keep], torch.tensor(float(pos_weight), device=logit.device),
                                       torch.tensor(1.0, device=logit.device))
    return F.binary_cross_entropy_with_logits(logit[keep], target, weight=weight)


def rot_loss(rot_logit: torch.Tensor, covers: torch.Tensor, rot_ok: torch.Tensor) -> torch.Tensor:
    """BCE of "this window's rotation is the right one" over the windows that
    cover the truth (`covers`)."""
    if not bool(covers.any()):
        return rot_logit.sum() * 0.0
    return F.binary_cross_entropy_with_logits(rot_logit[covers], rot_ok[covers].float())


def loc_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not bool(mask.any()):
        return pred.sum() * 0.0
    return F.smooth_l1_loss(pred[mask], target[mask])


def pool_softmax_loss(logit: torch.Tensor, pos: torch.Tensor, keep: torch.Tensor, temperature: float,
                      near: Optional[torch.Tensor] = None, near_target: Optional[torch.Tensor] = None
                      ) -> torch.Tensor:
    """`listwise_loss` for a pool too long to hold masks for twice: `logit`, `pos`, `keep` (every window of the
    softmax) and, if given, `near` and `near_target` are flat over the pool."""
    if not bool(pos.any()):
        return logit.sum() * 0.0
    z: torch.Tensor = logit / temperature
    log_all: torch.Tensor = torch.logsumexp(z[keep], dim=0)
    total: torch.Tensor = -(torch.logsumexp(z[pos], dim=0) - log_all)
    if near is None or near_target is None or not bool(near.any()):
        return total
    t: torch.Tensor = near_target[near]
    return (total - (t * (z[near] - log_all)).sum()) / (1.0 + t.sum())


def rerank_loss(out: Dict[str, torch.Tensor], pos: torch.Tensor, neg: torch.Tensor, covers: torch.Tensor,
                rot_ok: torch.Tensor, loc: torch.Tensor, loc_mask: torch.Tensor, weights: Dict[str, float],
                temperature: float, k: int, margin: float, pos_weight: float,
                near: Optional[torch.Tensor] = None,
                near_target: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """The terms of one list and their weighted `total`. `out` is the network's
    (`logit`, and `rot`, `loc` when it has them); a term with weight 0 is skipped. The near windows are in the
    listwise term only (the hinge and the BCE see positives and negatives)."""
    terms: Dict[str, torch.Tensor] = {'list': listwise_loss(out['logit'], pos, neg, temperature, near, near_target)}
    if weights.get('topk', 0.0):
        terms['topk'] = topk_hinge_loss(out['logit'], pos, neg, k, margin)
    if weights.get('bce', 0.0):
        terms['bce'] = bce_loss(out['logit'], pos, neg, pos_weight)
    if weights.get('rot', 0.0) and 'rot' in out:
        terms['rot'] = rot_loss(out['rot'], covers, rot_ok)
    if weights.get('loc', 0.0) and 'loc' in out:
        terms['loc'] = loc_loss(out['loc'], loc, loc_mask)
    total: torch.Tensor = terms['list']
    for name, term in terms.items():
        if name != 'list':
            total = total + weights[name] * term
    terms['total'] = total
    return terms
