'''`L_bal` / `L_ord_a` / `L_ord_b` -- `MppRoutingHead/spec.md`'s "Ordinal-aware
loss" formulas, adopted here from the start (spec.md's own words: "adopted
... verbatim, from the start"). One real difference from that project: `N`
(how many rungs one call's logits/target range over) VARIES per episode --
spec.md's N-way episode design deliberately excludes the full 6 during
training -- so every function here takes the EPISODE's own rung list, never
`DsLadder.DEFAULT_RUNGS`' fixed 6-tuple.

    logits: [N_query, K]            K = len(episode_rungs) for THIS episode
    target: [N_query]               LOCAL index into episode_rungs, 0..K-1
    episode_rungs: e.g. (1.0, 4.0, 16.0) for a drawn 3-way subset

    loss = compute_loss(logits, target, episode_rungs, 'ord_a', ...)

`w_c` (the class-imbalance weight) is recomputed PER EPISODE from this
episode's own query batch, not read from a table built once over the whole
training manifest the way `MppRoutingHead/Datasets.class_weights` does --
there is no single fixed rung distribution to build that table from once N
varies episode to episode. Cheap (one bincount over a small batch) and
adapts automatically to whichever subset was drawn AND however short the
actual yield came in for any one rung in it. Same sklearn 'balanced' formula
either way: `w_c = N_query / (K * n_c)`.
'''
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def episode_class_weights(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    '''`w_c = N / (K * n_c)` from THIS episode's own `target` batch -- the
    same formula `MppRoutingHead.Datasets.class_weights` uses, scoped to one
    episode instead of the whole training manifest. `num_classes` is `K`
    (this episode's own rung count), not a fixed 6.

    A class with zero examples in this batch gets weight 0, not `inf` --
    nothing to reweight if `target` never names it, and it cannot appear as
    a target this call anyway.
    '''
    counts = torch.bincount(target, minlength=num_classes).double()
    total = float(counts.sum())
    if total <= 0:
        raise ValueError('episode_class_weights: target is empty')
    weights = torch.where(counts > 0, total / (num_classes * counts.clamp_min(1.0)),
                          torch.zeros_like(counts))
    return weights.float().to(target.device)


def compute_loss(logits: torch.Tensor, target: torch.Tensor,
                 episode_rungs: Sequence[float], loss_kind: str,
                 ordinal_weight: float = 1.0,
                 ordinal_sigma: float = 1.0) -> torch.Tensor:
    '''`loss_kind`:

        'bal'   = -w_y log p_y                              (weighted CE)
        'ord_a' = L_bal + lambda * sum_i w_i (E_c[ell_c] - ell_y)^2 / sum_i w_i
                                                            (+ regression term, class-weighted)
        'ord_b' = w_y * (-sum_c q_c(y) log p_c)              (soft target, REPLACES
                                                              the one-hot CE target
                                                              rather than adding to it
                                                              -- see MppRoutingHead/
                                                              cli/train.py's own
                                                              _compute_loss for why)

    `ell_c = log2(episode_rungs[c])` -- geometric spacing, matching
    `analyze_stage1_metrics.nearest_rung`'s own reasoning throughout this
    repo. `ordinal_weight`/`ordinal_sigma` are unvalidated starting values
    (spec.md's own status for both).
    '''
    K = len(episode_rungs)
    weights = episode_class_weights(target, K)

    if loss_kind == 'bal':
        return F.cross_entropy(logits, target, weight=weights)

    log2_rungs = torch.log2(
        torch.tensor(episode_rungs, dtype=torch.float32, device=logits.device))

    if loss_kind == 'ord_a':
        l_bal = F.cross_entropy(logits, target, weight=weights)
        probs = F.softmax(logits.float(), dim=-1)
        expected_log_rung = (probs * log2_rungs).sum(dim=-1)
        true_log_rung = log2_rungs[target]
        # Weighted like L_bal: an unweighted mean would let the
        # rungs with the most examples decide the ordinal term alone.
        w = weights[target]
        l_ord = (w * (expected_log_rung - true_log_rung).pow(2)).sum() / w.sum()
        return l_bal + ordinal_weight * l_ord

    if loss_kind == 'ord_b':
        dist2 = (log2_rungs.unsqueeze(0) - log2_rungs[target].unsqueeze(1)).pow(2)
        soft_target = F.softmax(-dist2 / (2 * ordinal_sigma ** 2), dim=-1)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        per_sample = -(soft_target * log_probs).sum(dim=-1)
        w = weights[target]
        return (w * per_sample).sum() / w.sum()

    raise ValueError(f'unknown loss_kind {loss_kind!r}')
