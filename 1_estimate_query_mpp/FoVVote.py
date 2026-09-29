'''FoV patch-vote aggregation -- one FoV's `M` per-patch class-probability
vectors -> one FoV-level class prediction. `FoV_Vote.md` (this directory)
is the full comparison/derivation; this file is JUST the six formulas, no
model code, no I/O -- `ClassifierEstMpp.estimate()` and
`PrototypicalEstMpp.estimate()` both call `vote(...)` on their own
per-patch `probs` tensor rather than each carrying its own copy, which is
the whole reason this file exists: before it, `ClassifierEstMpp` had ONE
of these six inlined directly in `estimate()`, and a second estimator
needing the same aggregation would have meant a second inlined copy able
to drift from the first.

    predicted_class, extra = vote('mean_probability', probs)
    predicted_class, extra = vote('median_log_rung', probs, rungs=rungs)
    predicted_class, extra = vote('quality_weighted', probs, weights=w)

Every function takes `probs: [M, num_classes]` (M patches' own softmax
distributions, rows summing to 1) and returns `(predicted_class: int,
extra: Dict[str, float])`. `extra` is NOT the same shape across the six --
each carries whatever ITS OWN section of `FoV_Vote.md` calls out as worth
keeping (confidence, a median split flag, an effective sample size, ...),
because they are not measuring the same kind of thing (`FoV_Vote.md`'s own
closing line: "平均 winning probability 本身不是 FoV 正確率" -- forcing
one common `extra` shape across all six would either drop information or
paper over that difference with a field that means five different things
depending which function filled it in).
'''
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import torch


def mean_probability(probs: torch.Tensor, **_) -> Tuple[int, Dict[str, float]]:
    '''`FoV_Vote.md` #1 -- currently `ClassifierEstMpp`'s own method.
    Per-class arithmetic mean of every patch's softmax, argmax the mean.
    Keeps confidence (a single unconfident patch is outvoted by several
    confident ones), at the cost of one badly-uncalibrated confident patch
    being able to drag the mean past several weak-but-correct ones (see
    the doc's own failure case).
    '''
    mean = probs.mean(dim=0)                              # [C]
    c = int(mean.argmax())
    return c, dict(confidence=float(mean[c]))


def hard_majority(probs: torch.Tensor, **_) -> Tuple[int, Dict[str, float]]:
    '''`FoV_Vote.md` #2. Each patch's own argmax casts ONE vote; confidence
    is discarded entirely, which is the point -- an uncalibrated 0.99 on
    one patch cannot outweigh four patches that each only lean 0.20 toward
    the right answer, but four patches at a genuine but weak 0.20 CAN
    outvote two patches at a genuine, strong 0.94 (the doc's own failure
    case) -- confidence being gone cuts both ways.
    '''
    z = probs.argmax(dim=1)                                # [M]
    counts = torch.bincount(z, minlength=probs.shape[1])
    c = int(counts.argmax())
    return c, dict(vote_share=float(counts[c]) / probs.shape[0])


def patch_class_median(probs: torch.Tensor, tie_rule: str = 'lower',
                       **_) -> Tuple[int, Dict[str, float]]:
    '''`FoV_Vote.md` #3, "hard ordinal median". Each patch's own argmax
    class INDEX (not its probability), sorted, take the median -- uses the
    rungs' ordinal ordering (a class far away only counts as one sample no
    matter how confident), unlike hard majority which has no notion of
    "close" vs "far" wrong.

    `tie_rule`: with an even `M` there are two middle values
    (`lower`/`upper`); the doc's own "Danger" section for this method
    flags reporting BOTH separately as part of evaluating it, so this is a
    parameter, not a hardcoded choice -- default `'lower'` for
    determinism, not because the doc prefers it.
    '''
    z = probs.argmax(dim=1)                                # [M]
    sorted_z, _ = torch.sort(z)
    m = sorted_z.shape[0]
    if m % 2 == 1:
        c = int(sorted_z[m // 2])
        split = 0.0
    else:
        lower = int(sorted_z[m // 2 - 1])
        upper = int(sorted_z[m // 2])
        c = lower if tie_rule == 'lower' else upper
        split = float(abs(upper - lower))
    return c, dict(median_split=split)


def median_log_rung(probs: torch.Tensor, rungs: Sequence[float],
                    **_) -> Tuple[int, Dict[str, float]]:
    '''`FoV_Vote.md` #4, "soft ordinal median". Each patch's own EXPECTED
    log-rung (`sum_c p[i,c] * log2(rung_c)`, the full distribution, not
    just the argmax), then the median of that across patches, snapped to
    the nearest rung's own log2 value.

    On an exact tie between two rungs (the doc's own failure case: a
    bimodal patch distribution puts the expectation exactly between two
    rungs neither one individually supports), `argmin` on the distance
    keeps whichever rung comes FIRST in `rungs` -- a deterministic
    tie-break, not a claim that the lower rung is the right answer; the
    doc's own risk-flag for this method (`count(argmax_i = predicted) ==
    0`) is what actually catches this case, not this function's tie rule.
    '''
    log_rungs = torch.log2(torch.tensor([float(r) for r in rungs],
                                        dtype=probs.dtype, device=probs.device))
    e = (probs * log_rungs.unsqueeze(0)).sum(dim=1)          # [M]
    e_med = float(e.median())
    c = int((log_rungs - e_med).abs().argmin())
    return c, dict(expected_log_rung_median=e_med)


def sum_log_probability(probs: torch.Tensor, eps: float = 1e-12,
                        **_) -> Tuple[int, Dict[str, float]]:
    '''`FoV_Vote.md` #5, "product of evidence" in log space. Requires every
    patch to keep supporting the winning class -- one patch giving it
    near-zero probability acts as a near-veto (the doc's own failure
    case). The doc itself flags this one as a stress test, not part of the
    recommended first round: adjacent FoV patches are not independent
    samples, so the product-of-evidence independence assumption this
    formula relies on does not really hold -- kept here because the doc
    keeps it, not because it is recommended for routine use.
    '''
    s = torch.log(probs.clamp_min(eps)).sum(dim=0)           # [C]
    c = int(s.argmax())
    return c, dict(log_evidence=float(s[c]))


def quality_weighted(probs: torch.Tensor,
                     weights: Optional[torch.Tensor] = None,
                     **_) -> Tuple[int, Dict[str, float]]:
    '''`FoV_Vote.md` #6. `weights` (`[M]`, each in `[0,1]`, defaulting to
    all-ones -- i.e. plain mean probability with `weights=None`) down-
    weights or drops (`w_i=0`, "trimmed") patches a caller's own quality
    signal (tissue amount, blur, artefact, prediction agreement -- NOT
    computed here, this function only consumes the number) flags. The
    doc's own failure case is exactly the risk of building that quality
    signal FROM something scale-correlated (smooth coarse tissue mistaken
    for blur): this function has no opinion on how `weights` was made, it
    only reports `effective_sample_size` (`(sum w)^2 / sum(w^2)`) so a
    caller can see when the weighting collapsed onto too few patches.
    '''
    if weights is None:
        weights = torch.ones(probs.shape[0], dtype=probs.dtype, device=probs.device)
    w = weights.clamp_min(0)
    total_w = w.sum().clamp_min(1e-12)
    weighted = (probs * w.unsqueeze(1)).sum(dim=0) / total_w  # [C]
    c = int(weighted.argmax())
    ess = float((w.sum() ** 2) / (w ** 2).sum().clamp_min(1e-12))
    return c, dict(confidence=float(weighted[c]), effective_sample_size=ess)


#: `--vote`'s registry, same idiom `Runtime.GENERATOR_CHOICES`/
#: `ROUTING_HEAD_CHOICES` and `Heads._CLASSIFIER_REGISTRY` already use --
#: a config names one of these by string. `FoV_Vote.md`'s own "建議的第一
#: 輪比較" names five of the six for routine use; `sum_log_probability`
#: stays registered (a caller can still ask for it as a stress test) but
#: is not the default for either estimator.
VOTE_CHOICES: Dict[str, Callable[..., Tuple[int, Dict[str, float]]]] = {
    'mean_probability': mean_probability,
    'hard_majority': hard_majority,
    'patch_class_median': patch_class_median,
    'median_log_rung': median_log_rung,
    'sum_log_probability': sum_log_probability,
    'quality_weighted': quality_weighted,
}


def vote(name: str, probs: torch.Tensor, *, rungs: Optional[Sequence[float]] = None,
        weights: Optional[torch.Tensor] = None,
        tie_rule: str = 'lower') -> Tuple[int, Dict[str, float]]:
    '''Dispatch to one of `VOTE_CHOICES` by name. `rungs`/`weights`/
    `tie_rule` are passed through to whichever function actually reads
    them (`median_log_rung` needs `rungs`, `quality_weighted` needs
    `weights`, `patch_class_median` needs `tie_rule`); every other
    function's own `**_` ignores whatever it does not need, so a caller
    can pass all three unconditionally without knowing which vote it
    picked cares about which.
    '''
    if name not in VOTE_CHOICES:
        raise ValueError(f'vote: {name!r} not in {sorted(VOTE_CHOICES)}')
    if name == 'median_log_rung' and rungs is None:
        raise ValueError('vote: median_log_rung needs rungs=')
    return VOTE_CHOICES[name](probs, rungs=rungs, weights=weights, tie_rule=tie_rule)
