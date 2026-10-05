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

import numpy as np
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
    w = weights.to(probs.device, probs.dtype).clamp_min(0)
    # Every patch weighted out (a FoV of blank glass, or patches that all
    # disagree) leaves nothing to average; argmax of a zero vector would
    # silently answer class 0. Fall back to equal weights, and say so.
    all_zero = bool(w.sum() <= 0)
    if all_zero:
        w = torch.ones_like(w)
    weighted = (probs * w.unsqueeze(1)).sum(dim=0) / w.sum()  # [C]
    c = int(weighted.argmax())
    ess = float((w.sum() ** 2) / (w ** 2).sum().clamp_min(1e-12))
    return c, dict(confidence=float(weighted[c]), effective_sample_size=ess,
                   all_zero_weights=float(all_zero))


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


# ── quality signals for `quality_weighted` (FoV_Vote.md #6) ──────────────────
#
# `quality_weighted` consumes a weight per patch and has no opinion on where
# it came from; these are the two sources measured so far. Each takes the
# patches AND their probabilities, so one call shape serves both.

def tissue_weights(probs: torch.Tensor, patches: Sequence) -> torch.Tensor:
    '''Share of each patch's pixels that are tissue, by `TissueSegFunc.
    mask_hsv` -- the `hsv` mask recipe's own per-pixel rule, so no threshold
    is invented here. A patch of blank glass weighs 0.

    The doc's warning applies: a coarse-rung patch spans more slide and
    meets more background, so this weight can correlate with scale.
    `diagnose` reports corr(w, expected log-rung) to show whether it does.'''
    from TissueSegFunc import mask_hsv                       # noqa: PLC0415
    w = [float(mask_hsv(np.ascontiguousarray(p)).mean()) for p in patches]
    return torch.tensor(w, dtype=probs.dtype, device=probs.device)


def agreement_weights(probs: torch.Tensor, patches=None) -> torch.Tensor:
    '''Share of the OTHER patches whose argmax equals this patch's: 1 when
    every other patch picked the same class, 0 when none did. Leave-one-out
    so a patch does not vote for itself.

    Not an independent quality measure: it is computed from the same
    probabilities it then weights, so it leans the mean toward the plurality
    -- between mean_probability and hard_majority, not a cleaner signal.'''
    m = probs.shape[0]
    if m < 2:
        return torch.ones(m, dtype=probs.dtype, device=probs.device)
    z = probs.argmax(dim=1)
    counts = torch.bincount(z, minlength=probs.shape[1])
    return ((counts[z] - 1).to(probs.dtype) / (m - 1))


QUALITY_SIGNALS: Dict[str, Callable[..., torch.Tensor]] = {
    'tissue': tissue_weights,
    'agree': agreement_weights,
}


# ── what each rule's danger looks like on one FoV (FoV_Vote.md, 危險分布) ──
#
# RAW quantities only. Every flag in the doc compares one of these to a
# threshold, and the doc requires the thresholds to be fixed on validation
# before test is looked at -- so the comparison happens in
# analyze_stage1_metrics.py against a thresholds file fitted on val, never
# here.

def _margins(probs: torch.Tensor) -> torch.Tensor:
    top = probs.topk(min(2, probs.shape[1]), dim=1).values
    return top[:, 0] - top[:, 1] if top.shape[1] > 1 else top[:, 0]


def _median_or_nan(x: torch.Tensor) -> float:
    return float(x.median()) if x.numel() else float('nan')


def diagnose(name: str, probs: torch.Tensor, c: int, *,
             rungs: Optional[Sequence[float]] = None,
             weights: Optional[torch.Tensor] = None) -> Dict[str, float]:
    '''The quantities FoV_Vote.md's danger section names for rule `name`,
    which chose class `c` on this FoV. Two are common to every rule --
    `risk_winner_support` (share of patches whose argmax is `c`) and
    `risk_winner_prob` (mean probability on `c`) -- because the risk-coverage
    curve needs one confidence every rule has.'''
    p = probs.detach().float().cpu()
    m, n_cls = p.shape
    z = p.argmax(dim=1)
    counts = torch.bincount(z, minlength=n_cls)
    out = dict(risk_winner_support=float(counts[c]) / m,
               risk_winner_prob=float(p[:, c].mean()))
    log_r = (torch.log2(torch.tensor([float(r) for r in rungs]))
             if rungs is not None else None)

    if name == 'mean_probability':
        # #1: the mean winner is not the argmax mode, and fewer than half
        # the patches back it -- a few confident patches decided
        out['risk_mean_not_mode'] = float(c != int(counts.argmax()))
    elif name == 'hard_majority':
        # #2: weak majority, strong dissent; and plain ties
        margin = _margins(p)
        out['risk_margin_majority_p50'] = _median_or_nan(margin[z == c])
        out['risk_margin_dissent_p50'] = _median_or_nan(margin[z != c])
        top2 = counts.topk(min(2, n_cls)).values
        out['risk_vote_tie'] = float(top2.shape[0] > 1 and top2[0] == top2[1])
    elif name == 'patch_class_median':
        # #3: the median patch is unsure, or the two middle values split
        out['risk_top1_p50'] = float(p.max(dim=1).values.median())
        zs = torch.sort(z).values
        out['risk_median_split'] = (float(abs(int(zs[m // 2]) - int(zs[m // 2 - 1])))
                                    if m % 2 == 0 else 0.0)
    elif name == 'median_log_rung' and log_r is not None:
        # #4: the snapped class has little probability and no patch picked it
        e = (p * log_r).sum(dim=1)
        out['risk_support_count'] = float(counts[c])
        out['risk_snap_gap'] = abs(float(e.median()) - float(log_r[c]))
        q = torch.quantile(e, torch.tensor([0.25, 0.75]))
        out['risk_e_iqr'] = float(q[1] - q[0])
    elif name == 'sum_log_probability':
        # #5: one patch vetoes -- the winner's lowest patch probability, and
        # whether leaving any single patch out changes the answer
        logp = torch.log(p.clamp_min(1e-12))
        s = logp.sum(dim=0)
        out['risk_min_p_winner'] = float(p[:, c].min())
        out['risk_loo_unstable'] = float(any(int((s - logp[i]).argmax()) != c
                                             for i in range(m)) if m > 1 else 0.0)
    elif name == 'quality_weighted' and weights is not None:
        # #6: the weight follows scale, or collapses onto few patches; and
        # whether weighting changed the answer at all
        w = weights.detach().float().cpu()
        out['risk_ess_frac'] = float((w.sum() ** 2) / (w ** 2).sum().clamp_min(1e-12)) / m
        out['risk_flip'] = float(c != int(p.mean(dim=0).argmax()))
        if log_r is not None and m > 1 and float(w.std()) > 0:
            e = (p * log_r).sum(dim=1)
            if float(e.std()) > 0:
                out['risk_corr_w_e'] = float(torch.corrcoef(torch.stack([w, e]))[0, 1])
    return out
