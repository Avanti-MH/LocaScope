#!/usr/bin/env python3
"""The ordinal losses, in BOTH places they are written -- `training/
PrototypicalRoutingHead/Losses.compute_loss` and `training/MppRoutingHead/
cli/train._compute_loss` -- named after the formula rather than a module
because what it checks is that the two agree and that both weight the
regression term (user, 2026-09-24).

    python utilities/test_modules/test_ordinal_loss.py

No data, no model: hand-built logits whose answer is known.

WHAT THIS DEFENDS
-----------------
    ord_a weighting   the regression term is class-weighted like L_bal. The
                      probe: one rare-class sample predicted far off and many
                      common ones predicted right. Weighted, the rare error
                      dominates the term; unweighted (the decoy, computed here
                      by hand) it is diluted -- the two must differ by the
                      class-weight ratio, not by rounding
    agreement         the two implementations return the same number for the
                      same inputs, for bal, ord_a and ord_b
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))

from _paths import setup_import_paths                            # noqa: E402

setup_import_paths()

import torch                                                     # noqa: E402
import torch.nn.functional as F                                  # noqa: E402

from training.PrototypicalRoutingHead.Losses import (            # noqa: E402
    compute_loss, episode_class_weights)
from training.MppRoutingHead.cli.train import _compute_loss      # noqa: E402

RUNGS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
_RESULTS = []


def check(name, fn):
    try:
        out = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   {out}' if out else ''))
    except Exception as e:                                       # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n          {type(e).__name__}: {e}')


def _probe():
    """9 samples of rung 1 predicted right, 1 of rung 32 predicted as rung 1."""
    target = torch.tensor([0] * 9 + [5])
    logits = torch.full((10, 6), -8.0)
    logits[:, 0] = 8.0            # everyone says rung 1
    return logits, target


def _regression(logits, target, weights):
    probs = F.softmax(logits, dim=-1)
    l2 = torch.log2(torch.tensor(RUNGS))
    sq = ((probs * l2).sum(-1) - l2[target]).pow(2)
    if weights is None:
        return sq.mean()
    w = weights[target]
    return (w * sq).sum() / w.sum()


def t_ord_a_regression_term_is_class_weighted_and_the_decoy_is_not():
    logits, target = _probe()
    weights = episode_class_weights(target, 6)
    got = compute_loss(logits, target, RUNGS, 'ord_a', ordinal_weight=1.0) \
        - compute_loss(logits, target, RUNGS, 'bal')
    want = _regression(logits, target, weights)
    decoy = _regression(logits, target, None)
    assert torch.allclose(got, want, atol=1e-5), (float(got), float(want))
    assert float(want) > 2 * float(decoy), (
        f'weighted {float(want):.3f} vs unweighted {float(decoy):.3f}: the '
        f'probe cannot tell the two apart')
    return (f'term {float(got):.3f} (weighted) vs {float(decoy):.3f} unweighted; '
            f'rung 32 error 25 in log2^2')


def t_both_implementations_agree():
    torch.manual_seed(0)
    logits = torch.randn(40, 6)
    target = torch.randint(0, 6, (40,))
    weights = episode_class_weights(target, 6)
    for kind in ('bal', 'ord_a', 'ord_b'):
        a = compute_loss(logits, target, RUNGS, kind, 0.7, 1.3)
        b = _compute_loss(logits, target, weights, kind, 0.7, 1.3)
        assert torch.allclose(a, b, atol=1e-5), (kind, float(a), float(b))
    return 'bal, ord_a, ord_b equal to 1e-5'


# ══════════════════════════════════════════════════════════════════════════════

_TESTS = [t for n, t in sorted(globals().items()) if n.startswith('t_')]


def main() -> int:
    for fn in _TESTS:
        check(fn.__name__[2:].replace('_', ' '), fn)
    failed = [n for n, e in _RESULTS if e is not None]
    print(f'\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed')
    if failed:
        print('failed: ' + ', '.join(failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
