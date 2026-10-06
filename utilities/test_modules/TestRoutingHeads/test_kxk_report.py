#!/usr/bin/env python3
"""Tests for `kxk_report` and `improves_everywhere` in training/PrototypicalRoutingHead/cli/train.py --
the validation accuracy the checkpoints are chosen on.

    python utilities/test_modules/TestRoutingHeads/test_kxk_report.py

No model: hand-written per-pair, per-rung accuracies whose answer is worked
out below.

WHAT THIS DEFENDS
-----------------
    the order of the means
        rung in a combo   mean over the K x K pairs
        combo             mean of its rungs
        dataset           mean of its combos
        total             mean of the datasets
    scored against the DECOY an implementation is most likely to slip into:
    one pooled mean over every number, which on these inputs is a different
    value -- so matching it would fail the test.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))

from _paths import setup_import_paths                            # noqa: E402

setup_import_paths()

from training.PrototypicalRoutingHead.cli.train import (      # noqa: E402
    improves_everywhere, kxk_report)

_RESULTS = []


def check(name, fn):
    try:
        out = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   {out}' if out else ''))
    except Exception as e:                                       # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n          {type(e).__name__}: {e}')


def _res(rungs, pairs, n_resampled=0):
    return dict(ok=True, rungs=rungs, n_resampled=n_resampled,
                acc_by_pair_rung=[dict(zip(rungs, p)) for p in pairs])


SIX = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
RESULTS = {
    # dataset A: two combos
    'A': {
        # 2 pairs; rung 1 -> (1.0 + 0.0)/2 = 0.5, rung 2 -> 1.0; combo = 0.75
        '1+2': _res((1.0, 2.0), [(1.0, 1.0), (0.0, 1.0)]),
        # 1 pair over six rungs; combo = mean(1,1,1,1,1,0) = 5/6
        '1+2+4+8+16+32': _res(SIX, [(1, 1, 1, 1, 1, 0)], n_resampled=3),
    },
    # dataset B: one combo, all correct -> 1.0
    'B': {'1+2': _res((1.0, 2.0), [(1.0, 1.0), (1.0, 1.0)])},
}


def t_the_means_nest_in_the_stated_order_and_the_pooled_decoy_differs():
    combo_rows, rung_rows, summary = kxk_report(RESULTS, epoch=1, k=2)
    a = (0.75 + 5 / 6) / 2
    want_total = (a + 1.0) / 2
    assert abs(summary['by_dataset']['A'] - a) < 1e-12, summary
    assert abs(summary['by_dataset']['B'] - 1.0) < 1e-12
    assert abs(summary['total'] - want_total) < 1e-12, (summary['total'], want_total)
    pooled = [v for d in RESULTS.values() for r in d.values()
              for p in r['acc_by_pair_rung'] for v in p.values()]
    decoy = sum(pooled) / len(pooled)
    assert abs(decoy - want_total) > 0.01, 'the decoy agrees -- the probe is useless'
    rung1 = [r for r in rung_rows if r['val_dataset'] == 'A'
             and r['combo'] == '1+2' and r['rung'] == 1.0][0]
    assert abs(rung1['level_accuracy'] - 0.5) < 1e-12
    assert abs(summary['six_rung'] - 5 / 6) < 1e-12, 'six_rung is A only'
    # native: only combos with n_resampled == 0 -> A '1+2' 0.75, B '1+2' 1.0
    assert abs(summary['native'] - 0.875) < 1e-12, summary['native']
    assert len(combo_rows) == 3 and len(rung_rows) == 2 + 6 + 2
    return f'total {want_total:.4f} (pooled decoy {decoy:.4f})'


def t_a_combo_that_could_not_be_drawn_is_missing_not_zero():
    bad = {'A': {'1+2': dict(ok=False, rungs=(1.0, 2.0), acc_by_pair_rung=[],
                             n_resampled=0),
                 '1+4': _res((1.0, 4.0), [(1.0, 0.0)])}}
    _, _, summary = kxk_report(bad, epoch=1, k=1)
    assert abs(summary['total'] - 0.5) < 1e-12, (
        f'{summary["total"]}: a missing combo was averaged in as a number')
    return 'skipped, not scored as 0'


def t_best_needs_every_dataset_to_hold_and_one_to_rise():
    """`_best.pt` is chosen by `improves_everywhere`. The decoy is the rule it
    replaced, the mean going up, which the third case below satisfies while
    one dataset falls."""
    nan = float('nan')
    saved = {'bracs': 0.60, 'ki67': 0.80}
    cases = [
        ('one up, one level',         {'bracs': 0.65, 'ki67': 0.80}, True),
        ('both up',                   {'bracs': 0.65, 'ki67': 0.85}, True),
        ('mean up, one falls',        {'bracs': 0.75, 'ki67': 0.79}, False),
        ('both level',                {'bracs': 0.60, 'ki67': 0.80}, False),
        ('one falls, none rise',      {'bracs': 0.59, 'ki67': 0.80}, False),
        ('one cannot be scored',      {'bracs': 0.65, 'ki67': nan},  True),
        ('unscored one, other falls', {'bracs': 0.50, 'ki67': nan},  False),
    ]
    for label, now, want in cases:
        got = improves_everywhere(now, saved)
        assert got == want, f'{label}: {now} against {saved} gave {got}'
    mean_up = {'bracs': 0.75, 'ki67': 0.79}
    assert sum(mean_up.values()) > sum(saved.values()), 'the decoy must pass'
    assert improves_everywhere({'bracs': 0.5, 'ki67': 0.5}, {}), \
        'the first epoch has nothing to fall below'
    assert not improves_everywhere({'bracs': nan, 'ki67': nan}, {}), \
        'nothing scored is not an improvement'
    return f'{len(cases)} cases, and the mean-went-up decoy refused'


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
