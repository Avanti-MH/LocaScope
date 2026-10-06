#!/usr/bin/env python3
"""Tests for aiNNModel/models/common/Resume.py -- crash-resume state.

    python utilities/test_modules/TestAiNNModel/test_resume.py

No slide, no encoder: a two-layer toy model trained on data drawn from every
generator Resume.py saves (python's `random`, numpy, torch, and a named
`random.Random` of the caller's), so a resume that forgot one of them changes
the numbers.

WHAT THIS DEFENDS
-----------------
    exactness     2 epochs, stop, resume, 2 more == 4 epochs straight, bit for
                  bit -- weights, optimizer, scheduler, best score, the rows
                  accumulated so far. Scored against a DECOY resume that
                  restores the weights but not the RNG, which must differ, so
                  a test that could not see a lost generator cannot pass
    the rule      no --resume-dir: nothing written. A dir, no file: from
                  scratch. A finished run resumes into zero epochs
    identity      a file written under a different identity is refused, and
                  the refusal names the field
"""

from __future__ import annotations

import os
import random
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))

from _paths import setup_import_paths                            # noqa: E402

setup_import_paths()

import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402

from Resume import ResumeFile, ResumeMismatch                    # noqa: E402

_RESULTS = []
IDENTITY = {'loss': 'bal', 'lr': 0.01, 'rungs': [1, 2, 4]}


def check(name, fn):
    try:
        out = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   {out}' if out else ''))
    except Exception as e:                                       # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n          {type(e).__name__}: {e}')


def _build(seed=0):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(),
                                torch.nn.Dropout(0.3), torch.nn.Linear(8, 1))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', patience=0)
    return model, opt, sched


def _train(epochs, resume_dir=None, *, stop_after=None, restore_rng=True):
    """The training loop shape every CLI here uses. Returns the final state
    and how many epochs actually ran in this call."""
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    model, opt, sched = _build()
    mine = random.Random(7)
    best, rows, start = -1.0, [], 0
    rf = ResumeFile.for_model(resume_dir, 'toy')
    state = rf.load(IDENTITY)
    if state is not None:
        if restore_rng:
            ResumeFile.restore(state, modules={'m': model}, optimizers={'o': opt},
                               schedulers={'s': sched}, named_rngs={'mine': mine})
        else:        # the decoy: weights and optimizer, generators left fresh
            model.load_state_dict(state['modules']['m'])
            opt.load_state_dict(state['optimizers']['o'])
            sched.load_state_dict(state['schedulers']['s'])
        start, best, rows = state['epoch'], state['best'], list(state['extra'])
    ran = 0
    for epoch in range(start + 1, epochs + 1):
        model.train()
        n = 4 + mine.randrange(4) + random.randrange(3)
        x = torch.randn(n, 4) + float(np.random.rand())
        loss = model(x).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        score = -float(loss)
        sched.step(score)
        best = max(best, score)
        rows.append({'epoch': epoch, 'loss': float(loss)})
        rf.save(IDENTITY, epoch=epoch, modules={'m': model}, optimizers={'o': opt},
                schedulers={'s': sched}, best=best, extra=rows,
                named_rngs={'mine': mine})
        ran += 1
        if stop_after is not None and epoch == stop_after:
            break
    return model, opt, best, rows, ran


def _same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a.state_dict().values(),
                                                 b.state_dict().values()))


def t_resumed_run_equals_the_uninterrupted_one_and_the_decoy_does_not():
    straight, s_opt, s_best, s_rows, _ = _train(4)
    with tempfile.TemporaryDirectory() as d:
        _train(4, d, stop_after=2)
        resumed, r_opt, r_best, r_rows, ran = _train(4, d)
        assert ran == 2, f'resume ran {ran} epochs, not the remaining 2'
        assert _same(straight, resumed), 'weights differ after resume'
        assert r_best == s_best and r_rows == s_rows, 'best / rows differ'
        assert r_opt.state_dict()['state'].keys() == s_opt.state_dict()['state'].keys()
    with tempfile.TemporaryDirectory() as d:
        _train(4, d, stop_after=2)
        decoy, *_ = _train(4, d, restore_rng=False)
        assert not _same(straight, decoy), (
            'a resume that dropped the RNG still matched -- this test cannot '
            'see a lost generator')
    return 'bit-identical; the no-RNG decoy differs'


def t_no_dir_writes_nothing_and_an_empty_dir_starts_fresh():
    rf = ResumeFile.for_model(None, 'toy')
    assert not rf.enabled and rf.load(IDENTITY) is None
    with tempfile.TemporaryDirectory() as d:
        _, _, _, rows, ran = _train(3, d)
        assert ran == 3 and len(rows) == 3
        assert os.path.exists(os.path.join(d, 'toy_resume.pt'))
    assert rf.path is None, 'a disabled resume file has somewhere to write'
    _, _, _, rows, ran = _train(2, None)
    assert ran == 2 and len(rows) == 2
    return 'disabled: nothing; empty dir: 3 epochs from scratch, file written'


def t_a_finished_run_resumes_into_zero_epochs():
    with tempfile.TemporaryDirectory() as d:
        _train(3, d)
        _, _, _, rows, ran = _train(3, d)
        assert ran == 0, f'a finished run trained {ran} more epochs'
        assert len(rows) == 3, 'the rows of the finished run were lost'
    return '--epochs is the total'


def t_a_different_identity_is_refused_by_name():
    with tempfile.TemporaryDirectory() as d:
        _train(1, d)
        rf = ResumeFile.for_model(d, 'toy')
        try:
            rf.load({**IDENTITY, 'loss': 'ord_a'})
        except ResumeMismatch as e:
            assert 'loss' in str(e) and 'ord_a' in str(e), str(e)
        else:
            raise AssertionError('an ord_a run resumed a bal file')
        assert rf.load({**IDENTITY, 'rungs': (1, 2, 4)}) is not None, (
            'a tuple and a list of the same values were called different')
    return 'loss named; tuple == list'


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
