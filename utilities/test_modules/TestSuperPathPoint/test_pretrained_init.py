#!/usr/bin/env python3
"""Tests for KeypointNet.upstream_state_dict / load_upstream. spec.md 13.

    python utilities/test_modules/TestSuperPathPoint/test_pretrained_init.py
    python .../test_pretrained_init.py --with-weights     # needs sp_v6

THIS TEST EXISTS TO BLOCK AN ARM. `--pretrained` is one of the four arms of
spec.md 12 step 6, and it has a failure mode that produces a NUMBER rather than
an error: under `strict=False` the keys that did not match stay randomly
initialised, the network trains, the loss falls, and the run reads as
"pretraining did not help" instead of "pretraining did not happen". So the load
is strict, and this file checks the strictness from both sides.

WHAT WOULD RUN AND BE WRONG
-----------------------------
1. A prefix rename that misses a group. `backbone.` -> `backbone.stages.` is
   three separate rewrites (`detector.` and `descriptor.` grow a `head.`), and
   getting one wrong leaves that whole head random while the trunk transfers.
   A trunk-only transfer is still much better than random, so the arm still
   "works".
2. The RGB inflation left as a plain repeat, without the /3. Then the
   3-channel network sees three times the activation of the 1-channel one at
   every layer -- and BatchNorm mostly absorbs it, so it trains.
3. Both sides empty. A comparison of two networks that both answer zero passes
   any equality check, which is why every equality here is scored against a
   DECOY: shuffle one layer and the SAME assertion has to fail.

Sections:
  1. keys      -- the rename covers every parameter, and only by renaming
  2. rgb       -- repeat-and-divide is exact on luma, against a no-/3 decoy
  3. strict    -- a missing or extra key is refused, not absorbed
  4. weights   -- --with-weights only: the real sp_v6 against the real teacher
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))

from _paths import setup_import_paths                            # noqa: E402

setup_import_paths()

import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402

from SuperPoint.KeypointNet import (KeypointNetConfig,           # noqa: E402
                                    load_upstream,
                                    upstream_state_dict)

_RESULTS = []


def check(name, fn):
    try:
        out = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   {out}' if out else ''))
    except Exception as e:                                       # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n          {type(e).__name__}: {e}')


def _student(in_channels=1):
    return KeypointNetConfig.wired(in_channels=in_channels).build('cpu')


def _fake_upstream(in_channels=1):
    """A state dict keyed the way upstream keys one, with the student's shapes.

    Built by taking a student apart rather than by importing upstream: the
    shapes are the thing under test everywhere else, and a fake that got them
    from the same place cannot disagree about them here.
    """
    net = _student(in_channels)
    out = {}
    for key, value in net.state_dict().items():
        for new, old in (('backbone.stages.', 'backbone.'),
                         ('detector.head.', 'detector.'),
                         ('descriptor.head.', 'descriptor.')):
            if key.startswith(new):
                key = old + key[len(new):]
                break
        if not value.is_floating_point():
            out[key] = value                       # num_batches_tracked
        elif key.endswith('running_var'):
            # POSITIVE, and this is not fussiness. A randn here is a negative
            # variance, `sqrt(var + eps)` is NaN, and every forward below
            # answers NaN -- which an `abs().max() < 1e-5` reports as a failure
            # of the thing under test rather than of its fixture. It cost one
            # run to find out.
            out[key] = torch.rand_like(value) + 0.5
        else:
            # SMALL, so the detector's softmax does not saturate. At randn
            # scale it does: the first version of this file reported `gap 0,
            # decoy 1` -- both exact integers, which a float32 network cannot
            # produce -- because every prob_map had collapsed to one-hot. The
            # comparison then tested WHERE the peak was and not what the
            # probabilities were, and a wrong divisor that leaves the argmax
            # alone would have passed. `check_fixture` below is what refuses
            # to let that happen silently again.
            out[key] = torch.randn_like(value) * 0.05
    return out


def check_fixture(state, in_channels=1):
    """Refuse a fixture that cannot test what it is used for.

    Three fixture bugs in one session -- a float32 edge that could not hold
    0.95, a mask that was solid tissue so every candidate fell in one bucket,
    a randn `running_var` that made every forward NaN -- and all three were
    reported as failures of the SUBJECT. The dangerous version is the same bug
    the other way: a fixture too degenerate to exercise the code, which passes.

    So the fixture states its own preconditions, and they fail as `fixture`.
    """
    bad = [k for k, v in state.items()
           if v.is_floating_point() and not torch.isfinite(v).all()]
    assert not bad, f'fixture has non-finite tensors: {bad[:3]}'
    var = [k for k in state if k.endswith('running_var')]
    assert var, 'fixture has no BatchNorm buffers; it is not this architecture'
    assert all((state[k] > 0).all() for k in var), 'a running_var is <= 0'

    net = _student(in_channels)
    net.load_state_dict(upstream_state_dict(state, in_channels), strict=True)
    net.eval()
    with torch.no_grad():
        prob = net(torch.rand(1, in_channels, 64, 64)).prob_map
    assert torch.isfinite(prob).all(), 'fixture produces a non-finite prob_map'
    # NOT one-hot: a saturated map compares argmax only.
    spread = float(prob.max() - prob.min())
    assert 1e-4 < spread < 0.999, (
        f'prob_map spread {spread:.3g} -- a saturated or constant map cannot '
        f'tell a wrong scale from a right one')
    return f'{len(state)} tensors, prob spread {spread:.3g}'


# ── 0. fixture ───────────────────────────────────────────────────────────────

def t_the_fixture_can_test_what_it_is_used_for():
    """The fixture's own preconditions, so a broken one fails as `fixture`
    rather than as the subject -- or, worse, quietly passes."""
    return check_fixture(_fake_upstream(1), 1)


# ── 1. keys ──────────────────────────────────────────────────────────────────

def t_the_rename_covers_every_parameter():
    """Set equality against the student's own keys, both directions."""
    net = _student()
    mapped = set(upstream_state_dict(_fake_upstream(), 1))
    wanted = set(net.state_dict())
    assert mapped == wanted, (
        f'missing {sorted(wanted - mapped)[:3]}, extra {sorted(mapped - wanted)[:3]}')
    return f'{len(wanted)} keys, exactly'


def t_every_group_is_renamed_and_not_just_the_trunk():
    """The decoy: a rename that only handled `backbone.` would still cover 3/4
    of the tensors, and a trunk-only transfer trains fine."""
    mapped = upstream_state_dict(_fake_upstream(), 1)
    for group in ('backbone.stages.', 'detector.head.', 'descriptor.head.'):
        assert any(k.startswith(group) for k in mapped), f'nothing under {group}'
    assert not any(k.startswith(('backbone.0', 'detector.0', 'descriptor.0'))
                   for k in mapped), 'an upstream-shaped key survived'
    return 'all three groups'


def t_the_rename_moves_no_values():
    """Renaming must not also transform. Same tensors, different names."""
    raw = _fake_upstream()
    mapped = upstream_state_dict(raw, 1)
    for key, value in raw.items():
        if key == 'backbone.0.0.conv.weight':
            continue                       # the only one inflation may touch
        moved = [v for k, v in mapped.items() if v.shape == value.shape
                 and torch.equal(v, value)]
        assert moved, f'{key} did not survive the rename intact'
    return f'{len(raw)} tensors unchanged'


# ── 2. rgb ───────────────────────────────────────────────────────────────────

# `t_repeat_and_divide_is_exact_on_luma` MOVED TO `with_weights` on 2026-08-31.
# A fake state dict cannot carry it. Three attempts: `randn` saturated the
# detector softmax to one-hot and compared only where the peak was (`gap 0,
# decoy 1`, two exact integers a float32 network cannot produce); `randn * 0.05`
# flattened the softmax so every difference compressed toward zero (decoy
# 9e-10); reading the trunk instead of prob_map hit the same scaling attenuating
# through four stages (decoy 7e-9). Each fix was a different weight scale, and
# choosing a scale that lands the output in a readable regime is calibrating the
# fixture, not testing the code. Real weights have a real scale.
#
# What stays here is everything that needs NO forward -- the rename, the
# inflation's refusals, strict -- and those passed unchanged through all three.


def t_an_undefined_inflation_is_refused():
    for want in (2, 4):
        try:
            upstream_state_dict(_fake_upstream(1), want)
        except ValueError as e:
            assert 'repeat-and-divide' in str(e), str(e)
        else:
            raise AssertionError(f'{want} channels was accepted')
    return '2 and 4 refused'


# ── 3. strict ────────────────────────────────────────────────────────────────

def t_a_missing_key_is_refused_rather_than_left_random():
    """The failure this whole file exists for."""
    net = _student()
    raw = _fake_upstream()
    raw.pop('detector.0.conv.weight')
    try:
        load_upstream(net, raw)
    except RuntimeError as e:
        assert 'Missing key' in str(e) or 'missing keys' in str(e).lower(), str(e)
        return 'refused'
    raise AssertionError(
        'a state dict short one tensor loaded anyway; that layer is now random '
        'and the arm would report "pretraining did not help"')


def t_an_unexpected_key_is_refused():
    net = _student()
    raw = _fake_upstream()
    raw['backbone.99.weight'] = torch.zeros(1)
    try:
        load_upstream(net, raw)
    except RuntimeError:
        return 'refused'
    raise AssertionError('an unmapped key was absorbed')


def t_a_good_load_changes_the_weights():
    """Guards against the vacuous pass: if `load_upstream` did nothing, every
    check above still passes."""
    net = _student()
    before = net.state_dict()['backbone.stages.0.0.conv.weight'].clone()
    load_upstream(net, _fake_upstream())
    after = net.state_dict()['backbone.stages.0.0.conv.weight']
    assert not torch.equal(before, after), 'load_upstream was a no-op'
    return 'weights moved'


# ── 4. weights ───────────────────────────────────────────────────────────────

def with_weights(device):
    """The real thing: sp_v6 into a student, against the teacher's own outputs.

    THREE CLAIMS, ALL AGAINST REAL WEIGHTS, because the fake state dict cannot
    carry any of them. Three attempts proved that: `randn` saturated the
    softmax to one-hot (`gap 0, decoy 1`), `randn * 0.05` flattened it
    (decoy 9e-10), and reading the trunk instead ran into the same scaling
    attenuating four stages down (decoy 7e-9). Each fix was a different weight
    scale, which is calibrating a fixture rather than testing the code
    (ClaudeRules section 8). Real weights have a real scale and need no tuning.

      1. DETECTOR   `student.prob_map` == `teacher.dense_prob`, elementwise.
      2. DESCRIPTOR `student.descriptors` == `teacher.dense_descriptors`.
                    Not covered by 1: `detector.head.0.conv.weight` and
                    `descriptor.head.0.conv.weight` are both [256, 128, 3, 3],
                    so a mis-mapping INSIDE the descriptor path leaves
                    `prob_map` untouched and passes `strict=True` as well.
      3. RGB        the 3-channel student on a repeated luma image == the
                    1-channel student on that image. This is what conv1's
                    repeat-and-divide-by-three buys, and without it
                    `rgb+pretrained` and `gray+pretrained` are two different
                    kinds of pretraining (spec.md 13).

    Each is scored against a decoy, and the decoys are different failures:
    a shuffled trunk layer for 1 and 2, the missing /3 for 3.
    """
    from SuperPoint.Teacher import SuperPointTeacher, TeacherConfig

    state, path = SuperPointTeacher.weights_state_dict()
    teacher = TeacherConfig().build(device)
    net = _student(1).to(device).eval()
    load_upstream(net, state)

    image = torch.rand(2, 1, 256, 256, device=device)
    with torch.no_grad():
        out = net(image)
        prob, desc = out.prob_map.float(), out.descriptors.float()
        their_prob = teacher.dense_prob(image).float()
        their_desc = teacher.dense_descriptors(image).float()

    # A guard before any comparison: two degenerate outputs agree perfectly.
    assert torch.isfinite(their_prob).all() and torch.isfinite(their_desc).all()
    assert float(their_prob.max() - their_prob.min()) > 1e-3, \
        'the teacher answers a flat map; nothing below compares anything'

    gap = (prob - their_prob).abs().max().item()
    gap_desc = (desc - their_desc).abs().max().item()

    # decoy for 1 and 2: shuffle one trunk layer, both must break
    upstream_key = 'backbone.2.0.conv.weight'
    spoiled = dict(state)
    w = spoiled[upstream_key]
    spoiled[upstream_key] = w.flatten()[torch.randperm(w.numel())].view_as(w)
    net2 = _student(1).to(device).eval()
    load_upstream(net2, spoiled)
    with torch.no_grad():
        bad = net2(image)
        decoy = (bad.prob_map.float() - their_prob).abs().max().item()
        decoy_desc = (bad.descriptors.float() - their_desc).abs().max().item()

    assert gap < 1e-4, (
        f'student and teacher differ by {gap:.3g} on the same input. The '
        f'weights loaded but the two models are not the same model')
    assert decoy > 100 * max(gap, 1e-9), (
        f'shuffling {upstream_key} moved prob_map by only {decoy:.3g}; this '
        f'comparison cannot tell a loaded net from a broken one')
    assert gap_desc < 1e-4, (
        f'the DESCRIPTOR heads differ by {gap_desc:.3g} while the detector '
        f'agrees. `strict=True` cannot see this: the two heads first blocks '
        f'are both [256, 128, 3, 3]')
    assert decoy_desc > 100 * max(gap_desc, 1e-9), (
        f'the shuffled trunk moved the descriptors by only {decoy_desc:.3g}')

    # 3. RGB: repeat-and-divide is exact on luma, against the no-divide decoy
    rgb = _student(3).to(device).eval()
    load_upstream(rgb, state)
    with torch.no_grad():
        mine_rgb = rgb(image.repeat(1, 3, 1, 1)).prob_map.float()
    gap_rgb = (mine_rgb - prob).abs().max().item()

    plain = upstream_state_dict(state, 3)
    plain['backbone.stages.0.0.conv.weight'] = (
        plain['backbone.stages.0.0.conv.weight'] * 3.0)
    rgb2 = _student(3).to(device).eval()
    rgb2.load_state_dict(plain, strict=True)
    with torch.no_grad():
        decoy_rgb = (rgb2(image.repeat(1, 3, 1, 1)).prob_map.float()
                     - prob).abs().max().item()

    assert gap_rgb < 1e-4, (
        f'the 3-channel student on a repeated luma image differs from the '
        f'1-channel student by {gap_rgb:.3g}; conv1 was not inflated exactly')
    assert decoy_rgb > 100 * max(gap_rgb, 1e-9), (
        f'dropping the /3 moved the output by only {decoy_rgb:.3g}; this '
        f'cannot tell repeat-and-divide from a plain repeat')

    return (f'{os.path.basename(path)}: prob {gap:.2g}/{decoy:.2g}  '
            f'desc {gap_desc:.2g}/{decoy_desc:.2g}  '
            f'rgb {gap_rgb:.2g}/{decoy_rgb:.2g}   (gap/decoy)')


_SECTIONS = {
    'fixture': ['t_the_fixture_can_test_what_it_is_used_for'],
    'keys': ['t_the_rename_covers_every_parameter',
             't_every_group_is_renamed_and_not_just_the_trunk',
             't_the_rename_moves_no_values'],
    'rgb': ['t_an_undefined_inflation_is_refused'],
    'strict': ['t_a_missing_key_is_refused_rather_than_left_random',
               't_an_unexpected_key_is_refused',
               't_a_good_load_changes_the_weights'],
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', choices=sorted(_SECTIONS))
    ap.add_argument('--with-weights', action='store_true',
                    help='load the real sp_v6 and compare against the teacher')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available()
                    else 'cpu')
    args = ap.parse_args()

    torch.manual_seed(0)
    for section in (args.only or list(_SECTIONS)):
        print(f'\n[{section}]')
        for name in _SECTIONS[section]:
            check(name[2:].replace('_', ' '), globals()[name])

    if args.with_weights:
        print('\n[weights]')
        check('sp_v6 into a student equals the teacher',
              lambda: with_weights(args.device))

    failed = [n for n, e in _RESULTS if e is not None]
    print(f'\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed')
    if failed:
        print('failed: ' + ', '.join(failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
