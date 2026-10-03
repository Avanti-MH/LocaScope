#!/usr/bin/env python3
"""Tests for training/PrototypicalRoutingHead/Episodes.py -- the mixed-pool
draws training and validation make.

    python utilities/test_modules/RoutingHeadsTest/test_episodes.py

No slide, no model, no GPU: the pools are hand-built ManifestRows whose
coordinates are chosen so the right answer is known.

WHAT THIS DEFENDS
-----------------
    overlap rule   two positions on one WSI whose footprints overlap never land
                   on opposite sides -- scored against the same draw with the
                   rule switched off, which DOES split them, so a test that
                   could not see a split cannot pass
    distinctness   no position appears twice in one draw, across every batch
    combinations   39 training combinations, none held out, each once per epoch
    K              `max_feasible_k` returns a K that draws and K+1 that does
                   not, on a pool whose answer is known by arithmetic
"""

from __future__ import annotations

import os
import random
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))

from _paths import setup_import_paths                            # noqa: E402

setup_import_paths()

from training.MppRoutingHead.Datasets import (                    # noqa: E402
    ManifestRow, pick_wsi_names)
from training.PrototypicalRoutingHead.Episodes import (          # noqa: E402
    HELD_OUT_COMBOS, batch_shape, draw, epoch_schedule, episodes_per_epoch,
    max_feasible_k, overlaps, pool_by_rung, reuse_pairs, training_combos)

_RESULTS = []


def check(name, fn):
    try:
        out = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   {out}' if out else ''))
    except Exception as e:                                       # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n          {type(e).__name__}: {e}')


def row(wsi, x, y, rung, fp=None):
    return ManifestRow(dataset='d', wsi_name=wsi, x=x, y=y, rung=float(rung),
                       footprint_l0=int(fp if fp is not None else 256 * rung))


def far_rows(rung, n, tag):
    """`n` rows at `rung`, each on its own WSI -- nothing can overlap."""
    return [row(f'{tag}{i}', 0, 0, rung) for i in range(n)]


# ── overlap rule ──────────────────────────────────────────────────────────

def t_overlaps_is_geometry_across_rungs():
    a = row('w', 0, 0, 1)                 # [0, 256)
    inside = row('w', 100, 100, 4)        # [100, 1124) -- contains part of a
    beside = row('w', 256, 0, 1)          # touches a's edge, no shared area
    other = row('v', 0, 0, 1)             # same place, another slide
    assert overlaps(a, inside) and overlaps(inside, a)
    assert not overlaps(a, beside), 'an edge is not an overlap'
    assert not overlaps(a, other), 'two slides never overlap'
    assert not overlaps(a, inside, max_ratio=1.0), 'ratio 1 disables the rule'
    return 'cross-rung yes, edge no, other slide no'


def _split_count(max_overlap):
    """How often the overlapping pair A (rung 1) / B (rung 4) lands on
    OPPOSITE sides over many draws, and how many draws succeeded."""
    A, B = row('w', 0, 0, 1), row('w', 100, 100, 4)
    pool = {1.0: [A, row('a2', 0, 0, 1)], 4.0: [B, row('b2', 0, 0, 4)]}
    rng = random.Random(0)
    split = ok = 0
    for _ in range(400):
        d = draw(pool, pool, rng, (1.0, 4.0), n_support=1, n_query=1, ks=1, kq=1,
                 max_overlap=max_overlap)
        if d is None:
            continue
        ok += 1
        s = {id(r) for b in d.supports for rows in b.values() for r in rows}
        if (id(A) in s) != (id(B) in s):
            split += 1
    return split, ok


def t_overlapping_positions_never_split_and_the_decoy_does():
    split, ok = _split_count(0.0)
    decoy_split, decoy_ok = _split_count(1.0)
    assert ok > 0, 'no draw succeeded -- the test saw nothing'
    assert split == 0, f'{split}/{ok} draws put A and B on opposite sides'
    assert decoy_split > 0, ('with the rule off A and B never split either, so '
                             'this test could not have seen a violation')
    return f'rule on: 0/{ok} split; rule off: {decoy_split}/{decoy_ok} split'


def t_no_position_appears_twice_in_a_draw():
    pool = {r: far_rows(r, 60, f'r{r:g}_') for r in (1.0, 2.0, 4.0)}
    rng = random.Random(1)
    for ks, kq in ((1, 5), (5, 1), (3, 4)):
        d = draw(pool, pool, rng, (1.0, 2.0, 4.0), n_support=2, n_query=3,
                 ks=ks, kq=kq)
        assert d is not None
        ids = [id(r) for side in (d.supports, d.queries) for b in side
               for rows in b.values() for r in rows]
        assert len(ids) == len(set(ids)), f'{ks}x{kq}: a position repeats'
        assert len(d.supports) == ks and len(d.queries) == kq
        assert all(len(rows) == 2 for b in d.supports for rows in b.values())
        assert all(len(rows) == 3 for b in d.queries for rows in b.values())
    return '1x5, 5x1, 3x4 all distinct, batch sizes exact'


# ── combinations and counts ───────────────────────────────────────────────

def t_39_training_combinations_none_held_out():
    combos = training_combos()
    assert len(combos) == 39, len(combos)
    assert len(set(combos)) == 39
    held = set(HELD_OUT_COMBOS)
    assert not (set(combos) & held), 'a held-out combination is trained on'
    assert Counter(len(c) for c in combos) == {3: 19, 4: 14, 5: 6}
    return '19 + 14 + 6'


def t_each_combination_once_per_epoch_and_k_rounds_under_none():
    combos = training_combos()
    rng = random.Random(2)
    one = epoch_schedule(combos, episodes_per_epoch('hold_q', 7, len(combos)), rng)
    assert sorted(one) == sorted(combos), 'an epoch is not every combination once'
    two = epoch_schedule(combos, episodes_per_epoch('hold_q', 7, len(combos)), rng)
    assert one != two, 'two epochs drew the same order'
    none = epoch_schedule(combos, episodes_per_epoch('none', 7, len(combos)), rng)
    assert Counter(none) == {c: 7 for c in combos}
    return 'hold: 39 draws, each once; none at K=7: 273, each 7 times'


def t_shapes_and_pairs():
    assert batch_shape('none', 9) == (1, 1)
    assert batch_shape('hold_s', 9) == (1, 9)
    assert batch_shape('hold_q', 9) == (9, 1)
    assert reuse_pairs(1, 3) == [(0, 0), (0, 1), (0, 2)]
    assert len(reuse_pairs(4, 4)) == 16
    for mode in ('none', 'hold_s', 'hold_q'):
        ks, kq = batch_shape(mode, 5)
        steps = episodes_per_epoch(mode, 5, 39) * len(reuse_pairs(ks, kq))
        assert steps == 39 * 5, (mode, steps)
    return 'every mode: 39 x K optimizer steps per epoch'


# ── K ─────────────────────────────────────────────────────────────────────

def t_max_feasible_k_is_exact_on_a_known_pool():
    """10 positions per rung on 10 slides, 1 support and 1 query per rung per
    batch: hold_s needs 1 + K, hold_q K + 1, so K = 9 draws and K = 10 cannot."""
    pool = {r: far_rows(r, 10, f'r{r:g}_') for r in (1.0, 2.0, 4.0)}
    combos = [(1.0, 2.0), (2.0, 4.0), (1.0, 2.0, 4.0)]
    shapes = [lambda k: batch_shape('hold_s', k), lambda k: batch_shape('hold_q', k)]
    k = max_feasible_k(lambda c: [(pool, pool)], combos, shapes,
                       n_support=1, n_query=1)
    assert k == 9, k
    rng = random.Random(0)
    assert draw(pool, pool, rng, combos[-1], n_support=1, n_query=1,
                ks=1, kq=9) is not None
    assert draw(pool, pool, rng, combos[-1], n_support=1, n_query=1,
                ks=1, kq=10) is None, 'K+1 drew: the maximum is not a maximum'
    val = max_feasible_k(lambda c: [(pool, pool)], combos, [lambda k: (k, k)],
                         n_support=1, n_query=1)
    assert val == 5, val          # 2K <= 10
    return 'hold K=9 (10 does not draw), val KxK K=5'


def t_mixed_pool_ignores_which_slide():
    rows = [row('a', 0, 0, 1), row('b', 0, 0, 1), row('a', 0, 0, 2)]
    pool = pool_by_rung(rows)
    assert len(pool[1.0]) == 2 and len(pool[2.0]) == 1
    return 'grouped by rung, across slides'


def t_a_capped_wsi_pick_is_random_fixed_and_a_full_cap_is_a_no_op():
    names = [f'BRACS_{i:04d}' for i in range(300)]
    a = pick_wsi_names(names, 50, seed=42)
    assert len(a) == 50 and set(a) <= set(names)
    assert a == pick_wsi_names(names, 50, seed=42), 'same seed, different slides'
    assert a == sorted(a), 'the picked slides lost their original order'
    assert a != pick_wsi_names(names, 50, seed=43), 'the seed does nothing'
    assert a != names[:50], 'this is the first N -- the decoy the pick replaced'
    assert max(names.index(n) for n in a) > 100, 'all picks from one end'
    assert pick_wsi_names(names, 300, seed=42) == names
    assert pick_wsi_names(names, 999, seed=42) == names
    assert pick_wsi_names(names, None, seed=42) == names
    return '50 of 300: fixed per seed, spread out, not the first 50; full cap = all, same order'


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
