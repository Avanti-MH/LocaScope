#!/usr/bin/env python3
"""Unit test for utilities/WsiFeaturesMapStore.

    python utilities/test_modules/test_wsi_features_map_store.py

No GPU, no slide, no model, about a second. Everything here is geometry and
tensor bookkeeping, which is the point: the module under test exists so that a
cached (slide, level) can be checked for free before anything expensive runs.

What it is for
--------------
GigaPathSlidingWinSimRot holds `list[FeaturesMap]`, one per tissue region.
FeatureStore holds five flat tensors. WsiFeaturesMapStore is the conversion, and
the gate that decides whether a stored one still describes the mask in hand.

The gate is the part worth testing hardest, because the alternative is trusting
`mask_id` -- a string a caller composes -- and a string cannot notice that
min_region_ratio changed. So the gate does not compare ids. It recomputes the
grid from the regions and compares the coordinates, which costs milliseconds and
can only agree if the mask and the scale agree.

Four checks:

    round trip          restore(flatten(x)) is x, scored against decoys
    a changed mask      narrower regions must be refused
    a changed scale     an L1 store asked for at L2 must be refused
    malformed input     fp32, or [N, D] where [N, n, D] belongs, must raise
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _d in ('aiNNModel', 'utilities'):
    p = str(_ROOT / _d)
    if p not in sys.path:
        sys.path.insert(0, p)

import torch                                                # noqa: E402

from PatchingLib import FeaturesMap, WsiFeaturesMap          # noqa: E402
from TissuesRegionsMask import TissueRegion                  # noqa: E402
import WsiFeaturesMapStore as WS                             # noqa: E402

_RESULTS = []


def check(name, fn):
    try:
        fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}')
    except Exception as e:                                   # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n        {type(e).__name__}: {e}')


def rejects(fn, needle=''):
    try:
        fn()
    except Exception as e:                                   # noqa: BLE001
        if needle and needle not in str(e):
            raise AssertionError(
                f'raised, but the message never mentions {needle!r}: {e}') from None
        return
    raise AssertionError('should have raised, returned normally')


# ── fixtures ──────────────────────────────────────────────────────────────────

DS, LEVEL, TILE, DIM = 4.0, 1, 256, 8

#: Three regions at level-0, sized so the grids differ from each other: a wide
#: one, a tall one, and a small one. Equal-sized regions would let a bug that
#: swaps two of them pass unnoticed.
#:
#: The small one is 2x1 tiles and not 1x1, which it was first. A single-patch
#: region has no order to get wrong, so the round trip's "roll the rows" decoy
#: became the identity and the test compared a difference against itself. A
#: decoy that cannot distinguish anything is not a weak check, it is no check.
REGIONS = [
    TissueRegion(x=0,     y=0,     w=int(TILE * DS * 4), h=int(TILE * DS * 2), index=0),
    TissueRegion(x=8192,  y=4096,  w=int(TILE * DS * 2), h=int(TILE * DS * 3), index=1),
    TissueRegion(x=20480, y=12288, w=int(TILE * DS * 2), h=int(TILE * DS * 1), index=2),
]


def grids(regions=None, ds=DS, level=LEVEL, tile=TILE, overlap=True):
    return WS.region_grids(regions if regions is not None else REGIONS,
                           ds=ds, level=level, tile_size=tile, overlap=overlap)


def wfm(regions=None, ds=DS, level=LEVEL, tile=TILE, overlap=True):
    """A WsiFeaturesMap over `regions` -- regions and maps paired by the type."""
    regions = REGIONS if regions is None else regions
    gs = grids(regions, ds=ds, level=level, tile=tile, overlap=overlap)
    return WsiFeaturesMap(regions, maps(gs), ds=ds, level=level,
                          tile_size=tile, overlap=overlap)


def maps(gs=None, seed=0):
    """One FeaturesMap per grid, values distinct per (region, patch, channel).

    Not random: a deterministic ramp means a mis-ordered restore produces a
    difference this file can print and a reader can recognise, rather than one
    more indistinguishable random tensor.
    """
    gs = gs if gs is not None else grids()
    out = []
    for r, g in enumerate(gs):
        n = len(g)
        base = torch.arange(n, dtype=torch.float32).unsqueeze(1) + 1000.0 * r
        feats = base + torch.arange(DIM, dtype=torch.float32).unsqueeze(0) * 0.001
        out.append(FeaturesMap(g, feats))
    return out


# ── round trip ────────────────────────────────────────────────────────────────

def t_round_trip():
    """restore(flatten(x)) is x, and the decoys say the comparison can fail.

    fp16 is FeatureStore's requirement, not a choice here, so the round trip is
    lossy by construction -- about 1e-3 relative on values of this size. That is
    why the check is a margin over decoys and not an equality: the decoys are
    two orders larger, so the test still fails on a real reordering.
    """
    gs = grids()
    w = wfm()
    before = w.maps
    tensors = WS.to_store_tensors(w)

    assert tensors['features'].dtype == torch.float16, \
        f"features must be fp16 for FeatureStore, got {tensors['features'].dtype}"
    assert tensors['features'].ndim == 3, \
        f"features must be [N, n, D], got {tuple(tensors['features'].shape)}"
    n_total = sum(len(g) for g in gs)
    assert tensors['features'].shape == (n_total, 1, DIM), tensors['features'].shape
    for name, dt in (('x', torch.int32), ('y', torch.int32),
                     ('region', torch.int16), ('grid_rc', torch.int32)):
        assert tensors[name].dtype == dt, f'{name} is {tensors[name].dtype}, want {dt}'

    back = WS.from_store_tensors(tensors, REGIONS, ds=DS, level=LEVEL,
                                 tile_size=TILE, overlap=True)
    after = back.maps
    assert len(after) == len(before), f'{len(after)} maps back, {len(before)} in'
    assert back.regions is REGIONS, 'the restored map lost its regions'
    assert (back.ds, back.level, back.overlap) == (DS, LEVEL, True), \
        'the restored map lost the scale it was built at'

    worst_gap = 0.0
    worst_decoy = float('inf')
    for i, (a, b) in enumerate(zip(after, before)):
        assert len(a.grid) == len(b.grid), f'region {i}: grid length changed'
        gap = float((a.features.float() - b.features).abs().max())
        # Two ways a restore goes wrong and neither raises: the per-region split
        # lands on the wrong boundary, or a region's rows come back rotated.
        decoys = {'roll rows': float((a.features.float()
                                      - b.features.roll(1, dims=0)).abs().max())}
        if i + 1 < len(before) and len(before[i + 1].grid) == len(b.grid):
            decoys['next region'] = float(
                (a.features.float() - before[i + 1].features).abs().max())
        worst_gap = max(worst_gap, gap)
        worst_decoy = min(worst_decoy, min(decoys.values()))
        print(f'        region {i}  n={len(a.grid)}  max|Δ| {gap:.3e}   '
              + '  '.join(f'{k} {v:.2e}' for k, v in decoys.items()))

    assert worst_gap * 100 < worst_decoy, (
        f'round trip differs by {worst_gap:.3e}, not clear of the nearest decoy '
        f'at {worst_decoy:.3e}')


def t_coordinates_survive():
    """x / y / region / grid_rc must describe the same patches the grids do.

    Checked against the grids rather than against a recorded copy: the store's
    job is to be readable by a process that has only the mask, so the columns
    have to agree with what geometry alone produces.
    """
    gs = grids()
    tensors = WS.to_store_tensors(wfm())
    at = 0
    for r, g in enumerate(gs):
        for k, info in enumerate(g.iter_infos()):
            assert int(tensors['region'][at]) == r, f'row {at}: region'
            assert int(tensors['x'][at]) == info.x, f'row {at}: x'
            assert int(tensors['y'][at]) == info.y, f'row {at}: y'
            assert tuple(tensors['grid_rc'][at].tolist()) == (info.row, info.col), \
                f'row {at}: grid_rc'
            at += 1
    assert at == tensors['x'].numel(), f'{at} rows walked, {tensors["x"].numel()} stored'


# ── the gate ──────────────────────────────────────────────────────────────────

def t_gate_accepts_the_same_geometry():
    tensors = WS.to_store_tensors(wfm())
    bad = WS.geometry_mismatch(tensors, grids())      # rebuilt, not reused
    assert bad == [], f'the gate rejected an unchanged mask: {bad}'


def t_gate_refuses_a_narrowed_mask():
    """Dropping a region has to be caught.

    This is what mask_id cannot do. 'hest@ds4' is the same string whether
    min_region_ratio was 0.10 or 0.30, and the store written under one would be
    loaded under the other with a region's worth of features missing and every
    later region's index shifted by one.
    """
    tensors = WS.to_store_tensors(wfm())
    narrowed = grids(regions=REGIONS[:2])
    bad = WS.geometry_mismatch(tensors, narrowed)
    assert bad, 'a mask with one fewer region was accepted'
    print(f'        {bad[0]}')


def t_gate_refuses_a_moved_region():
    """A region of the same size at a different place must also be caught.

    Region COUNT is the easy half. Counting alone would pass a mask whose
    regions merged differently but happened to come out three again.
    """
    tensors = WS.to_store_tensors(wfm())
    moved = [TissueRegion(x=r.x + int(TILE * DS), y=r.y, w=r.w, h=r.h, index=r.index)
             for r in REGIONS]
    bad = WS.geometry_mismatch(tensors, grids(regions=moved))
    assert bad, 'a mask with a shifted region was accepted'
    print(f'        {bad[0]}')


def t_gate_refuses_another_scale():
    """An L1 store asked for at L2.

    ds changes the tile count of every region, so this is the loudest of the
    three -- but it is also the one a caller is most likely to trigger, because
    build_wsi_features rebuilds at a second scale by design.
    """
    tensors = WS.to_store_tensors(wfm())
    bad = WS.geometry_mismatch(tensors, grids(ds=DS * 4, level=LEVEL + 1))
    assert bad, 'a store from another level was accepted'
    print(f'        {bad[0]}')


def t_gate_refuses_a_changed_overlap():
    """overlap=False halves the patch count without moving a single region."""
    tensors = WS.to_store_tensors(wfm())
    bad = WS.geometry_mismatch(tensors, grids(overlap=False))
    assert bad, 'a store written with the overlap grid was accepted without it'
    print(f'        {bad[0]}')


# ── malformed input ───────────────────────────────────────────────────────────

def t_refuses_malformed():
    gs = grids()
    good = maps(gs)
    restore = dict(ds=DS, level=LEVEL, tile_size=TILE, overlap=True)

    # The pairing check moved into WsiFeaturesMap's constructor, which is the
    # point of the type: to_store_tensors no longer has two lists to compare.
    rejects(lambda: WsiFeaturesMap(REGIONS, good[:2], **restore), 'paired')
    rejects(lambda: WsiFeaturesMap(REGIONS, good, ds=DS * 4, level=LEVEL + 1,
                                   tile_size=TILE, overlap=True), 'scales')

    # On the LENGTH, not on a corrupted `region` column: from_store_tensors
    # splits by the grids' own lengths and documents that it ignores the stored
    # region ids, so a test that expects it to notice one is testing a promise
    # the module never made. The first version did exactly that and passed
    # nothing.
    full = WS.to_store_tensors(wfm())
    rejects(lambda: WS.from_store_tensors(
        {**full, 'features': full['features'][:-1]}, REGIONS, **restore), 'rows')
    rejects(lambda: WS.from_store_tensors(
        {**full, 'features': full['features'][:, 0]}, REGIONS, **restore),
        '[N, n, D]')

    # A FeaturesMap whose features do not match its grid cannot even be built,
    # which is the right place for that check -- assert it stays true so this
    # module is not tempted to re-implement it.
    rejects(lambda: FeaturesMap(gs[0], torch.zeros(len(gs[0]) + 1, DIM)), 'count')
    rejects(lambda: FeaturesMap(gs[0], torch.zeros(len(gs[0]))), '[N, D]')


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    argparse.ArgumentParser().parse_args()

    print('flatten / restore')
    check('round trip survives fp16',        t_round_trip)
    check('coordinates match the grids',     t_coordinates_survive)

    print('the geometry gate')
    check('accepts an unchanged mask',       t_gate_accepts_the_same_geometry)
    check('refuses a narrowed mask',         t_gate_refuses_a_narrowed_mask)
    check('refuses a moved region',          t_gate_refuses_a_moved_region)
    check('refuses another scale',           t_gate_refuses_another_scale)
    check('refuses a changed overlap',       t_gate_refuses_a_changed_overlap)

    print('malformed input')
    check('refuses what cannot be stored',   t_refuses_malformed)

    bad = [n for n, e in _RESULTS if e is not None]
    print(f'\n{len(_RESULTS) - len(bad)}/{len(_RESULTS)} passed')
    if bad:
        print('failed: ' + ', '.join(bad))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
