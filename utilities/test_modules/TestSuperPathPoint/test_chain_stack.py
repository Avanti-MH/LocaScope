#!/usr/bin/env python3
"""Tests for `ChainStack`: `FStack`/`RStack`/`CStack`. spec.md 3.2.

    python utilities/test_modules/TestSuperPathPoint/test_chain_stack.py

WHAT THIS BLOCKS
==================
Everything downstream -- `SurvivalProcess`, tau calibration, the eventual
相依型 comparison -- reads coordinates this module produces. A wrong scale
factor or a pyramid whose children do not actually tile their parent fills
the survival table with plausible numbers instead of failing loudly (the
same shape of risk `test_survival.py`'s own header names for `Patterns` and
`Attribution`). This file is pure geometry and local-disk-cache arithmetic
only -- no GPU, no WSI handle, no store -- which is what lets it run in
seconds and be the thing checked BEFORE any real tile is ever extracted
(ClaudeRules 8).

Sections:
  scale      `rung_scale` vs `rung_shrink` -- two quantities, equal on 'F'
  degrade    the 'R' degradation is the one in TileSampler, not a copy
  mother     CStack's root tile IS FStack's footprint, not a second definition
  pyramid    the reconstruction property, the overlap's exact 1/4 share, growth
  nearest    picked by centre distance, not by containment
  cache      the local tile cache round-trips
  own        from_tile/base_rung compounding, derive() refusing 'own', and
             FStack/RStack/CStack.from_own against a fake PreTileStore
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))

from _paths import setup_import_paths                           # noqa: E402

setup_import_paths('SuperPathPoint')

from pathlib import Path                                      # noqa: E402

import numpy as np                                           # noqa: E402

from Store import (PreTileCorpus, PreTileMeta, PreTileRecord,  # noqa: E402
                   PreTileStore)
from TileSampler import PRE_TILE_FACTOR, pre_tile_px          # noqa: E402
from SlideReader import degrade_resolution                     # noqa: E402
from PatchingLib import PatchInfo                             # noqa: E402
from SurvivalAnalysis import ChainStack                    # noqa: E402
FStack, RStack, CStack = ChainStack.FStack, ChainStack.RStack, ChainStack.CStack

_RESULTS = []
_TILE = 64          # small enough to be instant, even enough to have a centre
_FACTOR = PRE_TILE_FACTOR


def check(name, fn):
    try:
        out = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   {out}' if out else ''))
    except Exception as e:                                   # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n          {type(e).__name__}: {e}')


# ── 1. scale ─────────────────────────────────────────────────────────────────

def t_scale_and_shrink_agree_on_F_and_disagree_on_R():
    """The two quantities that are both `ds` on one axis and are not the same.

    `rung_scale` is level-0 px per output PIXEL -- the mapping. `rung_shrink`
    is how far a position can be off -- the tolerance. On 'F' both are `ds`; on
    'R' the mapping is 1.0 (the frame never moves) while the tolerance is still
    `ds` (the image was degraded). Using `ds` as the 'R' mapping scatters every
    coarse point `ds` times too far and the table still fills.
    """
    for ds in (1.0, 4.0, 32.0):
        if ChainStack.rung_scale(ds, 'F') != ds:
            raise AssertionError(f"F scale at ds {ds} is not ds")
        if ChainStack.rung_shrink(ds, 'F') != ds:
            raise AssertionError(f"F shrink at ds {ds} is not ds")
        if ChainStack.rung_scale(ds, 'R') != 1.0:
            raise AssertionError(
                f"R scale at ds {ds} came back "
                f"{ChainStack.rung_scale(ds, 'R')}, not 1.0")
        if ChainStack.rung_shrink(ds, 'R') != ds:
            raise AssertionError(f"R shrink at ds {ds} is not ds")
    return "F: scale == shrink == ds.  R: scale 1.0, shrink ds"


def t_C_scale_and_shrink_match_F_not_R():
    """'C' is real reads exactly like 'F', just tiled recursively.

    A descendant tile at rung `ds` covers `tile * ds` level-0 px the same as
    an 'F' tile there -- the ONLY thing 'C' changes is where the pixels come
    from, not what one output pixel means. If this test disagreed, every
    coverage-confirmation or 相依型 distance computed on a 'C' tile would be
    scaled wrong while still looking like a plausible number.
    """
    for ds in (1.0, 4.0, 32.0):
        if ChainStack.rung_scale(ds, 'C') != ChainStack.rung_scale(ds, 'F'):
            raise AssertionError(f"C scale at ds {ds} disagrees with F")
        if ChainStack.rung_shrink(ds, 'C') != ChainStack.rung_shrink(ds, 'F'):
            raise AssertionError(f"C shrink at ds {ds} disagrees with F")
    return "C agrees with F at every ds checked"


# ── 2. degrade ───────────────────────────────────────────────────────────────

def t_the_r_degradation_loses_detail_and_keeps_the_frame():
    """'R' at ds d: same size out, `tile/d` real samples in it.

    Checked against a decoy that would pass a shape assertion: an image that
    was resized and resized back with the SAME filter both ways keeps more
    high-frequency content than the INTER_AREA/INTER_LINEAR pair, so a
    'degradation' that lost nothing would show up here as a variance that did
    not drop.
    """
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    out = degrade_resolution(img, 8.0, 256)
    if out.shape != img.shape:
        raise AssertionError(f'{out.shape} != {img.shape}; the frame moved')
    before = float(np.var(np.diff(img[..., 0].astype(np.float32), axis=1)))
    after = float(np.var(np.diff(out[..., 0].astype(np.float32), axis=1)))
    if after >= before * 0.25:
        raise AssertionError(
            f'horizontal detail variance {before:.0f} -> {after:.0f}; a ds 8 '
            f'degradation should remove most of it')
    same = degrade_resolution(img, 1.0, 256)
    if not np.array_equal(same, img):
        raise AssertionError(
            'ds 1 changed the image. ds 1 is the identity on both axes, and '
            'that is the assertion the whole F/R comparison rests on')
    return f'detail variance {before:.0f} -> {after:.0f}, ds 1 is exact'


# ── 3. mother ────────────────────────────────────────────────────────────────

def t_mother_is_exactly_FStack_footprint_not_a_second_definition():
    """`CStack.mother` delegates to `FStack.footprint`. If someone re-inlines
    the formula later (a plausible "simplification"), this catches the two
    drifting apart -- which would mean 'C''s root tile is no longer the same
    rectangle as the chain's own 'F' tile at that rung."""
    for cx, cy, ds, tile in [(1000.0, 2000.0, 32.0, 256),
                              (0.0, 0.0, 4.0, 128)]:
        a = CStack.mother(cx, cy, ds, tile=tile)
        b = FStack.footprint(cx, cy, ds, tile=tile)
        if a != b:
            raise AssertionError(f'CStack.mother {a} != FStack.footprint {b}')
    return "mother(cx,cy,ds) == FStack.footprint(cx,cy,ds) at every case"


def t_mother_is_centred_and_sized_right():
    """Hand-computed: 256 tile at ds 32 is an 8192 px square centred on (cx,cy)."""
    m = CStack.mother(10_000.0, 10_000.0, 32.0, tile=256)
    if m.size_px != 8192:
        raise AssertionError(f'size_px {m.size_px} != 8192')
    if (m.x, m.y) != (10_000 - 4096, 10_000 - 4096):
        raise AssertionError(f'top-left {(m.x, m.y)} != (5904, 5904)')
    return f'8192 px square at ({m.x}, {m.y})'


def t_R_footprint_ignores_ds_and_always_equals_F_at_ds_1():
    """'R' has no growing footprint -- spec.md 3.2 'R 沒有這個代價'. Every rung
    reads the same `tile` level-0 px, degraded by different amounts; the
    footprint argument is accepted (every rung asks for it) but must not
    change anything."""
    base = FStack.footprint(500.0, 500.0, 1.0, tile=256)
    for ds in (1.0, 8.0, 32.0):
        r = RStack.footprint(500.0, 500.0, ds, tile=256)
        if r != base:
            raise AssertionError(f'RStack.footprint at ds {ds} is {r}, '
                                 f'expected the ds-1 footprint {base}')
    return "R footprint is the ds-1 footprint at every ds checked"


# ── 4. pyramid ───────────────────────────────────────────────────────────────

def t_one_transition_is_exactly_4_main_1_overlap():
    """The geometry `CStack`'s whole design leans on: `DsLadder`'s ladder steps
    exactly 2x, so every parent -> child transition tiles a `2*tile x 2*tile`
    footprint with `tile`-sized children -- always 4 main, always 1 overlap,
    regardless of which two adjacent rungs."""
    groups = CStack.pyramid(0.0, 0.0, (16.0, 32.0), tile=256)
    if set(groups) != {16.0}:
        raise AssertionError(f'expected one child rung 16.0, got {set(groups)}')
    if len(groups[16.0]) != 1:
        raise AssertionError(f'expected 1 group, got {len(groups[16.0])}')
    g = groups[16.0][0]
    if len(g.main) != 4:
        raise AssertionError(f'expected 4 main tiles, got {len(g.main)}')
    return "4 main + 1 overlap, one transition"


def t_children_union_reconstructs_the_parent_exactly():
    """Union of a rung's main tiles == the parent's own rectangle, in INTEGER
    level-0 px -- not a tolerance. This is the property `demo_survival_
    analysis.py`'s chains_stack part prints as its "reconstruction check";
    here it is a real assertion instead of something a human has to read off
    a printout."""
    groups = CStack.pyramid(12_345.0, 6_789.0, (1.0, 2.0, 4.0, 8.0), tile=256)
    for ds, gs in groups.items():
        for g in gs:
            xs = [m.x for m in g.main] + [m.x + m.size_px for m in g.main]
            ys = [m.y for m in g.main] + [m.y + m.size_px for m in g.main]
            got = (min(xs), min(ys), max(xs), max(ys))
            want = (g.parent.x, g.parent.y,
                    g.parent.x + g.parent.size_px, g.parent.y + g.parent.size_px)
            if got != want:
                raise AssertionError(
                    f'ds {ds}: children union {got} != parent {want}')
    return f"{sum(len(gs) for gs in groups.values())} groups reconstruct exactly"


def t_overlap_shares_exactly_a_quarter_with_each_of_its_four_mains():
    """The geometric definition 相依型 comparison is built on: the overlap
    tile's intersection with EACH of its four main neighbours is exactly
    `size_px**2 / 4`, computed by hand from the two rectangles -- not asserted
    via `PatchGrid` again, which would just check the library agrees with
    itself."""
    groups = CStack.pyramid(0.0, 0.0, (16.0, 32.0), tile=256)
    g = groups[16.0][0]
    o = g.overlap
    for m in g.main:
        ix = max(0, min(o.x + o.size_px, m.x + m.size_px) - max(o.x, m.x))
        iy = max(0, min(o.y + o.size_px, m.y + m.size_px) - max(o.y, m.y))
        area = ix * iy
        want = (o.size_px ** 2) / 4
        if area != want:
            raise AssertionError(
                f'overlap {o} vs main {m}: intersection {area} != {want}')
    return f"overlap {o.size_px}x{o.size_px} shares 1/4 with all 4 mains"


def t_pyramid_grows_4x_per_transition():
    """`4 ** k` main tiles after `k` transitions -- exponential BY DESIGN
    (`ChainStack.py`'s `CStack` docstring), not something that should ever
    read as a bug. Three transitions here: 4, 16, 64."""
    groups = CStack.pyramid(0.0, 0.0, (4.0, 8.0, 16.0, 32.0), tile=256)
    counts = {ds: sum(len(g.main) for g in gs) for ds, gs in groups.items()}
    want = {16.0: 4, 8.0: 16, 4.0: 64}
    if counts != want:
        raise AssertionError(f'main-tile counts {counts} != {want}')
    return f"{counts}"


def t_a_non_2x_step_is_refused_not_silently_regrouped():
    """Skipping a rung (e.g. asking for 4 -> 32 directly, a 8x step) would need
    per-corner grouping `_children_of` does not implement -- see the class
    docstring. It must raise, not produce a group whose `main` no longer means
    "this overlap's four partners"."""
    try:
        CStack.pyramid(0.0, 0.0, (4.0, 32.0), tile=256)
    except ValueError:
        return "raised on an 8x step, as required"
    raise AssertionError('a non-2x step should have raised ValueError')


# ── 5. nearest ───────────────────────────────────────────────────────────────

def t_nearest_is_by_centre_distance():
    """Two non-overlapping tiles, a point closer to each in turn -- confirms
    `nearest` picks by CENTRE distance and not, say, the first tile in the
    list or the one whose top-left is closest. Built from hand-placed
    `PatchInfo`, not from `pyramid()`, so the test does not depend on the
    thing it is partly there to protect.

    `RStack.derive(source='C')` picks among tiles that DO overlap (main and
    overlap tiles share a corner), where containment alone could not break
    the tie at all -- this test only pins the distance rule itself.
    """
    a = PatchInfo(row=0, col=0, x=0, y=0, size_px=100, kind='main', ds=1.0)
    b = PatchInfo(row=0, col=1, x=100, y=0, size_px=100, kind='main', ds=1.0)
    near = CStack.nearest([a, b], cx=51.0, cy=50.0)
    if near is not a:
        raise AssertionError('centre (51, 50) is closer to a (centre 50,50) '
                             'than to b (centre 150,50), got the wrong tile')
    near2 = CStack.nearest([a, b], cx=149.0, cy=50.0)
    if near2 is not b:
        raise AssertionError('centre (149, 50) should have picked b')
    return "picks by centre distance on both sides"


# ── 6. cache ─────────────────────────────────────────────────────────────────

def t_the_local_cache_round_trips():
    """`_cache_put` then `_cache_get` returns the same pixels -- the primitive
    `CStack.read_one`/`RStack.derive(source='C')` depend on to avoid a second
    WSI hit. Lossless PNG, so this is exact equality, not a tolerance."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        ChainStack._cache_put(tmp, 'slideA', 100, 200, 4.0, 64, img)
        got = ChainStack._cache_get(tmp, 'slideA', 100, 200, 4.0, 64)
        if got is None or not np.array_equal(got, img):
            raise AssertionError('round-trip did not return the same image')
        miss = ChainStack._cache_get(tmp, 'slideA', 999, 999, 4.0, 64)
        if miss is not None:
            raise AssertionError('a position never written should miss')
    return "put -> get round-trips exactly, a miss stays a miss"


def t_cache_root_none_disables_the_cache_rather_than_erroring():
    """`cache_root=None`/`''` is a valid choice (every caller can opt out), and
    it must be a silent no-op, not a crash on a `None` path."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    ChainStack._cache_put(None, 'slideA', 0, 0, 1.0, 32, img)   # must not raise
    if ChainStack._cache_get(None, 'slideA', 0, 0, 1.0, 32) is not None:
        raise AssertionError('cache_root=None must never return a hit')
    return "cache_root=None is a no-op on both put and get"


# ── 7. own ───────────────────────────────────────────────────────────────────

def t_from_tile_is_identity_at_ds_1_and_compounds_above_base_rung():
    """`base_rung` does NOT spare rungs up to itself -- only `ds<=1.0` is ever
    the untouched tile: `base_rung=4` still fully re-degrades at `ds=4`, it does not
    treat `ds<=base_rung` as "already there". Checked against the SAME `ds=4`
    built from `base_rung=1` -- the `base_rung=4` one must lose MORE detail,
    not the same amount, because it degrades an already-blurred tile again.

    `from_tile` never reads `base_rung` in its own arithmetic -- it is a label
    the CALLER asserts about `image`'s own provenance (see the docstring), so
    the test has to actually hand it an already-degraded image for
    `base_rung=4` to mean anything; passing the same sharp noise for both
    makes the two calls bitwise
    identical and asserts nothing about compounding at all.
    """
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    already_ds4 = degrade_resolution(img, 4.0, 256)   # a REAL base_rung=4 tile

    out = RStack.from_tile(already_ds4, 4.0, [1.0, 4.0, 8.0], tile=256)
    if not np.array_equal(out[1.0], already_ds4):
        raise AssertionError('ds 1 must be the identity regardless of base_rung')

    def _detail(a):
        return float(np.var(np.diff(a[..., 0].astype(np.float32), axis=1)))

    sharp_ds4 = RStack.from_tile(img, 1.0, [4.0], tile=256)[4.0]
    blurred_ds4 = out[4.0]
    if _detail(blurred_ds4) >= _detail(sharp_ds4):
        raise AssertionError(
            f'base_rung=4 at ds=4 (detail {_detail(blurred_ds4):.0f}) should '
            f'be MORE degraded than base_rung=1 at ds=4 '
            f'(detail {_detail(sharp_ds4):.0f}), not the same tile undegraded')
    return f'ds1 identity, ds4 compounds past the base_rung=1 baseline'


def t_footprint_base_rung_reports_the_true_window():
    """`RStack.footprint(..., base_rung=B)` must equal `FStack.footprint` at
    `B`, not always the ds-1 window -- otherwise a `base_rung=4` stack would
    claim a footprint the array does not actually have."""
    a = RStack.footprint(500.0, 500.0, 8.0, tile=256, base_rung=4.0)
    b = FStack.footprint(500.0, 500.0, 4.0, tile=256)
    if a != b:
        raise AssertionError(f'{a} != {b}')
    return f'base_rung=4 footprint == FStack.footprint at ds 4'


def t_derive_refuses_own_rather_than_guessing():
    """`source='own'` cannot fit `derive`'s signature (no `Chain` -- see the
    class docstring) -- it must say so, not silently do something with a
    `chain` argument it will never receive from an own tile."""
    try:
        RStack.derive(None, [1.0], tile=256, source='own')
    except ValueError as e:
        if 'from_own' not in str(e):
            raise AssertionError(f'error does not point at from_own: {e}')
        return "raised ValueError pointing at from_own"
    raise AssertionError("source='own' should have raised")


@contextlib.contextmanager
def _tree():
    """`Cache.RESULT_DIR` pointed at a temporary directory for the length of a
    test; yields the job the corpora are written under."""
    import Cache                                                  # noqa: PLC0415
    with tempfile.TemporaryDirectory() as tmp:
        saved, Cache.RESULT_DIR = Cache.RESULT_DIR, tmp
        try:
            yield 'TestChainStack'
        finally:
            Cache.RESULT_DIR = saved


def _corpus(root, sampler_id):
    """A corpus address in job `root`'s tree; only `sampler_id` varies
    between tests."""
    return PreTileCorpus(root, 'seg0', 'reg0', sampler_id, 'ladder-test', _FACTOR)


def _make_store(root, wsi_stem, ds, tile, sampler_id, records):
    """A finished rung of `_corpus(root, sampler_id)` with `records` (list of
    kwargs for `PreTileRecord`), each tile a uniform patch coloured by its own
    index -- so a round-trip through `centre_crop`/`from_tile` is checkable by
    colour alone, the same trick `test_store.py` uses.
    """
    corpus = _corpus(root, sampler_id)
    meta = PreTileMeta(wsi_stem=wsi_stem, ds=float(ds), tile=int(tile),
                      seg_id=corpus.seg_id, region_id=corpus.region_id,
                      sampler_id=sampler_id, plan=corpus.plan, seed=0,
                      segmenter_id='deadbeef',
                      pre_tile_factor=_FACTOR, level=0, level_ds=float(ds),
                      shrink=1.0, read_size=pre_tile_px(tile, _FACTOR))
    folder = PreTileStore.create(corpus, meta)
    out = []
    pre = pre_tile_px(tile, _FACTOR)
    for kwargs in records:
        rec = PreTileRecord(**kwargs)
        image = np.full((pre, pre, 3), rec.index, dtype=np.uint8)
        PreTileStore.save_tile(folder, rec, image, meta)
        out.append(rec)
    PreTileStore.write_index(folder, out)
    return folder, meta


def t_FStack_from_own_is_lazy_and_reads_a_chain_on_getitem():
    """`chains()` (metadata only) finds the chain; `x[inherit_id]` is where
    `FStack.read` -- a real pixel read -- actually happens."""
    with _tree() as root:
        _make_store(root, 'SLIDE_A', 1.0, _TILE, 'aaaa1111',
                   [dict(index=0, x=1000, y=1000, inherit_id=7)])
        _make_store(root, 'SLIDE_A', 2.0, _TILE, 'aaaa1111',
                   [dict(index=0, x=900, y=900, inherit_id=7)])
        own = FStack.from_own(_corpus(root, 'aaaa1111'), 'SLIDE_A', tile=_TILE, rungs=[1.0, 2.0])
        if len(own) != 1 or list(own) != [7]:
            raise AssertionError(f'expected one chain keyed 7, got {list(own)}')
        stack = own[7]
        if set(stack) != {1.0, 2.0}:
            raise AssertionError(f'expected rungs {{1.0, 2.0}}, got {set(stack)}')
        if stack[1.0][0, 0, 0] != 0 or stack[2.0][0, 0, 0] != 0:
            raise AssertionError('pixels did not round-trip (both records are index 0)')
    return "1 chain found, x[7] reads both rungs"


def t_RStack_from_own_scans_every_rung_not_just_one():
    """own's batch can span more than one base_rung in a single run -- both
    must show up, not just whichever folder `find` happens to see first."""
    with _tree() as root:
        _make_store(root, 'SLIDE_A', 1.0, _TILE, 'bbbb2222',
                   [dict(index=0, x=0, y=0), dict(index=1, x=200, y=200)])
        _make_store(root, 'SLIDE_A', 4.0, _TILE, 'bbbb2222',
                   [dict(index=0, x=1000, y=1000)])
        own = RStack.from_own(_corpus(root, 'bbbb2222'), 'SLIDE_A', [1.0, 4.0], tile=_TILE)
        if len(own) != 3:
            raise AssertionError(f'expected 3 own tiles across both rungs, got {len(own)}')
        stacks = [own[i] for i in own]
        if not all(set(s) == {1.0, 4.0} for s in stacks):
            raise AssertionError('every own tile should produce both output rungs')
    return "3 records across ds 1/ds 4, each -> a 2-rung stack"


def t_RStack_from_own_cache_off_by_default_on_when_asked():
    """`cache_root` defaults to `None` (unlike every other `cache_root` in
    this module) -- degrade is cheap, see `from_own`'s docstring. Passing one
    must make a second access hit the cache instead of recomputing."""
    with _tree() as root:
        cache_root = 'TestChainStackTiles'
        tile_dir = ChainStack._cache_dir(cache_root, 'SLIDE_A')
        _make_store(root, 'SLIDE_A', 1.0, _TILE, 'cccc3333',
                   [dict(index=5, x=300, y=400)])
        off = RStack.from_own(_corpus(root, 'cccc3333'), 'SLIDE_A', [1.0, 2.0], tile=_TILE)
        off[0]
        if ChainStack._cache_get(tile_dir, 'SLIDE_A', 300, 400, 2.0, _TILE) is not None:
            raise AssertionError('a fresh cache_root must start empty')
        on = RStack.from_own(_corpus(root, 'cccc3333'), 'SLIDE_A', [1.0, 2.0], tile=_TILE,
                             cache_root=cache_root)
        on[0]
        if ChainStack._cache_get(tile_dir, 'SLIDE_A', 300, 400, 2.0, _TILE) is None:
            raise AssertionError('cache_root was given but nothing was written')
    return "cache_root=None writes nothing, cache_root=<job> does"


def t_CStack_from_own_builds_forest_geometry_without_a_wsi():
    """The whole point of building geometry up front: `from_own` must not
    need a real `wsi` at all (only `__getitem__` does, for the descendants) --
    passing `None` and never touching it proves the split is real, not just
    documented."""
    with _tree() as root:
        _make_store(root, 'SLIDE_A', 16.0, _TILE, 'dddd4444',
                   [dict(index=0, x=10_000, y=10_000)])
        forest = CStack.from_own(_corpus(root, 'dddd4444'), 'SLIDE_A', [4.0, 8.0, 16.0], None,
                                 tile=_TILE)
        if len(forest) != 1:
            raise AssertionError(f'expected 1 tree, got {len(forest)}')
        _folder, _record, meta, mother, groups_by_ds = forest.items[0]
        if mother.ds != 16.0:
            raise AssertionError(f'mother ds {mother.ds} != the record\'s own 16.0')
        want = CStack.pyramid(*mother_centre(mother), [4.0, 8.0, 16.0], tile=_TILE)
        if set(groups_by_ds) != set(want):
            raise AssertionError(f'{set(groups_by_ds)} != {set(want)}')
    return "1 tree, geometry matches CStack.pyramid, no wsi touched"


def mother_centre(mother):
    return (mother.x + 0.5 * mother.size_px, mother.y + 0.5 * mother.size_px)


_SECTIONS = {
    'scale':   ['t_scale_and_shrink_agree_on_F_and_disagree_on_R',
               't_C_scale_and_shrink_match_F_not_R'],
    'degrade': ['t_the_r_degradation_loses_detail_and_keeps_the_frame'],
    'mother':  ['t_mother_is_exactly_FStack_footprint_not_a_second_definition',
               't_mother_is_centred_and_sized_right',
               't_R_footprint_ignores_ds_and_always_equals_F_at_ds_1'],
    'pyramid': ['t_one_transition_is_exactly_4_main_1_overlap',
               't_children_union_reconstructs_the_parent_exactly',
               't_overlap_shares_exactly_a_quarter_with_each_of_its_four_mains',
               't_pyramid_grows_4x_per_transition',
               't_a_non_2x_step_is_refused_not_silently_regrouped'],
    'nearest': ['t_nearest_is_by_centre_distance'],
    'cache':   ['t_the_local_cache_round_trips',
               't_cache_root_none_disables_the_cache_rather_than_erroring'],
    'own':     ['t_from_tile_is_identity_at_ds_1_and_compounds_above_base_rung',
               't_footprint_base_rung_reports_the_true_window',
               't_derive_refuses_own_rather_than_guessing',
               't_FStack_from_own_is_lazy_and_reads_a_chain_on_getitem',
               't_RStack_from_own_scans_every_rung_not_just_one',
               't_RStack_from_own_cache_off_by_default_on_when_asked',
               't_CStack_from_own_builds_forest_geometry_without_a_wsi'],
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', choices=sorted(_SECTIONS))
    args = ap.parse_args()

    for section in (args.only or list(_SECTIONS)):
        print(f'\n[{section}]')
        for name in _SECTIONS[section]:
            check(name[2:].replace('_', ' '), globals()[name])

    failed = [n for n, e in _RESULTS if e is not None]
    print(f'\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed')
    if failed:
        print('failed: ' + ', '.join(failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
