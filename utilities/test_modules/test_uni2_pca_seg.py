#!/usr/bin/env python3
"""Tests for aiNNModel/Uni2PcaSegFunc.py.

    python utilities/test_modules/test_uni2_pca_seg.py
    python utilities/test_modules/test_uni2_pca_seg.py --with-model
    python utilities/test_modules/test_uni2_pca_seg.py --with-model --wsi slide.svs

Three tiers, and only the first runs everywhere:

    (default)     config, identity, and the sampling helpers. No weights, no
                  GPU, no slide. Seconds.
    --with-model  builds the encoder, so it downloads/loads UNI2. Still no
                  slide: what it checks is the shape of one forward and the
                  refusal to run unfitted.
    --wsi         fits on a real slide and then asks the question this module
                  exists to answer.

WHAT THE --wsi TIER IS FOR
---------------------------
One assertion carries this module: **after fit(wsi), segmenting a plane in one
call and segmenting it in four quadrants must agree pixel for pixel.**

That is the property `from_wsi` already documents as the dividing line between
methods it may tile and methods it may not -- "_mask_otsu derives its threshold
globally, and per tile it would threshold blank glass against its own noise" --
and it is the entire reason the PCA is fitted in `fit` rather than in
`__call__`. If it holds, `seg_chunk_px` and `read_chunk_px` are sound here. If
it does not, every mask produced with those set has seams that nothing reports.

The decoy is the design that was rejected: refit the PCA on each quadrant's own
features, which is what a lazily-fitting `__call__` would do. It is scored
alongside, so the check reports a MARGIN rather than passing a tolerance -- and
the margin is the measurement of how wrong the rejected design would have been.

WHAT IS DELIBERATELY NOT ASSERTED
----------------------------------
That the mask is correct. There is no tissue ground truth here, and inventing a
threshold on the foreground fraction would be exactly the guess
ClaudeRules section 8 forbids. What the --wsi tier does instead is PRINT
`fit_report` next to the tissue fractions this project has already measured
(BRACS 38.2 / 24.4 / 23.1 percent, Ki67 9.2 / 5.9 / 4.6 / 3.5) and assert only
that the result is not degenerate. A number near 0.5 on a Ki67 slide means the
PCA found something other than tissue -- position, or scanner banding -- and
that is for a human to look at, not for an assert to decide.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np                                              # noqa: E402

from _paths import setup_import_paths                           # noqa: E402

setup_import_paths()

from ConfigIdentity import config_from, registered              # noqa: E402
from Uni2PcaSegFunc import (Uni2PcaSegConfig,                   # noqa: E402
                            Uni2PcaSegmenter, _UNI2_PCA_BASELINE,
                            scanned_bounds, stratified_positions,
                            tile_saturation)

_RESULTS = []


def check(name, fn):
    try:
        out = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   {out}' if out else ''))
    except Exception as e:                                       # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n          {type(e).__name__}: {e}')


# ══════════════════════════════════════════════════════════════════════════════
#  a slide that is not a slide
# ══════════════════════════════════════════════════════════════════════════════

class _FakeSlide:
    """Enough of an OpenSlide for the helpers that only do arithmetic on it.

    Not a mock of the model -- there is no model in this tier. It exists so that
    `scanned_bounds` and `tile_saturation` can be checked against an image whose
    answer is known by construction, which no real slide provides.
    """

    def __init__(self, image: np.ndarray, bounds=None):
        self._image = image
        height, width = image.shape[:2]
        self.level_dimensions = [(width, height)]
        self.level_downsamples = [1.0]
        self.properties = {}
        if bounds is not None:
            x, y, w, h = bounds
            self.properties = {'openslide.bounds-x': str(x),
                               'openslide.bounds-y': str(y),
                               'openslide.bounds-width': str(w),
                               'openslide.bounds-height': str(h)}

    def get_best_level_for_downsample(self, downsample):
        return 0

    def read_region_rgb(self, location, level, size):
        x, y = location
        w, h = size
        return self._image[y:y + h, x:x + w]


def _split_image(size=512):
    """Left half strongly coloured, right half grey. Saturation knows the answer."""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, :size // 2] = (200, 40, 160)      # saturated
    image[:, size // 2:] = (150, 150, 150)     # grey: saturation 0
    return image


# ══════════════════════════════════════════════════════════════════════════════
#  1. config
# ══════════════════════════════════════════════════════════════════════════════

def t_registered_under_its_name():
    """`config_from('uni2-pca-seg')` reaches this class.

    The registry fills by import side effect, so the usual failure is not a typo
    but a module nobody imported -- which is why `TissueSegFunc` names this
    module at its own bottom. Importing THIS test imports the module directly,
    so what is actually checked here is that the decorator name and the
    `method` default agree.
    """
    assert 'uni2-pca-seg' in registered(), \
        f'not registered. Registered: {registered()}'
    cfg = config_from('uni2-pca-seg')
    assert isinstance(cfg, Uni2PcaSegConfig), type(cfg)
    assert cfg.method == 'uni2-pca-seg', (
        f"the registered name and the method default disagree: "
        f"'uni2-pca-seg' vs {cfg.method!r}. TissueSegConfig.build dispatches on "
        f"method, so they cannot differ")
    return f'{len(registered())} configs registered'


def t_every_identity_field_is_in_the_baseline():
    """A field missing from BASELINE re-hashes every store ever written.

    `parts_against` includes any field ABSENT from the baseline unconditionally
    -- there is nothing for it to equal -- so forgetting an entry does not
    silently drop the field, it silently invalidates every existing id.
    ConfigIdentity's own docstring calls that "a recompute, not a wrong answer",
    which is survivable and still worth catching before a 20 GB cache is
    rebuilt for nothing.

    NOT_IDENTITY is the other side: those must NOT be in the baseline, or the
    zero point would name a field that is never compared against it.
    """
    fields = {f.name for f in dataclasses.fields(Uni2PcaSegConfig)}
    skipped = set(Uni2PcaSegConfig.NOT_IDENTITY)
    baseline = set(_UNI2_PCA_BASELINE)

    missing = fields - skipped - baseline
    assert not missing, f'fields with no baseline entry: {sorted(missing)}'
    stray = baseline - fields
    assert not stray, f'baseline names fields the config does not have: {sorted(stray)}'
    overlap = baseline & skipped
    assert not overlap, f'NOT_IDENTITY fields are in the baseline: {sorted(overlap)}'
    return f'{len(baseline)} identity fields, {len(skipped)} excluded'


def t_every_identity_field_moves_the_hash():
    """Change any hashed field and the parts change; change batch_tiles and they do not.

    The dead-field check. A knob that no longer reaches the code it names still
    sits in the config looking applied -- this project has had three of those at
    once -- and the cheapest way to notice is that it is either absent from the
    hash or present and inert. This catches the first.

    It also pins NOT_IDENTITY from the other direction: `batch_tiles` cannot
    change a single tile's features, because a ViT normalises per sample, so a
    run that only differs there must land on the same id.
    """
    base = Uni2PcaSegConfig()
    reference = base.identity_parts(_UNI2_PCA_BASELINE)

    unmoved = []
    for field in dataclasses.fields(Uni2PcaSegConfig):
        value = getattr(base, field.name)
        if isinstance(value, bool):
            bumped = not value
        elif isinstance(value, float):
            bumped = value * 2 + 1
        elif isinstance(value, int):
            bumped = value + 1
        elif isinstance(value, str):
            bumped = value + '_x'
        else:
            continue
        parts = dataclasses.replace(base, **{field.name: bumped}) \
            .identity_parts(_UNI2_PCA_BASELINE)
        moved = parts != reference
        if field.name in Uni2PcaSegConfig.NOT_IDENTITY:
            assert not moved, (
                f'{field.name} is in NOT_IDENTITY but changing it moved the '
                f'hash; one of the two is wrong')
        elif not moved:
            unmoved.append(field.name)

    assert not unmoved, (
        f'these fields do not reach the identity: {unmoved}. Either add them to '
        f'NOT_IDENTITY with the reason, or find out why parts_against skips them')
    return f'{len(dataclasses.fields(Uni2PcaSegConfig))} fields, all accounted for'


# ══════════════════════════════════════════════════════════════════════════════
#  2. helpers
# ══════════════════════════════════════════════════════════════════════════════

def t_scanned_bounds_honours_limit_bounds():
    """With bounds-* set, the rectangle is the scanned one; without, the canvas.

    Why it matters here rather than being a formality: on a MIRAX the canvas
    outside `openslide.bounds-*` is stage travel range with no image data. A ViT
    handed a blank tile has only its positional embedding varying between
    patches, so a PCA fitted on those finds POSITION, and the mask comes out in
    bands.
    """
    slide = _FakeSlide(np.zeros((400, 600, 3), np.uint8), bounds=(50, 30, 200, 100))
    assert scanned_bounds(slide, True) == ((50, 30), (200, 100))
    assert scanned_bounds(slide, False) == ((0, 0), (600, 400))

    bare = _FakeSlide(np.zeros((400, 600, 3), np.uint8))
    assert scanned_bounds(bare, True) == ((0, 0), (600, 400)), \
        'a slide with no bounds-* should fall back to the whole canvas'
    return 'bounds respected, absent bounds fall back to the canvas'


def t_tile_saturation_finds_the_coloured_half():
    """Saturation per tile position, against an image built to have an answer."""
    image = _split_image(512)
    slide = _FakeSlide(image)
    saturation = tile_saturation(slide, (0, 0), (512, 512), tile=128, ds=1.0)

    assert saturation.shape == (4, 4), saturation.shape
    left, right = saturation[:, :2], saturation[:, 2:]
    assert left.min() > 0.5, f'the coloured half read {left.min():.3f}'
    assert right.max() < 0.05, f'the grey half read {right.max():.3f}'
    return (f'coloured half {left.mean():.2f}, grey half {right.mean():.3f}')


def t_stratified_covers_every_band_where_uniform_misses_some():
    """What stratifying actually buys: no quantile band goes unrepresented.

    THIS CHECK REPLACES ONE THAT ASSERTED SOMETHING FALSE, and the correction is
    worth keeping. The first version claimed stratifying oversamples a rare
    class, and measured 3.5 percent against uniform's 2.7 -- barely a
    difference, because there is no difference to find:

        equal picks per QUANTILE bin is, in expectation, the same composition as
        uniform sampling. A class covering 3.5 percent of the area lands
        entirely in the top bin, is 35 percent OF that bin, and that bin takes
        10 percent of the picks -- so 3.5 percent of the sample, identically.

    The docstring's discussion of tissue fractions is the MOTIVATION for why
    blank tiles dominate; it is not a claim that quantile binning fixes the
    imbalance, and reading it as one was the error.

    What quantile binning does give is guaranteed COVERAGE at small n: every
    band contributes, where a uniform draw of the same size leaves bands empty
    by chance. That is a variance property, and it is testable.

    THE DECOY IS UNIFORM SAMPLING of the same count, averaged over trials so the
    comparison is not one lucky draw.
    """
    rng = np.random.default_rng(0)
    grid = rng.random((100, 100)).astype(np.float32)
    bins = 10
    edges = np.quantile(grid, np.linspace(0, 1, bins + 1)[1:-1])

    picked = stratified_positions(grid, 15, bins=bins, seed=0)
    covered = len(np.unique(np.digitize([grid[r, c] for r, c in picked], edges)))

    trials = 200
    uniform_covered = []
    for _ in range(trials):
        idx = rng.choice(grid.size, size=len(picked), replace=False)
        uniform_covered.append(
            len(np.unique(np.digitize(grid.ravel()[idx], edges))))
    decoy = float(np.mean(uniform_covered))

    assert covered == bins, (
        f'stratified left {bins - covered} of {bins} quantile bands empty; '
        f'covering all of them is the one thing it is for')
    assert decoy < bins - 0.5, (
        f'uniform sampling of {len(picked)} covered {decoy:.1f} of {bins} bands '
        f'on average, which is essentially all of them -- so this grid is too '
        f'small a test to tell the two apart')
    return (f'{len(picked)} positions: stratified {covered}/{bins} bands, '
            f'uniform {decoy:.1f}/{bins} over {trials} trials')


def t_stratified_is_deterministic():
    """Same seed, same positions. The fit basis depends on this.

    `fit_seed` is in the identity hash, so a config claiming to name a fit has
    to actually determine it. If this were nondeterministic, two runs would
    share an id and hold different bases -- one name, two things.
    """
    rng = np.random.default_rng(1)
    grid = rng.random((60, 80)).astype(np.float32)
    a = stratified_positions(grid, 200, bins=10, seed=7)
    b = stratified_positions(grid, 200, bins=10, seed=7)
    c = stratified_positions(grid, 200, bins=10, seed=8)
    assert a == b, 'same seed gave different positions'
    assert a != c, 'different seeds gave identical positions'
    return f'{len(a)} positions, stable under the seed'


def t_read_rgb_refuses_a_plain_openslide():
    """A handle without read_region_rgb is refused, by name.

    `.convert("RGB")` only drops alpha, and pixels the scanner never wrote carry
    RGB 0 -- so every hole becomes a black rectangle. A ViT reading one sees a
    hard edge that exists in no tissue, and the PCA spends a component on it.
    """
    from Uni2PcaSegFunc import _read_rgb

    class _Bare:
        pass

    try:
        _read_rgb(_Bare(), (0, 0), 0, (16, 16))
    except TypeError as e:
        assert 'SafeSlide' in str(e), f'unhelpful message: {e}'
        return 'refused, and names SafeSlide'
    raise AssertionError('a handle with no read_region_rgb was accepted')


# ══════════════════════════════════════════════════════════════════════════════
#  3. model  (--with-model; loads UNI2, needs no slide)
# ══════════════════════════════════════════════════════════════════════════════

def t_unfitted_call_refuses(seg):
    """__call__ before fit must raise, and say what to call.

    The one failure mode a caller can hit by simply forgetting a line. Fitting
    lazily instead would be the rejected design, and it would produce a mask
    rather than an error.
    """
    assert not seg.fitted, 'this check needs an unfitted segmenter'
    try:
        seg(np.zeros((seg.cfg.tile, seg.cfg.tile, 3), np.uint8))
    except RuntimeError as e:
        assert 'fit(' in str(e), f'the message does not name fit(): {e}'
        return 'raises RuntimeError and names fit(wsi)'
    raise AssertionError('an unfitted segmenter produced a mask')


def t_cells_have_the_shape_the_grid_implies(seg):
    """One forward: [B, tile, tile, 3] -> [B, cells_per_side**2, D].

    Cheap, and it is the assertion that would catch the prefix tokens being
    mis-stripped. UNI2 carries nine, and dropping the wrong number still
    divides cleanly into a plausible grid -- `forward_intermediates` is used
    precisely so nobody has to get that right by hand, and this pins that it
    did.
    """
    import torch

    tile = seg.cfg.tile
    batch = np.zeros((2, tile, tile, 3), dtype=np.uint8)
    cells = seg._cells(batch)

    expected = seg._cells_per_side ** 2
    assert cells.shape[0] == 2, cells.shape
    assert cells.shape[1] == expected, (
        f'{cells.shape[1]} cells for a {tile} px tile, expected {expected} '
        f'({seg._cells_per_side}x{seg._cells_per_side} at {seg._cell_px} px)')
    assert cells.dtype == torch.float32, cells.dtype
    return (f'[2, {cells.shape[1]}, {cells.shape[2]}] at {seg._cell_px} px '
            f'per cell')


def t_mask_ds_is_the_patch_size(seg):
    """Everything reads at level 0, so the mask lands at ds = cell_px = 14.

    Not a parameter and not a choice: `slide_pca_mask`'s "nothing here chooses a
    resolution, the patch grid does". A `plane_ds` field and then a `level`
    argument both used to make this answerable only after a fit, and both are
    gone -- `Uni2PcaSegFunc.LEVEL` records why and the measurement that allowed
    it.

    14 is FINER than the ds 32 hsv masks in use, which is what makes the
    granularity worry moot: a 256 px tile at ds 1 spans 18 cells, so
    `tissue_ratio` has room to be a knob.
    """
    from Uni2PcaSegFunc import LEVEL

    assert LEVEL == 0, f'LEVEL is {LEVEL}; this check assumes level 0'
    assert seg.mask_ds == seg._cell_px, (seg.mask_ds, seg._cell_px)
    tile_at_ds1 = 256
    return (f'ds {seg.mask_ds:.0f} per cell; a {tile_at_ds1} px tile at ds 1 '
            f'spans {tile_at_ds1 / seg.mask_ds:.0f} cells')


# ══════════════════════════════════════════════════════════════════════════════
#  4. slide  (--wsi; fits, then the assertion this module exists for)
# ══════════════════════════════════════════════════════════════════════════════

def t_fit_report_is_not_degenerate(seg):
    """Print what the fit saw; assert only that it is not vacuous.

    No tissue ground truth exists here, so "the mask is right" is not assertable
    and a threshold on the foreground fraction would be a guess. What IS
    assertable is that the PCA separated something: a fraction of exactly 0 or
    exactly 1 means PC1 landed entirely on one side of the threshold and the
    mask carries no information at all.

    The measured tissue fractions on this project's slides, for reading the
    printed number against: BRACS 38.2 / 24.4 / 23.1 percent, Ki67 9.2 / 5.9 /
    4.6 / 3.5 percent.
    """
    report = seg.fit_report
    assert report is not None, 'fit() did not record a report'
    print(f'        level {report["level"]} (ds {report["level_ds"]:.4g}), '
          f'{report["positions"]} positions, {report["cells"]} cells')
    print(f'        explained variance, first three: '
          f'{report["explained_variance_top3"]:.1%}')
    print(f'        foreground fraction in the fit sample: '
          f'{report["foreground_fraction_in_sample"]:.1%}   '
          f'(measured tissue: BRACS 23-38%, Ki67 3.5-9%)')
    print(f'        mask_ds {report["mask_ds"]:.0f} level-0 px per cell')

    fraction = report['foreground_fraction_in_sample']
    assert 0.0 < fraction < 1.0, (
        f'PC1 put every sampled cell on one side of the threshold '
        f'({fraction:.3f}); the mask carries no information. Try the other '
        f'larger_pca_as_fg, or a different background_threshold')
    return f'foreground {fraction:.1%}, explained {report["explained_variance_top3"]:.1%}'


def t_tiling_is_invariant_and_refitting_is_not(seg, wsi, args):
    """One call over a plane == four calls over its quadrants. THE check.

    This is the property that makes `from_wsi`'s `seg_chunk_px` and
    `read_chunk_px` paths sound for this segmenter, and it holds only because
    the PCA was fitted in `fit`. `from_wsi` documents the boundary itself:
    "_mask_otsu derives its threshold globally, and per tile it would threshold
    blank glass against its own noise ... _mask_hsv is per-pixel and HEST is
    fully convolutional, so both are unaffected."

    THE DECOY is the design that was rejected -- refit the PCA on each
    quadrant's own features, which is what a lazily-fitting `__call__` would do.
    Scoring it alongside turns "the pre-fitted version agrees" into a MARGIN:
    the disagreement of the decoy is the measurement of how wrong that design
    would have been, and without it a passing test would only mean the two calls
    happened to be equal.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler
    from Uni2PcaSegFunc import _GpuProjection

    cfg = seg.cfg
    level = seg.fit_report['level']
    side = cfg.tile * args.plane_tiles          # even split into quadrants
    origin, span = scanned_bounds(wsi, cfg.limit_bounds)
    level_ds = seg.fit_report['level_ds']

    # A plane over the middle of the scanned rectangle, where there is content.
    x = origin[0] + int(span[0] / 2 - side * level_ds / 2)
    y = origin[1] + int(span[1] / 2 - side * level_ds / 2)
    plane = wsi.read_region_rgb((x, y), level, (side, side))

    whole = seg(plane)
    assert whole.shape == plane.shape[:2], (whole.shape, plane.shape)

    half = side // 2
    stitched = np.zeros_like(whole)
    for row in (0, 1):
        for col in (0, 1):
            quadrant = plane[row * half:(row + 1) * half,
                             col * half:(col + 1) * half]
            stitched[row * half:(row + 1) * half,
                     col * half:(col + 1) * half] = seg(quadrant)

    disagree = int((whole != stitched).sum())

    # The decoy: the same four quadrants, each with a PCA fitted on itself.
    refit = np.zeros_like(whole)
    saved = seg._project
    try:
        for row in (0, 1):
            for col in (0, 1):
                quadrant = plane[row * half:(row + 1) * half,
                                 col * half:(col + 1) * half]
                tiles = np.stack([
                    quadrant[r * cfg.tile:(r + 1) * cfg.tile,
                             c * cfg.tile:(c + 1) * cfg.tile]
                    for r in range(half // cfg.tile)
                    for c in range(half // cfg.tile)])
                cells = seg._cells(tiles).cpu().numpy()
                cells = cells.reshape(-1, cells.shape[-1])
                pca = PCA(n_components=cfg.components,
                          random_state=cfg.fit_seed).fit(cells)
                scaler = MinMaxScaler(clip=True).fit(pca.transform(cells))
                seg._project = _GpuProjection(pca, scaler, seg.device)
                refit[row * half:(row + 1) * half,
                      col * half:(col + 1) * half] = seg(quadrant)
    finally:
        seg._project = saved

    decoy_disagree = int((whole != refit).sum())
    total = whole.size

    assert disagree == 0, (
        f'{disagree} of {total} pixels ({100 * disagree / total:.2f}%) differ '
        f'between one call and four. The PCA is not being shared across calls, '
        f'so from_wsi with seg_chunk_px set would produce seams and report '
        f'nothing')
    assert decoy_disagree > 0, (
        f'refitting the PCA per quadrant changed nothing either, so this check '
        f'cannot tell a shared basis from a per-tile one. Either the plane is '
        f'too uniform to fit differently -- try a larger --plane-tiles -- or '
        f'the projection is not being applied at all')
    return (f'{side}x{side} plane: pre-fitted 0/{total} pixels differ, '
            f'per-quadrant refit {decoy_disagree}/{total} '
            f'({100 * decoy_disagree / total:.1f}%)')


def t_fit_report_agrees_with_mask_ds(seg, wsi, args):
    """One granularity, one number, whichever way it is asked for.

    Two places compute it -- the property from `_cell_px * level_ds`, the report
    from the same -- and a quantity computed twice is one that eventually
    differs in one of them. They agree here because there is now only one input,
    the level the fit actually read; before `plane_ds` was removed the property
    used the REQUEST and the report used the resolution, and on BRACS those are
    never the same number.
    """
    report = seg.fit_report
    assert abs(seg.mask_ds - report['mask_ds']) < 1e-9, (
        f'mask_ds says {seg.mask_ds} and fit_report says {report["mask_ds"]}; '
        f'they are the same quantity')
    assert abs(seg.mask_ds - seg._cell_px * report['level_ds']) < 1e-9
    return (f'{seg.mask_ds:.0f} level-0 px per cell   '
            f'(level {report["level"]}, ds {report["level_ds"]:.4g}, '
            f'cell {seg._cell_px} px)')


def t_mask_is_binary_and_full_size(seg, wsi, args):
    """uint8, values in {0, 1}, and the input's spatial size.

    `_tiled_apply:307-311` slices the returned mask with the INPUT tile's pixel
    offsets and assigns into a full-size array, so anything smaller raises there
    on broadcast -- while the single-call path would have accepted a coarse mask
    and derived a coarser ds from its shape. A cell-resolution return would
    therefore work until someone passed `seg_chunk_px`; this pins that it does
    not happen.
    """
    cfg = seg.cfg
    level = seg.fit_report['level']
    origin, span = scanned_bounds(wsi, cfg.limit_bounds)
    # Deliberately NOT a multiple of the tile: the padding path has to be
    # exercised, and the crop back has to land on the requested size.
    height, width = cfg.tile * 2 + 37, cfg.tile * 3 - 11
    x = origin[0] + int(span[0] / 2)
    y = origin[1] + int(span[1] / 2)
    plane = wsi.read_region_rgb((x, y), level, (width, height))

    mask = seg(plane)
    assert mask.shape == (height, width), (mask.shape, (height, width))
    assert mask.dtype == np.uint8, mask.dtype
    assert set(np.unique(mask)) <= {0, 1}, np.unique(mask)
    return f'{width}x{height} in, {mask.shape[1]}x{mask.shape[0]} out, {mask.mean():.1%} tissue'


# ══════════════════════════════════════════════════════════════════════════════

_SECTIONS = {
    'config':  ['t_registered_under_its_name',
                't_every_identity_field_is_in_the_baseline',
                't_every_identity_field_moves_the_hash'],
    'helpers': ['t_scanned_bounds_honours_limit_bounds',
                't_tile_saturation_finds_the_coloured_half',
                't_stratified_covers_every_band_where_uniform_misses_some',
                't_stratified_is_deterministic',
                't_read_rgb_refuses_a_plain_openslide'],
}

_MODEL_CHECKS = ['t_unfitted_call_refuses', 't_cells_have_the_shape_the_grid_implies',
                 't_mask_ds_is_the_patch_size']
_SLIDE_CHECKS = ['t_fit_report_is_not_degenerate',
                 't_fit_report_agrees_with_mask_ds',
                 't_tiling_is_invariant_and_refitting_is_not',
                 't_mask_is_binary_and_full_size']


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', choices=sorted(_SECTIONS),
                    help='run only these no-model sections')
    ap.add_argument('--with-model', action='store_true',
                    help='build the encoder (loads UNI2) and check one forward')
    ap.add_argument('--wsi', help='fit on this slide and run the tiling checks; '
                                  'implies --with-model')
    ap.add_argument('--fit-level', type=int, default=None,
                    help='override the fit level. Only for the escape hatch; '
                         'the segmenter reads level 0 and has no level knob')
    ap.add_argument('--fit-tiles', type=int, default=200,
                    help='fit sample size; lower than the 1000 default so a '
                         'test is minutes rather than tens of minutes')
    ap.add_argument('--plane-tiles', type=int, default=4,
                    help='the tiling check reads a plane_tiles x plane_tiles '
                         'tile square; must be even so it splits in quadrants')
    args = ap.parse_args()

    for section in (args.only or sorted(_SECTIONS)):
        print(f'\n[{section}]')
        for name in _SECTIONS[section]:
            check(name[2:].replace('_', ' '), globals()[name])

    if args.only:
        return _report()

    if not (args.with_model or args.wsi):
        print('\n[model]  skipped -- pass --with-model to load UNI2, '
              '--wsi to fit on a slide')
        return _report()

    if args.plane_tiles % 2:
        print(f'\n--plane-tiles must be even to split into quadrants, '
              f'got {args.plane_tiles}')
        return 1

    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n[model]  device {device}')
    cfg = Uni2PcaSegConfig(fit_tiles=args.fit_tiles, workers=0)
    seg = cfg.build(device)
    print(f'         identity {seg.identity_id()}  '
          f'cells {seg._cells_per_side}x{seg._cells_per_side} '
          f'at {seg._cell_px} px')
    for name in _MODEL_CHECKS:
        check(name[2:].replace('_', ' '), lambda f=globals()[name]: f(seg))

    if not args.wsi:
        print('\n[slide]  skipped -- pass --wsi to fit and run the tiling checks')
        return _report()

    from SafeSlide import SafeSlide

    print(f'\n[slide]  {os.path.basename(args.wsi)}')
    wsi = SafeSlide(args.wsi)
    try:
        seg.fit(wsi) if args.fit_level is None else seg.fit(wsi, args.fit_level)
        for name in _SLIDE_CHECKS:
            fn = globals()[name]
            check(name[2:].replace('_', ' '),
                  (lambda f=fn: f(seg)) if name == 't_fit_report_is_not_degenerate'
                  else (lambda f=fn: f(seg, wsi, args)))
    finally:
        wsi.close()

    return _report()


def _report():
    failed = [n for n, e in _RESULTS if e is not None]
    print(f'\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed')
    if failed:
        print('failed: ' + ', '.join(failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
