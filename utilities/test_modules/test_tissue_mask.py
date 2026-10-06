#!/usr/bin/env python3
"""Tests for the three mask modules, one direction:

    TissueMaskConfig  ->  TissueSegFunc (producer)  ->  TissueMask (product)
    recipe + cache        slide -> SlideMask            SlideMask + regions

    python utilities/test_modules/test_tissue_mask.py              # synthetic only
    python utilities/test_modules/test_tissue_mask.py --wsi <slide> \\
        [--seg hsv hest uni2_pca] [--sweep] [--ops] [--tiling]      # + a figure

The synthetic half needs no weights, no GPU and no slide: the slides and the
segmenter are stand-ins, every cache lives in a temporary directory, and it
runs in seconds. It still imports torch -- the recipes module imports the HEST
and UNI2 segmenter modules, which do.

The real-slide half draws what each recipe makes of one slide, and optionally
what each region-prep step does to it (--ops), how the mask moves with the
segmentation ds (--sweep) and whether a tiled read stitches without seams
(--tiling). It asserts only what a real slide must satisfy; the rest is for
reading.

WHAT THE SYNTHETIC HALF DEFENDS
--------------------------------
    TissueMask       regions in ABSOLUTE level-0 coordinates whatever the mask's
                     origin and resolution; lengths never shifted by the origin;
                     no negative index wrapping to the far corner; views that
                     cannot leak into the mask they came from
    TissueSegFunc    the level picked by ratio, not by openslide's strict rule;
                     a tiled read that equals the whole read; a method that does
                     not run reading nothing
    TissueMaskConfig a cache that serves the wrong mask (two recipes, one
                     directory; two slides, one stem); a cache that resegments
                     for nothing (a hit that builds a segmenter; a region-only
                     change); the silent hsv default coming back
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import tempfile
from dataclasses import dataclass
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))

from _paths import job_result_dir, setup_import_paths            # noqa: E402

setup_import_paths()

import numpy as np                                               # noqa: E402

from Cache import CacheMismatch                                  # noqa: E402
from TissueMask import SlideMask, TissueMask, TissueRegion       # noqa: E402
from TissueMaskConfig import (MASK_RECIPES, MaskMaker,           # noqa: E402
                              TissueMaskConfig, add_mask_args,
                              mask_cfg_from_args)
from TissueSegFunc import (PlaneSegConfig, TissueSegConfig,      # noqa: E402
                           TissueSegmenter, plane_geometry, tiled_apply)
from ReadGeometry import nearest_level                           # noqa: E402

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
#  stand-ins
# ══════════════════════════════════════════════════════════════════════════════

class _Slide:
    """What TissueMask and the plane read ask of a slide, and nothing else.

    Not an `openslide.OpenSlide`: a real slide would make these checks depend
    on a file, a reader and a minute of IO to assert arithmetic. `plane`, when
    given, is the pixel content of every level -- one level-0 image, sampled
    down on read -- so a read at the wrong place shows as the wrong pixels.
    """

    def __init__(self, width: int, height: int, mpp: float = 0.25,
                 downsamples=(1.0, 4.0, 16.0), bounds=None, plane=None,
                 path='/data/Group_A/SLIDE_A.svs'):
        self._filename = path
        self.level_downsamples = [float(d) for d in downsamples]
        self.level_dimensions = [(int(width / d), int(height / d))
                                 for d in self.level_downsamples]
        self.level_dimensions[0] = (int(width), int(height))
        self.properties = {'openslide.mpp-x': str(mpp),
                           'openslide.mpp-y': str(mpp)}
        self.base_mpp = float(mpp)
        self.dimensions = self.level_dimensions[0]
        if bounds is not None:
            for key, value in zip(('x', 'y', 'width', 'height'), bounds):
                self.properties[f'openslide.bounds-{key}'] = str(value)
        self.plane = plane
        self.reads = []

    def get_best_level_for_downsample(self, ds):
        best = 0
        for i, d in enumerate(self.level_downsamples):
            if d <= ds:
                best = i
        return best

    def read_region_rgb(self, location, level, size):
        self.reads.append((tuple(location), level, tuple(size)))
        w, h = size
        if self.plane is None:
            out = np.zeros((h, w, 3), np.uint8)
            out[..., 0] = 200
            return out
        ds = self.level_downsamples[level]
        x0, y0 = location
        ys = (y0 + np.arange(h) * ds).astype(int).clip(0, self.plane.shape[0] - 1)
        xs = (x0 + np.arange(w) * ds).astype(int).clip(0, self.plane.shape[1] - 1)
        return self.plane[np.ix_(ys, xs)]


def blobs(H: int, W: int, boxes) -> np.ndarray:
    """A bool (H, W) mask with rectangular blobs; boxes are (row, col, h, w)."""
    mask = np.zeros((H, W), dtype=bool)
    for r, c, h, w in boxes:
        mask[r:r + h, c:c + w] = True
    return mask


def make_mask(mask: np.ndarray, ds: float, origin=(0, 0), mpp: float = 0.25,
              downsamples=(1.0,)) -> TissueMask:
    """A TissueMask over `mask` at `ds`, starting at `origin`, on a canvas that
    is exactly origin + span."""
    rows, cols = mask.shape
    span = (int(cols * ds), int(rows * ds))
    wsi = _Slide(origin[0] + span[0], origin[1] + span[1], mpp=mpp,
                 downsamples=downsamples)
    return TissueMask(wsi, SlideMask(mask=mask, origin=origin, span=span,
                                     mask_ds=float(ds)))


def boxes_of(trm):
    return sorted((r.x, r.y, r.w, r.h) for r in trm.tissue_regions)


# ══════════════════════════════════════════════════════════════════════════════
#  TissueMask -- the product
# ══════════════════════════════════════════════════════════════════════════════

def t_mask_tissue_fraction():
    half = np.zeros((10, 10), bool)
    half[:5] = True
    assert abs(make_mask(half, 1.0).tissue_fraction() - 0.5) < 1e-9
    assert make_mask(np.ones((8, 8), bool), 1.0).tissue_fraction() == 1.0
    assert make_mask(np.zeros((8, 8), bool), 1.0).tissue_fraction() == 0.0
    return '0, 0.5, 1'


def t_mask_search_finds_blobs_in_level0():
    """Mask 50x100 at ds 4, two blobs: level-0 box = mask box x 4. A blob
    under `min_area_px` is dropped."""
    a, b = (5, 10, 10, 20), (30, 60, 8, 15)
    regions = sorted(TissueMask._search_tissue_regions(
        blobs(50, 100, [a, b]), 4.0, 4.0), key=lambda r: r.x)
    assert len(regions) == 2, len(regions)
    for r, (row, col, h, w) in zip(regions, (a, b)):
        assert (r.x, r.y, r.w, r.h) == (col * 4, row * 4, w * 4, h * 4), \
            (r.x, r.y, r.w, r.h)
    tiny = TissueMask._search_tissue_regions(blobs(50, 100, [(0, 0, 5, 5)]),
                                             4.0, 4.0, min_area_px=100)
    assert tiny == [], 'a 25 px blob survived min_area_px=100'
    return '2 blobs, tiny dropped'


def t_mask_origin_offset():
    """A mask that starts somewhere other than level-0 (0, 0) -- what a MIRAX
    crop produces. Every bug it can hide is silent: regions in the wrong place,
    a size shifted by the offset because a length went through a position
    conversion, or a rect entirely before the origin reading the far corner."""
    OX, OY, DS = 1000, 500, 2.0
    trm = make_mask(blobs(20, 20, [(4, 6, 10, 12)]), DS, origin=(OX, OY))
    assert len(trm.tissue_regions) == 1
    r = trm.tissue_regions[0]
    assert (r.x, r.y) == (OX + 6 * DS, OY + 4 * DS), (r.x, r.y)
    assert (r.w, r.h) == (12 * DS, 10 * DS), 'a size carried the offset'
    assert trm.to_mask_xy(r.x, r.y) == (6, 4)
    assert trm.region_box(r) == (6, 4, 12, 10)
    assert trm._to_mask_len(r.w, r.h) == (12, 10), 'a length lost the origin twice'
    assert trm.has_tissue_l0(r.x, r.y, r.w, r.h), 'blob not found at its own bbox'
    assert not trm.has_tissue_l0(OX, OY, r.w, r.h), 'corner is 30% tissue, not 50%'

    # Entirely before the origin. Unclamped this is mask[-8:-4, -8:-4] =
    # mask[12:16, 12:16], a real slice of the far corner -- so the tissue goes
    # exactly there and the answer must still be False.
    far = make_mask(blobs(20, 20, [(12, 12, 4, 4)]), DS, origin=(OX, OY))
    assert not far.has_tissue_l0(OX - 16, OY - 16, 8, 8), \
        'a negative mask index read tissue from the opposite corner'
    return 'absolute regions, unshifted lengths, no wrap'


def t_mask_geometry_is_derived():
    """mask_ds comes from the shape against the span the mask COVERS. Both are
    silent when wrong: a ds-14 mask read as ds 32 puts every region in the wrong
    place with the right shape."""
    W0, H0 = 4480, 2240
    got = {}
    for ds in (14.0, 28.0):
        mask = blobs(int(H0 / ds), int(W0 / ds),
                     [(int(560 / ds), int(1120 / ds), int(560 / ds), int(1120 / ds))])
        trm = TissueMask(_Slide(W0, H0), SlideMask(mask, (0, 0), (W0, H0), ds))
        assert (trm.mask_ds_x, trm.mask_ds_y) == (ds, ds)
        got[ds] = boxes_of(trm)
    assert got[14.0] == [(1120, 560, 1120, 560)] == got[28.0], got

    # span is what the mask COVERS: 276 tiles of 224 inside a 61,879 px
    # rectangle. The canvas is the decoy, 0.09 percent away.
    COVERED, CANVAS = 276 * 224, 61879
    wsi = _Slide(CANVAS, 2240)
    mask = blobs(100, 276, [(0, 270, 100, 6)])
    right = TissueMask(wsi, SlideMask(mask, (0, 0), (COVERED, 2240), 224.0))
    wrong = TissueMask(wsi, SlideMask(mask, (0, 0), (CANVAS, 2240), 224.0))
    drift = abs(right.tissue_regions[0].x - wrong.tissue_regions[0].x)
    assert right.mask_ds_x == 224.0 and drift >= 50, (right.mask_ds_x, drift)

    # x and y independently: a non-square mask on a non-square span.
    c = TissueMask(_Slide(4480, 2240),
                   SlideMask(blobs(80, 320, [(20, 80, 20, 80)]), (0, 0),
                             (4480, 2240), 14.0))
    assert (c.mask_ds_x, c.mask_ds_y) == (14.0, 28.0)
    assert boxes_of(c) == [(1120, 560, 1120, 560)], 'x and y were swapped'
    assert abs(c.mask_mpp - 0.25 * 21.0) < 1e-9, c.mask_mpp
    assert (c.wsi_width, c.wsi_height) == (4480, 2240)
    assert c.wsi_level_downsamples == [1.0, 4.0, 16.0]

    # Any dtype, and the read-only broadcast a segmenter that does not run
    # hands over (0 bytes).
    ref = boxes_of(c)
    for arr in (np.asarray(c.main_mask, np.uint8), np.asarray(c.main_mask, np.uint8) * 255):
        other = TissueMask(_Slide(4480, 2240), SlideMask(arr, (0, 0), (4480, 2240), 14.0))
        assert other.main_mask.dtype == np.bool_ and boxes_of(other) == ref
    full = TissueMask(_Slide(1024, 1024),
                      SlideMask(np.broadcast_to(True, (64, 64)), (0, 0), (1024, 1024), 16.0))
    assert full.tissue_fraction() == 1.0 and full.main_mask.shape == (64, 64)
    return f'ds 14 = ds 28, canvas decoy drifts {drift} px, x/y independent'


def t_mask_tissue_gate():
    """The gate the sampler and the camera use. `>=`, so exactly half passes;
    one mask pixel less flips it; an origin does not wrap."""
    DS, TILE = 16.0, 256
    CELL = int(TILE / DS)
    mask = np.zeros((64, 64), bool)
    mask[:CELL // 2, 0:CELL] = True                     # 8/16
    mask[:CELL // 2 - 1, CELL:2 * CELL] = True          # 7/16
    mask[:CELL // 2 + 2, 2 * CELL:3 * CELL] = True      # 10/16
    trm = make_mask(mask, DS)
    assert trm.has_tissue_l0(0, 0, TILE, TILE, 0.5), 'exactly half must pass'
    assert not trm.has_tissue_l0(TILE, 0, TILE, TILE, 0.5)
    assert trm.has_tissue_l0(2 * TILE, 0, TILE, TILE, 0.5)
    mask2 = mask.copy()
    mask2[CELL // 2 - 1, 0] = False
    assert not make_mask(mask2, DS).has_tissue_l0(0, 0, TILE, TILE, 0.5), \
        'the gate is reading a different rectangle than the coordinates name'

    # Two tiles before the origin converts to mask columns -32..-16, a real
    # non-empty slice of rows 32..48 if unclamped -- so the tissue goes there.
    OX, OY = 1000, 500
    far = np.zeros((64, 64), bool)
    far[32:48, 32:48] = True
    trm3 = make_mask(far, DS, origin=(OX, OY))
    r = trm3.tissue_regions[0]
    assert (r.x, r.y) == (OX + 512, OY + 512)
    assert not trm3.has_tissue_l0(OX - 2 * TILE, OY - 2 * TILE, TILE, TILE, 0.5)
    return '8/16 passes, 7/16 not, one pixel flips it'


def t_mask_backdrop_covers_the_mask():
    """read_matching_rgb at a mask_ds that is NOT a level. At ds 32 on a 4x
    pyramid a wrong backdrop scale would still pass -- 32 is a level -- so the
    check is at ds 14, UNI2's patch grid, where it would zoom the backdrop
    3.5x under every overlay."""
    DS, W, H = 14.0, 200, 100
    wsi = _Slide(int(W * DS), int(H * DS), downsamples=(1., 4., 16., 32.))
    trm = TissueMask(wsi, SlideMask(np.ones((H, W), bool), (0, 0),
                                    (int(W * DS), int(H * DS)), DS))
    out = trm.read_matching_rgb(wsi)
    (_, lv, size) = wsi.reads[-1]
    assert lv == 1 and size == (int(round(W * DS / 4)), int(round(H * DS / 4))), (lv, size)
    assert out.shape[:2] == (H, W), out.shape
    wsi32 = _Slide(W * 32, H * 32, downsamples=(1., 4., 16., 32.))
    TissueMask(wsi32, SlideMask(np.ones((H, W), bool), (0, 0), (W * 32, H * 32),
                                32.0)).read_matching_rgb(wsi32)
    assert wsi32.reads[-1][2] == (W, H), 'the case that was already right moved'
    return f'ds 14 reads {size} at level 1'


def _sized(sides):
    """Square regions of exactly these level-0 sides, at ds 1."""
    pitch = max(sides) + 1000
    return make_mask(blobs(max(sides), pitch * len(sides),
                           [(0, i * pitch, n, n) for i, n in enumerate(sides)]), 1.0)


def t_mask_views_do_not_leak():
    """Every region step is a view. The source keeps its regions whatever is
    derived from it."""
    trm = _sized([300, 2000, 100])
    before = boxes_of(trm)
    views = [trm.patchable(256), trm.filtered(0.5), trm.merged(), trm.raw()]
    assert boxes_of(trm) == before, 'a view changed its source'
    assert len(views[0]) == 2 and len(views[1]) == 1, (len(views[0]), len(views[1]))
    assert views[0].main_mask is trm.main_mask, 'a view copied the raster'
    return 'source untouched by patchable / filtered / merged / raw'


def t_mask_patchable_round_trip():
    """fine -> coarse -> fine returns the fine answer. patchable is monotone
    in the footprint, so a coarse pass that reached the source would hide what
    it dropped from every later fine pass."""
    trm = _sized([300, 2000, 100])
    side = lambda s: sorted(r.w for r in trm.patchable(s).tissue_regions)  # noqa: E731
    fine, coarse, again = side(256), side(1024), side(256)
    assert (fine, coarse, again) == ([300, 2000], [2000], [300, 2000]), \
        (fine, coarse, again)
    return f'{fine} -> {coarse} -> {again}'


def t_mask_merged_chains_partial_overlaps():
    """A-B-C chain merges to one union box; D isolated and E nested in B stay
    (nested is `filtered`'s job). Indices are renumbered."""
    base = make_mask(np.ones((500, 500), bool), 1.0)
    trm = base._with([TissueRegion(0, 0, 100, 100, 0), TissueRegion(50, 50, 100, 100, 1),
                      TissueRegion(120, 120, 100, 100, 2), TissueRegion(400, 400, 50, 50, 3),
                      TissueRegion(60, 60, 20, 20, 4)]).merged()
    assert boxes_of(trm) == sorted([(0, 0, 220, 220), (400, 400, 50, 50),
                                    (60, 60, 20, 20)]), boxes_of(trm)
    assert [r.index for r in trm.tissue_regions] == [0, 1, 2]
    assert len(base._with([]).merged()) == 0
    return 'A-B-C -> one, D and E kept'


def t_mask_filtered_is_area_then_containment():
    trm = make_mask(np.ones((500, 500), bool), 1.0)._with([
        TissueRegion(0, 0, 200, 200), TissueRegion(10, 10, 50, 50),     # nested
        TissueRegion(300, 300, 100, 100), TissueRegion(450, 450, 10, 10)])
    assert boxes_of(trm.filtered(0.05)) == [(0, 0, 200, 200), (300, 300, 100, 100)]
    return 'small and nested dropped'


def _slide_mask(components: bool = True) -> SlideMask:
    rng = np.random.default_rng(0)
    rows, cols, k = 12, 20, 4
    return SlideMask(
        mask=(rng.random((rows, cols)) < 0.25), origin=(1000, 2000),
        span=(cols * 14, rows * 14), mask_ds=14.0, report={'cells': 7},
        components=(rng.random((rows, cols, k)).astype(np.float16)
                    if components else None))


def t_slide_mask_round_trip():
    """What a cache stores comes back as what went in, geometry TYPED:
    `f'{mask_ds:.0f}'` on a string raises (a lazy annotation decoded as
    str)."""
    sm = _slide_mask(components=False)
    with tempfile.TemporaryDirectory() as root:
        back = SlideMask.load(sm.save(os.path.join(root, 'mask.safetensors')))
    assert np.array_equal(back.mask.astype(bool), np.asarray(sm.mask)), 'mask changed'
    assert (back.origin, back.span) == (sm.origin, sm.span), (back.origin, back.span)
    assert isinstance(back.mask_ds, float) and f'{back.mask_ds:.0f}' == '14', back.mask_ds
    assert back.report == sm.report, (back.report, sm.report)
    assert back.geometry() == sm.geometry(), (back.geometry(), sm.geometry())
    return 'mask, typed geometry, report'


def t_slide_mask_components():
    """Components ride along, come back only when asked, are refused when
    absent and must line up with the mask."""
    with tempfile.TemporaryDirectory() as root:
        sm = _slide_mask()
        path = sm.save(os.path.join(root, 'a.safetensors'))
        assert SlideMask.load(path).components is None, 'components were read unasked'
        back = SlideMask.load(path, with_components=True)
        assert np.array_equal(back.components, sm.components), 'components changed'
        assert back.geometry() == sm.geometry(), (back.geometry(), sm.geometry())
        path = _slide_mask(components=False).save(os.path.join(root, 'b.safetensors'))
        try:
            SlideMask.load(path, with_components=True)
            raise AssertionError('absent components answered with_components')
        except CacheMismatch:
            pass
        try:
            dataclasses.replace(sm, components=np.zeros((20, 12, 4), np.float16)) \
                .save(os.path.join(root, 'c.safetensors'))
            raise AssertionError('transposed components were accepted')
        except ValueError:
            pass
    return 'round trip, absent refused, transposed refused'


# ══════════════════════════════════════════════════════════════════════════════
#  TissueSegFunc -- the producer
# ══════════════════════════════════════════════════════════════════════════════

def t_seg_level_is_nearest_by_ratio():
    """A level reporting 4.00003 is the level for ds 4; openslide's strict rule
    (the decoy) picks level 0, which on BRACS_1228 is 6.58 Gpx to segment
    instead of 411 Mpx."""
    wsi = _Slide(4096, 4096, downsamples=(1.0, 4.00003, 16.0001))
    assert wsi.get_best_level_for_downsample(4.0) == 0, 'the decoy did not bite'
    lds = wsi.level_downsamples
    assert nearest_level(lds, 4.0) == 1 and plane_geometry(wsi, 4.0).level == 1
    assert nearest_level(lds, 7.0) == 1 and nearest_level(lds, 9.0) == 2
    return '4.0 -> level 1 (openslide: 0)'


def t_seg_plane_is_the_scanned_rect():
    wsi = _Slide(10000, 8000, downsamples=(1.0, 4.0), bounds=(1200, 800, 4000, 2000))
    g = plane_geometry(wsi, 4.0, limit_bounds=True)
    assert (g.level, g.origin, g.span, g.shape) == (1, (1200, 800), (4000, 2000), (500, 1000)), g
    g = plane_geometry(wsi, 4.0, limit_bounds=False)
    assert (g.origin, g.span) == ((0, 0), (10000, 8000)), g
    return 'bounds honoured, shape in level px'


def t_seg_tiled_apply_is_the_whole_answer():
    """A per-pixel method stitched from tiles is the method on the whole image.
    Grid and trimming are the only moving parts, so an off-by-one in either
    shows up as a mismatched band."""
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (97, 131, 3), dtype=np.uint8)
    method = lambda a: (a[..., 0] > 127).astype(np.uint8)          # noqa: E731
    whole = method(img)
    tiled = tiled_apply(method, 97, 131, lambda y0, x0, y1, x1: img[y0:y1, x0:x1],
                        budget_px=700, stitch_overlap=5)
    assert np.array_equal(whole, tiled), f'{int((whole != tiled).sum())} px differ'
    return 'budget 700 px on a 97x131 plane'


def t_seg_plane_read_tiled_equals_whole():
    """PlaneSegmenter end to end on a fake slide: the read+seg tiled path and
    the single read agree, and the mask lands on the scanned rectangle."""
    plane = np.full((2000, 3000, 3), 245, np.uint8)
    plane[400:1200, 600:2200] = (180, 60, 140)                   # stained tissue
    wsi = _Slide(3000, 2000, downsamples=(1.0, 4.0), plane=plane)
    whole = PlaneSegConfig('hsv', ds=4.0, seg_chunk_px=None,
                           read_chunk_px=None).build().segment_slide(wsi)
    tiled = PlaneSegConfig('hsv', ds=4.0, seg_chunk_px=20_000,
                           read_chunk_px=20_000).build().segment_slide(wsi)
    assert whole.shape == (500, 750) and whole.mask_ds == 4.0, (whole.shape, whole.mask_ds)
    assert np.array_equal(whole.mask, tiled.mask), \
        f'{int((whole.mask != tiled.mask).sum())} px differ between tiled and whole'
    trm = TissueMask(wsi, whole)
    assert boxes_of(trm) == [(600, 400, 1600, 800)], boxes_of(trm)
    return f'{len(wsi.reads)} reads, tiled == whole'


# ── fingerprints (ConfigIdentity rule 3) ─────────────────────────────────────
#
# Each producer's output on one fixed synthetic slide, pinned to its VERSION.
# A behaviour change that forgets to bump the VERSION fails here instead of
# leaving cached output that a record would still call current.

FP_PLANE = {0: '364d3749d2221a33'}
FP_REGIONS = {0: '7a5868decd5e243c'}
FP_READER = {0: '908e7b4cd71b9c3a'}


def _textured_slide():
    """3000 x 2000, two stained sections and a pale one on noisy glass, with a
    4x level: something every producer below has to get right."""
    rng = np.random.default_rng(7)
    plane = (235 + rng.integers(-8, 8, (2000, 3000, 3))).astype(np.uint8)
    plane[300:900, 400:1500] = (175, 70, 150)
    plane[1100:1700, 1800:2700] = (150, 90, 160)
    plane[1300:1500, 300:700] = (215, 185, 205)
    noise = rng.integers(-20, 20, (2000, 3000, 3))
    plane = np.clip(plane.astype(int) + noise * (plane[..., :1] < 230), 0, 255
                    ).astype(np.uint8)
    return _Slide(3000, 2000, downsamples=(1.0, 4.0), plane=plane)


def t_fingerprint_plane_segmenter():
    from ConfigIdentity import check_fingerprint, digest
    sm = PlaneSegConfig('hsv', seg_chunk_px=20_000,
                        read_chunk_px=20_000).build().segment_slide(_textured_slide())
    return check_fingerprint(PlaneSegConfig, digest(np.asarray(sm.mask)), FP_PLANE)


def t_fingerprint_region_prep():
    from ConfigIdentity import check_fingerprint, digest
    wsi = _textured_slide()
    sm = PlaneSegConfig('hsv').build().segment_slide(wsi)
    regions = MASK_RECIPES['hsv'].regions(wsi, sm)
    return check_fingerprint(TissueMaskConfig,
                             digest(np.array(boxes_of(regions), np.int64)),
                             FP_REGIONS)


def t_fingerprint_slide_reader():
    """A native read, a shrink through each filter, a forced level: the pixels
    every cache made of slide pixels is made of."""
    from ConfigIdentity import check_fingerprint, digest
    from ReadGeometry import ReadSpec
    from SlideReader import SlideReader
    wsi = _textured_slide()
    reads = [SlideReader(wsi, resize=r).read(380, 290, ReadSpec(96, 64), ds,
                                             level=lv)
             for r, ds, lv in (('lanczos', 4.0, None), ('lanczos', 3.0, None),
                               ('area', 3.0, None), ('lanczos', 6.0, 1))]
    assert all(r is not None for r in reads), 'a fixture read ran off the slide'
    return check_fingerprint(SlideReader, digest(*reads), FP_READER)


def t_seg_none_reads_nothing():
    """method '' is one region over the plane, and nothing is read to get
    there -- not a method returning ones, which read the whole level for it."""
    wsi = _Slide(4000, 2000, downsamples=(1.0, 4.0))
    sm = PlaneSegConfig('').build().segment_slide(wsi)
    assert wsi.reads == [], f'{len(wsi.reads)} reads for a mask that is all True'
    assert sm.mask.all() and sm.shape == (500, 1000)
    assert len(TissueMask(wsi, sm)) == 1
    return 'no read, one region'


def t_seg_refusals():
    try:
        PlaneSegConfig('otsu')
        raise AssertionError('a tiled otsu was accepted')
    except ValueError:
        pass
    PlaneSegConfig('otsu', seg_chunk_px=None, read_chunk_px=None)
    try:
        TissueSegConfig().build()
        raise AssertionError('the base config built something')
    except TypeError:
        pass
    return 'tiled otsu, the bare base'


# ══════════════════════════════════════════════════════════════════════════════
#  TissueMaskConfig -- the recipe and the cache
# ══════════════════════════════════════════════════════════════════════════════

ROWS, COLS, CELL = 12, 20, 14


class FakeSegmenter(TissueSegmenter):
    """A slide segmenter that counts its calls. No model, no plane."""
    calls = 0

    def __init__(self, cfg):
        self.cfg = cfg

    def identity_id(self):
        return f'fake-{self.cfg.tag}'

    def segment_slide(self, wsi):
        # 10 x 16 = 160 cells, over `_search_tissue_regions`' min_area_px of
        # 100 -- a smaller blob comes back as NO region, and every "the hit
        # placed the same regions" check then compares two empty lists.
        FakeSegmenter.calls += 1
        mask = np.zeros((ROWS, COLS), bool)
        mask[1:11, 2:18] = True
        return SlideMask(mask=mask, origin=(0, 0),
                         span=(COLS * CELL, ROWS * CELL), mask_ds=float(CELL))


@dataclass(frozen=True)
class FakeSegConfig(TissueSegConfig):
    method: str = 'fake'
    tag: int = 0
    builds = 0

    def build(self, device=None):
        type(self).builds += 1
        return FakeSegmenter(self)


def _cfg(**over) -> TissueMaskConfig:
    return TissueMaskConfig(seg=FakeSegConfig(), **over)


def _slide(path='/data/Group_A/SLIDE_A.svs'):
    return _Slide(COLS * CELL, ROWS * CELL, downsamples=(1.0,), path=path)


def _fresh():
    FakeSegmenter.calls = 0
    FakeSegConfig.builds = 0


def t_cfg_seg_has_no_default():
    try:
        TissueMaskConfig()
    except TypeError:
        return 'TissueMaskConfig() refused'
    raise AssertionError('TissueMaskConfig() built -- the silent default is back')


def t_cfg_hest_is_the_baseline():
    """The hest recipe is every baseline at once: the mask's own fields and
    HestSegConfig's. Only its ModelConfig speaks, against ModelConfig's own
    baseline (a timm encoder), because HEST is a torchvision network."""
    parts = MASK_RECIPES['hest'].identity_parts()
    assert all(p.startswith('seg.model.') for p in parts), parts
    assert MASK_RECIPES['hest'].seg_id().startswith('hest-')
    assert MASK_RECIPES['hsv'].seg_id().startswith('hsv-')
    assert MASK_RECIPES['none'].seg_id().startswith('none-')
    ids = {name: cfg.seg_id() for name, cfg in MASK_RECIPES.items()}
    assert len(set(ids.values())) == len(ids), ids
    return ', '.join(f'{v}' for v in ids.values())


def t_cfg_region_fields_do_not_move_seg_id():
    base, other = _cfg(), _cfg(min_region_ratio=0.3, merge=False)
    assert base.seg_id() == other.seg_id(), 'a region-only change moved seg_id'
    assert base.region_id() != other.region_id()
    return f'seg_id {base.seg_id()}'


def t_cfg_seg_fields_move_seg_id_and_not_region_id():
    """The decoy: two recipes identical but for how the plane is read must not
    share a directory. The split is the nesting, so no field can sit on the
    wrong side of it."""
    base = MASK_RECIPES['hsv']
    for field, value in (('ds', 32.0), ('read_chunk_px', 1_000_000),
                         ('seg_chunk_px', None), ('stitch_overlap', 64),
                         ('limit_bounds', False)):
        other = dataclasses.replace(base, seg=dataclasses.replace(base.seg, **{field: value}))
        assert base.seg_id() != other.seg_id(), field
        assert base.region_id() == other.region_id(), field
    assert _cfg().seg_id() != TissueMaskConfig(seg=FakeSegConfig(tag=1)).seg_id()
    return 'ds, read_chunk_px, seg_chunk_px, stitch_overlap, limit_bounds, seg'


def t_cfg_weights_content_moves_seg_id():
    """A finetune overwritten IN PLACE under the same path. The path cannot see
    it and a cache hit builds no model to hash, so the key reads the file."""
    from HestSegFunc import HestSegConfig                        # noqa: PLC0415
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, 'finetune.ckpt')
        with open(path, 'wb') as handle:
            handle.write(b'first weights')
        seg = HestSegConfig(model=dataclasses.replace(HestSegConfig().model, weights=path))
        cfg = TissueMaskConfig(seg=seg)
        first = cfg.seg_id()
        with open(path, 'wb') as handle:
            handle.write(b'second weights, longer')
        second = cfg.seg_id()
    assert first != second, 'the weights changed and the key did not'
    assert first != MASK_RECIPES['hest'].seg_id()
    return f'{first} -> {second}'


def t_cache_miss_then_hit_builds_nothing():
    """A hit reads the mask back without building a segmenter -- the whole
    point of caching a model's output."""
    _fresh()
    with tempfile.TemporaryDirectory() as root:
        cfg, wsi = _cfg(), _slide()
        with MaskMaker(cfg, root) as masks:
            a, hit_a = masks.mask(wsi)
        with MaskMaker(cfg, root) as masks:
            b, hit_b = masks.mask(wsi)
        assert (hit_a, hit_b) == (False, True), (hit_a, hit_b)
        assert (FakeSegmenter.calls, FakeSegConfig.builds) == (1, 1), \
            (FakeSegmenter.calls, FakeSegConfig.builds)
        assert len(a) == 1, f'{len(a)} regions -- the fixture must have one'
        assert boxes_of(a) == boxes_of(b), 'the hit placed different regions'
        assert (masks.slide_dir('SLIDE_A') / 'mask_meta.json').exists()
    return f'{len(a)} regions, one build'


def t_cache_region_change_reuses_the_mask():
    _fresh()
    with tempfile.TemporaryDirectory() as root:
        MaskMaker(_cfg(), root).mask(_slide())
        _, hit = MaskMaker(_cfg(min_region_ratio=0.3), root).mask(_slide())
        assert hit and FakeSegmenter.calls == 1, (hit, FakeSegmenter.calls)
    return 'min_region_ratio changed, no resegmentation'


def t_cache_version_bump_resegments():
    """Same address, other segmenter code: the stored record no longer
    matches, so the mask is made again in place. The decoy is the same maker
    after the version is restored, which must hit what was just rewritten."""
    _fresh()
    with tempfile.TemporaryDirectory() as root:
        MaskMaker(_cfg(), root).mask(_slide())
        saved = FakeSegConfig.VERSION
        try:
            FakeSegConfig.VERSION = saved + 1
            _, hit = MaskMaker(_cfg(), root).mask(_slide())
            assert not hit and FakeSegmenter.calls == 2, (hit, FakeSegmenter.calls)
        finally:
            FakeSegConfig.VERSION = saved
        _, hit = MaskMaker(_cfg(), root).mask(_slide())
        assert not hit, 'the restored version hit the bumped mask'
        _, hit = MaskMaker(_cfg(), root).mask(_slide())
        assert hit and FakeSegmenter.calls == 3, (hit, FakeSegmenter.calls)
    return 'bump resegments, restore resegments once, then hits'


def t_cache_segmenter_lives_with_the_maker():
    _fresh()
    with tempfile.TemporaryDirectory() as root:
        masks = MaskMaker(_cfg(), root)
        for name in ('A', 'B', 'C'):
            masks.mask(_slide(f'/data/g/SLIDE_{name}.svs'))
        assert (FakeSegmenter.calls, FakeSegConfig.builds) == (3, 1)
        masks.close()
        masks.mask(_slide('/data/g/SLIDE_D.svs'))
        assert FakeSegConfig.builds == 2, 'close did not drop the segmenter'
    return '3 slides, 1 build; close forces a rebuild'


def t_cache_same_stem_elsewhere_is_refused():
    _fresh()
    with tempfile.TemporaryDirectory() as root:
        masks = MaskMaker(_cfg(), root)
        masks.mask(_slide('/data/Group_A/SLIDE_A.svs'))
        masks.mask(_slide('/other/mount/Group_A/SLIDE_A.svs'))     # a remount: fine
        try:
            masks.mask(_slide('/data/Group_B/SLIDE_A.svs'))
        except CacheMismatch as e:
            assert 'SLIDE_A' in str(e), str(e)
            return 'remount hits, same stem elsewhere refused'
    raise AssertionError('a different slide with the same stem got a cached mask')


def t_cache_missing_sidecar_is_a_miss():
    """The sidecar is written last; a job killed between the two files leaves
    a miss, not a half-written hit."""
    _fresh()
    with tempfile.TemporaryDirectory() as root:
        masks = MaskMaker(_cfg(), root)
        masks.mask(_slide())
        (masks.slide_dir('SLIDE_A') / 'mask_meta.json').unlink()
        _, hit = masks.mask(_slide())
        assert not hit and FakeSegmenter.calls == 2
    return 'no sidecar -> resegmented'


def t_cache_no_root_always_segments():
    _fresh()
    masks = MaskMaker(_cfg())
    assert [masks.mask(_slide())[1] for _ in range(2)] == [False, False]
    assert (FakeSegmenter.calls, FakeSegConfig.builds) == (2, 1)
    return 'two calls, two segmentations, one build'


def t_cfg_overrides_from_args():
    """`--mask-ds` and friends make a DIFFERENT recipe -- a different seg_id --
    so an override can never be served a mask made without it; and they are
    refused on a segmenter they do not apply to."""
    def args(**kw):
        base = dict(seg='hest', mask_ds=None, seg_chunk_px=None,
                    read_chunk_px=None, min_region_ratio=None)
        return SimpleNamespace(**{**base, **kw})
    assert mask_cfg_from_args(args()) == MASK_RECIPES['hest']
    ds1 = mask_cfg_from_args(args(mask_ds=1.0, read_chunk_px=0))
    assert ds1.seg.ds == 1.0 and ds1.seg.read_chunk_px is None
    assert ds1.seg_id() != MASK_RECIPES['hest'].seg_id()
    ratio = mask_cfg_from_args(args(min_region_ratio=0.2))
    assert ratio.seg_id() == MASK_RECIPES['hest'].seg_id()
    assert ratio.region_id() != MASK_RECIPES['hest'].region_id()
    try:
        mask_cfg_from_args(args(seg='uni2_pca', mask_ds=4.0))
        raise AssertionError('--mask-ds was accepted for a slide segmenter')
    except ValueError:
        pass
    return 'new seg_id per override, uni2_pca refuses plane overrides'


# ══════════════════════════════════════════════════════════════════════════════
#  real slide -- a figure, and what a real slide must satisfy
# ══════════════════════════════════════════════════════════════════════════════

def _check_real(name: str, trm: TissueMask) -> None:
    assert trm.main_mask.dtype == bool
    assert trm.tissue_fraction() > 0.0, f'{name}: no tissue on a real slide'
    assert len(trm.tissue_regions) > 0, f'{name}: no region on a real slide'
    for i, r in enumerate(trm.tissue_regions):
        assert r.w > 0 and r.h > 0, f'{name}: region {i} has zero size'
        assert 0 <= r.x < trm.wsi_width and 0 <= r.y < trm.wsi_height, \
            f'{name}: region {i} at ({r.x}, {r.y}) is off the slide'


def real_slide(args):
    """Row 0: each recipe's mask and backdrop. --ops: raw, then each region
    step alone, then the recipe's pipeline plus patchable. --sweep: the first
    plane recipe at each --sweep-ds. --tiling: the first plane recipe read
    whole against read tiled, the tile grid, and a budget sweep."""
    import matplotlib                                            # noqa: PLC0415
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt                              # noqa: PLC0415
    import torch                                                 # noqa: PLC0415
    from SafeSlide import SafeSlide                              # noqa: PLC0415

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    recipes = {}
    for name in args.seg:
        cfg = mask_cfg_from_args(SimpleNamespace(
            seg=name, mask_ds=args.mask_ds, seg_chunk_px=None,
            read_chunk_px=None, min_region_ratio=None))
        if name == 'uni2_pca' and args.pca_fit_tiles:
            cfg = dataclasses.replace(cfg, seg=dataclasses.replace(
                cfg.seg, fit_tiles=args.pca_fit_tiles))
        recipes[name] = cfg
    planes = [n for n, c in recipes.items() if isinstance(c.seg, PlaneSegConfig)]

    panels = {'top': [], 'sweep': [], 'ops': [], 'tiling': []}
    wsi = SafeSlide(args.wsi)
    try:
        bounds = (int(wsi.properties.get('openslide.bounds-x', 0)),
                  int(wsi.properties.get('openslide.bounds-y', 0)),
                  int(wsi.properties.get('openslide.bounds-width', wsi.dimensions[0])),
                  int(wsi.properties.get('openslide.bounds-height', wsi.dimensions[1])))
        for name, cfg in recipes.items():
            print(f'\n[{name}] {cfg.seg_id()}', flush=True)
            with MaskMaker(cfg, device=device) as masks:
                slide_mask, _ = masks.slide_mask(wsi)
            raw = TissueMask(wsi, slide_mask)
            trm = cfg.regions(wsi, slide_mask)
            _check_real(name, trm)
            print(f'  tissue {trm.tissue_fraction():.1%}  {len(raw)} raw -> '
                  f'{len(trm)} regions  mask {trm.main_mask.shape} at ds '
                  f'{trm.mask_ds_x:.2f}', flush=True)
            panels['top'].append(('mask', trm, f'{name} mask'))
            panels['top'].append(('thumb', (trm, trm.read_matching_rgb(wsi)),
                                  f'{name} backdrop'))
            if args.ops:
                side = args.ops_patch_tile * args.ops_patch_ds
                for label, view in (
                        ('raw', raw),
                        (f'filtered({cfg.min_region_ratio})', raw.filtered(cfg.min_region_ratio)),
                        ('merged', raw.merged()),
                        (f'patchable({side:g})', raw.patchable(side)),
                        ('recipe + patchable', trm.patchable(side))):
                    panels['ops'].append(('mask', view, f'[{name}] {label}  '
                                          f'{len(raw)}->{len(view)}'))

        if args.sweep and planes:
            base = recipes[planes[0]]
            for ds in args.sweep_ds:
                cfg = dataclasses.replace(base, seg=dataclasses.replace(base.seg, ds=ds))
                with MaskMaker(cfg, device=device) as masks:
                    trm, _ = masks.mask(wsi)
                print(f'  [sweep {planes[0]}] ds {ds:g}: mask {trm.main_mask.shape}  '
                      f'{len(trm)} regions', flush=True)
                panels['sweep'].append(('mask', trm, f'[sweep {planes[0]}] ds {ds:g}'))

        if args.tiling and planes:
            base = recipes[planes[0]]
            seg = dataclasses.replace(base.seg, ds=args.tiling_ds,
                                      stitch_overlap=args.tiling_overlap)
            whole = dataclasses.replace(base, seg=dataclasses.replace(
                seg, seg_chunk_px=None, read_chunk_px=None))
            with MaskMaker(whole, device=device) as masks:
                ref, _ = masks.mask(wsi)
            panels['tiling'].append(('mask', ref, f'[{planes[0]}] whole read'))
            for budget in args.seg_chunk_px_sweep:
                cfg = dataclasses.replace(base, seg=dataclasses.replace(
                    seg, seg_chunk_px=budget, read_chunk_px=budget))
                with MaskMaker(cfg, device=device) as masks:
                    trm, _ = masks.mask(wsi)
                diff = float((trm.main_mask != ref.main_mask).mean())
                print(f'  [tiling {planes[0]}] budget {budget / 1e6:g}M: '
                      f'{diff:.3%} of pixels differ from the whole read', flush=True)
                panels['tiling'].append(('grid', (trm, budget), f'budget {budget / 1e6:g}M  '
                                         f'diff {diff:.2%}'))
    finally:
        wsi.close()

    rows = [panels['top']] + [panels[k][i:i + args.per_row]
                              for k in ('sweep', 'ops', 'tiling')
                              for i in range(0, len(panels[k]), args.per_row)]
    n_cols = max(len(r) for r in rows)
    fig, axes = plt.subplots(len(rows), n_cols, squeeze=False,
                             figsize=(args.figure_scale[0] * n_cols,
                                      args.figure_scale[1] * len(rows)))
    for ax in axes.ravel():
        ax.axis('off')
    for r, row in enumerate(rows):
        for c, (kind, payload, title) in enumerate(row):
            _draw(axes[r, c], kind, payload, title, bounds, args)
    fig.tight_layout()
    out = args.out or os.path.join(job_result_dir('TestTissueMask'),
                                   'tissue_mask.png')
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {out}')


def _draw(ax, kind, payload, title, bounds, args):
    """Region boxes in mask coordinates; red inside the scanned rectangle,
    magenta where a region reaches the MIRAX padding that holds no pixels."""
    import matplotlib.patches as mpatches                        # noqa: PLC0415
    if kind == 'thumb':
        trm, image = payload
        ax.imshow(image)
    elif kind == 'grid':
        trm, budget = payload
        ax.imshow(trm.main_mask, cmap='gray', vmin=0, vmax=1)
        H, W = trm.main_mask.shape
        n_h = n_w = 1
        while (H // n_h) * (W // n_w) > budget:
            if H // n_h >= W // n_w:
                n_h *= 2
            else:
                n_w *= 2
        for i in range(1, n_h):
            ax.axhline(i * (H // n_h), color='cyan', linewidth=0.8)
        for j in range(1, n_w):
            ax.axvline(j * (W // n_w), color='cyan', linewidth=0.8)
        title = f'{title}\n{n_h}x{n_w} tiles'
    else:
        trm = payload
        ax.imshow(trm.main_mask, cmap='gray', vmin=0, vmax=1)
    bx, by, bw, bh = bounds
    outside = 0
    for r in trm.tissue_regions:
        out = (r.x < bx or r.y < by or r.x + r.w > bx + bw or r.y + r.h > by + bh)
        outside += int(out)
        mx, my, mw, mh = trm.region_box(r)
        ax.add_patch(mpatches.Rectangle((mx, my), mw, mh, fill=False,
                                        edgecolor='magenta' if out else 'red',
                                        linewidth=args.bbox_lw))
        if args.region_index:
            ax.text(mx + 2, my + 8, str(r.index), color='yellow', fontsize=7)
    mx, my = trm.to_mask_xy(bx, by)
    ax.add_patch(mpatches.Rectangle((mx, my), bw / trm.mask_ds_x, bh / trm.mask_ds_y,
                                    fill=False, edgecolor='cyan', linestyle='--',
                                    linewidth=args.bbox_lw * 1.5))
    ax.set_title(f'{title}\n{len(trm)} regions, tissue '
                 f'{trm.tissue_fraction():.1%}, {outside} outside the scan',
                 fontsize=9)


# ══════════════════════════════════════════════════════════════════════════════

_TESTS = [t for n, t in sorted(globals().items(), key=lambda kv: kv[0])
          if n.startswith('t_')]


def _pixels(s: str) -> int:
    s = s.strip().upper()
    scale = {'M': 1_000_000, 'K': 1_000}.get(s[-1:], 1)
    return int(float(s.rstrip('MK')) * scale)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--wsi', default=None,
                    help='a real slide: adds the figure. Omitted: synthetic only')
    ap.add_argument('--seg', nargs='+', default=['hsv', 'hest'],
                    choices=sorted(MASK_RECIPES),
                    help='recipes to draw, one mask + backdrop each')
    ap.add_argument('--mask-ds', type=float, default=None,
                    help="override every plane recipe's segmentation ds")
    ap.add_argument('--pca-fit-tiles', type=int, default=200,
                    help='uni2_pca only: the fit sample. Hashed, so this '
                         'figure is not the production mask (1000)')
    ap.add_argument('--ops', action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument('--ops-patch-tile', type=int, default=256)
    ap.add_argument('--ops-patch-ds', type=float, default=1.0)
    ap.add_argument('--sweep', action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument('--sweep-ds', type=lambda s: [float(x) for x in s.split(',')],
                    default=[4.0, 16.0, 32.0, 64.0])
    ap.add_argument('--tiling', action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument('--tiling-ds', type=float, default=32.0)
    ap.add_argument('--tiling-overlap', type=int, default=128)
    ap.add_argument('--seg-chunk-px-sweep',
                    type=lambda s: [_pixels(x) for x in s.split(',')],
                    default=[16_000_000, 4_000_000, 1_000_000])
    ap.add_argument('--per-row', type=int, default=4)
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--figure-scale', type=lambda s: [float(x) for x in s.split(',')],
                    default=[7.0, 5.0])
    ap.add_argument('--region-index', action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument('--bbox-lw', type=float, default=1.0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    for fn in _TESTS:
        check(fn.__name__[2:].replace('_', ' '), fn)
    failed = [n for n, e in _RESULTS if e is not None]
    print(f'\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed')
    if failed:
        print('failed: ' + ', '.join(failed))
        return 1

    if args.wsi:
        real_slide(args)
    return 0


# add_mask_args is the CLI the tools use; imported so a reader of this file sees
# which helper `t_cfg_overrides_from_args` stands in for.
_ = add_mask_args

if __name__ == '__main__':
    sys.exit(main())
