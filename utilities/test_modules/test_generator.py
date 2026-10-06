#!/usr/bin/env python3
"""Tests for `query_sim/generator.py`'s `FovSupply`: WHERE a camera's FoVs are.

    python utilities/test_modules/test_generator.py --wsi <slide> [--level 1]

Run through `jobscripts/TestReadPath.sh`, which resolves slide names and loops.

FoVs placed by richness bucket, reproducible whatever the camera's own seed;
iterating draws new batches (next seed) and only repeats once the slide is
out of positions; a plan or a config that is not this camera's is refused;
and a `Render` carries no mask and no sampler.

(The `equiv` section that pinned FovSupply against a frozen copy of the
Camera methods it replaced -- 116 shots, 0 different -- was deleted with that
copy on 2026-10-03, once the move was accepted. The sampler's own arithmetic
for a rectangular camera is `test_tile_sampler.py`, section fov; the render
is `test_camera.py`.)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_ROOT))

from _paths import setup_import_paths                            # noqa: E402

setup_import_paths()

import numpy as np                                               # noqa: E402

from camera import Render                                        # noqa: E402
from config import DomainGapConfig                               # noqa: E402
from generator import FovSupply, fov_plan_of                     # noqa: E402
from SafeSlide import SafeSlide                                  # noqa: E402
from SlideReader import SlideReader                              # noqa: E402
from TileSampler import SamplerConfig, camera_plan               # noqa: E402
from TissueMaskConfig import MASK_RECIPES, MaskMaker             # noqa: E402


def fov_cfg(query_mpp: float) -> DomainGapConfig:
    """The camera the window bench uses: optics and colour on, geometry off."""
    return DomainGapConfig(
        wh_ratio='45:32', MPixels=1.47456, query_mpp=query_mpp,
        rotation_choices=(0,), angle_jitter_deg=0.0, scale_range=(1.0, 1.0),
        query_mpp_jitter=0.0, stage_shift_max=0)


def _same(a, b) -> bool:
    return a is not None and b is not None and np.array_equal(a, b)


class _Expect:
    def __init__(self):
        self.failures = []

    def __call__(self, ok, what):
        print(f'  {"ok  " if ok else "FAIL"} {what}', flush=True)
        if not ok:
            self.failures.append(what)


def run_supply(args, wsi, mask, level_mpp) -> int:
    n_asked = 8
    sampler_cfg = SamplerConfig(n_per_rung=n_asked, seed=5)
    reader = SlideReader(wsi)

    def camera(seed):
        return Render(reader, fov_cfg(level_mpp), seed=seed)

    def build(seed):
        return FovSupply(camera(seed), mask, sampler_cfg)

    supply = build(0)
    cam = supply.cam
    try:
        bank = supply.bank()
    except RuntimeError as exc:
        if 'No FoV position' not in str(exc):
            raise
        print(f'  SKIPPED  no FoV position at level {args.level} on this slide:'
              f'\n{exc}')
        return 0
    sampler = supply.sampler
    metas = [s.meta for s in sampler]
    print(f'  {len(metas)} positions drawn, {len(bank)} rendered '
          f'({n_asked} asked) at ds {metas[0].ds:g}')
    print(sampler.reports[metas[0].ds].line())
    expect = _Expect()

    expect(1 <= len(bank) <= n_asked, f'the bank holds 1..{n_asked} shots')
    shape = (cam.output_h, cam.output_w, 3)
    expect(all(tuple(b.image.shape) == shape and b.image.dtype == np.uint8
               for b in bank), f'every shot is {shape} uint8')
    rects = {(m.x, m.y): m.fov_rect for m in metas}
    expect(all(any((r[0], r[1]) == (b.gt_x, b.gt_y) for r in rects.values())
               for b in bank), 'every shot sits at a position the sampler drew')
    names = set(sampler.cfg.richness.names)
    expect(all(b.bucket in names and b.fov_background is not None
               and b.draw_index == 0 and b.pass_index == 0 for b in bank),
           'every shot carries its bucket, background, draw and pass')
    expect(all(b.fov_background < 0.85 for b in bank),
           'no FoV above the 0.85 background the default caps forbid')

    # The pixels depend on the position and the round, not on the camera's own
    # seed and not on what was drawn before them.
    other = build(99).bank()
    expect(len(other) == len(bank)
           and all(_same(a.image, b.image) for a, b in zip(bank, other)),
           "a camera with another seed renders the same FoVs (positions come "
           "from the sampler's seed, the domain gap from the position)")

    # The iterator: the first draw is the bank, then a NEW draw (next seed) of
    # positions not yet shown, and only once the slide is out of positions
    # does it repeat, each pass with a fresh domain gap.
    bank_xy = {(b.gt_x, b.gt_y) for b in bank}
    n_more = min(len(bank), 3)
    it = iter(supply)
    first_draw = [next(it) for _ in bank]
    more = [next(it) for _ in range(n_more)]
    expect(all(_same(a.image, b.image) for a, b in zip(bank, first_draw)),
           'iterating starts with the bank')
    expect(all((m.gt_x, m.gt_y) not in bank_xy and m.pass_index == 0
               and m.draw_index >= 1 for m in more),
           'the next shots are new positions from a later draw, not repeats')
    expect(len({(m.gt_x, m.gt_y) for m in more}) == len(more),
           'and are distinct from each other')
    it2 = iter(build(7))
    again = [next(it2) for _ in range(len(bank) + n_more)]
    expect(all(_same(a.image, b.image)
               for a, b in zip(first_draw + more, again)),
           'the same sequence on another camera (positions and pixels)')

    # Exhaustion, forced: a supply that only ever offers two positions. The
    # shots then repeat them, each pass a new domain gap, reproducibly.
    if len(bank) == len(metas) and len(metas) >= 2:
        two = metas[:2]

        def scarce(seed):
            s = build(seed)
            s.draw = lambda k: (sampler, two if k == 0 else [])
            return s

        seq = iter(scarce(0))
        p0 = [next(seq) for _ in two]
        p1 = [next(seq) for _ in two]
        p2 = [next(seq) for _ in two]
        expect([m.pass_index for m in p0 + p1 + p2] == [0, 0, 1, 1, 2, 2]
               and all((a.gt_x, a.gt_y) == (b.gt_x, b.gt_y)
                       for a, b in zip(p0, p1)),
               'out of positions: the same two, passes 0 / 1 / 2')
        expect(not any(_same(a.image, b.image) for a, b in zip(p0, p1))
               and not any(_same(a.image, b.image) for a, b in zip(p1, p2)),
               'every repeat is a different domain gap')
        seq2 = iter(scarce(9))
        q = [next(seq2) for _ in range(6)]
        expect(all(_same(a.image, b.image) for a, b in zip(p0 + p1 + p2, q)),
               'and the repeats reproduce on another camera')
    else:
        print('  SKIPPED  exhaustion check: fewer than two renderable positions')

    # No config: SamplerConfig's own defaults; the FoV's size from the camera.
    plain = FovSupply(camera(0), mask)
    expect(plain.cfg == SamplerConfig(),
           'without a config the default SamplerConfig is used')
    first = next(iter(plain))
    expect(first.bucket is not None and first.draw_index == 0,
           'a supply with the default config yields placed FoVs')
    try:
        FovSupply(cam, mask, object())
        expect(False, 'a config that is not a SamplerConfig is refused')
    except TypeError:
        expect(True, 'a config that is not a SamplerConfig is refused')
    derived = camera_plan(wsi.level_downsamples, cam.spec,
                          ds=cam.rect_w_l0 / float(cam.output_w), level=cam.level)
    expect(fov_plan_of(cam) == derived, 'fov_plan_of is camera_plan of the spec')
    given = FovSupply(camera(0), mask, sampler_cfg, plan=derived)
    expect([(s.meta.x, s.meta.y) for s in given.sampler]
           == [(m.x, m.y) for m in metas],
           'a plan passed in equals the plan derived from the camera')
    try:
        FovSupply(cam, mask, sampler_cfg, plan=camera_plan(
            wsi.level_downsamples, cam.spec, ds=99.0, level=0))
        expect(False, 'a plan for another FoV is refused')
    except ValueError:
        expect(True, 'a plan for another FoV is refused')
    expect(cam.spec.long_side == max(cam.output_w, cam.output_h)
           and (cam.spec.sensor_w, cam.spec.sensor_h)
           == (cam.output_w, cam.output_h),
           "the camera's spec is its own sensor")
    # Placing and reading are not the renderer's: no mask, no sampler, no
    # read of its own, and the tissue-ratio parameters before that are gone.
    expect(not any(hasattr(cam, n) for n in (
        'mask', 'set_sampler', 'FoV_bank', '_fov_draw', '_sampler_cfg', 'qfw',
        'read', 'tissue_ratio', 'region_protrusion_ratio',
        'max_consecutive_fail', 'required_region_side_l0')),
        'a Render carries no mask, no sampler, no read of its own')
    try:
        iter(cam)
        expect(False, 'a Render is not iterable')
    except TypeError:
        expect(True, 'a Render is not iterable')

    return 1 if expect.failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--wsi', required=True)
    ap.add_argument('--level', type=int, default=1)
    args = ap.parse_args()

    wsi = SafeSlide(args.wsi)
    level_mpp = wsi.base_mpp * wsi.level_downsamples[args.level]
    print(f'{Path(args.wsi).name}  L{args.level}  base_mpp={wsi.base_mpp:.4f}  '
          f'level_mpp={level_mpp:.4f}')
    with MaskMaker(MASK_RECIPES['hsv']) as masks:
        mask, _ = masks.mask(wsi)
    print('\n======== [supply] ========', flush=True)
    rc = run_supply(args, wsi, mask, level_mpp)
    print(f'\n======== {"ok" if rc == 0 else "FAIL"} ========')
    return rc


if __name__ == '__main__':
    sys.exit(main())
