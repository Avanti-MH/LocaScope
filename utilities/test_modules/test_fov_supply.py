#!/usr/bin/env python3
"""Tests for `query_sim/FovSupply.py`: a FoV's flow, TileSampler -> Render.

    python utilities/test_modules/test_fov_supply.py --wsi <slide> [--level 1]

Run through `jobscripts/TestReadPath.sh`, which resolves slide names and loops.

`equiv` pins FovSupply over a PlanSpec against a frozen one-camera FovSupply
and the `camera_plan` it drew over: given that rung as a PlanSpec, the same
positions, the same pixels, the same gap (effective_mpp aside, which names the
objective's ds), and the frozen shot's bucket / background / origin / overlap
equal to the meta they came from.
`supply` checks what it is now: one supply across several rungs, each through
its own objective; reproducible whatever the camera's own seed; a photo the
same alone or in order; the cached draw the same as the drawn one; a plan for
another camera, or not a PlanSpec, refused; and a Render that carries no mask
and no sampler. (The sampler's arithmetic for a rectangular camera is
`test_tile_sampler.py`, section fov; the render is `test_camera.py`.)
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
import tempfile
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
from FovSupply import FovSupply                                  # noqa: E402
from DsLadder import RungPlan                                    # noqa: E402
from ReadGeometry import (REAL_PHOTO_SENSOR as SENSOR, ReadSpec,  # noqa: E402
                          level_px)
from SafeSlide import SafeSlide                                  # noqa: E402
from SlideReader import SlideReader                              # noqa: E402
from TileSampler import PlanSpec, SamplerConfig, TileSampler, with_camera  # noqa: E402
from TissueMaskConfig import MASK_RECIPES, MaskMaker             # noqa: E402


def fov_cfg() -> DomainGapConfig:
    """The camera the window bench uses: optics and colour on, geometry off."""
    return DomainGapConfig(
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


# ── equiv: a frozen one-camera FovSupply ────────────────────────────────────

def _old_camera_plan(level_downsamples, camera, ds: float, level: int) -> RungPlan:
    """FROZEN: TileSampler.camera_plan -- the camera's one 'F' rung, footprint
    `long_side * ds`, at the camera's own level."""
    tile = camera.long_side
    level_ds = float(level_downsamples[level])
    plan = RungPlan(rung_ds=float(ds), level=int(level), level_ds=level_ds,
                    shrink=float(ds) / level_ds, tile_size=tile,
                    read_size=level_px(tile, ds, level_ds),
                    footprint_l0=float(tile) * float(ds), stack_kind='F')
    return with_camera(plan, camera)


def _old_bank(cam, mask, cfg):
    """FROZEN: the one-camera FovSupply(cam, mask, cfg).bank() -- draw 0 over
    fov_plan_of(cam), every position rendered by `cam` itself at its FoV
    rectangle, rng of (seed, x, y, ds, pass 0)."""
    plan = _old_camera_plan(cam.wsi.level_downsamples, cam.spec,
                            ds=cam.rect_w_l0 / float(cam.output_w), level=cam.level)
    sampler = TileSampler(cam.wsi, mask, cfg).sample([plan])
    out = []
    for meta in [s.meta for s in sampler]:
        key = f'{cfg.seed}|{meta.x}|{meta.y}|{meta.ds:g}|{0}'
        rng = random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))
        x0, y0, _, _ = meta.fov_rect
        image, params = cam.capture_with_gt(x0, y0, rng=rng)
        if image is None:
            continue
        out.append(dict(image=image, params=params, gt=(x0, y0),
                        bucket=meta.bucket, fov_background=meta.score,
                        origin=meta.origin, overlap_max=meta.overlap_max,
                        x=meta.x, y=meta.y, ds=meta.ds))
    return out


def run_equiv(cam, mask, cfg, expect) -> list:
    old = _old_bank(cam, mask, cfg)
    rung = cam.rect_w_l0 / float(cam.output_w)
    new = list(FovSupply(cam, PlanSpec('ladder', (rung,), camera=cam.spec), cfg, mask))
    print(f'  {len(new)} FoVs (old bank {len(old)}) at rung {rung:g}', flush=True)
    expect(len(new) == len(old) and len(new) > 0, 'as many FoVs as the old bank')
    expect(all((m.x, m.y, m.ds) == (o['x'], o['y'], o['ds'])
               and tuple(m.fov_rect[:2]) == o['gt'] for (m, _, _), o in zip(new, old)),
           'the old rung as a PlanSpec: the same positions, in the same order')
    expect(all(_same(img, o['image']) for (_, img, _), o in zip(new, old)),
           'the same pixels, bit for bit')

    def drawn(p):
        return {k: v for k, v in p.items() if k != 'effective_mpp'}
    expect(all(drawn(p) == drawn(o['params']) for (_, _, p), o in zip(new, old)),
           "the same gap drawn (effective_mpp aside: it names the objective's ds)")
    expect(all(m.bucket == o['bucket'] and m.score == o['fov_background']
               and m.origin == o['origin'] and m.overlap_max == o['overlap_max']
               for (m, _, _), o in zip(new, old)),
           "the old shot's bucket / background / origin / overlap are the meta's")
    return new


# ── supply: what it is now ──────────────────────────────────────────────────

def run_supply(wsi, mask, level_ds, new, expect) -> None:
    reader = SlideReader(wsi)
    cam = Render(reader, SENSOR, fov_cfg(), ds=level_ds, seed=0)
    shape = (cam.output_h, cam.output_w, 3)
    expect(all(tuple(img.shape) == shape and img.dtype == np.uint8 for _, img, _ in new),
           f'every FoV {shape} uint8')

    # One supply over several rungs, each through its own objective.
    full = DomainGapConfig()
    microscope = Render(reader, SENSOR, full, ds=1.0, seed=0)
    rungs = tuple(float(d) for d in wsi.level_downsamples[:2])
    plan = PlanSpec('ladder', rungs, camera=microscope.spec)
    cfg = SamplerConfig(n_per_rung=3, seed=3)
    supply = FovSupply(microscope, plan, cfg, mask)
    got = list(supply)
    seen = sorted({float(m.ds) for m, _, _ in got})
    checked = 0
    for m, img, _ in got:
        ref = Render(reader, SENSOR, full, ds=float(m.ds))
        want, _ = ref.capture_with_gt(
            m.fov_rect[0], m.fov_rect[1],
            rng=random.Random(int(hashlib.sha256(
                f'3|{m.x}|{m.y}|{m.ds:g}|0'.encode()).hexdigest()[:16], 16)))
        checked += _same(img, want)
    expect(len(got) > 0 and checked == len(got),
           f'one supply across rungs {seen}: every FoV is Render at its own ds '
           f'({checked}/{len(got)})')

    # The picture depends on the draw's seed and the position only.
    other = list(FovSupply(Render(reader, SENSOR, full, ds=1.0, seed=99), plan,
                           cfg, mask))
    expect(len(other) == len(got) and all(_same(a[1], b[1]) for a, b in zip(got, other)),
           'a microscope with another seed takes the same photos')
    metas = [m for m, _, _ in got]
    alone = [supply.photo(m)[0] for m in reversed(metas)][::-1]
    expect(all(_same(a, b[1]) for a, b in zip(alone, got)),
           'a photo is the same taken alone, in reverse order')

    # The cached draw is the drawn one: a miss writes it, a hit reads it back.
    with tempfile.TemporaryDirectory() as root:
        with MaskMaker(MASK_RECIPES['hsv']) as masks:
            miss = FovSupply.cached(microscope, plan, cfg, root, masks=masks)
            first = [(m.x, m.y, m.ds) for m, _, _ in miss]
            hit = FovSupply.cached(microscope, plan, cfg, root, masks=masks)
            again = list(hit)
        expect(first == [(m.x, m.y, m.ds) for m in metas]
               and [(m.x, m.y, m.ds) for m, _, _ in again] == first
               and all(_same(a[1], b[1]) for a, b in zip(again, got))
               and hit.sampler.cache_info['samples_hit'],
               'FovSupply.cached: the same positions and photos, the second a cache hit')

    # Plans and configs that are not this camera's.
    try:
        FovSupply(microscope, PlanSpec('ladder', rungs, camera=ReadSpec(256, 256)),
                  cfg, mask)
        expect(False, 'a plan for another camera is refused')
    except ValueError:
        expect(True, 'a plan for another camera is refused')
    try:
        FovSupply(microscope, plan.plans_for(wsi)[0], cfg, mask)
        expect(False, 'a RungPlan in place of a PlanSpec is refused')
    except TypeError:
        expect(True, 'a RungPlan in place of a PlanSpec is refused')
    try:
        FovSupply(microscope, plan, object(), mask)
        expect(False, 'a config that is not a SamplerConfig is refused')
    except TypeError:
        expect(True, 'a config that is not a SamplerConfig is refused')

    # Placing and reading are not the renderer's.
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--wsi', required=True)
    ap.add_argument('--level', type=int, default=1)
    args = ap.parse_args()

    wsi = SafeSlide(args.wsi)
    level_ds = float(wsi.level_downsamples[args.level])
    level_mpp = wsi.base_mpp * level_ds
    print(f'{Path(args.wsi).name}  L{args.level}  base_mpp={wsi.base_mpp:.4f}  '
          f'level_mpp={level_mpp:.4f}')
    with MaskMaker(MASK_RECIPES['hsv']) as masks:
        mask, _ = masks.mask(wsi)
    expect = _Expect()
    cam = Render(SlideReader(wsi), SENSOR, fov_cfg(), ds=level_ds, seed=0)
    cfg = SamplerConfig(n_per_rung=8, seed=5)
    rung = cam.rect_w_l0 / float(cam.output_w)
    try:
        FovSupply(cam, PlanSpec('ladder', (rung,), camera=cam.spec), cfg, mask).sampler
    except RuntimeError as exc:
        if 'No FoV position' not in str(exc):
            raise
        print(f'  SKIPPED  no FoV position at level {args.level} on this slide:'
              f'\n{exc}')
        return 0
    print('\n======== [equiv] ========', flush=True)
    new = run_equiv(cam, mask, cfg, expect)
    print('\n======== [supply] ========', flush=True)
    run_supply(wsi, mask, level_ds, new, expect)
    rc = 1 if expect.failures else 0
    print(f'\n======== {"ok" if rc == 0 else "FAIL"} ========')
    return rc


if __name__ == '__main__':
    sys.exit(main())
