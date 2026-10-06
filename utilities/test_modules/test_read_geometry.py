"""test_read_geometry -- ReadGeometry: the sampler reserves what the reader reads.

No slide, no GPU: every check is arithmetic on the numbers `SlideReader` reads by.
A reserve is right when the read fits inside the footprint box grown by it,
and TIGHT when one pixel less does not fit -- so each check also runs a decoy
reserve that lets positions fall off the slide and requires it to fail:

    manifest   footprint-only reserve
    fov_plan   int(fp * hypot / long side)

The read the reserve is checked against is written out independently of
`read_rect` (`_independent_read`), so a reserve and a read computed by the
same function cannot pass by agreeing with themselves.

    python utilities/test_modules/test_read_geometry.py
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))

from _paths import setup_import_paths                            # noqa: E402

setup_import_paths()

from DsLadder import RungPlan                                    # noqa: E402
from ReadGeometry import (ReadSpec, FovGeometry, ReadRect,     # noqa: E402
                          level_for, level_px, reserve_margin)
from TileSampler import PlanSpec, _margin_of, with_camera       # noqa: E402

RUNGS = (1, 2, 4, 8, 16, 32)
#: A margin, output px, as a still FoV camera reads one (`camera.read_margin`).
MARGIN = 64
#: downsamples that are not a ladder rung: a measured BRACS level, an
#: arbitrary query magnification, and one whose products are never integers
DS_AWKWARD = (4.00014, 3.9, 7.3)
TILE = 256

_RESULTS = []


def check(name, fn):
    try:
        said = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   ({said})' if said else ''))
    except Exception as exc:                                     # noqa: BLE001
        _RESULTS.append((name, exc))
        print(f'  FAIL  {name}: {exc}')
        traceback.print_exc()


def _plan(rung: float, tile: int = TILE) -> RungPlan:
    fp = float(tile) * rung
    return RungPlan(rung_ds=float(rung), level=0, level_ds=1.0, shrink=float(rung),
                    tile_size=tile, read_size=int(fp), footprint_l0=fp)


def _fits(read: ReadRect, box: int, margin: int) -> bool:
    return read.inside(-margin, -margin, box + margin, box + margin)


# ── the read itself ──────────────────────────────────────────────────────────

def t_fov_geometry_is_query_from_wsi_arithmetic():
    """The four numbers, written out: the rect is `int(output * ds)`, the
    square its ceiled diagonal. In ds, so no mpp round trip can cost a px:
    `base * rung / base` is not always `rung` in floating point, and a
    `(mpp, base_mpp)` form would truncate whatever it came back as."""
    import math
    for rung in RUNGS:
        g = FovGeometry.of(TILE, TILE, rung)
        rect = int(TILE * rung)
        assert (g.rect_w_l0, g.rect_h_l0) == (rect, rect)
        assert g.square_out == 363
        assert g.square_l0 == 363 * int(rung), (rung, g.square_l0)
        assert g.square_l0 >= math.hypot(rect, rect), 'the square cannot rotate'
    g = FovGeometry.of(1440, 1024, 4.0)
    assert (g.rect_w_l0, g.rect_h_l0, g.square_out) == (5760, 4096, 1767)
    return 'rect, square_out, square_l0 = square_out * ds'


def t_the_square_centred_is_the_square_read():
    """The rotating read's origin is centred for `square_l0`; its length is
    `level_px(square_out)` level px. The two must be one square: at most one
    level px apart in level-0 terms, at every ds and level. The decoy is the
    square of the level-0 diagonal, `ceil(hypot(rect_l0))`: 725 at ds 2, where
    726 level-0 px are read."""
    import math
    worst = 0.0
    for ds, lv in ((1.0, 1.0), (2.0, 1.0), (4.0, 4.00014), (8.0, 4.00014),
                   (32.0, 16.0011), (3.9, 1.0), (7.3, 4.00014)):
        g = FovGeometry.of(TILE, TILE, ds)
        read_l0 = level_px(g.square_out, ds, lv) * lv
        gap = abs(read_l0 - g.square_l0)
        assert gap <= lv, f'ds {ds} level {lv}: read {read_l0:.1f}, square {g.square_l0}'
        worst = max(worst, gap / lv)
    old = int(math.ceil(math.hypot(512, 512)))
    assert old == 725 and level_px(363, 2.0, 1.0) == 726, 'the decoy'
    return f'largest gap {worst:.2f} level px; the old square was 725 for 726 read'


def t_the_level_and_its_pixels():
    """`level_for` never upsamples; `level_px` rounds, and a native read is
    the output size exactly. The decoys: a 5% window, which takes a coarser
    level and blows it up, and an `int` read size, 255 px for a 256 tile off
    BRACS's ds 4.00014 level."""
    bracs = (1.0, 4.00014, 16.0011, 32.0045)
    assert level_for(bracs, 4.0) == 1 and level_for(bracs, 2.0) == 0
    assert level_for(bracs, 3.9) == 0, 'a level 2.6% coarser was taken'
    old_window = abs(4.00014 - 3.9) / 4.00014 < 0.05
    assert old_window, 'the decoy: the old rule would have read level 1 here'
    assert level_px(TILE, 4.0, 4.00014) == TILE
    assert int(TILE * 4.0 / 4.00014) == TILE - 1, 'the decoy: int is one short'
    assert level_px(363, 4.0, 4.00014) == 363
    assert level_px(TILE, 8.0, 4.00014) == 2 * TILE
    try:
        level_for(bracs, 0.5)
    except ValueError:
        return 'level 1 for ds 4, level 0 for ds 3.9; 256 px, not 255'
    raise AssertionError('a ds finer than level 0 was accepted')


def t_read_rect_is_centred_and_shaped_as_the_three_crops():
    import math
    g = FovGeometry.of(1440, 1024, 4.0)
    sq = g.read_rect(1000, 2000, rotates=True)
    assert (sq.w, sq.h) == (g.square_l0, g.square_l0)
    assert sq.x0 == 1000 - (g.square_l0 - g.rect_w_l0) // 2
    assert sq.y0 == 2000 - (g.square_l0 - g.rect_h_l0) // 2
    plain = g.read_rect(1000, 2000, rotates=False)
    assert (plain.x0, plain.y0, plain.w, plain.h) == (1000, 2000, 5760, 4096)
    pad = g.read_rect(1000, 2000, rotates=False, margin_out=MARGIN)
    m = int(round(MARGIN * (5760 / 1440)))
    assert (pad.x0, pad.y0, pad.w, pad.h) == (1000 - m, 2000 - m, 5760 + 2 * m,
                                              4096 + 2 * m)
    big = g.read_rect(1000, 2000, rotates=True, margin_out=17)
    side = int(math.ceil((g.square_out + 34) * 4.0 - 1e-6))
    assert (big.w, big.h) == (side, side) and big.w > sq.w
    assert big.x0 == 1000 - (side - g.rect_w_l0) // 2
    assert g.read_out(True, 17) == (g.square_out + 34,) * 2
    assert g.read_out(False, MARGIN) == (1440 + 2 * MARGIN, 1024 + 2 * MARGIN)
    return f'square {g.square_l0}, +17 out px {side}, margin {m} l0 px'


# ── what a sampler reserves for a camera: ReadSpec.place / with_camera ─────

#: Every camera the project places: the routing heads' rotating tile, a plain
#: tile, SuperPathPoint's 3x pre-tile, and the microscope FoV rotating and not.
CAMERAS = (ReadSpec(TILE, TILE, rotates=True),
           ReadSpec(TILE, TILE),
           ReadSpec(TILE, TILE, margin_out=TILE),
           ReadSpec(1440, 1024, rotates=True),
           ReadSpec(1440, 1024, rotates=True, margin_out=17),
           ReadSpec(1440, 1024, margin_out=MARGIN))


def _independent_read(cam: ReadSpec, x: int, y: int, ds: float) -> ReadRect:
    """What the camera reads, written out by hand from the sensor and ds --
    NOT through read_rect, so agreement with the reserve is evidence and not
    a function compared with itself. The rotating square is the sensor's
    diagonal times ds (`FovGeometry`'s docstring says why not the level-0
    rectangle's own diagonal)."""
    import math
    rw, rh = int(cam.sensor_w * ds), int(cam.sensor_h * ds)
    if cam.rotates:
        square_out = math.ceil(math.hypot(cam.sensor_w, cam.sensor_h))
        side = int(math.ceil((square_out + 2 * cam.margin_out) * ds - 1e-6))
        return ReadRect(x - (side - rw) // 2, y - (side - rh) // 2, side, side)
    m = int(round(cam.margin_out * (rw / cam.sensor_w)))
    return ReadRect(x - m, y - m, rw + 2 * m, rh + 2 * m)


def t_the_reserve_holds_the_camera_read_and_is_tight():
    worst, n = 0, 0
    for cam in CAMERAS:
        for rung in RUNGS + DS_AWKWARD:
            plan = with_camera(_plan(rung, cam.long_side), cam)
            m = _margin_of(plan)
            box = int(plan.footprint_l0)
            fw = plan.fov_w_l0 or box
            fh = plan.fov_h_l0 or box
            ox, oy = cam.fov_offset(box, fw, fh)
            read = _independent_read(cam, ox, oy, rung)
            where = f'{cam.key()} ds {rung}'
            assert _fits(read, box, m), f'{where}: the read leaves the reserve'
            if m:
                assert not _fits(read, box, m - 1), f'{where}: not tight'
            worst, n = max(worst, m), n + 1
    return f'{n} cases, largest margin {worst} px'


def t_the_old_manifest_reserve_does_not_hold_it():
    """The decoy: the footprint alone. At rung 32 the read overhangs by about
    the whole margin -- the 1697 px measured on bracs/test."""
    plan = _plan(32)
    read = FovGeometry.of(TILE, TILE, 32).read_rect(0, 0, rotates=True)
    m = _margin_of(plan)
    assert m == 0 and not _fits(read, int(plan.footprint_l0), m)
    over = reserve_margin(read, 0, 0, int(plan.footprint_l0))
    assert over > 1600, over
    return f'overhang {over} px with no reserve'


def t_the_old_fov_reserve_misses_somewhere():
    """The second decoy: the FoV sampler's old reserve, `fp * hypot / long side`
    truncated for a rotating camera and the footprint alone for a still one --
    the 1 px rotation shortfall and the never-reserved sensor margin."""
    import math
    missed = []
    for cam in (ReadSpec(1440, 1024, rotates=True),
                ReadSpec(1440, 1024, margin_out=MARGIN)):
        for ds in (1.0, 4.0, 16.0, 32.0) + DS_AWKWARD:
            fp = 1440.0 * ds
            box = int(fp)
            old = fp * (math.hypot(1440, 1024) / 1440 if cam.rotates else 1.0)
            m_old = (int(old) - box) // 2
            geo = cam.geometry(ds)
            ox, oy = cam.fov_offset(box, geo.rect_w_l0, geo.rect_h_l0)
            if not _fits(_independent_read(cam, ox, oy, ds), box, m_old):
                missed.append((cam.key(), ds))
    assert missed, 'the old reserve held everywhere: the decoy proves nothing'
    return f'{len(missed)} of {2 * (4 + len(DS_AWKWARD))} cases miss'


def t_plan_spec_key_names_the_camera():
    keys = {cam.key(): PlanSpec('ladder', RUNGS, camera=cam).key() for cam in CAMERAS}
    assert len(set(keys.values())) == len(CAMERAS), keys
    assert all(k.startswith('ladder-1-2-4-8-16-32-') for k in keys.values()), keys
    try:
        PlanSpec('ladder', RUNGS)
    except ValueError:
        return ', '.join(sorted(keys.values()))
    raise AssertionError('a PlanSpec with no camera was accepted')


_SECTIONS = {
    'read': ['t_fov_geometry_is_query_from_wsi_arithmetic',
             't_the_level_and_its_pixels',
             't_the_square_centred_is_the_square_read',
             't_read_rect_is_centred_and_shaped_as_the_three_crops'],
    'place': ['t_the_reserve_holds_the_camera_read_and_is_tight',
              't_the_old_manifest_reserve_does_not_hold_it',
              't_the_old_fov_reserve_misses_somewhere',
              't_plan_spec_key_names_the_camera'],
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
