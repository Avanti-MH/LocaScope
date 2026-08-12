#!/usr/bin/env python3
"""Pin Camera.output_to_level0 against pixels, not against its own derivation.

    python utilities/test_modules/test_camera_output_to_level0.py \
        --wsi /work/u26130998/datasets/.../BRACS_1228.svs [--level 1]

Why a pixel test and not an arithmetic one
------------------------------------------
The pooling experiment builds its queries by photographing a FoV and cutting
tiles out of it, then looks up each tile's answer by the level-0 coordinate this
method returns. If the mapping is wrong, every query's answer points somewhere
else and the experiment reports that no pooling can find anything -- a result
that looks like a finding rather than a bug.

The part most likely to be wrong is the sign of the inverse rotation, and a sign
error is invisible at 0 and 180 degrees. So the test does not check the formula;
it takes the shot, cuts a tile, rotates it back, and asks whether the WSI at the
computed place actually looks like that tile -- and whether it looks like it MORE
than the sign-flipped and one-tile-shifted alternatives do.

The camera is built photometric=False, scale fixed at 1 and no angle jitter, so
the shot is a pure lossless rotation of the source. Any residual difference is
resampling in the bounding-square read, not augmentation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _d in ('query_sim', 'utilities'):
    p = str(_ROOT / _d)
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                          # noqa: E402
import openslide                                            # noqa: E402

from camera import Camera                                   # noqa: E402
from config import DomainGapConfig                          # noqa: E402
from augment.geometry import apply_rotation                 # noqa: E402

TILE = 256
ROTS = (0, 90, 180, 270)


def geometry_only_cfg(query_mpp: float) -> DomainGapConfig:
    """A camera that only moves pixels around, so a mismatch means the map is wrong."""
    return DomainGapConfig(
        wh_ratio='45:32', MPixels=1.47456,      # the real photos' shape
        query_mpp=query_mpp,
        photometric=False,                      # no colour / vignette / lens / noise
        angle_jitter_deg=0.0,                   # exact multiples of 90 only
        scale_range=(1.0, 1.0),
        query_mpp_jitter=0.0,
    )


def pick_textured_position(cam, wsi, tries=40, seed=0):
    """A FoV position whose content actually varies.

    On blank glass every crop looks like every other crop, so a wrong mapping
    would score just as well as the right one and the test would pass while
    meaning nothing.
    """
    rng = np.random.default_rng(seed)
    q = cam.qfw
    w, h = wsi.dimensions
    best, best_std = None, -1.0
    for _ in range(tries):
        x = int(rng.integers(0, max(1, w - q.rect_w_l0 - q.bounding_square_side_l0)))
        y = int(rng.integers(0, max(1, h - q.rect_h_l0 - q.bounding_square_side_l0)))
        img = cam.capture(x, y, rotation=0)
        if img is None:
            continue
        s = float(np.asarray(img, dtype=np.float32).std())
        if s > best_std:
            best, best_std = (x, y), s
    if best is None or best_std < 8.0:
        raise RuntimeError(
            f'no textured FoV found in {tries} tries (best std {best_std:.1f}); '
            f'pass a different --seed or a slide with more tissue')
    return best, best_std


def read_at(wsi, level, ds, cx, cy):
    """A TILE-sized crop of the WSI centred on level-0 point (cx, cy)."""
    x0 = int(round(cx - TILE * ds / 2.0))
    y0 = int(round(cy - TILE * ds / 2.0))
    return np.asarray(wsi.read_region((x0, y0), level, (TILE, TILE)).convert('RGB'),
                      dtype=np.float32)


def mad(a, b):
    return float(np.abs(a - b).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--wsi', required=True)
    ap.add_argument('--level', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    wsi = openslide.OpenSlide(args.wsi)
    base_mpp = (float(wsi.properties.get(openslide.PROPERTY_NAME_MPP_X, 0.25))
                + float(wsi.properties.get(openslide.PROPERTY_NAME_MPP_Y, 0.25))) / 2
    level_mpp = base_mpp * wsi.level_downsamples[args.level]
    print(f'{Path(args.wsi).name}  L{args.level}  base_mpp={base_mpp:.4f}  '
          f'level_mpp={level_mpp:.4f}')

    cam = Camera(wsi, cfg=geometry_only_cfg(level_mpp), seed=args.seed)
    q = cam.qfw
    ds = q.rect_w_l0 / q.output_w            # level-0 px per output px
    print(f'output {q.output_w}x{q.output_h}  chosen_level={q.chosen_level}  '
          f'ds={ds:.3f}  tiles {q.output_h // TILE}x{q.output_w // TILE}')

    (x, y), std = pick_textured_position(cam, wsi, seed=args.seed)
    print(f'FoV at ({x}, {y})  std={std:.1f}\n')

    failures = []
    for rot in ROTS:
        img, params = cam.capture_with_gt(x, y, rotation=rot)
        assert img is not None, f'capture failed at rot={rot}'
        arr = np.asarray(img, dtype=np.float32)

        wins = 0
        rows = list(cam.output_tile_origins(x, y, TILE, rot_deg=rot, scale=1.0))
        for r, c, u, v, cx, cy in rows:
            tile = arr[v:v + TILE, u:u + TILE]
            # A tile of the rotated output is a rotated tile of the source, so
            # undo the rotation before comparing with the WSI.
            back = np.asarray(apply_rotation(tile.astype(np.uint8), -rot),
                              dtype=np.float32)

            here = mad(back, read_at(wsi, q.chosen_level, ds, cx, cy))
            # The two ways this could be wrong, scored the same way.
            flip = mad(back, read_at(wsi, q.chosen_level, ds,
                                     2 * (x + q.rect_w_l0 / 2) - cx,
                                     2 * (y + q.rect_h_l0 / 2) - cy))
            shift = mad(back, read_at(wsi, q.chosen_level, ds,
                                      cx + TILE * ds, cy))
            if here < flip and here < shift:
                wins += 1
            else:
                failures.append((rot, r, c, here, flip, shift))

        tag = 'ok  ' if wins == len(rows) else 'FAIL'
        print(f'  {tag} rot={rot:3d}  {wins}/{len(rows)} tiles matched their '
              f'computed position best')

    if failures:
        print('\nfirst few mismatches (MAD: computed / sign-flipped / shifted):')
        for rot, r, c, a, b, cc in failures[:6]:
            print(f'  rot={rot:3d} tile({r},{c})  {a:7.2f} / {b:7.2f} / {cc:7.2f}')
        print('\nIf 0 and 180 pass but 90 and 270 fail, the inverse rotation in '
              'Camera.output_to_level0 has the wrong sign -- swap the signs on '
              'du_s / dv_s.')
        return 1

    print('\nall rotations map back to the right place')
    return 0


if __name__ == '__main__':
    sys.exit(main())
