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
import random
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
from SafeSlide import SafeSlide                             # noqa: E402
from config import DomainGapConfig                          # noqa: E402
from augment.geometry import apply_rotation                 # noqa: E402

TILE = 256
ROTS = (0, 90, 180, 270)

#: A decoy position only counts as BEATING the computed one if it wins by
#: both of these -- a decoy that merely ties is not evidence of a wrong map.
#:
#: Why two conditions and not just the ratio: on a near-blank tile every
#: candidate matches to within rounding, so the two MADs are both noise and
#: their RATIO is unbounded -- 0.01 against 0.02 is a coin flip that happens
#: to read as "twice as good". A ratio test alone cannot tell that regime
#: from a real one at any constant. The absolute term is what says "both of
#: these are zero, so neither won"; it is measured on the DECISION (how far
#: apart the two matches landed), not on the tile's texture.
#:
#: The margins are wide because the real signal is not subtle. A wrong
#: inverse rotation scored MAD 5.2 against the correct position's 47.5 when
#: this file caught it (see output_to_level0's docstring) -- ratio 0.11, gap
#: 42.3. A correct map on textured tissue scores 0.02 against a decoy's 0.32.
#: Both regimes sit an order of magnitude clear of these cuts.
DECOY_RATIO = 0.5    # decoy must match at least twice as well ...
DECOY_ABS = 1.0      # ... and by more than 1 MAD unit on a 0-255 scale


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


def full_gap_cfg(query_mpp: float) -> DomainGapConfig:
    """Every augmentation ON -- the opposite of `geometry_only_cfg`.

    The mapping test wants a camera that only moves pixels, so it turns the
    photometric stage off. The reproducibility check below wants the exact
    opposite: `photometric=False` returns before the sensor stage runs, so it
    would never execute `apply_noise` at all, and a check that never runs the
    noise cannot notice the noise being irreproducible.
    """
    return DomainGapConfig(wh_ratio='45:32', MPixels=1.47456,
                           query_mpp=query_mpp)


def check_same_seed_same_pixels(wsi, query_mpp, x, y, seed) -> list:
    """Two Cameras, one seed, one position -> must be BIT-IDENTICAL.

    Nothing asserted this until 2026-09-16, which is how `apply_stage_shift`
    and `apply_noise` drew from the process-global `np.random` for as long as
    they did: `Camera.__init__` used to call `np.random.seed(seed)`, so in the
    build-one-camera-then-shoot order every existing caller happens to use,
    the global draws came out reproducible anyway and nothing looked wrong.

    So the order here is deliberate and is the part that must not be
    "simplified": BOTH cameras are built BEFORE EITHER shoots. Under the old
    code that alone breaks it -- building B re-seeds the global, A's shot then
    consumes it, and B's shot gets the advanced state. Building A, shooting A,
    building B, shooting B would have passed on the buggy code and tested
    nothing. The `np.random` call between the two shots is the same argument
    made louder: a shot must not depend on global state at all, so disturbing
    it must not change the pixels.

    That interleaved order is not hypothetical -- it is exactly what
    `training/MppRoutingHead/Datasets.py`'s `CameraBank` does (one Camera per
    rung of a slide, all built before any of them shoots).
    """
    failures = []
    cfg = full_gap_cfg(query_mpp)

    cam_a = Camera(wsi, cfg=cfg, seed=seed)
    cam_b = Camera(wsi, cfg=cfg, seed=seed)          # built BEFORE a shoots
    img_a, params_a = cam_a.capture_with_gt(x, y)
    np.random.random(1000)                            # disturb the global state
    img_b, params_b = cam_b.capture_with_gt(x, y)

    if img_a is None or img_b is None:
        failures.append(('same-seed', 'capture returned None'))
    elif not np.array_equal(img_a, img_b):
        differing = int((np.asarray(img_a) != np.asarray(img_b)).any(axis=2).sum())
        failures.append((
            'same-seed',
            f'{differing} px differ; '
            f'stage_shift a={params_a["stage_shift_dx"]},{params_a["stage_shift_dy"]} '
            f'b={params_b["stage_shift_dx"]},{params_b["stage_shift_dy"]}  '
            f'noise_seed a={params_a["noise_seed"]} b={params_b["noise_seed"]}'))
    print(f'  {"ok  " if not failures else "FAIL"} two Cameras, same seed, '
          f'built before either shoots -> identical pixels')

    # The property training/MppRoutingHead/Datasets.py's eval split depends on:
    # one Camera, a per-call rng derived from the sample's identity, same
    # answer every time regardless of what ran in between.
    before = len(failures)
    cam_c = Camera(wsi, cfg=cfg, seed=None)
    img_1 = cam_c.capture(x, y, rng=random.Random(99))
    np.random.random(1000)
    img_2 = cam_c.capture(x, y, rng=random.Random(99))
    if img_1 is None or img_2 is None:
        failures.append(('per-call-rng', 'capture returned None'))
    elif not np.array_equal(img_1, img_2):
        differing = int((np.asarray(img_1) != np.asarray(img_2)).any(axis=2).sum())
        failures.append(('per-call-rng', f'{differing} px differ'))
    print(f'  {"ok  " if len(failures) == before else "FAIL"} one Camera, same '
          f'capture(rng=) twice -> identical pixels')

    return failures


def read_at(wsi, level, ds, cx, cy):
    """A TILE-sized crop of the WSI centred on level-0 point (cx, cy)."""
    x0 = int(round(cx - TILE * ds / 2.0))
    y0 = int(round(cy - TILE * ds / 2.0))
    # read_region_rgb: `.convert('RGB')` drops the alpha and paints every
    # unphotographed pixel black, which this check would then compare against
    # a rotated copy of the same black and call agreement.
    return np.asarray(wsi.read_region_rgb((x0, y0), level, (TILE, TILE)),
                      dtype=np.float32)


def mad(a, b):
    return float(np.abs(a - b).mean())


def beats(here: float, decoy: float) -> bool:
    """Did `decoy` match the tile better than the computed position, by
    enough that it cannot be noise? See DECOY_RATIO / DECOY_ABS."""
    return decoy < here * DECOY_RATIO and (here - decoy) > DECOY_ABS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--wsi', required=True)
    ap.add_argument('--level', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    wsi = SafeSlide(args.wsi)
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
            if not any(beats(here, decoy) for decoy in (flip, shift)):
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
        print(f'\nA decoy only counts as winning if it matches at least '
              f'{1 / DECOY_RATIO:.0f}x better AND by more than {DECOY_ABS} MAD, '
              f'so anything listed above is a decisive loss, not a tie.')
        print('If the SIGN-FLIPPED column is the one winning, and it wins at 90 '
              'and 270 while 0 and 180 pass, the inverse rotation in '
              'Camera.output_to_level0 has the wrong sign -- swap the signs on '
              'du_s / dv_s. (0 and 180 cannot see that error: the two forms '
              'coincide there.) If the SHIFTED column is the one winning, the '
              'offset is wrong rather than the rotation, and the sign is not '
              'the thing to touch.')
        return 1

    print('\nall rotations map back to the right place')

    print('\nseed reproducibility (every augmentation ON):')
    seed_failures = check_same_seed_same_pixels(wsi, level_mpp, x, y, args.seed)
    if seed_failures:
        print('\nmismatches:')
        for what, detail in seed_failures:
            print(f'  {what}: {detail}')
        print('\nA shot must depend only on the rng it was given. If this fails, '
              'something in the augment chain is drawing from the process-global '
              '`np.random` again -- grep query_sim/ for `np.random.` and check '
              'that every draw goes through `_sample_params`\'s rng (an offset '
              'recorded in `params`, or a seed recorded there) instead.')
        return 1
    print('  same seed and same capture(rng=) both reproduce exactly')
    return 0


if __name__ == '__main__':
    sys.exit(main())
