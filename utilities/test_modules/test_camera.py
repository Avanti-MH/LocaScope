#!/usr/bin/env python3
"""Everything that checks `query_sim/camera.py`, in one file.

    python utilities/test_modules/test_camera.py --wsi <slide> [--level 1]
    python utilities/test_modules/test_camera.py --wsi <slide> --only map seed

Run through `jobscripts/TestReadPath.sh`, which resolves slide names and loops.

Placing FoVs is `FovSupply`'s, tested in `test_fov_supply.py`. Each section
runs on its own and the exit status is the worst of them, so a failing map
check does not hide the seed check.

    map         `Render.output_to_level0` against pixels, not against its own
                derivation. The sign of the inverse rotation is invisible at 0
                and 180 degrees, so all four of 0/90/180/270 are checked.
    seed        two Renders from one seed, and one Render with the same
                `capture(rng=)` twice, must give bit-identical pixels.
    augment     the optimised `query_sim/augment/` bodies against the legacy
                ones, value for value, plus what the whole capture costs.

Two slides, two formats, on purpose. BRACS is SVS and steps 4x per pyramid
level; Ki67 is MIRAX and steps 2x (CLAUDE.md, "Pyramid spacing decides how hard
stage 1 is"). `SlideReader.level_of` picks the level by searching for the coarsest
level not coarser than the requested ds, so the two pyramids send it down different
branches -- a test on one format alone leaves the other's arithmetic
unexercised. `TestReadPath.sh` runs both.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_ROOT))

from _paths import job_result_dir, setup_import_paths            # noqa: E402

setup_import_paths()

import numpy as np                                               # noqa: E402
import openslide                                                 # noqa: E402

from camera import Render                                        # noqa: E402
from ReadGeometry import SENSOR_MARGIN                           # noqa: E402
from ReadGeometry import ReadSpec                                # noqa: E402
from SlideReader import SlideReader                              # noqa: E402
from config import DomainGapConfig                               # noqa: E402
from SafeSlide import SafeSlide                                  # noqa: E402
from augment import field, geometry, lens                        # noqa: E402
from augment.geometry import apply_rotation                      # noqa: E402


# ==============================================================================
#  map / seed -- shared
# ==============================================================================


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
#: inverse rotation scores MAD 5.2 against the correct position's 47.5 (see
#: output_to_level0's docstring) -- ratio 0.11, gap
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


# ==============================================================================
#  map
# ==============================================================================


def pick_textured_position(cam, wsi, tries=40, seed=0):
    """A FoV position whose content actually varies.

    On blank glass every crop looks like every other crop, so a wrong mapping
    would score just as well as the right one and the test would pass while
    meaning nothing.
    """
    rng = np.random.default_rng(seed)
    q = cam
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


def run_map(args, wsi, level_mpp) -> int:
    cam = Render(SlideReader(wsi), cfg=geometry_only_cfg(level_mpp), seed=args.seed)
    q = cam
    ds = q.rect_w_l0 / q.output_w            # level-0 px per output px
    print(f'output {q.output_w}x{q.output_h}  chosen_level={q.level}  '
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

            here = mad(back, read_at(wsi, q.level, ds, cx, cy))
            # The two ways this could be wrong, scored the same way.
            flip = mad(back, read_at(wsi, q.level, ds,
                                     2 * (x + q.rect_w_l0 / 2) - cx,
                                     2 * (y + q.rect_h_l0 / 2) - cy))
            shift = mad(back, read_at(wsi, q.level, ds,
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
              'Render.output_to_level0 has the wrong sign -- swap the signs on '
              'du_s / dv_s. (0 and 180 cannot see that error: the two forms '
              'coincide there.) If the SHIFTED column is the one winning, the '
              'offset is wrong rather than the rotation, and the sign is not '
              'the thing to touch.')
        return 1

    print('\nall rotations map back to the right place')
    return 0



# ==============================================================================
#  seed
# ==============================================================================


def check_same_seed_same_pixels(wsi, query_mpp, x, y, seed) -> list:
    """Two Renders, one seed, one position -> must be BIT-IDENTICAL.

    The order here is deliberate and is the part that must not be
    "simplified": BOTH are built BEFORE EITHER shoots. A constructor that
    seeded the global `np.random` would pass a build-shoot-build-shoot order
    and fail this one -- building B re-seeds the global, A's shot consumes it,
    and B's shot gets the advanced state. The `np.random` call between the two shots is the same argument
    made louder: a shot must not depend on global state at all, so disturbing
    it must not change the pixels.

    That interleaved order is not hypothetical -- it is exactly what
    `training/MppRoutingHead/Datasets.py`'s `CameraBank` does (one Render per
    rung of a slide, all built before any of them shoots).
    """
    failures = []
    cfg = full_gap_cfg(query_mpp)

    cam_a = Render(SlideReader(wsi), cfg=cfg, seed=seed)
    cam_b = Render(SlideReader(wsi), cfg=cfg, seed=seed)          # built BEFORE a shoots
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
    print(f'  {"ok  " if not failures else "FAIL"} two Renders, same seed, '
          f'built before either shoots -> identical pixels')

    # The property training/MppRoutingHead/Datasets.py's eval split depends on:
    # one Render, a per-call rng derived from the sample's identity, same
    # answer every time regardless of what ran in between.
    before = len(failures)
    cam_c = Render(SlideReader(wsi), cfg=cfg, seed=None)
    img_1 = cam_c.capture(x, y, rng=random.Random(99))
    np.random.random(1000)
    img_2 = cam_c.capture(x, y, rng=random.Random(99))
    if img_1 is None or img_2 is None:
        failures.append(('per-call-rng', 'capture returned None'))
    elif not np.array_equal(img_1, img_2):
        differing = int((np.asarray(img_1) != np.asarray(img_2)).any(axis=2).sum())
        failures.append(('per-call-rng', f'{differing} px differ'))
    print(f'  {"ok  " if len(failures) == before else "FAIL"} one Render, same '
          f'capture(rng=) twice -> identical pixels')

    return failures


def run_seed(args, wsi, level_mpp) -> int:
    cam = Render(SlideReader(wsi), cfg=geometry_only_cfg(level_mpp), seed=args.seed)
    (x, y), _ = pick_textured_position(cam, wsi, seed=args.seed)
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



# ==============================================================================
#  augment
# ==============================================================================


#: Timed repeats per op per shot. The report takes the median, so one
#: scheduler hiccup does not become the measurement.
REPEATS = 3


#: Below this the crop is glass or an empty margin. On a constant image every
#: rewrite agrees trivially -- `np.where` against a gather, a cached grid
#: against a rebuilt one -- so the comparison would pass while meaning nothing.
MIN_STD = 12.0


def augment_cfg(query_mpp: float) -> DomainGapConfig:
    """The camera the benches use: every photometric effect on.

    Rotation and scale are pinned the way `bench_offgrid_score` pins them,
    which is exactly the case where the two `.copy()` calls fire, so the
    optimisation is measured under the conditions that motivated it.
    """
    return DomainGapConfig(
        wh_ratio='45:32', MPixels=1.47456,      # the real photos' shape
        query_mpp=query_mpp,
        photometric=True,
        angle_jitter_deg=0.0,
        scale_range=(1.0, 1.0),
        query_mpp_jitter=0.0,
        stage_shift_max=0,
    )


def textured_shots(camera, wsi, n_shots: int, seed: int) -> list:
    """(xy, raw square, params) from places whose pixels actually vary.

    `reader.read` with the sensor margin returns the array the augment chain receives at rotation 0 --
    the FoV rect plus SENSOR_MARGIN, which is what the camera reads once it knows
    the exposure does not turn -- and `capture_with_gt` returns the parameter
    set the chain would have been given. Both come straight off the camera, so
    nothing in this file invents an input.
    """
    rng = np.random.default_rng(seed)
    q = camera
    width, height = wsi.dimensions
    span_x = max(1, width - q.bounding_square_side_l0 - 1)
    span_y = max(1, height - q.bounding_square_side_l0 - 1)

    out = []
    for _ in range(400):
        if len(out) >= n_shots:
            break
        x = int(rng.integers(0, span_x))
        y = int(rng.integers(0, span_y))
        raw = q.reader.read(x, y, ReadSpec(q.output_w, q.output_h,
                                           margin_out=SENSOR_MARGIN), q.ds)
        if raw is None:
            continue
        array = np.array(raw)
        if float(array.std()) < MIN_STD:
            continue
        _, params = camera.capture_with_gt(x, y, rotation=0)
        if params is None:
            continue
        out.append(((x, y), array, params))
    return out


def compare(legacy_fn, fast_fn, image, *args) -> dict:
    """Run both bodies on the same pixels and say how far apart they land.

    Both are called once before timing, so a lazily built cache is not charged
    to the fast path's per-shot cost. That one-time cost is real, but it is
    paid once per frame size and amortised over the 1089 displacements of a
    grid point; billing it to every shot would be the wrong number.
    """
    a = np.asarray(legacy_fn(image, *args))
    b = np.asarray(fast_fn(image, *args))

    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    n_differ = int((diff > 0).sum())

    def median_ms(function):
        times = []
        for _ in range(REPEATS):
            start = time.perf_counter()
            function(image, *args)
            times.append((time.perf_counter() - start) * 1e3)
        return float(np.median(times))

    return {'max_abs': int(diff.max()), 'n_differ': n_differ,
            'n_values': int(diff.size), 'frac_differ': n_differ / diff.size,
            'legacy_ms': median_ms(legacy_fn), 'fast_ms': median_ms(fast_fn)}


def vignette_float32(image, strength):
    """The inexact variant, run with the flag on and then restored."""
    previous = field.VIGNETTE_FLOAT32
    field.VIGNETTE_FLOAT32 = True
    try:
        return field._apply_vignette_fast(image, strength)
    finally:
        field.VIGNETTE_FLOAT32 = previous


def cases(params: dict) -> list:
    """(name, legacy, fast, args, gated) for one sampled parameter set.

    `gated` marks the rewrites whose claim is exact equality. The float32
    vignette is not gated: it is here to be measured, not to pass.
    """
    return [
        ('rotation @ 0', geometry._apply_rotation_legacy,
         geometry._apply_rotation_fast, (0.0,), True),
        ('scale @ 1', geometry._apply_scale_legacy,
         geometry._apply_scale_fast, (1.0,), True),
        ('vignette f64', field._apply_vignette_legacy,
         field._apply_vignette_fast, (params['vignette_strength'],), True),
        ('distortion', lens._apply_distortion_legacy,
         lens._apply_distortion_fast,
         (params['distortion_k1'], params['distortion_k2']), True),
        ('vignette f32 *', field._apply_vignette_legacy,
         vignette_float32, (params['vignette_strength'],), False),
    ]


# ══════════════════════════════════════════════════════════════════════════════
#  The whole capture
# ══════════════════════════════════════════════════════════════════════════════

def set_fast(enabled: bool) -> None:
    """Flip all three augment modules at once.

    The public functions dispatch on these flags at call time and every caller
    binds the dispatcher rather than a body, so this changes what a Render
    capture actually runs without importing anything from `pipeline`.
    """
    field.USE_FAST = lens.USE_FAST = geometry.USE_FAST = enabled


def time_captures(wsi, cfg, positions, seed: int) -> dict:
    """`Render.capture_with_gt` timed both ways, on identical parameter draws.

    A capture is read + augment + centre crop, and only the middle term
    changes. Including the other two is the point: the per-op table says what
    the rewrite saves, this says what fraction of a real shot that is, which is
    the number that decides whether a bench gets shorter.

    Both passes build a fresh Render from the SAME seed, so `_py_rng` hands
    them the same sequence of domain-gap parameters. Without that one pass
    could draw a larger `k1` or `vignette_strength` more often than the other
    and the difference would be luck rather than code -- the parameters are
    redrawn on every capture by design.

    Each position is timed legacy-then-fast at even indices and fast-then-
    legacy at odd ones, so the page cache warming on the first of a pair does
    not systematically favour the second. A warm-up pass runs first anyway, so
    neither side is charged for the cold read of a WSI block.
    """
    def capture_all(fast: bool, record: list | None):
        set_fast(fast)
        camera = Render(SlideReader(wsi), cfg=cfg, seed=seed)
        for x, y in positions:
            start = time.perf_counter()
            camera.capture_with_gt(x, y, rotation=0)
            if record is not None:
                record.append((time.perf_counter() - start) * 1e3)

    capture_all(True, None)                       # warm the page cache
    capture_all(False, None)

    legacy, fast = [], []
    for index, (x, y) in enumerate(positions):
        order = (False, True) if index % 2 == 0 else (True, False)
        for use_fast in order:
            set_fast(use_fast)
            camera = Render(SlideReader(wsi), cfg=cfg, seed=seed + index)
            start = time.perf_counter()
            camera.capture_with_gt(x, y, rotation=0)
            elapsed = (time.perf_counter() - start) * 1e3
            (fast if use_fast else legacy).append(elapsed)

    # The read alone, so the report can say how much of a capture is the part
    # no rewrite in this file touches.
    set_fast(True)
    cam = Render(SlideReader(wsi), cfg=cfg, seed=seed)
    pad = ReadSpec(cam.output_w, cam.output_h, margin_out=SENSOR_MARGIN)
    reads = []
    for x, y in positions:
        start = time.perf_counter()
        cam.reader.read(x, y, pad, cam.ds)
        reads.append((time.perf_counter() - start) * 1e3)

    set_fast(True)
    return {'legacy_ms': float(np.median(legacy)),
            'fast_ms': float(np.median(fast)),
            'read_ms': float(np.median(reads)),
            'n': len(positions)}


def print_capture(timing: dict) -> None:
    legacy, fast, read = timing['legacy_ms'], timing['fast_ms'], timing['read_ms']
    saved = legacy - fast
    print(f'\nwhole Camera.capture_with_gt, median of {timing["n"]} shots')
    print('-' * 87)
    print(f'  legacy      {legacy:8.1f} ms')
    print(f'  fast        {fast:8.1f} ms      '
          f'{saved:.1f} ms saved, {saved / legacy * 100:.1f}% of a shot, '
          f'{legacy / fast:.2f}x')
    print(f'  of which read {read:6.1f} ms      '
          f'{read / legacy * 100:.0f}% of the capture is the WSI read, which no '
          f'rewrite here touches')
    print(f'{"":16}augment chain alone: {legacy - read:.1f} -> {fast - read:.1f} '
          f'ms ({(legacy - read) / max(1e-9, fast - read):.2f}x)')
    print(f'\n  NOTE  "legacy" here is the CURRENT pipeline running the old op '
          f'bodies. The crop-first\n        order, the narrow read and the '
          f'removal of field_mask are in BOTH numbers, so\n        this '
          f'{saved:.0f} ms is the body rewrites alone.')


def summarise(rows: list) -> list:
    """Worst case per op across shots, not the average.

    An op that agrees on four photographs and disagrees on the fifth has a
    problem, and a mean would bury it under the four.
    """
    by_op = {}
    for row in rows:
        entry = by_op.setdefault(row['op'], {
            'op': row['op'], 'gated': row['gated'], 'shots': 0, 'max_abs': 0,
            'n_differ': 0, 'n_values': 0, 'legacy_ms': [], 'fast_ms': []})
        entry['shots'] += 1
        entry['max_abs'] = max(entry['max_abs'], row['max_abs'])
        entry['n_differ'] += row['n_differ']
        entry['n_values'] += row['n_values']
        entry['legacy_ms'].append(row['legacy_ms'])
        entry['fast_ms'].append(row['fast_ms'])

    out = []
    for entry in by_op.values():
        legacy = float(np.median(entry['legacy_ms']))
        fast = float(np.median(entry['fast_ms']))
        out.append({
            'op': entry['op'], 'gated': entry['gated'], 'shots': entry['shots'],
            'max_abs': entry['max_abs'], 'n_differ': entry['n_differ'],
            'frac_differ': entry['n_differ'] / max(1, entry['n_values']),
            'legacy_ms': round(legacy, 2), 'fast_ms': round(fast, 2),
            'speedup': round(legacy / fast, 2) if fast > 0 else float('inf'),
            'saved_ms': round(legacy - fast, 2)})
    return out


def print_table(summary: list) -> None:
    print(f'\n{"op":<16}{"max|Δ|":>8}{"differ":>12}{"fraction":>12}'
          f'{"legacy":>11}{"fast":>11}{"x":>7}{"saved":>10}')
    print('-' * 87)
    for row in summary:
        print(f'{row["op"]:<16}{row["max_abs"]:>8}{row["n_differ"]:>12,}'
              f'{row["frac_differ"]:>12.2e}{row["legacy_ms"]:>10.2f}m'
              f'{row["fast_ms"]:>10.2f}m{row["speedup"]:>7.2f}'
              f'{row["saved_ms"]:>9.2f}m')
    print('-' * 87)
    total_saved = sum(r['saved_ms'] for r in summary if r['gated'])
    print(f'  max|Δ| is the worst pixel of every shot, not an average.')
    print(f'  gated rewrites save {total_saved:.1f} ms per shot together.')
    print(f'  * vignette f32 is knowingly inexact: reported, not gated.')


def run_augment(args, wsi, level_mpp) -> int:
    cfg = augment_cfg(level_mpp)
    camera = Render(SlideReader(wsi), cfg=cfg, seed=args.seed)
    q = camera

    read_w = q.output_w + 2 * SENSOR_MARGIN
    read_h = q.output_h + 2 * SENSOR_MARGIN
    side = q.geometry.square_out
    print(f'{Path(args.wsi).name}  L{args.level}  level_mpp={level_mpp:.4f}')
    print(f'sensor {q.output_w}x{q.output_h}   read {read_w}x{read_h} = '
          f'{read_w * read_h / 1e6:.2f} Mpx  ({read_w * read_h * 3 / 1e6:.2f} MB/op)')
    print(f'  (the {side}^2 = {side * side / 1e6:.2f} Mpx bounding square is read '
          f'only when the exposure rotates; these shots do not)')

    shots = textured_shots(camera, wsi, args.shots, args.seed)
    if not shots:
        print(f'no crop reached std {MIN_STD} -- nothing worth comparing')
        return 1
    print(f'{len(shots)} textured shots\n')

    rows = []
    for index, ((x, y), raw, params) in enumerate(shots):
        for name, legacy_fn, fast_fn, extra, gated in cases(params):
            rows.append({'shot': index, 'x': x, 'y': y, 'op': name,
                         'gated': gated,
                         **compare(legacy_fn, fast_fn, raw, *extra)})
        print(f'  shot {index} at ({x},{y})  vignette='
              f'{params["vignette_strength"]:.3f}  '
              f'k1={params["distortion_k1"]:.4f}', flush=True)

    summary = summarise(rows)
    print_table(summary)

    timing = time_captures(wsi, cfg, [xy for xy, _, _ in shots], args.seed)
    print_capture(timing)
    summary.append({
        'op': 'capture (whole)', 'gated': False, 'shots': timing['n'],
        'max_abs': 0, 'n_differ': 0, 'frac_differ': 0.0,
        'legacy_ms': round(timing['legacy_ms'], 2),
        'fast_ms': round(timing['fast_ms'], 2),
        'speedup': round(timing['legacy_ms'] / timing['fast_ms'], 2),
        'saved_ms': round(timing['legacy_ms'] - timing['fast_ms'], 2)})

    # The point of the geometry rewrite is that a no-op returns the input
    # itself. Values alone cannot show that -- a copy has the same values --
    # so it is stated here as the fact it is.
    probe = shots[0][1]
    print(f'\nno-op returns the input object itself: '
          f'rotation {geometry._apply_rotation_fast(probe, 0.0) is probe}   '
          f'scale {geometry._apply_scale_fast(probe, 1.0) is probe}')

    out_dir = Path(args.out or job_result_dir('TestReadPath'))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'augment_equivalence_{Path(args.wsi).stem}.csv'
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f'  {path}')

    failures = [f'{r["op"]}: {r["n_differ"]:,} values differ, max {r["max_abs"]}'
                f' -- this rewrite claimed exact equality'
                for r in summary if r['gated'] and r['n_differ'] != 0]
    if failures:
        print(f'\n{len(failures)} FAILURE(S):')
        for message in failures:
            print(f'  {message}')
        return 1
    print('\nall gated rewrites exact')
    return 0



# ══════════════════════════════════════════════════════════════════════════════

SECTIONS = ('map', 'seed', 'augment')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--wsi', required=True)
    ap.add_argument('--level', type=int, default=1,
                    help='pyramid level for map / seed / augment')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--only', nargs='+', choices=SECTIONS, default=None,
                    help='sections to run (default: all)')
    ap.add_argument('--shots', type=int, default=5, help='augment: shots')
    ap.add_argument('--out', default=None,
                    help='augment: where the csv goes. Default: '
                         'result/<SLURM_JOB_NAME or TestReadPath>/')
    args = ap.parse_args()
    only = args.only or list(SECTIONS)

    wsi = SafeSlide(args.wsi)
    base_mpp = (float(wsi.properties.get(openslide.PROPERTY_NAME_MPP_X, 0.25))
                + float(wsi.properties.get(openslide.PROPERTY_NAME_MPP_Y, 0.25))) / 2
    level_mpp = base_mpp * wsi.level_downsamples[args.level]
    print(f'{Path(args.wsi).name}  L{args.level}  base_mpp={base_mpp:.4f}  '
          f'level_mpp={level_mpp:.4f}')

    runners = {'map': lambda: run_map(args, wsi, level_mpp),
               'seed': lambda: run_seed(args, wsi, level_mpp),
               'augment': lambda: run_augment(args, wsi, level_mpp)}
    status = {}
    for name in only:
        print(f'\n======== [{name}] ========', flush=True)
        status[name] = runners[name]()

    print('\n======== summary ========')
    for name, rc in status.items():
        print(f'  {"ok  " if rc == 0 else "FAIL"}  {name}' + ('' if rc == 0 else f'  (exit {rc})'))
    return max(status.values()) if status else 0


if __name__ == '__main__':
    sys.exit(main())
