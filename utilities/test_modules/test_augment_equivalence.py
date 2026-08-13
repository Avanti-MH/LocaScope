#!/usr/bin/env python3
"""Do the optimised augment bodies produce the same pixels as the old ones?

    python utilities/test_modules/test_augment_equivalence.py \
        --wsi /work/u26130998/datasets/.../BRACS_1228.svs [--level 1] [--shots 5]

Why this exists
---------------
`query_sim/augment/` was rewritten for speed, not for behaviour. A shot spends
about 0.9 s in the read plus `_apply_params`, against 0.03 s in
the encoder, so the augment chain is where a displacement bench's hours go.
The rewrites are:

    geometry   rotation at 0 and scale at 1 stop calling `.copy()`
    field      the vignette's distance / exp term cached per frame size, and
               its `np.clip` dropped where the gain provably cannot push a
               uint8 out of range
    lens       the distortion's mgrid / normalisation / radius square cached

Each is an argument that two expressions are equal, and an argument is not
evidence. This runs both bodies on photographs the Camera actually produced
and prints how far apart the two answers land.

One op at a time settles CORRECTNESS
------------------------------------
Every one of these is a deterministic function of (image, params). If each
agrees pixel for pixel on the same input, the chain that composes them agrees
too, so there is no separate end-to-end case to run for equality -- and running
one would mean driving `simulate_with_gt` twice with both random generators
pinned, machinery that could fail on its own and be blamed on the rewrite.

The whole capture settles COST
------------------------------
Per-op milliseconds do not say whether a bench gets shorter. `capture_with_gt`
is read + augment + centre crop, and only the middle term changed, so the
report times the whole thing both ways and prints the read separately. 158 ms
saved is a different fact depending on whether a shot costs 300 ms or 900 ms,
and the second number is the one that multiplies by 32,670 displacements.

Both passes build a Camera from the same seed so they draw the same domain-gap
parameters -- those are redrawn on every capture by design, and an unpinned
comparison would be measuring which pass got luckier.

The images come from `Camera.qfw.crop_padded`, the exact array
`capture_with_gt` hands to the augment chain for a shot that does not rotate,
and the parameters come from `Camera.capture_with_gt` itself. Both sides of the
comparison are what production sees, not values invented here -- which matters
for the timings, since the frame is now the sensor plus a margin rather than
the 2.12x bounding square.

What "the same" means
---------------------
Not a tolerance. Each op is compared value for value and the report gives the
largest absolute difference and how many of the h*w*3 values moved at all.
Four of the five should show exactly zero and the gate fails if they do not --
a tolerance would let a real regression through as "close enough".

The fifth is knowingly inexact and is measured, not gated. `VIGNETTE_FLOAT32`
computes the gain in float32, halving the two largest allocations in the whole
chain. `.astype(np.uint8)` truncates rather than rounds, so a product float64
puts at 200.0000001 and float32 puts at 199.9999999 becomes 200 against 199 --
a difference of 1, on the order of 1e-4 of pixels. Whether that is worth the
memory is a decision to take from the number, not from either assumption.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _d in ('query_sim', 'utilities', 'utilities/test_modules'):
    _p = str(_ROOT / _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                          # noqa: E402
import openslide                                            # noqa: E402

from camera import Camera, SENSOR_MARGIN                    # noqa: E402
from SafeSlide import SafeSlide                             # noqa: E402
from config import DomainGapConfig                          # noqa: E402
from augment import field, geometry, lens                   # noqa: E402
from _paths import job_result_dir                           # noqa: E402

#: Timed repeats per op per shot. The report takes the median, so one
#: scheduler hiccup does not become the measurement.
REPEATS = 3

#: Below this the crop is glass or an empty margin. On a constant image every
#: rewrite agrees trivially -- `np.where` against a gather, a cached grid
#: against a rebuilt one -- so the comparison would pass while meaning nothing.
MIN_STD = 12.0


def full_gap_cfg(query_mpp: float) -> DomainGapConfig:
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

    `crop_padded` returns the array the augment chain receives at rotation 0 --
    the FoV rect plus SENSOR_MARGIN, which is what Camera reads once it knows
    the exposure does not turn -- and `capture_with_gt` returns the parameter
    set the chain would have been given. Both come straight off the Camera, so
    nothing in this file invents an input.
    """
    rng = np.random.default_rng(seed)
    q = camera.qfw
    width, height = wsi.dimensions
    span_x = max(1, width - q.bounding_square_side_l0 - 1)
    span_y = max(1, height - q.bounding_square_side_l0 - 1)

    out = []
    for _ in range(400):
        if len(out) >= n_shots:
            break
        x = int(rng.integers(0, span_x))
        y = int(rng.integers(0, span_y))
        raw = q.crop_padded(x, y, margin=SENSOR_MARGIN)
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
    binds the dispatcher rather than a body, so this changes what a Camera
    capture actually runs without importing anything from `pipeline`.
    """
    field.USE_FAST = lens.USE_FAST = geometry.USE_FAST = enabled


def time_captures(wsi, cfg, positions, seed: int) -> dict:
    """`Camera.capture_with_gt` timed both ways, on identical parameter draws.

    A capture is read + augment + centre crop, and only the middle term
    changes. Including the other two is the point: the per-op table says what
    the rewrite saves, this says what fraction of a real shot that is, which is
    the number that decides whether a bench gets shorter.

    Both passes build a fresh Camera from the SAME seed, so `_py_rng` hands
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
        camera = Camera(wsi, cfg=cfg, seed=seed)
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
            camera = Camera(wsi, cfg=cfg, seed=seed + index)
            start = time.perf_counter()
            camera.capture_with_gt(x, y, rotation=0)
            elapsed = (time.perf_counter() - start) * 1e3
            (fast if use_fast else legacy).append(elapsed)

    # The read alone, so the report can say how much of a capture is the part
    # no rewrite in this file touches.
    set_fast(True)
    reader = Camera(wsi, cfg=cfg, seed=seed).qfw
    reads = []
    for x, y in positions:
        start = time.perf_counter()
        reader.crop_padded(x, y, margin=SENSOR_MARGIN)
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Are the optimised augment bodies the same pixels?')
    parser.add_argument('--wsi', required=True)
    parser.add_argument('--level', type=int, default=1)
    parser.add_argument('--shots', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    wsi = SafeSlide(args.wsi)
    base_mpp = (float(wsi.properties.get(openslide.PROPERTY_NAME_MPP_X, 0.25))
                + float(wsi.properties.get(openslide.PROPERTY_NAME_MPP_Y, 0.25))) / 2
    level_mpp = base_mpp * wsi.level_downsamples[args.level]
    cfg = full_gap_cfg(level_mpp)
    camera = Camera(wsi, cfg=cfg, seed=args.seed)
    q = camera.qfw

    read_w = q.output_w + 2 * SENSOR_MARGIN
    read_h = q.output_h + 2 * SENSOR_MARGIN
    side = q.bounding_square_side_out
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

    out_dir = Path(args.out or job_result_dir('AugmentEquivalence'))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'augment_equivalence.csv'
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


if __name__ == '__main__':
    sys.exit(main())
