#!/usr/bin/env python3
"""Read a bench_locascope metrics.csv and say how the pipeline actually performs.

No GPU, no WSI, no torch -- it only reads the csv, so it runs on a login node
and takes under a second.

Two things are measured and they must not be confused:

  MAIN PATH      What LocaScopePipeline actually outputs today: retrieval's
                 rank-1 window, refined by SIFT. Judged on refine_center_err_um.
                 This is the number that describes the product.

  TOP-K + VERIFY What a system that ran SIFT on the first --sift-topk
                 candidates and picked by inlier count WOULD output. It exists
                 only inside bench_locascope; LocaScopePipeline never calls
                 top_k. Reporting it as the pipeline's accuracy would be wrong,
                 which is why section E labels it as headroom.

Why the centre and not the corner
---------------------------------
Every distance here is centre-to-centre. A corner is not comparable across
rotations -- the recorded ground-truth corner and the predicted corner are
different corners of the same footprint, which once inflated one shot's error
to 901 um when the true centre error was 0.8 um.

Why distances are also given in tiles
-------------------------------------
um and L0 px are absolute, so the same number means different things at
different levels: 500 um is most of a tile at L0 and a rounding error at L6.
Every distance is therefore also printed in tiles, where

    one tile = --tile-size px at the level the query really came from
             = tile_size * nominal_mpp um

and so, since err_um = err_px * base_mpp,

    err_in_tiles = err_um / (tile_size * nominal_mpp)

base_mpp cancels, and nominal_mpp comes from the ground truth rather than from
the level the pipeline routed to. The tile unit is therefore invariant to a
routing mistake. The routed-level tile (tile_size * retrieval.ds, recorded as
retr_tile_l0) is deliberately NOT used: it grows with the routing error, so a
shot mis-routed to L6 would show a huge px error as a fraction of a tile and
look accurate.

Usage:
    python utilities/cli/analyze_locascope_metrics.py \\
        result/BenchLocaScope/metrics.csv

    # a stricter idea of "correct", and deeper verification
    python utilities/cli/analyze_locascope_metrics.py \\
        result/BenchLocaScope/metrics.csv --tol-um 10

    # one slide only
    python utilities/cli/analyze_locascope_metrics.py \\
        result/BenchLocaScope/metrics.csv --wsi BRACS_1936
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import sys

#: _paths imports os and sys and nothing else, so it costs this file none of
#: what it is careful about -- no torch, no openslide, still under a second on a
#: login node. Deriving OUTPUT_ROOT a second time here would have been the
#: version that eventually disagrees with the first.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'test_modules'))
import _paths                                                       # noqa: E402

BAR = '=' * 74


# ── csv access ────────────────────────────────────────────────────────────────

def cell(row: dict, key: str):
    """The raw string, or None for a blank. csv has no NULL, so '' means None."""
    v = row.get(key, '')
    return None if v in ('', 'None', None) else v


def num(row: dict, key: str):
    v = cell(row, key)
    if v is None:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def rank(row: dict, key: str):
    v = cell(row, key)
    try:
        return int(v) if v is not None else None
    except ValueError:
        return None


def is_correct(row: dict, tol: float) -> bool:
    """Final centre error within tol. A missing error counts as wrong.

    Written out rather than as `(num(...) or BIG) <= tol`, which is the bug this
    function exists to prevent: an error of exactly 0.0 um is falsy in Python,
    so the `or` fires and a pixel-perfect localisation is scored as a failure.
    Ten shots in the first run of this script were miscounted that way.
    """
    e = num(row, 'refine_center_err_um')
    return e is not None and e <= tol


def tile_um(row: dict, tile_size: int):
    """One tile's physical width, in um, at the level the query came from.

    Uses nominal_mpp (ground truth) and not the routed level, so the unit does
    not stretch when routing is wrong. See the module docstring.
    """
    m = num(row, 'nominal_mpp')
    return tile_size * m if m else None


def in_tiles(row: dict, err_um, tile_size: int):
    """An um distance expressed in tiles of the query's own level."""
    t = tile_um(row, tile_size)
    return None if (err_um is None or not t) else err_um / t


def pctl(values: list, p: int) -> float:
    """Nearest-rank percentile. Small n here, so no interpolation is warranted."""
    v = sorted(x for x in values if x is not None)
    if not v:
        return float('nan')
    return v[min(len(v) - 1, int(len(v) * p / 100))]


# ── sections ──────────────────────────────────────────────────────────────────

def section_a(rows: list, tol: float, ts: int) -> None:
    print(BAR)
    print('A. WHAT THE PIPELINE ACTUALLY OUTPUTS')
    print('   main path = retrieval rank 1 -> SIFT refine. No top-K verification:')
    print('   LocaScopePipeline never calls top_k, that lives only in the bench.')
    print(BAR)
    err = [num(r, 'refine_center_err_um') for r in rows]
    err = [e for e in err if e is not None]
    if not err:
        print('  no refine_center_err_um in this csv')
        return
    px = [e for e in (num(r, 'refine_center_err_px') for r in rows) if e is not None]
    tl = [e for e in (in_tiles(r, num(r, 'refine_center_err_um'), ts) for r in rows)
          if e is not None]
    print(f'  final centre error        n={len(err)}     '
          f'1 tile = {ts} px at the level the query came from')
    print(f'     {"":5s}{"um":>13s}{"L0 px":>13s}{"tiles":>11s}')
    for p in (10, 25, 50, 60, 70, 75, 80, 90):
        t_str = f'{pctl(tl, p):11,.2f}' if tl else f'{"-":>11s}'
        p_str = f'{pctl(px, p):13,.1f}' if px else f'{"-":>13s}'
        print(f'     p{p:<4d}{pctl(err, p):13,.1f}{p_str}{t_str}')
    print()
    for t in (10, 50, 100, 500, 1000):
        n = sum(1 for e in err if e <= t)
        mark = '  <-- --tol-um' if t == tol else ''
        print(f'  error <= {t:6d} um   : {n:5d}/{len(err)}  '
              f'({n / len(err) * 100:5.1f}%){mark}')
    if tl:
        print()
        for t in (0.25, 0.5, 1.0, 2.0, 4.0):
            n = sum(1 for e in tl if e <= t)
            print(f'  error <= {t:6.2f} tile : {n:5d}/{len(tl)}  '
                  f'({n / len(tl) * 100:5.1f}%)')
    lo, hi = pctl(err, 70), pctl(err, 75)
    if hi > 20 * max(lo, 1e-9):
        print(f'\n  BIMODAL: p70={lo:,.1f} um jumps to p75={hi:,.1f} um. Almost nothing')
        print('  lands in between, so the pass rate barely moves with the threshold')
        print('  -- the number below is robust rather than threshold-chosen.')


def section_b(rows: list, tol: float, ts: int) -> None:
    print('\n' + BAR)
    print(f'B. PER SLIDE AND LEVEL   (correct = final centre error <= {tol:g} um)')
    print(BAR)
    g = collections.defaultdict(list)
    for r in rows:
        e = num(r, 'refine_center_err_um')
        if e is not None:
            g[(os.path.basename(r['wsi_path'])[:24], r['level'])].append(
                (e, in_tiles(r, e, ts)))
    print(f'  {"slide":26s}{"lvl":>4s}{"n":>6s}{"correct":>9s}'
          f'{"p50 um":>11s}{"p50 tile":>11s}')
    for k in sorted(g):
        v = [e for e, _ in g[k]]
        t = [x for _, x in g[k] if x is not None]
        n = sum(1 for e in v if e <= tol)
        t_str = f'{pctl(t, 50):>11.2f}' if t else f'{"-":>11s}'
        print(f'  {k[0]:26s}{"L" + k[1]:>4s}{len(v):>6d}'
              f'{n / len(v) * 100:>8.0f}%{pctl(v, 50):>11.1f}{t_str}')


def section_c(rows: list, tol: float) -> None:
    print('\n' + BAR)
    print('C. WHERE THE FAILURES STOP')
    print(BAR)
    ok = lambda r: is_correct(r, tol)   # noqa: E731
    bad = [r for r in rows if not ok(r)]
    if not bad:
        print('  nothing failed')
        return
    c = collections.Counter()
    for r in bad:
        if cell(r, 'routed_level') != r['level']:
            c['1. routed to the wrong level'] += 1
        elif cell(r, 'retr_hit_rank') != '1':
            c['2. routed right, retrieval rank 1 wrong'] += 1
        else:
            c['3. routed right, rank 1 right, SIFT did not finish it'] += 1
    for k in sorted(c):
        print(f'  {c[k]:5d}  ({c[k] / len(bad) * 100:4.1f}%)  {k}')
    print(f'  {len(bad):5d}          total failures of {len(rows)}')

    sub = [r for r in bad if cell(r, 'routed_level') == r['level']
           and cell(r, 'retr_hit_rank') != '1']
    if sub:
        print('\n  category 2, split by where the truth actually ranked:')
        d = collections.Counter(
            'not in the top-K at all' if not cell(r, 'retr_hit_rank')
            else f'rank {cell(r, "retr_hit_rank")}' for r in sub)
        for k, v in d.most_common(8):
            print(f'    {v:5d}  {k}')
        n_in = sum(v for k, v in d.items() if k != 'not in the top-K at all')
        print(f'    -> {n_in} of these are reachable by verifying more candidates')

    print('\n  routing, by the level the shot really came from:')
    print(f'    {"true":>6s}{"routed":>8s}{"n":>7s}{"correct":>10s}')
    for lv in sorted({r['level'] for r in rows}):
        same = [r for r in rows if r['level'] == lv]
        for rl, n in sorted(collections.Counter(
                cell(r, 'routed_level') for r in same).items(),
                key=lambda x: (x[0] is None, x[0])):
            s = [r for r in same if cell(r, 'routed_level') == rl]
            note = ''
            if rl is not None and rl != lv:
                note = '  too fine' if int(rl) < int(lv) else '  too coarse'
            print(f'    {"L" + lv:>6s}{"L" + str(rl):>8s}{n:>7d}'
                  f'{sum(1 for r in s if ok(r)) / len(s) * 100:>9.0f}%{note}')


def section_d(rows: list, tol: float) -> None:
    print('\n' + BAR)
    print('D. DOES THE PIPELINE KNOW WHEN IT IS WRONG')
    print('   refine_success is the only self-check available at inference time:')
    print('   it needs no ground truth, so it is the flag a deployment can act on.')
    print(BAR)
    ok = lambda r: is_correct(r, tol)   # noqa: E731
    claimed = [r for r in rows if r.get('refine_success') == 'True']
    gave_up = [r for r in rows if r.get('refine_success') == 'False']
    for lab, s in (('refine_success=True ', claimed), ('refine_success=False', gave_up)):
        if not s:
            continue
        n = sum(1 for r in s if ok(r))
        print(f'  {lab}  n={len(s):5d}   actually correct {n:5d}  ({n / len(s) * 100:5.1f}%)')
    if claimed:
        wrong = sum(1 for r in claimed if not ok(r))
        print(f'\n  coverage  {len(claimed)}/{len(rows)} = {len(claimed)/len(rows)*100:.1f}%'
              '   (how often it answers at all)')
        print(f'  precision {len(claimed)-wrong}/{len(claimed)} = '
              f'{(len(claimed)-wrong)/len(claimed)*100:.1f}%'
              '   (of the answers it gives, how many are right)')
        print(f'  confidently wrong {wrong} = {wrong/len(rows)*100:.1f}% of all shots')

    # inlier ratio: free, needs no ground truth, and recorded for every row
    rr = [(num(r, 'refine_inliers') / num(r, 'refine_matches'), ok(r))
          for r in claimed
          if num(r, 'refine_matches') and num(r, 'refine_matches') > 0]
    if rr:
        yes = [x for x, o in rr if o]
        no = [x for x, o in rr if not o]
        print('\n  refine_inliers / refine_matches, among the answers it gave:')
        if yes:
            print(f'    correct  n={len(yes):5d}  min {min(yes):.3f}  p50 {pctl(yes,50):.3f}')
        if no:
            print(f'    wrong    n={len(no):5d}  min {min(no):.3f}  max {max(no):.3f}')
        if yes and no and max(no) < min(yes):
            print(f'    -> separated on this data: wrong tops out at {max(no):.3f},')
            print(f'       correct bottoms out at {min(yes):.3f}. Treat as a hypothesis,')
            print('       not a calibrated threshold, until the wrong ones come from')
            print('       more than one root cause.')


def section_e(rows: list, tol: float) -> None:
    """Top-K + SIFT verification: what it would add, bounded honestly.

    sift_hit_rank records the FIRST candidate SIFT localised correctly, not
    every one. So a picker that chose a LATER rank cannot be scored: the later
    candidate may be the same place seen by an overlapping window -- top_k
    suppresses duplicates only within one rotation, so the same spot legitimately
    appears several times. Those rows are reported as undecidable rather than
    guessed at, which is why the accuracy comes out as a range.
    """
    print('\n' + BAR)
    print('E. HEADROOM: TOP-K + SIFT VERIFICATION')
    print('   NOT what the pipeline does today. Implemented in bench_locascope')
    print('   only. Read as "what wiring this in would buy", not as accuracy.')
    print(BAR)
    scored = [r for r in rows if cell(r, 'sift_topk_n')]
    if not scored:
        print('  this csv has no sift_topk_n -- the bench ran without --sift-topk')
        return
    n = len(rows)
    depth = collections.Counter(cell(r, 'sift_topk_n') for r in scored)
    print(f'  candidates SIFT verified per shot: '
          f'{", ".join(f"{k} on {v} shots" for k, v in depth.most_common(4))}')

    def judge(pick, r):
        """right / wrong / unknown / abstain, deducing only what is deducible."""
        truth = rank(r, 'sift_hit_rank')
        if pick is None:
            return 'abstain'
        if truth is None:
            return 'wrong'          # none of the verified candidates was correct
        if pick == truth:
            return 'right'
        if pick < truth:
            return 'wrong'          # the first correct one is further down
        return 'unknown'            # may be the same place at another rank

    def first_to_clear(r):
        return rank(r, 'sift_verified_rank')

    def most_inliers(r):
        inl = num(r, 'sift_best_inliers')
        # sift_best_rank does not itself check min_inliers, so a shot where
        # nothing cleared would otherwise be handed an answer it should refuse.
        floor = num(r, 'refine_inliers') and 10
        return rank(r, 'sift_best_rank') if inl and inl >= (floor or 10) else None

    print(f'\n  {"picker":26s}{"right":>8s}{"wrong":>8s}{"undecid":>9s}'
          f'{"abstain":>9s}   accuracy')
    results = {}
    for lab, f in (('first to clear min_inliers', first_to_clear),
                   ('most inliers in the list', most_inliers)):
        c = collections.Counter(judge(f(r), r) for r in rows)
        results[lab] = c
        lo = c['right'] / n * 100
        hi = (c['right'] + c['unknown']) / n * 100
        span = f'{lo:.1f}%' if c['unknown'] == 0 else f'{lo:.1f}% .. {hi:.1f}%'
        print(f'  {lab:26s}{c["right"]:>8d}{c["wrong"]:>8d}'
              f'{c["unknown"]:>9d}{c["abstain"]:>9d}   {span}')

    ok = lambda r: is_correct(r, tol)   # noqa: E731
    main = sum(1 for r in rows if ok(r))
    best = max(c['right'] for c in results.values())
    print(f'\n  main path today          : {main}/{n} = {main/n*100:.1f}%')
    print(f'  best verified picker     : {best}/{n} = {best/n*100:.1f}%'
          f'   ({(best-main)/n*100:+.1f} points)')

    rescue = [r for r in rows if not ok(r) and cell(r, 'sift_hit_rank')]
    print(f'\n  shots the main path got wrong but a verifier could rescue: {len(rescue)}')
    deeper = [r for r in rows
              if not ok(r) and not cell(r, 'sift_hit_rank')
              and rank(r, 'retr_hit_rank') and cell(r, 'sift_topk_n')
              and rank(r, 'retr_hit_rank') > int(cell(r, 'sift_topk_n'))]
    print(f'  wrong, truth in the top-K but PAST --sift-topk (never examined): '
          f'{len(deeper)}')
    floor = [r for r in rows if not ok(r) and not cell(r, 'retr_hit_rank')]
    print(f'  wrong, truth not in the top-K at all -- the hard floor          : '
          f'{len(floor)}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('metrics_csv', nargs='?',
                    default=os.path.join(_paths.RESULT_DIR, 'BenchLocaScope', 'metrics.csv'))
    ap.add_argument('--tol-um', type=float, default=100.0,
                    help='final centre error at or under which a shot counts as '
                         'correct (default 100). The distribution is bimodal, so '
                         'anything from 50 to 1000 gives nearly the same answer.')
    ap.add_argument('--tile-size', type=int, default=256,
                    help='px per tile, to express distances in tiles as well as '
                         'in um (default 256, matching LocaScopePipeline). The '
                         'tile is taken at the level the query came from, not at '
                         'the routed level -- see the module docstring.')
    ap.add_argument('--wsi', default=None, metavar='SUBSTR',
                    help='keep only rows whose wsi_path contains SUBSTR')
    ap.add_argument('--level', type=int, nargs='+', default=None)
    ap.add_argument('--sections', default='abcde',
                    help='which sections to print, e.g. ae')
    args = ap.parse_args()

    with open(args.metrics_csv, newline='') as f:
        allrows = list(csv.DictReader(f))
    rows = [r for r in allrows if not r.get('error')]
    print(f'{args.metrics_csv}')
    print(f'{len(allrows)} rows, {len(rows)} usable, '
          f'{len(allrows) - len(rows)} with an error')

    if args.wsi:
        rows = [r for r in rows if args.wsi in r['wsi_path']]
    if args.level is not None:
        keep = {str(x) for x in args.level}
        rows = [r for r in rows if r['level'] in keep]
    if args.wsi or args.level is not None:
        print(f'filtered to {len(rows)} rows')
    if not rows:
        sys.exit('no rows left after filtering')

    s = args.sections.lower()
    if 'a' in s: section_a(rows, args.tol_um, args.tile_size)
    if 'b' in s: section_b(rows, args.tol_um, args.tile_size)
    if 'c' in s: section_c(rows, args.tol_um)
    if 'd' in s: section_d(rows, args.tol_um)
    if 'e' in s: section_e(rows, args.tol_um)


if __name__ == '__main__':
    main()
