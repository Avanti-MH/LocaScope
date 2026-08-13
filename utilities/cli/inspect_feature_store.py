#!/usr/bin/env python3
"""Say what a feature store contains, and whether its answers point anywhere.

    python utilities/cli/inspect_feature_store.py result/cache/features/*.safetensors
    python utilities/cli/inspect_feature_store.py result/cache/features --pairs

Reading the metadata is header-only, so listing a directory of 30 GB stores is
instant. `--pairs` is the part that costs anything: it loads the index tensors of
each query store and checks them against the reference store they name.

Why the pairing check exists
----------------------------
A query store carries `ans_main` / `ans_ovlp`: indices into a DIFFERENT file,
the reference store for the same slide and level. If those are wrong, every
question in the experiment is unanswerable, every pooling scores the same zero,
and the report reads "no pooling improves recall" -- a finding, not a bug.

The coordinate mapping that produces them has already been wrong once (the
inverse rotation's sign, caught by test_camera_output_to_level0), so this is a
demonstrated failure path rather than a hypothetical one.

Three things are checked, all from the stores alone -- no WSI, no model:

  in range      0 <= ans < len(reference)
  consistent    the stored delta equals (query x/y - reference x/y) / ds.
                delta and x/y are written by different lines of the dump, so
                agreement between them is evidence the pairing logic ran.
  distributed   nearest-of-both |delta| <= 128 px, and not all zero. 128 is the
                covering radius of the two grids together -- main at (256i,256j)
                and overlap at (256i+128,256j+128) form a checkerboard whose
                deep holes, like (128,0), sit 128 from both. The dump enforces
                this by construction, so a violation here means the two disagree
                about what is legal. All-zero means the FoVs landed exactly on
                grid points, so delta is not a variable at all and the
                experiment quietly lost a dimension.

Why a single grid is NOT held to 181 px
---------------------------------------
256/sqrt(2) = 181 is the covering radius of one grid, but only inside its
extent. PatchGrid lays whole tiles, so a region keeps a margin of up to 255 px
at its right and bottom with no main grid point in it, and the overlap grid is
inset another 128 -- a query there is legitimately 300+ px from that grid.

That is a different fact from a broken coordinate inverse, and the two are told
apart by WHERE the answer is: margin tiles answer to the outermost ring of the
grid, a sign error answers to interior points. So an over-181 delta is reported
and only fails when its answer is not on that ring.

Measured before this rule was written: 254 over-181 queries across 23
combinations, 254 of them on the outermost ring, main almost never and overlap
always more -- the asymmetry the 128 px inset predicts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import numpy as np                                          # noqa: E402
import FeatureStore as FS                                   # noqa: E402

#: Covering radius of ONE grid: a square lattice of spacing 256 has its farthest
#: point at the cell centre, 256/sqrt(2) away. This holds only INSIDE the grid's
#: extent, so exceeding it is a question ("is the answer on the outermost ring?")
#: rather than a verdict -- see the module docstring.
DELTA_ONE_GRID = 256 / 2 ** 0.5          # 181.0

#: Covering radius of the two grids TOGETHER. Main sits at (256i, 256j) and
#: overlap at (256i+128, 256j+128), so in units of 128 the union is every point
#: whose coordinates share a parity -- a checkerboard whose deep holes, like
#: (128, 0), are 128 from every occupied point. This is the bound that means
#: something: it is how far a query can be from the nearest position retrieval
#: actually scores.
DELTA_UNION = 128.0

#: Share of queries allowed to land in the uncovered margin. Region edges make a
#: few unavoidable -- 0.52% measured over 23 combinations, 1.7% in the worst one.
#: A broken coordinate inverse would not be a few.
MARGIN_FRAC_MAX = 5.0


def expand(paths) -> list:
    """Paths to inspect, complaining about what is missing rather than treating
    an absent directory as an unreadable file."""
    out, missing = [], []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            hits = sorted(p.glob('*.safetensors'))
            if not hits:
                missing.append(f'{p} is a directory with no .safetensors in it')
            out.extend(hits)
        elif p.exists():
            out.append(p)
        else:
            missing.append(f'{p} does not exist '
                           f'(note /tmp is node-local on this cluster -- a store '
                           f'written by a job on another node is not here)')
    for m in missing:
        print(f'[skip] {m}')
    return out


def show(path: Path) -> FS.StoreMeta:
    m = FS.load_meta(path)
    size = path.stat().st_size / 1e9
    grid = f'{m.token_grid[0]}x{m.token_grid[1]}' if m.token_grid else '-'
    is_query = 'query' in m.pooling
    print(f'{path.name}')
    print(f'  slide     {m.wsi_stem}  L{m.level}   ds={m.ds:g}  '
          f'mpp={m.mpp:.4f}  base_mpp={m.base_mpp:.4f}')
    print(f'  features  pooling={m.pooling}  slots={len(m.slots)}  '
          f'layout={m.slot_layout}  dim={m.dim}  '
          f'token_grid={grid}  num_prefix={m.num_prefix}')
    print(f'  made by   encoder={m.encoder_id}   mask={m.mask_id}   '
          f'tile={m.tile_size}   overlap={m.overlap}')
    print(f'  identity  cfg_hash={m.cfg_hash()}   {size:.2f} GB   {m.created_at}')
    if is_query:
        # "fraction of the grid covered" is a reference-store idea. A query store
        # is not drawn from the grid at all -- its n_available is a placeholder
        # that satisfies the sample/all validation, and printing it as a
        # percentage said "covers 100% of the grid", which is nonsense.
        print(f'  content   {m.n_tiles:,} query tiles   seed={m.sample_seed}')
    else:
        # "over", not "of": the figure can exceed 100% and that is information,
        # not an error. A displaced or inherited tile does not sit on this
        # level's grid, so a store can hold more tiles than the grid offers --
        # and seeing 103.9% is the quickest way to notice that a level ran out
        # of positions and had some synthesised. `origin` in the extras gives
        # the exact count.
        pct = (m.n_tiles / m.n_available * 100) if m.n_available else float('nan')
        note = ('   <- NOT a complete cache; require coverage="all" to refuse it'
                if m.coverage == 'sample' else '')
        print(f'  content   {m.n_tiles:,} over {m.n_available:,} grid positions '
              f'({pct:.1f}%)   coverage={m.coverage}   seed={m.sample_seed}{note}')
    return m


#: Round edges, because these are read by eye every time. linspace over the
#: covering radius gave 0-23-45-68, which needs arithmetic before it means
#: anything. Both run past their bound on purpose: for a single grid the margin
#: band legitimately reaches ~500, and clipping it would hide the shape that
#: shows it IS a margin band; for the union anything past 128 is a bug, and a
#: bug should be visible rather than clipped.
EDGES_ONE_GRID = list(range(0, 526, 25))      # 0..525, past the 181 bound
EDGES_UNION = list(range(0, 145, 16))         # 0..144, covers the 128 bound


def summarise(v: np.ndarray, label: str, ok: bool = True) -> None:
    print(f'    {label:22s} n={len(v):5d}  min={v.min():6.1f}  '
          f'p50={np.median(v):6.1f}  max={v.max():6.1f}  {"ok" if ok else "FAIL"}')


def histogram(v: np.ndarray, edges, label: str) -> None:
    counts, _ = np.histogram(v, bins=edges)
    wide = max(counts.max(), 1)
    print(f'    {label}')
    for lo, hi, c in zip(edges[:-1], edges[1:], counts):
        if c == 0:
            continue                       # empty bins are noise, not evidence
        bar = '#' * int(40 * c / wide)
        print(f'      {lo:4d}-{hi:<4d} {c:6d} {bar}')


def on_outer_ring(ans: np.ndarray, reg: np.ndarray, rc: np.ndarray,
                  kind: np.ndarray) -> np.ndarray:
    """For each answer index, is that grid position on the outermost row or
    column of its own (region, kind) grid?

    This is what separates "the query sat in the margin PatchGrid does not
    cover" from "the coordinate inverse is wrong": the first can only ever
    answer to the edge of the grid, the second lands anywhere.
    """
    out = np.zeros(len(ans), dtype=bool)
    for i, a in enumerate(ans):
        sel = (reg == reg[a]) & (kind == kind[a])
        rows, cols = rc[sel, 0], rc[sel, 1]
        out[i] = (rc[a, 0] in (rows.min(), rows.max())
                  or rc[a, 1] in (cols.min(), cols.max()))
    return out


def check_pair(qpath: Path, root: Path, show_hist: bool = False) -> bool:
    """Verify one query store against the reference store it belongs to."""
    qm = FS.load_meta(qpath)
    refs = FS.find(root, wsi_stem=qm.wsi_stem, level=qm.level, pooling='tokens')
    refs = [r for r in refs if FS.load_meta(r).cfg_hash() == qm.cfg_hash()]
    if not refs:
        print(f'  PAIR  no reference store for {qm.wsi_stem} L{qm.level} '
              f'cfg={qm.cfg_hash()}')
        return False
    rpath = refs[0]
    rm = FS.load_meta(rpath)

    q, _ = FS.load(qpath, keys=('x', 'y', 'ans_main', 'ans_ovlp',
                                'delta_main', 'delta_ovlp'))
    r, _ = FS.load(rpath, keys=('x', 'y', 'region', 'grid_rc', 'kind'))
    n_ref = len(r['x'])
    rreg = r['region'].numpy()
    rrc = r['grid_rc'].numpy()
    rkind = r['kind'].numpy()
    ok = True
    print(f'  PAIR  {rpath.name}   ref n={n_ref:,}')

    mags = {}
    for side in ('main', 'ovlp'):
        ans = q[f'ans_{side}'].numpy().astype(np.int64)
        dlt = q[f'delta_{side}'].numpy().astype(np.float64)

        bad = int(((ans < 0) | (ans >= n_ref)).sum())
        if bad:
            print(f'    {side}: {bad} of {len(ans)} indices out of range '
                  f'[0, {n_ref}) -- every one of those questions is unanswerable')
            ok = False
            continue

        # delta and x/y come from different lines of the dump; agreement is
        # evidence the pairing ran, not just that something was written.
        want = np.stack([q['x'].numpy()[np.arange(len(ans))] - r['x'].numpy()[ans],
                         q['y'].numpy()[np.arange(len(ans))] - r['y'].numpy()[ans]],
                        axis=1) / rm.ds
        drift = np.abs(want - dlt).max() if len(ans) else 0.0
        if drift > 1.5:                       # int16 rounding is under a pixel
            print(f'    {side}: stored delta disagrees with x/y by up to '
                  f'{drift:.1f} px -- the pairing and the coordinates were not '
                  f'computed from the same thing')
            ok = False

        mag = np.hypot(dlt[:, 0], dlt[:, 1])
        mags[side] = mag
        side_ok = True
        over = mag > DELTA_ONE_GRID + 1
        if over.any():
            # Margin tile or sign error? The answer's position decides. The ring
            # is read off the stored sample rather than the full grid, which can
            # only shrink it -- with thousands of positions per region every
            # extreme row and column is represented many times over.
            ring = on_outer_ring(ans[over], rreg, rrc, rkind)
            n_in = int((~ring).sum())
            if n_in:
                print(f'    {side}: {n_in} of {int(over.sum())} deltas past '
                      f'{DELTA_ONE_GRID:.0f} px answer to an INTERIOR grid point '
                      f'-- the nearest-neighbour search or the coordinate '
                      f'inverse is wrong (max {mag.max():.1f} px)')
                side_ok = ok = False
            else:
                print(f'    {side}: {int(over.sum())} of {len(mag)} past '
                      f'{DELTA_ONE_GRID:.0f} px, all answering to the outermost '
                      f'ring -- the uncovered margin at a region edge, expected')
        if mag.max() < 1e-6:
            print(f'    {side}: every delta is zero -- queries landed exactly on '
                  f'grid points, so delta is not a variable and the experiment '
                  f'lost that axis without saying so')
            side_ok = ok = False
        summarise(mag, f'{side} |delta| px', side_ok)
        # Quiet when healthy, expansive when not: the shape only matters once
        # something is off, and 25 stores at 8 bins each is 600 lines nobody
        # reads.
        if show_hist or not side_ok:
            histogram(mag, EDGES_ONE_GRID, f'{side} |delta| px')

    # The quantity that means something. Either grid alone can leave a query
    # 181 px from its nearest position, but retrieval scores both, so what it
    # actually has to cope with is the smaller of the two -- and that cannot
    # exceed 128 unless the pairing is wrong.
    if len(mags) == 2:
        nearest = np.minimum(mags['main'], mags['ovlp'])
        margin = nearest > DELTA_UNION
        near_ok = True
        # Both the dump and eval drop these, so the population worth describing
        # is the one that will be scored. What is left to check is the RATE: a
        # broken coordinate inverse would put most queries in the margin, not
        # half a percent of them. Measured 0.52% over 23 combinations, 1.7% in
        # the worst single one.
        if margin.any():
            frac = margin.mean() * 100
            print(f'    {int(margin.sum())} of {len(nearest)} ({frac:.1f}%) lie '
                  f'past {DELTA_UNION:.0f} px from BOTH grids -- the margin '
                  f'PatchGrid leaves at a region edge. dump and eval drop them.')
            if frac > MARGIN_FRAC_MAX:
                print(f'      that is over {MARGIN_FRAC_MAX:.0f}%, far more than '
                      f'edges can explain -- suspect the pairing, not the grid')
                near_ok = ok = False
        kept = nearest[~margin]
        if len(kept):
            summarise(kept, 'nearest of both px', near_ok)
        if show_hist or not near_ok:
            histogram(nearest, EDGES_UNION,
                      'nearest of both  <- how far retrieval has to reach')

    # delta varies per FoV, not per tile: the 20 tiles of one shot sit on the
    # same 256 lattice as the grid, so they share one offset. Distinct values
    # therefore count FoVs x rotations, and a run with few FoVs has a coarse
    # delta axis however many query tiles it holds.
    n_distinct = len(np.unique(np.round(np.stack(
        [mags['main'], mags['ovlp']], 1), 1), axis=0))
    print(f'    {n_distinct} distinct delta value(s) among {len(mags["main"])} '
          f'queries -- delta resolution comes from the FoV count, not the tile count')

    return ok


# ── which sampler wrote this? ─────────────────────────────────────────────────
#
# A store's file date says when it was written, and the source file's mtime says
# when someone last touched the code -- neither says which sampling rule ran.
# The store itself does, in three places written by different lines, which is
# what makes them worth reading together:
#
#   sampler_id   the SamplerConfig hash. Empty means the writer had no sampling
#                rule to record. Necessary, not sufficient: it also comes out
#                empty for a store written before the field existed.
#
#   origin       0 grid / 1 displaced / 2 inherited. Any 1 or 2 is decisive --
#                only the quota sampler can produce a tile that is not on the
#                grid. All-zero is NOT decisive the other way: a level whose
#                buckets all filled from the grid legitimately has no
#                displacements.
#
#   white_frac   the distribution is what says whether quotas BOUND. Uniform
#                sampling inside tissue regions measured p50 = 0.72 with 46% at
#                ~1.0 on BRACS_1228 L0; a quota run caps the `full` bucket, so a
#                store still showing ~46% pure background was not drawn under
#                one whatever its other fields say.
#
# Reads the header plus a few [N] vectors. The features tensor is never touched,
# so this stays seconds over 44 stores totalling tens of GB.

#: Bucket names, duplicated from ReferenceSampler rather than imported: this
#: file must stay readable against stores written by older code, and importing
#: the current names would silently relabel a store whose bucket ids meant
#: something else. Length mismatch is reported instead.
BUCKET_NAMES = ('lt15', 'mid', 'gt70', 'gt80', 'full')

ORIGIN_NAMES = ('grid', 'jitter', 'inherit')


def sampling_row(path: Path) -> dict:
    """One store's sampling evidence. Header plus [N] vectors only."""
    from safetensors import safe_open

    meta = FS.load_meta(path)
    with safe_open(str(path), framework='pt') as f:
        present = set(f.keys())
        wanted = {'white_frac', 'origin', 'bucket', 'valid_frac',
                  'parent_x', 'parent_y'}
        got = {k: f.get_tensor(k).numpy() for k in sorted(wanted & present)}

    row = dict(name=path.name, wsi=meta.wsi_stem, level=meta.level,
               is_query='query' in meta.pooling, n=meta.n_tiles,
               sampler_id=meta.sampler_id, created=meta.created_at,
               keys=sorted(present - {'features', 'x', 'y', 'region',
                                      'grid_rc'}))

    white = got.get('white_frac')
    if white is not None and len(white):
        white = white.astype(np.float64)
        row['white_p50'] = float(np.median(white))
        row['white_full_frac'] = float((white >= 0.99).mean())
    origin = got.get('origin')
    if origin is not None and len(origin):
        row['origin'] = {ORIGIN_NAMES[i] if i < len(ORIGIN_NAMES) else f'?{i}':
                         int((origin == i).sum())
                         for i in sorted(set(origin.tolist()))}
    bucket = got.get('bucket')
    if bucket is not None and len(bucket):
        row['bucket'] = {BUCKET_NAMES[i] if i < len(BUCKET_NAMES) else f'?{i}':
                         int((bucket == i).sum())
                         for i in sorted(set(bucket.tolist()))}
    return row


def report_sampling(paths) -> None:
    rows, unreadable = [], []
    for p in paths:
        try:
            rows.append(sampling_row(p))
        except Exception as e:                              # noqa: BLE001
            unreadable.append((p, e))

    print(f'{"store":52s} {"n":>6s}  {"sampler_id":12s} '
          f'{"white p50":>9s} {"pure bg":>8s}  origin')
    print('-' * 118)
    for r in sorted(rows, key=lambda r: (r['wsi'], r['level'], r['is_query'])):
        white_p50 = (f'{r["white_p50"]:9.3f}' if 'white_p50' in r
                     else '        -')
        pure = (f'{r["white_full_frac"] * 100:7.1f}%'
                if 'white_full_frac' in r else '       -')
        origin = ('  '.join(f'{k}={v}' for k, v in r['origin'].items())
                  if 'origin' in r else '-')
        print(f'{r["name"][:52]:52s} {r["n"]:6d}  '
              f'{(r["sampler_id"] or "(empty)"):12s} {white_p50} {pure}  '
              f'{origin}')
        # Which extras are present dates the store against the code more
        # precisely than any file time can: valid_frac appears only once hole
        # filtering existed, parent_x/parent_y only after the parent was stored
        # as a coordinate. A store missing them was written before those lines,
        # whatever the source file's mtime says today.
        print(f'{"":52s} {"":6s}  extras: {", ".join(r["keys"]) or "(none)"}')

    print()
    with_id = [r for r in rows if r['sampler_id']]
    displaced = [r for r in rows
                 if any(k != 'grid' for k in r.get('origin', {}))]
    print(f'{len(rows)} store(s):  {len(with_id)} carry a sampler_id, '
          f'{len(displaced)} contain a tile that is not on the grid')

    # The verdict is stated as what the evidence supports, not as a guess about
    # which code was current. All-grid with no sampler_id is consistent with
    # both a uniform draw and a quota run that never needed a displacement --
    # the white_frac column is what separates them, so it is named here rather
    # than left for the reader to remember.
    if not with_id and not displaced:
        print('  -> no store records a sampling rule and none holds an '
              'off-grid tile.')
        print('     Read the "pure bg" column: ~46% at L0 is the uniform '
              'draw; a quota run holds it near its lt15/mid/gt70 shares.')
    elif displaced:
        print('  -> at least one store was written by the quota sampler: '
              'a displaced or inherited tile cannot come from a uniform draw.')

    for p, e in unreadable:
        print(f'[unreadable] {p.name}: {type(e).__name__}: {e}')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='*', default=['result/cache/features'],
                    help='stores or directories of them '
                         '(default result/cache/features)')
    ap.add_argument('--pairs', action='store_true',
                    help='also verify each query store against its reference')
    ap.add_argument('--sampling', action='store_true',
                    help='one line per store: which sampling rule wrote it, '
                         'and whether the background quotas actually bound')
    ap.add_argument('--hist', action='store_true',
                    help='show delta histograms even when the checks pass; '
                         'failures print them regardless')
    args = ap.parse_args()

    paths = expand(args.paths)
    if not paths:
        sys.exit('no .safetensors found')

    if args.sampling:
        report_sampling(paths)
        return 0

    metas, failed = {}, []
    for p in paths:
        try:
            metas[p] = show(p)
        except Exception as e:                              # noqa: BLE001
            print(f'{p.name}\n  UNREADABLE: {type(e).__name__}: {e}')
            failed.append(p)
        print()

    if args.pairs:
        print('=' * 60)
        for p, m in metas.items():
            if 'query' not in m.pooling:
                continue
            print(f'{p.name}')
            if not check_pair(p, p.parent, show_hist=args.hist):
                failed.append(p)
            print()

    print(f'{len(metas)} store(s) read, {len(failed)} problem(s)')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
