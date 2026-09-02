#!/usr/bin/env python3
"""spec.md 12 step 3c: cut the pre-tiles the training set is made of.

    python training/SuperPathPoint/cli/extract_pretiles.py \
        --tile 256 --n 500

Outputs (in result/cache/tiles/ by default):
    <wsi_stem>__ds<d>__t<tile>__<cfg8>/000000.png ... index.csv, meta.json
    extract_pretiles.csv        in result/<SLURM_JOB_NAME or ExtractPreTiles>/

argparse, a loop, and printed progress. Everything that decides anything is in
`utilities/PreTileStore.py` (the format, the identity, the centre-crop
geometry),
`common/DsLadder.py` (which level to read) and `utilities/TileSampler.py` (which
positions exist). CLAUDE.md: a library layer that prints cannot be called by a
bench.

WHAT COMES OUT IS A PRE-TILE, NOT A TILE
-----------------------------------------
Each PNG is `tile * factor` on a side, centred on a position the sampler's
richness buckets admitted. The tile itself is never written: it is
`PreTileStore.centre_crop(pre, tile)` and lives only in the training loop.

The reason is spec.md 6.6 -- a production homography needs 1.78x the source it
is given, so a warp of a bare tile is a third pure black, and pure black is a
straight maximum-contrast edge with two right angles, which is exactly what a
corner detector fires on. A photograph cannot avoid that. A WSI can: the tissue
continues past the tile.

WHY THE SAMPLER GATES ON THE TILE AND THE READ COVERS THE PRE-TILE
-------------------------------------------------------------------
Two different squares, on purpose:

    the richness  is scored over tile * ds           (what we train on)
    the read      covers tile * factor * ds           (warp context only)

Gating on the pre-tile would multiply the rejection-sampling footprint by 3 and,
at tile 256, drop the reachable ladder from ds 32 to ds 11 -- losing the two
coarsest rungs, which is where Stage C's relative-survival labels carry the most
information. The pre-tile has to be READABLE, not tissue.

THE COST USED TO BE RECORDED; IT IS NOW REFUSED. Near the edge of the scanned
rectangle a pre-tile can run off the slide, and `clip_px` on each record used to
say by how much -- those positions kept the tile at the centre (`PreTileStore`
never slides the window inward, because that would move the tile off centre and
every crop downstream would be of the wrong place) and carried some background
at one edge.

`TileSampler` is now given the pre-tile as `reserve_l0`, so the lattice never
OFFERS such a position: `filter_patchable` gets the reserve rather than the
tile, and the first legal corner sits a margin inside the region. `clip_px` is
therefore 0 on every record, and the loop below asserts it rather than writing
it. A non-zero clip now means the reserve stopped binding, and every tile after
it is suspect -- which is a thing to stop on, not a column to fill in.

The 2026-08-26 corpus was cut before that and does carry clips: 295 of 500 at
ds 32 on one slide. Those are not wrong, they are the older contract.

RESUMABLE ON PURPOSE
---------------------
`index.csv` is written last, and its presence is what marks a directory
complete. A job killed at walltime therefore leaves directories that `find()`
skips and that a re-run rebuilds, rather than a short index that reads as a
small dataset.
"""

from __future__ import annotations

import argparse
import dataclasses
import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, '..', '..', '..', 'utilities'),
           os.path.join(_HERE, '..')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _paths import RESULT_DIR, job_result_dir, setup_import_paths  # noqa: E402

setup_import_paths()

import cv2                                                        # noqa: E402

import MaskStore                                                  # noqa: E402
from SafeSlide import SafeSlide                                    # noqa: E402
from TileSampler import (InheritConfig, OverlapConfig,             # noqa: E402
                         RichnessConfig, SamplerConfig, TileSampler,
                         caps_for_tissue_ratio)
from TissuesRegionsMask import TissuesRegionsMask                  # noqa: E402

import PreTileStore                                    # noqa: E402
from DsLadder import DEFAULT_RUNGS, DsLadder                # noqa: E402
from PreTileStore import (PRE_TILE_FACTOR, PreTileMeta,     # noqa: E402
                                 PreTileRecord, pre_tile_px)

DEFAULT_MASK_ROOT = os.path.join(RESULT_DIR, 'cache', 'masks')
DEFAULT_TILE_ROOT = os.path.join(RESULT_DIR, 'cache', 'tiles')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mask-root', default=DEFAULT_MASK_ROOT,
                    help='where build_mask_store.py wrote the masks')
    ap.add_argument('--root', default=DEFAULT_TILE_ROOT,
                    help='where the pre-tiles go (default: result/cache/tiles/)')
    ap.add_argument('--wsi', nargs='*', default=None,
                    help='slide paths. Default: every mask in the store')
    ap.add_argument('--tile', type=int, default=256,
                    help='what the model sees. v1 is 256; 512 and 1024 are '
                         'separate models and separate extractions (spec.md 6.5)')
    ap.add_argument('--pre-tile-factor', type=int, default=PRE_TILE_FACTOR,
                    help='pre-tile side / tile side. 3 is derived (spec.md 6.6, '
                         'bound 2.49) and is NOT a knob to tune for disk: it is '
                         'an identity field, so a different value is a different '
                         'dataset, not a cheaper version of this one')
    ap.add_argument('--ds', type=float, nargs='+', default=list(DEFAULT_RUNGS),
                    help='the ladder rungs to extract')
    ap.add_argument('--n', type=int, default=500,
                    help='tiles per (slide, ds). The probe of step 3b says '
                         'which cells can actually supply this')
    # ── the three sampling axes (utilities/TileSampler.py) ──
    #
    # All four go into `sampler_id`, so a corpus cut at one setting is not the
    # corpus cut at another. The defaults are the disjoint lattice: step equal
    # to the tile, no overlap admitted at all.
    # ── inheritance: the chains Stage B reads (spec.md 3.2) ──
    #
    # A chain is one level-0 centre with a tile at EVERY rung. It was never
    # reachable from this CLI before 2026-09-01, which is why the corpus of
    # 2026-08-27 has `inherit_id = -1` on all 6,388 rows -- not a setting that
    # was wrong, an option that was not wired.
    ap.add_argument('--inherit-share', type=float, default=0.0,
                    help='fraction of each rung that comes from chains. The '
                         'number of CENTRES is share * n, capped by how many '
                         'the source rung admits -- the rest of the rung is '
                         'filled by its own sampling, so a high share does not '
                         'shrink the corpus. 0 = off, which is the default '
                         'because a corpus that nothing analyses by stack does '
                         'not need chains')
    ap.add_argument('--inherit-source-rung', type=float, default=None,
                    help='which rung the centres are chosen at. None = the '
                         'finest, which has the most candidates but does not '
                         'guarantee they fit anywhere coarser. A COARSE source '
                         'guarantees the fit at every finer rung, because the '
                         'footprint only shrinks going down -- what it cannot '
                         'guarantee is tissue there, and a centre whose fine '
                         'window is glass is refused by the zero caps and the '
                         'chain truncates. `n_inherit_refused` per rung is how '
                         'much that costs')
    ap.add_argument('--tissue-ratio', type=float, default=None,
                    help='admit only tiles with at least this much tissue, as '
                         'CAPS on the richness buckets -- every bucket wholly '
                         'at or below 1-ratio background gets cap 1 and the '
                         'rest get 0 (`caps_for_tissue_ratio`). The ratio must '
                         'land on a bucket edge or it is refused rather than '
                         'rounded. Default: the settled seven-bucket contract, '
                         'whose gate is at 85 per cent background.\n'
                         'IT ALSO DECIDES WHERE CHAIN CENTRES COME FROM: '
                         '`_choose_centres` draws uniformly from whatever the '
                         'caps admit, so under the default a centre may sit in '
                         'a window that is 84 per cent glass -- and its tile '
                         'at the finest rung then lands in a zero-capped '
                         'bucket and truncates the chain. Measured on '
                         'BRACS_1598 (24 per cent tissue): 20 centres asked, '
                         '13 complete chains')
    ap.add_argument('--bucket-frame', default='per_rung',
                    choices=('per_rung', 'at_inherit'),
                    help="where a chain's richness bucket is decided. "
                         "'per_rung' recomputes it at each rung, so each "
                         "rung's distribution is exactly what the contract "
                         'asks and a chain has NO single bucket. '
                         "'at_inherit' fixes it at the source rung and carries "
                         'it, so a chain has one bucket -- which is what a '
                         'survival analysis stratified by bucket needs, and '
                         'what it costs is the per-rung distribution')
    ap.add_argument('--candidates', default='lattice',
                    choices=('lattice', 'random'),
                    help="'random' is the sampler this replaced, kept as the "
                         'control arm. It produced 202,420 overlapping pairs '
                         'over the 2026-08-26 corpus and 69.2 per cent of '
                         'tiles touching another')
    ap.add_argument('--grid-step', type=int, default=0,
                    help='lattice step in OUTPUT px. 0 means the tile, i.e. '
                         'disjoint, and 0 is the only spelling of that -- '
                         'writing the tile size out is refused, because two '
                         'spellings of one lattice are two sampler_ids over '
                         'one corpus. Half the tile is a deliberate 50 per '
                         'cent lattice and needs --max-overlap raised to match')
    ap.add_argument('--max-overlap', type=float, default=0.0,
                    help='largest area fraction any two tiles of a rung may '
                         'share')
    ap.add_argument('--overlapping-share', type=float, default=0.0,
                    help='largest share of a rung that may overlap anything '
                         'at all. 0 forbids it outright')
    ap.add_argument('--max-tries', type=int, default=2500,
                    help='rejection budget per cell, 5x n')
    ap.add_argument('--seed', type=int, default=0,
                    help='identity, not convenience: two seeds are two datasets '
                         'and get two directories')
    ap.add_argument('--overwrite', action='store_true',
                    help='replace directories that already hold a finished '
                         'extraction with this identity')
    ap.add_argument('--out', default=None,
                    help='directory for the summary CSV')
    args = ap.parse_args()

    out_dir = args.out or job_result_dir('ExtractPreTiles')
    os.makedirs(out_dir, exist_ok=True)

    pre_px = pre_tile_px(args.tile, args.pre_tile_factor)
    print(f'tile {args.tile}   pre-tile {pre_px}   factor '
          f'{args.pre_tile_factor}', flush=True)
    _rich = RichnessConfig()
    print('  richness  ' + '  '.join(
        f'{nm}:{f:.0%}/{c:.0%}' for nm, f, c
        in zip(_rich.names, _rich.floors, _rich.caps)) + '   (floor/cap)',
        flush=True)
    print(f'masks {args.mask_root}\ntiles {args.root}', flush=True)

    paths = args.wsi
    # A PATH, NOT A STEM, and the difference used to surface four frames down
    # as openslide's "Unsupported or missing image file" -- which reads as a
    # corrupt slide, not as a wrong argument. The stem is what every OTHER
    # thing here is keyed by (the mask store, the tile store, --wsi-stem in
    # make_ha_labels), so reaching for it is the expected mistake.
    for candidate in paths or ():
        if not os.path.exists(candidate):
            hit = MaskStore.find(args.mask_root)
            known = sorted(MaskStore.load_meta(p).wsi_path for p in hit)
            match = [k for k in known
                     if MaskStore.wsi_stem_of(k) == candidate]
            ap.error(
                f'--wsi takes slide PATHS, not stems, and {candidate!r} is not '
                f'a file.' + (f' Did you mean {match[0]}?' if match else
                              f' Known: {", ".join(os.path.basename(k) for k in known[:4])}'
                              f'{" ..." if len(known) > 4 else ""}'))

    if not paths:
        found = MaskStore.find(args.mask_root)
        if not found:
            print(f'no masks under {args.mask_root}. Run '
                  f'utilities/cli/build_mask_store.py first.')
            return 1
        paths = [MaskStore.load_meta(p).wsi_path for p in found]
        print(f'{len(paths)} slides from the mask store', flush=True)

    rows, failures = [], []
    for index, wsi_path in enumerate(paths, 1):
        stem = MaskStore.wsi_stem_of(wsi_path)
        print(f'\n[{index}/{len(paths)}] {stem}', flush=True)
        try:
            mask_path = MaskStore.find_one(args.mask_root, wsi_stem=stem)
        except Exception as e:                                   # noqa: BLE001
            print(f'    no mask: {e}', flush=True)
            failures.append((stem, str(e)))
            continue

        slide_mask, mask_meta = MaskStore.load(mask_path)
        print(f'    mask {mask_meta.rows}x{mask_meta.cols} at ds '
              f'{mask_meta.mask_ds:.0f}, tissue {mask_meta.fraction:.1%}   '
              f'({mask_meta.segmenter_id})', flush=True)

        with SafeSlide(wsi_path) as wsi:
            trm = TissuesRegionsMask.from_mask(wsi, slide_mask.mask,
                                               slide_mask.origin, slide_mask.span)
            # ONE SAMPLER FOR THE WHOLE SLIDE. Inheritance fixes a set of
            # centres BEFORE any rung is filled and validates it at each; a
            # sampler per rung chooses its own centres and no two rungs share
            # one, which is why the 2026-08-27 corpus has `inherit_id = -1` on
            # all 6,388 rows. Per-rung resume still works -- the sampling is
            # redone, the writes are skipped -- see `_extract_slide`.
            rows += _extract_slide(wsi, trm, slide_mask, mask_meta, args,
                                   stem, failures)

    summary = os.path.join(out_dir, 'extract_pretiles.csv')
    if rows:
        with open(summary, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        total = sum(r['n_got'] for r in rows)
        gb = sum(r['bytes'] for r in rows) / 1e9
        print(f'\nSaved {summary}   ({len(rows)} cells, {total} pre-tiles, '
              f'{gb:.1f} GB on disk)')
        # PNG against raw. Measured 2026-08-27 on the v1 corpus: 45.1 per
        # cent, 14.2 GB on disk for 17,784 pre-tiles of 768 px against 31.5 GB
        # uncompressed. Printed on every run rather than asserted -- it is a
        # fact about the data, and a slide set with more glass would compress
        # further. spec.md 6.5 carries what it decides for 512 and 1024.
        raw = sum(r['n_got'] for r in rows) * pre_px * pre_px * 3
        if raw:
            print(f'PNG is {gb * 1e9 / raw:.1%} of raw '
                  f'({raw / 1e9:.1f} GB uncompressed)')

    if failures:
        print(f'\n{len(failures)} cell(s) failed:')
        for what, why in failures:
            print(f'  {what}: {why}')
    return 1 if failures else 0


def _plans_for(wsi, args):
    """`(sampler plans, per-rung pre-tile plans, pre_px)`, all rungs at once.

    Two plans per rung and they are not interchangeable:

        plan_tile  gates the sampler. `tile_size=plan.read_size` hands
                   TileSampler the LEVEL pixel count whose level-0 footprint is
                   exactly `tile * ds`, which is what the richness buckets
                   have to be scored over.
        plan_pre   drives the read. Same level -- the level depends only on the
                   rung -- but `read_size` is the pre-tile's.

    ASCENDING IN ds, WHICH `TileSampler.sample` REQUIRES AND SAYS WHY: a chain
    truncates at the rung where it first lands in a zero-capped bucket and
    every coarser rung is then skipped, which is only expressible if the
    coarser ones have not been filled yet.
    """
    pre_px = pre_tile_px(args.tile, args.pre_tile_factor)
    rungs = sorted(float(d) for d in args.ds)
    ladder = DsLadder(rungs=tuple(rungs))
    tiles = ladder.plan(wsi.level_downsamples, args.tile)
    pres = ladder.plan(wsi.level_downsamples, pre_px)

    plans, pre_plans = [], {}
    for plan_tile, plan_pre in zip(tiles, pres):
        if plan_pre.level != plan_tile.level:
            raise AssertionError(
                f'ds {plan_tile.rung_ds:g}: the tile plan reads level '
                f'{plan_tile.level} and the pre-tile plan level '
                f'{plan_pre.level}. DsLadder picks the level from the rung '
                f'alone, so this cannot happen unless that changed -- and if '
                f'it did, the crop would be at a different resolution than the '
                f'gate')
        # The reserve is the PRE-tile, handed to the sampler rather than
        # repaired afterwards: a lattice that reserves it never offers a
        # position whose pre-tile runs off the region, and `clip_px` stops
        # being a repair for something the geometry could have refused.
        plans.append(dataclasses.replace(
            plan_tile,
            reserve_l0=plan_tile.footprint_l0 * args.pre_tile_factor))
        pre_plans[float(plan_tile.rung_ds)] = plan_pre
    return plans, pre_plans, pre_px


def _richness(args) -> RichnessConfig:
    """The seven-bucket contract, with `--tissue-ratio` as caps if given.

    THE FLAG IS BACK AND IT MEANS SOMETHING ELSE NOW. The old `--tissue-ratio`
    was a SECOND gate that scored the same quantity as the buckets, and the two
    disagreeing produced the 475/500 corpus of 2026-08-26: 22.5 per cent of
    every rung was reserved for buckets the gate had already emptied. This one
    is not a second gate -- it is written INTO the caps, so there is one
    mechanism and one place a tile can be refused.

    THE FLOORS ARE DROPPED WITH IT. A floor asks a rung to supply a share of a
    bucket the caps have just closed, which no slide can do, and the run comes
    back short with the shortfall reported as a property of the slides -- the
    2026-08-26 failure again, by a different route. So a ratio replaces the
    whole contract rather than being layered on it.
    """
    if args.tissue_ratio is None:
        return RichnessConfig(bucket_frame=args.bucket_frame)
    caps = caps_for_tissue_ratio(float(args.tissue_ratio))
    return RichnessConfig(caps=caps, floors=tuple(0.0 for _ in caps),
                          bucket_frame=args.bucket_frame)


def _sampler_config(plans, args) -> SamplerConfig:
    """One config for every rung, which is what inheritance requires.

    ONE SAMPLER OVER ALL RUNGS, NOT ONE PER RUNG, and that is the change that
    makes chains possible at all. `_choose_centres` runs once, before any rung
    is filled, and `_place_inherited` then validates each centre at each rung;
    a sampler per rung would choose its own centres and no two rungs would
    share one. The corpus of 2026-08-27 has `inherit_id = -1` on all 6,388
    rows for exactly this reason -- not because the option was set wrong, but
    because it was never reachable.

    `tile` is taken from the FINEST plan. Every plan has the same
    `tile_size` -- it is `args.tile` in level pixels -- and asserting that here
    beats letting one rung's differing value decide the whole config silently.
    """
    sizes = {int(q.tile_size) for q in plans}
    if len(sizes) != 1:
        raise AssertionError(
            f'the rungs disagree about tile_size: {sorted(sizes)}. '
            f'SamplerConfig.tile is one number for the whole ladder, so one '
            f'rung would be gated on a square the others are not')

    return SamplerConfig(
        tile=sizes.pop(), n_per_rung=args.n, seed=args.seed,
        candidates=args.candidates,
        max_tries_per_tile=max(1, args.max_tries // max(args.n, 1)),
        overlap=OverlapConfig(grid_step=args.grid_step,
                              max_overlap_ratio=args.max_overlap,
                              overlapping_share=args.overlapping_share),
        richness=_richness(args),
        inherit=InheritConfig(stack_kind='F', share=args.inherit_share,
                              source_rung=args.inherit_source_rung))


def _extract_slide(wsi, trm, slide_mask, mask_meta, args, stem, failures):
    """Every rung of one slide, from ONE sampler. Returns a row per rung written.

    The sampler runs once and is then split by rung into the per-rung stores.
    The store layout does not change -- one directory per (slide, ds) -- but
    every directory now carries the same `sampler_id`, which is the honest
    thing: the rungs were cut by one decision, and under inheritance they are
    not independent of each other.

    RESUME IS PER RUNG AND THE SAMPLING IS NOT SKIPPED. A rung whose directory
    already exists is skipped at the WRITE, not at the sample -- because the
    inheritance set is chosen across all rungs at once and cannot be rebuilt
    from a subset. So a re-run after a walltime kill pays the sampling again
    and none of the reads, which is where the hours are.
    """
    plans, pre_plans, pre_px = _plans_for(wsi, args)
    cfg = _sampler_config(plans, args)
    sampler = TileSampler(wsi, trm, cfg).sample(plans)

    by_rung = {}
    for sample in sampler:
        by_rung.setdefault(float(sample.meta.ds), []).append(sample)

    chains = len({s.meta.inherit_id for s in sampler
                  if s.meta.inherit_id >= 0})
    print(f'    sampler {cfg.sampler_id()}   {len(sampler)} tiles over '
          f'{len(plans)} rungs, {chains} chains', flush=True)

    rows = []
    for plan in plans:
        ds = float(plan.rung_ds)
        try:
            rows.append(_write_rung(wsi, slide_mask, mask_meta, args, cfg,
                                    pre_plans[ds], pre_px, ds,
                                    by_rung.get(ds, []),
                                    sampler.reports.get(ds)))
        except PreTileStore.PreTileMismatch as e:
            # An existing finished directory. Not a failure -- it is what
            # --overwrite is for, and skipping is what makes this script safe
            # to re-run after a walltime kill.
            print(f'    ds {ds:g}: have it   '
                  f'({e.args[0].splitlines()[0]})', flush=True)
        except Exception as e:                                   # noqa: BLE001
            print(f'    ds {ds:g}: FAILED  {type(e).__name__}: {e}',
                  flush=True)
            failures.append((f'{stem} ds{ds:g}', f'{type(e).__name__}: {e}'))
    return rows


def _write_rung(wsi, slide_mask, mask_meta, args, cfg, plan_pre, pre_px, ds,
                samples, report):
    """One (slide, ds) directory, from samples the shared sampler already chose."""
    # BUILT BEFORE THE META, because `sampler_id` is part of the store's
    # identity and the meta cannot be assembled without it. It replaced
    # `tissue_ratio`, which named a gate the sampler no longer has -- and which
    # covered only one of the three axes, so two corpora differing in their
    # bucket floors used to share a directory.
    meta = PreTileMeta.of(wsi, plan_pre, tile=args.tile,
                          sampler_id=cfg.sampler_id(), seed=args.seed,
                          segmenter_id=mask_meta.segmenter_id,
                          factor=args.pre_tile_factor, n_requested=args.n)
    folder = PreTileStore.create(args.root, meta, overwrite=args.overwrite)
    origin, span = slide_mask.origin, slide_mask.span

    records, written = [], 0
    for i, sample in enumerate(samples):
        info = sample.meta
        px, py = info.reserve_origin_l0

        # `clip` is an ASSERTION, not a repair. The lattice was handed
        # `reserve_l0`, so it never offered a position whose pre-tile runs off
        # the region -- and if one appears anyway, the reserve stopped binding
        # and every tile after it is suspect.
        # `info.reserve`, not a reserve recomputed from `meta` here: that was a
        # THIRD spelling of one number, and the assertion is worth nothing if
        # it checks a different rectangle than the one that was read.
        reserve = int(info.reserve)
        clip = max(0,
                   origin[0] - px, origin[1] - py,
                   px + reserve - (origin[0] + span[0]),
                   py + reserve - (origin[1] + span[1]))
        if clip:
            raise AssertionError(
                f'pre-tile {i} at level-0 ({px}, {py}) runs {clip} px off the '
                f'scanned region, but the sampler reserved '
                f'{int(info.reserve)} px around every tile. The reserve is '
                f'not binding -- check that plan.reserve_l0 reached '
                f'filter_patchable and that the mask origin is the one the '
                f'lattice used')

        # The RESERVE, not the tile: the store holds pre-tiles and the tile is
        # their centre crop. `materialise` reads it through the same numbers
        # the lattice honoured, so the read cannot disagree with the geometry
        # that placed it.
        image = sample.materialise(wsi, extent='reserve').image
        if image.shape[0] != pre_px:
            image = cv2.resize(image, (pre_px, pre_px),
                               interpolation=cv2.INTER_AREA)

        # The three axes come off the sampler, not out of a second computation
        # here: `bucket` depends on the scorer and the edges, `overlap_max` on
        # what else that rung took, `inherit_id` on a set fixed before any rung
        # was filled. None is recoverable from (x, y).
        record = PreTileRecord(
            index=i, x=int(info.x), y=int(info.y), clip_px=0,
            bucket=info.bucket, score=float(info.score),
            overlap_max=float(info.overlap_max),
            inherit_id=int(info.inherit_id), origin=info.origin,
            parent_x=int(info.parent_x), parent_y=int(info.parent_y))
        path = PreTileStore.save_tile(folder, record, image, meta)
        written += os.path.getsize(path)
        records.append(record)
        sample.release()          # streaming: the pixels are on disk now

    PreTileStore.write_index(folder, records, meta)

    chains = sum(1 for r in records if r.inherit_id >= 0)
    # `n_inherit_refused` IS THE COST OF `on_incomplete='drop'`, PER RUNG.
    # A centre chosen at `source_rung` is guaranteed to FIT at every finer
    # rung, but not to have tissue there: `caps[bucket] <= 0` on the top two
    # buckets is the retired tissue gate, and it binds the inherited set too.
    # A chain refused at any rung truncates and is then dropped whole by
    # `stacks()`, so this column is the only place the loss is visible.
    refused = int(getattr(report, 'n_inherit_refused', 0)) if report else 0
    breaching = int(getattr(report, 'n_inherit_breaching', 0)) if report else 0

    print(f'    ds {ds:g}  level {plan_pre.level}  read '
          f'{plan_pre.read_size} -> {pre_px}   {len(records)}/{args.n} tiles, '
          f'{chains} in chains, {refused} refused, {written / 1e6:.0f} MB',
          flush=True)

    return {'wsi_stem': meta.wsi_stem, 'ds': meta.ds, 'tile': meta.tile,
            'pre_px': pre_px, 'sampler_id': meta.sampler_id,
            'level': meta.level, 'read_size': meta.read_size,
            'footprint_l0': int(meta.tile_footprint_l0),
            'n_requested': args.n, 'n_got': len(records), 'n_clipped': 0,
            'n_chain': chains, 'n_inherit_refused': refused,
            'n_inherit_breaching': breaching,
            'bytes': written, 'dir': os.path.basename(str(folder))}


if __name__ == '__main__':
    sys.exit(main())
