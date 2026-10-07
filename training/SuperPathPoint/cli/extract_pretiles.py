#!/usr/bin/env python3
"""spec.md 12 step 3c: cut the pre-tiles the training set is made of.

    python training/SuperPathPoint/cli/extract_pretiles.py \
        --tile 256 --n 500

Outputs:
    result/cache/<--pretile-cache-job>_pretiles/<seg_id>/<slide>/
        <region_id>_<sampler_id>_<plan>/f<factor>/ds<d>/000000.png ... index.csv, meta.json
    extract_pretiles.csv        in result/<SLURM_JOB_NAME or ExtractPreTiles>/

argparse, a loop, and printed progress. Everything that decides anything is in
`utilities/Store.py` (the format and the corpus address), `utilities/TileSampler.py`
(which positions exist, and the centre-crop geometry), `utilities/DsLadder.py`
(which level to read) and `common/Corpora.py` (the sampler config). CLAUDE.md: a library layer that prints cannot be called by a
bench.

WHAT COMES OUT IS A PRE-TILE, NOT A TILE
-----------------------------------------
Each PNG is `tile * factor` on a side, centred on a position the sampler's
richness buckets admitted. The tile itself is never written: it is
`TileSampler.centre_crop(pre, tile)` and lives only in the training loop.

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

A CLIP IS REFUSED, NOT RECORDED. Near the edge of the scanned rectangle a
pre-tile could run off the slide (`PreTileStore` never slides the window
inward, because that would move the tile off centre and every crop downstream
would be of the wrong place). `TileSampler` is given the pre-tile as
`reserve_l0`, so the lattice never OFFERS such a position: `patchable` gets the reserve rather than the
tile, and the first legal corner sits a margin inside the region. `clip_px` is
therefore 0 on every record, and the loop below asserts it rather than writing
it. A non-zero clip means the reserve stopped binding, and every tile after
it is suspect -- which is a thing to stop on, not a column to fill in.

RESUMABLE ON PURPOSE
---------------------
`index.csv` is written last, and its presence is what marks a directory
complete. A job killed at walltime therefore leaves directories that
`PreTileCorpus.rung_dirs` skips and that a re-run rebuilds, rather than a short index that reads as a
small dataset.
"""

from __future__ import annotations

import argparse
import dataclasses
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', '..', 'utilities'))
import _paths                                                     # noqa: E402
_paths.setup_import_paths('SuperPathPoint')

from cli import (add_pretile_args, job_result_dir, mask_root,     # noqa: E402
                 pretile_root)


import cv2                                                        # noqa: E402
import torch                                                      # noqa: E402

from Cache import find, read_meta, wsi_stem_of                    # noqa: E402
from SafeSlide import SafeSlide                                    # noqa: E402
from TileSampler import SamplerConfig, TileSampler, pre_tile_px     # noqa: E402
from SlideReader import SlideReader                                # noqa: E402
from TissueMaskConfig import MASK_RECIPES, MaskMaker              # noqa: E402

from DsLadder import DEFAULT_RUNGS, DsLadder                      # noqa: E402
from Store import (PreTileCorpus, PreTileMeta, PreTileRecord,     # noqa: E402
                   PreTileStore, StoreMismatch)
from common.Corpora import ladder, pretile_spec, sampler_config   # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # --pretile-cache-job, --mask-cache-job, --seg, --pre-tile-factor, --tile.
    # --tile is what the model sees: v1 is 256, and 512 and 1024 are separate
    # models and separate extractions (spec.md 6.5). --pre-tile-factor 3 is
    # derived (spec.md 6.6, bound 2.49) and is NOT a knob to tune for disk.
    add_pretile_args(ap)
    ap.add_argument('--wsi', nargs='*', default=None,
                    help='slide paths. Default: every mask in the mask cache')
    ap.add_argument('--rungs', type=float, nargs='+', default=list(DEFAULT_RUNGS),
                    help='the ladder rungs to extract. Part of the corpus '
                         'address: a chain is only a chain over rungs sampled '
                         'together, so another list is another corpus')
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
    # A chain is one level-0 centre with a tile at EVERY rung.
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
    ap.add_argument('--step', type=float, default=1.0,
                    help='lattice step as a fraction of the tile. 1.0 is '
                         'disjoint, and the only spelling of that. 0.5 is a '
                         'deliberate 50 per '
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
    # THE CONFIG AND THE CORPUS ARE KNOWN BEFORE ANY SLIDE IS OPENED. Every
    # knob is slide-independent, so the directory this run writes is fixed
    # here and printed -- the key a reader passes to find it again.
    cfg = sampler_config(
        n=args.n, seed=args.seed, candidates=args.candidates,
        max_tries=args.max_tries, step=args.step,
        max_overlap=args.max_overlap,
        overlapping_share=args.overlapping_share,
        bucket_frame=args.bucket_frame,
        inherit_share=args.inherit_share,
        inherit_source_rung=args.inherit_source_rung)
    rich = cfg.richness
    print('  richness  ' + '  '.join(
        f'{nm}:{f:.0%}/{c:.0%}' for nm, f, c
        in zip(rich.names, rich.floors, rich.caps)) + '   (floor/cap)',
        flush=True)
    mask_cfg = MASK_RECIPES[args.seg]
    corpus = PreTileCorpus.of(pretile_root(args), mask_cfg, cfg,
                              ladder(args.rungs, args.tile, args.pre_tile_factor),
                              args.pre_tile_factor)
    seg_dir = mask_root(args) / mask_cfg.seg_id()
    print(f'masks  {seg_dir}\ntiles  {corpus.root}\ncorpus {corpus.key}',
          flush=True)

    paths = args.wsi
    # A PATH, NOT A STEM, and the difference would surface four frames down
    # as openslide's "Unsupported or missing image file" -- which reads as a
    # corrupt slide, not as a wrong argument. The stem is what every OTHER
    # thing here is keyed by (the mask cache, the pre-tile cache, --wsi-stem in
    # make_ha_labels), so reaching for it is the expected mistake.
    for candidate in paths or ():
        if not os.path.exists(candidate):
            hit = find(seg_dir, '*/mask_meta.json')
            known = sorted(read_meta(p)['wsi_path'] for p in hit)
            match = [k for k in known if wsi_stem_of(k) == candidate]
            ap.error(
                f'--wsi takes slide PATHS, not stems, and {candidate!r} is not '
                f'a file.' + (f' Did you mean {match[0]}?' if match else
                              f' Known: {", ".join(os.path.basename(k) for k in known[:4])}'
                              f'{" ..." if len(known) > 4 else ""}'))

    if not paths:
        found = find(seg_dir, '*/mask_meta.json')
        if not found:
            print(f'no masks under {seg_dir}. Run '
                  f'utilities/cli/build_cache/build_mask_store.py first, or '
                  f'pass --wsi and the mask is made on the way.')
            return 1
        paths = [read_meta(p)['wsi_path'] for p in found]
        print(f'{len(paths)} slides from the mask cache', flush=True)

    rows, failures = [], []
    # A MISSING MASK IS MADE, NOT REFUSED: MaskMaker segments on a miss and
    # writes it to the same cache a later run hits. The segmenter is built
    # only on that first miss and released when the loop ends.
    # THE DEVICE IS NOT OPTIONAL. `Uni2PcaSegConfig.build(None)` (and hest's)
    # falls back to CPU, and a ViT over ~10^5 tiles on CPU ran at ~2.5 tiles/s
    # against the ~650 the GPU does: one slide took over ten hours, all of it
    # lost when the job was cancelled, because a mask is written only when its
    # slide is finished.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'mask segmenter on {device}', flush=True)
    with MaskMaker(mask_cfg, mask_root(args), device) as masks:
        for index, wsi_path in enumerate(paths, 1):
            stem = wsi_stem_of(wsi_path)
            print(f'\n[{index}/{len(paths)}] {stem}', flush=True)
            with SafeSlide(wsi_path) as wsi:
                rows += _extract_slide(
                    wsi, masks, cfg, corpus, args.rungs, tile=args.tile, n=args.n,
                    overwrite=args.overwrite, failures=failures)

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
        # PNG against raw. Measured on the v1 corpus: 45.1 per
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


def _plans_for(wsi, *, tile: int, pre_tile_factor: int, rungs):
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

    KEYWORD ARGUMENTS, NOT `args`: `cli/prepare_chain_stack.py` calls this
    and `_extract_slide` in process for F's and C's own corpora, with values
    that never went through argparse.
    """
    pre_px = pre_tile_px(tile, pre_tile_factor)
    rungs = sorted(float(d) for d in rungs)
    dsl = DsLadder(rungs=tuple(rungs))
    # The tile plans come from the pre-tile CAMERA, so the reserve is what it
    # reads (`ReadSpec.place`) rather than a footprint times the factor
    # computed here a second time.
    tiles = ladder(rungs, tile, pre_tile_factor).plans_for(wsi)
    pres = dsl.plan(wsi.level_downsamples, pre_px)

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
        plans.append(plan_tile)
        pre_plans[float(plan_tile.rung_ds)] = plan_pre
    return plans, pre_plans, pre_px


def _extract_slide(wsi, masks: MaskMaker, cfg: SamplerConfig,
                   corpus: PreTileCorpus, rungs, *, tile: int, n: int, overwrite: bool,
                   failures: list):
    """Every rung of one slide, from ONE sampler. Returns a row per rung written.

    The sampler runs once and is then split by rung into the per-rung
    directories of `corpus`. Every rung carries the same `sampler_id`, which
    is the honest thing: the rungs were cut by one decision, and under
    inheritance they are not independent of each other.

    RESUME IS PER RUNG AND THE SAMPLING IS NOT SKIPPED. A rung whose directory
    is already finished is skipped at the WRITE, not at the sample -- because
    the inheritance set is chosen across all rungs at once and cannot be
    rebuilt from a subset. So a re-run after a walltime kill pays the sampling
    again and none of the reads, which is where the hours are.

    `cfg` and `corpus` ARE BUILT BY THE CALLER: `main()` from argparse,
    `cli/prepare_chain_stack.py` from `common/Corpora.RECIPES`. `rungs` must
    be the ladder `corpus` was addressed with, and that is checked -- a rung
    list that disagrees with the address would file one corpus's tiles under
    another's key.
    """
    if ladder(rungs, tile, corpus.factor).key() != corpus.plan:
        raise AssertionError(
            f'rungs {sorted(rungs)} are plan {ladder(rungs, tile, corpus.factor).key()}, but the '
            f'corpus is addressed as {corpus.plan}')
    stem = wsi_stem_of(wsi)
    tile, pre_tile_factor = int(tile), int(corpus.factor)
    mask, _hit = masks.mask(wsi)
    segmenter_id = read_meta(
        masks.slide_dir(stem) / 'mask_meta.json').get('segmenter_id', '')
    frac = float(mask.main_mask.mean())
    print(f'    mask {mask.main_mask.shape[0]}x{mask.main_mask.shape[1]}, '
          f'tissue {frac:.1%}, {len(mask.tissue_regions)} regions   '
          f'({segmenter_id})', flush=True)

    plans, pre_plans, pre_px = _plans_for(wsi, tile=tile,
                                          pre_tile_factor=pre_tile_factor,
                                          rungs=rungs)
    sampler = TileSampler(wsi, mask, cfg).sample(plans)
    reader = SlideReader(wsi, resize='area')

    by_rung = {}
    for sample in sampler:
        by_rung.setdefault(float(sample.meta.ds), []).append(sample)

    chains = len({s.meta.inherit_id for s in sampler
                  if s.meta.inherit_id >= 0})
    print(f'    sampler {cfg.identity_id()}   {len(sampler)} tiles over '
          f'{len(plans)} rungs, {chains} chains', flush=True)

    rows = []
    for plan in plans:
        ds_ = float(plan.rung_ds)
        try:
            rows.append(_write_rung(wsi, mask.slide_mask, segmenter_id, cfg,
                                    corpus, pre_plans[ds_], pre_px, ds_,
                                    by_rung.get(ds_, []),
                                    sampler.reports.get(ds_),
                                    tile=tile, n=n, overwrite=overwrite,
                                    reader=reader))
        except StoreMismatch as e:
            # An existing finished directory. Not a failure -- it is what
            # --overwrite is for, and skipping is what makes this script safe
            # to re-run after a walltime kill.
            print(f'    ds {ds_:g}: have it   '
                  f'({e.args[0].splitlines()[0]})', flush=True)
        except Exception as e:                                   # noqa: BLE001
            print(f'    ds {ds_:g}: FAILED  {type(e).__name__}: {e}',
                  flush=True)
            failures.append((f'{stem} ds{ds_:g}', f'{type(e).__name__}: {e}'))
    return rows


def _write_rung(wsi, slide_mask, segmenter_id: str, cfg: SamplerConfig,
                corpus: PreTileCorpus, plan_pre, pre_px, ds, samples, report, *,
                tile: int, n: int, overwrite: bool, reader):
    """One (slide, ds) directory, from samples the shared sampler already chose.
    `reader` is the slide's `SlideReader` ('area' filter)."""
    meta = PreTileMeta.of(wsi, plan_pre, corpus, tile=int(tile),
                          seed=cfg.seed, segmenter_id=segmenter_id,
                          n_requested=n)
    folder = PreTileStore.create(corpus, meta, overwrite=overwrite)
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
                f'the patchable view and that the mask origin is the one the '
                f'lattice used')

        # The RESERVE, not the tile: the store holds pre-tiles and the tile is
        # their centre crop. The reader reads the tile grown by the pre-tile
        # margin -- the very read `pretile_spec` told the sampler to reserve
        # (`ReadSpec.place`) -- so the read cannot disagree with the geometry
        # that placed it.
        image = reader.read(info.x, info.y, pretile_spec(tile, corpus.factor),
                            info.ds, stack=info.stack_kind)
        if image is None:
            raise AssertionError(f'pre-tile {i} at level-0 ({info.x}, {info.y}) '
                                 f'reads off the slide')
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

    PreTileStore.write_index(folder, records)

    chains = sum(1 for r in records if r.inherit_id >= 0)
    # `n_inherit_refused` IS THE COST OF `on_incomplete='drop'`, PER RUNG.
    # A centre chosen at `source_rung` is guaranteed to FIT at every finer
    # rung, but not to have tissue there: `caps[bucket] <= 0` on the top two
    # buckets is the tissue gate, and it binds the inherited set too.
    # A chain refused at any rung truncates and is then dropped whole by
    # `stacks()`, so this column is the only place the loss is visible.
    refused = int(getattr(report, 'n_inherit_refused', 0)) if report else 0
    breaching = int(getattr(report, 'n_inherit_breaching', 0)) if report else 0

    print(f'    ds {ds:g}  level {plan_pre.level}  read '
          f'{plan_pre.read_size} -> {pre_px}   {len(records)}/{n} tiles, '
          f'{chains} in chains, {refused} refused, {written / 1e6:.0f} MB',
          flush=True)

    return {'wsi_stem': meta.wsi_stem, 'ds': meta.ds, 'tile': meta.tile,
            'pre_px': pre_px, 'sampler_id': meta.sampler_id,
            'level': meta.level, 'read_size': meta.read_size,
            'footprint_l0': int(meta.tile_footprint_l0),
            'n_requested': n, 'n_got': len(records), 'n_clipped': 0,
            'n_chain': chains, 'n_inherit_refused': refused,
            'n_inherit_breaching': breaching,
            'bytes': written, 'corpus': corpus.key}


if __name__ == '__main__':
    sys.exit(main())
