#!/usr/bin/env python3
"""Build a slide's reference set under quota, and write it to a FeatureStore.

The reference behind stage 1 is currently 40 tiles per level, drawn uniformly
inside the tissue mask and rebuilt in memory on every LocaScopePipeline.build().
That is enough for a KNN to saturate (result/MppEstimate: 40 -> 640 buys 2.7
points) but it controls nothing about what those tiles contain and it leaves no
record of how they were chosen. The pooling stores showed the cost: up to 13% of
the tiles in one slide's deep banks were bit-identical twins, every one of them a
hole or unscanned canvas that SafeSlide fills with a flat colour, which therefore
encodes to the same vector at every level and tells the estimator nothing.

This tool picks the tiles under an explicit background quota (ReferenceSampler),
reads them, drops the ones the scanner never photographed, encodes what is left,
and stores the coordinates together with WHY each one is there -- its background
fraction, its bucket, whether it came from the grid or was displaced, and which
level-0 location it inherits. None of that survives a build that only lives in
memory, and without it the sampling rules can only be believed, not checked.

Two passes, and the first one is free
-------------------------------------
    --dry-run   mask, geometry, quotas. Reads no tile. Answers "will this level
                fall short, and of what" in the time it takes to segment the
                slide, instead of an hour into encoding.

    full        the above, then the inheritance set is validated at EVERY level
                by reading it -- a hole is a property of (location, level), so a
                location can be solid at level 0 and missing at level 3, and a
                correspondence with holes in it is not a correspondence -- and
                only then are the per-level quotas filled.

Cost note: --pooling tokens keeps all 197 tokens and costs roughly 605 KB per
tile, so 1000 tiles across ten levels is about 6 GB per slide. --pooling cls
keeps one vector and costs about 61 MB. Encoding time is the same either way;
the difference is disk and every later read.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _d in ('utilities', 'aiNNModel'):
    p = str(_ROOT / _d)
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402

import FeatureStore as FS                                           # noqa: E402
import ReferenceSampler as RS                                       # noqa: E402
from ReferenceSampler import BUCKETS, SamplerConfig                 # noqa: E402
from SafeSlide import SafeSlide                                     # noqa: E402
from TissuesRegionsMask import TissuesRegionsMask                   # noqa: E402
from TissueSegFunc import HestSegConfig                             # noqa: E402
from TileEncoderFunc import encoder_config, encoder_names           # noqa: E402
import _paths                                                       # noqa: E402
from _paths import encoder_tag, job_result_dir                      # noqa: E402


def _record(s, i: int) -> dict:
    """One row of a LevelSample, as a plain dict, so replacements can be
    appended to the queue as first-class candidates."""
    return dict(x=int(s.x[i]), y=int(s.y[i]), region=int(s.region[i]),
                rc=(int(s.grid_rc[i, 0]), int(s.grid_rc[i, 1])),
                kind=int(s.kind[i]), white=float(s.white_frac[i]),
                bucket=int(s.bucket[i]), origin=int(s.origin[i]),
                parent_x=int(s.parent_x[i]), parent_y=int(s.parent_y[i]),
                inherit=int(s.inherit_id[i]))


def _record_from_geom(g, i: int) -> dict:
    return dict(x=int(g.xy[i, 0]), y=int(g.xy[i, 1]), region=int(g.region[i]),
                rc=(int(g.grid_rc[i, 0]), int(g.grid_rc[i, 1])),
                kind=int(g.kind[i]), white=float(g.white[i]),
                bucket=int(g.bucket[i]), origin=0,
                parent_x=-1, parent_y=-1, inherit=-1)


def read_level(slide, sampler, sample, geom, level, cfg, verbose=True):
    """Read every chosen tile; replace the ones that were never photographed.

    A rejected tile is replaced from its OWN bucket, not from whichever bucket
    happens to have room: holes come in contiguous patches -- the twin
    measurement put them 0.7 to 1.2 tile steps apart -- so a whole neighbourhood
    is refused at once, and topping up from anywhere would quietly move the
    background mix that the quotas exist to hold.

    Returns (images, records, stats).
    """
    queue = [_record(sample, i) for i in range(len(sample))]
    imgs, kept, rejected, replaced, unreplaced = [], [], 0, 0, 0
    valids = []
    at = 0
    while at < len(queue):
        r = queue[at]
        at += 1
        rgb, valid = slide.read_region_valid((r['x'], r['y']), level,
                                             (cfg.tile, cfg.tile))
        vf = float(valid.mean())
        if vf < cfg.min_valid:
            rejected += 1
            more = sampler.replace(level, r['bucket'], n=1)
            if more:
                x, y, gi = more[0]
                queue.append(_record_from_geom(geom, gi))
                replaced += 1
            else:
                unreplaced += 1
            continue
        r['valid'] = vf
        valids.append(vf)
        imgs.append(rgb)
        kept.append(r)

    stats = dict(rejected=rejected, replaced=replaced, unreplaced=unreplaced,
                 kept=len(kept),
                 valid_min=float(min(valids)) if valids else float('nan'))
    if verbose and rejected:
        print(f'      holes: {rejected} tiles below valid {cfg.min_valid:.2f}, '
              f'{replaced} replaced, {unreplaced} could not be', flush=True)
    return imgs, kept, stats


def to_sample(records, level, ds):
    n = len(records)
    return RS.LevelSample(
        level=level, ds=ds,
        x=np.array([r['x'] for r in records], np.int64),
        y=np.array([r['y'] for r in records], np.int64),
        region=np.array([r['region'] for r in records], np.int32),
        grid_rc=np.array([r['rc'] for r in records], np.int32).reshape(n, 2),
        kind=np.array([r['kind'] for r in records], np.int8),
        white_frac=np.array([r['white'] for r in records], np.float32),
        bucket=np.array([r['bucket'] for r in records], np.int8),
        origin=np.array([r['origin'] for r in records], np.int8),
        parent_x=np.array([r['parent_x'] for r in records], np.int64),
        parent_y=np.array([r['parent_y'] for r in records], np.int64),
        inherit_id=np.array([r['inherit'] for r in records], np.int32),
        valid_frac=np.array([r['valid'] for r in records], np.float32))


def bucket_line(s) -> str:
    got = [f'{BUCKETS[b]}={int((s.bucket == b).sum())}'
           for b in range(len(BUCKETS))]
    o = s.origin
    return ('  '.join(got) +
            f'   |  grid={int((o == 0).sum())} jitter={int((o == 1).sum())} '
            f'inherit={int((o == 2).sum())}')


def build_slide(wsi_path, args, cfg, encoder, spec, device, hest_method,
                encoder_id, mask_id, out_root, rows, reports) -> None:
    """Build one slide. Appends a row per level to `rows` and the pre-flight
    text to `reports`, so the run leaves a record under result/<job> rather than
    only in a log that the next job overwrites."""
    stem = Path(wsi_path).stem
    slide = SafeSlide(str(wsi_path))
    try:
        t0 = time.time()
        mask = TissuesRegionsMask.from_wsi(
            slide, level=args.mask_level, method=hest_method,
            seg_chunk_px=int(args.seg_chunk_px), stitch_overlap=128,
            read_chunk_px=(int(args.read_chunk_px)
                           if args.read_chunk_px else None))
        n_raw = len(mask.tissue_regions)
        # The same two stages LocaScopePipeline.build() runs, in the same order.
        # Level-independent, so they belong here and the per-level "can this
        # region host a tile" test stays inside build_level_geoms.
        mask.filter_regions(min_ratio=args.min_region_ratio)
        mask.merge_overlapping()
        print(f'  mask: tissue={mask.tissue_fraction() * 100:.1f}%  '
              f'regions {n_raw} -> {len(mask.tissue_regions)}  '
              f'({time.time() - t0:.0f}s)', flush=True)
        if not mask.tissue_regions:
            print('  no region survived the filters -- skipped', flush=True)
            return

        levels = (args.levels if args.levels
                  else list(range(slide.level_count)))
        geoms = RS.build_level_geoms(mask, levels, slide.level_downsamples, cfg)
        if not geoms:
            print('  no level can host a tile -- skipped', flush=True)
            return

        rng = np.random.default_rng(cfg.seed)
        inherit = RS.pick_inheritance(geoms, mask, cfg, rng)
        plans = {lv: RS.plan_level(g, cfg, inherited=0)
                 for lv, g in geoms.items()}
        report = RS.render_preflight(stem, cfg, geoms, plans, inherit)
        print(report)
        reports.append(report)
        unusable = [lv for lv, p in sorted(plans.items()) if p.unusable]

        if args.dry_run:
            print('  --dry-run: no tile was read. The per-level quotas above do'
                  ' not yet subtract\n  the inheritance set, whose final size'
                  ' needs the read validation.\n', flush=True)
            return unusable

        # The one read before the plan is fixed. Geometry has already cut the
        # candidates down, so this costs the intersection rather than the pool.
        t0 = time.time()
        inherit = RS.validate_inheritance(
            inherit, geoms,
            lambda x, y, lv, t: float(
                slide.read_region_valid((x, y), lv, (t, t))[1].mean()),
            cfg)
        n_inh = len(inherit.xy0)
        print(f'  inheritance: {n_inh} locations survive at every level '
              f'({time.time() - t0:.0f}s)', flush=True)

        sampler = RS.ReferenceSampler(geoms, cfg, mask=mask, inherit=inherit)
        base_mpp = slide.base_mpp  # SafeSlide.base_mpp: mean of mpp-x/y, one definition

        for lv in sorted(geoms):
            g = geoms[lv]
            plan = RS.plan_level(g, cfg, inherited=n_inh)
            if plan.unusable:
                # Skipped, not fatal, and the other levels are unaffected. A
                # deep level running out of grid positions is a fact about the
                # slide -- BRACS_1228's level 3 offers 116 in total -- not a
                # misconfiguration, and it is no reason for level 0's thousand
                # to go unbuilt.
                print(f'    L{lv}: {plan.got} tiles achievable, below '
                      f'min_useful {cfg.min_useful} -- level skipped',
                      flush=True)
                rows.append(dict(wsi_stem=stem, level=lv, verdict='UNUSABLE',
                                 n_grid=len(g.xy), n_target=plan.n_target,
                                 planned=plan.got, built=0))
                continue
            s = sampler.plan(lv, plan)
            t0 = time.time()
            imgs, records, stats = read_level(slide, sampler, s, g, lv, cfg)
            if not records:
                print(f'    L{lv}: nothing survived the read -- skipped',
                      flush=True)
                continue
            t_read = time.time() - t0

            # .pooled() reduces inside the batch loop, so the 197-token
            # intermediate never crosses to the host: 86 KB per tile instead of
            # 1.21 MB. Same numbers -- test_encoders scores this against
            # pooling the tokens afterwards, for every encoder rather than one.
            feats = encoder.pooled(imgs, args.pooling)
            # The names come from the mode and the spec, not from a second
            # reduction -- pooled_spec reads the tensor it is handed and checks
            # the slot count against what pool_slots named.
            fs = encoder.pooled_spec(feats, args.pooling)
            slots, layout = fs.slots, fs.slot_layout
            final = to_sample(records, lv, g.ds)

            meta = FS.StoreMeta(
                wsi_stem=stem, wsi_path=str(wsi_path), level=lv, ds=g.ds,
                mpp=base_mpp * g.ds, base_mpp=base_mpp, tile_size=cfg.tile,
                overlap=True, pooling=args.pooling, slots=tuple(slots),
                slot_layout=layout, dim=spec['dim'],
                feat_hw=tuple(spec['feat_hw']),
                num_prefix=spec['num_prefix'], encoder_id=encoder_id,
                mask_id=mask_id, coverage='sample', n_available=len(g.xy),
                sample_seed=cfg.seed, n_tiles=len(final),
                sampler_id=cfg.sampler_id())
            path = FS.save(out_root, meta=meta,
                           **RS.to_store_args(final, feats))

            print(f'    L{lv}: {len(final)}/{plan.n_target} tiles   '
                  f'read {t_read:.0f}s  encode {time.time() - t0 - t_read:.0f}s',
                  flush=True)
            print(f'      {bucket_line(final)}', flush=True)
            for note in sampler.notes(lv):
                print(f'      note: {note}', flush=True)
            print(f'      -> {Path(path).name}', flush=True)

            row = dict(wsi_stem=stem, level=lv,
                       verdict='SHORT' if plan.short else 'OK',
                       n_grid=len(g.xy), n_target=plan.n_target,
                       planned=plan.got, built=len(final),
                       inherited=int((final.origin == 2).sum()),
                       from_grid=int((final.origin == 0).sum()),
                       from_jitter=int((final.origin == 1).sum()),
                       **{f'n_{b}': int((final.bucket == i).sum())
                          for i, b in enumerate(BUCKETS)},
                       **{k: stats[k] for k in
                          ('rejected', 'replaced', 'unreplaced', 'valid_min')},
                       store=Path(path).name, sampler_id=cfg.sampler_id())
            rows.append(row)
        return unusable
    finally:
        print(f'  slide holes: {slide.hole_summary()}', flush=True)
        slide.close()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('wsi', nargs='+', help='WSI paths')
    # Absolute, off _paths.RESULT_DIR. A repo-relative default here resolved
    # against the caller's cwd and so wrote INSIDE the checkout -- the one thing
    # the result/ move exists to prevent. RefStore.sh passes no --out, so this
    # default is what it used.
    # default=None rather than the path itself, so that "the user named a
    # directory" stays distinguishable from "we chose one". Only the second
    # gets the encoder level appended.
    ap.add_argument('--out', default=None,
                    help='store root, used verbatim. Default '
                         'result/cache/features/<encoder>/ -- the same cache '
                         'the pooling stores use. They coexist because '
                         'sampler_id and mask_id are both in cfg_hash, so the '
                         'filenames differ')
    ap.add_argument('--report-dir', default=None,
                    help='where the per-level CSV and the pre-flight text go, '
                         'used verbatim; default '
                         'result/<SLURM_JOB_NAME>/<encoder>/')
    ap.add_argument('--levels', type=int, nargs='*', default=None,
                    help='levels to build; default every level')
    ap.add_argument('--dry-run', action='store_true',
                    help='mask, geometry and quotas only -- reads no tile')
    ap.add_argument('--pooling', default='cls',
                    help="'cls' keeps one vector per tile (~61 MB/slide); "
                         "'tokens' keeps all 197 (~6 GB/slide)")
    ap.add_argument(
        '--encoder', default='gigapath', choices=encoder_names(),
        help='which tile encoder. Only the module for THIS one is imported: '
             'every implementation sets HF_HOME above its own timm import and '
             'setdefault is first-one-wins, so importing all three would point '
             'two of them at the wrong weight cache -- silently. See '
             'TileEncoderFunc._IMPLEMENTATIONS. A store carries encoder_id in '
             'its identity, so one built here can only be read back by a run '
             'using the same encoder; that is a refusal, not a wrong answer.')
    ap.add_argument(
        '--head', default='',
        help="which exit of the model, empty for its own default. Only CONCH "
             "has two, and it needs --head trunk here: this writes pooled "
             "features, and pooling needs a token axis that CONCH's default "
             "attentional pooler does not have -- it hands back ONE 512-d "
             "vector. trunk is the bare ViT, the same shape GigaPath and UNI2 "
             "have. The head reaches identity_id, so the two never compare "
             "equal.")

    ap.add_argument('--n-target', type=int, default=1000)
    ap.add_argument('--tile', type=int, default=256)
    ap.add_argument('--over', type=float, default=1.25)
    ap.add_argument('--jitter-cap', type=float, default=0.20)
    ap.add_argument('--inherit-frac', type=float, default=0.50)
    ap.add_argument('--min-valid', type=float, default=0.95)
    ap.add_argument('--max-miss', type=int, default=20)
    ap.add_argument('--min-useful', type=int, default=200,
                    help='a level delivering fewer than this is reported '
                         'UNUSABLE and the exit code is non-zero. Falling '
                         'SHORT of the target is not a failure')
    ap.add_argument('--seed', type=int, default=42)

    ap.add_argument('--mask-level', type=int, default=1,
                    help='WSI level the tissue mask is segmented on')
    # Named for the resource it bounds, not for its unit. The old name was
    # --mask-max-pixels, one of four things in this repo called "max pixels"
    # that limit different resources at different stages, and it cost a run: the
    # default here was 6e8 while LocaScopePipeline.py already used 4_000_000 for
    # the same knob, and the mismatch asked a ResNet for a 110 GiB allocation.
    ap.add_argument('--seg-chunk-px', type=float, default=4_000_000,
                    help='pixels per segmentation forward pass -- bounds VRAM. '
                         'Splitting changes nothing about the mask that comes '
                         'out, only how many passes it takes')
    ap.add_argument('--read-chunk-px', type=float, default=0,
                    help='pixels per slide read -- bounds host RAM. 0 reads the '
                         'whole mask level at once, which is fine at ds=4 and '
                         'not at ds=1. When both are set the grid collapses to '
                         'min(), so setting this also caps the segmentation')
    ap.add_argument('--min-region-ratio', type=float, default=0.01)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    cfg = SamplerConfig(tile=args.tile, n_target=args.n_target, over=args.over,
                        jitter_cap=args.jitter_cap,
                        inherit_frac=args.inherit_frac,
                        min_valid=args.min_valid, max_miss=args.max_miss,
                        min_useful=args.min_useful,
                        seed=args.seed)

    # The default store root gets the tag but NOT a job name: it is a cache
    # shared across jobs on purpose, and a store's whole value is being reusable
    # by the next run. The readers glob one directory non-recursively
    # (FeatureStore.py:492), so this is also the level they must be pointed at.
    # An explicit --out is used verbatim, tag included or not, as the caller
    # wrote it.
    enc_tag = encoder_tag(args.encoder, args.head)
    if args.out:
        out_root = (args.out if os.path.isabs(args.out)
                    else str(_ROOT / args.out))
    else:
        out_root = os.path.join(_paths.RESULT_DIR, 'cache', 'features', enc_tag)
    os.makedirs(out_root, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    hest_method = HestSegConfig().build(device)
    mask_id = f'hest@L{args.mask_level}'

    # None, not a placeholder string. --dry-run builds no model, so there is no
    # identity_id to derive, and nothing on that path reads this: _build_slide
    # returns at its own `if args.dry_run` before StoreMeta is ever constructed,
    # and the preflight report is rendered from cfg and geometry alone.
    #
    # It used to be 'prov-gigapath@fp32tokens', which was dead either way -- the
    # real one overwrites it below -- and would have become a lie the moment
    # --encoder took a second value. A plausible-looking dead string is the
    # dangerous kind: the next person to read it on this path gets an answer
    # instead of an error. None makes that use fail where it is written.
    encoder = spec = encoder_id = None
    if not args.dry_run:
        # fp32: what the free function defaulted to (GigaPathFunc_old) when
        # this wrote its
        # existing stores, and changing it would silently orphan them. It stays
        # fixed across --encoder for the same reason -- the precision is part of
        # identity_id, so letting it follow the encoder would move gigapath's
        # own name too.
        over = {'head': args.head} if args.head else {}
        encoder = encoder_config(args.encoder, batch_size=args.batch_size,
                                 **over)\
            .with_model(dtype='fp32').build(device)
        # The model_spec itself: StoreMeta and pooling_kinds read dim / feat_hw /
        # num_prefix off it by name, so there is nothing to convert.
        spec = encoder.model_spec
        # Derived, not typed: the old literal could not notice a changed
        # checkpoint, a changed precision or a changed transform.
        encoder_id = encoder.identity_id()

    print(f'out       {out_root}')
    print(f'sampler   {cfg.sampler_id()}   seed {cfg.seed}   '
          f'target {cfg.n_target}/level   pooling {args.pooling}')
    print(f'mask      {mask_id}\n')

    failures, thin, rows, reports = [], [], [], []
    for path in args.wsi:
        print(f'== {Path(path).stem}', flush=True)
        try:
            bad = build_slide(path, args, cfg, encoder, spec, device, hest_method,
                              encoder_id, mask_id, out_root, rows, reports)
            if bad:
                thin.append((Path(path).stem, bad))
        except Exception as e:                              # noqa: BLE001
            failures.append((Path(path).stem, f'{type(e).__name__}: {e}'))
            print(f'  FAILED: {type(e).__name__}: {e}', flush=True)
            traceback.print_exc()

    if failures:
        print(f'\n{len(failures)} slide(s) failed:')
        for stem, msg in failures:
            print(f'  {stem}: {msg}')
    if thin:
        print(f'\nlevels below min_useful={cfg.min_useful}, skipped:')
        for stem, lv in thin:
            print(f'  {stem}: L{lv}')

    # Under result/<job>, per the standing rule that analysis outputs go there
    # and only SLURM's stdout goes to log/. The pre-flight is the answer to
    # "why did this level only get 296 tiles", and a log is overwritten by the
    # next job with the same name.
    report_dir = args.report_dir or job_result_dir('RefStore', encoder=enc_tag)
    os.makedirs(report_dir, exist_ok=True)
    if reports:
        txt = os.path.join(report_dir, 'refstore_preflight.txt')
        with open(txt, 'w') as f:
            f.write('\n\n'.join(reports) + '\n')
        print(f'\n  {os.path.basename(txt)} -> {txt}')
    if rows:
        keys = sorted({k for r in rows for k in r})
        keys = (['wsi_stem', 'level', 'verdict'] +
                [k for k in keys if k not in ('wsi_stem', 'level', 'verdict')])
        csv_path = os.path.join(report_dir, 'refstore_levels.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f'  {os.path.basename(csv_path)}  {len(rows)} rows -> {csv_path}')

    # A thin level is NOT a failure: it is skipped, the other levels are built,
    # and refstore_levels.csv records which and why. Only an exception is.
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
