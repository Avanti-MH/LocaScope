#!/usr/bin/env python3
"""Build a slide's reference set under a richness contract, and store it.

    python utilities/cli/build_cache/build_reference_store.py <wsi>... [--dry-run]

The reference behind stage 1 is 40 tiles per level, drawn inside the tissue
mask and rebuilt in memory on every LocaScopePipeline.build(). That is enough
for a KNN to saturate (result/MppEstimate: 40 -> 640 buys 2.7 points) but it
leaves no record of how the tiles were chosen. This tool draws them with
TileSampler under the reference-bank richness contract (`KnnEstMpp.
REFERENCE_BANK_RICHNESS`), one native rung per pyramid level, drops the ones
the scanner never photographed, encodes what is left, and stores the
coordinates together with WHY each one is there -- its background score, its
bucket, whether it is a grid, jitter or inherited tile, and its chain.

    result/cache/<job>_features/<encoder>/<seg_id>/<slide>/<region_id>/<draw>/ds<d>_<pooling>.safetensors

`<draw>` is `<sampler_id>_<plan>` and is printed at the start; a reader names
it (`--draw`) to find these stores.

Two passes, and the first one is free
-------------------------------------
    --dry-run   mask and per-level supply (TileSampler.preflight). Reads no
                tile. Answers "will this level fall short, and of what" in the
                time it takes to segment the slide.

    full        the draw, the reads, the encode. A level the sampler fills with
                fewer than --min-useful tiles is skipped, not fatal: a deep
                level running out of positions is a fact about the slide.

Every drawn tile is read and kept: the sampler places tiles on the mask, and an
unscanned block is glass to the mask, so nothing is filtered after the read.

Cost note: --pooling tokens keeps all 197 tokens, roughly 605 KB per tile;
--pooling cls keeps one vector, about 61 MB per slide.
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
_ROOT = _HERE.parent.parent.parent
for _d in ('utilities', 'aiNNModel', ''):     # '' = the root: stage packages
    p = str(_ROOT / _d)
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402

import Cache                                                        # noqa: E402
from stage1_estimation.KnnEstMpp import REFERENCE_BANK_RICHNESS                       # noqa: E402
from SafeSlide import SafeSlide                                     # noqa: E402
from ReadGeometry import ReadSpec                                   # noqa: E402
from SlideReader import SlideReader                                 # noqa: E402
from Store import FeatureStore as FS                                # noqa: E402
from TileEncoderFunc import encoder_config, encoder_names           # noqa: E402
from TileSampler import (InheritConfig, OverlapConfig,              # noqa: E402
                         SamplerConfig, TileSampler, native_plans)
from TissueMaskConfig import MaskMaker, add_mask_args, mask_cfg_from_args  # noqa: E402
from _paths import encoder_tag, job_result_dir                      # noqa: E402

ORIGIN_CODE = {'grid': 0, 'jitter': 1, 'inherit': 2}


def plan_label(levels, tile: int) -> str:
    """The rung plan: every native level, or the ones asked for, for a plain
    `tile` px camera. Part of the draw's address, because an inherited chain
    spans exactly these rungs -- and the tile size is the camera's, not the
    sampler's, so it is named here or two tile sizes would share one."""
    from ReadGeometry import ReadSpec                             # noqa: PLC0415
    base = 'native' if levels is None else 'native-L' + '-'.join(map(str, sorted(levels)))
    return f'{base}-{ReadSpec(int(tile), int(tile)).key()}'


def build_slide(wsi_path, args, cfg, plan, encoder, spec, masks, out_root,
                rows, report_dir) -> list:
    """One slide. Appends a row per level to `rows`; returns the levels that
    were too thin to build."""
    stem = Cache.wsi_stem_of(wsi_path)
    slide = SafeSlide(str(wsi_path))
    reader = SlideReader(slide)
    thin = []
    try:
        t0 = time.time()
        mask, _ = masks.mask(slide)
        print(f'  mask: tissue={mask.tissue_fraction() * 100:.1f}%  '
              f'{len(mask.tissue_regions)} regions  ({time.time() - t0:.0f}s)',
              flush=True)
        if not mask.tissue_regions:
            print('  no region survived the filters -- skipped', flush=True)
            return thin

        rungs = [p for p in native_plans(slide, args.tile)
                 if args.levels is None or p.level in args.levels]
        sampler = TileSampler(slide, mask, cfg, slide=stem)
        if args.dry_run:
            for report in sampler.preflight(rungs):
                print(f'    ds {report.ds:g}: {report.n_admissible} admissible of '
                      f'{report.n_candidates} candidates   supply '
                      + '  '.join(f'{k}={v}' for k, v in report.supply.items()),
                      flush=True)
            return thin

        sampler.sample(rungs)
        sampler.write_report(report_dir)
        names = list(cfg.richness.names)
        base_mpp = slide.base_mpp
        for rung in rungs:
            metas = [s.meta for s in sampler if s.meta.ds == rung.rung_ds]
            t0 = time.time()
            kept = metas
            imgs = reader.read_samples(metas, ReadSpec(args.tile, args.tile))
            t_read = time.time() - t0
            row = dict(wsi_stem=stem, level=rung.level, ds=rung.rung_ds,
                       n_target=cfg.n_per_rung, kept=len(kept))
            if len(kept) < args.min_useful:
                print(f'    L{rung.level}: {len(kept)} tiles drawn, '
                      f'below --min-useful {args.min_useful} -- level skipped',
                      flush=True)
                rows.append(dict(row, verdict='UNUSABLE'))
                thin.append(rung.level)
                continue

            feats = encoder.pooled(imgs, args.pooling)
            fs = encoder.pooled_spec(feats, args.pooling)
            column = lambda f, dtype: torch.from_numpy(                   # noqa: E731
                np.array([f(m) for m in kept], dtype=dtype))
            meta = FS.Meta(
                wsi_stem=stem, wsi_path=str(wsi_path), level=rung.level,
                ds=float(rung.rung_ds), mpp=base_mpp * rung.rung_ds,
                base_mpp=base_mpp, tile_size=args.tile, overlap=False,
                pooling=args.pooling, slots=tuple(fs.slots),
                slot_layout=fs.slot_layout, dim=spec['dim'],
                feat_hw=tuple(spec['feat_hw']), num_prefix=spec['num_prefix'],
                encoder_id=encoder.identity_id(), seg_id=masks.cfg.seg_id(),
                region_id=masks.cfg.region_id(), coverage='sample',
                n_available=sampler.reports[rung.rung_ds].n_admissible,
                n_tiles=len(kept), sampler_id=cfg.identity_id(), plan=plan,
                sample_seed=cfg.seed, buckets=tuple(names))
            path = FS.save(
                out_root, meta=meta, features=feats.to(torch.float16),
                x=column(lambda m: m.x, np.int32),
                y=column(lambda m: m.y, np.int32),
                extra={'bucket': column(lambda m: names.index(m.bucket), np.int8),
                       'white_frac': column(lambda m: m.score, np.float32),
                       'origin': column(lambda m: ORIGIN_CODE[m.origin], np.int8),
                       'parent_x': column(lambda m: m.parent_x, np.int64),
                       'parent_y': column(lambda m: m.parent_y, np.int64),
                       'inherit_id': column(lambda m: m.inherit_id, np.int32)})
            origins = [m.origin for m in kept]
            print(f'    L{rung.level}: {len(kept)}/{cfg.n_per_rung} tiles   '
                  f'read {t_read:.0f}s  encode {time.time() - t0 - t_read:.0f}s   '
                  f'grid={origins.count("grid")} jitter={origins.count("jitter")} '
                  f'inherit={origins.count("inherit")}   -> {path}', flush=True)
            rows.append(dict(row, verdict='SHORT' if len(kept) < cfg.n_per_rung else 'OK',
                             inherited=origins.count('inherit'),
                             **{f'n_{b}': sum(1 for m in kept if m.bucket == b)
                                for b in names}))
        return thin
    finally:
        print(f'  slide holes: {slide.hole_summary()}', flush=True)
        slide.close()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('wsi', nargs='+', help='WSI paths')
    ap.add_argument('--out', default=None,
                    help='store root, used verbatim. Default result/cache/'
                         '<--features-cache-job>_features/<encoder>/')
    ap.add_argument('--features-cache-job', default=None,
                    help='the <made_by> of the feature cache. Default: this job')
    ap.add_argument('--report-dir', default=None,
                    help='where the per-level CSV and the sampler reports go; '
                         'default result/<SLURM_JOB_NAME or RefStore>/<encoder>/')
    ap.add_argument('--levels', type=int, nargs='*', default=None,
                    help='pyramid levels to build; default every level')
    ap.add_argument('--dry-run', action='store_true',
                    help='mask and per-level supply only -- reads no tile')
    ap.add_argument('--pooling', default='cls',
                    help="'cls' keeps one vector per tile (~61 MB/slide); "
                         "'tokens' keeps all 197 (~6 GB/slide)")
    ap.add_argument('--encoder', default='gigapath', choices=encoder_names(),
                    help='which tile encoder. Only the module for THIS one is '
                         'imported: every implementation sets HF_HOME above its '
                         'own timm import, first one wins')
    ap.add_argument('--head', default='',
                    help="which exit of the model; CONCH needs --head trunk "
                         'here, since pooling needs a token axis')
    ap.add_argument('--n-target', type=int, default=1000, help='tiles per level')
    ap.add_argument('--tile', type=int, default=256)
    ap.add_argument('--inherit-share', type=float, default=0.50,
                    help='share of each level carried by chains -- the same '
                         'level-0 centre at every level')
    ap.add_argument('--min-useful', type=int, default=200,
                    help='a level keeping fewer than this is skipped. Falling '
                         'SHORT of --n-target is not a failure')
    ap.add_argument('--seed', type=int, default=42)
    add_mask_args(ap)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    cfg = SamplerConfig(n_per_rung=args.n_target, seed=args.seed,
                        richness=REFERENCE_BANK_RICHNESS, overlap=OverlapConfig(),
                        inherit=InheritConfig(stack_kind='F', share=args.inherit_share))
    plan = plan_label(args.levels, args.tile)
    enc_tag = encoder_tag(args.encoder, args.head)
    out_root = Path(args.out) if args.out else (
        Cache.cache_root(args.features_cache_job or Cache.job_name('RefStore'),
                         'features') / enc_tag)
    report_dir = args.report_dir or job_result_dir('RefStore', encoder=enc_tag)
    os.makedirs(report_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    masks = MaskMaker(mask_cfg_from_args(args), device=device)
    encoder = spec = None
    if not args.dry_run:
        # fp32 and fixed across --encoder: the precision is part of
        # identity_id, so letting it follow the encoder would move its name.
        over = {'head': args.head} if args.head else {}
        encoder = encoder_config(args.encoder, batch_size=args.batch_size, **over)\
            .with_model(dtype='fp32').build(device)
        spec = encoder.model_spec

    print(f'out       {out_root}')
    print(f'draw      {FS.sample_key(cfg.identity_id(), plan)}   '
          f'(pass --draw {FS.sample_key(cfg.identity_id(), plan)} to a reader)')
    print(f'mask      {masks.cfg.seg_id()}/{masks.cfg.region_id()}   '
          f'target {cfg.n_per_rung}/level   pooling {args.pooling}\n')

    failures, thin, rows = [], [], []
    for path in args.wsi:
        print(f'== {Path(path).stem}', flush=True)
        try:
            bad = build_slide(path, args, cfg, plan, encoder, spec, masks,
                              out_root, rows, report_dir)
            if bad:
                thin.append((Path(path).stem, bad))
        except Exception as e:                              # noqa: BLE001
            failures.append((Path(path).stem, f'{type(e).__name__}: {e}'))
            print(f'  FAILED: {type(e).__name__}: {e}', flush=True)
            traceback.print_exc()
    masks.close()

    if failures:
        print(f'\n{len(failures)} slide(s) failed:')
        for stem, msg in failures:
            print(f'  {stem}: {msg}')
    if thin:
        print(f'\nlevels below --min-useful {args.min_useful}, skipped:')
        for stem, levels in thin:
            print(f'  {stem}: L{levels}')
    if rows:
        keys = sorted({k for r in rows for k in r})
        keys = ['wsi_stem', 'level', 'verdict'] + [
            k for k in keys if k not in ('wsi_stem', 'level', 'verdict')]
        csv_path = os.path.join(report_dir, 'refstore_levels.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f'  refstore_levels.csv  {len(rows)} rows -> {csv_path}')

    # A thin level is NOT a failure: it is skipped, the others are built, and
    # the CSV records which and why. Only an exception is.
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
