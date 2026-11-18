#!/usr/bin/env python3
"""Pre-warm the mask cache: segment every named slide once, so a later job that
samples from it never waits on the GPU for a mask.

    python -m locascope.cli.build_cache.build_mask_store --seg uni2_pca <wsi-path>...
    python -m locascope.cli.build_cache.build_mask_store --seg hest --wsi-names Ki67_pure_0001 ...
    python -m locascope.cli.build_cache.build_mask_store --seg hest --mask-ds 2 --all      # every registered slide, HEST at ds 2
    python -m locascope.cli.build_cache.build_mask_store --seg hest --datasets @pure-train # a dataset id, registered or virtual

Outputs:
    result/cache/<cache-job>/slide=<slide>/mask/mask_<seg_id>.safetensors + record_<seg_id>.json
    build_mask_store.csv          in result/<SLURM_JOB_NAME or BuildMaskStore>/

argparse, a loop, and printed progress. Everything that decides anything is
`TissueMaskConfig.MaskMaker.slide_mask`: the recipe names the directory, a hit
is read back, a miss is segmented and written atomically. This is the same call
a sampler makes on a miss, so a mask written here IS the mask a job pointed at
`--mask-cache-job <cache-job>` reads -- there is no second format to keep in
step with the first.

WHY PRE-WARM AT ALL
-------------------
UNI2-PCA costs 3.5 to 6 minutes of GPU per slide (measured, `Uni2PcaSeg.
LEVEL`) and HEST is seconds to minutes. A training job that misses on its first
epoch spends that inside its own walltime, on the card it reserved for the
encoder. Running this first moves the cost to a job that does nothing else.

WHAT THE CSV IS FOR
-------------------
One row per slide: the tissue fraction and, for UNI2-PCA, the explained variance
and the foreground fraction the fit saw. None of it is asserted -- there is no
tissue ground truth here -- but the fractions are readable against what this
project has already measured, and a slide that comes out at 0.5 on a Ki67 is the
signal that PC1 found position or scanner banding rather than tissue.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
from locascope.paths import job_result_dir   # noqa: E402


import torch                                                    # noqa: E402

from locascope.core.slides.AccessDatasets import dataset_ids, list_names, locate  # noqa: E402
from locascope.core.cache.Cache import job_name, wsi_stem_of                            # noqa: E402
from locascope.core.slides.SafeSlide import SafeSlide                                  # noqa: E402
from locascope.core.backbones.TissueSeg import PlaneSegConfig                         # noqa: E402
from locascope.core.tiles.TissueMaskConfig import MASK_RECIPES, MaskMaker             # noqa: E402
from locascope.core.backbones.Uni2PcaSeg import Uni2PcaSegConfig                     # noqa: E402

#: The measured tissue fractions on this project's own slides, for reading the
#: printed number against. BRACS from test_EoMT's stratified_positions docstring
#: and the SlideWinTest log; Ki67 from the same docstring.
_REFERENCE = 'BRACS 20.8-38.2%, Ki67 3.5-9.2%'


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('wsi', nargs='*', default=[],
                    help='one or more slide PATHS')
    ap.add_argument('--wsi-names', nargs='*', default=[],
                    help='one or more locascope/core/slides/AccessDatasets.py names '
                        '(e.g. Ki67_pure_0001), resolved to paths -- an '
                        'alternative to the positional paths, not a '
                        'replacement; the two lists are concatenated')
    ap.add_argument('--seg', choices=sorted(MASK_RECIPES), default='uni2_pca',
                    help='mask recipe (TissueMaskConfig.MASK_RECIPES)')
    ap.add_argument('--datasets', nargs='*', default=[],
                    help="every slide of these AccessDatasets ids -- a registered "
                         "dataset or a virtual one ('@pure-train') -- added to the "
                         "slides above")
    ap.add_argument('--all', action='store_true',
                    help='every slide of every registered dataset')
    ap.add_argument('--part', type=int, default=None,
                    help='with --parts N: only the slides whose position in the list is '
                         'this number mod N (an array job runs one part each)')
    ap.add_argument('--parts', type=int, default=1)
    ap.add_argument('--mask-ds', type=float, default=None,
                    help="override the recipe's segmentation ds (plane segmenters: "
                         "hsv / hest / none). A different ds is a different seg_id, so "
                         "it is stored beside the recipe's own mask, not over it")
    ap.add_argument('--hest-dtype', choices=('fp32', 'fp16'), default=None,
                    help="hest only: the network's precision. fp16 is a different "
                         "seg_id (identity), so its masks sit beside the fp32 ones")
    ap.add_argument('--seg-batch', type=int, default=None,
                    help='tiles of one shape per forward pass (speed only, same seg_id)')
    ap.add_argument('--read-workers', type=int, default=None,
                    help='threads reading the next tiles during the forward pass (speed only)')
    ap.add_argument('--cache-job', default=None,
                    help='whose cache the masks go to: result/cache/<made_by>/. '
                         'Default: this job (SLURM_JOB_NAME or BuildMaskStore)')
    ap.add_argument('--fit-tiles', type=int, default=None,
                    help='uni2_pca only: tiles the PCA is fitted on. Default: '
                         "the recipe's own. A different value is a different "
                         'seg_id and so a different directory, which is correct')
    ap.add_argument('--workers', type=int, default=None,
                    help='uni2_pca only: DataLoader workers for reading tiles')
    ap.add_argument('--components', type=int, default=None)
    ap.add_argument('--background-threshold', type=float, default=None)
    ap.add_argument('--larger-pca-as-fg', action=argparse.BooleanOptionalAction,
                    default=None,
                    help='which side of PC1 is tissue. Decided by '
                         'locascope/cli/diagnostics/inspect_pca_seg.py and now the config'
                         "'s own default, so this flag is here to override it "
                         'rather than to repeat it. default=None and not False: '
                         'a CLI default that restates a config default is a '
                         'second place for the answer to live, and the two drift')
    ap.add_argument('--read-level', type=int, default=None,
                    help='uni2_pca recipes: the pyramid level UNI2 reads at (Uni2PcaSegConfig.read_level; identity)')
    ap.add_argument('--skip-blank-sd', type=float, default=None,
                    help='uni2_pca recipes: tiles of flat glass all around (grey sd below this at ds 8) are background '
                         'without being encoded (Uni2PcaSegConfig.skip_blank_sd; identity; -1 = off)')
    ap.add_argument('--fit-tissue-cache', default=None,
                    help="uni2_pca recipes: whose cache the fit's HEST ds 4 mask is read from / written to; '' = none, made "
                         'afresh and kept nowhere (Uni2PcaSegConfig.fit_tissue_cache; not identity)')
    ap.add_argument('--overwrite', action='store_true',
                    help="delete the slide's cached mask first. Draws already "
                         'made from the old mask live under the seg= level of '
                         'each job\'s tree and are NOT touched '
                         '(purge_cache.py --level seg=<seg_id>): '
                         'delete those too, or they outlive the mask they came '
                         'from')
    ap.add_argument('--out', default=None,
                    help='directory for the summary CSV. Empty means '
                         'result/<SLURM_JOB_NAME or BuildMaskStore>/')
    args = ap.parse_args()

    # (dataset, name) pairs: a name found in two datasets resolves by the one it was listed under
    pairs = [(None, name) for name in args.wsi_names]
    for dataset in (dataset_ids() if args.all else args.datasets):
        pairs += [(dataset, name) for name in list_names(dataset=dataset)]
    paths = list(args.wsi)
    for dataset, name in dict.fromkeys(pairs):
        path = locate(name, dataset=dataset).path
        if path not in paths:
            paths.append(path)
    if not paths:
        ap.error('give at least one slide: paths, --wsi-names, --datasets or --all')
    if args.part is not None:
        if not 0 <= args.part < args.parts:
            ap.error(f'--part {args.part} is outside 0..{args.parts - 1}')
        paths = paths[args.part::args.parts]

    mask_cfg = MASK_RECIPES[args.seg]
    overrides = {k: v for k, v in (
        ('fit_tiles', args.fit_tiles), ('workers', args.workers),
        ('components', args.components),
        ('background_threshold', args.background_threshold),
        ('larger_pca_as_fg', args.larger_pca_as_fg),
        ('read_level', args.read_level),
        ('skip_blank_sd', args.skip_blank_sd),
        ('fit_tissue_cache', args.fit_tissue_cache)) if v is not None}
    if overrides:
        if not isinstance(mask_cfg.seg, Uni2PcaSegConfig):
            ap.error(f'{sorted(overrides)} only apply to the uni2_pca recipes')
        mask_cfg = dataclasses.replace(
            mask_cfg, seg=dataclasses.replace(mask_cfg.seg, **overrides))

    if args.hest_dtype is not None:
        if args.seg != 'hest':
            ap.error('--hest-dtype applies to --seg hest only')
        mask_cfg = dataclasses.replace(mask_cfg, seg=dataclasses.replace(
            mask_cfg.seg, model=dataclasses.replace(mask_cfg.seg.model, dtype=args.hest_dtype)))

    if args.mask_ds is not None:
        if not isinstance(mask_cfg.seg, PlaneSegConfig):
            ap.error(f'--mask-ds applies to the plane segmenters (none / hsv / hest), not {args.seg}')
        mask_cfg = dataclasses.replace(
            mask_cfg, seg=dataclasses.replace(mask_cfg.seg, ds=args.mask_ds))

    made_by = args.cache_job or job_name('BuildMaskStore')
    out_dir = args.out or job_result_dir('BuildMaskStore')
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'recipe    {args.seg}   seg_id {mask_cfg.seg_id()}   device {device}',
          flush=True)
    print(f'cache     {made_by}', flush=True)

    masks = MaskMaker(mask_cfg, made_by, device)
    if args.seg_batch is not None or args.read_workers is not None:
        # speed knobs live on the built segmenter, not in the config: building it
        # here costs the model load even when every slide turns out to be cached
        seg = masks.segmenter
        if args.seg_batch is not None:
            seg.batch_tiles = args.seg_batch
        if args.read_workers is not None:
            seg.read_workers = args.read_workers
        print(f'speed     batch {getattr(seg, "batch_tiles", 1)} tiles, '
              f'{getattr(seg, "read_workers", 0)} read workers', flush=True)
    rows, failures = [], []
    for index, wsi_path in enumerate(paths, 1):
        stem = wsi_stem_of(wsi_path)
        print(f'\n[{index}/{len(paths)}] {stem}', flush=True)
        if args.overwrite:
            masks.entry(stem).drop(mask_cfg.seg_id())
        t_slide = time.perf_counter()
        try:
            with SafeSlide(wsi_path) as wsi:
                slide_mask, hit = masks.slide_mask(wsi)
        except Exception as e:                                   # noqa: BLE001
            # One unreadable slide must not lose the ones already done. The
            # cache is written per slide, so what is on disk stays valid.
            print(f'    FAILED  {type(e).__name__}: {e}', flush=True)
            failures.append((stem, f'{type(e).__name__}: {e}'))
            continue

        geo, report = slide_mask.geometry(), slide_mask.report or {}
        print(f'    {"have it" if hit else "segmented"}   {geo["rows"]} x '
              f'{geo["cols"]} cells at ds {geo["mask_ds"]:.0f}   tissue '
              f'{geo["fraction"]:.1%}   ({_REFERENCE})', flush=True)
        seconds = time.perf_counter() - t_slide
        print(f'    time {seconds:.1f} s for this slide (open + segment + write; a cache hit is the read back)', flush=True)
        if report:
            print(f'    fit: {report.get("cells", "?")} cells, explained '
                  f'{report.get("explained_variance_top3", 0):.1%}, foreground '
                  f'in sample {report.get("foreground_fraction_in_sample", 0):.1%}',
                  flush=True)
        rows.append({'wsi_stem': stem, 'seg': args.seg,
                     'seconds': round(seconds, 1),
                     'seg_id': mask_cfg.seg_id(),
                     'entry': str(masks.entry(stem).dir),
                     **geo,
                     'fit_cells': report.get('cells', ''),
                     'explained_top3': report.get('explained_variance_top3', ''),
                     'fit_foreground': report.get('foreground_fraction_in_sample', ''),
                     'reused': int(hit)})
    masks.close()

    summary = os.path.join(out_dir, 'build_mask_store.csv')
    if rows:
        with open(summary, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f'\nSaved {summary}   ({len(rows)} slides)')

    if failures:
        print(f'\n{len(failures)} slide(s) failed:')
        for stem, why in failures:
            print(f'  {stem}: {why}')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
