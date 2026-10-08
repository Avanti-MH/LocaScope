#!/usr/bin/env python3
"""Pre-warm the mask cache: segment every named slide once, so a later job that
samples from it never waits on the GPU for a mask.

    python utilities/cli/build_cache/build_mask_store.py --seg uni2_pca <wsi-path>...
    python utilities/cli/build_cache/build_mask_store.py --seg hest --wsi-names Ki67_pure_0001 ...

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
UNI2-PCA costs 3.5 to 6 minutes of GPU per slide (measured, `Uni2PcaSegFunc.
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

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, '..', '..'),
          os.path.join(_HERE, '..', '..', '..', 'aiNNModel')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _paths import job_result_dir, setup_import_paths            # noqa: E402

setup_import_paths()

import torch                                                    # noqa: E402

from AccessDatasets import locate                                 # noqa: E402
from Cache import job_name, wsi_stem_of                            # noqa: E402
from SafeSlide import SafeSlide                                  # noqa: E402
from TissueMaskConfig import MASK_RECIPES, MaskMaker             # noqa: E402

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
                    help='one or more utilities/AccessDatasets.py names '
                        '(e.g. Ki67_pure_0001), resolved to paths -- an '
                        'alternative to the positional paths, not a '
                        'replacement; the two lists are concatenated')
    ap.add_argument('--seg', choices=sorted(MASK_RECIPES), default='uni2_pca',
                    help='mask recipe (TissueMaskConfig.MASK_RECIPES)')
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
                         'utilities/cli/diagnostics/inspect_pca_seg.py and now the config'
                         "'s own default, so this flag is here to override it "
                         'rather than to repeat it. default=None and not False: '
                         'a CLI default that restates a config default is a '
                         'second place for the answer to live, and the two drift')
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

    paths = list(args.wsi) + [locate(name).path for name in args.wsi_names]
    if not paths:
        ap.error('give at least one slide, as a path or via --wsi-names')

    mask_cfg = MASK_RECIPES[args.seg]
    overrides = {k: v for k, v in (
        ('fit_tiles', args.fit_tiles), ('workers', args.workers),
        ('components', args.components),
        ('background_threshold', args.background_threshold),
        ('larger_pca_as_fg', args.larger_pca_as_fg)) if v is not None}
    if overrides:
        if args.seg != 'uni2_pca':
            ap.error(f'{sorted(overrides)} only apply to --seg uni2_pca')
        mask_cfg = dataclasses.replace(
            mask_cfg, seg=dataclasses.replace(mask_cfg.seg, **overrides))

    made_by = args.cache_job or job_name('BuildMaskStore')
    out_dir = args.out or job_result_dir('BuildMaskStore')
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'recipe    {args.seg}   seg_id {mask_cfg.seg_id()}   device {device}',
          flush=True)
    print(f'cache     {made_by}', flush=True)

    masks = MaskMaker(mask_cfg, made_by, device)
    rows, failures = [], []
    for index, wsi_path in enumerate(paths, 1):
        stem = wsi_stem_of(wsi_path)
        print(f'\n[{index}/{len(paths)}] {stem}', flush=True)
        if args.overwrite:
            masks.entry(stem).drop(mask_cfg.seg_id())
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
        if report:
            print(f'    fit: {report.get("cells", "?")} cells, explained '
                  f'{report.get("explained_variance_top3", 0):.1%}, foreground '
                  f'in sample {report.get("foreground_fraction_in_sample", 0):.1%}',
                  flush=True)
        rows.append({'wsi_stem': stem, 'seg': args.seg,
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
