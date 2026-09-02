#!/usr/bin/env python3
"""Fill the mask store: one tissue mask per slide, written once and reused.

    python utilities/cli/build_mask_store.py <wsi>... [--fit-tiles 1000]

Outputs (in result/cache/masks/ by default):
    <wsi_stem>__<method>__<cfg8>.safetensors
    build_mask_store.csv          in result/<SLURM_JOB_NAME or BuildMaskStore>/

argparse, a loop, and printed progress. Everything that decides anything is in
`utilities/MaskStore.py` -- `build_one` runs the segmenter, `MaskMeta.of`
assembles the identity, `save` validates and writes atomically. The split is
`FeatureStore` and `cli/build_reference_store.py`'s, and CLAUDE.md's reason: a
library layer that prints cannot be called by a bench.

WHY THIS RUNS ONCE AND NOT THREE TIMES
---------------------------------------
The mask costs 3.5 to 6 minutes of GPU per slide (measured, `Uni2PcaSegFunc.
LEVEL`), and three later steps read it: the sampling probe of spec.md 12 step
3b, the pre-tile extraction of 3c, and any bench that wants the same regions the
training tiles came from. Recomputing is not the problem; three recomputations
that could quietly differ is.

The store is keyed on the segmenter's `identity_id()`, so re-running with a
changed config writes a SECOND file rather than overwriting the first, and a
reader that asks for one gets an error rather than the other.

WHAT THE CSV IS FOR
-------------------
One row per slide: the tissue fraction, the explained variance, and the
foreground fraction the fit saw. None of it is asserted -- there is no tissue
ground truth here -- but the fractions are readable against what this project
has already measured, and a slide that comes out at 0.5 on a Ki67 is the signal
that PC1 found position or scanner banding rather than tissue.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, '..'), os.path.join(_HERE, '..', '..', 'aiNNModel')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _paths import RESULT_DIR, job_result_dir, setup_import_paths  # noqa: E402

setup_import_paths()

import torch                                                    # noqa: E402

import MaskStore                                                # noqa: E402
from MaskStore import MaskMeta                                   # noqa: E402
from SafeSlide import SafeSlide                                  # noqa: E402
from Uni2PcaSegFunc import Uni2PcaSegConfig                      # noqa: E402


#: result/cache/ rather than result/<job>/, because several jobs read it and it
#: is not the byproduct of any one of them. `make clean-job JOB=cache` is then
#: the one obvious way to purge it (ClaudeRules section 6).
DEFAULT_ROOT = os.path.join(RESULT_DIR, 'cache', 'masks')

#: The measured tissue fractions on this project's own slides, for reading the
#: printed number against. BRACS from test_EoMT's stratified_positions docstring
#: and the SlideWinTest log; Ki67 from the same docstring.
_REFERENCE = 'BRACS 20.8-38.2%, Ki67 3.5-9.2%'


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('wsi', nargs='+', help='one or more slides')
    ap.add_argument('--root', default=DEFAULT_ROOT,
                    help='where the store lives (default: result/cache/masks/)')
    ap.add_argument('--fit-tiles', type=int, default=1000,
                    help="tiles the PCA is fitted on. The config's own default; "
                         'a smaller value is a different identity and so a '
                         'different file, which is correct')
    ap.add_argument('--workers', type=int, default=8,
                    help='DataLoader workers for reading tiles. The read is the '
                         'cost here, not the model')
    ap.add_argument('--components', type=int, default=16)
    ap.add_argument('--background-threshold', type=float, default=0.5)
    ap.add_argument('--larger-pca-as-fg', action=argparse.BooleanOptionalAction,
                    default=None,
                    help='which side of PC1 is tissue. Decided by '
                         'utilities/cli/inspect_pca_seg.py and now the config'
                         "'s own default, so this flag is here to override it "
                         'rather than to repeat it. default=None and not False: '
                         'a CLI default that restates a config default is a '
                         'second place for the answer to live, and the two drift')
    ap.add_argument('--overwrite', action='store_true',
                    help='rebuild even when a mask with this identity exists')
    ap.add_argument('--out', default=None,
                    help='directory for the summary CSV. Empty means '
                         'result/<SLURM_JOB_NAME or BuildMaskStore>/')
    args = ap.parse_args()

    out_dir = args.out or job_result_dir('BuildMaskStore')
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    overrides = {} if args.larger_pca_as_fg is None else {
        'larger_pca_as_fg': args.larger_pca_as_fg}
    cfg = Uni2PcaSegConfig(fit_tiles=args.fit_tiles,
                           components=args.components,
                           background_threshold=args.background_threshold,
                           workers=args.workers, **overrides)
    # Built ONCE and reused across slides. The encoder's weights are the
    # expensive part of construction and they do not depend on the slide; the
    # PCA basis does, and `mask_wsi` refits it per slide.
    segmenter = cfg.build(device)
    print(f'segmenter {segmenter.identity_id()}   method {cfg.method}   '
          f'device {device}', flush=True)
    print(f'store     {args.root}', flush=True)

    rows, failures = [], []
    for index, wsi_path in enumerate(args.wsi, 1):
        stem = MaskStore.wsi_stem_of(wsi_path)
        print(f'\n[{index}/{len(args.wsi)}] {stem}', flush=True)

        existing = MaskStore.find(args.root, wsi_stem=stem,
                                  segmenter_id=segmenter.identity_id())
        if existing and not args.overwrite:
            print(f'    have it: {existing[0].name}   (--overwrite to rebuild)',
                  flush=True)
            slide_mask, meta = MaskStore.load(existing[0])
            rows.append(_row(stem, existing[0], meta, reused=True))
            continue

        try:
            with SafeSlide(wsi_path) as wsi:
                slide_mask = MaskStore.build_one(wsi, segmenter)
                meta = MaskMeta.of(slide_mask, wsi, segmenter)
                path = MaskStore.save(args.root, slide_mask, meta)
        except Exception as e:                                   # noqa: BLE001
            # One unreadable slide must not lose the ones already done. The
            # store is written per slide, so what is on disk stays valid.
            print(f'    FAILED  {type(e).__name__}: {e}', flush=True)
            failures.append((stem, f'{type(e).__name__}: {e}'))
            continue

        report = slide_mask.report or {}
        print(f'    {meta.rows} x {meta.cols} cells at ds {meta.mask_ds:.0f}   '
              f'tissue {meta.fraction:.1%}   ({_REFERENCE})', flush=True)
        if report:
            print(f'    fit: {report.get("cells", "?")} cells, explained '
                  f'{report.get("explained_variance_top3", 0):.1%}, foreground '
                  f'in sample {report.get("foreground_fraction_in_sample", 0):.1%}',
                  flush=True)
        print(f'    wrote {path.name}', flush=True)
        rows.append(_row(stem, path, meta, reused=False))

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


def _row(stem, path, meta, *, reused):
    report = meta.report
    return {'wsi_stem': stem, 'file': os.path.basename(str(path)),
            'method': meta.method, 'segmenter_id': meta.segmenter_id,
            'mask_ds': meta.mask_ds, 'rows': meta.rows, 'cols': meta.cols,
            'origin_x': meta.origin_x, 'origin_y': meta.origin_y,
            'span_w': meta.span_w, 'span_h': meta.span_h,
            'tissue_fraction': meta.fraction,
            'n_components': meta.n_components,
            'fit_cells': report.get('cells', ''),
            'explained_top3': report.get('explained_variance_top3', ''),
            'fit_foreground': report.get('foreground_fraction_in_sample', ''),
            'reused': int(reused)}


if __name__ == '__main__':
    sys.exit(main())
