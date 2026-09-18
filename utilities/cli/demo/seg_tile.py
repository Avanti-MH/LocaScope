#!/usr/bin/env python3
"""Run one or more tissue segmenters on one or more TILE images and save one
combined quick-look figure PER TILE -- no MaskStore write, and no whole-slide
read except for `--method uni2`, which needs one to FIT its PCA basis first
(see below).

Neither existing tool answers "what would these methods say about this one
tile": `utilities/cli/build_cache/build_mask_store.py` always fits + segments
a WHOLE slide and always writes the result to `MaskStore`; `utilities/cli/
diagnostics/inspect_pca_seg.py` is uni2-pca-seg only, and whole-slide too.
This is the direct, no-cache, any-combination-of-the-four-methods way to look
at one or more tiles.

    python utilities/cli/demo/seg_tile.py --image tile.png --method hsv
    python utilities/cli/demo/seg_tile.py --image tile.png --method hest
    python utilities/cli/demo/seg_tile.py --image a.png b.png \\
        --method uni2 hest hsv otsu

Outputs (in result/<SLURM_JOB_NAME or SegTile>/ unless --out):
    seg_<method1>-<method2>-...__<tile stem>.png      one per --image file,
                                                       one panel per method

A SWEEP IS ONE JOB, NOT ONE PER METHOD
=========================================
`--method` takes one or more values. Each `--image` file gets ONE figure --
the tile plus one overlay panel per requested method side by side -- so
comparing all four is a single job instead of four separate ones with four
separate figures to flip between.

--METHOD UNI2 ACROSS TILES FROM DIFFERENT SLIDES
===================================================
uni2 is the only method that needs a slide-wide PCA basis FIRST
(`Uni2PcaSegmenter.fit`) -- fitting on the single tile itself would give it
its OWN basis, exactly the failure `Uni2PcaSegmenter.fit`'s own docstring
documents for a per-tile PCA (an all-tissue tile lands its threshold inside
tissue). So each tile needs to know which slide it came from.

That slide name is recovered automatically from the tile's own folder, IF it
follows the `{slide}__ds{ds}__t{tile}__{hash}` convention
(`training/FewShotEoMT/Dataset.py`'s/`cli/infer.py`'s own tile cache layout)
-- so a batch of tiles from several different slides fits each slide's basis
once (cached and reused across every tile from that slide) without the
caller having to say which tile belongs to which slide. `--wsi-name` is a
FALLBACK for tiles whose folder does not encode this -- used only when a
tile's own slide cannot be recovered. If a uni2 fit cannot be resolved for a
given tile (no folder match and no --wsi-name), that tile's uni2 panel is
skipped with a printed note rather than failing the whole run.

METHOD CHOICES -- TissueSegFunc.py's own table has the full contract
=======================================================================
    'hsv'    saturation/value thresholds. No model, no fit, per-pixel.
    'otsu'   Otsu on grayscale, excluding near-black. No model, no fit.
             Not tiling-safe IN GENERAL (the threshold depends on the
             histogram of whatever it is shown) -- but a single tile here
             already IS the whole plane it gets thresholded against, so
             that caveat is about stitching multiple tiles, not about
             this tool.
    'hest'   DeepLabV3 + ResNet-50 (MahmoodLab/hest-tissue-seg). No fit --
             fully convolutional, any tile size, one forward pass.
    'uni2'   Uni2PcaSegFunc's PCA-on-UNI2-features method -- see above.
             Costs 3.5-6 minutes of GPU per slide fit
             (`Uni2PcaSegFunc.LEVEL`'s own measurement), once per DISTINCT
             slide in the batch, not once per tile.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, '..', '..'),
          os.path.join(_HERE, '..', '..', '..', 'aiNNModel')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                               # noqa: E402
import matplotlib                                                 # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                  # noqa: E402
import torch                                                      # noqa: E402
from PIL import Image                                             # noqa: E402

from _paths import job_result_dir, setup_import_paths            # noqa: E402

setup_import_paths()

from AccessDatasets import locate                                 # noqa: E402
from SafeSlide import SafeSlide                                   # noqa: E402
from TissueSegFunc import TissueSegConfig                         # noqa: E402
from HestSegFunc import HestSegConfig                             # noqa: E402
from Uni2PcaSegFunc import Uni2PcaSegConfig                       # noqa: E402


#: `{slide}__ds{ds}__t{tile}__{hash}` -- the tile cache folder-naming
#: convention (`training/FewShotEoMT/Dataset.py`'s/`cli/infer.py`'s
#: `_TILE_FOLDER_RE`, duplicated here rather than imported: that file lives
#: under `training/FewShotEoMT/`, a different package, and this is a two-line
#: regex, not a shared abstraction worth a cross-package import for.
_TILE_FOLDER_RE = re.compile(
    r'^(?P<slide>.+)__ds(?P<ds>[\d.]+)__t(?P<tile>\d+)__[0-9a-fA-F]+$')


def _recover_slide(image_path: str) -> Optional[str]:
    """The slide name encoded in the tile's own folder, or None.

    Lets `--method uni2` fit the RIGHT slide per tile even when one batch
    mixes tiles from several slides, without the caller having to say which
    tile belongs to which slide.
    """
    match = _TILE_FOLDER_RE.match(os.path.basename(os.path.dirname(image_path)))
    return match.group('slide') if match else None


def _build_no_fit_segmenters(methods, device) -> dict:
    """hest/hsv/otsu -- built ONCE, independent of any slide or tile."""
    segs = {}
    if 'hest' in methods:
        segs['hest'] = HestSegConfig().build(device)
    for m in ('hsv', 'otsu'):
        if m in methods:
            segs[m] = TissueSegConfig(method=m).build()
    return segs


def _fit_uni2(slide_name: str, args, device):
    cfg = Uni2PcaSegConfig(fit_tiles=args.fit_tiles,
                           components=args.components,
                           background_threshold=args.background_threshold,
                           larger_pca_as_fg=args.larger_pca_as_fg,
                           workers=args.workers)
    seg = cfg.build(device)
    entry = locate(slide_name)
    with SafeSlide(entry.path, warn=False) as wsi:
        print(f'  fitting uni2-pca-seg on {args.fit_tiles} tiles of '
             f'{slide_name}...', flush=True)
        seg.fit(wsi)
    report = seg.fit_report
    print(f'  fit done ({slide_name}): {report["cells"]} cells, explained '
         f'{report["explained_variance_top3"]:.1%}, foreground in sample '
         f'{report["foreground_fraction_in_sample"]:.1%}', flush=True)
    return seg


def _overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """GREEN = selected, matching `inspect_pca_seg.py`'s own convention -- a
    grey mask reads backwards the moment the selected region is the
    background (2026-08-26), an overlay on the actual tile cannot."""
    over = image.copy()
    selected = mask.astype(bool)
    over[selected] = (over[selected] * 0.55
                      + np.array([0, 200, 80]) * 0.45).astype(np.uint8)
    return over


def _plot_multi(image: np.ndarray, results: "Dict[str, Optional[np.ndarray]]",
                out_path: str) -> None:
    methods = list(results)
    fig, axes = plt.subplots(1, 1 + len(methods), figsize=(4 * (1 + len(methods)), 4))
    axes = np.atleast_1d(axes)
    axes[0].imshow(image)
    axes[0].set_title('tile', fontsize=10)
    for ax, method in zip(axes[1:], methods):
        mask = results[method]
        if mask is None:
            ax.imshow(np.zeros_like(image))
            ax.set_title(f'{method}\n(skipped -- no slide)', fontsize=10)
        else:
            ax.imshow(_overlay(image, mask))
            ax.set_title(f'{method}   frac {mask.mean():.1%}', fontsize=10)
    for ax in axes:
        ax.axis('off')
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--image', nargs='+', required=True,
                    help='one or more tile image FILEs')
    ap.add_argument('--method', nargs='+', required=True,
                    choices=('uni2', 'hest', 'hsv', 'otsu'),
                    help='one or more methods -- each --image gets ONE '
                         'combined figure with a panel per method, so a '
                         'sweep is one job instead of one per method')

    ap.add_argument('--wsi-name', default=None,
                    help='FALLBACK slide for --method uni2, used only when '
                         "a tile's own folder name does not encode one -- "
                         'see _recover_slide / the module docstring')
    ap.add_argument('--fit-tiles', type=int, default=1000)
    ap.add_argument('--components', type=int, default=16)
    ap.add_argument('--background-threshold', type=float, default=0.5)
    ap.add_argument('--larger-pca-as-fg', action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument('--workers', type=int, default=8)

    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default=None,
                    help='output directory. Empty means '
                         'result/<SLURM_JOB_NAME or SegTile>/')
    args = ap.parse_args()

    methods = list(dict.fromkeys(args.method))       # de-dup, keep order
    out_dir = args.out or job_result_dir('SegTile')
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device)
    # hest/hsv/otsu: built ONCE, reused for every tile. uni2: fit lazily,
    # once PER DISTINCT SLIDE seen across the batch, cached and reused for
    # every other tile from that same slide.
    no_fit = _build_no_fit_segmenters(methods, device)
    uni2_cache: Dict[str, object] = {}

    for image_path in args.image:
        image = np.array(Image.open(image_path).convert('RGB'))
        stem = os.path.splitext(os.path.basename(image_path))[0]
        print(f'-- {image_path}', flush=True)

        results: Dict[str, Optional[np.ndarray]] = {}
        for method in methods:
            if method == 'uni2':
                slide = _recover_slide(image_path) or args.wsi_name
                if slide is None:
                    print(f'  no slide recoverable from the folder name and '
                         f'no --wsi-name given -- skipping uni2 for this tile',
                         flush=True)
                    results['uni2'] = None
                    continue
                if slide not in uni2_cache:
                    uni2_cache[slide] = _fit_uni2(slide, args, device)
                seg = uni2_cache[slide]
            else:
                seg = no_fit[method]
            results[method] = seg(image).astype(bool)

        out_path = os.path.join(out_dir, f"seg_{'-'.join(methods)}__{stem}.png")
        _plot_multi(image, results, out_path)
        for method, mask in results.items():
            if mask is not None:
                print(f'  {method:<5} tissue frac {mask.mean():.1%}', flush=True)
        print(f'  -> {out_path}', flush=True)

    return 0


if __name__ == '__main__':
    sys.exit(main())
