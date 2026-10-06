#!/usr/bin/env python3
"""Look at the masks the pre-tiles were actually cut from. No model, no re-fit.

    python utilities/cli/inspect_cache_store/inspect_mask_store.py
    python utilities/cli/inspect_cache_store/inspect_mask_store.py --with-thumb

One row per stored mask: the mask itself, the tissue regions drawn on it, and
optionally the slide thumbnail beside it at the same extent.

WHY THIS IS NOT `TestTissueMask.sh` AND NOT `InspectPcaSeg.sh`
---------------------------------------------------------------
Three tools, three different questions, and the difference is which mask each
of them is looking at:

    TestTissueMask.sh    runs test_tissue_mask.py, which SEGMENTS the
                         slide itself with Otsu, HSV or HEST. It never touches
                         Uni2PcaSegFunc and it never touches the store, so its
                         pictures are of a mask nothing downstream ever used
    InspectPcaSeg.sh     RE-FITS a PCA per slide to answer "which side of PC1 is
                         tissue". Minutes and a GPU, and the basis it fits is
                         not the basis the stored mask was cut with unless the
                         fit sample matched
    this                 reads the mask cache (`result/cache/<job>_mask/
                         <seg_id>/<slide>/mask.safetensors`) and draws it.
                         Seconds, no GPU. The mask here IS the one
                         `probe_tile_yield` gated on and `extract_pretiles` cut
                         17,784 pre-tiles from

So this is the one to look at when the question is "what did the sampler see".
The other two answer "would a different segmenter do better" and "is PC1 tissue
at all", and neither of those is a question about the corpus on disk.

WHAT TO READ
-------------
The printed foreground fraction against the measured tissue -- BRACS 23-38 per
cent, Ki67 3.5-9. A Ki67 slide reading near 0.5 means PC1 found scanner banding
or position rather than tissue, and every tile sampled from it is suspect. That
number is also in the sidecar's `fraction`, so this prints it beside the one it
recomputes from the array: they are written by different lines and agreeing is
evidence the file is the file it says it is.
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))          # utilities/

import numpy as np                                              # noqa: E402
import matplotlib                                               # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                 # noqa: E402
import matplotlib.patches as mpatches                           # noqa: E402

from _paths import job_result_dir, setup_import_paths          # noqa: E402

setup_import_paths()

from Cache import cache_root, find, read_meta                    # noqa: E402
from TissueMaskConfig import MASK_RECIPES                        # noqa: E402
from TissueMask import SlideMask, TissueMask     # noqa: E402


def draw_regions(axis, trm, linewidth: float = 0.8) -> None:
    """Region boxes in MASK coordinates, which is what the axis is showing.

    `region_box` is what converts, and it is the function to use rather than
    dividing by `mask_ds` here: the regions are stored in ABSOLUTE level-0
    coordinates and a mask with an origin -- every MIRAX -- needs the origin
    subtracted as well as the scale divided. Doing that by hand at a call site
    is how a Ki67 slide gets boxes that are all shifted by the same few hundred
    pixels and still look plausible.
    """
    for region in trm.tissue_regions:
        x, y, w, h = trm.region_box(region)
        axis.add_patch(mpatches.Rectangle((x, y), w, h, fill=False,
                                          edgecolor='#d93025',
                                          linewidth=linewidth))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seg', choices=sorted(MASK_RECIPES), default='uni2_pca',
                    help='which recipe\'s masks to draw')
    ap.add_argument('--mask-cache-job', default='BuildMaskStore',
                    help='the job that made the cache: result/cache/'
                         '<this>_mask/')
    ap.add_argument('--wsi-stem', nargs='*', default=None)
    ap.add_argument('--min-ratio', type=float, default=0.01,
                    help="`filtered` cutoff. 0.01 is the recipes' own, so the "
                         'default panel is what the sampler saw')
    ap.add_argument('--ops', action=argparse.BooleanOptionalAction, default=True,
                    help='draw filtered / merged / patchable each on its own, '
                         'and then the three in sequence -- the recipe\'s two '
                         'and then the one TileSampler takes per rung')
    ap.add_argument('--patch-tile', type=int, default=256,
                    help='patchable tile size, in OUTPUT px')
    ap.add_argument('--patch-ds', type=float, default=32.0,
                    help='patchable target rung. The COARSEST rung is '
                         'the one that empties cells, so it is the one worth '
                         'looking at')
    ap.add_argument('--with-thumb', action='store_true',
                    help='read the slide for a thumbnail at the mask extent. '
                         'Needs the slides to be mounted and costs seconds each')
    ap.add_argument('--dpi', type=int, default=150)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    seg_dir = cache_root(args.mask_cache_job, 'mask') / MASK_RECIPES[args.seg].seg_id()
    paths = find(seg_dir, '*/mask_meta.json')
    if args.wsi_stem:
        paths = [p for p in paths if p.parent.name in args.wsi_stem]
    if not paths:
        raise SystemExit(f'no masks under {seg_dir}')

    # mask | baseline | [ops x4] | [thumb]
    ops_names = ['[1] filtered', '[2] merged', '[3] patchable',
                 'pipeline [1]->[2]->[3]']
    cols = 2 + (len(ops_names) if args.ops else 0) + (1 if args.with_thumb else 0)
    fig, axes = plt.subplots(len(paths), cols,
                             figsize=(5.0 * cols, 4.4 * len(paths)))
    axes = np.atleast_2d(axes)

    print(f'{len(paths)} masks under {seg_dir}\n')
    for row, path in enumerate(paths):
        meta = read_meta(path)
        slide_mask = SlideMask.load(path.parent / 'mask.safetensors')
        mask = np.asarray(slide_mask.mask).astype(bool)
        stem = path.parent.name

        # `fraction` as stored, against the same number recomputed from the
        # array. Two lines, one fact: if they disagree the file is not what its
        # metadata says.
        recomputed = float(mask.mean())
        agree = abs(recomputed - meta['fraction']) < 1e-4
        print(f'{stem:26s} {args.seg:14s} ds {meta["mask_ds"]:6.2f}  '
              f'{meta["rows"]}x{meta["cols"]}  fg {recomputed:6.2%} '
              f'(stored {meta["fraction"]:6.2%}{"" if agree else "  MISMATCH"})')

        # The regions need the slide, not for pixels but for the four numbers
        # `TissueMask` reads off it -- level_dimensions, mpp and downsamples --
        # so a missing slide costs the region boxes and nothing else. The mask
        # panel still draws, which is the point of storing the mask separately.
        trm = None
        wsi_path = meta.get('wsi_path', '')
        if wsi_path and os.path.exists(wsi_path):
            import openslide                                    # noqa: PLC0415
            wsi = openslide.OpenSlide(wsi_path)
            try:
                trm = TissueMask(wsi, slide_mask)
                print(f'{"":26s} {len(trm.tissue_regions)} regions')
                if args.with_thumb:
                    axes[row, cols - 1].imshow(trm.read_matching_rgb(wsi))
            finally:
                wsi.close()
        else:
            print(f'{"":26s} slide not mounted at {wsi_path!r}; '
                  f'no regions')

        axes[row, 0].imshow(mask, cmap='gray', interpolation='nearest')
        axes[row, 0].set_title(f'{stem}\n{args.seg}  ds '
                               f'{meta["mask_ds"]:.2f}  fg {recomputed:.1%}',
                               fontsize=9)

        axes[row, 1].imshow(mask, cmap='gray', interpolation='nearest')
        col = 2
        if trm is None:
            axes[row, 1].set_title('slide not mounted; no regions', fontsize=9)
        else:
            n0 = len(trm.tissue_regions)
            draw_regions(axes[row, 1], trm)
            axes[row, 1].set_title(f'baseline  {n0} regions', fontsize=9)

            if args.ops:
                # Each operation ALONE, then the three in sequence. Alone is
                # what says which one did the work; the sequence is what the
                # sampler actually sees, and the two differ because `merged`
                # has fewer, larger boxes to work on once `filtered` has run.
                # Every one is a view of `trm`, so each panel carries its own
                # result and none leaks into the next.
                side = int(args.patch_tile * args.patch_ds)
                for name, t in (
                        (ops_names[0], trm.filtered(args.min_ratio)),
                        (ops_names[1], trm.merged()),
                        (ops_names[2], trm.patchable(side)),
                        (ops_names[3], trm.filtered(args.min_ratio).merged()
                                          .patchable(side))):
                    axes[row, col].imshow(mask, cmap='gray',
                                          interpolation='nearest')
                    draw_regions(axes[row, col], t)
                    axes[row, col].set_title(f'{name}\n{n0} -> '
                                             f'{len(t.tissue_regions)}',
                                             fontsize=9)
                    print(f'{"":26s} {name:24s} {n0} -> '
                          f'{len(t.tissue_regions)}')
                    col += 1

        if args.with_thumb and trm is not None:
            axes[row, col].set_title('slide, same extent', fontsize=9)
            draw_regions(axes[row, col], trm)

        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])

    out = args.out or os.path.join(job_result_dir('InspectMaskStore'),
                                   'mask_store.png')
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved  {out}')
    print('\n  Read the foreground fraction against the measured tissue:')
    print('  BRACS 23-38 per cent, Ki67 3.5-9. A Ki67 slide near 0.5 means PC1')
    print('  found banding or position rather than tissue, and every tile cut')
    print('  from it is suspect.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
