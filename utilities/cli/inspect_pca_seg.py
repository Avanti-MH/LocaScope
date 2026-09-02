#!/usr/bin/env python3
"""Which side of PC1 is tissue -- and is PC1 tissue at all?

    python utilities/cli/inspect_pca_seg.py <wsi>... [--fit-tiles 200] [--workers 8]

Outputs (in result/<SLURM_JOB_NAME or InspectPcaSeg>/):
    pca_seg_polarity__<slide>_L0.png
    pca_seg_polarity__<slide>_L0_definitions.csv
    pca_seg_polarity.csv          one row per (slide, polarity)

WHAT THIS DECIDES
-----------------
`Uni2PcaSegConfig.larger_pca_as_fg`. The sign of a principal component is
arbitrary, so the flag exists and nothing in the fit chooses it. On BRACS_1228
the fit reported a foreground fraction of 68.0 percent where the slide's
measured tissue is 20.8 -- and 100 - 68 = 32, which lands inside the 23-38
percent range the three BRACS slides cover. That arithmetic is a hypothesis,
not a finding: the mechanism it assumes is "PC1 separates tissue from glass with
an arbitrary sign", and the thing that decides it is the mask, which is what
this draws.

TWO OUTCOMES, AND THEY ARE NOT "one of the two flags is right"
---------------------------------------------------------------
    A   one polarity's mask lies on the tissue and the other on the glass, and
        the first one's fraction and agreement with the HSV reference are both
        high. -> set that value, record it, move on.

    B   NEITHER lies on the tissue, or the PC1 histogram is unimodal. -> PC1 is
        not the tissue/glass axis on this slide, and the polarity flag is the
        wrong knob entirely. What to reach for then is the fit sample,
        `feature_norm`, or a component other than the first -- and the RGB panel is
        there to say which, because a PC1 that found POSITION comes out as
        bands and one that found the scan grid comes out as stripes.

The histogram is the panel that separates A from B, and it is the reason this
tool draws the continuous field and not just two masks. Two masks always look
like a choice between two answers even when neither is one.

HSV IS A REFERENCE, NOT GROUND TRUTH
-------------------------------------
`agree_hsv` below is agreement with `TissueSegConfig('hsv')`, which this project
has shipped and whose numbers appear in its logs -- 20.8 percent tissue on
BRACS_1228. It is a threshold on saturation and it is wrong at fat, at fold
shadows and at pale sections. So a high agreement means "this polarity is the
one that behaves like the segmenter we already had", which is what the question
asks, and it does NOT mean the mask is correct. Nothing here measures that.

NO MAGNIFICATION FLAG, BECAUSE THE SEGMENTER HAS NO MAGNIFICATION
------------------------------------------------------------------
This tool had `--plane-ds`, then `--level`, then `--crop`, and all three
described how an earlier version read a plane on the segmenter's behalf rather
than anything the segmenter does. `Uni2PcaSegFunc.LEVEL` records the removal and
the measurement behind it: whole slide, every tile, ~3.5 min for BRACS_1003691
and ~5.8 min for a Ki67, so there was never a cost that justified reading
coarser.

What runs now is one call -- `components_wsi` fits on 1000 stratified tiles and
streams every tile of the slide through `utilities/WsiTileLoader`, assembling
only the CELL grid. That is 1/196 of the pixels (`cell_px` squared), which is
what makes a whole Ki67 slide a few hundred MB instead of 10 GB, and it is what
`test_EoMT.slide_pca_mask` has always done.

The mask therefore lands at ds 14 and everything below the projection stays at
cell resolution -- masks, histogram, hsv reference, fractions. Upsampling only
to compare would turn a 22 MB array into a 4.4 GB one to answer the same
question.
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

import numpy as np                                              # noqa: E402
import matplotlib                                               # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                 # noqa: E402

from _paths import job_result_dir, setup_import_paths           # noqa: E402

setup_import_paths()

import cv2                                                      # noqa: E402
import torch                                                    # noqa: E402

from SafeSlide import SafeSlide                                  # noqa: E402
from TissueSegFunc import TissueSegConfig                        # noqa: E402

from Uni2PcaSegFunc import Uni2PcaSegConfig                     # noqa: E402


#: Every string a reader will see, and what it actually computes. Written to
#: <figure>_definitions.csv on every run so the definitions cannot drift away
#: from the code that produced them (ClaudeRules section 12).
DEFINITIONS = [
    ('slide, per-cell mean colour',
     'the scanned rectangle with each cell replaced by the mean colour of the '
     'pixels it covers. Same array shape as the components, so the slide and '
     'the mask cannot slip out of register'),
    ('PC 1-3 as RGB',
     'the first three PCA components per cell, each MinMax-scaled to [0,1] and '
     'shown as one colour channel. Bands or stripes here mean PC1 found '
     'position or the scan grid rather than tissue'),
    ('PC1', 'the first component per cell, MinMax-scaled to [0,1] by the '
            'scaler fitted with the PCA. This is the quantity both masks '
            'threshold'),
    ('PC1 histogram',
     'distribution of that value over every cell of the plane, with '
     'background_threshold marked. Bimodal means a threshold can separate two '
     'populations; unimodal means no value of the flag helps'),
    ('fg=False', 'the mask with larger_pca_as_fg=False, i.e. tissue is PC1 < '
                 'background_threshold'),
    ('fg=True', 'the mask with larger_pca_as_fg=True, i.e. tissue is PC1 > '
                'background_threshold'),
    ('hsv reference',
     "TissueSegConfig('hsv') on the per-cell mean colour, so it lands on the "
     'same grid. A saturation and value '
     'threshold, not ground truth -- it is wrong at fat, at fold shadows and '
     'at pale sections'),
    ('frac', 'fraction of CELLS the mask calls tissue. Cells, not pixels: '
             'the mask is decided per cell and upsampling before counting '
             'would only restate the same fraction'),
    ('cells', 'how many cells the figure covers'),
    ('agree_hsv',
     'intersection over union between the mask and the hsv reference. High '
     'means "behaves like the segmenter we already had", NOT "correct"'),
]


def _iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else float('nan')


def _show(axis, image, title, cmap=None):
    axis.imshow(image, cmap=cmap)
    axis.set_title(title, fontsize=9)
    axis.axis('off')


def _thumb(array, longest=900):
    """Downscale for display only. INTER_AREA so a binary mask shown small does
    not alias into a dotted pattern that reads as texture."""
    height, width = array.shape[:2]
    scale = longest / max(height, width)
    if scale >= 1:
        return array
    return cv2.resize(array, (max(1, int(width * scale)), max(1, int(height * scale))),
                      interpolation=cv2.INTER_AREA)


def analyse(wsi_path, args, out_dir):
    """One slide: fit, project the whole thing, threshold both ways, draw.

    No magnification argument, because the segmenter has none. Level 0, mask at
    ds 14, and the only knobs left are the ones that change what the PCA sees.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stem = os.path.splitext(os.path.basename(wsi_path))[0]

    cfg = Uni2PcaSegConfig(fit_tiles=args.fit_tiles,
                           components=args.components,
                           background_threshold=args.background_threshold,
                           workers=args.workers)
    seg = cfg.build(device)
    wsi = SafeSlide(wsi_path)
    try:
        print(f'  fitting on {args.fit_tiles} tiles, then projecting the slide...',
              flush=True)
        # `components_wsi` fits on first use and streams every tile. The plane
        # is never materialised: at cell resolution the scanned rectangle is
        # 1/196 of the pixels, which is what makes a whole Ki67 slide a 547 MB
        # array instead of a 10 GB one.
        components, thumb_cells = seg.components_wsi(wsi, thumbnail=True)
        report = seg.fit_report
        print(f'    level {report["level"]}, {report["cells"]} fit cells, '
              f'explained {report["explained_variance_top3"]:.1%}, '
              f'mask_ds {report["mask_ds"]:.0f}', flush=True)
    finally:
        wsi.close()

    hsv_mask = None
    pc1 = components[..., 0].astype(np.float32)

    # EVERYTHING BELOW IS AT CELL RESOLUTION. The mask's information lives in
    # the cells, `mask_ds` says how coarse that is, and upsampling only to
    # compare would multiply every array by 196 to answer the same question --
    # 22 MB against 4.4 GB on BRACS_1003691.
    #
    # hsv runs on the per-cell mean colour so the reference lands on the same
    # grid. `mask_hsv` is per-pixel, so running it on a downsampled image is
    # sound in a way `mask_otsu` would not be -- `from_wsi`'s read_chunk_px
    # docstring makes the same distinction for the same reason.
    hsv_mask = TissueSegConfig('hsv').build()(thumb_cells).astype(bool)

    masks = {'fg=False': pc1 < cfg.background_threshold,
             'fg=True': pc1 > cfg.background_threshold}

    rows = []
    for name, mask in masks.items():
        rows.append({'wsi': stem,
                     'level': report['level'], 'level_ds': report['level_ds'],
                     'mask_ds': report['mask_ds'], 'polarity': name,
                     'cells': int(pc1.size), 'frac': float(mask.mean()),
                     'agree_hsv': _iou(mask, hsv_mask),
                     'frac_hsv': float(hsv_mask.mean()),
                     'explained_top3': report['explained_variance_top3'],
                     'fit_cells': report['cells']})
        print(f'    {name:<9} frac {rows[-1]["frac"]:.1%}  '
              f'agree_hsv {rows[-1]["agree_hsv"]:.3f}   '
              f'(hsv itself {rows[-1]["frac_hsv"]:.1%})', flush=True)

    # ── figure ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    flat = axes.flatten()

    _show(flat[0], _thumb(thumb_cells),
          f'slide, per-cell mean colour\n'
          f'{pc1.shape[1]} x {pc1.shape[0]} cells of '
          f'{report["mask_ds"]:.0f} level-0 px')

    rgb = components[..., :3]
    lo, hi = rgb.min(axis=(0, 1)), rgb.max(axis=(0, 1))
    rgb = (rgb - lo) / np.maximum(hi - lo, 1e-6)
    _show(flat[1], _thumb(rgb.astype(np.float32)),
          'PC 1-3 as RGB\nbands or stripes = not tissue')

    _show(flat[2], _thumb(pc1.astype(np.float32)),
          f'PC1   explained(1-3) {report["explained_variance_top3"]:.1%}\n'
          f'one cell = {report["mask_ds"]:.0f} level-0 px', cmap='viridis')

    axis = flat[3]
    axis.hist(pc1.ravel(), bins=120, color='0.4')
    axis.axvline(cfg.background_threshold, color='crimson', linewidth=1.5)
    axis.set_title(f'PC1 histogram over {pc1.size} cells\n'
                   f'threshold {cfg.background_threshold} in red -- '
                   f'bimodal?', fontsize=9)
    axis.set_yticks([])

    # SELECTED IS GREEN, ON THE SLIDE. Not a grey mask, which is what these two
    # panels were and which got read backwards on 2026-08-26: with cmap='gray'
    # white is 1, and when the selected region is the BACKGROUND the eye reads
    # the dark shape as the answer -- the tissue-shaped speckle in an fg=False
    # panel is value 0, and it looks exactly like a tissue mask.
    #
    # An overlay cannot be read that way round: whatever is tinted is what the
    # mask holds. `test_EoMT.draw:611` does the same thing for the same reason.
    for axis, (name, mask) in zip(flat[4:], masks.items()):
        row = next(r for r in rows if r['polarity'] == name)
        over = thumb_cells.copy()
        selected = mask.astype(bool)
        over[selected] = (over[selected] * 0.55
                          + np.array([0, 200, 80]) * 0.45).astype(np.uint8)
        _show(axis, _thumb(over),
              f'{name}   GREEN = selected\n'
              f'frac {row["frac"]:.1%}   agree_hsv {row["agree_hsv"]:.3f}')

    fig.suptitle(
        f'{stem}   level {report["level"]}, whole slide, mask ds '
        f'{report["mask_ds"]:.0f} -- which side of PC1 is tissue?\n'
        f'hsv reference here: {rows[0]["frac_hsv"]:.1%} tissue.  On a pale '
        f'section hsv UNDER-counts -- it is a saturation threshold, so '
        f'agree_hsv being low does not make the mask wrong', fontsize=12)
    fig.text(0.5, 0.005,
             'agree_hsv is IoU against a saturation threshold, not against '
             'ground truth. If the histogram is unimodal, neither polarity is '
             'the answer. If the PC1 map shows a regular lattice over blank '
             'glass, the components carry the tile grid and no threshold fixes '
             'that.',
             ha='center', fontsize=8, color='0.35')
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))

    tag = f"L{report['level']}"
    path = os.path.join(out_dir, f'pca_seg_polarity__{stem}_{tag}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'    saved {path}', flush=True)

    with open(path.replace('.png', '_definitions.csv'), 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['term', 'means'])
        writer.writerows(DEFINITIONS)
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('wsi', nargs='+', help='one or more slides')
    # NO --plane-ds, --level or --crop. The segmenter reads at level 0 and
    # produces a ds 14 mask over the whole slide, and none of those three
    # described anything it does -- they described how an earlier version read
    # a plane on its behalf. See Uni2PcaSegFunc.LEVEL for the measurement that
    # removed the last reason to keep them.
    ap.add_argument('--workers', type=int, default=8,
                    help='DataLoader workers for reading tiles. 0 reads in the '
                         'parent, which is right on a login node and wrong for '
                         'a whole slide')
    ap.add_argument('--fit-tiles', type=int, default=200)
    ap.add_argument('--components', type=int, default=16)
    ap.add_argument('--background-threshold', type=float, default=0.5)
    ap.add_argument('--out', default=None,
                    help='output directory. Empty means '
                         'result/<SLURM_JOB_NAME or InspectPcaSeg>/')
    args = ap.parse_args()

    out_dir = args.out or job_result_dir('InspectPcaSeg')
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for wsi_path in args.wsi:
        print(f'\n{os.path.basename(wsi_path)}', flush=True)
        rows.extend(analyse(wsi_path, args, out_dir))

    summary = os.path.join(out_dir, 'pca_seg_polarity.csv')
    with open(summary, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nSaved {summary}   ({len(rows)} rows)')

    # The verdict, said out loud rather than left in a CSV. Deliberately not an
    # assertion: nothing here knows which answer is right, and a tool that
    # printed "use fg=True" would be claiming it did.
    print('\nRead it this way:')
    print('  - histogram bimodal, one polarity clearly on the tissue, its '
          'agree_hsv well above the other -> that is the value')
    print('  - histogram unimodal, or both agree_hsv similar and low -> PC1 is '
          'not the tissue axis here; the flag is the wrong knob and the RGB '
          'panel says what it found instead')
    print('  - the two slides disagree -> the polarity is per-stain, not one '
          'flag, and a single ladder covering both has a problem')
    return 0


if __name__ == '__main__':
    sys.exit(main())
