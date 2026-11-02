#!/usr/bin/env python3
"""Three independent demos, selected with `--parts` (all three run by
default). spec.md 3.2, plan.md 2.1②.

    python training/SuperPathPoint/cli/demo_survival_analysis.py
    python training/SuperPathPoint/cli/demo_survival_analysis.py --parts chains_stack
    python training/SuperPathPoint/cli/demo_survival_analysis.py --parts merge_grid
    python training/SuperPathPoint/cli/demo_survival_analysis.py --parts merge_grid \
        --checkpoint <.pt> --wsi-name BRACS_1228 --real-flow
    python training/SuperPathPoint/cli/demo_survival_analysis.py --parts visualize
    python training/SuperPathPoint/cli/demo_survival_analysis.py --parts visualize \
        --checkpoint <.pt> --wsi-name BRACS_1228 --c-rungs 4 8 16
    python training/SuperPathPoint/cli/demo_survival_analysis.py --parts scale_diagnostic \
        --checkpoint <.pt> --wsi-name BRACS_1228 --n-scale-tiles 10

The parts share `--wsi-name`/`--tile`/`--c-rungs`/`--chainstack-cache-job`/`--out` and nothing
else, so this file keeps them as independent functions (`_run_chains_stack`,
`_run_merge_grid`, `_run_visualize`) dispatched from one `main()`, never
forcing one part's logic through another's. `visualize` (a PPT demo of this
stage) is the same pattern again: synthetic, step-by-
step GIFs of keypoints -> anchor merge -> alive, with an optional real-data
closing figure -- see that section's own module comment.

`--out` is now a DIRECTORY shared by both parts (`result/<job>/`, default
`job_result_dir('DemoSurvivalAnalysis')`) rather than each part resolving
its own default -- `chains_stack` writes under `<out>/figures/`, exactly as
before; `merge_grid` writes `<out>/merge_grid_demo*.png`, exactly as before.
Neither part's own filenames changed, only where the shared root comes from.

=====================================================================
PART "chains_stack" -- smoke-test all three axes' OWN and REUSE-F paths
against a REAL slide
=====================================================================
`test_chain_stack.py` (2.1①) proved the geometry is correct against synthetic
coordinates, and `own`'s wiring against a fake corpus fixture. What
neither can catch: a wrong pyramid LEVEL, a transposed axis, an off-by-one in
`_read_wsi_tile`'s level resolution -- anything that still produces a
plausible `tile x tile` RGB array. Those only show up against real tissue,
which is what this part runs.

FIVE PATHS, NOT TWO
    F           `FStack.from_own`                     -- own, F's only source
    R  own      `RStack.from_own`                     -- stageA, independent
    R  reuse-F  `RStack.derive(chain, source='F')`     -- same chain as F's own
    C  own      `CStack.from_own`                      -- stageB-cOwn, independent
    C  reuse-F  `CStack.from_mother` + `FStack.read`   -- same chain's ds-16 tile

`own` and `reuse-F` are NOT the same physical point for R/C: `own`'s centre is
wherever its own independent draw landed; `reuse-F`'s centre is F's own
chain's centre. Each is checked and PLOTTED on its own, never overlaid.

ONE QUANTITATIVE CHECK, ON BOTH C TREES: crop the mother's own image down to
where a real child tile sits, and correlate that crop against the child's own
independent read -- against a DECOY (the SAME crop vs a DIFFERENT child), the
way every criterion in this project is a margin over a decoy and not a bare
threshold.

EACH AXIS'S CORPUS IS `prepare_chain_stack.axis_corpus` -- imported, not
re-derived here, so both address the same directories.

=====================================================================
PART "merge_grid" -- three implementations of `_merge_within_radius`
compared
=====================================================================
ADOPTED. `A` (grid hash) is what `SurvivalProcess._merge_within_radius` runs
now -- the evidence this part produced (identical kept-index sets on
synthetic data; 131x faster on a real C-tree's raw ds=1 detections, 34503
points; 0% anchor mismatch and bit-identical aggregated curves deployed
end-to-end in `cli/survival_alpha_analysis.py`'s own C-axis flow, 1.5x
faster overall) is what cleared it. `B`, called here through
`AlphaCalibration.merge_anchors`, now ALSO runs the grid version underneath
(it calls the same module attribute) -- so `B` and `A` will always agree
going forward; the two are kept as SEPARATE, independently-written copies
specifically so a future edit to one that silently diverges from the other
shows up as a mismatch here, not as a downstream number nobody double-checks.
`original` remains the one genuinely different, genuinely slower baseline.

No model, no GPU, no slide by default -- synthetic keypoints. `--checkpoint`
(+ `--wsi-name`) adds a THIRD comparison on REAL detections: one C-tree's
actual keypoints at one rung, read the same way
`cli/survival_alpha_analysis.py`'s C axis does. `--real-flow` adds a FOURTH:
each implementation deployed (monkey-patched) into
`survival_alpha_analysis.py`'s actual C-axis flow -- anchors, probe,
`alpha_curve`, `aggregate_curves` -- so the comparison is not just "does the
merge primitive agree in isolation" but "does swapping it change one number
the real pipeline reports."

FOUR COMPARISONS, because they answer different questions -- every one
reports a QUANTITATIVE mismatch (a count/fraction of points, or a max abs
diff on aggregated numbers), never just a boolean "agree":
    the FIGURE (`--n-clusters` etc.)   small enough to plot and look at.
    the TIMING (`--timing-n-clusters`) `--repeats` INDEPENDENT synthetic
                                       draws, a timing DISTRIBUTION and a
                                       disagreement RATE.
    REAL DATA (`--checkpoint`)         one C-tree's actual detections.
    REAL FLOW (`--real-flow`)          each implementation deployed into the
                                       real C-axis flow.

Deferred imports (torch, AccessDatasets, SafeSlide, ChainStack,
SurvivalProcess, prepare_chain_stack, reeval_density, survival_alpha_analysis)
inside `merge_grid`'s real-data/real-flow functions ONLY -- so a
`--parts merge_grid` run with no `--checkpoint` stays torch-free. `chains_
stack` needs all of those unconditionally (it always reads a real slide), so
they sit at module level.

Outputs (in `result/<SLURM_JOB_NAME or DemoSurvivalAnalysis>/`):
    figures/r_stack_{own,reuseF}.png                      [chains_stack]
    figures/pyramid_{lineage,overview}_{own,reuseF}.png   [chains_stack]
    merge_grid_demo.png                                   [merge_grid]
    merge_grid_demo_real.png        only with --checkpoint --plot-real
    merge_grid_demo_real_flow.png   only with --real-flow
    visualize_keypoints.gif                               [visualize]
    visualize_anchor_merge.gif                             [visualize]
    visualize_alive.gif                                    [visualize]
    visualize_real_example.png      only with --checkpoint  [visualize]
    figures/scale_diagnostic_tile{k:02d}.png   [scale_diagnostic]
    figures/scale_diagnostic_decay_vs_survival.png   [scale_diagnostic]
    figures/scale_synthetic_control.png   [scale_synthetic_control]
    figures/scale_synthetic_control_examples.png   [scale_synthetic_control]

=====================================================================
PART "scale_diagnostic" -- does RStack's own resampling filter matter, or
does the six-pattern classification track pure SCALE?
=====================================================================
Takes `--n-scale-tiles` real F-chain tiles (ds=1, sharpest read), builds TWO
R-shaped stacks per tile from the SAME real pixels -- `RStack.from_tile`
(shrink + grow back) and `classical_blur_stack` (plain Gaussian blur,
`sigma=ds/2`, an approximation, not a derived equivalence -- see that
function's own docstring) -- runs the real detector + the production R-axis
survival flow (`detect_all_rungs` -> `anchors_of` -> `probe_real` ->
`Patterns.classify`) on both, and marks only three of the six patterns on a
2-row (RStack / classical blur) x len(--rungs) column (ds=1 first, most
degraded last) figure per tile: 只在一階 green X, 一直存活 black X, 不連續
(flicker) grey X. An anchor's marker sits at the SAME pixel in every column
of its own row (R's footprint never moves), so watching it fade into the
blur shows exactly where its row's `alive[]` vector says it died. 細部存活/
晚生型/中間帶 are drawn too, as an unfilled circle coloured by that rung's
own score (dark = high, `_SCALE_CIRCLE_CMAP`) rather than a sixth fixed
colour.

A SECOND, POOLED FIGURE (`scale_diagnostic_decay_vs_survival.png`, 2026-09-
14): for every anchor across every tile and both rows, `_decay_rate`
probes a CLASSICAL, non-learned `|Laplacian|` response on the classical
blur stack at several ds (Theil-Sen slope of log(R) vs log(ds), negated),
and `_survival_breadth` reads off the coarsest ds that anchor's own
`alive[]` vector still says yes to. Scatter of one against the other, with
the real Spearman correlation reported alongside a PERMUTATION test (shuffle
the pairing `--decay-permutations` times, report what fraction of shuffles
correlate at least as strongly) -- this project's own "margin over a decoy"
convention, applied to a correlation instead of a group difference, and the
reason `一直存活` was NOT singled out as a hand-picked comparison group: a
continuous correlation over every anchor uses more of the data and does not
require deciding which pattern counts as the baseline.

NEEDS --checkpoint, ALWAYS -- unlike the other three parts this has no
synthetic-only mode and is never in the default `--parts` list.

=====================================================================
PART "scale_synthetic_control" -- is `_decay_rate` a working ruler at all?
=====================================================================
`--n-synthetic-points` Gaussian blobs of KNOWN, controlled radius (log-
uniform, same 1..16 range `--rungs` spans), well separated on one flat
canvas, put through the SAME `classical_blur_stack`/`_laplacian_maps`/
`_decay_rate` pipeline `scale_diagnostic` uses on real anchors -- except
here the "true scale" is not inferred, it is the radius the blob was drawn
with. Reports the same Spearman rho + permutation p, on ground truth.

WHY THIS EXISTS: a null correlation on real data (only) is ambiguous
between "the scale hypothesis is wrong" and "this particular measurement
cannot see it" -- this part is the way to tell those apart. If it ALSO
shows near-zero correlation here, on a KNOWN size difference, the real
result says nothing about the hypothesis; only if this control shows a
strong, clean negative correlation does the real-data null become real
evidence. No checkpoint, no slide, no torch -- pure numpy/cv2, fast.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', '..', 'utilities'))
import _paths                                                     # noqa: E402
_paths.setup_import_paths('SuperPathPoint')

from cli import (add_chainstack_args, add_corpus_args,  # noqa: E402
                 chainstack_root, job_result_dir)


import matplotlib                                              # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                # noqa: E402
import matplotlib.colors                                        # noqa: E402
import matplotlib.cm                                            # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter     # noqa: E402
from matplotlib.lines import Line2D                              # noqa: E402
from matplotlib.patches import Rectangle                        # noqa: E402
import numpy as np                                              # noqa: E402
import cv2                                                       # noqa: E402

import AccessDatasets                                              # noqa: E402
from SafeSlide import SafeSlide                                  # noqa: E402
from SlideReader import SlideReader                              # noqa: E402
from ReadGeometry import ReadSpec                                # noqa: E402
from SurvivalAnalysis import ChainStack, SurvivalProcess          # noqa: E402
FStack, RStack, CStack = ChainStack.FStack, ChainStack.RStack, ChainStack.CStack
from SurvivalAnalysis.AlphaCalibration import merge_anchors        # noqa: E402
from SurvivalAnalysis.AliveCandidates import (                     # noqa: E402
    probe_via_probability_map, alive_probability_map, _bilinear_sample)
from SurvivalAnalysis.Patterns import ASCII_NAMES, alive_from, classify  # noqa: E402
import prepare_chain_stack                                        # noqa: E402

DEFAULT_WSI_NAME = 'BRACS_1228'


# =============================================================================
#  chains_stack
# =============================================================================

def _looks_like_tissue(img) -> bool:
    """Cheap sanity, not a real assertion: a blank/near-uniform read (wrong
    level reading past the slide edge, a black hole) has near-zero std."""
    return float(np.std(img)) > 5.0


def _correlate(a, b) -> float:
    a = a.astype(np.float32).ravel() - a.astype(np.float32).mean()
    b = b.astype(np.float32).ravel() - b.astype(np.float32).mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def _cache_stats(cache_root):
    """`(n_files, n_bytes)` under `cache_root`, ANY depth (`rglob`, not
    `glob` -- tiles sit in `<dir>/<Axis>Stack/<pyramid>/*.png`)."""
    if not cache_root or not os.path.isdir(cache_root):
        return 0, 0
    files = list(Path(cache_root).rglob('*.png'))
    return len(files), sum(f.stat().st_size for f in files)


def check_C(mother, mother_img, groups_by_ds, wsi, wsi_stem, tile, cache_root,
           lineage_index, label):
    """One lineage walk down from `mother`, PLUS the quantitative check:
    crop the mother's own image down to where a real child tile sits, and
    correlate that crop against the child's own independent read -- against a
    DECOY (the SAME crop vs a DIFFERENT child). Geometry (`mother`,
    `groups_by_ds`) is the caller's responsibility -- `own`'s comes from
    `CStack.from_own`, `reuse-F`'s from `CStack.from_mother` -- this only
    reads and checks, so both trees run through the exact same code.
    """
    t_read0 = time.perf_counter()
    lineage = {}                      # ds -> (group, [main imgs], overlap img)
    node = mother
    n_reads = 0
    for ds in sorted(groups_by_ds, reverse=True):
        group = next(g for g in groups_by_ds[ds] if g.parent == node)
        main_imgs = [CStack.read_one(m, wsi, wsi_stem=wsi_stem, root=mother,
                                     finest=min(groups_by_ds), tile=tile,
                                     cache_root=cache_root)
                    for m in group.main]
        overlap_img = CStack.read_one(group.overlap, wsi, wsi_stem=wsi_stem,
                                      root=mother, finest=min(groups_by_ds),
                                      tile=tile, cache_root=cache_root)
        n_reads += len(group.main) + 1
        lineage[ds] = (group, main_imgs, overlap_img)
        node = group.main[min(lineage_index, len(group.main) - 1)]
    t_read = time.perf_counter() - t_read0

    print(f"C  {label}: read {n_reads} real descendant tiles (one lineage) "
          f"in {t_read*1000:.1f} ms   ({t_read/max(n_reads,1)*1000:.1f} ms/tile)")

    # ── the quantitative check: mother crop vs child read, against a decoy ──
    first_ds = max(groups_by_ds)
    group0, mains0, _ = lineage[first_ds]
    scale = tile / mother.size_px
    def _crop_of(child):
        x0 = int(round((child.x - mother.x) * scale))
        y0 = int(round((child.y - mother.y) * scale))
        s = max(1, int(round(child.size_px * scale)))
        patch = mother_img[y0:y0 + s, x0:x0 + s]
        return cv2.resize(patch, (tile, tile), interpolation=cv2.INTER_AREA)

    crop = _crop_of(group0.main[0])
    real_corr = _correlate(crop, mains0[0])
    decoy_corr = _correlate(crop, mains0[-1])         # same crop, WRONG child
    passed = real_corr > decoy_corr + 0.1
    print(f"C  {label}: mother-crop vs its real child:  real={real_corr:.3f}  "
          f"decoy(wrong child)={decoy_corr:.3f}   "
          f"{'OK -- real wins' if passed else 'FAIL -- geometry/pixels disagree'}")

    return lineage


def read_whole_tree(wsi, wsi_stem, groups_by_ds, mother, mother_img, tile,
                    cache_root):
    """Every tile in the pyramid, GROUPED BY RUNG (`{ds: [(info, img), ...]}`).
    For the overview figure -- NOT what `check_C`'s own numbers cost, which
    only reads one lineage. Cached, so a second run of the same centre is
    nearly free, but the FIRST run at the default 4-transition `--c-rungs` is
    ~425 real reads.
    """
    t0 = time.perf_counter()
    finest = min(groups_by_ds)
    by_ds = {mother.ds: [(mother, mother_img)]}
    n = 1
    for ds, groups in groups_by_ds.items():
        items = []
        for g in groups:
            for m in g.main:
                items.append((m, CStack.read_one(m, wsi, wsi_stem=wsi_stem,
                                                 root=mother, finest=finest,
                                                 tile=tile,
                                                 cache_root=cache_root)))
            items.append((g.overlap,
                         CStack.read_one(g.overlap, wsi, wsi_stem=wsi_stem,
                                        root=mother, finest=finest, tile=tile,
                                        cache_root=cache_root)))
        by_ds[ds] = items
        n += len(items)
    dt = time.perf_counter() - t0
    print(f"C  overview: read all {n} real tiles in {dt:.2f} s "
          f"({dt/n*1000:.1f} ms/tile)")
    return by_ds


def _strip_figure(images_by_ds, title, out_path, footprints=None):
    """Left-to-right, finest to coarsest -- one panel per rung, no nesting."""
    rungs = sorted(images_by_ds)
    fig, axes = plt.subplots(1, len(rungs), figsize=(2.3 * len(rungs), 2.8))
    if len(rungs) == 1:
        axes = [axes]
    for ax, ds in zip(axes, rungs):
        ax.imshow(images_by_ds[ds])
        label = f'ds={ds:g}'
        if footprints is not None:
            label += f'\n{footprints[ds]:.0f} L0 px'
        ax.set_title(label, fontsize=9)
        ax.axis('off')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _draw_tile_box(ax, info, *, color, style):
    ax.add_patch(Rectangle((info.x, info.y), info.size_px, info.size_px,
                            fill=False, edgecolor=color, linewidth=0.8,
                            linestyle=style))


#: One colour for every line in the overview, deliberately -- see plan.md's
#: case history: a colour-by-rung continuous colormap failed on its own
#: terms once the finest rung's sheer tile count dominated every panel.
_LINE_COLOR = '#d6273c'


def _overview_figure(by_ds, mother, title, out_path):
    """One panel per rung, coarsest (left) to finest (right), SAME physical
    area in every panel -- the tile count growing is the whole point."""
    rungs = sorted(by_ds, reverse=True)
    fig, axes = plt.subplots(1, len(rungs), figsize=(3.2 * len(rungs), 3.6))
    if len(rungs) == 1:
        axes = [axes]
    for ax, ds in zip(axes, rungs):
        for info, img in by_ds[ds]:
            ax.imshow(img, extent=(info.x, info.x + info.size_px,
                                   info.y + info.size_px, info.y))
            _draw_tile_box(ax, info, color=_LINE_COLOR,
                          style='-' if info.kind == 'main' else '--')
        ax.set_xlim(mother.x, mother.x + mother.size_px)
        ax.set_ylim(mother.y + mother.size_px, mother.y)
        ax.set_aspect('equal')
        n_main = sum(1 for info, _ in by_ds[ds] if info.kind == 'main')
        ax.set_title(f'ds={ds:g}\n{n_main} main tile'
                     f'{"s" if n_main != 1 else ""}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


#: Fixed, hand-picked list rather than a continuous colormap -- same
#: complaint as `_LINE_COLOR`: `viridis`/`plasma` both route through yellow
#: and through blue-and-purple-together.
_LINEAGE_COLORS = ('#d6273c', '#e07b1a', '#1a7a3c', '#6b4423', '#555555')


def _lineage_figure(mother, mother_img, lineage, title, out_path):
    """Real pixels, not empty rectangles -- the whole point of this run."""
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(mother_img, extent=(mother.x, mother.x + mother.size_px,
                                  mother.y + mother.size_px, mother.y))
    _draw_tile_box(ax, mother, color='black', style='-')

    rungs = sorted(lineage, reverse=True)
    for i, ds in enumerate(rungs):
        group, main_imgs, overlap_img = lineage[ds]
        color = _LINEAGE_COLORS[i % len(_LINEAGE_COLORS)]
        for m, img in zip(group.main, main_imgs):
            ax.imshow(img, extent=(m.x, m.x + m.size_px, m.y + m.size_px, m.y))
            _draw_tile_box(ax, m, color=color, style='-')
        o = group.overlap
        ax.imshow(overlap_img, extent=(o.x, o.x + o.size_px,
                                       o.y + o.size_px, o.y), alpha=0.6)
        _draw_tile_box(ax, o, color=color, style='--')

    ax.set_xlim(mother.x, mother.x + mother.size_px)
    ax.set_ylim(mother.y + mother.size_px, mother.y)
    ax.set_aspect('equal')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _run_chains_stack(args, out_dir: str) -> None:
    wsi_name = args.wsi_name or DEFAULT_WSI_NAME
    if args.wsi is None:
        entry = AccessDatasets.locate(wsi_name)
        wsi_path, wsi_stem = entry.path, entry.name
    else:
        wsi_path, wsi_stem = args.wsi, args.wsi_stem

    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    cache_root = chainstack_root(args)
    mother_ds = max(args.c_rungs)

    # prepare_chain_stack's own addresses -- imported, not re-derived, see
    # the module docstring.
    f_corpus, r_corpus, c_corpus = (prepare_chain_stack.axis_corpus(a, args)
                                    for a in 'FRC')

    print(f"slide: {wsi_path}")
    # this slide's tile cache in the job's tree
    tile_dir = ChainStack._cache_dir(cache_root, wsi_stem) if cache_root else None
    files_before, bytes_before = _cache_stats(tile_dir)

    r_own_stack = None
    c_own_mother = c_own_img = c_own_groups = lineage_own = whole_tree_own = None

    with SafeSlide(wsi_path) as wsi:
        # ── F: own is F's only source ───────────────────────────────────
        f_own = FStack.from_own(f_corpus, wsi_stem, tile=args.tile,
                                rungs=args.rungs)
        if not len(f_own):
            raise RuntimeError(
                f'F has no complete chain for {wsi_stem} in '
                f'{f_corpus.key}: its draw holds no chain over every rung')
        chain_id = next(iter(f_own))
        chain = f_own.chains[chain_id]
        print(f"centre: ({chain.cx:.0f}, {chain.cy:.0f}), chain {chain_id}")
        t0 = time.perf_counter()
        f_stack = f_own[chain_id]
        dt = time.perf_counter() - t0
        oks = [_looks_like_tissue(f_stack[d]) for d in f_stack]
        print(f"F  own: chain {chain_id}, {len(f_stack)} rungs read in "
              f"{dt*1000:.1f} ms   {'OK' if all(oks) else 'FAIL'}")

        # ── R: own (stageA, independent) and reuse-F (this same chain) ──
        # R's own tiles are a resize of a stored pre-tile: never cached
        # (RStack.from_own says why), whatever --chainstack-cache-job is
        r_own = RStack.from_own(r_corpus, wsi_stem, args.rungs,
                                tile=args.tile, cache_root=None)
        if len(r_own):
            t0 = time.perf_counter()
            r_own_stack = r_own[0]
            dt = time.perf_counter() - t0
            oks = [_looks_like_tissue(r_own_stack[d]) for d in r_own_stack]
            print(f"R  own: tile 0, {len(r_own_stack)} rungs derived in "
                  f"{dt*1000:.1f} ms   {'OK' if all(oks) else 'FAIL'}")
        else:
            print(f"R  own: 0 tiles found for {wsi_stem} under "
                  f"{r_corpus.key} -- skipped")

        t0 = time.perf_counter()
        r_stack = RStack.derive(chain, args.rungs, tile=args.tile, source='F')
        dt = time.perf_counter() - t0
        oks = [_looks_like_tissue(r_stack[d]) for d in args.rungs]
        print(f"R  reuse-F: derived {len(args.rungs)} rungs from chain "
              f"{chain_id} in {dt*1000:.1f} ms   {'OK' if all(oks) else 'FAIL'}")

        # ── C: own (stageB-cOwn, independent) ────────────────────────────
        c_own = CStack.from_own(c_corpus, wsi_stem, args.c_rungs,
                                wsi, tile=args.tile, cache_root=cache_root)
        if len(c_own):
            t0 = time.perf_counter()
            c_own_mother, c_own_img, c_own_groups, _ = c_own[0]
            dt = time.perf_counter() - t0
            print(f"C  own: tree 0, mother ds {c_own_mother.ds:g} read in "
                  f"{dt:.2f} s")
            lineage_own = check_C(c_own_mother, c_own_img, c_own_groups, wsi,
                                  wsi_stem, args.tile, cache_root,
                                  args.lineage_index, label='own')
            whole_tree_own = read_whole_tree(wsi, wsi_stem, c_own_groups,
                                             c_own_mother, c_own_img,
                                             args.tile, cache_root)
        else:
            print(f"C  own: 0 trees found for {wsi_stem} under "
                  f"{c_corpus.key} -- skipped")

        # ── C: reuse-F (this same chain's own ds-16 read as the mother) ──
        if mother_ds not in chain.members:
            raise RuntimeError(
                f"chain {chain_id} has no ds {mother_ds:g} member -- C's "
                f"reuse-F path needs F's own chain to cover C's mother rung")
        reuse_mother_img = FStack.read(chain, tile=args.tile)[mother_ds]
        t_geom0 = time.perf_counter()
        mother, mother_img, groups_by_ds, _ = CStack.from_mother(
            reuse_mother_img, chain.cx, chain.cy, args.c_rungs, wsi,
            wsi_stem=wsi_stem, tile=args.tile, cache_root=cache_root)
        t_geom = time.perf_counter() - t_geom0
        n_groups = sum(len(g) for g in groups_by_ds.values())
        print(f"C  reuse-F: pyramid ({n_groups} groups, mother ds "
              f"{mother.ds:g}) built + read in {t_geom:.2f} s")
        lineage_reuse = check_C(mother, mother_img, groups_by_ds, wsi,
                                wsi_stem, args.tile, cache_root,
                                args.lineage_index, label='reuse-F')
        whole_tree_reuse = read_whole_tree(wsi, wsi_stem, groups_by_ds,
                                          mother, mother_img, args.tile,
                                          cache_root)

    files_after, bytes_after = _cache_stats(tile_dir)
    print(f"local cache: +{files_after - files_before} files, "
          f"+{(bytes_after - bytes_before)/1024:.0f} KB THIS RUN "
          f"({files_after} files, {bytes_after/1024:.0f} KB total on disk)")

    # ONE C TREE AT A TIME, on its own -- the question "how big is one
    # pyramid" needs the pyramid's OWN directory (`_pyramid_dir`), not the
    # run-wide delta above, which also includes R's tiles and both trees.
    for label, root_mother in (('own', c_own_mother), ('reuse-F', mother)):
        if root_mother is None:
            continue
        c_dir = ChainStack._pyramid_dir(cache_root, 'C', wsi_stem,
                                        root_mother, args.tile,
                                        min(args.c_rungs))
        c_files, c_bytes = _cache_stats(c_dir)
        print(f"C  {label} tree on disk: {c_files} files, "
              f"{c_bytes/1024:.0f} KB ({c_bytes/1024/1024:.2f} MB)  -> {c_dir}")

    _strip_figure(r_stack, 'R axis (reuse-F), real tissue -- fixed footprint, '
                 'degraded resolution',
                 os.path.join(fig_dir, 'r_stack_reuseF.png'))
    if r_own_stack is not None:
        _strip_figure(r_own_stack, 'R axis (own), real tissue -- fixed '
                     'footprint, degraded resolution',
                     os.path.join(fig_dir, 'r_stack_own.png'))

    _lineage_figure(mother, mother_img, lineage_reuse,
                    'C axis (reuse-F), real tissue -- one lineage down from '
                    'the mother\nsolid = main   dashed = overlap (半透明, '
                    'shows the 1/4 share)',
                    os.path.join(fig_dir, 'pyramid_lineage_reuseF.png'))
    _overview_figure(whole_tree_reuse, mother,
                     'C axis (reuse-F), real tissue -- SAME area every '
                     'panel, coarsest (left) to finest (right)\n'
                     'solid = main (recurses)   dashed = overlap (leaf)',
                     os.path.join(fig_dir, 'pyramid_overview_reuseF.png'))
    if lineage_own is not None:
        _lineage_figure(c_own_mother, c_own_img, lineage_own,
                        'C axis (own), real tissue -- one lineage down from '
                        'the mother\nsolid = main   dashed = overlap (半透明, '
                        'shows the 1/4 share)',
                        os.path.join(fig_dir, 'pyramid_lineage_own.png'))
        _overview_figure(whole_tree_own, c_own_mother,
                         'C axis (own), real tissue -- SAME area every '
                         'panel, coarsest (left) to finest (right)\n'
                         'solid = main (recurses)   dashed = overlap (leaf)',
                         os.path.join(fig_dir, 'pyramid_overview_own.png'))

    print(f"\nfigures -> {fig_dir}/"
          f"{{r_stack_own,r_stack_reuseF,"
          f"pyramid_lineage_own,pyramid_overview_own,"
          f"pyramid_lineage_reuseF,pyramid_overview_reuseF}}.png")


# =============================================================================
#  merge_grid
# =============================================================================

def merge_within_radius_original(points: np.ndarray, radius: float,
                                 priority: Optional[np.ndarray] = None
                                 ) -> np.ndarray:
    """The pre-fix version. `kept_xy` is a Python list of points, rebuilt into
    a fresh array with `np.asarray` on every iteration -- that rebuild's own
    cost is O(k) per iteration, so summed over all n iterations it is its own
    O(n^2), stacked on top of the O(n*k) the comparisons already cost.
    """
    n = len(points)
    if not n or float(radius) <= 0.0:
        return np.arange(n, dtype=np.int64)
    order = (np.argsort(-np.asarray(priority)) if priority is not None
            else np.arange(n))
    kept: List[int] = []
    kept_xy: List[np.ndarray] = []
    for i in order:
        i = int(i)
        if kept_xy:
            near = np.linalg.norm(np.asarray(kept_xy) - points[i], axis=1)
            if float(near.min()) <= float(radius):
                continue
        kept.append(i)
        kept_xy.append(points[i])
    return np.sort(np.asarray(kept, np.int64))


def merge_within_radius_grid(points: np.ndarray, radius: float,
                             priority: Optional[np.ndarray] = None
                             ) -> np.ndarray:
    """Method A. Same contract as `SurvivalProcess._merge_within_radius`:
    indices to KEEP after a greedy, priority-ordered radius merge -- visit
    points highest priority first (input order if `priority` is None), keep
    a point only if no already-kept point is within `radius` of it.

    Only the SEARCH SCOPE changes. A candidate compares against the
    already-kept points in its own radius-sized grid cell and the
    surrounding 3x3, not every already-kept point: any point within `radius`
    of a cell of side `radius` must fall in that 3x3 neighbourhood, so
    nothing is missed. That is where the speed difference comes from, not
    from a looser merge.
    """
    n = len(points)
    if not n or float(radius) <= 0.0:
        return np.arange(n, dtype=np.int64)
    order = (np.argsort(-np.asarray(priority)) if priority is not None
            else np.arange(n))
    cell = float(radius)
    grid: Dict[Tuple[int, int], List[int]] = {}
    kept: List[int] = []
    for i in order:
        i = int(i)
        cx = int(np.floor(points[i, 0] / cell))
        cy = int(np.floor(points[i, 1] / cell))
        near = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((cx + dx, cy + dy), ()):
                    if np.linalg.norm(points[j] - points[i]) <= radius:
                        near = True
                        break
                if near:
                    break
            if near:
                break
        if near:
            continue
        kept.append(i)
        grid.setdefault((cx, cy), []).append(i)
    return np.sort(np.asarray(kept, dtype=np.int64))


def synthetic_points(n_clusters: int, max_per_cluster: int, jitter: float,
                     extent: float, seed: int
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Fake keypoints: `n_clusters` physical points, each hit by 1..
    `max_per_cluster` near-duplicate detections (jitter = the sub-pixel
    disagreement a real point gets across rungs/tiles) scattered over an
    `extent` x `extent` canvas. Returns `(points, cluster_id)` --
    `cluster_id` is for the figure's ground-truth colouring only, never fed to
    any of the three merges.
    """
    rng = np.random.default_rng(seed)
    centres = rng.uniform(0, extent, size=(n_clusters, 2))
    pts: List[np.ndarray] = []
    cid: List[int] = []
    for k, c in enumerate(centres):
        m = int(rng.integers(1, max_per_cluster + 1))
        pts.append(c + rng.normal(0.0, jitter, size=(m, 2)))
        cid += [k] * m
    return np.concatenate(pts, axis=0), np.asarray(cid, dtype=np.int64)


def _load_real_points(args) -> Tuple[np.ndarray, float, str]:
    """Real detections from one C-tree, one rung -- read the same way `cli/
    survival_alpha_analysis.py`'s C axis does. Returns `(points, radius,
    label)`; `radius` is `net.cfg.nms_radius`, the same value production
    passes to `anchors_of_generations` for this exact call, not a demo value.

    Deferred imports: none of this loads (no torch, no model) unless
    `--checkpoint` is actually given, so the synthetic-only invocation stays
    model-free.
    """
    import torch                                                  # noqa: PLC0415
    from SurvivalAnalysis import SurvivalProcess                    # noqa: PLC0415
    from reeval_density import load_arm                              # noqa: PLC0415

    if not args.wsi_name:
        raise SystemExit('--wsi-name is required with --checkpoint')

    entry = AccessDatasets.locate(args.wsi_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net, _identity, _ = load_arm(args.checkpoint, device)
    threshold = (args.score_threshold if args.score_threshold is not None
                else float(net.cfg.detection_threshold))

    c_corpus = prepare_chain_stack.axis_corpus('C', args)

    with SafeSlide(entry.path) as wsi:
        forest = ChainStack.CStack.from_own(
            c_corpus, entry.name, args.c_rungs, wsi, tile=args.tile,
            cache_root=chainstack_root(args))
        mother, mother_image, groups_by_ds, images_by_ds = forest[args.tree_index]
        _per_rung_tiles, per_rung = SurvivalProcess.detect_all_generations(
            mother, mother_image, groups_by_ds, images_by_ds, net,
            score_threshold=threshold)

    rung = args.real_rung if args.real_rung is not None else min(args.c_rungs)
    if rung not in per_rung:
        raise SystemExit(f'--real-rung {rung} not in --c-rungs {args.c_rungs}')
    xy0, _score = per_rung[rung]
    label = f'{entry.name}  C-tree {args.tree_index}  ds={rung:g}'
    return np.asarray(xy0, np.float64), float(net.cfg.nms_radius), label


def _run_real_flow_comparison(args) -> Dict[str, Tuple[Optional[dict], float,
                                                       int, List[np.ndarray]]]:
    """Deploy `original` / `B` / `A` into `cli/survival_alpha_analysis.py`'s
    own C-axis flow (`anchors_of_generations` -> `probe_real` ->
    `probe_decoy` -> `alpha_curve` -> `aggregate_curves`), monkey-patching
    `SurvivalProcess._merge_within_radius` for the duration of each candidate
    -- both `anchors_of_generations` and `AlphaCalibration.merge_anchors`
    look this name up on the `SurvivalProcess` module at CALL time, so
    replacing the module attribute redirects every caller without touching
    either file.

    Detection (`detect_all_generations`, the GPU part) runs ONCE per tree and
    is shared by all three candidates -- only the merge onward is repeated
    per candidate, both because that isolates what actually changed and
    because it is the only way this stays a demo and not three independent
    full analysis runs.

    `rng` is reseeded from `args.seed` before each candidate so the SAME
    decoy draw is used for all three -- any numeric difference in the
    aggregated curves can then only come from the merge implementation, not
    from a different random decoy.
    """
    import survival_alpha_analysis as saa                          # noqa: PLC0415
    import torch                                                    # noqa: PLC0415
    from SurvivalAnalysis import SurvivalProcess                    # noqa: PLC0415
    from reeval_density import load_arm                                # noqa: PLC0415

    if not args.wsi_name:
        raise SystemExit('--wsi-name is required with --real-flow')

    entry = AccessDatasets.locate(args.wsi_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net, _identity, _ = load_arm(args.checkpoint, device)
    threshold = (args.score_threshold if args.score_threshold is not None
                else float(net.cfg.detection_threshold))
    c_corpus = prepare_chain_stack.axis_corpus('C', args)
    order = sorted(float(r) for r in args.c_rungs)
    alphas = np.arange(0.5, 4.0 + 0.125, 0.25)     # survival_alpha_analysis.py's own default

    with SafeSlide(entry.path) as wsi:
        forest = ChainStack.CStack.from_own(
            c_corpus, entry.name, args.c_rungs, wsi, tile=args.tile,
            cache_root=chainstack_root(args))
        chainstacks = [forest[i] for i in forest][:args.real_flow_n_trees]

        tree_data = []
        tree_max_n = []
        for mother, mother_image, groups_by_ds, images_by_ds in chainstacks:
            per_rung_tiles, per_rung = SurvivalProcess.detect_all_generations(
                mother, mother_image, groups_by_ds, images_by_ds, net,
                score_threshold=threshold)
            tree_data.append((per_rung_tiles, per_rung))
            tree_max_n.append(max(len(xy0) for xy0, _sc in per_rung.values()))

    # `original` gets the SAME guard the other three comparisons already
    # have -- real C-tree point counts run into the tens of thousands (see
    # the [real data] section), where `original`'s extra O(n^2) rebuild cost
    # on top of B's O(n*k) can take far longer than B itself, per tree, with
    # no progress printed inside the merge to show it is still working.
    skip_original = any(n > args.skip_original_above for n in tree_max_n)
    if skip_original:
        print(f'  original skipped -- max {max(tree_max_n)} raw points in a '
             f'rung exceeds --skip-original-above {args.skip_original_above}',
             flush=True)

    saved_impl = SurvivalProcess._merge_within_radius
    candidates = {'original': merge_within_radius_original,
                 'B': None,
                 'A': merge_within_radius_grid}
    results: Dict[str, Tuple[Optional[dict], Optional[float], int,
                            List[np.ndarray]]] = {}
    try:
        for name, fn in candidates.items():
            if name == 'original' and skip_original:
                results[name] = (None, None, len(tree_data), [])
                continue
            SurvivalProcess._merge_within_radius = (fn if fn is not None
                                                    else saved_impl)
            rng = np.random.default_rng(args.seed)
            t0 = time.perf_counter()
            curves = []
            anchors_per_tree: List[np.ndarray] = []
            for per_rung_tiles, per_rung in tree_data:
                anchors, _ = SurvivalProcess.anchors_of_generations(
                    per_rung_tiles, order, net.cfg.nms_radius)
                anchors_per_tree.append(anchors)
                dist, score, _rival, _ = SurvivalProcess.probe_real(
                    anchors, per_rung, order)
                decoy_shift = saa._decoy_shift_per_rung(
                    'fixed', 8.0, order, 'C', rng)
                decoy_dist, decoy_score = saa.probe_decoy(
                    anchors, per_rung, order, decoy_shift)
                curves.append(saa.alpha_curve(
                    dist, score, decoy_dist, decoy_score, rungs=order,
                    alphas=alphas, tau_floor=0.0, threshold=threshold))
            dt = time.perf_counter() - t0
            bucket = saa.aggregate_curves(curves) if curves else None
            results[name] = (bucket, dt, len(tree_data), anchors_per_tree)
    finally:
        SurvivalProcess._merge_within_radius = saved_impl

    return results


def _anchor_mismatch(ref_anchors: List[np.ndarray], cmp_anchors: List[np.ndarray]
                     ) -> float:
    """Mean, across trees, of the symmetric difference between two
    candidates' anchor sets (rounded to 1e-6 -- these are deterministic
    floating-point results, so identical inputs through the identical
    algorithm produce identical values; rounding only guards against
    reordered-summation noise) as a fraction of the reference tree's anchor
    count. 0.0 means every anchor in every tree matched exactly, not just
    that the COUNTS matched.
    """
    fracs = []
    for a_ref, a_cmp in zip(ref_anchors, cmp_anchors):
        set_ref = set(map(tuple, np.round(a_ref, 6).tolist()))
        set_cmp = set(map(tuple, np.round(a_cmp, 6).tolist()))
        n_total = max(len(set_ref), 1)
        fracs.append(len(set_ref ^ set_cmp) / n_total)
    return float(np.mean(fracs)) if fracs else 0.0


def _plot_real_flow(anchors_by_candidate: Dict[str, np.ndarray],
                    curves_by_candidate: Dict[str, Optional[dict]],
                    order: List[float], label: str, out_path: str) -> None:
    """Two panels: the first tree's anchors from all three candidates
    OVERLAID (identical implementations put identical markers on top of each
    other -- a visible offset IS a disagreement), and the finest rung's
    match_rate vs alpha, same overlay logic, for the aggregated curves.
    """
    colours = {'original': 'steelblue', 'B': 'crimson', 'A': 'seagreen'}
    markers = {'original': 'x', 'B': 'o', 'A': '+'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    for name, anchors in anchors_by_candidate.items():
        ax1.scatter(anchors[:, 0], anchors[:, 1], s=24, c=colours[name],
                   marker=markers[name], alpha=0.6,
                   label=f'{name} ({len(anchors)})')
    ax1.set_title('anchors, first C-tree (overlaid)', fontsize=9)
    ax1.invert_yaxis()
    ax1.set_aspect('equal')
    ax1.legend(fontsize=7)

    finest = order[0]
    for name, bucket in curves_by_candidate.items():
        if bucket is None:
            continue
        ax2.plot(bucket['alphas'], bucket['match_rate_mean'][0, :],
                color=colours[name], marker=markers[name], ms=4,
                label=name)
    ax2.set_xlabel('alpha')
    ax2.set_ylabel('match_rate')
    ax2.set_title(f'match_rate vs alpha, finest rung ds={finest:g} (overlaid)',
                 fontsize=9)
    ax2.legend(fontsize=7)

    fig.suptitle(f'real flow -- {label}', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _draw_scatter_only(ax, points: np.ndarray, kept_idx: np.ndarray,
                       title: str) -> None:
    """Same colouring as `_draw_merge_grid_figure`, no per-point circles and
    no grid lines -- real detection counts run into the thousands, where
    circles for every kept point would be unreadable clutter rather than an
    aid.
    """
    ax.set_title(title, fontsize=9)
    dropped = np.setdiff1d(np.arange(len(points)), kept_idx)
    ax.scatter(points[dropped, 0], points[dropped, 1], s=4, c='lightgray',
              label=f'merged away ({len(dropped)})')
    ax.scatter(points[kept_idx, 0], points[kept_idx, 1], s=10, c='crimson',
              label=f'kept ({len(kept_idx)})', zorder=3)
    ax.set_aspect('equal')
    ax.invert_yaxis()                    # image row order, level-0 y grows down
    ax.legend(fontsize=7, loc='upper right', markerscale=2)


def _mismatch(a: np.ndarray, b: np.ndarray, n_total: int) -> Tuple[int, float]:
    """Symmetric difference between two KEPT index sets: how many points one
    implementation kept that the other dropped, or vice versa, as a count
    and a fraction of `n_total`. `len(a) == len(b)` is not enough -- two
    same-SIZE sets can still disagree on WHICH points survived, and a bare
    boolean `agree` says nothing about how far apart a disagreement is.
    """
    diff = len(set(a.tolist()) ^ set(b.tolist()))
    return diff, (diff / n_total if n_total else 0.0)


def _time_it(fn, points, radius, priority, repeats: int) -> Tuple[np.ndarray, float]:
    """Best-of-`repeats` wall time -- the min, not the mean, because a stray
    scheduler hiccup can only make a run SLOWER, never faster than its true
    cost.
    """
    best = None
    idx = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        idx = fn(points, radius, priority=priority)
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return idx, best


def _timing_multi_draw(args) -> bool:
    """`--repeats` INDEPENDENT synthetic draws -- a fresh seed every time, not
    the same data timed repeatedly (that would only measure scheduler noise
    on ONE layout). Each draw is timed once per implementation and checked
    for mismatch against B; the report is a timing DISTRIBUTION across draws
    plus a disagreement RATE (how many of the draws produced ANY mismatch),
    which a single boolean 'agree' from one draw cannot tell you.
    """
    times: Dict[str, List[float]] = {'B': [], 'A': [], 'original': []}
    mismatch_ba: List[float] = []
    mismatch_ob: List[float] = []
    n_skipped = 0

    for d in range(args.repeats):
        seed = args.seed + 1000 + d
        t_points, _ = synthetic_points(
            args.timing_n_clusters, args.max_per_cluster, args.jitter,
            args.timing_extent, seed)
        t_priority = np.random.default_rng(seed + 1).uniform(size=len(t_points))
        n_t = len(t_points)

        t0 = time.perf_counter()
        kept_b = merge_anchors(t_points, args.radius, priority=t_priority)
        times['B'].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        kept_a = merge_within_radius_grid(t_points, args.radius, priority=t_priority)
        times['A'].append(time.perf_counter() - t0)
        _, frac_ba = _mismatch(kept_b, kept_a, n_t)
        mismatch_ba.append(frac_ba)

        if n_t <= args.skip_original_above:
            t0 = time.perf_counter()
            kept_o = merge_within_radius_original(t_points, args.radius,
                                                  priority=t_priority)
            times['original'].append(time.perf_counter() - t0)
            _, frac_ob = _mismatch(kept_o, kept_b, n_t)
            mismatch_ob.append(frac_ob)
        else:
            n_skipped += 1

    def _stats(name: str) -> str:
        arr = np.asarray(times[name])
        return f'mean {arr.mean():.4f}s (std {arr.std():.4f}s)' if len(arr) else 'n/a'

    b_mean = float(np.mean(times['B']))
    a_mean = float(np.mean(times['A']))
    print(f'\n[timing set]  {args.repeats} independent draws, '
         f'~{args.timing_n_clusters} clusters each, radius {args.radius:g}, '
         f'clustering re-drawn (different seed) every repeat')
    print(f'  B (array buffer, retired) -> {_stats("B")}')
    print(f'  A (grid, production)         -> {_stats("A")}   '
         f'({(b_mean / a_mean if a_mean > 0 else float("inf")):.1f}x vs B, '
         f'mean over draws)')
    n_disagree_ba = sum(1 for x in mismatch_ba if x > 0)
    print(f'  B vs A mismatch: mean {np.mean(mismatch_ba):.2%} of kept '
         f'points, disagreement in {n_disagree_ba}/{args.repeats} draws')

    agree = (n_disagree_ba == 0)
    if times['original']:
        o_mean = float(np.mean(times['original']))
        b_mean_paired = float(np.mean(times['B'][:len(times['original'])]))
        print(f'  original                      -> {_stats("original")}   '
             f'({(o_mean / b_mean_paired if b_mean_paired > 0 else float("inf")):.1f}'
             f'x slower than B)')
        n_disagree_ob = sum(1 for x in mismatch_ob if x > 0)
        print(f'  B vs original mismatch: mean {np.mean(mismatch_ob):.2%} of '
             f'kept points, disagreement in {n_disagree_ob}/{len(mismatch_ob)} draws')
        agree = agree and (n_disagree_ob == 0)
    if n_skipped:
        print(f'  original skipped on {n_skipped}/{args.repeats} draws -- '
             f'point count exceeded --skip-original-above '
             f'{args.skip_original_above}')

    return agree


def _draw_merge_grid_figure(ax, points: np.ndarray, kept_idx: np.ndarray,
                            radius: float, extent: float, title: str) -> None:
    ax.set_title(title, fontsize=9)
    dropped = np.setdiff1d(np.arange(len(points)), kept_idx)
    ax.scatter(points[dropped, 0], points[dropped, 1], s=14, c='lightgray',
              label=f'merged away ({len(dropped)})')
    ax.scatter(points[kept_idx, 0], points[kept_idx, 1], s=28, c='crimson',
              marker='*', label=f'kept ({len(kept_idx)})', zorder=3)
    for x, y in points[kept_idx]:
        ax.add_patch(plt.Circle((x, y), radius, fill=False, color='crimson',
                                alpha=0.25, lw=0.8))
    n_cells = int(np.ceil(extent / radius)) + 1
    for k in range(n_cells + 1):
        ax.axvline(k * radius, color='steelblue', lw=0.4, alpha=0.35)
        ax.axhline(k * radius, color='steelblue', lw=0.4, alpha=0.35)
    ax.set_xlim(-radius, extent + radius)
    ax.set_ylim(-radius, extent + radius)
    ax.set_aspect('equal')
    ax.legend(fontsize=7, loc='upper right')


def _run_merge_grid(args, out_dir: str) -> int:
    # ── small set: figure + correctness ──────────────────────────────────────
    points, _cluster_id = synthetic_points(
        args.n_clusters, args.max_per_cluster, args.jitter, args.extent,
        args.seed)
    priority = np.random.default_rng(args.seed + 1).uniform(size=len(points))

    kept_orig = merge_within_radius_original(points, args.radius, priority=priority)
    kept_ref = merge_anchors(points, args.radius, priority=priority)
    kept_grid = merge_within_radius_grid(points, args.radius, priority=priority)

    n_diff_ob, frac_ob = _mismatch(kept_orig, kept_ref, len(points))
    n_diff_ba, frac_ba = _mismatch(kept_ref, kept_grid, len(points))
    agree_small = (n_diff_ob == 0 and n_diff_ba == 0)
    print(f'[figure set]  {len(points)} raw points, {args.n_clusters} true '
         f'clusters, radius {args.radius:g}')
    print(f'  original -> {len(kept_orig)} kept')
    print(f'  B (array buffer, retired) -> {len(kept_ref)} kept')
    print(f'  A (grid, production)  -> {len(kept_grid)} kept')
    print(f'  mismatch  orig-vs-B: {n_diff_ob} pts ({frac_ob:.2%})   '
         f'B-vs-A: {n_diff_ba} pts ({frac_ba:.2%})')
    print(f'  all three agree: {agree_small}')

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
    _draw_merge_grid_figure(axes[0], points, kept_orig, args.radius, args.extent,
                           f'original (list rebuild)\n{len(kept_orig)} anchors')
    _draw_merge_grid_figure(axes[1], points, kept_ref, args.radius, args.extent,
                           f'B: production (array buffer)\n{len(kept_ref)} anchors')
    _draw_merge_grid_figure(axes[2], points, kept_grid, args.radius, args.extent,
                           f'A: grid\n{len(kept_grid)} anchors, agree={agree_small}')
    fig.suptitle('_merge_within_radius: three implementations, same input',
                fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out_path = os.path.join(out_dir, 'merge_grid_demo.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved  {out_path}')

    # ── large set: timing distribution + disagreement rate, not plotted ─────
    agree_large = _timing_multi_draw(args)

    # ── real data: timing + agreement, plotted only if asked ────────────────
    agree_real = True
    if args.checkpoint:
        print(f'\n[real data]  {args.wsi_name}  C-tree {args.tree_index}  '
             f'detecting...', flush=True)
        r_points, r_radius, r_label = _load_real_points(args)
        n_r = len(r_points)
        print(f'[real data]  {r_label}  {n_r} raw detections, '
             f'radius {r_radius:g} (net.cfg.nms_radius), '
             f'best of {args.repeats}')

        kept_b_r, dt_b_r = _time_it(
            lambda p, r, priority: merge_anchors(p, r, priority=priority),
            r_points, r_radius, None, args.repeats)
        print(f'  B (array buffer, retired) -> {len(kept_b_r)} kept   '
             f'{dt_b_r:.4f}s')

        kept_grid_r, dt_grid_r = _time_it(merge_within_radius_grid, r_points,
                                          r_radius, None, args.repeats)
        speedup = f'{dt_b_r / dt_grid_r:.1f}x vs B' if dt_grid_r > 0 else 'too fast to time'
        print(f'  A (grid, production)         -> {len(kept_grid_r)} kept   '
             f'{dt_grid_r:.4f}s   ({speedup})')

        n_diff_ba_r, frac_ba_r = _mismatch(kept_b_r, kept_grid_r, n_r)
        agree_real = (n_diff_ba_r == 0)
        print(f'  B vs A mismatch: {n_diff_ba_r} pts ({frac_ba_r:.2%})')

        if n_r <= args.skip_original_above:
            kept_orig_r, dt_orig_r = _time_it(
                merge_within_radius_original, r_points, r_radius, None,
                args.repeats)
            slower = (f'{dt_orig_r / dt_b_r:.1f}x slower than B'
                     if dt_b_r > 0 else 'n/a')
            print(f'  original                      -> {len(kept_orig_r)} '
                 f'kept   {dt_orig_r:.4f}s   ({slower})')
            n_diff_ob_r, frac_ob_r = _mismatch(kept_orig_r, kept_b_r, n_r)
            print(f'  B vs original mismatch: {n_diff_ob_r} pts ({frac_ob_r:.2%})')
            agree_real = agree_real and (n_diff_ob_r == 0)
            print(f'  all three agree: {agree_real}')
        else:
            print(f'  original skipped -- {n_r} points exceeds '
                 f'--skip-original-above {args.skip_original_above}')

        if args.plot_real:
            fig_r, axes_r = plt.subplots(1, 2, figsize=(11, 5.5))
            _draw_scatter_only(axes_r[0], r_points, kept_b_r,
                              f'B: production\n{len(kept_b_r)} anchors')
            _draw_scatter_only(axes_r[1], r_points, kept_grid_r,
                              f'A: grid\n{len(kept_grid_r)} anchors, '
                              f'agree={agree_real}')
            fig_r.suptitle(f'real data -- {r_label}', fontsize=11)
            fig_r.tight_layout(rect=(0, 0, 1, 0.94))
            real_out = os.path.join(out_dir, 'merge_grid_demo_real.png')
            fig_r.savefig(real_out, dpi=150, bbox_inches='tight')
            plt.close(fig_r)
            print(f'Saved  {real_out}')

    # ── real flow: deploy each implementation, compare the aggregated result ─
    agree_flow = True
    if args.real_flow:
        print(f'\n[real flow]  {args.wsi_name}  up to '
             f'{args.real_flow_n_trees} C-trees  (survival_alpha_analysis.py'
             f'\'s own C-axis flow, each implementation deployed in turn)')
        flow_results = _run_real_flow_comparison(args)
        ref_bucket, ref_dt, ref_n, ref_anchors = flow_results['B']
        print(f'  B (retired)     -> {ref_n} ChainStacks   {ref_dt:.4f}s   '
             f'{sum(len(a) for a in ref_anchors)} anchors total')
        for name in ('original', 'A'):
            bucket, dt, n_trees, anchors_per_tree = flow_results[name]
            if dt is None:
                print(f'  {name:8s}        -> skipped (see above)')
                continue
            speed_note = f'{ref_dt / dt:.1f}x vs B' if dt > 0 else 'n/a'
            print(f'  {name:8s}        -> {n_trees} ChainStacks   {dt:.4f}s'
                 f'   ({speed_note})')

            anchor_mismatch = _anchor_mismatch(ref_anchors, anchors_per_tree)
            print(f'    anchor mismatch vs B: mean {anchor_mismatch:.2%} '
                 f'across {len(ref_anchors)} trees')

            if ref_bucket is None or bucket is None:
                same = (ref_bucket is None) == (bucket is None)
                print(f'    aggregated curves match B: {same}')
            else:
                keys = ('match_rate_mean', 'decoy_rate_mean', 'gap_mean',
                       'margin_mean')
                max_diffs = {k: float(np.nanmax(np.abs(ref_bucket[k] - bucket[k])))
                            for k in keys}
                same = all(np.allclose(ref_bucket[k], bucket[k], equal_nan=True)
                          for k in keys)
                print(f'    aggregated curves match B: {same}   max abs diff: '
                     + ', '.join(f'{k.replace("_mean", "")}={v:.2e}'
                                for k, v in max_diffs.items()))
            agree_flow = agree_flow and same and (anchor_mismatch == 0.0)
        print(f'  all three match: {agree_flow}')

        anchors_by_candidate = {name: flow_results[name][3][0]
                                for name in ('original', 'B', 'A')
                                if flow_results[name][3]}
        curves_by_candidate = {name: flow_results[name][0]
                               for name in ('original', 'B', 'A')}
        if anchors_by_candidate:
            order = sorted(float(r) for r in args.c_rungs)
            flow_out = os.path.join(out_dir, 'merge_grid_demo_real_flow.png')
            _plot_real_flow(anchors_by_candidate, curves_by_candidate, order,
                           f'{args.wsi_name}, first C-tree', flow_out)
            print(f'Saved  {flow_out}')

    return 0 if (agree_small and agree_large and agree_real and agree_flow) else 1


# =============================================================================
#  visualize -- synthetic, step-by-step GIFs for a PPT: keypoints -> anchor
#  merge -> alive.
# =============================================================================
#
# SYNTHETIC, NOT REAL DETECTIONS, AND DELIBERATELY NOT RANDOM EITHER. The
# point is to show the GEOMETRY/LOGIC clearly (an exact merge radius, an
# exact tau crossing), which real data's own noise would obscure -- and
# because this file cannot be executed to iterate on (ClaudeRules 2: only
# `python -m py_compile` runs directly here), every distance in the scene
# below is chosen by hand with enough margin that it cannot flip a point
# into the wrong pattern, rather than drawn from an RNG and hoped to land
# right. `_run_visualize` re-checks the hand-derived geometry against the
# REAL `SurvivalProcess.anchors_of` before trusting any of it (see its own
# comment) -- a wrong picture should fail loudly, not render.
#
# SIX POINTS, ONE PER `Patterns.PATTERNS` ENTRY: which rungs each one is
# "visible" at was chosen so `Patterns.classify` on the resulting alive
# vector lands on that point's OWN name -- the six-pattern legend at the end
# of the alive animation is therefore a real classification result, not a
# label pasted on afterward. Placed >=120 level-0 px apart, more than double
# the widest radius (tau or merge) this demo ever compares against, so one
# point's own jitter can never be mistaken for a different point's.

_VIZ_ORDER = (1.0, 2.0, 4.0, 8.0)          # finest first, C/F-axis convention
_VIZ_FOOTPRINT = 32.0

#: name -> which rungs that point is "visible" (detected) at. Positions are
#: assigned separately, below, by `_place_viz_points` -- kept apart from
#: this dict because WHICH rung a point lives at is a fixed, semantic
#: choice (it is what makes `Patterns.classify` land on that point's own
#: name), while WHERE it sits is not semantic at all and can be laid out
#: however reads best.
_VIZ_PATTERN_RUNGS = {
    '一直存活': (1.0, 2.0, 4.0, 8.0),
    '細部存活': (1.0, 2.0),
    '晚生型':   (4.0, 8.0),
    '只在一階': (2.0,),
    '中間帶':   (2.0, 4.0),
    '不連續':   (1.0, 4.0),
}


def _place_viz_points() -> Dict[str, Tuple[float, float]]:
    """name -> level-0 xy for the six points, via rejection sampling with a
    FIXED seed -- less rigid than a hand-picked grid ("could be more randomly
    distributed"), while staying reproducible
    across runs of this file (a fresh, ungoverned random draw every run
    would make the GIF a different scene each time, which is not what a
    PPT asset should be).

    `_TARGET_MIN_DIST=9.0` is a GENEROUS placement target, not the safety
    bound itself -- six points needing pairwise distance >=9 fit easily
    inside a 26x26 usable square (packing theory's own ceiling for 6 points
    in a square that size is close to 16), so this succeeds in a handful of
    attempts almost always, with a large attempt budget as a backstop
    rather than a real expectation of needing it. The ACTUAL safety bound
    is computed AFTER this runs, from whatever it actually produced (see
    `_VIZ_MIN_SPACING`/`_VIZ_ALPHA_DEFAULT` below and `_run_visualize`'s own
    runtime check) -- this function does not need to hit an exact number,
    only to not produce two points absurdly close together.
    """
    rng = np.random.default_rng(20260911)
    margin, target_min_dist = 3.0, 9.0
    names = list(_VIZ_PATTERN_RUNGS)
    placed: List[np.ndarray] = []
    for _ in names:
        for _attempt in range(5000):
            candidate = rng.uniform(margin, _VIZ_FOOTPRINT - margin, size=2)
            if all(float(np.hypot(*(candidate - p))) >= target_min_dist
                  for p in placed):
                placed.append(candidate)
                break
        else:
            raise RuntimeError(
                '_place_viz_points could not place all six points at '
                f'target_min_dist={target_min_dist} inside a '
                f'{_VIZ_FOOTPRINT:g}x{_VIZ_FOOTPRINT:g} canvas -- widen the '
                'canvas or loosen target_min_dist')
    return {name: (float(xy[0]), float(xy[1]))
           for name, xy in zip(names, placed)}


#: name -> (true level-0 xy, the rungs it is "visible" -- i.e. detected --
#: at). Positions from `_place_viz_points` (seeded random, see its own
#: docstring), rung visibility from `_VIZ_PATTERN_RUNGS` (fixed, semantic).
_VIZ_POSITIONS = _place_viz_points()
_VIZ_TRUE_POINTS = {name: (_VIZ_POSITIONS[name], rungs)
                   for name, rungs in _VIZ_PATTERN_RUNGS.items()}

#: The actual closest-pair distance THIS run's placement produced, and the
#: `--viz-alpha` ceiling that follows from it -- computed here, once, from
#: whatever `_VIZ_TRUE_POINTS` actually is, rather than a hand-typed
#: literal that has to be remembered and re-typed by hand every time the
#: scene changes, and drifts out of sync with it. Deriving the default FROM
#: the scene makes that impossible: the
#: default is always exactly as large as this scene allows, whatever this
#: scene currently is.
_VIZ_MIN_SPACING = min(
    float(np.hypot(xy_a[0] - xy_b[0], xy_a[1] - xy_b[1]))
    for (xy_a, _), (xy_b, _) in itertools.combinations(
        _VIZ_TRUE_POINTS.values(), 2))
#: 1.5x safety margin, same factor `_run_visualize`'s own check uses; 0.9x
#: on top of that so the DEFAULT sits under the hard ceiling with room to
#: spare, not exactly on it.
_VIZ_ALPHA_DEFAULT = round(
    _VIZ_MIN_SPACING / 1.5 / max(_VIZ_ORDER) * 0.9, 2)

#: Rung colour AND marker shape, finest to coarsest -- separate palette from
#: `survival_alpha_analysis.PATTERN_COLORS` (duplicated below, not imported,
#: so a synthetic-only `--parts visualize` run never pulls in that file's
#: heavier import chain) since "which rung" and "which pattern" are two
#: different questions a viewer needs to tell apart in the same figure. A
#: SHAPE per rung, not just a colour, is what actually solves "every point
#: looks exactly overlapped": a rung's own
#: jitter from its neighbour is a fraction of a level-0 px by design (the
#: whole reason a merge RADIUS is needed instead of exact-match), so no
#: marker size fix alone makes two near-coincident dots read as two --
#: drawn UNFILLED (`facecolors='none'`), overlapping markers show as
#: overlapping OUTLINES/rings instead of one solid blob.
#: 16.0 included even though the synthetic scene only uses up to ds=8 --
#: `_run_visualize_real_example` reuses these two dicts for REAL data,
#: whose `--c-rungs` default includes 16.0.
_RUNG_COLORS = {1.0: '#8B5CF6', 2.0: '#3B82F6', 4.0: '#10B981', 8.0: '#F59E0B',
                16.0: '#E23670'}
_RUNG_MARKERS = {1.0: 'o', 2.0: '^', 4.0: 's', 8.0: 'D', 16.0: 'P'}

#: Same six colours as the "六態存活譜" reference figure
#: (claude.ai/code/artifact/501f65f3-bd59-4a04-92ca-958602f85270) --
#: duplicated from `survival_alpha_analysis.PATTERN_COLORS` rather than
#: imported for the same synthetic-stays-light reason as `_RUNG_COLORS`.
_VIZ_PATTERN_COLORS = {
    '一直存活': '#1D4E77', '細部存活': '#2B7A8C', '晚生型': '#4E9C6C',
    '只在一階': '#94BD3E', '中間帶': '#CE9A24', '不連續': '#C0453A',
}

def _marker_area(target_radius_units: float, data_range: float,
                 fig_inches: float) -> float:
    """matplotlib scatter `s` (points^2) for a marker whose ON-SCREEN
    RADIUS is `target_radius_units` DATA units, given the axes' own data
    RANGE and the figure's PHYSICAL width in inches -- derived, not a bare
    `s=N` literal, because `s` is a fixed physical size (points, 1/72 inch)
    that does NOT automatically track a change in axes range or figure
    size, and `visualize_real_example.png`'s crop has a user-set range
    (`--viz-real-crop`) -- this formula self-adjusts to
    any figure/range combination instead of needing to be re-tuned by hand
    every time either changes.

    `points_per_unit = fig_inches / data_range * 72` (a matplotlib point is
    1/72 inch); `s = pi * (target_radius_units * points_per_unit)^2`.
    """
    points_per_unit = fig_inches / data_range * 72.0
    return float(np.pi * (target_radius_units * points_per_unit) ** 2)


#: Target on-screen radii, in DATA UNITS of the 32-wide synthetic canvas --
#: 0.5/0.7/1.1 chosen against `_place_viz_points`'s own `target_min_dist=
#: 9.0` (comfortably smaller than half that, so two DIFFERENT points' own
#: markers never touch) and `_viz_offset`'s own max magnitude (~2.0 at
#: ds=8, so a MED marker's ~0.7 radius stays smaller than a same-point
#: cross-rung offset, letting overlapping outlines still read as two).
#: `_SYNTH_FIG_IN=6.3` is the representative single-panel figure width used
#: across `_animate_keypoints`/`_animate_anchor_merge`/`_animate_alive`
#: (whose two side-by-side panels are close enough to this per-panel width
#: not to need a separate constant).
_SYNTH_FIG_IN = 6.3
#: 0.5 = diameter of 1 level-0 px (radius 0.5) -- all three set to the SAME
#: "one pixel" size on request; split them apart again (e.g. make LARGE
#: bigger for anchors) if the uniform size reads as hard to tell apart.
_VIZ_MARKER_SMALL = _marker_area(0.25, _VIZ_FOOTPRINT, _SYNTH_FIG_IN)
_VIZ_MARKER_MED = _marker_area(0.3, _VIZ_FOOTPRINT, _SYNTH_FIG_IN)
_VIZ_MARKER_LARGE = _marker_area(0.35, _VIZ_FOOTPRINT, _SYNTH_FIG_IN)


def _viz_offset(ds: float, point_index: int) -> np.ndarray:
    """A per-(point, rung) offset standing in for real detection noise /
    downsampling quantisation -- deterministic (not `np.random`), UNLIKE
    `_place_viz_points`: the offset has to stay under the real merge
    radius EXACTLY, a bound this file cannot re-roll and check by running,
    so it is worked through by hand once here rather than sampled per run.

    Magnitude `0.25*ds` is pushed as close to that bound as a comfortable
    margin allows ("pull the cross-rung jitter
    apart more -- radius is tied to ds/2, there's room"). Checked for
    every rung PAIR (not just the finest, since the merge visits
    finest-first and a later rung always compares against whatever
    survived, not necessarily its immediate neighbour): worst case for a
    pair `(ds_i, ds_j)` is `0.25*ds_i + 0.25*ds_j` (triangle inequality,
    two offsets pointing straight at each other) against a merge radius of
    `max(ds_i,ds_j)//2`. Every "adjacent" pair (`(1,2)`, `(2,4)`, `(4,8)`)
    ties for tightest at a 1.33x margin (`0.25+0.5=0.75` vs radius `1.0`,
    etc. -- both sides scale together, so the ratio repeats); every other
    pair has more room (`(1,8)`: 1.78x). `_run_visualize`'s own comparison
    against the real `SurvivalProcess.anchors_of` is the backstop this
    relies on in addition to the hand check: if this margin were ever cut
    too fine, a same-point cluster failing to fully merge would show up as
    more than 6 anchors and raise `AssertionError` there, not render
    silently.

    THE ANGLE MATTERS AS MUCH AS THE MAGNITUDE:
    the margin above is a TRIANGLE-INEQUALITY upper bound (`m_i+m_j`, two
    offsets pointing straight at each other) -- it says the offsets CANNOT
    exceed the radius, not that they are actually spread apart. An angle of
    `ds*0.05` differs by only 0.05-0.4 RADIANS between ds=1..8, so offsets
    would point in nearly the SAME direction (`|m_i-m_j|`, near 0). The angle
    is keyed to `_VIZ_ORDER`'s INDEX, not `ds` itself: a fixed
    90-degree step per rung spreads any two rungs at least 90 degrees
    apart, which puts the realised distance near
    `sqrt(m_i^2+m_j^2)` (perpendicular) up to `m_i+m_j` (opposite,
    non-adjacent rung pairs) -- both comfortably bigger, and
    the safety bound above (which never assumed a particular angle) still
    holds exactly as derived.
    """
    ds_index = _VIZ_ORDER.index(ds)
    angle = point_index * 1.3 + ds_index * (np.pi / 2.0)
    mag = 0.25 * ds
    return mag * np.array([np.cos(angle), np.sin(angle)])


def _synthetic_scene():
    """`(points_by_rung, label_by_rung, score_by_rung)`, each keyed by ds.
    `points_by_rung[ds]` is `[n_ds, 2]` -- every `_VIZ_TRUE_POINTS` entry
    "visible" at `ds`, offset by `_viz_offset`. `score_by_rung[ds]` is a
    flat 0.85 for every detection: this demo is entirely about tau/merge
    GEOMETRY, not the score threshold, so score is held constant on purpose.
    """
    points_by_rung = {ds: [] for ds in _VIZ_ORDER}
    label_by_rung = {ds: [] for ds in _VIZ_ORDER}
    for k, (pattern, (xy, rungs)) in enumerate(_VIZ_TRUE_POINTS.items()):
        xy = np.asarray(xy, np.float64)
        for ds in rungs:
            points_by_rung[ds].append(xy + _viz_offset(ds, k))
            label_by_rung[ds].append(pattern)
    points_by_rung = {ds: (np.stack(pts) if pts else np.zeros((0, 2)))
                      for ds, pts in points_by_rung.items()}
    score_by_rung = {ds: np.full(len(points_by_rung[ds]), 0.85)
                     for ds in _VIZ_ORDER}
    return points_by_rung, label_by_rung, score_by_rung


def _place_viz_noise_peaks(seed: int, n: int = 10) -> np.ndarray:
    """`[n, 2]` -- scattered background peak centres, standing in for a
    REAL raw probability map's own clutter (noise, background texture,
    weaker near-duplicate responses) that NMS and the score threshold
    throw away before a coordinate ever becomes one of `detect()`'s
    keypoints (`SurvivalProcess._detect_one` runs NMS on `prob` AFTER the
    network produces it -- the map candidate 1 reads is that PRE-NMS
    field, not the sparse post-NMS result). Without any noise the field
    would be the post-NMS picture -- clean bumps and nothing else -- which
    does not show what candidate 1 actually has to pick a winner out of.

    `seed` differs PER RUNG (`_VIZ_NOISE_PEAKS` below) -- a real network
    forward pass produces an independent field at every rung, so the same
    noise layout repeating at all four ds would misrepresent that as one
    fixed pattern. Still a fixed seed per rung, not `np.random`
    freshly drawn per run, for the same reproducibility reason `_place_
    viz_points` gives.

    Needs NO keep-out geometry the way `_place_viz_points`/`_viz_offset`
    do: `_VIZ_NOISE_AMPLITUDE` (below) caps every noise bump safely under
    `score_threshold=0.5` regardless of where it lands, so a noise peak
    can never flip an alive/dead decision no matter how close it sits to
    a real anchor -- correctness comes from the AMPLITUDE cap, not from
    where these are placed.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(2.0, _VIZ_FOOTPRINT - 2.0, size=(n, 2))


#: One INDEPENDENT noise layout per rung (different seed each) -- see
#: `_place_viz_noise_peaks`'s own docstring for why sharing one set across
#: all four ds would be wrong.
_VIZ_NOISE_PEAKS = {ds: _place_viz_noise_peaks(2026091201 + i)
                   for i, ds in enumerate(_VIZ_ORDER)}
#: Comfortably under `score_threshold=0.5` (used throughout `_animate_
#: alive`) with margin to spare, so these can never read as alive no
#: matter how close to an anchor they land.
_VIZ_NOISE_AMPLITUDE = 0.32


def _synthetic_probability_field(points: np.ndarray, *,
                                 noise_peaks: Optional[np.ndarray] = None,
                                 sigma: float = 1.5,
                                 size: int = int(_VIZ_FOOTPRINT)
                                 ) -> np.ndarray:
    """`[size, size]`, `field[y, x]` -- MAX (not sum) of a Gaussian bump per
    point, so each bump's own peak stays exactly 1.0 regardless of how many
    other bumps exist, instead of two nearby bumps inflating each other's
    height. `[y, x]` indexing matches `AliveCandidates.probe_via_
    probability_map`'s own `h, w = combined_map.shape` / `map_[y0, x0]`
    convention exactly -- checked against that function's body, not assumed.

    `sigma=1.5` (a crisp bump, not a wide soft one) is load-bearing: the
    nearest two true points are 11 apart and the widest alive window
    (`--viz-alpha 0.75 * ds=8 = tau 6`) can reach 6 from an anchor, so a
    neighbour's bump has to have decayed to nothing by 11-6=5 away from ITS
    OWN centre or a `probe_via_probability_map` window could pick up a
    neighbour's response instead of "nothing here". At `sigma=1.5`, `d=5`
    gives `exp(-25/4.5) ~= 4e-3` -- nowhere near `score_threshold=0.5`. A
    wider sigma would also be the "too uniform, no clear peaks" heatmap:
    on a 32-wide canvas a `sigma=2.5`+
    bump is wide enough to wash out most of the frame.

    `noise_peaks`, if given (a caller passes `_VIZ_NOISE_PEAKS[ds]` -- this
    function does not default to any particular rung's own set, since it
    has no way to know which rung `points` came from), additionally folds
    those in at `_VIZ_NOISE_AMPLITUDE` -- background clutter a real PRE-NMS
    probability map would have, capped low enough to
    never change an alive/dead decision (see `_place_viz_noise_peaks`'s
    own docstring). Real `points` always win the max-combine regardless of
    proximity, since `_VIZ_NOISE_AMPLITUDE < 1.0`.
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    field = np.zeros((size, size), np.float64)
    for x, y in points:
        bump = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma ** 2))
        field = np.maximum(field, bump)
    if noise_peaks is not None:
        for x, y in noise_peaks:
            bump = _VIZ_NOISE_AMPLITUDE * np.exp(
                -((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma ** 2))
            field = np.maximum(field, bump)
    return field


def _trace_cross_rung_merge(points: np.ndarray, rung_id: np.ndarray,
                            rung_scale: np.ndarray, base: float = 0.0
                            ) -> Tuple[List[dict], List[int]]:
    """Step-by-step trace of exactly `SurvivalProcess._merge_within_
    radius`'s cross-rung algorithm (same grid-hash, same finest-first
    visiting order `points` is already sorted in, same radius formula) --
    kept as a SEPARATE copy for animation only, the same reason this file's
    own `merge_grid` part keeps `merge_within_radius_grid` as an
    independent copy: so a future edit to the real function that silently
    diverges from this one shows up as a WRONG PICTURE, not a correct
    picture animating the wrong thing. `_run_visualize` checks this trace's
    own kept set against the real `SurvivalProcess.anchors_of` before
    trusting it for a GIF -- not just assumed to match.

    Returns `(frames, kept)`. Each frame is `{'i', 'kept_so_far', 'alive',
    'radius', 'rival'}` -- `i` the point just decided, `kept_so_far` the
    anchor indices BEFORE this decision, `alive` whether it survived,
    `radius` the radius it was checked against, `rival` the already-kept
    point that merged it away (`None` if it survived or had nothing to
    check against).
    """
    n = len(points)
    cell = max(float(base) + float(np.max(rung_scale)) // 2.0, 1.0)
    grid: Dict[Tuple[int, int], List[int]] = {}
    kept: List[int] = []
    frames: List[dict] = []
    for i in range(n):
        cx = int(np.floor(points[i, 0] / cell))
        cy = int(np.floor(points[i, 1] / cell))
        near = False
        rival = None
        radius = 0.0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((cx + dx, cy + dy), ()):
                    if rung_id[j] == rung_id[i]:
                        continue
                    r = float(base) + max(rung_scale[i], rung_scale[j]) // 2.0
                    if np.linalg.norm(points[j] - points[i]) <= r:
                        near, rival, radius = True, j, r
                        break
                if near:
                    break
            if near:
                break
        frames.append({'i': i, 'kept_so_far': list(kept), 'alive': not near,
                       'radius': radius, 'rival': rival})
        if not near:
            kept.append(i)
            grid.setdefault((cx, cy), []).append(i)
    return frames, kept


def _save_frame_png(fig, frame_dir: Optional[str], k: int) -> None:
    """Shared by all three `_animate_*` functions: if `frame_dir` is given,
    snapshot the CURRENT figure state as its own PNG -- called from inside
    each animation's `frame(k)` callback, right after that frame's drawing,
    so the standalone PNGs are pixel-identical to what the GIF shows at
    that step (one drawing routine, two outputs), never a second, possibly-
    drifted render path.
    """
    if frame_dir:
        fig.savefig(os.path.join(frame_dir, f'frame_{k:02d}.png'), dpi=140)


def _draw_rung_points(ax, pts: np.ndarray, ds: float, *, label: bool = True,
                      size: Optional[float] = None) -> None:
    """One rung's raw points, UNFILLED (`facecolors='none'`) with a shape+
    colour unique to `ds` (`_RUNG_MARKERS`/`_RUNG_COLORS`) -- unfilled so
    that two near-coincident points (a rung's own jitter from its
    neighbour, deliberately small -- see `_viz_offset`) still read as two
    overlapping OUTLINES rather than one solid blob, which a filled marker
    of any size cannot show.

    `size` defaults to `_VIZ_MARKER_MED` (sized for the 32-wide synthetic
    canvas) -- callers on a DIFFERENT data range (`_run_visualize_real_
    example`'s crop, a user-set `--viz-real-crop`) must pass their own
    `_marker_area(...)` result instead, or markers sized for one canvas
    render at the wrong physical size on another.
    """
    if not len(pts):
        return
    ax.scatter(pts[:, 0], pts[:, 1], s=(size if size is not None
                                       else _VIZ_MARKER_MED),
              marker=_RUNG_MARKERS.get(ds, 'o'), facecolors='none',
              edgecolors=_RUNG_COLORS.get(ds, '0.5'), linewidths=1.4,
              zorder=3, label=(f'ds={ds:g} detection' if label else None))


def _animate_keypoints(points_by_rung: Dict[float, np.ndarray], out_path: str,
                       fps: float, frame_dir: Optional[str] = None) -> None:
    """(b) -- each rung's own detections fade in, one rung per frame, all in
    the SAME fixed level-0 coordinate frame: the picture behind "convert
    every rung's keypoints into one shared coordinate system and overlay
    them", before any merging happens. Legend spells out shape+colour ->
    rung explicitly.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    def frame(k):
        ax.clear()
        ax.set_xlim(0, _VIZ_FOOTPRINT)
        ax.set_ylim(_VIZ_FOOTPRINT, 0)
        ax.set_aspect('equal')
        for ds in _VIZ_ORDER[:k + 1]:
            _draw_rung_points(ax, points_by_rung[ds], ds)
        ax.legend(loc='upper right', fontsize=7)
        ax.set_title('raw detections per rung, overlaid in one coordinate '
                     f'frame\nadded ds={_VIZ_ORDER[k]:g}', fontsize=10)
        _save_frame_png(fig, frame_dir, k)

    anim = FuncAnimation(fig, frame, frames=len(_VIZ_ORDER))
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def _viz_merge_legend() -> List[Line2D]:
    """Proxy handles shared by every frame of `_animate_anchor_merge` --
    built once, not re-derived per frame: raw shapes, the kept star's colour,
    the search-radius circle and the merged-away mark, all spelled out.
    """
    handles = [Line2D([0], [0], marker=_RUNG_MARKERS[ds], color='w',
                      markerfacecolor='none', markeredgecolor=_RUNG_COLORS[ds],
                      markersize=7, label=f'ds={ds:g} raw point')
              for ds in _VIZ_ORDER]
    handles.append(Line2D([0], [0], marker='*', color='w',
                          markerfacecolor='0.6', markeredgecolor='0.3',
                          markersize=10,
                          label='anchor (fill colour = contributing rung)'))
    handles.append(Line2D([0], [0], marker='x', color='0.65', linestyle='None',
                          markersize=7, label='merged away'))
    handles.append(Line2D([0], [0], color='crimson', lw=1.2,
                          label='radius being checked right now'))
    return handles


def _animate_anchor_merge(all_pts: np.ndarray, rung_id: np.ndarray,
                          frames: List[dict], out_path: str, fps: float,
                          frame_dir: Optional[str] = None) -> None:
    """(c) -- THREE PHASES ("draw one figure per
    ds first, then one figure with everything overlaid" before the merge
    itself): (1) each rung's raw points fade in on their own, one rung per
    frame (same picture as `_animate_keypoints`, repeated here so this GIF
    stands on its own); (2) one frame with all rungs overlaid, nothing
    decided yet; (3) the greedy grid-hash cross-rung merge itself, one
    point decided per frame -- the candidate's merge radius drawn as the
    CIRCLE it actually is (the grid-hash's 3x3-neighbour cell check is the
    same physical test, just a faster way to ask it -- see
    `_merge_within_radius`'s own docstring). A kept point becomes a STAR
    coloured by WHICH RUNG it came from (`rung_id`), not a single uniform
    colour, so it is visible which rung became the anchor.
    """
    # Wider than tall so the legend has its own space to the right of the
    # coordinate axes instead of sitting inside it: an in-axes legend covers
    # real points on this small canvas.
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    fig.subplots_adjust(right=0.72)
    n = len(all_pts)
    n_reveal = len(_VIZ_ORDER)
    n_decide = len(frames)
    n_total = n_reveal + 1 + n_decide + 1     # +1 result frame at the end
    legend_handles = _viz_merge_legend()
    # The final kept set, for the closing "result" frame -- `frames[-1]`'s
    # own `kept_so_far` is the state BEFORE its own decision, so the true
    # final set adds that last point too if it survived.
    final_kept = list(frames[-1]['kept_so_far'])
    if frames[-1]['alive']:
        final_kept.append(frames[-1]['i'])

    def frame(k):
        ax.clear()
        ax.set_xlim(0, _VIZ_FOOTPRINT)
        ax.set_ylim(_VIZ_FOOTPRINT, 0)
        ax.set_aspect('equal')

        if k < n_reveal:
            for ds in _VIZ_ORDER[:k + 1]:
                _draw_rung_points(ax, all_pts[rung_id == ds], ds, label=False)
            ax.set_title('phase 1/3 -- raw detections per rung, one at a '
                         f'time\nadded ds={_VIZ_ORDER[k]:g}', fontsize=10)
        elif k == n_reveal:
            for ds in _VIZ_ORDER:
                _draw_rung_points(ax, all_pts[rung_id == ds], ds, label=False)
            ax.set_title(f'phase 2/3 -- all {n} raw points overlaid, '
                         'about to merge', fontsize=10)
        elif k == n_total - 1:
            # Continues straight from the LAST decision frame, not a clean
            # cut to just the survivors ("the circle
            # should turn into an X, that's what makes it connect") -- every
            # merged-away point (including the one the previous frame was
            # still highlighting with a circle) settles into the same plain
            # X every other merged-away point already has; only the circle
            # and the "here's the one just decided" highlight go away.
            for j in range(n):
                if j in final_kept:
                    ax.scatter(*all_pts[j], s=_VIZ_MARKER_LARGE,
                              color=_RUNG_COLORS[rung_id[j]], marker='*',
                              edgecolors='0.2', linewidths=0.8, zorder=4)
                else:
                    ax.scatter(*all_pts[j], s=_VIZ_MARKER_SMALL, color='0.7',
                              marker='x', zorder=2)
            ax.set_title(f'result -- {n} raw points merged into '
                         f'{len(final_kept)} anchors', fontsize=10)
        else:
            f = frames[k - n_reveal - 1]
            i, kept_so_far, alive, radius, rival = (
                f['i'], f['kept_so_far'], f['alive'], f['radius'], f['rival'])
            for j in range(n):
                if j == i:
                    continue
                if j in kept_so_far:
                    ax.scatter(*all_pts[j], s=_VIZ_MARKER_MED,
                              color=_RUNG_COLORS[rung_id[j]], marker='*',
                              edgecolors='0.3', linewidths=0.6, zorder=4)
                elif j < i:
                    ax.scatter(*all_pts[j], s=_VIZ_MARKER_SMALL, color='0.7',
                              marker='x', zorder=2)
                else:
                    _draw_rung_points(ax, all_pts[j:j + 1], rung_id[j],
                                      label=False)
            cx, cy = all_pts[i]
            if radius > 0:
                ax.add_patch(plt.Circle((cx, cy), radius, fill=False,
                                        color='crimson', lw=1.2, zorder=5))
            if alive:
                color, marker = _RUNG_COLORS[rung_id[i]], '*'
            else:
                color, marker = 'crimson', 'x'
            ax.scatter(cx, cy, s=_VIZ_MARKER_LARGE, color=color, marker=marker,
                      edgecolors='0.2' if alive else None,
                      linewidths=0.8 if alive else 0, zorder=6)
            if not alive and rival is not None:
                rx, ry = all_pts[rival]
                ax.plot([cx, rx], [cy, ry], '--', color='crimson', lw=1,
                       zorder=5)
            status = ('kept -> becomes an anchor' if alive
                      else f'merged away -- a kept point is within radius '
                           f'{radius:.2f}')
            j_in_phase = k - n_reveal
            ax.set_title(f'phase 3/3 -- point {j_in_phase}/{len(frames)}  '
                         f'(ds={rung_id[i]:g})\n{status}', fontsize=10)

        ax.legend(handles=legend_handles, loc='upper left',
                 bbox_to_anchor=(1.02, 1.0), fontsize=7, ncol=1,
                 borderaxespad=0.0)
        _save_frame_png(fig, frame_dir, k)

    anim = FuncAnimation(fig, frame, frames=n_total)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def _trace_probability_map_peak(field: np.ndarray, anchor_xy: np.ndarray,
                                tau: float, sample_step: float = 0.5
                                ) -> Tuple[np.ndarray, np.ndarray, float,
                                          float, float]:
    """Re-derives `probe_via_probability_map`'s exact sample grid (SAME
    formula, SAME `_bilinear_sample` -- imported and reused, not
    reimplemented, so any interpolation bug shows up identically in both
    rather than being a second, possibly-wrong copy) around ONE anchor, at
    `scale=1.0`/`map_origin=(0,0)` (this demo's own convention
    throughout) -- for VISUALIZATION only: the production function
    returns `peak_value`/`peak_dist` (a distance, not a coordinate), so it
    alone cannot say WHERE the winning candidate actually is. `_animate_
    alive` checks this trace's own `peak_value` against the real
    function's before trusting it to draw anything, the same backstop
    pattern `_trace_cross_rung_merge` uses against `SurvivalProcess.
    anchors_of`.

    Returns `(xs, ys, best_x, best_y, peak_value)` -- `xs`/`ys` the WHOLE
    sample grid (for drawing every candidate checked), `best_x`/`best_y`
    the winning one's own coordinate, `peak_value` its interpolated value
    (compared against the real function's own return as the backstop
    above).
    """
    r = float(tau)
    steps = np.arange(-r, r + sample_step, sample_step)
    dx, dy = np.meshgrid(steps, steps)
    within_circle = (dx ** 2 + dy ** 2) <= r ** 2
    dx, dy = dx[within_circle], dy[within_circle]
    cx, cy = anchor_xy
    xs, ys = cx + dx, cy + dy
    h, w = field.shape
    values = _bilinear_sample(field, xs, ys, h, w)
    best = int(np.argmax(values))
    return xs, ys, float(xs[best]), float(ys[best]), float(values[best])


def _animate_alive(anchors: np.ndarray, anchor_pattern: List[str],
                   anchor_rung: np.ndarray,
                   points_by_rung: Dict[float, np.ndarray],
                   score_by_rung: Dict[float, np.ndarray],
                   maps_by_rung: Dict[float, np.ndarray], viz_alpha: float,
                   out_path: str, fps: float,
                   frame_dir: Optional[str] = None) -> None:
    """(d) -- for the SAME six anchors, animate alive[j] across all four
    rungs for two of the five candidates side by side: 絕對雙門檻法
    (baseline, `Patterns.alive_from`, a line to the nearest REAL detected
    point) and 機率圖法 (candidate 1, `probe_via_probability_map` +
    `alive_probability_map`, a circle on the raw field). Chosen because the
    difference between "is there a real NMS-kept point nearby" and "is the
    raw field itself high nearby" is the most visually direct of the five
    -- not because the other three (candidates 2/3/4) are less interesting;
    they are text, not pictures, in the PPT for now.

    FILL COLOUR CHANGES MEANING ONCE, AT THE CLOSING FRAME: for the four per-rung frames, an anchor's fill is
    `anchor_rung` -- which rung its own coordinate came from (the same
    colour it had as a star at the end of `_animate_anchor_merge`, so this
    picks up where that GIF left off). Only the FIFTH, closing frame
    switches every anchor's fill to `anchor_pattern`'s colour -- the
    six-pattern classification, which is not knowable until the alive[]
    vector across ALL FOUR rungs has been seen, so it cannot honestly be
    shown any earlier than this.
    """
    n = len(anchors)
    rung_colors = [_RUNG_COLORS[ds] for ds in anchor_rung]
    pattern_colors = [_VIZ_PATTERN_COLORS[p] for p in anchor_pattern]
    n_rungs = len(_VIZ_ORDER)
    fig, (ax_b, ax_p) = plt.subplots(1, 2, figsize=(12.5, 6.8))

    # The six-pattern colour key, spelled out ON the figure (naming it in
    # the suptitle is not the same as labelling it) -- added ONCE, outside frame(): a fig-level
    # legend is not inside the axes `ax.clear()` wipes each frame, so it
    # does not need to be re-added every frame the way the per-axes
    # alive/dead legends do. Only becomes the ACTIVE meaning at the final
    # frame -- see the docstring above -- but stays visible throughout as
    # a reference a viewer can read ahead to.
    pattern_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
              markeredgecolor='0.3', markersize=9, label=ASCII_NAMES[p])
        for p, color in _VIZ_PATTERN_COLORS.items()]
    fig.legend(handles=pattern_legend, loc='lower center', ncol=6,
              fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.0),
              title='fill colour from the FINAL frame on = which pattern '
                    'this anchor is', title_fontsize=7)
    fig.subplots_adjust(bottom=0.18, top=0.86)

    #: Same rung-colour key `_viz_merge_legend` uses -- shown on the
    #: per-rung frames only (the fig-level legend above takes over once
    #: fill switches to pattern colour at the final frame).
    rung_fill_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=_RUNG_COLORS[ds],
              markeredgecolor='0.3', markersize=8, label=f'ds={ds:g} anchor')
        for ds in _VIZ_ORDER]

    def frame(k):
        is_final = (k == n_rungs)
        ds = _VIZ_ORDER[-1] if is_final else _VIZ_ORDER[k]
        colors = pattern_colors if is_final else rung_colors
        tau = viz_alpha * ds
        rung_pts = points_by_rung[ds]
        rung_score = score_by_rung[ds]

        if len(rung_pts):
            dist, score = SurvivalProcess.nearest_detection(
                rung_pts, rung_score, anchors)
        else:
            dist, score = np.full(n, -1.0), np.zeros(n)
        alive_b = alive_from(score, dist, score_threshold=0.5,
                             tau=np.full(n, tau))

        field = maps_by_rung[ds]
        peak_value, peak_dist = probe_via_probability_map(
            field, anchors, map_origin=(0.0, 0.0), scale=1.0, tau=tau,
            sample_step=0.5)
        alive_p = alive_probability_map(
            peak_value.reshape(1, -1), peak_dist.reshape(1, -1),
            score_threshold=0.5, tau=np.array([tau]))[0]

        ax_b.clear(); ax_p.clear()
        # vmin/vmax FIXED at 0/1, not auto-scaled per frame: a max-combined
        # Gaussian field's own peak is always exactly 1.0 by construction
        # (`_synthetic_probability_field`'s own docstring), so auto-scaling
        # to each frame's min/max is a no-op for the peak but lets a frame
        # with a smaller/absent bump stretch its own noise floor up to look
        # like a real signal -- fixed limits keep every frame's colour
        # meaning the SAME number.
        ax_p.imshow(field, origin='upper',
                   extent=(0, _VIZ_FOOTPRINT, _VIZ_FOOTPRINT, 0),
                   cmap='viridis', vmin=0.0, vmax=1.0, alpha=0.7, zorder=1)
        for ax, title in ((ax_b, 'baseline (nearest real detection)'),
                          (ax_p, 'probability map (circular window on raw '
                                 'field)')):
            ax.set_xlim(0, _VIZ_FOOTPRINT); ax.set_ylim(_VIZ_FOOTPRINT, 0)
            ax.set_aspect('equal')
            ax.set_title(f'{title}\nds={ds:g}  tau={tau:.1f}', fontsize=10)

        if len(rung_pts) and not is_final:
            ax_b.scatter(rung_pts[:, 0], rung_pts[:, 1],
                        s=_VIZ_MARKER_SMALL, color=_RUNG_COLORS[ds],
                        marker='s', zorder=2,
                        label=f'{len(rung_pts)} real detections (ds={ds:g})')

        for i in range(n):
            x, y = anchors[i]
            # On the FINAL frame, the alive/dead-per-rung edge colour and
            # the detection-matching visuals (real-detection squares, the
            # match line, the tau window) stop meaning anything -- this
            # frame is the CLASSIFICATION result, not another rung check --
            # so both collapse to a plain neutral edge and nothing else is
            # drawn on top of the anchors.
            edge = '0.3' if is_final else ('limegreen' if alive_b[i] else 'crimson')
            ax_b.scatter(x, y, s=_VIZ_MARKER_LARGE, facecolor=colors[i],
                        edgecolor=edge, linewidth=2, zorder=4)
            if not is_final and alive_b[i] and len(rung_pts):
                d = np.linalg.norm(rung_pts - np.array([x, y]), axis=1)
                nx, ny = rung_pts[int(np.argmin(d))]
                ax_b.plot([x, nx], [y, ny], '-', color='0.35', lw=1, zorder=3)

            edge = '0.3' if is_final else ('limegreen' if alive_p[i] else 'crimson')
            ax_p.scatter(x, y, s=_VIZ_MARKER_LARGE, facecolor=colors[i],
                        edgecolor=edge, linewidth=2, zorder=4)
            if not is_final:
                ax_p.add_patch(plt.Circle((x, y), tau, fill=False,
                                          color='white', lw=1, alpha=0.8,
                                          zorder=3))
                # The sample grid `probe_via_probability_map` actually
                # checks (every candidate, real signal or noise alike),
                # plus the winning one, marked: a bare circle only implies a
                # search radius, it does not show candidate1's mechanism itself.
                gxs, gys, best_x, best_y, traced_val = \
                    _trace_probability_map_peak(field, (x, y), tau)
                if not np.isclose(traced_val, peak_value[i], atol=1e-6):
                    raise AssertionError(
                        f'visualize\'s _trace_probability_map_peak '
                        f'({traced_val:.6f}) disagrees with the real '
                        f'probe_via_probability_map ({peak_value[i]:.6f}) '
                        f'for anchor {i} at ds={ds:g} -- the trace has '
                        f'drifted from production logic, fix it before '
                        f'trusting the sample-grid overlay')
                ax_p.scatter(gxs, gys, s=2, color='white', alpha=0.35,
                           zorder=2, linewidths=0)
                ax_p.scatter([best_x], [best_y], s=40, marker='x',
                           color='yellow', linewidths=1.6, zorder=5)

        if is_final:
            # Neither alive/dead nor detection-matching means anything on
            # this frame -- just the anchors and what their fill means.
            ax_b.legend(handles=[Line2D(
                [0], [0], marker='o', color='w', markerfacecolor='0.6',
                markeredgecolor='0.3', markersize=8,
                label='anchor (fill = pattern, see legend below)')],
                fontsize=6, loc='upper right')
            ax_p.legend(handles=[Line2D(
                [0], [0], marker='o', color='w', markerfacecolor='0.6',
                markeredgecolor='0.3', markersize=8,
                label='anchor (fill = pattern, see legend below)')],
                fontsize=6, loc='upper right')
        else:
            alive_legend = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='0.6',
                      markeredgecolor='limegreen', markeredgewidth=2,
                      markersize=8, label='alive (this rung)'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='0.6',
                      markeredgecolor='crimson', markeredgewidth=2,
                      markersize=8, label='dead (this rung)'),
            ]
            ax_b.legend(handles=alive_legend + rung_fill_legend + ([Line2D(
                [0], [0], marker='s', color='w',
                markerfacecolor=_RUNG_COLORS[ds], markeredgecolor='0.3',
                markersize=6,
                label=f'{len(rung_pts)} real detections (ds={ds:g})')]
                if len(rung_pts) else []),
                fontsize=6, loc='upper right')
            ax_p.legend(handles=alive_legend + rung_fill_legend + [
                Line2D([0], [0], color='white', lw=1,
                      label='tau search window'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='0.9',
                      markeredgecolor='none', markersize=4, alpha=0.7,
                      label='sample grid (every candidate checked)'),
                Line2D([0], [0], marker='x', color='yellow', linestyle='None',
                      markersize=7, markeredgewidth=1.6,
                      label='peak found (this anchor)'),
            ], fontsize=6, loc='upper right')

        if is_final:
            fig.suptitle(f'final classification (fill switches to '
                        f'pattern colour; ds={ds:g} still shown)',
                        fontsize=11)
        else:
            fig.suptitle(f'anchor alive decision, ds={ds:g}  '
                        '(fill = contributing rung)', fontsize=11)
        _save_frame_png(fig, frame_dir, k)

    anim = FuncAnimation(fig, frame, frames=n_rungs + 1)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def _run_visualize_real_example(args, out_dir: str) -> None:
    """A single static real-data figure closing the synthetic story: a
    native-resolution crop of the actual tissue (`SlideReader.read` at
    ds 1 -- NOT the C-tree's own coarser mother
    image, which is downsampled and would render as a couple of blurry
    blown-up pixels at a crop this small) as the BOTTOM layer, with the
    real per-rung detections and the anchors `anchors_of_generations`
    builds from them drawn on top. Skipped (with a printed note) unless
    `--checkpoint` is given -- deferred torch import keeps a synthetic-only
    `--parts visualize` run model-free, same as `merge_grid`. Pass
    `--c-rungs 4 8 16` (etc.) to keep the DETECTION to the coarse rungs
    only; `--viz-real-crop` controls the DISPLAY crop size separately. Anchors are
    the BOTTOM scatter layer (drawn on top they hide everything else), the
    tissue image bottom of everything, per-rung dots on top.
    """
    if not args.checkpoint:
        print('  [visualize] --checkpoint not given -- skipping the real-'
             'data example figure (synthetic GIFs still saved)')
        return
    import torch                                                  # noqa: PLC0415
    from reeval_density import load_arm                              # noqa: PLC0415

    wsi_name = args.wsi_name or DEFAULT_WSI_NAME
    entry = AccessDatasets.locate(wsi_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net, _identity, _ = load_arm(args.checkpoint, device)
    threshold = float(net.cfg.detection_threshold)
    c_corpus = prepare_chain_stack.axis_corpus('C', args)
    order = sorted(float(r) for r in args.c_rungs)
    crop = int(args.viz_real_crop)

    with SafeSlide(entry.path) as wsi:
        forest = ChainStack.CStack.from_own(
            c_corpus, entry.name, args.c_rungs, wsi, tile=args.tile,
            cache_root=chainstack_root(args))
        mother, mother_image, groups_by_ds, images_by_ds = forest[0]
        per_rung_tiles, per_rung = SurvivalProcess.detect_all_generations(
            mother, mother_image, groups_by_ds, images_by_ds, net,
            score_threshold=threshold)

        anchors, _ = SurvivalProcess.anchors_of_generations(
            per_rung_tiles, order, net.cfg.nms_radius)

        # Crop centred on the anchors' own centroid (falls back to the
        # mother's centre if this tree produced none) -- a native, level-0
        # `SlideReader.read` at ds 1, NOT `mother_image` (which
        # is downsampled to `args.tile` px covering the WHOLE mother
        # footprint, so a crop this small into it would be a couple of
        # blown-up, blurry source pixels).
        if len(anchors):
            cx, cy = float(anchors[:, 0].mean()), float(anchors[:, 1].mean())
        else:
            cx, cy = mother.x + mother.size_px / 2, mother.y + mother.size_px / 2
        slide_w, slide_h = wsi.dimensions
        x0 = min(max(0, int(cx - crop / 2)), slide_w - crop)
        y0 = min(max(0, int(cy - crop / 2)), slide_h - crop)
        crop_img = SlideReader(wsi).read(x0, y0, ReadSpec(crop, crop), 1.0)

    fig_in = 7.0
    # Scaled to THIS crop, not the synthetic canvas's constants -- `crop`
    # is a user-set `--viz-real-crop` and can be anything (32, 256, ...),
    # so a marker size baked in for the 32-wide synthetic scene would be
    # the wrong physical size here (see `_marker_area`'s own docstring).
    anchor_size = _marker_area(crop * 0.02, crop, fig_in)
    detection_size = _marker_area(crop * 0.011, crop, fig_in)

    fig, ax = plt.subplots(figsize=(fig_in, fig_in))
    ax.imshow(crop_img, extent=(x0, x0 + crop, y0 + crop, y0), zorder=0)
    if len(anchors):
        ax.scatter(anchors[:, 0], anchors[:, 1], s=anchor_size,
                  color='#D4A017', marker='*', edgecolors='0.2',
                  linewidths=0.6, label=f'anchor ({len(anchors)} total)',
                  zorder=1)
    for ds in order:
        xy0, _score = per_rung[ds]
        if len(xy0):
            _draw_rung_points(ax, xy0, ds, label=False, size=detection_size)
            ax.scatter([], [], marker=_RUNG_MARKERS.get(ds, 'o'),
                      facecolors='none', edgecolors=_RUNG_COLORS.get(ds, '0.5'),
                      label=f'ds={ds:g} detection ({len(xy0)} total)')
    ax.set_xlim(x0, x0 + crop)
    ax.set_ylim(y0 + crop, y0)
    ax.set_aspect('equal')
    ax.legend(fontsize=6, loc='upper right')
    ax.set_title(f'real example -- {entry.name}, C-tree 0, rungs {order}\n'
                f'{crop}x{crop} level-0 px crop at native resolution')
    out_path = os.path.join(out_dir, 'visualize_real_example.png')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved  {out_path}')


def _run_visualize(args, out_dir: str) -> None:
    """PARTS=visualize -- synthetic, step-by-step GIFs (plus every frame as
    its own PNG) for a PPT: (b) each rung's own keypoints overlaid, (c) the
    cross-rung anchor merge (one point decided per frame), (d) alive[j] for
    two of the five candidates side by side, then a real-data closing
    figure if `--checkpoint` is given. See this section's own module
    comment for why synthetic, hand-placed, non-random data drives the main
    story.
    """
    # `--viz-alpha`'s ceiling for THIS scene: the widest tau this many-a
    # point can ever reach (`viz_alpha * max(ds)`) has to stay well under
    # the closest two true points' own distance, or a rung's real detection
    # of ONE point could be mistaken for a NEIGHBOURING point's alive
    # signal. Computed from `_VIZ_TRUE_POINTS` itself (not a hard-coded
    # "11") so a future edit to the scene re-checks itself rather than
    # silently invalidating this bound.
    min_spacing = min(
        float(np.hypot(xy_a[0] - xy_b[0], xy_a[1] - xy_b[1]))
        for (xy_a, _), (xy_b, _) in itertools.combinations(
            _VIZ_TRUE_POINTS.values(), 2))
    tau_max = args.viz_alpha * max(_VIZ_ORDER)
    if tau_max * 1.5 >= min_spacing:
        raise SystemExit(
            f'--viz-alpha {args.viz_alpha} gives tau_max={tau_max:.1f}, too '
            f'close to the closest two synthetic points\' own spacing '
            f'({min_spacing:.1f}) for a 1.5x safety margin -- a real '
            f'detection at a NEIGHBOURING point could be mistaken for this '
            f'point\'s own, breaking the six-pattern story. Keep --viz-alpha '
            f'below {min_spacing / 1.5 / max(_VIZ_ORDER):.2f} for this scene.')

    points_by_rung, _label_by_rung, score_by_rung = _synthetic_scene()
    order = list(_VIZ_ORDER)

    # ── verify the hand-placed geometry against the REAL merge before
    # trusting any of it for a GIF (this file cannot be run to iterate) ────
    real_anchors, _ = SurvivalProcess.anchors_of(points_by_rung, order,
                                                 merge_radius_l0=0.0)
    all_pts = np.concatenate([points_by_rung[ds] for ds in order], axis=0)
    rung_id = np.concatenate([np.full(len(points_by_rung[ds]), ds)
                              for ds in order])
    frames, kept = _trace_cross_rung_merge(all_pts, rung_id, rung_id.copy())
    traced_anchors = all_pts[kept]
    #: which rung EACH anchor's own coordinate came from -- the finest rung
    #: it survived at, since the merge visits finest-first (module comment
    #: above: "a point kept from a finer rung's list is visited before the
    #: same physical point's coarser-rung duplicate"). Feeds `_animate_
    #: alive`'s early frames, which fill each anchor by this rung before
    #: its closing frame switches the fill to the pattern classification.
    anchor_rung = rung_id[kept]

    if len(traced_anchors) != 6 or len(real_anchors) != 6:
        raise AssertionError(
            f'visualize\'s synthetic scene no longer produces 6 anchors '
            f'(traced {len(traced_anchors)}, real {len(real_anchors)}) -- '
            f'check _VIZ_TRUE_POINTS/_viz_offset before trusting the GIFs')
    set_traced = set(map(tuple, np.round(traced_anchors, 4).tolist()))
    set_real = set(map(tuple, np.round(real_anchors, 4).tolist()))
    if set_traced != set_real:
        raise AssertionError(
            'visualize\'s merge trace (_trace_cross_rung_merge) disagrees '
            'with the real SurvivalProcess.anchors_of on the anchor '
            'coordinates -- the trace has drifted from production logic, '
            'fix it before trusting the anchor-merge GIF')

    anchor_pattern = []
    for x, y in traced_anchors:
        best_p, best_d = None, np.inf
        for p, (xy, _rungs) in _VIZ_TRUE_POINTS.items():
            d = float(np.hypot(x - xy[0], y - xy[1]))
            if d < best_d:
                best_p, best_d = p, d
        anchor_pattern.append(best_p)

    # One frames/ subdirectory per animation -- every GIF frame ALSO saved
    # as its own PNG there (per the user's ask), rendered by the exact same
    # drawing code as the GIF (`_save_frame_png`, called from inside each
    # animation's own frame() callback), never a second render path.
    kp_dir = os.path.join(out_dir, 'visualize_keypoints_frames')
    merge_dir = os.path.join(out_dir, 'visualize_anchor_merge_frames')
    alive_dir = os.path.join(out_dir, 'visualize_alive_frames')
    for d in (kp_dir, merge_dir, alive_dir):
        os.makedirs(d, exist_ok=True)

    _animate_keypoints(points_by_rung,
                       os.path.join(out_dir, 'visualize_keypoints.gif'),
                       fps=args.viz_fps, frame_dir=kp_dir)
    print(f'Saved  {os.path.join(out_dir, "visualize_keypoints.gif")}  '
         f'(+ frames in {kp_dir}/)')

    _animate_anchor_merge(all_pts, rung_id, frames,
                          os.path.join(out_dir, 'visualize_anchor_merge.gif'),
                          fps=args.viz_fps, frame_dir=merge_dir)
    print(f'Saved  {os.path.join(out_dir, "visualize_anchor_merge.gif")}  '
         f'(+ frames in {merge_dir}/)')

    maps_by_rung = {ds: _synthetic_probability_field(
                       points_by_rung[ds], noise_peaks=_VIZ_NOISE_PEAKS[ds])
                    for ds in order}
    _animate_alive(traced_anchors, anchor_pattern, anchor_rung,
                  points_by_rung, score_by_rung, maps_by_rung, args.viz_alpha,
                  os.path.join(out_dir, 'visualize_alive.gif'),
                  fps=args.viz_fps, frame_dir=alive_dir)
    print(f'Saved  {os.path.join(out_dir, "visualize_alive.gif")}  '
         f'(+ frames in {alive_dir}/)')

    _run_visualize_real_example(args, out_dir)


# =============================================================================
#  dispatch
# =============================================================================

# =============================================================================
#  scale_diagnostic -- does RStack's shrink-and-grow agree with a plain
#  Gaussian blur on WHICH points are 只在一階/一直存活/不連續? It tests
#  the hypothesis that 只在一階 tracks a point's own
#  characteristic SCALE (classical scale-space theory) rather than something
#  specific to RStack's own resampling filter.
# =============================================================================
#
# NOT A THIRD AXIS ALONGSIDE F/R/C. Both stacks built here are R-SHAPED --
# fixed footprint, `rung_scale('R', ds) == 1.0` at every rung -- one is
# `RStack.from_tile` itself, the other a from-scratch degradation using the
# SAME real tile. Neither is written to any store; this is a comparison, not
# a corpus.

def classical_blur_stack(image: np.ndarray, rungs: Sequence[float], *,
                         tile: int) -> Dict[float, np.ndarray]:
    """Same shape as `ChainStack.RStack.from_tile`: one real tile -> `{ds:
    image}`, `ds<=1.0` untouched (mirrors `degrade_resolution`'s own
    shortcut -- "no degradation" has to mean the same thing on both rows, or
    a difference at column 0 would masquerade as a finding about column 1+).

    `sigma = ds / 2`, AN APPROXIMATION, NOT A DERIVED EQUIVALENCE.
    `degrade_resolution` (INTER_AREA shrink by `ds`, INTER_LINEAR grow back)
    is a box-filter-shaped low-pass with no single sigma of its own; `ds/2`
    puts a Gaussian blur's radius in the same rough order of magnitude as
    that box filter's support (side `ds` px). The two rows are meant to be
    comparable in DEGREE of information removed, not identical in
    mechanism -- the mechanism difference is the whole point of running
    this comparison at all. Nobody has fit `sigma` against RStack's own
    frequency response; treat this the way this project treats any
    PENDING-MEASUREMENT constant.
    """
    out: Dict[float, np.ndarray] = {}
    for ds in sorted(float(r) for r in rungs):
        img = image
        if img.shape[0] != tile:
            img = cv2.resize(img, (tile, tile), interpolation=cv2.INTER_AREA)
        if ds > 1.0:
            sigma = float(ds) / 2.0
            img = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
        out[ds] = img
    return out


#: The three patterns this diagnostic marks with a fixed-colour X -- the
#: SCALE hypothesis (只在一階) and its two natural contrasts (survives
#: everywhere / survives nowhere consistently).
_SCALE_MARK_COLORS = {'只在一階': '#2ECC40', '一直存活': '#111111',
                      '不連續': '#888888'}

#: The other three patterns (drawn too, but as a circle
#: coloured by that rung's own score rather than a fixed per-pattern
#: colour) -- 'Blues' runs light-to-dark, matching "darker = higher score"
#: with no inversion needed.
_SCALE_CIRCLE_PATTERNS = ('細部存活', '晚生型', '中間帶')
_SCALE_CIRCLE_CMAP = plt.get_cmap('Blues')


def _scale_diagnostic_anchors(stack: Dict[float, np.ndarray], net, *,
                              order: List[float], score_threshold: float,
                              alpha: float, tau_floor: float
                              ) -> Tuple[np.ndarray, List[str], np.ndarray,
                                        np.ndarray]:
    """One R-shaped stack -> `(anchors [N,2] tile px, patterns [N], alive
    [N, len(order)] bool, score [N, len(order)] float)`. Same R-axis flow
    `survival_alpha_analysis._one_r_tile` runs in production
    (`detect_all_rungs` -> `anchors_of(..., rung_scale=lambda ds: 1.0)` ->
    `probe_real`), with `origin=(0,0)`/`scale=1.0` at every rung instead of
    a real WSI position: this demo never leaves one tile's own pixel grid,
    so tile-pixel and level-0 coordinates coincide throughout.

    `alive`/`score` are returned alongside `patterns` (not just the
    classification) because the figure marks an anchor only at the RUNGS it
    actually survived, coloured by that rung's own score for the three
    patterns drawn as circles -- `patterns[i]` alone cannot say which
    columns those are or how confident each one was.

    `tau = max(tau_floor, alpha * ds)` -- the SAME formula
    `survival_alpha_analysis.py` calibrates `alpha` against (plan.md 2.2
    settled on alpha~3 for production); `alpha`/`tau_floor` are this
    function's own arguments rather than a re-import, so a demo run is never
    silently coupled to that file's current default changing under it.
    """
    origins = {d: (0.0, 0.0) for d in order}
    scales = {d: 1.0 for d in order}
    detections, per_rung, maps = SurvivalProcess.detect_all_rungs(
        stack, net, rungs=order, origins=origins, scales=scales,
        score_threshold=score_threshold)
    anchors, _source_rung = SurvivalProcess.anchors_of(
        detections, order, net.cfg.nms_radius, rung_scale=lambda ds: 1.0)
    dist, score, _rival, _rival_dist = SurvivalProcess.probe_real(
        anchors, per_rung, order, maps, origins=origins, scales=scales,
        nms_radius=net.cfg.nms_radius)
    tau = np.maximum(float(tau_floor), float(alpha) * np.asarray(order, np.float64))
    alive = alive_from(score, dist, score_threshold=score_threshold, tau=tau)
    patterns = [classify(row)[0] for row in alive]
    return anchors, patterns, alive, score


def _draw_scale_diagnostic_figure(r_stack, r_anchors, r_patterns, r_alive,
                                  r_score, c_stack, c_anchors, c_patterns,
                                  c_alive, c_score, order: List[float],
                                  score_threshold: float, title: str,
                                  out_path: str) -> None:
    """2 rows (RStack / classical blur) x len(order) columns (ds=1, no
    degradation, first; most-degraded last). An anchor's marker is drawn
    ONLY on the columns where its OWN `alive[]` row says it actually
    survived --
    只在一階 therefore appears in exactly one column, 不連續 (flicker) in
    whichever non-contiguous columns it survived, and 一直存活 in every
    column, which is what "alive everywhere" already means.

    細部存活/晚生型/中間帶 are drawn too, but as an
    unfilled CIRCLE rather than a fixed-colour X, coloured by that rung's
    OWN score (`_SCALE_CIRCLE_CMAP`, dark = high, light = low, normalised
    over `[score_threshold, 1.0]` since nothing alive ever scores below the
    threshold) -- these three share one visual channel (score) rather than
    three more fixed colours, because this run is not asking "which of
    these three is it" the way it is for the other three.
    """
    n_cols = len(order)
    # +1.0 in width is the colorbar's OWN reserved strip (added below via
    # add_axes), not shared with the column axes -- fig.colorbar(..., ax=
    # axes) would borrow room from the axes it
    # is given, shrinking column n_cols-1 every time this runs.
    fig, axes = plt.subplots(2, n_cols, figsize=(3.2 * n_cols + 1.0, 7.2),
                             squeeze=False)
    norm = matplotlib.colors.Normalize(vmin=float(score_threshold), vmax=1.0)
    rows = ((r_stack, r_anchors, r_patterns, r_alive, r_score,
            'RStack (shrink + grow)'),
           (c_stack, c_anchors, c_patterns, c_alive, c_score,
            'classical blur (sigma=ds/2)'))
    for row_i, (stack, anchors, patterns, alive, score, row_label) \
            in enumerate(rows):
        for col_i, ds in enumerate(order):
            ax = axes[row_i][col_i]
            ax.imshow(stack[ds])
            for i, ((x, y), pattern) in enumerate(zip(anchors, patterns)):
                if not alive[i, col_i]:
                    continue
                color = _SCALE_MARK_COLORS.get(pattern)
                if color is not None:
                    ax.scatter([x], [y], s=36, c=color, marker='x',
                              linewidths=1.6, zorder=3)
                elif pattern in _SCALE_CIRCLE_PATTERNS:
                    ax.scatter([x], [y], s=48, facecolors='none',
                              edgecolors=_SCALE_CIRCLE_CMAP(norm(
                                  score[i, col_i])),
                              linewidths=1.4, zorder=3)
            if row_i == 0:
                ax.set_title(f'ds={ds:g}' + ('  (no degradation)'
                                             if ds <= 1.0 else ''),
                            fontsize=8)
            if col_i == 0:
                ax.set_ylabel(row_label, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])

    # ASCII_NAMES ONLY, not the Chinese pattern name -- matplotlib's default
    # font has no CJK glyphs (Patterns.py's own docstring). Putting
    # the Chinese name in the label even alongside the ASCII one still
    # renders that half as empty boxes.
    handles = [Line2D([0], [0], marker='x', color=color, linestyle='None',
                      markersize=7, markeredgewidth=1.6,
                      label=ASCII_NAMES[p])
              for p, color in _SCALE_MARK_COLORS.items()]
    handles.append(Line2D([0], [0], marker='o', color='w',
                          markerfacecolor='none',
                          markeredgecolor=_SCALE_CIRCLE_CMAP(0.5),
                          markersize=8, markeredgewidth=1.4,
                          label='fine-only / late-born / mid-band '
                                '(circle, colour = score)'))
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=9,
              frameon=False, bbox_to_anchor=(0.46, 0.0))
    # right=0.90 reserves the rightmost 10% of the FIGURE (not of any axes)
    # for the colorbar strip added next -- the column axes fill 0..0.90 and
    # are never touched by the colorbar call, unlike ax=axes above.
    fig.subplots_adjust(bottom=0.16, top=0.92, right=0.90, wspace=0.06,
                        hspace=0.08)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=_SCALE_CIRCLE_CMAP)
    sm.set_array([])
    cbar_ax = fig.add_axes((0.92, 0.16, 0.015, 0.76))     # own strip, not axes'
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('score (circle colour)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=13)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


#: 5x5 patch (half-width 2) around an anchor's own coordinate -- a single
#: bilinear-sampled pixel of a Laplacian response is noisy (Laplacian is a
#: high-frequency operator, sensitive to sub-pixel sampling position); a
#: small patch mean is not.
_DECAY_PATCH_HALF = 2


def _laplacian_maps(stack: Dict[float, np.ndarray],
                    order: List[float]) -> Dict[float, np.ndarray]:
    """One `|Laplacian|` response map per rung, float32 grayscale -- computed
    ONCE per tile from `classical_blur_stack`'s own real Gaussian-blurred
    images, then reused for every anchor's own decay-rate probe rather than
    recomputed per anchor.
    """
    out: Dict[float, np.ndarray] = {}
    for ds in order:
        gray = cv2.cvtColor(stack[ds], cv2.COLOR_RGB2GRAY).astype(np.float32)
        out[ds] = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    return out


def _patch_mean(response_map: np.ndarray, x: float, y: float, *,
                half: int = _DECAY_PATCH_HALF) -> float:
    """Mean of `response_map` in a `(2*half+1)`-square window centred on
    `(x, y)`, clipped to the map's own bounds."""
    h, w = response_map.shape
    xi, yi = int(round(x)), int(round(y))
    x0, x1 = max(0, xi - half), min(w, xi + half + 1)
    y0, y1 = max(0, yi - half), min(h, yi + half + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(response_map[y0:y1, x0:x1].mean())


def _decay_rate(lap_maps: Dict[float, np.ndarray], order: List[float],
                xy) -> float:
    """One anchor's decay rate: `-slope` of a Theil-Sen fit of `log(R[ds])`
    against `log(ds)`, where `R[ds] = _patch_mean(lap_maps[ds], *xy)`.

    `R ~ C * ds**slope` -- `slope` negative means R falls as ds grows
    (decay), so `-slope` is a "bigger number = decays faster" quantity.
    NOT normalised by `R[ds=1]`: a per-point multiplicative constant (this
    point's own contrast/stain intensity) only shifts the log-log
    INTERCEPT, never the slope, so the raw values already carry no such
    bias to correct for.

    Theil-Sen (median of all pairwise slopes), not ordinary least squares:
    `order` has as few as 5 rungs, where one noisy `R[ds]` can drag an OLS
    fit far off; Theil-Sen's median is unmoved by any single pair.
    """
    from scipy.stats import theilslopes                          # noqa: PLC0415
    x, y = xy
    r = np.array([_patch_mean(lap_maps[ds], x, y) for ds in order], np.float64)
    r = np.clip(r, 1e-6, None)
    slope, _intercept, _lo, _hi = theilslopes(
        np.log(r), np.log(np.asarray(order, np.float64)))
    return float(-slope)


def _survival_breadth(alive_row: np.ndarray,
                      order: List[float]) -> Optional[float]:
    """The COARSEST ds this anchor is still alive at -- not a count of alive
    rungs. Matches classical scale-space's own "characteristic scale"
    definition directly (the largest degradation a structure survives),
    and does not conflate a point that flickers (alive at a fine AND a
    coarse rung, dead between) with one that survives a contiguous range
    the way a raw count of alive rungs would. `None` (never alive at any
    rung) excludes the anchor from the correlation -- should not happen for
    a real anchor, which is detected at at least its own source rung, but
    guarded against rather than assumed.
    """
    alive_ds = [ds for ds, is_alive in zip(order, alive_row) if is_alive]
    return max(alive_ds) if alive_ds else None


def _permutation_test_spearman(decay: np.ndarray, breadth: np.ndarray, *,
                               n_permutations: int, seed: int
                               ) -> Tuple[float, float]:
    """`(real Spearman rho, permutation p)` -- the margin-over-a-decoy this
    project's own convention asks for, applied to a correlation instead of a
    group difference: `p` is the fraction of `n_permutations` random
    re-pairings of `breadth` against the SAME `decay` values whose `|rho|`
    is >= the real `|rho|`. A small `p` means the real pairing is more
    correlated than shuffling the two apart typically produces by chance,
    not just "a p-value below 0.05" from a parametric assumption.
    """
    from scipy.stats import spearmanr                             # noqa: PLC0415
    rho, _p_parametric = spearmanr(decay, breadth)
    rng = np.random.default_rng(seed)
    shuffled = np.array(breadth, copy=True)
    hits = 0
    for _ in range(int(n_permutations)):
        rng.shuffle(shuffled)
        r, _ = spearmanr(decay, shuffled)
        if abs(r) >= abs(rho):
            hits += 1
    return float(rho), hits / max(int(n_permutations), 1)


def _draw_decay_vs_survival_figure(decay: np.ndarray, breadth: np.ndarray,
                                   rho: float, p_perm: float,
                                   n_permutations: int, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.4))
    ax.scatter(decay, breadth, s=16, alpha=0.5, c='#2563EB', edgecolors='none')
    ax.set_xlabel('decay rate  (faster decay ->)', fontsize=10)
    ax.set_ylabel('survival breadth  (coarsest ds still alive)', fontsize=10)
    ax.set_title(f'Spearman rho = {rho:.3f}   permutation p = {p_perm:.3f}   '
                f'(n={len(decay)}, {n_permutations} shuffles)', fontsize=10)
    ax.grid(alpha=0.25)

    formula = ('decay rate = -slope,  slope = Theil-Sen median slope of  '
              'log(R[ds])  vs  log(ds)   (R ~ C * ds^slope)\n'
              f'R[ds] = mean |Laplacian(classical_blur_stack[ds])| over a '
              f'{2 * _DECAY_PATCH_HALF + 1}x{2 * _DECAY_PATCH_HALF + 1} '
              'patch centred on the anchor -- computed on the CLASSICAL '
              'blur stack for both rows, since RStack\'s own filter is '
              'exactly what this whole comparison is checking is not the '
              'reason a point decays')
    fig.text(0.5, 0.01, formula, ha='center', va='bottom', fontsize=7.5,
             family='monospace', wrap=True)
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _run_scale_diagnostic(args, out_dir: str) -> None:
    if not args.checkpoint:
        raise SystemExit('--parts scale_diagnostic needs --checkpoint (a '
                         'trained KeypointNet .pt) -- it runs the real '
                         'detector, there is no synthetic-only mode here')
    import torch                                                  # noqa: PLC0415
    from reeval_density import load_arm                              # noqa: PLC0415

    wsi_name = args.wsi_name or DEFAULT_WSI_NAME
    entry = AccessDatasets.locate(wsi_name)
    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net, _identity, _ = load_arm(args.checkpoint, device)
    threshold = (args.score_threshold if args.score_threshold is not None
                else float(net.cfg.detection_threshold))
    order = sorted(float(r) for r in args.rungs)

    f_corpus = prepare_chain_stack.axis_corpus('F', args)
    f_own = FStack.from_own(f_corpus, entry.name, tile=args.tile,
                            rungs=args.rungs)
    chain_ids = list(f_own)[:args.n_scale_tiles]
    if not chain_ids:
        raise SystemExit(f'no complete F chain for {entry.name} in '
                         f'{f_corpus.key}: its draw holds no chain over '
                         f'every rung')
    if len(chain_ids) < args.n_scale_tiles:
        print(f'  only {len(chain_ids)} complete chains available, wanted '
             f'{args.n_scale_tiles}', flush=True)

    print(f'[scale_diagnostic]  {entry.name}  {len(chain_ids)} tiles  '
         f'rungs {order}  alpha={args.scale_alpha:g}  '
         f'score_threshold={threshold:g}', flush=True)

    # Pooled across every tile AND both rows (RStack-detected anchors +
    # classical-blur-detected anchors) -- a per-tile correlation would have
    # only ~80-90 anchors to work with; pooling gives the permutation test
    # real statistical power. The two rows are different anchor SETS (each
    # stack's own detector run finds its own points), which is why this
    # concatenates rather than requiring anchors to match 1:1 across rows.
    all_decay: List[float] = []
    all_breadth: List[float] = []

    for k, chain_id in enumerate(chain_ids):
        base_image = f_own[chain_id][1.0]
        r_stack = RStack.from_tile(base_image, 1.0, order, tile=args.tile)
        c_stack = classical_blur_stack(base_image, order, tile=args.tile)

        r_anchors, r_patterns, r_alive, r_score = _scale_diagnostic_anchors(
            r_stack, net, order=order, score_threshold=threshold,
            alpha=args.scale_alpha, tau_floor=args.scale_tau_floor)
        c_anchors, c_patterns, c_alive, c_score = _scale_diagnostic_anchors(
            c_stack, net, order=order, score_threshold=threshold,
            alpha=args.scale_alpha, tau_floor=args.scale_tau_floor)

        counts_r = {p: r_patterns.count(p) for p in _SCALE_MARK_COLORS}
        counts_c = {p: c_patterns.count(p) for p in _SCALE_MARK_COLORS}
        print(f'  tile {k:02d} (chain {chain_id})  '
             f'RStack {len(r_anchors)} anchors {counts_r}   '
             f'classical {len(c_anchors)} anchors {counts_c}', flush=True)

        out_path = os.path.join(fig_dir, f'scale_diagnostic_tile{k:02d}.png')
        _draw_scale_diagnostic_figure(
            r_stack, r_anchors, r_patterns, r_alive, r_score,
            c_stack, c_anchors, c_patterns, c_alive, c_score,
            order, threshold, f'{entry.name}  chain {chain_id}', out_path)
        print(f'  saved {out_path}', flush=True)

        # Decay rate is always probed on the CLASSICAL blur stack (see
        # _draw_decay_vs_survival_figure's own formula note) regardless of
        # which row an anchor came from -- RStack's own filter is exactly
        # what this whole comparison exists to rule in or out as the cause
        # of a point's decay, so it cannot also be the substrate the decay
        # itself is measured on.
        lap_maps = _laplacian_maps(c_stack, order)
        for anchors, alive in ((r_anchors, r_alive), (c_anchors, c_alive)):
            for i, xy in enumerate(anchors):
                breadth = _survival_breadth(alive[i], order)
                if breadth is None:
                    continue
                all_decay.append(_decay_rate(lap_maps, order, xy))
                all_breadth.append(breadth)

    print(f'\nfigures -> {fig_dir}/scale_diagnostic_tile*.png')

    if len(all_decay) < 3:
        print(f'\n[decay vs survival]  only {len(all_decay)} anchors pooled '
             f'-- too few for a correlation, skipping that figure')
        return

    decay_arr = np.asarray(all_decay, np.float64)
    breadth_arr = np.asarray(all_breadth, np.float64)
    rho, p_perm = _permutation_test_spearman(
        decay_arr, breadth_arr, n_permutations=args.decay_permutations,
        seed=args.seed)
    print(f'\n[decay vs survival]  n={len(decay_arr)}  Spearman rho={rho:.3f}  '
         f'permutation p={p_perm:.3f}  ({args.decay_permutations} shuffles)')

    decay_out = os.path.join(fig_dir, 'scale_diagnostic_decay_vs_survival.png')
    _draw_decay_vs_survival_figure(decay_arr, breadth_arr, rho, p_perm,
                                   args.decay_permutations, decay_out)
    print(f'  saved {decay_out}')


#: See the module docstring's new PART section for why this exists: the real
#: `scale_diagnostic` run showed near-zero correlation (rho=0.031, n=2917)
#: between decay rate and survival breadth, and this is the check for
#: whether that means the scale hypothesis is wrong or the RULER is --
#: before trusting either, does `_decay_rate` even track a KNOWN, controlled
#: size difference on synthetic blobs. No checkpoint, no slide, no torch.

def _synthetic_blob_canvas(n_points: int, *, min_radius: float = 1.0,
                           max_radius: float = 16.0, seed: int = 0,
                           spacing: float = 60.0, background: float = 90.0,
                           amplitude: float = 130.0
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`n_points` Gaussian blobs of KNOWN, controlled radius (log-uniform in
    `[min_radius, max_radius]` -- the same 1..16 range `order` spans, so
    this control probes the same scale range the real run did), each on its
    own grid cell `spacing` px apart (jittered) so no blob's blurred
    appearance ever reaches another's -- one big flat-background canvas.

    Returns `(canvas [H,W,3] uint8, xy [n_points,2], radius [n_points])`.
    `radius` IS the ground truth `_decay_rate` is being asked to recover --
    a real anchor never has one, which is exactly why this needs synthetic
    data rather than one more pass over real tiles.
    """
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(n_points)))
    canvas_side = int(side * spacing)
    canvas = np.full((canvas_side, canvas_side), float(background), np.float64)

    radius = np.exp(rng.uniform(np.log(min_radius), np.log(max_radius),
                                size=n_points))
    xy = np.zeros((n_points, 2), np.float64)
    for k in range(n_points):
        row, col = divmod(k, side)
        cx = (col + 0.5) * spacing + rng.uniform(-0.15, 0.15) * spacing
        cy = (row + 0.5) * spacing + rng.uniform(-0.15, 0.15) * spacing
        xy[k] = (cx, cy)
        r = float(radius[k])
        half = int(np.ceil(4 * r))
        x0, x1 = max(0, int(cx) - half), min(canvas_side, int(cx) + half + 1)
        y0, y1 = max(0, int(cy) - half), min(canvas_side, int(cy) + half + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        canvas[y0:y1, x0:x1] += amplitude * np.exp(
            -((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * r ** 2))

    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    return np.stack([canvas] * 3, axis=-1), xy, radius


def _draw_decay_vs_radius_figure(decay: np.ndarray, radius: np.ndarray,
                                 rho: float, p_perm: float,
                                 n_permutations: int, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.4))
    ax.scatter(decay, radius, s=10, alpha=0.4, c='#DC2626', edgecolors='none')
    ax.set_yscale('log')
    ax.set_xlabel('decay rate  (faster decay ->)', fontsize=10)
    ax.set_ylabel('KNOWN blob radius (px, log scale)', fontsize=10)
    ax.set_title(f'SYNTHETIC POSITIVE CONTROL   Spearman rho = {rho:.3f}   '
                f'permutation p = {p_perm:.3f}\n(n={len(decay)}, '
                f'{n_permutations} shuffles) -- ground truth, not a real '
                'anchor', fontsize=10)
    ax.grid(alpha=0.25, which='both')
    formula = ('same _decay_rate/_laplacian_maps as scale_diagnostic\'s own '
              'figure -- only the input differs: Gaussian blobs of KNOWN '
              'radius instead of a real anchor\'s unknown one. A strong '
              'negative correlation here (small radius -> fast decay) is '
              'the sanity check the real-data null result needs before it '
              'can be read as evidence against the scale hypothesis rather '
              'than evidence the measurement cannot see it')
    fig.text(0.5, 0.01, formula, ha='center', va='bottom', fontsize=7.5,
             family='monospace', wrap=True)
    fig.subplots_adjust(bottom=0.24)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _draw_synthetic_examples_figure(stack: Dict[float, np.ndarray],
                                    xy: np.ndarray, radius: np.ndarray,
                                    order: List[float], indices: np.ndarray,
                                    crop_half: int, out_path: str) -> None:
    """`len(indices)` rows (one example blob each, sorted by KNOWN radius) x
    `len(order)` columns (ds=1 first, most degraded last) -- same
    "rows=example, columns=ds" layout `_draw_scale_diagnostic_figure` uses
    for real tiles, except each panel here is a small CROP centred on one
    chosen blob rather than a whole tile: the full synthetic canvas is
    mostly flat background between widely-spaced blobs (`_synthetic_blob_
    canvas`'s own `spacing`), not informative to look at as one image.
    """
    n_rows, n_cols = len(indices), len(order)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.0 * n_cols, 2.0 * n_rows),
                             squeeze=False)
    for row_i, idx in enumerate(indices):
        cx, cy = xy[idx]
        for col_i, ds in enumerate(order):
            ax = axes[row_i][col_i]
            img = stack[ds]
            h, w = img.shape[:2]
            x0 = max(0, int(cx) - crop_half)
            x1 = min(w, int(cx) + crop_half + 1)
            y0 = max(0, int(cy) - crop_half)
            y1 = min(h, int(cy) + crop_half + 1)
            ax.imshow(img[y0:y1, x0:x1])
            if row_i == 0:
                ax.set_title(f'ds={ds:g}' + ('  (no degradation)'
                                             if ds <= 1.0 else ''),
                            fontsize=9)
            if col_i == 0:
                ax.set_ylabel(f'r={radius[idx]:.1f}px', fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('synthetic positive control -- example blobs across '
                'degradation stages (radius KNOWN, sorted small to large)',
                fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _run_scale_synthetic_control(args, out_dir: str) -> None:
    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    order = sorted(float(r) for r in args.rungs)

    canvas, xy, radius = _synthetic_blob_canvas(
        args.n_synthetic_points, seed=args.seed)
    print(f'[scale_synthetic_control]  {args.n_synthetic_points} blobs  '
         f'canvas {canvas.shape[0]}x{canvas.shape[1]}  '
         f'radius range [{radius.min():.1f}, {radius.max():.1f}] px  '
         f'rungs {order}', flush=True)

    stack = classical_blur_stack(canvas, order, tile=canvas.shape[0])
    lap_maps = _laplacian_maps(stack, order)
    decay = np.array([_decay_rate(lap_maps, order, xy[k])
                      for k in range(len(xy))], np.float64)

    rho, p_perm = _permutation_test_spearman(
        decay, radius, n_permutations=args.decay_permutations, seed=args.seed)
    print(f'  Spearman rho={rho:.3f}  permutation p={p_perm:.3f}  '
         f'({args.decay_permutations} shuffles)', flush=True)
    if rho > -0.3 or p_perm > 0.05:
        print('  WARNING: this is the ruler failing to see a KNOWN, '
             'controlled size difference -- the real scale_diagnostic '
             'null result should be read as "this measurement could not '
             'detect it", not as evidence the scale hypothesis is wrong',
             flush=True)

    out_path = os.path.join(fig_dir, 'scale_synthetic_control.png')
    _draw_decay_vs_radius_figure(decay, radius, rho, p_perm,
                                 args.decay_permutations, out_path)
    print(f'  saved {out_path}', flush=True)

    # A handful of actual example blobs, sorted small to large, so a human
    # can look at what the canvas/blur stack actually looks like -- the
    # scatter figure above never shows a single pixel of it.
    sort_idx = np.argsort(radius)
    n_examples = min(8, len(sort_idx))
    pick = sort_idx[np.linspace(0, len(sort_idx) - 1, n_examples).round()
                    .astype(int)]
    crop_half = int(4 * radius.max())
    examples_out = os.path.join(fig_dir, 'scale_synthetic_control_examples.png')
    _draw_synthetic_examples_figure(stack, xy, radius, order, pick,
                                    crop_half, examples_out)
    print(f'  saved {examples_out}', flush=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--parts', nargs='+',
                    choices=['chains_stack', 'merge_grid', 'visualize',
                            'scale_diagnostic', 'scale_synthetic_control'],
                    default=['chains_stack', 'merge_grid', 'visualize'],
                    help='which demo(s) to run -- the original three by '
                         'default; scale_diagnostic needs --checkpoint and '
                         'is never included by default (it is a real '
                         'detector run, not a cheap synthetic check)')

    # ── shared ────────────────────────────────────────────────────────────
    ap.add_argument('--wsi-name', default=None,
                    help='resolved via AccessDatasets -- chains_stack '
                         f'defaults to {DEFAULT_WSI_NAME} if unset; '
                         'merge_grid needs it only with --checkpoint/'
                         '--real-flow')
    add_corpus_args(ap, corpus=False)
    prepare_chain_stack.add_axis_corpus_args(ap)
    ap.add_argument('--c-rungs', type=float, nargs='+',
                    default=[1.0, 2.0, 4.0, 8.0, 16.0],
                    help="chains_stack: C's mother is always the coarsest "
                         'of these. merge_grid: which rungs a real C-tree '
                         'covers')
    add_chainstack_args(ap, 'DemoSurvivalAnalysis', on=True)
    ap.add_argument('--out', default=None,
                    help='output DIRECTORY (default: '
                         "job_result_dir('DemoSurvivalAnalysis')) -- shared "
                         'root for both parts, each keeps its own filenames')

    # ── chains_stack only ────────────────────────────────────────────────
    ap.add_argument('--wsi', default=None,
                    help='[chains_stack] bypass AccessDatasets, use this '
                         'path directly (needs --wsi-stem too)')
    ap.add_argument('--wsi-stem', default=None)
    ap.add_argument('--rungs', type=float, nargs='+',
                    default=[1.0, 2.0, 4.0, 8.0, 16.0],
                    help='[chains_stack] F chain completeness + R output '
                         "rungs -- must match prepare_chain_stack.py's own "
                         '--rungs')
    ap.add_argument('--lineage-index', type=int, default=0,
                    help='[chains_stack]')

    # ── merge_grid only ───────────────────────────────────────────────────
    ap.add_argument('--n-clusters', type=int, default=40,
                    help='[merge_grid] figure only -- kept small enough to plot')
    ap.add_argument('--max-per-cluster', type=int, default=6,
                    help='[merge_grid]')
    ap.add_argument('--jitter', type=float, default=1.5, help='[merge_grid]')
    ap.add_argument('--extent', type=float, default=100.0, help='[merge_grid]')
    ap.add_argument('--radius', type=float, default=6.0, help='[merge_grid]')
    ap.add_argument('--seed', type=int, default=0, help='[merge_grid]')
    ap.add_argument('--timing-n-clusters', type=int, default=800,
                    help='[merge_grid] timing/agreement only -- not plotted')
    ap.add_argument('--timing-extent', type=float, default=400.0,
                    help='[merge_grid]')
    ap.add_argument('--skip-original-above', type=int, default=20_000,
                    help='[merge_grid] point count above which the O(n^2) '
                         'original is skipped in the timing comparison')
    ap.add_argument('--repeats', type=int, default=3, help='[merge_grid]')
    ap.add_argument('--checkpoint', default=None,
                    help='[merge_grid] enables the real-data comparison: a '
                         'trained KeypointNet .pt, loaded to detect on one '
                         'C-tree')
    ap.add_argument('--tree-index', type=int, default=0,
                    help='[merge_grid] which C-tree in the forest to read')
    ap.add_argument('--real-rung', type=float, default=None,
                    help='[merge_grid] which c-rung to pull detections from '
                         '(default: the finest, min(--c-rungs))')
    ap.add_argument('--score-threshold', type=float, default=None,
                    help="[merge_grid] defaults to the checkpoint's own "
                         'cfg.detection_threshold')
    ap.add_argument('--plot-real', action='store_true',
                    help='[merge_grid] also save a scatter of the real '
                         'point cloud')
    ap.add_argument('--real-flow', action='store_true',
                    help='[merge_grid] enables the FOURTH comparison: '
                         'deploy each of the three implementations into '
                         "survival_alpha_analysis.py's own C-axis flow. "
                         'Requires --checkpoint and --wsi-name')
    ap.add_argument('--real-flow-n-trees', type=int, default=2,
                    help='[merge_grid] how many C-trees to run the flow on')

    # ── visualize only ───────────────────────────────────────────────────
    ap.add_argument('--viz-alpha', type=float, default=_VIZ_ALPHA_DEFAULT,
                    help='[visualize] tau = viz_alpha * ds for the alive '
                         'animation -- real analysis found 2.5-3.5 to work '
                         'well, but this demo\'s compact synthetic scene '
                         'needs a smaller value to keep every point safely '
                         'separated. Default is DERIVED from the actual '
                         'point layout (_VIZ_ALPHA_DEFAULT), not a literal, '
                         'so it never drifts out of sync with the scene '
                         'the way a hand-typed one did on 2026-09-11 -- '
                         'raise it past the safe ceiling and _run_'
                         'visualize\'s own check raises SystemExit with the '
                         'exact number instead of silently rendering an '
                         'unsafe scene')
    ap.add_argument('--viz-fps', type=float, default=1.2,
                    help='[visualize] GIF playback speed -- kept slow on '
                         'purpose, these are meant to be read frame by '
                         'frame, not watched at speed')
    ap.add_argument('--viz-real-crop', type=int, default=256,
                    help='[visualize] side length (level-0 px) of the '
                         'native-resolution tissue crop in '
                         'visualize_real_example.png, centred on the '
                         'anchors\' own centroid')

    # ── scale_diagnostic only ────────────────────────────────────────────
    ap.add_argument('--n-scale-tiles', type=int, default=10,
                    help='[scale_diagnostic] how many F chains\' own ds=1 '
                         'tile to run both stacks on')
    ap.add_argument('--scale-alpha', type=float, default=3.0,
                    help='[scale_diagnostic] tau = max(scale_tau_floor, '
                         'scale_alpha * ds) for the alive decision -- 3.0 '
                         'is plan.md 2.2\'s settled production value, not '
                         'this file\'s own default')
    ap.add_argument('--scale-tau-floor', type=float, default=0.0,
                    help='[scale_diagnostic]')
    ap.add_argument('--decay-permutations', type=int, default=500,
                    help='[scale_diagnostic, scale_synthetic_control] '
                         'shuffles for the decay-rate correlation\'s '
                         'permutation test (--seed is reused as this '
                         'shuffle\'s own seed)')
    ap.add_argument('--n-synthetic-points', type=int, default=2917,
                    help='[scale_synthetic_control] how many known-radius '
                         'blobs -- 2917 matches the pooled anchor count '
                         'scale_diagnostic reported on BRACS_1228 with the '
                         'default --n-scale-tiles; override to match '
                         'whatever a different scale_diagnostic run '
                         'actually produced')
    args = ap.parse_args()

    out_dir = args.out or job_result_dir('DemoSurvivalAnalysis')
    os.makedirs(out_dir, exist_ok=True)

    status = 0
    if 'chains_stack' in args.parts:
        print('======== [chains_stack] ========')
        _run_chains_stack(args, out_dir)
    if 'merge_grid' in args.parts:
        print('\n======== [merge_grid] ========')
        status = max(status, _run_merge_grid(args, out_dir))
    if 'visualize' in args.parts:
        print('\n======== [visualize] ========')
        _run_visualize(args, out_dir)
    if 'scale_diagnostic' in args.parts:
        print('\n======== [scale_diagnostic] ========')
        _run_scale_diagnostic(args, out_dir)
    if 'scale_synthetic_control' in args.parts:
        print('\n======== [scale_synthetic_control] ========')
        _run_scale_synthetic_control(args, out_dir)
    return status


if __name__ == '__main__':
    sys.exit(main())
